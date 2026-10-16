"""Flask web app: forms, CSRF, uploads, background jobs, report rendering and security headers.
The pipeline is the FAKE one from conftest (simulated evidence)."""

import io
import re

import pytest

import webapp
from tests.conftest import FakeRetriever, FakeLLM, ev


READY_STATUS = {
    "provider": "fake", "model": "fake-model", "llm_ready": True, "llm_key_name": "OPENAI_API_KEY",
    "search_ready": True, "transcription_ready": True, "transcription_reason": "", "whisper_model": "base",
    "max_audio_seconds": 600, "vision_ready": True, "ffmpeg_ready": False}


class SyncExecutor:
    def submit(self, fn, *a, **kw):
        fn(*a, **kw)


@pytest.fixture
def client(fake_pipeline, monkeypatch):
    monkeypatch.setattr(webapp, "_executor", SyncExecutor())
    monkeypatch.setattr(webapp, "_fc_instance", fake_pipeline)
    monkeypatch.setattr(webapp, "config_status", lambda: dict(READY_STATUS))
    webapp._jobs.clear()
    webapp.app.config["TESTING"] = True
    return webapp.app.test_client()


def csrf(c):
    html = c.get("/").get_data(as_text=True)
    return re.search(r'name="csrf_token" value="([^"]+)"', html).group(1)


def submit(c, **data):
    data.setdefault("csrf_token", csrf(c))
    data.setdefault("country", "Uganda")
    data.setdefault("language", "en")
    data.setdefault("output_language", "en")
    return c.post("/check", data=data, content_type="multipart/form-data")


def test_home_page_and_security_headers(client):
    r = client.get("/")
    assert r.status_code == 200 and b"Kweli Lens AI" in r.data
    assert "script-src 'self'" in r.headers["Content-Security-Policy"]
    assert r.headers["X-Frame-Options"] == "DENY" and r.headers["Referrer-Policy"] == "no-referrer"


@pytest.mark.parametrize("path, status", [("/", 200), ("/about", 200), ("/report/missing", 404)])
def test_pages_carry_kweli_lens_branding(client, path, status):
    r = client.get(path)
    html = r.get_data(as_text=True)
    assert r.status_code == status
    assert "<title>" in html and "Kweli Lens AI</title>" in html  # browser tab
    assert re.search(r'class="brand"[^>]*>.*Kweli Lens AI</a>', html)  # header
    assert "Kweli Lens AI assists journalists" in html  # footer
    # The upstream attribution is not shown in the UI (the MIT notice stays in LICENSE).
    assert "Loki (OpenFactVerification)" not in html and "MIT licence" not in html
    assert "TruthLens" not in html


def test_about_page_documents_verdict_rules(client):
    html = client.get("/about").get_data(as_text=True)
    assert "Unverified" in html and "never means" in html
    assert "Related reports are not counted" in html  # the related-media rule is documented
    assert "faster-whisper" in html


def test_csrf_is_required(client):
    client.get("/")
    r = client.post("/check", data={"input_type": "text", "text": "Kampala is in Uganda."})
    assert r.status_code == 400 and b"session expired" in r.data


def test_missing_keys_give_clear_message(client, monkeypatch):
    status = {"provider": "gpt", "model": "m", "llm_ready": False, "llm_key_name": "OPENAI_API_KEY",
              "search_ready": False, "transcription_ready": False, "transcription_reason": "off", "whisper_model": "base",
              "max_audio_seconds": 600, "vision_ready": False, "ffmpeg_ready": False}
    monkeypatch.setattr(webapp, "config_status", lambda: status)
    r = submit(client, input_type="text", text="Kampala is in Uganda.")
    assert r.status_code == 503 and b"OPENAI_API_KEY" in r.data


def test_text_check_end_to_end_report(client):
    FakeRetriever.corpus = {"Kampala": [
        ev("[S] Kampala is the capital of Uganda.", "https://www.gov.ug/about", "official"),
        ev("[C] Background on Uganda.", "javascript:alert(document.cookie)", "unknown"),
    ]}
    r = submit(client, input_type="text", text="Kampala is the capital of Uganda. <script>alert('x')</script>")
    assert r.status_code == 303
    job_url = r.headers["Location"]
    r = client.get(job_url)
    assert r.status_code == 302 and "/report/" in r.headers["Location"]
    report_url = r.headers["Location"]
    html = client.get(report_url).get_data(as_text=True)
    assert "Verdict: True" in html and "✔" in html  # verdict shown as icon + text, not colour only
    # the submitted claim, then the verdict, then the explanation, then the sources
    assert html.index("You asked us to check") < html.index("Verdict: True") < html.index(">Why<") < html.index("Key sources")
    assert 'href="https://www.gov.ug/about"' in html and 'rel="noopener noreferrer nofollow"' in html
    assert 'href="javascript' not in html  # unsafe link never becomes clickable
    assert "<script>alert('x')</script>" not in html and "&lt;script&gt;" in html  # user input escaped
    assert "Editorial reminder" in html and "Verification performed" in html
    assert "<title>Fact-check report - Kweli Lens AI</title>" in html
    status = client.get(job_url.replace("/job/", "/api/job/")).get_json()
    assert status["state"] == "done"
    j = client.get(report_url + ".json")
    assert j.status_code == 200 and j.get_json()["claim_detail"][0]["verdict"] == "SUPPORTED"
    assert "filename=kweli-lens-report-" in j.headers["Content-Disposition"]


def test_reports_are_private_to_the_session(client):
    r = submit(client, input_type="text", text="Kampala is the capital of Uganda.")
    job_url = r.headers["Location"]
    other = webapp.app.test_client()
    assert other.get(job_url).status_code == 404
    assert other.get(job_url.replace("/job/", "/report/")).status_code == 404


def test_pipeline_errors_are_shown_without_internals(client):
    from factcheck.core.Retriever import RetrievalError

    FakeRetriever.error = RetrievalError("The evidence search service rejected the API key. Check SERPER_API_KEY.")
    r = submit(client, input_type="text", text="Kampala is the capital of Uganda.")
    status = client.get(r.headers["Location"].replace("/job/", "/api/job/")).get_json()
    assert status["state"] == "error" and "rejected the API key" in status["error"]
    html = client.get(r.headers["Location"]).get_data(as_text=True)
    assert "The check failed" in html


def test_invalid_inputs_rejected(client):
    assert submit(client, input_type="text", text="   ").status_code == 400
    assert submit(client, input_type="nonsense", text="x").status_code == 400
    r = submit(client, input_type="url", url="http://127.0.0.1/admin")
    assert r.status_code == 400 and b"local or internal" in r.data
    r = submit(client, input_type="image", file=(io.BytesIO(b"MZ..."), "virus.exe"))
    assert r.status_code == 400 and b"Unsupported image file type" in r.data
    r = submit(client, input_type="image")
    assert r.status_code == 400 and b"choose a image file" in r.data


def test_image_upload_flow_and_temp_cleanup(client, tmp_path, monkeypatch):
    import cv2
    import numpy as np
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    ok, png = cv2.imencode(".png", np.full((50, 80, 3), 255, np.uint8))
    FakeRetriever.corpus = {"districts": [ev("[S] Uganda has 146 districts.", "https://www.gov.ug/d", "official")]}
    r = submit(client, input_type="image", file=(io.BytesIO(png.tobytes()), "post.png"))
    assert r.status_code == 303
    html = client.get(r.headers["Location"].replace("/job/", "/report/")).get_data(as_text=True)
    assert "Text read from the image" in html and "Uganda has 146 districts." in html
    assert not any(p.name.startswith("truthlens_") for p in tmp_path.iterdir())  # upload deleted


def test_fake_image_content_is_rejected_during_processing(client):
    r = submit(client, input_type="image", file=(io.BytesIO(b"this is not really a png"), "fake.png"))
    status = client.get(r.headers["Location"].replace("/job/", "/api/job/")).get_json()
    assert status["state"] == "error" and "Unsupported image format" in status["error"]


def test_oversized_upload(client):
    webapp.app.config["MAX_CONTENT_LENGTH"] = 1000
    try:
        r = submit(client, input_type="image", file=(io.BytesIO(b"\x89PNG" + b"0" * 5000), "big.png"))
        assert r.status_code == 413
    finally:
        webapp.app.config["MAX_CONTENT_LENGTH"] = 101 * 1024 * 1024


def test_safe_href_filter():
    assert webapp.safe_href("https://a.com/x") == "https://a.com/x"
    for bad in ("javascript:alert(1)", "data:text/html,x", "", None, "//evil.com"):
        assert webapp.safe_href(bad) == ""


def test_gemini_provider_resolves_with_default_model(monkeypatch):
    monkeypatch.setitem(webapp.SETTINGS, "provider", None)
    monkeypatch.setitem(webapp.SETTINGS, "model", None)
    monkeypatch.setenv("TRUTHLENS_PROVIDER", "gemini")
    monkeypatch.setenv("TRUTHLENS_MODEL", "")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    assert webapp.resolve_provider() == ("gemini", "gemini-3.8-flash")
    status = webapp.config_status()
    assert status["llm_ready"] and status["llm_key_name"] == "GEMINI_API_KEY" and status["vision_ready"]


def test_claude_provider_wires_all_five_components(monkeypatch):
    """With TRUTHLENS_PROVIDER=claude every pipeline step uses Claude (no Gemini/OpenAI client is created)."""
    from factcheck import FactCheck
    from factcheck.utils.llmclient import ClaudeClient

    monkeypatch.setitem(webapp.SETTINGS, "provider", None)
    monkeypatch.setitem(webapp.SETTINGS, "model", None)
    monkeypatch.setenv("TRUTHLENS_PROVIDER", "claude")
    monkeypatch.setenv("TRUTHLENS_MODEL", "claude-haiku-5-5")
    provider, model = webapp.resolve_provider()
    assert (provider, model) == ("claude", "claude-haiku-5-5")
    fc = FactCheck(default_model=model, client=provider, api_config={"ANTHROPIC_API_KEY": "sk-ant-test", "SERPER_API_KEY": "k"})
    for step in ("decompose_model", "checkworthy_model", "query_generator_model", "evidence_retrieval_model", "claim_verify_model"):
        client = getattr(fc, step)
        assert isinstance(client, ClaudeClient) and client.model == "claude-haiku-5-5", step
    assert fc.vision_client is fc.decompose_model


def test_claude_api_failure_ends_the_job_with_a_readable_error(client, fake_pipeline, monkeypatch):
    """A failed Claude call must mark the job as failed (not leave it running) and show the cause."""
    from factcheck.utils.llmclient import LLMProviderError

    def fail(*a, **kw):
        raise LLMProviderError("Your Anthropic credit balance is too low.")

    monkeypatch.setattr(fake_pipeline.decomposer, "getclaims", fail)
    r = submit(client, input_type="text", text="Uganda has 146 districts.")
    status = client.get(r.headers["Location"].replace("/job/", "/api/job/")).get_json()
    assert status["state"] == "error" and "credit balance" in status["error"]


# --------------------------------------------------------------------------- audio/video -> transcript review -> fact-check
def make_wav_bytes(seconds=1):
    import math
    import struct
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000)
        w.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(i / 10))) for i in range(16000 * seconds)))
    return buf.getvalue()


@pytest.fixture
def fake_whisper(monkeypatch):
    from factcheck.utils import transcription
    from tests.test_multimodal import FakeTranscriber

    t = FakeTranscriber("President Ruto is dead.")
    monkeypatch.setattr(transcription, "get_transcriber", lambda: t)
    return t


def test_audio_upload_is_transcribed_then_reviewed_before_fact_checking(client, fake_whisper, tmp_path, monkeypatch):
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    r = submit(client, input_type="audio", language="auto", file=(io.BytesIO(make_wav_bytes()), "clip.wav"))
    assert r.status_code == 303
    job_url = r.headers["Location"]
    r = client.get(job_url)
    assert r.status_code == 302 and "/transcript/" in r.headers["Location"]  # review step, not a report
    review_url = r.headers["Location"]
    assert not any(p.name.startswith("truthlens_") for p in tmp_path.iterdir())  # recording deleted
    assert client.get(job_url.replace("/job/", "/api/job/")).get_json()["next_url"].endswith(review_url)

    html = client.get(review_url).get_data(as_text=True)
    assert "Review the transcript" in html and "Nothing has been fact-checked yet" in html
    assert "President Ruto is dead." in html and 'name="transcript_id"' in html
    assert not FakeLLM.prompts_seen  # no AI or search calls before the user confirms
    assert client.get(job_url.replace("/job/", "/report/")).status_code == 302  # a transcript job has no report

    # the user corrects the transcript and submits it
    tid = review_url.rsplit("/", 1)[1]
    FakeRetriever.corpus = {"Ruto": [ev("[X][R] A video claiming Ruto died is doctored.", "https://pesacheck.org/v", "fact_checker")]}
    r = submit(client, input_type="text", text="President William Ruto is dead.", transcript_id=tid)
    assert r.status_code == 303
    report_html = client.get(r.headers["Location"].replace("/job/", "/report/")).get_data(as_text=True)
    assert "You submitted this from the audio transcript" in report_html and "clip.wav" in report_html
    assert "edited or selected part of the transcript" in report_html
    assert "Verdict: Unverified" in report_html and "Related reports and context" in report_html
    assert 'href="https://pesacheck.org/v"' in report_html
    assert report_html.index("Verdict: Unverified") < report_html.index("Related reports and context")


def test_segments_with_timestamps_are_offered_for_selection(client, monkeypatch):
    from factcheck.utils import transcription
    from factcheck.utils.transcription import Transcript

    class TwoSegments:
        def transcribe(self, path, language=None, progress=None):
            segs = [{"start": 0.0, "end": 4.0, "text": "Good evening."},
                    {"start": 65.0, "end": 70.5, "text": "Fuel prices will rise by 10% next month."}]
            return Transcript(text="Good evening. Fuel prices will rise by 10% next month.", segments=segs,
                              language="en", language_probability=0.99, duration=71.0, transcribed_seconds=71.0, model="base")

    monkeypatch.setattr(transcription, "get_transcriber", lambda: TwoSegments())
    r = submit(client, input_type="audio", file=(io.BytesIO(make_wav_bytes()), "news.wav"))
    html = client.get(client.get(r.headers["Location"]).headers["Location"]).get_data(as_text=True)
    assert html.count('class="seg-pick"') == 2
    assert "1:05–1:10" in html and 'data-text="Fuel prices will rise by 10% next month."' in html
    assert "Detected language: en" in html and "faster-whisper base" in html


def test_transcript_cannot_be_used_from_another_session(client, fake_whisper):
    r = submit(client, input_type="audio", file=(io.BytesIO(make_wav_bytes()), "clip.wav"))
    tid = client.get(r.headers["Location"]).headers["Location"].rsplit("/", 1)[1]
    other = webapp.app.test_client()
    assert other.get(f"/transcript/{tid}").status_code == 404
    r = submit(other, input_type="text", text="President Ruto is dead.", transcript_id=tid)
    assert r.status_code == 400 and b"transcript has expired" in r.data


def test_audio_upload_when_transcription_is_off(client, monkeypatch):
    monkeypatch.setattr(webapp, "config_status", lambda: dict(
        READY_STATUS, transcription_ready=False,
        transcription_reason="Audio/video transcription is turned off on this hosted server."))
    r = submit(client, input_type="audio", file=(io.BytesIO(make_wav_bytes()), "clip.wav"))
    assert r.status_code == 503 and b"turned off on this hosted server" in r.data
    assert b"turned off on this hosted server" in client.get("/").data  # also explained on the form


def test_unsupported_and_silent_audio_are_reported(client, fake_whisper):
    r = submit(client, input_type="audio", file=(io.BytesIO(b"x" * 100), "clip.exe"))
    assert r.status_code == 400 and b"Unsupported audio file type" in r.data
    r = submit(client, input_type="audio", file=(io.BytesIO(b"MZ not audio"), "fake.mp3"))
    status = client.get(r.headers["Location"].replace("/job/", "/api/job/")).get_json()
    assert status["state"] == "error" and "Unsupported audio format" in status["error"]
    fake_whisper.text = ""
    r = submit(client, input_type="audio", file=(io.BytesIO(make_wav_bytes()), "quiet.wav"))
    status = client.get(r.headers["Location"].replace("/job/", "/api/job/")).get_json()
    assert status["state"] == "error" and "No speech was detected" in status["error"]


def test_multi_claim_report_has_overview_and_unverified_never_says_false(client):
    r = submit(client, input_type="text", text="The Nile is 10 kilometres long. Kampala is the capital of Uganda.")
    html = client.get(r.headers["Location"].replace("/job/", "/report/")).get_data(as_text=True)
    assert "2 claims found" in html and "This does not mean the claim is false." in html
