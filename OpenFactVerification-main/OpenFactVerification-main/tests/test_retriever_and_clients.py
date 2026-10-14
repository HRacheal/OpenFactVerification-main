"""Serper retriever and LLM provider clients. All HTTP / SDK calls are mocked; no API costs."""

from types import SimpleNamespace

import pytest
import requests

from factcheck.core.Retriever import serper_retriever
from factcheck.core.Retriever.serper_retriever import SerperEvidenceRetriever, RetrievalError
from factcheck.utils.llmclient import GPTClient, ClaudeClient, GeminiClient, MissingCredentialsError, LLMProviderError


class FakePost:
    def __init__(self, status=200, payload=None, exc=None):
        self.status, self.payload, self.exc = status, payload, exc
        self.sent = None

    def __call__(self, url, headers=None, json=None, timeout=None):
        self.sent = {"url": url, "headers": headers, "json": json, "timeout": timeout}
        if self.exc:
            raise self.exc
        return SimpleNamespace(status_code=self.status, json=lambda: self.payload)


SERPER_RESPONSE = [
    {
        "searchParameters": {"q": "Uganda districts"},
        "organic": [
            {"title": "Districts of Uganda", "link": "https://www.statistics.go.ug/districts", "snippet": "Uganda has 146 districts according to the 2024 census report published by the bureau of statistics.", "date": "Jun 1, 2024"},
            {"title": "Fake", "link": "javascript:alert(1)", "snippet": "bad"},
            {"title": "Facebook post", "link": "https://www.facebook.com/p/1", "snippet": "Uganda has 200 districts!!"},
        ],
    },
    {"searchParameters": {"q": "q2"}, "answerBox": {"answer": "146"}, "organic": []},
]


def make_retriever(monkeypatch, post, pages=None):
    monkeypatch.setattr(serper_retriever.requests, "post", post)
    pages = pages or {}
    monkeypatch.setattr(SerperEvidenceRetriever, "_read_page", lambda self, url: pages.get(url))
    return SerperEvidenceRetriever(api_config={"SERPER_API_KEY": "test-key"})


def test_metadata_is_preserved_and_country_bias_sent(monkeypatch):
    post = FakePost(payload=SERPER_RESPONSE)
    page_text = "Intro text. " * 20 + "Uganda has 146 districts according to the 2024 census report published by the bureau of statistics. More detail follows here."
    r = make_retriever(monkeypatch, post, {"https://www.statistics.go.ug/districts": {"text": page_text, "title": "", "site_name": "UBOS", "published_date": "2024-06-01"}})
    r.set_context(country="Uganda", language="en")
    out = r.retrieve_evidence({"Uganda has 146 districts.": ["Uganda districts", "q2"]})
    assert post.sent["json"][0]["gl"] == "ug" and post.sent["headers"]["X-API-KEY"] == "test-key"
    evs = out["Uganda has 146 districts."]
    official = next(e for e in evs if "statistics.go.ug" in e["url"])
    assert official["title"] == "Districts of Uganda"
    assert official["source_type"] == "official" and official["full_text_fetched"] is True
    assert official["published_date"] == "2024-06-01" and official["retrieved_at"]
    assert not any("javascript" in e["url"] for e in evs)
    fb = next(e for e in evs if "facebook" in e["url"])
    assert fb["source_type"] == "social" and fb["full_text_fetched"] is False  # page not readable -> snippet only
    box = next(e for e in evs if e["url"] == "")
    assert box["text"] == "146" and box["publisher"] == "Search engine"  # no fake URL like "Google Answer Box"


def test_missing_key_raises_clear_error(monkeypatch):
    r = SerperEvidenceRetriever(api_config={})
    with pytest.raises(RetrievalError, match="SERPER_API_KEY"):
        r.retrieve_evidence({"c": ["c"]})


@pytest.mark.parametrize(
    "post,match",
    [
        (FakePost(status=403), "rejected the API key"),
        (FakePost(status=429), "quota"),
        (FakePost(status=500), "HTTP 500"),
        (FakePost(exc=requests.exceptions.Timeout()), "timed out"),
        (FakePost(exc=requests.exceptions.ConnectionError()), "Could not reach"),
        (FakePost(payload={"unexpected": True}), "unexpected response"),
    ],
)
def test_provider_failures_raise_instead_of_inventing_evidence(monkeypatch, post, match):
    r = make_retriever(monkeypatch, post)
    with pytest.raises(RetrievalError, match=match):
        r.retrieve_evidence({"Claim one.": ["Claim one."], "Claim two.": ["Claim two."]})


def test_empty_search_results_give_no_evidence(monkeypatch):
    r = make_retriever(monkeypatch, FakePost(payload=[{"organic": []}]))
    assert r.retrieve_evidence({"c": ["c"]}) == {"c": []}


# ---------------------------------------------------------------- LLM clients
def test_missing_keys_raise_missing_credentials():
    with pytest.raises(MissingCredentialsError, match="OPENAI_API_KEY"):
        GPTClient(api_config={})
    with pytest.raises(MissingCredentialsError, match="ANTHROPIC_API_KEY"):
        ClaudeClient(api_config={})
    with pytest.raises(MissingCredentialsError, match="GEMINI_API_KEY"):
        GeminiClient(api_config={})


def _gpt_response(content, finish="stop"):
    return SimpleNamespace(
        choices=[SimpleNamespace(finish_reason=finish, message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
    )


def test_gpt_client_returns_text_and_logs_usage():
    c = GPTClient(api_config={"OPENAI_API_KEY": "sk-test"})
    c.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: _gpt_response('{"a":1}'))))
    assert c.call([[{"role": "user", "content": "hi"}]]) == '{"a":1}'
    assert c.usage.prompt_tokens == 11 and c.usage.completion_tokens == 7


def test_gpt_client_retries_transient_errors_then_fails_cleanly(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = {"n": 0}

    def flaky(**kw):
        calls["n"] += 1
        raise TimeoutError("network")

    c = GPTClient(api_config={"OPENAI_API_KEY": "sk-test"})
    c.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=flaky)))
    with pytest.raises(LLMProviderError) as exc:
        c.call([[{"role": "user", "content": "hi"}]], num_retries=3)
    assert calls["n"] == 3 and "sk-test" not in str(exc.value)


def _claude_response(text="{}", stop="end_turn"):
    return SimpleNamespace(
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
        stop_reason=stop,
        usage=SimpleNamespace(input_tokens=20, output_tokens=4),
    )


def test_claude_client_uses_current_model_fallback_and_parses_text():
    c = ClaudeClient(api_config={"ANTHROPIC_API_KEY": "sk-ant-test"})
    assert c.model == "claude-opus-5-5"
    seen = {}

    def create(**kw):
        seen.update(kw)
        return _claude_response('{"ok": true}')

    c.client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)), messages=None)
    out = c.call(c.construct_message_list(["Return JSON"]))
    assert out == '{"ok": true}'
    assert seen["fallbacks"] == "default" and seen["betas"] == ["server-side-fallback-2026-07-01"]
    assert seen["output_config"] == {"effort": "medium"} and "JSON" in seen["system"]
    assert c.usage.prompt_tokens == 20


def test_claude_refusal_is_reported():
    c = ClaudeClient(model="claude-haiku-4-5", api_config={"ANTHROPIC_API_KEY": "sk-ant-test"})
    c.client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: _claude_response("", "refusal")))
    with pytest.raises(LLMProviderError, match="declined"):
        c.call(c.construct_message_list(["x"]))


def test_claude_vision_sends_base64_image():
    c = ClaudeClient(model="claude-haiku-4-5", api_config={"ANTHROPIC_API_KEY": "sk-ant-test"})
    seen = {}
    c.client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: seen.update(kw) or _claude_response('{"visible_text":"x"}')))
    c.vision("Read this", [(b"\x89PNG....", "image/png")])
    block = seen["messages"][0]["content"][0]
    assert block["type"] == "image" and block["source"]["media_type"] == "image/png"
    assert "output_config" not in seen  # Haiku 4.5 does not accept effort


def test_multi_call_raises_when_every_call_fails():
    from tests.conftest import FakeLLM

    class Broken(FakeLLM):
        def _call(self, messages, **kw):
            raise MissingCredentialsError("no key")

    with pytest.raises(MissingCredentialsError):
        Broken().multi_call([[{"role": "user", "content": "a"}]])


def _gemini_client(seen, text='{"a":1}', **response):
    c = GeminiClient(api_config={"GEMINI_API_KEY": "test-key"})

    def generate_content(**kw):
        seen.update(kw)
        return SimpleNamespace(
            text=text, candidates=[SimpleNamespace(finish_reason="STOP")], prompt_feedback=None,
            usage_metadata=SimpleNamespace(prompt_token_count=5, candidates_token_count=3), **response,
        )

    c.client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    return c


def test_gemini_client_requests_json_and_logs_usage():
    seen = {}
    c = _gemini_client(seen)
    assert c.call(c.construct_message_list(["hi"])) == '{"a":1}'
    assert seen["model"] == "gemini-3.8-flash"
    assert seen["config"].response_mime_type == "application/json"
    assert "JSON" in seen["config"].system_instruction
    assert c.usage.prompt_tokens == 5 and c.usage.completion_tokens == 3


def test_gemini_vision_sends_inline_image():
    seen = {}
    c = _gemini_client(seen)
    c.vision("Read this", [(b"fake-png-bytes", "image/png")])
    parts = seen["contents"][0].parts
    assert parts[0].inline_data.mime_type == "image/png" and parts[-1].text == "Read this"


def test_gemini_bad_request_is_not_retried():
    from google.genai import errors

    calls = []
    c = GeminiClient(api_config={"GEMINI_API_KEY": "test-key"})

    def generate_content(**kw):
        calls.append(1)
        raise errors.ClientError(404, {"error": {"code": 404, "message": "model not found", "status": "NOT_FOUND"}})

    c.client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    with pytest.raises(LLMProviderError, match="404"):
        c.call(c.construct_message_list(["hi"]))
    assert len(calls) == 1


@pytest.mark.parametrize("configured, sent", [
    ("", "gemini-3.8-flash"),
    ("gemini-1.5-flash", "gemini-3.8-flash"),
    ("gemini/gemini-2.5-flash", "gemini-2.5-flash"),
    ("models/gemini-2.5-flash", "gemini-2.5-flash"),
    (" 'gemini-2.5-pro' ", "gemini-2.5-pro"),
])
def test_gemini_model_names_are_normalized(configured, sent):
    assert GeminiClient(model=configured, api_config={"GEMINI_API_KEY": "k"}).model == sent


def test_gemini_404_switches_to_an_available_model_once(caplog):
    from google.genai import errors

    seen = []
    c = GeminiClient(model="gemini-9-gone", api_config={"GEMINI_API_KEY": "test-key"})

    def generate_content(model, **kw):
        seen.append(model)
        if model == "gemini-9-gone":
            raise errors.ClientError(404, {"error": {"code": 404, "message": "not found", "status": "NOT_FOUND"}})
        return SimpleNamespace(text='{"ok":1}', candidates=[], prompt_feedback=None, usage_metadata=None)

    listed = [SimpleNamespace(name=f"models/{n}", supported_actions=["generateContent"])
              for n in ("gemini-2.0-flash", "gemini-3.0-flash", "gemini-3.0-flash-lite")]
    c.client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content, list=lambda: listed))
    assert c.call(c.construct_message_list(["hi"])) == '{"ok":1}'
    assert seen == ["gemini-9-gone", "gemini-3.0-flash"] and c.model == "gemini-3.0-flash"


# ---------------------------------------------------------------------------- Claude HTTP errors and retries
# These drive the real Anthropic SDK against a mocked HTTP transport, so the SDK's own status -> exception
# mapping and retry loop are exercised. Nothing leaves the machine.
import anthropic  # noqa: E402
import httpx2  # noqa: E402

from factcheck.utils.llmclient import claude_client  # noqa: E402


def _claude_with_transport(statuses, message="error", model="claude-haiku-5-5"):
    """A ClaudeClient whose HTTP layer answers with each status in turn (the last one repeats)."""
    calls = []

    def handler(request):
        status = statuses[min(len(calls), len(statuses) - 1)]
        calls.append(status)
        if status == 200:
            return httpx2.Response(200, json={
                "id": "msg_1", "type": "message", "role": "assistant", "model": model,
                "content": [{"type": "text", "text": '{"ok": true}'}], "stop_reason": "end_turn",
                "stop_sequence": None, "usage": {"input_tokens": 3, "output_tokens": 2},
            })
        return httpx2.Response(status, headers={"retry-after-ms": "1"},
                               json={"type": "error", "error": {"type": "api_error", "message": message}})

    c = ClaudeClient(model=model, api_config={"ANTHROPIC_API_KEY": "sk-ant-test"})
    c.client = anthropic.Anthropic(
        api_key="sk-ant-test", max_retries=claude_client.SDK_MAX_RETRIES,
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    return c, calls


@pytest.mark.parametrize("status, match", [
    (401, "rejected the API key"),
    (403, "not allowed to use"),
    (404, "not found .*retired"),
    (400, "rejected the request .HTTP 400"),
])
def test_claude_permanent_errors_are_sent_once_with_a_clear_message(status, match):
    c, calls = _claude_with_transport([status])
    with pytest.raises(LLMProviderError, match=match):
        c.call(c.construct_message_list(["x"]))
    assert calls == [status]  # never retried: neither by the SDK nor by BaseClient


def test_claude_exhausted_credits_are_not_retried():
    c, calls = _claude_with_transport([400], message="Your credit balance is too low to access the Anthropic API.")
    with pytest.raises(LLMProviderError, match="credit balance is too low"):
        c.call(c.construct_message_list(["x"]))
    assert calls == [400]


@pytest.mark.parametrize("status, match", [
    (429, "rate limit"), (500, "temporarily unavailable"), (503, "temporarily unavailable"),
    (504, "temporarily unavailable"), (529, "temporarily unavailable"),
])
def test_claude_transient_errors_get_bounded_retries_then_a_clear_message(status, match):
    c, calls = _claude_with_transport([status])
    with pytest.raises(LLMProviderError, match=match):
        c.call(c.construct_message_list(["x"]), num_retries=3)
    # 1 attempt + SDK_MAX_RETRIES, even though BaseClient was allowed 3 attempts: only one retry layer.
    assert calls == [status] * (1 + claude_client.SDK_MAX_RETRIES)


def test_claude_recovers_when_a_transient_error_clears():
    c, calls = _claude_with_transport([529, 429, 200])
    assert c.call(c.construct_message_list(["x"]), num_retries=1) == '{"ok": true}'
    assert calls == [529, 429, 200]


def test_claude_network_errors_are_reported_clearly():
    c = ClaudeClient(model="claude-haiku-5-5", api_config={"ANTHROPIC_API_KEY": "sk-ant-test"})
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")

    def timeout(**kw):
        raise anthropic.APITimeoutError(request=request)

    def offline(**kw):
        raise anthropic.APIConnectionError(request=request)

    for create, match in ((timeout, "did not answer in time"), (offline, "Could not reach")):
        c.client = SimpleNamespace(messages=SimpleNamespace(create=create))
        with pytest.raises(LLMProviderError, match=match):
            c.call(c.construct_message_list(["x"]))


@pytest.mark.parametrize("value, expected", [("xhigh", "xhigh"), (" LOW ", "low"), ("bogus", "medium"), ("", "medium")])
def test_claude_effort_is_validated(monkeypatch, value, expected):
    monkeypatch.setenv("CLAUDE_EFFORT", value)
    c = ClaudeClient(model="claude-haiku-5-5", api_config={"ANTHROPIC_API_KEY": "sk-ant-test"})
    assert c._request_kwargs() == {"output_config": {"effort": expected}}  # Haiku 5.5: effort, no fallback


def test_claude_client_uses_sdk_retries():
    c = ClaudeClient(api_config={"ANTHROPIC_API_KEY": "sk-ant-test"})
    assert c.client.max_retries == claude_client.SDK_MAX_RETRIES == 2
