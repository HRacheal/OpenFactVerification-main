"""Image, audio and video processing with locally generated media files. AI services and the Whisper model
are mocked unless stated; OpenCV and (if installed) ffmpeg run for real."""

import math
import os
import struct
import subprocess
import wave

import cv2
import numpy as np
import pytest

from factcheck.utils import multimodal, transcription
from factcheck.utils.multimodal import sniff_type, image2text, voice2text, video2text, MediaError, find_ffmpeg
from factcheck.utils.transcription import Transcript, TranscriptionError, WhisperTranscriber
from factcheck import inputs
from factcheck.inputs import InputError
from tests.conftest import FakeLLM


class FakeTranscriber:
    """Stands in for WhisperTranscriber (same interface, no model)."""

    def __init__(self, text="Rais alisema bei ya mafuta itapanda.", error=None):
        self.text, self.error, self.paths, self.languages = text, error, [], []

    def transcribe(self, path, language=None, progress=None):
        self.paths.append(path)
        self.languages.append(language)
        if self.error:
            raise self.error
        if progress:
            progress(1.0, 2.0)
        segments = [{"start": 0.0, "end": 2.0, "text": self.text}] if self.text else []
        return Transcript(text=self.text, segments=segments, language="sw", language_probability=0.97,
                          duration=2.0, transcribed_seconds=2.0, model="fake")


def make_png(path, w=200, h=100):
    img = np.full((h, w, 3), 255, np.uint8)
    cv2.putText(img, "Hello", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
    cv2.imwrite(str(path), img)
    return path


def make_wav(path, seconds=1):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000)
        w.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(i / 10))) for i in range(16000 * seconds)))
    return path


def make_mp4(path, seconds=3, fps=10):
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    for i in range(seconds * fps):
        frame = np.full((120, 160, 3), (i * 8) % 255, np.uint8)
        out.write(frame)
    out.release()
    return path


def test_sniff_type_uses_content_not_extension():
    assert sniff_type(b"\x89PNG\r\n\x1a\n0000") == "image/png"
    assert sniff_type(b"\xff\xd8\xff\xe0") == "image/jpeg"
    assert sniff_type(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "image/webp"
    assert sniff_type(b"RIFF\x00\x00\x00\x00WAVEfmt ") == "audio/wav"
    assert sniff_type(b"ID3\x04\x00") == "audio/mpeg"
    assert sniff_type(b"\x00\x00\x00\x18ftypmp42") == "video/mp4"
    assert sniff_type(b"MZ\x90\x00 executable") is None


def test_image_text_extraction(tmp_path):
    res = image2text(str(make_png(tmp_path / "a.png")), FakeLLM())
    assert res.text == "Uganda has 146 districts."
    assert res.extracted["description"] and any("check it against the original" in n for n in res.notes)


def test_large_image_is_downscaled(tmp_path):
    captured = {}

    class Spy(FakeLLM):
        def _call_vision(self, prompt, images, **kw):
            data, mime = images[0]
            captured["shape"] = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR).shape
            return super()._call_vision(prompt, images)

    image2text(str(make_png(tmp_path / "big.png", 4000, 3000)), Spy())
    assert max(captured["shape"][:2]) <= multimodal.VISION_MAX_SIDE


def test_fake_image_with_wrong_content_is_rejected(tmp_path):
    p = tmp_path / "evil.png"
    p.write_bytes(b"MZ this is not an image")
    with pytest.raises(MediaError, match="Unsupported image"):
        image2text(str(p), FakeLLM())


def test_image_without_text_uses_description_and_says_so(tmp_path):
    class NoText(FakeLLM):
        def _call_vision(self, prompt, images, **kw):
            return '{"visible_text": "", "description": "A crowd at a rally.", "unclear_parts": ["sign is blurry"]}'

    res = image2text(str(make_png(tmp_path / "a.png")), NoText())
    assert res.text == "A crowd at a rally."
    assert any("No readable text" in n for n in res.notes) and any("blurry" in n for n in res.notes)


def test_audio_transcription_keeps_segments_and_language(tmp_path):
    t = FakeTranscriber()
    messages = []
    res = voice2text(str(make_wav(tmp_path / "a.wav")), t, language="sw", progress=messages.append)
    assert res.text == t.text and res.extracted["transcript"] == t.text
    assert res.extracted["segments"][0] == {"start": 0.0, "end": 2.0, "text": t.text}
    assert res.extracted["spoken_language"] == "sw" and res.extracted["speech_model"] == "faster-whisper fake"
    assert t.languages == ["sw"] and any("0:01 of 0:02" in m for m in messages)
    assert any("may contain errors" in n for n in res.notes)


def test_audio_when_transcription_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setenv("TRUTHLENS_TRANSCRIPTION", "off")
    with pytest.raises(MediaError, match="turned off"):
        voice2text(str(make_wav(tmp_path / "a.wav")), None)


def test_silent_audio_gives_clear_error(tmp_path):
    with pytest.raises(MediaError, match="No speech was detected"):
        voice2text(str(make_wav(tmp_path / "a.wav")), FakeTranscriber(text=""))


def test_transcription_failure_is_reported_safely(tmp_path):
    t = FakeTranscriber(error=TranscriptionError("Speech transcription failed while processing this file."))
    with pytest.raises(MediaError, match="Speech transcription failed") as e:
        voice2text(str(make_wav(tmp_path / "a.wav")), t)
    assert str(tmp_path) not in str(e.value)  # no server paths in user-facing messages


def test_unsupported_audio_content_is_rejected(tmp_path):
    p = tmp_path / "x.mp3"
    p.write_bytes(b"MZ\x90\x00 not audio at all")
    with pytest.raises(MediaError, match="Unsupported audio format"):
        voice2text(str(p), FakeTranscriber())


def test_luganda_and_capped_audio_are_flagged(tmp_path):
    class Long(FakeTranscriber):
        def transcribe(self, path, language=None, progress=None):
            t = super().transcribe(path, language, progress)
            t.duration, t.transcribed_seconds = 1200.0, 600.0
            return t

    res = voice2text(str(make_wav(tmp_path / "a.wav")), Long(), language="lg")
    assert any("first 10:00 of a 20:00" in n for n in res.notes)
    assert any("no Luganda support" in n for n in res.notes)


@pytest.mark.parametrize("env, expected", [
    ({"TRUTHLENS_TRANSCRIPTION": "off"}, False),
    ({"RENDER": "true"}, False),  # hosted free instance: off unless explicitly enabled
    ({"RENDER": "true", "TRUTHLENS_TRANSCRIPTION": "on"}, True),
    ({}, True),
])
def test_transcription_enabled_switch(monkeypatch, env, expected):
    for k in ("TRUTHLENS_TRANSCRIPTION", "RENDER"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    ok, reason = transcription.enabled()
    assert ok is expected and (ok or reason)


def test_transcriber_loads_model_once_with_configured_settings(monkeypatch):
    created = []

    class FakeModel:
        def __init__(self, name, **kw):
            created.append((name, kw))

    import faster_whisper

    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeModel)
    monkeypatch.setenv("TRUTHLENS_WHISPER_MODEL", "tiny")
    t = WhisperTranscriber()
    t.load(), t.load()
    assert len(created) == 1
    name, kw = created[0]
    assert name == "tiny" and kw["device"] == "cpu" and kw["compute_type"] == "int8"


def test_model_download_failure_gives_clear_message(monkeypatch):
    def boom(*a, **kw):
        raise OSError("connection refused to huggingface.co at C:\\secret\\path")

    import faster_whisper

    monkeypatch.setattr(faster_whisper, "WhisperModel", boom)
    with pytest.raises(TranscriptionError, match="needs internet access") as e:
        WhisperTranscriber().load()
    assert "secret" not in str(e.value)


@pytest.mark.skipif(not find_ffmpeg(), reason="ffmpeg not installed")
def test_corrupt_audio_is_rejected_by_decoder(tmp_path):
    p = tmp_path / "broken.mp3"
    p.write_bytes(b"ID3\x04\x00" + os.urandom(4000))
    with pytest.raises(TranscriptionError, match="could not be decoded|too short"):
        WhisperTranscriber().transcribe(str(p))


def _whisper_model_cached() -> bool:
    try:
        from faster_whisper import WhisperModel

        WhisperModel(transcription.settings()["model"], device="cpu", compute_type="int8", local_files_only=True)
        return True
    except Exception:
        return False


@pytest.mark.slow
@pytest.mark.skipif(not _whisper_model_cached(), reason="Whisper model not downloaded (python -m factcheck.utils.transcription --download)")
def test_real_whisper_transcribes_demo_speech():
    """Runs the real local model on the bundled 8-second demo recording."""
    res = voice2text("demo_data/speech.mp3", WhisperTranscriber(), language="en")
    text = res.text.lower()
    assert "nile" in text and "africa" in text
    assert res.extracted["segments"] and res.extracted["spoken_language"] == "en"


def test_new_audio_formats_are_sniffed():
    assert sniff_type(b"fLaC\x00\x00\x00\x22") == "audio/flac"
    assert sniff_type(b"\xff\xf1\x50\x80") == "audio/aac"
    assert sniff_type(b"\xff\xfb\x90\x00") == "audio/mpeg"
    assert sniff_type(b"OggS\x00\x02") == "audio/ogg"


def test_video_frames_sampled_with_limits(tmp_path, monkeypatch):
    seen = {}

    class Spy(FakeLLM):
        def _call_vision(self, prompt, images, **kw):
            seen["n"] = len(images)
            return super()._call_vision(prompt, images)

    monkeypatch.setattr(multimodal, "find_ffmpeg", lambda: None)  # simulate missing ffmpeg
    res = video2text(str(make_mp4(tmp_path / "v.mp4")), Spy(), FakeTranscriber())
    assert seen["n"] == multimodal.VIDEO_FRAME_SAMPLES
    assert "BREAKING: Kampala floods" in res.text
    assert any("audio track could not be extracted" in n for n in res.notes)
    assert any("may not agree" in n for n in res.notes)


def test_long_video_only_partially_analysed(tmp_path, monkeypatch):
    monkeypatch.setattr(multimodal, "MAX_VIDEO_SECONDS", 1)
    monkeypatch.setattr(multimodal, "find_ffmpeg", lambda: None)
    res = video2text(str(make_mp4(tmp_path / "v.mp4", seconds=4)), FakeLLM(), None)
    assert any("Only the first" in n for n in res.notes)


@pytest.mark.skipif(not find_ffmpeg(), reason="ffmpeg not installed")
def test_video_audio_track_is_extracted_and_transcribed(tmp_path):
    src = tmp_path / "av.mp4"
    cmd = [find_ffmpeg(), "-nostdin", "-y", "-f", "lavfi", "-i", "color=c=blue:s=160x120:d=2", "-f", "lavfi",
           "-i", "sine=frequency=440:duration=2", "-shortest", "-c:v", "mpeg4", "-c:a", "aac", str(src)]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=120)
    t = FakeTranscriber("Waziri alisema shule zitafunguliwa Jumatatu.")
    res = video2text(str(src), FakeLLM(), t)
    assert t.paths and t.paths[0].endswith("audio.wav")
    assert not os.path.exists(t.paths[0])  # extracted audio is deleted after transcription
    assert "Spoken in the video (transcript):\nWaziri alisema" in res.text
    # speech and visuals stay separate sources
    assert res.extracted["transcript"].startswith("Waziri") and res.extracted["on_screen_text"] == "BREAKING: Kampala floods"


@pytest.mark.skipif(not find_ffmpeg(), reason="ffmpeg not installed")
def test_video_without_audio_track_says_so(tmp_path):
    t = FakeTranscriber()
    res = video2text(str(make_mp4(tmp_path / "silent.mp4")), FakeLLM(), t)
    assert not t.paths  # nothing to transcribe
    assert any("no audio track" in n for n in res.notes)
    assert "BREAKING: Kampala floods" in res.text  # on-screen text still extracted


def test_upload_limits_and_extensions(tmp_path):
    with pytest.raises(InputError, match="too large"):
        inputs.check_upload_size("image", 11 * 1024 * 1024)
    with pytest.raises(InputError, match="empty"):
        inputs.check_upload_size("audio", 0)
    with pytest.raises(InputError, match="Unsupported"):
        inputs.check_extension("image", "malware.exe")
    inputs.check_extension("video", "clip.MP4")


def test_demo_files_are_recognised():
    for path, expected in [("demo_data/image.webp", "image/webp"), ("demo_data/speech.mp3", "audio/mpeg"), ("demo_data/video.m4v", "video/mp4")]:
        with open(path, "rb") as f:
            assert sniff_type(f.read(16)) == expected, path
