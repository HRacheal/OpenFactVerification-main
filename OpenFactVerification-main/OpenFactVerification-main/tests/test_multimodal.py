"""Image, audio and video processing with locally generated media files. AI services are mocked;
OpenCV and (if installed) ffmpeg run for real."""

import math
import struct
import subprocess
import wave

import cv2
import numpy as np
import pytest

from factcheck.utils import multimodal
from factcheck.utils.multimodal import sniff_type, image2text, voice2text, video2text, MediaError, find_ffmpeg
from factcheck import inputs
from factcheck.inputs import InputError
from tests.conftest import FakeLLM


class FakeTranscriber:
    def __init__(self, text="Rais alisema bei ya mafuta itapanda."):
        self.text, self.paths = text, []

    def transcribe(self, path, model=None):
        self.paths.append(path)
        return self.text


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


def test_audio_transcription(tmp_path):
    t = FakeTranscriber()
    res = voice2text(str(make_wav(tmp_path / "a.wav")), t)
    assert res.text == t.text and res.extracted["transcript"] == t.text


def test_audio_needs_openai_key(tmp_path):
    with pytest.raises(MediaError, match="OPENAI_API_KEY"):
        voice2text(str(make_wav(tmp_path / "a.wav")), None)


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
    assert t.paths and t.paths[0].endswith("audio.mp3")
    assert "Spoken in the video (transcript):\nWaziri alisema" in res.text


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
