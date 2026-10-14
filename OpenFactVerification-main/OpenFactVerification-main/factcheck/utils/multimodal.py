"""Turn images, audio and video into text for the fact-checking pipeline.

- Images: a vision-capable LLM (OpenAI, Claude or Gemini) transcribes the visible text and briefly describes the image.
- Audio: OpenAI speech-to-text (Whisper by default). Requires OPENAI_API_KEY even when Claude is the main model.
- Video: OpenCV samples a small, fixed number of frames (never the whole video) for on-screen text, and
  ffmpeg extracts the audio track for transcription.

Extracted text is NOT evidence. It only tells us what the media says/shows; the claims in it still need
to be verified. Everything returned keeps transcript, on-screen text and description separate so they
are not treated as automatically consistent.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field

from .logger import CustomLogger
from .json_utils import parse_llm_json, expect_dict, LLMOutputError

logger = CustomLogger(__name__).getlog()

MAX_VIDEO_SECONDS = int(os.environ.get("TRUTHLENS_MAX_VIDEO_SECONDS", "600"))
VIDEO_FRAME_SAMPLES = int(os.environ.get("TRUTHLENS_VIDEO_FRAMES", "8"))
VISION_MAX_SIDE = 1568
VISION_MAX_BYTES = 4 * 1024 * 1024


class MediaError(RuntimeError):
    """Media could not be processed. Message is safe to show users."""


@dataclass
class MediaResult:
    text: str  # text passed to claim extraction
    extracted: dict = field(default_factory=dict)  # fields shown to the journalist (transcript, visible text ...)
    notes: list = field(default_factory=list)  # limitations


# --------------------------------------------------------------------------- file type sniffing
def sniff_type(header: bytes) -> str | None:
    """Identify a file from its first bytes (do not trust the file extension)."""
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "image/webp"
    if header[:4] == b"RIFF" and header[8:12] == b"WAVE":
        return "audio/wav"
    if header.startswith(b"ID3") or header[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "audio/mpeg"
    if header.startswith(b"OggS"):
        return "audio/ogg"
    if header[4:8] == b"ftyp":
        brand = header[8:12]
        if brand in (b"M4A ", b"M4B "):
            return "audio/mp4"
        if brand == b"qt  ":
            return "video/quicktime"
        return "video/mp4"  # isom, mp41, mp42, M4V, avc1 ...
    if header.startswith(b"\x1a\x45\xdf\xa3"):
        return "video/webm"
    return None


# --------------------------------------------------------------------------- images
IMAGE_PROMPT = """You are helping a fact-checker read an image (often a screenshot of a social media post).
1. "visible_text": transcribe ALL readable text exactly as written, in its original language, keeping
   names, numbers, dates and spelling unchanged. Keep line breaks between separate posts/captions. Do not translate.
2. "description": in one or two neutral sentences, describe what the image shows (e.g. "Screenshot of a
   Facebook post by an account named X with a photo of a crowd"). Do not identify private people by face.
3. "unclear_parts": list any text that is blurry, cut off or uncertain.
Do NOT say whether anything in the image is true. Return JSON only with these three keys."""


def _prepare_image(data: bytes, mime: str) -> tuple[bytes, str]:
    """Downscale/re-encode large images so they fit provider limits."""
    import cv2
    import numpy as np

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise MediaError("The image could not be decoded. It may be corrupt.")
    h, w = img.shape[:2]
    if max(h, w) <= VISION_MAX_SIDE and len(data) <= VISION_MAX_BYTES and mime in ("image/png", "image/jpeg"):
        return data, mime
    scale = min(1.0, VISION_MAX_SIDE / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise MediaError("The image could not be re-encoded.")
    return buf.tobytes(), "image/jpeg"


def image2text(path: str, vision_client) -> MediaResult:
    with open(path, "rb") as f:
        data = f.read()
    mime = sniff_type(data[:16])
    if mime not in ("image/png", "image/jpeg", "image/webp"):
        raise MediaError("Unsupported image format. Please upload PNG, JPEG or WebP.")
    data, mime = _prepare_image(data, mime)
    raw = vision_client.vision(IMAGE_PROMPT, [(data, mime)])
    try:
        info = expect_dict(parse_llm_json(raw))
    except LLMOutputError:
        raise MediaError("The image analysis returned an unreadable answer. Please try again.")
    visible = str(info.get("visible_text") or "").strip()
    description = str(info.get("description") or "").strip()
    unclear = info.get("unclear_parts") or []
    unclear = [str(u) for u in unclear] if isinstance(unclear, list) else [str(unclear)]
    notes = ["Text was read from the image by AI; check it against the original image before relying on it."]
    if unclear:
        notes.append("Some text in the image was unclear: " + "; ".join(unclear[:5]))
    if visible:
        text = visible
    elif description:
        text = description
        notes.append("No readable text was found; claims were extracted from an AI description of the image.")
    else:
        raise MediaError("No text or describable content could be extracted from the image.")
    return MediaResult(text=text, extracted={"visible_text": visible, "description": description}, notes=notes)


# --------------------------------------------------------------------------- audio
def find_ffmpeg() -> str | None:
    return os.environ.get("FFMPEG_PATH") or shutil.which("ffmpeg")


def voice2text(path: str, transcriber, model: str = None) -> MediaResult:
    """Transcribe an audio file. `transcriber` must provide .transcribe(path, model=...)."""
    with open(path, "rb") as f:
        header = f.read(16)
    if sniff_type(header) not in ("audio/wav", "audio/mpeg", "audio/mp4", "audio/ogg", "video/webm"):
        raise MediaError("Unsupported audio format. Please upload WAV, MP3, M4A or OGG.")
    if transcriber is None:
        raise MediaError("Audio transcription needs OPENAI_API_KEY (OpenAI speech-to-text). Add it to your .env file.")
    model = model or os.environ.get("TRUTHLENS_TRANSCRIBE_MODEL", "whisper-1")
    transcript = (transcriber.transcribe(path, model=model) or "").strip()
    if not transcript:
        raise MediaError("No speech could be transcribed from the audio.")
    return MediaResult(
        text=transcript,
        extracted={"transcript": transcript},
        notes=["The transcript was produced automatically and may contain errors, especially for names and for Luganda."],
    )


# --------------------------------------------------------------------------- video
VIDEO_PROMPT = """These are {n} frames sampled at roughly even intervals from a video, in order (timestamps: {stamps}).
For a fact-checker:
1. "on_screen_text": transcribe captions, headlines, banners and other readable text exactly, in the original
   language, without translating. Merge repeated text.
2. "description": one or two neutral sentences describing what the frames show.
Do NOT say whether anything is true. Return JSON only with keys "on_screen_text" and "description"."""


def _sample_frames(path: str, max_seconds: int, n_frames: int):
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise MediaError("The video could not be opened. It may be corrupt or use an unsupported codec.")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = total / fps if fps > 0 and total > 0 else 0.0
        analysed = min(duration, max_seconds) if duration else 0.0
        frames, stamps = [], []
        if analysed > 0:
            positions = [analysed * (i + 0.5) / n_frames for i in range(n_frames)]
            for t in positions:
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ok, frame = cap.read()
                if not ok:
                    continue
                h, w = frame.shape[:2]
                scale = min(1.0, 768 / max(h, w))
                if scale < 1.0:
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    frames.append((buf.tobytes(), "image/jpeg"))
                    stamps.append(f"{int(t // 60)}:{int(t % 60):02d}")
        return frames, stamps, duration, analysed
    finally:
        cap.release()


def _extract_audio(path: str, max_seconds: int, workdir: str) -> str | None:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        return None
    out = os.path.join(workdir, "audio.mp3")
    cmd = [ffmpeg, "-nostdin", "-y", "-i", path, "-t", str(max_seconds), "-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k", out]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180, check=True)
    except (subprocess.SubprocessError, OSError):
        return None
    return out if os.path.exists(out) and os.path.getsize(out) > 1024 else None


def video2text(path: str, vision_client, transcriber=None) -> MediaResult:
    with open(path, "rb") as f:
        header = f.read(16)
    if sniff_type(header) not in ("video/mp4", "video/quicktime", "video/webm"):
        raise MediaError("Unsupported video format. Please upload MP4, M4V, MOV or WebM.")

    notes, extracted = [], {}
    frames, stamps, duration, analysed = _sample_frames(path, MAX_VIDEO_SECONDS, VIDEO_FRAME_SAMPLES)
    if duration and duration > MAX_VIDEO_SECONDS:
        notes.append(
            f"Only the first {MAX_VIDEO_SECONDS // 60} minutes of a {duration / 60:.0f}-minute video were analysed."
        )
    extracted["duration_seconds"] = round(duration, 1)

    on_screen, description = "", ""
    if frames and vision_client is not None:
        try:
            raw = vision_client.vision(VIDEO_PROMPT.format(n=len(frames), stamps=", ".join(stamps)), frames)
            info = expect_dict(parse_llm_json(raw))
            on_screen = str(info.get("on_screen_text") or "").strip()
            description = str(info.get("description") or "").strip()
            notes.append(
                f"Visual content was judged from {len(frames)} sampled frames only; text shown briefly between them may be missed."
            )
        except Exception as e:
            logger.warning(f"Video frame analysis failed: {type(e).__name__}")
            notes.append("The video frames could not be analysed.")
    elif not frames:
        notes.append("No frames could be read from the video.")

    transcript = ""
    with tempfile.TemporaryDirectory(prefix="truthlens_av_") as workdir:
        audio = _extract_audio(path, MAX_VIDEO_SECONDS, workdir)
        if audio is None:
            notes.append("The audio track could not be extracted (ffmpeg missing, or the video has no audio).")
        elif transcriber is None:
            notes.append("Spoken content was not transcribed: OPENAI_API_KEY is required for speech-to-text.")
        else:
            try:
                transcript = (transcriber.transcribe(audio, model=os.environ.get("TRUTHLENS_TRANSCRIBE_MODEL", "whisper-1")) or "").strip()
                notes.append("The transcript was produced automatically and may contain errors.")
            except Exception as e:
                logger.warning(f"Video transcription failed: {type(e).__name__}")
                notes.append("The audio track could not be transcribed.")

    extracted.update({"transcript": transcript, "on_screen_text": on_screen, "description": description})
    sections = []
    if transcript:
        sections.append(f"Spoken in the video (transcript):\n{transcript}")
    if on_screen:
        sections.append(f"Text shown on screen in the video:\n{on_screen}")
    if not sections:
        raise MediaError("Neither speech nor on-screen text could be extracted from the video. " + " ".join(notes))
    notes.append("Spoken content and on-screen text are listed separately; they may not agree with each other.")
    return MediaResult(text="\n\n".join(sections), extracted=extracted, notes=notes)


def modal_normalization(modal="text", input=None, vision_client=None, transcriber=None) -> str:
    """Backward-compatible CLI helper: returns the text for a given modality."""
    logger.info(f"== Processing: Modal: {modal}")
    if modal == "string":
        return str(input)
    if modal == "text":
        with open(input, "r", encoding="utf-8") as f:
            return f.read()
    if modal == "speech":
        return voice2text(input, transcriber).text
    if modal == "image":
        return image2text(input, vision_client).text
    if modal == "video":
        return video2text(input, vision_client, transcriber).text
    raise NotImplementedError(f"Unknown modal {modal}")
