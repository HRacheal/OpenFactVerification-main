"""Turn images, audio and video into text for the fact-checking pipeline.

- Images: a vision-capable LLM (OpenAI, Claude or Gemini) transcribes the visible text and briefly describes the image.
- Audio: local speech-to-text with faster-whisper (see transcription.py). No API key and no paid service.
- Video: OpenCV samples a small, fixed number of frames (never the whole video) for on-screen text, and
  ffmpeg extracts the audio track, which faster-whisper transcribes. Speech and visuals are separate sources.

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
    if header.startswith(b"fLaC"):
        return "audio/flac"
    if len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xF6) == 0xF0:
        return "audio/aac"  # ADTS frame
    if header.startswith(b"ID3") or (len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0):
        return "audio/mpeg"  # ID3 tag or MPEG audio frame sync
    if header.startswith(b"OggS"):
        return "audio/ogg"  # Ogg Vorbis / Opus
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
AUDIO_TYPES = ("audio/wav", "audio/mpeg", "audio/mp4", "audio/ogg", "audio/flac", "audio/aac", "video/webm", "video/mp4")


def find_ffmpeg() -> str | None:
    return os.environ.get("FFMPEG_PATH") or shutil.which("ffmpeg")


def media_duration(path: str) -> float | None:
    """Duration in seconds from the container metadata (PyAV), or None if unknown."""
    try:
        import av

        with av.open(path) as container:
            return container.duration / 1_000_000 if container.duration else None
    except Exception:
        return None


def transcription_unavailable_reason() -> str:
    from .transcription import enabled

    return enabled()[1] or "Speech transcription is not available."


def transcribe_to_result(path: str, transcriber, language: str = None, progress=None) -> MediaResult:
    """Run the transcriber on a decodable audio file and package the transcript, segments and limits."""
    from .transcription import TranscriptionError, format_timestamp

    def on_segment(done, total):
        if progress:
            progress(f"Transcribing speech: {format_timestamp(done)} of {format_timestamp(total)} processed")

    if progress:
        progress("Loading the speech model and transcribing (this can take a few minutes on a CPU)")
    try:
        t = transcriber.transcribe(path, language=language, progress=on_segment)
    except TranscriptionError as e:
        raise MediaError(str(e))
    if not t.text.strip():
        raise MediaError("No speech was detected in the recording. It may be silent, music only, or too quiet.")
    notes = ["The transcript was produced automatically by a speech-recognition model and may contain errors, "
             "especially in names, numbers and local languages. Review and correct it before relying on it."]
    if t.duration and t.transcribed_seconds + 1 < t.duration:
        notes.append(f"Only the first {format_timestamp(t.transcribed_seconds)} of a {format_timestamp(t.duration)} "
                     "recording was transcribed.")
    if language == "lg":
        notes.append("The speech model has no Luganda support, so Luganda speech was transcribed with automatic "
                     "language detection and is likely to contain many errors.")
    elif t.language_probability and t.language_probability < 0.6 and (language or "auto") == "auto":
        notes.append(f"The spoken language could not be identified with confidence (best guess: {t.language}). "
                     "Choosing the language on the form may improve the transcript.")
    extracted = {
        "transcript": t.text,
        "segments": t.segments,
        "spoken_language": t.language,
        "language_probability": t.language_probability,
        "duration_seconds": t.duration,
        "transcribed_seconds": t.transcribed_seconds,
        "speech_model": f"faster-whisper {t.model}",
    }
    return MediaResult(text=t.text, extracted=extracted, notes=notes)


def voice2text(path: str, transcriber, language: str = None, progress=None) -> MediaResult:
    """Transcribe an audio file. `transcriber` provides .transcribe(path, language=, progress=) -> Transcript."""
    with open(path, "rb") as f:
        header = f.read(16)
    if sniff_type(header) not in AUDIO_TYPES:
        raise MediaError("Unsupported audio format. Please upload MP3, WAV, M4A, OGG/Opus, FLAC, AAC or WebM audio.")
    if transcriber is None:
        raise MediaError(transcription_unavailable_reason())
    return transcribe_to_result(path, transcriber, language=language, progress=progress)


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


def _extract_audio(path: str, max_seconds: int, workdir: str) -> tuple[str | None, str]:
    """Extract the first audio track as 16 kHz mono WAV (the format Whisper uses internally).
    Returns (wav path or None, reason it failed)."""
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        return None, ("The audio track could not be extracted because FFmpeg is not installed "
                      "(install it or set FFMPEG_PATH).")
    out = os.path.join(workdir, "audio.wav")
    # Argument list, no shell. -map 0:a:0 fails cleanly when the video has no audio stream.
    cmd = [ffmpeg, "-nostdin", "-y", "-hide_banner", "-loglevel", "error", "-i", path, "-map", "0:a:0",
           "-t", str(max_seconds), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", out]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300, check=True)
    except subprocess.TimeoutExpired:
        return None, "Extracting the audio track took too long and was stopped."
    except (subprocess.SubprocessError, OSError):
        return None, "The video has no audio track, or its audio could not be read."
    if not os.path.exists(out) or os.path.getsize(out) <= 1024:
        return None, "The video has no usable audio track."
    return out, ""


def video2text(path: str, vision_client, transcriber=None, language: str = None, progress=None) -> MediaResult:
    with open(path, "rb") as f:
        header = f.read(16)
    if sniff_type(header) not in ("video/mp4", "video/quicktime", "video/webm"):
        raise MediaError("Unsupported video format. Please upload MP4, M4V, MOV or WebM.")

    notes, extracted = [], {}
    if progress:
        progress("Sampling video frames for on-screen text")
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
    else:
        notes.append("Video frames were not analysed because no vision-capable AI model is configured.")

    transcript = ""
    if transcriber is None:
        notes.append("Spoken content was not transcribed. " + transcription_unavailable_reason())
    else:
        # Temporary folder is deleted (with the extracted audio) as soon as transcription ends.
        with tempfile.TemporaryDirectory(prefix="truthlens_av_") as workdir:
            if progress:
                progress("Extracting the audio track with FFmpeg")
            audio, reason = _extract_audio(path, MAX_VIDEO_SECONDS, workdir)
            if audio is None:
                notes.append(reason)
            else:
                try:
                    speech = transcribe_to_result(audio, transcriber, language=language, progress=progress)
                    transcript = speech.text
                    extracted.update({k: v for k, v in speech.extracted.items() if k != "duration_seconds"})
                    notes.extend(speech.notes)
                except MediaError as e:
                    notes.append(f"The audio track could not be transcribed: {e}")

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
