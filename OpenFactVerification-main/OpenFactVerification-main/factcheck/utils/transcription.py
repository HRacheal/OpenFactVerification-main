"""Local speech-to-text with faster-whisper (open source, no API key, no paid service).

The Whisper model is downloaded once from the Hugging Face Hub on first use (internet needed for that
download only) and cached on disk. After that, audio never leaves this machine.

Configuration (environment variables, all optional):
  TRUTHLENS_TRANSCRIPTION        on | off | auto (default auto: on, except on Render where it is off unless set)
  TRUTHLENS_WHISPER_MODEL        tiny | base | small | medium | large-v3 ... (default base)
  TRUTHLENS_WHISPER_DEVICE       cpu | cuda | auto (default cpu)
  TRUTHLENS_WHISPER_COMPUTE_TYPE int8 | int8_float16 | float16 | float32 (default int8, good for CPU)
  TRUTHLENS_WHISPER_CPU_THREADS  number of CPU threads (default 0 = library default)
  TRUTHLENS_WHISPER_BEAM_SIZE    beam size (default 5; 1 is faster but slightly less accurate)
  TRUTHLENS_WHISPER_CACHE_DIR    folder for downloaded models (default: the Hugging Face cache)
  TRUTHLENS_WHISPER_LOCAL_ONLY   1 = never download; use only a model already in the cache
  TRUTHLENS_MAX_AUDIO_SECONDS    only the first N seconds are transcribed (default 600)

Pre-download the model (e.g. before going offline):  python -m factcheck.utils.transcription --download
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import dataclass, field

from .logger import CustomLogger

logger = CustomLogger(__name__).getlog()

SAMPLE_RATE = 16000
KNOWN_MODEL_SIZES = ("tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en",
                     "large-v1", "large-v2", "large-v3", "large-v3-turbo", "turbo", "distil-large-v3")
# Whisper has no Luganda model; Luganda audio is transcribed with automatic language detection.
WHISPER_LANGUAGES = {"en": "en", "sw": "sw"}


class TranscriptionError(RuntimeError):
    """Transcription failed. Message is safe to show users (no paths or internals)."""


@dataclass
class Transcript:
    text: str
    segments: list = field(default_factory=list)  # [{"start": float, "end": float, "text": str}]
    language: str = ""
    language_probability: float = 0.0
    duration: float = 0.0  # seconds of audio in the file
    transcribed_seconds: float = 0.0  # seconds actually transcribed (may be capped)
    model: str = ""


def _env(name: str, default: str) -> str:
    return (os.environ.get(name) or default).strip().strip("\"'")


def settings() -> dict:
    return {
        "model": _env("TRUTHLENS_WHISPER_MODEL", "base"),
        "device": _env("TRUTHLENS_WHISPER_DEVICE", "cpu"),
        "compute_type": _env("TRUTHLENS_WHISPER_COMPUTE_TYPE", "int8"),
        "cpu_threads": int(_env("TRUTHLENS_WHISPER_CPU_THREADS", "0")),
        "beam_size": max(1, int(_env("TRUTHLENS_WHISPER_BEAM_SIZE", "5"))),
        "cache_dir": _env("TRUTHLENS_WHISPER_CACHE_DIR", "") or None,
        "local_only": _env("TRUTHLENS_WHISPER_LOCAL_ONLY", "0") in ("1", "true", "yes"),
        "max_seconds": max(1, int(_env("TRUTHLENS_MAX_AUDIO_SECONDS", "600"))),
    }


def library_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
    except Exception:
        return False
    return True


def enabled() -> tuple[bool, str]:
    """(is transcription enabled, reason shown to users when it is not)."""
    mode = _env("TRUTHLENS_TRANSCRIPTION", "auto").lower()
    if mode in ("off", "0", "false", "no"):
        return False, "Audio/video transcription is turned off on this server (TRUTHLENS_TRANSCRIPTION=off)."
    if mode == "auto" and os.environ.get("RENDER"):
        # Render free instances (512 MB RAM, 0.1 CPU, no persistent disk) cannot run Whisper reliably.
        return False, ("Audio/video transcription is turned off on this hosted server because it runs speech "
                       "recognition locally and needs more memory and CPU than the server has. Run Kweli Lens "
                       "on your own computer to transcribe recordings, or paste the transcript as text.")
    if not library_available():
        return False, "Audio/video transcription needs the faster-whisper package: pip install faster-whisper"
    return True, ""


class WhisperTranscriber:
    """Loads the Whisper model once (lazily, thread-safe) and transcribes one file at a time,
    so several uploads cannot multiply CPU and memory use."""

    def __init__(self, config: dict | None = None):
        self.config = config or settings()
        self._model = None
        self._load_lock = threading.Lock()
        self._run_lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self.config["model"]

    def load(self):
        with self._load_lock:
            if self._model is not None:
                return self._model
            try:
                from faster_whisper import WhisperModel
            except Exception:
                raise TranscriptionError("Transcription needs the faster-whisper package: pip install faster-whisper")
            c = self.config
            logger.info(f"Loading Whisper model {c['model']!r} on {c['device']} ({c['compute_type']})")
            started = time.time()
            try:
                self._model = WhisperModel(
                    c["model"],
                    device=c["device"],
                    compute_type=c["compute_type"],
                    cpu_threads=c["cpu_threads"],
                    download_root=c["cache_dir"],
                    local_files_only=c["local_only"],
                )
            except Exception as e:
                logger.error(f"Whisper model load failed: {type(e).__name__}: {e}")
                if c["model"] not in KNOWN_MODEL_SIZES and not os.path.isdir(c["model"]):
                    raise TranscriptionError(
                        f"Unknown speech model {c['model']!r}. Set TRUTHLENS_WHISPER_MODEL to tiny, base, small or medium.")
                if c["local_only"]:
                    raise TranscriptionError(
                        "The speech-to-text model is not downloaded yet and downloads are disabled "
                        "(TRUTHLENS_WHISPER_LOCAL_ONLY=1). Run: python -m factcheck.utils.transcription --download")
                raise TranscriptionError(
                    "The speech-to-text model could not be loaded. The first use downloads it once "
                    "(about 75 MB for 'tiny', 145 MB for 'base', 480 MB for 'small'), which needs internet access "
                    "and free disk space. Check the connection and try again.")
            logger.info(f"Whisper model ready in {time.time() - started:.1f}s")
            return self._model

    @staticmethod
    def decode(path: str, max_seconds: int):
        """Decode an audio file to 16 kHz mono float32, at most `max_seconds` long.

        Uses the ffmpeg program when available (stops reading at the limit, so long files are not loaded
        whole into memory); otherwise PyAV, which faster-whisper installs. Returns (audio, file_duration)."""
        import numpy as np

        from .multimodal import find_ffmpeg, media_duration

        duration = media_duration(path)
        ffmpeg = find_ffmpeg()
        if ffmpeg:
            # Argument list, no shell: the path is never interpreted by a shell.
            cmd = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-i", path, "-t", str(max_seconds),
                   "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-acodec", "pcm_s16le", "pipe:1"]
            try:
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=300, check=True)
            except (subprocess.SubprocessError, OSError) as e:
                logger.warning(f"ffmpeg audio decoding failed: {type(e).__name__}")
                raise TranscriptionError("The audio could not be decoded. The file may be corrupt, empty or in an unsupported format.")
            audio = np.frombuffer(proc.stdout, np.int16).astype(np.float32) / 32768.0
        else:
            from faster_whisper import decode_audio

            try:
                audio = decode_audio(path, sampling_rate=SAMPLE_RATE)[: int(max_seconds * SAMPLE_RATE)]
            except Exception as e:
                logger.warning(f"Audio decoding failed: {type(e).__name__}")
                raise TranscriptionError("The audio could not be decoded. The file may be corrupt, empty or in an unsupported format.")
        return audio, max(duration or 0.0, len(audio) / SAMPLE_RATE)

    def transcribe(self, path: str, language: str | None = None, progress=None) -> Transcript:
        """Transcribe `path`. `language` is "en", "sw", "lg" or None/"auto".
        `progress(done_seconds, total_seconds)` is called as segments are produced."""
        audio, duration = self.decode(path, self.config["max_seconds"])
        total = len(audio) / SAMPLE_RATE
        if total < 0.3:
            raise TranscriptionError("The recording is empty or too short to contain speech.")

        model = self.load()
        with self._run_lock:
            try:
                segments_iter, info = model.transcribe(
                    audio,
                    language=WHISPER_LANGUAGES.get(language or ""),
                    beam_size=self.config["beam_size"],
                    vad_filter=True,  # skip silence; reduces made-up text on silent parts
                    condition_on_previous_text=False,  # limits repetition loops on long files
                )
                segments = []
                for s in segments_iter:  # transcription happens lazily while iterating
                    text = (s.text or "").strip()
                    if text:
                        segments.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": text})
                    if progress:
                        progress(min(s.end, total), total)
            except TranscriptionError:
                raise
            except Exception as e:
                logger.error(f"Whisper transcription failed: {type(e).__name__}: {e}")
                raise TranscriptionError("Speech transcription failed while processing this file. Please try another file.")

        text = " ".join(s["text"] for s in segments).strip()
        return Transcript(
            text=text,
            segments=segments,
            language=getattr(info, "language", "") or "",
            language_probability=round(float(getattr(info, "language_probability", 0.0) or 0.0), 3),
            duration=round(duration, 1),
            transcribed_seconds=round(total, 1),
            model=self.model_name,
        )


_shared: WhisperTranscriber | None = None
_shared_lock = threading.Lock()


def get_transcriber() -> WhisperTranscriber | None:
    """The process-wide transcriber, or None when transcription is disabled/unavailable."""
    global _shared
    ok, _ = enabled()
    if not ok:
        return None
    with _shared_lock:
        if _shared is None:
            _shared = WhisperTranscriber()
        return _shared


def format_timestamp(seconds: float) -> str:
    seconds = int(seconds or 0)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Kweli Lens local speech-to-text (faster-whisper)")
    parser.add_argument("--download", action="store_true", help="download/cache the configured model and exit")
    parser.add_argument("file", nargs="?", help="audio file to transcribe")
    parser.add_argument("--language", default=None, choices=["en", "sw", "lg", "auto"])
    args = parser.parse_args()
    t = WhisperTranscriber()
    if args.download or not args.file:
        t.load()
        print(f"Model {t.model_name!r} is ready.")
        return
    started = time.time()
    result = t.transcribe(args.file, language=args.language)
    print(f"[{result.language} p={result.language_probability}] {result.transcribed_seconds}s of audio "
          f"in {time.time() - started:.1f}s with model {result.model}")
    for s in result.segments:
        print(f"[{format_timestamp(s['start'])}-{format_timestamp(s['end'])}] {s['text']}")


if __name__ == "__main__":
    main()
