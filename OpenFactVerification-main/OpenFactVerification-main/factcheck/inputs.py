"""Shared input handling: turn a submission (text, URL, image, audio, video) into text for the pipeline.

Used by both the web app (webapp.py) and the command line (python -m factcheck).
"""

from __future__ import annotations

import os

from factcheck.utils.multimodal import MediaResult, MediaError, image2text, voice2text, video2text
from factcheck.utils.safe_fetch import fetch_url, extract_readable_text, UnsafeURLError, FetchError
from factcheck.utils.sources import extract_page_metadata, publisher_for, get_domain

MAX_TEXT_CHARS = 20000
MAX_ARTICLE_CHARS = 12000

INPUT_TYPES = ("text", "url", "image", "audio", "video")

# Upload limits in bytes and the file extensions shown to users.
UPLOAD_LIMITS = {
    "image": (10 * 1024 * 1024, ("png", "jpg", "jpeg", "webp")),
    "audio": (25 * 1024 * 1024, ("wav", "mp3", "m4a", "ogg")),  # 25 MB = OpenAI transcription limit
    "video": (100 * 1024 * 1024, ("mp4", "m4v", "mov", "webm")),
}


class InputError(ValueError):
    """The submission is invalid. Message is safe to show users."""


def prepare_text(text: str) -> MediaResult:
    text = (text or "").strip()
    if not text:
        raise InputError("Please enter some text to check.")
    if len(text) > MAX_TEXT_CHARS:
        raise InputError(f"The text is too long ({len(text)} characters). The limit is {MAX_TEXT_CHARS}.")
    return MediaResult(text=text)


def prepare_url(url: str) -> MediaResult:
    try:
        page = fetch_url(url)
    except (UnsafeURLError, FetchError) as e:
        raise InputError(str(e))
    article = extract_readable_text(page.html, max_chars=MAX_ARTICLE_CHARS + 1)
    if len(article) < 200:
        raise InputError(
            "Readable article text could not be extracted from this page. It may require a login or "
            "subscription, or load its content with JavaScript. Try pasting the article text instead."
        )
    meta = extract_page_metadata(page.html)
    notes = []
    if len(article) > MAX_ARTICLE_CHARS:
        article = article[:MAX_ARTICLE_CHARS]
        notes.append(f"The article is long; only the first {MAX_ARTICLE_CHARS} characters were checked.")
    if not meta["published_date"]:
        notes.append("The page did not state a publication date in its metadata.")
    extracted = {
        "url": url,
        "final_url": page.final_url,
        "domain": get_domain(page.final_url),
        "title": meta["title"],
        "publisher": publisher_for(page.final_url, meta["site_name"]),
        "published_date": meta["published_date"],
        "article_text": article,
    }
    return MediaResult(text=article, extracted=extracted, notes=notes)


def check_upload_size(input_type: str, size: int) -> None:
    limit, exts = UPLOAD_LIMITS[input_type]
    if size <= 0:
        raise InputError("The uploaded file is empty.")
    if size > limit:
        raise InputError(f"The file is too large. The limit for {input_type} files is {limit // (1024 * 1024)} MB.")


def check_extension(input_type: str, filename: str) -> None:
    _, exts = UPLOAD_LIMITS[input_type]
    ext = os.path.splitext(filename or "")[1].lower().lstrip(".")
    if ext not in exts:
        raise InputError(f"Unsupported {input_type} file type. Allowed: {', '.join(exts).upper()}.")


def prepare_media(input_type: str, path: str, vision_client=None, transcriber=None) -> MediaResult:
    check_upload_size(input_type, os.path.getsize(path))
    try:
        if input_type == "image":
            if vision_client is None:
                raise MediaError("Image analysis needs a vision-capable AI model to be configured.")
            return image2text(path, vision_client)
        if input_type == "audio":
            return voice2text(path, transcriber)
        if input_type == "video":
            return video2text(path, vision_client, transcriber)
    except MediaError as e:
        raise InputError(str(e))
    raise InputError(f"Unknown input type {input_type!r}.")
