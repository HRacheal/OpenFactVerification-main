"""Kweli Lens AI web application (Flask).

Flow: index form -> POST /check (validated, CSRF-protected) -> background job -> /job/<id> progress page
-> /report/<id>. Audio/video first go through a transcription job (local faster-whisper) -> /transcript/<id>,
where the user reviews/corrects the transcript and picks what to fact-check -> POST /check as text. Results are kept only in server memory for a limited time and are tied to the browser
session that created them (no shared response.json file).

Run:  python webapp.py            (then open http://127.0.0.1:2024)
"""

from __future__ import annotations

import argparse
import os
import secrets
import shutil
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from urllib.parse import urlparse

from flask import Flask, request, render_template, jsonify, session, redirect, url_for, abort, Response
from werkzeug.exceptions import RequestEntityTooLarge
import json

from factcheck.utils.api_config import load_api_config  # also loads .env
from factcheck.utils.llmclient import CLIENTS, LLMError, MissingCredentialsError
from factcheck.utils.llmclient.gemini_client import normalize_model as normalize_gemini_model
from factcheck.utils.logger import CustomLogger
from factcheck.utils.utils import load_yaml
from factcheck.core.Retriever import RetrievalError
from factcheck.core.Verdict import (
    VERDICT_LABELS, VERDICT_HEADLINES, SOURCE_WEIGHT, BASIS_WEIGHT, SNIPPET_ONLY_WEIGHT, RECENT_DAYS,
)
from factcheck.utils.sources import SOURCE_TYPE_LABELS
from factcheck import inputs
from factcheck.inputs import InputError, UPLOAD_LIMITS
from factcheck.utils import transcription

logger = CustomLogger(__name__).getlog()

COUNTRIES = ["Kenya", "Uganda", "Rwanda", "Tanzania", "Burundi", "South Sudan", "Ethiopia", "Somalia", "DR Congo"]
LANGUAGES = {"auto": "Detect automatically", "en": "English", "sw": "Swahili (Kiswahili)", "lg": "Luganda"}
OUTPUT_LANGUAGES = {"en": "English", "sw": "Swahili (Kiswahili)", "lg": "Luganda (experimental)"}
DEFAULT_MODELS = {"gpt": "gpt-4.1-mini", "claude": "claude-opus-5-5", "gemini": "gemini-3.8-flash", "local_openai": "vicuna-7b"}
# A TRUTHLENS_MODEL that doesn't match the provider (e.g. a leftover claude-* id) falls back to the default.
MODEL_PREFIXES = {"gpt": ("gpt", "chatgpt", "o1", "o3", "o4"), "claude": ("claude-",), "gemini": ("gemini-", "gemini/", "models/gemini-")}
LLM_KEY_NAMES = {"gpt": "OPENAI_API_KEY", "claude": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY", "local_openai": "LOCAL_API_URL"}
JOB_TTL_SECONDS = int(os.environ.get("TRUTHLENS_RESULT_TTL", "3600"))
MAX_JOBS = 100

app = Flask(__name__, static_folder="assets")
app.config.update(
    SECRET_KEY=os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32),
    MAX_CONTENT_LENGTH=max(limit for limit, _ in UPLOAD_LIMITS.values()) + 1024 * 1024,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)

# --------------------------------------------------------------------------- settings / pipeline
SETTINGS = {
    "provider": None,
    "model": None,
    "prompt": "truthlens_prompt",
    "retriever": "serper",
    "api_config": {},
}
_fc_lock = threading.Lock()
_fc_instance = None
_executor = ThreadPoolExecutor(max_workers=int(os.environ.get("TRUTHLENS_WORKERS", "2")))
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def resolve_provider() -> tuple[str, str]:
    cfg = load_api_config(SETTINGS["api_config"])
    provider = (SETTINGS["provider"] or os.environ.get("TRUTHLENS_PROVIDER") or "").strip().strip("\"'").lower()
    if not provider:
        if cfg.get("OPENAI_API_KEY"):
            provider = "gpt"
        elif cfg.get("ANTHROPIC_API_KEY"):
            provider = "claude"
        elif cfg.get("GEMINI_API_KEY"):
            provider = "gemini"
        else:
            provider = "gpt"
    provider = {"openai": "gpt", "anthropic": "claude", "google": "gemini", "local": "local_openai"}.get(provider, provider)
    if provider not in CLIENTS:
        raise MissingCredentialsError(f"Unknown TRUTHLENS_PROVIDER {provider!r}. Use openai, claude, gemini or local.")
    model = (SETTINGS["model"] or os.environ.get("TRUTHLENS_MODEL") or "").strip().strip("\"'")
    if not model or (provider in MODEL_PREFIXES and not model.lower().startswith(MODEL_PREFIXES[provider])):
        if model:
            logger.warning(f"TRUTHLENS_MODEL={model!r} does not belong to provider {provider!r}; "
                           f"using {DEFAULT_MODELS[provider]!r} instead.")
        model = DEFAULT_MODELS[provider]
    if provider == "gemini":
        model = normalize_gemini_model(model)  # strips gemini/ or models/ prefixes, replaces retired ids
    return provider, model


def get_factcheck():
    """Create the pipeline lazily so the site still loads (and explains) when keys are missing."""
    global _fc_instance
    with _fc_lock:
        if _fc_instance is None:
            from factcheck import FactCheck

            provider, model = resolve_provider()
            _fc_instance = FactCheck(
                default_model=model,
                client=provider,
                api_config=SETTINGS["api_config"],
                prompt=SETTINGS["prompt"],
                retriever=SETTINGS["retriever"],
            )
        return _fc_instance


def config_status() -> dict:
    cfg = load_api_config(SETTINGS["api_config"])
    try:
        provider, model = resolve_provider()
    except LLMError:
        provider, model = "unknown", ""
    key_name = LLM_KEY_NAMES.get(provider)
    from factcheck.utils.multimodal import find_ffmpeg

    transcription_ready, transcription_reason = transcription.enabled()
    return {
        "provider": provider,
        "model": model,
        "llm_ready": bool(key_name and cfg.get(key_name)),
        "llm_key_name": key_name,
        "search_ready": bool(cfg.get("SERPER_API_KEY")),
        "transcription_ready": transcription_ready,
        "transcription_reason": transcription_reason,
        "whisper_model": transcription.settings()["model"],
        "max_audio_seconds": transcription.settings()["max_seconds"],
        "vision_ready": provider in ("gpt", "claude", "gemini") and bool(key_name and cfg.get(key_name)),
        "ffmpeg_ready": bool(find_ffmpeg()),
    }


# --------------------------------------------------------------------------- jobs
def _purge_jobs():
    now = time.time()
    with _jobs_lock:
        for jid in [j for j, v in _jobs.items() if now - v["created"] > JOB_TTL_SECONDS]:
            _jobs.pop(jid, None)
        while len(_jobs) > MAX_JOBS:
            oldest = min(_jobs, key=lambda j: _jobs[j]["created"])
            _jobs.pop(oldest, None)


def _update(job_id, **fields):
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(fields)


def run_job(job_id: str, submission: dict, workdir: str | None):
    def progress(stage, message):
        _update(job_id, stage=stage, message=message)

    try:
        itype = submission["input_type"]
        progress("input", "Reading the submitted content")
        if itype == "text":
            media = inputs.prepare_text(submission["text"])
        elif itype == "url":
            progress("input", "Fetching and reading the web page")
            media = inputs.prepare_url(submission["url"])
        else:  # image (audio/video are transcribed first, then submitted as text)
            fc = get_factcheck()
            progress("input", "Reading text in the image")
            media = inputs.prepare_media(itype, submission["path"], vision_client=fc.vision_client)
        fc = get_factcheck()
        result = fc.check_text(
            media.text,
            country=submission["country"] or None,
            language=submission["language"],
            output_language=submission["output_language"],
            progress=progress,
        )
        source = submission.get("source")  # set when the text is a reviewed audio/video transcript
        result["input"] = {
            "type": source["type"] if source else itype,
            "filename": source["filename"] if source else submission.get("filename", ""),
            "extracted": source["extracted"] if source else media.extracted,
            "notes": source["notes"] if source else media.notes,
        }
        _update(job_id, state="done", stage="done", message="Report ready", result=result)
    except (InputError, RetrievalError, LLMError) as e:
        _update(job_id, state="error", error=str(e))
    except Exception:
        logger.error(f"Job {job_id} failed:\n{traceback.format_exc()}")
        _update(job_id, state="error", error="An unexpected error occurred while checking this content. See the server log for details.")
    finally:
        if workdir:
            shutil.rmtree(workdir, ignore_errors=True)  # uploaded files are never kept


def run_transcription_job(job_id: str, submission: dict, workdir: str):
    """Transcribe uploaded audio/video locally. The result waits for the user's review; nothing is fact-checked yet."""
    def progress(message):
        _update(job_id, stage="transcribe", message=message)

    try:
        itype = submission["input_type"]
        vision_client = None
        if itype == "video" and submission.get("vision_ready"):
            try:
                vision_client = get_factcheck().vision_client
            except LLMError:
                vision_client = None
        progress("Preparing the recording")
        media = inputs.prepare_media(itype, submission["path"], vision_client=vision_client,
                                     transcriber=transcription.get_transcriber(), language=submission["language"],
                                     progress=progress)
        _update(job_id, state="done", stage="done", message="Transcript ready for review", transcript={
            "type": itype,
            "filename": submission.get("filename", ""),
            "extracted": media.extracted,
            "notes": media.notes,
            "settings": {k: submission[k] for k in ("country", "language", "output_language")},
        })
    except InputError as e:
        _update(job_id, state="error", error=str(e))
    except Exception:
        logger.error(f"Transcription job {job_id} failed:\n{traceback.format_exc()}")
        _update(job_id, state="error", error="An unexpected error occurred while transcribing this file. See the server log for details.")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)  # the recording and any extracted audio are never kept


def get_job_or_404(job_id: str) -> dict:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job or not secrets.compare_digest(job["owner"], session.get("owner", "")):
        abort(404)
    return job


# --------------------------------------------------------------------------- security helpers
def csrf_token() -> str:
    if "csrf" not in session:
        session["csrf"] = secrets.token_urlsafe(32)
    return session["csrf"]


def ensure_owner() -> str:
    if "owner" not in session:
        session["owner"] = secrets.token_urlsafe(16)
    return session["owner"]


@app.after_request
def security_headers(resp):
    resp.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
        "object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
    )
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    if request.path.startswith(("/report", "/job", "/api")):
        resp.headers["Cache-Control"] = "no-store"
    return resp


def safe_href(url) -> str:
    """Only allow http(s) links in templates (blocks javascript: and data: URLs from search results)."""
    try:
        return url if isinstance(url, str) and urlparse(url).scheme in ("http", "https") else ""
    except ValueError:
        return ""


def fmt_date(value) -> str:
    if not value:
        return ""
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).strftime("%d %b %Y")
    except ValueError:
        return str(value)


app.jinja_env.filters["safe_href"] = safe_href
app.jinja_env.filters["fmt_date"] = fmt_date
app.jinja_env.globals.update(
    csrf_token=csrf_token,
    VERDICT_LABELS=VERDICT_LABELS,
    VERDICT_HEADLINES=VERDICT_HEADLINES,
    RECENT_DAYS=RECENT_DAYS,
    format_timestamp=transcription.format_timestamp,
    SOURCE_TYPE_LABELS=SOURCE_TYPE_LABELS,
    LANGUAGES=LANGUAGES,
    OUTPUT_LANGUAGES=OUTPUT_LANGUAGES,
    COUNTRIES=COUNTRIES,
    UPLOAD_LIMITS=UPLOAD_LIMITS,
    MAX_TEXT_CHARS=inputs.MAX_TEXT_CHARS,
)


# --------------------------------------------------------------------------- routes
@app.route("/", methods=["GET"])
def index():
    ensure_owner()
    return render_template("index.html", status=config_status(), form={}, error=None)


@app.route("/about")
def about():
    return render_template(
        "about.html", source_weight=SOURCE_WEIGHT, basis_weight=BASIS_WEIGHT, snippet_weight=SNIPPET_ONLY_WEIGHT,
        recent_days=RECENT_DAYS,
    )


def _form_error(message: str, form: dict, code: int = 400):
    return render_template("index.html", status=config_status(), form=form, error=message), code


@app.route("/check", methods=["POST"])
def check():
    owner = ensure_owner()
    form = {k: request.form.get(k, "") for k in ("input_type", "text", "url", "country", "language", "output_language")}
    transcript_id = request.form.get("transcript_id", "")
    token = request.form.get("csrf_token", "")
    if not token or not secrets.compare_digest(token, session.get("csrf", "")):
        return _form_error("Your session expired. Please submit the form again.", form, 400)

    itype = form["input_type"]
    if itype not in inputs.INPUT_TYPES:
        return _form_error("Please choose an input type.", form)
    country = form["country"] if form["country"] in COUNTRIES or form["country"] == "International" else ""
    language = form["language"] if form["language"] in LANGUAGES else "auto"
    output_language = form["output_language"] if form["output_language"] in OUTPUT_LANGUAGES else "en"

    status = config_status()
    media_upload = itype in ("audio", "video")
    if media_upload:
        # Transcription runs locally and needs no API key; the fact-check that follows does.
        if itype == "audio" and not status["transcription_ready"]:
            return _form_error(status["transcription_reason"], form, 503)
        if itype == "video" and not status["transcription_ready"] and not status["vision_ready"]:
            return _form_error(status["transcription_reason"], form, 503)
    else:
        if not status["llm_ready"]:
            return _form_error(f"The AI model is not configured: set {status['llm_key_name']} in your .env file and restart.", form, 503)
        if not status["search_ready"]:
            return _form_error("Evidence search is not configured: set SERPER_API_KEY in your .env file and restart.", form, 503)

    submission = {"input_type": itype, "country": country if country != "International" else "", "language": language,
                  "output_language": output_language, "vision_ready": status["vision_ready"]}
    workdir = None
    try:
        if itype == "text":
            text = form["text"]
            if transcript_id:
                source = _transcript_source(transcript_id)
                on_screen = source["extracted"].get("on_screen_text") or ""
                if request.form.get("include_onscreen") and on_screen:
                    text = (text.strip() + "\n\nText shown on screen in the video:\n" + on_screen).strip()
                if text.strip() != (source["extracted"].get("transcript") or "").strip():
                    source["notes"].append("You fact-checked an edited or selected part of the transcript.")
                submission["source"] = source
            submission["text"] = inputs.prepare_text(text).text
        elif itype == "url":
            from factcheck.utils.safe_fetch import validate_url, UnsafeURLError

            try:
                submission["url"] = validate_url(form["url"])
            except UnsafeURLError as e:
                raise InputError(str(e))
        else:
            # The form has one file field per media type; use the one that actually contains a file.
            upload = next((f for f in request.files.getlist("file") if f and f.filename), None)
            if upload is None:
                raise InputError(f"Please choose a {itype} file to upload.")
            inputs.check_extension(itype, upload.filename)
            workdir = tempfile.mkdtemp(prefix="truthlens_")
            # Never use the user's filename on disk.
            path = os.path.join(workdir, "upload" + os.path.splitext(upload.filename)[1].lower())
            upload.save(path)
            inputs.check_upload_size(itype, os.path.getsize(path))
            submission["path"] = path
            submission["filename"] = os.path.basename(upload.filename)[:120]
    except InputError as e:
        if workdir:
            shutil.rmtree(workdir, ignore_errors=True)
        return _form_error(str(e), form)

    _purge_jobs()
    job_id = secrets.token_urlsafe(16)
    kind = "transcribe" if media_upload else "check"
    with _jobs_lock:
        _jobs[job_id] = {"owner": owner, "created": time.time(), "state": "running", "stage": "queued", "kind": kind,
                         "message": "Waiting to start", "error": None, "result": None, "transcript": None}
    _executor.submit(run_transcription_job if media_upload else run_job, job_id, submission, workdir)
    return redirect(url_for("job_page", job_id=job_id), code=303)


def _transcript_source(transcript_id: str) -> dict:
    """The reviewed transcript's provenance (file name, transcript, on-screen text), for the report."""
    with _jobs_lock:
        job = _jobs.get(transcript_id)
    if (not job or job.get("kind") != "transcribe" or job["state"] != "done"
            or not secrets.compare_digest(job["owner"], session.get("owner", ""))):
        raise InputError("This transcript has expired. Please upload the recording again.")
    t = job["transcript"]
    return {"type": t["type"], "filename": t["filename"], "extracted": dict(t["extracted"]), "notes": list(t["notes"])}


@app.route("/job/<job_id>")
def job_page(job_id):
    job = get_job_or_404(job_id)
    if job["state"] == "done":
        return redirect(_done_url(job_id, job))
    return render_template("progress.html", job_id=job_id, job=job, next_url=_done_url(job_id, job))


def _done_url(job_id: str, job: dict) -> str:
    return url_for("transcript_page" if job.get("kind") == "transcribe" else "report", job_id=job_id)


@app.route("/api/job/<job_id>")
def job_status(job_id):
    job = get_job_or_404(job_id)
    return jsonify({**{k: job[k] for k in ("state", "stage", "message", "error")}, "next_url": _done_url(job_id, job)})


@app.route("/transcript/<job_id>")
def transcript_page(job_id):
    job = get_job_or_404(job_id)
    if job.get("kind") != "transcribe":
        abort(404)
    if job["state"] != "done":
        return redirect(url_for("job_page", job_id=job_id))
    return render_template("transcript.html", t=job["transcript"], job_id=job_id, status=config_status())


@app.route("/report/<job_id>")
def report(job_id):
    job = get_job_or_404(job_id)
    if job.get("kind") == "transcribe" or job["state"] != "done":
        return redirect(url_for("job_page", job_id=job_id))
    return render_template("report.html", r=job["result"], job_id=job_id)


@app.route("/report/<job_id>.json")
def report_json(job_id):
    job = get_job_or_404(job_id)
    if job["state"] != "done" or job.get("kind") == "transcribe":
        abort(404)
    body = json.dumps(job["result"], ensure_ascii=False, indent=2)
    return Response(body, mimetype="application/json",
                    headers={"Content-Disposition": f"attachment; filename=kweli-lens-report-{job_id[:8]}.json"})


@app.errorhandler(RequestEntityTooLarge)
def too_large(e):
    return _form_error("The uploaded file is too large.", {}, 413)


@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", message="This page or report was not found. Reports expire after an hour and can only be opened in the browser that created them."), 404


def main():
    parser = argparse.ArgumentParser(description="Kweli Lens AI web app")
    parser.add_argument("--model", type=str, default=None, help="Model name (default from TRUTHLENS_MODEL)")
    parser.add_argument("--client", type=str, default=None, choices=CLIENTS.keys(), help="Provider client")
    parser.add_argument("--prompt", type=str, default="truthlens_prompt")
    parser.add_argument("--retriever", type=str, default="serper")
    parser.add_argument("--api_config", type=str, default=None, help="Optional YAML file with API keys")
    parser.add_argument("--host", type=str, default=os.environ.get("TRUTHLENS_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("TRUTHLENS_PORT", "2024")))
    args = parser.parse_args()

    api_config = {}
    if args.api_config:
        try:
            api_config = load_yaml(args.api_config) or {}
        except Exception as e:
            print(f"Error loading api config: {e}")
    SETTINGS.update(provider=args.client, model=args.model, prompt=args.prompt, retriever=args.retriever, api_config=api_config)

    status = config_status()
    print(f"Kweli Lens AI | provider={status['provider']} model={status['model']} | "
          f"LLM key: {'OK' if status['llm_ready'] else 'MISSING'} | Serper key: {'OK' if status['search_ready'] else 'MISSING'}")
    speech = f"faster-whisper {status['whisper_model']}" if status["transcription_ready"] else "OFF - " + status["transcription_reason"]
    print(f"Speech transcription: {speech} | FFmpeg: {'found' if status['ffmpeg_ready'] else 'NOT FOUND (needed for video audio)'}")
    try:
        resolve_provider()
    except LLMError as e:
        print(f"Configuration problem: {e}")
    if status["llm_key_name"] and not status["llm_ready"]:
        print(f"Configuration problem: {status['llm_key_name']} is empty. Set it in .env and restart.")
    print(f"Open http://{args.host}:{args.port} in your browser.")
    # debug=False: the Werkzeug debugger allows remote code execution if exposed.
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
