import os
import re
import threading

from google import genai
from google.genai import errors, types
from .base import BaseClient, LLMProviderError, logger

SYSTEM_PROMPT = (
    "You are a careful fact-checking assistant. Respond with a single valid JSON object only - "
    "no markdown fences and no text before or after the JSON."
)

# HTTP codes that will never succeed on retry (bad request, bad key, no access, unknown model).
_NON_RETRYABLE_CODES = (400, 401, 403, 404)

# Google's current stable Flash model (gemini-2.5-flash now returns 404 "no longer available to new users").
DEFAULT_MODEL = "gemini-3.8-flash"
# The SDK does not retry unless retry_options is set. Google recommends exponential backoff for 408/429/5xx
# (e.g. 503 "model is currently experiencing high demand"), which the SDK implements with jitter.
DEFAULT_RETRY_ATTEMPTS = 5  # including the first call; override with TRUTHLENS_GEMINI_RETRIES
# "Please retry in 7h37m50.157s." in 429 messages; the fraction is dropped for display.
_RETRY_IN = re.compile(r"retry in ((?:\d+h)?(?:\d+m)?\d+(?:\.\d+)?s|(?:\d+h)?\d+m)", re.I)
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
# Prefixes used by other wrappers (LiteLLM "gemini/", REST "models/") that the google-genai SDK must not receive.
_FOREIGN_PREFIXES = ("gemini/", "google/", "models/")
# Model families Google has shut down; requests for them return 404 NOT_FOUND.
_RETIRED = re.compile(r"^gemini-(1\.0|1\.5)(-|$)|^gemini-pro(-vision)?$")
# Stable "gemini-<version>-flash" ids, used to pick a replacement when the configured model 404s.
_STABLE_FLASH = re.compile(r"^gemini-(\d+(?:\.\d+)?)-flash$")


def normalize_model(name: str) -> str:
    """Turn whatever TRUTHLENS_MODEL holds into a model id the Gemini API accepts."""
    model = (name or "").strip().strip("\"'")
    for prefix in _FOREIGN_PREFIXES:
        if model.lower().startswith(prefix):
            model = model[len(prefix):]
    if not model:
        return DEFAULT_MODEL
    if _RETIRED.match(model.lower()):
        logger.warning(f"Gemini model {model!r} has been retired by Google; using {DEFAULT_MODEL!r} instead.")
        return DEFAULT_MODEL
    return model


def describe_api_error(e: errors.APIError, secret: str = None) -> str:
    """'HTTP 503 UNAVAILABLE: <message>' for logs, with the API key removed and the length capped."""
    message = " ".join(str(getattr(e, "message", "") or "").split())
    if secret:
        message = message.replace(secret, "***")
    return f"HTTP {getattr(e, 'code', '?')} {getattr(e, 'status', '') or ''}: {message[:600]}".strip()


def _retry_attempts() -> int:
    try:
        return max(1, int(os.environ.get("TRUTHLENS_GEMINI_RETRIES", DEFAULT_RETRY_ATTEMPTS)))
    except ValueError:
        return DEFAULT_RETRY_ATTEMPTS


class GeminiClient(BaseClient):
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_config: dict = None,
        max_requests_per_minute=15,  # free-tier limits are low; keep concurrency modest
        request_window=60,
        timeout: float = 120.0,
    ):
        super().__init__(normalize_model(model), api_config, max_requests_per_minute, request_window)
        self._model_lock = threading.Lock()
        self._model_checked = False  # set once a 404 has triggered a lookup of the models this key can use
        key = self.require_key(self.api_config, "GEMINI_API_KEY", "Google Gemini")
        self._key = key  # only used to scrub error messages before logging
        self.retry_attempts = _retry_attempts()
        # The SDK timeout is in milliseconds. The SDK retries 408/429/5xx with exponential backoff and jitter;
        # callers such as Decompose ask BaseClient for a single attempt, so this is the only retry layer.
        retry = types.HttpRetryOptions(attempts=self.retry_attempts, initial_delay=2.0, max_delay=30.0)
        self.client = genai.Client(
            api_key=key, http_options=types.HttpOptions(timeout=int(timeout * 1000), retry_options=retry)
        )

    def _generate(self, contents, seed: int = None) -> str:
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            seed=seed,
        )
        try:
            response = self.client.models.generate_content(model=self.model, contents=contents, config=config)
        except errors.APIError as e:
            code = getattr(e, "code", None)
            if code != 404:
                logger.error(f"Gemini request failed (model={self.model!r}): {describe_api_error(e, self._key)}")
            if code == 404:
                requested = self.model
                logger.error(f"Gemini 404 NOT_FOUND: POST {API_URL.format(model=requested)} (model={requested!r}): "
                             f"{getattr(e, 'message', '')}")
                if self._switch_to_available_model(requested):
                    return self._generate(contents, seed)
                raise LLMProviderError(
                    f"Gemini model {requested!r} was not found (HTTP 404). Set TRUTHLENS_MODEL to a model your key "
                    f"can use, e.g. {DEFAULT_MODEL}."
                ) from e
            if code in _NON_RETRYABLE_CODES:
                hint = ""
                if "api key" in str(getattr(e, "message", "")).lower() or code in (401, 403):
                    hint = " Check that GEMINI_API_KEY in your .env is a valid key from https://aistudio.google.com/apikey."
                raise LLMProviderError(f"Gemini rejected the request (HTTP {code}: {e.status}).{hint}") from e
            # 429 / 5xx: the SDK has already retried with backoff, so report what Google said.
            if code == 429:
                # e.g. "...free_tier_requests, limit: 20, model: gemini-3.8-flash Please retry in 7h37m50.15s."
                wait = _RETRY_IN.search(str(getattr(e, "message", "")))
                delay = re.sub(r"\.\d+", "", wait.group(1)) if wait else ""
                when = f"Google says to retry in about {delay}." if wait else "Wait a minute and try again."
                raise LLMProviderError(
                    f"Gemini rate limit or free-tier quota reached (HTTP 429: {e.status}). {when} "
                    f"Check your limits at https://aistudio.google.com/rate-limit (daily quotas reset at midnight "
                    f"Pacific time)."
                ) from e
            if isinstance(code, int) and code >= 500:
                raise LLMProviderError(
                    f"Gemini is temporarily unavailable (HTTP {code}: {e.status}) after {self.retry_attempts} attempts. "
                    f"This is usually a short-lived overload on Google's side; please try again shortly."
                ) from e
            raise

        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            self._log_usage(usage.prompt_token_count, usage.candidates_token_count)

        feedback = getattr(response, "prompt_feedback", None)
        if feedback is not None and getattr(feedback, "block_reason", None):
            raise LLMProviderError("Gemini declined to answer this request (prompt blocked).")
        candidates = getattr(response, "candidates", None) or []
        if candidates and str(getattr(candidates[0], "finish_reason", "")).endswith(("SAFETY", "PROHIBITED_CONTENT")):
            raise LLMProviderError("Gemini declined to answer this request (safety filter).")
        return response.text or ""

    def _switch_to_available_model(self, failed: str) -> bool:
        """After a 404, ask the API which models this key can use and switch to the newest stable flash model.
        Done at most once per client; returns True if self.model changed and the request should be resent."""
        with self._model_lock:
            if self.model != failed:  # another concurrent call already switched
                return True
            if self._model_checked:
                return False
            self._model_checked = True
            try:
                available = [
                    m.name.removeprefix("models/") for m in self.client.models.list()
                    if "generateContent" in (m.supported_actions or [])
                ]
            except Exception as e:
                logger.warning(f"Could not list Gemini models ({type(e).__name__}): {e}")
                return False
            flash = sorted(
                (n for n in available if _STABLE_FLASH.match(n)),
                key=lambda n: float(_STABLE_FLASH.match(n).group(1)), reverse=True,
            )
            candidates = [n for n in [DEFAULT_MODEL] + flash if n in available and n != failed]
            if not candidates:
                logger.error(f"No usable Gemini flash model found for this key. Available: {available}")
                return False
            logger.warning(f"Gemini model {failed!r} is unavailable; switching to {candidates[0]!r}. "
                           f"Set TRUTHLENS_MODEL={candidates[0]} to skip this lookup.")
            self.model = candidates[0]
            self.usage.model = self.model
            return True

    def _call(self, messages, **kwargs):
        return self._generate(messages, seed=kwargs.get("seed", 42))

    def _call_vision(self, prompt: str, images: list, **kwargs) -> str:
        parts = [types.Part.from_bytes(data=data, mime_type=mime) for data, mime in images]
        parts.append(types.Part.from_text(text=prompt))
        return self._generate([types.Content(role="user", parts=parts)])

    def construct_message_list(self, prompt_list: list[str], system_role: str = None):
        # The system prompt is sent via GenerateContentConfig.system_instruction in _generate().
        return [[types.Content(role="user", parts=[types.Part.from_text(text=p)])] for p in prompt_list]
