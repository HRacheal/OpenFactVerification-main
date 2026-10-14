import base64
import os

import anthropic
from anthropic import Anthropic
from .base import BaseClient, LLMProviderError, logger

SYSTEM_PROMPT = (
    "You are a careful fact-checking assistant. Respond with a single valid JSON object only - "
    "no markdown fences and no text before or after the JSON."
)

# Models that accept output_config.effort and the server-side refusal fallback ("default" routing).
_EFFORT_PREFIXES = ("claude-opus-5", "claude-sonnet-5", "claude-haiku-5", "claude-fable-5", "claude-opus-4-8", "claude-opus-4-7")
_EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
_FALLBACK_MODELS = ("claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5")
_FALLBACK_BETA = "server-side-fallback-2026-07-01"
# The SDK retries 408/409/429/5xx and connection errors with exponential backoff (honouring retry-after).
# It is the only retry layer for those errors: _create() converts whatever is left into an LLMProviderError,
# which BaseClient never retries, so bad keys, exhausted credits and unknown models are never re-sent.
SDK_MAX_RETRIES = 2


def _effort_from_env() -> str:
    effort = os.environ.get("CLAUDE_EFFORT", "").strip().strip("\"'").lower() or "medium"
    if effort not in _EFFORT_LEVELS:
        logger.warning(f"CLAUDE_EFFORT={effort!r} is not one of {', '.join(_EFFORT_LEVELS)}; using 'medium'.")
        effort = "medium"
    return effort


def describe_api_error(e: anthropic.APIError, model: str) -> str:
    """A user-safe explanation of an Anthropic API error (the SDK never includes the API key in messages)."""
    detail = " ".join(str(getattr(e, "message", "") or "").split())[:300]
    if isinstance(e, anthropic.APITimeoutError):
        return "The Claude API did not answer in time, even after retrying. Please try again."
    if isinstance(e, anthropic.APIConnectionError):
        return "Could not reach the Claude API. Check your internet connection."
    if isinstance(e, anthropic.AuthenticationError):
        return "Anthropic rejected the API key (HTTP 401). Check ANTHROPIC_API_KEY in your .env file."
    if isinstance(e, anthropic.PermissionDeniedError):
        return f"Your Anthropic API key is not allowed to use {model!r} (HTTP 403). {detail}"
    if isinstance(e, anthropic.NotFoundError):
        return (f"Claude model {model!r} was not found (HTTP 404); it may be retired or misspelled. "
                f"Set TRUTHLENS_MODEL to a current model such as claude-haiku-5-5 or claude-opus-5-5.")
    if isinstance(e, anthropic.RateLimitError):
        return "The Claude API rate limit was reached (HTTP 429), even after retrying. Wait a minute and try again."
    if isinstance(e, anthropic.BadRequestError):
        if "credit balance" in detail.lower():
            return "Your Anthropic credit balance is too low. Add credits at https://console.anthropic.com/ and try again."
        return f"The Claude API rejected the request (HTTP 400): {detail}"
    status = getattr(e, "status_code", None)
    if isinstance(status, int) and status >= 500:
        return f"The Claude API is temporarily unavailable (HTTP {status}), even after retrying. Please try again shortly."
    return f"The Claude API returned an error (HTTP {status}): {detail}"


class ClaudeClient(BaseClient):
    def __init__(
        self,
        model: str = "claude-opus-5-5",
        api_config: dict = None,
        max_requests_per_minute=200,
        request_window=60,
        timeout: float = 120.0,
    ):
        super().__init__(model, api_config, max_requests_per_minute, request_window)
        key = self.require_key(self.api_config, "ANTHROPIC_API_KEY", "Anthropic Claude")
        self.client = Anthropic(api_key=key, timeout=timeout, max_retries=SDK_MAX_RETRIES)
        self.effort = _effort_from_env()
        self.use_fallback = os.environ.get("CLAUDE_REFUSAL_FALLBACK", "1") != "0"

    def _request_kwargs(self) -> dict:
        kwargs = {}
        if self.model.startswith(_EFFORT_PREFIXES) and self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        if self.use_fallback and self.model in _FALLBACK_MODELS:
            kwargs["betas"] = [_FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        return kwargs

    def _create(self, messages: list) -> str:
        kwargs = self._request_kwargs()
        create = self.client.beta.messages.create if "betas" in kwargs else self.client.messages.create
        try:
            response = create(model=self.model, max_tokens=16000, system=SYSTEM_PROMPT, messages=messages, **kwargs)
        except anthropic.APIError as e:
            message = describe_api_error(e, self.model)
            logger.error(f"Claude request failed (model={self.model!r}, {type(e).__name__}): {message}")
            raise LLMProviderError(message) from e
        if getattr(response, "usage", None) is not None:
            self._log_usage(response.usage.input_tokens, response.usage.output_tokens)
        if response.stop_reason == "refusal":
            raise LLMProviderError("Claude declined to answer this request.")
        return "".join(block.text for block in response.content if block.type == "text")

    def _call(self, messages, **kwargs):
        # `seed` is not supported by the Anthropic API and is ignored here.
        return self._create(messages)

    def _call_vision(self, prompt: str, images: list, **kwargs) -> str:
        content = []
        for data, mime in images:
            content.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": mime, "data": base64.standard_b64encode(data).decode("ascii")},
                }
            )
        content.append({"type": "text", "text": prompt})
        return self._create([{"role": "user", "content": content}])

    def construct_message_list(self, prompt_list: list[str], system_role: str = None):
        # The system prompt is sent via the top-level `system` parameter in _create().
        return [[{"role": "user", "content": p}] for p in prompt_list]
