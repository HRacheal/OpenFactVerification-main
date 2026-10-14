import time
import asyncio
import threading
from abc import abstractmethod
from functools import partial
from collections import deque

from ..data_class import TokenUsage
from ..logger import CustomLogger

logger = CustomLogger(__name__).getlog()


class LLMError(RuntimeError):
    """Base error for LLM provider problems. Messages are safe to show users (no secrets)."""


class MissingCredentialsError(LLMError):
    """Raised when the API key for the selected provider is not configured."""


class LLMProviderError(LLMError):
    """Raised when the provider fails (network, rate limit, refusal, bad request) after retries."""


class BaseClient:
    # Errors that will never succeed on retry (subclasses fill these with SDK exception classes).
    non_retryable_errors: tuple = ()

    def __init__(
        self,
        model: str,
        api_config: dict,
        max_requests_per_minute: int,
        request_window: int,
    ) -> None:
        self.model = model
        self.api_config = api_config or {}
        self.max_requests_per_minute = max_requests_per_minute
        self.request_window = request_window
        self.traffic_queue = deque()
        self.total_traffic = 0
        self.usage = TokenUsage(model=model)
        self._usage_lock = threading.Lock()

    @staticmethod
    def require_key(api_config: dict, key: str, provider: str) -> str:
        value = (api_config or {}).get(key)
        if not value:
            raise MissingCredentialsError(
                f"{provider} is selected but {key} is not configured. "
                f"Add it to your .env file (see .env.example) and restart the app."
            )
        return value

    @abstractmethod
    def _call(self, messages, **kwargs):
        """Internal function to call the API. Returns the response text."""

    @abstractmethod
    def _call_vision(self, prompt: str, images: list, **kwargs) -> str:
        """Send a text prompt plus images [(bytes, mime_type), ...]; return text."""

    def _log_usage(self, prompt_tokens: int = 0, completion_tokens: int = 0):
        with self._usage_lock:
            self.usage.prompt_tokens += int(prompt_tokens or 0)
            self.usage.completion_tokens += int(completion_tokens or 0)

    def get_usage(self):
        return self.usage

    def reset_usage(self):
        with self._usage_lock:
            self.usage.prompt_tokens = 0
            self.usage.completion_tokens = 0

    @abstractmethod
    def construct_message_list(self, prompt_list: list[str]) -> list:
        """Construct a list of messages for the function self.multi_call."""

    def get_request_length(self, messages):
        return 1

    def _with_retries(self, fn, num_retries=3, waiting_time=1.0):
        last_error = None
        for attempt in range(max(1, num_retries)):
            try:
                result = fn()
                if result:
                    return result
                last_error = LLMProviderError("The AI provider returned an empty response.")
            except LLMError:
                raise
            except self.non_retryable_errors as e:
                raise LLMProviderError(f"AI provider rejected the request ({type(e).__name__}).") from e
            except Exception as e:  # network errors, rate limits, 5xx
                last_error = e
                logger.warning(f"LLM call failed ({type(e).__name__}); attempt {attempt + 1}/{num_retries}.")
            time.sleep(waiting_time * (2**attempt))
        if isinstance(last_error, LLMError):
            raise last_error
        raise LLMProviderError(
            f"AI provider request failed after {num_retries} attempts ({type(last_error).__name__})."
        ) from last_error

    def call(self, messages: list, num_retries=3, waiting_time=1, **kwargs):
        seed = kwargs.get("seed", 42)
        assert type(seed) is int, "Seed must be an integer."
        assert len(messages) == 1, "Only one message is allowed for this function."
        return self._with_retries(partial(self._call, messages[0], seed=seed), num_retries, waiting_time)

    def vision(self, prompt: str, images: list, num_retries=2) -> str:
        """Ask the model about one or more images. images: list of (bytes, mime_type)."""
        return self._with_retries(partial(self._call_vision, prompt, images), num_retries)

    def set_model(self, model: str):
        self.model = model

    async def _async_call(self, messages: list, **kwargs):
        """Calls the API asynchronously, tracks traffic, and enforces rate limits."""
        while len(self.traffic_queue) >= self.max_requests_per_minute:
            await asyncio.sleep(1)
            self._expire_old_traffic()

        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, partial(self._with_retries, partial(self._call, messages, **kwargs)))

        self.total_traffic += self.get_request_length(messages)
        self.traffic_queue.append((time.time(), self.get_request_length(messages)))

        return response

    def multi_call(self, messages_list, **kwargs):
        """Run several calls concurrently. Failed calls come back as "" so callers can retry/fallback;
        if every call fails, the first error is raised so configuration problems are not hidden."""
        if not messages_list:
            return []

        async def _gather():
            tasks = [self._async_call(messages=messages, **kwargs) for messages in messages_list]
            return await asyncio.gather(*tasks, return_exceptions=True)

        # A fresh event loop per call: safe from worker threads (web app jobs) and the main thread.
        results = asyncio.run(_gather())
        errors = [r for r in results if isinstance(r, BaseException)]
        if errors and len(errors) == len(results):
            first = errors[0]
            raise first if isinstance(first, LLMError) else LLMProviderError(str(type(first).__name__))
        for e in errors:
            logger.warning(f"One concurrent LLM call failed: {type(e).__name__}: {e}")
        return ["" if isinstance(r, BaseException) else r for r in results]

    def _expire_old_traffic(self):
        """Expires traffic older than the request window."""
        current_time = time.time()
        while self.traffic_queue and self.traffic_queue[0][0] + self.request_window < current_time:
            self.total_traffic -= self.traffic_queue.popleft()[1]
