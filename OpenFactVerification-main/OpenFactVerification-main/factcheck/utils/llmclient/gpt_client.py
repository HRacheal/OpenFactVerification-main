import base64

import openai
from openai import OpenAI
from .base import BaseClient, LLMProviderError

DEFAULT_SYSTEM_ROLE = "You are a careful fact-checking assistant. Always answer with a single valid JSON object."


class GPTClient(BaseClient):
    non_retryable_errors = (
        openai.AuthenticationError,
        openai.PermissionDeniedError,
        openai.BadRequestError,
        openai.NotFoundError,
    )

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        api_config: dict = None,
        max_requests_per_minute=200,
        request_window=60,
        timeout: float = 90.0,
    ):
        super().__init__(model, api_config, max_requests_per_minute, request_window)
        key = self.require_key(self.api_config, "OPENAI_API_KEY", "OpenAI")
        # SDK retries are disabled here because BaseClient implements its own retry/backoff.
        self.client = OpenAI(api_key=key, timeout=timeout, max_retries=0)

    def _call(self, messages, **kwargs):
        seed = kwargs.get("seed", 42)
        response = self.client.chat.completions.create(
            response_format={"type": "json_object"},
            seed=seed,
            model=self.model,
            messages=messages,
        )
        choice = response.choices[0]
        if getattr(choice, "finish_reason", None) == "content_filter":
            raise LLMProviderError("The AI provider declined to answer this request (content filter).")
        if getattr(response, "usage", None) is not None:
            self._log_usage(response.usage.prompt_tokens, response.usage.completion_tokens)
        return choice.message.content or ""

    def _call_vision(self, prompt: str, images: list, **kwargs) -> str:
        content = [{"type": "text", "text": prompt}]
        for data, mime in images:
            b64 = base64.b64encode(data).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
        response = self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": DEFAULT_SYSTEM_ROLE}, {"role": "user", "content": content}],
        )
        if getattr(response, "usage", None) is not None:
            self._log_usage(response.usage.prompt_tokens, response.usage.completion_tokens)
        return response.choices[0].message.content or ""

    def transcribe(self, file_path: str, model: str = "whisper-1") -> str:
        """Speech-to-text for an audio file on disk."""

        def _do():
            with open(file_path, "rb") as f:
                result = self.client.audio.transcriptions.create(model=model, file=f)
            return getattr(result, "text", "") or ""

        return self._with_retries(_do, num_retries=2)

    def construct_message_list(self, prompt_list: list[str], system_role: str = DEFAULT_SYSTEM_ROLE):
        return [[{"role": "system", "content": system_role}, {"role": "user", "content": p}] for p in prompt_list]
