from openai import OpenAI
from .base import BaseClient, LLMProviderError


class LocalOpenAIClient(BaseClient):
    """Support Local host LLM chatbot with OpenAI API.
    see https://github.com/lm-sys/FastChat/blob/main/docs/openai_api.md for example usage.
    """

    def __init__(
        self,
        model: str = "",
        api_config: dict = None,
        max_requests_per_minute=200,
        request_window=60,
    ):
        super().__init__(model, api_config, max_requests_per_minute, request_window)
        base_url = self.require_key(self.api_config, "LOCAL_API_URL", "Local OpenAI-compatible server")
        # Previously this mutated the global `openai` module, which would also redirect any GPT client.
        self.client = OpenAI(api_key=self.api_config.get("LOCAL_API_KEY") or "not-needed", base_url=base_url, max_retries=0)

    def _call(self, messages, **kwargs):
        seed = kwargs.get("seed", 42)
        response = self.client.chat.completions.create(
            response_format={"type": "json_object"},
            seed=seed,
            model=self.model,
            messages=messages,
        )
        if getattr(response, "usage", None) is not None:
            self._log_usage(response.usage.prompt_tokens, response.usage.completion_tokens)
        return response.choices[0].message.content or ""

    def _call_vision(self, prompt, images, **kwargs):
        raise LLMProviderError("Image analysis is not supported with the local model client.")

    def construct_message_list(
        self,
        prompt_list: list[str],
        system_role: str = "You are a helpful assistant designed to output JSON.",
    ):
        return [[{"role": "system", "content": system_role}, {"role": "user", "content": p}] for p in prompt_list]
