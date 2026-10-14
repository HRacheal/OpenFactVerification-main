from .base import LLMError, MissingCredentialsError, LLMProviderError  # noqa: F401
from .gpt_client import GPTClient
from .claude_client import ClaudeClient
from .local_openai_client import LocalOpenAIClient
from .gemini_client import GeminiClient

# fmt: off
CLIENTS = {
    "gpt": GPTClient,
    "claude": ClaudeClient,
    "gemini": GeminiClient,
    "local_openai": LocalOpenAIClient
}
# fmt: on


def model2client(model_name: str):
    """If the client is not specified, use this function to map the model name to the corresponding client."""
    name = model_name.lower()
    if name.startswith(("gpt", "chatgpt", "o1", "o3", "o4")):
        return GPTClient
    elif name.startswith("claude"):
        return ClaudeClient
    elif name.startswith("gemini"):
        return GeminiClient
    elif name.startswith("vicuna"):
        return LocalOpenAIClient
    else:
        raise ValueError(f"Model {model_name} not supported. Set TRUTHLENS_PROVIDER to choose a client explicitly.")
