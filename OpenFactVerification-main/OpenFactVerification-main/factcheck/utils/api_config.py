"""API key configuration.

Precedence (highest first):
  1. a non-empty value in the YAML config file passed with --api_config (optional, never commit real keys)
  2. environment variables, including those loaded from a local `.env` file in the project root
Keys are never logged or sent to the browser.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

try:
    from dotenv import find_dotenv, load_dotenv

    # Real environment variables win over .env (override=False). The project-root .env is loaded first, then
    # the nearest .env above the working directory, so an earlier file's values are never replaced.
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    load_dotenv(find_dotenv(usecwd=True), override=False)
except ImportError:  # python-dotenv is optional
    pass

# Define all keys for the API configuration
keys = [
    "SERPER_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "LOCAL_API_KEY",
    "LOCAL_API_URL",
]


def load_api_config(api_config: dict = None):
    """Merge API keys from the config dict and the environment (config dict takes precedence)."""
    if api_config is None:
        api_config = dict()
    assert type(api_config) is dict, "api_config must be a dictionary."

    merged_config = {}
    for key in keys:
        value = api_config.get(key) or os.environ.get(key) or ""
        merged_config[key] = str(value).strip().strip("\"'") or None

    for key in api_config.keys():
        if key not in keys:
            merged_config[key] = api_config[key]
    return merged_config
