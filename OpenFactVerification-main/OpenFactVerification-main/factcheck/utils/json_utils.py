"""Safe parsing helpers for structured LLM output.

LLM responses are untrusted text. They must never be passed to ``eval``.
These helpers only ever *parse data* (JSON, or Python literals via
``ast.literal_eval`` which cannot call functions), then validate its shape.
"""

from __future__ import annotations

import ast
import json
import re


class LLMOutputError(ValueError):
    """Raised when an LLM response cannot be parsed into the expected structure."""


_FENCE_RE = re.compile(r"^```(?:json|JSON)?\s*(.*?)\s*```$", re.DOTALL)


def _strip_wrappers(text: str) -> str:
    text = text.strip()
    m = _FENCE_RE.match(text)
    if m:
        text = m.group(1).strip()
    return text


def _first_json_object(text: str) -> str | None:
    """Return the first balanced {...} or [...] substring, respecting strings."""
    start = None
    stack = []
    in_str = False
    escape = False
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            if start is None:
                start = i
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            if ch != stack.pop():
                return None
            if not stack and start is not None:
                return text[start : i + 1]
    return None


def parse_llm_json(text) -> object:
    """Parse an LLM response into a Python object without executing code.

    Accepts plain JSON, JSON wrapped in markdown code fences, JSON surrounded by
    prose, and (as a last resort) Python-literal dicts such as single-quoted keys.
    """
    if not isinstance(text, str) or not text.strip():
        raise LLMOutputError("Empty LLM response.")
    cleaned = _strip_wrappers(text)
    candidates = [cleaned]
    extracted = _first_json_object(cleaned)
    if extracted and extracted != cleaned:
        candidates.append(extracted)
    for cand in candidates:
        try:
            return json.loads(cand)
        except (json.JSONDecodeError, TypeError):
            pass
        try:
            # literal_eval only accepts literals (dict/list/str/num/bool/None); it cannot run code.
            value = ast.literal_eval(cand)
            if isinstance(value, (dict, list)):
                return value
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            pass
    raise LLMOutputError(f"Could not parse LLM response as JSON: {text[:200]!r}")


def expect_dict(value, required_keys=()) -> dict:
    if not isinstance(value, dict):
        raise LLMOutputError(f"Expected a JSON object, got {type(value).__name__}.")
    missing = [k for k in required_keys if k not in value]
    if missing:
        raise LLMOutputError(f"JSON object missing keys: {missing}")
    return value


def expect_str_list(value, name="list") -> list[str]:
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise LLMOutputError(f"Expected {name} to be a list of strings.")
    return [x.strip() for x in value if x.strip()]
