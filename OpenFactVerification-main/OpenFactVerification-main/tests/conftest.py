"""Shared test fixtures. ALL tests here are offline: LLMs, search and web pages are mocked."""

import json
import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # the logger writes to ./log

# Make sure no real keys leak into tests. The variables are set to "" rather than removed: importing
# factcheck loads .env files with override=False, which would otherwise fill them back in with real keys.
for _k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "SERPER_API_KEY", "LOCAL_API_KEY", "LOCAL_API_URL",
           "TRUTHLENS_PROVIDER", "TRUTHLENS_MODEL"):
    os.environ[_k] = ""
# Fixed Claude settings, independent of the shell (Claude Code, for one, sets CLAUDE_EFFORT itself).
os.environ["CLAUDE_EFFORT"] = "medium"
os.environ["CLAUDE_REFUSAL_FALLBACK"] = "1"

from factcheck.utils.llmclient.base import BaseClient  # noqa: E402


class FakeLLM(BaseClient):
    """A deterministic stand-in for an LLM. It recognises which Kweli Lens prompt it received and answers
    in the expected JSON shape. Evidence text containing [S], [R], [P] or [C] decides the stance."""

    prompts_seen = []

    def __init__(self, model="fake-model", api_config=None, **kw):
        super().__init__(model, api_config or {}, 1000, 60)
        self.overrides = {}

    def construct_message_list(self, prompt_list, system_role=None):
        return [[{"role": "user", "content": p}] for p in prompt_list]

    def _call(self, messages, **kwargs):
        prompt = messages[-1]["content"]
        FakeLLM.prompts_seen.append(prompt)
        self._log_usage(10, 5)
        for key, value in self.overrides.items():
            if key in prompt:
                return value
        if "You extract verifiable factual claims" in prompt:
            doc = prompt.split("\nText: ")[-1].rsplit("\nOutput (JSON only):", 1)[0].strip()
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", doc) if s.strip() and not s.strip().endswith("?")]
            return json.dumps({"claims": sents}, ensure_ascii=False)
        if "split the text into chunks" in prompt:
            facts = prompt.split("\nFacts: ")[-1].rsplit("\nOutput (JSON only):", 1)[0].strip()
            import ast

            claims = ast.literal_eval(facts)
            return json.dumps({c: c for c in claims}, ensure_ascii=False)
        if "Decide, for each statement" in prompt:
            block = prompt.split("Statements:\n")[-1].rsplit("\n\nOutput (JSON only):", 1)[0]
            out = {}
            for line in block.strip().splitlines():
                stmt = re.sub(r"^\d+\.\s*", "", line).strip()
                out[stmt] = "No (Opinion.)" if re.search(r"\b(best|worst)\b", stmt, re.I) else "Yes (Checkable fact.)"
            return json.dumps(out, ensure_ascii=False)
        if "You write web search queries" in prompt:
            claim = prompt.split("Claim: ")[1].split("\n")[0]
            return json.dumps({"Questions": [claim + " fact check"]}, ensure_ascii=False)
        if "You assess ONE piece of evidence" in prompt:
            text = prompt.split("[evidence text]: ")[1].split("\n\nDecide the relationship")[0]
            rel = {"[S]": "SUPPORTS", "[R]": "REFUTES", "[P]": "PARTIAL", "[C]": "CONTEXT"}
            relationship = next((v for k, v in rel.items() if k in text), "IRRELEVANT")
            scope = "related" if "[X]" in text else "direct"  # [X]: about a related video/post, not the claim
            for marker in ("[S]", "[R]", "[P]", "[C]", "[X]"):
                text = text.replace(marker, "")
            quote = text.strip()[:60]
            return json.dumps({"reasoning": f"Fake reasoning: {relationship}", "relationship": relationship,
                               "basis": "explicit" if relationship != "IRRELEVANT" else "inferred", "quote": quote,
                               "scope": scope})
        if "You write a short, neutral explanation" in prompt:
            return json.dumps({"explanation": "Fake explanation."})
        return "{}"

    def _call_vision(self, prompt, images, **kwargs):
        FakeLLM.prompts_seen.append(prompt)
        if "frames sampled" in prompt:
            return json.dumps({"on_screen_text": "BREAKING: Kampala floods", "description": "News broadcast."})
        return json.dumps({"visible_text": "Uganda has 146 districts.", "description": "Screenshot of a post.", "unclear_parts": []})


class FakeRetriever:
    """Returns canned evidence per claim. Evidence is chosen by keywords in the claim."""

    corpus = {}
    error = None

    def __init__(self, llm_client=None, api_config=None):
        self.llm_client = llm_client
        self.context = {}

    def set_context(self, country=None, language=None):
        self.context = {"country": country, "language": language}

    def retrieve_evidence(self, claim_queries_dict):
        if FakeRetriever.error:
            raise FakeRetriever.error
        out = {}
        for claim in claim_queries_dict:
            items = []
            for keyword, evs in FakeRetriever.corpus.items():
                if keyword.lower() in claim.lower():
                    items.extend(dict(e) for e in evs)
            out[claim] = items
        return out


def ev(text, url, source_type="news", full=True, date="2025-01-10"):
    from factcheck.utils.sources import get_domain

    return {"text": text, "url": url, "title": f"Title for {get_domain(url)}", "domain": get_domain(url),
            "publisher": get_domain(url), "published_date": date, "retrieved_at": "2026-10-13T10:00:00+00:00",
            "source_type": source_type, "full_text_fetched": full, "query": "q"}


@pytest.fixture
def fake_pipeline(monkeypatch):
    from factcheck.utils import llmclient
    from factcheck.core import Retriever
    from factcheck import FactCheck

    monkeypatch.setitem(llmclient.CLIENTS, "fake", FakeLLM)
    monkeypatch.setitem(Retriever.retriever_map, "fake", FakeRetriever)
    FakeRetriever.corpus = {}
    FakeRetriever.error = None
    FakeLLM.prompts_seen = []
    return FactCheck(default_model="fake-model", client="fake", retriever="fake", api_config={})
