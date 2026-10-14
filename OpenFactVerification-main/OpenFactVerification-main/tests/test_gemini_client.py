"""Gemini client: retries, sanitized diagnostics and pipeline wiring.

SIMULATED: requests go through the real google-genai SDK (including its retry logic), but the HTTP layer is an
httpx.MockTransport, so nothing reaches Google and no key or quota is needed.
"""

import json
import logging

import httpx
import pytest
from google import genai

from factcheck.utils.llmclient import GeminiClient, LLMProviderError
from factcheck.utils.llmclient import gemini_client
from tests.conftest import FakeLLM, FakeRetriever, ev

KEY = "AIzaTEST-not-a-real-key-0123456789"
OVERLOADED = {"error": {"code": 503, "status": "UNAVAILABLE", "message": (
    "This model is currently experiencing high demand. Spikes in demand are usually temporary. "
    "Please try again later.")}}
BAD_KEY = {"error": {"code": 400, "status": "INVALID_ARGUMENT", "message": "API key not valid. Please pass a valid API key."}}


def ok(text):
    return httpx.Response(200, json={
        "candidates": [{"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 3},
    })


@pytest.fixture
def gemini_http(monkeypatch):
    """Route every genai.Client built by GeminiClient through a fake transport, with near-zero backoff."""
    state = {"handler": None, "requests": []}
    real_client = genai.Client

    def transport(request):
        state["requests"].append(request)
        return state["handler"](request)

    def make_client(api_key, http_options):
        http_options.httpx_client = httpx.Client(transport=httpx.MockTransport(transport))
        http_options.retry_options.initial_delay = 0.01
        http_options.retry_options.max_delay = 0.02
        return real_client(api_key=api_key, http_options=http_options)

    monkeypatch.setattr(gemini_client.genai, "Client", make_client)
    return state


def prompt_of(request):
    return json.loads(request.content)["contents"][-1]["parts"][-1]["text"]


# --------------------------------------------------------------------------- configuration
def test_retries_are_enabled_on_the_sdk_client(monkeypatch):
    monkeypatch.delenv("TRUTHLENS_GEMINI_RETRIES", raising=False)
    c = GeminiClient(model="gemini-3.8-flash", api_config={"GEMINI_API_KEY": KEY})
    retry = c.client._api_client._http_options.retry_options
    assert retry is not None and retry.attempts == 5
    monkeypatch.setenv("TRUTHLENS_GEMINI_RETRIES", "2")
    assert GeminiClient(api_config={"GEMINI_API_KEY": KEY}).client._api_client._http_options.retry_options.attempts == 2
    monkeypatch.setenv("TRUTHLENS_GEMINI_RETRIES", "not-a-number")
    assert GeminiClient(api_config={"GEMINI_API_KEY": KEY}).retry_attempts == 5


def test_env_selects_gemini_and_configured_model(monkeypatch):
    import webapp

    monkeypatch.setitem(webapp.SETTINGS, "provider", None)
    monkeypatch.setitem(webapp.SETTINGS, "model", None)
    monkeypatch.setenv("TRUTHLENS_PROVIDER", "gemini")
    monkeypatch.setenv("TRUTHLENS_MODEL", "gemini-3.8-flash")
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    assert webapp.resolve_provider() == ("gemini", "gemini-3.8-flash")
    status = webapp.config_status()
    assert status["llm_ready"] and status["llm_key_name"] == "GEMINI_API_KEY"
    assert KEY not in json.dumps(status)


# --------------------------------------------------------------------------- error handling
def test_transient_503_is_retried_then_succeeds(gemini_http):
    replies = iter([httpx.Response(503, json=OVERLOADED), httpx.Response(503, json=OVERLOADED), ok('{"a": 1}')])
    gemini_http["handler"] = lambda r: next(replies)
    c = GeminiClient(model="gemini-3.8-flash", api_config={"GEMINI_API_KEY": KEY})
    # num_retries=1 is what Decompose/CheckWorthy pass; the SDK layer must still retry.
    assert c.call(c.construct_message_list(["hi"]), num_retries=1) == '{"a": 1}'
    assert len(gemini_http["requests"]) == 3
    assert "models/gemini-3.8-flash:generateContent" in str(gemini_http["requests"][0].url)
    assert c.usage.prompt_tokens == 7


def test_persistent_503_gives_sanitized_diagnostic(gemini_http, caplog):
    gemini_http["handler"] = lambda r: httpx.Response(503, json=OVERLOADED)
    c = GeminiClient(model="gemini-3.8-flash", api_config={"GEMINI_API_KEY": KEY})
    with caplog.at_level(logging.ERROR, logger="FactCheck"), pytest.raises(LLMProviderError) as exc:
        c.call(c.construct_message_list(["hi"]), num_retries=1)
    assert len(gemini_http["requests"]) == 5
    # Browser message: status, no secrets, no raw provider dump.
    assert "HTTP 503" in str(exc.value) and "UNAVAILABLE" in str(exc.value) and "5 attempts" in str(exc.value)
    assert KEY not in str(exc.value)
    # Local log: status plus Google's explanation, key scrubbed.
    assert "HTTP 503 UNAVAILABLE" in caplog.text and "high demand" in caplog.text
    assert KEY not in caplog.text


def test_invalid_key_is_not_retried_and_points_at_env(gemini_http, caplog):
    gemini_http["handler"] = lambda r: httpx.Response(400, json=BAD_KEY)
    c = GeminiClient(model="gemini-3.8-flash", api_config={"GEMINI_API_KEY": KEY})
    with caplog.at_level(logging.ERROR, logger="FactCheck"), pytest.raises(LLMProviderError, match="GEMINI_API_KEY") as exc:
        c.call(c.construct_message_list(["hi"]))
    assert len(gemini_http["requests"]) == 1
    assert "HTTP 400" in str(exc.value) and KEY not in str(exc.value)
    assert "API key not valid" in caplog.text and KEY not in caplog.text
    # The key travels only in the request header, never the URL.
    assert KEY not in str(gemini_http["requests"][0].url)


def test_key_echoed_by_provider_is_scrubbed_from_log(gemini_http, caplog):
    body = {"error": {"code": 500, "status": "INTERNAL", "message": f"internal error for key {KEY}"}}
    gemini_http["handler"] = lambda r: httpx.Response(500, json=body)
    c = GeminiClient(api_config={"GEMINI_API_KEY": KEY})
    with caplog.at_level(logging.ERROR, logger="FactCheck"), pytest.raises(LLMProviderError):
        c.call(c.construct_message_list(["hi"]))
    assert "internal error for key ***" in caplog.text and KEY not in caplog.text


def test_quota_429_reports_quota_and_reset_time(gemini_http):
    # Shape of the real free-tier daily-limit error.
    body = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": (
        "You exceeded your current quota, please check your plan and billing details. * Quota exceeded for metric: "
        "generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-3.8-flash "
        "Please retry in 7h37m50.157457866s.")}}
    gemini_http["handler"] = lambda r: httpx.Response(429, json=body)
    c = GeminiClient(api_config={"GEMINI_API_KEY": KEY})
    with pytest.raises(LLMProviderError, match="quota") as exc:
        c.call(c.construct_message_list(["hi"]))
    assert "retry in about 7h37m50s" in str(exc.value)


def test_quota_429_without_retry_hint(gemini_http):
    body = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "Quota exceeded."}}
    gemini_http["handler"] = lambda r: httpx.Response(429, json=body)
    c = GeminiClient(api_config={"GEMINI_API_KEY": KEY})
    with pytest.raises(LLMProviderError, match="Wait a minute"):
        c.call(c.construct_message_list(["hi"]))


def test_unknown_model_404_gives_useful_message(gemini_http):
    def handler(request):
        if request.method == "GET":  # model list used by the 404 fallback: nothing usable
            return httpx.Response(200, json={"models": []})
        return httpx.Response(404, json={"error": {"code": 404, "status": "NOT_FOUND", "message": "not found"}})

    gemini_http["handler"] = handler
    c = GeminiClient(model="gemini-0-typo", api_config={"GEMINI_API_KEY": KEY})
    with pytest.raises(LLMProviderError, match="TRUTHLENS_MODEL"):
        c.call(c.construct_message_list(["hi"]))


# --------------------------------------------------------------------------- pipeline components
def test_claim_extraction_survives_a_503(gemini_http):
    from factcheck.core import Decompose
    from factcheck.utils.prompt import prompt_mapper

    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(503, json=OVERLOADED)
        return ok(json.dumps({"claims": ["Kampala is the capital of Uganda."]}))

    gemini_http["handler"] = handler
    client = GeminiClient(model="gemini-3.8-flash", api_config={"GEMINI_API_KEY": KEY})
    claims = Decompose(llm_client=client, prompt=prompt_mapper("truthlens_prompt")).getclaims(
        "Kampala is the capital of Uganda.")
    assert claims == ["Kampala is the capital of Uganda."] and len(calls) == 2


def test_full_pipeline_runs_on_gemini_clients(gemini_http, monkeypatch):
    """All five steps use GeminiClient + the real SDK; replies come from the offline FakeLLM, search from FakeRetriever."""
    from factcheck import FactCheck
    from factcheck.core import Retriever

    monkeypatch.setitem(Retriever.retriever_map, "fake", FakeRetriever)
    FakeRetriever.error = None
    FakeRetriever.corpus = {"Kampala": [ev("[S] Kampala is the capital of Uganda.", "https://www.gov.ug/about", "official")]}
    brain = FakeLLM()
    first = []

    def handler(request):
        if not first:  # the very first Gemini request is overloaded
            first.append(1)
            return httpx.Response(503, json=OVERLOADED)
        return ok(brain._call([{"role": "user", "content": prompt_of(request)}]))

    gemini_http["handler"] = handler
    fc = FactCheck(default_model="gemini-3.8-flash", client="gemini", retriever="fake",
                   api_config={"GEMINI_API_KEY": KEY})
    for step in ("decompose_model", "checkworthy_model", "query_generator_model",
                 "evidence_retrieval_model", "claim_verify_model"):
        client = getattr(fc, step)
        assert isinstance(client, GeminiClient) and client.model == "gemini-3.8-flash"
    res = fc.check_text("Kampala is the capital of Uganda.", country="Uganda", language="en")
    claims = {c["claim"]: c for c in res["claim_detail"]}
    assert claims["Kampala is the capital of Uganda."]["verdict"] == "SUPPORTED"
    assert all("models/gemini-3.8-flash:" in str(r.url) for r in gemini_http["requests"])
