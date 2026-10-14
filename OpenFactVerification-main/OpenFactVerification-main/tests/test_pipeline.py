"""End-to-end pipeline with a FAKE LLM and FAKE search results (simulated; no real verification)."""

import json
import os

import pytest

from factcheck.core.Retriever import RetrievalError
from tests.conftest import FakeRetriever, FakeLLM, ev

DATA = os.path.join(os.path.dirname(__file__), "data", "multilingual_claims.json")


def test_full_pipeline_produces_connected_report(fake_pipeline):
    FakeRetriever.corpus = {
        "Kampala": [ev("[S] Kampala is the capital of Uganda.", "https://www.gov.ug/about", "official")],
        "Kilimanjaro": [
            ev("[R] Mount Kilimanjaro is in Tanzania, not Kenya.", "https://www.nationalgeographic.com/k", "unknown"),
            ev("[R] Kilimanjaro, Tanzania's highest peak.", "https://www.thecitizen.co.tz/k", "news"),
            ev("[R] Tanzania is home to Kilimanjaro.", "https://www.bbc.com/k", "news"),
        ],
    }
    text = "Kampala is the capital of Uganda. Mount Kilimanjaro is in Kenya. Rwanda has the best coffee."
    res = fake_pipeline.check_text(text, country="Uganda", language="en", output_language="en")
    claims = {c["claim"]: c for c in res["claim_detail"]}
    assert claims["Kampala is the capital of Uganda."]["verdict"] == "SUPPORTED"
    assert claims["Mount Kilimanjaro is in Kenya."]["verdict"] == "CONTRADICTED"
    opinion = claims["Rwanda has the best coffee."]
    assert opinion["verdict"] == "NOT_CHECKABLE" and opinion["evidences"] == []
    # evidence stays linked to its source
    e = claims["Kampala is the capital of Uganda."]["evidences"][0]
    assert e["url"] == "https://www.gov.ug/about" and e["published_date"] and e["retrieved_at"] and e["quote"]
    assert res["summary"]["verdict_counts"]["SUPPORTED"] == 1
    assert res["verified_at"] and res["context"]["country"] == "Uganda"
    assert claims["Kampala is the capital of Uganda."]["explanation"] == "Fake explanation."


def test_no_evidence_gives_insufficient_never_false(fake_pipeline):
    res = fake_pipeline.check_text("The Nile is 10 kilometres long.")
    c = res["claim_detail"][0]
    assert c["verdict"] == "INSUFFICIENT_EVIDENCE"
    assert any("no usable results" in n for n in c["notes"])


def test_conflicting_evidence_is_mixed(fake_pipeline):
    FakeRetriever.corpus = {"fees": [
        ev("[S] Fees were cut by half.", "https://www.education.go.ke/a", "official"),
        ev("[R] Fees were not cut.", "https://pesacheck.org/b", "fact_checker"),
    ]}
    res = fake_pipeline.check_text("University fees were cut by 50% in 2026.")
    assert res["claim_detail"][0]["verdict"] == "MIXED"


def test_content_without_claims(fake_pipeline):
    res = fake_pipeline.check_text("Is this true?")
    assert res["claim_detail"] == [] and res["warnings"]


def test_search_failure_propagates_as_error(fake_pipeline):
    FakeRetriever.error = RetrievalError("The evidence search service rejected the API key.")
    with pytest.raises(RetrievalError):
        fake_pipeline.check_text("Kampala is the capital of Uganda.")


def test_country_and_language_reach_every_step(fake_pipeline):
    fake_pipeline.check_text("Nairobi ni mji mkuu wa Kenya.", country="Kenya", language="sw", output_language="sw")
    assert fake_pipeline.evidence_crawler.context == {"country": "Kenya", "language": "sw"}
    decomposition = [p for p in FakeLLM.prompts_seen if "You extract verifiable factual claims" in p][0]
    assert "Country/region: Kenya" in decomposition and "Swahili" in decomposition
    checkworthy = [p for p in FakeLLM.prompts_seen if "Decide, for each statement" in p][0]
    assert "written in Swahili (Kiswahili)" in checkworthy
    queries = [p for p in FakeLLM.prompts_seen if "You write web search queries" in p][0]
    assert "Claim: Nairobi ni mji mkuu wa Kenya." in queries


def load_multilingual():
    with open(DATA, encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.parametrize("case", load_multilingual()["cases"], ids=lambda c: c["id"])
def test_multilingual_claims_preserved_and_routed(fake_pipeline, case):
    """Simulated: checks that Swahili/Luganda text is not translated or altered on its way through the
    pipeline and that the verdict logic responds to the (fake) evidence. Real language quality must be
    measured with script/eval_multilingual.py and live API keys."""
    stance = {"SUPPORTED": "[S]", "CONTRADICTED": "[R]"}[case["expected_verdict"]]
    FakeRetriever.corpus = {case["claim"][:12]: [
        ev(f"{stance} {e['excerpt']}", e["url"], e["source_type"]) for e in case["reference_evidence"]
    ]}
    res = fake_pipeline.check_text(case["claim"], country=case["country"], language=case["language"], output_language=case["language"])
    c = res["claim_detail"][0]
    assert c["claim"] == case["claim"]  # wording unchanged (numbers, names, diacritics)
    assert c["verdict"] == case["expected_verdict"]
