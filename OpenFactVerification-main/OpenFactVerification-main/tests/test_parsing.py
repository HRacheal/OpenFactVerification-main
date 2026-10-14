"""Safe handling of AI responses: parsing, validation, and the core pipeline steps' fallbacks. (Mocked.)"""

import json

import pytest

from factcheck.utils.json_utils import parse_llm_json, LLMOutputError, expect_str_list
from factcheck.core import Decompose, Checkworthy, QueryGenerator, ClaimVerify
from factcheck.utils.prompt import prompt_mapper
from tests.conftest import FakeLLM

PROMPT = prompt_mapper("truthlens_prompt")


def test_parses_plain_fenced_and_wrapped_json():
    assert parse_llm_json('{"a": 1}') == {"a": 1}
    assert parse_llm_json('```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}
    assert parse_llm_json('Sure! Here it is: {"claims": ["x {y}"]} Hope that helps.') == {"claims": ["x {y}"]}
    assert parse_llm_json("{'claims': ['single quoted']}") == {"claims": ["single quoted"]}


def test_never_executes_code(tmp_path):
    marker = tmp_path / "pwned"
    payload = f"__import__('pathlib').Path(r'{marker}').write_text('x')"
    with pytest.raises(LLMOutputError):
        parse_llm_json(payload)
    assert not marker.exists()


@pytest.mark.parametrize("bad", ["", "   ", "not json at all", "{unclosed", None, 42])
def test_malformed_responses_raise(bad):
    with pytest.raises(LLMOutputError):
        parse_llm_json(bad)


def test_expect_str_list_rejects_non_strings():
    with pytest.raises(LLMOutputError):
        expect_str_list(["ok", 3])


def test_decompose_falls_back_to_sentences_on_garbage():
    llm = FakeLLM()
    llm.overrides["You extract verifiable factual claims"] = "I cannot comply."
    claims = Decompose(llm, PROMPT).getclaims("Nairobi is in Kenya. Is it big? Kampala is in Uganda.", num_retries=2)
    assert claims == ["Nairobi is in Kenya.", "Kampala is in Uganda."]  # the question is not turned into a claim


def test_decompose_preserves_swahili_wording():
    claims = Decompose(FakeLLM(), PROMPT).getclaims("Mlima Kilimanjaro uko Kenya.")
    assert claims == ["Mlima Kilimanjaro uko Kenya."]


def test_restore_claims_always_returns_every_claim():
    llm = FakeLLM()
    llm.overrides["split the text into chunks"] = json.dumps({"other": "zzz"})
    out = Decompose(llm, PROMPT).restore_claims("A is B.", ["A is B."], num_retries=1)
    assert list(out) == ["A is B."]


def test_checkworthy_filters_opinions_and_handles_failure():
    cw = Checkworthy(FakeLLM(), PROMPT)
    worthy, reasons = cw.identify_checkworthiness(["Uganda has 146 districts.", "This is the best government."])
    assert worthy == ["Uganda has 146 districts."]
    assert reasons["This is the best government."].startswith("No")

    broken = FakeLLM()
    broken.overrides["Decide, for each statement"] = "garbage"
    worthy, reasons = Checkworthy(broken, PROMPT).identify_checkworthiness(["X happened."], num_retries=1)
    assert worthy == ["X happened."] and "could not be assessed" in reasons["X happened."]


def test_query_generator_accepts_legacy_key_and_caps_queries():
    llm = FakeLLM()
    llm.overrides["You write web search queries"] = json.dumps({"Question": ["a", "b", "c", "d", "e"]})
    q = QueryGenerator(llm, PROMPT, max_query_per_claim=3).generate_query(["Claim X."])
    assert q["Claim X."] == ["Claim X.", "a", "b"]


def test_query_generator_includes_claim_even_if_llm_fails():
    llm = FakeLLM()
    llm.overrides["You write web search queries"] = "nope"
    q = QueryGenerator(llm, PROMPT).generate_query(["Claim X."], generating_time=1)
    assert q == {"Claim X.": ["Claim X."]}


def test_verify_rejects_unknown_relationship_and_invented_quotes():
    with pytest.raises(LLMOutputError):
        ClaimVerify.validate_assessment({"reasoning": "r", "relationship": "TRUE"}, "text")
    out = ClaimVerify.validate_assessment(
        {"reasoning": "r", "relationship": "supports", "basis": "explicit", "quote": "a quote that is not there"},
        "The evidence says something else entirely.",
    )
    assert out["relationship"] == "SUPPORTS" and out["quote"] == "" and out["basis"] == "inferred"
    ok = ClaimVerify.validate_assessment(
        {"reasoning": "r", "relationship": "REFUTES", "basis": "explicit", "quote": "the capital  of Uganda"},
        "Kampala is the capital of Uganda.",
    )
    assert ok["basis"] == "explicit" and ok["quote"]


def test_verify_unparseable_response_becomes_irrelevant_not_false():
    llm = FakeLLM()
    llm.overrides["You assess ONE piece of evidence"] = "<<broken>>"
    res = ClaimVerify(llm, PROMPT).verify_claims({"C.": [{"text": "[R] something", "url": "https://a.com"}]})
    e = res["C."][0]
    assert e.relationship == "IRRELEVANT" and "could not be read" in e.reasoning
