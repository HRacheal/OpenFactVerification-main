from __future__ import annotations

import re
from dataclasses import fields

from factcheck.utils.logger import CustomLogger
from factcheck.utils.data_class import Evidence, STANCES
from factcheck.utils.json_utils import parse_llm_json, expect_dict, LLMOutputError

logger = CustomLogger(__name__).getlog()

_EVIDENCE_FIELDS = {f.name for f in fields(Evidence)}
_MAX_EVIDENCE_CHARS = 4000  # keep prompts bounded


def _normalise(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def quote_in_text(quote: str, text: str) -> bool:
    """True if `quote` really appears in `text` (whitespace/case-insensitive)."""
    q = _normalise(quote).strip(" .\"'“”")
    return bool(q) and len(q) >= 8 and q in _normalise(text)


def format_source(evidence: dict) -> str:
    parts = []
    if evidence.get("title"):
        parts.append(f"title: {evidence['title']}")
    if evidence.get("publisher") or evidence.get("domain"):
        parts.append(f"publisher: {evidence.get('publisher') or evidence.get('domain')}")
    parts.append(f"published: {evidence.get('published_date') or 'unknown'}")
    if not evidence.get("full_text_fetched"):
        parts.append("NOTE: this is a search-result snippet only; the full page could not be read")
    return "; ".join(parts)


class ClaimVerify:
    def __init__(self, llm_client, prompt):
        """Initialize the ClaimVerify class

        Args:
            llm_client (BaseClient): The LLM client used for verifying the factuality of claims.
            prompt (BasePrompt): The prompt used for verifying the factuality of claims.
        """
        self.llm_client = llm_client
        self.prompt = prompt

    def verify_claims(
        self, claim_evidences_dict, prompt: str = None, context: str = "None given", output_language: str = "English"
    ) -> dict[str, list[Evidence]]:
        """Assess each (claim, evidence) pair.

        Returns:
            dict: claim -> list[Evidence] with relationship, reasoning, basis and quote filled in.
        """
        return self._verify_all_claims(claim_evidences_dict, prompt=prompt, context=context, output_language=output_language)

    @staticmethod
    def validate_assessment(raw: dict, evidence_text: str) -> dict:
        data = expect_dict(raw, ["reasoning", "relationship"])
        relationship = str(data["relationship"]).strip().upper()
        if relationship not in STANCES:
            raise LLMOutputError(f"Unknown relationship {relationship!r}.")
        reasoning = str(data.get("reasoning", "")).strip()
        basis = str(data.get("basis", "inferred")).strip().lower()
        basis = basis if basis in ("explicit", "inferred") else "inferred"
        quote = str(data.get("quote", "") or "").strip()
        # Guard against invented quotes: a quote that is not in the evidence is discarded and the
        # judgement is downgraded to "inferred".
        if quote and not quote_in_text(quote, evidence_text):
            quote = ""
            basis = "inferred"
        if not quote and basis == "explicit":
            basis = "inferred"
        # "related": the evidence is about a related video/post/rumour/event rather than the claim itself.
        # Verdict.apply_relevance_rules turns such evidence into context; unknown values count as direct.
        scope = str(data.get("scope", "direct") or "direct").strip().lower()
        scope = "related" if scope == "related" else "direct"
        return {"reasoning": reasoning, "relationship": relationship, "basis": basis, "quote": quote, "scope": scope}

    def _verify_all_claims(
        self,
        claim_evidences_dict: dict[str, list[dict]],
        num_retries=3,
        prompt: str = None,
        context: str = "None given",
        output_language: str = "English",
    ) -> dict[str, list[Evidence]]:
        template = prompt or self.prompt.verify_prompt
        claim_evidence_list = []
        messages_list = []
        for claim, _evidences in claim_evidences_dict.items():
            for e in _evidences:
                user_input = template.format(
                    claim=claim,
                    evidence=(e.get("text") or "")[:_MAX_EVIDENCE_CHARS],
                    source=format_source(e),
                    context=context,
                    output_language=output_language,
                )
                claim_evidence_list.append((claim, e))
                messages_list.append(user_input)
        factual_results = [None] * len(messages_list)

        attempts = 0
        while (attempts < num_retries) and (None in factual_results):
            _indices = [i for i, r in enumerate(factual_results) if r is None]
            _messages = [messages_list[i] for i in _indices]
            _response_list = self.llm_client.multi_call(self.llm_client.construct_message_list(_messages))
            for _response, _index in zip(_response_list, _indices):
                try:
                    evidence_text = claim_evidence_list[_index][1].get("text") or ""
                    factual_results[_index] = self.validate_assessment(parse_llm_json(_response), evidence_text)
                except LLMOutputError as e:
                    logger.info(f"Warning: LLM verification parse fail ({e}), retry {attempts}.")
            attempts += 1

        _template_results = {
            "reasoning": "[System warning] The AI response for this evidence could not be read, so it was not used.",
            "relationship": "IRRELEVANT",
            "basis": "inferred",
            "quote": "",
        }

        claim_verifications_dict = {k: [] for k in claim_evidences_dict.keys()}
        for (claim, evidence), verification in zip(claim_evidence_list, factual_results):
            verification = verification or _template_results
            meta = {k: v for k, v in evidence.items() if k in _EVIDENCE_FIELDS and k not in verification and k != "claim"}
            claim_verifications_dict[claim].append(Evidence(claim=claim, **meta, **verification))
        return claim_verifications_dict
