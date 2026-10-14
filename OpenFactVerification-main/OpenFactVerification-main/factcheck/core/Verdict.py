"""Kweli Lens verdict rules.

These rules are deliberately simple, deterministic and documented (see docs/verdict_rules.md) so a
journalist can see exactly why a verdict was given. The AI only labels individual pieces of evidence;
it does NOT choose the final verdict.

Summary of the rules
--------------------
1. Only evidence labelled SUPPORTS, REFUTES or PARTIAL counts. CONTEXT and IRRELEVANT never move a verdict.
2. Each counted piece gets a weight:
       weight = basis weight x source weight x text weight
   basis:  explicit (quote verified in the source) = 1.0, inferred = 0.5
   source: official / academic / fact_checker = 1.0, news = 0.8, unknown = 0.5, social media = 0.25
   text:   full page read = 1.0, search snippet only = 0.7
3. Independence: evidence is grouped by website (registrable domain, so en.wikipedia.org and
   sw.wikipedia.org are the same site). Per domain, only the single strongest supporting
   weight and the single strongest contradicting weight are kept, so ten pages from one website
   count as one source. PARTIAL adds its weight to both sides.
4. S = total support weight, R = total contradiction weight. A side is "established" when its
   weight is at least 1.0 (e.g. one explicit statement from an official source, or two independent
   explicit statements from news outlets).
5. Verdict:
       neither side established                    -> INSUFFICIENT_EVIDENCE
       both sides established                      -> MIXED
       one side established, other side > 0 and
         the established side is less than 2x it   -> MIXED
       only support established                    -> SUPPORTED
       only contradiction established              -> CONTRADICTED
6. "Evidence strength" (low / moderate / high) describes how much weighted evidence backs the verdict.
   It is NOT a probability that the claim is true.
"""

from __future__ import annotations

import re

from factcheck.utils.sources import site_key, get_domain
from factcheck.utils.data_class import (
    VERDICT_SUPPORTED,
    VERDICT_CONTRADICTED,
    VERDICT_MIXED,
    VERDICT_INSUFFICIENT,
)

BASIS_WEIGHT = {"explicit": 1.0, "inferred": 0.5}
SOURCE_WEIGHT = {"official": 1.0, "academic": 1.0, "fact_checker": 1.0, "news": 0.8, "unknown": 0.5, "social": 0.25}
SNIPPET_ONLY_WEIGHT = 0.7
ESTABLISHED = 1.0
DOMINANCE_RATIO = 2.0

VERDICT_LABELS = {
    VERDICT_SUPPORTED: "Supported",
    VERDICT_CONTRADICTED: "Contradicted",
    VERDICT_MIXED: "Mixed / conflicting",
    VERDICT_INSUFFICIENT: "Insufficient evidence",
    "NOT_CHECKABLE": "Not checkable",
}


def _get(e, name, default=None):
    return e.get(name, default) if isinstance(e, dict) else getattr(e, name, default)


def evidence_weight(e) -> float:
    w = BASIS_WEIGHT.get(_get(e, "basis", "inferred"), 0.5)
    w *= SOURCE_WEIGHT.get(_get(e, "source_type", "unknown"), 0.5)
    if not _get(e, "full_text_fetched", False):
        w *= SNIPPET_ONLY_WEIGHT
    return round(w, 3)


def _strength_label(weight: float) -> str:
    if weight >= 3.0:
        return "high"
    if weight >= 1.5:
        return "moderate"
    return "low"


def decide_verdict(evidences: list) -> tuple[str, str, dict]:
    """Apply the Kweli Lens rules. Returns (verdict, rule_explanation, strength_details)."""
    per_domain: dict[str, dict] = {}
    counts = {"SUPPORTS": 0, "REFUTES": 0, "PARTIAL": 0, "CONTEXT": 0, "IRRELEVANT": 0}
    for e in evidences:
        rel = _get(e, "relationship", "IRRELEVANT")
        counts[rel] = counts.get(rel, 0) + 1
        if rel not in ("SUPPORTS", "REFUTES", "PARTIAL"):
            continue
        domain = site_key(_get(e, "domain") or get_domain(_get(e, "url") or "")) or "unknown-source"
        w = evidence_weight(e)
        slot = per_domain.setdefault(domain, {"support": 0.0, "refute": 0.0})
        if rel in ("SUPPORTS", "PARTIAL"):
            slot["support"] = max(slot["support"], w)
        if rel in ("REFUTES", "PARTIAL"):
            slot["refute"] = max(slot["refute"], w)

    S = round(sum(d["support"] for d in per_domain.values()), 3)
    R = round(sum(d["refute"] for d in per_domain.values()), 3)
    support_sources = sum(1 for d in per_domain.values() if d["support"] > 0)
    refute_sources = sum(1 for d in per_domain.values() if d["refute"] > 0)
    s_ok, r_ok = S >= ESTABLISHED, R >= ESTABLISHED

    if s_ok and r_ok:
        verdict = VERDICT_MIXED
        reason = "Credible evidence was found both supporting and contradicting the claim."
    elif s_ok:
        if R > 0 and S < DOMINANCE_RATIO * R:
            verdict, reason = VERDICT_MIXED, "Supporting evidence is not clearly stronger than the contradicting evidence."
        else:
            verdict, reason = VERDICT_SUPPORTED, "Enough independent, relevant evidence supports the claim."
    elif r_ok:
        if S > 0 and R < DOMINANCE_RATIO * S:
            verdict, reason = VERDICT_MIXED, "Contradicting evidence is not clearly stronger than the supporting evidence."
        else:
            verdict, reason = VERDICT_CONTRADICTED, "Enough independent, relevant evidence contradicts the claim."
    else:
        verdict = VERDICT_INSUFFICIENT
        if S == 0 and R == 0:
            reason = "No retrieved evidence directly supports or contradicts the claim. This does not mean the claim is false."
        else:
            reason = "Some relevant evidence was found, but not enough strong, independent evidence to reach a conclusion."

    winning = {VERDICT_SUPPORTED: S, VERDICT_CONTRADICTED: R, VERDICT_MIXED: max(S, R)}.get(verdict, max(S, R))
    strength = {
        "support_weight": S,
        "contradict_weight": R,
        "support_sources": support_sources,
        "contradict_sources": refute_sources,
        "counts": counts,
        "label": _strength_label(winning) if verdict != VERDICT_INSUFFICIENT else "low",
    }
    return verdict, reason, strength


_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")


def evidence_notes(claim: str, evidences: list) -> list[str]:
    """Limitations worth flagging to the journalist (no effect on the verdict)."""
    notes = []
    relevant = [e for e in evidences if _get(e, "relationship") in ("SUPPORTS", "REFUTES", "PARTIAL")]
    if evidences and relevant and all(not _get(e, "full_text_fetched", False) for e in relevant):
        notes.append("All deciding evidence comes from search snippets; the full pages could not be read.")
    if relevant and all(_get(e, "basis") == "inferred" for e in relevant):
        notes.append("No deciding evidence was backed by an exact quote; the assessments are inferences.")
    if any(_get(e, "source_type") == "social" for e in relevant):
        notes.append("Some deciding evidence comes from social media or user-generated sites.")
    years = [int(y) for y in _YEAR_RE.findall(claim)]
    if years and relevant:
        claim_year = max(years)
        older = [e for e in relevant if (_get(e, "published_date") or "")[:4].isdigit() and int(_get(e, "published_date")[:4]) < claim_year]
        if older:
            notes.append(f"{len(older)} deciding source(s) were published before {claim_year}, the year mentioned in the claim.")
    if not any(_get(e, "published_date") for e in relevant) and relevant:
        notes.append("Publication dates were not available for the deciding sources.")
    return notes
