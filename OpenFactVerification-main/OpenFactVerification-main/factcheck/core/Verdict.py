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
7. Relevance guards, applied before the verdict (see apply_relevance_rules):
   - Evidence the assessor marks as being about something RELATED (e.g. a fake video, a rumour or post
     about the claim, a different event or period) is shown as context, never counted for or against.
   - Claims about a current situation that can change (someone has died, been arrested, resigned ...)
     with no date in the claim are "time-sensitive": deciding evidence published more than RECENT_DAYS
     before the check is shown as context, because it cannot show the situation now. Exception: an
     older report that someone died still supports a claim that they are dead (death is permanent).
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta

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

RECENT_DAYS = 30

# Plain-language labels shown to users. The internal verdict codes are unchanged (JSON compatibility).
VERDICT_LABELS = {
    VERDICT_SUPPORTED: "True",
    VERDICT_CONTRADICTED: "False",
    VERDICT_MIXED: "Partly true / disputed",
    VERDICT_INSUFFICIENT: "Unverified",
    "NOT_CHECKABLE": "Not checkable",
}
# One-sentence meaning of each verdict, shown under the verdict.
VERDICT_HEADLINES = {
    VERDICT_SUPPORTED: "Reliable evidence found during this check supports the claim.",
    VERDICT_CONTRADICTED: "Reliable evidence found during this check contradicts the claim.",
    VERDICT_MIXED: "Part of the claim is supported but part is not, or reliable sources disagree.",
    VERDICT_INSUFFICIENT: "There is not enough reliable evidence to confirm or rule out this claim. "
                          "This does not mean the claim is false.",
    "NOT_CHECKABLE": "This is an opinion, a prediction or too vague to check against evidence.",
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


# --------------------------------------------------------------------------- relevance and time
_DEATH = (r"dead|died|dies|death|passed away|killed|murdered|assassinated|"
          r"amefariki|amekufa|ameaga dunia|ameuawa|afudde|attiddwa")
_CHANGEABLE = (r"arrested|detained|jailed|imprisoned|in custody|kidnapped|abducted|missing|hospitali[sz]ed|"
               r"in hospital|critically ill|in a coma|resigned|sacked|fired|impeached|ousted|overthrown|fled|"
               r"in exile|coup|amekamatwa|amejiuzulu|amefutwa kazi|amelazwa|ametekwa|amepotea|akwatiddwa")
_DEATH_RE = re.compile(rf"\b({_DEATH})\b", re.I)
_CHANGEABLE_RE = re.compile(rf"\b({_DEATH}|{_CHANGEABLE})\b", re.I)
_DATE_IN_CLAIM_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b|\b\d{1,2}(st|nd|rd|th)?\s+(jan|feb|mar|apr|may|jun|jul|aug|"
                               r"sep|oct|nov|dec)[a-z]*\b", re.I)


def is_time_sensitive(claim: str) -> bool:
    """A claim about a current, changeable situation (alive/dead, detained, in office ...) with no stated date."""
    return bool(_CHANGEABLE_RE.search(claim or "")) and not _DATE_IN_CLAIM_RE.search(claim or "")


def is_death_claim(claim: str) -> bool:
    return bool(_DEATH_RE.search(claim or ""))


_REL_DATE_RE = re.compile(r"(\d+)\s*(minute|hour|day|week|month|year)s?\s+ago", re.I)
_TEXT_DATE_FORMATS = ("%b %d, %Y", "%B %d, %Y", "%d %b %Y", "%d %B %Y", "%b %d %Y", "%Y/%m/%d", "%d/%m/%Y")


def parse_date(value, today: date | None = None) -> date | None:
    """Parse the date formats search providers and page metadata use ("2026-10-14", "Oct 14, 2026",
    "3 days ago"). Returns None when unknown; never guesses."""
    if not value:
        return None
    text = str(value).strip()
    today = today or date.today()
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", text)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    m = _REL_DATE_RE.search(text)
    if m:
        n, unit = int(m.group(1)), m.group(2).lower()
        days = {"minute": 0, "hour": 0, "day": n, "week": 7 * n, "month": 30 * n, "year": 365 * n}[unit]
        return today - timedelta(days=days)
    for fmt in _TEXT_DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _set(e, name, value):
    if isinstance(e, dict):
        e[name] = value
    else:
        setattr(e, name, value)


def apply_relevance_rules(claim: str, evidences: list, today: date | None = None) -> list:
    """Downgrade evidence that does not address the claim itself to CONTEXT, with a note explaining why.
    Mutates and returns `evidences`."""
    today = today or date.today()
    time_sensitive = is_time_sensitive(claim)
    death_claim = is_death_claim(claim)
    for e in evidences:
        rel = _get(e, "relationship")
        if rel not in ("SUPPORTS", "REFUTES", "PARTIAL"):
            continue
        if _get(e, "scope", "direct") == "related":
            _set(e, "relationship", "CONTEXT")
            _set(e, "note", "This source is about a related video, post, rumour or event, not the claim itself, "
                            "so it was not counted for or against the claim.")
            continue
        if not time_sensitive:
            continue
        published = parse_date(_get(e, "published_date"), today)
        if published is None:
            _set(e, "note", "Publication date unknown; this source may be out of date for a claim about the current situation.")
        elif (today - published).days > RECENT_DAYS and not (death_claim and rel == "SUPPORTS"):
            _set(e, "relationship", "CONTEXT")
            _set(e, "note", f"Published {published.isoformat()}, more than {RECENT_DAYS} days before this check. "
                            "It cannot show the current situation, so it was not counted.")
    return evidences


_REL_RANK = {"SUPPORTS": 0, "REFUTES": 0, "PARTIAL": 0, "CONTEXT": 1, "IRRELEVANT": 2}


def rank_evidence(evidences: list, today: date | None = None) -> list:
    """Most useful first: deciding evidence, then context, then irrelevant; within a group by weight,
    then by most recent publication date."""
    today = today or date.today()

    def key(e):
        d = parse_date(_get(e, "published_date"), today)
        return (_REL_RANK.get(_get(e, "relationship"), 3), -evidence_weight(e), -(d.toordinal() if d else 0))

    return sorted(evidences, key=key)
