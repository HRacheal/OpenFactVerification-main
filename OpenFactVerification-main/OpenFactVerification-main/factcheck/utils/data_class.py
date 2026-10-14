from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class TokenUsage:
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: Optional[int] = 0


@dataclass
class PipelineUsage:
    decomposer: TokenUsage = None
    checkworthy: TokenUsage = None
    query_generator: TokenUsage = None
    evidence_crawler: TokenUsage = None
    claimverify: TokenUsage = None


# Stance labels produced by the evidence assessor.
STANCES = ("SUPPORTS", "REFUTES", "PARTIAL", "CONTEXT", "IRRELEVANT")

# Kweli Lens claim-level verdicts.
VERDICT_SUPPORTED = "SUPPORTED"
VERDICT_CONTRADICTED = "CONTRADICTED"
VERDICT_MIXED = "MIXED"
VERDICT_INSUFFICIENT = "INSUFFICIENT_EVIDENCE"
VERDICT_NOT_CHECKABLE = "NOT_CHECKABLE"
VERDICTS = (VERDICT_SUPPORTED, VERDICT_CONTRADICTED, VERDICT_MIXED, VERDICT_INSUFFICIENT, VERDICT_NOT_CHECKABLE)


@dataclass
class Evidence:
    """One evidence excerpt and its assessment against one claim.

    Required: claim, text, url, reasoning, relationship. The rest is source metadata that may be
    unavailable; missing values stay None/"" and are never invented.
    """

    claim: str = None
    text: str = None  # evidence excerpt, original language
    url: str = None  # "" when the provider returned no link (e.g. some answer boxes)
    reasoning: str = None
    relationship: str = None  # one of STANCES
    basis: str = "inferred"  # "explicit" | "inferred"
    quote: str = ""  # exact passage from `text` the assessment relies on (verified to exist)
    title: str = ""
    domain: str = ""
    publisher: str = ""
    published_date: Optional[str] = None  # as reported by the search provider or page metadata
    retrieved_at: Optional[str] = None  # ISO timestamp when Kweli Lens retrieved it
    source_type: str = "unknown"  # official | academic | fact_checker | news | social | unknown
    full_text_fetched: bool = False  # False => assessment based on the search snippet only
    query: str = ""

    def attribute_check(self) -> bool:
        for name in ("claim", "text", "url", "reasoning", "relationship"):
            if getattr(self, name) is None:
                print(f"Field {name} is None")
                return False
        return True


@dataclass
class ClaimDetail:
    """Dataclass to store the details of a claim.

    factuality (legacy Loki field): "Nothing to check.", "No evidence found.", or float in [0, 1]
    verdict: one of VERDICTS (Kweli Lens)
    """

    id: int = None
    claim: str = None
    checkworthy: bool = None
    checkworthy_reason: str = None
    origin_text: str = None
    start: int = None
    end: int = None
    queries: List[str] = None
    evidences: List[Evidence] = None
    factuality: any = None
    verdict: str = VERDICT_NOT_CHECKABLE
    verdict_reason: str = ""  # deterministic rule explanation
    explanation: str = ""  # plain-language explanation (may be AI-written, in the output language)
    strength: dict = field(default_factory=dict)  # evidence weights used by the verdict rules
    notes: List[str] = field(default_factory=list)  # limitations / warnings

    def attribute_check(self) -> bool:
        for name in ("id", "claim", "checkworthy", "checkworthy_reason", "origin_text", "start", "end", "queries", "evidences", "factuality"):
            if getattr(self, name) is None:
                print(f"Field {name} is None")
                return False
        for evidence in self.evidences:
            if not evidence.attribute_check():
                return False
        return True


@dataclass
class FCSummary:
    num_claims: int = None
    num_checkworthy_claims: int = None
    num_verified_claims: int = None
    num_supported_claims: int = None
    num_refuted_claims: int = None
    num_controversial_claims: int = None
    factuality: float = None
    verdict_counts: dict = field(default_factory=dict)

    def attribute_check(self) -> bool:
        for f in self.__dataclass_fields__.values():
            if getattr(self, f.name) is None:
                print(f"Field {f.name} is None")
                return False
        return True


@dataclass
class FactCheckOutput:
    raw_text: str = None
    token_count: int = None
    usage: PipelineUsage = None
    claim_detail: List[ClaimDetail] = None
    summary: FCSummary = None
    context: dict = field(default_factory=dict)  # country, language, output_language
    verified_at: str = ""
    warnings: List[str] = field(default_factory=list)

    def attribute_check(self) -> bool:
        for name in ("raw_text", "token_count", "usage", "claim_detail", "summary"):
            if getattr(self, name) is None:
                print(f"Field {name} is None")
                return False
        for claim in self.claim_detail:
            if not claim.attribute_check():
                return False
        return self.summary.attribute_check()
