import concurrent.futures
import time
from datetime import datetime, timezone

from dataclasses import asdict
from factcheck.utils.llmclient import CLIENTS, model2client
from factcheck.utils.prompt import prompt_mapper
from factcheck.utils.logger import CustomLogger
from factcheck.utils.api_config import load_api_config
from factcheck.utils.json_utils import parse_llm_json, expect_dict, LLMOutputError
from factcheck.utils.data_class import (
    PipelineUsage,
    FactCheckOutput,
    ClaimDetail,
    FCSummary,
    VERDICTS,
    VERDICT_NOT_CHECKABLE,
    VERDICT_INSUFFICIENT,
)
from factcheck.core import (
    Decompose,
    Checkworthy,
    QueryGenerator,
    retriever_mapper,
    ClaimVerify,
)
from factcheck.core.Verdict import (
    decide_verdict,
    evidence_notes,
    apply_relevance_rules,
    rank_evidence,
    is_time_sensitive,
    VERDICT_LABELS,
    VERDICT_HEADLINES,
    RECENT_DAYS,
)

logger = CustomLogger(__name__).getlog()

LANGUAGE_NAMES = {"en": "English", "sw": "Swahili (Kiswahili)", "lg": "Luganda"}

try:
    import tiktoken

    _ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception:  # tiktoken downloads its vocabulary on first use; not essential
    _ENCODING = None


def count_tokens(text: str) -> int:
    if _ENCODING is not None:
        try:
            return len(_ENCODING.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)


class FactCheck:
    def __init__(
        self,
        default_model: str = "gpt-4.1-mini",
        client: str = None,
        prompt: str = "truthlens_prompt",
        retriever: str = "serper",
        decompose_model: str = None,
        checkworthy_model: str = None,
        query_generator_model: str = None,
        evidence_retrieval_model: str = None,
        claim_verify_model: str = None,
        api_config: dict = None,
        num_seed_retries: int = 3,
    ):
        self.prompt = prompt_mapper(prompt_name=prompt)
        self.load_config(api_config=api_config)
        self.default_model = default_model
        self.client_name = client

        step_models = {
            "decompose_model": decompose_model,
            "checkworthy_model": checkworthy_model,
            "query_generator_model": query_generator_model,
            "evidence_retrieval_model": evidence_retrieval_model,
            "claim_verify_model": claim_verify_model,
        }

        for key, _model_name in step_models.items():
            _model_name = default_model if _model_name is None else _model_name
            logger.info(f"== Init {key} with model: {_model_name}")
            LLMClient = CLIENTS[client] if client is not None else model2client(_model_name)
            setattr(self, key, LLMClient(model=_model_name, api_config=self.api_config))

        # sub-modules
        self.decomposer = Decompose(llm_client=self.decompose_model, prompt=self.prompt)
        self.checkworthy = Checkworthy(llm_client=self.checkworthy_model, prompt=self.prompt)
        self.query_generator = QueryGenerator(llm_client=self.query_generator_model, prompt=self.prompt)
        self.evidence_crawler = retriever_mapper(retriever_name=retriever)(
            llm_client=self.evidence_retrieval_model, api_config=self.api_config
        )
        self.claimverify = ClaimVerify(llm_client=self.claim_verify_model, prompt=self.prompt)
        self.attr_list = ["decomposer", "checkworthy", "query_generator", "evidence_crawler", "claimverify"]
        self.num_seed_retries = num_seed_retries

        logger.info("===Sub-modules Init Finished===")

    def load_config(self, api_config: dict) -> None:
        self.api_config = load_api_config(api_config)

    @property
    def vision_client(self):
        """The LLM client used for image/video-frame analysis (same provider/model as the pipeline)."""
        return self.decompose_model

    @staticmethod
    def build_context(country: str = None, language: str = "auto", today: str = None) -> str:
        today = today or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        lang = LANGUAGE_NAMES.get(language, "detect automatically")
        return f"Country/region: {country or 'not specified'}. Input language: {lang}. Today's date: {today}."

    def check_text(
        self,
        raw_text: str,
        country: str = None,
        language: str = "auto",
        output_language: str = "en",
        progress=None,
    ):
        """Run the full fact-checking pipeline on text.

        Args:
            raw_text: the content to check (already extracted from URL/image/audio/video if needed)
            country: optional country/region context, e.g. "Uganda"
            language: input language code ("en", "sw", "lg") or "auto"
            output_language: language for explanations ("en", "sw", "lg")
            progress: optional callable(stage: str, message: str)
        """

        def report(stage, message):
            logger.info(f"== {stage}: {message}")
            if progress:
                progress(stage, message)

        self._reset_usage()
        out_lang = LANGUAGE_NAMES.get(output_language, "English")
        today = datetime.now(timezone.utc).date()
        context = self.build_context(country, language, today=today.isoformat())
        if hasattr(self.evidence_crawler, "set_context"):
            self.evidence_crawler.set_context(country=country, language=language)
        ctx = {"country": country or "", "language": language, "output_language": output_language}

        st_time = time.time()
        report("claims", "Extracting claims from the content")
        claims = self.decomposer.getclaims(doc=raw_text, num_retries=self.num_seed_retries, context=context)
        if not claims:
            return self._finalize_factcheck(
                raw_text, [], ctx, warnings=["No verifiable factual claims were found in the submitted content."]
            )

        report("checkworthy", f"Found {len(claims)} claim(s); deciding which can be verified")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_claim2doc = executor.submit(
                self.decomposer.restore_claims, doc=raw_text, claims=claims, num_retries=self.num_seed_retries
            )
            future_checkworthy = executor.submit(
                self.checkworthy.identify_checkworthiness,
                claims,
                num_retries=self.num_seed_retries,
                context=context,
                output_language=out_lang,
            )
            future_queries = executor.submit(self.query_generator.generate_query, claims=claims, context=context)
            claim2doc = future_claim2doc.result()
            checkworthy_claims, claim2checkworthy = future_checkworthy.result()
            claim_queries_dict = future_queries.result()

        checkworthy_set = set(checkworthy_claims)
        claim_queries_dict = {k: v for k, v in claim_queries_dict.items() if k in checkworthy_set}

        claim_evidences_dict, claim_verifications_dict = {}, {}
        if claim_queries_dict:
            report("search", f"Searching the web for evidence on {len(claim_queries_dict)} claim(s)")
            if hasattr(self.evidence_crawler, "set_recent_claims"):
                # Claims about a current situation (alive/dead, detained ...) also get recent-news searches.
                self.evidence_crawler.set_recent_claims({c for c in claim_queries_dict if is_time_sensitive(c)})
            claim_evidences_dict = self.evidence_crawler.retrieve_evidence(claim_queries_dict=claim_queries_dict)
            n_evidence = sum(len(v) for v in claim_evidences_dict.values())
            report("verify", f"Assessing {n_evidence} evidence excerpt(s)")
            claim_verifications_dict = self.claimverify.verify_claims(
                claim_evidences_dict=claim_evidences_dict, context=context, output_language=out_lang
            )

        report("verdict", "Applying verdict rules and writing explanations")
        claim_detail = self._merge_claim_details(
            claim2doc=claim2doc,
            claim2checkworthy=claim2checkworthy,
            claim2queries=claim_queries_dict,
            claim2verifications=claim_verifications_dict,
            today=today,
        )
        self._add_explanations(claim_detail, out_lang, today)
        logger.info(f"== Done in {time.time() - st_time:.1f}s")
        return self._finalize_factcheck(raw_text, claim_detail, ctx)

    def _get_usage(self):
        return PipelineUsage(**{attr: getattr(self, attr).llm_client.usage for attr in self.attr_list})

    def _reset_usage(self):
        for attr in self.attr_list:
            getattr(self, attr).llm_client.reset_usage()

    def _merge_claim_details(
        self, claim2doc: dict, claim2checkworthy: dict, claim2queries: dict, claim2verifications: dict, today=None, **_
    ) -> list[ClaimDetail]:
        claim_details = []
        for i, (claim, origin) in enumerate(claim2doc.items()):
            common = dict(
                id=i,
                claim=claim,
                checkworthy_reason=claim2checkworthy.get(claim, "No reason provided."),
                origin_text=origin["text"],
                start=origin["start"],
                end=origin["end"],
            )
            if claim in claim2verifications:
                evidences = apply_relevance_rules(claim, claim2verifications.get(claim, []), today)
                evidences = rank_evidence(evidences, today)  # most useful sources first
                verdict, reason, strength = decide_verdict(evidences)
                labels = [e.relationship for e in evidences]
                sup, ref = labels.count("SUPPORTS"), labels.count("REFUTES")
                factuality = "No evidence found." if sup + ref == 0 else sup / (sup + ref)
                notes = evidence_notes(claim, evidences)
                if not evidences:
                    notes.insert(0, "The search returned no usable results for this claim.")
                if is_time_sensitive(claim):
                    notes.insert(0, f"This claim is about a situation that can change, so only reports from the last "
                                    f"{RECENT_DAYS} days were counted for or against it. Older reports are shown as context.")
                claim_obj = ClaimDetail(
                    **common,
                    checkworthy=True,
                    queries=claim2queries.get(claim, []),
                    evidences=evidences,
                    factuality=factuality,
                    verdict=verdict,
                    verdict_reason=reason,
                    strength=strength,
                    notes=notes,
                )
            else:
                claim_obj = ClaimDetail(
                    **common,
                    checkworthy=False,
                    queries=[],
                    evidences=[],
                    factuality="Nothing to check.",
                    verdict=VERDICT_NOT_CHECKABLE,
                    verdict_reason="This statement was judged not objectively verifiable (see reason), so it was not fact-checked.",
                )
            claim_details.append(claim_obj)
        return claim_details

    def _add_explanations(self, claim_details: list[ClaimDetail], out_lang: str, today=None) -> None:
        """Ask the LLM for a short plain-language explanation of each (rule-decided) verdict."""
        targets = [c for c in claim_details if c.checkworthy]
        template = getattr(self.prompt, "summary_prompt", None)
        for c in targets:
            c.explanation = self._fallback_explanation(c)
        if not template or not targets:
            return
        stance_words = {"SUPPORTS": "supports the claim", "REFUTES": "contradicts the claim",
                        "PARTIAL": "partly supports the claim", "CONTEXT": "context only (not counted)"}
        prompts = []
        for c in targets:
            lines = [
                f"- {stance_words.get(e.relationship, e.relationship)}; "
                f"{'about a related video/post/rumour/event, not the claim itself' if e.scope == 'related' else 'about the claim itself'}; "
                f"{e.publisher or e.domain or 'unknown source'}; {e.published_date or 'date unknown'}; {e.reasoning}"
                f"{'; caveat: ' + e.note if e.note else ''}"
                for e in c.evidences  # already ranked: deciding evidence first
                if e.relationship != "IRRELEVANT"
            ] or ["- (no relevant evidence was found)"]
            prompts.append(
                template.format(
                    claim=c.claim,
                    today=(today or datetime.now(timezone.utc).date()).isoformat(),
                    verdict=f"{VERDICT_LABELS[c.verdict]}: {VERDICT_HEADLINES[c.verdict]} Rule applied: {c.verdict_reason}",
                    evidence="\n".join(lines[:12]),
                    output_language=out_lang,
                )
            )
        try:
            responses = self.claimverify.llm_client.multi_call(self.claimverify.llm_client.construct_message_list(prompts))
        except Exception as e:  # explanations are helpful but not essential
            logger.warning(f"Explanation generation failed: {type(e).__name__}")
            return
        for c, r in zip(targets, responses):
            try:
                text = expect_dict(parse_llm_json(r), ["explanation"])["explanation"]
                if isinstance(text, str) and text.strip():
                    c.explanation = text.strip()
            except LLMOutputError:
                pass

    @staticmethod
    def _fallback_explanation(c: ClaimDetail) -> str:
        """Plain-language explanation used when the AI explanation is unavailable."""
        s = c.strength or {}
        related = sum(1 for e in c.evidences if e.scope == "related" or (e.note and e.relationship == "CONTEXT"))
        parts = [c.verdict_reason]
        if s.get("support_sources") or s.get("contradict_sources"):
            parts.append(f"{s.get('support_sources', 0)} independent source(s) support it and "
                         f"{s.get('contradict_sources', 0)} contradict it.")
        if related:
            parts.append(f"{related} report(s) concern related videos, posts or older events and were not counted.")
        return " ".join(parts)

    def _finalize_factcheck(
        self, raw_text: str, claim_detail: list[ClaimDetail], context: dict, warnings: list = None, return_dict: bool = True
    ):
        verified_claims = [c for c in claim_detail if not isinstance(c.factuality, str)]
        num_claims = len(claim_detail)
        num_checkworthy_claims = len([c for c in claim_detail if c.checkworthy])
        num_verified_claims = len(verified_claims)
        num_supported_claims = len([c for c in verified_claims if c.factuality == 1])
        num_refuted_claims = len([c for c in verified_claims if c.factuality == 0])
        factuality = sum(c.factuality for c in verified_claims) / num_verified_claims if num_verified_claims else 0
        verdict_counts = {v: 0 for v in VERDICTS}
        for c in claim_detail:
            verdict_counts[c.verdict] = verdict_counts.get(c.verdict, 0) + 1

        summary = FCSummary(
            num_claims,
            num_checkworthy_claims,
            num_verified_claims,
            num_supported_claims,
            num_refuted_claims,
            num_verified_claims - num_supported_claims - num_refuted_claims,
            factuality,
            verdict_counts,
        )
        output = FactCheckOutput(
            raw_text=raw_text,
            token_count=count_tokens(raw_text),
            usage=self._get_usage(),
            claim_detail=claim_detail,
            summary=summary,
            context=context or {},
            verified_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            warnings=list(warnings or []),
        )
        if not output.attribute_check():
            raise ValueError("Output attribute check failed.")
        return asdict(output) if return_dict else output
