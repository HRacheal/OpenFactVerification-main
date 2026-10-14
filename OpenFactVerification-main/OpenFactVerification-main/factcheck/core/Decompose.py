import re

from factcheck.utils.logger import CustomLogger
from factcheck.utils.json_utils import parse_llm_json, expect_dict, expect_str_list, LLMOutputError

logger = CustomLogger(__name__).getlog()

# Sentence boundary: ., ! or ? followed by whitespace. Used only as a fallback when the LLM fails.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


class Decompose:
    def __init__(self, llm_client, prompt):
        """Initialize the Decompose class

        Args:
            llm_client (BaseClient): The LLM client used for decomposing documents into claims.
            prompt (BasePrompt): The prompt used for fact checking.
        """
        self.llm_client = llm_client
        self.prompt = prompt
        self.doc2sent = self._regex_doc2sent

    def _regex_doc2sent(self, text: str):
        """Split the document into sentences (dependency-free fallback; works for English/Swahili/Luganda)."""
        sentences = _SENT_SPLIT_RE.split(text)
        return [s.strip() for s in sentences if len(s.strip()) >= 3 and not s.strip().endswith("?")]

    def getclaims(self, doc: str, num_retries: int = 3, prompt: str = None, context: str = "None given") -> list[str]:
        """Use the LLM to decompose a document into claims.

        Returns:
            list: a list of claims (possibly empty if the text contains no factual assertions)
        """
        template = prompt or self.prompt.decompose_prompt
        user_input = template.format(doc=doc, context=context).strip()

        messages = self.llm_client.construct_message_list([user_input])
        for i in range(num_retries):
            response = self.llm_client.call(messages=messages, num_retries=1, seed=42 + i)
            try:
                data = expect_dict(parse_llm_json(response), ["claims"])
                claims = expect_str_list(data["claims"], "claims")
                # de-duplicate while keeping order
                return list(dict.fromkeys(claims))
            except LLMOutputError as e:
                logger.error(f"Parse LLM response error: {e}")
        logger.info("The LLM did not return a valid claim list; falling back to sentence splitting.")
        return self.doc2sent(doc)

    def restore_claims(self, doc: str, claims: list, num_retries: int = 3, prompt: str = None) -> dict[str, dict]:
        """Use the LLM to map claims back to the document

        Returns:
            dict: a dictionary of claims and their corresponding text spans and start/end indices.
        """

        def restore(claim2doc):
            claim2doc_detail = {}
            flag = True
            for claim, sent in claim2doc.items():
                if not isinstance(sent, str):
                    flag = False
                    continue
                st = doc.find(sent)
                if st != -1:
                    claim2doc_detail[claim] = {"text": sent, "start": st, "end": st + len(sent)}
                else:
                    flag = False

            cur_pos = -1
            for k, v in claim2doc_detail.items():
                if v["start"] < cur_pos + 1 and v["end"] > cur_pos:
                    v["start"] = cur_pos + 1
                    flag = False
                elif v["start"] < cur_pos + 1 and v["end"] <= cur_pos:
                    v["start"] = v["end"]  # temporarily ignore this span
                    flag = False
                elif v["start"] > cur_pos + 1:
                    v["start"] = cur_pos + 1
                    flag = False
                v["text"] = doc[v["start"] : v["end"]]
                claim2doc_detail[k] = v
                cur_pos = v["end"]

            return claim2doc_detail, flag

        template = prompt or self.prompt.restore_prompt
        user_input = template.format(doc=doc, claims=claims).strip()
        messages = self.llm_client.construct_message_list([user_input])

        tmp_restore = {}
        for i in range(num_retries):
            response = self.llm_client.call(messages=messages, num_retries=1, seed=42 + i)
            try:
                claim2doc = expect_dict(parse_llm_json(response))
                if len(claim2doc) != len(claims):
                    raise LLMOutputError("Number of restored spans does not match number of claims.")
                claim2doc_detail, flag = restore(claim2doc)
                if flag:
                    return self._ensure_all(claim2doc_detail, claims)
                tmp_restore = claim2doc_detail
            except LLMOutputError as e:
                logger.error(f"Parse LLM response error: {e}")

        return self._ensure_all(tmp_restore, claims)

    @staticmethod
    def _ensure_all(claim2doc_detail: dict, claims: list) -> dict:
        """Guarantee every extracted claim appears in the result (spans may be empty), in claim order."""
        result = {}
        for c in claims:
            result[c] = claim2doc_detail.get(c, {"text": "", "start": 0, "end": 0})
        return result
