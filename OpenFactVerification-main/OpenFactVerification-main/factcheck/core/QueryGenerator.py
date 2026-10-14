from factcheck.utils.logger import CustomLogger
from factcheck.utils.json_utils import parse_llm_json, expect_dict, LLMOutputError

logger = CustomLogger(__name__).getlog()


class QueryGenerator:
    def __init__(self, llm_client, prompt, max_query_per_claim: int = 4):
        """Initialize the QueryGenerator class

        Args:
            llm_client (BaseClient): The LLM client used for generating questions.
            prompt (BasePrompt): The prompt used for generating questions.
            max_query_per_claim: cap on search queries per claim (each query costs one search request).
        """
        self.llm_client = llm_client
        self.prompt = prompt
        self.max_query_per_claim = max_query_per_claim

    @staticmethod
    def _parse_questions(response: str) -> list[str]:
        data = expect_dict(parse_llm_json(response))
        # The original Loki prompt examples used both "Questions" and "Question".
        questions = data.get("Questions", data.get("Question"))
        if not isinstance(questions, list):
            raise LLMOutputError("Missing 'Questions' list.")
        cleaned = [q.strip() for q in questions if isinstance(q, str) and q.strip()]
        if not cleaned:
            raise LLMOutputError("Empty question list.")
        return cleaned

    def generate_query(
        self, claims: list[str], generating_time: int = 3, prompt: str = None, context: str = "None given"
    ) -> dict[str, list[str]]:
        """Generate search queries for the given claims.

        Returns:
            dict: claim -> list of queries. The claim itself is always the first query.
        """
        generated_questions = [[] for _ in claims]
        attempts = 0
        template = prompt or self.prompt.qgen_prompt
        messages_list = [template.format(claim=claim, context=context) for claim in claims]

        while (attempts < generating_time) and any(q == [] for q in generated_questions):
            _indices = [i for i, q in enumerate(generated_questions) if q == []]
            _messages = [messages_list[i] for i in _indices]

            _message_list = self.llm_client.construct_message_list(_messages)
            _response_list = self.llm_client.multi_call(_message_list)

            for _response, _index in zip(_response_list, _indices):
                try:
                    generated_questions[_index] = self._parse_questions(_response)
                except LLMOutputError as e:
                    logger.info(f"Warning: query generation parse fail ({e}), retry {attempts}.")
            attempts += 1

        claim_query_dict = {}
        for claim, questions in zip(claims, generated_questions):
            queries = [claim] + [q for q in questions if q != claim]
            claim_query_dict[claim] = list(dict.fromkeys(queries))[: self.max_query_per_claim]
        return claim_query_dict
