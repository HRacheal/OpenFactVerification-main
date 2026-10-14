from factcheck.utils.logger import CustomLogger
from factcheck.utils.json_utils import parse_llm_json, expect_dict, LLMOutputError

logger = CustomLogger(__name__).getlog()


class Checkworthy:
    def __init__(self, llm_client, prompt):
        """Initialize the Checkworthy class

        Args:
            llm_client (BaseClient): The LLM client used for identifying checkworthiness of claims.
            prompt (BasePrompt): The prompt used for identifying checkworthiness of claims.
        """
        self.llm_client = llm_client
        self.prompt = prompt

    def identify_checkworthiness(
        self,
        texts: list[str],
        num_retries: int = 3,
        prompt: str = None,
        context: str = "None given",
        output_language: str = "English",
    ):
        """Decide which claims are worth fact-checking.

        If the LLM cannot return a valid answer, all claims are treated as checkworthy (so nothing is
        silently dropped) and the reason says so.

        Returns:
            (list[str], dict[str, str]): checkworthy claims, and a reason per claim
        """
        if not texts:
            return [], {}
        joint_texts = "\n".join([str(i + 1) + ". " + j for i, j in enumerate(texts)])
        template = prompt or self.prompt.checkworthy_prompt
        user_input = template.format(texts=joint_texts, context=context, output_language=output_language)

        messages = self.llm_client.construct_message_list([user_input])
        for i in range(num_retries):
            response = self.llm_client.call(messages, num_retries=1, seed=42 + i)
            try:
                raw = expect_dict(parse_llm_json(response))
                answers = {k.strip(): v for k, v in raw.items() if isinstance(v, str)}
                if not all(v.strip().startswith(("Yes", "No")) for v in answers.values()):
                    raise LLMOutputError("Every value must start with 'Yes' or 'No'.")
                claim2checkworthy = {}
                for t in texts:
                    if t in answers:
                        claim2checkworthy[t] = answers[t].strip()
                    else:
                        claim2checkworthy[t] = "Yes (Not assessed by the model; checked to be safe.)"
                checkworthy_claims = [t for t in texts if claim2checkworthy[t].startswith("Yes")]
                return checkworthy_claims, claim2checkworthy
            except LLMOutputError as e:
                logger.error(f"====== Checkworthiness parse error: {e}")
        reason = "Yes (Checkworthiness could not be assessed automatically; checked to be safe.)"
        return list(texts), {t: reason for t in texts}
