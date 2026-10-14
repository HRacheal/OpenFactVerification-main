"""Kweli Lens AI prompts (provider-neutral: used for both OpenAI and Claude).

Based on Loki's ChatGPT prompts, extended for:
- multilingual input (English, Swahili, Luganda) without silent translation,
- preserving numbers, dates, units, qualifiers and attribution,
- regional (East African) context in search queries,
- grounded, quote-backed evidence assessment with a CONTEXT/PARTIAL stance.

Every prompt asks for JSON only. Literal braces are doubled for str.format().
Placeholders: {doc} {claims} {texts} {claim} {context} {output_language} {evidence}
"""

decompose_prompt = """You extract verifiable factual claims from content submitted to a fact-checking desk.

Context supplied by the journalist: {context}

Rules:
1. Return a JSON object with a single key "claims" whose value is a list of strings.
2. Each claim must be one self-contained factual assertion that appears in the text. Replace pronouns
   ("he", "it", "the ministry") with the full name used in the text when the text makes it clear.
3. Write each claim in the SAME LANGUAGE as the text it comes from. Do NOT translate.
4. Preserve exactly: names, numbers, percentages, currencies, units, dates, places and qualifiers
   such as "about", "up to", "reportedly", "allegedly", "according to X". Never round or convert numbers.
5. Keep attribution: if the text says "the minister said X", the claim is "The minister said X" unless X
   itself is stated as fact by the author.
6. Do NOT turn questions, opinions, predictions, jokes or uncertain statements into asserted facts.
   If a sentence is a question or an opinion, either skip it or keep its hedged form
   (e.g. "The post asks whether ...", "The author believes ...").
7. Split compound statements into separate claims (e.g. a policy introduction and its effect).
8. If the text contains no factual assertions, return {{"claims": []}}.

Example
Text: Kenya introduced a new education policy in 2026, reducing university fees by 50%.
Output: {{"claims": ["Kenya introduced a new education policy in 2026.", "Kenya's new 2026 education policy reduced university fees by 50%."]}}

Example
Text: Serikali ya Uganda imesema bei ya mafuta itapanda kwa asilimia 10 mwezi ujao. Je, hii ni kweli?
Output: {{"claims": ["Serikali ya Uganda imesema bei ya mafuta itapanda kwa asilimia 10 mwezi ujao."]}}

Text: {doc}
Output (JSON only):
"""

restore_prompt = """Given a text and a list of facts derived from the text, split the text into chunks that derive each fact.
For each fact, find the corresponding continuous span in the original text that contains the information
to derive the fact. Return a JSON object where the keys are the facts (copied exactly) and the values are
the spans copied exactly from the original text, in the original language. The returned spans, in order,
should concatenate to the full original text.

Example
Text: Mary is a five-year old girl, she likes playing piano and she doesn't like cookies.
Facts: ["Mary is a five-year old girl.", "Mary likes playing piano.", "Mary doesn't like cookies."]
Output:
{{"Mary is a five-year old girl.": "Mary is a five-year old girl,",
"Mary likes playing piano.": "she likes playing piano",
"Mary doesn't like cookies.": "and she doesn't like cookies."}}

Text: {doc}
Facts: {claims}
Output (JSON only):
"""

checkworthy_prompt = """Decide, for each statement, whether it contains factual information that a journalist could
objectively verify against evidence, regardless of whether it is true.

Context supplied by the journalist: {context}

Guidelines:
1. Opinions, value judgements, predictions and rhetorical questions are NOT verifiable ("No").
2. Statements with unclear references (e.g. "he is a professor" with no name) are NOT verifiable ("No").
3. Statements that assert a checkable fact, statistic, date, quote or event ARE verifiable ("Yes"),
   even if broad or likely false.
4. If a statement is verifiable only with missing context (no date, no place), answer "Yes" but say
   which context is missing.

Return a JSON object: each statement copied exactly as the key, and as the value a string that starts with
"Yes" or "No" followed by a short reason in parentheses written in {output_language}.

Example
Statements:
1. Uganda's population census was conducted in 2024.
2. This government is the worst ever.
3. He was arrested last week.
Output:
{{
  "Uganda's population census was conducted in 2024.": "Yes (A specific, checkable event and date.)",
  "This government is the worst ever.": "No (A subjective opinion, not a verifiable fact.)",
  "He was arrested last week.": "No (It is unclear who 'he' is.)"
}}

Statements:
{texts}

Output (JSON only):
"""

qgen_prompt = """You write web search queries that a fact-checker would use to find evidence about a claim.

Claim: {claim}
Context supplied by the journalist: {context}

Rules:
1. Return a JSON object with a single key "Questions" whose value is a list of 2 to 4 short search queries.
2. Include the key entities, places, numbers and dates from the claim. Add the country from the context
   when the claim does not name one.
3. If the claim is not in English, include at least one query in the claim's original language AND at least
   one in English, preserving names and numbers exactly.
4. For statistics, add a query aimed at the original source (official statistics bureau, ministry report,
   dataset or survey) and the time period.
5. Where useful, add one query aimed at existing fact-checks (e.g. "<topic> fact check").
6. Do not answer the claim. Do not invent facts that are not in the claim.

Example
Claim: Kenya's new 2026 education policy reduced university fees by 50%.
Context: Country: Kenya
Output: {{"Questions": ["Kenya 2026 education policy university fees reduced 50%", "Kenya university fees 2026 Ministry of Education announcement", "Kenya university fees cut 50 percent fact check"]}}

Output (JSON only):
"""

verify_prompt = """You assess ONE piece of evidence against ONE claim for a professional fact-checker.

[claim]: {claim}
[context]: {context}
[evidence source]: {source}
[evidence text]: {evidence}

Decide the relationship using ONLY the evidence text above, not your own background knowledge:
- "SUPPORTS": the evidence clearly states something that confirms the claim.
- "REFUTES": the evidence clearly states something that conflicts with the claim.
- "PARTIAL": the evidence confirms part of the claim but conflicts with another part (e.g. right event, wrong number).
- "CONTEXT": the evidence is on-topic and useful background but does not confirm or contradict the claim
  (e.g. different year, different population, different measure, or only mentions the topic).
- "IRRELEVANT": the evidence is off-topic or too vague to use.

Important:
- Search snippets may be truncated. Do not treat a vague or partial snippet as conclusive.
- For numbers: check that the evidence refers to the same place, population, time period and measurement.
  If it does not, use "CONTEXT", not "REFUTES".
- Absence of a fact in the evidence is NOT a refutation.
- "basis" is "explicit" if the evidence directly states it, "inferred" if you had to reason beyond what it says.
- "quote" must be a short passage copied EXACTLY from the evidence text (in its original language) that your
  judgement relies on. Use "" if there is none.
- Write "reasoning" in {output_language}, in one to three plain sentences.

Return JSON only with keys "reasoning", "relationship", "basis", "quote".

Example
[claim]: Kampala is the capital of Uganda.
[evidence text]: Kampala is the capital and largest city of Uganda.
Output: {{"reasoning": "The evidence directly states that Kampala is Uganda's capital.", "relationship": "SUPPORTS", "basis": "explicit", "quote": "Kampala is the capital and largest city of Uganda."}}

Output (JSON only):
"""

summary_prompt = """You write a short, neutral explanation of a fact-check result for a journalist.

Claim: {claim}
Verdict (already decided by fixed rules — do not change it): {verdict}
Evidence assessments (stance, source, reasoning):
{evidence}

Write 2 to 4 plain sentences in {output_language} that explain why this verdict was reached, mention any
disagreement between sources, and state what is still uncertain or missing. Only use the information above.
Do not add facts. Keep names, numbers and dates exactly as they appear.

Return JSON only: {{"explanation": "..."}}
"""


class TruthLensPrompt:
    decompose_prompt = decompose_prompt
    restore_prompt = restore_prompt
    checkworthy_prompt = checkworthy_prompt
    qgen_prompt = qgen_prompt
    verify_prompt = verify_prompt
    summary_prompt = summary_prompt
