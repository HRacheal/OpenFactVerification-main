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
7. If the claim is about a situation that can change (whether someone is alive or dead, detained, ill, in
   office, missing), search for the person's LATEST news and official statements (e.g. "<name> latest news",
   "<name> today", "<office> official statement"), not only for the rumour. Do not add a year that the claim
   does not state.

Example
Claim: President Museveni has died.
Context: Country: Uganda
Output: {{"Questions": ["President Museveni latest news", "Museveni State House Uganda statement", "Museveni death rumour fact check"]}}

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
- Words in common are not enough: the evidence must be about the SAME fact as the claim.
- "scope" is "direct" if the evidence speaks about the claim itself, or "related" if it is about something
  connected but different: a video, photo, post or rumour about the claim (e.g. "a video claiming X is fake"),
  an earlier or different event, or a different person or place. A report that a video or post is fake or
  manipulated shows that the MEDIA is false; it does not by itself show whether the underlying claim is true
  or false. Such evidence is "related" and normally "CONTEXT", unless it ALSO directly states the fact
  (e.g. "X is alive and addressed a rally on 14 October"), in which case it is "direct".
- Time matters. Use today's date from [context] and the publication date in [evidence source]. For a claim about
  the current situation (someone is dead, arrested, ill, in office ...), evidence only confirms or contradicts it
  if it describes the situation at or after the time the claim is about. Older evidence is "CONTEXT".
  Mention the relevant date in your reasoning.
- Write "reasoning" in {output_language}, in one to three plain sentences.

Return JSON only with keys "reasoning", "relationship", "basis", "quote", "scope".

Example
[claim]: Kampala is the capital of Uganda.
[evidence text]: Kampala is the capital and largest city of Uganda.
Output: {{"reasoning": "The evidence directly states that Kampala is Uganda's capital.", "relationship": "SUPPORTS", "basis": "explicit", "quote": "Kampala is the capital and largest city of Uganda.", "scope": "direct"}}

Example
[claim]: The Kenyan Health Minister has died.
[evidence text]: A video circulating on TikTok that claims to show the minister's funeral is fake. The footage is from a 2019 funeral in Nigeria.
Output: {{"reasoning": "The evidence shows that a video is fake, but it does not say whether the minister is alive or dead.", "relationship": "CONTEXT", "basis": "explicit", "quote": "claims to show the minister's funeral is fake", "scope": "related"}}

Output (JSON only):
"""

summary_prompt = """You write a short, neutral explanation of a fact-check result for an ordinary reader with no
fact-checking background.

Claim: {claim}
Date of this check: {today}
Verdict (already decided by fixed rules; do not change or soften it): {verdict}
Evidence assessments (stance; scope; publisher; publication date; what the source says; caveat):
{evidence}

Write 2 to 4 short sentences in simple {output_language}:
1. Why this verdict: what the most important evidence shows, naming the publisher and date when known
   ("Reuters reported on 14 October 2026 that ...").
2. If any evidence is about a related video, post or rumour rather than the claim itself, say plainly that it
   is about that media and does not by itself prove or disprove the claim.
3. What is still uncertain or missing, if anything. If the verdict is "Unverified", say that this does not
   mean the claim is false.
Rules: use only the information above; never add facts, sources or dates. Keep names, numbers and dates exactly
as they appear. No jargon (do not say "weight", "stance" or "scope").

Return JSON only: {{"explanation": "..."}}
"""


class TruthLensPrompt:
    decompose_prompt = decompose_prompt
    restore_prompt = restore_prompt
    checkworthy_prompt = checkworthy_prompt
    qgen_prompt = qgen_prompt
    verify_prompt = verify_prompt
    summary_prompt = summary_prompt
