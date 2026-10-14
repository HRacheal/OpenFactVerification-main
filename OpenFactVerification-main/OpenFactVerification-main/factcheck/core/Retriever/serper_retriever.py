"""Evidence retrieval through the Serper.dev Google Search API.

For every search result Kweli Lens keeps the link between the excerpt and its source: title, URL, domain,
publisher, publication date (when available), retrieval time, source type, and whether the full page
was read or only the search snippet. Nothing is invented: missing metadata stays empty.

If Serper fails (bad key, quota, network), a RetrievalError is raised so the user sees a clear error
instead of a report silently built on no evidence.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import requests

from factcheck.utils.logger import CustomLogger
from factcheck.utils.safe_fetch import fetch_url, extract_readable_text, UnsafeURLError, FetchError
from factcheck.utils.sources import get_domain, classify_source, publisher_for, extract_page_metadata

logger = CustomLogger(__name__).getlog()

SERPER_URL = "https://google.serper.dev/search"

# Country name -> Google "gl" (geolocation) code used to bias results.
COUNTRY_CODES = {
    "kenya": "ke",
    "uganda": "ug",
    "rwanda": "rw",
    "tanzania": "tz",
    "burundi": "bi",
    "south sudan": "ss",
    "ethiopia": "et",
    "somalia": "so",
    "dr congo": "cd",
}


class RetrievalError(RuntimeError):
    """Evidence search failed. Message is safe to show to users."""


class SerperEvidenceRetriever:
    def __init__(self, llm_client=None, api_config: dict = None, top_k: int = 3, max_evidence_per_claim: int = 8,
                 fetch_full_text: bool = True, timeout: float = 20.0):
        """Initialize the Serper evidence retriever."""
        api_config = api_config or {}
        self.serper_key = api_config.get("SERPER_API_KEY")
        self.llm_client = llm_client
        self.top_k = top_k
        self.max_evidence_per_claim = max_evidence_per_claim
        self.fetch_full_text = fetch_full_text
        self.timeout = timeout
        self.country = None
        self.language = None

    def set_context(self, country: str = None, language: str = None):
        self.country = country
        self.language = language

    # ------------------------------------------------------------------ search
    def _request_serper_api(self, queries: list[str]) -> list[dict]:
        if not self.serper_key:
            raise RetrievalError("Evidence search is not configured: SERPER_API_KEY is missing. Add it to your .env file.")
        gl = COUNTRY_CODES.get((self.country or "").strip().lower())
        payload = []
        for q in queries:
            item = {"q": q, "autocorrect": False}
            if gl:
                item["gl"] = gl
            if self.language == "sw":
                item["hl"] = "sw"
            payload.append(item)
        try:
            resp = requests.post(
                SERPER_URL,
                headers={"X-API-KEY": self.serper_key, "Content-Type": "application/json"},
                json=payload,
                timeout=self.timeout,
            )
        except requests.exceptions.Timeout:
            raise RetrievalError("The evidence search service (Serper) timed out. Please try again.")
        except requests.exceptions.RequestException:
            raise RetrievalError("Could not reach the evidence search service (Serper). Check your internet connection.")
        if resp.status_code in (401, 403):
            raise RetrievalError("The evidence search service rejected the API key. Check SERPER_API_KEY.")
        if resp.status_code == 429:
            raise RetrievalError("The evidence search service rate limit or credit quota was reached.")
        if resp.status_code != 200:
            raise RetrievalError(f"The evidence search service returned an error (HTTP {resp.status_code}).")
        try:
            data = resp.json()
        except ValueError:
            raise RetrievalError("The evidence search service returned an unreadable response.")
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list) or len(data) != len(queries):
            raise RetrievalError("The evidence search service returned an unexpected response.")
        return data

    @staticmethod
    def _results_from_response(query: str, response: dict, top_k: int) -> list[dict]:
        results = []
        box = response.get("answerBox") if isinstance(response, dict) else None
        if isinstance(box, dict) and (box.get("answer") or box.get("snippet")):
            results.append(
                {
                    "text": str(box.get("answer") or box.get("snippet")),
                    "url": box.get("link") if str(box.get("link", "")).lower().startswith(("http://", "https://")) else "",
                    "title": box.get("title") or "Search engine answer box",
                    "published_date": box.get("date"),
                    "query": query,
                }
            )
        for r in (response.get("organic") or [])[:top_k]:
            if not isinstance(r, dict) or not str(r.get("link", "")).lower().startswith(("http://", "https://")):
                continue  # only keep real web links
            results.append(
                {
                    "text": re.sub(r"\n+", "\n", r.get("snippet") or ""),
                    "url": r["link"],
                    "title": r.get("title") or "",
                    "published_date": r.get("date"),
                    "query": query,
                }
            )
        return results

    # ------------------------------------------------------------------ page reading
    @staticmethod
    def _best_excerpt(page_text: str, snippet: str, query: str, window: int = 1500) -> str | None:
        """Find the passage of the page that the search snippet came from, or the most query-relevant passage."""
        if not page_text:
            return None
        probe = re.sub(r"\s+", " ", snippet or "").strip(" .…")[:80]
        norm_page = re.sub(r"\s+", " ", page_text)
        if len(probe) >= 30:
            idx = norm_page.lower().find(probe[:60].lower())
            if idx != -1:
                start = max(0, idx - 300)
                return norm_page[start : start + window]
        terms = {t for t in re.findall(r"\w{4,}", query.lower())}
        if not terms:
            return None
        best, best_score = None, 0
        for start in range(0, max(1, len(norm_page) - 200), 500):
            chunk = norm_page[start : start + window]
            score = sum(1 for t in terms if t in chunk.lower())
            if score > best_score:
                best, best_score = chunk, score
        return best if best_score >= max(2, len(terms) // 2) else None

    def _read_page(self, url: str) -> dict | None:
        try:
            page = fetch_url(url, timeout=(4, 8), max_bytes=2 * 1024 * 1024)
        except (UnsafeURLError, FetchError) as e:
            logger.info(f"Could not read evidence page {get_domain(url)}: {e}")
            return None
        except Exception as e:  # never let one bad page break the whole check
            logger.info(f"Unexpected error reading {get_domain(url)}: {type(e).__name__}")
            return None
        meta = extract_page_metadata(page.html)
        return {"text": extract_readable_text(page.html, max_chars=60000), **meta}

    # ------------------------------------------------------------------ main entry
    def retrieve_evidence(self, claim_queries_dict: dict, top_k: int = None, snippet_extend_flag: bool = None) -> dict:
        """Retrieve evidence for each claim. Returns claim -> list of evidence dicts."""
        top_k = top_k or self.top_k
        fetch_full = self.fetch_full_text if snippet_extend_flag is None else snippet_extend_flag
        logger.info("Collecting evidences ...")
        query_list = [q for queries in claim_queries_dict.values() for q in queries]
        if not query_list:
            return {claim: [] for claim in claim_queries_dict}

        responses = []
        for i in range(0, len(query_list), 100):
            responses += self._request_serper_api(query_list[i : i + 100])

        retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        claim_evidence_dict, i = {}, 0
        for claim, queries in claim_queries_dict.items():
            seen, evidences = set(), []
            for q, resp in zip(queries, responses[i : i + len(queries)]):
                for r in self._results_from_response(q, resp, top_k):
                    key = r["url"] or r["text"]
                    if key in seen:
                        continue
                    seen.add(key)
                    evidences.append(r)
            i += len(queries)
            claim_evidence_dict[claim] = evidences[: self.max_evidence_per_claim]

        # Read the full pages (each URL once) so the assessment is not based on snippets alone.
        urls = sorted({e["url"] for evs in claim_evidence_dict.values() for e in evs if e["url"]})
        pages = {}
        if fetch_full and urls:
            with ThreadPoolExecutor(max_workers=8) as pool:
                pages = dict(zip(urls, pool.map(self._read_page, urls)))

        for claim, evs in claim_evidence_dict.items():
            for e in evs:
                page = pages.get(e["url"]) if e["url"] else None
                e["full_text_fetched"] = False
                if page:
                    excerpt = self._best_excerpt(page["text"], e["text"], e["query"] + " " + claim)
                    if excerpt:
                        e["text"] = excerpt
                        e["full_text_fetched"] = True
                    if not e["title"] and page.get("title"):
                        e["title"] = page["title"]
                    if page.get("published_date"):
                        e["published_date"] = page["published_date"]
                e["domain"] = get_domain(e["url"]) if e["url"] else ""
                e["publisher"] = publisher_for(e["url"], (page or {}).get("site_name", "")) if e["url"] else "Search engine"
                e["source_type"] = classify_source(e["url"]) if e["url"] else "unknown"
                e["retrieved_at"] = retrieved_at
            claim_evidence_dict[claim] = [e for e in evs if e["text"].strip()]

        logger.info("Collect evidences done!")
        return claim_evidence_dict
