"""Source metadata helpers: domain, source type, and publication-date extraction.

The source type is a *hint* derived only from the domain name. Being on one of these lists does not
make a source reliable, and being absent does not make it unreliable. The journalist must judge.
"""

from __future__ import annotations

import json
import re
from urllib.parse import urlparse

import bs4

# Established fact-checking organisations (IFCN signatories or long-running desks).
FACT_CHECKERS = {
    "pesacheck.org",
    "africacheck.org",
    "factcheck.afp.com",
    "fullfact.org",
    "politifact.com",
    "snopes.com",
    "factcheck.org",
    "dubawa.org",
    "211check.org",
    "piga-firimbi.africauncensored.online",
}

# Reputable East African and international news outlets (relevant to the initial focus region).
NEWS_OUTLETS = {
    # Uganda
    "monitor.co.ug": "Daily Monitor",
    "newvision.co.ug": "New Vision",
    "observer.ug": "The Observer",
    "chimpreports.com": "ChimpReports",
    "nilepost.co.ug": "Nile Post",
    # Kenya
    "nation.africa": "Nation",
    "standardmedia.co.ke": "The Standard",
    "the-star.co.ke": "The Star",
    "citizen.digital": "Citizen Digital",
    "kbc.co.ke": "KBC",
    "businessdailyafrica.com": "Business Daily",
    "theeastafrican.co.ke": "The EastAfrican",
    # Rwanda
    "newtimes.co.rw": "The New Times",
    "ktpress.rw": "KT Press",
    "igihe.com": "IGIHE",
    # Tanzania
    "thecitizen.co.tz": "The Citizen",
    "dailynews.co.tz": "Daily News",
    # International
    "bbc.com": "BBC",
    "bbc.co.uk": "BBC",
    "reuters.com": "Reuters",
    "apnews.com": "Associated Press",
    "aljazeera.com": "Al Jazeera",
    "theguardian.com": "The Guardian",
    "dw.com": "DW",
    "voanews.com": "VOA",
}

SOCIAL_AND_UGC = {
    "facebook.com",
    "x.com",
    "twitter.com",
    "tiktok.com",
    "instagram.com",
    "youtube.com",
    "reddit.com",
    "quora.com",
    "medium.com",
    "blogspot.com",
    "wordpress.com",
    "pinterest.com",
    "linkedin.com",
    "t.me",
}

# Official / academic suffixes, including East African government and academic second-level domains.
OFFICIAL_SUFFIXES = (".gov", ".go.ke", ".go.ug", ".gov.rw", ".go.tz", ".gov.uk", ".gov.za", ".int", ".mil", ".gouv.fr")
OFFICIAL_DOMAINS = {"who.int", "un.org", "worldbank.org", "imf.org", "afdb.org", "eac.int", "au.int", "unicef.org", "undp.org"}
ACADEMIC_SUFFIXES = (".edu", ".ac.ke", ".ac.ug", ".ac.rw", ".ac.tz", ".ac.uk", ".ac.za")

SOURCE_TYPE_LABELS = {
    "official": "Government / intergovernmental",
    "academic": "Academic / research",
    "fact_checker": "Fact-checking organisation",
    "news": "News outlet",
    "social": "Social media / user-generated",
    "unknown": "Other website",
}


def get_domain(url: str) -> str:
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


_SECOND_LEVEL = {"co", "go", "ac", "or", "ne", "gov", "com", "org", "edu", "net", "mil", "sc"}


def site_key(domain: str) -> str:
    """Approximate registrable domain, so subdomains of one organisation count as ONE source.
    e.g. en.wikipedia.org -> wikipedia.org, www.monitor.co.ug -> monitor.co.ug"""
    parts = [p for p in (domain or "").lower().split(".") if p]
    if len(parts) >= 3 and parts[-2] in _SECOND_LEVEL and len(parts[-1]) == 2:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:]) if len(parts) >= 2 else (domain or "")


def _matches(domain: str, candidates) -> str | None:
    for c in candidates:
        if domain == c or domain.endswith("." + c):
            return c
    return None


def classify_source(url: str) -> str:
    domain = get_domain(url)
    if not domain:
        return "unknown"
    if _matches(domain, FACT_CHECKERS) or "/fact-check" in url.lower() or "/factcheck" in url.lower():
        return "fact_checker"
    if domain.endswith(OFFICIAL_SUFFIXES) or _matches(domain, OFFICIAL_DOMAINS):
        return "official"
    if domain.endswith(ACADEMIC_SUFFIXES):
        return "academic"
    if _matches(domain, NEWS_OUTLETS):
        return "news"
    if _matches(domain, SOCIAL_AND_UGC):
        return "social"
    return "unknown"


def publisher_for(url: str, page_site_name: str = "") -> str:
    domain = get_domain(url)
    match = _matches(domain, NEWS_OUTLETS)
    if match:
        return NEWS_OUTLETS[match]
    return page_site_name.strip() or domain


_DATE_META_KEYS = (
    "article:published_time",
    "og:article:published_time",
    "datepublished",
    "publishdate",
    "pubdate",
    "date",
    "dc.date",
    "dc.date.issued",
    "sailthru.date",
    "parsely-pub-date",
)
_ISO_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def extract_page_metadata(html: str) -> dict:
    """Extract title, site name and publication date from HTML metadata. Missing values are ''/None."""
    meta = {"title": "", "site_name": "", "published_date": None}
    try:
        soup = bs4.BeautifulSoup(html, "html.parser")
    except Exception:
        return meta
    og_title = soup.find("meta", attrs={"property": "og:title"})
    if og_title and og_title.get("content"):
        meta["title"] = og_title["content"].strip()
    elif soup.title and soup.title.string:
        meta["title"] = soup.title.string.strip()
    site = soup.find("meta", attrs={"property": "og:site_name"})
    if site and site.get("content"):
        meta["site_name"] = site["content"].strip()

    for tag in soup.find_all("meta"):
        key = (tag.get("property") or tag.get("name") or tag.get("itemprop") or "").lower()
        if key in _DATE_META_KEYS and tag.get("content"):
            m = _ISO_DATE_RE.search(tag["content"])
            if m:
                meta["published_date"] = m.group(1)
                return meta
    # JSON-LD (schema.org NewsArticle etc.)
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            data = json.loads(script.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        items = data if isinstance(data, list) else data.get("@graph", [data]) if isinstance(data, dict) else []
        for item in items:
            if isinstance(item, dict) and item.get("datePublished"):
                m = _ISO_DATE_RE.search(str(item["datePublished"]))
                if m:
                    meta["published_date"] = m.group(1)
                    return meta
    time_tag = soup.find("time", attrs={"datetime": True})
    if time_tag:
        m = _ISO_DATE_RE.search(time_tag["datetime"])
        if m:
            meta["published_date"] = m.group(1)
    return meta
