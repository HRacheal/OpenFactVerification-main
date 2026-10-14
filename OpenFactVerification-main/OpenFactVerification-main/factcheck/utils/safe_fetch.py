"""SSRF-resistant web fetching and readable-text extraction.

Used for (a) URLs submitted by journalists and (b) evidence pages returned by the search provider.

Protections:
- only http/https, only ports 80/443, no user:password@ in URLs;
- every hostname is resolved and ALL resulting IPs must be public (no loopback, private, link-local,
  multicast, reserved or cloud-metadata addresses);
- redirects are followed manually (max 5) and every hop is re-validated;
- connect/read timeouts, a download size cap, and an HTML/text content-type allow-list.

Residual risk (documented in README): DNS rebinding between our check and the actual connection is not
fully prevented; run the app on a trusted machine/network.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import bs4
import requests

USER_AGENT = "Mozilla/5.0 (compatible; KweliLensAI/1.0; fact-checking research tool)"
ALLOWED_SCHEMES = {"http", "https"}
ALLOWED_PORTS = {None, 80, 443}
ALLOWED_CONTENT_TYPES = ("text/html", "application/xhtml+xml", "text/plain")
MAX_REDIRECTS = 5
DEFAULT_MAX_BYTES = 3 * 1024 * 1024
MAX_URL_LENGTH = 2048


class UnsafeURLError(ValueError):
    """The URL is malformed or points somewhere Kweli Lens refuses to fetch."""


class FetchError(RuntimeError):
    """The page could not be fetched or read. Message is safe to show to users."""


@dataclass
class FetchedPage:
    requested_url: str
    final_url: str
    status: int
    content_type: str
    html: str


def _ip_is_public(ip: str) -> bool:
    addr = ipaddress.ip_address(ip)
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
        addr = addr.ipv4_mapped
    return addr.is_global and not addr.is_multicast


def validate_url(url: str, resolve: bool = True) -> str:
    """Return the normalised URL or raise UnsafeURLError."""
    if not isinstance(url, str) or not url.strip():
        raise UnsafeURLError("Please enter a URL.")
    url = url.strip()
    if len(url) > MAX_URL_LENGTH:
        raise UnsafeURLError("The URL is too long.")
    if re.search(r"[\s\x00-\x1f]", url):
        raise UnsafeURLError("The URL contains spaces or control characters.")
    try:
        parsed = urlparse(url)
        port = parsed.port
    except ValueError:
        raise UnsafeURLError("The URL is not valid.")
    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise UnsafeURLError("Only http:// and https:// links are supported.")
    if parsed.username or parsed.password:
        raise UnsafeURLError("URLs containing usernames or passwords are not allowed.")
    if port not in ALLOWED_PORTS:
        raise UnsafeURLError("Only standard web ports (80 and 443) are allowed.")
    host = (parsed.hostname or "").rstrip(".").lower()
    if not host:
        raise UnsafeURLError("The URL has no website address.")
    if host == "localhost" or host.endswith((".localhost", ".local", ".internal", ".lan", ".home.arpa")):
        raise UnsafeURLError("Links to local or internal network addresses are not allowed.")

    # Literal IP address in the URL
    try:
        ipaddress.ip_address(host.strip("[]"))
        is_literal = True
    except ValueError:
        is_literal = False
    if is_literal:
        if not _ip_is_public(host.strip("[]")):
            raise UnsafeURLError("Links to local or internal network addresses are not allowed.")
        return url

    if resolve:
        try:
            infos = socket.getaddrinfo(host, port or (443 if parsed.scheme == "https" else 80), proto=socket.IPPROTO_TCP)
        except socket.gaierror:
            raise UnsafeURLError("The website address could not be found (DNS lookup failed).")
        ips = {info[4][0] for info in infos}
        if not ips or not all(_ip_is_public(ip) for ip in ips):
            raise UnsafeURLError("Links to local or internal network addresses are not allowed.")
    return url


def fetch_url(url: str, timeout: tuple = (5, 10), max_bytes: int = DEFAULT_MAX_BYTES, session=None) -> FetchedPage:
    """Fetch an HTML/text page safely. Raises UnsafeURLError or FetchError."""
    http = session or requests
    current = validate_url(url)
    for _ in range(MAX_REDIRECTS + 1):
        try:
            resp = http.get(
                current,
                headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml,text/plain;q=0.8"},
                timeout=timeout,
                allow_redirects=False,
                stream=True,
            )
        except requests.exceptions.Timeout:
            raise FetchError("The website took too long to respond.")
        except requests.exceptions.RequestException:
            raise FetchError("The website could not be reached.")
        try:
            if resp.status_code in (301, 302, 303, 307, 308):
                location = resp.headers.get("Location")
                if not location:
                    raise FetchError("The website sent an invalid redirect.")
                current = validate_url(urljoin(current, location))
                continue
            if resp.status_code in (401, 403):
                raise FetchError("The website refused access (it may require a login or block automated tools).")
            if resp.status_code == 404:
                raise FetchError("The page was not found (404).")
            if resp.status_code == 402 or resp.status_code == 451:
                raise FetchError("The page is not available (paywall or legal restriction).")
            if resp.status_code >= 400:
                raise FetchError(f"The website returned an error (HTTP {resp.status_code}).")
            ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if ctype and not ctype.startswith(ALLOWED_CONTENT_TYPES):
                raise FetchError(f"Unsupported content type ({ctype}). Only web pages can be analysed.")
            declared = resp.headers.get("Content-Length")
            if declared and declared.isdigit() and int(declared) > max_bytes:
                raise FetchError("The page is too large to analyse.")
            chunks, total = [], 0
            for chunk in resp.iter_content(chunk_size=65536):
                total += len(chunk)
                if total > max_bytes:
                    raise FetchError("The page is too large to analyse.")
                chunks.append(chunk)
            raw = b"".join(chunks)
            encoding = resp.encoding if resp.encoding and resp.encoding.lower() != "iso-8859-1" else None
            html = raw.decode(encoding or "utf-8", errors="replace")
            return FetchedPage(url, current, resp.status_code, ctype, html)
        finally:
            resp.close()
    raise FetchError("The website redirected too many times.")


_REMOVE_TAGS = ["script", "style", "noscript", "nav", "footer", "header", "aside", "form", "iframe", "svg", "button"]


def extract_readable_text(html: str, max_chars: int = 20000) -> str:
    """Extract the main readable text of an article-like page."""
    soup = bs4.BeautifulSoup(html, "html.parser")
    for tag in soup(_REMOVE_TAGS):
        tag.decompose()
    for tag in soup.select("sup.reference, .mw-editsection, .noprint"):  # citation markers / edit links
        tag.decompose()
    root = soup.find("article") or soup.find("main") or soup.body or soup
    blocks = []
    for el in root.find_all(["h1", "h2", "h3", "p", "li", "blockquote"]):
        # No separator: inline tags (<a>, <b>) must not split words or detach punctuation.
        text = " ".join(el.get_text().split())
        if len(text) >= 25 or el.name in ("h1", "h2"):
            blocks.append(text)
    text = "\n".join(dict.fromkeys(blocks))  # drop duplicate blocks
    if len(text) < 200:  # pages without <p> structure
        text = " ".join(root.get_text(" ", strip=True).split())
    return text[:max_chars]
