"""URL validation, SSRF protection and article extraction. Network and DNS are mocked."""

import socket

import pytest

from factcheck.utils import safe_fetch
from factcheck.utils.safe_fetch import validate_url, fetch_url, extract_readable_text, UnsafeURLError, FetchError
from factcheck.utils.sources import classify_source, extract_page_metadata, publisher_for


def fake_dns(mapping):
    def getaddrinfo(host, port, *a, **k):
        if host not in mapping:
            raise socket.gaierror("not found")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port)) for ip in mapping[host]]

    return getaddrinfo


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/x",
        "javascript:alert(1)",
        "http://localhost/admin",
        "http://127.0.0.1:80/",
        "http://10.0.0.5/",
        "http://192.168.1.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/",
        "http://user:pass@example.com/",
        "http://example.com:22/",
        "http://printer.local/",
        "",
        "http://exa mple.com/",
    ],
)
def test_rejects_unsafe_urls(url, monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", fake_dns({"example.com": ["93.184.216.34"]}))
    with pytest.raises(UnsafeURLError):
        validate_url(url)


def test_rejects_hostname_that_resolves_to_private_ip(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", fake_dns({"evil.example": ["93.184.216.34", "10.1.2.3"]}))
    with pytest.raises(UnsafeURLError):
        validate_url("https://evil.example/")


def test_accepts_public_url(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", fake_dns({"www.monitor.co.ug": ["104.18.1.1"]}))
    assert validate_url(" https://www.monitor.co.ug/news ") == "https://www.monitor.co.ug/news"


class FakeResp:
    def __init__(self, status=200, headers=None, body=b"", encoding="utf-8"):
        self.status_code = status
        self.headers = headers or {"Content-Type": "text/html; charset=utf-8"}
        self._body = body
        self.encoding = encoding

    def iter_content(self, chunk_size=65536):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i : i + chunk_size]

    def close(self):
        pass


class FakeSession:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def get(self, url, **kw):
        self.calls.append(url)
        assert kw["allow_redirects"] is False
        return self.responses[url]


@pytest.fixture
def public_dns(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", fake_dns({"news.example": ["93.184.216.34"], "other.example": ["93.184.216.35"]}))


def test_redirect_to_internal_address_is_blocked(public_dns):
    s = FakeSession({"https://news.example/a": FakeResp(302, {"Location": "http://127.0.0.1/secret"})})
    with pytest.raises(UnsafeURLError):
        fetch_url("https://news.example/a", session=s)


def test_follows_safe_redirect(public_dns):
    s = FakeSession({
        "https://news.example/a": FakeResp(301, {"Location": "https://other.example/b"}),
        "https://other.example/b": FakeResp(200, body=b"<html><p>Hello</p></html>"),
    })
    page = fetch_url("https://news.example/a", session=s)
    assert page.final_url == "https://other.example/b" and "Hello" in page.html


def test_size_limit_and_content_type(public_dns):
    big = FakeSession({"https://news.example/a": FakeResp(200, body=b"x" * 5000)})
    with pytest.raises(FetchError, match="too large"):
        fetch_url("https://news.example/a", session=big, max_bytes=1000)
    exe = FakeSession({"https://news.example/a": FakeResp(200, {"Content-Type": "application/octet-stream"}, b"MZ")})
    with pytest.raises(FetchError, match="Unsupported content type"):
        fetch_url("https://news.example/a", session=exe)
    paywall = FakeSession({"https://news.example/a": FakeResp(403)})
    with pytest.raises(FetchError, match="refused access"):
        fetch_url("https://news.example/a", session=paywall)


def test_timeout_is_reported(public_dns):
    import requests

    class Slow:
        def get(self, *a, **k):
            raise requests.exceptions.Timeout()

    with pytest.raises(FetchError, match="too long"):
        fetch_url("https://news.example/a", session=Slow())


ARTICLE = """<html><head><title>Fees cut | Daily Monitor</title>
<meta property="og:site_name" content="Daily Monitor">
<meta property="article:published_time" content="2026-03-04T08:00:00+03:00"></head>
<body><nav>Home | Sports | Login to read more</nav><script>var x=1;</script>
<article><h1>Government announces fee changes</h1>
<p>The Ministry of Education said on Tuesday that tuition fees at public universities will be reduced.</p>
<p>Officials did not say by how much the fees would fall, but promised details next month.</p></article>
<footer>Copyright</footer></body></html>"""


def test_extracts_article_text_and_metadata():
    text = extract_readable_text(ARTICLE)
    assert "Ministry of Education" in text and "var x" not in text and "Login to read" not in text
    meta = extract_page_metadata(ARTICLE)
    assert meta["published_date"] == "2026-03-04" and meta["site_name"] == "Daily Monitor"


def test_missing_date_is_not_invented():
    assert extract_page_metadata("<html><p>No dates here</p></html>")["published_date"] is None


def test_source_classification():
    assert classify_source("https://www.statistics.go.ug/report") == "official"
    assert classify_source("https://pesacheck.org/some-check") == "fact_checker"
    assert classify_source("https://www.monitor.co.ug/uganda/news") == "news"
    assert classify_source("https://www.facebook.com/post/1") == "social"
    assert classify_source("https://mak.ac.ug/research") == "academic"
    assert classify_source("https://random-blog.example") == "unknown"
    assert publisher_for("https://www.newvision.co.ug/x") == "New Vision"
