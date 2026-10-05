"""
Offline tests for the Gemini-grounded brand-email lookup (fail-closed).

Pins the contracts that keep fabrications out of outreach:
  * no API key          -> unavailable, zero emails
  * no grounding URLs   -> zero emails (ungrounded output never trusted)
  * placeholder tokens  -> rejected
  * valid grounded JSON -> parsed, typed (hr vs press/contact), evidence URL attached
  * malformed / HTTP errors -> fail closed with an empty email list
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests  # noqa: E402

from src import email_lookup  # noqa: E402


def _grounded_text():
    return (
        '{"emails": ['
        '{"email": "press@samsung.com", "type": "press", "source": "https://www.samsung.com/press/"},'
        '{"email": "hr.bangalore@samsung.com", "type": "hr", "source": "https://www.samsung-careers.com/contact"},'
        '{"email": "yourname@example.com", "type": "contact", "source": "https://www.samsung.com/press/"}'
        "]}"
    )


def test_no_api_key_is_unavailable():
    saved = email_lookup._api_key()
    try:
        email_lookup.clear_cache()
        email_lookup.os.environ.pop("GEMINI_API_KEY", None)
        out = email_lookup.lookup_brand_emails("SAMSUNG")
        assert out["status"] == "unavailable"
        assert out["emails"] == []
    finally:
        email_lookup.clear_cache()
        if saved:
            email_lookup._set_env("GEMINI_API_KEY", saved)


def test_ungrounded_answer_yields_no_emails():
    # A well-formed JSON answer with NO grounding chunks and NO grounding-redirect
    # sources must fail closed.
    out = email_lookup.parse_brand_emails(
        _grounded_text(), urls=[], supports=[],
    )
    assert out == []


def test_grounding_redirect_source_requires_api_returned_url():
    # `source` is untrusted model output. A grounding-redirect URL is only
    # evidence when the API actually returned it in groundingMetadata.
    redirect = (
        "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbC123"
    )
    text = (
        '{"emails": [{"email": "seuk.pr@samsung.com", "type": "press", '
        f'"source": "{redirect}"}}]}}'
    )
    # Returned by the API -> accepted, even with no groundingChunks.
    out = email_lookup.parse_brand_emails(text, urls=[redirect], supports=[])
    assert out[0]["email"] == "seuk.pr@samsung.com"
    assert out[0]["source"] == redirect

    # Fabricated by the model, never returned -> dropped. Matching the redirect
    # *prefix* must not be enough.
    out2 = email_lookup.parse_brand_emails(text, urls=[], supports=[])
    assert out2 == []


def test_grounded_json_parses_and_types():
    urls = [
        "https://www.samsung.com/press/",
        "https://www.samsung-careers.com/contact",
    ]
    out = email_lookup.parse_brand_emails(_grounded_text(), urls=urls, supports=[])
    emails = {e["email"] for e in out}
    assert emails == {"press@samsung.com", "hr.bangalore@samsung.com"}
    # Placeholder address dropped.
    assert "yourname@example.com" not in emails
    types = {e["email"]: e["type"] for e in out}
    assert types["hr.bangalore@samsung.com"] == "hr"
    assert types["press@samsung.com"] in ("press", "contact")
    # Every address is traced to a real grounding URL.
    assert all(e["source"].startswith("https://") for e in out)


def test_source_forced_to_grounding_uri():
    urls = ["https://www.samsung.com/press/"]
    # source that matches a real grounding chunk is kept verbatim...
    out = email_lookup.parse_brand_emails(
        '{"emails": [{"email": "pr@samsung.com", "type": "pr", "source": "https://www.samsung.com/press/"}]}',
        urls=urls,
        supports=[],
    )
    assert out == [{
        "email": "pr@samsung.com",
        "type": "pr",
        "source": "https://www.samsung.com/press/",
    }]
    # ...a source with no grounding to trace to is dropped (never re-cited).
    out2 = email_lookup.parse_brand_emails(
        '{"emails": [{"email": "pr@samsung.com", "type": "pr", "source": "https://unrelated.example/x"}]}',
        urls=urls,
        supports=[],
    )
    assert out2 == []


def test_raw_text_scan_dropped_even_with_grounding():
    # No per-address source means any provenance we attach would be invented,
    # so raw-text scanning yields nothing regardless of grounding.
    out = email_lookup.parse_brand_emails(
        "no json here just press@samsung.com and hr@samsung.com",
        urls=["https://www.samsung.com/"],
        supports=[],
    )
    assert out == []


def test_placeholder_validation():
    for bad in (
        "yourname@example.com",
        "name@yourbrand.com",
        "user@domain.invalid",
        "email@example.org",
        "not an email",
        "plaindomain.com",
        "two..dots@a.com",
    ):
        assert not email_lookup._is_valid_email(bad), bad
    assert email_lookup._is_valid_email("pr@samsung.co.in")
    assert email_lookup._is_valid_email("hr.bangalore@samsung.com")


def test_http_error_fails_closed(monkeypatch):
    email_lookup._set_env("GEMINI_API_KEY", "FAKEKEY")
    email_lookup.clear_cache()

    def _fake(*a, **k):
        class _R:
            status_code = 500

            @staticmethod
            def text():
                return "boom"
        return _R()

    monkeypatch.setattr(requests, "post", _fake)
    out = email_lookup.lookup_brand_emails("SAMSUNG")
    assert out["status"] == "error"
    assert out["emails"] == []


def test_success_path(monkeypatch):
    email_lookup._set_env("GEMINI_API_KEY", "FAKEKEY")
    email_lookup.clear_cache()

    def _fake(*a, **k):
        class _R:
            status_code = 200

            @staticmethod
            def json():
                return {
                    "candidates": [{
                        "content": {"parts": [{"text": _grounded_text()}]},
                        "groundingMetadata": {
                            "groundingChunks": [
                                {"web": {"uri": "https://www.samsung.com/press/"}},
                                {"web": {"uri": "https://www.samsung-careers.com/contact"}},
                            ],
                            "groundingSupports": [],
                        },
                    }]
                }
        return _R()

    monkeypatch.setattr(requests, "post", _fake)
    out = email_lookup.lookup_brand_emails("SAMSUNG")
    assert out["status"] == "ok"
    assert {e["email"] for e in out["emails"]} == {
        "press@samsung.com", "hr.bangalore@samsung.com",
    }
    assert "hr.bangalore@samsung.com" in out["hr_emails"]
    assert any(e["url"].startswith("https://www.samsung.com") for e in out["evidence"])

# ============================================================================
# Item 5 — a brand's contact must be on the brand's own domain
# ============================================================================

class TestEmailDomainCheck:
    """A grounded search often surfaces a contact's personal address (or an
    agency's) as if it were the brand's. Every consumer webmail address passes
    the RFC-shaped regex, so without this an outreach draft goes to
    recruiter@gmail.com and cites a real URL as provenance."""

    @staticmethod
    def _ok(addr, brand="Nike"):
        from src.email_lookup import _is_valid_email, _brand_domain
        return _is_valid_email(addr, _brand_domain(brand))

    def test_rejects_personal_webmail(self):
        for addr in ("recruiter@gmail.com", "info@yahoo.com", "press@hotmail.com",
                     "a@outlook.com", "x@icloud.com", "y@gmail.co.uk"):
            assert self._ok(addr) is False, addr

    def test_rejects_personal_provider_subdomain(self):
        assert self._ok("press@mail.gmail.com") is False

    def test_accepts_official_brand_addresses(self):
        assert self._ok("press@nike.com") is True
        assert self._ok("press@sony.co.uk", "Sony") is True   # sibling ccTLD
        assert self._ok("hr@mail.nike.com") is True           # subdomain

    def test_rejects_a_different_brand_domain(self):
        assert self._ok("pr@adidas.com") is False
        assert self._ok("press@notsony.com", "Sony") is False
        assert self._ok("press@sony.com.evil.net", "Sony") is False

    def test_non_catalog_brand_keeps_the_provider_check_only(self):
        # We never curated a domain, so we must not reject a legitimate
        # address on that basis — but a gmail address is still wrong.
        assert self._ok("press@acme-industrial.io", "Totally Unknown Co") is True
        assert self._ok("recruiter@gmail.com", "Totally Unknown Co") is False

    def test_parse_drops_off_domain_and_webmail_addresses(self):
        from src.email_lookup import parse_brand_emails, _brand_domain
        urls = ["https://nike.com/press", "https://news.ycombinator.com/x"]
        payload = ('{"emails": ['
                   '{"email": "press@nike.com", "type": "press", "source": "https://nike.com/press"},'
                   '{"email": "recruiter@gmail.com", "type": "pr", "source": "https://nike.com/press"},'
                   '{"email": "pr@adidas.com", "type": "pr", "source": "https://news.ycombinator.com/x"}'
                   ']}')
        out = parse_brand_emails(payload, urls, [], brand_domain=_brand_domain("Nike"))
        assert [e["email"] for e in out] == ["press@nike.com"]
