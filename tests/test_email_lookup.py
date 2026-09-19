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


def test_grounding_redirect_source_accepted_without_chunks():
    # Google returns causes as grounding-api-redirect URLs even when
    # groundingChunks is empty — that IS the citation evidence.
    text = (
        '{"emails": [{"email": "seuk.pr@samsung.com", "type": "press", '
        '"source": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbC123"}]}'
    )
    out = email_lookup.parse_brand_emails(text, urls=[], supports=[])
    assert out[0]["email"] == "seuk.pr@samsung.com"
    assert out[0]["source"].startswith("https://vertexaisearch.cloud.google.com/grounding-api-redirect/")


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


def test_raw_text_scan_needs_grounding():
    # Raw-text fallback still requires grounding evidence to exist.
    out = email_lookup.parse_brand_emails(
        "no json here just press@samsung.com and hr@samsung.com",
        urls=["https://www.samsung.com/"],
        supports=[],
    )
    emails = {e["email"] for e in out}
    assert emails == {"press@samsung.com", "hr@samsung.com"}
    # Without chunks AND without redirect sources the scan yields nothing.
    out2 = email_lookup.parse_brand_emails(
        "no json here just press@samsung.com",
        urls=[],
        supports=[],
    )
    assert out2 == []


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