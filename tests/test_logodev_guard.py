"""
Tests for the logo.dev client (B2a/B2b) and the server's external
brand-validation guard (Part A / B2b remediation).

These tests must never touch the network or any real key: they assert the
fail-closed behavior when no LOGO_DEV key is configured, and exercise the
outreach endpoint with a stubbed client. Only status "verified" may ever be
presented as a real brand.
"""

import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent))


def _clear_validate_cache():
    import server

    server._BRAND_VALIDATION_CACHE.clear()


def test_validate_brand_fails_closed_without_key():
    """No LOGO_DEV_SECRET_KEY configured -> status 'unavailable', never verified."""
    from src.logodev import LogoDevClient

    client = LogoDevClient()
    # Ensure the process has no key set (the test env must not carry one).
    import os

    assert not os.environ.get("LOGO_DEV_SECRET_KEY")
    assert client.available is False
    result = client.validate_brand("NIKE")
    assert result["status"] == "unavailable"
    assert result["status"] != "verified"


def test_outreach_generate_requires_external_verification():
    """A detected brand (count > 0) still needs a 'verified' logo.dev result."""
    import server

    _clear_validate_cache()
    fake_job = {
        "status": "done",
        "dashboard": {
            "title": "TEST CHANNEL",
            "creator": "TESTER",
            "products": [
                {
                    "brand": "NIKE",
                    "product": "Nike Air",
                    "category": "APPAREL",
                    "appearance_count": 3,
                    "appearances": ["SCENE 001", "SCENE 004"],
                }
            ],
        },
    }
    server.JOBS["TESTJOB-XYZ"] = fake_job
    client = server.app.test_client()
    original = server.OUTREACH_ENABLED
    server.OUTREACH_ENABLED = True

    try:
        with mock.patch.object(
            server.LogoDevClient, "validate_brand",
            return_value={"status": "unverified", "brand": "NIKE", "domain": None},
        ):
            resp = client.post("/api/outreach/generate", json={
                "job_id": "TESTJOB-XYZ", "brand": "NIKE",
            })
            assert resp.status_code == 400
            body = resp.get_json()
            assert body["brand_validation"]["status"] == "unverified"

        _clear_validate_cache()
        with mock.patch.object(
            server.LogoDevClient, "validate_brand",
            return_value={"status": "verified", "brand": "NIKE",
                          "domain": "nike.com"},
        ):
            resp = client.post("/api/outreach/generate", json={
                "job_id": "TESTJOB-XYZ", "brand": "NIKE",
            })
            assert resp.status_code == 200
            body = resp.get_json()
            assert body["brand"] == "NIKE"
            assert body["brand_validation"]["status"] == "verified"
    finally:
        server.OUTREACH_ENABLED = original
        server.JOBS.pop("TESTJOB-XYZ", None)
        _clear_validate_cache()


def test_outreach_generate_blocks_zero_appearance_brands():
    """Knowledge-graph SUGGESTED brands (count 0) must never get a draft."""
    import server

    _clear_validate_cache()
    fake_job = {
        "status": "done",
        "dashboard": {
            "title": "TEST CHANNEL",
            "creator": "TESTER",
            "products": [
                {
                    "brand": "ROLEX",
                    "product": "Rolex Submariner",
                    "category": "LUXURY",
                    "appearance_count": 0,
                    "appearances": [],
                }
            ],
        },
    }
    server.JOBS["TESTJOB-ROL"] = fake_job
    client = server.app.test_client()
    original = server.OUTREACH_ENABLED
    server.OUTREACH_ENABLED = True

    try:
        with mock.patch.object(
            server.LogoDevClient, "validate_brand",
            return_value={"status": "verified", "brand": "ROLEX",
                          "domain": "rolex.com"},
        ):
            resp = client.post("/api/outreach/generate", json={
                "job_id": "TESTJOB-ROL", "brand": "ROLEX",
            })
            assert resp.status_code == 400
            body = resp.get_json()
            assert "NO REAL ON-SCREEN APPEARANCES" in body["error"]
    finally:
        server.OUTREACH_ENABLED = original
        server.JOBS.pop("TESTJOB-ROL", None)
        _clear_validate_cache()
