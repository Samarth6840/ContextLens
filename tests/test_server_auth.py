import os
import tempfile
from unittest import mock

import server
from src.auth import hash_password
from src.store import JobStore

# ── Auth (feature-flagged on for these tests) ──────────────────────────


def _enable_auth(admin_password="hunter2"):
    server.AUTH_ENABLED = True
    server.ADMIN_CREDENTIALS = {
        "user": "admin",
        "password_hash": hash_password(admin_password),
    }
    server.SESSION_MANAGER._tokens.clear()


def test_login_and_protected_access():
    _enable_auth()
    client = server.app.test_client()
    try:
        # Unauthenticated access to /api/jobs is rejected while auth is on.
        resp = client.get("/api/jobs")
        assert resp.status_code == 401

        # Wrong password rejected.
        bad = client.post("/api/login", json={
            "username": "admin", "password": "wrong"
        })
        assert bad.status_code == 401

        # Correct login issues a bearer token.
        good = client.post("/api/login", json={
            "username": "admin", "password": "hunter2"
        })
        assert good.status_code == 200
        token = good.get_json()["token"]

        # Authorized access now succeeds.
        authed = client.get("/api/jobs", headers={
            "Authorization": f"Bearer {token}"
        })
        assert authed.status_code == 200

        # /api/me reports authenticated.
        me = client.get("/api/me", headers={"Authorization": f"Bearer {token}"})
        assert me.get_json()["authenticated"] is True

        # Logout revokes the token.
        client.post("/api/logout", headers={"Authorization": f"Bearer {token}"})
        revoked = client.get("/api/jobs", headers={
            "Authorization": f"Bearer {token}"
        })
        assert revoked.status_code == 401
    finally:
        server.AUTH_ENABLED = False
        server.ADMIN_CREDENTIALS = None
        server.SESSION_MANAGER._tokens.clear()


def test_login_fails_closed_when_unconfigured_or_disabled():
    client = server.app.test_client()
    # Auth disabled -> login refused.
    server.AUTH_ENABLED = False
    server.ADMIN_CREDENTIALS = {"user": "admin",
                                "password_hash": hash_password("x")}
    try:
        resp = client.post("/api/login", json={
            "username": "admin", "password": "x"
        })
        assert resp.status_code == 403
        assert "AUTH DISABLED" in resp.get_json()["error"]
    finally:
        server.AUTH_ENABLED = False
        server.ADMIN_CREDENTIALS = None


def test_me_reports_disabled_when_auth_off():
    server.AUTH_ENABLED = False
    try:
        resp = server.app.test_client().get("/api/me")
        body = resp.get_json()
        assert body["authenticated"] is False
        assert body["auth_enabled"] is False
    finally:
        server.AUTH_ENABLED = False


# ── Database / archive endpoints ────────────────────────────────────────


def _fresh_store():
    tmpdir = tempfile.mkdtemp()
    return JobStore(os.path.join(tmpdir, "archive.db"))


def test_archive_list_and_get_endpoints():
    store = _fresh_store()
    store.save("ARCH1", {
        "job_id": "ARCH1", "title": "ARCHIVED", "status": "done",
        "creator": "TESTER", "created_at": "2026-01-01T00:00:00Z",
        "finished_at": "2026-01-01T00:00:01Z",
        "dashboard": {"confidence": 0.9},
        "result": {"layer3": {"recommendations": [{"brand": "NIKE"}]}},
    })
    client = server.app.test_client()
    with mock.patch.object(server, "JOB_STORE", store):
        listing = client.get("/api/jobs/archive")
        assert listing.status_code == 200
        assert any(j["job_id"] == "ARCH1"
                   for j in listing.get_json()["jobs"])

        detail = client.get("/api/jobs/archive/ARCH1")
        assert detail.status_code == 200
        body = detail.get_json()
        assert body["dashboard"]["confidence"] == 0.9
        assert body["result"]["layer3"]["recommendations"] == [{"brand": "NIKE"}]

        missing = client.get("/api/jobs/archive/NOPE")
        assert missing.status_code == 404
    store.close()


def test_health_reports_ok():
    resp = server.app.test_client().get("/api/health")
    assert resp.status_code == 200
    assert resp.get_json()["ok"] is True
