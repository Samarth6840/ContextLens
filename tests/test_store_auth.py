import os
import tempfile

from src.auth import (
    SessionManager,
    hash_password,
    verify_password,
)
from src.store import JobStore, prune_for_store


# ── auth ────────────────────────────────────────────────────────────────


def test_password_hash_roundtrip():
    record = hash_password("s3cret!")
    assert verify_password("s3cret!", record) is True
    assert verify_password("wrong", record) is False


def test_hash_is_randomized_and_not_plaintext():
    a = hash_password("same pass")
    b = hash_password("same pass")
    assert a["hash"] != b["hash"]  # fresh salt each time
    assert "same pass" not in a["hash"]


def test_verify_rejects_corrupt_record():
    assert verify_password("x", {}) is False
    assert verify_password("x", {"salt": "zz", "iterations": "1",
                                 "hash": "nothex(y)"}) is False


def test_session_create_validate_and_revoke():
    mgr = SessionManager(ttl_seconds=3600)
    token = mgr.create("alice")
    assert mgr.validate(token) == "alice"
    assert mgr.active_sessions() == 1
    mgr.revoke(token)
    assert mgr.validate(token) is None
    assert mgr.active_sessions() == 0


def test_session_rejects_garbage_and_expiry():
    mgr = SessionManager(ttl_seconds=-1)  # already expired
    token = mgr.create("bob")
    assert mgr.validate(token) is None  # expired on access
    assert mgr.validate("bogus-token") is None
    assert mgr.validate(None) is None


# ── store ───────────────────────────────────────────────────────────────


def _tmp_db():
    tmpdir = tempfile.mkdtemp()
    return os.path.join(tmpdir, "test.db")


def test_store_save_load_roundtrip():
    store = JobStore(_tmp_db())
    job = {"job_id": "J1", "status": "done", "title": "T",
           "dashboard": {"confidence": 0.9}}
    store.save("J1", job)
    loaded = store.load("J1")
    assert loaded["job_id"] == "J1"
    assert loaded["dashboard"]["confidence"] == 0.9
    store.close()


def test_store_survives_reopen():
    path = _tmp_db()
    store = JobStore(path)
    store.save("J1", {"job_id": "J1", "status": "done", "title": "T"})
    store.close()

    reopened = JobStore(path)
    assert reopened.load("J1")["title"] == "T"
    assert reopened.count() == 1
    reopened.close()


def test_store_list_and_delete():
    store = JobStore(_tmp_db())
    store.save("A", {"job_id": "A", "created_at": "2026-01-01T00:00:00Z",
                     "status": "done"})
    store.save("B", {"job_id": "B", "created_at": "2026-01-02T00:00:00Z",
                     "status": "error"})
    assert store.count() == 2
    ids = [j["job_id"] for j in store.list()]
    assert ids == ["B", "A"]  # newest first
    assert store.delete("A") is True
    assert store.load("A") is None
    assert store.delete("MISSING") is False
    store.close()


def test_prune_for_store_drops_heavy_embeds():
    result = {
        "video_path": "/tmp/v.mp4",
        "layer2b": {"confidence": 0.8},
        "layer2a": {"fused_embed": [0.1] * 1000},
        "layer1": {"num_embeddings": 500},
        "layer3": {"recommendations": [{"brand": "NIKE"}]},
        "layer2d": {"creator_profile": {"handle": "@r"}},
        "layer2c": {"memory_brands": ["NIKE"], "memory_size": 1},
    }
    pruned = prune_for_store(result)
    assert "fused_embed" not in pruned
    assert "layer2a" not in pruned
    assert pruned["layer3"]["recommendations"] == [{"brand": "NIKE"}]
    assert pruned["layer2d"]["creator_profile"]["handle"] == "@r"
    assert pruned["layer2c"]["memory_brands"] == ["NIKE"]
