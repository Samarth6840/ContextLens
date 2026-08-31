"""Tests for the Layer 3 affinity training pathway (affinity_trainer).

Verifies interaction extraction from persisted job history and that the
resulting model actually trains and can be attached for ranking.
"""

import pytest

try:
    from src.layer3.affinity import CreatorBrandAffinityModel
    HAS_TORCH = True
except Exception:  # noqa: BLE001
    CreatorBrandAffinityModel = None
    HAS_TORCH = False

from src.layer3.affinity_trainer import (
    extract_interactions,
    train_affinity_from_jobs,
)

pytestmark = pytest.mark.skipif(
    not HAS_TORCH, reason="torch not available"
)


def _job(creator, brand_tallies=None, recs=None):
    result = {
        "layer2d": {"creator_profile": {
            "creator_id": creator,
            "brand_tallies": brand_tallies or {},
        }},
        "layer3": {"recommendations": recs or []},
    }
    return {"creator": creator, "result": result}


def test_extract_interactions_from_profile_tallies():
    jobs = [
        _job("c1", {"NIKE": 2, "ADIDAS": 1}),
        _job("c1", {"NIKE": 1}),
        _job("c2", {"SONY": 3}),
    ]
    interactions = extract_interactions(jobs)
    assert ("c1", "NIKE", 3.0) in interactions  # 2 + 1 summed
    assert ("c1", "ADIDAS", 1.0) in interactions
    assert ("c2", "SONY", 3.0) in interactions


def test_extract_interactions_falls_back_to_direct_recs():
    jobs = [
        _job("c1", recs=[
            {"brand": "PUMA", "type": "DIRECT"},
            {"brand": "ADIDAS", "type": "SUGGESTED"},  # ignored
        ]),
    ]
    interactions = extract_interactions(jobs)
    assert ("c1", "PUMA", 1.0) in interactions
    assert all(b != "ADIDAS" for _c, b, _w in interactions)


def test_train_no_interactions_is_honest_cold_start():
    model, summary = train_affinity_from_jobs([_job("c1", {})], epochs=2)
    assert summary["status"] == "no_interactions"
    assert model.is_fitted is False
    assert model.predict_affinity("c1", ["NIKE"]) is None


def test_train_produces_fitted_model_and_scores():
    jobs = [
        _job("c1", {"NIKE": 2, "ADIDAS": 1}),
        _job("c1", {"NIKE": 1}),
        _job("c2", {"SONY": 3}),
    ]
    model, summary = train_affinity_from_jobs(
        jobs, embed_dim=16, n_layers=2, lr=1e-2, epochs=20, seed=1
    )
    assert summary["status"] == "trained"
    assert model.is_fitted is True
    assert summary["interactions"] == 3
    scores = model.predict_affinity("c1", ["NIKE", "SONY"])
    assert scores is not None
    assert scores[0] > scores[1]  # interacted brand ranks above cold brand
