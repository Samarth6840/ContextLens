"""Tests for Layer 3 CreatorBrandAffinityModel (LightGCN).

Verifies the learned model trains, distinguishes interacted from
non-interacted brands, and fails closed (returns None) on cold start so callers
fall back to the graph heuristic.
"""

import pytest

try:
    from src.layer3.affinity import CreatorBrandAffinityModel
    HAS_TORCH = True
except Exception:  # noqa: BLE001
    CreatorBrandAffinityModel = None
    HAS_TORCH = False


pytestmark = pytest.mark.skipif(
    not HAS_TORCH, reason="torch not available"
)


@pytest.fixture
def trained_model():
    m = CreatorBrandAffinityModel(embed_dim=16, n_layers=2, lr=1e-2)
    summary = m.fit(
        ["c1", "c1", "c1", "c2", "c2", "c3"],
        ["NIKE", "ADIDAS", "PUMA", "NIKE", "ADIDAS", "SONY"],
        epochs=25,
        seed=1,
    )
    return m, summary


def test_untrained_model_scores_nothing(trained_model):
    m, _ = trained_model
    fresh = CreatorBrandAffinityModel()
    assert fresh.is_fitted is False
    assert fresh.predict_affinity("c1", ["NIKE"]) is None


def test_fit_honestly_sets_fitted(trained_model):
    m, summary = trained_model
    assert m.is_fitted is True
    assert summary["epochs"] == 25
    assert summary["creators"] == 3


def test_interacted_brand_scores_higher_than_non_interacted(trained_model):
    m, _ = trained_model
    # c1 interacted with NIKE/ADIDAS/PUMA; SONY no.
    scores = m.predict_affinity("c1", ["NIKE", "SONY"])
    assert scores[0] > scores[1]


def test_unknown_brand_scores_zero(trained_model):
    m, _ = trained_model
    scores = m.predict_affinity("c1", ["NOT_A_BRAND"])
    assert scores[0] == 0.0


def test_cold_start_unseen_creator_returns_none(trained_model):
    m, _ = trained_model
    assert m.predict_affinity("nobody_here", ["NIKE"]) is None


def test_empty_fit_scores_nothing():
    m = CreatorBrandAffinityModel()
    m.fit([], [])
    assert m.is_fitted is False


def test_scores_are_in_unit_range(trained_model):
    m, _ = trained_model
    for b in ["NIKE", "SONY", "PUMA"]:
        s = m.predict_affinity("c1", [b])[0]
        assert 0.0 <= s <= 1.0


def test_deterministic_with_seed():
    a = CreatorBrandAffinityModel(embed_dim=8, n_layers=1, lr=1e-2)
    b = CreatorBrandAffinityModel(embed_dim=8, n_layers=1, lr=1e-2)
    a.fit(["c1", "c2"], ["NIKE", "PUMA"], epochs=5, seed=7)
    b.fit(["c1", "c2"], ["NIKE", "PUMA"], epochs=5, seed=7)
    sa = a.predict_affinity("c1", ["NIKE", "PUMA"])
    sb = b.predict_affinity("c1", ["NIKE", "PUMA"])
    for x, y in zip(sa, sb):
        assert x == pytest.approx(y, abs=1e-4)
