"""Tests for Layer 3: KnowledgeGraph + BrandRecommender.

These are the core Phase 3 deliverables (graph-based cold-start recommendation)
and previously had zero coverage. Tests cover:
  - knowledge-graph adjacency / suggestion / shared-category behavior
  - DIRECT recommendations (brands detected in the video)
  - SUGGESTED recommendations (adjacent brands that never appeared)
  - explainability (every recommendation carries non-empty `reasons`)
  - cold-start baseline: a suggested brand gets at least a floor score
"""

from src.layer3.knowledge_graph import KnowledgeGraph
from src.layer3.recommender import BrandRecommender


# ─────────────────────────────────────────────────────────────
# KnowledgeGraph
# ─────────────────────────────────────────────────────────────

def _mini_catalog():
    return {
        "NIKE": {"product": "Nike Air", "category": "APPAREL",
                 "categories": ["APPAREL", "FOOTWEAR", "SPORTS"]},
        "ADIDAS": {"product": "Adidas Samba", "category": "APPAREL",
                   "categories": ["APPAREL", "FOOTWEAR", "SPORTS"]},
        "PUMA": {"product": "Puma Suede", "category": "APPAREL",
                 "categories": ["APPAREL", "FOOTWEAR", "SPORTS"]},
        "SONY": {"product": "PlayStation", "category": "ELECTRONICS",
                 "categories": ["ELECTRONICS", "GAMING"]},
        "APPLE": {"product": "iPhone", "category": "ELECTRONICS",
                  "categories": ["ELECTRONICS", "SMARTPHONE"]},
    }


def test_neighbors_share_category():
    g = KnowledgeGraph(_mini_catalog())
    nbrs = g.neighbors("NIKE")
    assert "ADIDAS" in nbrs and "PUMA" in nbrs
    assert "SONY" not in nbrs and "APPLE" not in nbrs


def test_neighbors_exclude_self_and_unknown():
    g = KnowledgeGraph(_mini_catalog())
    assert "NIKE" not in g.neighbors("NIKE")
    assert g.neighbors("NONEXISTENT") == []


def test_suggest_for_never_appeared_brand():
    g = KnowledgeGraph(_mini_catalog())
    # Content features Nike on screen; Puma/Adidas should be suggested, Apple not.
    suggested = g.suggest_for(["NIKE"])
    assert "PUMA" in suggested and "ADIDAS" in suggested
    assert "APPLE" not in suggested
    # Detected brands themselves are never suggested.
    assert "NIKE" not in suggested


def test_suggest_excludes_all_detected():
    g = KnowledgeGraph(_mini_catalog())
    suggested = g.suggest_for(["NIKE", "ADIDAS", "PUMA"])
    assert "NIKE" not in suggested
    assert "ADIDAS" not in suggested
    assert "PUMA" not in suggested
    # The apparel cluster is fully detected; SONY/APPLE share no category with
    # it, so nothing adjacent remains to suggest.
    assert suggested == []


def test_shared_categories():
    g = KnowledgeGraph(_mini_catalog())
    assert set(g.shared_categories("NIKE", "PUMA")) == {"APPAREL", "FOOTWEAR", "SPORTS"}
    assert g.shared_categories("NIKE", "SONY") == []


def test_unknown_brand_defaults_to_general_category():
    g = KnowledgeGraph(_mini_catalog())
    assert g.categories_for("NOPE") == ["GENERAL"]


# ─────────────────────────────────────────────────────────────
# BrandRecommender
# ─────────────────────────────────────────────────────────────

def _timeline():
    """A Nike-only video (logo + spoken mention) → DIRECT Nike, SUGGESTED others."""
    return {
        "NIKE": {
            "appearance_count": 2,
            "modalities": ["logo", "speech"],
            "cross_scene": True,
            "appearances": [
                {"frame_index": 10, "timestamp": 1.0, "modality": "logo", "confidence": 0.9},
                {"frame_index": 400, "timestamp": 20.0, "modality": "speech", "confidence": 0.8},
            ],
        },
    }


def test_direct_recommendation_ranked_first():
    rec = BrandRecommender(graph=KnowledgeGraph(_mini_catalog()), category_affinity=0.7)
    out = rec.recommend(_timeline(), top_k=12)
    # DIRECT Nike (evidence 0.9) should outrank the cold-start SUGGESTED baseline.
    assert out[0]["brand"] == "NIKE"
    assert out[0]["type"] == "DIRECT"


def test_direct_has_explainable_reasons():
    rec = BrandRecommender(graph=KnowledgeGraph(_mini_catalog()))
    out = rec.recommend(_timeline(), top_k=12)
    nike = next(r for r in out if r["brand"] == "NIKE")
    assert nike["reasons"]
    joined = " ".join(nike["reasons"]).upper()
    assert "LOGO" in joined
    assert "SPOKEN" in joined


def test_suggested_brands_include_adjacent_not_detected():
    rec = BrandRecommender(graph=KnowledgeGraph(_mini_catalog()))
    out = rec.recommend(_timeline(), top_k=12)
    suggested = [r for r in out if r["type"] == "SUGGESTED"]
    suggested_brands = {r["brand"] for r in suggested}
    # Adjacent to Nike and not on screen.
    assert "PUMA" in suggested_brands
    assert "ADIDAS" in suggested_brands
    assert "NIKE" not in suggested_brands


def test_suggested_brand_gets_cold_start_floor_score():
    rec = BrandRecommender(graph=KnowledgeGraph(_mini_catalog()), category_affinity=0.7)
    out = rec.recommend(_timeline(), top_k=12)
    puma = next(r for r in out if r["brand"] == "PUMA")
    assert puma["type"] == "SUGGESTED"
    assert puma["score"] >= 0.15  # never a bare 0 recommendation


def test_suggested_reasons_reference_driving_brand():
    rec = BrandRecommender(graph=KnowledgeGraph(_mini_catalog()))
    out = rec.recommend(_timeline(), top_k=12)
    puma = next(r for r in out if r["brand"] == "PUMA")
    assert "NIKE" in " ".join(puma["reasons"])


def test_top_k_respected_and_sorted_by_score():
    rec = BrandRecommender(graph=KnowledgeGraph(_mini_catalog()))
    out = rec.recommend(_timeline(), top_k=3)
    assert len(out) <= 3
    scores = [r["score"] for r in out]
    assert scores == sorted(scores, reverse=True)


def test_no_detected_brands_yields_empty_or_baseline_only():
    # With no timeline brand, suggest_for([]) is empty and no DIRECT exists.
    rec = BrandRecommender(graph=KnowledgeGraph(_mini_catalog()))
    out = rec.recommend({}, top_k=12)
    assert out == []


def test_recommender_with_real_catalog():
    # Smoke test against the real curated catalog: Nike appears, get suggestions.
    rec = BrandRecommender()
    out = rec.recommend(_timeline(), top_k=12)
    assert out[0]["brand"] == "NIKE"
    for r in out:
        assert r["brand"]
        assert r["product"]
        assert r["category"]
        assert r["reasons"]


def test_evidence_override_drives_direct_ranking():
    # If brand_evidence gives favorite a higher score, it should lead.
    rec = BrandRecommender(graph=KnowledgeGraph(_mini_catalog()))
    out = rec.recommend(
        _timeline(),
        brand_evidence={"NIKE": 0.99},
        top_k=12,
    )
    assert out[0]["brand"] == "NIKE"
    assert out[0]["score"] == round(min(1.0, 0.99), 3)


def test_confidence_mirrors_score_on_records():
    rec = BrandRecommender(graph=KnowledgeGraph(_mini_catalog()))
    for r in rec.recommend(_timeline(), top_k=12):
        assert r["confidence"] == r["score"]
