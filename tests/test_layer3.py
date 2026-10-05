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


def test_relation_same_category():
    g = KnowledgeGraph(_mini_catalog())
    rel, weight = g.relation("NIKE", "PUMA")
    assert rel == "same_category"
    assert weight == 1.0


def test_relation_none_when_no_shared_or_complementary():
    g = KnowledgeGraph(_mini_catalog())
    assert g.relation("NIKE", "SONY")[0] == "none"


def test_relation_complementary_across_related_category_cluster():
    # NIKE (APPAREL/FOOTWEAR/SPORTS) and a mock OUTDOOR/DRINKWARE brand are
    # complementary via the apparel/outdoor cluster.
    catalog = {
        "NIKE": {"category": "APPAREL", "categories": ["APPAREL", "SPORTS"]},
        "YETI": {"category": "OUTDOOR", "categories": ["OUTDOOR", "DRINKWARE"]},
    }
    g = KnowledgeGraph(catalog)
    rel, _ = g.relation("NIKE", "YETI")
    assert rel == "complementary"


def test_suggested_with_relation_types_drivers():
    # NIKE on screen -> PUMA is same_category (competitor) driven by NIKE;
    # a YETI-like brand in the catalog is complementary.
    catalog = {
        "NIKE": {"category": "APPAREL", "categories": ["APPAREL", "FOOTWEAR", "SPORTS"]},
        "PUMA": {"category": "APPAREL", "categories": ["APPAREL", "FOOTWEAR", "SPORTS"]},
        "STANLEY": {"category": "OUTDOOR", "categories": ["OUTDOOR", "DRINKWARE"]},
        "SONY": {"category": "ELECTRONICS", "categories": ["ELECTRONICS"]},
    }
    g = KnowledgeGraph(catalog)
    sugg = g.suggested_with_relation(["NIKE"])
    # PUMA: same_category, driven by NIKE
    puma_rels = [r for r in sugg["PUMA"] if r[0] == "same_category"]
    assert puma_rels and puma_rels[0][2] == "NIKE"
    # STANLEY: complementary
    stanley_rels = [r for r in sugg["STANLEY"] if r[0] == "complementary"]
    assert stanley_rels
    # SONY shares no category and is not complementary to Nike.
    assert "SONY" not in sugg or all(r[0] == "none" for r in sugg["SONY"])


def test_best_relation_to_prefers_same_category_over_complementary():
    catalog = {
        "NIKE": {"category": "APPAREL", "categories": ["APPAREL", "FOOTWEAR", "SPORTS"]},
        "PUMA": {"category": "APPAREL", "categories": ["APPAREL", "FOOTWEAR", "SPORTS"]},
        "STANLEY": {"category": "OUTDOOR", "categories": ["OUTDOOR", "DRINKWARE"]},
    }
    g = KnowledgeGraph(catalog)
    rel, w, det = g.best_relation_to("PUMA", ["NIKE"])
    assert rel == "same_category" and det == "NIKE"
    rel2, w2, _ = g.best_relation_to("STANLEY", ["NIKE"])
    assert rel2 == "complementary"
    assert w2 < w  # complementary is weighted below same-category


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


def test_affinity_model_modulates_ranking_when_fitted():
    from src.layer3.affinity import CreatorBrandAffinityModel

    aff = CreatorBrandAffinityModel(embed_dim=16, n_layers=2, lr=1e-2)
    # The creator has strong history with ADIDAS and PUMA.
    aff.fit(
        ["creator", "creator", "creator", "creator"],
        ["ADIDAS", "PUMA", "NIKE", "NIKE"],
        epochs=30,
        seed=3,
    )
    rec = BrandRecommender(
        graph=KnowledgeGraph(_mini_catalog()),
        affinity_model=aff,
        affinity_blend=0.6,
    )
    out = rec.recommend(_timeline(), top_k=12, creator_id="creator")
    suggested = {r["brand"]: r for r in out if r["type"] == "SUGGESTED"}
    # With learned affinity, ADIDAS/PUMA (past history) should outrank SONY/APPLE
    # which the creator never engaged with.
    if "ADIDAS" in suggested and "APPLE" in suggested:
        assert suggested["ADIDAS"]["score"] > suggested["APPLE"]["score"]
    assert any("AFFINITY" in " ".join(r["reasons"]).upper() for r in out)


def test_affinity_ignored_on_cold_start_creator():
    from src.layer3.affinity import CreatorBrandAffinityModel

    aff = CreatorBrandAffinityModel(embed_dim=16, n_layers=1, lr=1e-2)
    aff.fit(["known"], ["NIKE", "ADIDAS"], epochs=5, seed=1)
    rec = BrandRecommender(
        graph=KnowledgeGraph(_mini_catalog()),
        affinity_model=aff,
        affinity_blend=0.6,
    )
    # An unseen creator must fall back to pure graph/evidence scores.
    out = rec.recommend(_timeline(), top_k=12, creator_id="brand_new_creator")
    assert out[0]["brand"] == "NIKE"
    # Pinned against a blend-free run rather than a hardcoded evidence number:
    # the intent is "affinity contributes nothing", which stays true as the
    # evidence rule itself evolves.
    plain = BrandRecommender(graph=KnowledgeGraph(_mini_catalog())).recommend(
        _timeline(), top_k=12
    )
    assert out[0]["score"] == plain[0]["score"]
    assert not any("AFFINITY" in r.upper() for r in out[0]["reasons"])


def _speech_only_timeline(n_mentions, conf=0.0):
    """A brand that is only ever named out loud — never shown on screen."""
    appearances = [
        {"frame_index": None, "timestamp": 10.0 * i, "modality": "speech", "confidence": 1.0}
        for i in range(n_mentions)
    ]
    if conf:
        appearances.append(
            {"frame_index": 5, "timestamp": 1.0, "modality": "logo", "confidence": conf}
        )
    return {
        "NIKE": {
            "appearance_count": len(appearances),
            "modalities": ["logo", "speech"] if conf else ["speech"],
            "cross_scene": False,
            "appearances": appearances,
        },
    }


def test_repeated_spoken_mentions_outrank_a_single_one():
    """A brand named ten times should not score the same as one named once."""
    from src.layer3.recommender import BrandRecommender as R

    rec = R(graph=KnowledgeGraph(_mini_catalog()))
    # Select NIKE, not [0]: a SUGGESTED brand with 0.7 category affinity can
    # legitimately outrank a weakly-evidenced DIRECT brand.
    nike = lambda tl: next(r for r in rec.recommend(tl, top_k=12) if r["brand"] == "NIKE")
    once, many = nike(_speech_only_timeline(1)), nike(_speech_only_timeline(10))
    assert many["score"] > once["score"]


def test_spoken_evidence_saturates():
    """Past the saturation point extra mentions must not keep adding score."""
    from src.layer3.recommender import BrandRecommender as R

    rec = R(graph=KnowledgeGraph(_mini_catalog()))
    at_cap = rec.recommend(_speech_only_timeline(8), top_k=12)[0]
    way_past = rec.recommend(_speech_only_timeline(40), top_k=12)[0]
    assert at_cap["score"] == way_past["score"]


def test_spoken_evidence_stays_below_logo_strength():
    """Naming a brand is weaker than showing it: a speech-only brand must never
    reach the 0.5 strong-evidence line on mentions alone."""
    from src.layer3.recommender import SPEECH_CEILING

    rec = BrandRecommender(graph=KnowledgeGraph(_mini_catalog()))
    out = rec.recommend(_speech_only_timeline(50), top_k=12)[0]
    assert out["score"] <= SPEECH_CEILING
    assert not any("STRONG EVIDENCE" in r for r in out["reasons"])


def test_speech_fills_headroom_without_clamping_strong_brands():
    """A strong logo plus one mention must not flatten to 1.0 and erase the
    ranking between strong brands."""
    from src.layer3.recommender import BrandRecommender as R

    rec = R(graph=KnowledgeGraph(_mini_catalog()))
    strong = rec.recommend(_speech_only_timeline(1, conf=0.9), top_k=12)[0]
    assert strong["score"] > 0.9
    assert strong["score"] < 1.0


def test_mention_count_is_reported_in_reasons():
    """The count is what makes the claim defensible, so it must be stated."""
    rec = BrandRecommender(graph=KnowledgeGraph(_mini_catalog()))
    nike = lambda n: next(
        r for r in rec.recommend(_speech_only_timeline(n), top_k=12)
        if r["brand"] == "NIKE"
    )
    assert "NAMED 4x IN THE AUDIO" in " ".join(nike(4)["reasons"])
    assert "NAMED ONCE IN THE AUDIO" in " ".join(nike(1)["reasons"])


def test_explicit_brand_evidence_is_not_double_counted():
    """Layer 2b's fusion already counts spoken mentions as an evidence source, so
    an explicit strength must be returned untouched."""
    rec = BrandRecommender(graph=KnowledgeGraph(_mini_catalog()))
    out = rec.recommend(_speech_only_timeline(20), brand_evidence={"NIKE": 0.42}, top_k=12)
    assert out[0]["score"] == 0.42
