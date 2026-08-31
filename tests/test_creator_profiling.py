"""Tests for Layer 2d Creator Profiling.

Focus: content-niche profiling and the suppression case (prompt §6) — a creator
who casually wears Nike once but whose content is cooking should NOT be
recommended a Nike collaboration by the graph, while a creator who actually
does fitness gets it.
"""

from src.layer2.creator_profiling import (
    CreatorProfile,
    apply_niche_suppression,
    build_profile_from_results,
    niche_fit_score,
)


def _profile_with_categories(cats):
    p = CreatorProfile("creator")
    p.add_video(categories=cats, brands={"NIKE": 1})
    return p


def test_dominant_category():
    p = _profile_with_categories({"FOOD": 40, "APPAREL": 1, "SPORTS": 1})
    assert p.dominant_category() == "FOOD"
    assert p.category_share("FOOD") == 40 / 42


def test_accumulation_across_videos():
    p = CreatorProfile("c")
    p.add_video(categories={"FOOD": 10, "BEVERAGE": 2}, brands={"NIKE": 1})
    p.add_video(categories={"FOOD": 30, "BEVERAGE": 4})
    assert p.videos_analyzed == 2
    assert p.categories["FOOD"] == 40
    assert p.brand_tallies["NIKE"] == 1


def test_engagement_rate():
    p = CreatorProfile("c")
    p.add_video(followers=100_000, engagements=5_000)
    assert p.engagement_rate == 0.05


def test_niche_fit_low_for_stray_brand():
    # Cooking creator (FOOD-dominant) who once featured NIKE (APPAREL/FOOTWEAR).
    p = _profile_with_categories({"FOOD": 45, "APPAREL": 1})
    fit = niche_fit_score(p, "NIKE")
    assert fit is not None
    assert fit < 0.15


def test_niche_fit_high_for_recurring_brand():
    p = _profile_with_categories({"APPAREL": 30, "FOOTWEAR": 20, "SPORTS": 10})
    fit = niche_fit_score(p, "NIKE")
    assert fit is not None
    assert fit >= 0.5


def test_suppression_downgrades_mismatched_suggested_only():
    p = _profile_with_categories({"FOOD": 45, "APPAREL": 1})
    recs = [
        {"brand": "NIKE", "type": "DIRECT", "score": 0.9},
        {"brand": "PUMA", "type": "SUGGESTED", "score": 0.6},
    ]
    out = apply_niche_suppression(recs, p)
    # DIRECT never suppressed.
    assert out[0]["score"] == 0.9
    assert out[0].get("niche_suppressed", False) is False
    # SUGGESTED mismatched -> suppressed.
    assert out[1]["niche_suppressed"] is True
    assert out[1]["score"] < 0.6


def test_no_suppression_without_content_signal():
    p = CreatorProfile("c")  # no categories -> niche_fit returns None
    recs = [{"brand": "PUMA", "type": "SUGGESTED", "score": 0.6}]
    out = apply_niche_suppression(recs, p)
    assert out[0].get("niche_suppressed", False) is False
    assert out[0]["score"] == 0.6


def test_build_profile_from_results():
    results = [
        {
            "recommendations": [
                {"brand": "NIKE"},
                {"brand": "ADIDAS"},
            ],
        },
        {
            "recommendations": [{"brand": "NIKE"}],
        },
    ]
    p = build_profile_from_results(results, "creator", handle="@cook")
    assert p.handle == "@cook"
    assert p.brand_tallies["NIKE"] == 2
    assert p.brand_tallies["ADIDAS"] == 1
    assert p.videos_analyzed == 2
    assert p.dominant_category() in ("APPAREL", "FOOTWEAR", "SPORTS")


def test_profile_categories_helper():
    from src.pipeline import _profile_categories

    timeline = {
        "NIKE": {"appearance_count": 2},
        "STARBUCKS": {"appearance_count": 1},
    }
    cats = _profile_categories(timeline)
    # Nike -> APPAREL/FOOTWEAR/SPORTS (x2); Starbucks -> BEVERAGE/FOOD.
    assert cats["APPAREL"] == 2
    assert cats["BEVERAGE"] == 1
