from src.outreach import (
    PersonalizedOutreachGenerator,
    _lookup_recommendation,
    generate_personalized_outreach,
)


def _result():
    recs = [
        {
            "brand": "NIKE",
            "product": "running sneakers",
            "category": "FOOTWEAR",
            "type": "DIRECT",
            "score": 0.92,
            "confidence": 0.92,
            "appearances": 3,
            "reasons": [
                "LOGO / ON-SCREEN DETECTED — 3 appearance(s)",
                "STRONG EVIDENCE — CONFIDENCE 92%",
                "CREATOR-BRAND AFFINITY — 88%",
            ],
        },
        {
            "brand": "ADIDAS",
            "product": "training shoes",
            "category": "FOOTWEAR",
            "type": "SUGGESTED",
            "score": 0.61,
            "reasons": [
                "SAME CATEGORY (FOOTWEAR) AS NIKE",
                "COMPLEMENTARY TO NIKE",
            ],
        },
    ]
    profile = {
        "creator_id": "c1",
        "handle": "@runner",
        "followers": 120000,
        "engagement_rate": 0.047,
        "categories": {"FOOTWEAR": 8, "APPAREL": 5},
        "dominant_category": "FOOTWEAR",
        "brand_tallies": {"NIKE": 2},
        "videos_analyzed": 10,
        "production_quality": 0.8,
    }
    return {
        "layer3": {"recommendations": recs},
        "layer2d": {"creator_profile": profile},
        "layer2c": {"brand_memory": {"brands": ["NIKE", "PUMA"], "size": 2}},
    }


def test_generate_direct_brand_ok():
    out = generate_personalized_outreach(_result(), "NIKE", target="brands@nike.com")
    assert out["status"] == "ok"
    assert out["brand"] == "NIKE"
    assert "NIKE" in out["subject"]
    assert "running sneakers" in out["body"]
    assert out["rationale"]["type"] == "DIRECT"
    assert "run" in out["body"].lower()
    assert "@runner" in out["body"]


def test_generate_suggested_brand_ok():
    out = generate_personalized_outreach(_result(), "ADIDAS")
    assert out["status"] == "ok"
    assert out["type"] == "SUGGESTED"
    assert any("adjacent" in f for f in out["rationale"]["evidence_facts"])


def test_fail_closed_for_unrecommended_brand():
    out = generate_personalized_outreach(_result(), "GUCCI")
    assert out["status"] == "not_recommended"
    assert "error" in out


def test_always_emits_all_tone_variants():
    out = generate_personalized_outreach(_result(), "NIKE")
    assert set(out["tones"].keys()) == {
        "professional", "creator_voice", "data_driven"
    }
    for t in out["tones"].values():
        assert t["subject"] and t["body"]


def test_creator_voice_tone_is_distinct():
    out = generate_personalized_outreach(_result(), "NIKE", tone="creator_voice")
    assert out["tones"]["creator_voice"]["subject"] == out["subject"]
    assert "Hey NIKE" in out["body"]


def test_cross_video_memory_signal_grounded():
    out = generate_personalized_outreach(_result(), "NIKE")
    assert out["rationale"]["cross_video_memory"] is True
    assert "more than one piece of content" in out["body"]


def test_handler_interface():
    gen = PersonalizedOutreachGenerator()
    out = gen.generate(_result(), "ADIDAS")
    assert out["status"] == "ok"
    assert out["brand"] == "ADIDAS"


def test_lookup_is_case_and_space_insensitive():
    recs = [{"brand": " NIKE "}]
    hit = _lookup_recommendation(recs, "nike")
    assert hit is not None
    assert _lookup_recommendation(recs, "") is None


def test_no_recommendations_fails_closed():
    out = generate_personalized_outreach({}, "NIKE")
    assert out["status"] == "not_recommended"
