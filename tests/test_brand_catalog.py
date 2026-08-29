"""
Unit tests for the shared brand catalog + Layer 2c brand resolver +
Layer 3 knowledge graph / recommender.

These tests use only in-memory fixtures — the production data path
(./data per config.yaml) is never touched.
"""

import sys
from pathlib import Path

import numpy as np

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.brand_catalog import (
    BRAND_CATALOG,
    build_text_queries,
    canonical_name,
    categories_for,
    contact_for,
    find_brand_mentions,
    lookup,
    match_brand,
    normalize_text,
    product_for,
)
from src.layer2.brand_resolver import (
    BrandResolver,
    brand_evidence_from_timeline,
    build_brand_timeline,
)
from src.layer3.knowledge_graph import KnowledgeGraph
from src.layer3.recommender import BrandRecommender


# ============================================================
# Catalog — matching
# ============================================================

class TestBrandCatalog:
    def test_catalog_is_populated(self):
        assert len(BRAND_CATALOG) >= 30
        for brand, info in BRAND_CATALOG.items():
            assert info.get("product")
            assert info.get("category")
            assert info.get("categories")
            assert info.get("aliases")
            assert info.get("contact_email")

    def test_canonical_name_exact(self):
        assert canonical_name("NIKE") == "NIKE"
        assert canonical_name("Coca-Cola") == "COCA-COLA"
        assert canonical_name("Levi's") == "LEVI'S"

    def test_canonical_name_unknown(self):
        assert canonical_name("text logo") is None

    def test_match_brand_from_detector_class(self):
        assert match_brand("Samsung logo") == "SAMSUNG"
        assert match_brand("Nike logo") == "NIKE"
        assert match_brand("generic text logo") is None

    def test_match_brand_alias(self):
        assert match_brand("check out this swoosh") == "NIKE"
        assert match_brand("grab a coke") == "COCA-COLA"

    def test_match_brand_short_alias_not_match(self):
        # Aliases shorter than 3 chars are ignored (avoid false positives)
        assert match_brand("lg") is None

    def test_find_brand_mentions(self):
        mentions = find_brand_mentions(
            "I love my new Samsung and my Adidas shoes"
        )
        brands = [m["brand"] for m in mentions]
        assert "SAMSUNG" in brands
        assert "ADIDAS" in brands
        positions = [m["position"] for m in mentions]
        assert positions == sorted(positions)

    def test_find_brand_mentions_empty(self):
        assert find_brand_mentions("") == []
        assert find_brand_mentions("nothing here at all") == []

    def test_lookup_and_contact(self):
        info = lookup("Nike")
        assert info is not None
        assert info["category"] == "APPAREL"
        assert lookup("text logo") is None

    def test_contact_for(self):
        contact = contact_for("NIKE")
        assert contact["email"] == "partnerships@nike.com"
        assert "nike" in contact["website"]
        assert contact["verified"] is False

    def test_product_and_categories(self):
        assert product_for("NIKE") == "Nike Air"
        assert "FOOTWEAR" in categories_for("NIKE")
        assert categories_for("UNKNOWN BRAND") == ["GENERAL"]

    def test_text_queries_cover_all_brands(self):
        queries = build_text_queries()
        assert len(queries) == len(BRAND_CATALOG)
        assert "NIKE logo" in queries
        assert all(q.endswith(" logo") for q in queries)

    def test_normalize(self):
        assert normalize_text("  Coca-Cola  ") == "COCA COLA"
        assert normalize_text("") == ""

    # ── Multilingual (Devanagari) mention detection ─────────────
    def test_devanagari_normalize_preserved(self):
        # Devanagari aliases must survive normalization (matras + anusvara)
        assert normalize_text("सैमसंग") == "सैमसंग"
        assert normalize_text("नाइके के जूते") == "नाइके के जूते"

    def test_find_mentions_devanagari_exact(self):
        mentions = find_brand_mentions(
            "मैंने सैमसंग गैलेक्सी एस24 रिव्यू किया है"
        )
        assert "SAMSUNG" in [m["brand"] for m in mentions]

    def test_find_mentions_devanagari_multiple(self):
        mentions = find_brand_mentions(
            "मुझे नाइके के जूते और एडिडास दोनों पसंद हैं"
        )
        brands = {m["brand"] for m in mentions}
        assert brands == {"NIKE", "ADIDAS"}

    def test_find_mentions_devanagari_negative(self):
        assert find_brand_mentions("आज का दिन बहुत अच्छा था") == []

    def test_find_mentions_code_switched(self):
        mentions = find_brand_mentions(
            "भाई ये Samsung का फोन है बहुत बढ़िया है"
        )
        assert "SAMSUNG" in [m["brand"] for m in mentions]

    def test_fuzzy_off_by_default_no_phonetic_variant(self):
        # 'सैमसं' (missing trailing ग) is distance-1 from 'सैमसंग' but is NOT
        # an explicit alias — the default exact path must NOT match it.
        assert find_brand_mentions("इस सैमसं फोन की बैटरी अच्छी है") == []

    def test_fuzzy_on_catches_phonetic_variant(self):
        mentions = find_brand_mentions(
            "इस सैमसं फोन की बैटरी अच्छी है", fuzzy=True, max_distance=1
        )
        assert "SAMSUNG" in [m["brand"] for m in mentions]


# ============================================================
# Layer 2c — BrandResolver
# ============================================================

class TestBrandResolver:
    def _frame(self, h=100, w=160):
        return np.zeros((h, w, 3), dtype=np.uint8)

    def test_resolve_from_class_name(self):
        resolver = BrandResolver()
        dets = [[{"class_name": "Nike logo", "bbox": [0, 0, 10, 10], "confidence": 0.9}]]
        out = resolver.resolve(dets, [self._frame()])
        assert out[0][0]["brand"] == "NIKE"
        assert out[0][0]["class_name"] == "NIKE"

    def test_generic_logo_stays_unresolved_without_ocr(self):
        resolver = BrandResolver()
        dets = [[{"class_name": "text logo", "bbox": [0, 0, 10, 10], "confidence": 0.9}]]
        out = resolver.resolve(dets, [self._frame()])
        assert out[0][0]["brand"] is None

    def test_crop_ocr_resolves_generic_logo(self):
        class FakeOCR:
            def extract_text(self, crop):
                return [{"text": "adidas"}]
        resolver = BrandResolver(ocr_extractor=FakeOCR())
        dets = [[{"class_name": "text logo", "bbox": [10, 10, 40, 30], "confidence": 0.9}]]
        out = resolver.resolve(dets, [self._frame()])
        assert out[0][0]["brand"] == "ADIDAS"

    def test_crop_ocr_no_match_stays_unresolved(self):
        class FakeOCR:
            def extract_text(self, crop):
                return [{"text": "qwerty nonsense"}]
        resolver = BrandResolver(ocr_extractor=FakeOCR())
        dets = [[{"class_name": "brand logo", "bbox": [10, 10, 40, 30], "confidence": 0.9}]]
        out = resolver.resolve(dets, [self._frame()])
        assert out[0][0]["brand"] is None

    def test_resolver_counts_logged(self):
        resolver = BrandResolver()
        dets = [
            [{"class_name": "Adidas logo", "bbox": [0, 0, 5, 5], "confidence": 0.8}],
            [{"class_name": "text logo", "bbox": [0, 0, 5, 5], "confidence": 0.8}],
        ]
        frames = [self._frame(), self._frame()]
        out = resolver.resolve(dets, frames)
        assert out[0][0]["brand"] == "ADIDAS"
        assert out[1][0]["brand"] is None


# ============================================================
# Layer 2c — brand timeline
# ============================================================

class TestBrandTimeline:
    def test_timeline_from_logos_only(self):
        logos = [
            [{"brand": "NIKE", "confidence": 0.9}],
            [],
            [{"brand": "NIKE", "confidence": 0.8}],
        ]
        tl = build_brand_timeline(logos, [], video_fps=1.0)
        nike = tl["NIKE"]
        assert nike["appearance_count"] == 2
        assert nike["modalities"] == ["logo"]
        assert nike["cross_scene"] is False
        assert nike["first_seen"] == 0.0
        assert nike["last_seen"] == 2.0

    def test_cross_scene_flag(self):
        logos = [[{"brand": "NIKE", "confidence": 0.9}]]
        mentions = [{"brand": "NIKE", "position": 5}]
        tl = build_brand_timeline(logos, mentions, transcript="hello world")
        assert tl["NIKE"]["cross_scene"] is True
        assert tl["NIKE"]["modalities"] == ["logo", "speech"]

    def test_unresolved_logos_ignored(self):
        logos = [[{"brand": None, "confidence": 0.9}]]
        tl = build_brand_timeline(logos, [])
        assert tl == {}

    def test_evidence_from_timeline(self):
        logos = [[{"brand": "ADIDAS", "confidence": 0.8}]]
        mentions = [{"brand": "NIKE", "position": 5}]
        tl = build_brand_timeline(logos, mentions, transcript="hello nike")
        ev = brand_evidence_from_timeline(tl)
        assert ev["ADIDAS"] == 0.8
        assert ev["NIKE"] == 0.6  # speech-only floor


# ============================================================
# Layer 3 — knowledge graph + recommender
# ============================================================

class TestKnowledgeGraph:
    def test_neighbors_share_category(self):
        g = KnowledgeGraph()
        assert "PUMA" in g.neighbors("NIKE")
        assert "NIKE" not in g.neighbors("NIKE")

    def test_suggest_for_excludes_detected(self):
        g = KnowledgeGraph()
        suggested = g.suggest_for(["NIKE"])
        assert "NIKE" not in suggested
        assert "PUMA" in suggested

    def test_shared_categories(self):
        g = KnowledgeGraph()
        assert "FOOTWEAR" in g.shared_categories("NIKE", "PUMA")


class TestBrandRecommender:
    def test_direct_recommendation(self):
        tl = build_brand_timeline(
            [[{"brand": "NIKE", "confidence": 0.9}]], [],
            video_fps=1.0,
        )
        recs = BrandRecommender().recommend(tl)
        assert recs
        top = recs[0]
        assert top["brand"] == "NIKE"
        assert top["type"] == "DIRECT"
        assert any("LOGO" in r for r in top["reasons"])
        assert top["appearances"] == 1

    def test_suggested_recommendation(self):
        tl = build_brand_timeline(
            [[{"brand": "NIKE", "confidence": 0.9}]], [],
            video_fps=1.0,
        )
        recs = BrandRecommender().recommend(tl, top_k=50)
        suggested = [r for r in recs if r["type"] == "SUGGESTED"]
        assert any(r["brand"] == "PUMA" for r in suggested)
        puma = next(r for r in suggested if r["brand"] == "PUMA")
        assert any("SAME CATEGORY" in r for r in puma["reasons"])
        assert puma["appearances"] == 0

    def test_ranked_by_score_desc(self):
        tl = build_brand_timeline(
            [[{"brand": "NIKE", "confidence": 0.9}]], [],
            video_fps=1.0,
        )
        recs = BrandRecommender().recommend(tl, top_k=50)
        scores = [r["score"] for r in recs]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_respected(self):
        tl = build_brand_timeline(
            [[{"brand": "NIKE", "confidence": 0.9}]], [],
            video_fps=1.0,
        )
        recs = BrandRecommender().recommend(tl, top_k=5)
        assert len(recs) <= 5

    def test_empty_timeline_no_recs(self):
        recs = BrandRecommender().recommend({})
        assert recs == []
