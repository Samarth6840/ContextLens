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
from src.layer1.logo_retrieval import _canonical, LogoRetrievalIndex
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

    def test_ocr_crosscheck_overrides_spurious_high_conf_class(self):
        # YOLO-World sometimes CONFIDENTLY (>= gate) mislabels a logo region as a
        # spurious brand ('SUPREME logo' on an Apple wordmark at 0.6). The
        # crop-OCR cross-check must prefer the real on-screen wordmark.
        class FakeOCR:
            def extract_text(self, crop):
                return [{"text": "Apple Intelligence"}]
        resolver = BrandResolver(ocr_extractor=FakeOCR(), class_confidence=0.40)
        dets = [[{"class_name": "SUPREME logo", "bbox": [10, 10, 50, 50], "confidence": 0.6}]]
        out = resolver.resolve(dets, [self._frame()])
        assert out[0][0]["brand"] == "APPLE"
        assert out[0][0]["class_confirmed"] is False
        assert out[0][0]["resolved_vs_class"]["class_brand"] == "SUPREME"

    def test_class_brand_kept_when_ocr_reads_nothing(self):
        # When OCR finds no wordmark, the (above-gate) class label stands.
        class EmptyOCR:
            def extract_text(self, crop):
                return []
        resolver = BrandResolver(ocr_extractor=EmptyOCR(), class_confidence=0.40)
        dets = [[{"class_name": "Supreme logo", "bbox": [10, 10, 50, 50], "confidence": 0.6}]]
        out = resolver.resolve(dets, [self._frame()])
        assert out[0][0]["brand"] == "SUPREME"
        assert out[0][0]["class_confirmed"] is True

    # ── CLIP-retrieval path (Phase 1/2) ──────────────────────────────────
    # Fusion priority: OCR > CLIP retrieval > class-label. YOLO-World is weak on
    # icon-only/stylized marks (no wordmark for OCR) and can CONFIDENTLY mislabel
    # them (e.g. 'SUPREME logo' on a Samsung foldable at 0.45-0.60). CLIP image
    # retrieval against a per-brand reference bank is the primary classifier for
    # that case. These use a STUB index (real CLIP is heavy/non-deterministic for
    # unit tests) returning controlled candidates.

    class FakeRetrieval:
        def __init__(self, candidates, min_sim=0.22):
            self.candidates = candidates
            self.min_sim = min_sim
            self.is_empty = False

        def query(self, crop):
            return [(b, s) for (b, s) in self.candidates if s >= self.min_sim]

    def test_retrieval_resolves_icon_only_when_ocr_and_class_empty(self):
        # Icon-only logo: OCR empty, class label generic -> CLIP retrieval
        # supplies the brand (the at-risk case CLIP is meant to fix).
        class EmptyOCR:
            def extract_text(self, crop):
                return []
        ri = self.FakeRetrieval([("SAMSUNG", 0.78), ("SONY", 0.60)])
        resolver = BrandResolver(ocr_extractor=EmptyOCR(), class_confidence=0.40,
                                 retrieval_index=ri, retrieval_min_similarity=0.22)
        dets = [[{"class_name": "brand logo", "bbox": [10, 10, 50, 50], "confidence": 0.9}]]
        out = resolver.resolve(dets, [self._frame()])
        assert out[0][0]["brand"] == "SAMSUNG"
        assert out[0][0]["resolution_source"] == "clip_retrieval"
        assert out[0][0]["retrieval_similarity"] == 0.78
        assert out[0][0]["retrieval_top3"][0] == ("SAMSUNG", 0.78)

    def test_retrieval_corrects_spurious_high_conf_class(self):
        # Class confidently says SUPREME (>= gate) on an icon-only logo, OCR
        # silent; retrieval wins over the spurious class label (the fix).
        class EmptyOCR:
            def extract_text(self, crop):
                return []
        ri = self.FakeRetrieval([("SAMSUNG", 0.81)])
        resolver = BrandResolver(ocr_extractor=EmptyOCR(), class_confidence=0.40,
                                 retrieval_index=ri, retrieval_min_similarity=0.22)
        dets = [[{"class_name": "SUPREME logo", "bbox": [10, 10, 50, 50], "confidence": 0.6}]]
        out = resolver.resolve(dets, [self._frame()])
        assert out[0][0]["brand"] == "SAMSUNG"
        assert out[0][0]["resolution_source"] == "clip_retrieval"
        assert out[0][0]["resolved_vs_class"]["class_brand"] == "SUPREME"

    def test_ocr_beats_retrieval(self):
        # OCR reads a real wordmark; it must win even if retrieval prefers a
        # different brand (ground-truth-adjacent beats similarity).
        class FakeOCR:
            def extract_text(self, crop):
                return [{"text": "samsung galaxy"}]
        ri = self.FakeRetrieval([("SONY", 0.85)])
        resolver = BrandResolver(ocr_extractor=FakeOCR(), class_confidence=0.40,
                                 retrieval_index=ri, retrieval_min_similarity=0.22)
        dets = [[{"class_name": "SUPREME logo", "bbox": [10, 10, 50, 50], "confidence": 0.6}]]
        out = resolver.resolve(dets, [self._frame()])
        assert out[0][0]["brand"] == "SAMSUNG"
        assert out[0][0]["resolution_source"] == "ocr"

    def test_retrieval_below_threshold_rejected(self):
        # Retrieval candidate below min_similarity is rejected; no brand.
        class EmptyOCR:
            def extract_text(self, crop):
                return []
        ri = self.FakeRetrieval([("SAMSUNG", 0.10)], min_sim=0.22)
        resolver = BrandResolver(ocr_extractor=EmptyOCR(), class_confidence=0.40,
                                 retrieval_index=ri, retrieval_min_similarity=0.22)
        dets = [[{"class_name": "brand logo", "bbox": [10, 10, 50, 50], "confidence": 0.9}]]
        out = resolver.resolve(dets, [self._frame()])
        assert out[0][0]["brand"] is None


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


# ============================================================
# Layer 1 — CLIP logo-retrieval index (Phase 1/2)
# ============================================================

class TestLogoRetrievalValidation:
    """The brand-name guard in LogoRetrievalIndex must accept every real catalog
    brand — including hyphenated (COCA-COLA) and apostrophe (LEVI'S) marks that a
    naive [A-Z0-9 ] regex silently drops (that was a live bug: 80 reference crops
    for those two brands vanished from the index)."""

    def _blank(self, h=16, w=16):
        return np.zeros((h, w, 3), dtype=np.uint8)

    def test_canonical_uppercases_and_strips(self):
        assert _canonical("  Under Armour ") == "UNDER ARMOUR"

    def test_hyphenated_brand_accepted(self):
        assert LogoRetrievalIndex().add_brand("COCA-COLA", [self._blank()]) == 1

    def test_apostrophe_brand_accepted(self):
        assert LogoRetrievalIndex().add_brand("LEVI'S", [self._blank()]) == 1

    def test_accents_accepted(self):
        assert LogoRetrievalIndex().add_brand("NESCAFÉ", [self._blank()]) == 1

    def test_empty_brand_rejected(self):
        assert LogoRetrievalIndex().add_brand("", [self._blank()]) == 0
