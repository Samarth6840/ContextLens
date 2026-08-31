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
    UNKNOWN_BRAND,
    brand_evidence_from_timeline,
    build_brand_timeline,
    group_unknown_logo_regions,
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

    def test_match_brand_z_fold_family(self):
        # The foldable phone video names the device "Z Fold 8 Ultra" (spoken in
        # English within a Hindi narration, transcribed verbatim by ASR). "Z Fold"
        # / "Galaxy Z Fold" are Samsung's foldable family -> SAMSUNG.
        assert match_brand("Z Fold 8 Ultra") == "SAMSUNG"
        assert match_brand("Galaxy Z Fold8 Ultra Folas G") == "SAMSUNG"
        # A generic "foldable" (no Z Fold / Galaxy brand) must NOT match.
        assert match_brand("this is a generic foldable smartphone") is None

    def test_find_mentions_z_fold(self):
        from src.brand_catalog import find_brand_mentions
        # Real ASR snippet from the foldable video: no explicit "samsung", the
        # device is named only as "Z Fold 8 Ultra" -> must resolve to SAMSUNG.
        mentions = find_brand_mentions(
            "ये सबसे पतला फोल्डिंग स्मार्टफोन, Z Fold 8 Ultra."
        )
        assert "SAMSUNG" in [m["brand"] for m in mentions]

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

    def test_resolve_from_class_name_unconfirmed_without_corroboration(self):
        # A brand asserted from YOLO-World's zero-shot class label ALONE is the
        # fabrication path (small spurious 'GUCCI logo'/'REEBOK logo'/'SUPREME
        # logo' boxes, no readable text, no retrieval match -> a fabricated
        # DIRECT recommendation). With corroboration required and neither OCR
        # nor CLIP naming a brand, the class label is NOT asserted; the detection
        # is tagged class_unconfirmed and left unresolved (fail-closed).
        resolver = BrandResolver()
        dets = [[{"class_name": "Nike logo", "bbox": [0, 0, 10, 10], "confidence": 0.9}]]
        out = resolver.resolve(dets, [self._frame()])
        assert out[0][0]["brand"] is None
        assert out[0][0]["class_unconfirmed"] is True

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

    def test_full_frame_editorial_box_suppressed_even_when_ocr_reads_brand(self):
        # A detection box covering most of the frame (title card / full-screen
        # editorial / screen recording) is NOT a compact brand wordmark. OCR over
        # such a box reads the surrounding scene text, and resolving it to a brand
        # produces a false DIRECT recommendation (observed: a ~0.58-area box whose
        # headline text resolved to GOOGLE). It must be suppressed fail-closed no
        # matter which merged text OCR returns.
        class FakeOCR:
            def extract_text(self, crop):
                return [{"text": "The $176 Billion Google Accounting Trick"}]
        resolver = BrandResolver(ocr_extractor=FakeOCR())
        dets = [[{"class_name": "brand logo", "bbox": [0, 0, 160, 100],
                  "confidence": 0.9}]]  # covers 100% of the 160x100 frame
        out = resolver.resolve(dets, [self._frame()])
        assert out[0][0]["brand"] is None
        assert out[0][0]["editorial_box"] is True

    def test_compact_logo_unaffected_by_editorial_box_gate(self):
        # A genuinely compact wordmark (small fraction of frame area) is far below
        # the full-frame bar and must still resolve normally via OCR.
        class FakeOCR:
            def extract_text(self, crop):
                return [{"text": "adidas"}]
        resolver = BrandResolver(ocr_extractor=FakeOCR())
        dets = [[{"class_name": "text logo", "bbox": [10, 10, 40, 30],
                  "confidence": 0.9}]]  # ~4% of frame
        out = resolver.resolve(dets, [self._frame()])
        assert out[0][0]["brand"] == "ADIDAS"
        assert out[0][0].get("editorial_box", False) is False

    def test_editorial_box_gate_can_be_disabled(self):
        # Operators who accept the risk can restore the legacy behavior (resolve
        # even full-frame boxes) by raising the threshold above 1.0.
        class FakeOCR:
            def extract_text(self, crop):
                return [{"text": "adidas"}]
        resolver = BrandResolver(ocr_extractor=FakeOCR(), max_logo_area_fraction=1.5)
        dets = [[{"class_name": "brand logo", "bbox": [0, 0, 160, 100],
                  "confidence": 0.9}]]
        out = resolver.resolve(dets, [self._frame()])
        assert out[0][0]["brand"] == "ADIDAS"
        assert out[0][0].get("editorial_box", False) is False

    def test_resolver_counts_logged(self):
        resolver = BrandResolver()
        dets = [
            [{"class_name": "Adidas logo", "bbox": [0, 0, 5, 5], "confidence": 0.8}],
            [{"class_name": "text logo", "bbox": [0, 0, 5, 5], "confidence": 0.8}],
        ]
        frames = [self._frame(), self._frame()]
        out = resolver.resolve(dets, frames)
        assert out[0][0]["brand"] is None
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

    def test_class_brand_unconfirmed_when_ocr_reads_nothing(self):
        # When OCR finds no wordmark AND no retrieval corroborates the class
        # label, the above-gate class label is NOT trusted alone (fail-closed):
        # brand stays None, tagged class_unconfirmed so it can never become a
        # DIRECT recommendation. This is the fabrication leak the corroboration
        # gate closes.
        class EmptyOCR:
            def extract_text(self, crop):
                return []
        resolver = BrandResolver(ocr_extractor=EmptyOCR(), class_confidence=0.40)
        dets = [[{"class_name": "Supreme logo", "bbox": [10, 10, 50, 50], "confidence": 0.6}]]
        out = resolver.resolve(dets, [self._frame()])
        assert out[0][0]["brand"] is None
        assert out[0][0]["class_unconfirmed"] is True

    def test_class_brand_corroborated_by_retrieval_asserted_via_retrieval(self):
        # When CLIP retrieval independently names the same brand as the class
        # label, the brand IS asserted — but via the retrieval path (the trusted
        # signal), resolving the icon-only case CLIP is meant to fix.
        class EmptyOCR:
            def extract_text(self, crop):
                return []
        ri = self.FakeRetrieval([("SAMSUNG", 0.78), ("SONY", 0.30)])
        resolver = BrandResolver(ocr_extractor=EmptyOCR(), class_confidence=0.40,
                                 retrieval_index=ri, retrieval_min_similarity=0.22,
                                 retrieval_min_margin=0.10)
        dets = [[{"class_name": "Samsung logo", "bbox": [10, 10, 50, 50], "confidence": 0.6}]]
        out = resolver.resolve(dets, [self._frame()])
        assert out[0][0]["brand"] == "SAMSUNG"
        assert out[0][0]["resolution_source"] == "clip_retrieval"

    def test_class_corroboration_can_be_disabled(self):
        # Operators who accept the fabrication risk can restore the legacy
        # class-only assertion explicitly.
        class EmptyOCR:
            def extract_text(self, crop):
                return []
        resolver = BrandResolver(ocr_extractor=EmptyOCR(), class_confidence=0.40,
                                 class_require_corroboration=False)
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

    # ── Margin guard on CLIP retrieval (Phase 1 precision fix) ────────────
    # Data from a real 65s foldable-video run: every one of 124 noise hits had a
    # top1-top2 margin <= 0.052 (86/124 <= 0.02) — a stack of near-tied random
    # brands (ADIDAS 0.9 / NEW BALANCE 0.89 / ASICS 0.88) that passes ANY
    # absolute similarity floor. A genuine match is unambiguous (wide margin to
    # every other brand). So resolution requires a margin, not just a floor.

    def test_retrieval_tight_margin_rejected_as_noise(self):
        # Two brands near-tied (e.g. 0.90 vs 0.89) = noise dressed as confidence.
        class EmptyOCR:
            def extract_text(self, crop):
                return []
        ri = self.FakeRetrieval([("ADIDAS", 0.90), ("NEW BALANCE", 0.89)])
        resolver = BrandResolver(ocr_extractor=EmptyOCR(), class_confidence=0.40,
                                 retrieval_index=ri, retrieval_min_similarity=0.22,
                                 retrieval_min_margin=0.10)
        dets = [[{"class_name": "brand logo", "bbox": [10, 10, 50, 50], "confidence": 0.9}]]
        out = resolver.resolve(dets, [self._frame()])
        assert out[0][0]["brand"] is None

    def test_retrieval_wide_margin_accepted(self):
        # Top-1 far above every other brand (e.g. SAMSUNG 0.9 / others 0.3).
        class EmptyOCR:
            def extract_text(self, crop):
                return []
        ri = self.FakeRetrieval([("SAMSUNG", 0.90), ("SONY", 0.30), ("ADIDAS", 0.28)])
        resolver = BrandResolver(ocr_extractor=EmptyOCR(), class_confidence=0.40,
                                 retrieval_index=ri, retrieval_min_similarity=0.22,
                                 retrieval_min_margin=0.10)
        dets = [[{"class_name": "brand logo", "bbox": [10, 10, 50, 50], "confidence": 0.9}]]
        out = resolver.resolve(dets, [self._frame()])
        assert out[0][0]["brand"] == "SAMSUNG"
        assert out[0][0]["resolution_source"] == "clip_retrieval"

    def test_retrieval_single_candidate_accepted(self):
        # Only one brand clears the absolute floor -> every other brand in the
        # bank fell weak, which is a strong genuine-match signal (real noise
        # stacks always drag 2+ brands over the floor).
        class EmptyOCR:
            def extract_text(self, crop):
                return []
        ri = self.FakeRetrieval([("SAMSUNG", 0.81)])
        resolver = BrandResolver(ocr_extractor=EmptyOCR(), class_confidence=0.40,
                                 retrieval_index=ri, retrieval_min_similarity=0.22,
                                 retrieval_min_margin=0.10)
        dets = [[{"class_name": "brand logo", "bbox": [10, 10, 50, 50], "confidence": 0.9}]]
        out = resolver.resolve(dets, [self._frame()])
        assert out[0][0]["brand"] == "SAMSUNG"
        assert out[0][0]["resolution_source"] == "clip_retrieval"

    # ── OCR-context screen-content gate (Phase 1) ──────────────────────────
    # A phone video shows the DEVICE with its own on-screen UI (app drawer,
    # Messages/2FA screen, camera watermark). YOLO flags those screen regions as
    # logos and OCR reads the app names off them, producing a "brand" (META/
    # GOOGLE/SUPREME) that is on-screen content, not a brand appearance. Same
    # trust-class as a spurious CLIP result: suppress fail-closed rather than
    # report it as a detected brand.

    class WordOCR:
        def __init__(self, text):
            self.text = text

        def extract_text(self, crop):
            return [{"text": self.text}]

    def test_screen_ui_ocr_brand_suppressed(self):
        # App-drawer read ("Store / Play Store / Gaming Hub / Instagram") names a
        # brand but is phone-screen content -> suppressed, never a brand.
        resolver = BrandResolver(ocr_extractor=self.WordOCR(
            "Store Play Store Gaming Hub Instagram Voice"),
            class_confidence=0.40)
        dets = [[{"class_name": "META", "bbox": [10, 10, 50, 50], "confidence": 0.3}]]
        out = resolver.resolve(dets, [self._frame()])
        assert out[0][0]["brand"] is None
        assert out[0][0]["screen_content"] is True
        assert out[0][0]["screen_content_reason"] == "ocr_text_reads_phone_ui"

    def test_legit_wordmark_not_suppressed(self):
        # A clean title-card wordmark has no UI tokens -> kept as a brand.
        resolver = BrandResolver(ocr_extractor=self.WordOCR(
            "The all-new Galaxy Z Fold8 Ultra"),
            class_confidence=0.40)
        dets = [[{"class_name": "text logo", "bbox": [10, 10, 50, 50], "confidence": 0.3}]]
        out = resolver.resolve(dets, [self._frame()])
        assert out[0][0]["brand"] == "SAMSUNG"
        assert out[0][0]["resolution_source"] == "ocr"
        assert out[0][0].get("screen_content") is not True

    def test_screen_content_filter_can_be_disabled(self):
        # Gate off -> UI-text OCR read is allowed through as a brand (for tuning/
        # fallback), not suppressed.
        resolver = BrandResolver(ocr_extractor=self.WordOCR(
            "Store Play Store Gaming Hub Instagram Voice"),
            class_confidence=0.40, screen_content_filter=False)
        dets = [[{"class_name": "META", "bbox": [10, 10, 50, 50], "confidence": 0.4}]]
        out = resolver.resolve(dets, [self._frame()])
        assert out[0][0]["brand"] == "META"


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

    def test_unknown_brand_sentinel_never_becomes_evidence(self):
        # The reserved UNKNOWN_BRAND grouping must never leak into evidence
        # (which drives DIRECT/SUGGESTED recommendations).
        tl = {
            "NIKE": {
                "appearances": [{"modality": "logo", "confidence": 0.9}],
                "modalities": ["logo"],
            },
            UNKNOWN_BRAND: {
                "appearances": [{"modality": "logo", "confidence": 0.9}],
                "modalities": ["logo"],
            },
        }
        ev = brand_evidence_from_timeline(tl)
        assert ev.get("NIKE") == 0.9
        assert UNKNOWN_BRAND not in ev

    def test_unknown_grouping_merges_persistent_unresolved_region(self):
        # The same physical unresolved mark held across frames collapses into ONE
        # unknown region (spatial overlap + small frame gap), not N boxes.
        logos = [
            [{"brand": None, "bbox": [10, 10, 40, 40], "confidence": 0.9}],
            [{"brand": None, "bbox": [12, 10, 42, 40], "confidence": 0.9}],
        ]
        regions = group_unknown_logo_regions(logos)
        assert len(regions) == 1
        r = regions[0]
        assert r["brand"] == UNKNOWN_BRAND
        assert r["appearance_count"] == 2
        assert r["frames"] == [0, 1]

    def test_unknown_grouping_separates_distinct_regions(self):
        # Two separate unresolved marks resolve to two distinct unknown regions.
        logos = [
            [
                {"brand": None, "bbox": [10, 10, 40, 40], "confidence": 0.9},
                {"brand": None, "bbox": [60, 60, 90, 90], "confidence": 0.9},
            ]
        ]
        regions = group_unknown_logo_regions(logos)
        assert len(regions) == 2
        assert all(r["brand"] == UNKNOWN_BRAND for r in regions)

    def test_unknown_grouping_fills_a_skipped_frame_within_gap(self):
        # A one-frame suppression gap is bridged, so the mark stays one region.
        logos = [
            [{"brand": None, "bbox": [10, 10, 40, 40], "confidence": 0.9}],
            [],
            [{"brand": None, "bbox": [12, 10, 42, 40], "confidence": 0.9}],
        ]
        regions = group_unknown_logo_regions(logos)
        assert len(regions) == 1
        assert regions[0]["appearance_count"] == 2

    def test_unknown_grouping_ignores_resolved_brands(self):
        # A detection that resolved to a real brand is not an unknown region.
        logos = [
            [
                {"brand": "SAMSUNG", "bbox": [10, 10, 40, 40], "confidence": 0.9},
                {"brand": None, "bbox": [60, 60, 90, 90], "confidence": 0.9},
            ]
        ]
        regions = group_unknown_logo_regions(logos)
        assert len(regions) == 1
        assert all(r["brand"] == UNKNOWN_BRAND for r in regions)


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

    def test_query_aggregates_per_brand_for_margin(self, monkeypatch):
        # Many brands have several reference crops; raw rows would emit duplicate
        # brand candidates and corrupt the top1-vs-top2 margin. query() must
        # return each brand at most once, at its BEST crop similarity.
        idx = LogoRetrievalIndex()
        # Reference rows: 3 crops (SAMSUNG x2, SONY x1). The query embedding is
        # chosen so SAMSUNG-best > SONY-best, and second SAMSUNG crop stays below.
        ref = np.array([
            [1.0, 0.0],   # SAMSUNG crop 0
            [0.9, 0.0],   # SAMSUNG crop 1 (dup of same brand)
            [0.0, 1.0],   # SONY crop 0
        ], dtype=np.float32)

        def fake_embed(crops):
            # At query time the only call is the single query crop.
            if len(crops) == 1:
                return np.array([[1.0, 0.4]], dtype=np.float32)
            return ref

        monkeypatch.setattr(idx, "_embed_image_batch", fake_embed)
        idx._embeddings = ref
        idx._brand_of_row = ["SAMSUNG", "SAMSUNG", "SONY"]
        idx._brand_rows = {"SAMSUNG": [0, 1], "SONY": [2]}

        res = idx.query(self._blank())
        brands = [b for b, _ in res]
        # SAMSUNG appears ONCE (best crop), ranked above SONY; no dup brand.
        assert brands == ["SAMSUNG", "SONY"]
        # Best-of aggregation keeps the top SAMSUNG sim (1.0), not the weaker dup.
        assert abs(res[0][1] - 1.0) < 1e-3
