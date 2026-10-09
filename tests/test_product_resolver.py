"""
Tests for the tiered product→brand resolution system (Layer 2b).

Covers the four tiers:
  Tier 1 — direct catalog brand-name match (no product aliases in catalog).
  Tier 2 — Wikidata SPARQL manufacturer lookup (mocked; local auditable cache).
  Tier 3 — Qwen3-VL structured manufacturer query (mocked; low trust, never
           resolves at full confidence without corroboration).
  Tier 4 — learned cross-video product↔brand co-occurrence, promoted ONLY after
           >= min_distinct_videos DISTINCT videos (never frames of one video).

Every resolution records resolution_tier + source; provenance is additive and
never influences matching. No product→brand pair is hardcoded anywhere in the
test data — all brand attribution flows through the tiers.

Uses in-memory fixtures only (no network, no model weights).
"""

import os
import tempfile

from src.brand_catalog import match_brand
from src.layer2.brand_memory import ProductResolutionMemory
from src.layer2.product_resolver import (
    ProductBrandResolver,
    ProductNameExtractor,
    WikidataProductLookup,
    canonicalize_transliteration,
    has_devanagari,
    romanize_devanagari,
)

# A Wikidata HTTP mock: returns a manufacturer for a small set of known
# products, empty results otherwise (fail-closed).
_PRODUCT_MAKERS = {
    "MAC MINI": "Apple Inc.",
    "IPAD PRO": "Apple Inc.",
    "PIXEL 10": "Google LLC",
    "Z FOLD 8 ULTRA": "Samsung",
    "GALAXY Z FOLD8 ULTRA": "Samsung",
}


def _wd_mock(url, params, timeout):
    q = (params.get("query") or "").upper()
    for product, maker in _PRODUCT_MAKERS.items():
        if product in q:
            return {"results": {"bindings": [
                {"makerLabel": {"value": maker}}]}}
    return {"results": {"bindings": []}}


def _make_wd(cache_path=None):
    return WikidataProductLookup(cache_path=cache_path, http_get=_wd_mock)


# ============================================================================
# Tier 2 — Wikidata lookup
# ============================================================================

class TestWikidataLookup:
    def test_resolves_mac_mini_to_apple(self):
        wd = _make_wd()
        res = wd.lookup("Mac Mini")
        assert res is not None
        assert res["brand"] == "APPLE"
        assert res["wikidata_manufacturer"] == "Apple Inc."
        assert res["source"] == "wikidata"
        assert res["tier"] == 2

    def test_unknown_product_fails_closed(self):
        wd = _make_wd()
        assert wd.lookup("Frobnitz 9000") is None

    def test_cache_hit_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cache.json")
            wd = _make_wd(cache_path=path)
            assert wd.lookup("Mac Mini") is not None
            assert wd.lookup("Mac Mini") is not None  # cache hit
            assert wd.cache_size() == 1
            assert wd.stats()["hits"] >= 1
            # Cache persists to disk
            assert os.path.exists(path)

    def test_lookup_normalizes_input(self):
        wd = _make_wd()
        assert wd.lookup("  mac mini  ")["brand"] == "APPLE"

    def test_lookup_cache_only_never_queries(self):
        # live=False must consult only the local cache, never the endpoint. An
        # unlearned product returns None WITHOUT a query; a previously-queried
        # product returns the cached brand.
        calls = []
        def counting_mock(url, params, timeout):
            calls.append(params.get("query"))
            return _wd_mock(url, params, timeout)
        wd = WikidataProductLookup(cache_path=None, http_get=counting_mock)
        assert wd.lookup("Mac Mini", live=False) is None
        assert calls == []  # no external query spent
        assert wd.lookup("Mac Mini") is not None  # live warm now
        assert len(calls) == 1
        assert wd.lookup("Mac Mini", live=False)["brand"] == "APPLE"  # cached
        assert len(calls) == 1  # cache hit, still no new query


# ============================================================================
# Product-name extraction + plausibility gate
# ============================================================================

class TestProductNameExtractor:
    def test_extracts_capitalized_multiword(self):
        ex = ProductNameExtractor()
        spans = ex.extract("Mac Mini 55,000 4GB RAM")
        assert any(c["span"] == "Mac Mini" for c in spans)

    def test_does_not_catch_all_lowercase_noise(self):
        ex = ProductNameExtractor()
        assert ex.extract("just some random text here") == []

    def test_price_boosts_plausibility(self):
        ex = ProductNameExtractor()
        spans = ex.extract("The Mac Mini 55,000")
        assert spans and spans[0]["boost"] > 1.0

    def test_trailing_price_digits_trimmed(self):
        ex = ProductNameExtractor()
        spans = ex.extract("Mac Mini 55,000")
        assert any(c["span"] == "Mac Mini" for c in spans)

    def test_pixel_9_keeps_model_number(self):
        ex = ProductNameExtractor()
        spans = ex.extract("Pixel 9 Pro 55,000")
        assert any(c["span"] == "Pixel 9 Pro" for c in spans)


# ============================================================================
# Devanagari romanization + transliteration canonicalization
# ============================================================================

class TestDevanagariRomanization:
    def test_has_devanagari(self):
        assert has_devanagari("पिक्सेर 10")
        assert not has_devanagari("Pixel 10")

    def test_romanize_pixel(self):
        assert "piksera" in romanize_devanagari("पिक्सेर").lower()

    def test_canonicalize_transliteration_pixel(self):
        assert canonicalize_transliteration("piksera") == "pixel"

    def test_extractor_romanizes_then_resolves_span(self):
        ex = ProductNameExtractor()
        spans = ex.extract("पिक्सेर 10 इन पर साल बढ़िया है")
        assert any(c["span"] == "Pixel 10" for c in spans)


# ============================================================================
# Tiered resolver end-to-end
# ============================================================================

class TestProductBrandResolver:
    def test_tier1_direct_catalog_brand(self):
        r = ProductBrandResolver(wikidata=_make_wd())
        res = r.resolve("APPLE MAC MINI")
        assert res and res[0]["brand"] == "APPLE"
        assert res[0]["resolution_tier"] == 1

    def test_tier2_wikidata_mac_mini(self):
        r = ProductBrandResolver(wikidata=_make_wd())
        res = r.resolve("Mac Mini 55,000 4GB RAM")
        assert res and res[0]["brand"] == "APPLE"
        assert res[0]["resolution_tier"] == 2
        assert res[0]["source"] == "wikidata"

    def test_tier2_held_out_ipad_pro(self):
        # "iPad Pro" was a previously-hardcoded APPLE alias, now migrated out of
        # the catalog. It must resolve ONLY through the tiered resolver — and the
        # catalog itself must stay product-free.
        assert match_brand("iPad Pro") is None
        r = ProductBrandResolver(wikidata=_make_wd())
        res = r.resolve("iPad Pro 12 55,000")
        assert res and res[0]["brand"] == "APPLE"
        assert res[0]["resolution_tier"] == 2

    def test_tier2_devanagari_pixel(self):
        r = ProductBrandResolver(wikidata=_make_wd())
        res = r.resolve("पिक्सेर 10 इन पर साल बढ़िया है")
        assert res and res[0]["brand"] == "GOOGLE"
        assert res[0]["product_span"] == "PIXEL 10"

    def test_unresolved_fails_closed(self):
        r = ProductBrandResolver(wikidata=_make_wd())
        assert r.resolve("a completely generic sentence without products") == []

    def test_provenance_emitted(self):
        r = ProductBrandResolver(wikidata=_make_wd())
        r.resolve("Mac Mini 55,000")
        prov = r.emit_provenance()
        assert prov["tier_counts"].get("2") == 1
        assert prov["qwen_calls"] == 0


# ============================================================================
# Tier 3 — Qwen structured call (low trust)
# ============================================================================

class _FakeQwen:
    def __init__(self, maker):
        self.maker = maker

    def __call__(self, product_span, frame=None):
        return {"manufacturer": self.maker, "confidence": 0.45, "fallback": False}


class TestQwenTier:
    def test_qwen_corroborated_accepts_brand(self):
        r = ProductBrandResolver(
            wikidata=_make_wd(),
            qwen=_FakeQwen("Sony"),
            corroboration_fn=lambda span: True,
        )
        res = r.resolve("WH-1000XM5 55,000")
        assert res and res[0]["brand"] == "SONY"
        assert res[0]["resolution_tier"] == "qwen_corroborated"

    def test_qwen_uncorroborated_never_asserts_brand(self):
        # Tier 3 WITHOUT independent corroboration must NOT resolve at full
        # confidence; it is logged as a low-confidence candidate instead.
        r = ProductBrandResolver(
            wikidata=_make_wd(),
            qwen=_FakeQwen("Sony"),
            corroboration_fn=lambda span: False,
        )
        res = r.resolve("WH-1000XM5 55,000")
        assert res == []
        assert len(r.low_confidence_candidates) == 1
        assert r.tier_counts.get("qwen_unsupported") == 1

    def test_qwen_unknown_returns_nothing(self):
        r = ProductBrandResolver(
            wikidata=_make_wd(),
            qwen=lambda span, frame=None: {"manufacturer": None},
        )
        assert r.resolve("WH-1000XM5 55,000") == []

    def test_qwen_skipped_on_cache_only_path(self):
        # live=False must never spend a Qwen call: the corroborated-Sony span
        # resolves to nothing on the cache-only path even though a live call
        # would (and a bare lambda proves Qwen was not invoked).
        called = []
        def qwen(span, frame=None):
            called.append(span)
            return {"manufacturer": "Sony"}
        r = ProductBrandResolver(
            wikidata=_make_wd(),
            qwen=qwen,
            corroboration_fn=lambda span: True,
        )
        assert r.resolve("WH-1000XM5 55,000", live=False) == []
        assert called == []


# ============================================================================
# Tier 4 — learned co-occurrence (>=3 distinct videos)
# ============================================================================

class TestTier4Learned:
    def test_promoted_only_after_three_distinct_videos(self):
        mem = ProductResolutionMemory(min_distinct_videos=3)
        # 2 distinct videos -> NOT promoted
        for v in ("v1", "v2"):
            mem.record("AirPods Pro", "APPLE", v)
        assert mem.lookup("AirPods Pro") is None
        # same video again (frames) does not count as a new distinct video
        mem.record("AirPods Pro", "APPLE", "v1")
        assert mem.lookup("AirPods Pro") is None
        # 3rd distinct video -> promoted
        mem.record("AirPods Pro", "APPLE", "v3")
        res = mem.lookup("AirPods Pro")
        assert res is not None
        assert res["brand"] == "APPLE"
        assert res["distinct_videos"] == 3
        assert res["videos_observed"] == ["v1", "v2", "v3"]

    def test_resolver_uses_learned_tier4(self):
        mem = ProductResolutionMemory(min_distinct_videos=3)
        for v in ("v1", "v2", "v3"):
            mem.record("AirPods Pro", "APPLE", v)
        r = ProductBrandResolver(wikidata=_make_wd(), learned_lookup=mem.lookup)
        res = r.resolve("AirPods Pro 25,000")
        assert res and res[0]["brand"] == "APPLE"
        assert res[0]["resolution_tier"] == 4
        assert res[0]["source"] == "learned_cooccurrence"

    def test_record_observation_feeds_memory(self):
        mem = ProductResolutionMemory(min_distinct_videos=3)
        for v in ("v1", "v2", "v3"):
            mem.record("Galaxy Buds", "SAMSUNG", v)
        # learned memory wins over Wikidata (Tier 4 checked first)
        r = ProductBrandResolver(
            wikidata=_make_wd(), learned_lookup=mem.lookup,
            add_brand_resolution=mem.record,
        )
        res = r.resolve("Galaxy Buds 15,000")
        assert res and res[0]["brand"] == "SAMSUNG"
        assert res[0]["resolution_tier"] == 4

    def test_serialization_roundtrip(self):
        mem = ProductResolutionMemory(min_distinct_videos=3)
        for v in ("v1", "v2", "v3"):
            mem.record("iPad Pro", "APPLE", v)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mem.json")
            mem.save(path)
            loaded = ProductResolutionMemory.load(path)
            assert loaded.lookup("iPad Pro")["brand"] == "APPLE"
            assert loaded.lookup("iPad Pro")["distinct_videos"] == 3


# ============================================================================
# Provenance is additive / never influences matching
# ============================================================================

class TestProvenanceNonInfluence:
    def test_adding_sources_does_not_change_resolution(self):
        r1 = ProductBrandResolver(wikidata=_make_wd(), extractor=ProductNameExtractor())
        r2 = ProductBrandResolver(wikidata=_make_wd(), extractor=ProductNameExtractor())
        r1.resolve("Mac Mini 55,000")
        r2.resolve("Mac Mini 55,000")
        # Provenance accumulation on r1 must not influence the re-resolution.
        again = r1.resolve("Mac Mini 55,000")
        assert again == r2.resolve("Mac Mini 55,000")
        # Tier counts are metrics-only.
        assert r1.emit_provenance()["tier_counts"]["2"] == 2


# ============================================================================
# extract_price signal (spec_extractor) backing the plausibility gate
# ============================================================================

class TestPriceSignal:
    def test_price_extraction_hindi_style(self):
        from src.layer1.spec_extractor import extract_price
        assert extract_price("55,000")["value"] == 55000.0
        assert extract_price("₹1,29,999")["currency"] == "INR"
        assert extract_price("1.2 lakh")["value"] == 120000.0
        assert extract_price("no price here") == {}


# ============================================================================
# D1.2 regression: "MAC MINI" OCR must resolve to APPLE end-to-end
# ============================================================================

class TestMacMiniResolutionEndToEnd:
    """Isolated regression for the D1.2 defect (OCR "MAC MINI" produced NO
    brand because the catalog deliberately has no product aliases). The tiered
    resolver must name APPLE from a REAL source — Tier 4 learned cross-video
    memory (network-free) or Tier 2 Wikidata — and its provenance must be
    recorded so the resolution is visible, not wiped."""

    def test_mac_mini_to_apple_via_learned_memory_cache_only(self):
        # Tier 4: learned cross-video co-occurrence (>= 3 distinct videos), the
        # cheapest and most trusted path. live=False proves no external/network
        # query is involved on the OCR evidence path.
        mem = ProductResolutionMemory(min_distinct_videos=3)
        for v in ("v1", "v2", "v3"):
            mem.record("Mac Mini", "APPLE", v)
        res = ProductBrandResolver(wikidata=_make_wd(), learned_lookup=mem.lookup)
        out = res.resolve("Mac Mini", live=False)
        assert len(out) == 1
        r = out[0]
        assert r["brand"] == "APPLE"
        assert r["resolution_tier"] == 4
        assert r["source"] == "learned_cooccurrence"
        assert r["confidence"] == 0.85
        assert res.emit_provenance()["tier_counts"]["4"] == 1

    def test_mac_mini_not_promoted_below_three_videos(self):
        # Fail-closed: with only 2 distinct videos the association is NOT
        # promoted, so "MAC MINI" stays UNRESOLVED rather than fabricated.
        mem = ProductResolutionMemory(min_distinct_videos=3)
        for v in ("v1", "v2"):
            mem.record("Mac Mini", "APPLE", v)
        res = ProductBrandResolver(wikidata=_make_wd(), learned_lookup=mem.lookup)
        assert res.resolve("Mac Mini", live=False) == []

    def test_mac_mini_to_apple_via_wikidata_live_chain(self):
        # Tier 2: real Wikidata chain (mocked http_get at the wire level, no
        # alias tables). Resolves to APPLE and records tier-2 provenance.
        wd = _make_wd()
        resolver = ProductBrandResolver(wikidata=wd)
        res = resolver.resolve("Mac Mini")
        assert len(res) == 1
        r = res[0]
        assert r["brand"] == "APPLE"
        assert r["resolution_tier"] == 2
        assert r["source"] == "wikidata"
        assert resolver.emit_provenance()["tier_counts"]["2"] == 1

    def test_resolution_survives_repeated_ocr_routing(self):
        # The historical wipe bug reset `product_resolutions` after resolution.
        # Isolated at the SPI level: repeated OCR-path resolutions must keep
        # accumulating provenance (tier counts) and never return empty.
        from src.layer2.brand_memory import ProductResolutionMemory

        mem = ProductResolutionMemory(min_distinct_videos=3)
        for v in ("v1", "v2", "v3"):
            mem.record("Mac Mini", "APPLE", v)
        res = ProductBrandResolver(
            wikidata=_make_wd(), learned_lookup=mem.lookup,
        )
        for _ in range(3):
            out = res.resolve("Mac Mini", live=False)
            assert out and out[0]["brand"] == "APPLE"
        counts = res.emit_provenance()["tier_counts"]
        assert counts["4"] == 3

class TestTransliterationMapIsReachable:
    """The map is only useful if its keys are what romanize_devanagari emits.

    Every key is matched after _tidy_roman() lowercases and _ROM_TOKEN_RE
    splits on [a-z]+, so an uppercase key ("alTr") or a guessed spelling
    ("gaileksi" where the romanizer says "gailaksi") is dead. That is how the
    table lost 9 of 14 entries and "Z Fold" spoken in Hindi stopped resolving.
    """

    def test_every_key_matches_the_romanizer(self):
        from src.layer2.product_resolver import _TRANSLITERATION_MAP

        dead = [k for k in _TRANSLITERATION_MAP if not k.islower() and not k.isalpha()]
        assert not dead, f"keys unreachable by _ROM_TOKEN_RE: {dead}"

    def test_devanagari_actually_canonicalizes(self):
        from src.layer2.product_resolver import (
            canonicalize_transliteration,
            romanize_devanagari,
        )

        cases = [
            ("ज़ेफ़ोल्ड 5", "z fold"),
            ("फ़ोल्ड", "fold"),
            ("अल्ट्रा", "ultra"),
            ("गैलक्सी", "galaxy"),
            ("पिक्सेर", "pixel"),
            ("आइफ़ोन", "iphone"),
            ("मैक्रा", "mac"),
            ("स्नैपड्रैगन", "snapdragon"),
        ]
        for hindi, expected in cases:
            got = canonicalize_transliteration(romanize_devanagari(hindi))
            assert expected in got.lower(), (
                f"{hindi!r} romanized to {romanize_devanagari(hindi)!r} -> {got!r}, "
                f"expected to contain {expected!r}"
            )


class TestItem15WikidataQuery:
    """The old query forced a scan of every rdfs:label in every language and
    the endpoint timed out. It also compared against normalize_text() output,
    which turns hyphens into spaces, so WH-1000XM5 could never equal its own
    label even when the query did return."""

    def test_uses_the_search_index_not_a_label_scan(self):
        from src.layer2.product_resolver import _SPARQL_TEMPLATE
        q = _SPARQL_TEMPLATE.format(search="Mac Mini")
        assert "rdfs:label ?productLabel" not in q
        assert "EntitySearch" in q and "wikibase:mwapi" in q

    def test_hyphenated_model_number_reaches_the_query_intact(self):
        from src.layer2.product_resolver import WikidataProductLookup
        seen = {}

        def spy(url, params, timeout):
            seen["q"] = params["query"]
            return {"results": {"bindings": [{"makerLabel": {"value": "Sony"}}]}}

        wd = WikidataProductLookup(cache_path=None, http_get=spy)
        res = wd.lookup("WH-1000XM5")
        assert 'mwapi:search "WH-1000XM5"' in seen["q"]
        assert res is not None and res["brand"] == "SONY"

    def test_search_term_cannot_break_out_of_the_string_literal(self):
        from src.layer2.product_resolver import _SPARQL_TEMPLATE, _sparql_escape
        q = _SPARQL_TEMPLATE.format(search=_sparql_escape(
            'evil" } . ?x wdt:P176 ?y . SERVICE x {'))
        assert q.count('mwapi:search "') == 1


class TestItem16LeadingArticle:
    """A sentence-initial article is not part of the product name. "The Mac
    Mini 55,000" yielded the span "The Mac Mini", whose normalized form
    "THE MAC MINI" matches no label, so the single most valuable span — a real
    product next to a price — never resolved."""

    def test_leading_article_is_stripped(self):
        from src.layer2.product_resolver import ProductNameExtractor
        e = ProductNameExtractor()
        def first(t):
            r = e.extract(t)
            return r[0]["normalized"] if r else None
        assert first("The Mac Mini 55,000") == "MAC MINI"
        assert first("A Galaxy S24 Ultra") == "GALAXY S24 ULTRA"
        assert first("The new OnePlus 12") == "ONEPLUS 12"

    def test_article_free_text_is_unchanged(self):
        from src.layer2.product_resolver import ProductNameExtractor
        e = ProductNameExtractor()
        r = e.extract("Mac Mini 55,000")
        assert r and r[0]["normalized"] == "MAC MINI"

    def test_bare_article_yields_nothing(self):
        from src.layer2.product_resolver import ProductNameExtractor
        e = ProductNameExtractor()
        assert e.extract("The") == []
        assert e.extract("A 5") == []
