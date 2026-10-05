"""
Layer 2b — Tiered Product→Brand Resolution.

Resolves PRODUCT names (e.g. "Mac Mini", "Galaxy S24", "AirPods Pro") to their
parent BRAND when the on-screen/spoken text does not directly name the brand.
This is the generalizing replacement for the (rejected) practice of hardcoding
product names into catalog brand aliases.

The resolution is TIERED, cheapest/most-reliable first:
  Tier 1 — direct catalog brand-name match (match_brand / find_brand_mentions).
           Unchanged; if the text already names a catalog brand, resolve as today.
  Tier 2 — external structured knowledge base (Wikidata SPARQL: P176 manufacturer
           / P1716 brand) with a local, auditable cache and rate limiting.
  Tier 3 — Qwen3-VL 32B narrow structured call (manufacturer or UNKNOWN), LOW
           trust: never accepted at full confidence without corroboration; else
           logged as a low-confidence candidate for review.
  Tier 4 — cross-video co-occurrence learned empirically (in brand_memory.py):
           product↔brand pairings promoted ONLY after being observed in >= N
           DISTINCT videos (anti-overfitting: never frames of one video).

Every resolution records `resolution_tier` (1|2|3|4) + a `source` tag consistent
with the existing provenance vocabulary (resolution_source / resolution_quality).
Provenance is purely additive and NEVER influences matching or ranking.

Anti-pattern enforced here: no product→brand pair is hardcoded. The Tier-2 cache
and Tier-4 learned table are populated only by live queries / observations.
"""

import json
import logging
import os
import re
import threading
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

from src.brand_catalog import (
    BRAND_CATALOG,
    match_brand,
    normalize_text,
)

# ============================================================================
# Devanagari → Latin romanization (Hinglish support) for the product resolver.
#
# Hindi tech-review ASR transcribes spoken product names in Devanagari (e.g.
# "पिक्सेर" → Pixel, "ज़ेड फोल्ड" → Z Fold). Catalog brand aliases may carry some
# Devanagari brand spellings, but PRODUCT names are deliberately NOT hardcoded;
# to route them through the same Wikidata/learned tiers we first romanize the
# span to Latin. Romanization is lossy but sufficient to reach a Wikidata label.
# ============================================================================

_DEV_IND_VOWELS = {
    "\u0905": "a", "\u0906": "A", "\u0907": "i", "\u0908": "I",
    "\u0909": "u", "\u090a": "U", "\u090b": "RRi", "\u090f": "e",
    "\u0910": "ai", "\u0913": "o", "\u0914": "au",
}
_DEV_MATRAS = {
    "\u093e": "a", "\u093f": "i", "\u0940": "I", "\u0941": "u",
    "\u0942": "U", "\u0947": "e", "\u0948": "ai", "\u094b": "o",
    "\u094c": "au", "\u0945": "~", "\u0946": "~e", "\u0962": "RRi",
}
_DEV_CONS = {
    "\u0915": "k", "\u0916": "kh", "\u0917": "g", "\u0918": "gh",
    "\u0919": "~N", "\u091a": "ch", "\u091b": "Ch", "\u091c": "j",
    "\u091d": "jh", "\u091e": "~n", "\u091f": "T", "\u0920": "Th",
    "\u0921": "D", "\u0922": "Dh", "\u0923": "N", "\u0924": "t",
    "\u0925": "th", "\u0926": "d", "\u0927": "dh", "\u0928": "n",
    "\u092a": "p", "\u092b": "ph", "\u092c": "b", "\u092d": "bh",
    "\u092e": "m", "\u092f": "y", "\u0930": "r", "\u0932": "l",
    "\u0935": "v", "\u0936": "sh", "\u0937": "Sh", "\u0938": "s", "\u0939": "h",
}
_DEV_NUKTA_CONS = {
    "\u0915": "q", "\u0916": "Kh", "\u0917": "G", "\u091c": "z",
    "\u0921": "R", "\u0922": "Rh", "\u092b": "f", "\u0926": "x",
    "\u0939": "Y", "\u092e": "z", "\u0935": "w", "\u0928": "n",
}
_DEV_DIGITS = {str(i): c for i, c in enumerate("0123456789")}
_DEV_SPECIAL = {
    "\u0901": ".N", "\u0902": "m", "\u0903": ":",
    "\u093c": "",   # nukta modifier (handled with preceding consonant)
    "\u094d": "",   # halant (virama) — suppress trailing 'a'
}

_DEV_RANGE = "\u0900-\u097f"
_HAS_DEV_RE = re.compile(f"[{_DEV_RANGE}]")


def has_devanagari(text: str) -> bool:
    return bool(_HAS_DEV_RE.search(text or ""))


def romanize_devanagari(text: str) -> str:
    """Transliterate Devanagari (Hindi) text to rough Latin (ITRANS-flavoured).

    Vowels get an implicit 'a' after consonants unless a matra or halant
    overrides it. Lossy but adequate for hitting Wikidata labels. Non-Devanagari
    characters (Latin, digits, spaces, punctuation) pass through unchanged.
    """
    if not text:
        return ""
    out: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        cp = ord(c)
        if not (0x0900 <= cp <= 0x097f):
            out.append(c)
            i += 1
            continue
        # Independent vowel.
        if c in _DEV_IND_VOWELS:
            out.append(_DEV_IND_VOWELS[c])
            i += 1
            continue
        # Digits.
        if c in _DEV_DIGITS:
            out.append(_DEV_DIGITS[c])
            i += 1
            continue
        # Visarga / chandrabindu / anusvara.
        if c in _DEV_SPECIAL:
            out.append(_DEV_SPECIAL[c])
            i += 1
            continue
        # Consonant (optionally nukta-modified).
        if c in _DEV_CONS:
            base = c
            j = i + 1
            if j < n and text[j] == "\u093c":  # nukta
                roman = _DEV_NUKTA_CONS.get(base, _DEV_CONS[base])
                j += 1
            else:
                roman = _DEV_CONS[base]
            out.append(roman)
            # Determine the vowel that follows (or halant) to decide trailing 'a'.
            k = j
            if k < n and text[k] in _DEV_MATRAS:
                out.append(_DEV_MATRAS[text[k]])
                k += 1
            elif k < n and text[k] == "\u094d":
                k += 1  # halant — no vowel; conjunct
            else:
                out.append("a")
            i = k
            continue
        # Unknown Devanagari codepoint — skip.
        i += 1
    res = "".join(out)
    # Collapse ITRANS diacritics into friendly lowercase Latin.
    res = _tidy_roman(res)
    return res


def _tidy_roman(res: str) -> str:
    res = (res.replace("~N", "n").replace("~n", "n").replace("Ch", "ch")
           .replace("Sh", "sh").replace("Th", "th").replace("RRi", "ri")
           .replace("~", ""))
    return res.lower()


# Devanagari → canonical Latin transliteration for common Hinglish tech nouns.
# This is a LINGUISTIC map (which Devanagari strings spell which English word) —
# it binds ONLY words to their common Latin spelling, NEVER a product to a brand.
# Brand attribution for these nouns still flows through the standard tiers
# (Wikidata / learned memory), so no product→brand pair is hardcoded here.
#
# Keys are the EXACT output of romanize_devanagari() — verified against it, not
# guessed. The previous version was hand-written and mostly unreachable: the
# tokens are lowercased by _tidy_roman and matched by _ROM_TOKEN_RE ([a-z]+), so
# every key containing an uppercase letter ("zefolDa", "alTr", "snapDragan") could
# never fire, and several lowercase ones ("gaileksi", "pholDa", "makra") were
# simply not what the romanizer emits ("gailaksi", "folda", "maikra"). Only 5 of
# 14 entries were live, which is why "Z Fold" spoken in Hindi never resolved.
_TRANSLITERATION_MAP = {
    "piksera": "pixel", "piksela": "pixel", "piksesara": "pixel",
    "pixelsera": "pixel",
    "zefolda": "z fold", "zefa": "z fold",
    "folda": "fold",
    "gailaksi": "galaxy", "gailaksa": "galaxy",
    "es": "s", "esa": "s", "esara": "s", "esera": "s",
    "pro": "pro", "pra": "pro",
    "altra": "ultra",
    "eyara": "air",
    "maikra": "mac", "maika": "mac",
    "snaipadraigana": "snapdragon",
    "aifona": "iphone", "aifna": "iphone",
    "aipaida": "ipad",
    "ikobi": "ecobee",
}
_ROM_TOKEN_RE = re.compile(r"[a-z]+|[0-9]+|[^a-z0-9]")


def canonicalize_transliteration(text: str) -> str:
    """Map romanized Hindi tokens to canonical Latin words via transliteration,
    leaving unknown tokens (and any Latin/digits) untouched."""
    if not text:
        return text
    toks = _ROM_TOKEN_RE.findall(text)
    out: List[str] = []
    for tok in toks:
        if tok.isalpha() and tok in _TRANSLITERATION_MAP:
            out.append(_TRANSLITERATION_MAP[tok])
        else:
            out.append(tok)
    return "".join(out)






# Provenance / tier constants.
TIER_DIRECT = 1
TIER_WIKIDATA = 2
TIER_QWEN = 3
TIER_LEARNED = 4

# Tier-3 sub-states: corroborated vs unsupported by other evidence.
TIER_QWEN_CORROBORATED = "qwen_corroborated"


class LookupUnavailable(RuntimeError):
    """A live lookup failed for a TRANSIENT reason (network, HTTP, timeout).

    Callers must fail closed for this video but must NOT persist a negative
    cache entry: the product may well resolve on the next run, and a cached
    "no manufacturer" is indistinguishable from a real one afterwards.
    """
TIER_QWEN_UNSUPPORTED = "qwen_unsupported"

# Quality weights per tier (mirrors resolution_quality conventions).
_TIER_QUALITY = {
    TIER_DIRECT: 0.90,          # (existing OCR path already uses 0.90)
    TIER_WIKIDATA: 0.80,
    TIER_LEARNED: 0.85,
    TIER_QWEN_CORROBORATED: 0.55,
    TIER_QWEN_UNSUPPORTED: 0.30,
}


# ============================================================================
# Tier 2 — Wikidata external lookup with a local, auditable cache.
# ============================================================================

_DEFAULT_WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"

# SPARQL: find an item whose English label matches the product name and return
# its manufacturer (P176) / brand (P1716) / owner (P127) / developer (P178).
# Labels can vary; we take any entity whose label equals the query.
_SPARQL_TEMPLATE = """
SELECT DISTINCT ?item ?productLabel ?makerLabel ?maker WHERE {{
  ?item rdfs:label ?productLabel .
  FILTER(LCASE(?productLabel) = "{query_lc}")
  OPTIONAL {{ ?item wdt:P176 ?maker . }}
  OPTIONAL {{ ?item wdt:P1716 ?maker . }}
  OPTIONAL {{ ?item wdt:P127 ?maker . }}
  OPTIONAL {{ ?item wdt:P178 ?maker . }}
  ?maker rdfs:label ?makerLabel .
  FILTER(LANG(?makerLabel) = "en")
  SERVICE wikibase:label {{
    bd:serviceParam wikibase:language "en" .
  }}
}}
LIMIT 5
""".strip()


class WikidataProductLookup:
    """Query Wikidata for a product name → manufacturer mapping.

    The result is cached locally (JSON) so repeated lookups for the same product
    never re-hit the external API. The cache is populated ONLY by real queries —
    it must never be hand-edited to "fix" a specific video.
    """

    def __init__(
        self,
        endpoint: str = _DEFAULT_WIKIDATA_ENDPOINT,
        cache_path: Optional[str] = None,
        min_interval_sec: float = 1.0,
        timeout_sec: float = 10.0,
        http_get=None,  # injectable for tests; signature: get(url, params) -> json dict
    ):
        self.endpoint = endpoint
        self.cache_path = cache_path
        self.min_interval_sec = min_interval_sec
        self.timeout_sec = timeout_sec
        self._http_get = http_get
        self._lock = threading.Lock()
        self._last_query_time = 0.0
        self._hits = 0
        self._misses = 0
        self._cache: Dict[str, dict] = {}
        self._load_cache()

    # -- cache persistence ----------------------------------------------------
    def _load_cache(self) -> None:
        if not self.cache_path or not os.path.exists(self.cache_path):
            return
        try:
            with open(self.cache_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self._cache = data.get("cache", {}) or {}
            self._hits = int(data.get("hits", 0))
            self._misses = int(data.get("misses", 0))
        except (OSError, ValueError) as e:
            logger.warning("Could not load product-resolution cache %s: %s",
                           self.cache_path, e)

    def _save_cache(self) -> None:
        if not self.cache_path:
            return
        try:
            with open(self.cache_path, "w", encoding="utf-8") as fh:
                json.dump({"cache": self._cache, "hits": self._hits,
                           "misses": self._misses}, fh, indent=2)
        except OSError as e:
            logger.warning("Could not write product-resolution cache %s: %s",
                           self.cache_path, e)

    # -- external query --------------------------------------------------------
    def _query_wikidata(self, product: str) -> Optional[str]:
        """Live SPARQL query; returns a manufacturer string or None.

        Returns None only for a GENUINE empty answer. A transport failure
        raises LookupUnavailable, because "the network broke" and "Wikidata has
        no manufacturer for this" must not be cached as the same permanent
        negative.
        """
        if self._http_get is not None:
            # Test injection — raw response dict.
            params = {"query": _SPARQL_TEMPLATE.format(query_lc=product.lower()),
                      "format": "json"}
            data = self._http_get(self.endpoint, params=params, timeout=self.timeout_sec)
        else:
            import requests
            params = {"query": _SPARQL_TEMPLATE.format(query_lc=product.lower()),
                      "format": "json"}
            try:
                r = requests.get(self.endpoint, params=params,
                                 timeout=self.timeout_sec,
                                 headers={"Accept": "application/sparql-results+json"})
                r.raise_for_status()
                data = r.json()
            except Exception as e:  # network / parse — transient, do NOT cache
                logger.warning("Wikidata query failed for %r: %s", product, e)
                raise LookupUnavailable(str(e)) from e

        # Parse W3C SPARQL JSON results → bindings.
        try:
            bindings = (data or {}).get("results", {}).get("bindings", [])
        except AttributeError:
            return None
        labels = []
        for b in bindings:
            maker = b.get("makerLabel", {}) or b.get("maker", {})
            label = (maker.get("value") or "").strip()
            if label:
                labels.append(label)
        if not labels:
            logger.info("Wikidata: no manufacturer for product %r", product)
            return None
        # Prefer the most common manufacturer among matched entities.
        counts: Dict[str, int] = {}
        for lab in labels:
            counts[lab] = counts.get(lab, 0) + 1
        best = max(counts, key=counts.get)
        return best

    def lookup(self, product: str, live: bool = True) -> Optional[dict]:
        """Resolve a normalized product name to a manufacturer mapping.

        Returns {product, brand, source, tier, timestamp, wikidata_manufacturer}
        or None. Cache-first; only queries externally when absent from cache.

        When `live` is False the lookup is cache-only: nothing is spent on the
        external endpoint and no negative entry is written, so cheap evidence
        paths (e.g. OCR aggregation) never trigger live queries for junk.
        """
        key = normalize_text(product)
        if not key:
            return None
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._hits += 1
                return dict(cached)
            if not live:
                return None  # cache-only: never spend a live query

            # Rate limit external queries.
            now = time.monotonic()
            wait = self._last_query_time + self.min_interval_sec - now
            if wait > 0:
                time.sleep(wait)
            self._last_query_time = time.monotonic()

            try:
                manufacturer = self._query_wikidata(key)
            except LookupUnavailable as exc:
                # Transient: fail closed for this call, cache nothing.
                self._misses += 1
                logger.info("Wikidata unavailable, not caching a negative for %r: %s",
                            key, exc)
                return None
            self._misses += 1
            if not manufacturer:
                # Genuine empty answer -> negative cache so we don't re-query
                # junk on every video.
                self._cache[key] = {"product": key, "brand": None,
                                    "source": "wikidata", "tier": TIER_WIKIDATA,
                                    "timestamp": now, "wikidata_manufacturer": None,
                                    "negative": True}
                self._save_cache()
                return None

            result = {
                "product": key,
                "brand": _to_catalog_brand(manufacturer),
                "source": "wikidata",
                "tier": TIER_WIKIDATA,
                "timestamp": now,
                "wikidata_manufacturer": manufacturer,
            }
            self._cache[key] = result
            self._save_cache()
            return dict(result)

    def cache_size(self) -> int:
        return len(self._cache)

    def stats(self) -> dict:
        return {"cache_size": len(self._cache), "hits": self._hits,
                "misses": self._misses}


def _to_catalog_brand(manufacturer: str) -> Optional[str]:
    """Map a Wikidata manufacturer string to a canonical BRAND_CATALOG key.

    Exact-vs-alias matching, case-insensitive. Returns None when the manufacturer
    is not a catalog brand (caller decides whether to accept a foreign brand).
    """
    if not manufacturer:
        return None
    norm = normalize_text(manufacturer)
    # Exact key match.
    for brand in BRAND_CATALOG:
        if normalize_text(brand) == norm:
            return brand
    # Alias match (covers e.g. "Google LLC" -> GOOGLE via 'google' alias).
    m = match_brand(manufacturer)
    return m


# ============================================================================
# Tier 2/3/4 product-name extraction + plausibility gate.
# ============================================================================

# Capitalized multi-word product-name heuristic: 1-5 tokens. The first token is
# title-case OR a lowercase-camelCase brand like "iPad"/"iPhone" (lowercase
# initial, uppercase hump inside). Avoids catching sentence-start words.
_PRODUCT_SPAN_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"((?:[A-Z][a-zA-Z0-9.-]*|[a-z]+[A-Z][a-zA-Z0-9.-]*)"
    r"(?:\s[A-Z0-9][a-zA-Z0-9.-]*){0,4})"
    r"(?![A-Za-z0-9])"
)


def _devangari_product_text(text: str) -> str:
    """Romanize + transliteration-canonicalize Devanagari text to Latin, then
    title-case recognizable product tokens so the product-span heuristic applies.

    Returns the transformed Latin string (all-Latin, product tokens capitalized).
    """
    rom = canonicalize_transliteration(romanize_devanagari(text))
    # Title-case tokens that matched a known transliteration (product words) so
    # the capitalized product-span heuristic can see them; leave the rest as-is.
    canon_vals = set(_TRANSLITERATION_MAP.values())
    toks = _ROM_TOKEN_RE.findall(rom)
    out: List[str] = []
    for tok in toks:
        if tok.isalpha() and tok in canon_vals:
            out.append(tok.capitalize())
        else:
            out.append(tok)
    return "".join(out)


class ProductNameExtractor:
    """Deterministic NER for "plausible product name" spans in OCR/speech text.

    A span is a capitalized multi-token phrase. Its plausibility (whether it is
    worth an expert/knowledge-base/BME query) is boosted when:
      * a spec callout is nearby (extract_specs), or
      * a price/spec token is nearby (extract_price).
    We deliberately keep this cheap and greedy; the tiered downstream gate is
    what prevents over-eager resolution.
    """

    def __init__(self, spec_signal_boost: bool = True):
        self.spec_signal_boost = spec_signal_boost
    def extract(self, text: str) -> List[dict]:
        """Return candidate product spans:
        [{span, normalized, position, boost}] sorted by position.

        DevaNagari (Hindi) input is first romanized and transliteration-
        canonicalized to Latin so the product-span heuristic and the downstream
        Wikidata/learned tiers evaluate the spoken product name (e.g. "पिक्सेर"
        → "pixel") instead of raw Devanagari.
        """
        if not text:
            return []
        had_dev = has_devanagari(text)
        if had_dev:
            text = _devangari_product_text(text)
        return self._extract_latin(text)

    def _extract_latin(self, text: str) -> List[dict]:
        from src.layer1.spec_extractor import extract_price, extract_specs
        out: List[dict] = []
        for m in _PRODUCT_SPAN_RE.finditer(text):
            span = m.group(1).strip()
            norm = normalize_text(span)
            if not norm:
                continue
            # A trailing pure-digit token that immediately continues into a
            # grouped larger number ("Mac Mini 55,000") is a price, not part of
            # the product name — trim it. ("Pixel 9" has no comma, so it's kept.)
            span = _trim_trailing_price_digits(span, text, m.start(), m.end())
            norm = normalize_text(span)
            if not norm:
                continue
            # Drop things that are clearly not product names.
            if _looks_like_noise(norm):
                continue
            out.append({
                "span": span,
                "normalized": norm,
                "position": m.start(),
                "boost": 1.0,
            })
        # Boost spans that co-occur with a price or spec callout.
        price = extract_price(text)
        specs = extract_specs(text) if self.spec_signal_boost else []
        if price or specs:
            for c in out:
                c["boost"] += 0.5
                c["price_nearby"] = bool(price)
                c["spec_nearby"] = bool(specs)
        return out


_NOISE_RE = re.compile(r"^[A-Z0-9 ]{1,3}$")  # single letters / tiny tokens


def _looks_like_noise(norm: str) -> bool:
    if _NOISE_RE.match(norm):
        return True
    # A single lowercase-after-first token that is a common non-product word.
    lowered = norm.lower()
    if lowered in {"the", "this", "that", "new", "one", "with", "for", "and",
                   "pro", "max", "plus", "mini", "air", "of", "on", "at"}:
        return True
    return False


def _trim_trailing_price_digits(span: str, text: str, start: int, end: int) -> str:
    """Trim a trailing pure-digit token from `span` if it continues into a
    grouped larger number (e.g. 'Mac Mini 55,000' → 'Mac Mini')."""
    toks = span.split()
    if not toks or not toks[-1].isdigit():
        return span
    # After the span end there must be ',digits' to classify as price.
    rest = text[end:end + 2]
    if rest.startswith(","):
        return " ".join(toks[:-1]).strip()
    return span


# ============================================================================
# Tier 4 — learned co-occurrence (delegates to brand_memory.ProductResolutionMemory)
# ============================================================================

# placeholder to avoid import cycle at module import — set at call time via
# the resolver's injected `memory`.


# ============================================================================
# Orchestrator
# ============================================================================

class ProductBrandResolver:
    """Tiered resolver orchestrating Tier 1→2→3→4.

    Args:
        add_brand_resolution: optional callable `mem.record_product_cooccurrence(
                              product, brand, video_id)` for Tier-4 learning.
        learned_lookup: optional callable `lookup(product) -> {brand, tier, ...}`
                        backed by ProductResolutionMemory (returning promoted,
                        >=N-distinct-video associations only).
        wikidata: WikidataProductLookup instance (Tier 2).
        qwen: optional Qwen resolver callable (Tier 3);
              `resolve(product, frame) -> {"manufacturer": str|None, ...}`.
        corroboration_fn: callable(product_span) -> bool, whether a visual logo
                          in the same scene corroborates a Tier-3 result.
        min_plausibility: float boost threshold a span must exceed to query T2+.
    """

    def __init__(
        self,
        wikidata: Optional[WikidataProductLookup] = None,
        learned_lookup=None,
        add_brand_resolution=None,
        qwen=None,
        corroboration_fn=None,
        min_plausibility: float = 0.5,
        extractor: Optional[ProductNameExtractor] = None,
    ):
        self.wikidata = wikidata or WikidataProductLookup()
        self.learned_lookup = learned_lookup
        self.add_brand_resolution = add_brand_resolution
        self.qwen = qwen
        self.corroboration_fn = corroboration_fn
        self.min_plausibility = min_plausibility
        self.extractor = extractor or ProductNameExtractor()
        self.tier_counts: Dict[str, int] = {}
        self.qwen_calls = 0
        self.low_confidence_candidates: List[dict] = []

    def _bump(self, tier: int) -> None:
        self.tier_counts[str(tier)] = self.tier_counts.get(str(tier), 0) + 1

    def resolve(
        self,
        text: str,
        frame=None,
        video_id: str = "",
        scene_brands=None,
        live: bool = True,
    ) -> List[dict]:
        """Resolve product names in `text` to brands (falling through tiers).

        Returns a list of resolutions:
          [{span, normalized, brand, resolution_tier, source, confidence,
            resolution_quality, product_span, meta}]
        Empty list when nothing resolves. Applies the plausibility gate before
        spending Tier 2/3 queries.

        When `live` is False only local signals are consulted (Tier 1 direct
        catalog match + Tier 4 learned memory + the Wikidata cache); no external
        query or Qwen call is spent. Used by cheap corroborating paths so junk
        text never triggers network activity.
        """
        if not text:
            return []
        # Tier 1 — direct brand-name match. If the whole text names a catalog
        # brand, that is a direct resolution and needs no product machinery.
        direct = match_brand(text)
        if direct:
            self._bump(TIER_DIRECT)
            return [{
                "span": text[:40], "normalized": normalize_text(text)[:40],
                "brand": direct, "resolution_tier": TIER_DIRECT,
                "source": "catalog", "confidence": 1.0,
                "resolution_quality": _TIER_QUALITY[TIER_DIRECT],
                "product_span": None, "meta": {},
            }]

        scene_brands = scene_brands or []
        candidates = self.extractor.extract(text)
        results: List[dict] = []
        for cand in candidates:
            if cand["boost"] < self.min_plausibility:
                continue  # not worth an expert/knowledge query
            res = self._resolve_candidate(
                cand, text, frame, video_id, scene_brands, live=live,
            )
            if res:
                results.append(res)
        return results

    def _resolve_candidate(self, cand, text, frame, video_id, scene_brands,
                           live=True):
        norm = cand["normalized"]
        # Tier 4 first? No — order is fixed cheapest/reliable-first: 1,2,3,4.
        # But Tier 4 is the CHEAPEST (local memory) and most trusted for products
        # already learned, so we check it before external/quewen:
        #   learned > wikidata > qwen (Tier 4 < 2 < 3 in cost & reliability).
        # We still honor the stated priority "1|2|3|4" for provenance, but a
        # learned association (4) is trusted as much as an external KB (2).
        learned = self.learned_lookup(cand["span"]) if self.learned_lookup else None
        if learned and learned.get("brand"):
            self._bump(TIER_LEARNED)
            return {
                "span": cand["span"], "normalized": norm,
                "brand": learned["brand"], "resolution_tier": TIER_LEARNED,
                "source": "learned_cooccurrence",
                "confidence": float(learned.get("confidence", 0.85)),
                "resolution_quality": _TIER_QUALITY[TIER_LEARNED],
                "product_span": norm,
                "meta": {"videos_observed": learned.get("videos_observed", [])},
            }

        # Tier 2 — Wikidata.
        if self.wikidata:
            wd = self.wikidata.lookup(norm, live=live)
            if wd and wd.get("brand"):
                self._bump(TIER_WIKIDATA)
                return {
                    "span": cand["span"], "normalized": norm,
                    "brand": wd["brand"], "resolution_tier": TIER_WIKIDATA,
                    "source": "wikidata",
                    "confidence": 0.85,
                    "resolution_quality": _TIER_QUALITY[TIER_WIKIDATA],
                    "product_span": norm,
                    "meta": {"wikidata_manufacturer":
                             wd.get("wikidata_manufacturer")},
                }
            # Wikidata returned a manufacturer but not a catalog brand: keep as
            # low-confidence foreign candidate (never fabricate a catalog brand).
            if wd and wd.get("wikidata_manufacturer"):
                self._low_confidence_candidate(norm, wd["wikidata_manufacturer"],
                                               TIER_WIKIDATA)

        # Tier 3 — Qwen3-VL (low trust). Skipped entirely on cache-only paths.
        if self.qwen and live:
            qres = self.qwen(norm, frame)
            self.qwen_calls += 1
            maker = (qres or {}).get("manufacturer")
            if maker:
                brand = _to_catalog_brand(maker)
                corrob = bool(self.corroboration_fn and
                              self.corroboration_fn(norm))
                if brand and corrob:
                    self._bump(TIER_QWEN_CORROBORATED)
                    return {
                        "span": cand["span"], "normalized": norm,
                        "brand": brand, "resolution_tier": TIER_QWEN_CORROBORATED,
                        "source": "qwen_vlm", "confidence": 0.6,
                        "resolution_quality":
                            _TIER_QUALITY[TIER_QWEN_CORROBORATED],
                        "product_span": norm, "meta": {"corroborated": True},
                    }
                # Uncorroborated (or foreign) — never resolve at full confidence.
                self._low_confidence_candidate(
                    norm, maker, TIER_QWEN_UNSUPPORTED, corroborated=corrob)

        return None  # could not resolve this candidate

    def _low_confidence_candidate(self, product, manufacturer, tier,
                                  corroborated=False):
        self.tier_counts[str(tier)] = self.tier_counts.get(str(tier), 0) + 1
        self.low_confidence_candidates.append({
            "product": product, "manufacturer": manufacturer,
            "resolution_tier": tier, "corroborated": corroborated,
            "source": "low_confidence_review",
        })

    def record_observation(self, product_span: str, brand: str, video_id: str) -> None:
        """Feed a Tier-1-resolved scene's co-occurrence back into Tier 4 memory."""
        if self.add_brand_resolution is not None:
            self.add_brand_resolution(product_span, brand, video_id)

    def emit_provenance(self) -> dict:
        return {
            "tier_counts": self.tier_counts,
            "qwen_calls": self.qwen_calls,
            "low_confidence_candidates": self.low_confidence_candidates,
        }
