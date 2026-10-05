"""
Central Brand Catalog — single source of truth for brand identities.

Used by Layer 1 (YOLO-World text queries, OCR text matching), Layer 2
(evidence, timeline) and Layer 3 (knowledge graph, contact lookup).

Each brand entry carries:
    product          — canonical flagship product label
    category         — primary category
    categories       — categories the brand belongs to (drives Layer 3 graph)
    aliases          — OCR/ASR-friendly alias strings used for matching
    contact_website  — official public domain

There is deliberately NO contact_email field. The catalog previously carried
guessed addresses (pr@brand.com, partnerships@brand.com, brand@brand.com)
that were never sourced from anywhere — the same fabrication class as the
hallucinated logo brands this project exists to stop. A guessed address that
reaches a creator's outreach list is worse than no address at all, because it
looks real. Get a real address from a verified source (logo.dev response,
brand press page) and store it in the CRM, not in this file.

The catalog is deliberately small and manually curated for a first version
(the prompt's Section 7 specifies manual curation for v1, LLM-assisted mining
later). Extend BRAND_CATALOG in one place and every layer follows.
"""

import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Canonical brand name -> metadata.
BRAND_CATALOG: Dict[str, dict] = {
    "NIKE": {
        "product": "Nike Air",
        "category": "APPAREL",
        "categories": ["APPAREL", "FOOTWEAR", "SPORTS"],
        "aliases": ['nike', 'swoosh', 'नाइके'],
        "contact_website": "https://www.nike.com",
    },
    "ADIDAS": {
        "product": "Adidas Samba",
        "category": "APPAREL",
        "categories": ["APPAREL", "FOOTWEAR", "SPORTS"],
        "aliases": ['adidas', 'three stripes', 'एडिडास'],
        "contact_website": "https://www.adidas.com",
    },
    "PUMA": {
        "product": "Puma Suede",
        "category": "APPAREL",
        "categories": ["APPAREL", "FOOTWEAR", "SPORTS"],
        "aliases": ['puma', 'cat logo', 'पूमा'],
        "contact_website": "https://us.puma.com",
    },
    "ASICS": {
        "product": "Asics Gel-Kayano",
        "category": "FOOTWEAR",
        "categories": ["FOOTWEAR", "SPORTS"],
        "aliases": ['asics', 'एसिक्स'],
        "contact_website": "https://www.asics.com",
    },
    "REEBOK": {
        "product": "Reebok Club C",
        "category": "FOOTWEAR",
        "categories": ["FOOTWEAR", "SPORTS", "APPAREL"],
        "aliases": ['reebok', 'रीबॉक'],
        "contact_website": "https://www.reebok.com",
    },
    "NEW BALANCE": {
        "product": "New Balance 574",
        "category": "FOOTWEAR",
        "categories": ["FOOTWEAR", "SPORTS", "APPAREL"],
        "aliases": ['new balance', 'न्यू बैलेंस'],
        "contact_website": "https://www.newbalance.com",
    },
    "UNDER ARMOUR": {
        "product": "Under Armour HOVR",
        "category": "APPAREL",
        "categories": ["APPAREL", "FOOTWEAR", "SPORTS"],
        "aliases": ['under armour', 'underarmour', 'अंडर आर्मर'],
        "contact_website": "https://www.underarmour.com",
    },
    "DECATHLON": {
        "product": "Decathlon Sports Gear",
        "category": "SPORTS",
        "categories": ["SPORTS", "OUTDOOR", "APPAREL"],
        "aliases": ['decathlon', 'डेकाथलॉन'],
        "contact_website": "https://www.decathlon.com",
    },
    "APPLE": {
        "product": "Apple Vision Pro",
        "category": "ELECTRONICS",
        "categories": ["ELECTRONICS", "TECH"],
        "aliases": ['apple', 'एप्पल'],
        "contact_website": "https://www.apple.com",
    },
    "SAMSUNG": {
        "product": "Samsung Galaxy",
        "category": "ELECTRONICS",
        "categories": ["ELECTRONICS", "TECH"],
        "aliases": ['samsung', 'सैमसंग', 'सैमसन', 'सामसंग'],
        "contact_website": "https://www.samsung.com",
    },
    "SONY": {
        "product": "Sony WH-1000XM5",
        "category": "ELECTRONICS",
        "categories": ["ELECTRONICS", "TECH"],
        "aliases": ['sony', 'सोनी'],
        "contact_website": "https://www.sony.com",
    },
    "QUALCOMM": {
        "product": "Snapdragon",
        "category": "ELECTRONICS",
        "categories": ["ELECTRONICS", "TECH"],
        "aliases": ['qualcomm', 'क्वालकॉम'],
        "contact_website": "https://www.qualcomm.com",
    },
    "LG": {
        "product": "LG OLED",
        "category": "ELECTRONICS",
        "categories": ["ELECTRONICS", "TECH"],
        "aliases": ["lg", "lg electronics", "lge"],
        "contact_website": "https://www.lg.com",
    },
    "GOOGLE": {
        "product": "Google Pixel",
        "category": "ELECTRONICS",
        "categories": ["ELECTRONICS", "TECH"],
        "aliases": ['google', 'गूगल'],
        "contact_website": "https://about.google",
    },
    "NVIDIA": {
        "product": "NVIDIA GeForce",
        "category": "ELECTRONICS",
        "categories": ["ELECTRONICS", "SEMICONDUCTOR", "TECH"],
        "aliases": ['nvidia', 'एनवीडिया'],
        "contact_website": "https://www.nvidia.com",
    },
    "XIAOMI": {
        "product": "Xiaomi Redmi",
        "category": "ELECTRONICS",
        "categories": ["ELECTRONICS", "TECH"],
        "aliases": ['xiaomi', 'श्याओमी', 'ज़ियाओमी'],
        "contact_website": "https://www.mi.com/global",
    },
    "OPPO": {
        "product": "OPPO Find",
        "category": "ELECTRONICS",
        "categories": ["ELECTRONICS", "TECH"],
        "aliases": ['oppo', 'ओप्पो', 'ओपो'],
        "contact_website": "https://www.oppo.com",
    },
    "VIVO": {
        "product": "Vivo V Series",
        "category": "ELECTRONICS",
        "categories": ["ELECTRONICS", "TECH"],
        "aliases": ['vivo', 'वीवो'],
        "contact_website": "https://www.vivo.com",
    },
    "MICROSOFT": {
        "product": "Microsoft Surface",
        "category": "ELECTRONICS",
        "categories": ["ELECTRONICS", "TECH"],
        "aliases": ['microsoft', 'माइक्रोसॉफ्ट'],
        "contact_website": "https://www.microsoft.com",
    },
    "META": {
        "product": "Meta Quest",
        "category": "ELECTRONICS",
        "categories": ["ELECTRONICS", "TECH", "SOCIAL"],
        "aliases": ['meta', 'facebook', 'instagram', 'इंस्टाग्राम', 'फेसबुक'],
        "contact_website": "https://about.meta.com",
    },
    "AMAZON": {
        "product": "Amazon Devices",
        "category": "ELECTRONICS",
        "categories": ["ELECTRONICS", "TECH", "RETAIL"],
        "aliases": ['amazon', 'अमेज़न', 'अमेजन'],
        "contact_website": "https://www.amazon.com",
    },
    "NESCAFÉ": {
        "product": "NESCAFÉ Gold",
        "category": "BEVERAGE",
        "categories": ["BEVERAGE", "FOOD"],
        "aliases": ["nescafe", "nescafé"],
        "contact_website": "https://www.nescafe.com",
    },
    "COCA-COLA": {
        "product": "Coca-Cola Zero",
        "category": "BEVERAGE",
        "categories": ["BEVERAGE", "FOOD"],
        "aliases": ['coca cola', 'coca-cola', 'coke', 'कोका कोला'],
        "contact_website": "https://www.coca-cola.com",
    },
    "PEPSI": {
        "product": "Pepsi Max",
        "category": "BEVERAGE",
        "categories": ["BEVERAGE", "FOOD"],
        "aliases": ['pepsi', 'पेप्सी'],
        "contact_website": "https://www.pepsi.com",
    },
    "RED BULL": {
        "product": "Red Bull Energy",
        "category": "BEVERAGE",
        "categories": ["BEVERAGE", "ENERGY", "SPORTS"],
        "aliases": ['red bull', 'रेड बुल'],
        "contact_website": "https://www.redbull.com",
    },
    "STARBUCKS": {
        "product": "Starbucks Cup",
        "category": "BEVERAGE",
        "categories": ["BEVERAGE", "FOOD"],
        "aliases": ['starbucks', 'स्टारबक्स'],
        "contact_website": "https://www.starbucks.com",
    },
    "STANLEY": {
        "product": "Stanley Mug",
        "category": "DRINKWARE",
        "categories": ["DRINKWARE", "OUTDOOR"],
        "aliases": ["stanley"],
        "contact_website": "https://www.stanley1913.com",
    },
    "YETI": {
        "product": "YETI Rambler",
        "category": "DRINKWARE",
        "categories": ["DRINKWARE", "OUTDOOR"],
        "aliases": ["yeti"],
        "contact_website": "https://www.yeti.com",
    },
    "MERCEDES": {
        "product": "Mercedes C-Class",
        "category": "AUTOMOTIVE",
        "categories": ["AUTOMOTIVE", "LUXURY"],
        "aliases": ['mercedes', 'mercedes-benz', 'mercedes benz', 'benz', 'मर्सिडीज', 'बेंज'],
        "contact_website": "https://www.mercedes-benz.com",
    },
    "BMW": {
        "product": "BMW 5 Series",
        "category": "AUTOMOTIVE",
        "categories": ["AUTOMOTIVE", "LUXURY"],
        "aliases": ['bmw', 'बीएमडब्ल्यू', 'बी एम डब्ल्यू'],
        "contact_website": "https://www.bmw.com",
    },
    "TESLA": {
        "product": "Tesla Model 3",
        "category": "AUTOMOTIVE",
        "categories": ["AUTOMOTIVE", "ELECTRONICS"],
        "aliases": ['tesla', 'टेस्ला'],
        "contact_website": "https://www.tesla.com",
    },
    "SUPREME": {
        "product": "Supreme Box Logo",
        "category": "APPAREL",
        "categories": ["APPAREL", "STREETWEAR"],
        "aliases": ["supreme"],
        "contact_website": "https://www.supremenewyork.com",
    },
    "GUCCI": {
        "product": "Gucci Bag",
        "category": "APPAREL",
        "categories": ["APPAREL", "LUXURY"],
        "aliases": ['gucci', 'गुच्ची'],
        "contact_website": "https://www.gucci.com",
    },
    "ROLEX": {
        "product": "Rolex Submariner",
        "category": "LUXURY",
        "categories": ["LUXURY", "ACCESSORIES"],
        "aliases": ['rolex', 'रोलेक्स'],
        "contact_website": "https://www.rolex.com",
    },
    "LEVI'S": {
        "product": "Levi's 501",
        "category": "APPAREL",
        "categories": ["APPAREL", "DENIM"],
        "aliases": ["levis", "levi's", "levi s"],
        "contact_website": "https://www.levi.com",
    },
    "ZARA": {
        "product": "Zara Collection",
        "category": "APPAREL",
        "categories": ["APPAREL", "RETAIL"],
        "aliases": ["zara"],
        "contact_website": "https://www.zara.com",
    },
    "LULULEMON": {
        "product": "Lululemon Align",
        "category": "APPAREL",
        "categories": ["APPAREL", "SPORTS"],
        "aliases": ["lululemon", "lulu"],
        "contact_website": "https://shop.lululemon.com",
    },
    "VISA": {
        "product": "Visa Card",
        "category": "FINANCE",
        "categories": ["FINANCE", "PAYMENTS"],
        "aliases": ["visa"],
        "contact_website": "https://www.visa.com",
    },
    "MASTERCARD": {
        "product": "Mastercard",
        "category": "FINANCE",
        "categories": ["FINANCE", "PAYMENTS"],
        "aliases": ['mastercard', 'master card', 'मास्टरकार्ड'],
        "contact_website": "https://www.mastercard.com",
    },
}

# Generic YOLO-World fallback queries used alongside the catalog queries.
GENERIC_LOGO_QUERIES = [
    "brand logo",
    "company logo",
    "text logo",
    "product logo",
    "label",
    "wordmark",
]


def canonical_name(name: str) -> Optional[str]:
    """Return the canonical brand name if `name` normalizes into one.

    A full-name equality check (not alias matching) — used when a detector
    already reports a specific class like 'Samsung logo'.
    """
    norm = normalize_text(name)
    if not norm:
        return None
    for brand in BRAND_CATALOG:
        if normalize_text(brand) == norm:
            return brand
    return None


def normalize_text(text: str) -> str:
    """Normalize arbitrary text for comparison.

    Uppercases, strips punctuation, collapses whitespace.

    Unicode-aware: letters from any script (including Devanagari) and their
    combining marks (matras, anusvara) are preserved, so multilingual brand
    aliases like "सैमसंग" survive normalization. Only ASCII/non-letter
    punctuation and whitespace are removed.
    """
    if not text:
        return ""
    s = str(text).upper()
    # Keep Unicode word chars, combining marks, Devanagari vowel signs and
    # spaces; everything else (punctuation) becomes a space.
    s = re.sub(r"[^\w\u0300-\u036f\u0900-\u097f\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


_ALIAS_PATTERNS: Dict[str, re.Pattern] = {}


def _ordered_aliases(info: dict) -> list:
    """Aliases longest-first, so overlapping matches prefer the full phrase.

    "Mercedes-Benz" would otherwise match 'mercedes' AND 'mercedes-benz' AND
    'benz' (three mentions for one brand). Longest-first with span dedupe in
    `find_brand_mentions` yields one mention per region.
    """
    return sorted(info["aliases"], key=lambda a: len(normalize_text(a)), reverse=True)


def _alias_pattern(alias_norm: str) -> re.Pattern:
    pattern = _ALIAS_PATTERNS.get(alias_norm)
    if pattern is None:
        # The guard must cover Devanagari too. `[A-Z0-9]` only blocks ASCII
        # letters/digits, so सोनी (SONY) matched inside सोनीपत (a different
        # word) — an alias has to be bounded by *any* letter or digit, not
        # just a Latin one.
        pattern = re.compile(
            rf"(?<![A-Z0-9ऀ-ॿ]){re.escape(alias_norm)}(?![A-Z0-9ऀ-ॿ])"
        )
        _ALIAS_PATTERNS[alias_norm] = pattern
    return pattern


def match_brand(text: str) -> Optional[str]:
    """Return the canonical brand name found in `text`, or None.

    Uses letter-boundary alias matching on normalized text. Requires the alias
    to be at least 2 characters: word boundaries are enforced on both sides
    (e.g. "lg" won't match inside "BLOG"), so a 2-char alias is a deliberate
    real brand (LG) rather than a false-positive risk. Single characters are
    still too ambiguous to ever match.

    When several brands match, the LONGEST matched alias wins and ties break
    toward the leftmost occurrence. Returning the first brand in catalog order
    (as this used to) made the answer depend on dict insertion order: "Samsung
    vs Apple" returned APPLE purely because APPLE is cataloged first, which is
    not a property of the text.
    """
    norm = normalize_text(text)
    if not norm:
        return None
    best: Optional[tuple] = None  # (alias_len, -start, brand)
    for brand, info in BRAND_CATALOG.items():
        for alias in _ordered_aliases(info):
            alias_norm = normalize_text(alias)
            if len(alias_norm) < 2:
                continue
            m = _alias_pattern(alias_norm).search(norm)
            if m is None:
                continue
            cand = (len(alias_norm), -m.start(), brand)
            if best is None or cand > best:
                best = cand
    return best[2] if best else None


def _levenshtein_bounded(a: str, b: str, max_dist: int) -> int:
    """Bounded Levenshtein distance (early-exits above max_dist)."""
    if abs(len(a) - len(b)) > max_dist:
        return max_dist + 1
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        row_min = cur[0]
        for j, cb in enumerate(b, start=1):
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (0 if ca == cb else 1),
            )
            if cur[j] < row_min:
                row_min = cur[j]
        if row_min > max_dist:
            return max_dist + 1
        prev = cur
    return prev[-1]


def _fuzzy_token_match(
    token_norm: str, max_distance: int
) -> Optional[tuple]:
    """Best (brand, distance) for a token via bounded edit distance.

    Guards against over-eager fuzzy matching:
      - token and alias must both be >= 4 characters,
      - edit distance <= max_distance,
      - first code point must agree (the leading consonant/letter matches),
    which removes most accidental distance-1 collisions on short words.
    Returns None when no alias is close enough.
    """
    if len(token_norm) < 4:
        return None
    best: Optional[tuple] = None
    for brand, info in BRAND_CATALOG.items():
        for alias in info["aliases"]:
            alias_norm = normalize_text(alias)
            if len(alias_norm) < 4:
                continue
            if alias_norm[:1] != token_norm[:1]:
                continue
            d = _levenshtein_bounded(token_norm, alias_norm, max_distance)
            if d <= max_distance:
                if best is None or d < best[1]:
                    best = (brand, d)
    return best


def find_brand_mentions(
    text: str,
    fuzzy: bool = False,
    max_distance: int = 1,
) -> List[dict]:
    """Find all known-brand mentions in raw text.

    Returns a list of dicts (sorted by position):
        {"brand": canonical, "position": int, "snippet": str}
    Word-boundary matching on the raw text (case-insensitive), covering both
    Latin and non-Latin (e.g. Devanagari) brand aliases from the catalog.

    Args:
        text: The raw transcript (any script).
        fuzzy: When True, also match tokens within `max_distance` edits of a
               catalog alias (phonetic/transliteration variants such as
               "सैमसन" for "सैमसंग"). Default False — the exact word-boundary
               path is the primary, zero-false-positive matcher; fuzzy is an
               opt-in supplement so it cannot silently inflate evidence.
        max_distance: Max bounded Levenshtein distance for fuzzy matching.
    """
    if not text:
        return []
    found = []
    # brand -> list of (start, end) spans already recorded, so overlapping
    # aliases for the same brand ("mercedes" + "mercedes-benz" + "benz" in
    # "Mercedes-Benz") collapse into one mention instead of inflating the count.
    spans_by_brand: dict = {}
    for brand, info in BRAND_CATALOG.items():
        for alias in _ordered_aliases(info):
            alias_norm = normalize_text(alias)
            if len(alias_norm) < 2:
                continue
            raw_pattern = re.compile(
                rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])",
                re.IGNORECASE,
            )
            for m in raw_pattern.finditer(text):
                span = (m.start(), m.end())
                if any(s <= span[0] and span[1] <= e for s, e in spans_by_brand.get(brand, ())):
                    continue
                found.append({
                    "brand": brand,
                    "position": m.start(),
                    "snippet": text[
                        max(0, m.start() - 20): m.end() + 20
                    ],
                })
                spans_by_brand.setdefault(brand, []).append(span)

    if fuzzy and max_distance > 0:
        # Supplement: token-level phonetic matching for transliteration
        # variants the exact matcher missed. One mention per token, using the
        # best-matching alias, and never over tokens that already matched.
        for m in re.finditer(r"\S+", text):
            token = m.group(0)
            start = m.start()
            if any(s <= start and m.end() <= e for spans in spans_by_brand.values() for s, e in spans):
                continue
            best = _fuzzy_token_match(normalize_text(token), max_distance)
            if best is None:
                continue
            brand, _dist = best
            found.append({
                "brand": brand,
                "position": start,
                "snippet": text[
                    max(0, start - 20): m.end() + 20
                ],
            })

    found.sort(key=lambda x: x["position"])
    return found


def lookup(brand: str) -> Optional[dict]:
    """Return catalog metadata for a (possibly non-canonical) brand name."""
    name = canonical_name(brand)
    if name:
        return BRAND_CATALOG[name]
    matched = match_brand(brand)
    if matched:
        return BRAND_CATALOG[matched]
    return None


def contact_for(brand: str) -> Optional[dict]:
    """Return contact metadata for a brand, or None.

    `email` is always None: the catalog carries no contact addresses, because
    every address it used to carry was guessed rather than sourced (see the
    module docstring). Callers must source a real address from a verified
    lookup before any outreach. `website` is the official public domain, which
    IS curated and safe to show.
    """
    info = lookup(brand)
    if not info:
        return None
    return {
        "email": None,
        "website": info.get("contact_website"),
        "verified": False,
    }


def build_text_queries() -> List[str]:
    """Build YOLO-World text queries: '<Brand> logo' per catalog brand.

    These let the zero-shot logo detector match specific brands directly
    (e.g. 'Nike logo') instead of only generic 'brand logo' boxes.
    """
    queries = []
    for brand in BRAND_CATALOG:
        queries.append(f"{brand} logo")
    return queries


def all_brand_names() -> List[str]:
    """All canonical brand names — used for speech-mention scanning."""
    return list(BRAND_CATALOG.keys())


def product_for(brand: str) -> str:
    info = lookup(brand)
    return info.get("product", brand) if info else brand


def categories_for(brand: str) -> List[str]:
    info = lookup(brand)
    if not info:
        return ["GENERAL"]
    return info.get("categories") or [info.get("category", "GENERAL")]
