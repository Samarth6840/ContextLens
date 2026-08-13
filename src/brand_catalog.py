"""
Central Brand Catalog — single source of truth for brand identities.

Used by Layer 1 (YOLO-World text queries, OCR text matching), Layer 2
(evidence, timeline) and Layer 3 (knowledge graph, contact lookup).

Each brand entry carries:
    product          — canonical flagship product label
    category         — primary category
    categories       — categories the brand belongs to (drives Layer 3 graph)
    aliases          — OCR/ASR-friendly alias strings used for matching
    contact_email    — best-effort public contact (placeholder, edit in CRM)
    contact_website  — official public domain
    contact_verified — False → the email is a placeholder to replace

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
        "contact_email": "partnerships@nike.com",
        "contact_website": "https://www.nike.com",
        "contact_verified": False,
    },
    "ADIDAS": {
        "product": "Adidas Samba",
        "category": "APPAREL",
        "categories": ["APPAREL", "FOOTWEAR", "SPORTS"],
        "aliases": ['adidas', 'three stripes', 'एडिडास'],
        "contact_email": "partnerships@adidas.com",
        "contact_website": "https://www.adidas.com",
        "contact_verified": False,
    },
    "PUMA": {
        "product": "Puma Suede",
        "category": "APPAREL",
        "categories": ["APPAREL", "FOOTWEAR", "SPORTS"],
        "aliases": ['puma', 'cat logo', 'पूमा'],
        "contact_email": "brand@puma.com",
        "contact_website": "https://us.puma.com",
        "contact_verified": False,
    },
    "ASICS": {
        "product": "Asics Gel-Kayano",
        "category": "FOOTWEAR",
        "categories": ["FOOTWEAR", "SPORTS"],
        "aliases": ['asics', 'एसिक्स'],
        "contact_email": "contact@asics.com",
        "contact_website": "https://www.asics.com",
        "contact_verified": False,
    },
    "REEBOK": {
        "product": "Reebok Club C",
        "category": "FOOTWEAR",
        "categories": ["FOOTWEAR", "SPORTS", "APPAREL"],
        "aliases": ['reebok', 'रीबॉक'],
        "contact_email": "pr@reebok.com",
        "contact_website": "https://www.reebok.com",
        "contact_verified": False,
    },
    "NEW BALANCE": {
        "product": "New Balance 574",
        "category": "FOOTWEAR",
        "categories": ["FOOTWEAR", "SPORTS", "APPAREL"],
        "aliases": ['new balance', 'न्यू बैलेंस'],
        "contact_email": "mediarelations@newbalance.com",
        "contact_website": "https://www.newbalance.com",
        "contact_verified": False,
    },
    "UNDER ARMOUR": {
        "product": "Under Armour HOVR",
        "category": "APPAREL",
        "categories": ["APPAREL", "FOOTWEAR", "SPORTS"],
        "aliases": ['under armour', 'underarmour', 'अंडर आर्मर'],
        "contact_email": "pr@underarmour.com",
        "contact_website": "https://www.underarmour.com",
        "contact_verified": False,
    },
    "DECATHLON": {
        "product": "Decathlon Sports Gear",
        "category": "SPORTS",
        "categories": ["SPORTS", "OUTDOOR", "APPAREL"],
        "aliases": ['decathlon', 'डेकाथलॉन'],
        "contact_email": "partners@decathlon.com",
        "contact_website": "https://www.decathlon.com",
        "contact_verified": False,
    },
    "APPLE": {
        "product": "Apple Vision Pro",
        "category": "ELECTRONICS",
        "categories": ["ELECTRONICS", "TECH"],
        "aliases": ['apple', 'एप्पल', 'आईफोन'],
        "contact_email": "press@apple.com",
        "contact_website": "https://www.apple.com",
        "contact_verified": False,
    },
    "SAMSUNG": {
        "product": "Samsung Galaxy",
        "category": "ELECTRONICS",
        "categories": ["ELECTRONICS", "TECH"],
        "aliases": ['samsung', 'galaxy', 'सैमसंग', 'सैमसन'],
        "contact_email": "pr@samsung.com",
        "contact_website": "https://www.samsung.com",
        "contact_verified": False,
    },
    "SONY": {
        "product": "Sony WH-1000XM5",
        "category": "ELECTRONICS",
        "categories": ["ELECTRONICS", "TECH"],
        "aliases": ['sony', 'सोनी'],
        "contact_email": "pr@sony.com",
        "contact_website": "https://www.sony.com",
        "contact_verified": False,
    },
    "LG": {
        "product": "LG OLED",
        "category": "ELECTRONICS",
        "categories": ["ELECTRONICS", "TECH"],
        "aliases": ["lg"],
        "contact_email": "pr@lg.com",
        "contact_website": "https://www.lg.com",
        "contact_verified": False,
    },
    "GOOGLE": {
        "product": "Google Pixel",
        "category": "ELECTRONICS",
        "categories": ["ELECTRONICS", "TECH"],
        "aliases": ['google', 'pixel', 'गूगल'],
        "contact_email": "press@google.com",
        "contact_website": "https://about.google",
        "contact_verified": False,
    },
    "MICROSOFT": {
        "product": "Microsoft Surface",
        "category": "ELECTRONICS",
        "categories": ["ELECTRONICS", "TECH"],
        "aliases": ['microsoft', 'माइक्रोसॉफ्ट'],
        "contact_email": "rapidresponse@microsoft.com",
        "contact_website": "https://www.microsoft.com",
        "contact_verified": False,
    },
    "META": {
        "product": "Meta Quest",
        "category": "ELECTRONICS",
        "categories": ["ELECTRONICS", "TECH", "SOCIAL"],
        "aliases": ['meta', 'facebook', 'instagram', 'इंस्टाग्राम', 'फेसबुक'],
        "contact_email": "press@meta.com",
        "contact_website": "https://about.meta.com",
        "contact_verified": False,
    },
    "AMAZON": {
        "product": "Amazon Devices",
        "category": "ELECTRONICS",
        "categories": ["ELECTRONICS", "TECH", "RETAIL"],
        "aliases": ['amazon', 'prime', 'अमेज़न', 'अमेजन'],
        "contact_email": "press@amazon.com",
        "contact_website": "https://www.amazon.com",
        "contact_verified": False,
    },
    "NESCAFÉ": {
        "product": "NESCAFÉ Gold",
        "category": "BEVERAGE",
        "categories": ["BEVERAGE", "FOOD"],
        "aliases": ["nescafe", "nescafé"],
        "contact_email": "consumer.services@nescafe.com",
        "contact_website": "https://www.nescafe.com",
        "contact_verified": False,
    },
    "COCA-COLA": {
        "product": "Coca-Cola Zero",
        "category": "BEVERAGE",
        "categories": ["BEVERAGE", "FOOD"],
        "aliases": ['coca cola', 'coca-cola', 'coke', 'कोका कोला'],
        "contact_email": "pr@coca-cola.com",
        "contact_website": "https://www.coca-cola.com",
        "contact_verified": False,
    },
    "PEPSI": {
        "product": "Pepsi Max",
        "category": "BEVERAGE",
        "categories": ["BEVERAGE", "FOOD"],
        "aliases": ['pepsi', 'पेप्सी'],
        "contact_email": "pr@pepsico.com",
        "contact_website": "https://www.pepsi.com",
        "contact_verified": False,
    },
    "RED BULL": {
        "product": "Red Bull Energy",
        "category": "BEVERAGE",
        "categories": ["BEVERAGE", "ENERGY", "SPORTS"],
        "aliases": ['red bull', 'रेड बुल'],
        "contact_email": "media@redbull.com",
        "contact_website": "https://www.redbull.com",
        "contact_verified": False,
    },
    "STARBUCKS": {
        "product": "Starbucks Cup",
        "category": "BEVERAGE",
        "categories": ["BEVERAGE", "FOOD"],
        "aliases": ['starbucks', 'स्टारबक्स'],
        "contact_email": "partnerrelations@starbucks.com",
        "contact_website": "https://www.starbucks.com",
        "contact_verified": False,
    },
    "STANLEY": {
        "product": "Stanley Mug",
        "category": "DRINKWARE",
        "categories": ["DRINKWARE", "OUTDOOR"],
        "aliases": ["stanley"],
        "contact_email": "pr@stanley.com",
        "contact_website": "https://www.stanley1913.com",
        "contact_verified": False,
    },
    "YETI": {
        "product": "YETI Rambler",
        "category": "DRINKWARE",
        "categories": ["DRINKWARE", "OUTDOOR"],
        "aliases": ["yeti"],
        "contact_email": "pr@yeti.com",
        "contact_website": "https://www.yeti.com",
        "contact_verified": False,
    },
    "MERCEDES": {
        "product": "Mercedes C-Class",
        "category": "AUTOMOTIVE",
        "categories": ["AUTOMOTIVE", "LUXURY"],
        "aliases": ['mercedes', 'mercedes-benz', 'mercedes benz', 'benz', 'मर्सिडीज', 'बेंज'],
        "contact_email": "pr@mercedes-benz.com",
        "contact_website": "https://www.mercedes-benz.com",
        "contact_verified": False,
    },
    "BMW": {
        "product": "BMW 5 Series",
        "category": "AUTOMOTIVE",
        "categories": ["AUTOMOTIVE", "LUXURY"],
        "aliases": ['bmw', 'बीएमडब्ल्यू', 'बी एम डब्ल्यू'],
        "contact_email": "pr@bmwgroup.com",
        "contact_website": "https://www.bmw.com",
        "contact_verified": False,
    },
    "TESLA": {
        "product": "Tesla Model 3",
        "category": "AUTOMOTIVE",
        "categories": ["AUTOMOTIVE", "TECH"],
        "aliases": ['tesla', 'टेस्ला'],
        "contact_email": "press@tesla.com",
        "contact_website": "https://www.tesla.com",
        "contact_verified": False,
    },
    "SUPREME": {
        "product": "Supreme Box Logo",
        "category": "APPAREL",
        "categories": ["APPAREL", "STREETWEAR"],
        "aliases": ["supreme"],
        "contact_email": "info@supremenewyork.com",
        "contact_website": "https://www.supremenewyork.com",
        "contact_verified": False,
    },
    "GUCCI": {
        "product": "Gucci Bag",
        "category": "APPAREL",
        "categories": ["APPAREL", "LUXURY"],
        "aliases": ['gucci', 'गुच्ची'],
        "contact_email": "pr@gucci.com",
        "contact_website": "https://www.gucci.com",
        "contact_verified": False,
    },
    "ROLEX": {
        "product": "Rolex Submariner",
        "category": "LUXURY",
        "categories": ["LUXURY", "ACCESSORIES"],
        "aliases": ['rolex', 'रोलेक्स'],
        "contact_email": "pr@rolex.com",
        "contact_website": "https://www.rolex.com",
        "contact_verified": False,
    },
    "LEVI'S": {
        "product": "Levi's 501",
        "category": "APPAREL",
        "categories": ["APPAREL", "DENIM"],
        "aliases": ["levis", "levi's", "levi s"],
        "contact_email": "pr@levi.com",
        "contact_website": "https://www.levi.com",
        "contact_verified": False,
    },
    "ZARA": {
        "product": "Zara Collection",
        "category": "APPAREL",
        "categories": ["APPAREL", "RETAIL"],
        "aliases": ["zara"],
        "contact_email": "pr@zara.com",
        "contact_website": "https://www.zara.com",
        "contact_verified": False,
    },
    "LULULEMON": {
        "product": "Lululemon Align",
        "category": "APPAREL",
        "categories": ["APPAREL", "SPORTS"],
        "aliases": ["lululemon", "lulu"],
        "contact_email": "pr@lululemon.com",
        "contact_website": "https://shop.lululemon.com",
        "contact_verified": False,
    },
    "VISA": {
        "product": "Visa Card",
        "category": "FINANCE",
        "categories": ["FINANCE", "PAYMENTS"],
        "aliases": ["visa"],
        "contact_email": "pr@visa.com",
        "contact_website": "https://www.visa.com",
        "contact_verified": False,
    },
    "MASTERCARD": {
        "product": "Mastercard",
        "category": "FINANCE",
        "categories": ["FINANCE", "PAYMENTS"],
        "aliases": ['mastercard', 'master card', 'मास्टरकार्ड'],
        "contact_email": "media@mastercard.com",
        "contact_website": "https://www.mastercard.com",
        "contact_verified": False,
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


def _alias_pattern(alias_norm: str) -> re.Pattern:
    pattern = _ALIAS_PATTERNS.get(alias_norm)
    if pattern is None:
        pattern = re.compile(
            rf"(?<![A-Z0-9]){re.escape(alias_norm)}(?![A-Z0-9])"
        )
        _ALIAS_PATTERNS[alias_norm] = pattern
    return pattern


def match_brand(text: str) -> Optional[str]:
    """Return the canonical brand name found in `text`, or None.

    Uses word-boundary alias matching on normalized text. Requires the
    alias to be at least 3 characters to avoid trivial false positives.
    """
    norm = normalize_text(text)
    if not norm:
        return None
    for brand, info in BRAND_CATALOG.items():
        for alias in info["aliases"]:
            alias_norm = normalize_text(alias)
            if len(alias_norm) < 3:
                continue
            if _alias_pattern(alias_norm).search(norm):
                return brand
    return None


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
    matched_positions: set = set()
    for brand, info in BRAND_CATALOG.items():
        for alias in info["aliases"]:
            alias_norm = normalize_text(alias)
            if len(alias_norm) < 3:
                continue
            raw_pattern = re.compile(
                rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])",
                re.IGNORECASE,
            )
            for m in raw_pattern.finditer(text):
                found.append({
                    "brand": brand,
                    "position": m.start(),
                    "snippet": text[
                        max(0, m.start() - 20): m.start() + len(alias) + 20
                    ],
                })
                matched_positions.add(m.start())

    if fuzzy and max_distance > 0:
        # Supplement: token-level phonetic matching for transliteration
        # variants the exact matcher missed. One mention per token, using the
        # best-matching alias, and never over tokens that already matched.
        for token in re.split(r"\s+", text):
            start = text.find(token)
            if start in matched_positions or start < 0:
                continue
            best = _fuzzy_token_match(normalize_text(token), max_distance)
            if best is None:
                continue
            brand, _dist = best
            found.append({
                "brand": brand,
                "position": start,
                "snippet": text[
                    max(0, start - 20): start + len(token) + 20
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
    """Return contact metadata for a brand, or None."""
    info = lookup(brand)
    if not info:
        return None
    return {
        "email": info.get("contact_email"),
        "website": info.get("contact_website"),
        "verified": bool(info.get("contact_verified")),
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
