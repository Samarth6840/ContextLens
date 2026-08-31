"""
Layer 1 — Spec / Creator Attribute Extraction (Phase 3).

A creator's on-screen text overlays fall into three buckets that the pipeline
previously treated as one undifferentiated "LOGO REGION":
  1. brand wordmark      -> BrandResolver / catalog path (existing)
  2. spec callout        -> structured product data (screen size, thickness,
                            battery, material, dimensions, processor)
  3. creator attribution -> creator handle / follower count (outreach side)

This module implements (2) and (3) as LIGHTWEIGHT, deterministic, pure-regex
extractors over OCR text. No new models, no brand-catalog dependency, and it
FAILS CLOSED (returns nothing) for anything it cannot confidently extract —
spec data is never fabricated or guessed.

Callouts are the "free" signal the plan calls out: high-value structured product
data sitting in the video with zero brand-catalog dependency. We surface the raw
text plus a normalized value/unit pair where one is unambiguous.
"""

import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Spec-callout patterns ────────────────────────────────────
# Each pattern captures a (value, unit) pair. Values are normalized to a plain
# string; units are canonicalized. All are anchored loosely so they match text
# within a larger OCR line (e.g. "8 Inches Inner Display").

_SPEC_PATTERNS = [
    # Screen / display size — "<N> inch(es)" (case-insensitive)
    {
        "field": "screen_size",
        "regex": re.compile(r"(\d+(?:\.\d+)?)\s*(?:inch|inches|in\b)", re.I),
        "unit": "in",
    },
    # Thickness / dimensions — "<N> mm"
    {
        "field": "thickness",
        "regex": re.compile(r"(\d+(?:\.\d+)?)\s*mm\b", re.I),
        "unit": "mm",
    },
    # Battery capacity — "<N> mAh"
    {
        "field": "battery",
        "regex": re.compile(r"(\d+(?:\.\d+)?)\s*mah\b", re.I),
        "unit": "mAh",
    },
    # Weight — "<N> g" or "<N> grams"
    {
        "field": "weight",
        "regex": re.compile(r"(\d+(?:\.\d+)?)\s*(?:g|grams?)\b", re.I),
        "unit": "g",
    },
    # Storage / RAM — "<N> GB" / "<N> TB"
    {
        "field": "storage",
        "regex": re.compile(r"(\d+(?:\.\d+)?)\s*(gb|tb)\b", re.I),
        "unit": lambda m: m.group(2).upper(),
    },
    # Refresh rate — "<N> Hz"
    {
        "field": "refresh_rate",
        "regex": re.compile(r"(\d+(?:\.\d+)?)\s*hz\b", re.I),
        "unit": "Hz",
    },
    # Megapixels — "<N> MP"
    {
        "field": "camera",
        "regex": re.compile(r"(\d+(?:\.\d+)?)\s*mp\b", re.I),
        "unit": "MP",
    },
]

# Material / feature callouts — matched as free text, normalized to title case.
_MATERIAL_TERMS = [
    "titanium alloy", "stainless steel", "aluminum alloy", "aluminium alloy",
    "glass back", "ceramic", "gorilla glass", "leather", "carbon fiber",
    "vegan leather",
]

# Processor / chip callouts — "Snapdragon", "A18", "Exynos", "Tensor", "Dimensity".
_CHIP_TERMS = [
    "snapdragon", "exynos", "dimensity", "tensor", "a-series", "ryzen",
]

# ── Creator-attribution patterns ─────────────────────────────
# "@<handle>" anywhere; follower counts like "4.3m followers" / "1.2M followers".
_HANDLE_RE = re.compile(r"@([A-Za-z0-9_.]{1,30})", re.I)
_FOLLOWERS_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*([km]?)\s*followers?", re.I
)


def _norm_num(value: str) -> float:
    return float(value)


def extract_specs(text: str) -> List[Dict]:
    """Extract structured spec callouts from OCR text.

    Returns a list of {field, value, unit, raw} dicts, deduped by field keeping
    the first occurrence. Fails closed: returns [] when nothing matches.
    """
    if not text:
        return []
    norm_text = " " + " ".join((text or "").split()) + " "
    found: Dict[str, Dict] = {}
    for spec in _SPEC_PATTERNS:
        for m in spec["regex"].finditer(norm_text):
            unit = spec["unit"](m) if callable(spec["unit"]) else spec["unit"]
            key = spec["field"]
            if key in found:
                continue  # keep first occurrence per field
            try:
                value = _norm_num(m.group(1))
            except ValueError:
                continue
            found[key] = {
                "field": key,
                "value": value,
                "unit": unit,
                "raw": m.group(0).strip(),
            }

    # Material / feature (free-text, normalized to title case).
    low = " " + norm_text.lower() + " "
    for term in _MATERIAL_TERMS:
        if term in low:
            key = "material"
            if key not in found:
                found[key] = {
                    "field": key,
                    "value": term.title(),
                    "unit": "",
                    "raw": term,
                }
            break

    # Processor / chip.
    for term in _CHIP_TERMS:
        if term in low:
            key = "processor"
            if key not in found:
                # Capture the full chip name word (e.g. "Snapdragon 8 Elite Gen 5").
                term_match = re.search(re.escape(term), norm_text, re.I)
                if term_match is None:
                    continue
                tail = norm_text[term_match.start():]
                name_match = re.match(
                    r"([A-Za-z0-9 .\-]{2,40})", tail
                )
                name = name_match.group(1).strip() if name_match else term.title()
                found[key] = {
                    "field": key,
                    "value": name,
                    "unit": "",
                    "raw": term,
                }
            break

    return list(found.values())


def extract_creator(text: str) -> Dict:
    """Extract creator-attribution info from OCR text.

    Returns a dict with optional keys {handle, followers, followers_label}.
    Fails closed (empty dict) when nothing matches.
    """
    if not text:
        return {}
    out: Dict = {}
    hm = _HANDLE_RE.search(text)
    if hm:
        out["handle"] = hm.group(1)
    fm = _FOLLOWERS_RE.search(text)
    if fm:
        followers = _norm_num(fm.group(1))
        scale = (fm.group(2) or "").lower()
        if scale == "k":
            followers *= 1_000
        elif scale == "m":
            followers *= 1_000_000
        out["followers"] = round(followers)
        out["followers_label"] = fm.group(0).strip()
    return out
