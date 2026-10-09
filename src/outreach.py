"""
Layer 4 — Personalized outreach automation (Phase 3).

The ContextLens review flagged that the original outreach draft was a static
uppercase template keyed only to "a brand appeared on screen". This module turns
outreach generation into a data-driven, explainable step that consumes the full
Phase 1 + Phase 2 pipeline output:

  * layer3.recommendations — ranked, reason-bearing brand collaborations
    (DIRECT on-screen vs SUGGESTED adjacent, with category/complementary/affinity
    drivers and CREATOR-BRAND AFFINITY signals).
  * layer2d.creator_profile — recurring content categories, dominant niche,
    followers, engagement rate, production quality, videos analyzed.
  * layer2c.brand_memory — whether the target brand has appeared across prior
    videos (indirect-reference resolutions, cross-video persistence).

The generator:

  1. Selects the target brand's recommendation from the pipeline (falling back to
     a look-up by name).
  2. Collates concrete PERSONALIZATION facts (the "evidence" strings that ground
     each copy line), so no claim is fabricated.
  3. Renders copy in several tone variants (professional / creator-voice /
     data-driven) using plain-string templates — no external LLM required, and
     therefore deterministic and auditable.
  4. Emits an explicit `rationale` listing which pipeline signals were used, so
     outreach decisions can be traced to real evidence.

Every generated value is derived from the input result; nothing is invented.
If the pipeline lacks the required context, the generator fails closed (returns
an error dict rather than guessing).
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _lookup_recommendation(recommendations: List[dict], brand: str) -> Optional[dict]:
    """Find the recommendation for `brand` (canonical, case-insensitive)."""
    key = (brand or "").strip().upper()
    if not key or not recommendations:
        return None
    for rec in recommendations:
        if str(rec.get("brand", "")).strip().upper() == key:
            return rec
    return None


def _pick_personalization_facts(rec: dict) -> List[str]:
    """Turn a recommendation's `reasons` into grounded, mail-friendly facts."""
    facts: List[str] = []
    rec_type = str(rec.get("type", "")).upper()
    reasons = rec.get("reasons", []) or []
    product = rec.get("product") or rec.get("brand")

    if rec_type == "DIRECT":
        facts.append(f"your {product} already appeared on screen in the content")
    else:
        facts.append(
            f"your {product} is a natural adjacent fit for the content's niche"
        )
    for reason in reasons:
        r = str(reason)
        if "LOGO / ON-SCREEN" in r:
            facts.append("it was detected on screen (logo visible)")
        elif "SPOKEN CONTENT" in r:
            facts.append("it was mentioned in the spoken content")
        elif "CROSS-SCENE" in r:
            facts.append("visual + spoken evidence were linked across scenes")
        elif "STRONG EVIDENCE" in r:
            facts.append("detection confidence was strong")
        elif "AFFINITY" in r:
            facts.append("the learned creator–brand affinity model also ranks it highly")
        elif "COMPLEMENTARY TO" in r:
            facts.append("it complements another brand the creator already features")
        elif "SAME CATEGORY" in r:
            facts.append("it sits in the same category as the creator's recurring content")
        elif "FITS THE CONTENT NICHE" in r:
            facts.append("the category is a proven content niche for the creator")
    # dedupe, keep order
    seen: set = set()
    uniq = []
    for f in facts:
        if f not in seen:
            seen.add(f)
            uniq.append(f)
    return uniq


def _profile_blurb(profile: Optional[dict]) -> str:
    """One-line human descriptor of the creator's content profile."""
    if not profile:
        return ""
    parts: List[str] = []
    niche = profile.get("dominant_category")
    if niche:
        parts.append(f"content centered on {niche}")
    n = profile.get("videos_analyzed")
    if n:
        parts.append(f"{n} videos analyzed")
    fans = profile.get("followers")
    if fans:
        parts.append(f"{fans:,} followers")
    er = profile.get("engagement_rate")
    if er:
        parts.append(f"{er:.1%} engagement")
    if not parts:
        return ""
    return ", ".join(parts)


def _tone_templates(tone: str) -> Dict[str, str]:
    """Return (subject_prefix, intro_opener, closing) for the requested tone."""
    base = {
        "professional": (
            "PARTNERSHIP OPPORTUNITY",
            "Hello {brand} Team,",
            "Best regards,",
        ),
        "creator_voice": (
            "Let's do something {brand} × {creator}",
            "Hey {brand} — quick one from {creator},",
            "Talk soon,",
        ),
        "data_driven": (
            "DATA-BACKED PLACEMENT — {brand} × {creator}",
            "To the {brand} partnerships team,",
            "Regards,",
        ),
    }
    return base.get(tone, base["professional"])


def _subject(tone, brand, creator, rec_type):
    prefix, _, _ = _tone_templates(tone)
    # creator_voice / data_driven prefixes carry literal {brand}/{creator}
    # placeholders; without this they ship in the subject line as raw braces.
    prefix = prefix.format(brand=brand, creator=creator)
    kind = "DIRECT PLACEMENT" if rec_type == "DIRECT" else "CONTENT-FIT OPPORTUNITY"
    return f"{prefix} — {kind}: {brand} × {creator}"


def generate_personalized_outreach(
    result: Dict[str, Any],
    brand: str,
    target: str = "",
    tone: str = "professional",
    creator_name: str = "",
) -> Dict[str, Any]:
    """Generate personalized outreach copy from a full pipeline result.

    Args:
        result: pipeline output dict (layer3.recommendations, layer2d.creator_profile,
                layer2c.brand_memory).
        brand: target brand name.
        target: recipient address/name (optional; empty -> blank line).
        tone: professional | creator_voice | data_driven.
        creator_name: override for the creator display name.

    Returns a dict with subject/body/rationale + all tones, or an error dict
    (fail-closed) when the brand has no grounded recommendation.
    """
    recommendations = (result.get("layer3") or {}).get("recommendations") or []
    profile = (result.get("layer2d") or {}).get("creator_profile")
    memory = (result.get("layer2c") or {}).get("brand_memory") or {}

    rec = _lookup_recommendation(recommendations, brand)
    if rec is None:
        return {
            "error": (
                "BRAND NOT RECOMMENDED — NO GROUNDED COLLABORATION DATA. "
                "Refusing to fabricate outreach copy for an unvalidated brand."
            ),
            "brand": brand,
            "status": "not_recommended",
        }

    key = str(rec.get("brand", "")).strip().upper()
    rec_type = str(rec.get("type", "")).upper()
    product = rec.get("product") or key

    # Creator identity — prefer explicit override, else job/creator, else handle.
    handle = (profile or {}).get("handle") or "the channel"
    creator = creator_name or handle or "the channel"
    blurb = _profile_blurb(profile)

    facts = _pick_personalization_facts(rec)
    memory_brands = memory.get("brands") or []
    cross_video = key in memory_brands

    rationale = {
        "brand": key,
        "type": rec_type,
        "score": rec.get("score"),
        "product": product,
        "recommendation_reasons": rec.get("reasons", []),
        "evidence_facts": facts,
        "creator_profile": blurb or None,
        "cross_video_memory": cross_video,
        "tone": tone,
    }

    # Personalization sentence block.
    lines: List[str] = [
        f"I run content that is a natural fit for {brand}; {blurb}."
        if blurb
        else f"I feature brands in a space that {brand} fits naturally.",
    ]
    if cross_video:
        lines.append(
            f"{brand} has turned up across more than one piece of content we analyzed "
            f"({len(memory_brands)} brands tracked in our brand memory)."
        )
    lines.extend(f"• {f}." for f in facts)
    personal = "\n".join(lines)

    # Compose the primary requested tone.
    prefix, opener, closing = _tone_templates(tone)
    body = (
        opener.format(brand=key, creator=creator)
        + "\n\n"
        + personal
        + "\n\n"
        + "I'd like to explore a native integration or sponsored segment that "
        "matches how the content is actually watched. Happy to share full "
        "view counts, audience demographics, and the complete scene breakdown."
        + "\n\n"
        + closing
        + "\n"
        + creator
    )
    if target:
        body = f"TO: {target}\n\n" + body

    return {
        "status": "ok",
        "brand": key,
        "product": product,
        "type": rec_type,
        "target": target,
        "subject": _subject(tone, key, creator, rec_type),
        "body": body,
        "rationale": rationale,
        "tones": {
            t: _render_tone(t, key, creator, target, personal, cross_video,
                            rec_type, blurb, facts)
            for t in ("professional", "creator_voice", "data_driven")
        },
    }


def _render_tone(tone, brand, creator, target, personal, cross_video,
                 rec_type, blurb, facts):
    prefix, opener, closing = _tone_templates(tone)
    body = (
        opener.format(brand=brand, creator=creator)
        + "\n\n"
        + personal
        + "\n\n"
        + "I'd like to explore a native integration or sponsored segment that "
        "matches how the content is actually watched. Happy to share full "
        "view counts, audience demographics, and the complete scene breakdown."
        + "\n\n"
        + closing
        + "\n"
        + creator
    )
    if target:
        body = f"TO: {target}\n\n" + body
    return {"subject": _subject(tone, brand, creator, rec_type), "body": body}


class PersonalizedOutreachGenerator:
    """Thin wrapper around generate_personalized_outreach for server use."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def generate(self, result: Dict[str, Any], brand: str, **kwargs) -> Dict[str, Any]:
        return generate_personalized_outreach(result, brand=brand, **kwargs)
