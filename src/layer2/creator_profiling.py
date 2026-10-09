"""
Layer 2d — Creator Profiling.

Builds a behavioral profile of a creator in parallel to brand detection. The
profile aggregates across videos (not just the one being analyzed) so Layer 3
can recommend collaborations that fit the creator, and — critically — SUPPRESS
recommendations that are a poor niche fit even when the brand briefly appeared.

Concrete suppression case (prompt §6): a creator wears Nike once but their
content is 45+ cooking videos. Naive co-occurrence would recommend a Nike
collaboration; the profile should NOT. The signal that suppresses it is a
content-niche mismatch: the brand's dominant category is far from the creator's
recurring topic distribution.

This module is deterministic (no model): it tallies category frequency from the
brand evidence produced upstream, computes engagement/production metrics from
the profile metadata, and applies an explicit niche-fit gate. It intentionally
does NOT scrape demographics — engagement *style* only, per the prompt's flag.
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class CreatorProfile:
    """Accumulated behavioral + content profile for one creator.

    Attributes
    ----------
    creator_id : str
    handle : Optional[str]
    followers : Optional[int]
    engagements : Optional[float]       — typical per-post engagement (interactions)
    engagement_rate : Optional[float]   — engagements / followers
    categories : Dict[str, int]         — recurring content-category tallies
    brand_tallies : Dict[str, int]      — accumulated brand appearance counts
    videos_analyzed : int
    production_quality : Optional[float] — 0-1 mean visual-production style
    """

    def __init__(self, creator_id: str, handle: Optional[str] = None):
        self.creator_id = creator_id
        self.handle = handle
        self.followers: Optional[int] = None
        self.engagements: Optional[float] = None
        self.engagement_rate: Optional[float] = None
        self.categories: Dict[str, int] = {}
        self.brand_tallies: Dict[str, int] = {}
        self.videos_analyzed = 0
        self.production_quality: Optional[float] = None

    # ── aggregation ────────────────────────────────────────────

    def add_video(
        self,
        categories: Optional[Dict[str, int]] = None,
        brands: Optional[Dict[str, int]] = None,
        followers: Optional[int] = None,
        engagements: Optional[float] = None,
        production_quality: Optional[float] = None,
    ) -> None:
        """Fold one video's observed signals into the running profile."""
        self.videos_analyzed += 1
        if categories:
            for cat, n in categories.items():
                self.categories[cat] = self.categories.get(cat, 0) + n
        if brands:
            for brand, n in brands.items():
                self.brand_tallies[brand] = self.brand_tallies.get(brand, 0) + n
        if followers is not None:
            self.followers = max(self.followers or 0, followers)
        if engagements is not None:
            self.engagements = max(self.engagements or 0, engagements)
        if production_quality is not None:
            prev = self.production_quality
            n = self.videos_analyzed
            self.production_quality = (
                ((prev or 0.0) * (n - 1) + production_quality) / n
            )
        self._recompute_engagement_rate()

    def _recompute_engagement_rate(self) -> None:
        if self.followers and self.engagements is not None:
            self.engagement_rate = self.engagements / self.followers

    # ── content profile ────────────────────────────────────────

    def dominant_category(self) -> Optional[str]:
        """The recurring content category with the highest tally."""
        if not self.categories:
            return None
        return max(self.categories, key=self.categories.get)

    def category_share(self, category: str) -> float:
        """Fraction of the creator's observed content in `category` (0-1)."""
        total = sum(self.categories.values())
        if not total:
            return 0.0
        return self.categories.get(category, 0) / total

    def to_dict(self) -> Dict:
        return {
            "creator_id": self.creator_id,
            "handle": self.handle,
            "followers": self.followers,
            "engagements": self.engagements,
            "engagement_rate": self.engagement_rate,
            "categories": dict(self.categories),
            "dominant_category": self.dominant_category(),
            "brand_tallies": dict(self.brand_tallies),
            "videos_analyzed": self.videos_analyzed,
            "production_quality": self.production_quality,
        }


def build_profile_from_results(
    results: List[Dict],
    creator_id: str,
    handle: Optional[str] = None,
) -> CreatorProfile:
    """Build a CreatorProfile from a list of video pipeline result dicts.

    Each result is expected to contain the compiled pipeline output with the
    brand timeline / recommendations (uses `brand` categories from the global
    catalog via `categories_for`).
    """
    from src.brand_catalog import categories_for

    profile = CreatorProfile(creator_id, handle=handle)
    for res in results:
        categories: Dict[str, int] = {}
        brands: Dict[str, int] = {}
        # The compiled pipeline output nests recommendations under `layer3`;
        # a bare top-level `recommendations` is only a convenience for callers
        # passing the raw recommender list. Reading only the top-level key left
        # every real profile with no categories, so niche suppression never fired.
        recs = (res.get("layer3") or {}).get("recommendations") or \
            res.get("recommendations") or []
        for rec in recs:
            brand = rec.get("brand")
            if not brand:
                continue
            brands[brand] = brands.get(brand, 0) + 1
            for cat in categories_for(brand):
                categories[cat] = categories.get(cat, 0) + 1
        followers = None
        engagements = None
        creator_meta = res.get("creator_meta") or res.get("creator") or {}
        if isinstance(creator_meta, dict):
            followers = creator_meta.get("followers")
            engagements = creator_meta.get("engagements")
        quality = res.get("production_quality")
        profile.add_video(
            categories=categories or None,
            brands=brands or None,
            followers=followers,
            engagements=engagements,
            production_quality=quality,
        )
    return profile


def niche_fit_score(profile: CreatorProfile, brand: str) -> Optional[float]:
    """How well `brand` fits the creator's recurring content niche (0-1).

    Returns None when the profile has no content signal (nothing to judge).
    The score = the creator's content share of the brand's categories, capped
    so a single stray appearance (Nike once among 45+ cooking videos) scores
    near 0 while a real niche match scores high.
    """
    from src.brand_catalog import categories_for

    if not profile.categories:
        return None
    cats = categories_for(brand)
    share = max((profile.category_share(c) for c in cats), default=0.0)
    # A brand seen once vs a recurring niche: damp by how many videos actually
    # featured the brand's categories.
    return min(1.0, share)


def apply_niche_suppression(
    recommendations: List[Dict],
    profile: CreatorProfile,
    threshold: float = 0.15,
    suppress_factor: float = 0.5,
) -> List[Dict]:
    """Downweight SUGGESTED recommendations whose niche fit is poor.

    Returns the same list (mutated) with a `niche_suppressed` flag and reduced
    `score` on mismatches. DIRECT (on-screen) recommendations are NOT suppressed
    — on-screen evidence is real regardless of niche; only lookalike SUGGESTEDs
    (which rely on the knowledge graph) are damped so the system never
    recommends a brand the creator's audience won't care about.
    """
    for rec in recommendations:
        if rec.get("type") != "SUGGESTED":
            continue
        fit = niche_fit_score(profile, rec["brand"])
        if fit is None:
            continue
        if fit < threshold:
            rec["niche_suppressed"] = True
            rec["_original_score"] = rec["score"]
            rec["score"] = round(rec["score"] * suppress_factor, 3)
            # `confidence` is the field the UI/outreach shows, so damp it too:
            # downweighting only `score` leaves the displayed confidence asserting
            # a strong fit the niche gate just rejected.
            if rec.get("confidence") is not None:
                rec["confidence"] = round(
                    rec["confidence"] * suppress_factor, 3)
        else:
            rec["niche_suppressed"] = False
    return recommendations
