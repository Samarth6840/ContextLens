"""
Layer 3 — Brand Recommender.

Ranks brand collaboration opportunities from the Layer 2 output:

  DIRECT     — brands detected in this video (logo / speech / OCR evidence)
  SUGGESTED  — brands that never appeared on screen but share a category with
               a detected brand (the prompt's Puma-for-a-Nike-creator case)

Explainability is built in: every recommendation carries `reasons` that state
which evidence (Layer 2b/timeline) or which graph relationship (Layer 3)
drove it — never a bare ranked list.

The ranking is intentionally simple (category affinity + evidence strength) so
it stays honest as a cold-start baseline. Phase 2 upgrades it with a
LightGCN-style affinity model and/or an LLM-as-ranker (prompt §7).
"""

import logging
from typing import Dict, List, Optional

import numpy as np

from src.layer3.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)

DEFAULT_CATEGORY_AFFINITY = 0.7


class BrandRecommender:
    """Ranked, explainable brand recommendations."""

    def __init__(
        self,
        graph: Optional[KnowledgeGraph] = None,
        category_affinity: float = DEFAULT_CATEGORY_AFFINITY,
    ):
        self.graph = graph or KnowledgeGraph()
        self.category_affinity = category_affinity

    def recommend(
        self,
        timeline: Dict[str, dict],
        brand_evidence: Optional[Dict[str, float]] = None,
        top_k: int = 12,
    ) -> List[dict]:
        """Rank recommendations from the brand timeline + evidence strengths.

        Args:
            timeline: brand_timeline dict from build_brand_timeline()
            brand_evidence: brand -> evidence strength (0-1); defaults to the
                            per-brand mean logo confidence.
            top_k: max recommendations to return

        Returns:
            Ranked list of dicts:
                brand, product, category, type (DIRECT|SUGGESTED),
                score, confidence, appearances, reasons[]
        """
        brand_evidence = brand_evidence or {}
        detected = sorted(timeline.keys())

        def _evidence(brand: str) -> float:
            """Evidence for a brand: explicit strength or mean logo confidence."""
            if brand in brand_evidence:
                return float(brand_evidence[brand])
            entry = timeline.get(brand, {})
            confs = [
                a.get("confidence") for a in entry.get("appearances", [])
                if a.get("modality") == "logo" and a.get("confidence") is not None
            ]
            return float(np.mean(confs)) if confs else 0.0

        recs: List[dict] = []

        # ── DIRECT — brands with evidence in this video ─────────────────
        for brand in detected:
            info = self.graph.catalog.get(brand, {})
            entry = timeline[brand]
            ev = max(_evidence(brand), entry.get("confidence", 0.0))
            modalities = entry.get("modalities", [])
            reasons = []
            n_logo = sum(
                1 for a in entry.get("appearances", [])
                if a.get("modality") == "logo"
            )
            if n_logo:
                reasons.append(f"LOGO / ON-SCREEN DETECTED — {n_logo} appearance(s)")
            if "speech" in modalities:
                reasons.append("MENTIONED IN SPOKEN CONTENT")
            if entry.get("cross_scene"):
                reasons.append(
                    "CROSS-SCENE — VISUAL + SPOKEN EVIDENCE LINKED"
                )
            if ev >= 0.5:
                reasons.append(f"STRONG EVIDENCE — CONFIDENCE {ev:.0%}")
            recs.append({
                "brand": brand,
                "product": info.get("product", brand),
                "category": (info.get("categories") or [info.get("category", "GENERAL")])[0],
                "type": "DIRECT",
                "score": round(min(1.0, ev), 3),
                "confidence": round(min(1.0, ev), 3),
                "appearances": entry.get("appearance_count", 0),
                "reasons": reasons or ["DETECTED IN VIDEO"],
            })

        # ── SUGGESTED — brands adjacent to detected ones (never on screen) ─
        for brand in self.graph.suggest_for(detected):
            info = self.graph.catalog.get(brand, {})
            cats = info.get("categories") or [info.get("category", "GENERAL")]
            drivers: List[str] = []
            driver_cats: set = set()
            for d in detected:
                shared = self.graph.shared_categories(brand, d)
                if shared:
                    drivers.append(d)
                    driver_cats |= set(shared)
            ev = 0.0
            for d in drivers:
                ev = max(ev, _evidence(d))
            score = round(ev * self.category_affinity, 3)

            reasons = []
            top_cat = driver_cats or set(cats[:1])
            for c in sorted(top_cat)[:2]:
                if drivers:
                    reasons.append(
                        f"SAME CATEGORY ({c}) AS {', '.join(drivers[:3])}"
                    )
                else:
                    reasons.append(f"CATEGORY ({c}) FITS THE CONTENT NICHE")
            if score < 0.15:
                score = 0.15  # category-affinity baseline for cold start

            recs.append({
                "brand": brand,
                "product": info.get("product", brand),
                "category": cats[0],
                "type": "SUGGESTED",
                "score": round(score, 3),
                "confidence": round(score, 3),
                "appearances": 0,
                "reasons": reasons or ["KNOWLEDGE-GRAPH CATEGORY FIT"],
            })

        recs.sort(key=lambda r: r["score"], reverse=True)
        return recs[:top_k]
