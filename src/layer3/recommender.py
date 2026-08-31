"""
Layer 3 — Brand Recommender.

Ranks brand collaboration opportunities from the Layer 2 output:

  DIRECT     — brands detected in this video (logo / speech / OCR evidence /
               product-catalog match)
  SUGGESTED  — brands that never appeared on screen but share a category with
               a detected brand (the prompt's Puma-for-a-Nike-creator case)

What "knowledge-graph ranked" actually means here (honest description):
  The upstream Layer 3 component (src/layer3/knowledge_graph.py) is a REAL,
  lightweight category-adjacency graph derived from the curated brand catalog:
  brand → categories → adjacent brands. It is the graph that supplies SUGGESTED
  brands and drives their score via category affinity. It is NOT a learned
  product-relation / competitor graph (no embedding-trained edges, no LLM-mined
  relations yet). DIRECT recommendations are ranked by evidence strength; only
  SUGGESTED recommendations are graph-ranked. If "knowledge-graph ranked" were
  claimed for the full output, that would overstate it.

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
# When the learned affinity model is fitted AND knows the creator, its score is
# blended with the graph/evidence score with this weight (the rest goes to the
# explainable graph/evidence signal). 0.0 disables the learned signal entirely.
DEFAULT_AFFINITY_BLEND = 0.5


class BrandRecommender:
    """Ranked, explainable brand recommendations.

    Ranking combines:
      - evidence strength (DIRECT) / category affinity (SUGGESTED) from the
        knowledge graph — always on, explainable;
      - an optional learned creator-brand affinity score (LightGCN) that is
        blended in ONLY when the creator is known to a fitted model; otherwise
        it is skipped (honest cold start).
    """

    def __init__(
        self,
        graph: Optional[KnowledgeGraph] = None,
        category_affinity: float = DEFAULT_CATEGORY_AFFINITY,
        affinity_model=None,
        affinity_blend: float = DEFAULT_AFFINITY_BLEND,
    ):
        self.graph = graph or KnowledgeGraph()
        self.category_affinity = category_affinity
        self.affinity_model = affinity_model
        self.affinity_blend = affinity_blend

    def _affinity_scores(self, creator_id, brands) -> Dict[str, float]:
        """Return {brand: affinity(0-1)} for a creator, or {} on cold start."""
        if not creator_id or self.affinity_model is None:
            return {}
        try:
            fitted = getattr(self.affinity_model, "is_fitted", False)
        except Exception:  # noqa: BLE001
            fitted = False
        if not fitted:
            return {}
        try:
            scores = self.affinity_model.predict_affinity(creator_id, brands)
        except Exception:  # noqa: BLE001
            return {}
        if scores is None:
            return {}
        return {b: float(s) for b, s in zip(brands, scores)}

    @staticmethod
    def _blend(graph_score: float, affinity: float, blend: float) -> float:
        if affinity <= 0.0:
            return graph_score
        return (1.0 - blend) * graph_score + blend * affinity

    def recommend(
        self,
        timeline: Dict[str, dict],
        brand_evidence: Optional[Dict[str, float]] = None,
        top_k: int = 12,
        creator_id: Optional[str] = None,
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

        # Optional learned-affinity scores for the creator across all candidate
        # brands (detected + suggested). Empty/None on cold start.
        all_candidates = set(detected) | set(self.graph.suggest_for(detected))
        affinity = self._affinity_scores(creator_id, sorted(all_candidates))

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
            aff_val = affinity.get(brand, 0.0)
            if aff_val > 0.0:
                reasons.append(f"CREATOR-BRAND AFFINITY — {aff_val:.0%}")
            final = self._blend(ev, aff_val, self.affinity_blend)
            recs.append({
                "brand": brand,
                "product": info.get("product", brand),
                "category": (info.get("categories") or [info.get("category", "GENERAL")])[0],
                "type": "DIRECT",
                "score": round(min(1.0, final), 3),
                "confidence": round(min(1.0, final), 3),
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
            base = ev * self.category_affinity
            if base < 0.15:
                base = 0.15  # category-affinity baseline for cold start

            reasons = []
            top_cat = driver_cats or set(cats[:1])
            for c in sorted(top_cat)[:2]:
                if drivers:
                    reasons.append(
                        f"SAME CATEGORY ({c}) AS {', '.join(drivers[:3])}"
                    )
                else:
                    reasons.append(f"CATEGORY ({c}) FITS THE CONTENT NICHE")
            # Complementary (related-category) relationship is worth surfacing.
            for d in detected:
                rel, _w = self.graph.relation(d, brand)
                if rel == "complementary":
                    reasons.append(f"COMPLEMENTARY TO {d}")
                    break
            aff_val = affinity.get(brand, 0.0)
            if aff_val > 0.0:
                reasons.append(f"CREATOR-BRAND AFFINITY — {aff_val:.0%}")
            score = self._blend(base, aff_val, self.affinity_blend)

            recs.append({
                "brand": brand,
                "product": info.get("product", brand),
                "category": cats[0],
                "type": "SUGGESTED",
                "score": round(score, 3),
                "confidence": round(score, 3),
                "appearances": 0,
                "reasons": reasons or ["CATEGORY-AFFINITY KNOWLEDGE-GRAPH FIT"],
            })

        recs.sort(key=lambda r: r["score"], reverse=True)
        return recs[:top_k]
