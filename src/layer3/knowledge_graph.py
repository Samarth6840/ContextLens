"""
Layer 3 — Knowledge Graph (brand → category → adjacent brands).

Enables the core product value: recommending brands that NEVER appeared on
screen. A fitness creator whose video shows Nike gets Puma/Asics/Decathlon
suggested because they share categories in the graph.

For v1 the graph is derived from the curated brand catalog (each brand lists
its categories). The prompt (§7) specifies this manual-curation-first approach,
with LLM-assisted construction / product-taxonomy mining as the follow-up.

Edges are "shares a category". Category weights can be tuned later; v1 treats
every shared category equally.

Phase 2 enrichment: relations are now TYPED and WEIGHTED so the graph can
distinguish a direct competitor (same category) from a complementary brand
(related category — an accessory for a detected flagship product, a payment
network next to a phone brand, etc.). This makes recommendations more targeted:
a detected phone brand surfaces complimenting accessories, not just rival phones.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from src.brand_catalog import BRAND_CATALOG

logger = logging.getLogger(__name__)

# Weight for a "same category" (direct competitor) edge.
SAME_CATEGORY_WEIGHT = 1.0
# Weight for a "complementary" (related category) edge.
COMPLEMENTARY_WEIGHT = 0.55

# Category clusters that are treated as "complementary" to each other.
# A brand in one member can be a natural recommendation for a creator whose
# content featured a brand in another member. Inferred, curated for v1.
RELATED_CATEGORY_CLUSTERS: List[Set[str]] = [
    {"APPAREL", "FOOTWEAR", "ACCESSORIES", "SPORTS", "STREETWEAR", "DENIM", "OUTDOOR"},
    {"ELECTRONICS", "TECH", "SEMICONDUCTOR"},
    {"BEVERAGE", "FOOD", "DRINKWARE", "ENERGY"},
    {"FINANCE", "PAYMENTS"},
]

# Map each category to every other category it is complementary with.
_COMPLEMENTARY_CACHE: Dict[str, Set[str]] = {}


def _complementary_categories(cat: str) -> Set[str]:
    """Return the set of categories complementary to `cat` (excluding itself)."""
    if not _COMPLEMENTARY_CACHE:
        for cluster in RELATED_CATEGORY_CLUSTERS:
            for member in cluster:
                _COMPLEMENTARY_CACHE.setdefault(member, set()).update(cluster - {member})
    return set(_COMPLEMENTARY_CACHE.get(cat, set()))


class KnowledgeGraph:
    """Typed, weighted, category-based knowledge graph over the brand catalog.

    Two edge kinds:
      - same-category (detected brand <-> competitor sharing a category)
      - complementary (detected brand <-> brand in a related category cluster)

    All v1 methods (`categories_for`, `neighbors`, `shared_categories`,
    `suggest_for`) keep their previous semantics so callers (recommender, tests)
    remain compatible; the enriched methods add relation typing.
    """

    def __init__(self, catalog: Optional[Dict[str, dict]] = None):
        self.catalog = catalog if catalog is not None else BRAND_CATALOG
        self._category_brands: Dict[str, Set[str]] = defaultdict(set)
        for brand, info in self.catalog.items():
            for cat in self.categories_for(brand):
                self._category_brands[cat].add(brand)

    def categories_for(self, brand: str) -> List[str]:
        info = self.catalog.get(brand)
        if not info:
            return ["GENERAL"]
        return info.get("categories") or [info.get("category", "GENERAL")]

    def neighbors(self, brand: str) -> List[str]:
        """All catalog brands sharing at least one category with `brand`."""
        out: Set[str] = set()
        for cat in self.categories_for(brand):
            out |= self._category_brands.get(cat, set())
        out.discard(brand)
        return sorted(out)

    def suggest_for(self, detected: List[str]) -> List[str]:
        """All catalog brands adjacent to any detected brand, excluding detected."""
        detected = set(detected)
        adj: Set[str] = set()
        for brand in detected:
            adj |= set(self.neighbors(brand))
        adj -= detected
        return sorted(adj)

    def shared_categories(self, a: str, b: str) -> List[str]:
        return sorted(set(self.categories_for(a)) & set(self.categories_for(b)))

    # ── Phase 2 enriched API ────────────────────────────────────────────────

    def relation(self, a: str, b: str) -> Tuple[str, float]:
        """Classify the edge between two catalog brands, if any.

        Returns (relation, weight) where relation is one of:
          "none"            — no edge
          "same_category"   — direct competitor
          "complementary"   — related-category (accessory / ecosystem fit)
        """
        if a not in self.catalog or b not in self.catalog:
            return "none", 0.0
        if self.shared_categories(a, b):
            return "same_category", SAME_CATEGORY_WEIGHT
        cats_a = set(self.categories_for(a))
        cats_b = set(self.categories_for(b))
        if any(cb in _complementary_categories(ca) for ca in cats_a for cb in cats_b):
            return "complementary", COMPLEMENTARY_WEIGHT
        return "none", 0.0

    def suggested_with_relation(
        self, detected: List[str]
    ) -> Dict[str, List[Tuple[str, str, float]]]:
        """Suggest brands with typed relations.

        Returns {suggested_brand: [(relation_type, weight, triggered_by_brand), ...]}.
        A suggested brand can relate to several detected brands (e.g. share a
        category with one and be complementary to another).
        """
        detected = set(detected)
        by_cand: Dict[str, Dict[Tuple[str, str], float]] = defaultdict(dict)
        for det in detected:
            for cand in self.catalog:
                if cand in detected:
                    continue
                rel, weight = self.relation(det, cand)
                if rel == "none":
                    continue
                key = (det, cand)
                # Keep the strongest of parallel edges from the same (det, cand).
                if weight > by_cand[cand].get(key, 0.0):
                    by_cand[cand][key] = weight
        out: Dict[str, List[Tuple[str, str, float]]] = {}
        for cand, edges in by_cand.items():
            out[cand] = [
                (rel, w, det)
                for (det, _cand), w in edges.items()
                for rel in ("same_category", "complementary")
                if self.relation(det, cand)[0] == rel and self.relation(det, cand)[1] == w
            ]
        return out

    def best_relation_to(self, cand: str, detected: List[str]) -> Tuple[str, float, str]:
        """Best-scored relation from any detected brand to `cand`."""
        best = ("none", 0.0, "")
        for det in detected:
            rel, weight = self.relation(det, cand)
            if weight > best[1]:
                best = (rel, weight, det)
        return best

