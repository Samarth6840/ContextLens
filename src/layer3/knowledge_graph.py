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
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set

from src.brand_catalog import BRAND_CATALOG

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """Category-based knowledge graph over the brand catalog."""

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
