"""The resolver must say WHY a crop went unresolved.

Every identity failure used to exit one silent `return out`, so an unresolved
crop was indistinguishable from a detector false positive. That made brand
accuracy unreportable: 3-of-3 named read as perfect while the other 1148 crops
vanished with no attributable cause. These tests pin the cause onto the result.
"""

import numpy as np
import pytest

from src.layer2.brand_resolver import BrandResolver


class _Index:
    """Minimal LogoRetrievalIndex stand-in: fixed (brand, similarity) ranking."""

    def __init__(self, ranked):
        self._ranked = ranked
        self.is_empty = not ranked

    def query(self, _crop):
        return list(self._ranked)


def _resolver(ranked, **kw):
    return BrandResolver(
        ocr_extractor=None,
        class_confidence=0.4,
        retrieval_index=_Index(ranked),
        retrieval_min_similarity=kw.get("min_sim", 0.22),
        retrieval_min_margin=kw.get("min_margin", 0.10),
    )


def _crop():
    return np.full((64, 64, 3), 255, dtype=np.uint8)


DET = {"bbox": [0, 0, 32, 32], "confidence": 0.5, "class_name": "logo"}


def test_small_margin_rejection_is_attributable():
    # Two brands clear the floor but are too close to call.
    r = _resolver([("BMW", 0.30), ("AUDI", 0.28)])
    out = r.resolve([[DET]], [_crop()])[0][0]
    assert out["brand"] is None
    assert out["unresolved_reason"] == "small_margin"
    assert out["retrieval_diag"]["top1"] == "BMW"
    assert out["retrieval_diag"]["margin"] == pytest.approx(0.02, abs=1e-3)


def test_below_floor_rejection_is_attributable():
    r = _resolver([("BMW", 0.05)])
    out = r.resolve([[DET]], [_crop()])[0][0]
    assert out["brand"] is None
    assert out["unresolved_reason"] == "below_similarity_floor"


def test_missing_index_is_attributable():
    r = _resolver([])
    out = r.resolve([[DET]], [_crop()])[0][0]
    assert out["brand"] is None
    assert out["unresolved_reason"] == "no_retrieval_index"


def test_unambiguous_retrieval_still_names_a_brand():
    r = _resolver([("BMW", 0.31), ("AUDI", 0.20)])
    out = r.resolve([[DET]], [_crop()])[0][0]
    assert out["brand"] == "BMW"
    assert out["resolution_source"] == "clip_retrieval"
    assert "unresolved_reason" not in out


def test_sole_brand_above_floor_is_trusted():
    # One unique brand over the floor: treat as a genuine match, not a rejection.
    r = _resolver([("BMW", 0.25), ("BMW", 0.24)])
    out = r.resolve([[DET]], [_crop()])[0][0]
    assert out["brand"] == "BMW"
