"""
Tests for the embedding-version guard on persisted vector indexes.

The DINOv2->DINOv3 upgrade kept the same output dim (768), so a stale index
built under one model would silently return plausible-but-wrong nearest
neighbors against a live model from another space. These tests assert the
guard FAILS LOUDLY on a version mismatch (not just dimension mismatch).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.layer1.product_index import ProductEmbeddingIndex
from src.layer1.visual_embeddings import assert_embedding_version

DIM = 768


@pytest.fixture()
def ref_dir(tmp_path):
    from PIL import Image

    brand_dir = tmp_path / "refs" / "NIKE"
    brand_dir.mkdir(parents=True)
    Image.new("RGB", (16, 16), (30, 30, 30)).save(brand_dir / "n.png")
    return str(tmp_path / "refs")


def _make_extractor(version_tag):
    """Minimal FakeExtractor standing in for VisualEmbeddingExtractor.

    Contract: `embedding_version` names the embedding space; `extract_batch`
    returns unit-normalized vectors of the product-index dim.
    """

    class FakeExtractor:
        def __init__(self):
            self.embedding_version = version_tag
            self.output_dim = DIM
            self._rng = np.random.RandomState(0)

        def extract_batch(self, frames, batch_size=8):
            n = len(frames)
            embs = self._rng.randn(n, self.output_dim).astype(np.float32)
            embs /= np.linalg.norm(embs, axis=1, keepdims=True)
            return embs.astype(np.float32)

    return FakeExtractor()


def _rand_vec():
    rng = np.random.RandomState(1)
    v = rng.randn(1, DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


# ── Shared guard helper ────────────────────────────────────


def test_assert_embedding_version_mismatch_raises():
    with pytest.raises(ValueError, match="version mismatch"):
        assert_embedding_version("embed:dinov3", "embed:dinov2", "ProductIndex")


def test_assert_embedding_version_same_version_ok():
    # Same space passes.
    assert_embedding_version("embed:dinov3", "embed:dinov3", "ProductIndex")


def test_assert_embedding_version_missing_tag_is_noop():
    # Legacy index (no version tag) plus a live tag cannot assert -> skip.
    assert_embedding_version(None, "embed:dinov3", "ProductIndex")
    # Live query without a tag but tagged index -> skip.
    assert_embedding_version("embed:dinov3", None, "ProductIndex")
    assert_embedding_version(None, None, "ProductIndex")


# ── ProductEmbeddingIndex integration ──────────────────────


def test_product_index_build_stamps_version(ref_dir):
    extractor = _make_extractor("embed:dinov3-vitb16")
    index = ProductEmbeddingIndex(ref_dir)
    assert index.build(extractor, batch_size=8) >= 1
    assert index.embedding_version == "embed:dinov3-vitb16"

    matches = index.query(
        _rand_vec(), [0], top_k=1, similarity_threshold=0.0,
        embedding_version="embed:dinov3-vitb16",
    )
    assert isinstance(matches, list)


def test_product_index_query_rejects_mismatch(ref_dir):
    build_extractor = _make_extractor("embed:dinov3-vitb16")
    query_extractor = _make_extractor("embed:dinov2-base")

    index = ProductEmbeddingIndex(ref_dir)
    index.build(build_extractor, batch_size=8)

    with pytest.raises(ValueError, match="version mismatch"):
        index.query(
            query_extractor.extract_batch([_rand_vec()[0]]),
            [0], top_k=1, similarity_threshold=0.0,
            embedding_version="embed:dinov2-base",
        )

    # Omitting the version stays backward compatible (no assertion).
    assert isinstance(
        index.query(
            query_extractor.extract_batch([_rand_vec()[0]]),
            [0], top_k=1, similarity_threshold=0.0,
        ),
        list,
    )

# ── Resolver acceptance canary ─────────────────────────────


def test_resolver_acceptance_canary_happy_path():
    from src.pipeline import resolver_acceptance_canary

    dets = [
        [
            {"brand": "NIKE", "bbox": [0, 0, 1, 1]},
            {"bbox": [2, 2, 3, 3]},              # unresolved (no brand)
        ],
        [
            {"brand": "ADIDAS", "bbox": [0, 0, 1, 1]},
        ],
    ]
    out = resolver_acceptance_canary(dets)
    assert out["total_logo_detections"] == 3
    assert out["resolved"] == 2
    assert out["unresolved"] == 1
    assert out["acceptance_rate"] == pytest.approx(2 / 3, abs=1e-4)


def test_resolver_acceptance_canary_excludes_unknown_sentinel():
    from src.layer2.brand_resolver import UNKNOWN_BRAND
    from src.pipeline import resolver_acceptance_canary

    dets = [
        [
            {"brand": UNKNOWN_BRAND, "bbox": [0, 0, 1, 1]},
            {"brand": "SONY", "bbox": [2, 2, 3, 3]},
        ],
    ]
    out = resolver_acceptance_canary(dets)
    assert out["resolved"] == 1
    assert out["unresolved"] == 1


def test_resolver_acceptance_canary_empty_input():
    from src.pipeline import resolver_acceptance_canary

    out = resolver_acceptance_canary([])
    assert out["total_logo_detections"] == 0
    assert out["resolved"] == 0
    assert out["unresolved"] == 0
    assert out["acceptance_rate"] == 0.0
