"""
Tests for the open-set brand identification module (escalation Task 3).

These tests must never touch the network: they assert the cost-gated candidate
selection, the deterministic name derivation (no LLM guess), and the
fail-closed behavior when no reverse-image-search backend / logo.dev key is
available. A candidate name is only ever lower-trust evidence — never a
presented detection — until logo.dev validates it.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402

from src.openset import (  # noqa: E402
    OpenSetBrandIdentifier,
    ReverseImageResult,
    _brand_likeness,
    _derive_candidate_name,
)

_CROP_DIR = str(Path(__file__).parent.parent / "static" / "openset_crops")

# Test-local filter vocabulary mirroring config open_set.generic_*_filter.
# The shipped module takes these from config and never hardcodes them.
_GENERIC_WORDS = ["logo", "tshirt", "t-shirt", "футболка", "логотип"]
_GENERIC_DOMAINS = ["google.com", "facebook.com"]


def _identifier(backend, min_conf: float = 0.01) -> OpenSetBrandIdentifier:
    return OpenSetBrandIdentifier(
        backend=backend,
        min_logo_confidence=min_conf,
        max_candidates_per_video=5,
        crop_cache_dir=_CROP_DIR,
        generic_tag_filter=_GENERIC_WORDS,
        generic_domain_filter=_GENERIC_DOMAINS,
        logodev_timeout=2.0,
    )


def _tiny_video() -> str:
    """Build a real, readable 1-second video for candidate-frame extraction."""
    import cv2

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    writer = cv2.VideoWriter(
        tmp.name, cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (64, 64)
    )
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[0:20, 0:20] = (255, 255, 255)
    for _ in range(2):
        writer.write(frame)
    writer.release()
    return tmp.name


class _FakeBackend:
    name = "fake_backend"
    available = True

    def __init__(self, results=None):
        self._results = results or []

    def search_crop(self, crop):
        return self._results


def test_derive_candidate_uses_real_wordmark_tags_not_guess():
    """Candidate name comes only from real engine-read tags/URLs."""
    results = [
        ReverseImageResult(title="best express", url="", source="yandex_cbir_tags"),
        ReverseImageResult(title="express china", url="", source="yandex_cbir_tags"),
        ReverseImageResult(title="tshirt", url="", source="yandex_cbir_tags"),
    ]
    assert _derive_candidate_name(results, _GENERIC_WORDS, _GENERIC_DOMAINS) == "BEST EXPRESS"


def test_derive_candidate_rejects_generic_tags():
    """Generic descriptors (t-shirt, logo) never become a brand candidate."""
    results = [
        ReverseImageResult(title="футболка", url="", source="yandex_cbir_tags"),
        ReverseImageResult(title="логотип", url="", source="yandex_cbir_tags"),
    ]
    assert _derive_candidate_name(results, _GENERIC_WORDS, _GENERIC_DOMAINS) is None


def test_derive_candidate_rejects_generic_domains():
    """A page from a generic/portal domain never becomes the candidate name."""
    results = [
        ReverseImageResult(
            title="", url="https://www.facebook.com/watch/?v=1", source="yandex_cbir_similar",
        ),
    ]
    assert _derive_candidate_name(results, _GENERIC_WORDS, _GENERIC_DOMAINS) is None


def test_brand_likeness_orders_real_wordmarks_over_generic():
    assert _brand_likeness("best express", _GENERIC_WORDS) > _brand_likeness("tshirt", _GENERIC_WORDS)
    assert _brand_likeness("express china", _GENERIC_WORDS) > _brand_likeness("logo", _GENERIC_WORDS)


def test_identify_fails_closed_without_backend():
    """No runnable backend -> available=False, zero candidates surfaced."""
    identifier = _identifier(backend=_UnavailableBackend())
    result = {"layer1": {"logo_detections": [[
        {"bbox": [0, 0, 10, 10], "confidence": 0.99},
    ]]}}
    out = identifier.identify(result, "/nonexistent.mp4")
    assert out["available"] is False
    assert out["candidates"] == []


def test_identify_fails_closed_without_configured_gate():
    """An identifier with no real threshold/cache config is unusable."""
    import pytest

    with pytest.raises(ValueError):
        OpenSetBrandIdentifier(
            backend=_UnavailableBackend(),
            min_logo_confidence=0.0,
            max_candidates_per_video=5,
            crop_cache_dir=_CROP_DIR,
            generic_tag_filter=_GENERIC_WORDS,
            generic_domain_filter=_GENERIC_DOMAINS,
            logodev_timeout=2.0,
        )


def test_identify_surfaces_candidate_without_verification():
    """A real search hit yields a candidate, but logo.dev (no key) keeps it
    lower-trust — never 'verified', never outreach-eligible."""
    identifier = _identifier(backend=_FakeBackend([
        ReverseImageResult(title="best express", url="", source="yandex_cbir_tags"),
    ]))
    identifier._logodev_client = _FakeLogodev()
    result = {"layer1": {"logo_detections": [[
        {"bbox": [0, 0, 20, 20], "confidence": 0.99},
    ]]}}
    video = _tiny_video()
    try:
        out = identifier.identify(result, video)
    finally:
        Path(video).unlink(missing_ok=True)
    assert out["available"] is True
    assert len(out["candidates"]) == 1
    cand = out["candidates"][0]
    assert cand["candidate_name"] == "BEST EXPRESS"
    assert cand["status"] == "candidate_unverified"
    assert cand["logo_dev_validation"]["status"] == "unavailable"
    assert cand["logo_dev_validation"]["status"] != "verified"


class _UnavailableBackend:
    name = "fake_unavailable"
    available = False

    def search_crop(self, crop):
        raise AssertionError("must not be called when unavailable")


class _FakeLogodev:
    def validate_brand(self, brand):
        return {"status": "unavailable", "brand": brand, "domain": None}
