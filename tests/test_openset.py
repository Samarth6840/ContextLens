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
    GeminiGroundedBackend,
    OpenSetBrandIdentifier,
    ReverseImageResult,
    _brand_likeness,
    _derive_candidate_name,
    _is_void_name,
    create_backend,
)

_CROP_DIR = str(Path(__file__).parent.parent / "static" / "openset_crops")

# Test-local filter vocabulary mirroring config open_set.generic_*_filter.
# The shipped module takes these from config and never hardcodes them.
_GENERIC_WORDS = ["logo", "tshirt", "t-shirt", "футболка", "логотип"]
_GENERIC_DOMAINS = ["google.com", "facebook.com"]


def _identifier(
    backend,
    min_conf: float = 0.01,
    min_crop_area: float = 100.0,
    max_crop_aspect: float = 5.0,
    consent_upload: bool = True,
) -> OpenSetBrandIdentifier:
    return OpenSetBrandIdentifier(
        backend=backend,
        consent_upload=consent_upload,
        min_logo_confidence=min_conf,
        min_crop_area=min_crop_area,
        max_crop_aspect=max_crop_aspect,
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
            consent_upload=True,
            min_logo_confidence=0.0,
            min_crop_area=0.0,
            max_crop_aspect=0.0,
            max_candidates_per_video=5,
            crop_cache_dir=_CROP_DIR,
            generic_tag_filter=_GENERIC_WORDS,
            generic_domain_filter=_GENERIC_DOMAINS,
            logodev_timeout=2.0,
        )


def test_consent_false_never_uploads_crop():
    """consent_upload=False must keep every crop off the wire.

    The backend raises if touched, so any call fails the test outright. Local
    work still happens: the crop is cached and the reason is surfaced.
    """
    class _TripwireBackend:
        name = "tripwire"
        available = True

        def search_crop(self, crop):
            raise AssertionError("crop was uploaded without consent")

    identifier = _identifier(backend=_TripwireBackend(), consent_upload=False)
    identifier._logodev_client = _FakeLogodev()
    result = {"layer1": {"logo_detections": [[
        {"bbox": [0, 0, 20, 20], "confidence": 0.99},
    ]]}}
    video = _tiny_video()
    try:
        out = identifier.identify(result, video)
    finally:
        Path(video).unlink(missing_ok=True)
    assert out["consent_upload"] is False
    # A backend WAS available — so this must not read as a missing dependency.
    assert out["available"] is True
    assert "consent_upload" in out["reason"]
    assert len(out["candidates"]) == 1
    cand = out["candidates"][0]
    assert cand["status"] == "consent_withheld"
    assert cand["search_results"] == []
    assert cand["candidate_name"] is None
    # Local evidence trail survives the withheld upload.
    assert cand["crop_id"]


def test_consent_must_be_a_real_bool():
    """Omitting consent must fail loudly rather than default to leaking."""
    import pytest

    for bad in (None, "true", 1, 0):
        with pytest.raises(ValueError):
            OpenSetBrandIdentifier(
                backend=_UnavailableBackend(),
                consent_upload=bad,
                min_logo_confidence=0.3,
                min_crop_area=3000.0,
                max_crop_aspect=3.0,
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


class _FakeGeminiResponse:
    """Minimal requests-style stand-in with a real Google API body shape."""

    status_code = 200

    def __init__(self, text=""):
        self._text = text

    def json(self):
        return {
            "candidates": [{
                "content": {"parts": [{"text": self._text}]},
                "groundingMetadata": {
                    "groundingChunks": [
                        {"web": {"uri": "https://beste.com/", "title": "BestE Home"}},
                        {"web": {"uri": "https://wiki.org/BestE", "title": "BestE — Wiki"}},
                    ],
                },
            }]
        }


def _gemini_backend(monkeypatch) -> GeminiGroundedBackend:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    return GeminiGroundedBackend()


def test_gemini_backend_registered_in_factory(monkeypatch):
    """create_backend resolves the gemini_grounded backend."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    backend = create_backend("gemini_grounded")
    assert backend.name == "gemini_grounded"
    assert backend.available is True


def test_gemini_backend_unavailable_without_key(monkeypatch):
    """No GEMINI_API_KEY -> fail closed, never a fabricated name."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    backend = GeminiGroundedBackend()
    assert backend.available is False


def test_gemini_search_parses_grounded_chunks_into_real_sources(monkeypatch):
    """Real grounding chunks become citable ReverseImageResults."""
    import requests as real_requests

    backend = _gemini_backend(monkeypatch)
    crop = np.zeros((16, 16, 3), dtype=np.uint8)
    captured = {}

    def fake_post(url, params=None, json=None, timeout=None):
        captured["payload"] = json
        assert "google_search" in [t for t in [json["tools"][0]]][0]
        return _FakeGeminiResponse(
            "This is BestE logistics. BRAND: BestE\nSources confirm it."
        )

    monkeypatch.setattr(real_requests, "post", fake_post)
    results = backend.search_crop(crop)
    urls = [r.url for r in results if r.url]
    assert "https://beste.com/" in urls
    assert "https://wiki.org/BestE" in urls
    web = [r for r in results if r.source == "gemini_grounded_web"]
    assert all(r.url for r in web)
    tags = [r for r in results if r.source == "gemini_grounded_tags"]
    assert tags and tags[0].title == "BestE"
    assert any(r.source == "gemini_grounded_desc" for r in results)


def test_gemini_identification_line_parses_brand_format():
    assert GeminiGroundedBackend._identification_line(
        "It's a courier firm. BRAND: BestE"
    ) == "BestE"
    # No BRAND: line means no brand claim. This used to return "bestey express"
    # verbatim, i.e. an unparseable model reply became a ReverseImageResult
    # title that then scored as a brand. The description is still surfaced
    # separately as gemini_grounded_desc, so nothing is lost.
    assert GeminiGroundedBackend._identification_line("bestey express") == ""
    assert GeminiGroundedBackend._identification_line("  ") == ""
    assert GeminiGroundedBackend._identification_line("BRAND:  FastShip  ") == "FastShip"
    # A period inside the name must not truncate it.
    assert GeminiGroundedBackend._identification_line("BRAND: Dr. Pepper") == "Dr. Pepper"


def test_gemini_prose_response_never_becomes_a_brand_result():
    """Item 20: verbose prose used to be returned as a brand candidate."""
    prose = ("The image appears to show a television screen displaying an "
             "advertisement for a well-known consumer electronics brand.")
    assert GeminiGroundedBackend._identification_line(prose) == ""


def test_gemini_tags_feed_candidate_name_deterministically(monkeypatch):
    """Grounded Gemini tags flow into the same deterministic name pipeline."""
    import requests as real_requests

    backend = _gemini_backend(monkeypatch)

    def fake_post(url, params=None, json=None, timeout=None):
        return _FakeGeminiResponse("A courier service logo. BRAND: BestE")

    monkeypatch.setattr(real_requests, "post", fake_post)
    crop = np.zeros((16, 16, 3), dtype=np.uint8)
    results = backend.search_crop(crop)
    name = _derive_candidate_name(results, _GENERIC_WORDS, _GENERIC_DOMAINS)
    assert name == "BESTE"


def test_derive_candidate_never_surfaces_void_names():
    """'unknown'/'UNRESOLVED' engine output is a failed identification, not a
    brand label — surfacing it would be a fabricated name."""
    for void in ("unknown", "UNRESOLVED", "none", "n/a", "Unknown brand"):
        results = [
            ReverseImageResult(title=void, url="", source="gemini_grounded_tags"),
            ReverseImageResult(title="logo", url="", source="yandex_cbir_tags"),
        ]
        assert _derive_candidate_name(results, _GENERIC_WORDS, _GENERIC_DOMAINS) is None


def test_derive_candidate_never_surfaces_negated_identifications():
    """'NOT IDENTIFIABLE' is a failed lookup, not a brand. Regression: it was
    reaching logo.dev and coming back verified as identifiable.ca."""
    for void in (
        "NOT IDENTIFIABLE",
        "not identifiable",
        "The brand is not identifiable",
        "cannot identify",
        "unidentifiable",
        "no logo detected",
    ):
        results = [
            ReverseImageResult(title=void, url="", source="gemini_grounded_tags"),
            ReverseImageResult(title="logo", url="", source="yandex_cbir_tags"),
        ]
        assert _derive_candidate_name(results, _GENERIC_WORDS, _GENERIC_DOMAINS) is None


def test_void_name_matching_keeps_real_not_prefixed_brands():
    """Phrase matching must not swallow real brands that start with 'not'."""
    for real in ("Notion", "Nottington", "Nokia"):
        assert not _is_void_name(real)


def test_identify_records_banner_shape_rejection():
    """A wide-analogue bbox (full-width strap) is rejected pre-frame-read and
    the reason is recorded for the UI's open-set status states."""
    identifier = _identifier(backend=_UnavailableBackend(), max_crop_aspect=3.0)
    result = {"layer1": {"logo_detections": [[
        {"bbox": [0, 0, 40, 8], "confidence": 0.99},
    ]]}}
    out = identifier.identify(result, "/nonexistent.mp4")
    assert out["candidates"] == []
    assert out["rejected"][0]["reason"] == "banner_shape"
    assert out["rejected"][0]["aspect"] >= 3.0
    assert out["rejected"][0]["frame_index"] == 0


def test_identify_records_too_small_rejection():
    """A crop below the minimum searchable area is rejected with a reason."""
    identifier = _identifier(
        backend=_UnavailableBackend(), min_crop_area=100.0, max_crop_aspect=5.0
    )
    result = {"layer1": {"logo_detections": [[
        {"bbox": [0, 0, 8, 8], "confidence": 0.99},
    ]]}}
    video = _tiny_video()
    try:
        out = identifier.identify(result, video)
    finally:
        Path(video).unlink(missing_ok=True)
    assert out["candidates"] == []
    assert out["rejected"][0]["reason"] == "too_small"
    assert out["rejected"][0]["width"] * out["rejected"][0]["height"] < 100.0


def test_identify_void_search_result_is_no_name_not_a_brand():
    """When search runs but only yields void/generic output, the crop is
    recorded as a real search with no confident match — never a named brand."""
    identifier = _identifier(backend=_FakeBackend([
        ReverseImageResult(title="unknown", url="", source="yandex_cbir_tags"),
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
    assert len(out["candidates"]) == 1
    cand = out["candidates"][0]
    assert cand["candidate_name"] is None
    assert cand["status"] == "candidate_no_name"
    assert cand["search_results"]
