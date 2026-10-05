"""Detector saturation reporting for the open-set funnel.

The defect these guard: when the detector is saturated, its confidence
distribution is flat, the confidence gate rejects nearly everything, and the
dashboard reports a funnel of zeros. Read on its own that looks like the
reverse-image resolver failing to identify brands. It is not - the gates are
behaving as documented, they are just being fed a noise floor.

So saturation is measured where the detector's proposals enter the pipeline,
before any gate can discard one, and reported alongside the funnel. Nothing
here drops, reorders or caps a candidate.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.openset import OpenSetBrandIdentifier, ReverseImageResult  # noqa: E402

_CROP_DIR = str(Path(__file__).parent.parent / "static" / "openset_crops")


class _DeadBackend:
    """Never called: these tests exercise reporting, not identification."""

    name = "dead"

    def available(self) -> bool:
        return False

    def search_crop(self, crop):
        raise AssertionError("diagnostics tests must not call a backend")


def _identifier(saturated_at: int = 30) -> OpenSetBrandIdentifier:
    return OpenSetBrandIdentifier(
        backend=_DeadBackend(),
        consent_upload=True,
        min_logo_confidence=0.30,
        min_crop_area=3000.0,
        max_crop_aspect=3.0,
        max_candidates_per_video=30,
        crop_cache_dir=_CROP_DIR,
        generic_tag_filter=[],
        generic_domain_filter=[],
        logodev_timeout=2.0,
        saturated_proposals_per_frame=saturated_at,
    )


def _result(det_counts, conf=0.148, bbox=(10, 10, 60, 60)):
    """A finished-job layer1 with the given proposals per frame."""
    return {
        "layer1": {
            "logo_detections": [
                [
                    {"bbox": list(bbox), "confidence": conf, "class_name": "logo"}
                    for _ in range(n)
                ]
                for n in det_counts
            ]
        }
    }


def test_saturated_frame_is_reported():
    # The measured failure: 112 proposals in a frame, all near-identical low
    # confidence. A sparse frame must not be flagged the same way.
    ident = _identifier(saturated_at=30)
    out = ident.identify(_result([112, 4, 0]), "/nonexistent.mp4")
    dd = out["detector_diagnostics"]
    assert dd["max_proposals_in_a_frame"] == 112
    assert dd["median_proposals_in_a_frame"] == 4
    assert dd["saturated_frames"] == [0]
    assert dd["saturated_frame_count"] == 1


def test_proposals_counted_before_any_gate():
    # Every proposal is counted even though the 0.30 confidence gate rejects
    # all of them. If this were measured after the gate, saturation would read
    # as zero and the whole point of the report would be inverted.
    ident = _identifier(saturated_at=30)
    out = ident.identify(_result([50, 50]), "/nonexistent.mp4")
    dd = out["detector_diagnostics"]
    assert dd["proposals_total"] == 100
    assert dd["saturated_frame_count"] == 2
    assert out["skipped_counts"]["below_min_confidence"] == 100
    assert not out["candidates"]


def test_no_behaviour_change_to_candidates():
    # A video under the saturation threshold must produce exactly the same
    # candidates and rejections as before the report existed. The report is
    # observation; if it changed the funnel it would be a bug, not a feature.
    ident = _identifier(saturated_at=10_000)
    out = ident.identify(_result([2, 2], conf=0.9), "/nonexistent.mp4")
    assert out["candidates"] == []
    assert len(out["rejected"]) == 4
    assert all(r["reason"] == "unreadable_frame" for r in out["rejected"])
    assert out["detector_diagnostics"]["saturated_frame_count"] == 0


def test_saturation_threshold_is_validated():
    # A zero threshold would flag every non-empty frame and make the warning
    # permanent noise, which is the same failure as not reporting at all.
    try:
        _identifier(saturated_at=0)
    except ValueError as exc:
        assert "saturated_proposals_per_frame" in str(exc)
    else:
        raise AssertionError("saturated_proposals_per_frame <= 0 must be rejected")


def test_reason_counts_cover_every_drop_path():
    # The taxonomy is only useful if no drop path is silent. A detection with
    # no bbox used to vanish entirely. banner_shape and too_small are already
    # covered against a real video in test_openset.py; those two need geometry
    # that a missing video file cannot supply, so they are not re-litigated here.
    ident = _identifier()
    res = {
        "layer1": {
            "logo_detections": [
                [{"bbox": None, "confidence": 0.9, "class_name": "logo"}],
                [{"bbox": [0, 0, 200, 2], "confidence": 0.9, "class_name": "logo"}],
                [{"bbox": [0, 0, 10, 10], "confidence": 0.9, "class_name": "logo"}],
            ]
        }
    }
    out = ident.identify(res, "/nonexistent.mp4")
    reasons = out["rejected_reason_counts"]
    assert reasons["no_bbox"] == 1
    assert reasons["banner_shape"] == 1
    assert reasons["unreadable_frame"] == 1
    assert sum(reasons.values()) == 3
