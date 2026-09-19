"""Tests for TemporalBrandSmoother (Phase 1 stability fix).

A static on-screen overlay (wordmark held for many frames) gets region-proposal
confidence that flaps around the cut-off, so identical pixels resolve a brand
one frame and 'text logo'/nothing the next. The smoother back-fills unresolved
logo boxes from a solid, persistent resolved brand on a spatially-overlapping
region, without ever weakening an already-resolved detection.
"""

from src.layer2.brand_resolver import TemporalBrandSmoother


def _det(bbox, brand=None, conf=0.0, cls_name="text logo"):
    d = {
        "bbox": list(bbox),
        "confidence": conf,
        "class_name": cls_name if brand is None else brand,
        "text_prompt": "text logo",
    }
    if brand:
        d["brand"] = brand
        d["resolution_source"] = "ocr"
    return d


# Stable wordmark region around x=[40,120], y=[30,60].
WORDMARK = [40.0, 30.0, 120.0, 60.0]


def test_gap_frames_backfilled_from_resolved_neighbours():
    # Frames 0 and 3 resolve SAMSUNG on the wordmark; frames 1,2 are the same
    # pixels but flickered to an unresolved generic box (confidence flapping).
    logos = [
        [_det(WORDMARK, "SAMSUNG", 0.50)],
        [_det(WORDMARK, None, 0.13)],
        [_det(WORDMARK, None, 0.13)],
        [_det(WORDMARK, "SAMSUNG", 0.50)],
    ]
    out = TemporalBrandSmoother(window=2, min_votes=2, min_iou=0.3).smooth(logos)
    # Frames 1 and 2 should inherit SAMSUNG, marked as temporal smoothing.
    assert out[1][0]["brand"] == "SAMSUNG"
    assert out[2][0]["brand"] == "SAMSUNG"
    assert out[1][0]["resolution_source"] == "temporal_smoothing"
    assert out[1][0]["smoothed_votes"] >= 2
    # Already-resolved frames are untouched by the smoother.
    assert out[0][0]["brand"] == "SAMSUNG"
    assert out[0][0]["resolution_source"] == "ocr"


def test_already_resolved_brand_is_never_weakened():
    # Frame 1 already resolves APPLE independently; a SAMSUNG region nearby must
    # not overwrite it even if it spatially overlaps.
    logos = [
        [_det(WORDMARK, "SAMSUNG", 0.50)],
        [_det(WORDMARK, "APPLE", 0.55)],
    ]
    out = TemporalBrandSmoother(window=2, min_votes=2, min_iou=0.3).smooth(logos)
    assert out[1][0]["brand"] == "APPLE"


def test_single_resolved_frame_does_not_backfill():
    # Only one resolved frame — not enough votes, so unresolved neighbours stay
    # unresolved (no guessing with insufficient evidence).
    logos = [
        [_det(WORDMARK, None, 0.10)],
        [_det(WORDMARK, "SAMSUNG", 0.50)],
        [_det(WORDMARK, None, 0.10)],
    ]
    out = TemporalBrandSmoother(window=2, min_votes=2, min_iou=0.3).smooth(logos)
    assert out[0][0].get("brand") is None
    assert out[2][0].get("brand") is None
    assert out[1][0]["brand"] == "SAMSUNG"


def test_non_overlapping_boxes_not_merged():
    # Two distinct spatial regions (wordmark + chip card) — smoothing must not
    # cross-attribute a brand from one region to the other.
    chip = [200.0, 30.0, 420.0, 60.0]
    logos = [
        [_det(WORDMARK, "SAMSUNG", 0.50), _det(chip, None, 0.12)],
        [_det(WORDMARK, "SAMSUNG", 0.50), _det(chip, None, 0.12)],
        [_det(WORDMARK, "SAMSUNG", 0.50), _det(chip, None, 0.12)],
    ]
    out = TemporalBrandSmoother(window=2, min_votes=2, min_iou=0.3).smooth(logos)
    # The chip region never overlaps the wordmark, so no brand leaks into it.
    for frame in out:
        for det in frame:
            if det["bbox"] == list(chip):
                assert det.get("brand") is None


def test_votes_need_supporting_frames():
    # SAMSUNG resolves strongly on frames 0 and 1 only; far-apart resolved frame
    # 5 shares no overlap with frame 2 in-window, so frame 2 stays unresolved.
    logos = [
        [_det(WORDMARK, "SAMSUNG", 0.40)],
        [_det(WORDMARK, "SAMSUNG", 0.40)],
        [_det(WORDMARK, None, 0.14)],
        [_det(WORDMARK, None, 0.14)],
        [_det(WORDMARK, None, 0.14)],
        [_det(WORDMARK, "SAMSUNG", 0.40)],
    ]
    out = TemporalBrandSmoother(window=1, min_votes=2, min_iou=0.3).smooth(logos)
    # window=1: frame 2 is outside frames 0/1's window, so not back-filled.
    assert out[2][0].get("brand") is None
