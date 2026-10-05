"""_crop_hash must dedupe NEAR-identical crops, not just byte-identical ones.

It hashed raw grayscale bytes, so two frames of the same logo — the detection
box shifting a pixel or two — produced different hashes and nothing ever
deduped, so every frame of a persistent logo spent its own search budget.

Both halves are pinned: near-identical collapses, different logo survives.
The margins are measured, not eyeballed — see the constants in openset.py.
"""

import cv2
import numpy as np
import pytest

from src.openset import (
    CROP_HASH_BITS,
    CROP_HASH_MAX_DISTANCE,
    _crop_hash,
    _crop_hash_distance,
)

BRANDS = ["SAMSUNG", "NIKE", "ADIDAS", "SONY", "APPLE", "TOYOTA", "BMW", "PEPSI"]


def _wordmark(text: str, scale: float = 1.0):
    img = np.zeros((48, 200, 3), dtype=np.uint8)
    cv2.putText(
        img, text, (6, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9 * scale, (235, 235, 235), 2, cv2.LINE_AA
    )
    return img


def _shift(img, dx: int, dy: int = 0):
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]))


def _noisy(img, sigma: float = 8.0):
    rng = np.random.default_rng(0)
    return np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(
        np.uint8
    )


def test_near_identical_crops_dedupe():
    """Same logo across frames: box shifted, sensor noise, small rescale."""
    base = _crop_hash(_wordmark("SAMSUNG"))
    variants = [(_shift(_wordmark("SAMSUNG"), dx, dy), "") for dx, dy in
                [(1, 0), (2, 1), (3, -1), (4, 2), (5, 3)]]
    variants += [(_noisy(_shift(_wordmark("SAMSUNG"), dx, dy)), "") for dx, dy in
                 [(1, 0), (2, 1), (3, -1)]]
    variants += [(_wordmark("SAMSUNG", s), "") for s in (0.9, 0.95, 1.05)]

    for crop, _ in variants:
        d = _crop_hash_distance(base, _crop_hash(crop))
        assert d <= CROP_HASH_MAX_DISTANCE, (
            f"same wordmark variant hashed to distance {d} "
            f"(> {CROP_HASH_MAX_DISTANCE} of {CROP_HASH_BITS} bits); dedup "
            "regressed and every frame of a persistent logo will re-search"
        )


@pytest.mark.parametrize("other", [b for b in BRANDS if b != "SAMSUNG"])
def test_different_logos_stay_distinct(other):
    d = _crop_hash_distance(_crop_hash(_wordmark("SAMSUNG")), _crop_hash(_wordmark(other)))
    assert d > CROP_HASH_MAX_DISTANCE, (
        f"SAMSUNG vs {other} collided at distance {d} — a genuinely "
        "different candidate crop is being silently dropped as a duplicate"
    )


def test_threshold_sits_between_the_two_distributions():
    """Guard the constant: same-logo noise must stay well under the threshold,
    and the closest cross-logo pair well over it."""
    base = _crop_hash(_wordmark("SAMSUNG"))
    same = max(
        [_crop_hash_distance(base, _crop_hash(_shift(_wordmark("SAMSUNG"), dx, dy)))
         for dx, dy in [(1, 0), (2, 1), (3, -1), (4, 2)]]
        + [_crop_hash_distance(_crop_hash(_noisy(_wordmark("SAMSUNG"))), base)]
        + [_crop_hash_distance(base, _crop_hash(_wordmark("SAMSUNG", s)))
           for s in (0.9, 1.05)]
    )
    cross = min(
        _crop_hash_distance(_crop_hash(_wordmark(a)), _crop_hash(_wordmark(b)))
        for i, a in enumerate(BRANDS) for b in BRANDS[i + 1:]
    )
    assert same < CROP_HASH_MAX_DISTANCE < cross, (
        f"threshold {CROP_HASH_MAX_DISTANCE} is not in the gap: same-logo "
        f"reaches {same}, closest different pair is {cross} — retune "
        "CROP_HASH_MAX_DISTANCE or the hashing, don't move the goalposts"
    )


def test_hash_is_stable_across_calls():
    a = _wordmark("SAMSUNG")
    assert _crop_hash(a) == _crop_hash(a.copy())
    assert len(_crop_hash(a)) == CROP_HASH_BITS // 4
