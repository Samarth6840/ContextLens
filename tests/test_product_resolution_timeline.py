"""Tests for timed product resolutions (speech + OCR) and scene captions.

Covers the two dashboard features added in this session:

  * PLAY-@-time clip chips for product resolutions:
      - pipeline `_timestamp_product_resolutions` (real speech timestamps
        propagated from timed brand mentions)
      - pipeline `_collect_ocr_product_resolutions` (on-screen product names,
        cache-only, no network / Qwen)
      - server `_build_product_resolutions` (frame index -> real timestamp)
  * Deterministic scene caption line: server `_scene_caption`.

All tests are offline: the resolver is faked and no network or model is hit.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.brand_catalog import match_brand  # noqa: E402
from src.pipeline import (  # noqa: E402
    _collect_ocr_product_resolutions,
    _timestamp_product_resolutions,
)
from server import _build_product_resolutions, _scene_caption  # noqa: E402


class _FakeResolver:
    """Stand-in that resolves product spans to brands via a dict lookup."""

    def __init__(self, answer):
        self.answer = answer  # {span: {brand, tier, source, conf, product_span}}
        self.calls = []

    def resolve(self, text, live=True):
        self.calls.append({"text": text, "live": live})
        hits = []
        for span, r in self.answer.items():
            if span.lower() in text.lower() and r.get("brand"):
                hits.append({
                    "span": span,
                    "normalized": span,
                    "brand": r["brand"],
                    "resolution_tier": r.get("tier"),
                    "source": r.get("source"),
                    "confidence": r.get("conf", 0.8),
                    "resolution_quality": r.get("quality"),
                    "product_span": r.get("product_span"),
                    "meta": {},
                })
        return hits


# ── pipeline: speech timestamps on product resolutions ───────


def test_timestamp_product_resolutions_propagates_real_times():
    resolutions = [{
        "brand": "APPLE",
        "span": "Mac Mini",
        "product_span": "Mac Mini",
        "resolution_tier": "T2",
        "source": "wikidata",
    }]
    mentions = [{
        "brand": "APPLE",
        "product_span": "Mac Mini",
        "start_time": 12.5,
        "end_time": 14.0,
        "frame_index": 40,
    }]
    out = _timestamp_product_resolutions(resolutions, mentions)
    assert out[0]["start_time"] == 12.5
    assert out[0]["end_time"] == 14.0
    assert out[0]["frame_index"] == 40
    assert out[0]["brand"] == "APPLE"  # original fields preserved


def test_timestamp_product_resolutions_matching_is_brand_plus_span():
    resolutions = [{
        "brand": "SAMSUNG",
        "span": "Z Fold 8 Ultra",
        "product_span": "Z Fold 8 Ultra",
    }]
    mentions = [{
        "brand": "SAMSUNG",
        "product_span": "Washing Machine",  # different product: must not match
        "start_time": 3.0,
        "end_time": 4.0,
    }]
    out = _timestamp_product_resolutions(resolutions, mentions)
    assert out[0].get("start_time") is None


def test_timestamp_product_resolutions_unmatched_keeps_no_time():
    resolutions = [{"brand": "XIAOMI", "span": "Buds Pro"}]
    out = _timestamp_product_resolutions(resolutions, [])
    assert out[0].get("start_time") is None
    assert out[0]["brand"] == "XIAOMI"


def test_timestamp_product_resolutions_never_mutates_input():
    resolutions = [{"brand": "APPLE", "span": "Mac Mini"}]
    mentions = [{"brand": "APPLE", "product_span": "Mac Mini",
                 "start_time": 1.0, "end_time": 2.0}]
    _timestamp_product_resolutions(resolutions, mentions)
    assert "start_time" not in resolutions[0]


# ── pipeline: OCR product resolutions (cache-only) ───────────


def test_collect_ocr_resolves_products_but_skips_direct_brands():
    resolver = _FakeResolver({
        "Mac Mini": {"brand": "APPLE", "tier": "T2", "source": "wikidata",
                     "product_span": "Mac Mini"},
    })
    ocr = [
        [{"text": "ROLEX"}],           # direct brand name -> skipped
        [{"text": "Mac Mini review"}],  # product -> resolved
        [],
    ]
    out = _collect_ocr_product_resolutions(ocr, resolver, match_brand)
    assert len(out) == 1
    r = out[0]
    assert r["brand"] == "APPLE"
    assert r["mode"] == "ocr"
    assert r["frame_index"] == 1
    assert r["frames"] == [1]
    assert all(call["live"] is False for call in resolver.calls)


def test_collect_ocr_dedupes_by_brand_and_span_across_frames():
    resolver = _FakeResolver({
        "Z Fold": {"brand": "SAMSUNG", "tier": "T4", "source": "learned_cooccurrence"},
    })
    ocr = [
        [{"text": "Z Fold 8"}],
        [{"text": "unrelated text"}],
        [{"text": "Z Fold for real"}],
    ]
    out = _collect_ocr_product_resolutions(ocr, resolver, match_brand)
    assert len(out) == 1
    assert out[0]["frame_index"] == 0  # first frame recorded
    assert out[0]["frames"] == [0, 2]


def test_collect_ocr_never_fabricates():
    resolver = _FakeResolver({"Mac Mini": {"brand": "APPLE"}})
    ocr = [[{"text": "come and take it"}], [{"text": "hello world"}]]
    out = _collect_ocr_product_resolutions(ocr, resolver, match_brand)
    assert out == []


def test_collect_ocr_respects_distinct_and_frame_caps():
    resolver = FakeResolverWithCap()
    ocr = [[{"text": "prod %d" % i}] for i in range(8)]
    out = _collect_ocr_product_resolutions(
        ocr, resolver, match_brand, max_distinct=3, max_frames=2,
    )
    assert len(out) == 3
    assert all(len(r["frames"]) == 1 for r in out)


class FakeResolverWithCap:
    def resolve(self, text, live=True):
        import re
        m = re.search(r"prod (\d+)", text)
        if not m:
            return []
        return [{
            "brand": "BRAND" + m.group(1),
            "span": text,
            "product_span": text,
            "resolution_tier": "T2",
            "source": "wikidata",
            "confidence": 0.8,
        }]


def test_collect_ocr_handles_ocr_failures_per_frame():
    resolver = _FakeResolver({})
    ocr = [[{"text": "Mac Mini"}], None, []]
    out = _collect_ocr_product_resolutions(ocr, resolver, match_brand)
    assert out == []


# ── server: scene caption (deterministic, real detections) ───


def test_scene_caption_synthesized_from_real_detections():
    objects = [
        {"class_name": "person"}, {"class_name": "cell phone"},
        {"class_name": "person"}, {"class_name": "wristwatch"},
    ]
    logos = [
        {"brand": "SAMSUNG", "confidence": 0.9},
        {"brand": "UNKNOWN BRAND", "confidence": 0.1},
    ]
    cap = _scene_caption(objects, logos, ["Mac Mini", "Mind blown"])
    assert "person, cell phone, wristwatch" in cap   # distinct, real labels
    assert "SAMSUNG" in cap                           # resolved brand present
    assert "UNKNOWN BRAND" not in cap                 # unresolved excluded
    assert "on-screen text 'Mac Mini Mind blown'" in cap


def test_scene_caption_empty_when_nothing_detected():
    assert _scene_caption([], [], []) == ""
    assert _scene_caption(None, None, None) == ""


def test_scene_caption_caps_parts():
    objects = [{"class_name": "obj%d" % i} for i in range(10)]
    logos = [{"brand": "BRAND%d" % i} for i in range(10)]
    cap = _scene_caption(objects, logos, [], max_objects=2, max_brands=1)
    assert cap.count(",") <= 1          # <= 2 objects joined
    assert "BRAND0" in cap and "BRAND9" not in cap


def test_scene_caption_never_leaks_raw_class_label_as_brand():
    # A logo box the resolver could NOT name has class_name "ROLEX logo" and no
    # `brand`. It must NEVER print that label as if it were a confirmed brand —
    # the dashboard scene chip for the same box shows UNKNOWN BRAND.
    logos = [
        {"class_name": "ROLEX logo", "confidence": 0.1},   # unresolved
        {"class_name": "SAMSUNG logo", "brand": "SAMSUNG", "confidence": 0.9},
    ]
    cap = _scene_caption([], logos, [])
    assert "ROLEX" not in cap
    assert "SAMSUNG" in cap
    assert "ROLEX logo" not in cap


# ── server: product_resolutions dashboard block ──────────────


def _src_ts(idx, stride=3, fps=30.0):
    return (idx * stride) / fps


def test_build_product_resolutions_maps_ocr_frame_to_timestamp():
    l1 = {"product_resolutions": [{
        "brand": "APPLE", "span": "Mac Mini", "product_span": "Mac Mini",
        "resolution_tier": "T2", "source": "wikidata", "confidence": 0.85,
        "mode": "ocr", "frame_index": 10, "frames": [10, 12],
    }]}
    def nearest(ts):
        return ts  # passthrough for the assertion below
    out = _build_product_resolutions(
        l1,
        src_timestamp_fn=lambda idx: _src_ts(idx, stride=3, fps=30.0),
        nearest_frame_fn=nearest,
    )
    assert len(out) == 1
    r = out[0]
    assert r["start_time"] == pytest.approx(1.0)  # 10 * 3 / 30
    assert r["brand"] == "APPLE"
    assert r["product"] == "Mac Mini"
    assert r["mode"] == "ocr"


def test_build_product_resolutions_speech_uses_real_timestamp():
    l1 = {"product_resolutions": [{
        "brand": "SAMSUNG", "span": "Z Fold", "product_span": "Z Fold",
        "resolution_tier": "T4", "source": "learned_cooccurrence",
        "mode": "speech", "start_time": 42.0, "end_time": 43.5,
    }]}
    out = _build_product_resolutions(
        l1,
        src_timestamp_fn=lambda idx: _src_ts(idx),
        nearest_frame_fn=lambda ts: 99,
    )
    assert out[0]["start_time"] == 42.0
    assert out[0]["end_time"] == 43.5
    assert out[0]["frame_index"] == 99  # nearest scene frame for clip jump


def test_build_product_resolutions_fail_closed_empty():
    assert _build_product_resolutions(
        {}, src_timestamp_fn=lambda i: i, nearest_frame_fn=lambda t: t,
    ) == []


def test_build_product_resolutions_sorts_timestampless_last():
    l1 = {"product_resolutions": [
        {"brand": "X", "mode": "speech", "start_time": 5.0},
        {"brand": "Y", "mode": "ocr", "frame_index": 0, "span": "thing"},
        {"brand": "Z", "mode": "ocr", "span": "no frame at all"},
    ]}
    def src_ts(idx):
        return (idx * 3) / 30.0
    out = _build_product_resolutions(
        l1, src_timestamp_fn=src_ts, nearest_frame_fn=lambda t: t,
    )
    assert [r["brand"] for r in out] == ["Y", "X", "Z"]  # 0.0s, 5.0s, then no-time
    assert out[0]["start_time"] == 0.0
    assert out[-1]["start_time"] is None