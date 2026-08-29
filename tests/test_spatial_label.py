"""
Tests for Layer 2c spatial / temporal label stabilization.

Covers:
  - TemporalObjectSmoother: flapping COCO labels across adjacent frames of the
    same device are smoothed to the majority label; singletons left alone.
  - apply_spatial_brand_context: object detections overlapping a resolved
    brand's on-screen location are tagged with the brand; distant ones aren't.
"""
import pytest

from src.layer2.spatial_label import (
    TemporalObjectSmoother,
    apply_spatial_brand_context,
)


def _det(cls, bbox, conf=0.8):
    return {"class_name": cls, "bbox": list(bbox), "confidence": conf}


class TestTemporalObjectSmoother:
    def test_flapping_label_smoothed_to_majority(self):
        # Same device at (50,50,150,150) across 4 frames, labels flapping.
        detections = [
            [_det("backpack", [50, 50, 150, 150])],
            [_det("remote", [50, 50, 150, 150])],
            [_det("mouse", [50, 50, 150, 150])],
            [_det("mouse", [50, 50, 150, 150])],
        ]
        out = TemporalObjectSmoother(min_track_len=2).smooth(detections)
        # Majority label is "mouse" (appears 2x); isolated flips suppressed.
        assert [d[0]["class_name"] for d in out] == [
            "mouse", "mouse", "mouse", "mouse",
        ]
        # Original labels preserved for diagnosis where they changed.
        assert out[0][0]["smoothed_label"] == "backpack"

    def test_singleton_frame_left_alone(self):
        detections = [[_det("laptop", [10, 10, 100, 100])]]
        out = TemporalObjectSmoother(min_track_len=2).smooth(detections)
        assert out[0][0]["class_name"] == "laptop"
        assert "smoothed_label" not in out[0][0]

    def test_two_distinct_blobs_not_merged(self):
        detections = [
            [
                _det("person", [0, 0, 40, 40]),
                _det("laptop", [200, 200, 300, 300]),
            ],
            [
                _det("person", [0, 0, 40, 40]),
                _det("laptop", [200, 200, 300, 300]),
            ],
        ]
        out = TemporalObjectSmoother(min_track_len=2).smooth(detections)
        assert out[0][0]["class_name"] == "person"
        assert out[0][1]["class_name"] == "laptop"


class TestSpatialBrandContext:
    def test_overlapping_object_tagged_with_brand(self):
        detections = [[_det("mouse", [50, 50, 150, 150])]]
        resolved = [[{"brand": "SAMSUNG", "bbox": [60, 60, 140, 140]}]]
        out = apply_spatial_brand_context(detections, resolved, window=1)
        assert out[0][0]["brand_context"] == "SAMSUNG"
        # Underlying label preserved (honesty).
        assert out[0][0]["class_name"] == "mouse"

    def test_distant_object_not_tagged(self):
        detections = [[_det("person", [300, 300, 400, 400])]]
        resolved = [[{"brand": "SAMSUNG", "bbox": [10, 10, 100, 100]}]]
        out = apply_spatial_brand_context(detections, resolved, window=1)
        assert "brand_context" not in out[0][0]

    def test_beyond_window_not_tagged(self):
        detections = [[] for _ in range(7)]
        detections[5] = [_det("mouse", [50, 50, 150, 150])]
        resolved = [[{"brand": "SAMSUNG", "bbox": [60, 60, 140, 140]}]] + [[] for _ in range(6)]
        # Brand at frame 0, object at frame 5 -> outside window=2.
        out = apply_spatial_brand_context(detections, resolved, window=2)
        assert "brand_context" not in out[5][0]

    def test_no_brand_regions_no_change(self):
        detections = [[_det("mouse", [50, 50, 150, 150])]]
        resolved = [[{"brand": None, "bbox": None}]]
        out = apply_spatial_brand_context(detections, resolved)
        assert out == detections
