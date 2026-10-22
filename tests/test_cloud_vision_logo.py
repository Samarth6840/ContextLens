"""Cloud Vision logo adapter and its resolver trust boundary."""

import numpy as np

from src.layer1 import logo_detector
from src.layer1.logo_detector import GoogleCloudVisionLogoDetector
from src.layer2.brand_resolver import BrandResolver


def test_cloud_vision_parses_named_logo_and_clips_bbox():
    parsed = GoogleCloudVisionLogoDetector._parse_response(
        {
            "logoAnnotations": [{
                "description": "Samsung",
                "score": 0.93,
                "boundingPoly": {"vertices": [
                    {"x": -10, "y": 4}, {"x": 60, "y": 4},
                    {"x": 60, "y": 40}, {"x": -10, "y": 40},
                ]},
            }]
        },
        (50, 100, 3),
    )
    assert parsed == [{
        "bbox": [0.0, 4.0, 60.0, 40.0],
        "confidence": 0.93,
        "text_prompt": "Samsung",
        "class_name": "Samsung",
        "class_confirmed": True,
        "detection_source": "google_cloud_vision",
    }]


def test_cloud_vision_batches_images_and_preserves_alignment(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_VISION_API_KEY", "test-key")
    detector = GoogleCloudVisionLogoDetector("local.pt", 0.1, "cpu", 640)
    calls = []

    class Response:
        def __init__(self, count):
            self.count = count

        def raise_for_status(self):
            pass

        def json(self):
            one = {"logoAnnotations": [{
                    "description": "Google", "score": 0.9,
                    "boundingPoly": {"vertices": [
                        {"x": 1, "y": 2}, {"x": 20, "y": 2},
                        {"x": 20, "y": 15}, {"x": 1, "y": 15},
                    ]},
                }]}
            return {"responses": [one] + [{} for _ in range(self.count - 1)]}

    def fake_post(url, **kwargs):
        calls.append(kwargs)
        return Response(len(kwargs["json"]["requests"]))

    monkeypatch.setattr("requests.post", fake_post)
    frames = [np.zeros((32, 40, 3), np.uint8) for _ in range(3)]
    out = detector.detect_batch(frames, batch_size=2)
    assert [len(row) for row in out] == [1, 0, 1]
    assert len(calls) == 2
    assert len(calls[0]["json"]["requests"]) == 2


def test_cloud_vision_failure_uses_local_fallback_without_leaking_key(monkeypatch, caplog):
    monkeypatch.setenv("GOOGLE_CLOUD_VISION_API_KEY", "test-key-must-not-leak")
    detector = GoogleCloudVisionLogoDetector("local.pt", 0.1, "cpu", 640)

    class Fallback:
        def detect_batch(self, frames, text_queries, batch_size):
            return [[{"class_name": "logo", "confidence": 0.2, "bbox": [1, 1, 5, 5]}]
                    for _ in frames]

    detector._fallback = Fallback()

    def denied(*args, **kwargs):
        raise RuntimeError(
            "403 https://vision.googleapis.com/v1/images:annotate?key=test-key-must-not-leak"
        )

    monkeypatch.setattr("requests.post", denied)
    out = detector.detect_batch([np.zeros((32, 32, 3), np.uint8)])
    assert out[0][0]["class_name"] == "logo"
    assert "test-key-must-not-leak" not in caplog.text


def test_cloud_vision_label_is_independent_brand_evidence():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detector = {
        "bbox": [10, 10, 80, 60],
        "confidence": 0.93,
        "class_name": "Samsung",
        "detection_source": "google_cloud_vision",
        "class_confirmed": True,
    }
    resolved = BrandResolver(ocr_extractor=None).resolve([[detector]], [frame])
    assert resolved[0][0]["brand"] == "SAMSUNG"
    assert resolved[0][0]["resolution_source"] == "google_cloud_vision"
