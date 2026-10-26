"""OCR-primary brand detections require compact, readable wordmarks."""

import numpy as np

from src.layer2.brand_resolver import BrandResolver
from src.pipeline import _logo_detector_enabled, _ocr_brand_detections


def _region(text, confidence=0.95, bbox=None):
    return {
        "text": text,
        "confidence": confidence,
        "bbox": bbox or [[10, 10], [80, 10], [80, 30], [10, 30]],
    }


def test_ocr_primary_accepts_wordmark_and_rejects_article_and_phone_ui():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    resolver = BrandResolver(screen_content_filter=True)
    detections = _ocr_brand_detections(
        [[
            _region("CMF PHONE 1"),
            _region("Price of every Apple product that got costlier in India"),
            _region("Instagram"),
            _region("HONDA", confidence=0.50),
        ]],
        [frame],
        resolver,
    )

    assert len(detections) == 1
    assert [d["brand"] for d in detections[0]] == ["CMF"]
    assert detections[0][0]["resolution_source"] == "full_frame_ocr"
    assert detections[0][0]["bbox"] == [10.0, 10.0, 80.0, 30.0]


def test_ocr_primary_preserves_frame_alignment_and_skips_bad_geometry():
    frames = [
        np.zeros((100, 200, 3), dtype=np.uint8),
        np.zeros((100, 200, 3), dtype=np.uint8),
    ]
    detections = _ocr_brand_detections(
        [
            [_region("realme P4 5G")],
            [_region("HONDA", bbox=[[10, 10]], confidence=0.99)],
        ],
        frames,
        BrandResolver(),
    )

    assert len(detections) == 2
    assert [d["brand"] for d in detections[0]] == ["REALME"]
    assert detections[1] == []


def test_ocr_primary_can_keep_detector_only_for_open_set_proposals():
    assert _logo_detector_enabled("ocr_primary", {"use_detector_proposals": True})
    assert not _logo_detector_enabled("ocr_primary", {"use_detector_proposals": False})
    assert _logo_detector_enabled("hybrid", {"use_detector_proposals": False})


def test_shipped_config_uses_ocr_identity_and_online_proposals():
    import yaml
    from pathlib import Path

    config = yaml.safe_load(Path("config/config.yaml").read_text())
    logo = config["layer1"]["logo_detection"]
    assert logo["identity_mode"] == "ocr_primary"
    assert logo["use_detector_proposals"] is True
