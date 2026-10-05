"""detect_batch must return exactly one detection list per input frame.

pipeline.py assigns per-frame results by index
(`all_logo_detections[idx] = dets`), so a flattened or short return silently
mis-assigns boxes to the wrong frames — and then BrandResolver iterates
detections that are not dicts. The single-image eval path cannot catch this
because it never calls detect_batch.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.layer1.logo_detector import YOLOLogoDetector


def _fake_results(n_frames, boxes_per_frame=2):
    out = []
    for _ in range(n_frames):
        arr = np.array([[10.0, 10.0, 50.0, 50.0]] * boxes_per_frame, dtype=np.float32)
        out.append(
            SimpleNamespace(
                boxes=SimpleNamespace(
                    xyxy=torch.from_numpy(arr),
                    conf=torch.tensor([0.9] * boxes_per_frame, dtype=torch.float32),
                    cls=torch.zeros(boxes_per_frame, dtype=torch.int32),
                ),
                names={0: "logo"},
            )
        )
    return out


def _detector(monkeypatch, n_frames, boxes_per_frame=2):
    det = YOLOLogoDetector.__new__(YOLOLogoDetector)  # skip model load
    det.confidence_threshold = 0.25
    det.device = "cpu"
    det._current_queries = None
    det.max_detections = 100
    seen = {}

    def _fake_model(images, **kw):
        seen["n"] = len(images)
        seen["max_det"] = kw.get("max_det")
        return _fake_results(len(images), boxes_per_frame)

    det.model = _fake_model
    return det, seen


@pytest.mark.parametrize("n_frames,batch", [(1, 8), (5, 8), (8, 8), (7, 4), (10, 3)])
def test_detect_batch_preserves_frame_alignment(monkeypatch, n_frames, batch):
    det, _ = _detector(monkeypatch, n_frames)
    frames = [np.zeros((20, 20, 3), dtype=np.uint8) for _ in range(n_frames)]
    out = det.detect_batch(frames, batch_size=batch)
    assert len(out) == n_frames
    for per_frame in out:
        assert isinstance(per_frame, list)
        for d in per_frame:
            assert isinstance(d, dict) and "bbox" in d and "confidence" in d


def test_detect_batch_never_sets_text_prompt():
    """A trained detector has no text prompts; None must not become a class hint."""
    det, _ = _detector(monkeypatch=None, n_frames=3)
    frames = [np.zeros((20, 20, 3), dtype=np.uint8) for _ in range(3)]
    for per_frame in det.detect_batch(frames):
        for d in per_frame:
            assert d["text_prompt"] is None
            assert d["class_name"] == "logo"


def test_detect_batch_empty_input():
    det, _ = _detector(monkeypatch=None, n_frames=1)
    assert det.detect_batch([]) == []


def test_max_detections_is_bounded_and_forwarded(monkeypatch):
    """Item 38: the detector emits boxes densely enough to reach Ultralytics'
    default 300-per-frame cap and trip "NMS time limit exceeded". Every extra
    box is also another crop-OCR call, the stage that dominated the 18-minute
    job. max_det must be passed through BOTH the single-image and batch paths
    or the bound silently applies to half the workload."""
    det, seen = _detector(monkeypatch, 4)
    det.max_detections = 7
    det.detect(np.zeros((20, 20, 3), dtype=np.uint8))
    assert seen["max_det"] == 7, "detect() did not forward max_det"

    det2, seen2 = _detector(monkeypatch, 4)
    det2.max_detections = 7
    det2.detect_batch([np.zeros((20, 20, 3), dtype=np.uint8)] * 4, batch_size=2)
    assert seen2["max_det"] == 7, "detect_batch() did not forward max_det"

    # Default exists even when __init__ was skipped.
    assert YOLOLogoDetector.max_detections > 0
