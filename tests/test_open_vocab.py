"""
Tests for the open-vocabulary scene-object detection pass (Layer 1).

The pass is ADDITIONAL to the COCO detector, never a replacement: YOLO-World
is prompted with config-driven product queries ("wristwatch", "wireless
earbuds", "sneaker", ...) that the fixed COCO vocabulary cannot represent.
Merging is spatial (IoU)-based — on high IoU overlap the curated open-vocab
label wins for that region; there is no shape/aspect-ratio heuristic. Every
detection carries an informational `detection_source` tag that never
influences matching.
"""


def _coco(box, name="cell phone", conf=0.8):
    return {
        "bbox": list(box),
        "class_name": name,
        "confidence": conf,
        "class_id": 67,
        "detection_source": "coco",
    }


def _ov(box, name, conf=0.6):
    return {
        "bbox": list(box),
        "class_name": name,
        "confidence": conf,
        "class_id": -1,
        "detection_source": "open_vocab",
    }


def _noop_yolo():
    """Patch target for src.layer1.detector.YOLO: never loads real weights."""

    class _FakeYOLO:
        def __init__(self, *args, **kwargs):
            self.names = {}

        def to(self, device):
            return self

    return _FakeYOLO


class TestMergeOpenVocabDetections:
    def test_curated_label_replaces_coco_label_on_high_iou(self):
        from src.layer1.detector import merge_open_vocab_detections

        coco = [_coco([0, 0, 100, 100])]
        ov = {0: [_ov([5, 5, 95, 95], "wristwatch")]}
        out = merge_open_vocab_detections([coco], ov, min_merge_iou=0.4)
        dets = out[0]
        # One fused entry: the curated label won, tag provenance is open_vocab.
        assert len(dets) == 1
        assert dets[0]["class_name"] == "wristwatch"
        assert dets[0]["detection_source"] == "open_vocab"
        # Confidence is the better of the two, never a fabricated blend.
        assert dets[0]["confidence"] == 0.8

    def test_non_overlapping_open_vocab_is_added(self):
        from src.layer1.detector import merge_open_vocab_detections

        coco = [_coco([0, 0, 100, 100])]
        ov = {0: [_ov([300, 300, 400, 400], "wireless earbuds")]}
        out = merge_open_vocab_detections([coco], ov, min_merge_iou=0.4)
        dets = out[0]
        assert len(dets) == 2
        assert dets[1]["class_name"] == "wireless earbuds"
        assert dets[1]["detection_source"] == "open_vocab"
        assert dets[1]["class_id"] == -1

    def test_below_iou_gate_is_added_not_merged(self):
        from src.layer1.detector import merge_open_vocab_detections

        coco = [_coco([0, 0, 50, 50])]
        # Touching-but-separate region: IoU 0 -> must not clobber the COCO label.
        ov = {0: [_ov([60, 0, 110, 50], "sneaker")]}
        out = merge_open_vocab_detections([coco], ov, min_merge_iou=0.4)
        dets = out[0]
        assert len(dets) == 2
        assert dets[0]["class_name"] == "cell phone"

    def test_absent_overlap_with_no_coco_detections_is_added(self):
        from src.layer1.detector import merge_open_vocab_detections

        ov = {0: [_ov([0, 0, 50, 50], "wall clock")]}
        out = merge_open_vocab_detections([[]], ov, min_merge_iou=0.4)
        assert len(out[0]) == 1
        assert out[0][0]["detection_source"] == "open_vocab"

    def test_out_of_range_index_is_ignored(self):
        from src.layer1.detector import merge_open_vocab_detections

        coco = [_coco([0, 0, 50, 50])]
        out = merge_open_vocab_detections([coco], {7: [_ov([0, 0, 50, 50], "x")]})
        # index 7 does not exist -> unchanged, no crash
        assert len(out) == 1
        assert out[0][0]["class_name"] == "cell phone"

    def test_empty_open_vocab_is_noop(self):
        from src.layer1.detector import merge_open_vocab_detections

        coco = [_coco([0, 0, 50, 50])]
        out = merge_open_vocab_detections([coco], {})
        assert out[0][0]["class_name"] == "cell phone"
        assert out[0][0]["detection_source"] == "coco"


class TestDetectorOpenVocabFailClosed:
    def test_no_queries_returns_empty_without_loading_model(self):
        """With an empty query list the detector must not even build a YOLO-World
        model (no reason to load weights when there is nothing to prompt)."""
        from unittest.mock import patch

        import numpy as np

        from src.layer1.detector import SceneObjectDetector

        class _NoModel:
            pass

        with patch("src.layer1.detector.YOLO", _noop_yolo()):
            det = SceneObjectDetector(
                model_name="yolov8x.pt",
                device="cpu",
                open_vocab_queries=[],
            )
        det.model = _NoModel()
        det._open_vocab_model = _NoModel()
        frames = [np.zeros((64, 64, 3), dtype=np.uint8)]
        assert det.detect_open_vocab(frames, []) == [[]]

    def test_coco_detections_are_tagged(self):
        """detect_batch stamps detection_source='coco' so the provenance contract
        holds for both passes."""
        from unittest.mock import patch

        import numpy as np

        from src.layer1.detector import SceneObjectDetector

        class _T:
            def __init__(self, v):
                self._v = v

            def cpu(self):
                return self

            def numpy(self):
                return self._v

            def __array__(self):
                return self._v

        class _FakeBoxes:
            def __init__(self, boxes, confs, clss, names):
                self.xyxy = _T(boxes)
                self.conf = _T(confs)
                self.cls = _T(clss)
                self.names = names

        class _FakeResult:
            def __init__(self):
                import numpy as np

                self.boxes = _FakeBoxes(
                    np.array([[0, 0, 10, 10]], dtype=np.float32),
                    np.array([[0.9]], dtype=np.float32),
                    np.array([0], dtype=np.int64),
                    {0: "person"},
                )
                self.names = {0: "person"}

        class _FakeModel(_noop_yolo()):
            def __call__(self, frames, conf=None, iou=None, classes=None, device=None, verbose=None):
                return [_FakeResult() for _ in frames]

        with patch("src.layer1.detector.YOLO", _FakeModel):
            det = SceneObjectDetector(model_name="yolov8x.pt", device="cpu")
        det._open_vocab_model = None
        out = det.detect_batch([np.zeros((64, 64, 3), dtype=np.uint8)])
        assert out[0][0]["class_name"] == "person"
        assert out[0][0]["detection_source"] == "coco"

    def test_pipeline_detector_factory_passes_open_vocab_queries(self):
        """The pipeline factory reads the config block and forwards queries +
        thresholds to the detector (config-driven, not hardcoded)."""
        from unittest.mock import patch

        from src.pipeline import Phase1Pipeline

        p = Phase1Pipeline(device_override="cpu")
        od_cfg = p.cfg["layer1"]["object_detection"]
        ov_cfg = od_cfg.get("open_vocab", {}) or {}
        assert ov_cfg.get("enabled") is True
        assert any("wristwatch" in q for q in (ov_cfg.get("text_queries") or []))

        captured = {}

        class _FakeDetector:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        with patch("src.layer1.detector.SceneObjectDetector", _FakeDetector):
            p._detector_factory()
        assert captured.get("open_vocab_queries") == ov_cfg.get("text_queries")
        assert (
            captured.get("open_vocab_confidence")
            == ov_cfg.get("confidence_threshold")
        )
        assert captured.get("open_vocab_model_name") == ov_cfg.get(
            "model", "yolov8s-worldv2.pt"
        )