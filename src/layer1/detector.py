"""
Layer 1 — Scene Object Detection Module
Uses YOLOv8 (ultralytics) for generic object detection (COCO classes).
These detections are used for frame weighting, OCR filtering, and scene context —
NOT for brand/logo detection. See logo_detector.py for real brand detection.
Loads real model weights — no mock/stub/placeholder inference.
"""

import logging
from typing import List, Optional

import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class SceneObjectDetector:
    """
    Scene object detection using YOLOv8 (COCO pretrained).
    Detects generic objects: person, cell phone, laptop, etc.
    Used for frame weighting and OCR filtering — NOT for brand detection.
    For brand/logo detection, see src.layer1.logo_detector.
    Loads actual pretrained weights — no hardcoded returns.
    """

    def __init__(
        self,
        model_name: str = "yolov8x.pt",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
        open_vocab_queries: Optional[List[str]] = None,
        open_vocab_confidence: Optional[float] = None,
        open_vocab_model_name: str = "yolov8s-worldv2.pt",
    ):
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.open_vocab_queries = list(open_vocab_queries or [])
        self.open_vocab_confidence = (
            open_vocab_confidence
            if open_vocab_confidence is not None
            else confidence_threshold
        )
        self.open_vocab_model_name = open_vocab_model_name
        self._open_vocab_model = None

        # Load real model weights — not a stub
        logger.info(f"Loading YOLO model '{model_name}' on {device}")
        self.model = YOLO(model_name)
        self.model.to(device)
        logger.info(f"YOLO model loaded successfully on {device}")

    def detect_batch(
        self,
        frames: List[np.ndarray],
        batch_size: int = 8,
        classes: Optional[List[int]] = None,
    ) -> List[List[dict]]:
        """
        Run detection on a batch of frames using true batched YOLO inference.

        Args:
            frames: List of RGB images
            batch_size: Number of frames per batch
            classes: Optional class filter

        Returns:
            List of detection lists, one per frame
        """
        all_detections = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i : i + batch_size]
            # Pipeline frames are RGB; Ultralytics interprets raw NumPy arrays
            # as BGR, so channel-swap at the model boundary (RGB -> BGR).
            results = self.model(
                [frame[:, :, ::-1] for frame in batch],
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                classes=classes,
                device=self.device,
                verbose=False,
            )
            for result in results:
                detections = []
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    for box, conf, cls_id in zip(boxes, confidences, class_ids):
                        detections.append({
                            "bbox": box.tolist(),
                            # ravel: ultralytics returns conf as (N,) today and
                            # (N,1) in some builds; float() on an ndim>0 array
                            # is deprecated and errors in numpy>=1.25.
                            "confidence": float(np.ravel(conf)[0]),
                            "class_id": int(cls_id),
                            "class_name": result.names[int(cls_id)],
                            "detection_source": "coco",
                        })
                all_detections.append(detections)
        return all_detections

    def detect_open_vocab(
        self,
        frames: List[np.ndarray],
        text_queries: Optional[List[str]] = None,
        batch_size: int = 8,
    ) -> List[List[dict]]:
        """
        Zero-shot open-vocabulary object detection (additional pass, not a
        replacement for the COCO pass). Prompts YOLO-World with config-driven
        product queries ("wristwatch", "wireless earbuds", "sneaker", ...) that
        the fixed COCO vocabulary cannot represent. On high-IoU overlap the
        curated label replaces the COCO label at merge time — never a shape/
        aspect-ratio heuristic override.

        Returns:
            Per-frame detection lists; every entry carries class_name,
            confidence, bbox and `detection_source: "open_vocab"`.
        """
        queries = list(text_queries or self.open_vocab_queries)
        if not queries:
            return [[] for _ in frames]

        if self._open_vocab_model is None:
            from src.layer1.logo_detector import YOLOWorldLogoDetector

            logger.info(
                "Loading open-vocab YOLO-World model '%s' on %s (queries: %s)",
                self.open_vocab_model_name, self.device, queries,
            )
            self._open_vocab_model = YOLOWorldLogoDetector(
                model_name=self.open_vocab_model_name,
                confidence_threshold=self.open_vocab_confidence,
                device=self.device,
                text_queries=queries,
            )

        per_frame = self._open_vocab_model.detect_batch(frames, queries, batch_size)
        for dets in per_frame:
            for d in dets:
                d["class_id"] = -1
                d["detection_source"] = "open_vocab"
        return per_frame


def _iou(box_a, box_b) -> float:
    """Intersection-over-union of two [x1, y1, x2, y2] boxes (pixel space)."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def merge_open_vocab_detections(
    all_detections: List[List[dict]],
    open_vocab_by_index: dict,
    min_merge_iou: float = 0.4,
) -> List[List[dict]]:
    """
    Merge open-vocabulary detections into the COCO per-frame detection lists.

    Matching rule (additional pass, curated label wins on real spatial overlap):
    - For each open-vocab box, find the COCO box with the highest IoU on the
      same frame.
    - If that IoU >= min_merge_iou, the curated open-vocab label REPLACES the
      COCO label for that region (a wristwatch correctly labeled as a watch
      rather than misnamed 'cell phone'), and the entry's confidence is the
      better of the two. The detection keeps `detection_source: "open_vocab"`.
    - Otherwise the open-vocab box is ADDED as its own detection.

    Every detection is tagged `detection_source` ("coco" or "open_vocab"); the
    tag is informational provenance and never influences matching.
    """
    # Deep-copy the inner dicts too: mutating `best` below used to reach back
    # into the caller's detection dicts (a `list(d)` only copies the outer list),
    # so a later re-merge saw already-overwritten labels.
    merged = [[dict(d) for d in frame] for frame in all_detections]
    for idx, ov_dets in (open_vocab_by_index or {}).items():
        idx = int(idx)
        if idx < 0 or idx >= len(merged) or not ov_dets:
            continue
        for ov in ov_dets:
            best = None
            best_iou = 0.0
            for coco in merged[idx]:
                iou = _iou(ov.get("bbox"), coco.get("bbox"))
                if iou > best_iou:
                    best_iou = iou
                    best = coco
            if best is not None and best_iou >= min_merge_iou:
                best["class_name"] = ov.get("class_name", best.get("class_name"))
                best["detection_source"] = "open_vocab"
                best["confidence"] = float(
                    max(best.get("confidence", 0.0), ov.get("confidence", 0.0))
                )
            else:
                merged[idx].append({
                    "bbox": ov.get("bbox"),
                    "confidence": float(ov.get("confidence", 0.0)),
                    "class_id": int(ov.get("class_id", -1)),
                    "class_name": ov.get("class_name"),
                    "detection_source": "open_vocab",
                })
    return merged