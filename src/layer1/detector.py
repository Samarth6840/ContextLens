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
    ):
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

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
            results = self.model(
                batch,
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
                            "confidence": float(conf),
                            "class_id": int(cls_id),
                            "class_name": result.names[int(cls_id)],
                        })
                all_detections.append(detections)
        return all_detections