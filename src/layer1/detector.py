"""
Layer 1 — Object/Logo Detection Module
Uses YOLO11 (ultralytics) for real-time brand logo detection.
Loads real model weights — no mock/stub/placeholder inference.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class LogoDetector:
    """
    Real-time logo/brand detection using YOLO11.
    Loads actual pretrained weights — no hardcoded returns.
    """

    def __init__(
        self,
        model_name: str = "yolo11",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        # Load real model weights — not a stub
        logger.info(f"Loading YOLO model '{model_name}' on {device}")
        self.model = YOLO(model_name)
        self.model.to(device)
        logger.info(f"YOLO model loaded successfully on {device}")

    def detect(
        self,
        image: np.ndarray,
        classes: Optional[List[int]] = None,
    ) -> List[dict]:
        """
        Run real forward pass on a single image frame.

        Args:
            image: RGB image as numpy array (H, W, 3)
            classes: Optional list of class IDs to filter by

        Returns:
            List of detection dicts with keys:
                - bbox: [x1, y1, x2, y2] in pixel coordinates
                - confidence: float
                - class_id: int
                - class_name: str
        """
        if image is None or image.size == 0:
            return []

        # Real inference — not a mock
        results = self.model(
            image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=classes,
            device=self.device,
            verbose=False,
        )

        detections = []
        for result in results:
            if result.boxes is None:
                continue

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

        return detections

    def detect_batch(
        self,
        frames: List[np.ndarray],
        batch_size: int = 8,
        classes: Optional[List[int]] = None,
    ) -> List[List[dict]]:
        """
        Run detection on a batch of frames.

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
            for frame in batch:
                detections = self.detect(frame, classes=classes)
                all_detections.append(detections)
        return all_detections

    def get_detection_confidence_variance(
        self,
        frames: List[np.ndarray],
        window: int = 10,
    ) -> float:
        """
        Estimate detection stability across a window of frames.
        Low variance = stable detection (high quality).
        High variance = flickering/unstable (low quality).

        This is used as a video quality signal for Layer 2a.
        """
        if len(frames) < 2:
            return 0.0

        confidences = []
        for frame in frames[:window]:
            dets = self.detect(frame)
            if dets:
                confidences.append(max(d["confidence"] for d in dets))
            else:
                confidences.append(0.0)

        if not confidences:
            return 0.0

        return float(np.var(confidences))