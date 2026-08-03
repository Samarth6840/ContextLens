"""
Layer 1 — Brand/Logo Detection Module

Zero-shot backend (ships now): YOLO-World prompted with text queries
like "Samsung logo", "brand logo". Returns real bounding boxes and real
model-output confidence — no post-processing into suspiciously round numbers.

Configuration:
    layer1.logo_detection.backend: "yolo_world"
    layer1.logo_detection.text_queries: list of text prompts
    layer1.logo_detection.confidence_threshold: float

All inference is real — no mock/stub/placeholder.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class LogoDetectionBackend(ABC):
    """Abstract base for logo detection backends."""

    @abstractmethod
    def detect(
        self,
        image: np.ndarray,
        text_queries: Optional[List[str]] = None,
    ) -> List[dict]:
        """
        Run logo detection on a single image.

        Returns list of dicts:
            - bbox: [x1, y1, x2, y2] pixel coords
            - confidence: float — real model output, not post-processed
            - text_prompt: str — which query matched
        """
        ...

    @abstractmethod
    def detect_batch(
        self,
        frames: List[np.ndarray],
        text_queries: Optional[List[str]] = None,
        batch_size: int = 8,
    ) -> List[List[dict]]:
        """Run detection on a batch of frames."""
        ...


class YOLOWorldLogoDetector(LogoDetectionBackend):
    """
    Zero-shot logo detection using YOLO-World.

    Prompts the model with text queries like "Samsung logo", "brand logo"
    and returns real bounding boxes + real model-output confidence scores.
    """

    DEFAULT_QUERIES = [
        "Samsung logo",
        "brand logo",
        "company logo",
        "text logo",
        "product logo",
    ]

    def __init__(
        self,
        model_name: str = "yolov8s-worldv2.pt",
        confidence_threshold: float = 0.30,
        device: Optional[str] = None,
        text_queries: Optional[List[str]] = None,
    ):
        import torch
        from ultralytics import YOLO

        self.confidence_threshold = confidence_threshold
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self._current_queries: Optional[List[str]] = None

        logger.info("Loading YOLO-World model '%s' on %s", model_name, self.device)
        self.model = YOLO(model_name)
        self.model.to(self.device)

        # Set classes once at init — avoids re-encoding text with CLIP per call
        queries = text_queries or self.DEFAULT_QUERIES
        self.model.set_classes(queries)
        self._current_queries = list(queries)
        logger.info("YOLO-World model loaded successfully (classes set: %s)", queries)

    def detect(
        self,
        image: np.ndarray,
        text_queries: Optional[List[str]] = None,
    ) -> List[dict]:
        if image is None or image.size == 0:
            return []

        queries = text_queries or self._current_queries or self.DEFAULT_QUERIES
        # Only re-encode text with CLIP if queries changed
        if queries != self._current_queries:
            self.model.set_classes(queries)
            self._current_queries = list(queries)

        results = self.model(
            image,
            conf=self.confidence_threshold,
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
                class_name = result.names[int(cls_id)]
                detections.append({
                    "bbox": box.tolist(),
                    "confidence": float(conf),
                    "text_prompt": class_name,
                    "class_name": class_name,
                })

        return detections

    def detect_batch(
        self,
        frames: List[np.ndarray],
        text_queries: Optional[List[str]] = None,
        batch_size: int = 8,
    ) -> List[List[dict]]:
        # Only re-encode text with CLIP if queries changed (avoid per-batch cost)
        queries = text_queries or self._current_queries or self.DEFAULT_QUERIES
        if queries != self._current_queries:
            self.model.set_classes(queries)
            self._current_queries = list(queries)

        all_detections = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i : i + batch_size]
            results = self.model(
                batch,
                conf=self.confidence_threshold,
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
                        class_name = result.names[int(cls_id)]
                        detections.append({
                            "bbox": box.tolist(),
                            "confidence": float(conf),
                            "text_prompt": class_name,
                            "class_name": class_name,
                        })
                all_detections.append(detections)

        return all_detections


def create_logo_detector(
    backend: str = "yolo_world",
    model_name: str = "yolov8s-worldv2.pt",
    confidence_threshold: float = 0.30,
    device: Optional[str] = None,
    text_queries: Optional[List[str]] = None,
    **kwargs,
) -> LogoDetectionBackend:
    """
    Factory: create the configured logo detection backend.

    Args:
        backend: "yolo_world"
        model_name: model weights path
        confidence_threshold: minimum confidence for detections
        device: "cuda", "cpu", or None for auto
        text_queries: optional override for YOLO-World text prompts

    Returns:
        LogoDetectionBackend instance
    """
    if backend == "yolo_world":
        return YOLOWorldLogoDetector(
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            device=device,
            text_queries=text_queries,
        )
    else:
        raise ValueError(
            f"Unknown logo detection backend: '{backend}'. "
            "Choose 'yolo_world'."
        )
