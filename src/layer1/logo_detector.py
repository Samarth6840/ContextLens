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

import base64
import logging
import os
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
            image[:, :, ::-1],  # pipeline frames are RGB; Ultralytics wants BGR
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
                [frame[:, :, ::-1] for frame in batch],  # RGB in, Ultralytics wants BGR
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


class YOLOLogoDetector(LogoDetectionBackend):
    """
    Single-class supervised logo detector (Ultralytics YOLO), e.g. weights
    trained by scripts/train_logo_detector.py on LogoDet-3K.

    Classes come from the checkpoint and there is no CLIP text encoder to
    re-encode, so `set_classes` is never called and `text_queries` is ignored.
    A trained class name is a category ("logo"), never a brand verdict.
    """

    # Class-level defaults so the attributes exist on any instance, including
    # one built via __new__ in tests that skip the model load.
    max_detections: int = 100
    imgsz: int = 640

    def __init__(
        self,
        model_name: str = "weights/logo_detector/train/weights/best.pt",
        confidence_threshold: float = 0.25,
        device: Optional[str] = None,
        text_queries: Optional[List[str]] = None,
        max_detections: int = 100,
        imgsz: int = 640,
    ):
        import torch
        from ultralytics import YOLO

        self.confidence_threshold = confidence_threshold
        # Serve at the checkpoint's OWN training resolution. The current
        # best.pt was trained at imgsz=640; evaluating/serving it at 960 costs
        # mAP (val mAP50 0.067 -> 0.048 in a controlled comparison). Keep this
        # equal to config.yaml's logo_detector.imgsz, and bump both together
        # with a retrained 960 model.
        self.imgsz = imgsz
        # Bound the candidate set per frame. Ultralytics defaults to 300, and
        # the nested-label pathology in the training data makes this detector
        # emit boxes densely enough that the count reaches the NMS clock and
        # trips "NMS time limit ... exceeded". Every extra box is also another
        # crop-OCR call downstream, so the cap is a cost control as much as a
        # stability one. 100 is ~2.5x what a healthy frame produces here, so it
        # only discards the pathological tail.
        self.max_detections = max_detections
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self._current_queries = None  # fixed classes, nothing to re-prompt

        logger.info("Loading supervised logo detector '%s' on %s", model_name, self.device)
        self.model = YOLO(model_name)
        self.model.to(self.device)
        logger.info("Supervised logo detector loaded (classes: %s)", self.model.names)

    def detect(
        self,
        image: np.ndarray,
        text_queries: Optional[List[str]] = None,
    ) -> List[dict]:
        if image is None or image.size == 0:
            return []
        results = self.model(
            image[:, :, ::-1],  # pipeline frames are RGB; Ultralytics wants BGR
            conf=self.confidence_threshold,
            imgsz=self.imgsz,
            device=self.device,
            max_det=self.max_detections,
            verbose=False,
        )
        # ultralytics >=8.4 returns a LIST even for a single image; older
        # versions returned the bare Results. Accept both.
        if isinstance(results, list):
            results = results[0] if results else None
        return self._to_detections(results) if results is not None else []

    def detect_batch(
        self,
        frames: List[np.ndarray],
        text_queries: Optional[List[str]] = None,
        batch_size: int = 8,
    ) -> List[List[dict]]:
        if not frames:
            return []
        # One entry per frame, in order. Callers index this against the frame
        # list (pipeline.py assigns all_logo_detections[idx] = dets), so
        # flattening the batch here silently mis-assigns every box.
        out: List[List[dict]] = []
        for i in range(0, len(frames), batch_size):
            results = self.model(
                [f[:, :, ::-1] for f in frames[i : i + batch_size]],  # RGB in -> BGR for YOLO
                conf=self.confidence_threshold,
                imgsz=self.imgsz,
                device=self.device,
                max_det=self.max_detections,
                verbose=False,
            )
            for result in results:
                out.append(self._to_detections(result))
        # Guard the contract the pipeline relies on rather than trusting it.
        if len(out) != len(frames):
            raise RuntimeError(
                f"detect_batch returned {len(out)} frames for {len(frames)} inputs"
            )
        return out

    @staticmethod
    def _to_detections(result) -> List[dict]:
        detections = []
        if result.boxes is None:
            return detections
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            detections.append(
                {
                    "bbox": box.tolist(),
                    "confidence": float(conf),
                    "text_prompt": None,
                    "class_name": result.names[int(cls_id)],
                }
            )
        return detections


def create_logo_detector(
    backend: str = "yolo_world",
    model_name: str = "yolov8s-worldv2.pt",
    confidence_threshold: float = 0.30,
    device: Optional[str] = None,
    text_queries: Optional[List[str]] = None,
    imgsz: int = 640,
    **kwargs,
) -> LogoDetectionBackend:
    """
    Factory: create the configured logo detection backend.

    Args:
        backend: "yolo_world" (zero-shot, prompted) | "yolo" (trained single-class)
        model_name: model weights path
        confidence_threshold: minimum confidence for detections
        device: "cuda", "cpu", or None for auto
        text_queries: optional override for YOLO-World text prompts

    Returns:
        LogoDetectionBackend instance
    """
    if backend == "google_cloud_vision":
        return GoogleCloudVisionLogoDetector(
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            device=device,
            imgsz=imgsz,
            timeout=float(kwargs.get("timeout", 20.0)),
            min_score=float(kwargs.get("min_score", 0.25)),
        )
    if backend == "yolo_world":
        return YOLOWorldLogoDetector(
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            device=device,
            text_queries=text_queries,
        )
    if backend == "yolo":
        return YOLOLogoDetector(
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            device=device,
            imgsz=imgsz,
        )
    raise ValueError(
        f"Unknown logo detection backend: '{backend}'. "
        "Choose 'google_cloud_vision', 'yolo_world', or 'yolo'."
    )


class GoogleCloudVisionLogoDetector(LogoDetectionBackend):
    """Cloud Vision named-logo detector with a local YOLO fallback."""

    API_URL = "https://vision.googleapis.com/v1/images:annotate"

    def __init__(
        self,
        model_name: str,
        confidence_threshold: float,
        device: Optional[str],
        imgsz: int,
        timeout: float = 20.0,
        min_score: float = 0.25,
    ):
        self.confidence_threshold = float(confidence_threshold)
        self.device = device
        self.imgsz = int(imgsz)
        self.timeout = float(timeout)
        self.min_score = float(min_score)
        self._fallback_args = (model_name, confidence_threshold, device, imgsz)
        self._fallback = None
        self._remote_disabled = False
        # Reuse the project's env/.env loader without exposing key material.
        from src.openset import _env, _safe_external_error
        self._api_key = os.environ.get("GOOGLE_CLOUD_VISION_API_KEY") or _env(
            "GOOGLE_CLOUD_VISION_API_KEY"
        )
        self._safe_error = _safe_external_error

    @property
    def available(self) -> bool:
        return bool(self._api_key) and not self._remote_disabled

    @staticmethod
    def _parse_response(response: dict, frame_shape) -> List[dict]:
        h, w = frame_shape[:2]
        out = []
        for annotation in response.get("logoAnnotations") or []:
            try:
                score = float(annotation.get("score", 0.0))
                vertices = (annotation.get("boundingPoly") or {}).get("vertices") or []
                points = [
                    (float(point.get("x", 0)), float(point.get("y", 0)))
                    for point in vertices
                ]
            except (TypeError, ValueError):
                continue
            if len(points) < 2 or not np.isfinite(points).all():
                continue
            bbox = [
                max(0.0, min(float(w), min(p[0] for p in points))),
                max(0.0, min(float(h), min(p[1] for p in points))),
                max(0.0, min(float(w), max(p[0] for p in points))),
                max(0.0, min(float(h), max(p[1] for p in points))),
            ]
            name = str(annotation.get("description") or "").strip()
            if name and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                out.append({
                    "bbox": bbox,
                    "confidence": score,
                    "text_prompt": name,
                    "class_name": name,
                    "class_confirmed": True,
                    "detection_source": "google_cloud_vision",
                })
        return out

    def _local_fallback(self):
        if self._fallback is None:
            model_name, confidence, device, imgsz = self._fallback_args
            self._fallback = YOLOLogoDetector(
                model_name=model_name,
                confidence_threshold=confidence,
                device=device,
                imgsz=imgsz,
            )
        return self._fallback

    def detect(self, image: np.ndarray, text_queries=None) -> List[dict]:
        return self.detect_batch([image], text_queries=text_queries)[0]

    def detect_batch(
        self,
        frames: List[np.ndarray],
        text_queries: Optional[List[str]] = None,
        batch_size: int = 8,
    ) -> List[List[dict]]:
        if not frames:
            return []
        if not self.available:
            logger.warning("Cloud Vision is not configured or was disabled; using local logo proposals")
            return self._local_fallback().detect_batch(frames, text_queries, batch_size)

        import cv2
        import requests

        output: List[List[dict]] = [[] for _ in frames]
        try:
            chunk_size = min(max(1, int(batch_size)), 8)
            for start in range(0, len(frames), chunk_size):
                batch = frames[start:start + chunk_size]
                payloads = []
                positions = []
                for local_idx, frame in enumerate(batch):
                    if frame is None or frame.size == 0:
                        continue
                    ok, encoded = cv2.imencode(
                        ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 92],
                    )
                    if not ok:
                        raise ValueError("could not encode frame for Cloud Vision")
                    payloads.append({
                        "image": {"content": base64.b64encode(encoded).decode("ascii")},
                        "features": [{"type": "LOGO_DETECTION", "maxResults": 10}],
                    })
                    positions.append(local_idx)
                if not payloads:
                    continue
                response = requests.post(
                    self.API_URL,
                    params={"key": self._api_key},
                    json={"requests": payloads},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                responses = response.json().get("responses") or []
                if len(responses) != len(positions):
                    raise RuntimeError("Cloud Vision returned a misaligned response batch")
                for local_idx, item in zip(positions, responses):
                    if item.get("error"):
                        logger.warning("Cloud Vision failed on one frame: %s", item["error"])
                        continue
                    output[start + local_idx] = [
                        det for det in self._parse_response(item, batch[local_idx].shape)
                        if det["confidence"] >= self.min_score
                    ]
            return output
        except Exception as exc:  # noqa: BLE001 — remote failure must not break a video job
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code in {401, 403, 404}:
                # A bad key/service configuration won't recover during this
                # process. Stop issuing one doomed cloud call per uploaded video.
                self._remote_disabled = True
            logger.warning(
                "Cloud Vision logo detection failed; using local proposals: %s",
                self._safe_error(exc),
            )
            return self._local_fallback().detect_batch(frames, text_queries, batch_size)
