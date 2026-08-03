"""
Layer 1 — OCR Module
Uses PaddleOCR for text detection and recognition in video frames.
Loads real model weights — no mock/stub/placeholder inference.
"""

import logging
from typing import List

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class OCRExtractor:
    """
    OCR text extraction using PaddleOCR.
    Detects and recognizes text in video frames — real inference.
    """

    def __init__(
        self,
        lang: str = "en",
        use_angle_cls: bool = True,
        det_db_thresh: float = 0.3,
        rec_batch_num: int = 6,
    ):
        self.lang = lang

        # Lazy import PaddleOCR — it has heavy dependencies
        logger.info(f"Initializing PaddleOCR (lang={lang})")
        from paddleocr import PaddleOCR as _PaddleOCR

        self.model = _PaddleOCR(
            use_angle_cls=use_angle_cls,
            lang=lang,
            det_db_thresh=det_db_thresh,
            rec_batch_num=rec_batch_num,
        )
        logger.info("PaddleOCR initialized successfully")

    def extract_text(self, image: np.ndarray) -> List[dict]:
        """
        Extract text from a single image frame.

        Args:
            image: RGB image as numpy array (H, W, 3)

        Returns:
            List of OCR result dicts with keys:
                - text: recognized string
                - confidence: float
                - bbox: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] polygon
        """
        if image is None or image.size == 0:
            return []

        # PaddleOCR expects BGR input; VideoProcessor.load_video produces RGB
        if image.ndim == 3 and image.shape[2] == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image

        # PaddleOCR 3.7.0: use predict() instead of deprecated ocr()
        results = self.model.predict(image_bgr)

        if results is None:
            return []

        # PaddleOCR 3.7.0+ predict() returns OCRResult objects (attribute access)
        # Older versions may return dict-like objects. Handle both formats.
        ocr_results = []
        for result in results:
            if hasattr(result, "get"):
                # Dict-like object (legacy format)
                rec_texts = result.get("rec_texts", [])
                rec_scores = result.get("rec_scores", [])
                rec_polys = result.get("rec_polys", [])
            else:
                # OCRResult dataclass (PaddleOCR 3.7.0+)
                rec_texts = getattr(result, "rec_texts", [])
                rec_scores = getattr(result, "rec_scores", [])
                rec_polys = getattr(result, "rec_polys", [])

            for text, score, poly in zip(rec_texts, rec_scores, rec_polys):
                ocr_results.append({
                    "text": text,
                    "confidence": float(score),
                    "bbox": poly.tolist() if hasattr(poly, "tolist") else list(poly),
                })

        return ocr_results

    def extract_text_batch(
        self,
        frames: List[np.ndarray],
    ) -> List[List[dict]]:
        """Run OCR on a batch of frames."""
        return [self.extract_text(frame) for frame in frames]
