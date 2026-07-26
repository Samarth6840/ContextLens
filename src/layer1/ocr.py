"""
Layer 1 — OCR Module
Uses PaddleOCR for text detection and recognition in video frames.
Loads real model weights — no mock/stub/placeholder inference.
"""

import logging
from typing import List, Optional, Tuple

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
        self.use_angle_cls = use_angle_cls
        self.det_db_thresh = det_db_thresh
        self.rec_batch_num = rec_batch_num

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

        # Real OCR inference — not a mock
        results = self.model.ocr(image, cls=self.use_angle_cls)

        if results is None or results[0] is None:
            return []

        ocr_results = []
        for line in results[0]:
            bbox, (text, confidence) = line
            ocr_results.append({
                "text": text,
                "confidence": float(confidence),
                "bbox": bbox,
            })

        return ocr_results

    def extract_text_batch(
        self,
        frames: List[np.ndarray],
    ) -> List[List[dict]]:
        """Run OCR on a batch of frames."""
        return [self.extract_text(frame) for frame in frames]

    def extract_brand_texts(
        self,
        image: np.ndarray,
        brand_keywords: Optional[List[str]] = None,
    ) -> List[dict]:
        """
        Extract text and filter for potential brand mentions.

        Args:
            image: RGB image
            brand_keywords: Optional list of brand names to look for.
                            If None, returns all OCR results.

        Returns:
            List of OCR results matching brand keywords (or all if no filter)
        """
        ocr_results = self.extract_text(image)

        if brand_keywords is None:
            return ocr_results

        brand_keywords_lower = [kw.lower() for kw in brand_keywords]
        filtered = []
        for result in ocr_results:
            text_lower = result["text"].lower()
            if any(kw in text_lower for kw in brand_keywords_lower):
                filtered.append(result)

        return filtered