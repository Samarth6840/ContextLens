"""
Layer 1 — Free.ai API Client
All-in-one client for OCR, STT, Vision, and Embeddings via Free.ai API.
30K free tokens/day, no credit card required.
"""

import io
import logging
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import requests

logger = logging.getLogger(__name__)

# Free.ai API configuration
FREE_AI_BASE_URL = "https://api.free.ai/v1"


class FreeAIClient:
    """
    Client for Free.ai API — OCR, STT, Vision, Embeddings.
    Uses a single API key for all modalities.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Free.ai client.

        Args:
            api_key: Free.ai API key. If not provided, reads from FREE_AI_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("FREE_AI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Free.ai API key required. Set FREE_AI_API_KEY env var or pass api_key parameter."
            )
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
        })
        logger.info("Free.ai client initialized")

    def _image_to_bytes(self, image: np.ndarray) -> bytes:
        """Convert numpy array to JPEG bytes."""
        if image.ndim == 3 and image.shape[2] == 3:
            # RGB to BGR for OpenCV
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image
        _, buffer = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return buffer.tobytes()

    def ocr(self, image: np.ndarray) -> List[dict]:
        """
        Extract text from image using Free.ai OCR.

        Args:
            image: RGB image as numpy array (H, W, 3)

        Returns:
            List of OCR result dicts with keys:
                - text: recognized string
                - confidence: float
        """
        if image is None or image.size == 0:
            return []

        try:
            image_bytes = self._image_to_bytes(image)
            response = self.session.post(
                f"{FREE_AI_BASE_URL}/ocr",
                files={"image": ("frame.jpg", image_bytes, "image/jpeg")},
                data={"model": "paddleocr-vl"},
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()

            # Parse Free.ai OCR response format
            ocr_results = []
            text = result.get("text", "")
            if text and not str(text).startswith("[OCR unavailable"):
                ocr_results.append({
                    "text": text,
                    "confidence": result.get("confidence", 0.9),
                })
            for item in result.get("results", []):
                ocr_results.append({
                    "text": item.get("text", ""),
                    "confidence": item.get("confidence", 0.9),
                })

            return ocr_results

        except Exception as e:
            logger.error(f"Free.ai OCR failed: {e}")
            return []

    def ocr_batch(self, frames: List[np.ndarray]) -> List[List[dict]]:
        """Run OCR on a batch of frames."""
        return [self.ocr(frame) for frame in frames]

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe audio using Free.ai STT (Whisper).

        Args:
            audio: Audio waveform as numpy array
            sample_rate: Sample rate in Hz

        Returns:
            Transcribed text string
        """
        if audio is None or len(audio) == 0:
            return ""

        try:
            # Convert numpy array to WAV bytes
            import soundfile as sf

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
                sf.write(tmp_path, audio, sample_rate)

            with open(tmp_path, "rb") as f:
                response = self.session.post(
                    f"{FREE_AI_BASE_URL}/stt",
                    files={"audio": ("audio.wav", f, "audio/wav")},
                    data={"model": "whisper", "language": "auto"},
                    timeout=60,
                )

            os.unlink(tmp_path)
            response.raise_for_status()
            result = response.json()

            text = result.get("text", "")
            if text:
                return text

            # Async job — poll the output URL for the transcription.
            output_url = result.get("output_url")
            if output_url:
                import time
                for _ in range(30):
                    time.sleep(1.0)
                    try:
                        out = self.session.get(output_url, timeout=15)
                        if out.status_code == 200 and out.text.strip():
                            return out.text.strip()
                    except Exception:
                        pass
            return result.get("text", "")

        except Exception as e:
            logger.error(f"Free.ai STT failed: {e}")
            return ""

    def describe_image(self, image: np.ndarray, prompt: str = "") -> dict:
        """
        Analyze image using Free.ai Vision.

        Args:
            image: RGB image as numpy array
            prompt: Optional analysis prompt

        Returns:
            Dict with analysis results
        """
        if image is None or image.size == 0:
            return {}

        try:
            image_bytes = self._image_to_bytes(image)
            payload = {
                "model": "qwen-vl",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self._bytes_to_base64(image_bytes)}"}},
                            {"type": "text", "text": prompt or "Describe this image in detail, focusing on any brands, logos, products, or text visible."},
                        ],
                    }
                ],
            }

            response = self.session.post(
                f"{FREE_AI_BASE_URL}/chat",
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            result = response.json()

            return {
                "description": result.get("choices", [{}])[0].get("message", {}).get("content", ""),
                "model": result.get("model", "unknown"),
            }

        except Exception as e:
            logger.error(f"Free.ai Vision failed: {e}")
            return {}

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get text embeddings using Free.ai Embeddings.

        Args:
            texts: List of text strings

        Returns:
            Embeddings as numpy array (n_texts, embed_dim)
        """
        if not texts:
            return np.array([])

        try:
            response = self.session.post(
                f"{FREE_AI_BASE_URL}/embeddings",
                json={
                    "model": "bge-m3",
                    "input": texts,
                },
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()

            embeddings = [item["embedding"] for item in result.get("data", [])]
            return np.array(embeddings, dtype=np.float32)

        except Exception as e:
            logger.error(f"Free.ai Embeddings failed: {e}")
            return np.array([])

    @staticmethod
    def _bytes_to_base64(data: bytes) -> str:
        """Convert bytes to base64 string."""
        import base64
        return base64.b64encode(data).decode("utf-8")


def create_freeai_client() -> Optional["FreeAIClient"]:
    """
    Create a Free.ai client if API key is available.

    Returns:
        FreeAIClient instance or None if no API key configured.
    """
    api_key = os.environ.get("FREE_AI_API_KEY")
    if not api_key:
        logger.warning("FREE_AI_API_KEY not set — Free.ai client disabled")
        return None

    try:
        return FreeAIClient(api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to create Free.ai client: {e}")
        return None


class FreeAIOCRExtractor:
    """
    Drop-in replacement for OCRExtractor (src/layer1/ocr.py) backed by
    Free.ai's cloud OCR API. Exposes the same interface so the pipeline can
    swap it in without changes.
    """

    def __init__(self, lang: str = "en", **kwargs):
        self.lang = lang
        self.client = create_freeai_client()
        if self.client is None:
            raise RuntimeError("Free.aith OCR requested but FREE_AI_API_KEY not set")
        logger.info("Free.ai OCR extractor initialized (lang=%s)", lang)

    def extract_text(self, image: np.ndarray) -> List[dict]:
        if image is None or image.size == 0:
            return []
        return self.client.ocr(image)

    def extract_text_batch(self, frames: List[np.ndarray]) -> List[List[dict]]:
        return [self.extract_text(frame) for frame in frames]


class FreeAISpeechToText:
    """
    Drop-in replacement for SpeechToText (src/layer1/audio.py) backed by
    Free.ai's cloud STT (Whisper) API. Exposes the same interface.
    """

    _backend = "freeai"

    def __init__(self, model_name: str = "large-v3", device: Optional[str] = None, **kwargs):
        self.model_name = model_name
        self.device = device
        self.client = create_freeai_client()
        if self.client is None:
            raise RuntimeError("Free.ai STT requested but FREE_AI_API_KEY not set")
        logger.info("Free.ai STT initialized (model=%s)", model_name)

    def transcribe_segment(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        if audio is None or len(audio) == 0:
            return ""
        return self.client.transcribe(audio, sample_rate=sample_rate)
