"""
Model download script for Phase 1.
Downloads real pretrained model weights — no stubs or placeholders.

Models:
- YOLOv8x: downloaded on first use by ultralytics (auto-download)
- YOLO-World (yolov8s-worldv2.pt): logo-detection backend; downloaded via ultralytics
- DINOv2: from HuggingFace transformers
- PaddleOCR: downloaded on first use by PaddleOCR (auto-download)
- Whisper medium: primary ASR model (mlx-whisper / faster-whisper / openai-whisper fallback chain)
- BEATs: requires manual download from Microsoft UNILM repository

This script triggers the auto-downloads and verifies they work.
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def download_yolo():
    """Trigger YOLO model download."""
    logger.info("Downloading YOLOv8x model...")
    from ultralytics import YOLO
    model = YOLO("yolov8x.pt")
    logger.info(f"YOLOv8x downloaded successfully. Model path: {model.ckpt_path}")
    return True


def download_yoloworld():
    """Trigger the YOLO-World zero-shot logo-detection model download.

    YOLO-World is the shipped logo-detection backend (config
    layer1.logo_detection.backend == "yolo_world"). Ultralytics auto-downloads
    the weights on first use; we trigger it here so cold start is explicit.
    """
    logger.info("Downloading YOLO-World model (yolov8s-worldv2.pt)...")
    from ultralytics import YOLO
    model = YOLO("yolov8s-worldv2.pt")
    logger.info(f"YOLO-World downloaded successfully. Model path: {model.ckpt_path}")
    return True


def download_dinov2():
    """Trigger DINOv2 model download from HuggingFace."""
    logger.info("Downloading DINOv2 model from HuggingFace...")
    from transformers import AutoImageProcessor, AutoModel
    model_name = "facebook/dinov2-base"  # matches config.yaml
    AutoImageProcessor.from_pretrained(model_name)
    AutoModel.from_pretrained(model_name)
    logger.info(f"DINOv2 model '{model_name}' downloaded successfully.")
    return True


def download_whisper():
    """Trigger Whisper model download (openai-whisper fallback)."""
    logger.info("Downloading Whisper medium (openai-whisper fallback)...")
    import whisper
    whisper.load_model("medium")
    logger.info("Whisper medium downloaded successfully.")
    return True


def download_paddleocr():
    """Trigger PaddleOCR model download."""
    logger.info("Downloading PaddleOCR models...")
    from paddleocr import PaddleOCR
    PaddleOCR(use_angle_cls=True, lang="en")
    logger.info("PaddleOCR models downloaded successfully.")
    return True


def check_beats():
    """Download the fine-tuned BEATs AudioSet tagging checkpoint if missing.

    Preferred source is the HuggingFace mirror (microsoft's Azure links have a
    history of intermittent failures — see microsoft/unilm#1492). The file is
    written to the repo root where config/layer1/audio_events.checkpoint points.
    """
    import urllib.request

    destination = Path("BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt")
    source_url = (
        "https://huggingface.co/WeiChihChen/"
        "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2/resolve/main/"
        "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
    )
    if destination.exists():
        logger.info(f"Fine-tuned BEATs checkpoint found at {destination}")
        return True
    logger.warning("Fine-tuned BEATs checkpoint missing — downloading %s", source_url)
    try:
        urllib.request.urlretrieve(source_url, destination)
        logger.info(
            "Downloaded fine-tuned BEATs checkpoint (%d bytes). "
            "Feature-extractor fallback BEATs_iter3_plus_AS2M.pt, if present, "
            "remains valid for the degraded path.",
            destination.stat().st_size,
        )
        return True
    except Exception as e:
        logger.error(f"BEATs auto-download failed: {e}")
        logger.info(
            "Manual download: %s\nPlace it as %s next to this repo's config, "
            "or keep BEATs_iter3_plus_AS2M.pt for the degraded audio path.",
            source_url, destination,
        )
        return False


def main():
    logger.info("=" * 60)
    logger.info("Phase 1 — Model Download Script")
    logger.info("=" * 60)

    success = True

    # Layer 1 models
    logger.info("\n--- Layer 1 Models ---")

    try:
        download_yolo()
    except Exception as e:
        logger.error(f"YOLO download failed: {e}")
        success = False

    try:
        download_yoloworld()
    except Exception as e:
        logger.error(f"YOLO-World download failed: {e}")
        success = False

    try:
        download_dinov2()
    except Exception as e:
        logger.error(f"DINOv2 download failed: {e}")
        success = False

    try:
        download_whisper()
    except Exception as e:
        logger.error(f"Whisper download failed: {e}")
        success = False

    try:
        download_paddleocr()
    except Exception as e:
        logger.error(f"PaddleOCR download failed: {e}")
        success = False

    # BEATs (optional)
    check_beats()

    logger.info("\n" + "=" * 60)
    if success:
        logger.info("All primary models downloaded successfully.")
    else:
        logger.error("Some models failed to download. Check errors above.")
    logger.info("=" * 60)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())