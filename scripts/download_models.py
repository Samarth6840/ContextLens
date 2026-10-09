"""
Model download script for Phase 1.
Downloads real pretrained model weights — no stubs or placeholders.

Models:
- YOLOv8x: downloaded on first use by ultralytics (auto-download)
- YOLO-World (yolov8s-worldv2.pt): logo-detection backend; downloaded via ultralytics
- DINOv3: from HuggingFace transformers (gated — run `huggingface-cli login`)
- PaddleOCR: downloaded here for the configured language
- Whisper (configured size): mlx-whisper primary + openai-whisper fallback
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


def _config() -> dict:
    """Load config/config.yaml so downloads match what the pipeline serves."""
    import yaml
    path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    return yaml.safe_load(path.read_text()) or {}


def download_dinov2():
    """Trigger the shipped visual backbone download (DINOv3, see config).

    DINOv3 is gated on HuggingFace, so `huggingface-cli login` (or
    HF_TOKEN) is required before this succeeds.
    """
    from transformers import AutoModel, DINOv3ViTImageProcessor
    model_name = (
        _config().get("layer1", {}).get("visual_embeddings", {})
        .get("model", "facebook/dinov3-vitb16-pretrain-lvd1689m")
    )
    logger.info("Downloading DINOv3 model '%s' from HuggingFace...", model_name)
    DINOv3ViTImageProcessor.from_pretrained(model_name)
    AutoModel.from_pretrained(model_name)
    logger.info("DINOv3 model '%s' downloaded successfully.", model_name)
    return True


def download_whisper():
    """Trigger Whisper download for the configured backend/fallbacks.

    mlx-whisper is primary on Apple Silicon; openai-whisper is the final
    fallback. Both are warmed so a runtime fallback is not a cold download.
    """
    model = _config().get("layer1", {}).get("speech_to_text", {}).get("model", "medium")
    try:
        import mlx_whisper.load_models  # type: ignore
        logger.info("Downloading Whisper '%s' (mlx-whisper, primary on MPS)...", model)
        mlx_whisper.load_models.load_model(model)
    except Exception as exc:  # noqa: BLE001 — mlx is Darwin-only; fallbacks below
        logger.info("mlx-whisper unavailable (%s); continuing with fallbacks.", exc)
    logger.info("Downloading Whisper '%s' (openai-whisper fallback)...", model)
    import whisper
    whisper.load_model(model)
    logger.info("Whisper '%s' downloaded successfully.", model)
    return True


def download_paddleocr():
    """Trigger PaddleOCR model download for the configured language."""
    lang = _config().get("layer1", {}).get("ocr", {}).get("lang", "hi")
    logger.info("Downloading PaddleOCR models (lang=%s)...", lang)
    from paddleocr import PaddleOCR
    # PaddleOCR 3.x renamed use_angle_cls -> use_textline_orientation.
    PaddleOCR(lang=lang, use_textline_orientation=True)
    logger.info("PaddleOCR models (lang=%s) downloaded successfully.", lang)
    return True


# Must match src/layer1/audio.py BEATS_TRUSTED_SHA256 — the sha256 (git-lfs oid)
# of the pinned third-party artifact that audio.py will pickle-load. Kept as a
# literal here so this standalone script does not have to import torch.
BEATS_TRUSTED_SHA256 = (
    "e5815275a04b6885e7b8af63d120b29bffae2cd2225cf4915e1ec6d819d3022c"
)


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_beats():
    """Download the fine-tuned BEATs AudioSet tagging checkpoint if missing.

    Preferred source is the HuggingFace mirror (microsoft's Azure links have a
    history of intermittent failures — see microsoft/unilm#1492). The file is
    written to the repo root where config/layer1/audio_events.checkpoint points.
    The download is verified against the pinned sha256 so a tampered mirror
    cannot seed a malicious pickle into the load path.
    """
    import urllib.request

    destination = Path("BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt")
    source_url = (
        "https://huggingface.co/WeiChihChen/"
        "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2/resolve/main/"
        "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
    )
    if destination.exists():
        if _sha256(destination) == BEATS_TRUSTED_SHA256:
            logger.info(f"Fine-tuned BEATs checkpoint verified at {destination}")
            return True
        logger.warning(
            "BEATs checkpoint at %s fails sha256 verification — re-downloading",
            destination,
        )
        destination.unlink()
    logger.warning("Fine-tuned BEATs checkpoint missing — downloading %s", source_url)
    try:
        urllib.request.urlretrieve(source_url, destination)
        digest = _sha256(destination)
        if digest != BEATS_TRUSTED_SHA256:
            destination.unlink()
            raise ValueError(
                f"downloaded BEATs checkpoint sha256 mismatch (got {digest}); "
                "deleted the untrusted file"
            )
        logger.info(
            "Downloaded and verified fine-tuned BEATs checkpoint (%d bytes). "
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