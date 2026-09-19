"""
Layer 1 — Visual Embeddings Module
Uses DINOv3 as a frozen backbone.
Extracts visual features from video frames — real forward pass, no stubs.
"""

import logging
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, DINOv3ViTImageProcessor

logger = logging.getLogger(__name__)


def embedding_model_version(model_name: str) -> str:
    """Return a stable, comparable tag identifying an embedding model instance.

    This is the single source of truth for the embedding space. Any consumer
    that stores vectors derived from an extractor (product index, logo
    retrieval index, persisted caches) MUST record this tag alongside the
    vectors and refuse to query them with a different version. Two models can
    share the same output dim yet live in incompatible vector spaces (e.g.
    DINOv2-base vs DINOv3-vitb16, both 768), so a dimension check is NOT a
    sufficient guard against silent nearest-neighbor corruption.
    """
    name = (model_name or "").strip() or "dinov3-vitb16-pretrain-lvd1689m"
    # Collapse to a coarse but stable identifier: family + size variant. This
    # keeps the tag human-readable while still distinguishing DINOv2/v2/v3 and
    # the base/large/small variants from each other.
    norm = name.lower().replace("facebook/", "").replace("/", "-")
    return f"embed:{norm}"


def assert_embedding_version(
    index_version: Optional[str],
    query_version: Optional[str],
    index_name: str = "embedding index",
) -> None:
    """Fail loudly if a live query's embedding space differs from the stored one.

    Both versions are Optional so the guard is a strict improvement over
    silence: when EITHER side lacks a version tag we cannot assert, and we skip
    (backward compatible with legacy indexes). When BOTH are present and differ,
    comparing vectors across incompatible spaces (e.g. DINOv2 vs DINOv3, same
    768-dim) would return plausible-but-wrong nearest neighbors forever — that is
    the silent corruption this prevents.
    """
    if query_version and index_version and query_version != index_version:
        raise ValueError(
            f"{index_name}: embedding-space version mismatch — stored vectors "
            f"were produced by '{index_version}' but the live query uses "
            f"'{query_version}'. Output dims can match while the spaces are "
            "incompatible (e.g. DINOv2 vs DINOv3). Rebuild the index with the "
            "active embedding model before querying it."
        )


class VisualEmbeddingExtractor:
    """
    Extracts visual embeddings from video frames using DINOv3 (ViT).
    Loads real pretrained weights — no hardcoded returns.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        output_dim: int = 768,
        device: Optional[str] = None,
    ):
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.output_dim = output_dim
        self.embedding_version = embedding_model_version(model_name)

        # Load real model weights
        logger.info(f"Loading DINOv3 model '{model_name}' on {device}")

        try:
            self.processor = DINOv3ViTImageProcessor.from_pretrained(model_name)
        except Exception as exc:
            logger.warning(
                "Falling back to built-in DINOv3 processor config "
                "(Hub image processor unavailable: %s)",
                exc,
            )
            self.processor = DINOv3ViTImageProcessor(
                do_resize=True,
                size={"height": 518, "width": 518},
                do_rescale=True,
                rescale_factor=1 / 255,
                do_normalize=True,
                image_mean=[0.485, 0.456, 0.406],
                image_std=[0.229, 0.224, 0.225],
                do_center_crop=True,
                crop_size={"height": 518, "width": 518},
            )

        # Try the cached weights first with local_files_only so an offline or
        # slow network never hangs (or fails) a job that has a working cache.
        # Only fall back to a network fetch if the weights are not present.
        self.model = self._load_model(model_name)
        self.model.to(device)
        self.model.eval()
        logger.info(f"DINOv3 model loaded successfully on {device}")

    def _load_model(self, model_name: str):
        """Load model weights, preferring the local HuggingFace cache.

        Tries `local_files_only` first so cached models load instantly and
        offline (a dead network can no longer hang the job on retry/backoff
        the way a bare `from_pretrained` does). Only if the model is missing
        locally does it attempt a network fetch. If both fail, raises a clear,
        actionable error instead of letting an opaque OSError propagate up and
        fail the whole analysis job.
        """
        try:
            return AutoModel.from_pretrained(model_name, local_files_only=True)
        except Exception as offline_exc:
            logger.info(
                "DINOv3 weights not in local cache (%s); attempting network fetch",
                offline_exc,
            )
            try:
                return AutoModel.from_pretrained(model_name)
            except Exception as net_exc:
                raise RuntimeError(
                    "DINOv3 visual-embedding model could not be loaded for "
                    f"'{model_name}'. It is not in the HuggingFace cache and "
                    "the Hub could not be reached. Fix by running e.g. "
                    f"`huggingface-cli download {model_name}` (or allowing "
                    "network access) and retrying. "
                    f"[cache error: {offline_exc}] [network error: {net_exc}]"
                ) from net_exc

    @torch.no_grad()
    def extract_batch(
        self,
        frames: List[np.ndarray],
        batch_size: int = 8,
    ) -> np.ndarray:
        """
        Extract embeddings for a batch of frames using true batched inference.

        Args:
            frames: List of RGB images
            batch_size: Number of frames per batch

        Returns:
            Array of shape (len(frames), output_dim)
        """
        embeddings = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i : i + batch_size]
            pil_images = [Image.fromarray(f) for f in batch]
            inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            batch_embs = F.normalize(outputs.last_hidden_state[:, 0, :], p=2, dim=-1)
            embeddings.append(batch_embs.cpu().numpy().astype(np.float32))

        if not embeddings:
            return np.zeros((0, self.output_dim), dtype=np.float32)

        return np.concatenate(embeddings, axis=0)
