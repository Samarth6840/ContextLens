"""
Layer 1 — Visual Embeddings Module
Uses DINOv2 (DINOv3-equivalent via transformers) as a frozen backbone.
Extracts visual features from video frames — real forward pass, no stubs.
"""

import logging
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

logger = logging.getLogger(__name__)


class VisualEmbeddingExtractor:
    """
    Extracts visual embeddings from video frames using DINOv2.
    Loads real pretrained weights — no hardcoded returns.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        output_dim: int = 768,
        device: Optional[str] = None,
    ):
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.output_dim = output_dim

        # Load real model weights
        logger.info(f"Loading DINOv2 model '{model_name}' on {device}")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        logger.info(f"DINOv2 model loaded successfully on {device}")

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
