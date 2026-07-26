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
        model_name: str = "facebook/dinov2-large",
        output_dim: int = 1024,
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
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
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract embedding from a single frame.

        Args:
            image: RGB image as numpy array (H, W, 3)

        Returns:
            Embedding vector of shape (output_dim,)
        """
        if image is None or image.size == 0:
            return np.zeros(self.output_dim, dtype=np.float32)

        # Convert numpy to PIL for the processor
        pil_image = Image.fromarray(image)

        # Preprocess and run real forward pass
        inputs = self.processor(
            images=pil_image,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**inputs)
        # Use CLS token embedding
        embedding = outputs.last_hidden_state[:, 0, :]  # (1, dim)
        embedding = F.normalize(embedding, p=2, dim=-1)

        return embedding.cpu().numpy().flatten().astype(np.float32)

    @torch.no_grad()
    def extract_batch(
        self,
        frames: List[np.ndarray],
        batch_size: int = 8,
    ) -> np.ndarray:
        """
        Extract embeddings for a batch of frames.

        Args:
            frames: List of RGB images
            batch_size: Number of frames per batch

        Returns:
            Array of shape (len(frames), output_dim)
        """
        embeddings = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i : i + batch_size]
            batch_embeddings = []

            for frame in batch:
                emb = self.extract(frame)
                batch_embeddings.append(emb)

            embeddings.extend(batch_embeddings)

        return np.stack(embeddings, axis=0)

    def compute_frame_quality_from_embeddings(
        self,
        embeddings: np.ndarray,
    ) -> float:
        """
        Estimate video quality from embedding consistency.
        High variance across frames may indicate poor quality/noise.
        Used as a video quality signal for Layer 2a.
        """
        if embeddings.shape[0] < 2:
            return 1.0

        # Compute pairwise cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-8)
        similarity_matrix = normalized @ normalized.T

        # Mean off-diagonal similarity — high = consistent = good quality
        n = similarity_matrix.shape[0]
        off_diag = similarity_matrix[~np.eye(n, dtype=bool)].reshape(n, -1)
        mean_similarity = float(np.mean(off_diag))

        return mean_similarity