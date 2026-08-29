"""
Layer 1 — Product-Catalog Visual Embedding Index.

Implements the DINOv2 -> product-catalog similarity search from the roadmap
(Phase 2.5): for each video frame embedding, run a nearest-neighbor (cosine)
search against a reference index of per-brand product images, and emit a
`visual_product_match` evidence type (brand + similarity + frame timestamp).

This is the piece that detects a product on camera even without a visible logo
or spoken name.

Honesty contract (matches the rest of the pipeline):
  - The index is built ONLY from real reference product/logo images on disk
    (globbed from the configured directory tree, one subdir per brand using its
    canonical catalog name). No synthetic/fabricated embeddings are injected.
  - If no reference images are present, the index is empty and every match
    returns zero — the module FAILS CLOSED rather than fabricating evidence.
  - Reference images are keyed by the SAME canonical brand names as
    src/brand_catalog.py, so a detected product match feeds evidence, the
    brand timeline, and recommendations coherently.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.brand_catalog import BRAND_CATALOG

logger = logging.getLogger(__name__)

# Supported image extensions for the reference index.
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class ProductEmbeddingIndex:
    """
    Lazy cosine-similarity index over per-brand reference product images.

    The index is a matrix of unit-normalized DINOv2 embeddings, one row per
    reference image, plus a parallel array of (brand, image_path) labels.

    build() must be called once (after the DINOv2 extractor is available)
    before query(). If no reference images are found, the index stays empty and
    query() returns [] for every frame (fail-closed).
    """

    def __init__(self, reference_dir: str = "benchmark/product_logos"):
        self.reference_dir = Path(reference_dir)
        self._matrix: Optional[np.ndarray] = None  # (N, D) unit-normalized
        self._labels: List[Tuple[str, str]] = []   # (brand, path) per row
        self._built = False

    def _discover_images(self) -> List[Tuple[str, Path]]:
        """Return [(brand, image_path)] by scanning for reference images.

        Two layouts are supported, so the configured reference dir works whether
        the images are organized as one subdir per canonical brand name, or as
        flat files named '<BRAND>_<n>.png' (e.g. 'SAMSUNG_0.png'):
          1. subdir whose name matches a canonical catalog brand -> all image
             files inside are that brand's reference set;
          2. flat image whose UPPERCASE basename starts with '<BRAND>_' (or
             equals '<BRAND>') -> that brand's reference image.

        Only catalog-brand names are accepted — anything else (e.g. unrelated
        crop dumps) is ignored, preventing mislabeled evidence.
        """
        if not self.reference_dir.is_dir():
            logger.warning(
                "Product reference dir %s not found — product-index evidence "
                "will be empty (fail-closed)",
                self.reference_dir,
            )
            return []
        found: List[Tuple[str, Path]] = []

        def _brand_for_dir(name: str) -> Optional[str]:
            return name.upper() if name.upper() in BRAND_CATALOG else None

        for sub in sorted(self.reference_dir.iterdir()):
            if sub.is_dir():
                brand = _brand_for_dir(sub.name)
                if not brand:
                    logger.debug(
                        "Ignoring reference subdir %s (not a catalog brand)", sub
                    )
                    continue
                imgs = [
                    p for p in sub.iterdir()
                    if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
                ]
                for img in imgs:
                    found.append((brand, img))
                continue

            # Flat file: '<BRAND>[_<n>].<ext>'
            if sub.suffix.lower() not in _IMAGE_EXTS:
                continue
            stem = sub.stem.upper()
            base = stem.split("_")[0]
            brand = base if base in BRAND_CATALOG else None
            if not brand:
                logger.debug(
                    "Ignoring flat reference file %s (no catalog-brand prefix)",
                    sub,
                )
                continue
            found.append((brand, sub))
        return found

    def build(self, embed_extractor, batch_size: int = 8) -> int:
        """Embed all discovered reference images and build the (N, D) index.

        Args:
            embed_extractor: VisualEmbeddingExtractor instance (DINOv2).
            batch_size: batch size for embedding.

        Returns:
            Number of reference images indexed (0 if none found / fail-closed).
        """
        images = self._discover_images()
        if not images:
            logger.warning(
                "No reference images found under %s — product-index evidence "
                "disabled (fail-closed)",
                self.reference_dir,
            )
            self._built = True
            self._matrix = None
            self._labels = []
            return 0

        # Load + embed in batches (reuse PIL path from the extractor).
        from PIL import Image
        frames = []
        batch_labels: List[Tuple[str, str]] = []
        all_embs: List[np.ndarray] = []
        for brand, path in images:
            try:
                img = Image.open(path).convert("RGB")
                frames.append(np.array(img))
                batch_labels.append((brand, str(path)))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping unreadable reference image %s: %s", path, exc)
            if len(frames) == batch_size:
                all_embs.append(embed_extractor.extract_batch(frames, batch_size))
                frames = []
        if frames:
            all_embs.append(embed_extractor.extract_batch(frames, batch_size))

        if not all_embs:
            self._built = True
            self._matrix = None
            self._labels = []
            return 0

        matrix = np.concatenate(all_embs, axis=0)
        # extract_batch already returns unit-normalized vectors; re-normalize to
        # be safe against any padding.
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix = matrix / norms

        self._matrix = matrix.astype(np.float32)
        self._labels = batch_labels
        self._built = True
        logger.info(
            "Product index built: %d reference image(s) for %d brand(s)",
            len(self._labels),
            len(set(b[0] for b in batch_labels)),
        )
        return len(self._labels)

    @property
    def is_empty(self) -> bool:
        return self._built and (self._matrix is None or len(self._labels) == 0)

    def query(
        self,
        frame_embeddings: np.ndarray,
        frame_indices: List[int],
        video_fps: float = 0.0,
        top_k: int = 1,
        similarity_threshold: float = 0.80,
    ) -> List[dict]:
        """Run cosine NN search for each frame embedding against the index.

        Args:
            frame_embeddings: (M, D) unit-normalized DINOv2 embeddings for the
                              sampled frames (same length as frame_indices).
            frame_indices: the original frame index for each embedding row.
            video_fps: for converting frame index -> timestamp.
            top_k: how many nearest brand matches to return per frame.
            similarity_threshold: minimum cosine similarity for a match to count.

        Returns:
            List of match dicts:
                {brand, similarity, frame_index, timestamp, reference_image}
            Empty list when the index is empty (fail-closed) or nothing clears
            the threshold.
        """
        if not self._built:
            logger.warning(
                "ProductEmbeddingIndex.query called before build() — returning []"
            )
            return []
        if self._matrix is None or len(self._labels) == 0:
            return []
        if frame_embeddings is None or len(frame_embeddings) == 0:
            return []

        matches: List[dict] = []
        for row, frame_idx in zip(frame_embeddings, frame_indices):
            # Cosine sim to every reference vector.
            sims = self._matrix @ row  # (N,)
            order = np.argsort(sims)[::-1][:top_k]
            for rank_idx in order:
                sim = float(sims[rank_idx])
                if sim < similarity_threshold:
                    continue
                brand, ref_path = self._labels[rank_idx]
                matches.append({
                    "brand": brand,
                    "similarity": round(sim, 4),
                    "frame_index": int(frame_idx),
                    "timestamp": round(frame_idx / video_fps, 1) if video_fps else int(frame_idx),
                    "reference_image": ref_path,
                })
        return matches
