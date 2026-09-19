"""
Layer 1 — CLIP-based logo brand retrieval index (Phase 1/2 of the
logo-recall/precision remediation).

CONTEXT
-------
YOLO-World zero-shot is weak on BOTH axes for logo detection (LogoDet-3K:
P=4.6%, R=8.2%, mAP50=2.0%), and it is confidently wrong on icon-only logos
where there is no text for OCR to rescue (e.g. it labels a Samsung foldable's
Samsung wordmark "SUPREME logo" at 0.45-0.60). OCR cross-check (3052ed6) only
saves the text-wordmark cases.

THIS MODULE makes CLIP-embedding retrieval the PRIMARY brand classifier:
  * build a per-brand CLIP image-embedding index from a reference bank
    (crops keyed by canonical brand name — same names as src/brand_catalog.py);
  * query a logo crop -> top-k (brand, similarity) candidates. The best match
    above `min_similarity` is the retrieval brand for that region.

Honesty / fail-closed (repo ethos):
  * If a brand has NO reference crops in the bank, it can never be retrieved —
    the index simply returns no candidate for it (no fabricated similarity).
  * The index stores raw per-brand crops + normalized embeddings; nothing is
    invented. Empty bank => empty retrieval (module returns None / empty).
  * Reference images are keyed by canonical catalog brands so adding a brand
    later is "add images under a <BRAND>/ dir and rebuild" — not a code change.

CLIP is loaded lazily (ViT-B/32 via open_clip if a checkpoint is configured,
else the built-in `clip` package), exactly mirroring the existing
scripts/logo_bench_backends.py RegionProposalCLIPBackend loading, so we never
load two divergent CLIP instances in one process.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.layer1.visual_embeddings import assert_embedding_version

logger = logging.getLogger(__name__)


def _canonical(brand: str) -> str:
    return (brand or "").strip().upper()


class LogoRetrievalIndex:
    """Flat CLIP image-embedding index mapping brand -> reference crops."""

    def __init__(
        self,
        reference_dir: str = "benchmark/reference_logos",
        min_similarity: float = 0.20,
        top_k: int = 3,
        checkpoint_path: Optional[str] = None,
    ):
        self.reference_dir = Path(reference_dir)
        self.min_similarity = float(min_similarity)
        self.top_k = int(top_k)
        self.checkpoint_path = checkpoint_path or None

        self._clip = None
        self._preprocess = None
        self._use_open_clip = False
        self._device = "cpu"

        # Built index state (None until build() succeeds).
        self._embeddings: Optional[np.ndarray] = None   # (N, D) L2-normalized
        self._brand_of_row: List[str] = []              # brand per reference crop
        self._brand_rows: Dict[str, List[int]] = {}     # brand -> row indices

    # False until _load() has run; lets us record which CLIP checkpoint/space
    # produced the stored embeddings so querying with a different one fails
    # loudly instead of silently corrupting nearest-neighbor results.
    @property
    def embedding_version(self) -> Optional[str]:
        if not self._use_open_clip and self._clip is None:
            return None
        if self._use_open_clip:
            source = self.checkpoint_path or "open_clip:ViT-B-32"
        else:
            source = "clip:ViT-B/32"
        return f"logo_clip:{source}"

    # ── CLIP lifecycle (lazy; mirrors RegionProposalCLIPBackend) ─────────
    def _load(self, device: str = "cpu"):
        self._device = device
        if self._clip is not None:
            return
        if self.checkpoint_path and Path(self.checkpoint_path).exists():
            import open_clip

            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained=self.checkpoint_path, device=device
            )
            model.eval()
            self._clip, self._preprocess, self._use_open_clip = model, preprocess, True
        else:
            import clip

            model, preprocess = clip.load("ViT-B/32", device=device)
            self._clip, self._preprocess, self._use_open_clip = model, preprocess, False

    def _embed_image_batch(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """L2-normalized CLIP image embeddings for an array of BGR crops."""
        import torch
        from PIL import Image

        if not crops:
            return np.zeros((0, 512), dtype=np.float32)
        self._load(self._device)
        imgs = []
        for c in crops:
            rgb = np.ascontiguousarray(c[:, :, ::-1])
            imgs.append(self._preprocess(Image.fromarray(rgb)))
        batch = torch.stack(imgs).to(self._device)
        with torch.no_grad():
            feats = self._clip.encode_image(batch).cpu().numpy()
        return feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)

    # ── Building the index ───────────────────────────────────────────────
    def add_brand(self, brand: str, crops: Sequence[np.ndarray]) -> int:
        """Embed and append reference crops for a canonical brand.

        Returns number of crops added (0 if the brand already has crops — we
        add offsets, caller manages lifecycle). Fails closed: bad brand name
        (not URL-sane) or empty crops adds nothing.
        """
        brand = _canonical(brand)
        if not brand or not re.match(r"^[\w'&.\-]+(?: [\w'&.\-]+)*$", brand):
            logger.warning("LogoRetrievalIndex: skipping invalid brand %r", brand)
            return 0
        crops = [c for c in crops if c is not None and c.size > 0]
        if not crops:
            return 0
        embs = self._embed_image_batch(crops)
        start = 0 if self._embeddings is None else self._embeddings.shape[0]
        rows = list(range(start, start + embs.shape[0]))
        self._embeddings = (
            embs if self._embeddings is None
            else np.concatenate([self._embeddings, embs], axis=0)
        )
        self._brand_of_row.extend([brand] * embs.shape[0])
        self._brand_rows.setdefault(brand, []).extend(rows)
        logger.info("LogoRetrievalIndex: +%d %s crops (bank total %d)",
                    embs.shape[0], brand, self._embeddings.shape[0])
        return embs.shape[0]

    def clear(self):
        self._embeddings = None
        self._brand_of_row = []
        self._brand_rows = {}

    @property
    def is_empty(self) -> bool:
        return self._embeddings is None or self._embeddings.shape[0] == 0

    @property
    def brands(self) -> List[str]:
        return sorted(self._brand_rows.keys())

    # ── Querying ─────────────────────────────────────────────────────────
    def query(
        self, crop: np.ndarray, top_k: Optional[int] = None,
        embedding_version: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """Return top-k (canonical_brand, similarity) candidates, desc by score.

        Aggregated per brand: each brand appears at most once, at its BEST crop
        similarity (many brands have several reference crops, so raw rows would
        produce duplicate brands and corrupt the ranking). This makes a
        top-1-vs-top-2 similarity "margin" meaningful (they are always distinct
        brands).

        `embedding_version` (when provided) must match this index's own
        CLIP-space version, else the caller is comparing vectors from different
        embedding spaces and we fail loudly rather than return wrong neighbors.

        Fail-closed: empty/None crop or empty index -> []. Candidates below
        `min_similarity` are dropped (unknown / not-in-bank brands give []).
        """
        if crop is None or crop.size == 0 or self.is_empty:
            return []
        assert_embedding_version(
            self.embedding_version, embedding_version, "LogoRetrievalIndex"
        )
        k = top_k or self.top_k
        q = self._embed_image_batch([crop])[0]          # (D,)
        sims = self._embeddings @ q                      # (N,)
        # Best similarity per brand.
        brand_sim: Dict[str, float] = {}
        for s, brand in zip(sims.tolist(), self._brand_of_row):
            s = float(s)
            if s < self.min_similarity:
                continue
            if s > brand_sim.get(brand, -1.0):
                brand_sim[brand] = s
        ranked = sorted(brand_sim.items(), key=lambda kv: kv[1], reverse=True)
        return [
            (brand, round(s, 4)) for brand, s in ranked[: max(k, 0)]
        ]

    @staticmethod
    def build_from_dir(
        reference_dir: str,
        device: str = "cpu",
        checkpoint_path: Optional[str] = None,
    ) -> "LogoRetrievalIndex":
        """Convenience builder: one brand per subdirectory of reference_dir.

        Each subdirectory name is the canonical brand (uppercase); every image
        inside contributes a reference crop. Mirrors the product/index layout
        expectations. Returns an index (possibly empty if the dir is missing).
        """
        index = LogoRetrievalIndex(
            reference_dir=reference_dir,
            checkpoint_path=checkpoint_path,
        )
        base = Path(reference_dir)
        if not base.is_dir():
            logger.warning("LogoRetrievalIndex: reference dir missing: %s", base)
            return index
        import cv2

        for sub in sorted(base.iterdir()):
            if not sub.is_dir():
                continue
            brand = _canonical(sub.name)
            crops = []
            for imgf in sorted(sub.glob("*")):
                if imgf.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
                    continue
                img = cv2.imread(str(imgf))
                if img is None:
                    continue
                crops.append(img)
            if crops:
                index.add_brand(brand, crops)
        return index
