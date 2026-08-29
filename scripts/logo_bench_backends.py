"""
Benchmark detection backends for the logo-detection rebuild (Part B).

Each backend implements the same interface:
    detect(image_bgr_or_rgb, text_queries=None) -> List[detections]
    detect_batch(frames, text_queries=None) -> List[List[detections]]
    name: str   # used as the results key in the benchmark report

detections: list of {"bbox":[x1,y1,x2,y2], "confidence": float,
                     "brand": canonical brand or None, "text_prompt": ...}

Implemented paradigms:
  * RegionProposalCLIPBackend — selective-search region proposals scored with
    CLIP against per-brand text prompts (zero-shot, no training).
  * SIFTLogoMatcherBackend — SIFT keypoint matching against a reference logo
    bank (baseline, no training, needs reference images).
  * YOLOWorldBackend — wraps the production YOLO-World detector (the current
    pipeline backend) as a benchmark participant.

Backends are deliberately kept in scripts/ (benchmark-only) — they are not
part of the production pipeline data path.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _to_bgr(image: np.ndarray) -> np.ndarray:
    """Normalize to BGR for OpenCV (detectors assume BGR input)."""
    if image.ndim != 3:
        return image
    h, w, c = image.shape
    if c == 4:
        return image[:, :, :3]
    return image


class RegionProposalCLIPBackend:
    """Selective-search proposals + CLIP zero-shot brand classification."""

    name = "region_proposal_clip"

    def __init__(
        self,
        device: str = "cpu",
        conf_threshold: float = 0.30,
        top_k: int = 200,
        min_proposal_area: float = 0.01,
        checkpoint_path: Optional[str] = None,
    ):
        self.conf_threshold = conf_threshold
        self.top_k = top_k
        self.min_proposal_area = min_proposal_area
        self.device = device
        self._clip = None
        self._use_open_clip = False
        self._preprocess = None
        self._checkpoint_path = checkpoint_path or os.environ.get("OPEN_CLIP_WEIGHTS")
        self._brand_names: List[str] = []

    def _load(self):
        if self._clip is None:
            import torch

            if self._checkpoint_path and os.path.exists(self._checkpoint_path):
                import open_clip

                model, _, preprocess = open_clip.create_model_and_transforms(
                    "ViT-B-32", pretrained=self._checkpoint_path, device=self.device
                )
                model.eval()
                self._clip = model
                self._use_open_clip = True
                self._preprocess = preprocess
            else:
                import clip

                model, preprocess = clip.load("ViT-B/32", device=self.device)
                self._clip = model
                self._use_open_clip = False
                self._preprocess = preprocess
        return self._clip

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        import torch

        clip = self._clip
        with torch.no_grad():
            if self._use_open_clip:
                import open_clip

                tok = open_clip.tokenize(texts).to(self.device)
                feats = clip.encode_text(tok).cpu().numpy()
            else:
                import clip as _openai_clip

                tok = _openai_clip.tokenize(texts)
                feats = clip.encode_text(tok).cpu().numpy()
        feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
        return feats

    def _proposals(self, img_bgr: np.ndarray) -> List[np.ndarray]:
        import cv2

        h, w = img_bgr.shape[:2]
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(img_bgr)
        ss.switchToSelectiveSearchFast()
        rects = ss.process()
        min_area = self.min_proposal_area * h * w
        out = []
        for x, y, rw, rh in rects[:self.top_k * 4]:
            if rw * rh < min_area:
                continue
            x2, y2 = min(w, x + rw), min(h, y + rh)
            if x2 - x < 4 or y2 - y < 4:
                continue
            out.append([int(x), int(y), int(x2), int(y2)])
            if len(out) >= self.top_k:
                break
        return out

    def _classify_proposal(
        self, img_bgr: np.ndarray, box: List[int], feats: np.ndarray, text_queries: List[str]
    ) -> tuple:
        """Return (max brand similarity, argmax brand) for a proposal box."""
        x1, y1, x2, y2 = box
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return 0.0, None
        f = self._embed_image(crop)
        sims = feats @ f  # (n_brands,)
        best_idx = int(np.argmax(sims))
        best = float(sims[best_idx])
        brand = None
        if text_queries:
            from src.brand_catalog import match_brand

            brand = match_brand(text_queries[best_idx]) or text_queries[best_idx]
        return best, brand

    def _embed_image(self, img_bgr: np.ndarray) -> np.ndarray:
        import torch
        from PIL import Image

        rgb = np.ascontiguousarray(img_bgr[:, :, ::-1])
        pil = Image.fromarray(rgb)
        x = self._preprocess(pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            f = self._clip.encode_image(x).cpu().numpy()[0]
        return f / np.linalg.norm(f)

    def detect(self, image, text_queries: Optional[List[str]] = None):
        self._load()
        if text_queries is None:
            text_queries = []
        feats = self._embed_texts(text_queries)
        img = _to_bgr(np.asarray(image))
        boxes = self._proposals(img)

        # Batch all proposal crops through the CLIP image encoder in one
        # forward pass instead of one pass per crop.
        import torch
        from PIL import Image

        crops = []
        valid_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            rgb = np.ascontiguousarray(crop[:, :, ::-1])
            crops.append(self._preprocess(Image.fromarray(rgb)))
            valid_boxes.append(box)
        detections = []
        if crops:
            batch = torch.stack(crops).to(self.device)
            with torch.no_grad():
                img_feats = self._clip.encode_image(batch).cpu().numpy()
            img_feats = img_feats / np.linalg.norm(img_feats, axis=1, keepdims=True)
            sims = img_feats @ feats.T  # (n_proposals, n_brands)
            for box, row in zip(valid_boxes, sims):
                best_idx = int(np.argmax(row))
                conf = float(row[best_idx])
                if conf < self.conf_threshold:
                    continue
                from src.brand_catalog import match_brand

                brand = match_brand(text_queries[best_idx]) or text_queries[best_idx]
                detections.append({
                    "bbox": box,
                    "confidence": conf,
                    "brand": brand,
                    "text_prompt": brand,
                })
        return detections

    def detect_batch(self, frames: List[np.ndarray], text_queries=None, batch_size=8):
        return [self.detect(f, text_queries) for f in frames]


class SIFTLogoMatcherBackend:
    """SIFT keypoint matching against a reference logo bank (baseline)."""

    name = "sift_match"

    def __init__(
        self,
        reference_bank_dir: str,
        conf_threshold: float = 0.05,
        top_k: int = 200,
    ):
        self.conf_threshold = conf_threshold
        self.top_k = top_k
        self.reference_bank_dir = reference_bank_dir
        self._refs = None  # list of {"brand", "descriptors", "kp", "img"}

    def _load_refs(self):
        if self._refs is not None:
            return self._refs
        import cv2
        from pathlib import Path

        bank = Path(self.reference_bank_dir)
        refs = []
        if bank.is_dir():
            for p in sorted(bank.glob("*.png")) + sorted(bank.glob("*.jpg")):
                img = cv2.imread(str(p))
                if img is None:
                    continue
                sift = cv2.SIFT_create()
                kp, des = sift.detectAndCompute(img, None)
                if des is None or len(kp) < 3:
                    continue
                refs.append({"brand": p.stem.upper(), "kp": kp, "des": des, "img": img})
        self._refs = refs
        return refs

    @staticmethod
    def _canonical_brand(stem_upper: str) -> str:
        from src.brand_catalog import match_brand

        return match_brand(stem_upper) or stem_upper.split("_")[0]

    def detect(self, image, text_queries: Optional[List[str]] = None):
        import cv2

        refs = self._load_refs()
        if not refs:
            return []
        img = _to_bgr(np.asarray(image))
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(img, None)
        if des is None or len(kp) < 5:
            return []

        # FLANN matcher with Lowe ratio test per reference logo.
        index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)

        detections = []
        for ref in refs:
            matches = matcher.knnMatch(ref["des"], des, k=2)
            good = []
            for pair in matches:
                if len(pair) == 2:
                    m, n = pair
                    if m.distance < 0.75 * n.distance:
                        good.append(m)
            if len(good) < 4:
                continue
            # Homography-based box: find region in query matching the reference.
            src_pts = np.float32([ref["kp"][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            try:
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            except cv2.error:
                continue
            if H is None or mask is None:
                continue
            h, w = ref["img"].shape[:2]
            corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
            x1, y1 = int(min(dst[:, 0])), int(min(dst[:, 1]))
            x2, y2 = int(max(dst[:, 0])), int(max(dst[:, 1]))
            img_h, img_w = img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)
            if x2 - x1 < 5 or y2 - y1 < 5:
                continue
            conf = float(len(good)) / 100.0  # normalized match count baseline
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "brand": self._canonical_brand(ref["brand"]),
                "text_prompt": f"sift:{ref['brand']}",
            })
        return detections

    def detect_batch(self, frames: List[np.ndarray], text_queries=None, batch_size=8):
        return [self.detect(f, text_queries) for f in frames]


class YOLOWorldBackend:
    """Wraps the production YOLO-World logo detector as a benchmark participant."""

    name = "yolo_world"

    def __init__(
        self,
        model_name: str = "yolov8s-worldv2.pt",
        confidence_threshold: float = 0.10,
        device: Optional[str] = None,
    ):
        import torch

        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        from src.layer1.logo_detector import YOLOWorldLogoDetector

        self._detector = YOLOWorldLogoDetector(
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            device=self.device,
            text_queries=[],
        )
        self._queries: Optional[List[str]] = None

    def _set_queries(self, queries: List[str]):
        if queries and queries != self._queries:
            self._detector.model.set_classes(queries)
            self._queries = list(queries)

    def detect(self, image, text_queries: Optional[List[str]] = None):
        if text_queries:
            self._set_queries(text_queries)
        dets = self._detector.detect(np.asarray(image), text_queries or None)
        for d in dets:
            d["brand"] = self._resolve_brand(d.get("class_name") or d.get("text_prompt"))
        return dets

    @staticmethod
    def _resolve_brand(class_name: Optional[str]) -> Optional[str]:
        if not class_name:
            return None
        from src.brand_catalog import match_brand

        return match_brand(class_name)

    def detect_batch(self, frames: List[np.ndarray], text_queries=None, batch_size=8):
        if text_queries:
            self._set_queries(text_queries)
        out = self._detector.detect_batch(
            [np.asarray(f) for f in frames], text_queries or None, batch_size
        )
        for dets in out:
            for d in dets:
                d["brand"] = self._resolve_brand(d.get("class_name") or d.get("text_prompt"))
        return out
