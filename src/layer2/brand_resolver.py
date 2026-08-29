"""
Layer 2c — Brand Resolution + Temporal Memory (lightweight).

Two jobs, both fixing the "text logo but not name" problem:

1. BrandResolver — converts generic logo detections ('text logo', 'brand logo')
   into actual brand names. Strategy, in priority order:
     a. The detection's own class_name already names a brand ('Samsung logo').
     b. Otherwise crop the logo bounding box and OCR the crop; the recognized
        text is matched against the brand catalog aliases.
     c. Otherwise the detection stays unresolved and is NOT reported as a brand
        product (avoids "TEXT LOGO" showing up as a brand name).

2. build_brand_timeline — temporal memory / cross-scene reasoning (prompt §5).
   Every resolved brand entity accumulates a running memory of appearances
   (frame index, timestamp, modality source). A speech mention of a brand that
   was also visually established is flagged cross_scene=True — the mention is
   resolved against stored visual context rather than treated independently.

The crop-OCR step reuses the already-loaded PaddleOCR extractor, so it adds no
new models — only a bounded amount of per-logo inference.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.brand_catalog import match_brand

logger = logging.getLogger(__name__)

# Generic labels that carry no brand identity and require OCR resolution.
GENERIC_LOGO_LABELS = {
    "brand logo", "company logo", "text logo", "product logo",
    "label", "wordmark", "logo", "brand",
}


class BrandResolver:
    """Resolve logo detections to canonical brand names."""

    # A per-brand class label ('SAMSUNG logo', 'Nike logo') from the detector is
    # only trusted as brand evidence above this confidence. Below it, the label
    # is treated as unconfirmed and we fall through to crop-OCR. This suppresses
    # YOLO-World's low-confidence spurious brand hits (e.g. 'SUPREME logo' on a
    # Samsung/Sony/Apple logo) that otherwise resolve to the wrong brand.
    DEFAULT_CLASS_CONFIDENCE = 0.40

    # Crop upscale factor applied before OCR so small/low-res logo text is more
    # likely to be read (the vanilla crop on a phone video can be tiny).
    DEFAULT_CROP_SCALE = 2.0

    def __init__(
        self,
        ocr_extractor=None,
        class_confidence=DEFAULT_CLASS_CONFIDENCE,
        crop_scale=DEFAULT_CROP_SCALE,
        retrieval_index=None,
        retrieval_min_similarity: float = 0.22,
    ):
        self.ocr = ocr_extractor
        self.class_confidence = float(class_confidence)
        self.crop_scale = float(crop_scale)
        # Optional CLIP logo-retrieval index (Phase 1/2). When provided it is
        # used as the icon-only fallback: if neither the class label nor OCR
        # names a brand, the crop is matched by CLIP retrieval against a
        # per-brand reference bank. Retrieval is the PRIMARY classifier for
        # icon-only / stylized marks where there is no readable text.
        self.retrieval_index = retrieval_index
        self.retrieval_min_similarity = float(retrieval_min_similarity)

    @staticmethod
    def _crop(frame: np.ndarray, bbox) -> Optional[np.ndarray]:
        """Crop a bounding box from a frame with a small margin."""
        if frame is None or frame.size == 0 or bbox is None:
            return None
        try:
            x1, y1, x2, y2 = (int(v) for v in bbox)
        except (TypeError, ValueError):
            return None
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        pad_x = int((x2 - x1) * 0.1) + 1
        pad_y = int((y2 - y1) * 0.1) + 1
        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        return frame[y1:y2, x1:x2]

    @staticmethod
    def _upscale(crop: np.ndarray, scale: float) -> np.ndarray:
        """Upscale a crop (bicubic) to aid small-text OCR.

        Only meaningful (and cheap) for genuinely small crops. Large crops —
        e.g. a detection box covering most of a frame — are already high-res
        and upscaling them just pushes PaddleOCR past its side limit and burns
        inference time. So the scale is applied only when the crop's max
        dimension is under a small threshold (SMALL_MAX), and the result is
        additionally capped so it never exceeds LARGE_MAX.
        """
        if scale <= 1.0:
            return crop
        import cv2

        SMALL_MAX = 192
        LARGE_MAX = 1200
        h, w = crop.shape[:2]
        if max(h, w) > SMALL_MAX:
            return crop
        nh, nw = int(h * scale), int(w * scale)
        if max(nh, nw) > LARGE_MAX:
            ratio = LARGE_MAX / max(nh, nw)
            nh, nw = int(nh * ratio), int(nw * ratio)
        return cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_CUBIC)

    def _ocr_crop_texts(self, crop: np.ndarray) -> List[str]:
        """OCR a crop and return the recognized text list.

        Tries the upscaled crop first (best for small logo text), and falls back
        to the original if upscaling produced nothing. Dedupes while preserving
        order.
        """
        if self.ocr is None or crop is None:
            return []
        candidates = []
        scaled = self._upscale(crop, self.crop_scale)
        candidates.append(scaled)
        if scaled is not crop:
            candidates.append(crop)
        seen = set()
        texts: List[str] = []
        for img in candidates:
            try:
                results = self.ocr.extract_text(img)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Crop-OCR failed for logo box: %s", exc)
                continue
            for r in results:
                t = r.get("text", "")
                if t and t not in seen:
                    seen.add(t)
                    texts.append(t)
            if texts:
                break
        return texts

    def _resolve_detection(self, det: dict, frame: np.ndarray) -> dict:
        """Resolve a single logo detection to a canonical brand name."""
        out = dict(det)
        out["brand"] = None
        class_name = str(det.get("class_name") or "")
        confidence = float(det.get("confidence", 0.0))
        class_brand = match_brand(class_name)

        # 1. Gather signals ONCE from the logo crop, reused by every path:
        #      * OCR  — reads the real on-screen wordmark (ground-truth-adjacent)
        #      * CLIP — image retrieval against the per-brand reference bank
        #      * class — YOLO-World's zero-shot brand guess (LEAST trustworthy:
        #        it confidently mislabels icon/stylized marks, e.g. "SUPREME logo"
        #        on a Samsung foldable at 0.45-0.60).
        #    Phase 3 visibility: every resolution records all three signals so a
        #    disagreement (the SUPREME-style failure mode) is never silent.
        crop = self._crop(frame, det.get("bbox"))
        ocr_brand, joined_ocr = None, ""
        retrieval_candidates: List[Tuple[str, float]] = []
        retrieval_brand, retrieval_sim = None, 0.0
        if crop is not None:
            if self.ocr is not None:
                texts = self._ocr_crop_texts(crop)
                joined_ocr = " ".join(texts)
                if texts:
                    ocr_brand = match_brand(joined_ocr)
                if texts and not ocr_brand:
                    self._log_near_misses(joined_ocr, class_name, confidence)
            if self.retrieval_index is not None and not self.retrieval_index.is_empty:
                retrieval_candidates = self._retrieval_query(crop)
                if retrieval_candidates:
                    retrieval_brand, retrieval_sim = retrieval_candidates[0]
                    retrieval_candidates = retrieval_candidates[:3]

        # 2. Fusion priority (plan §11): OCR wins when text is present and names
        #    a brand; else CLIP retrieval (icon-only / corrects spurious class
        #    labels); else the class label above the confidence gate; else keep
        #    the detection unresolved.
        if ocr_brand:
            out["brand"] = ocr_brand
            out["class_name"] = ocr_brand
            out["class_confirmed"] = False
            out["resolution_source"] = "ocr"
            out["ocr_text"] = joined_ocr[:40]
            out["retrieval_top3"] = retrieval_candidates
            if class_brand and class_brand != ocr_brand:
                out["resolved_vs_class"] = {
                    "class_brand": class_brand,
                    "class_confidence": round(confidence, 3),
                }
            logger.info(
                "Brand resolved via OCR: class='%s' (conf=%.2f) ocr_text=%r -> %s",
                class_name, confidence, joined_ocr[:40], ocr_brand,
            )
            return out

        if retrieval_brand:
            out["brand"] = retrieval_brand
            out["class_name"] = retrieval_brand
            out["class_confirmed"] = False
            out["resolution_source"] = "clip_retrieval"
            out["retrieval_top3"] = retrieval_candidates
            out["retrieval_similarity"] = retrieval_sim
            if class_brand and class_brand != retrieval_brand:
                out["resolved_vs_class"] = {
                    "class_brand": class_brand,
                    "class_confidence": round(confidence, 3),
                }
            logger.info(
                "Brand resolved via CLIP retrieval: class='%s' (conf=%.2f) "
                "crop->%s sim=%.3f candidates=%s",
                class_name, confidence, retrieval_brand, retrieval_sim,
                retrieval_candidates,
            )
            return out

        if class_brand and confidence >= self.class_confidence:
            out["brand"] = class_brand
            out["class_name"] = class_brand
            out["class_confirmed"] = True
            if retrieval_candidates:
                out["retrieval_top3"] = retrieval_candidates
            logger.debug(
                "Resolver: class '%s' -> %s (conf=%.2f, gate %.2f)",
                class_name, class_brand, confidence, self.class_confidence,
            )
            return out

        # 3. Unresolved — keep the raw label, brand stays None so it is
        #    excluded from brand products and recommendations.
        return out

    def _retrieval_query(self, crop: np.ndarray) -> List[Tuple[str, float]]:
        """Top retrieval candidates for a crop, filtered by min similarity.

        Returns [(brand, similarity), ...] desc by similarity, truncated to the
        index's top-k. Logs the top-3 with the YOLO class guess so a wrong
        zero-shot label is visible against the retrieval signal (Phase 3).
        """
        if self.retrieval_index is None:
            return []
        candidates = self.retrieval_index.query(crop)
        out = [
            (b, float(s)) for (b, s) in candidates
            if float(s) >= self.retrieval_min_similarity
        ]
        if out:
            logger.debug(
                "Retrieval top-3: %s", [(b, round(s, 3)) for b, s in out[:3]]
            )
        return out

    def _log_near_misses(self, joined: str, class_name: str, confidence: float) -> None:
        """Log OCR/class text that nearly matches a catalog brand (diagnostics)."""
        from src.brand_catalog import BRAND_CATALOG, normalize_text, _fuzzy_token_match

        norm = normalize_text(joined)
        near = []
        for token in set(norm.split()):
            if not token:
                continue
            hit = _fuzzy_token_match(token, 1)
            if hit:
                near.append({"token": token, "brand": hit[0], "dist": hit[1]})
        if near:
            logger.info(
                "Resolver near-miss: class='%s' (conf=%.2f) ocr_text=%r nearly "
                "matches %s",
                class_name, confidence, joined[:48],
                [(n["brand"], n["dist"]) for n in near],
            )

    def resolve(
        self,
        logo_detections: List[List[dict]],
        frames: List[np.ndarray],
    ) -> List[List[dict]]:
        """Resolve all logo detections across frames.

        Args:
            logo_detections: per-frame lists of logo detections
            frames: full list of RGB frames (same indexing)

        Returns:
            New per-frame detection lists, each detection carrying a `brand`
            key (None when unresolved) and `class_name` set to the canonical
            brand when resolved.
        """
        resolved = []
        n_resolved = 0
        n_total = 0
        class_outcomes: Dict[str, int] = {}
        resolved_by_class: Dict[str, str] = {}
        for frame, frame_dets in zip(frames, logo_detections):
            out = []
            for det in frame_dets:
                n_total += 1
                r = self._resolve_detection(det, frame)
                cls = str(det.get("class_name") or "?")
                class_outcomes[cls] = class_outcomes.get(cls, 0) + 1
                if r.get("brand"):
                    n_resolved += 1
                    resolved_by_class.setdefault(cls, r["brand"])
                out.append(r)
            resolved.append(out)
        logger.info(
            "Brand resolution: %d/%d logo detections mapped to a brand (gate=%.2f)",
            n_resolved, n_total, self.class_confidence,
        )
        detail = "; ".join(
            f"{cls}x{n}->{resolved_by_class.get(cls, 'UNRESOLVED')}"
            for cls, n in sorted(class_outcomes.items())
        )
        if detail:
            logger.info("Brand resolution per input class: %s", detail)
        return resolved


def build_brand_timeline(
    resolved_logos: List[List[dict]],
    brand_mentions: List[dict],
    video_fps: float = 0.0,
    transcript: Optional[str] = None,
    transcript_duration: Optional[float] = None,
) -> Dict[str, dict]:
    """Build a temporal memory of brand appearances (Layer 2c).

    Each brand accumulates a memory bank of appearances with modality source
    ('logo' from the visual track, 'speech' from ASR mentions). Cross-scene
    resolution: brands established both visually and verbally are flagged
    cross_scene=True, matching the prompt's "this phone at minute 9 → the
    Apple logo shown at minute 1" scenario.

    Returns a dict keyed by canonical brand:
        {
            "brand": str,
            "appearance_count": int,
            "first_seen": float|None, "last_seen": float|None,
            "modalities": [...], "cross_scene": bool,
            "appearances": [{frame_index, timestamp, modality, confidence}]
        }
    """
    timeline: Dict[str, dict] = {}

    def _entry(brand: str) -> dict:
        if brand not in timeline:
            timeline[brand] = {
                "appearances": [],
                "modalities": set(),
            }
        return timeline[brand]

    for idx, frame_dets in enumerate(resolved_logos):
        for det in frame_dets:
            brand = det.get("brand")
            if not brand:
                continue
            entry = _entry(brand)
            entry["appearances"].append({
                "frame_index": idx,
                "timestamp": round(idx / video_fps, 1) if video_fps else idx,
                "modality": "logo",
                "confidence": round(float(det.get("confidence", 0.0)), 3),
            })
            entry["modalities"].add("logo")

    for mention in brand_mentions:
        brand = mention.get("brand")
        if not brand:
            continue
        entry = _entry(brand)
        ts = None
        if transcript and transcript_duration and len(transcript) > 0:
            ts = transcript_duration * (
                mention.get("position", 0) / len(transcript)
            )
        entry["appearances"].append({
            "frame_index": None,
            "timestamp": round(ts, 1) if ts is not None else None,
            "modality": "speech",
            "confidence": 1.0,
        })
        entry["modalities"].add("speech")

    out: Dict[str, dict] = {}
    for brand, entry in timeline.items():
        apps = entry["appearances"]
        modalities = sorted(entry["modalities"])
        timestamps = [a["timestamp"] for a in apps if a.get("timestamp") is not None]
        out[brand] = {
            "brand": brand,
            "appearance_count": len(apps),
            "first_seen": timestamps[0] if timestamps else None,
            "last_seen": timestamps[-1] if timestamps else None,
            "modalities": modalities,
            "cross_scene": "logo" in modalities and "speech" in modalities,
            "appearances": apps,
        }
    return out


def brand_evidence_from_timeline(
    timeline: Dict[str, dict],
) -> Dict[str, float]:
    """Aggregate a per-brand evidence strength (0-1) from the timeline."""
    evidence: Dict[str, float] = {}
    for brand, entry in timeline.items():
        logo_confs = [
            a["confidence"] for a in entry["appearances"]
            if a["modality"] == "logo" and a.get("confidence") is not None
        ]
        base = float(np.mean(logo_confs)) if logo_confs else 0.0
        if "speech" in entry["modalities"]:
            base = max(base, 0.6)
        evidence[brand] = round(min(1.0, base), 3)
    return evidence
