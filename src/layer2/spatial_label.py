"""
Layer 2c — Spatial / Temporal label stabilization for scene objects.

Secondary remediation (ad-hoc, from the Samsung review regression):

1. TemporalObjectSmoother — the generic COCO detector (YOLOv8x) flips the
   label of one physical device across adjacent frames (e.g. scenes 013-017:
   backpack -> remote -> mouse -> mouse+laptop). Since the device is spatially
   stationary, we track boxes across consecutive sampled frames by box
   overlap/center proximity (ignoring the label), then assign each track its
   majority label. This suppresses isolated flapping labels without hurting the
   genuine per-frame detections.

2. apply_spatial_brand_context — once a brand is resolved (logo crop -> brand
   name with a frame + bbox), any COCO object detection that overlaps that
   brand's on-screen location in nearby frames is the SAME physical device. It
   is tagged with `brand_context` so the contradictory/flapping object label is
   anchored to the resolved brand instead of standing alone as noise. The label
   itself is kept (honesty), but the brand association is attached.

Both are pure functions of detections/boxes and are unit-testable.
"""

import logging
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


def _center(bbox) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _box_size(bbox) -> float:
    x1, y1, x2, y2 = bbox
    return max(1.0, max(x2 - x1, y2 - y1))


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    union = a_area + b_area - inter
    if union <= 0:
        return 0.0
    return inter / union


def _match_threshold(bbox) -> float:
    """Center-distance threshold for calling two boxes the same track.

    Scaled to box size so small devices (mouse) track tightly while larger
    objects tolerate more drift between adjacent sampled frames.
    """
    return max(12.0, _box_size(bbox) * 0.6)


class TemporalObjectSmoother:
    """Suppress flapping COCO labels across adjacent sampled frames."""

    def __init__(self, track_distance_factor: float = 0.6, min_track_len: int = 2):
        self.track_distance_factor = track_distance_factor
        self.min_track_len = min_track_len

    def smooth(
        self,
        detections: Sequence[Sequence[dict]],
    ) -> Sequence[Sequence[dict]]:
        """Return per-frame detection lists with flapping labels smoothed.

        Each detection dict that belongs to a spatial track spanning >=
        `min_track_len` frames has its `class_name` replaced by the majority
        label of that track (isolated flips are suppressed). The original label
        is preserved as `smoothed_label` when it differed, so the raw signal is
        not lost.
        """
        # Rebuild tracks deterministically, recording per (frame, idx) the run
        # of boxes it belongs to, then majority-vote each run's label.
        builds: List[Dict[int, List[dict]]] = [{} for _ in detections]
        active: List[dict] = []
        for f_idx, frame_dets in enumerate(detections):
            new_active: List[dict] = []
            for i, det in enumerate(frame_dets):
                bbox = det.get("bbox")
                if not bbox:
                    continue
                c = _center(bbox)
                thr = max(12.0, _box_size(bbox) * self.track_distance_factor)
                best, best_d = None, float("inf")
                for t in active:
                    d = ((t["center"][0] - c[0]) ** 2 + (t["center"][1] - c[1]) ** 2) ** 0.5
                    if d < best_d:
                        best_d, best = d, t
                if best is not None and best_d <= thr:
                    best["center"] = ((best["center"][0] + c[0]) / 2, (best["center"][1] + c[1]) / 2)
                    best["boxes"].append({"frame": f_idx, "i": i, "label": det.get("class_name", "")})
                    builds[f_idx][i] = best["boxes"]
                    active.remove(best)
                    new_active.append(best)
                else:
                    t = {
                        "center": c,
                        "boxes": [{"frame": f_idx, "i": i, "label": det.get("class_name", "")}],
                    }
                    builds[f_idx][i] = t["boxes"]
                    new_active.append(t)
            active = new_active

        out: List[List[dict]] = [list(fd) for fd in detections]
        for f_idx, frame_dets in enumerate(detections):
            for i, det in enumerate(frame_dets):
                boxes = builds[f_idx].get(i)
                if not boxes or len(boxes) < self.min_track_len:
                    continue
                counts: Dict[str, int] = {}
                for b in boxes:
                    lbl = b["label"] or "?"
                    counts[lbl] = counts.get(lbl, 0) + 1
                majority = max(counts, key=lambda k: counts[k])
                if majority and majority != det.get("class_name"):
                    out[f_idx][i] = dict(det)
                    out[f_idx][i]["class_name"] = majority
                    out[f_idx][i]["smoothed_label"] = det.get("class_name", "")
                    out[f_idx][i]["track_len"] = len(boxes)
        return out


def apply_spatial_brand_context(
    detections: Sequence[Sequence[dict]],
    resolved_logos: Sequence[Sequence[dict]],
    window: int = 3,
    min_iou: float = 0.05,
) -> Sequence[Sequence[dict]]:
    """Tag COCO object detections that overlap a resolved brand's on-screen
    location in nearby frames with `brand_context`.

    A resolved brand (logo -> brand, carrying `brand` + `bbox` on a frame)
    marks the spatial region of the physical device. Object detections in frames
    within +/- `window` whose box overlaps that region (IoU >= min_iou, or whose
    center falls inside the brand box) are the same device, so the flapping /
    contradictory COCO label is anchored to the brand instead of standing alone.

    Object detection dicts are copied and annotated; resolved brands are read
    but not mutated.
    """
    # Collect (frame, bbox, brand) for resolved logos.
    brand_regions: List[Tuple[int, List[float], str]] = []
    for f_idx, frame_logos in enumerate(resolved_logos):
        for det in frame_logos:
            brand = det.get("brand")
            bbox = det.get("bbox")
            if brand and bbox:
                brand_regions.append((f_idx, bbox, brand))

    if not brand_regions:
        return detections

    out: List[List[dict]] = [list(fd) for fd in detections]
    brand_centers: Dict[str, Tuple[float, float]] = {
        b: _center(bb) for _, bb, b in brand_regions
    }

    for f_idx, frame_dets in enumerate(detections):
        for i, det in enumerate(frame_dets):
            dbox = det.get("bbox")
            if not dbox:
                continue
            for (b_f, bbox, brand) in brand_regions:
                if abs(f_idx - b_f) > window:
                    continue
                overlap = _iou(dbox, bbox)
                dc = _center(dbox)
                in_box = (bbox[0] - 4 <= dc[0] <= bbox[2] + 4 and
                          bbox[1] - 4 <= dc[1] <= bbox[3] + 4)
                if overlap >= min_iou or in_box:
                    out[f_idx][i] = dict(out[f_idx][i])
                    out[f_idx][i]["brand_context"] = brand
                    break
    return out


# Small curated brand -> plausible product (COCO) category table (Phase 2c).
# A detection anchored to a brand (via `brand_context`) whose COCO label is NOT
# in the brand's plausible product set is flagged / reclassified, so a foldable
# phone being flapped to `mouse`/`remote`/`backpack` by the generic detector is
# corrected to the brand's primary product instead of standing alone as noise.
# Only brands we can reason about are listed; unknown brands are left untouched.
BRAND_PRODUCT_CATEGORIES: Dict[str, Dict[str, object]] = {
    "SAMSUNG": {
        "plausible": {"cell phone", "laptop", "tv"},
        "primary_label": "cell phone",
    },
    "APPLE": {
        "plausible": {"cell phone", "laptop", "tv"},
        "primary_label": "cell phone",
    },
    "MICROSOFT": {
        "plausible": {"laptop", "keyboard", "tv"},
        "primary_label": "laptop",
    },
    "SONY": {
        "plausible": {"cell phone", "tv", "laptop"},
        "primary_label": "cell phone",
    },
    # Brands without a curated category table are left untouched (no guessing).
}


def normalize_brand_name(brand: str) -> str:
    """Canonicalize a resolved brand name for table lookup.

    Resolved brands arrive in various forms ('Samsung', 'SAMSUNG', 'samsung').
    Map them to the canonical uppercase key used in BRAND_PRODUCT_CATEGORIES.
    """
    key = (brand or "").strip().upper()
    if key in BRAND_PRODUCT_CATEGORIES:
        return key
    # Fall back to a case-insensitive prefix match against the table keys.
    for table_key in BRAND_PRODUCT_CATEGORIES:
        if table_key.startswith(key) or key.startswith(table_key):
            return table_key
    return key


class SceneConsistencyResolver:
    """Layer 2c — brand-anchored contradiction correction (Phase 3).

    Fixes the same-device label flapping that TemporalObjectSmoother can't: a
    Microsoft logo resolved on a laptop being ALSO reported as `cell phone`, or
    a Samsung foldable phone being flapped to `mouse`/`remote`/`backpack`.

    Uses a small curated brand -> plausible-product table. A detection that:
      * carries a `brand_context` (spatially overlapped a resolved brand), AND
      * has a COCO `class_name` NOT in that brand's plausible product set, AND
      * clears `min_brand_confidence` (the underlying resolution is solid),
    is reclassified to the brand's `primary_label`. Both `raw_class` (original
    COCO label) and `corrected_class` are kept on the record for audit — the
    disagreement is never silently discarded.

    Reclassifying happens in-place on copied detection dicts (read-only input).
    """

    def __init__(
        self,
        min_brand_confidence: float = 0.30,
        min_detection_confidence: float = 0.10,
        categories: Optional[Dict[str, Dict[str, object]]] = None,
    ):
        self.min_brand_confidence = min_brand_confidence
        self.min_detection_confidence = min_detection_confidence
        self.categories = categories if categories is not None else BRAND_PRODUCT_CATEGORIES

    def resolve(
        self,
        detections: Sequence[Sequence[dict]],
        resolved_logos: Sequence[Sequence[dict]],
    ) -> Sequence[Sequence[dict]]:
        """Correct implausible brand-anchored object labels (Phase 2c).

        Args:
            detections: per-frame detection lists (post brand-context tagging).
            resolved_logos: per-frame brand-resolved logo detections.

        Returns new per-frame detection lists. Detections that are reclassified
        gain `raw_class`, `corrected_class`, and `reclassified_by`; the original
        COCO class is preserved as `raw_class`.
        """
        # Index resolved logos by (frame -> list of (brand, confidence)).
        brand_scores: Dict[int, List[Tuple[str, float]]] = {}
        for f_idx, frame_logos in enumerate(resolved_logos):
            for det in frame_logos:
                brand = normalize_brand_name(det.get("brand"))
                if brand in self.categories:
                    brand_scores.setdefault(f_idx, []).append(
                        (brand, float(det.get("confidence", 0.0)))
                    )
        if not brand_scores:
            return detections

        out: List[List[dict]] = [list(fd) for fd in detections]
        for f_idx, frame_dets in enumerate(detections):
            # Closest-in-time resolved brand window for this frame, so a brand
            # on an adjacent frame still anchors detections near it.
            best_brand, best_conf, best_delta = None, 0.0, None
            for b_f, entries in brand_scores.items():
                delta = abs(f_idx - b_f)
                # strongest-resolved brand closest in time anchors the frame
                if best_delta is None or delta < best_delta:
                    best_delta = delta
                    brand, conf = max(entries, key=lambda e: e[1])
                    best_brand, best_conf = brand, conf
            if best_delta is None or best_brand is None or best_delta > self._window:
                continue
            if best_conf < self.min_brand_confidence:
                continue

            spec = self.categories[best_brand]
            plausible = spec["plausible"]
            primary = spec.get("primary_label")

            for i, det in enumerate(frame_dets):
                cls = det.get("class_name", "")
                if det.get("brand_context") != best_brand:
                    continue
                if cls in plausible:
                    continue  # a plausible product (laptop/tv) is left alone
                if float(det.get("confidence", 0.0)) < self.min_detection_confidence:
                    continue
                if not primary or primary == cls:
                    continue
                out[f_idx][i] = dict(out[f_idx][i])
                out[f_idx][i]["raw_class"] = cls
                out[f_idx][i]["class_name"] = primary
                out[f_idx][i]["corrected_class"] = primary
                out[f_idx][i]["reclassified_by"] = "SceneConsistencyResolver"
                out[f_idx][i]["brand_context_consistency"] = "corrected"
        return out

    # Adjacent-frame brand window used for anchoring detections to a brand.
    _window = 3
