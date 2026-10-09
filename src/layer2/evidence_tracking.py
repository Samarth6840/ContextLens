"""Object/event evidence tracks — association across frames.

The fusion ledger treats a brand's evidence as a bag of per-frame votes. Two
frames of ONE physical object (a logo held on screen for 40 frames) therefore
look like 40 independent corroborating votes, which inflates a static overlay
and splits a moving object's contribution across buckets. This module marks the
same physical object across frames:

    detections -> greedy IoU/center association -> track_id per detection

Tracks are deliberately LABEL-AGNOSTIC (association uses boxes only, never the
resolved brand) so a track whose brand flickers SAMSUNG -> unresolved still
holds together; the fusion layer then collapses repeated observations of one
track into a single item carrying best strength + mean quality + persistence
(see src/layer2/evidence_fusion.py `_collapse_tracks`).

Association follows the ByteTrack idea of associating *every* box (including
low-confidence ones) rather than only high-scoring detections, which is what
keeps a fragmented trajectory from breaking into many short tracks. It is a
plain greedy matcher, not the full Kalman/ByteTrack model — no new deps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

Box = Tuple[float, float, float, float]


def _as_box(bbox) -> Optional[Box]:
    if bbox is None:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _center(a: Box) -> Tuple[float, float]:
    return ((a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0)


def _center_close(a: Box, b: Box, factor: float) -> bool:
    """True when two boxes are close by center distance relative to their size.

    Small motions between adjacent frames can drop IoU below the threshold
    (a compact logo moving a few pixels); center proximity recovers those
    without loosening IoU for large, overlapping-but-distinct objects.
    """
    cx_a, cy_a = _center(a)
    cx_b, cy_b = _center(b)
    scale = min(a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1])
    if scale <= 0:
        return False
    return ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5 <= factor * scale


@dataclass
class Track:
    """One physical object/event observed across frames."""

    track_id: str
    frames: List[int] = field(default_factory=list)
    boxes: List[Box] = field(default_factory=list)
    brands: List[Optional[str]] = field(default_factory=list)
    strengths: List[float] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.frames)

    def dominant_brand(self) -> Optional[str]:
        """Most frequent resolved brand along the track (None if never resolved)."""
        counts: Dict[str, int] = {}
        for b in self.brands:
            if b:
                counts[b] = counts.get(b, 0) + 1
        if not counts:
            return None
        return max(counts.items(), key=lambda kv: kv[1])[0]


def build_tracks(
    per_frame_detections: Sequence[Sequence[dict]],
    *,
    iou_threshold: float = 0.3,
    max_gap: int = 2,
    center_factor: float = 0.6,
) -> List[Track]:
    """Associate per-frame detections into tracks (greedy, ByteTrack-style).

    Args:
        per_frame_detections: one list of detection dicts per frame; each needs
            a `bbox` (and optionally `brand`/`confidence` for track metadata).
        iou_threshold: min IoU for a detection to join an existing track.
        max_gap: max skipped frames a detection may still join a track across.
        center_factor: center-distance tolerance (× min box side) as a fallback
            match when IoU is below threshold.

    Returns:
        Tracks in creation order. Detections without a usable bbox are ignored
        (they cannot be associated), which keeps noisy boxes out of the tracks.
    """
    tracks: List[Track] = []
    active: List[Tuple[Track, int, Box]] = []  # (track, last_frame, last_box)
    counter = 0

    for frame_idx, dets in enumerate(per_frame_detections):
        # One track may absorb at most one detection per frame. Without this,
        # two boxes in the same frame both match the same track (the first
        # updates last_frame to frame_idx, so `frame_idx - last_frame` is 0 for
        # the second and still passes), collapsing a whole swarm into a few
        # tracks and inflating persistence.
        claimed: set[str] = set()
        for det in dets:
            box = _as_box(det.get("bbox"))
            if box is None:
                continue
            best: Optional[Tuple[Track, int, Box]] = None
            best_score = -1.0
            for entry in active:
                tr, last_frame, last_box = entry
                if tr.track_id in claimed:
                    continue
                if frame_idx - last_frame > max_gap:
                    continue
                iou = _iou(box, last_box)
                if iou >= iou_threshold:
                    score = iou + 1.0  # prefer IoU matches over center matches
                elif _center_close(box, last_box, center_factor):
                    score = iou  # in [0, threshold)
                else:
                    continue
                if score > best_score:
                    best_score = score
                    best = entry
            if best is None:
                tr = Track(track_id=f"trk{counter}")
                counter += 1
                tr.frames.append(frame_idx)
                tr.boxes.append(box)
                tr.brands.append(det.get("brand"))
                tr.strengths.append(float(det.get("confidence", 0.0) or 0.0))
                tracks.append(tr)
                active.append((tr, frame_idx, box))
                claimed.add(tr.track_id)
            else:
                tr, _, _ = best
                tr.frames.append(frame_idx)
                tr.boxes.append(box)
                tr.brands.append(det.get("brand"))
                tr.strengths.append(float(det.get("confidence", 0.0) or 0.0))
                idx = active.index(best)
                active[idx] = (tr, frame_idx, box)
                claimed.add(tr.track_id)
        # Drop tracks that have gone silent for longer than max_gap.
        active = [e for e in active if frame_idx - e[1] <= max_gap]

    return tracks


def attach_track_ids(
    per_frame_detections: Sequence[Sequence[dict]],
    *,
    iou_threshold: float = 0.3,
    max_gap: int = 2,
    center_factor: float = 0.6,
) -> List[List[dict]]:
    """Copy detections with a `track_id` + `track_persistence` attached.

    Association is done on boxes only; each detection additionally receives the
    final length of its track as `track_persistence` so downstream fusion can
    credit a long-lived object without re-counting its frames. Detections whose
    box cannot be associated are returned unchanged (no track_id).
    """
    tracks = build_tracks(
        per_frame_detections,
        iou_threshold=iou_threshold,
        max_gap=max_gap,
        center_factor=center_factor,
    )
    # Map (frame_index, id-of-det) is fragile; map by frame+box identity instead.
    by_frame: Dict[int, List[Tuple[Box, Track]]] = {}
    for tr in tracks:
        for frame_idx, box in zip(tr.frames, tr.boxes):
            by_frame.setdefault(frame_idx, []).append((box, tr))

    out: List[List[dict]] = []
    for frame_idx, dets in enumerate(per_frame_detections):
        frame_out: List[dict] = []
        for det in dets:
            box = _as_box(det.get("bbox"))
            copy = dict(det)
            if box is not None:
                for cand_box, tr in by_frame.get(frame_idx, []):
                    if cand_box == box:
                        copy["track_id"] = tr.track_id
                        copy["track_persistence"] = tr.length
                        break
            frame_out.append(copy)
        out.append(frame_out)
    return out
