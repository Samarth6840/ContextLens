"""Collect frames worth labelling from already-analysed jobs (active learning).

The detector's recall is capped by its training data (see config/config.yaml,
layer1.logo_detection). The cheapest way to grow a useful label pool is to spend
human effort ONLY on the frames the current system gets wrong, then feed them to
`scripts/label_frames.py` and retrain (`scripts/train_logo_detector.py`).

This script scans analysed jobs (a SQLite JobStore or a directory of persisted
result JSONs) and emits a manifest of frames to label, each with a reason:

  * unresolved_logo  — a frame where the detector DID fire but the resolver could
                       not name any box. These are the hard negatives for the
                       identity stage and, where a mark is genuinely present, the
                       positives the resolver is missing.
  * missed_detection — a brand was spoken (ASR) near a frame with NO logo box at
                       all: a hard positive the detector missed entirely.

Output is JSONL, one row per (job, frame, reason):
    {"job_id", "video_path", "frame_index", "reason", "brand", "boxes", "max_conf"}

It does not modify any job; it only reads. With --extract-frames DIR it also
pulls exactly those frames out of each job's source video (cv2), writing
    <job_id>_f<frame_index>_<reason>.jpg
into DIR — a directory label_frames.py can be pointed at directly with
--frames DIR. NOTE: the job store prunes the temp upload video after analysis,
so extraction only works while the source file still exists; rows whose video
is gone are reported, not silently dropped.

Usage:
    python scripts/collect_active_learning_frames.py --store var/contextlens.db \
        --out var/active_learning.jsonl
    python scripts/collect_active_learning_frames.py --results var/results/ \
        --out var/active_learning.jsonl --extract-frames var/al_frames/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator

# Scripts are invoked as `python scripts/<name>.py`, so the repo root is not on
# sys.path by default; add it before importing `src.*`.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


UNKNOWN_BRAND = "UNKNOWN BRAND"


def _dedupe(cands: list[dict]) -> list[dict]:
    """De-dupe by (frame_index, reason), keeping first appearance."""
    seen = set()
    out = []
    for c in cands:
        key = (c["frame_index"], c["reason"])
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def frame_candidates(result: dict, min_conf: float = 0.25) -> list[dict]:
    """Frames worth labelling from a RAW pipeline result. Pure; no I/O.

    `result` is a pipeline result dict containing `layer1.logo_detections`
    (per-frame list of detections with `brand`, `confidence`) plus
    `layer1.brand_mentions`, `video_fps`, `video_stride`, `num_frames`.
    """
    l1 = result.get("layer1") or {}
    logos = l1.get("logo_detections") or []
    out: list[dict] = []

    frames_with_boxes = set()
    for fi, dets in enumerate(logos):
        if not dets:
            continue
        frames_with_boxes.add(fi)
        if any(d.get("brand") for d in dets):
            continue  # resolved -> nothing to learn here
        max_conf = max(
            (float(d.get("confidence", 0.0) or 0.0) for d in dets), default=0.0
        )
        out.append({
            "frame_index": fi,
            "reason": "unresolved_logo",
            "brand": None,
            "boxes": len(dets),
            "max_conf": round(max_conf, 4),
        })

    # Brand spoken/read but the detector found no box near that instant: a hard
    # positive the detector missed. Map the mention's real timestamp back to the
    # sampled frame index (same stride/fps conversion the pipeline uses).
    video_fps = float(result.get("video_fps") or 0.0)
    stride = int(result.get("video_stride", 1) or 1)
    num_frames = int(result.get("num_frames", 0) or 0)
    if video_fps > 0 and num_frames > 0:
        for m in l1.get("brand_mentions") or []:
            ts = m.get("start_time")
            if ts is None:
                continue
            fi = int(ts * video_fps / max(1, stride))
            fi = max(0, min(num_frames - 1, fi))
            if fi in frames_with_boxes:
                continue
            out.append({
                "frame_index": fi,
                "reason": "missed_detection",
                "brand": m.get("brand"),
                "boxes": 0,
                "max_conf": 0.0,
            })

    return _dedupe(out)


def dashboard_candidates(dash: dict, min_conf: float = 0.25) -> list[dict]:
    """Frames worth labelling from a persisted DASHBOARD block.

    The SQLite store keeps a pruned result (no `layer1`), so analysed jobs are
    recovered from the dashboard: scenes whose logos are all unresolved are hard
    negatives, and an ad/ASR mention whose nearest scene shows no resolved brand
    is a hard positive the detector missed. Pure; no I/O.
    """
    scenes = dash.get("scenes") or []
    out: list[dict] = []
    resolved_frames = set()
    for s in scenes:
        fi = s.get("frame_index")
        if fi is None:
            continue
        logos = s.get("logos") or []
        if any(l.get("class_name") and l.get("class_name") != UNKNOWN_BRAND
               for l in logos):
            resolved_frames.add(fi)
            continue
        if not logos:
            continue
        max_conf = max(
            (float(l.get("confidence", 0.0) or 0.0) for l in logos), default=0.0
        )
        out.append({
            "frame_index": fi,
            "reason": "unresolved_logo",
            "brand": None,
            "boxes": len(logos),
            "max_conf": round(max_conf, 4),
        })

    for a in dash.get("ads") or []:
        fi = a.get("frame_index")
        if fi is None or fi in resolved_frames:
            continue
        out.append({
            "frame_index": fi,
            "reason": "missed_detection",
            "brand": a.get("brand"),
            "boxes": 0,
            "max_conf": 0.0,
        })
    return _dedupe(out)


def snapshot_candidates(snapshot: dict, min_conf: float = 0.25) -> list[dict]:
    """Candidates from either a raw result or a persisted job snapshot."""
    if not isinstance(snapshot, dict):
        return []
    result = snapshot.get("result")
    if (isinstance(result, dict) and isinstance(result.get("layer1"), dict)
            and result.get("layer1")):
        return frame_candidates(result, min_conf)
    if isinstance(snapshot.get("layer1"), dict) and snapshot.get("layer1"):
        return frame_candidates(snapshot, min_conf)  # a bare result dict
    dash = snapshot.get("dashboard")
    if isinstance(dash, dict):
        return dashboard_candidates(dash, min_conf)
    return []


def _video_meta(data: dict) -> tuple[str, int]:
    """(video_path, video_stride) for a stored job / result JSON."""
    result = data.get("result") if isinstance(data.get("result"), dict) else {}
    video_path = data.get("video_path") or result.get("video_path") or ""
    stride = data.get("video_stride") or result.get("video_stride") or 1
    try:
        stride = max(1, int(stride))
    except (TypeError, ValueError):
        stride = 1
    return str(video_path), stride


def _safe_prefix(job_id: str) -> str:
    """Filename-safe job prefix.

    label_frames.py pins a queue with `name.split("_")[0]`, so the prefix must
    contain no underscore — hence "-" for anything outside [A-Za-z0-9.-].
    """
    safe = "".join(ch if (ch.isalnum() or ch in "-.") else "-"
                   for ch in str(job_id))
    return safe.strip("-.") or "job"


def extract_frames(video_path: str, stride: int, rows: list[dict],
                   out_dir: Path, prefix: str) -> dict:
    """Write each candidate row's source frame to out_dir as a JPEG.

    `frame_index` is a position in the pipeline's SAMPLED frame list; long
    videos are decoded with stride > 1, so sampled index i lives at source
    frame i*stride (the same conversion server._read_video_frame and
    openset._read_frame use). Pure selection in, files out: returns
    {"written", "missing_video", "read_failed"} counts.
    """
    counts = {"written": 0, "missing_video": 0, "read_failed": 0}
    if not rows:
        return counts
    if not video_path or not Path(video_path).is_file():
        counts["missing_video"] = len(rows)
        return counts

    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        counts["missing_video"] = len(rows)
        return counts
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_dir.mkdir(parents=True, exist_ok=True)
    for row in sorted(rows, key=lambda r: int(r.get("frame_index") or 0)):
        fi = int(row.get("frame_index") or 0)
        target = int(round(fi * max(1, int(stride))))
        if total > 0:
            target = min(target, total - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, target))
        ok, frame = cap.read()
        if not ok or frame is None:
            counts["read_failed"] += 1
            continue
        name = f"{prefix}_f{fi:06d}_{row.get('reason', 'frame')}.jpg"
        if cv2.imwrite(str(out_dir / name), frame):
            counts["written"] += 1
        else:
            counts["read_failed"] += 1
    cap.release()
    return counts


def _iter_results(args) -> Iterator[tuple[str, dict]]:
    if args.results:
        for p in sorted(Path(args.results).glob("*.json")):
            try:
                data = json.loads(p.read_text())
            except (OSError, ValueError):
                continue
            yield p.stem, data
    if args.store:
        from src.store import JobStore

        store = JobStore(args.store)
        for row in store.list():
            job = store.load(row.get("job_id"))
            if isinstance(job, dict):
                yield str(job.get("job_id")), job


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", help="SQLite JobStore path to scan")
    ap.add_argument("--results", help="directory of persisted result JSON files")
    ap.add_argument("--out", required=True, help="output JSONL manifest")
    ap.add_argument("--min-conf", type=float, default=0.25,
                    help="detector confidence floor of interest (informational)")
    ap.add_argument("--limit-per-job", type=int, default=0,
                    help="cap candidates per job (0 = no cap)")
    ap.add_argument("--extract-frames", metavar="DIR", default="",
                    help="also pull the listed frames out of each source video "
                         "into DIR, ready for scripts/label_frames.py --frames")
    args = ap.parse_args()

    if not args.store and not args.results:
        sys.exit("FATAL: pass --store or --results")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    extract_dir = Path(args.extract_frames) if args.extract_frames else None
    n_rows = 0
    n_jobs = 0
    reasons: dict[str, int] = {}
    extracted = {"written": 0, "missing_video": 0, "read_failed": 0}
    with out_path.open("w", encoding="utf-8") as fh:
        for job_id, data in _iter_results(args):
            if not isinstance(data, dict):
                continue
            if not (data.get("result") or data.get("dashboard") or data.get("layer1")):
                continue
            n_jobs += 1
            cands = snapshot_candidates(data, min_conf=args.min_conf)
            if args.limit_per_job:
                cands = cands[: args.limit_per_job]
            video_path, stride = _video_meta(data)
            for c in cands:
                fh.write(json.dumps({
                    "job_id": job_id,
                    "video_path": video_path,
                    **c,
                }) + "\n")
                reasons[c["reason"]] = reasons.get(c["reason"], 0) + 1
                n_rows += 1
            if extract_dir is not None and cands:
                for k, v in extract_frames(video_path, stride, cands,
                                           extract_dir,
                                           _safe_prefix(job_id)).items():
                    extracted[k] += v

    print(f"scanned {n_jobs} job(s); wrote {n_rows} frame candidate(s) -> {out_path}")
    if reasons:
        print("reasons:", reasons)
    if n_rows == 0:
        print("nothing to label — all detections resolved and no missed mentions.")
    if extract_dir is not None:
        print(f"extracted {extracted['written']} frame(s) -> {extract_dir}")
        if extracted["missing_video"]:
            print(f"  {extracted['missing_video']} row(s) skipped: source video "
                  "missing (the store prunes temp uploads) — re-upload and "
                  "re-analyse those videos, then re-run")
        if extracted["read_failed"]:
            print(f"  {extracted['read_failed']} row(s) failed to decode")
        if n_rows and not extracted["written"]:
            print("FATAL: --extract-frames was requested but no frame could be "
                  "read; the manifest alone is not a label queue.")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
