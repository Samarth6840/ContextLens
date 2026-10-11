"""Phase 2.5 acceptance harness: SigLIP2 vs CLIP on TRAIN videos.

Counts per-video brand resolutions (from pipeline run JSON) and detector
FP/free-frame (from a labeled export, box-level -> 1 FP detection on a
logo-FREE frame + matching GT box (IoU>=.5) counts; free-frame denominator).
Reports BEFORE (CLIP) vs AFTER (SigLIP2) rows when both runs exist.

MUST NOT read the holdout: any stem whose md5-prefix is in HOLDOUT_MD5 is
skipped and ASSERTED never touched (tune on train, confirm on ONE holdout
video manually per the 2.5 protocol — never sweep all 160).

Usage:
  python3 scripts/compare_encoder_2_5.py \
    --export <labeled_export_dir> \
    --yolo-dets benchmark/results/realvideo_v1_yolo.json \
    --before <run_clip.json> --after <run_siglip2.json>
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

HOLDOUT_MD5 = {"88b0502222", "df01d5a843"}


def read_all(root: str):
    from scripts.eval_detector_separation import read_split
    return read_split(Path(root), "all")  # [(stem, gts, is_logo, area, sub, sp)]


def fp_per_free_frame(dets, free_frames) -> float:
    """box-level: 1 FP per false detection on a logo-free frame; /free_frames."""
    from scripts.eval_logo import _iou
    fp = 0
    for r in free_frames:
        stem = r[0]
        v = dets.get(stem)
        if v is None:
            continue
        if isinstance(v, dict):
            v = v.get("dets") or []
        for x in v:
            if any(_iou(x[:4], g) >= 0.5 for g in r[1]):
                continue  # matched a GT box -> true positive, not a frame FP
            fp += 1
    return fp / len(free_frames) if free_frames else 0.0


def brand_resolutions(run: dict, md5: str) -> int:
    rows = run.get("videos") or run  # [ {video_md5: result} ] or {md5: result}
    if isinstance(rows, dict):
        res = rows.get(md5) or {}
        return len((res.get("layer2c") or {}).get("brand_timeline") or {})
    for r in rows:
        if str(r.get("video") or r.get("video_path", "")).startswith(md5):
            return len((r.get("layer2c") or {}).get("brand_timeline") or {})
    return -1  # run did not process this train video


def per_train_video(export: str, yolo: dict, run_before=None, run_after=None):
    all_rows = read_all(export)
    frames_by_md5 = {}
    for r in all_rows:
        frames_by_md5.setdefault(r[0].split("_", 1)[0], []).append(r)
    for md5 in HOLDOUT_MD5:
        if md5 in frames_by_md5:
            sys.exit(f"refusing to read holdout video {md5}")
    out = {}
    for md5, frames in frames_by_md5.items():
        free = [r for r in frames if not r[2]]  # is_logo flag = r[2]
        out[md5] = {
            "n_frames": len(frames),
            "n_logo_free": len(free),
            "fp_per_free_frame": fp_per_free_frame(yolo, free),
            "brand_resolutions_before": brand_resolutions(run_before, md5) if run_before else -1,
            "brand_resolutions_after": brand_resolutions(run_after, md5) if run_after else -1,
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--export", required=True)
    p.add_argument("--yolo-dets", required=True)
    p.add_argument("--before", help="pipeline run JSON under CLIP (baseline)")
    p.add_argument("--after", help="pipeline run JSON under SigLIP2")
    a = p.parse_args()

    yolo = json.loads(Path(a.yolo_dets).read_text())
    if yolo and isinstance(next(iter(yolo.values())), dict):
        yolo = {k: (v.get("dets") or []) for k, v in yolo.items()}
    run_b = json.loads(Path(a.before).read_text()) if a.before else None
    run_a = json.loads(Path(a.after).read_text()) if a.after else None

    rows = per_train_video(a.export, yolo, run_b, run_a)
    print(f"{'video(pooled)':<14} {'frames':>6} {'free':>5} {'YOLO FP/free':>12} "
          f"{'res.BEFORE':>10} {'res.AFTER':>10}")
    for md5, r in sorted(rows.items()):
        print(f"{md5:<14} {r['n_frames']:>6} {r['n_logo_free']:>5} "
              f"{r['fp_per_free_frame']:>12.3f} {r['brand_resolutions_before']:>10} "
              f"{r['brand_resolutions_after']:>10}")


if __name__ == "__main__":
    main()