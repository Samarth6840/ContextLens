"""Offline metrics over the archived job store (instrumentation-only).

Pure read-side analysis of what prune_for_store already persisted. It never
touches pipeline logic; every metric is derived from stored output so it works
even if the pipeline is unchanged from here on.

Usage (from the repo root):

    python3 scripts/offline_analysis.py                      # stats over all jobs
    python3 scripts/offline_analysis.py --latest 6           # last N jobs only
    python3 scripts/offline_analysis.py --labels benchmark/labels.csv

What it reports
---------------
1. Resolution acceptance trend  — acceptance_rate + corroboration_rate per job,
   the two canaries. A drift up in corroboration (more answers depending on
   multi-frame agreement) or down in acceptance is the earliest signal that a
   threshold/embedder change bit resolution quality.

2. Provenance / modality ablation — for every archived brand, the confidence
   reachable from each modality, so "what if we had dropped visual/speech" is
   answered post-hoc per job. Also the cross-scene ratio (brands established in
   both modalities) as a corroboration proxy.

3. Labeled evaluation (open-set ruler) — when --labels is given, a P/R/F1 report
   plus a reliability diagram (ECE) over prediction confidence. This is the
   accuracy measurement the review asked for; see benchmark/labels README for
   the CSV format.

Labels CSV format (header row required):
    video_id,brand,present
    01,_v1.mp4,APPLE,1
    01,_v1.mp4,NIKE,0
    02,_v2.mp4,__UNKNOWN__,1

  present: 1 = the brand genuinely appears in the video; 0 = it does not
           (labeled negative, counts toward FP). brand == "__UNKNOWN__" with
           present=1 marks a real, UNINDEXED brand that must stay unresolved —
           used to measure open-set precision (did the system guess a name it
           could not have known).
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.store import JobStore, _default_db_path

UNKNOWN = "__UNKNOWN__"

# --------------------------------------------------------------------------- #
# Label parsing (pure, testable)
# --------------------------------------------------------------------------- #
def parse_labels(text: str) -> Dict[str, Dict]:
    """Parse the labels CSV into {video_id: {present, absent, has_unknown}}."""
    out: Dict[str, Dict] = {}
    for row in csv.DictReader(__iter_lines(text)):
        vid = (row.get("video_id") or "").strip()
        brand = (row.get("brand") or "").strip().upper()
        raw = (row.get("present") or "").strip().lower()
        present = raw in {"1", "true", "present", "yes"}
        if not vid:
            continue
        rec = out.setdefault(vid, {"present": set(), "absent": set(),
                                   "has_unknown": False})
        if brand == UNKNOWN and present:
            rec["has_unknown"] = True
        elif present:
            rec["present"].add(brand)
        else:
            rec["absent"].add(brand)
    return out


def __iter_lines(text: str) -> List[str]:
    return [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]


# --------------------------------------------------------------------------- #
# Per-job / cross-job aggregation (pure, testable on dicts)
# --------------------------------------------------------------------------- #
def job_summary(job: Dict) -> dict:
    """One reusable row of metrics derivable from a pruned/stored job."""
    l2c = job.get("layer2c") or {}
    timeline = l2c.get("brand_timeline") or {}
    acc = job.get("resolver_acceptance") or {}
    corr = job.get("temporal_corroboration") or {}
    sources: Counter = Counter()
    conf_visual: List[float] = []
    conf_speech: List[float] = []
    cross_scene = 0
    for entry in timeline.values():
        if not isinstance(entry, dict):
            continue
        if entry.get("modalities"):
            if "logo" in entry["modalities"] and "speech" in entry["modalities"]:
                cross_scene += 1
        for app in entry.get("appearances", []):
            c = float(app.get("confidence") or 0.0)
            src = app.get("resolution_source")
            if src:
                sources[src] += 1
            if app.get("modality") == "speech":
                conf_speech.append(c)
            else:
                conf_visual.append(c)
    return {
        "job_id": job.get("job_id"),
        "video_path": job.get("video_path"),
        "created_at": job.get("created_at"),
        "acceptance_rate": acc.get("acceptance_rate"),
        "corroboration_rate": corr.get("corroboration_rate"),
        "brands": len(timeline),
        "cross_scene": cross_scene,
        "resolved_logos": acc.get("resolved"),
        "sources": dict(sources),
        "max_conf_visual": max(conf_visual) if conf_visual else None,
        "max_conf_speech": max(conf_speech) if conf_speech else None,
        "mean_conf_visual": (sum(conf_visual) / len(conf_visual)) if conf_visual else None,
        "indirect_resolutions": len(l2c.get("indirect_resolutions") or []),
    }


def modality_ablation(jobs: List[dict]) -> dict:
    """What each modality contributes, post-hoc, across archived jobs.

    For every brand, confidence reachable from visual-only vs speech-only vs
    both. cross_scene brands (both) are the only ones whose answer survives
    dropping either channel; the rest are single-modality dependent.
    """
    vis_only: List[float] = []
    sp_only: List[float] = []
    cross: List[float] = []
    for job in jobs:
        timeline = (job.get("layer2c") or {}).get("brand_timeline") or {}
        for entry in timeline.values():
            if not isinstance(entry, dict):
                continue
            mods = entry.get("modalities") or []
            confs = [a.get("confidence") or 0.0 for a in entry.get("appearances", [])]
            peak = max(confs) if confs else 0.0
            if "logo" in mods and "speech" in mods:
                cross.append(peak)
            elif "speech" in mods:
                sp_only.append(peak)
            else:
                vis_only.append(peak)
    def _agg(name, vals):
        return {
            "n": len(vals),
            "mean_conf": round(sum(vals) / len(vals), 4) if vals else None,
            "max_conf": round(max(vals), 4) if vals else None,
        }
    return {
        "visual_only": _agg("visual_only", vis_only),
        "speech_only": _agg("speech_only", sp_only),
        "cross_scene": _agg("cross_scene", cross),
    }


def _predicted_brands(job: Dict) -> Dict[str, float]:
    """Video-level brand presence with best available confidence.

    score = resolution_quality when the appearance carries one (the calibrated
    per-source trust), else the raw detector confidence.
    """
    timeline = (job.get("layer2c") or {}).get("brand_timeline") or {}
    pred = {}
    for brand, entry in timeline.items():
        if not isinstance(entry, dict):
            continue
        best = 0.0
        for app in entry.get("appearances", []):
            score = app.get("resolution_quality") or app.get("confidence") or 0.0
            best = max(best, float(score))
        pred[brand.upper()] = round(best, 4)
    return pred


def precision_recall_report(jobs: List[dict], labels: Dict[str, Dict]) -> dict:
    """Open-set P/R/F1 + expected calibration error (ECE) against labels."""
    tp = fp = fn = 0
    bins = defaultdict(lambda: {"count": 0, "correct": 0, "conf_sum": 0.0})
    unresolved_precision_errors = 0  # UNKNOWN label present, but a brand was resolved
    results = []
    for job in jobs:
        vid = job.get("video_id") or job.get("job_id") or job.get("video_path") or ""
        pred = _predicted_brands(job)
        lab = labels.get(vid)
        if lab is None:
            continue

        def _bin(conf: float, correct: float) -> None:
            if conf <= 0:
                return
            bi = min(9, int(conf * 10))
            bins[bi]["count"] += 1
            bins[bi]["correct"] += correct
            bins[bi]["conf_sum"] += conf

        for brand in lab["present"]:
            if brand in pred:
                tp += 1
                results.append((vid, brand, 1, pred[brand]))
                _bin(pred[brand], 1.0)
            else:
                fn += 1
                results.append((vid, brand, 0, 0.0))
        for brand in lab["absent"]:
            if brand in pred:
                fp += 1
                results.append((vid, brand, 0, pred[brand]))
                _bin(pred[brand], 0.0)
        if lab["has_unknown"] and pred:
            unresolved_precision_errors += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    ece = 0.0
    total = 0
    for i in range(10):
        b = bins[i]
        n = b["count"]
        total += n
        if n == 0:
            continue
        acc = b["correct"] / n
        conf = b["conf_sum"] / n
        ece += (n * abs(acc - conf))
    ece = round(ece / total, 4) if total else 0.0
    return {
        "labeled_videos": len({r[0] for r in results}),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
        "unresolved_precision_errors": unresolved_precision_errors,
        "ece": ece,
        "reliability": [
            {
                "bin": i,
                "count": bins[i]["count"],
                "accuracy": round(bins[i]["correct"] / bins[i]["count"], 4)
                if bins[i]["count"] else None,
                "mean_conf": round(bins[i]["conf_sum"] / bins[i]["count"], 4)
                if bins[i]["count"] else None,
            }
            for i in range(10)
        ],
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _pretty(d: dict, indent: int = 0) -> None:
    pad = "  " * indent
    for k, v in d.items():
        if isinstance(v, dict):
            print(f"{pad}{k}:")
            _pretty(v, indent + 1)
        else:
            print(f"{pad}{k}: {v}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=None, help="path to contextlens.db (default derived)")
    ap.add_argument("--latest", type=int, default=None, help="only the last N jobs")
    ap.add_argument("--labels", default=None, help="labels CSV for P/R + calibration")
    args = ap.parse_args()

    store = JobStore(args.db or _default_db_path())
    jobs = store.list()
    if args.latest:
        jobs = jobs[: args.latest]
    if not jobs:
        print("No stored jobs to analyze. (You must run a video through the pipeline first.)")
        store.close()
        return 1

    print(f"=== jobs analyzed: {len(jobs)} ===")
    print("\n--- resolution acceptance / corroboration per job ---")
    for s in (job_summary(j) for j in jobs):
        _pretty(s, 1)
    print("\n--- modality ablation (post-hoc, archived data) ---")
    _pretty(modality_ablation(jobs), 1)

    if args.labels:
        with open(args.labels) as fh:
            labels = parse_labels(fh.read())
        print(f"\n--- labeled eval ({len(labels)} labeled videos) ---")
        _pretty(precision_recall_report(jobs, labels), 1)
    else:
        print("\n(no --labels given; pass labels CSV to get precision/recall + calibration)")

    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())