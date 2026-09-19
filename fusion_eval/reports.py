"""Human-readable report: markdown + JSON artifact of the whole eval run."""

from __future__ import annotations

import json
import os
from typing import Dict, List

from .calibration import best_temperature
from .error_taxonomy import classify
from .metrics import compute_metrics, split_by_difficulty


def build_report(
    preds: List[dict],
    metrics: Dict[str, float],
    by_difficulty: Dict[str, Dict[str, float]],
    ablation: Dict,
    calibration: Dict,
    out_dir: str,
) -> str:
    rows = []
    for p in preds:
        rows.append({
            "video_id": p["video_id"],
            "difficulty": p.get("difficulty"),
            "gt": p.get("gt_brand"),
            "winner": p.get("winner"),
            "prob": p.get("prob"),
            "margin": p.get("margin"),
            "verdict": p.get("verdict"),
            "families": p.get("families"),
            "taxonomy": classify(p),
            "latency_s": p.get("latency_layer1"),
        })

    md = ["# Multimodal evidence fusion — evaluation report", ""]
    md.append("## Overall metrics")
    md.append("| metric | value |")
    md.append("|---|---|")
    for k, v in metrics.items():
        md.append(f"| {k} | {v} |")
    md.append("")

    md.append("## By difficulty")
    md.append("| difficulty | top1 | top1_covered | f1 | ece | abstention |")
    md.append("|---|---|---|---|---|---|")
    for d, m in by_difficulty.items():
        md.append(f"| {d} | {m['top1_accuracy']} | {m['top1_accuracy_covered']} | "
                  f"{m['f1']} | {m['ece']} | {m['abstention_rate']} |")
    md.append("")

    if ablation:
        md.append("## Ablations (delta vs ALL)")
        md.append("| combo | top1 | f1 | ece | abstention | Δtop1 | Δf1 | Δece |")
        md.append("|---|---|---|---|---|---|---|---|")
        for name, c in ablation["combos"].items():
            md.append(f"| {name} | {c['top1']} | {c['f1']} | {c['ece']} | {c['abstention']} "
                      f"| {c['delta_top1_vs_all']} | {c['delta_f1_vs_all']} | {c['delta_ece_vs_all']} |")
        md.append("")

    md.append("## Calibration")
    md.append(f"best temperature: {calibration['temperature']}  (ECE {calibration['ece']}, Brier {calibration['brier']})")
    md.append("")

    md.append("## Per-video verdicts")
    md.append("| video | difficulty | gt | winner | prob | verdict | taxonomy |")
    md.append("|---|---|---|---|---|---|---|")
    for r in rows:
        md.append(f"| {r['video_id']} | {r['difficulty']} | {r['gt']} | {r['winner']} | "
                  f"{r['prob']} | {r['verdict']} | {r['taxonomy']} |")
    md.append("")

    md_path = os.path.join(out_dir, "eval_report.md")
    open(md_path, "w").write("\n".join(md))
    json.dump({"metrics": metrics, "by_difficulty": by_difficulty, "ablation": ablation,
               "calibration": calibration, "rows": rows},
              open(os.path.join(out_dir, "predictions.json"), "w"), indent=2, default=str)
    return md_path