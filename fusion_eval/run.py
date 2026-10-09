"""Run the evaluation harness.

    python fusion_eval/run.py [--budget N] [--report-dir DIR] [--no-ablate]

Pipeline steps: build synthetic GT dataset -> run pipeline per entry ->
compute metrics, ablations, calibration, failure taxonomy -> write report.

Ablations re-fuse from the cached evidence ledger (no repeat detection).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

logging.disable(logging.INFO)

from fusion_eval.ablations import COMBOS
from fusion_eval.calibration import best_temperature
from fusion_eval.dataset import build_dataset, load_dataset
from fusion_eval.metrics import compute_metrics, split_by_difficulty, split_by_modality
from fusion_eval.reports import build_report

from src.pipeline import Phase1Pipeline


def _bank_root(pipeline) -> str:
    raw = pipeline.cfg["layer1"].get("product_index", {}).get("reference_dir", "benchmark/product_logos")
    if not os.path.isabs(raw):
        raw = os.path.join(os.path.dirname(pipeline.workspace), raw)
    return os.path.abspath(raw)


def _pred_from(entry: dict, result: dict, timing_total: float) -> dict:
    f = (result.get("layer2b") or {}).get("fusion") or {}
    top3 = [r["candidate"] for r in f.get("ranking", [])[:3]]
    return {
        "video_id": entry["video_id"],
        "gt_brand": entry.get("gt_brand"),
        "visible": bool(entry.get("visible_brand")),
        "difficulty": entry.get("difficulty"),
        "winner": f.get("winner"),
        "prob": (f.get("ranking") or [{}])[0].get("prob") if f.get("ranking") else 0.0,
        "margin": f.get("margin", 0.0),
        "verdict": f.get("verdict"),
        "families": (f.get("ranking") or [{}])[0].get("families", {}) if f.get("ranking") else {},
        "top3": top3,
        "latency_layer1": round(timing_total, 3),
        "gpu_mem_mb": 0.0,
        # Layer-2a gate path (learned_gating / fixed_heuristic / equal /
        # quality_proportional_fallback) so the report can show how often the
        # learned gate was actually used vs overridden.
        "weight_source": (result.get("layer2a") or {}).get("weight_source", "unknown"),
        # Which modalities actually carried GT evidence for this entry — drives
        # the accuracy-vs-modality-availability breakdown.
        "modalities": {
            "logo": bool(entry.get("logo")),
            "ocr": bool(entry.get("ocr")),
            "speech": bool(entry.get("speech")),
            "product": bool(entry.get("product")),
            "audio_event": bool(entry.get("audio_event")),
        },
        "_ledger": f.get("_ledger", {}),
        "_cfg": f,
    }


def _split_fit_test(rows):
    """Deterministic disjoint fit/test halves, stratified by hit/miss.

    Even/odd position within each correctness group, so a run where every
    prediction is wrong still yields two comparable halves instead of a fit set
    of only hits.
    """
    fit, test = [], []
    for correct in (True, False):
        grp = [r for r in rows if (r.get("winner") == r.get("gt_brand")) is correct]
        fit.extend(grp[0::2])
        test.extend(grp[1::2])
    return fit, test


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--report-dir", default="fusion_eval/out")
    ap.add_argument("--no-ablate", action="store_true")
    ap.add_argument("--dataset", default=None,
                    help="path to a labeled dataset JSON (entry list). When "
                         "omitted, a small synthetic GT set is synthesized from "
                         "the reference logo bank.")
    args = ap.parse_args()
    os.makedirs(args.report_dir, exist_ok=True)

    pipeline = Phase1Pipeline()
    if args.dataset:
        entries = load_dataset(args.dataset)
        print(f"dataset: {len(entries)} entries from {args.dataset}")
    else:
        bank = _bank_root(pipeline)
        entries = build_dataset(bank, args.report_dir, budget=args.budget)
        print(f"dataset: {len(entries)} entries from {bank}")

    preds = []
    for e in entries:
        r = pipeline.process_video(e["video_path"], frame_rate=25.0)
        timing = (r.get("timings") or {}).get("layer1",
                  sum((r.get("timings") or {}).values()))
        preds.append(_pred_from(e, r, float(timing) if timing else 0.0))
        p0 = preds[-1]
        print(f"  {e['video_id']:28s} gt={str(e.get('gt_brand')):12s} "
              f"winner={str(p0['winner']):12s} prob={p0['prob']:.3f} verdict={p0['verdict']}")

    by_diff = split_by_difficulty(preds)
    by_modality = split_by_modality(preds)

    ablate = {} if args.no_ablate else _run_ablations(pipeline, preds)

    calib_rows = [r for r in preds if r["verdict"] in ("confident", "ambiguous")]
    # Fit T and score it on disjoint halves, so the reported ECE is not the grid
    # search reporting its own minimum back to itself.
    fit_rows, test_rows = _split_fit_test(calib_rows)
    calibration = (
        best_temperature(fit_rows, test_rows=test_rows) if fit_rows
        else {"temperature": 1.0, "ece": 0.0, "brier": 0.0, "ece_all": {},
              "n_fit": 0, "n_test": 0}
    )

    metrics = compute_metrics(preds)
    path = build_report(preds, metrics, by_diff, ablate, calibration,
                        args.report_dir, by_modality=by_modality)
    print(f"\nTop-1(covered)={metrics['top1_accuracy_covered']}  F1={metrics['f1']}  "
          f"abstention={metrics['abstention_rate']}  ECE={metrics['ece']}  "
          f"calibration-T={calibration['temperature']}")
    print(f"report: {path}")
    if not args.no_ablate and ablate:
        for name, c in ablate["combos"].items():
            print(f"  {name:12s} top1={c['top1']}  Δ={c['delta_top1_vs_all']:+0.3f}  ece={c['ece']}")
    pipeline.ocr._shutdown()
    return 0


def _run_ablations(pipeline, base_preds) -> dict:
    """Re-fuse each entry's cached ledger with families removed (no re-detection)."""
    from fusion_eval.ablations import ablation_report, rerun_fusion

    cfg = pipeline.cfg["layer2b"]["fusion"]
    _MAP = {"logo_detected": "logo", "speech_mention": "speech",
            "ocr_hit": "ocr", "visual_product_match": "product",
            "product_retrieval": "product", "audio_event": "audio"}
    weights = {}
    for name, spec in (pipeline.cfg.get("layer2b", {}).get("evidence_sources") or {}).items():
        if isinstance(spec, dict) and name in _MAP and spec.get("status") == "implemented":
            weights[_MAP[name]] = float(spec.get("weight", 0.0))

    common = dict(
        weights=weights or None,
        temperature=float(cfg.get("temperature", 1.0)),
        time_bucket=float(cfg.get("time_bucket_seconds", 2.0)),
        accept=float(cfg.get("accept", 0.30)),
        margin_min=float(cfg.get("margin_min", 0.15)),
        agreement_bonus=float(cfg.get("agreement_bonus", 0.30)),
        contradiction_penalty=float(cfg.get("contradiction_penalty", 0.5)),
    )
    per_combo = {
        name: [rerun_fusion(p, disable, **common) for p in base_preds]
        for name, disable in COMBOS
    }
    return ablation_report(base_preds, per_combo)


if __name__ == "__main__":
    raise SystemExit(main())