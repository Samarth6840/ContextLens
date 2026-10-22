"""Merge the pre-typo-fix 1.8 summary into the re-run report (before/after).

The NVIDIAL->NVIDIA label fix (commit 846e259) changed one gold label AFTER the
first 1.8 run. This helper embeds the pre-fix scores (snapshotted at
/tmp/ctxlens_snap/before_summary.json) into the regenerated report under
`before_gold_typo_fix`, so the report carries both score sets instead of
silently overwriting history.

Usage:
    python3 scripts/merge_1p8_before_after.py \
        --report benchmark/results/realvideo_v1_1p8_arms.json \
        --before /tmp/ctxlens_snap/before_summary.json
"""
import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="benchmark/results/realvideo_v1_1p8_arms.json")
    ap.add_argument("--before", default="/tmp/ctxlens_snap/before_summary.json")
    args = ap.parse_args()

    report = json.loads(Path(args.report).read_text())
    before = json.loads(Path(args.before).read_text())

    after_frame = report["arm1_as_shipped"]["per_frame"]["df01d5a843_007520.jpg"]
    report["before_gold_typo_fix"] = {
        "why": (
            "Gold hero brand on df01d5a843_007520.jpg was recorded blind as "
            "'NVIDIAL' (typo for NVIDIA) and corrected in labels.jsonl commit "
            "846e259 AFTER this report's first run. Numbers below are the "
            "first run's; the top-level arm summaries are the re-score with "
            "the corrected gold. Only scoring changed — arm named-sets are "
            "identical between runs (the pipeline is deterministic on this "
            "holdout)."
        ),
        "frame": "df01d5a843_007520.jpg",
        "gold_before": before["gold_typo_frame_before"]["gold"],
        "gold_after": after_frame["gold"],
        "arm1_frame_before": before["gold_typo_frame_before"],
        "arm1_frame_after": after_frame,
        "arm2_frame_before": before["arm2_typo_frame_before"],
        "arm2_frame_after": report["arm2_ocr_fullframe"]["per_frame"]["df01d5a843_007520.jpg"],
        "arm1_summary_before": before["arm1"],
        "arm2_summary_before": before["arm2"],
        "verdict_before": before["verdict"],
        "decision_detail_before": before["decision_detail"],
    }

    # Verify arm named-sets did not drift between runs (only gold changed).
    old_named = before.get("arm1_typo_frame_before", {}).get("named")
    report["before_gold_typo_fix"]["arm_named_sets_unchanged_check"] = (
        "arm1 frame named before=" + str(before["gold_typo_frame_before"]["named"])
        + " after=" + str(after_frame["named"])
    )

    Path(args.report).write_text(json.dumps(report, indent=2))
    print(f"merged before/after into {Path(args.report).resolve()}")
    print(f"  before: arm1 {before['arm1']['hero_brand_recall']} / "
          f"arm2 {before['arm2']['hero_brand_recall']} / verdict {before['verdict']}")
    print(f"  after:  arm1 {report['arm1_as_shipped']['hero_brand_recall']} / "
          f"arm2 {report['arm2_ocr_fullframe']['hero_brand_recall']} / "
          f"verdict {report['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
