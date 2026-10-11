"""
One-shot test: does OWLv2 zero-shot grounding find logo regions at all?

Mirrors the measurement already in config.yaml (YOLO-World, 6 generic
queries, region_proposal_only=true baseline: 0 TP / 2 FP / F1 0.000 on
30 real LogoDet-3K frames / 41 logo boxes). Uses the SAME eval set and
SAME IoU/scoring logic as scripts/eval_logo.py so the numbers are directly
comparable.

Usage:
    python scripts/owlv2_recall_test.py --eval-set benchmark/eval_real --split test
"""
import argparse
import importlib.util as ilu
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Reuse the exact scoring functions from eval_logo.py so results are comparable.
_spec = ilu.spec_from_file_location(
    "eval_logo", Path(__file__).resolve().parent / "eval_logo.py"
)
eval_logo = ilu.module_from_spec(_spec)
_spec.loader.exec_module(eval_logo)

GENERIC_QUERIES = [
    "a brand logo",
    "a company logo",
    "a product logo",
    "a wordmark",
    "a label",
    "a text logo",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", default="benchmark/eval_real")
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", default="google/owlv2-base-patch16-ensemble")
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--out", default=None,
                    help="write predictions+gold JSON here so filtered re-analyses "
                         "can reuse them without re-running inference. "
                         "Defaults to <eval-set>/owlv2_preds_<split>.json")
    args = ap.parse_args()

    root = Path(args.eval_set)
    labels = json.load(open(root / "labels.json"))
    rows = labels[args.split]

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = Owlv2Processor.from_pretrained(args.model)
    model = Owlv2ForObjectDetection.from_pretrained(args.model).to(device)
    model.eval()

    t0 = time.monotonic()
    pred_rows = []
    gold = {}
    for i, r in enumerate(rows):
        img_path = root / "images" / args.split / r["image"]
        image = Image.open(img_path).convert("RGB")
        gold[r["image"]] = r["boxes"]

        inputs = processor(
            text=[GENERIC_QUERIES], images=image, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]]).to(device)
        results = processor.post_process_grounded_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=args.conf
        )[0]

        dets = []
        for box, score in zip(results["boxes"], results["scores"]):
            x1, y1, x2, y2 = box.tolist()
            dets.append({"bbox": [x1, y1, x2, y2], "conf": float(score)})
        pred_rows.append(
            {"image": r["image"], "negative": r["negative"], "dets": dets}
        )
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(rows)}  {time.monotonic()-t0:.1f}s")

    elapsed = time.monotonic() - t0
    report, wrong = eval_logo.evaluate(pred_rows, gold, iou_thr=args.iou)
    mAP50, n_gt = eval_logo.mean_ap(pred_rows, gold, iou_thr=args.iou)

    print(f"\n{'thr':>10} {'TP':>4} {'FP':>5} {'FN':>4} {'P':>7} {'R':>7} {'F1':>7}")
    for k, v in report.items():
        print(
            f"{k:>10} {v['tp']:>4} {v['fp']:>5} {v['fn']:>4} "
            f"{v['precision']:>7.4f} {v['recall']:>7.4f} {v['f1']:>7.4f}"
        )
    print(f"\nmAP50 ({n_gt} GT boxes): {mAP50}")
    print(f"{elapsed:.1f}s  ({1000*elapsed/max(1,len(rows)):.0f} ms/image)")

    out = Path(args.out or (root / f"owlv2_preds_{args.split}.json"))
    json.dump(
        {
            "model": args.model,
            "split": args.split,
            "iou": args.iou,
            "queries": GENERIC_QUERIES,
            "pred_rows": pred_rows,
            "gold": gold,
        },
        open(out, "w"),
        indent=1,
    )
    print(f"predictions -> {out}")


if __name__ == "__main__":
    main()
