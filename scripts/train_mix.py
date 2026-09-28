"""Fine-tune the corpus checkpoint on the human-labelled mix. Trains one run.

The previous mix run was launched as a bare `yolo detect train` line, so its
only record was runs/.../args.yaml, dug out of a save directory. This is the
same call with the arguments the diagnosis fixed, and the resolved args are
printed so a run can be read back from its own log.

Hyperparameters are defaults, not a config file: this is one experiment and
the numbers are in git history.

  python3 scripts/train_mix.py
  python3 scripts/train_mix.py --epochs 5 --patience 5      # smoke test

Writes to runs/detect/benchmark/train_mix/<name>. The failed unbalanced run
stays where it is under name=train: it is the evidence for the imbalance
diagnosis and must not be overwritten by the run that answers it.
"""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
FROZEN_VIDEO = "88b0502222"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="benchmark/train_mix/data.yaml")
    ap.add_argument("--base",
                    default="runs/detect/weights/logo_corpus/train/weights/best.pt",
                    help="checkpoint to fine-tune from, and the run's control")
    ap.add_argument("--name", default="train_1p5",
                    help="save-dir name under the project, so runs stay distinct")
    ap.add_argument("--project", default="benchmark/train_mix")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--lr0", type=float, default=0.0002)
    ap.add_argument("--optimizer", default="AdamW",
                    help="MUST be a named optimizer, never 'auto'. Ultralytics "
                         "discards lr0 and momentum under optimizer=auto and "
                         "substitutes lr_fit = 0.002*5/(4+nc), which is 0.002 for "
                         "this 1-class mix: a 10x miss that only the log reveals.")
    ap.add_argument("--freeze", type=int, default=10)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    data = ROOT / args.data
    text = data.read_text()
    if FROZEN_VIDEO in text:
        raise SystemExit(f"FATAL: {data} references the frozen test video")
    mix = json.loads((data.parent / "manifest.json").read_text())
    if mix["train"]["positives_from_one_video"]:
        print(f"WARNING: TRAIN positives come only from "
              f"{mix['train']['videos_carrying_positives']}. This run measures "
              f"whether hard-negative fine-tuning helps AT ALL on val/stress. It "
              f"is not evidence of generalization to an unseen video.")

    model = YOLO(args.base)
    model.train(data=str(data), project=args.project, name=args.name,
                exist_ok=True, epochs=args.epochs, patience=args.patience,
                lr0=args.lr0, optimizer=args.optimizer, freeze=args.freeze,
                imgsz=args.imgsz, batch=args.batch, device=args.device,
                seed=0, deterministic=True)

    t = model.trainer
    # The LR that actually ran, not the one that was asked for. Under
    # optimizer=auto these differ by 10x and the run looks fine until the
    # numbers come back wrong.
    if t.args.lr0 != args.lr0 or t.args.optimizer != args.optimizer:
        print(f"FATAL: requested lr0={args.lr0} optimizer={args.optimizer}, "
              f"trained with lr0={t.args.lr0} optimizer={t.args.optimizer}")
        return 1
    print(f"\nbase       {args.base}")
    print(f"candidate  {t.best}")
    print(f"save_dir   {t.save_dir}")
    print(f"trained    lr0={t.args.lr0} optimizer={t.args.optimizer} "
          f"freeze={t.args.freeze} epochs={t.epochs} patience={t.args.patience}")
    print(f"train      ratio neg:pos {mix['ratio_negative_to_positive']}:1 "
          f"({mix['train']['hard_negative_crops']} crops kept, "
          f"{mix['hard_negative_crops_dropped']} dropped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
