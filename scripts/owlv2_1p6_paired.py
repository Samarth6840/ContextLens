"""Phase 1.6: paired top-1 hit rate vs FP-per-free-frame, both arms, one split.

The 1.5 run scored OWLv2 and the fine-tuned YOLO detector with the same ledger
but reported only pooled box P/R/F1, which the annotation density of LogoDet-3K
shelf photos dominates. This re-scores the SAME saved detections on the two
distribution-robust quantities:

  * logo frames (n_gt > 0): TOP-1 hit rate -- did any detection match the
    LARGEST ground-truth box (the "hero" logo). Detections matching other GT
    boxes are ignored, not counted as FP (the 1.4b fix).
  * logo-free frames (n_gt == 0): FP per free frame at the same threshold.

Both are immune to how densely the other images are annotated, so the paired
columns are a fair OWLv2 vs YOLO comparison from a single ledger.

Usage:
    python scripts/owlv2_1p6_paired.py --root weights/logo_detector --split val \
        --out benchmark/results/owlv2_1p6_paired_val.json
"""
import argparse
import importlib.util as ilu
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_HERE = Path(__file__).resolve().parent


def _load(name):
    spec = ilu.spec_from_file_location(name, _HERE / f"{name}.py")
    mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sep = _load("eval_detector_separation")
ev = _load("eval_logo")
owl = _load("owlv2_separation")

THRS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)


def run_yolo(root, items, model_path, imgsz, device):
    from ultralytics import YOLO
    model = YOLO(model_path)
    rows = []
    for stem, gts, is_logo, img_area, sub, sp in items:
        img = root / sub / sp / f"{stem}.jpg"
        if not img.exists():
            continue
        r = model.predict(str(img), imgsz=imgsz, conf=0.001, device=device,
                          verbose=False)[0]
        d = r.boxes
        xyxy = d.xyxy.cpu().numpy() if d is not None and len(d) else np.zeros((0, 4))
        conf = d.conf.cpu().numpy() if d is not None and len(d) else np.zeros((0,))
        dets = np.hstack([xyxy, conf[:, None]]) if len(conf) else np.zeros((0, 5))
        rows.append({"image": stem, "group": sep.group_key(stem),
                     "is_logo_image": bool(is_logo), "sub": sub, "split": sp,
                     "n_gt": len(gts), "img_area": img_area,
                     "max_conf": float(conf.max()) if len(conf) else 0.0,
                     "dets": dets, "gts": np.asarray(gts, dtype=float)})
    return rows


def hero_metrics(rows, thrs, iou=0.5):
    logo = [r for r in rows if len(r["gts"])]
    free = [r for r in rows if not len(r["gts"])]
    out = {}
    for thr in thrs:
        hits = 0
        for r in logo:
            g = r["gts"]
            hero = g[((g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])).argmax()]
            d = r["dets"]
            d = d[d[:, 4] >= thr] if len(d) else d
            if any(ev._iou(det[:4], hero) >= iou for det in d):
                hits += 1
        fp = sum(int((r["dets"][:, 4] >= thr).sum()) if len(r["dets"]) else 0
                 for r in free)
        out[thr] = {"top1_hit_rate": round(hits / len(logo), 4) if logo else None,
                    "hits": hits, "n_logo": len(logo),
                    "fp_per_free_frame": round(fp / len(free), 4) if free else None,
                    "fp": fp, "n_free": len(free)}
    return out


def _serialize(rows):
    return [{**r, "dets": np.asarray(r["dets"]).tolist(),
             "gts": np.asarray(r["gts"]).tolist()} for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="weights/logo_detector")
    ap.add_argument("--split", default="val")
    ap.add_argument("--model-owlv2", default="google/owlv2-base-patch16-ensemble")
    ap.add_argument("--model-yolo", default="weights/logo_detector/train/weights/best.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--owlv2-dets", default=None)
    ap.add_argument("--yolo-dets", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    items = sep.read_split(root, args.split)
    print(f"{root} split={args.split}: {len(items)} images "
          f"({sum(1 for it in items if len(it[1]))} logo / "
          f"{sum(1 for it in items if not len(it[1]))} logo-free)")

    if args.yolo_dets and Path(args.yolo_dets).exists():
        rows_y = owl._rows_from_jsonable(json.load(open(args.yolo_dets)))
        print(f"YOLO dets loaded: {args.yolo_dets}")
    else:
        rows_y = run_yolo(root, items, args.model_yolo, args.imgsz, args.device)
        if args.yolo_dets:
            Path(args.yolo_dets).write_text(json.dumps({"rows": _serialize(rows_y)}))
            print(f"YOLO dets -> {args.yolo_dets}")

    if args.owlv2_dets and Path(args.owlv2_dets).exists():
        rows_o = owl._rows_from_jsonable(json.load(open(args.owlv2_dets)))
        print(f"OWLv2 dets loaded: {args.owlv2_dets}")
    else:
        rows_o, _ms = owl.run_owlv2(root, items, args.model_owlv2)
        if args.owlv2_dets:
            Path(args.owlv2_dets).write_text(json.dumps({"rows": _serialize(rows_o)}))
            print(f"OWLv2 dets -> {args.owlv2_dets}")

    m_o = hero_metrics(rows_o, THRS)
    m_y = hero_metrics(rows_y, THRS)
    print(f"\n=== paired (IoU 0.5) — logo frames: top-1 hit | logo-free: FP/frame ===")
    n_l, n_f = m_o[THRS[0]]["n_logo"], m_o[THRS[0]]["n_free"]
    print(f"  logo frames n={n_l}   logo-free frames n={n_f}")
    print(f"  {'conf':>6} | {'OWLv2 top1':>11} {'FP/free':>8} | "
          f"{'YOLO top1':>10} {'FP/free':>8}")
    for t in THRS:
        print(f"  {t:>6.2f} | {m_o[t]['top1_hit_rate']!s:>11} "
              f"{m_o[t]['fp_per_free_frame']!s:>8} | "
              f"{m_y[t]['top1_hit_rate']!s:>10} "
              f"{m_y[t]['fp_per_free_frame']!s:>8}")

    best_o = next((t for t in THRS if (m_o[t]["top1_hit_rate"] or 0) >= 0.3
                   and (m_o[t]["fp_per_free_frame"] if m_o[t]["fp_per_free_frame"]
                        is not None else 9) < 0.886), None)
    print(f"\n  OWLv2 threshold meeting top1>=0.3 AND FP/free<0.886: "
          f"{best_o if best_o is not None else 'NONE'}")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "root": str(root), "split": args.split,
            "owlv2": {"model": args.model_owlv2, "per_thr": m_o},
            "yolo": {"model": args.model_yolo, "per_thr": m_y},
            "pass_threshold_owlv2": best_o,
        }, indent=2))
        print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
