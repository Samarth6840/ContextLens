"""Phase 1.7: OWLv2 vs fine-tuned detector on the new realvideo_v1 set.

Same distribution-robust quantities as 1.6 (top-1 hit on logo frames, FP per
logo-free frame) but on real sampled video frames split by video:
  * logo frames: TOP-1 hit rate -- did any detection match the LARGEST GT box
  * logo-free frames: FP per free frame at the same threshold

Circularity: the detector arm is NOT graded on frames whose boxes all came from
that detector's own proposals (label_frames export writes model_prelabelled).
OWLv2 is a different model, so those frames stay in the OWLv2 arm.

Dets are frozen with --run-dets BEFORE the review (like 1.6), so grading later
is pure re-scoring against reviewed labels:

    python3 scripts/owlv2_1p7_grade.py \
        --frames benchmark/realvideo_v1/extract/images --run-dets
    # ... human review via label_frames.py ... then:
    python3 scripts/owlv2_1p7_grade.py \
        --frames benchmark/realvideo_v1/extract/images \
        --export benchmark/realvideo_v1/export
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


def items_from_frames(frames_root: Path):
    """Build read_split-shaped items from the raw frame tree (no labels yet)."""
    out = []
    for sp in ("train", "val", "test"):
        d = frames_root / sp
        if not d.is_dir():
            continue
        for img in sorted(d.glob("*.jpg")):
            w, h = sep._imsize(img)
            out.append((img.stem, np.zeros((0, 4)), False, w * h, "", sp))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="benchmark/realvideo_v1/extract/images")
    ap.add_argument("--export", default="benchmark/realvideo_v1/export")
    ap.add_argument("--run-dets", action="store_true")
    ap.add_argument("--yolo-dets", default="benchmark/results/realvideo_v1_yolo.json")
    ap.add_argument("--owlv2-dets", default="benchmark/results/realvideo_v1_owlv2.json")
    ap.add_argument("--model-yolo", default="weights/logo_detector/train/weights/best.pt")
    ap.add_argument("--model-owlv2", default="google/owlv2-base-patch16-ensemble")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    frames = Path(args.frames)
    items = items_from_frames(frames)
    print(f"{len(items)} frames across train/val/test "
          f"({sum(1 for i in items if i[2])} logo / "
          f"{sum(1 for i in items if not i[2])} logo-free by empty label)")

    if args.run_dets:
        rows_o, _ms = owl.run_owlv2(frames, items, args.model_owlv2)
        Path(args.owlv2_dets).write_text(json.dumps({
            "images": {r["image"]: {"split": r["split"],
                                    "dets": np.asarray(r["dets"]).tolist()}
                       for r in rows_o}}))
        print(f"  owlv2 dets ({len(rows_o)}) -> {args.owlv2_dets}  "
              f"mean {_ms.mean():.0f} ms/frame")
        # YOLO: reuse the same loop 1.6 uses (per-file predict, conf 0.001)
        roots: dict[str, list[tuple]] = {}
        for it in items:
            roots.setdefault(it[5], []).append(it)
        from ultralytics import YOLO
        model = YOLO(args.model_yolo)
        rows_y = {}
        for sp, its in roots.items():
            for stem, _g, _l, _a, sub, _s in its:
                img = frames / sub / sp / f"{stem}.jpg"
                r = model.predict(str(img), imgsz=640, conf=0.001,
                                  device=args.device, verbose=False)[0]
                d = r.boxes
                xy = d.xyxy.cpu().numpy() if d is not None and len(d) else \
                    np.zeros((0, 4))
                cf = d.conf.cpu().numpy() if d is not None and len(d) else \
                    np.zeros((0,))
                rows_y[stem] = np.hstack([xy, cf[:, None]]).tolist() if len(cf) \
                    else np.zeros((0, 5)).tolist()
        Path(args.yolo_dets).write_text(json.dumps(rows_y))
        print(f"  yolo dets ({len(rows_y)}) -> {args.yolo_dets}")
        print("dets frozen. run WITHOUT --run-dets after the review.")
        return 0

    yolo_dets = json.loads(Path(args.yolo_dets).read_text())
    owl_dets = json.loads(Path(args.owlv2_dets).read_text())["images"]
    export = Path(args.export)
    # circular exclusion for the DETECTOR arm, from label_frames export
    excl = set()
    mp = export / "model_prelabelled.json"
    if mp.is_file():
        excl = {r["file"].removesuffix(".jpg") for r in json.loads(mp.read_text())}
    items_exp = sep.read_split(export, "all")
    print(f"export: {len(items_exp)} reviewed frames "
          f"({sum(1 for i in items_exp if len(i[1]))} logo)")

    # Pass/fail runs on the POOLED holdout (val+test videos); per-video numbers
    # ride alongside because two 80-frame videos are too small each to carry a
    # call. Holdout identity is the video md5 (frozen in splits.json), NOT the
    # export dir name: exporting a val-only subset makes split_order(2) hand the
    # val video a "train" dir, so filtering by dir would silently drop it.
    HOLDOUT_MD5 = ("88b0502222", "df01d5a843")  # val + test, per splits.json

    def in_holdout(stem: str) -> bool:
        return stem.split("_", 1)[0] in HOLDOUT_MD5

    # OWLv2 persists per-frame {"split":..., "dets":[...]}; YOLO persists the
    # bare numeric list. Normalize to the (N,5) matrix once per frame.
    def det_rows(dets: dict, stem: str) -> np.ndarray:
        if stem not in dets:
            return np.zeros((0, 5), dtype=float)
        v = dets[stem]
        if isinstance(v, dict):
            v = v.get("dets") or []
        a = np.asarray(v, dtype=float)
        if a.ndim == 1:
            a = a.reshape(-1, 5)
        return a

    def grade(dets: dict, skip: set[str], holdout: bool = True):
        logo = [r for r in items_exp if len(r[1]) and r[0] not in skip and
                r[0] in dets and (not holdout or in_holdout(r[0]))]
        free = [r for r in items_exp if not len(r[1]) and r[0] in dets and
                (not holdout or in_holdout(r[0]))]
        out = {}
        for thr in THRS:
            hits = 0
            for r in logo:
                g = r[1]
                hero = g[((g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])).argmax()]
                d = det_rows(dets, r[0])
                d = d[d[:, 4] >= thr] if len(d) else d
                if any(ev._iou(x[:4], hero) >= 0.5 for x in d):
                    hits += 1
            fp = sum(int((det_rows(dets, r[0])[:, 4] >= thr).sum())
                     for r in free)
            out[thr] = {"top1_hit_rate": round(hits / len(logo), 4) if logo
                        else None,
                        "hits": hits, "n_logo": len(logo),
                        "fp_per_free_frame": round(fp / len(free), 4) if free
                        else None,
                        "fp": fp, "n_free": len(free)}
        return out

    # per-video (== per-md5) tables, then the pooled decision
    def grade_video(dets: dict, skip: set[str], vid: str):
        lg = [r for r in items_exp if len(r[1]) and r[0] not in skip and
              r[0] in dets and r[0].startswith(vid)]
        fr = [r for r in items_exp if not len(r[1]) and r[0] in dets and
              r[0].startswith(vid)]
        t = {}
        for thr in THRS:
            hits = 0
            for r in lg:
                hero = r[1][((r[1][:, 2] - r[1][:, 0]) *
                             (r[1][:, 3] - r[1][:, 1])).argmax()]
                d = det_rows(dets, r[0])
                d = d[d[:, 4] >= thr] if len(d) else d
                if any(ev._iou(x[:4], hero) >= 0.5 for x in d):
                    hits += 1
            fp = sum(int((det_rows(dets, r[0])[:, 4] >= thr).sum())
                     for r in fr)
            t[thr] = {"top1_hit_rate": round(hits / len(lg), 4) if lg else None,
                      "hits": hits, "n_logo": len(lg),
                      "fp_per_free_frame": round(fp / len(fr), 4) if fr else None,
                      "fp": fp, "n_free": len(fr)}
        return t

    per_split = {vid: {"owlv2": grade_video(owl_dets, set(), vid),
                       "yolo": grade_video(yolo_dets, excl, vid)}
                 for vid in HOLDOUT_MD5}
    mo = grade(owl_dets, set())
    my = grade(yolo_dets, excl)
    n_l, n_f = mo[THRS[0]]["n_logo"], mo[THRS[0]]["n_free"]
    print(f"\n=== held-out pooled = val+test: logo frames n={n_l}, "
          f"free frames n={n_f} ===")
    print(f"  {'conf':>6} | {'OWLv2 top1':>11} {'FP/free':>8} | "
          f"{'YOLO top1':>10} {'FP/free':>8}")
    for t in THRS:
        print(f"  {t:>6.2f} | {mo[t]['top1_hit_rate']!s:>11} "
              f"{mo[t]['fp_per_free_frame']!s:>8} | "
              f"{my[t]['top1_hit_rate']!s:>10} {my[t]['fp_per_free_frame']!s:>8}")
    for vid in HOLDOUT_MD5:
        a, b = per_split[vid]["owlv2"], per_split[vid]["yolo"]
        print(f"  [{vid}] logo n={a[THRS[0]]['n_logo']} free n="
              f"{a[THRS[0]]['n_free']}")
        for t in (0.05, 0.10, 0.25):
            print(f"    {t:>5.2f} OWLv2 {a[t]['top1_hit_rate']!s}/"
                  f"{a[t]['fp_per_free_frame']!s}   "
                  f"YOLO {b[t]['top1_hit_rate']!s}/{b[t]['fp_per_free_frame']!s}")
    # criterion: some OWLv2 threshold with top1>=0.3, FP/free below detector's at
    # matched top-1 recall (nearest threshold the detector reaches or exceeds),
    # by MORE THAN 10 FRAMES on the pooled 160. Ties fail.
    best = next((t for t in THRS if (mo[t]["top1_hit_rate"] or 0) >= 0.3), None)
    margin_frames = None
    if best is not None:
        det_match = next((t for t in THRS
                          if (my[t]["top1_hit_rate"] or 0) >= mo[best]["top1_hit_rate"]),
                         None)
        md = my[det_match] if det_match is not None else None
        print(f"\n  OWLv2 @conf {best}: top1 {mo[best]['top1_hit_rate']} "
              f"({mo[best]['hits']}/{mo[best]['n_logo']}) "
              f"FP/free {mo[best]['fp_per_free_frame']} "
              f"({mo[best]['fp']}/{mo[best]['n_free']})")
        if md is not None:
            margin_frames = md["fp"] - mo[best]["fp"]
            print(f"  YOLO at matched top1 (conf {det_match}): "
                  f"FP/free {md['fp_per_free_frame']} "
                  f"({md['fp']}/{md['n_free']})")
            print(f"  FP margin (YOLO - OWLv2): {margin_frames} FRAMES "
                  f"({'PASS' if margin_frames > 10 else 'FAIL: <=10, tie'})")
        else:
            print(f"  YOLO never reaches top1 {mo[best]['top1_hit_rate']} "
                  f"(best {my[THRS[0]]['top1_hit_rate']} at conf {THRS[0]})")
    else:
        print(f"\n  NO OWLv2 threshold reaches top-1 >= 0.3 "
              f"({mo[THRS[0]]['n_logo']} logo frames reviewed) "
              f"-> FAIL")
    out = {"pooled": {"owlv2": mo, "yolo": my}, "per_split": per_split,
           "excluded_circular_for_yolo": len(excl),
           "chosen_owlv2_conf": best,
           "fp_margin_frames": margin_frames,
           "pass": best is not None and margin_frames is not None
                   and margin_frames > 10,
           "note": "thresholds swept on the same frames flatter both arms about "
                   "equally (selection bias); df01d5a843 was the old val video "
                   "and 88b0502222 the old frozen_test, so both were seen in "
                   "earlier decisions - holdout is fresh-split, not unseen."}
    Path(args.yolo_dets).with_suffix(".grade.json").write_text(
        json.dumps(out, indent=2))
    print(f"  -> {Path(args.yolo_dets).with_suffix('.grade.json')}")


if __name__ == "__main__":
    main()