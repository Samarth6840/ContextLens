"""Synthetic dry-run for owlv2_1p7_grade.py — deterministic, known answers.

Layout mimics a post-export review dir:
  export/images/{val,test}/<md5>_<idx>.jpg   (200x100 solid grey)
  export/labels/{val,test}/<md5>_<idx>.txt    (YOLO cls cx cy bw bh | empty=free)
  export/model_prelabelled.json               (one logo frame -> circular-
                                               excluded from the YOLO arm only)

Scenario (all frames are synthetic; NO holdout):
  val  88b0502222: 4 logo + 2 free
  test df01d5a843: 4 logo + 2 free
  pooled: n_logo=8, n_free=4
  hero box (any 200x100 -> normalized): x1=40 y1=20 x2=120 y2=60
  OWLv2: every logo frame -> 1 hero det @ conf .5 ; free -> no dets
  YOLO: every logo frame -> 1 hero det @ conf .5 (one prelabelled frame
        excluded -> its det is a no-op for YOLO but STAYS for OWLv2);
        free -> N dets each at conf .5

Expectations:
  OWLv2 top1 @conf .05 = 8/8 = 1.0, FP/free = 0.
  best threshold = .05, top1 1.0 (hits 8/8).
  YOLO matched top1 = 1.0 (hits 7/7, prelabelled excluded).
  YOLO FP/free = N at conf .05 (fp = N*4).
  FP margin (YOLO - OWLv2) = N*4 - 0.
  pass = margin > 10  ->  N=2 (8) => FAIL tie;  N=3 (12) => PASS.
"""
import argparse, json, shutil, sys
from pathlib import Path
import cv2, numpy as np

p = argparse.ArgumentParser()
p.add_argument("--root", default="/tmp/owlv2_1p7_dryrun")
p.add_argument("--fps-per-free", type=int, default=2)
args = p.parse_args()

R = Path(args.root)
shutil.rmtree(R, ignore_errors=True)
W, H = 200, 100
hero = (40, 20, 120, 60)
img = np.full((H, W, 3), 128, np.uint8)

frames = []
for i in range(4):
    frames.append((f"88b0502222_{i:06d}", "val", "logo"))
for i in range(4, 6):
    frames.append((f"88b0502222_{i:06d}", "val", "free"))
for i in range(4):
    frames.append((f"df01d5a843_{i:06d}", "test", "logo"))
for i in range(4, 6):
    frames.append((f"df01d5a843_{i:06d}", "test", "free"))

prelabelled = ["88b0502222_000000"]
for stem, sp, kind in frames:
    (R / "images" / sp).mkdir(parents=True, exist_ok=True)
    (R / "labels" / sp).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(R / "images" / sp / f"{stem}.jpg"), img)
    if kind == "logo":
        txt = [f"0 {((hero[0]+hero[2])/2)/W:.6f} {((hero[1]+hero[3])/2)/H:.6f} "
               f"{(hero[2]-hero[0])/W:.6f} {(hero[3]-hero[1])/H:.6f}"]
        (R / "labels" / sp / f"{stem}.txt").write_text("\n".join(txt) + "\n")
    else:
        (R / "labels" / sp / f"{stem}.txt").write_text("")
(R / "model_prelabelled.json").write_text(
    json.dumps([{"file": s + ".jpg"} for s in prelabelled]))

def detbox():
    return [hero[0], hero[1], hero[2], hero[3], 0.5]

owlv2, yolo = {}, {}
for stem, sp, kind in frames:
    if kind == "logo":
        # OWLv2 sees the prelabelled frame too (different model, no circularity)
        owlv2[stem] = {"split": sp, "dets": [detbox()]}
        yolo[stem] = {"dets": [] if stem in prelabelled else [detbox()]}
    else:
        owlv2[stem] = {"split": sp, "dets": []}
        yolo[stem] = {"dets": [detbox()] * args.fps_per_free}
(R / "realvideo_v1_owlv2.json").write_text(json.dumps({"images": owlv2}))
(R / "realvideo_v1_yolo.json").write_text(json.dumps(yolo))

print(f"scenario: fps_per_free={args.fps_per_free} -> margin={args.fps_per_free*4}")
print("expect: OWLv2 top1=1.0 (8/8) FP/free=0.0 ; YOLO top1=1.0 (7/7) "
      f"FP/free={args.fps_per_free} (fp={args.fps_per_free*4}) ; "
      f"PASS iff {args.fps_per_free*4} > 10 -> "
      f"{'PASS' if args.fps_per_free*4 > 10 else 'FAIL tie'}")