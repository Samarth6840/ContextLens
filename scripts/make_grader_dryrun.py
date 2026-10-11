"""Synthetic dry-run for owlv2_1p7_grade.py — deterministic, known answers.

Layout mimics a post-export review dir:
  export/images/{val,test}/<md5>_<idx>.jpg   (200x100 solid grey)
  export/labels/{val,test}/<md5>_<idx>.txt    (YOLO cls cx cy bw bh | empty=free)
  export/model_prelabelled.json               (one logo frame -> circular-
                                               excluded from the YOLO arm only)
  export/heroes.json                          {stem: {x,y,w,h}} pixel coords —
                                               the reviewer's hero designation;
                                               may be a box that is NOT the
                                               largest (top-1 must use it anyway)

Scenario (all frames are synthetic; NO holdout):
  val  88b0502222: 4 logo + 2 free
  test df01d5a843: 4 logo + 2 free
  pooled: n_logo=8, n_free=4
  hero box (200x100 frame): x1=80 y1=40 x2=120 y2=60  (SMALL, designated hero)
  Every logo frame ALSO carries a LARGER, non-overlapping box (billboard)
  x1=0 y1=60 x2=199 y2=99. ioU(hero, billboard)=0, so a grader that fell back
  to argmax-area would pick the billboard and MISS the hero detection: the
  hero designation is the only box the top-1 hit can match.
  ONE logo frame (df01d5a843_000001) is BILLBOARD-ONLY: it carries the large
  non-hero box but NO hero designation. Protocol: it stays a logo frame
  (composition n_logo) but is excluded from the top-1 pool (n_hero).
  OWLv2: every logo frame -> 1 hit on the HERO box @ conf .5 ; free -> no dets
        (the billboard-only frame's own det is inert: not in the top-1 pool)
  YOLO: every logo frame -> 1 hit on the HERO box @ conf .5 ; free -> N dets
        each at conf .5. Frozen dets are the detector's output regardless of
        circularity; model_prelabelled.json ALONE decides the exclusion that
        removes one logo frame from the YOLO arm (when the file is present).

Expectations:
  n_logo=8, n_hero=7 (one billboard-only logo frame has no hero), n_free=4.
  OWLv2 top1 @conf .05 = 7/7 = 1.0 (hits the HERO, not the big billboard),
  FP/free = 0.
  YOLO matched top1 = 1.0 (hits 6/6, prelabelled excluded).
  FP margin (YOLO - OWLv2) = N*4 - 0.
  pass = margin > 10  ->  N=2 (8) => FAIL tie;  N=3 (12) => PASS.
"""
import argparse, json, shutil
from pathlib import Path
import cv2, numpy as np

p = argparse.ArgumentParser()
p.add_argument("--root", default="/tmp/owlv2_1p7_dryrun")
p.add_argument("--fps-per-free", type=int, default=2)
args = p.parse_args()

R = Path(args.root)
shutil.rmtree(R, ignore_errors=True)
W, H = 200, 100
hero = (80, 40, 120, 60)              # x1,y1,x2,y2 — DESIGNATED hero, small
billboard = (0, 60, 199, 99)          # larger, non-overlapping logo box
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
billboard_only = ["df01d5a843_000001"]
heroes = {}
for stem, sp, kind in frames:
    (R / "images" / sp).mkdir(parents=True, exist_ok=True)
    (R / "labels" / sp).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(R / "images" / sp / f"{stem}.jpg"), img)
    if kind == "logo":
        # two boxes per logo frame: the small designated hero AND a larger
        # non-overlapping billboard. ioU(hero,billboard)=0 -> only the grader
        # that respects the hero designation scores the hero det as a hit.
        # The billboard-only frame keeps only the large non-hero box: it is a
        # logo frame with NO hero, so it counts in n_logo but not in n_hero.
        boxes = (hero, billboard) if stem not in billboard_only else (billboard,)
        txt = [f"0 {(a[0]+a[2])/2/W:.6f} {(a[1]+a[3])/2/H:.6f} "
               f"{(a[2]-a[0])/W:.6f} {(a[3]-a[1])/H:.6f}"
               for a in boxes]
        (R / "labels" / sp / f"{stem}.txt").write_text("\n".join(txt) + "\n")
        if stem not in billboard_only:
            heroes[stem] = {"x": hero[0], "y": hero[1],
                            "w": hero[2] - hero[0], "h": hero[3] - hero[1]}
    else:
        (R / "labels" / sp / f"{stem}.txt").write_text("")
(R / "model_prelabelled.json").write_text(
    json.dumps([{"file": s + ".jpg"} for s in prelabelled]))
(R / "heroes.json").write_text(json.dumps(heroes))

def detbox():
    return [hero[0], hero[1], hero[2], hero[3], 0.5]

owlv2, yolo = {}, {}
for stem, sp, kind in frames:
    if kind == "logo":
        # OWLv2 sees the prelabelled frame too (different model, no circularity).
        owlv2[stem] = {"split": sp, "dets": [detbox()]}
        yolo[stem] = {"dets": [detbox()]}
    else:
        owlv2[stem] = {"split": sp, "dets": []}
        yolo[stem] = {"dets": [detbox()] * args.fps_per_free}
(R / "realvideo_v1_owlv2.json").write_text(json.dumps({"images": owlv2}))
(R / "realvideo_v1_yolo.json").write_text(json.dumps(yolo))

print(f"scenario: fps_per_free={args.fps_per_free} -> margin={args.fps_per_free*4}")
print("expect: n_logo=8 (7 hero + 1 billboard-only) n_free=4 ; "
      "OWLv2 top1=1.0 (7/7 hero hits, NOT the larger billboard) FP/free=0 ; "
      f"YOLO top1=1.0 (6/6); margin={args.fps_per_free*4} -> "
      f"{'PASS' if args.fps_per_free*4 > 10 else 'FAIL tie'}")