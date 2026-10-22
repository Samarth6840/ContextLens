"""Phase 1.8 — two-arm hero-brand recall + false-brand measurement.

Decides between Phase 3.2 (detector-based logo resolution) and an
OCR+CLIP-primary pipeline, per the pre-registered rule:

  Arm 1 (as shipped): logo detector -> BrandResolver (crop OCR + CLIP
    retrieval + fusion priority) -> temporal smoothing. The shipped brand
    surface for on-screen logos.
  Arm 2 (detector disabled): full-frame OCR matched against the catalog +
    full-frame CLIP retrieval (same gates as the resolver uses). OCR-primary.

Metrics on the SAME labeled holdout frames (159 of the 160 extracted —
`df01d5a843_017480.jpg` is an all-black tail frame that was deferred at
labeling and never judged; the run prints the accounting so the exclusion
cannot be silent):
  * hero-brand recall  = # of the 24 hero-designated frames where the arm's
    per-frame brand set contains the hero box's gold brand (from labels.jsonl).
  * false brands per free frame = brand attributions on the 135 logo-free
    frames, divided by 135 (a free frame carries no logo, so ANY brand the
    arm names on it is false).

Neither arm fits on the holdout; both reuse the shipped components and the
exact config the pipeline uses. ASR and the DINOv2 product index are shared
/ orthogonal channels and are excluded from BOTH arms so only the
detector-vs-OCR question moves the numbers.

Usage:
    python3 scripts/brand_recall_1p8.py \
        --labels benchmark/realvideo_v1/labels.jsonl \
        --out benchmark/results/realvideo_v1_1p8_arms.json
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.brand_catalog import match_brand  # noqa: E402


def load_rows(labels: Path):
    rows = [json.loads(l) for l in labels.read_text().splitlines() if l.strip()]
    holdout = {"88b0502222", "df01d5a843"}
    out = []
    for r in rows:
        if r["file"].split("_", 1)[0] not in holdout:
            continue
        v = r["verdict"]
        if v == "logo":
            hero = [b for b in r["boxes"] if b.get("hero")]
            brand = hero[0].get("brand") if hero else None
            if not brand:
                raise SystemExit(f"FATAL: hero frame {r['file']} lacks a brand")
            out.append({"file": r["file"], "video": r["video"],
                        "frame_index": r["frame_index"], "kind": "hero",
                        "gold": brand.upper()})
        elif v == "free":
            out.append({"file": r["file"], "video": r["video"],
                        "frame_index": r["frame_index"], "kind": "free", "gold": None})
    return out


def frame_paths(rows, images_root: Path) -> list[Path]:
    found = []
    stem = {Path(r["file"]).stem for r in rows}
    for sp in ("val", "test", "train"):
        for f in (images_root / sp).glob("*.jpg"):
            if f.stem in stem:
                found.append(f)
    return sorted(found, key=lambda p: p.name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="benchmark/realvideo_v1/labels.jsonl")
    ap.add_argument("--images", default="benchmark/realvideo_v1/extract/images")
    ap.add_argument("--out", default="benchmark/results/realvideo_v1_1p8_arms.json")
    args = ap.parse_args()

    rows = load_rows(Path(args.labels))
    heroes = [r for r in rows if r["kind"] == "hero"]
    frees = [r for r in rows if r["kind"] == "free"]
    rkey = {Path(r["file"]).stem: r for r in rows}
    print(f"holdout: {len(heroes)} hero frames, {len(frees)} free frames")
    # Extraction-vs-label accounting: the holdout extractor writes 80 frames per
    # video, but a deferred (SKIP) frame never receives a verdict row, so the
    # labeled pool can be smaller than the extraction. Report the gap loudly
    # instead of silently pooling over it (df01d5a843_017480.jpg, an all-black
    # tail frame, is the known case: 160 extracted, 159 labeled).
    images_root = Path(args.images)
    extracted = sorted(
        f.name for sp in ("val", "test") for f in (images_root / sp).glob("*.jpg")
    )
    labeled_files = {r["file"] for r in rows}
    unlabeled = [f for f in extracted if f not in labeled_files]
    if unlabeled:
        print(f"WARNING: {len(unlabeled)} extracted holdout frame(s) have no "
              f"label row (excluded from every metric below): {unlabeled}",
              flush=True)
    by_video = defaultdict(lambda: {"hero": 0, "free": 0})
    for r in rows:
        by_video[r["video"]][r["kind"]] += 1
    for v, c in sorted(by_video.items()):
        print(f"  {v}: {c['hero']} hero, {c['free']} free")

    # ---- build the shipped components (lazy) ----
    import cv2
    from src.pipeline import Phase1Pipeline
    from src.layer2.brand_resolver import BrandResolver, TemporalBrandSmoother

    p = Phase1Pipeline()
    _br = p.cfg["layer1"]["logo_detection"]
    _lr = p.cfg["layer1"]["logo_retrieval"]
    print("loading logo detector / OCR / retrieval ...", flush=True)

    files = frame_paths(rows, Path(args.images))
    want = {Path(r["file"]).stem for r in rows}
    files = [f for f in files if f.stem in want]
    print(f"loaded {len(files)} holdout frame images", flush=True)
    frames = [cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB) for f in files]

    resolver = BrandResolver(
        ocr_extractor=p.ocr,
        class_confidence=_br.get("class_confidence", 0.55),
        crop_scale=_br.get("crop_scale", 2.0),
        retrieval_index=p.logo_retrieval if _lr.get("enabled", True) else None,
        retrieval_min_similarity=_lr.get("min_similarity", 0.65),
        retrieval_min_margin=_lr.get("min_margin", 0.10),
        screen_content_filter=_br.get("screen_content_filter", {}).get("enabled", True),
        class_require_corroboration=_br.get("class_require_corroboration", True),
        max_logo_area_fraction=_br.get("max_logo_area_fraction", 0.50),
        max_unknown_per_frame=_br.get("max_unknown_per_frame", 10),
        superset_margin_ratio=_br.get("superset_margin_ratio", 0.45),
        progress=None,
    )
    smooth = TemporalBrandSmoother(
        window=int(_br.get("temporal_smoothing", {}).get("window", 2)),
        min_iou=float(_br.get("temporal_smoothing", {}).get("min_iou", 0.3)),
        min_votes=int(_br.get("temporal_smoothing", {}).get("min_votes", 2)),
    )

    # ---- ARM 1: detached logo boxes -> resolver -> smoother ----
    print("ARM 1: detector -> BrandResolver -> temporal smoothing ...", flush=True)
    dets = p.logo_detector.detect_batch(frames)
    resolved = resolver.resolve(dets, frames)
    resolved = smooth.smooth(resolved)
    arm1_brands = [
        {d.get("brand") for d in fd if d.get("brand")} for fd in resolved
    ]

    # ---- ARM 2: full-frame OCR + full-frame CLIP retrieval ----
    print("ARM 2: full-frame OCR + full-frame retrieval ...", flush=True)
    arm2_brands = [set() for _ in frames]
    ocr_res = p.ocr.extract_text_batch(frames)
    # Superset makes the full-frame read reusable for the whole frame; the
    # gate is the same one the resolver applies to a crop.
    for fi, fres in enumerate(ocr_res):
        for r in fres:
            b = match_brand(r.get("text", ""))
            if b:
                arm2_brands[fi].add(b.upper())
    for fi, frm in enumerate(frames):
        cands, _diag = resolver._retrieval_query(frm)
        if cands:
            arm2_brands[fi].add(cands[0][0].upper())

    # ---- score ----
    def score(brands_per_frame):
        hits, false_counts = 0, []
        per_frame = {}
        for f, bset in zip(files, brands_per_frame):
            info = rkey[f.stem]
            if info["kind"] == "hero":
                hit = info["gold"] in bset
                per_frame[f.name] = {"kind": "hero", "gold": info["gold"],
                                     "named": sorted(bset), "hero_named": hit}
                hits += int(hit)
            else:
                per_frame[f.name] = {"kind": "free", "named": sorted(bset),
                                     "false_count": len(bset)}
                false_counts.append(len(bset))
        n_free = max(1, len(false_counts))
        return hits, sum(false_counts) / n_free, sum(false_counts), per_frame

    arm1_hits, arm1_fpp, arm1_false, arm1_pf = score(arm1_brands)
    arm2_hits, arm2_fpp, arm2_false, arm2_pf = score(arm2_brands)

    # per-video breakdown
    def per_video(brands_per_frame):
        out = {}
        for f, bset in zip(files, brands_per_frame):
            info = rkey[f.stem]
            ent = out.setdefault(info["video"], {"hero": {"hits": 0, "n": 0},
                                                 "free": {"false": 0, "n": 0}})
            if info["kind"] == "hero":
                ent["hero"]["n"] += 1
                ent["hero"]["hits"] += info["gold"] in bset
            else:
                ent["free"]["n"] += 1
                ent["free"]["false"] += len(bset)
        return out

    rpt = {
        "frames": {"hero": len(heroes), "free": len(frees),
                   "labeled_total": len(rows), "extracted_holdout": len(extracted),
                   "unlabeled_excluded": unlabeled},
        "arm1_as_shipped": {
            "hero_brand_recall": f"{arm1_hits}/{len(heroes)}",
            "hero_brand_recall_frac": round(arm1_hits / max(1, len(heroes)), 4),
            "false_brands_per_free": round(arm1_fpp, 3),
            "false_brands_total": arm1_false,
            "per_video": per_video(arm1_brands),
            "per_frame": arm1_pf,
        },
        "arm2_ocr_fullframe": {
            "hero_brand_recall": f"{arm2_hits}/{len(heroes)}",
            "hero_brand_recall_frac": round(arm2_hits / max(1, len(heroes)), 4),
            "false_brands_per_free": round(arm2_fpp, 3),
            "false_brands_total": arm2_false,
            "per_video": per_video(arm2_brands),
            "per_frame": arm2_pf,
        },
        "decision": "OCR-primary wins if arm2 recall within 3 frames of arm1 or "
                    "better AND fewer false brands per free frame; Phase 3.2 only "
                    "if arm1 beats arm2 on recall by more than 3 frames.",
    }
    r1, r2 = arm1_hits, arm2_hits
    recall_ok = r2 >= r1 - 3
    fp_ok = arm2_fpp < arm1_fpp
    if recall_ok and fp_ok:
        rpt["verdict"] = "OCR_Primary"
    elif r1 - r2 > 3:
        rpt["verdict"] = "Detector_Fusion_3_2"
    else:
        rpt["verdict"] = "Inconclusive"
    n_fp = arm1_fpp if not fp_ok else arm2_fpp
    rpt["decision_detail"] = {
        "arm2_recall_within_3_or_better": recall_ok,
        "arm2_fewer_false_per_free": fp_ok,
        "recall_delta_arm1_minus_arm2": r1 - r2,
        "false_delta_arm1_minus_arm2": round(arm1_fpp - arm2_fpp, 3),
        "winning_false_rate": round(n_fp, 3),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rpt, indent=2))
    print("\n=== Phase 1.8 — two-arm brand recall ===")
    print(f"  ARM 1 (as shipped, detector):      hero recall {arm1_hits}/{len(heroes)}"
          f"   false brands/free {arm1_fpp:.3f}")
    print(f"  ARM 2 (OCR + full-frame retrieval): hero recall {arm2_hits}/{len(heroes)}"
          f"   false brands/free {arm2_fpp:.3f}")
    print(f"  VERDICT: {rpt['verdict']}")
    print(f"-> {Path(args.out).resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())