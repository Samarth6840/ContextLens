"""Phase 1.9 — failure-cause analysis for arm-2 (OCR + full-frame retrieval) hero misses.

Phase 1.8 measured THAT arm 2 misses 18 of 24 hero frames; this script explains
WHY, per frame, by replaying each stage of the arm-2 pipeline and recording what
each stage produced. Per-stage outputs are persisted to the result JSON first
(they were NOT persisted by the 1.8 harness), then each miss is classified into
exactly one cause, in pipeline order (first failing stage wins):

  1. brand_not_in_catalog  — gold brand absent from src/brand_catalog.py
                             (match_brand(gold) is None). Structural: no
                             OCR-channel naming can ever succeed, and unless
                             the brand is also in the CLIP reference bank the
                             retrieval channel cannot name it either.
  2. ocr_miss              — gold IS in the catalog, full-frame OCR ran, but no
                             OCR region's text resolves to the gold brand.
  3. match_brand_reject     — gold IS in the catalog and an OCR region's raw
                             text CONTAINS the gold string (normalized
                             substring), but match_brand() rejects/overrides it.
  4. retrieval_too_small    — gold IS in the retrieval bank; the retrieval query
                             ran but returned no accepted candidate for gold
                             (below_similarity_floor / small_margin / not
                             top-1). Persisted: top1/top2/margin/diag reason.
  5. fusion_suppressed      — a channel DID name gold but arm 2's final set
                             lacks it. By construction arm 2 is a plain union
                             (no smoothing, no abstention), so this must be 0;
                             a nonzero count is asserted against as a bug.

Causes are attributed to the FIRST stage that blocks gold along the OCR path,
then retrieval. A frame may have latent problems downstream of the assigned
cause (e.g. not-in-bank AND ocr_miss); every stage's raw output is persisted so
those remain auditable.

Usage:
    python3 scripts/brand_recall_1p9_misses.py \
        --labels benchmark/realvideo_v1/labels.jsonl \
        --arms benchmark/results/realvideo_v1_1p8_arms.json \
        --out benchmark/results/realvideo_v1_1p9_misses.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.brand_catalog import match_brand, normalize_text  # noqa: E402


def load_hero_rows(labels: Path):
    rows = [json.loads(l) for l in labels.read_text().splitlines() if l.strip()]
    holdout = {"88b0502222", "df01d5a843"}
    heroes = []
    for r in rows:
        if r["file"].split("_", 1)[0] not in holdout or r["verdict"] != "logo":
            continue
        hero = [b for b in r["boxes"] if b.get("hero")]
        if not hero or not hero[0].get("brand"):
            continue
        heroes.append({"file": r["file"], "gold": hero[0]["brand"].upper()})
    return heroes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="benchmark/realvideo_v1/labels.jsonl")
    ap.add_argument("--images", default="benchmark/realvideo_v1/extract/images")
    ap.add_argument("--arms", default="benchmark/results/realvideo_v1_1p8_arms.json")
    ap.add_argument("--out", default="benchmark/results/realvideo_v1_1p9_misses.json")
    args = ap.parse_args()

    arms = json.loads(Path(args.arms).read_text())
    arm2_pf = arms["arm2_ocr_fullframe"]["per_frame"]
    heroes = load_hero_rows(Path(args.labels))
    misses = [h for h in heroes if not arm2_pf.get(h["file"], {}).get("hero_named")]
    print(f"arm2 hero misses to explain: {len(misses)} / {len(heroes)}")

    # ---- build the shipped components (lazy), same wiring as 1.8 ----
    import cv2
    from src.pipeline import Phase1Pipeline
    from src.layer2.brand_resolver import BrandResolver

    p = Phase1Pipeline()
    _br = p.cfg["layer1"]["logo_detection"]
    _lr = p.cfg["layer1"]["logo_retrieval"]
    print("loading OCR / retrieval ...", flush=True)

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
    bank = set(getattr(resolver.retrieval_index, "brands", []) or []) \
        if resolver.retrieval_index is not None else set()

    images_root = Path(args.images)
    findings = {}
    counts = {}
    for h in misses:
        fpath = None
        for sp in ("val", "test", "train"):
            cand = images_root / sp / h["file"]
            if cand.exists():
                fpath = cand
                break
        if fpath is None:
            raise SystemExit(f"FATAL: image not found for {h['file']}")
        frame = cv2.cvtColor(cv2.imread(str(fpath)), cv2.COLOR_BGR2RGB)
        gold = h["gold"]

        # ---- stage A: catalog membership (upstream of everything) ----
        in_catalog = match_brand(gold) == gold

        # ---- stage B: full-frame OCR -> per-region match_brand ----
        ocr_regions = p.ocr.extract_text_batch([frame])[0]
        ocr_texts = [r.get("text", "") for r in ocr_regions]
        ocr_matches = [match_brand(t) for t in ocr_texts]
        ocr_named = sorted({m.upper() for m in ocr_matches if m})
        gold_norm = normalize_text(gold)
        contains_gold = [
            t for t in ocr_texts
            if gold_norm and gold_norm in normalize_text(t)
        ]

        # ---- stage C: full-frame retrieval (candidates + rejection diag) ----
        cands, diag = resolver._retrieval_query(frame)
        retrieval_named = cands[0][0].upper() if cands else None

        # ---- stage D: what arm 2 actually said (from the 1.8 report) ----
        arm2_named = arm2_pf[h["file"]]["named"]
        arm2_named_set = set(arm2_named)

        # ---- classify: first blocking stage wins ----
        channels_named_gold = (gold in ocr_named) or (retrieval_named == gold)
        if channels_named_gold and gold not in arm2_named_set:
            cause = "fusion_suppressed"
            detail = "a channel named gold but the final arm2 set lacks it (bug)"
        elif not in_catalog:
            cause = "brand_not_in_catalog"
            detail = (f"match_brand({gold!r}) is None; in_retrieval_bank="
                      f"{gold in bank}")
        elif gold in ocr_named:
            cause = "fusion_suppressed"
            detail = "ocr_named contains gold but final set does not (bug)"
        elif contains_gold:
            cause = "match_brand_reject"
            detail = (f"OCR text contains {gold!r} but match_brand rejected it; "
                      f"texts={contains_gold[:3]!r}")
        elif not ocr_texts:
            cause = "ocr_miss"
            detail = "full-frame OCR returned zero text regions"
        elif gold in bank:
            cause = "retrieval_too_small"
            detail = (f"gold in bank; retrieval did not accept it: "
                      f"{json.dumps(diag, default=str)}")
        else:
            cause = "ocr_miss"
            detail = (f"gold in catalog but not in bank; OCR regions did not "
                      f"resolve to gold; texts={ocr_texts[:5]!r}")

        counts[cause] = counts.get(cause, 0) + 1
        findings[h["file"]] = {
            "gold": gold,
            "cause": cause,
            "detail": detail,
            "stages": {
                "catalog": {"in_catalog": in_catalog,
                            "match_brand_gold": match_brand(gold),
                            "in_retrieval_bank": gold in bank},
                "ocr": {"regions": ocr_texts, "per_region_match": ocr_matches,
                        "named": ocr_named, "texts_containing_gold": contains_gold},
                "retrieval": {"named_top1": retrieval_named,
                              "candidates": [[b, s] for b, s in cands],
                              "diag": diag},
                "arm2_final_named": arm2_named,
            },
        }
        print(f"  {h['file']:38} gold={gold:12} -> {cause}")

    # arm 2 is a plain union of both channels: suppression must be impossible.
    suppressed = counts.get("fusion_suppressed", 0)
    if suppressed:
        raise SystemExit(
            f"FATAL: fusion_suppressed={suppressed} — arm2 has no fusion/"
            f"abstention stage; a nonzero count means the harness union is broken"
        )

    ordered = ["brand_not_in_catalog", "ocr_miss", "match_brand_reject",
               "retrieval_too_small", "fusion_suppressed"]
    out = {
        "arms_file": args.arms,
        "arm2_hero_hits": sum(1 for h in heroes
                              if arm2_pf.get(h["file"], {}).get("hero_named")),
        "arm2_hero_total": len(heroes),
        "misses_classified": len(misses),
        "cause_counts": {k: counts.get(k, 0) for k in ordered},
        "precedence": ordered,
        "misses": findings,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    print("\n=== Phase 1.9 — arm2 hero-miss causes ===")
    for k in ordered:
        print(f"  {k:24} {out['cause_counts'][k]}")
    print(f"  {'TOTAL':24} {len(misses)}")
    print(f"-> {Path(args.out).resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
