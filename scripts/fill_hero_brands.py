"""Fill the gold-box hero brand on each holdout logo frame — BLIND.

Shows each frame with ONLY the designated hero box drawn (no OCR text, no
detector boxes, no retrieval results) and prompts for the brand name on
stdin. Writes `brand` back into the hero box in labels.jsonl.

Usage:
    python3 scripts/fill_hero_brands.py \
        --labels benchmark/realvideo_v1/labels.jsonl

Per frame: press a key to advance (image window shows "brand? type on
terminal"), type the brand + Enter. Enter alone = skip this frame (brand
left blank, listed at the end). 'q' + Enter = quit and save what's done.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LABELS = ROOT / "benchmark/realvideo_v1/labels.jsonl"
DEFAULT_IMAGES = ROOT / "benchmark/realvideo_v1/extract/images"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(DEFAULT_LABELS))
    ap.add_argument("--images", default=str(DEFAULT_IMAGES))
    args = ap.parse_args()

    lp = Path(args.labels)
    print(f"writing to: {lp.resolve()}")
    rows = [json.loads(l) for l in lp.read_text().splitlines() if l.strip()]
    holdout = {"88b0502222", "df01d5a843"}
    targets = []
    for r in rows:
        md5 = r["file"].split("_", 1)[0]
        if md5 not in holdout:
            continue
        for b in r.get("boxes", []):
            if b.get("hero") and not b.get("brand"):
                targets.append((r, b))
    if not targets:
        print("no hero boxes missing a brand; nothing to do")
        return
    print(f"{len(targets)} hero boxes to brand (blind)")

    done = 0
    for i, (r, b) in enumerate(targets):
        stem = Path(r["file"]).stem
        for sp in ("test", "val", "train"):
            imgp = Path(args.images) / sp / f"{stem}.jpg"
            if imgp.is_file():
                break
        img = cv2.imread(str(imgp))
        if img is None:
            print(f"  ?? {r['file']} image not found; skipping")
            continue
        x, y, w, h = (int(round(b[k])) for k in ("x", "y", "w", "h"))
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 215, 255), 3)
        cv2.putText(img, "brand? (type on terminal)", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 215, 255), 1)
        cv2.imshow(f"hero {i+1}/{len(targets)} {r['file']}", img)
        cv2.waitKey(400)
        brand = input(f"{i+1}/{len(targets)} {r['file']}  brand (Enter=blank, "
                      f"q+Enter=quit): ").strip()
        if brand.lower() == "q":
            cv2.destroyAllWindows()
            break
        if brand:
            b["brand"] = brand.upper()
            done += 1
        # save after EVERY frame so an abort never loses work
        lp.write_text("".join(json.dumps(x) + "\n" for x in rows))
        cv2.destroyAllWindows()
    missing = [Path(r["file"]).stem for r, b in targets if not b.get("brand")]
    print(f"\nsaved {done}/{len(targets)} brands to {lp}")
    if missing:
        print("still blank:", missing)


if __name__ == "__main__":
    main()