"""Ground truth that silently disagrees with what was reviewed is worse than
no ground truth. Three ways that happened here:

  - Store.put appended, so re-judging a frame left its key twice in the file
    while the in-memory dict stayed correct. Line counts and key counts then
    disagree with the reviewer's actual work.
  - A frame marked logo with no boxes is a claim of "logo here" with no
    location, which cannot grade a detector.
  - A wall of SKIP rows drained the pool to 100/100 while 88 frames had no
    verdict at all, and a test set with no free frames cannot yield a
    false-positive rate at all.

So the freeze decision is a script, not a judgement call made at the moment it
matters. Exit 0 = safe to commit. Exit 1 = the file is not evaluation data yet.

    python3 scripts/check_label_store.py                  # consistency, all sets
    python3 scripts/check_label_store.py --freeze benchmark/frozen_test/labels.jsonl
"""
import argparse
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.label_frames import Store  # noqa: E402

TINY = 0.01   # a logo under 1% of frame area: the bucket the audit scores 0/9


def sizes(path: Path, rows: list[dict]) -> Counter:
    """Bucket every box by share of its own frame. Needs the real frame, since
    a crop and a full frame make the same box a different size class."""
    out: Counter = Counter()
    for r in rows:
        img = path.parent / "images" / r["file"]
        if not img.is_file():
            continue
        h, w = cv2.imread(str(img)).shape[:2]
        for b in r["boxes"]:
            a = b["w"] * b["h"] / (w * h)
            out["tiny" if a < TINY else "small" if a < 0.05
                else "medium" if a < 0.2 else "large"] += 1
    return out


def check(path: Path) -> None:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    keys = [r["file"] for r in rows]

    dupes = {k for k in keys if keys.count(k) > 1}
    assert not dupes, f"duplicate frame keys: {sorted(dupes)}"
    assert len(keys) == len(set(keys)), "line count != unique key count"

    empty_logo = [r["file"] for r in rows
                  if r["verdict"] == "logo" and not r["boxes"]]
    assert not empty_logo, f"logo verdict with zero boxes: {empty_logo}"

    boxes_on_free = [r["file"] for r in rows
                     if r["verdict"] == "free" and r["boxes"]]
    assert not boxes_on_free, f"free verdict carrying boxes: {boxes_on_free}"

    judged = [r for r in rows if r["verdict"] != "skip"]
    free = sum(1 for r in judged if r["verdict"] == "free")
    logo = sum(1 for r in judged if r["verdict"] == "logo")
    if judged and logo and not free:
        print(f"  WARN {path.name}: {logo} logo, 0 free -> no FP rate computable")
    if judged and free and not logo:
        print(f"  WARN {path.name}: {free} free, 0 logo -> no recall computable")
    print(f"  OK {path.name}: {len(rows)} rows, {len(judged)} judged "
          f"({logo} logo / {free} free / {len(rows)-len(judged)} skip), "
          f"{sum(len(r['boxes']) for r in rows)} boxes, keys unique")


def freeze_gate(path: Path) -> int:
    """Exit 0 only if this file is fit to be the one-shot final measurement."""
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    images = sorted((path.parent / "images").glob("*.jpg"))
    keys = [r["file"] for r in rows]
    judged = [r for r in rows if r["verdict"] != "skip"]
    v = Counter(r["verdict"] for r in rows)
    bucket = sizes(path, rows)
    fail: list[str] = []

    if len(keys) != len(set(keys)):
        fail.append(f"duplicate frame keys: {sorted({k for k in keys if keys.count(k) > 1})}")
    if len(set(keys)) != len(images):
        fail.append(f"{len(set(keys))} labelled keys vs {len(images)} frames on disk")
    if any(len(r["boxes"]) > 1 and r["verdict"] != "logo" for r in rows):
        fail.append("boxes on a non-logo verdict")
    if any(r["verdict"] == "logo" and not r["boxes"] for r in rows):
        fail.append("logo verdict with zero boxes")
    if v["free"] == 0:
        fail.append("free = 0: no false-positive rate is computable")
    if v["logo"] == 0:
        fail.append("logo = 0: no recall is computable")
    if v["skip"] > max(2, round(0.02 * len(images))):
        fail.append(f"skip = {v['skip']} of {len(images)}: "
                    f"a test set must be judged, not deferred")
    if not len(judged):
        fail.append("nothing judged")

    print(f"\nFREEZE GATE  {path}")
    print(f"  frames on disk   {len(images)}")
    print(f"  unique keys      {len(set(keys))}")
    print(f"  judged           {len(judged)}")
    print(f"  logo / free      {v['logo']} / {v['free']}")
    print(f"  skip             {v['skip']}")
    print(f"  boxes            {sum(len(r['boxes']) for r in rows)}")
    print(f"  box sizes        {dict(bucket) or 'none'}")
    if bucket["tiny"] == 0:
        # Not a failure. Report it as unrepresented rather than inventing cases
        # or quietly dropping the bucket from the writeup.
        print(f"  tiny <1%         0 - NOT REPRESENTED. Valid result: this video")
        print(f"                   has no tiny logos. Report tiny performance from")
        print(f"                   another labeled set; do not fabricate cases here.")
    if fail:
        print("\n  BLOCKED, do not freeze:")
        for f in fail:
            print(f"    - {f}")
        return 1
    print("\n  PASS - safe to freeze-commit. This file is the only")
    print("  artifact the final measurement is allowed to touch.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", metavar="LABELS_JSONL")
    a = ap.parse_args()

    if a.freeze:
        sys.exit(freeze_gate(Path(a.freeze)))

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "l.jsonl"
        s = Store(p)
        base = {"video": "v", "frame_index": 0, "boxes": []}
        s.put({**base, "file": "v_000000.jpg", "verdict": "skip"})
        s.put({**base, "file": "v_000200.jpg", "verdict": "free"})
        # re-judging the same frame must replace, never duplicate
        s.put({**base, "file": "v_000000.jpg", "verdict": "logo",
               "boxes": [{"x": 1, "y": 1, "w": 10, "h": 10, "src": "human"}]})
        check(p)
        for f in sorted(Path("benchmark").rglob("labels.jsonl")):
            check(f)
    print("store rewrite OK")
