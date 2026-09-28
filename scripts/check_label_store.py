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
    python3 scripts/check_label_store.py --pool   benchmark/eval_video/labels.jsonl
    python3 scripts/check_label_store.py --freeze benchmark/frozen_test/labels.jsonl

Two gates, because a working pool and a frozen test set are different claims:

  --pool    the labels are sound: unique keys, verdicts valid, boxes on logo
            rows only, every box human-drawn, every labelled frame present on
            disk. Unreviewed frames are expected and reported, not failed.

  --freeze  all of that, PLUS the set is fully judged. A frozen set is
            measured once and cannot grow, so an unjudged frame in it is a test
            case that will never run. That rule is wrong for a pool, which is
            reviewed incrementally and legitimately carries skips.
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


VALID_VERDICTS = {"free", "logo", "skip"}


def frame_index(path: Path) -> dict[str, Path]:
    """Every frame under a label store's images/ dir, keyed by bare filename.

    rglob, not glob. The extract pool keeps one subdirectory per video
    (images/a, images/b, images/c), so a flat glob("*.jpg") counts ZERO
    frames for it and every check built on that count fails for a reason that
    has nothing to do with the labels. The frozen test set is flat, which is
    exactly why the bug survived until a nested pool met the gate.
    """
    return {p.name: p for p in (path.parent / "images").rglob("*.jpg")
            if p.is_file()}


def sizes(frames: dict[str, Path], rows: list[dict]) -> Counter:
    """Bucket every box by share of its own frame. Needs the real frame, since
    a crop and a full frame make the same box a different size class."""
    out: Counter = Counter()
    for r in rows:
        img = frames.get(r["file"])
        if img is None:
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


def structural(path: Path, rows: list[dict], frames: dict[str, Path]) -> tuple[list[str], Counter]:
    """Checks that must hold for ANY label set, frozen or working.

    Returns (failures, verdict_counts). Every gate runs these; they describe
    the file disagreeing with itself or with the frames on disk, which is
    never legitimate.
    """
    keys = [r["file"] for r in rows]
    v = Counter(r["verdict"] for r in rows)
    fail: list[str] = []

    dupes = sorted({k for k in keys if keys.count(k) > 1})
    if dupes:
        fail.append(f"duplicate frame keys: {dupes}")

    orphans = sorted(set(keys) - set(frames))
    if orphans:
        # A label for a frame that is not there grades nothing, and the frame
        # it was meant for is silently unlabelled.
        fail.append(f"{len(orphans)} labels with no frame on disk: {orphans[:5]}"
                    + (" ..." if len(orphans) > 5 else ""))

    unlabelled = sorted(set(frames) - set(keys))
    if unlabelled:
        # Not a failure for a working pool - some frames are simply not
        # reviewed yet - but never silent, because an unjudged frame in a
        # FROZEN set is a missing test case.
        pass

    bad_verdict = sorted({r["verdict"] for r in rows} - VALID_VERDICTS)
    if bad_verdict:
        fail.append(f"unknown verdict values: {bad_verdict}")

    empty_logo = [r["file"] for r in rows
                  if r["verdict"] == "logo" and not r["boxes"]]
    if empty_logo:
        fail.append(f"logo verdict with zero boxes: {empty_logo[:5]}"
                    f"{' ...' if len(empty_logo) > 5 else ''}")

    boxes_on_free = [r["file"] for r in rows
                     if r["verdict"] != "logo" and r["boxes"]]
    if boxes_on_free:
        fail.append(f"boxes on a non-logo verdict: {boxes_on_free[:5]}"
                    f"{' ...' if len(boxes_on_free) > 5 else ''}")

    degenerate = [r["file"] for r in rows
                  for b in r["boxes"] if b.get("w", 0) < 1 or b.get("h", 0) < 1]
    if degenerate:
        fail.append(f"boxes too small to be a target: {sorted(set(degenerate))[:5]}")

    unprovenanced = [r["file"] for r in rows
                     for b in r["boxes"] if b.get("src") != "human"]
    if unprovenanced:
        # A box that is not human-drawn is model output. The whole point of
        # this loop is that the detector is graded on human ground truth, so a
        # model box in here makes every number downstream self-graded.
        fail.append(f"boxes not marked src=human: {sorted(set(unprovenanced))[:5]}"
                    f"{' ...' if len(unprovenanced) > 5 else ''}")

    return fail, v


def pool_gate(path: Path) -> int:
    """Exit 0 if this file is sound working-pool data: internally consistent
    and consistent with the frames on disk.

    Deliberately does NOT apply the judged-percentage rule. A working pool is
    reviewed incrementally and legitimately carries skips; the reviewer comes
    back to them. That rule exists for the FROZEN set, which is measured once
    and cannot grow, so an unjudged frame there is a test case that will never
    exist. Holding a working pool to it would mean deleting unreviewed frames
    to make a number look right.
    """
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    frames = frame_index(path)
    fail, v = structural(path, rows, frames)
    bucket = sizes(frames, rows)
    judged = [r for r in rows if r["verdict"] != "skip"]
    unlabelled = sorted(set(frames) - {r["file"] for r in rows})

    print(f"\nPOOL GATE  {path}")
    print(f"  frames on disk   {len(frames)}")
    print(f"  labelled rows    {len(rows)}   judged {len(judged)}"
          f"   skipped {v['skip']}")
    print(f"  logo / free      {v['logo']} / {v['free']}")
    print(f"  boxes            {sum(len(r['boxes']) for r in rows)}"
          f"   box sizes {dict(bucket) or 'none'}")
    if unlabelled:
        print(f"  not yet labelled {len(unlabelled)} frames on disk. Expected for a "
              f"working pool: they carry no verdict, so no build reads them.")
    if v["logo"] == 0 or v["free"] == 0:
        print(f"  NOTE: {v['logo']} logo / {v['free']} free. Both classes are "
              f"needed to compute recall AND a false-positive rate. A pool with "
              f"one class is fine as a negative or positive source, not as a "
              f"graded set.")

    if fail:
        print("\n  BLOCKED:")
        for f in fail:
            print(f"    - {f}")
        return 1
    print("\n  PASS - sound working-pool labels. This is NOT the frozen test set;")
    print("  see --freeze for the one-shot gate.")
    return 0


def freeze_gate(path: Path) -> int:
    """Exit 0 only if this file is fit to be the one-shot final measurement."""
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    frames = frame_index(path)
    keys = [r["file"] for r in rows]
    judged = [r for r in rows if r["verdict"] != "skip"]
    v = Counter(r["verdict"] for r in rows)
    bucket = sizes(frames, rows)
    # Structural checks, plus the judged-percentage rule that makes a frozen
    # set frozen: it is measured once, so anything unjudged is a test case that
    # will never be run.
    fail, _ = structural(path, rows, frames)

    if len(set(keys)) != len(frames):
        fail.append(f"{len(set(keys))} labelled keys vs {len(frames)} frames on disk")
    if v["free"] == 0:
        fail.append("free = 0: no false-positive rate is computable")
    if v["logo"] == 0:
        fail.append("logo = 0: no recall is computable")
    if v["skip"] > max(2, round(0.02 * len(frames))):
        fail.append(f"skip = {v['skip']} of {len(frames)}: "
                    f"a test set must be judged, not deferred")
    if not len(judged):
        fail.append("nothing judged")

    print(f"\nFREEZE GATE  {path}")
    print(f"  frames on disk   {len(frames)}")
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
    ap.add_argument("--freeze", metavar="LABELS_JSONL",
                    help="strict one-shot gate for the FROZEN test set: also "
                         "requires the set to be fully judged")
    ap.add_argument("--pool", metavar="LABELS_JSONL",
                    help="consistency gate for a working pool: sound labels, "
                         "in-progress review allowed")
    a = ap.parse_args()

    if a.freeze and a.pool:
        sys.exit("FATAL: --freeze and --pool are different gates; pick one")
    if a.freeze:
        sys.exit(freeze_gate(Path(a.freeze)))
    if a.pool:
        sys.exit(pool_gate(Path(a.pool)))

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
