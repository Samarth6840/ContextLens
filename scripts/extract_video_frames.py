"""Extract a frame pool from real ContextLens videos, split by video.

Split-by-video is the only honest way to evaluate a detector on video: adjacent
frames are near-duplicates, so a random frame split leaks the same scene into
train and test. Each source video is assigned wholesale to exactly one split.

We currently have only 2 distinct uploads (verified by md5 of file bytes; the
"Samsung thinnest foldable phone" clip was uploaded three times), so this
produces a 2-way holdout (train / test) rather than train / val / test. That is
enough to detect leakage but too little to tune on - see notes in the printout.

Frames are written unlabelled. They are a candidate pool, not a training set:
nothing here has reviewed logo ground truth yet.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent


def video_key(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()[:10]


def frozen_test_ids(root: Path) -> set[str]:
    """Every video id the frozen TEST set is built from.

    Read from its own PROVENANCE.json rather than hardcoded here, so the
    extractor cannot drift from a constant that build_detector_mix.py and
    train_mix.py each hold a private copy of. Returns an empty set when the
    frozen set does not exist yet (a fresh clone), which keeps the refusal
    honest instead of pretending to know the id.
    """
    prov = root / "benchmark/frozen_test/PROVENANCE.json"
    if not prov.is_file():
        return set()
    try:
        md5 = json.loads(prov.read_text()).get("video_md5")
    except json.JSONDecodeError:
        return set()
    return {md5} if isinstance(md5, str) else set()


def refuse_known_videos(videos: dict[str, Path], root: Path, out: Path) -> None:
    """Refuse anything that would move an already-placed video, by CONTENT.

    Three ways a re-extract silently rewrites the protocol, all of which
    happened or nearly happened:

      1. the frozen TEST video re-uploaded under a new name, or re-extracted.
         The frozen rule is that the detector is run on it exactly once, so it
         must never enter a pool that feeds checkpoint or threshold choice.
      2. a video already in the manifest, under any split. Re-extracting it
         re-runs split_order over a DIFFERENT set size, and _CYCLES[4] hands
         out ["train","train","val","test"] where _CYCLES[3] handed out
         ["train","val","test"]: one new upload reassigns every existing
         video's role, and the labels already reviewed against the old roles
         silently become wrong-role data.
      3. two different filenames with identical bytes, which is the same
         video counted twice.

    Checked on file content, not filename: a copy of the frozen TEST video
    saved under a different name has the same bytes and is still the frozen
    TEST video. Refuse rather than warn, and name the file that collided.
    """
    frozen = frozen_test_ids(root)
    known = {}
    man = out / "manifest.json"
    if man.is_file():
        try:
            for rec in json.loads(man.read_text()):
                if isinstance(rec, dict) and rec.get("video_md5"):
                    known[rec["video_md5"]] = rec.get("split") or rec.get("role")
        except json.JSONDecodeError:
            sys.exit(f"FATAL: {man} is not valid JSON; refusing to guess the "
                     f"protocol it records")

    problems = []
    for key, f in sorted(videos.items()):
        if key in frozen:
            problems.append(f"  {f.name}: {key} is the FROZEN TEST video")
        elif key in known:
            problems.append(f"  {f.name}: {key} is already in the manifest as "
                            f"{known[key]!r}; re-extracting reshuffles every "
                            f"video's split")
    if problems:
        print("\nREFUSING to extract:\n" + "\n".join(problems))
        print("\nThe split is a property of the protocol, not of the upload dir.")
        print("To add a genuinely new video, clear the role question first: see")
        print("benchmark/eval_video/manifest.json, then --out to a new root.")
        sys.exit(1)


def split_order(n_videos: int) -> list[str]:
    """Cycle of splits for n distinct videos, so each video lands in exactly one.

    3 distinct videos is the minimum for a real train/val/test holdout; below
    that a 2-way split is all the data supports, and anything more would put
    the same scene in two splits. Past 3, a 4th video joins TRAIN rather than
    VAL, because a val set of one video tunes on the same scene it reports.
    """
    if n_videos < 3:
        return ["train", "test"]
    return _CYCLES[min(n_videos, 5)]


_CYCLES = {3: ["train", "val", "test"],
           4: ["train", "train", "val", "test"],
           5: ["train", "train", "val", "test", "test"]}


def spread(n: int, budget: int) -> list[int]:
    """`budget` indices spread evenly over range(n), inclusive of both ends."""
    if budget >= n:
        return list(range(n))
    if budget <= 1:
        return [0]
    return sorted({round(i * (n - 1) / (budget - 1)) for i in range(budget)})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src",
                    default=os.environ.get("CONTEXTLENS_UPLOADS_DIR"),
                    help="directory of source .mp4 uploads "
                         "(default: $CONTEXTLENS_UPLOADS_DIR)")
    ap.add_argument("--out", default="benchmark/eval_video")
    ap.add_argument("--stride-train", type=int, default=8)
    ap.add_argument("--stride-test", type=int, default=40)
    ap.add_argument("--max-per-video", type=int, default=80,
                    help="cap frames per video so one upload cannot flood the pool")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        sys.exit(f"FATAL: {src} does not exist (uploads live in the OS temp dir)")

    by_key: dict[str, Path] = {}
    dupes: list[str] = []
    for f in sorted(src.glob("*.mp4")):
        k = video_key(f)
        if k in by_key:
            # Identical bytes under two names is one video, not two. Deduping
            # it silently would hide a real upload mistake (the same clip
            # re-saved, or a stray copy of the frozen TEST video), so say so.
            dupes.append(f"  {f.name}: identical content to {by_key[k].name}")
            continue
        by_key[k] = f
    print(f"{len(by_key)} distinct videos from {len(list(src.glob('*.mp4')))} files")
    if dupes:
        print("DUPLICATE CONTENT:\n" + "\n".join(dupes))
        print("Remove the redundant copies; each must be a genuinely distinct video.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Before any frame is written, not after: the leak that matters is the
    # frames landing in a split, and a warning after extraction is a report
    # nobody reads.
    refuse_known_videos(by_key, ROOT, out)
    manifest = []
    # Whole videos land in exactly one split. 3 distinct videos is the minimum
    # for a real train/val/test; below that a 2-way holdout is all the data
    # supports, and pretending otherwise leaks the same scene across splits.
    order = split_order(len(by_key))
    for i, (key, f) in enumerate(sorted(by_key.items())):
        split = order[i % len(order)]
        stride = args.stride_train if split == "train" else args.stride_test
        d = out / "images" / split
        d.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(f))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        # Evenly spread the budget over the whole video rather than taking the
        # first N: a cap on a prefix biases the pool to the opening scene.
        wanted = list(range(0, n, stride))
        if args.max_per_video:
            wanted = [wanted[i] for i in spread(len(wanted), args.max_per_video)]
        kept = 0
        for idx in wanted:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            cv2.imwrite(str(d / f"{key}_{idx:06d}.jpg"), frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            kept += 1
        cap.release()
        rec = {"video_md5": key, "source": f.name, "split": split,
               "width": w, "height": h, "source_frames": n, "fps": round(fps, 3),
               "stride": stride, "frames_written": kept}
        manifest.append(rec)
        print(f"  {key} -> {split:5s} {w}x{h} {n} frames, stride {stride}: {kept} written")

    # Merge with the existing manifest instead of overwriting. Writing only the
    # new videos drops every previously-recorded role, so refuse_known_videos
    # can no longer protect them on the next run and one upload silently
    # reassigns the split protocol.
    man_path = out / "manifest.json"
    existing: list = []
    if man_path.is_file():
        try:
            existing = json.loads(man_path.read_text())
        except json.JSONDecodeError:
            existing = []
    merged = {r["video_md5"]: r for r in existing
              if isinstance(r, dict) and r.get("video_md5")}
    for r in manifest:
        merged[r["video_md5"]] = r
    man_path.write_text(json.dumps(list(merged.values()), indent=2))
    tot = sum(r["frames_written"] for r in manifest)
    by_split: dict[str, int] = {}
    for r in manifest:
        by_split[r["split"]] = by_split.get(r["split"], 0) + 1
    print(f"\n{tot} frames -> {out}")
    print(f"videos per split: {by_split}")
    if len(by_key) < 3:
        print(f"NOTE: {len(by_key)} distinct video(s) — fewer than the 3 a real")
        print("      train/val/test holdout needs. More distinct uploads required.")
    print("NOTE: frames are UNLABELLED; nothing here has reviewed logo ground truth.")
    print("Frame names are <video_md5>_<source_frame_index>. The index is the ground")
    print("truth for 'which frame is this' - video_label_sheet.py prints it per tile.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
