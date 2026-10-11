# realvideo_v1 — rebuilt real-video set

Replaces the lost `benchmark/video_labelled` (gitignored, survived on no disk —
see the incident note in the last handover). Rebuilt from the 4 source videos
that survived in `~/Downloads`.

## Composition

| video md5 | source | res | frames | split |
|---|---|---|---|---|
| `88b0502222` | 5 Best Smartphones Under ₹20,000 | 640×360 | 80 | **val** |
| `df01d5a843` | Last Chance to Buy a Smartphone | 640×360 | 80 | **test** |
| `63fb2ac4d1` | Samsung's thinnest foldable phone | 720×1280 | 80 | train |
| `7864fa1c52` | Famous Tech Companies & Their Founders | 640×360 | 80 | train |

320 frames total. **val+test (160 frames, 2 videos) is the frozen holdout**;
see `splits.json`. The old `frozen_test` guarantee died with its data — this is
the fresh holdout, frozen in `splits.json` **before** any arm ran on the frames.

`extract/manifest.json` (written by `scripts/extract_video_frames.py`) is the
split authority; `splits.json` is the human-readable copy.

## Durability

- `MANIFEST.sha256` lists source-video hashes + an aggregate hash over all 320
  sampled frames. Committed to git.
- Source videos: copy the `source/` dir and the hash manifest to an external
  drive or a private HF dataset. The files also live in `~/Downloads` (uncommitted.
  Back them up before deleting anything.)
- Labels (`labels.jsonl` + the exported YOLO set) are committed once review is
  done.

## Ground truth flow (scripts/label_frames.py)

Frames land in `extract/images/{split}/<video_md5>_<frame_index>.jpg`, unlabelled.
One review pass decides per frame: LOGO (boxes) or NO LOGO (free) or SKIP.

    python3 scripts/label_frames.py \
        --frames benchmark/realvideo_v1/extract/images \
        --out benchmark/realvideo_v1/labels.jsonl \
        --propose weights/logo_detector/train/weights/best.pt \
        --port 8765
    # open http://127.0.0.1:8765, label; then:
    python3 scripts/label_frames.py --frames ... --out ... --export benchmark/realvideo_v1/export

The detector's proposals are review accelerators, never ground truth; confirmed
model-only frames land in `model_prelabelled.json` and are excluded when grading
that detector (circular). NO LOGO is a first-class verdict and is the only source
of FP/free-frame denominators.