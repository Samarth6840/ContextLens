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

One-command backup (run once a drive is mounted, or add `HF_TOKEN`):

    rsync -a benchmark/realvideo_v1/source/ \
        /Volumes/<YOUR_DRIVE>/contextlens/realvideo_v1/source/
    cp benchmark/realvideo_v1/MANIFEST.sha256 /Volumes/<YOUR_DRIVE>/contextlens/

    # private HF dataset (huggingface_hub is already installed):
    HF_TOKEN=hf_xxx python3 - <<'EOF'
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo("contextlens-realvideo-v1", private=True, repo_type="dataset")
    api.upload_folder(folder_path="benchmark/realvideo_v1/source",
                      repo_id="<user>/contextlens-realvideo-v1",
                      repo_type="dataset")
    api.upload_file(path_or_fileobj="benchmark/realvideo_v1/MANIFEST.sha256",
                    path_in_repo="MANIFEST.sha256",
                    repo_id="<user>/contextlens-realvideo-v1",
                    repo_type="dataset")
    EOF

## Ground truth flow (scripts/label_frames.py)

Frames land in `extract/images/{split}/<video_md5>_<frame_index>.jpg`, unlabelled.
One review pass decides per frame: LOGO (boxes) or NO LOGO (free) or SKIP.

### Label protocol (frozen — read before labeling)

- **Hero box** = the largest *clearly visible* brand logo or wordmark on a
  product, packaging, or the presenter's device. Billboards, background posters
  and UI overlays ARE labelled as logo boxes but are NOT hero candidates.
  The reviewer DESIGNATES the hero in the UI (click the box, then `h`); it is
  recorded in the label row and exported to `heroes.json`. The designated hero
  decides top-1 even when a second, larger box (a billboard) is present.
- **Top-1** means a detection matching the *designated hero box* — NOT the
  largest box by area (that definition is retired: in 1.4b a big background
  billboard could beat the small hero product mark). The grader reads
  `heroes.json`; a logo frame with no designated hero (e.g. only billboard
  boxes) stays in the composition count but is excluded from the top-1 pool.
- Frames with no readable logo are labelled **logo-free** (key `k` in the UI).
  The count matters: the holdout needs at least ~30 logo-free frames out of
  the 160, or FP/free-frame is a poor denominator. VERDICT SPLIT IS REPORTED
  TO THE LEAD EVERY TIME LABELING STOPS.
- SKIP (double-`s`) is reserved for genuinely unjudgeable frames (unreadable,
  truncated, ambiguous). A frame with no logo is NOT a skip.

### Blind holdout review (the 160 held-out frames — val+test)

Reviewer starts BLANK. No detector proposals are shown for the holdout: a box
already drawn on screen anchors the reviewer, and labels produced from the
detector's own boxes favor the detector arm even when edited (the circular-
frame exclusion only catches exact copies). Run the server per holdout video in
zero-circularity mode (no `--propose`, no `--use-cache`; the cached
`labels.proposals.json` is for the train videos only):

    python3 scripts/label_frames.py --frames benchmark/realvideo_v1/extract/images \
        --out benchmark/realvideo_v1/labels.jsonl \
        --only-video 88b0502222 --port 8765      # 5 Best Smartphones (old frozen_test)
    # label all 80, ctrl-C, then:
    python3 scripts/label_frames.py --frames benchmark/realvideo_v1/extract/images \
        --out benchmark/realvideo_v1/labels.jsonl \
        --only-video df01d5a843 --port 8765      # Last Chance (old val video)
    # label all 80, ctrl-C.

Exact-copy circularity cannot creep in: zero-circularity mode draws nothing and
the holdout frames are never pre-labelled. The train videos MAY be pre-labelled
(`--use-cache`) later for Phase 3.

Then export + grade pooled val+test:

    python3 scripts/label_frames.py --frames benchmark/realvideo_v1/extract/images \
        --out benchmark/realvideo_v1/labels.jsonl --export benchmark/realvideo_v1/export
    python3 scripts/owlv2_1p7_grade.py --frames benchmark/realvideo_v1/extract/images \
        --export benchmark/realvideo_v1/export

The export's detector proposals are never ground truth; confirmed model-only
frames land in `model_prelabelled.json` and are excluded when grading that
detector (circular). NO LOGO is a first-class verdict and is the only source
of FP/free-frame denominators.

### The 1.7 decision pools val+test (160 frames)

Neither arm is fit on this data (OWLv2 zero-shot, YOLO already trained), and one
80-frame video is too small AND too correlated to carry a pass/fail. Decision is
on the pooled val+test with per-video numbers reported alongside. Selection
bias: thresholds are swept on the same frames, which flatters both arms about
equally — stated in the decision note, not hidden. Contamination: `df01d5a843`
was the OLD val video and `88b0502222` the OLD frozen_test, so both were seen in
earlier decisions; this holdout is fresh-SPLIT, not unseen. Phase 2.5 reserves
one video for confirmation when it gets there.