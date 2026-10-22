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
  **KEY `h`**: the reviewer DESIGNATES the hero in the UI (click a box, then
  press `h`; a gold box + `H` tag marks it). The designation is recorded in the
  label row and exported to `heroes.json`. The designated hero decides top-1
  even when a second, larger box (a billboard) is present. **Do not skip `h`**
  — a forgotten hero silently shrinks the top-1 denominator with no error.
- **Top-1** means a detection matching the *designated hero box* — NOT the
  largest box by area (that definition is retired: in 1.4b a big background
  billboard could beat the small hero product mark). The grader reads
  `heroes.json`. A logo frame is perfectly valid with **zero heroes** (e.g.
  only billboard boxes): it counts as a logo frame, is reported separately in
  the composition as *logo without hero*, and is excluded from the top-1 pool.
- Frames with no readable logo are labelled **logo-free** (key `k` in the UI).
  The count matters: the holdout needs at least ~30 logo-free frames out of
  the 160, or FP/free-frame is a poor denominator. VERDICT SPLIT IS REPORTED
  TO THE LEAD EVERY TIME LABELING STOPS.
- **Composition is reported three ways** after labeling, per video and pooled:
  "N logo frames with hero, M logo frames without hero, K free." The grader
  prints this line for the pooled holdout and each video.
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

### The 1.7 decision pools val+test (159 labeled of 160 extracted)

Neither arm is fit on this data (OWLv2 zero-shot, YOLO already trained), and one
80-frame video is too small AND too correlated to carry a pass/fail. Decision is
on the pooled val+test with per-video numbers reported alongside. Selection
bias: thresholds are swept on the same frames, which flatters both arms about
equally — stated in the decision note, not hidden. Contamination: `df01d5a843`
was the OLD val video and `88b0502222` the OLD frozen_test, so both were seen in
earlier decisions; this holdout is fresh-SPLIT, not unseen. Phase 2.5 reserves
one video for confirmation when it gets there.

### 1.7 verdict (2026-10-11): FAIL — see `../results/realvideo_v1_yolo.grade.json`

OWLv2 zero-shot and the LogoDet-trained detector both lost on the real holdout.
Composition of the pooled labeled holdout: **24 logo frames with hero, 0 logo
frames without hero, 135 free** (159 labeled of 160 extracted — the one frame
without a label row is `df01d5a843_017480.jpg`, an all-black tail frame the
labeler deferred (SKIP records no verdict) and never judged; it is excluded
from every metric in this README and the harness prints the accounting on each
run). At its best threshold (conf 0.05) OWLv2 reached
top-1 recall **6/24 (0.25)** — below the pre-registered ≥0.3 bar, so no
threshold entered the FP-margin comparison and the decision fails outright
(`chosen_owlv2_conf: None`). The detector's best top-1 was **9/24 (0.375)** at
**239 FP per free frame**; even at conf 0.25 it still emits ~12 FP/free — a
domain-gap failure on real creator video, and the retired 0.886 bar now looks
optimistic. Logos are rare in real frames (15% of the holdout), so production
is dominated by rejecting free frames, where both arms are weakest, and the
training positives are starving.

Decision consequence: OWLv2 is dropped with no further rescoring; Phase 3.1
(cancelled, its verifier would sit on OWLv2 proposals); Phase 3 on hold until
Phase 1.8 (detector+OCR+retrieval+fusion vs OCR+retrieval-only on this same
holdout) decides between 3.2 and OCR+CLIP-primary.

### Accounting questions resolved (2026-10-12)

Three loose ends were chased down while preparing the 1.8 re-score; each had one
verifiable answer:

1. **160 extracted vs 159 labeled.** The frame without a label row is
   `df01d5a843_017480.jpg` — verified all-black (mean pixel 0.0, std 0.0), the
   video's tail frame. The labeler's SKIP records no verdict, so it was deferred
   at labeling time and never judged. It is excluded from every metric here; the
   1.8 harness prints the accounting as a WARNING on each run.
2. **42 SKIP on `7864fa1c52`.** Re-judged as logo-free in commit `0ad2fd2`
   after frame inspection (that video shows no on-screen logos). Current label
   counts: 44 free, 36 logo, 0 skip; the prior SKIP state is preserved in
   `labels.jsonl.bak_skip`.
3. **636 vs 626 tests collected.** The `wip/dedupe` branch adds exactly +10
   tests over `main` (14 vs 4 in `test_logo_batch_contract.py`); no other test
   file differs. 626 on `main` is the correct number.
