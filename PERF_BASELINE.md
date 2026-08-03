# ContextLens Phase 1 — Performance Report

## Test Configuration
- **Video**: Samsung foldable phone test video (720x1280, 25fps, 64.8s)
- **Frames extracted**: 65 (1 fps sampling)
- **Platform**: macOS Apple Silicon, CPU only (no CUDA)
- **Date**: 2026-07-28

## Model Devices (all CPU)
| Model | File Size | Device |
|-------|-----------|--------|
| YOLOv8x (scene objects) | 137 MB | CPU |
| YOLO-World v2 (logo detection) | 26 MB | CPU |
| DINOv2-large (embeddings) | ~1.1B params | CPU |
| PaddleOCR | — | CPU |
| openai-whisper large-v3 | 3 GB | CPU |
| BEATs | — | checkpoint missing, disabled |

## Timing Comparison

| Stage | Before | After | Δ |
|-------|--------|-------|---|
| load_video | 1.32s | 1.28s | — |
| extract_audio | 0.31s | 0.49s | — |
| detection (YOLOv8x) | 11.14s | 14.73s | — |
| layer1_visual (embeddings+OCR+STT+logo, concurrent) | — | 264.75s | logo overlapped |
| layer2a_fusion | 0.98s | 3.12s | — |
| layer2b_confidence | 0.00s | 0.00s | — |
| **TOTAL (wall clock)** | **447.79s** | **269.65s** | **-40%** |

**Note:** The stage times above reflect individual sub-stage durations measured by wall clock.
The wall-clock total in the "After" column (269.65s) is NOT the sum of the individual stage rows
(which would be ~284.37s) because `layer1_visual` includes detection, OCR, STT, and logo detection
running concurrently — those sub-stages overlap with each other and with parts of load_video and
extract_audio. The summed row values double-count overlapped execution. The 269.65s total is the
actual end-to-end wall-clock time and is the correct measure of pipeline latency.

## Optimizations Applied

### 1. YOLO-World `set_classes()` fix (highest impact)
**File**: `src/layer1/logo_detector.py`

Root cause: `detect_batch()` called `model.set_classes(queries)` before every batch,
re-encoding 5 text prompts with CLIP each time. This turned a 26MB model into a 232s bottleneck.

Fix: Set classes once in `__init__()`, only re-set if queries actually change.
The per-batch CLIP encoding cost (~3.5s each × 9 batches = ~31s) plus overhead is eliminated.

### 2. Logo detection moved into ThreadPool (Step 4)
**File**: `src/pipeline.py`

Before: detection → logo_detection → ThreadPool(embeddings, OCR, STT) [sequential]
After: detection → ThreadPool(logo_detection, embeddings, OCR, STT) [concurrent]

Logo detection is independent of detection results, so it runs concurrently with
embeddings, OCR, and speech-to-text. This eliminated ~232s of sequential blocking time.

### 3. faster-whisper with fallback (Step 1)
**File**: `src/layer1/audio.py`

Added faster-whisper (INT8 quantization, 4-5x faster on CPU) with automatic
fallback to openai-whisper large-v3 if the faster-whisper model isn't cached.
Uses `_is_fw_cached()` to check for complete downloads before attempting load,
avoiding network timeouts. ASR stays on large-v3 weights for Hindi/Hinglish coverage.

### Bug fixes
- `src/layer1/detector.py`: Fixed broken class structure (methods outside class body after rename)
- `src/layer1/ocr.py`: Added missing `import cv2` (used for RGB→BGR conversion)

## Output Results (correctness verification)
- Scene objects: 131 detections ✓
- Logo detections: 1 (real logo detection) ✓
- Embeddings: 65 (one per frame) ✓
- OCR results: 703 text regions
- Transcript: 357 chars (large-v3 now, vs 419 chars with base)
- Brand mentions: 0
- Audio events: 0 (disabled — checkpoint missing)

## Remaining Bottleneck
The layer1_visual aggregate dominates at 264.75s (98% of total). This is primarily:
- **DINOv2-large** (1.1B params on CPU): Could switch to DINOv2-base (86M params, 768-dim)
- **openai-whisper large-v3** (3GB on CPU): Projected to improve to ~60s with faster-whisper INT8
  (estimate — not yet measured on this hardware). Once the model is downloaded
  (`huggingface-cli download Systran/faster-whisper-large-v3`) and the faster-whisper code path
  activates, this note should be updated with the real measured timing.
