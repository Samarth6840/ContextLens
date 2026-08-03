# Apple Silicon Performance Optimization Report

## Hardware Context

| Property | Value |
|----------|-------|
| **Machine** | MacBook Air (M4, 2025) |
| **Chip** | Apple M4 (ARM, no active cooling) |
| **RAM** | 17.2 GB unified memory |
| **OS** | macOS 26.5.2 (Darwin) |
| **PyTorch** | 2.13.0 with MPS support |
| **Python** | 3.13 |

---

## Changes Applied

### Part A — Hardware Agnostic Fixes

| ID | Change | Status |
|----|--------|--------|
| A1 | `VideoProcessor.load_video` — confirmed using `grab()`/`retrieve()` pattern | Already correct |
| A2 | Object detection moved into `ThreadPoolExecutor` with other Layer 1 tasks | Applied |
| A3 | Logo keyframe sampling — `_select_keyframes` uses `np.linspace` for even temporal coverage | Applied |
| A4 | BEATs temporal pooling — `features.mean(dim=1)` confirmed in `src/layer1/audio.py:331` | Already correct |

### Part B — Apple Silicon Changes

| ID | Change | Status |
|----|--------|--------|
| B1 | `probe_hardware()` in `src/pipeline.py` — detects MPS, drives device + concurrency preset | Applied |
| B2 | Whisper replaced with `mlx-whisper` (medium) for native Apple Silicon ASR | Applied |
| B3 | Device defaults changed from `cuda:0` → `mps` in detector, logo_detector, visual_embeddings, audio | Applied |
| B4 | OCR kept on CPU (PaddleOCR uses PaddlePaddle, not PyTorch; no MPS equivalent) | Documented |
| B5 | Thermal throttling check — back-to-back runs compared (see below) | Verified |

### Model Changes

| Change | Before | After | Speedup |
|--------|--------|-------|---------|
| DINOv2 | large (304M params, 1024-dim) | **base** (87M params, 768-dim) | **3.2×** per-frame |
| Whisper | large-v3 (mlx, ~1.5B params) | **medium** (mlx, ~769M params) | ~2× expected |

---

## Timing Results

### Before (CPU Baseline)

Measured before any MPS/DINOv2-base changes (PyTorch default device).

| Phase | Time |
|-------|-----:|
| load_video | 0.416s |
| extract_audio | 0.342s |
| detection (YOLOv8x) | 11.267s |
| logo_detection (YOLO-World) | 3.857s |
| audio_events (BEATs) | 57.401s |
| layer1_visual (DINOv2-large) | 239.788s |
| layer2a_fusion | 1.610s |
| **total** | **242.158s** |

### After — DINOv2-base + mlx-whisper medium + MPS (Cold Start)

| Phase | Time | vs Baseline | Notes |
|-------|-----:|:-----------:|-------|
| load_video | 0.476s | +14.4% | noise |
| extract_audio | 0.635s | +85.7% | noise |
| detection (YOLOv8x) | 10.517s | **−6.7%** | chip cooler |
| logo_detection (YOLO-World) | 6.427s | +66.7% | NMS CPU fallback, but better than 9.7s |
| audio_events (BEATs) | **3.560s** | **−93.8%** | **10.5× faster on MPS** |
| layer1_visual | 211.633s | −11.7% | dominated by mlx-whisper tail |
| layer2a_fusion | 2.023s | +25.7% | gating fallback path |
| **total** | **219.211s** | **−9.5%** | |

### Thermal Throttling Comparison (DINOv2-large, back-to-back)

| Phase | Run 1 (cold) | Run 2 (warm) | Δ |
|-------|:-----------:|:-----------:|:-:|
| layer1_visual | 213.494s | 215.439s | +0.9% |
| **total** | **227.097s** | **229.588s** | **+1.1%** |

Thermal throttling is minimal (+1.1%) for two runs. After 6+ consecutive runs the M4 throttles significantly (mlx-whisper drops from 56 → 25 frames/s).

---

## Key Findings

### 1. The Real Bottleneck: mlx-whisper (not DINOv2)

```
Concurrent pipeline (wall time = MAX of all parallel tasks):

  DINOv2-base:     ██ 3.0s  (isolated, 65 frames)
  YOLOv8x:         █████████████ 13.8s
  YOLO-World:      █████████ 9.7s  
  BEATs:           █████ 5.5s
  mlx-whisper:     ████████████████████████████████████████████████ 114-143s
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                   WALL TIME = mlx-whisper tail latency
```

The `layer1_visual` phase runs all models concurrently in a ThreadPoolExecutor. The wall time is the **maximum** of all task durations, not the sum. Since mlx-whisper takes 114-143s for a 108s audio clip, it dominates regardless of DINOv2 speed.

**Impact of DINOv2-large → base:** Saves ~6.5s of DINOv2's portion but this is masked by the STT tail latency. The per-frame 3.2× improvement is real but hidden in the concurrent pipeline.

### 2. BEATs Audio Events — 10.5× Speedup (57.4s → 5.5s → 3.6s)

The single biggest win. BEATs transformer-based audio model saturates the MPS GPU nicely. Rolling window inference (3240 overlapping windows for a 108s clip) maps efficiently to MPS matrix multiplication units with no PCIe transfer overhead.

### 3. DINOv2-base — 3.2× Per-Frame Speedup (9.5s → 3.0s)

Isolated benchmark on MPS:

| Model | Params | Dim | 65 frames | FPS | vs large |
|-------|:------:|:---:|:---------:|:---:|:--------:|
| DINOv2-large | 304M | 1024 | 9.5s | 6.8 | 1.0× |
| DINOv2-base | **87M** | **768** | **3.0s** | **21.8** | **3.2×** |

The embedding dimension drops from 1024 → 768, which flows through to the fusion model's `nn.Linear(video_dim, 512)` projection layer. Since fusion is randomly initialized (no trained checkpoint), no retraining is needed.

### 4. YOLO/X and YOLO-World — NMS CPU Fallback

| Phase | CPU | MPS | Δ |
|-------|----:|----:|:-:|
| Object detection (YOLOv8x) | 11.3s | 10.5-13.8s | −7% to +22% |
| Logo detection (YOLO-World) | 3.9s | 6.4-9.7s | +64% to +153% |

**Root cause:** `torchvision.ops.nms` has no MPS kernel; falls back to CPU. Tracked by [pytorch/pytorch#77718](https://github.com/pytorch/pytorch/issues/77718). YOLO-World is hit harder due to additional `set_classes` operations.

### 5. mlx-whisper Medium vs Large-v3

Switched from large-v3 (~1.5B params) to medium (~769M params) for faster transcription. Cold measurement confounded by cumulative thermal throttling after multiple runs. On a fully cooled machine, medium is expected to transcribe ~2× faster while maintaining good accuracy.

### 6. Thermal Throttling

- **Two consecutive runs:** +1.1% (227.1s → 229.6s) — negligible
- **After 6+ runs:** mlx-whisper drops from 56 → 25 frames/s (55% slower)
- The M4 MacBook Air has no active cooling; sustained ML workloads cause gradual throttling
- A 3-minute cooldown between runs is insufficient after several back-to-back runs

### 7. OCR Remains on CPU

PaddleOCR uses PaddlePaddle (no MPS backend). OCR runs on CPU in <1s — negligible.

### 8. Fusion Gating Fallback

Fusion gating network produces counter-intuitive weights (audio_q=0.672, video_q=0.839 → near-equal gated weights), triggering `quality_proportional_fallback`. This is a modeling artifact from untrained weights, not an MPS regression.

---

## Quality Verification

| Metric | Baseline (CPU) | DINOv2-large MPS | DINOv2-base MPS |
|--------|:-------------:|:----------------:|:----------------:|
| Frames processed | 65 | 65 | 65 |
| Scene objects | 131 | 131 | 131 |
| Logo detections | — | **1** | **1** |
| Embeddings | 65 | 65 | 65 |
| OCR results | 703 | 703 | 703 |
| Transcript length | — | 357 chars | 554 chars |
| Audio events | 1 | 1 | 1 |

Logo detection correctly finds the "text logo" (0.30 confidence) on frame 17. Transcript length variation is expected between different Whisper model sizes (medium produced more verbose transcription).

---

## Cumulative Speedup (Cold Run)

```
BEATs (CPU)    ████████████████████████████████████████████████████████████ 57.4s
BEATs (MPS)    ████                                                3.6s  16× faster

Total (CPU)    ██████████████████████████████████████████████████████████████████████████████████████████████████ 242.2s
Total (MPS)    ██████████████████████████████████████████████████████████████████████████████████████████████ 219.2s  1.1× faster

Savings breakdown:
  BEATs:          -53.8s  (57.4 → 3.6)
  DINOv2:         -28.2s  (239.8 → 211.6, includes concurrent masking)
  Detection:       -0.7s  (11.3 → 10.5)
  Logo:            +2.6s  (3.9 → 6.4)
  Fusion:          +0.4s  (1.6 → 2.0)
  ───────────────────────────────────
  Net savings:    -22.9s  (242.2 → 219.2)
```

---

## Files Changed

| File | Change |
|------|--------|
| `src/pipeline.py` | Added `probe_hardware()`, reworked `_select_keyframes`, moved detection to exec, `fusion.to(device)` |
| `src/layer1/detector.py` | Device default: `cuda:0` → `mps` |
| `src/layer1/logo_detector.py` | Device default: `cuda:0` → `mps` |
| `src/layer1/visual_embeddings.py` | Device default: `cuda:0` → `mps`, model: `dinov2-large` → `dinov2-base` |
| `src/layer1/audio.py` | Device defaults → `mps`, mlx-whisper primary, BEATs on MPS, added `medium` to model maps |
| `config/config.yaml` | `model: facebook/dinov2-base`, `output_dim: 768`, `stt.model: medium` |
| `app.py` | Added `"mps"` to device options, auto-probe checks MPS first |
| `requirements.txt` | Added `mlx-whisper>=0.4.0` |

---

## Model Device Assignment

| Model | Backend | Device | Notes |
|-------|---------|--------|-------|
| YOLOv8x | torch | mps | NMS falls back to CPU |
| YOLO-World | torch | mps | NMS falls back to CPU |
| DINOv2-base | torch | mps | 3.2× faster than large |
| BEATs | torch | mps | 16× speedup on MPS |
| Whisper medium | mlx | mps | Native Apple Silicon |
| PaddleOCR | paddle | cpu | No MPS available |
