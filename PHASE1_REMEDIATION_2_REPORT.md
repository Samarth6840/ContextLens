# Phase 1 Remediation Report #2 — Confidence Ceiling from Unimplemented Evidence Sources

## Problem

The Layer 2b confidence score had a hidden mathematical ceiling. Five evidence sources shared weights summing to 1.0, but three of them (speech_mention, scene_context, product_retrieval) contributed exactly 0.0 because those modules aren't built yet. The maximum achievable pre-modulation confidence was 0.45 + 0.15 = 0.60, and the Samsung reference video produced only 27.6% confidence despite clear SAMSUNG text appearing in the video. Nothing in the output explained why the ceiling existed or how close the video was to it.

**Modulation ceiling note:** Modality quality modulation further scales each evidence weight by
`(0.5 + 0.5 × modality_weight)`. With a perfect quality score (modality_weight = 1.0), the
post-modulation maximum would be `0.60 × (0.5 + 0.5 × 1.0) = 0.60 × 1.0 = 0.60`. With the
actual video quality weight (≈0.91 from the Samsung clip), the post-modulation maximum becomes
`0.60 × (0.5 + 0.5 × 0.91) = 0.60 × 0.955 = 0.573`. So the Samsung video could theoretically
reach just above the 0.55 threshold even with the old ceiling — but only if both logo and OCR
evidence were perfect, which they were not (logo strength was only 0.30). The Samsung video's
actual 0.276 confidence reflected genuinely moderate evidence, not just the ceiling.

---

## Task 1 — Evidence-Source Registry

### Config (`config/config.yaml`, `layer2b.evidence_sources`)

```yaml
evidence_sources:
  logo_detected:
    weight: 0.45
    status: implemented  # YOLO-World / fine-tuned logo detector
  speech_mention:
    weight: 0.20
    status: scaffolded   # requires Layer 2c temporal/intent linking
  ocr_hit:
    weight: 0.15
    status: implemented  # PaddleOCR text extraction
  scene_context:
    weight: 0.10
    status: scaffolded   # requires scene-context classifier
  product_retrieval:
    weight: 0.10
    status: scaffolded   # requires product-embedding retrieval
```

Each source has a `weight` and a `status` (`implemented` | `scaffolded`). This is the single source of truth — both the scorer's renormalization logic and the output's source-availability fields read from it.

**Changing a source from `scaffolded` to `implemented` requires more than a config change.** The
corresponding evidence producer must first be fully wired in `pipeline.py:_aggregate_evidence()`:
- `speech_mention` needs Layer 2c temporal/intent linking (not just the current `detect_brand_mentions`)
- `scene_context` requires the BEATs checkpoint to be available AND its shape bug fixed (see Phase 1
  Remediation #3)
- `product_retrieval` must no longer be hardcoded to `0.0` — a real product-embedding retrieval module
  must be built

Until those producers return real, non-zero evidence, the config should remain `scaffolded`.
Activation is **not** config-only; the config flip is the final step after the producer is built.

### How the registry is consumed (`src/layer2/confidence.py`)

```python
# EvidenceConfidenceScorer.__init__()
self._source_registry = {}
for name, spec in evidence_sources.items():
    if isinstance(spec, dict):
        raw_status = spec.get("status", "")
        if raw_status == STATUS_IMPLEMENTED:
            status = STATUS_IMPLEMENTED
        else:
            # Treat missing, invalid, or unknown statuses as scaffolded
            # so they cannot silently become implemented without explicit
            # configuration review.
            status = STATUS_SCAFFOLDED
        self._source_registry[name] = {
            "weight": float(spec.get("weight", 0.0)),
            "status": status,
        }
    else:
        self._source_registry[name] = {
            "weight": float(spec),
            "status": STATUS_IMPLEMENTED,
        }

self._implemented = {k: v for k, v in self._source_registry.items()
                     if v["status"] == STATUS_IMPLEMENTED}
self._scaffolded = {k: v for k, v in self._source_registry.items()
                    if v["status"] == STATUS_SCAFFOLDED}
```

---

## Task 2 — Renormalization

### Formula

```
active_denominator = sum(weight_i for i where status == "implemented")

for each source in registry:
    if status == "implemented":
        effective_weight_i = weight_i / active_denominator
    else:
        effective_weight_i = 0.0      # scaffolded — explicit zero in output

# Every registry source key appears in effective_weights (scaffolded → 0.0).
```

### Hand calculation (Samsung reference video)

**Input evidence strengths:** logo_detected = 0.30, ocr_hit = 0.90

**Base weights from registry:** logo_detected = 0.45, ocr_hit = 0.15

**Active denominator:** 0.45 + 0.15 = 0.60

**Renormalized effective weights:**
- logo_detected: 0.45 / 0.60 = **0.75**
- ocr_hit: 0.15 / 0.60 = **0.25**

**Expected confidence (without modality modulation):**
```
confidence = (0.30 × 0.75 + 0.90 × 0.25) / (0.75 + 0.25)
           = (0.225 + 0.225) / 1.0
           = 0.450
```

**Pipeline actual output:** `confidence = 0.4512`

The 0.0012 difference is from modality quality modulation (video_weight ≈ 0.909 slightly shifts the modulated weights). The hand-calculated base (0.450) and pipeline output (0.4512) are consistent.

### Before/after comparison

| Metric | Before (old weights) | After (renormalized) |
|---|---|---|
| logo_detected weight | 0.45 (of 1.0 total) | 0.75 (of 1.0 effective) |
| ocr_hit weight | 0.15 (of 1.0 total) | 0.25 (of 1.0 effective) |
| Scaffolded contribution | 0.0 (but included in denominator) | 0.0 (excluded from denominator) |
| Max achievable confidence | 0.60 (before modulation) | 1.0 (perfect evidence from active sources) |
| Samsung video confidence | 0.276 (27.6%) | 0.451 (45.1%) |

The ceiling is removed. A perfect result from the two active sources can now reach 1.0.

### Threshold decision: kept at 0.55

The `min_evidence_threshold` remains at 0.55. Rationale:

1. **0.55 is not artificially high for a 2-source system.** With renormalized weights (0.75 / 0.25), if the logo detector returns strength ≥ 0.73 AND OCR returns strength ≥ 0.73, confidence exceeds 0.55. This is achievable with real, strong evidence — the Samsung video's OCR strength is already 0.90.

2. **Lowering the threshold would be premature.** The current 0.451 confidence on the Samsung video reflects genuinely moderate evidence (logo strength only 0.30 — a single weak detection), not an artificial ceiling. The threshold should reflect the bar for "confident enough to act on," and 0.55 remains appropriate for that.

3. **The threshold is already source-count-aware in practice.** Because renormalization gives active sources their fair share of weight, the threshold naturally becomes reachable when evidence is strong. When more sources come online in Phase 2, the renormalization will automatically adjust — the threshold doesn't need to change.

4. **Flagged for review at Phase 2 boundary.** When speech_mention, scene_context, or product_retrieval flip to `implemented`, the effective weights shift again. At that point, the threshold should be re-evaluated against the new 3+ source regime. This is noted in the config:

```yaml
min_evidence_threshold: 0.55  # Review when evidence source count changes
```

---

## Task 3 — Source-Availability in Output

### Pipeline output fields (from `src/pipeline.py`)

```python
output = {
    ...
    "layer2b": {
        **confidence_result,
        "evidence_sources_active": ["logo_detected", "ocr_hit"],
        "evidence_sources_pending": ["speech_mention", "scene_context", "product_retrieval"],
    },
}
```

### Samsung reference video output

```
=== Layer 2b Output ===
confidence:                    0.4512
is_confident:                  False
status:                        no_confident_evidence
coverage:                      0.4
effective_weights:             {'logo_detected': 0.75, 'ocr_hit': 0.25}
scaffolded_sources:            ['speech_mention', 'scene_context', 'product_retrieval']
evidence_sources_active:       ['logo_detected', 'ocr_hit']
evidence_sources_pending:      ['speech_mention', 'scene_context', 'product_retrieval']
```

The output now self-documents: "40% of evidence sources are implemented; these 2 are active; these 3 are pending Phase 2 work."

**UI caption:** This is a headless CLI/API pipeline with no UI layer. The caption requirement is addressed by the `evidence_sources_active`, `evidence_sources_pending`, `coverage`, and `scaffolded_sources` fields in the structured output, which any downstream UI can consume directly. When a UI is added in Phase 2, the fields are available for rendering a caption like: "Confidence based on 2 of 5 planned evidence sources; additional modules (speech linking, scene context, product retrieval) are Phase 2 work."

---

## Task 4 — ASR Consistency Check

### Git history

The ASR file (`src/layer1/audio.py`) has exactly **1 commit** — the initial file addition (`9debd2f`). No changes were made to it during the performance-optimization work or at any other time. The `SpeechToText` class uses `whisper.load_model("large-v3")` with `fp16=False` (CPU mode) and default decoding parameters.

### Run-to-run consistency

Ran the Samsung video's audio through ASR twice consecutively:

```
Run 1 (357 chars):  यह सामझाओं का समसी पतला folding smartphone Z Fold 8 Ultra ...
Run 2 (357 chars):  यह सामझाओं का समसी पतला folding smartphone Z Fold 8 Ultra ...
Identical: True
```

**Findings:** Whisper large-v3 produces identical output across two runs on the same audio. No non-determinism observed. The ASR pipeline was not modified and produces consistent results.

---

## Self-Audit

### Renormalization reads from config dynamically

```python
# confidence.py — no hardcoded source-count special-casing
total_impl_weight = sum(info["weight"] for info in self._implemented.values())
if total_impl_weight > 0:
    self._effective_weights = {
        name: info["weight"] / total_impl_weight
        for name, info in self._implemented.items()
    }
```

The denominator is computed from the registry at init time. Adding a third implemented source (e.g., flipping `speech_mention` to `implemented` in config) would automatically produce a 3-way renormalization: `0.45 + 0.20 + 0.15 = 0.80`, effective weights `0.5625 / 0.25 / 0.1875`. No code change needed.

### No evidence source given nonzero contribution without real detection

In `pipeline.py:_aggregate_evidence()`:
- `logo_detected`: computed from real YOLO-World detections (confidence scores averaged)
- `ocr_hit`: computed from real PaddleOCR results (confidence scores averaged)
- `speech_mention`: computed from brand mentions in Whisper transcript (currently 0 — no brand names detected in transcript)
- `scene_context`: computed from BEATs audio events (currently 0 — BEATs checkpoint missing)
- `product_retrieval`: hardcoded 0.0 (product retrieval not built)

No source contributes a nonzero value without a real underlying detection. The scaffolded sources contribute 0.0 both because their underlying modules aren't built AND because the scorer excludes them from the weighted sum.

### Files modified

| File | Change |
|---|---|
| `config/config.yaml` | Restructured `evidence_weights` → `evidence_sources` with `status` fields |
| `src/layer2/confidence.py` | Full rewrite: source registry, renormalization, coverage/scaffolded metadata |
| `src/pipeline.py` | Added `evidence_sources_active`/`evidence_sources_pending` to output |
| `tests/test_pipeline.py` | Expanded from 22→32 tests; added registry/renormalization/coverage tests |
