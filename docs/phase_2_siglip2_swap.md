# Phase 2 — SigLIP2 swap: implementation status + data

## Status (2026-10-11)
Swap implemented behind config, SHIPPED DEFAULT UNCHANGED (CLIP ViT-B-32
openai). The bank is re-embedded at pipeline start from
`benchmark/reference_logos_bank/reference` — there is no persisted index to
migrate, so "bank rebuild" for phase 2 is just launching with the new config.

## Deviation from the brief (accepted)
The lead specified `transformers.AutoModel` with `google/siglip2-base-patch16-224`
(a 224px checkpoint). The implementation uses open_clip's
`ViT-B-16-SigLIP2-256` pretrained `webli` instead — a different checkpoint AND
a different resolution (256 vs 224). Justification: it fits the encoder's
existing open_clip branch (`LogoRetrievalIndex._load`), so no second model
stack (transformers AutoModel + SigLIP2Processor) needs to be plumbed, and the
bank rebuild stays the same code path. Consequence to record for 2.5: numbers
below are for the 256px checkpoint; `google/siglip2-base-patch16-224` would
need its own embed-cost and similarity-scale measurement before being treated
as interchangeable.

## The knob
`config/config.yaml` → `layer1.logo_retrieval`:
- `model_name: "ViT-B-32"`, `pretrained: "openai"` (shipped default)
- override e.g. `model_name: "ViT-B-16-SigLIP2-256"`, `pretrained: "webli"`

`LogoRetrievalIndex.embedding_version` is now eager and embeds the encoder
identity (`logo_clip:open_clip:<model>:<pretrained>:rgb1`), so a CLIP-built
bank queried from a SigLIP2 process fails loudly (assert_embedding_version)
instead of returning corrupted nearest neighbours. Same 768-dim output, totally
different space.

## Version-guard coverage
- In-process indexes: `LogoRetrievalIndex.query` holds the guard
  (assert_embedding_version) and the pipeline query passes the live extractor
  tag. `ProductEmbeddingIndex.query` likewise asserts against the live
  DINOv3/DINOv2 tag.
- Persisted vector caches: `brand_memory.json` persists per-brand reference
  vectors ACROSS processes, which the in-process guards cannot see. It already
  stored an `embedding_model_version` provenance tag per entity but the cosine
  path never checked it. Fixed: `BrandMemoryBank.resolve_reference` now takes
  `embedding_version` and raises on any mismatch against a stored tag (legacy
  rows without a tag remain usable — fail-open only where there is no tag).
  The pipeline's indirect-mention call site passes the live extractor version.
  Note (a0dc08, tightened): an embedded row with NO `embedding_model_version`
  now also raises on any embedded lookup — a persisted vector that cannot be
  validated against the live space is never compared. Any runtime-written
  `brand_memory.json` from a deployed instance with untagged embedded rows
  will fail when queried; rebuild the bank under the active embedder.
- Nothing else persists vectors to disk (logo-retrieval and product indexes
  are rebuilt in-process at pipeline start; verified no `np.save/npz/pickle`
  of index matrices).

## Measured on the 543-crop / 16-brand bank (SigLIP2-B/16-256, webli)
- Embed cost on M1 CPU: 54 ms/crop, 29 s for the full bank — a config flip pays
  ~30 s at startup. Not a latency argument. Meaningful only as ms/crop × crops
  per frame: 2.5 must report the per-frame cost, not this bare rate.
- Within-brand top similarity: mean 0.953, min 0.683.
- Cross-brand top similarity: mean 0.775, max 0.907.
- Margin (within − cross): mean +0.18, min −0.15; 1% of crops have a negative
  margin (a wrong brand outranks the true one even at self-similarity).
- SCOPE: these are bank-vs-bank similarities on LogoDet reference crops —
  in-distribution, clean, no video noise. They are a useful NEGATIVE result
  (the 0.65 floor is meaningless) but must NOT be used to pick thresholds:
  video-crop noise (low-res crops, partial logos, motion blur) is precisely
  what is absent here, and a 1% negative-margin residue on clean data is a
  warning sign for video, not an all-clear.

## Implication for shipping (2.5 is still gated on 1.7)
The CLIP gates do NOT transfer, as the lead predicted:
- CLIP `min_similarity 0.65` is meaningless for SigLIP2 — genuine matches sit as
  low as 0.68 AND cross-brand noise reaches 0.91, so a 0.65 floor both rejects
  real hits and admits noise. SigLIP2 similarities cluster near 1.0.
- The margin idea still discriminates (mean 0.18) but its thresholds and the 1%
  negative-margin residue must be re-estimated on held-out real-video labels
  before the swap ships. Flip the config to SigLIP2 only after that retune.
- Currently the resolver's live `min_similarity 0.65 / min_margin 0.10` are
  tuned for CLIP; the swap would silently drop precision/recall otherwise.

## 2.5 acceptance + protocol (not run yet — gated on 1.7 labels)
- Acceptance requires a BEFORE/AFTER count of brand resolutions and false
  positives on real videos (SigLIP2 vs CLIP on the same jobs). CLIP stays the
  default until that table exists.
- 2.5 protocol (lead): TUNE thresholds on the TRAIN videos (once labeled),
  CONFIRM on ONE holdout video. Do NOT sweep all 160 holdout frames — that set
  is reserved for the OWLv2 1.7 decision.
- Report per-frame cost as ms/crop × crops per frame at 2.5, not the bare 54
  ms/crop.