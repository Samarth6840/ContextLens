# Phase 2 — SigLIP2 swap: implementation status + data

## Status (2026-10-11)
Swap implemented behind config, SHIPPED DEFAULT UNCHANGED (CLIP ViT-B-32
openai). The bank is re-embedded at pipeline start from
`benchmark/reference_logos_bank/reference` — there is no persisted index to
migrate, so "bank rebuild" for phase 2 is just launching with the new config.

## The knob
`config/config.yaml` → `layer1.logo_retrieval`:
- `model_name: "ViT-B-32"`, `pretrained: "openai"` (shipped default)
- override e.g. `model_name: "ViT-B-16-SigLIP2-256"`, `pretrained: "webli"`

`LogoRetrievalIndex.embedding_version` is now eager and embeds the encoder
identity (`logo_clip:open_clip:<model>:<pretrained>:rgb1`), so a CLIP-built
bank queried from a SigLIP2 process fails loudly (assert_embedding_version)
instead of returning corrupted nearest neighbours. Same 768-dim output, totally
different space.

## Measured on the 543-crop / 16-brand bank (SigLIP2-B/16-256, webli)
- Embed cost on M1 CPU: 54 ms/crop, 29 s for the full bank — a config flip pays
  ~30 s at startup. Not a latency argument against the swap.
- Within-brand top similarity: mean 0.953, min 0.683.
- Cross-brand top similarity: mean 0.775, max 0.907.
- Margin (within − cross): mean +0.18, min −0.15; 1% of crops have a negative
  margin (a wrong brand outranks the true one even at self-similarity).

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