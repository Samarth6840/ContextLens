# ContextLens
## Context-Aware Multimodal Brand Collaboration Recommendation System
### Capstone Project — Review II Report (Midterm Review)

Submitted by
Yashvi Patodi  [SAP ID]  [Roll No.]
Niyati Bansal  [SAP ID]  [Roll No.]
Samarth Joshi  [SAP ID]  [Roll No.]

Under the guidance of Dr. Vikas Khare

September 2026

> Note: the codebase uses the working codename **ADSCENE** (`server.py`, `config/config.yaml`). "ContextLens" is the project name used throughout this report. Sections 3 and 4 describe the system as actually implemented in this repository; every claim maps to a file.

---

# 1. Introduction and Problem Statement

## 1.1 Background

Content creators continuously show, wear and talk about branded products — a sneaker in a fit video, a flagship phone being reviewed, a brand name spoken while a laptop sits on screen. Deciding which brands a creator's content actually evidences, and which additional brands it would be logical to approach, is currently a manual process: someone watches the video, notes brands, and guesses at outreach targets. Existing video intelligence tools aimed at creators concentrate on editing, captioning or basic object tagging. Very few reason about the audio and video tracks **together**, and almost none combine detection, brand resolution, creator context and a brand relationship graph into a single pipeline.

When automated brand detection is attempted, it is usually unimodal: either a logo detector over frames, or a keyword scan over a transcript. Each alone is wrong in different ways. A logo is visible but never named; a brand is named but never shown; the audio is noisy, or the video is blurry, or the two are out of sync. A system that treats every modality as equally trustworthy fails exactly where real creator footage is shot — on a phone, indoors and outdoors, with whatever background noise and lighting happen to be present.

## 1.2 Problem Statement

We are designing and building **ContextLens**, a three-layer multimodal system that understands a creator's video and recommends brand collaborations from it. The system combines a perception layer (Layer 1) that detects objects, logos, speech and audio events; an intelligence layer (Layer 2) that estimates the *quality* of each signal, fuses the modalities through a learned, quality-aware gate, resolves brands from mixed evidence, and builds a cross-video memory of what a creator shows and says; and a recommendation layer (Layer 3) that turns this into a ranked, explainable list of DIRECT and SUGGESTED brand collaborations.

The specific research question the project tests is this: **does letting a fusion gate look at real, measured signal quality (audio noise, video blur, exposure, detection stability) rather than at learned features alone materially improve downstream confidence and recommendation quality when one modality is degraded?** This is the Layer 2a ablation (quality-gated dynamic weighting vs. fixed heuristic vs. equal weighting) that evaluates the project's central claim: that a system which knows *how trustworthy* each modality is degrades more gracefully than one that assumes equal trust.

## 1.3 Purpose and Need

The same robustness argument is what makes the project worth building at all. A creator's footage is rarely studio quality. An outreach recommendation built from a misread logo, or from speech the system could not hear because the audio was noisy, would be a fabricated claim sent to a real company. The design therefore treats signal-quality estimation and fail-closed behavior as first-class requirements: a brand name on the dashboard must always be backed by real, inspected evidence, and never be a guess. Where the pipeline cannot prove a brand, it must say so explicitly (unresolved / open-set candidate) rather than fill the gap.

## 1.4 Objectives

- Review existing multimodal fusion, quality-aware fusion, brand/logo detection and cold-start recommendation literature far enough to identify what current systems do well and where their fusion gates fall short (Section 2).
- Design and implement a three-layer system: Layer 1 multimodal perception; Layer 2 brand & context intelligence built around a modality-quality-aware fusion gate; Layer 3 knowledge-graph and affinity-driven recommendation (Section 3).
- Build the Layer 2a fusion gate in comparable versions of the same size — learned gating over measured signal quality, a fixed heuristic weighting, and equal weighting — so any improvement can be attributed to the weighting signal rather than to extra model capacity.
- Build brand resolution to be **fail-closed**: a brand is only asserted when independently corroborated by at least one of OCR text, CLIP logo retrieval against a reference bank, a corroborated detector class label, or a measured product match; everything else is left unresolved or escalated to the open-set module.
- Build the downstream recommendation layer and the product platform on top of the verified pipeline: ranked, explainable collaborations, draft outreach email generation with a human-in-the-loop approval gate, and a deployment-ready web application (API, dashboard, SQLite persistence, auth).
- Validate each stage against real data before committing to full-scale runs, and verify correctness with an automated test suite.

## 1.5 Scope

The project covers scene-level *brand and context understanding* of short-to-long creator video with audio and video used together, and brand-collaboration recommendation from that understanding. Scene classification itself is one *input signal* to the pipeline (the scene-context evidence source), not the project's end deliverable. The project does not automatically send email in the current phase: outreach is generated as drafts that a creator must review, and automatic dispatch is explicitly future work. Subtitles are not a third input in the current phase.

# 2. Literature Review and Market Survey

## 2.1 Academic Literature

Multimodal understanding of audio and video together is an active area. Self-supervised joint audio-visual representations such as **CAV-MAE** (Gong et al., 2023) share a transformer block across both modalities and learn general audio-visual features from large event datasets; such models motivate the two-stream-encoder design we use, though they target large event categories rather than the fine-grained brand semantics we need. **Cross-modal attention** (letting each modality attend to the other before a decision) has become the standard way to combine them, including in audio-visual scene classification (DCASE 2021 Task 1B; Xia, Yin & Dong, 2026, who reach 89.3% on the TAU Urban Audio-Visual Scenes 2021 benchmark with a learned gate on top of cross-modal attention).

That combination is the closest academic match to our Layer 2a design, and it exposes the exact gap we target. The learned gate in these systems is computed purely from the *attended features themselves*; the paper argues the gate should help when one modality is noisy or corrupted, but the robustness claim is never tested under added noise, blur or sync shift. Quality-aware and confidence-aware fusion is well studied in other fields — emotion recognition and speech recognition, e.g. QAAF-style work — but there the reliability estimate is itself *learned* (and usually over fine-tuned encoders), so it cannot be independently inspected or checked, and we found no version applied to brand/context understanding for creator-collaboration recommendation.

For the perception stack specifically, one-model-per-task is our choice over joint end-to-end learning: zero-shot open-vocabulary detection (YOLO-World), OCR for wordmarks (PaddleOCR), ASR for spoken brand mentions (Whisper-family), audio event detection (BEATs), frozen visual embeddings (DINOv3), and CLIP retrieval against a per-brand reference bank for icon-only marks. Each is best-in-class for its cost/accuracy trade-off on this task, and the layers communicate through typed, timestamp-aligned signals.

On the recommendation side, brand-collaboration matching is a genuine cold-start problem: new creators, sparse interaction history, and a large unseen brand space. Knowledge-graph adjacency (brand → category → adjacent brands) plus a collaborative signal (LightGCN-style affinity over a creator–brand graph) is the standard recipe we instantiate, with explainability as a hard requirement (every recommendation must state the evidence and graph edges that drove it).

## 2.2 Comparison of Related Work

| **Work** | **Task** | **How audio & video are combined** | **What it doesn't do** |
| --- | --- | --- | --- |
| Gong et al., CAV-MAE (2023) | General audio-visual representation learning | Joint transformer block, shared weights | Large event classes, not brand/context semantics |
| DCASE 2021 Task 1B baseline | Audio-visual scene classification | Pretrained visual features + spectrogram audio CNN, simple fusion | No gate; no test under noisy/shifted input |
| Xia, Yin & Dong (2026) | Audio-visual scene classification (TAU) | Cross-modal attention with a learned gate | Gate reads features only, not measured signal quality; robustness argued, never tested |
| QAAF-style quality-aware fusion (other fields) | Emotion / speech recognition | Learned reliability estimate reweights modalities | 'Reliability' is learned, not measured — cannot be inspected |
| Contextual video advertising (WACV-style systems) | Ad–content matching | Several separate expert models bolted together | No single shared representation to measure/improve |
| **ContextLens (this project)** | **Creator-brand collaboration recommendation from video** | **Frozen DINOv3 + BEATs feed a quality-aware fusion gate over *measured* signal quality, evidence-decomposed confidence, then KG + affinity recommendation** | **Full robustness ablation + training calibration still pending (Section 4.3)** |

*Table 2.1: Comparison of prior fusion and recommendation work and this project*

## 2.3 Market Survey

The closest product category is *contextual video advertising*, which matches ads to a video's content rather than to a viewer's profile. Published systems of this kind segment video, pull per-modality signals with several specialist models, and match against a brand-category database; they work but are collections of expert models with no one shared representation whose accuracy or robustness can be measured and improved as a whole. Our product takes the opposite stance: **one shared scene-understanding pipeline, built and tested as a whole, with recommendation and outreach built directly on top** rather than beside it.

For creators specifically, monetization tooling today means sponsorship marketplaces and some content tagging. Almost none actually *watch and listen* to a creator's video to derive which brands are evidenced and which adjacent brands are worth approaching. That gap — a content-grounded, explainable collaboration recommender for creators — is the market the platform half targets.

## 2.4 Research Gap

Three gaps drive this project. First, published fusion gates (for this and related tasks) look at *content*, not *measured signal quality*, and the learned "reliability" estimates in other fields cannot be inspected. Second, the robustness argument used to justify a gate in the first place is not tested under the very degradations it claims to fix (noise, blur, sync drift). Third, creator-brand collaboration recommendation from real video has no published system that reasons about audio and video together and exposes the evidence behind each recommendation. ContextLens addresses all three, and its Layer 2a ablation is deliberately built so the improvement (or absence of it) is attributable to the quality signal rather than to gate capacity.

# 3. Proposed System Design

## 3.1 Overall Architecture

The system is a three-layer pipeline. Layer 1 perceives; Layer 2 builds brand and context intelligence on top; Layer 3 recommends. All interfaces are typed, timestamp-aligned signals; every layer can be substituted without re-plumbing the others.

```
                    VIDEO + AUDIO INPUT (creator upload)
                    ─────────────────────────────────────
 LAYER 1 — MULTIMODAL PERCEPTION (one model per task)
   frames @1fps ─► YOLOv8x scene objects + YOLO-World open-vocab products
                ─► YOLO-World logo detection (catalog "<Brand> logo" queries)
                ─► PaddleOCR (Devanagari + Latin) on detections & crop supersets
                ─► CLIP logo retrieval vs per-brand reference bank (icon-only marks)
                ─► DINOv3 visual embeddings (frozen) → product-index NN
   audio       ─► ASR (mlx-whisper / faster-whisper / whisper fallback)
                ─► BEATs audio event detection
                ─► Audio/Video quality estimators (SNR, VAD, blur, exposure)
                    ▼
 LAYER 2 — BRAND & CONTEXT INTELLIGENCE
   2a Modality-quality-aware fusion        — learned gate over *measured* quality
   2b Evidence-based decomposed confidence — per-source weighted, auditable
   2c Temporal brand memory / resolution   — crop-OCR, retrieval, class corroboration,
                                              temporal smoothing, cross-video memory
      Tiered product→brand resolution      — aliases ▷ Wikidata ▷ Qwen3-VL ▷ memory
   2d Creator profiling + niche suppression
                    ▼
 LAYER 3 — RECOMMENDATION ENGINE
   knowledge graph (brand→category→adjacent) + LightGCN affinity (trained)
        ─► ranked DIRECT + SUGGESTED collaborations, each with reasons[]
                    ▼
       SERVER (Flask) — dashboard JSON, annotated scene thumbnails,
       ranked opportunities, open-set evidence cards,
       outward-generated draft emails (human-in-the-loop, fail-closed)
```

*Figure 3.1: ContextLens three-layer architecture*

## 3.2 Layer 1 — Multimodal Perception

One model per task, each kept frozen (none retrained in Phase 1):

| **Task** | **Model** | **Notes** |
| --- | --- | --- |
| Scene object detection | YOLOv8x (COCO, `yolov8x.pt`) + YOLO-World `yolov8s-worldv2.pt` open-vocab pass | Open-vocab promotes curated product labels (wristwatch, earbuds, sneaker …) that COCO cannot represent; curated label wins region on high IoU; every detection tagged `detection_source` |
| Logo detection | YOLO-World zero-shot | Catalog-generated `<BRAND> logo` text queries merged with generic fallbacks; scene-change keyframe sampling |
| OCR | PaddleOCR (PP-OCRv3) | `lang=hi` covers Devanagari + Latin wordmarks; logo crops upscaled ×2.0; full-frame OCR for products |
| Speech-to-text | mlx-whisper (MPS) → faster-whisper (INT8 CPU) → openai-whisper | Brand mention detection from shared catalog, exact word-boundary matching any script; fuzzy off by default |
| Audio events | BEATs (`BEATs_iter3_plus_AS2M.pt`) | 256-dim event summaries; ad/jingle cue evidence |
| Visual embeddings | DINOv3 (frozen, 768-dim) | Stride-sampled ≤30 frames |
| Product match | DINOv2 NN vs reference logo bank | Cosine NN; fails closed to 0 with empty bank |
| Logo classification | CLIP ViT-B/32 vs per-brand reference bank | Primary classifier for icon-only marks; requires `min_similarity 0.22` and `min_margin 0.10` |
| Central vision | Qwen3-VL 32B (gated, 8-bit) | Frame-filtered; loader fails closed until 32B weights wired |
| Cloud fallback | Free.ai | OCR/STT fallback when local models unavailable |

*Table 3.1: Layer 1 model-per-task assignment*

## 3.3 Layer 2a — The Modality-Quality-Aware Fusion Gate (the centerpiece)

Audio and video features first get a **measured** quality estimate, computed from the raw signal, not from learned features:

- Audio (`src/layer2/quality_estimator.py`, `AudioQualityEstimator`): SNR in dB via a percentile noise-floor estimate, and VAD confidence (fraction of active 25 ms frames) → 2 signals.
- Video (`VideoQualityEstimator`): Laplacian-variance blur, exposure (mean pixel, penalizing under/over-exposure), and detection-confidence variance → 3 signals.

These feed `LearnedGatingNetwork` — a small MLP (audio 2-dim → 64 → 64, video 3-dim → 64 → 64, concatenated → MLP → 2 logits, softmax) that outputs per-modality weights. The weighted representations then pass through `CrossAttentionFusion` (project both to 512-dim, prepend a learnable `[CLS]` token, 3-layer / 8-head transformer encoder, `[CLS]` output projection). A `FixedHeuristicWeighting` baseline (trust the modality above 0.3, equal otherwise) and an equal-weighting path provide the ablation controls.

Two design points matter. First, the gate's output is **sanity-checked per sample**: the direction and magnitude of its weights must agree with the measured quality; where the (initially untrained) gate disagrees, the code falls back to quality-proportional weighting rather than trusting random gating output. Second, all three weighting policies use the *same* fusion transformer, so the weight policy — and only the weight policy — is what differs between the compared arms.

*Figure 3.2: quality → weight → weighted cross-attention → fused representation*

## 3.4 Layer 2b — Evidence-Based, Decomposed Confidence

Confidence is not a single number. `evidence_breakdown` exposes per-source `{strength, weight, modulated_weight, contribution, status}`. Weights (config `layer2b`): logo 0.30, speech 0.20, OCR 0.18, visual product match 0.18, audio event 0.10; scene-context (0.04) and product-retrieval (0.0) are scaffolded and contribute zero. Weighted-sum aggregation with `min_evidence_threshold 0.55`: below it, the output is "no confident evidence" rather than a low-confidence guess. Speech evidence is modulated by estimated audio quality, so a speech-only mention counts for less when the audio is noisy.

## 3.5 Layer 2c — Brand Resolution and Temporal Memory

`BrandResolver` (`src/layer2/brand_resolver.py`, 990 lines) asserts a brand only through corroborated routes, in priority order:

1. **OCR on the logo crop / padded superset** — reads a real wordmark, matched against catalog aliases (any script); a multi-line brand card that splits across the box is retried on an upscaled superset.
2. **CLIP retrieval** — icon-only marks with no readable text are matched against the per-brand reference bank (from LogoDet-3K via `scripts/build_train_bank.py`), requiring similarity *and* top-1/top-2 margin floors.
3. **Corroborated class label** — a zero-shot class label (`Samsung logo`) is only trusted above 0.40 confidence *and* with an independent signal; otherwise it is `class_unconfirmed` and unresolved.

Fail-closed guards wrap all routes: a box exceeding 50% of frame area is treated as editorial/title-card content, not a brand appearance; OCR hits that read phone-screen/app-drawer UI (screen content) are suppressed; `TemporalBrandSmoother` stabilizes a persistent wordmark across frames instead of flickering, and never weakens an already-resolved brand. Unnameable persistent marks collapse into one `UNKNOWN BRAND` region that is shown but never becomes evidence.

Temporal memory extends across videos: `BrandMemory` persists resolved brands with provenance, recency decay and an embedding-model version tag (a stale index built under a different embedding model fails loudly). Product names spoken or read ("Mac Mini", "Z Fold 8") resolve to parent brands through **tiered product resolution** — Tier 1 catalog aliases, Tier 2 Wikidata SPARQL (auditable local cache, rate-limited), Tier 3 Qwen3-VL (budget-gated), Tier 4 cross-video learned co-occurrence (≥3 distinct videos) — with provenance recorded additively.

## 3.6 Layer 2d — Creator Profiling + Niche Suppression

A per-creator niche distribution is built from recurring content (scene evidence, detected categories). The recommendation layer downweights a SUGGESTED brand whose category fits poorly (e.g. a fitness brand for a cooking creator) via `niche_threshold 0.15 / suppress_factor 0.5`, so a single one-off brand mention cannot surprise a recommendation.

## 3.7 Layer 3 — Recommendation Engine

`KnowledgeGraph` derives brand → category → adjacent-brand adjacency from the curated catalog. `BrandRecommender` returns ranked recommendations of two kinds, each with `reasons[]`: **DIRECT** (evidence in this video) and **SUGGESTED** (never on screen; shares a category with an evidenced brand). Scores blend the evidence-driven base with a **LightGCN-based creator–brand affinity model** (`src/layer3/affinity.py` + `affinity_trainer.py`, trained on stored history via `_attach_affinity_model`) when the creator is known; otherwise it uses the explainable category/evidence heuristic — an honest cold start. Open-set brands beyond the catalog are handled by `src/openset.py`: real reverse-image search on the crop (Gemini-grounded or keyless browser-grounded backends) plus logo.dev external validation, cost-gated, and **fails closed with no backend** (no names are ever guessed).

## 3.8 Platform Integration and Safety

The trained pipeline feeds a Flask application (`server.py`, 1540 lines): upload → queued job → pipeline → dashboard JSON, annotated scene thumbnails, clip seek, ranked opportunities, open-set evidence cards, and outreach drafts. Draft email generation (`src/outreach.py`, 3 tones, grounded reasons) and Gemini-grounded brand contact lookup are **hard-gated**: a draft is only produced for a brand that a real logo.dev verification returns as `verified`, and the route fails closed without `LOGO_DEV_SECRET_KEY`; a brand never verified externally cannot reach a draft. Every draft is reviewed by the creator before anything is sent — the platform never emails a brand automatically. Jobs persist in SQLite; a PBKDF2-auth session layer protects the dashboard; deployment ships Docker, gunicorn and WSGI.

# 4. Implementation Status and Preliminary Results

## 4.1 What Has Been Built So Far

The full three-layer pipeline is implemented in code (`src/`), with a working server, frontend, infrastructure and tests:

- **Layer 1** — all perception modules implemented and wired through a concurrent pipeline (`src/pipeline.py`, 2347 lines): YOLOv8x + YOLO-World detection, logo detection with scene-change keyframe sampling, PaddleOCR, Whisper-family ASR, BEATs audio events, DINOv3 embeddings, DINOv2 product index, CLIP logo retrieval, Qwen3-VL (gated, fail-closed), Free.ai fallback. Frozen encoders: DINOv3, BEATs, YOLO, PaddleOCR, CLIP.
- **Layer 2** — modality-quality-aware fusion (`src/layer2/fusion.py`, 404 lines) and quality estimators (224 lines); evidence-decomposed confidence; brand resolution with temporal smoothing and the full fail-closed guard set; cross-video brand memory; tiered product→brand resolution; creator profiling + niche suppression.
- **Layer 3** — knowledge graph, explainable recommender, LightGCN affinity model with a training pathway and a demo seed database (`scripts/seed_db.py`).
- **Platform** — Flask server with job queue, SQLite persistence, auth, dashboard and insights APIs; outreach generation/forward (fail-closed); open-set identification with audited crops; vanilla-JS frontend ("Editorial Archival" design system); Docker/gunicorn/WSGI/Procfile; environment template `.env.example` (secrets never committed).
- **Brand catalog** — ~36 hand-curated brands with products, categories, per-language aliases (including Devanagari) and contact metadata (placeholders flagged `contact_verified: False`).
- **Tests** — 30 test modules covering auth, store, server jobs/draft-email/products/affinity, dashboard bounds, insights fallback, outreach persistence, pipeline, brand catalog (856+ lines), brand memory, temporal smoothing, spatial labeling, creator profiling, product resolution, embedding-version mismatch, open-set, logo.dev guard, email lookup, affinity training, layer-3, spec extraction, seed DB, and production infra.

## 4.2 Pipeline Validation Results

The validation run below was executed against this repository before writing this report; it is the *verified* state, not aspirational. Full multi-hour training and heavy-model runs remain (Section 4.3).

| **Component** | **Test performed** | **Outcome** |
| --- | --- | --- |
| Server module graph | `python3 -c "import server"` | Boots clean end-to-end (config + store bootstrap), exit 0 |
| Brand catalog + Layer 3 invariants | `pytest tests/test_brand_catalog.py tests/test_layer3.py` | **98 passed** (catalog/resolution + knowledge-graph/recommendation invariants hold) |
| BEATs checkpoint | On-disk verification `BEATs_iter3_plus_AS2M.pt` (361 MB) + vendored model code `BEATs/` | Present and loadable; config wired (`layer1.audio_events`) |
| Model weights | `yolov8x.pt`, `yolov8s-worldv2.pt` at repo root; `facebook/dinov3-vitb16-pretrain-lvd1689m` via transformers | Present; used by Layer 1 |
| Real video processing | Job store `var/contextlens.db` (jobs persisted) + 35 audited open-set crops in `static/openset_crops/` | Evidence the end-to-end platform processed real uploads (detection → crops → open-set pipeline) |
| CLIP retrieval tuning | Documented real 65 s stress-video run in `config.yaml` (logo_retrieval) | All 124 noise hits had margin ≤ 0.052; `min_margin 0.10` set from measurement; genuine matches (e.g. SAMSUNG 0.9 / others 0.3) retained |
| Fail-closed verification (outreach) | `test_logodev_guard.py`, draft-email tests, `test_outreach_forward_persist.py` | Without a valid logo.dev verification a draft is rejected — no fabricated brand reaches outreach |

*Table 4.1: Verifiable validation checks on the implemented pipeline*

These are correctness checks, not final research results. They confirm the system boots, the catalog/graph/recommendation invariants hold under test, the pretrained weights are genuine and wired, and real videos have passed end-to-end through the platform. They also record two early *qualitative* findings from the real stress videos the pipeline was tuned on: (a) zero-shot logo labels alone produce spurious high-confidence brand hits (e.g. `SUPREME` on Samsung/Apple marks) unless corroborated — which is why resolution requires OCR/retrieval corroboration; and (b) full-frame editorial content and phone-screen UI are routinely misread as brands unless gated — which is why the editorial-box and screen-content filters exist. Both findings changed the system (they motivated the fail-closed guards), which is exactly what validation-on-real-data is for.

## 4.3 What Remains

The infrastructure, code and verification are in place; what remains is execution of the research measurement and the final scaling steps.

| **Task** | **Target** |
| --- | --- |
| Fusion gate training script + trained checkpoint (config `layer2a.fusion.checkpoint` currently empty) | Before Review 3 |
| Layer 2a ablation on real creator video: learned-gate vs fixed-heuristic vs equal-weight, 5 seeds, bootstrap CI | Before Review 3 |
| Robustness suite: injected noise / blur / sync drift, measuring Δ confidence and Δ recommendation quality per arm | Final month |
| Qwen3-VL 32B weight wiring (currently fail-closed) and Tier-3 product resolution activation | Final month |
| Full heavy-model test-suite run (30 modules; two suites executed today) | Before Review 3 |
| Catalog scale-up, verified public brand contacts (`contact_verified`), closed first round of drafts | Final month |

*Table 4.2: Remaining work and target timeline*

# 5. Key Concepts, Technical Requirements and Future Work

## 5.1 Key Concepts Explained

- **Frozen pretrained model** — DINOv3, BEATs, YOLO, PaddleOCR and CLIP are used as-is; only the fusion gate and the affinity model are trained, keeping the system trainable on a single consumer GPU.
- **Modality-quality-aware fusion gate** — the part of Layer 2 that decides how much to trust each modality *per clip*, using *measured* signal quality (SNR, VAD, blur, exposure, detection stability) rather than learned features — the project's central research claim.
- **Fail-closed** — the operating principle that a brand name, recommendation or outreach draft must never be produced from a guess. Without corroborated evidence, the system outputs "unresolved"/"no confident evidence" instead.
- **Evidence decomposition** — confidence as an auditable vector of per-source contributions, so any low (or high) score can be traced to its cause.
- **Cold-start recommendation** — new creators have no interaction history; SUGGESTED recommendations come from the brand–category knowledge graph and a LightGCN affinity model where history exists, with `reasons[]` explaining every recommendation.
- **DIRECT vs SUGGESTED** — DIRECT = a brand evidenced in this video; SUGGESTED = a never-seen brand adjacent in the knowledge graph to an evidenced one (e.g. Puma recommended for a Nike-wearing creator).

## 5.2 Technical Requirements

- **Dataset / encoders**: creator video uploaded in-app; frozen pretrained DINOv3 (768-dim), BEATs (`BEATs_iter3_plus_AS2M.pt`, 256-dim), YOLOv8x/YOLO-World, PaddleOCR, CLIP ViT-B/32; per-brand reference logo bank built from LogoDet-3K.
- **Software**: Python, PyTorch, Hugging Face transformers, ultralytics, PaddleOCR, faster-whisper/mlx-whisper, librosa, OpenCV, Flask, SQLite, gunicorn/Docker.
- **Hardware**: a single consumer-grade GPU (or Apple Silicon via mlx-whisper/Metal) is sufficient; Layer 1 models are cached/frozen and Layer 2-3 is small.
- **Secrets**: `.env` (never committed) for `LOGO_DEV_SECRET_KEY`, `GEMINI_API_KEY`, admin password; `.env.example` documents the interface.

## 5.3 Future Extensions

- **Run the Layer 2a ablation** on real creator video across seeds with bootstrap confidence intervals, plus the degradation suite (noise, blur, sync shift) that quantifies whether the quality gate helps most exactly where it is designed to.
- **Train and ship the fusion gate checkpoint**, then wire Qwen3-VL 32B and activate Tier-3 product resolution with real budgets.
- **Scale the catalog** beyond ~36 curated brands using LLM-assisted mining, with verified public contacts.
- **Creator archive mode**: aggregate many videos per creator into a true layer-2d profile + affinity graph for long-horizon outreach targeting.
- **Graduate outreach** from reviewed drafts to a documented human-in-the-loop approval workflow with a real email/CRM integration.
- **Package the Layer 2a fusion result** for a research write-up alongside the final capstone report.

## 5.4 Summary

At this stage the project has moved from a research question into a complete, verified implementation: a three-layer system that perceives multimodal creator content, reasons about brand and context with fail-closed, quality-aware fusion, and recommends explainable brand collaborations through a knowledge graph and affinity model — served by a deployable web platform with audited, human-gated outreach. The immediate remaining work is the research measurement itself (fusion training, the ablation, and the robustness suite), for which the code, data paths and infrastructure are in place.