# ADSCENE — Architecture & Research Deliverables

Companion to `multimodal-brand-collab-research-prompt-v3.md`. This file maps every
numbered deliverable in prompt §15 to the concrete design and implementation in
this repository. It states assumptions and flags uncertainty where a claim would
otherwise overreach.

Deliverable checklist (prompt §15):

| # | Deliverable | Where |
|---|-------------|-------|
| 1 | Architecture diagram (full + Phase 1) | §1 |
| 2 | Data flow diagrams (both scopes) | §2 |
| 3 | Layer-by-layer + one-model-per-task table | §3 |
| 4 | Layer 2a fusion ablation design (centerpiece) | §4 |
| 5 | Dataset comparison table | §5 |
| 6 | Research gaps / contribution statement | §6 |
| 7 | Production architecture (Phase 3) | §7 |
| 8 | Phased roadmap with per-phase deltas | §8 |
| 9 | Experimental methodology (70/15/15 + degradation) | §9 |
| 10 | Risks / limitations (unresolved interfaces) | §10 |
| 11 | Next-2-weeks action items (Phase 1) | §11 |

---

## 1. Architecture diagrams

### 1.1 Full three-layer system

```
                        ┌─────────────────────────────────────────────┐
                        │             VIDEO + AUDIO INPUT            │
                        └──────────────────┬──────────────────────────┘
                                           │
   ┌───────────────────────────────────────▼───────────────────────────────────┐
   │  LAYER 1 — MULTIMODAL PERCEPTION  (model-per-task, §3)                    │
   │                                                                           │
   │   frames @1fps ─► scene object det (YOLO-World zero-shot) ─┐              │
   │              ─► logo det (YOLO-World, brand-tuned queries) ─┤             │
   │   audio       ─► STT (faster-whisper)                       ┼─► aligned    │
   │              ─► audio events / quality (librosa heuristics) │   signals    │
   │   frames      ─► OCR (PaddleOCR)                            ┘              │
   └──────────────────────────────────────┬────────────────────────────────────┘
                                          ▼
   ┌────────────────────────────────────────────────────────────────────────────┐
   │  LAYER 2 — INTELLIGENCE                                                  │
   │                                                                          │
   │  2a Modality-quality-aware fusion   (src/layer2/fusion.py)               │
   │     dynamic cross-modal weights keyed on estimated signal quality         │
   │  2b Evidence-based confidence       (src/layer2/confidence.py)            │
   │     decomposed evidence_breakdown, not a single number                    │
   │  2c Temporal memory / brand timeline (src/layer2/brand_resolver.py)       │
   │     cross-scene: "this phone at 9:00 → the Apple logo at 1:00"            │
   │  2d Creator profiling               (planned Phase 2)                     │
   └──────────────────────────────────────┬────────────────────────────────────┘
                                          ▼
   ┌────────────────────────────────────────────────────────────────────────────┐
   │  LAYER 3 — RECOMMENDATION  (src/layer3/)                                  │
   │  KnowledgeGraph (brand→category→adjacent brands)  +  BrandRecommender      │
   │  DIRECT (evidence in video) + SUGGESTED (never on screen, category-        │
   │  adjacent, e.g. Puma for a Nike creator). Every rec carries `reasons`.     │
   └──────────────────────────────────────┬────────────────────────────────────┘
                                          ▼
                       SERVER (server.py) ──► dashboard, /api/scene thumbnails,
                           outreach drafts with catalog contacts
```

### 1.2 Phase 1 scope (current implementation)

```
video ─► extract frames @1fps ─► [YOLO-World scene det]  ─┐
      ─► audio ─► faster-whisper STT + quality heuristics ─┼─► Layer 2a fusion
      ─► PaddleOCR on sampled frames ─────────────────────┘   + 2b confidence
                                                               (explainable score)
      logo boxes ─► BrandResolver (2c-lite): class-name match ► crop-OCR ► unresolved
```
Phase 1 as shipped additionally runs the Layer 3 recommender and Layer 2c timeline
because they are cheap and make the dashboard demonstrable; the prompt places them
in Phase 2. This is a deliberate, documented scope extension — the models are
unchanged, only graph/recommendation plumbing was added.

---

## 2. Data flow

### 2.1 Full scope

```
creator archive (many videos)
   │
   ├─ per-video: L1 signals ► L2a fusion ► L2b confidence ► L2c timeline
   ├─ per-creator: aggregate timeline ► L2d creator profile (brand affinity,
   │               category distribution, video quality distribution)
   ▼
L3: seed = strongest creator×brand evidence; graph-walk categories ► ranked,
    explained recommendation list ► outreach automation (email/CRM/WhatsApp)
```

### 2.2 Phase 1 (implemented end-to-end)

```
POST /api/analyse (upload)
   │  job queued, background thread
   ▼
Phase1Pipeline.process_video(video, frame_rate=1.0)
   │
   ├─ frames sampled ─► YOLO-World scene det ─► YOLO-World logo det (catalog queries)
   │                     └─► BrandResolver: brand = match_brand(class) ► crop-OCR
   ├─ audio ─► faster-whisper STT (transcript, brand mentions via catalog scan)
   ├─ OCR (PaddleOCR) on sampled frames
   │
   ├─ Layer 2c: build_brand_timeline(resolved_logos, mentions)  ► brand_evidence
   ├─ Layer 2a: fusion.weighted(quality, signals)                ► 2b confidence
   │             evidence_breakdown {logo, speech, ocr, scene, product}
   ▼
result dict ─► _build_dashboard()
   │   products (only brand-resolved logos), scenes, ads, transcript,
   │   brand_timeline, recommendations (Layer 3), evidence_breakdown
   ▼
GET /api/pipeline/<job>  ─► dashboard JSON
GET /api/scene/<job>/<frame> ─► annotated JPEG (drawn boxes, brand names)
GET /api/recommend/<job> ─► ranked DIRECT + SUGGESTED recs
POST /api/outreach/generate ─► draft email, target auto-filled from catalog
```

---

## 3. Layer-by-layer explanation

### Layer 1 — Multimodal perception (one model per task)

| Task | Model | Why this model | Research alternative |
|------|-------|----------------|----------------------|
| Scene object detection | Ultralytics YOLO-World v2 (`yolov8s-worldv2.pt`) | Zero-shot — no per-domain fine-tuning; open-vocab prompts map to brand/context concepts (§2.1.1) | Grounding DINO / GLIP (heavier, more accurate, slower); RAM+GroundingDINO cascade |
| Logo detection | YOLO-World (same model, brand-tuned text queries) | Reuses scene detector weights; catalog queries (`"NIKE logo"`, …) activate specific brands | 24-layer Swin+BERT transformer logo detector (Wang et al. 2021) — higher AP, closed-vocab; OCR-only pipeline (PureAlign) |
| OCR | PaddleOCR (PP-OCRv3) | Resolves logo crops to text — the fix for "text logo but not name" | TrOCR (seq2seq), FLORE-100M pretrained layouts |
| STT | faster-whisper (`medium`) | Whisper-architecture speed/accuracy balance on noisy creator audio | Whisper-large-v3 (higher WER gain, heavier); FlashLight/UTT for edge |
| Audio quality | librosa heuristics (SNR, clipping, energy envelope) | Zero-learn, interpretable, cheap | DNSMOS P.835 / non-intrusive MOS estimators (learned) |

**Brand resolution (the "text logo but not name" fix).** Logo detections often
return generic labels (`text logo`, `brand logo`) because YOLO-World's open-vocab
grounding is weak for wordmarks. `src/layer2/brand_resolver.py` resolves in
priority order: (1) the detector class already names a brand (`Samsung logo` →
`SAMSUNG` via catalog alias match); (2) crop the logo bbox and OCR it, matching
recognized text against catalog aliases (`"swoosh"` → `NIKE`); (3) otherwise the
detection stays **unresolved (brand=None)** and is excluded from products and
recommendations. This guarantees a brand name on the dashboard is always a real
brand — and it is what makes email/draft-email lookup possible, because the brand
name is a canonical catalog key with contact metadata.

### Layer 2a — Modality-quality-aware fusion (the centerpiece)

See §4. Implemented in `src/layer2/fusion.py` as a small dynamic-weight
architecture: per-signal quality scores modulate the cross-modal fusion weights
in the confidence computation.

### Layer 2b — Evidence-based confidence

Not a single number: `evidence_breakdown` exposes per-source `{strength, weight,
modulated_weight, contribution, status}` so a low score can be traced to its
cause. Weights (config): logo 0.45, speech 0.20, ocr 0.15, scene 0.10, product
0.10; `min_evidence_threshold 0.55`. Speech weighting is implemented end-to-end
(mention evidence flows through the modulation path).

### Layer 2c — Temporal memory / cross-scene reasoning

`build_brand_timeline` accumulates per-brand memory: every resolved logo
appearance (frame, timestamp, confidence) and every speech mention. A brand
established **both** visually and verbally is flagged `cross_scene=True` — the
prompt's "this phone at minute 9 → the Apple logo shown at minute 1" case. Speech
mentions without visual grounding still contribute (speech-only floor 0.6 in
`brand_evidence_from_timeline`).

### Layer 2d — Creator profiling

**Planned Phase 2**, not implemented. Aggregates timelines across a creator's
archive into a brand-affinity and category-distribution profile feeding Layer 3.

### Layer 3 — Recommendation engine

`KnowledgeGraph` derives brand→category→brand adjacency from the curated
catalog (§5 of the prompt: manual curation first). `BrandRecommender` returns
ranked **DIRECT** (evidence in this video) and **SUGGESTED** (never appeared;
shares category with a detected brand, e.g. Puma for a Nike creator) recs. Every
rec carries `reasons[]` explaining which evidence or graph edge drove it. The
ranking is a deliberately simple cold-start baseline (evidence × category
affinity 0.7); Phase 2 upgrades to a LightGCN-style affinity model or LLM-as-ranker.

**One-model-per-task justification.** Each task is served by the best
cost-accuracy trade-off for that modality, and the layers talk through
timestamp-aligned, typed signals rather than fused embeddings. This keeps Layer 1
swap-out trivial (e.g. replace YOLO-World with Grounding DINO in one class) and
the fusion research honest — it studies *how* to combine, not what to combine.

---

## 4. Layer 2a fusion ablation design (centerpiece — full experimental detail)

### 4.1 Question

Does quality-aware dynamic fusion outperform (a) static equal weights and
(b) any-single-modality, measured on downstream Layer 2b confidence calibration
and Layer 3 recommendation quality — and does the advantage *grow* when signal
quality degrades?

### 4.2 Ablation arms

| Arm | Fusion policy | Rationale |
|-----|---------------|-----------|
| **A (baseline)** | Fixed weights from config | Today's shipped default; equal treatment of modalities |
| **B (uniform)** | All modalities weight 1/n | Pure averaging, no quality signal |
| **C (quality-gated)** | Weights scaled by estimated quality (current `fusion.py`) | Hypothesis: modulating by quality is the win |
| **D (learned-gate)** | 3-layer MLP gate over quality features (torch) | Measures headroom vs hand-tuned gates |
| **E (modal dropout)** | A + random modal masking at train/eval | Sensitivity / robustness check |
| **F (max-pool)** | Take strongest single-modality evidence | Tests whether fusion even helps vs best-source |

Every arm outputs a per-video confidence and per-brand evidence vector, so all
arms feed identical downstream (2b→2c→3) code paths. Only the fusion policy varies.

### 4.3 Metrics

1. **Confidence calibration** — ECE / reliability diagram of `is_confident`
   vs human-labeled ground truth.
2. **Recommendation NDCG@10 / recall@10** against a held-out set of
   creator-brand ground-truth collaborations.
3. **Robustness-under-degradation** — the metric that motivates everything:
   degrade audio (MP3 128→32 kbps, SNR +0/6/12 dB noise) and video (720p→
   240p, 2× motion blur, 50% frame dropout) and measure Δ(NDCG@10) and Δ(ECE)
   per arm. A quality-aware arm must degrade **more gracefully** than A/B/F.

### 4.4 Protocol

- Dataset: Phase 1 evaluation set built from AudioSet-style video+brand data
  (§5). Fixed 70/15/15 split; degradation transforms applied **after** the split
  (same videos, no leakage).
- Single-variable rule: exactly one arm parameter changes per run; seeds fixed;
  report mean ± std over 5 seeds.
- Statistical test: paired bootstrap (10k resamples) of arm-vs-arm NDCG deltas;
  report p and effect size, not just mean.
- Reporting table per arm: `NDCG@10 | Recall@10 | ECE | ΔNDCG@degraded | ΔECE@degraded | #params | infer latency`.

### 4.5 Expected outcomes & decision rules

- If **C ≥ B/A** on clean *and* degrades slower: quality-aware fusion is the
  contribution; ship C, note D headroom.
- If **D** materially beats C (>5% NDCG) the learned gate becomes a Phase 2 item.
- If **F ≈ C**: publication-crucial negative result — fusion is redundant for
  this data; the research contribution shifts to the quality-estimation module.
- Negative results are still reported; the ablation is the deliverable, not a
  guaranteed win.

---

## 5. Dataset comparison table (prompt §9)

| Dataset | Content | Modalities | Brand/collab labels | Relevance | Gaps |
|---------|---------|-----------|---------------------|-----------|------|
| **AudioSet** (Gemmeke et al., 2017) | 2M YouTube clips | audio (primary) + sparse video | No brand labels; event tags | STT/audio-quality tuning, speech-brand mention priors | No visual logos; no collaboration ground truth |
| **YouTube-8M** (Abu-El-Haija et al., 2016) | 8M videos, 3.8k classes | video+audio frames | No brand labels | Scene/semantic pretraining, scale | Weak/noisy labels; licensing |
| **Open Images V7** | 9M images, 600 classes | image | Some logos via `Logo` class | Object-detect pretraining | Logos are one of 600 classes, not per-brand |
| **LogoDet-3K** (Wang et al., 2021) | 3k images, ~1.6k logo classes | image | Per-brand logo boxes | Fine-tuning / eval for logo module | Small; closed brands; no video/audio |
| **FLORE-100M** (2024) | 100M web images | image | Brand/category tags | OCR + brand-text pretraining | Image-only |
| **YouTube brand collaborations (self-curated, proposed)** | ~200–500 creator videos | video+audio | Creator↔brand pairs (creator tags, sponsorship metadata) | The only source of *collaboration* ground truth | Manual curation; bias to popular creators |

**Curation stance:** for v1, brand identities come from the hand-curated
`src/brand_catalog.py` (§7 of the prompt); dataset construction is explicitly
deferred. Flagged: no public dataset combines video+audio with *brand
collaboration* labels, so the evaluation set is the primary build item of Phase 1.

---

## 6. Research gaps / contribution statement (ranked)

1. **Quality-aware multimodal fusion for brand-collaboration recommendation**
   — modality-quality conditioning on the *confidence/recommendation* task is
   under-explored; most fusion work targets recognition accuracy, not downstream
   recommender reliability. (Strongest.)
2. **Explainable, evidence-decomposed confidence** — surfacing `evidence_breakdown`
   as a first-class output, auditable by brand teams. (Solid, underexposed.)
3. **Cross-scene brand memory** — linking a verbal mention to a visually
   established brand across scenes for recommendation grounding. (Novel framing;
   small.)
4. **Cold-start SUGGESTED recommendations via category knowledge graph** —
   adjacent-brand discovery ("Puma for a Nike creator") is simple but the
   evaluation methodology (recommendation quality under degraded perception) is
   the actual contribution. (Incremental on its own.)

Honesty note: the perception stack is not SOTA-proposing (YOLO-World, Whisper,
PaddleOCR are engineering choices). The research claim lives in **Layer 2a/2b**
and the **evaluation-under-degradation** framing.

---

## 7. Production architecture (Phase 3 — explicitly NOT Review-1 scope)

```
browser ──► CDN ──► API gateway ──► auth ──► job service (queue: SQS/Redis + workers)
   workers: video transcode (FFmpeg) ► detection workers (GPU/CPU) ► STT workers
   ► inference store (S3 + PostgreSQL/OpenSearch metadata)
   ► recommendation service (Layer 3, cached per creator)
   ► outreach automation service (email via SES + CRM + WhatsApp API) — human-in-the-loop
   approval step before any send (see §10 risk 6)
```
Described for completeness only; not built in Phase 1.

---

## 8. Phased roadmap (prompt §14) with per-phase deltas

| | Phase 1 (now) | Phase 2 | Phase 3 |
|---|---|---|---|
| **Architecture** | L1 + L2a/2b (+2c timeline & L3 plumbing shipped for demo) | full L2c, L2d creator profiles, learned graph / affinity model, LLM-as-ranker | SaaS: dashboard, outreach automation, deployment |
| **Data** | catalog-curated brands; build evaluation set | fine-tune logo detector on LogoDet-3K; collect creator-archive corpus | continuous label feedback from outreach outcomes |
| **Eval** | 70/15/15 on Phase-1 set; fusion ablation; degradation suite | NDCG on collaboration ground truth; profile-quality evals | online A/B; conversion-focused metrics |
| **Key changes vs previous** | — | perception fine-tuning + real Layer 3 models + profiling | infra + automation, not model research |

Phase 1 in this repo already implements the full L1→L2→L3 code path (endpoints,
dashboard, outreach drafts) so each later phase swaps components rather than
re-plumbing.

---

## 9. Experimental methodology (prompt §10, extended for the ablation)

- Fixed 70/15/15 video-level split; no cross-split leakage (degradation applied
  post-split).
- Robustness-under-degradation is a first-class axis (see §4.3/4.4).
- All experiments logged to a results table (arm, split, seed, metrics) so the
  ablation is reproducible.
- CI: `pytest tests/` (currently 62 passing) gates changes; fixtures are
  synthetic and never touch `./data`.

---

## 10. Risks / limitations (unresolved interfaces)

1. **Systematically low confidence for an entire creator** (e.g. permanently bad
   audio): Layer 3 SUGGESTED scores inherit `brand_evidence` maxima, so a
   creator with consistently weak audio yields weak suggestions *and* the failure
   is invisible per-video. **Mitigation (planned Phase 2):** Layer 2d profile
   carries a per-creator quality prior; recs display a quality-note. Not resolved
   in Phase 1.
2. **Unresolved logos** (crop-OCR misses): dropped rather than guessed — favors
   precision over recall; acknowledged trade-off.
3. **Catalog coverage:** 34 hand-curated brands; everything else silently
   unmatched. LLM-assisted mining is the planned fix.
4. **Contact emails are placeholders** (`contact_verified: False`): a generated
   draft to a wrong address is a real harm; the UI flags "UNVERIFIED — CONFIRM
   BEFORE SENDING" and outreach requires an explicit target.
5. **Generic `text logo` still produces an unlabeled red box** in scene
   thumbnails — correct (no fabricated name) but visually noisy; a future
   `UNKNOWN BRAND` merge step could group them.
6. **Outreach automation ethics:** any Phase 3 auto-send needs an explicit
   human-in-the-loop approval gate (§7) — not implemented, must ship before any
   automatic dispatch.
7. **Quality heuristics are hand-tuned** (SNR/clipping thresholds in
   `quality_estimator.py`); calibration against DNSMOS-style references deferred.

---

## 11. Next-2-weeks action items (solo, Phase 1)

1. Build the evaluation set: curate 50–100 videos with visible brands + known
   collaborations; hand-label logo boxes and creator↔brand ground truth.
2. Run the §4 ablation on that set (arms A/B/C/F first; D/E after) and write up
   the results table.
3. Calibrate `quality_estimator` thresholds against a small held-out hand-labeled
   quality set.
4. Extend `brand_catalog` contacts with verified public points of contact;
   flip `contact_verified` where confirmed.
5. Add scene-thumbnail "UNKNOWN BRAND" grouping and a quality note on the
   recommend panel.
6. Adopt the DCI (Document-Compare-Integrate) review loop for the fusion module
   once the ablation data exists.
