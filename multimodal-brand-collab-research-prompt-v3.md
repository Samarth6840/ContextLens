# Master Research + Build Prompt v3 — Context-Aware Multimodal Brand Collaboration Recommendation System
### (rebalanced: recommendation is the research contribution, perception is the input layer feeding it — not the other way around)

Use this with an AI research assistant. Two earlier drafts over-weighted perception (CV/OCR/ASR/
fusion) at the expense of the actual product value, which is the recommendation. This version fixes
that: it names the project by its recommendation contribution, structures everything into three
explicit layers, and puts modality-quality-aware fusion — the strongest single research angle —
front and center instead of buried in a fusion-strategy comparison list.

---

```text
Act as a Senior AI Researcher, Computer Vision Scientist, Speech/Audio ML Engineer, NLP Engineer,
Recommendation Systems Engineer, and SaaS Product Architect. Where you are uncertain whether a model
I list below is still current or has been superseded, say so explicitly — treat my model lists as a
2026 starting shortlist, not a final answer, and cite sources for any "state of the art" claim.

## 0. Project identity — read this before anything else

This is NOT "brand detection using multimodal learning." It is:

  **Context-Aware Multimodal Brand Collaboration Recommendation System**

Research question: How can multimodal AI understand creator content and recommend suitable brand
collaborations under noisy, incomplete, or conflicting modalities?

Detecting a brand is an intermediate signal, not the deliverable. The deliverable is a ranked,
explainable list of brands worth collaborating with — which depends on what's detected PLUS who the
creator is (topics, audience, engagement style) PLUS how those brands relate to each other and to
the creator's niche, even brands that never appeared on screen. Weight your answer accordingly: don't
let perception-layer detail crowd out the recommendation layer the way a generic multimodal-AI answer
would default to.

Split further into:
(A) BRAND & CONTEXT INTELLIGENCE (AI problem, Layers 1-2 below)
(B) COLLABORATION AUTOMATION (backend/product problem — email/CRM/WhatsApp APIs, Layer 3's output)
Treat (B) as ordinary engineering. Tag every section below with which problem/layer it belongs to.

## 1. Three-layer architecture — design each layer, and the interfaces between them

```
Video Upload
   │
   ▼
Layer 1 — Multimodal Understanding
   Visual: detection, logo detection, OCR, visual embeddings
   Audio: speech-to-text, non-speech audio events
   Text: NER, sentiment, intent (downstream of ASR)
   │
   ▼
Layer 2 — Brand & Context Intelligence
   Modality-quality estimation → dynamic modality weighting → quality-aware fusion
   → evidence aggregation → calibrated brand confidence
   Creator profiling (topics, audience signals, visual/content style) built in parallel
   Temporal memory / cross-scene reasoning (resolve "this phone" at minute 9 to the
   Apple logo shown at minute 1)
   │
   ▼
Layer 3 — Recommendation Engine
   Knowledge graph (brand → category → adjacent brands, so Puma/Asics/Decathlon can be
   recommended even if they never appeared) + creator-brand affinity model + vector
   retrieval → ranked, explainable brand recommendations → outreach automation
```

For each layer, specify: what it consumes from the layer below, what it hands to the layer above,
and what would break if that interface were skipped (e.g., what's lost if recommendation ran
directly off raw detections with no creator profile or knowledge graph in between).

## 2. Layer 1 — Multimodal Understanding (perception; keep this scoped, don't let it sprawl)

Pick ONE model per task for the actual build — list alternatives for the research-comparison
section, but commit to a single default so this stays buildable solo:

| Task               | Default pick                  | Why (and swap-test alternative)                    |
|--------------------|--------------------------------|----------------------------------------------------|
| Object/logo detect | RF-DETR or YOLO26              | real-time, strong domain transfer; swap-test vs. Grounding DINO for zero-shot rare brands |
| Visual embeddings  | DINOv3                         | frozen backbone, already your Review-1 choice       |
| OCR                | PaddleOCR (or GOT-OCR 2.0 if layout matters more than speed) | fast, wide language coverage |
| Speech-to-text     | Whisper large-v3               | broadest multilingual coverage; swap-test vs. Qwen3-ASR/Voxtral for Hindi/Hinglish WER |
| Audio events       | BEATs                          | your existing baseline; swap-test vs. ATST-SED/MAT-SED |
| Text (NER/sentiment/intent) | downstream of ASR, Phase 2 | not required for the Review-1 slice |

Confirm or challenge this table — if a swap-test alternative is clearly better for this specific use
case (Hindi/Hinglish creator speech, Indian product logos), say so and explain the trade-off instead
of defaulting to whichever is more famous.

## 3. Layer 2a — Modality-quality-aware fusion (the strongest research contribution — treat it as the centerpiece, not one bullet in a fusion-comparison list)

Design explicitly:

```
Audio  → quality estimate → weight
Video  → quality estimate → weight
                ↓
        Quality-aware fusion (e.g. video 95% quality / audio 22% quality
                                → video weighted ~80%, audio ~20%)
                ↓
        Cross-attention transformer over the weighted representations
```

This single mechanism is what should let the system degrade gracefully across noisy audio, Hindi/
Hinglish speech, missing modalities, poor lighting, and lip-sync mismatches — instead of needing a
bespoke fix per failure mode. Specify:
- What signal estimates "quality" per modality (SNR/VAD confidence for audio; blur/exposure/
  detection-confidence variance for video) — concretely, not just "a quality score."
- The actual weighting function (learned gating network vs. a fixed heuristic vs. attention-based
  soft weighting) and how it's trained/calibrated.
- How this compares to plain cross-attention fusion without quality weighting — this comparison
  (ablation: fused w/ dynamic weighting vs. fused w/o) IS the research contribution; make sure the
  evaluation plan in Section 8 actually measures this delta, not just fused-vs-unimodal.

## 4. Layer 2b — Evidence-based confidence (not a single number — a decomposed, explainable score)

```
Evidence            Weight
Logo detected        0.45
Speech mention       0.20
OCR hit              0.15
Scene context        0.10
Product retrieval    0.10
                    ------
Final confidence     0.90 (aggregation function — specify: weighted sum? learned
                            calibration layer? noisy-OR? — and how it's calibrated,
                            e.g. reliability diagrams / expected calibration error)
```

Below a minimum-evidence threshold, output "no confident evidence" rather than a low-confidence
guess. This decomposition should visibly reuse the per-modality quality weights from Section 3 where
relevant (e.g. a speech-only mention should count for less when audio quality was estimated low).

## 5. Layer 2c — Temporal memory / cross-scene reasoning

Explicit module: if a brand/product is visually established early (Apple logo at minute 1) and later
referred to indirectly ("this phone" at minute 9), the system must resolve the reference using stored
context rather than treating each frame/utterance independently. Specify:
- What gets stored as "memory" per detected entity (embedding + timestamp + modality source) and for
  how long/how far across a long video this should persist.
- The mechanism for resolving an indirect reference against that memory (attention over a running
  memory bank, a retrieval step against recent detections, or a temporal transformer with explicit
  memory tokens — compare options).
- How this differs from, and can reuse, the temporal modeling already planned for the perception
  layer — don't design two separate temporal mechanisms if one can serve both purposes.

## 6. Layer 2d — Creator profiling (parallel to brand detection, feeds Layer 3)

The system should not recommend collaborations off raw brand mentions alone. Build a creator profile
from: recurring topics/content category, audience signals (if available — engagement style, not
demographic scraping that raises its own privacy issues — flag this explicitly), visual/production
style, and brand affinity accumulated over multiple videos (not just the one being analyzed). Give a
concrete counterexample like: creator wears Nike once, but content is 45+ audience cooking videos
with low engagement — explain why the profile should suppress, not surprise, a Nike recommendation
in that case, and what signal specifically should suppress it.

## 7. Layer 3 — Recommendation engine

Combine, and be explicit about how each feeds the final ranking:
- Knowledge graph: brand → product category → adjacent brands/competitors, so the system can
  recommend Puma/Asics/Decathlon for a Nike-wearing fitness creator even though they never appeared
  on screen. Specify how this graph gets built/maintained (manual curation vs. mined from a product
  taxonomy vs. LLM-assisted graph construction) and kept current as new brands appear.
- Creator-brand affinity model: GNN-based collaborative filtering (LightGCN-style) over a
  creator–product–brand graph, capturing ecosystem affinity, not just co-occurrence.
- LLM-as-ranker / knowledge-graph-augmented LLM reasoning: since this is a genuine cold-start problem
  (new creators, new brands, sparse interaction history), evaluate whether an LLM-based ranking layer
  on top of embedding retrieval outperforms a pure learned ranking model here specifically, versus in
  a data-rich recommender setting where it might not be needed.
- Vector retrieval (Qdrant/Weaviate/Milvus — justify one pick for catalogue size and team capacity)
  against a brand/product embedding catalogue.
- Explainability: every recommendation should surface which evidence (Layer 2b), which creator-
  profile signals (Layer 6), and which graph relationships (this section) drove it — not just a
  ranked list with no rationale.

## 8. Real-world failure modes — verify the architecture above actually handles these

No audio track · no video track (podcast) · noisy audio · Hindi/Hinglish/code-switched speech ·
sarcasm/negative sentiment toward a named brand · brand never named explicitly · logo partially
occluded · product with no visible logo · brand shown briefly · poor resolution/lighting/motion blur
· AI-generated video with imperfect lip-sync · audio/video evidence disagreeing · brand only
recoverable via OCR · indirect reference needing temporal resolution (Section 5) · brand visible but
never spoken, or vice versa · frequent scene changes · multiple co-occurring brands · long-form video
where relevant moments are sparse · domain shift between training data and real creator video.

For each: which layer/module (1, 2a-d, or 3) is responsible for handling it, and does the
architecture above actually cover it, or does this failure mode expose a gap you should call out?

## 9. Datasets

Logo detection: LogoDet-3K (~200K annotated objects, 3,000 categories), QMUL-OpenLogo (352
categories), FlickrLogos-32 (evaluation-scale). OCR: ICDAR series, olmOCR-bench. Product: DeepFashion,
Stanford Online Products, Product-10K. Speech: Common Voice, LibriSpeech, plus a fresh check on
current Hindi/Hinglish/code-switched corpora (don't answer this from memory — verify what exists now).
Audio events: AudioSet (Review-1 base) + DESED for event-level synthetic soundscapes. Explain why
AudioSet alone is insufficient once scope expands into brand detection (its ontology has no brand
categories — it wasn't built for this). Recommendation/knowledge-graph: what exists for creator-brand
or product-category graphs vs. what you'd need to construct yourself (product taxonomy sources,
LLM-assisted graph construction, manual curation for a first version).
Propose synthetic-data methods for gaps: TTS-generated multilingual brand-mention audio, synthetic
occlusion/blur/compression for video, and how to validate synthetic data doesn't introduce its own
distribution gap.

## 10. Evaluation

Map metrics to layers, not just modules: Precision/Recall/F1/mAP → Layer 1 detection. ROC-AUC/ECE →
Layer 2b confidence calibration. The Layer 2a fusion ablation (dynamic-weighted vs. plain fusion,
under synthetic noise/blur/occlusion — tied to the existing <10% relative-performance-drop success
bar) is the centerpiece experiment; design it explicitly rather than folding it into a generic
ablation list. Top-k accuracy/Recall@K/NDCG/MRR → Layer 3 recommendation ranking. Latency/FPS/GPU
utilization → deployment. Specify a way to evaluate creator-profile-driven suppression (Section 6)
— e.g. a held-out set of creator/brand pairs where a naive co-occurrence model would over-recommend
and the profile-aware model shouldn't.

## 11. Deployment (Phase 3, not Review-1 scope)

Frontend/backend split, inference server, GPU scheduling, microservice boundaries, vector database
from Section 7, object storage, metadata database, monitoring, caching, CI/CD, and the email/CRM/
WhatsApp outreach automation layer. Flag what's overkill for solo/small-team execution.

## 12. Ethical, privacy, and legal considerations

Creator consent for content analysis, trademark/logo-usage concerns in automated detection, data
retention for uploaded video, disclosure obligations for AI-solicited sponsorships, and — specifically
flagged in Section 6 — the line between legitimate audience-engagement-style profiling and
demographic/personal-data scraping that raises its own privacy problems.

## 13. Research framing

The primary contribution is modality-quality-aware fusion for brand/context detection under
real-world creator-video degradation (Section 3), evaluated via the fused-with-weighting vs.
fused-without-weighting ablation. Secondary contributions: knowledge-graph-augmented, LLM-assisted
cold-start creator-brand recommendation (Section 7), and temporal cross-scene entity resolution
(Section 5). Be explicit and honest about which of these three is strong enough for a standalone
paper vs. which are solid engineering contributions supporting the main one. Suggest realistic
publication targets given project scale.

## 14. Phased roadmap — do not attempt all layers at once

Phase 1 (current/Review-1 scope): Layer 1 + Layer 2a/2b only — multimodal brand/context
understanding with quality-aware fusion and explainable confidence, evaluated on AudioSet-style data.
Phase 2: add Layer 2c (temporal memory), Layer 2d (creator profiling), and Layer 3's knowledge graph
+ affinity model.
Phase 3: SaaS features — dashboard, email/CRM/WhatsApp outreach automation, production deployment.
For each phase, state what changes in the architecture, training data, and evaluation harness
relative to the phase before it.

## 15. Deliverables

1. Text architecture diagram for the full three-layer system, plus a smaller diagram scoped to
   exactly Phase 1.
2. Data flow diagrams for both scopes.
3. Layer-by-layer explanation with the one-model-per-task table (Section 2) justified, plus
   research-comparison alternatives noted separately.
4. The Layer 2a fusion ablation design (Section 3) in full experimental detail — this is the
   centerpiece, give it more depth than any other single deliverable.
5. Dataset comparison table (Section 9).
6. Research gaps/contribution statement, ranked by strength (Section 13).
7. Production architecture (Section 11), explicitly marked Phase 3.
8. The phased roadmap (Section 14) with per-phase architecture/data/eval deltas.
9. Experimental methodology consistent with the existing 70/15/15 split and robustness-under-
   degradation testing, extended to explicitly cover the fusion ablation.
10. Risks/limitations — call out anywhere the three-layer design has an unresolved interface (e.g.
    what happens to Layer 3 recommendations when Layer 2 confidence is systematically low for an
    entire creator due to consistently poor audio, not a per-video accident).
11. Concrete next-2-weeks action items assuming solo/small-team execution, scoped to Phase 1 only.

Cite specific 2025/2026 papers, benchmarks, or model releases for any "best" or "state of the art"
claim, and flag uncertainty rather than asserting currency without a source.
```
