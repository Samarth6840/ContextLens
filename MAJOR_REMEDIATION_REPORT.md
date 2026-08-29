# MAJOR REMEDIATION REPORT

**Incident:** Shipped feature producing fabricated brand-collaboration data wired to live brand PR email addresses
**Scope:** Part A (live incident) · Part B (benchmarked logo detection rebuild) · Part C (real audio brand-mention detection)
**Status of this document:** Self-audited. No number in this report is mocked, LLM-invented, or hardcoded as a stand-in for a real run. Every metric comes from a real execution against real data; anything that could not be completed for real is explicitly marked **OPEN** rather than presented as complete.

---

## PART A — ACTIVE INCIDENT: FABRICATED BRAND-COLLABORATION DATA

### A1 — Immediate containment

**1. DRAFT EMAIL feature disabled — CONFIRMED DONE before any investigation.**
- `config/config.yaml` → `ui.outreach_enabled: false` (config/config.yaml:119). The flag defaults to **OFF even if config load fails** (server.py:56), so safety is the default state, not an opt-in.
- Server-side 403 gate: `_outreach_gate()` (server.py:74) returns 403 for `/api/outreach/generate` and `/api/outreach/forward` whenever outreach is disabled.
- Frontend: `static/app.js` gates every outreach control on `outreach_enabled`; when off, the outreach panel and editor render a "OUTREACH DISABLED" error box and all DRAFT EMAIL / GENERATE DRAFT / FORWARD controls are hidden and non-interactive (static/app.js:412-442).
- **Verified live:** `GET /api/health` on the running server (restarted on current code) returns:
  `{"jobs":0,"ok":true,"outreach_enabled":false,"outreach_reason":"DRAFT EMAIL DISABLED — PENDING DATA-INTEGRITY REVIEW (see MAJOR_REMEDIATION_REPORT.md)"}`

**2. Was any real email ever sent? — DEFINITIVE: NO.**
- There is no email-sending code anywhere in the repository. `grep` for `smtplib`, `sendgrid`, `boto3` (SES) across `server.py` and `src/` returns nothing.
- The only outreach "sending" artifact is `POST /api/outreach/forward`, which appends a stamp `{brand, at, request_id}` to an **in-memory** per-job `forwarded[]` list (server.py:694-700). It never opens an SMTP connection, never hits an email API, and never writes a mail queue. Data is lost on process restart.
- Therefore: **no email has ever been transmitted.** The exposure was the ability to generate a plausibly-worded, PR-address-targeted *draft* whose claims were not grounded in real detections. Escalation to "emails sent" is not required, but the draft-generation path itself is the incident and is the subject of A3.

### A2 — Trace of the data's actual origin

**Call chain (file by file):**
1. `src/layer1/logo_detector.py` — YOLO-World (`yolov8s-worldv2.pt`) runs real inference over sampled frames and returns real boxes + real model confidence for text queries (e.g. "brand logo", plus per-brand "<BRAND> logo" prompts).
2. `src/pipeline.py` + `src/layer2/brand_resolver.py` — logo detections are crop-OCR'd (PaddleOCR) and brand-resolved via `src/brand_catalog.py::match_brand()`, producing a brand timeline (`appearance_count`, per-frame `appearances`, confidence).
3. `server.py::_build_dashboard()` (server.py:174-230) — aggregates resolved brands into the dashboard `products` list, mapping real frame indexes to real `SCENE nnn` labels via `_scene_number_map()`/`fmt_scene()`.
4. `static/app.js::productsPanel()` — renders each product row with a **DRAFT EMAIL** button; `adsPanel()` renders "AD OPPORTUNITIES" cards with scene chips.
5. `server.py::outreach_generate()` (server.py:578+) — builds the email draft targeting `contact_for(brand)`'s real PR address (from the brand catalog, e.g. `pr@gucci.com`, `pr@rolex.com`).
6. `server.py::outreach_forward()` — records the in-memory "forwarded" stamp.

**Is an LLM involved? — NO, confirmed and ruled out explicitly.** There is no OpenAI/Anthropic/LLM API call anywhere in the pipeline or server (`grep -i "openai|anthropic|gpt|llm|langchain"` matches only the openai-whisper ASR library name). The brand names are **not** LLM-generated and **not** read from a hardcoded seed table.

**Root cause — unambiguous:**
The data path is a **real detector-driven pipeline**, but the pipeline had **no ground-truth-grounded validation of brand attribution** and the outreach template had **no integrity gate**:

- (a) **Brand attribution is the failure point.** YOLO-World was prompted with per-brand queries ("<BRAND> logo") and every brand-resolved detection box was reported as a product with real scene numbers. The Part B benchmark (below) now **quantifies** how unreliable this attribution is on a real held-out set: YOLO-World's localization recall is 8.2% and its **brand attribution accuracy is 0.0%** on matched boxes — i.e. the model fires brand classes (SUPREME, MASTERCARD, GUCCI, MICROSOFT … were its most frequent classes on the held-out images) at real scene numbers for logos that are actually other brands or not brands at all. A "Gucci / SCENE 003 / SCENE 013" row was a real frame-index of a **hallucinated** brand class.
- (b) **The outreach template asserted visibility it could not substantiate.** The original `outreach_generate` (pre-fix, in git HEAD) required only a target address and emitted: *"YOUR {product} APPEARED ON SCREEN ACROSS {count_txt} ({scene_txt}) — CLEARLY VISIBLE, UNPROMPTED, AND IN CONTEXT WITH THE CONTENT"* — and for a brand with **0 appearances** it substituted `"SEVERAL SCENES"`, fabricating the on-screen claim outright. With real PR addresses auto-filled for every catalog brand, this converted model hallucination (or zero detections) into a professional-looking outreach draft.
- (c) **No external existence check existed.** Any catalog brand name — real or not — could be presented as detected; there was no independent "does this brand actually exist" filter.

**Scene-number check (prompt question):** scene numbers were NOT synthetic placeholders — they were real `frame_index → SCENE nnn` mappings of real video frames. The fabrication was not the numbers; it was that **hallucinated brand classes were attached to real frames** and that **zero-appearance brands could be described as appearing "across SEVERAL SCENES"**. This is consistent with (and now proven by) the Part B benchmark.

**Does this code path connect to the real Layer 1/2 modules? — YES.** It is not a disconnected mockup feature. It is wired directly to the real YOLO-World detector, real PaddleOCR, real brand timeline, and real knowledge-graph recommender. The fix is therefore *not* "reconnect to the real pipeline" (already connected) — it is **"stop presenting unreliable brand attribution and zero-detection suggestions as verified on-screen appearances."**

**Was IntegrityAuditor ever run against this path?** No — no IntegrityAuditor component exists in the repository (searched `src/`, `scripts/`, `server.py`, `tests/`). The absence is the process gap: nothing audited that model brand classes were being asserted as verified brand appearances. That missing audit function is what Parts B (benchmark) and B2b (external validation) now supply.

### A3 — Fix applied

The incident is treated as **real output on a real detection path** (not a mockup), so the only acceptable fix is integrity-gating, not "tuning." Applied changes:

1. **Feature flag OFF (containment)** — `ui.outreach_enabled: false` + server 403 gate + frontend hides/disables all outreach controls. Outreach is *structurally* disabled in the shipped default.
2. **Table integrity labeling** — the brand-collaboration table and ad-opportunity cards are now explicitly labeled **UNVERIFIED DETECTION CANDIDATES — NOT CONFIRMED APPEARANCES** (server `_build_dashboard` returns `products_status`/`products_status_reason`; `static/app.js` renders the banner above the table and in the ads panel). Because Part B proves brand attribution is not yet production-validated, the incident's symptom (a plausible-looking brand table) can no longer be shown as verified output even in a demo.
3. **Appearance-count guard** — `outreach_generate` now returns 400 with an explicit message whenever `appearance_count <= 0` (server.py:653-667): *"SUGGESTED (KNOWLEDGE-GRAPH) BRANDS ARE NOT DETECTION OUTPUT."* Knowledge-graph SUGGESTED brands (which have `appearances: 0` by construction in `src/layer3/recommender.py:144`) can never receive an appearance-based draft.
4. **B2b external existence validation (new)** — before any draft is generated, the brand must pass `LogoDevClient().validate_brand()` with status `verified` (server.py:671-686). `unavailable` (no key / API down) is **fail-closed**: it is rejected, never promoted to verified. Per-brand results are cached for the process lifetime.
5. **Tests added** — `tests/test_logodev_guard.py` proves: (i) no key → `unavailable`, never `verified`; (ii) a detected brand with `count>0` but an unverified logo.dev status is rejected 400; (iii) the same brand with `verified` status passes; (iv) zero-appearance brands are blocked even when externally verified.

**When can outreach be re-enabled?** Only after (a) a logo.dev secret key is provisioned (so the `verified` gate can actually pass), and (b) the production default is switched to a detector whose brand attribution is benchmarked, per Part B. Until then it stays off — the safe default.

---

## PART B — REBUILD LOGO DETECTION WITH REAL BENCHMARK METHODOLOGY

### B1 — Real benchmark data

- **Dataset:** LogoDet-3K test split (HuggingFace `axonstan/LogoDet-3K`, parquet shard `test-00000-of-00002.parquet`). Downloaded and verified: 15,866 rows, 256 unique images, real ground-truth bounding boxes (xyxy) and per-object brand labels. Class index → name map (3,000 entries) parsed from the dataset README into `/tmp/adscene_bench/logodet3k_classes.json`.
- **Held-out split:** this IS the official test split — no image here was used for any fine-tuning (we did no fine-tuning; see OPEN items).
- **Catalog coverage in the evaluated subset:** after mapping LogoDet-3K labels to the project's shared catalog (`src/brand_catalog.py`), **101 images contain 159 ground-truth boxes across 14 catalog brands**: NEW BALANCE 34, APPLE 26, ZARA 17, COCA-COLA 14, SAMSUNG 11, LULULEMON 10, ASICS 10, UNDER ARMOUR 7, ADIDAS 6, ROLEX 6, BMW 6, PEPSI 5, LEVI'S 4, GOOGLE 3. (The other 155 shard images contain only non-catalog brands and are excluded by construction — the benchmark measures the catalog-retrieval task the product actually performs.)
- Dataset provenance recorded in every result file (`benchmark/results/benchmark_*.json`).

### B2 — Detection paradigms implemented and run

Each paradigm runs the **same held-out images** with the **same metrics** (`scripts/logo_bench_metrics.py`: IoU@0.5 matching, P/R/F1 at the backend's own operating threshold, mAP@0.5 via confidence-swept VOC precision envelope, plus a separate **brand-attribution accuracy** so localization and attribution are never conflated).

| Paradigm | Backend (script) | Status |
|---|---|---|
| Zero-shot open-vocabulary (current production approach) | `YOLOWorldBackend` — wraps the real production `YOLOWorldLogoDetector`, prompted with 34 "<BRAND> logo" + generic queries | **RUN, real** |
| Region-proposal + embedding retrieval (two-stage: selective-search proposals → CLIP text/image similarity vs per-brand prompts) | `RegionProposalCLIPBackend` — OpenCV selective search + OpenCLIP ViT-B-32 | **RUN, real** |
| Classical keypoint baseline (deterministic, no learning) | `SIFTLogoMatcherBackend` — SIFT + FLANN Lowe-ratio + RANSAC homography against a real reference logo bank | **RUN, real** |
| Fine-tuned end-to-end detector (YOLO/RF-DETR on the dataset train split) | — | **OPEN** (see below) |
| SSD / DeepLogo-style baseline (FlickrLogos-27) | — | **OPEN** (see below) |

**B2.2 / SSD / DeepLogo — explicit OPEN items, not simulated.** The prompt asks for a fine-tuned end-to-end detector and a DeepLogo-style SSD baseline (TensorFlow 1.x-era). **Neither was run.** Reasons stated plainly rather than papered over:
- Fine-tuning YOLO on the LogoDet-3K *train* split requires downloading the multi-GB training set and several hours of GPU training; this environment has no GPU training path and the dataset's train split was not retrieved. Running a randomly-initialized or untrained detector would produce meaningless "results"; fabricating trained-model numbers is prohibited. **OPEN.**
- The DeepLogo original codebase is TensorFlow 1.x and does not run on TF2 (noted in its own README); standing up an SSD reimplementation + FlickrLogos-27 download + training is a multi-day effort beyond this remediation's scope. **OPEN — documented, not simulated.**
- The benchmark therefore delivers three real paradigms (one classical + one region-proposal + the production zero-shot), which is the minimum comparison required to stop trusting a single model; the two OPEN rows are listed as future work, and the numbers below already falsify the "trust YOLO-World" assumption.

**B2a — logo.dev reference bank.** `src/logodev.py::LogoDevClient` implements search (`GET api.logo.dev/search`, Bearer secret), logo fetch (`GET img.logo.dev/<domain>?token=<publishable>`), and `build_reference_bank()`. **Security:** both keys come exclusively from the environment / `.env` (`_load_env_file()`); `.env`, `.env.*`, and `LOGO_DEV_*` are in `.gitignore`; `.env.example` ships placeholders only; the client never logs or persists keys. **Status: OPEN — no `LOGO_DEV_SECRET_KEY` or publishable token is present in this environment, so no live logo.dev call has been made and no logo.dev-sourced reference bank has been built.** The client + harness are complete and fail closed without keys.
- Instead, the **SIFT reference bank was built from real data**: 11 clean, real logo images for 4 catalog brands (APPLE ×3, SAMSUNG ×2, MICROSOFT ×3, SONY ×3) sourced from the real public `varun1212/logo-detection-dataset`, downloaded to `benchmark/reference_logos/`. This is real reference imagery — the retrieval-bank methodology is exercised for real, just not via logo.dev yet.

**B2b — real brand-name validation as a fabrication safeguard.** Wired into the outreach data path (see Part A3): only `verified` brands (external logo.dev existence) may receive a draft; `unavailable`/`unverified` are rejected. The report's honesty caveat from the prompt is respected: a logo.dev hit proves the brand exists, **not** that it appeared on screen — it is a validity filter, and the appearance gate is the separate `count>0` guard.

### B3 — Real quantitative comparison (all numbers from actual runs)

Evaluated on the same 101 images / 159 GT boxes. Metrics from `benchmark/results/benchmark_all_backends.json` (consolidated from real executed runs on this machine, all three backends on the same images; see also the per-run files `benchmark_20260805_152607.json`, `benchmark_20260805_221717.json`, `benchmark_20260805_222231.json`). No number is estimated.

| Backend | Precision | Recall | F1 | mAP@0.5 | TP | FP | FN | brand_accuracy (attribution of matched boxes) | boxes/img | latency ms/img |
|---|---|---|---|---|---|---|---|---|---|---|
| yolo_world (production) | 0.046 | 0.082 | 0.059 | 0.020 | 13 | 270 | 146 | **0.000** | 2.8 | 37 |
| region_proposal_clip | 0.027 | 0.069 | 0.039 | 0.023 | 11 | 399 | 148 | **0.182** | 4.1 | 2,910 |
| sift_match (reference bank) | 0.035 | 0.063 | 0.045 | 0.002 | 10 | 273 | 149 | **0.000** | 2.8 | 119 |

**Reading these honestly:** all three current paradigms perform poorly on the catalog-retrieval task. The two zero-shot/classical approaches find a few logo regions (recall 6–8%) but their **brand attribution is exactly 0% correct** on the boxes that do match ground truth. The region-proposal + CLIP retrieval stage — the first implementation that performs attribution as a *separate* step rather than trusting a detector class name — is the only one to attribute any box correctly (brand_accuracy 0.182), and it is ~80× slower than YOLO-World. This is the quantitative confirmation of the Part A root cause: **the current production path cannot be trusted to assert "BRAND X appeared on screen," and it is not in production-signed-off state.** The benchmark's job is precisely to make this visible instead of assumed. (Reported latency is warm-model; the first run in the original result file includes model load: yolo 67 ms, sift 139 ms.)

**Production selection:** **No production default is selected yet.** The correct action per the data is to keep logo-driven brand attribution **non-authoritative** (presented as detection candidates, not verified appearances) until a fine-tuned detector (OPEN) or a validated attribution stage beats this bar. This is a decision the benchmark forced, not one assumed. YOLO-World stays as the *candidate-generation* backend; the alternative (region-proposal + CLIP retrieval) is kept runnable via the existing config-driven backend pattern for swap-testing. Because attribution is not yet validated, the dashboard labels all products/ads as **UNVERIFIED DETECTION CANDIDATES** (server `products_status` field + prominent UI banner in `static/app.js`), so the incident's table can never be mistaken for confirmed appearances — even in a demo.

**B4 — Corrected positive control.** Superseded and closed by the real held-out result above. The production backend's real recall on the held-out test set is 8.2% (13/159 GT boxes) with 0% brand attribution — a proper quantitative result, replacing the earlier single-example fire/no-fire anecdote that was never completed. Stated explicitly: **no single-example positive control was ever completed in prior rounds; this benchmark supersedes and closes that open item.**

---

## PART C — REAL AUDIO BRAND-MENTION DETECTION

### C1 — Real brand-name source
`src/layer1/audio.py::detect_brand_mentions()` no longer scans against COCO object-class names. It now scans the ASR transcript against **`src/brand_catalog.py`** — the same single source of truth that drives Part B's logo-detection text queries and the knowledge graph (`find_brand_mentions()`). One shared brand list, config/data-driven (34 brands in the catalog, all data-driven via `BRAND_CATALOG`), no hardcoded inline list in the detection function. `detect_brand_mentions()` itself is deprecated (not called by the pipeline); the live path is `pipeline.py` → `find_brand_mentions()`.

### C2 — Real multilingual matching + Samsung-transcript verification
- `normalize_text()` is Unicode-aware: it preserves Devanagari letters (`\u0900–\u097f`) and combining marks, so transliterated aliases survive normalization (e.g. `सैमसंग` → matches SAMSUNG).
- The catalog carries 84 aliases including 31 Devanagari spellings (e.g. `सैमसंग`/`सैमसन` → SAMSUNG, `गूगल` → GOOGLE, `एडिडास` → ADIDAS).
- **Fuzzy/phonetic matching is opt-in and OFF by default** (`config/config.yaml`: `mention_fuzzy: false`, `mention_max_distance: 1`) — exact word-boundary matching (any script) is the zero-false-positive primary path; the bounded-Levenshtein supplement (`_fuzzy_token_match`) requires ≥4-char tokens, leading-code-point agreement, and is gated so it can never silently inflate speech evidence.
- **Concrete verification:** the Samsung reference video's transcript is not stored on disk, so the check was run on representative real-language Devanagari/Hinglish strings. `find_brand_mentions("इस वीडियो में मैं सैमसंग के फोन की समीक्षा कर रहा हूँ")` returns SAMSUNG (position 18, snippet with `सैमसंग`); the older COCO-class check would have returned nothing. **The report is explicit: the actual stored transcript is not available on disk, so the verification used representative real-language strings, not the literal transcript file.** The matcher is exercised and auditable; the specific transcript file is an OPEN artifact.
- 7 new multilingual tests cover Latin + Devanagari exact matches, fuzzy opt-in/opt-out behavior, and no-false-positive guards. Full suite: **72 passed** (was 69; +3 new B2b tests).

### C3 — Acoustic brand detection (documented future work, NOT built)
Sound-logo / jingle detection (Intel chime, Netflix "ta-dum") via audio-embedding matching against a reference sound-logo bank is a legitimate Phase 2+ direction, structurally analogous to the visual reference-bank retrieval in B2.3. **Explicitly scoped as future work — not built in this remediation** (C1/C2 were the active fabrication path and are fixed; C3 is not).

### C4 — Verify the fix doesn't move the fabrication
- `speech_mention` evidence is now produced only by real catalog-brand matches (`find_brand_mentions`), not the COCO-class check. 
- It is not inflated: fuzzy matching is OFF by default, so only exact word-boundary alias hits contribute; strength values come from the matched mentions' confidence aggregation in the pipeline, not a static number.
- Representative check: a transcript containing `सैमसंग` produces SAMSUNG speech evidence; a transcript with no catalog brand produces zero speech evidence. Auditable per mention via `{brand, position, snippet}`.

---

## SELF-AUDIT (all three parts)

0. **Independent re-verification pass (this document was re-audited end-to-end).** The benchmark numbers above were not taken on faith from the first run: the LogoDet-3K test shard was re-downloaded (313 MB parquet from HuggingFace `axonstan/LogoDet-3K`), the class-index map was re-extracted from the dataset README, and all three backends were re-run on this machine. The reproduced dataset subset is byte-for-byte the same (101 images / 159 GT boxes / 14 brands / identical per-brand counts), and YOLO-World and SIFT reproduced the originally-reported P/R/F1/mAP exactly (0.046/0.082/0.059/0.020 and 0.035/0.063/0.045/0.002). The previously-pending `region_proposal_clip` run was completed for real (0.027/0.069/0.039/0.023, brand_accuracy 0.182) and is now in the table. Result artifacts: `benchmark/results/benchmark_all_backends.json` + the three per-run files.
1. **No LLM-invented numbers.** All Part B metrics come from executed runs on real parquet data (result files on disk, reproduced in the audit pass). The two un-runnable paradigms are explicitly OPEN — no simulated rows.
2. **No hardcoded brand list dressed as config-driven.** The shared brand catalog is the single data source (`src/brand_catalog.py`), consumed by visual, audio, knowledge-graph, and outreach paths.
3. **Fuzzy-matching thresholds are not arbitrarily picked.** `max_distance:1` + ≥4-char + first-codepoint agreement guards are tested in `tests/test_brand_catalog.py` against real Devanagari/Hinglish strings.
4. **No new instance of the failure class.** The new logo.dev gate is fail-closed (no key ⇒ `unavailable` ⇒ rejection, never promotion). The benchmark's brand_accuracy metric makes attribution visible rather than assumed. The Part C path cannot inflate (fuzzy off by default). The dashboard now labels every product/ad row as an **UNVERIFIED DETECTION CANDIDATE**, so the Part A table cannot be mistaken for confirmed appearances.
5. **What remains genuinely OPEN (not silently incomplete):**
   - Fine-tuned end-to-end detector (LogoDet-3K train split + GPU) — not run.
   - SSD/DeepLogo-style baseline (FlickrLogos-27) — not run.
   - Live logo.dev calls + logo.dev-sourced reference bank — blocked on provisioning `LOGO_DEV_SECRET_KEY`/publishable token; client + wiring complete and fail-closed.
   - The literal Samsung video transcript file for C2 — not stored on disk; verification used representative real-language strings.
   - Outreach re-enablement — intentionally blocked until a benchmarked attribution path + logo.dev key exist.

**Sign-off gate for re-enabling outreach:** (1) logo.dev key provisioned and `validate_brand` returning `verified` for catalog brands; (2) a production logo backend whose brand attribution is benchmarked >0% on the held-out set; (3) review of this report. Until then `ui.outreach_enabled: false` is the shipped default.
