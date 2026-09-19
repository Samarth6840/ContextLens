# Plan: Tiered Product→Brand Resolution (replacing hardcoded product aliases)

## Objective
Resolve PRODUCT names (e.g. "Mac Mini", "Galaxy S24", "AirPods Pro") to their parent
BRAND when text does not directly name the brand, using mechanisms that generalize to
products we never manually entered — NOT a hardcoded product→brand table.

This is a **redesign** that supersedes the previously-scoped Defect-1.2 fix. The D1.2
approach (adding `mac mini`/`macbook`/... to `APPLE.aliases`) IS the rejected hardcoded
pattern and must be reverted.

## Core principle / anti-pattern
- NO product→brand pairs are hardcoded anywhere (code, catalog aliases, config).
- Tier 2 cache and Tier 4 learned table populated ONLY by live queries/observations.
- Every resolution carries provenance: `resolution_tier` (1|2|3|4) + `source`,
  consistent with the existing `resolution_source` / `resolution_quality` / `sources`
  conventions enforced by `tests/test_instrumentation.py`.

---

## Architecture overview

```
                        OCR/speech text
                              |
                              v
                 Tier 1: match_brand (unchanged)
                 └─ hit? -> resolve (tier=1)  [existing behavior]
                 └─ miss?
                              v
              [product-span + plausibility gate]  (new: ProductNameExtractor)
                              |
                  ┌───────────┼──────────────┐
                  v           v              v
             Tier 2        Tier 3         Tier 4
          Wikidata        Qwen-VL       learned
          (cache)      (structured,     (brand_memory
                        low trust,      co-occurrence)
                        corroborated)         |
                  └───────────┼──────────────┘
                              v
                  resolve with resolution_tier + source
```

New module: **`src/layer2/product_resolver.py`** — the single choke point consulted by
the OCR path, the speech path, and evidence aggregation after Tier 1 misses.

---

## Files to change (new + modified)

### 1. Migrate ALL product/device aliases out of the hardcoded catalog → `src/brand_catalog.py`
Remove EVERY product/device-name alias and move resolution responsibility to the tiered
resolver. Confirmed scope: **ALL product-family aliases** (user-approved), not just the
D1.2 addition.

Remove from `aliases` (these are product/device names, not brand-name spellings):
- **APPLE**: `आईफोन`, `mac mini`, `macbook`, `macbook pro`, `macbook air`, `ipad`,
  `iphone`, `airpods`, `vision pro` → keep `['apple', 'एप्पल']`
- **SAMSUNG**: `galaxy`, `z fold`, `galaxy z fold` → keep the Devanagari + `samsung`
- **QUALCOMM**: `snapdragon` → keep `['qualcomm', 'क्वालकॉम']`
- **GOOGLE**: `pixel`, `पिक्सेल`, `पिक्सल`, `पिक्सेर` → keep `['google', 'गूगल']`
- **XIAOMI**: `redmi` → keep `['xiaomi', 'श्याओमी', 'ज़ियाओमी']`
- **AMAZON**: `prime` → keep `['amazon', 'अमेज़न', 'अमेजन']`
- **LG**: `lg oled` → keep `['lg', 'lg electronics', 'lge']`

Keep as aliases (these are BRAND nicknames / logo-descriptors, NOT product models):
`swoosh` (NIKE), `three stripes` (ADIDAS), `cat logo` (PUMA), `coke` (COCA-COLA),
`lulu` (LULULEMON). These denote the brand mark itself and are not product-name spans.

Also remove the 3 tests I added to `tests/test_brand_catalog.py`
(`test_match_mac_mini_to_apple`, `test_find_mentions_apple_product_line`,
`test_apple_product_alias_negative`) — they asserted the hardcoded behavior.

**Regression risk & fallback:** removing `galaxy`/`pixel`/`snapdragon`/`redmi` from
aliases means existing tests (`test_z_fold_family`, `test_find_mentions_pixel`,
`test_find_mentions_phone_brands`) will now FAIL because tier 1 no longer matches them.
These tests must be updated to assert the tiered path (they become Tier-2/Tier-4
resolutions). To avoid a hard regression when the external path is down, the resolver
must FAIL-CLOSED to "no resolution" (never a fabricated brand), and Tier-4 learnings
must populate fast enough to cover these previously-working cases. This is an accepted
behavioral shift per the user's decision — verify each affected test's new expectation
explicitly in review.

### 2. NEW `src/layer2/product_resolver.py`
Responsibilities:
- `ProductNameExtractor`: deterministic NER for "plausible product name" spans.
  - Title-case / capitalized multi-token spans (e.g. "Mac Mini", "Samsung Galaxy S24").
  - Boost span plausibility when a spec value is near (`extract_specs` signals) OR a
    price token is near (new `extract_price`).
  - Returns candidate `{span, position, confidence, reasons[]}`.
- `ProductBrandResolver` (the tier orchestrator):
  - `resolve(text, frame=None, scene_brands=None, video_id=None, budget)`
    returns `{brand, resolution_tier, product_span, source, confidence, meta}`.
  - Tier 1 passthrough: if `match_brand(text)` hits, return `(tier=1, brand)`.
  - Tier 2: `WikidataProductLookup.lookup(product_span)` (cache-first).
  - Tier 3: `QwenProductLookup.resolve(product_span, frame)` — corroborated.
  - Tier 4: `product_resolution_memory.lookup(product_span)` — promoted associations.
- `WikidataProductLookup`:
  - SPARQL vs `https://query.wikidata.org/sparql`: query label→P176 (manufacturer) /
    P1716 (brand); request JSON.
  - Local cache: `config/product_resolutions.json` (or SQLite table) keyed by normalized
    span → `{brand, source:"wikidata", timestamp, wikidata_uri, manufacturer}`.
  - Rate-limit (min interval between external hits), fail-closed on network error.
  - Map manufacturer string → canonical catalog brand via lower-cased normalization;
    if not a catalog brand, still return manufacturer name but mark
    `in_catalog=false` (outreach/OOM decision — do NOT fabricate a catalog brand).
- `ProductResolutionMemory` (Tier 4) — see #5.

### 3. NEW price signal → `src/layer1/spec_extractor.py`
Add `extract_price(text) -> Optional[{value, currency, raw}]` using a small regex for
`₹`/`$`/`€`/number+`k`/`lakh`/`cr` etc. (Hinglish videos use ₹ and "lakh"/"cr").
Expose via `extract_specs` too (`{"field":"price",...}`) or as a separate function —
used by the plausibility gate. Minimal, fails-closed.

### 4. Tier 3 → `src/layer1/qwen3vl.py`
Add a narrow structured method (keep separate from `analyze_frame`):
- `Qwen3VLAbstract.resolve_product_manufacturer(product_span, frame=None) -> dict`
  returning `{"manufacturer": str|None, "confidence": float, "fallback": bool}`.
- Implement in `Qwen3VL32B`:
  - Inject a `system` message in the chat template (the current template has no system
    role — this is the minimal change to add one).
  - Strict JSON wrapper: prompt asks for ONLY `{"manufacturer": "..."}` or
    `{"manufacturer": null}`; parse + validate the single field.
  - `max_new_tokens` small (e.g. 40). Log separately; increment a counter + token/latency.
- Also handle the fail-closed fallback when model weights aren't wired
  (current `fallback` returns empty tokens → return `manufacturer=None`).

### 5. Tier 4 → `src/layer2/brand_memory.py`
Extend `BrandMemoryBank` with a co-occurrence table (new, parallel to entities):
- `_product_cooccurrence: Dict[str, Dict[str, Dict]]` =
  `{normalized_product: {brand: {videos: set[str], observations: int, first_seen, ...}}}`
- `record_product_cooccurrence(product_span, brand, video_id)` — called whenever a
  Tier-1 brand resolves in a scene that also contains a plausible product span.
  (Observation rule: extract product spans from OCR text in the SAME frame/scene where a
  brand was resolved via Tier 1/2. Insert into the set of distinct videos.)
- `promoted_product_brands(product_span) -> Optional[str]` — returns the brand only when
  the `(product, brand)` pair has been observed in **>= 3 distinct videos**
  (config `min_distinct_videos: 3`), plus provenance
  `{brand, source:"learned_cooccurrence", tier:4, videos_observed:[...], confidence}`.
- Serialization: extend `to_dict`/`from_dict`/`save`/`load` to include the new table.
- **Fix persistence gap**: invoke `self.save_brand_memory()` after recording in the
  pipeline so the learned table is durable across runs (currently never called).
- Provenance must remain **additive / non-influencing** of matching/ranking — the Tier-4
  promotion is gated ONLY on the empirical distinct-video count, + tests lock this in.

### 6. Integration → `src/pipeline.py` + `src/layer2/brand_resolver.py`
- Construct `ProductBrandResolver` in the pipeline (or resolver) with config + the
  brand memory + the Qwen getter.
- `_resolve_detection`: after `match_brand` misses on crop OCR / superset OCR, call
  `resolver.resolve(joined_ocr, frame, scene_brands, video_id, budget)`. If resolved,
  stamp `brand`, `resolution_source = resolver.source` (e.g. `"wikidata"`,
  `"qwen_vlm"`, `"learned_cooccurrence"`), `resolution_quality` by tier
  (2 → 0.80, 3 → 0.55 corroborated / 0.30 uncorroborated, 4 → 0.85), and
  **`resolution_tier`**. Tier-3-uncorroborated results are recorded as
  low-confidence candidates, not accepted.
- `_aggregate_evidence`: for full-frame OCR text that `match_brand` misses, also route
  through the resolver so product-resolved OCR can contribute to `ocr_hit`
  (only for accepted tiers).
- Speech path: in `process_video`, augment `find_brand_mentions` results by running the
  resolver over the transcript for product spans Tier 1 missed → new
  `product_resolutions` list, emitted in the result dict.
- Emit new provenance in result: `layer1.product_resolutions`,
  `layer2c.product_resolution_tiers` (counter dict), and per-tier cost/latency.

### 7. Config → `config/config.yaml`
Add a `layer1.product_resolution` block:
```yaml
product_resolution:
  enabled: True
  wikidata_enabled: True
  wikidata_endpoint: "https://query.wikidata.org/sparql"
  wikidata_min_interval_sec: 1.0      # rate limit
  cache_path: "config/product_resolutions.json"
  qwen_enabled: True
  qwen_max_calls_per_video: 10        # Tier-3 budget
  qwen_prompt: "Return ONLY JSON {\"manufacturer\": name-or-null} for this product."
  min_distinct_videos_for_promotion: 3
  spec_signal_boost: True
```

### 8. Tests → `tests/test_product_resolver.py` + updated existing tests
Mock external APIs (no network in CI) + follow existing conventions:
- **Update existing tests in `tests/test_brand_catalog.py`** that relied on removed
  product aliases: `test_match_brand_z_fold_family`, `test_find_mentions_z_fold`,
  `test_find_mentions_hindi_speaking_pixel`, `test_find_mentions_phone_brands`
  (galaxy/pixel/redmi). These are moved to assert the tiered path:
  e.g. `resolve("Z Fold 8 Ultra")` → SAMSUNG via tier 2 (mocked Wikidata) or via a
  pre-seeded tier-4 learned association.
- **Tier 1 passthrough**: `match_brand("Samsung logo")` still returns SAMSUNG, tier 1.
- **Tier 2 (mocked SPARQL)**: monkeypatch the Wikidata client to return
  P176=Apple for "Mac Mini"; assert `resolve("Mac Mini")` → APPLE, tier 2,
  source="wikidata"; and that the local cache is written + read back without re-query
  (cache-hit path).
- **Held-out test**: "iPad Pro" (an Apple product NOT in the reverted aliases) resolves
  to APPLE via mocked Tier 2 — proving generalization, no developer-provided answer.
- **Tier 3**: mocked Qwen returns `{"manufacturer":"Sony"}` for "WH-1000XM5"; assert
  accepted ONLY when corroborated (a low-conf logo in same scene), else logged as
  low-confidence candidate, brand stays None.
- **Tier 4**: record `("Mac Mini", "APPLE")` across 3 distinct videos → promoted;
  across 2 videos (same video repeated frames) → NOT promoted;
  provenance has `source:"learned_cooccurrence"` + `videos_observed` length 3.
- **Migration coverage**: seeded Tier-4 associations for `galaxy`/`pixel`/`redmi`/
  `z fold` must resolve the previously-alias-matching cases (so the migrated tests
  pass once learned, and fail-closed before learning — document both states).
- **Price/spec gate**: "mini" alone or generic lowercase text does NOT trigger Tier 2/3.
- **Provenance non-influence**: add test that `sources`/tier metadata never changes
  ranking (extend `test_instrumentation.py` pattern).
- **Persistence**: test that Tier-4 co-occurrence round-trips through
  `to_dict`/`from_dict`/`save`/`load`, and that the pipeline calls save.

---

## Verification
1. `python3 -m pytest tests/ -q -x` — all existing (note: count drops by 3 from the
   reverted tests, then rises with new ones) + new `test_product_resolver.py` pass;
   PROVENANCE thread (`test_instrumentation.py`) still green.
2. Held-out test proves Tier 2 generalization for a never-cataloged product.
3. Offline sanity: run the resolver on the archived job's "MAC MINI" OCR string and
   assert it resolves to APPLE via Tier 2 (mocked), confirming the original D1.2 defect
   is fixed in the GENERALIZING way — not via hardcoding.
4. Confirm tests exercise the Tier-4 cross-video (not cross-frame) guardrail.

## Explicit guardrails (re-verified during review)
- No manual `product→brand` entry in any file.
- Tier 3 never resolves at full confidence without corroboration.
- Tier 4 promotion requires >= 3 DISTINCT videos.
- Everything logged with `resolution_tier` + `source` for the measurement/paper.
- Provenance is additive and never influences ranking (existing convention preserved).
