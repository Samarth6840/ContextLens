# Logo identification product evaluation

This is the product-facing quality gate for logo recognition and unknown-brand
lookup. The goal is to measure proposal recall, brand accuracy, grounding, false
positives, latency, and provider usage separately. A model answer without a
matching source is not a verified brand.

## Existing data status

`realvideo_v1` has a frozen video-level val/test split, with 24 hero-logo frames
and 135 logo-free frames in its held-out videos. One extracted black tail frame
has no label and is excluded. It currently has no `logo-without-hero` examples,
only two held-out source videos, and no dedicated icon-only/wordmark strata, so
it is a useful smoke/acceptance set, not a production-accuracy claim.

The reference bank has 543 images across 16 folders. Visual audit found
systematic class contamination: Apple cereal packaging appears in `APPLE/`
(including `APPLE-006.png`, a Chex cereal package), the `SAMSUNG/` folder
contains many "Galaxy Counters" food packages (including `SAMSUNG-006.png`),
and the `GOOGLE/` folder includes unrelated TwistedPixel and Microsoft Studios
graphics. The original files are preserved for manual repair, but both logo
retrieval and product-index matching are disabled by default until a clean bank
is rebuilt and passes held-out evaluation.

Existing Phase 1.8 result (`benchmark/results/realvideo_v1_1p8_arms.json`) is a
warning about the gap: detector/resolver top-1 recall was 6/24 with 0.844 false
brand attributions per logo-free frame; full-frame OCR+retrieval was 6/24 with
0.119 false brands per free frame. These results are historical pipeline
measurements, not a provider benchmark, and must not be advertised as product
accuracy.

The first live Gemini pilot (`benchmark/results/open_set_provider_gemini_pilot.json`)
attempted all 14 crops: 4 requests completed, 10 failed from a mix of HTTP 503
overload and HTTP 429 quota. Across all 14 assigned samples, top-1 was 2/14
(14.3%), top-5 was 3/14 (21.4%), and only 1/14 (7.1%) returned any source URL.
Among the four successful responses alone, top-1 was 2/4; that conditional
number is not a release metric because it excludes provider failures. This
pilot rules Gemini out as the sole online identification provider in the
current account/configuration.

## Provider pilot

Build the deterministic pilot manifest from one largest human-designated hero
crop per held-out brand:

```bash
python3 scripts/build_logo_product_eval_manifest.py
```

The committed manifest, `serpapi_google_lens_pilot.jsonl`, records only file
references and gold boxes; it does not copy source footage. It contains 14
brands from the current held-out labels. Run a no-upload credential/budget
check first:

```bash
python3 scripts/benchmark_open_set_providers.py
```

To actually benchmark, configure provider credentials in `.env` and explicitly
allow the labeled logo crops to be uploaded:

```bash
python3 scripts/benchmark_open_set_providers.py --live --max-samples 14
```

The runner uploads only the human-labeled hero crop, not a full frame and not
the expected brand label. It reports top-1/top-5 title-match accuracy, URL/source
return rate, latency, errors, and per-sample evidence. Run each provider against
the same manifest. The first results are a small pilot; broaden the human gold
set before using the numbers as a release gate.

## Provider plan

| Role | Provider | Product assessment |
|---|---|---|
| Managed logo + web evidence | Google Cloud Vision [`LOGO_DETECTION`](https://docs.cloud.google.com/vision/docs/detecting-logos) + [`WEB_DETECTION`](https://cloud.google.com/vision/docs/detecting-web) | First-party API; this adapter now requests both features on a crop. Google currently lists Logo Detection at $1.50/1,000 units and Web Detection at $3.50/1,000 units after each feature's monthly free tier ([pricing](https://cloud.google.com/vision/pricing)). The current project key returned 403 and must be rotated/reconfigured before live use. |
| Reverse-image fallback | SerpApi Google Lens | Implemented using its [temporary image-upload flow](https://serpapi.com/image-api) (up to 500 KB, image ID expires after 10 minutes) and cited visual-match URLs. SerpApi currently advertises 250 free searches/month and a $25/month 1,000-search starter plan ([pricing](https://serpapi.com/pricing)). Enable only after checking its current privacy/retention settings and terms for the product's market. |
| Supplemental grounded reasoning | Gemini with Google Search | Keep as an optional fallback. In the live known-logo probe in this environment it returned a brand name without source URLs, so the app must continue to treat that as unresolved. |
| Retired | Bing Visual Search API | Removed from auto-routing. Microsoft's Bing Search APIs were retired August 11, 2025 ([retirement notice](https://learn.microsoft.com/en-us/lifecycle/announcements/bing-search-api-retirement)); Azure Foundry's newer text-grounding product is not a reverse-image-search replacement. |

## Release gates to add as the set grows

Track detection separately from identification. Add independently reviewed
videos from different creators, devices, lighting, languages, and logo scales;
include icon-only marks, wordmarks, product/packaging sub-brands, background
logos, logo-free scenes, and logos with no designated hero. Split by whole
video/source creator to avoid adjacent-frame leakage. Use a double-reviewed
test split, freeze it before tuning, report per-category recall and false
positive rate, and require citations or direct provider evidence plus the
existing independent brand-validation gate before marking an unknown brand
verified.
