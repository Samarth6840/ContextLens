"""
Layer 2c — Brand Resolution + Temporal Memory (lightweight).

Two jobs, both fixing the "text logo but not name" problem:

1. BrandResolver — converts generic logo detections ('text logo', 'brand logo')
   into actual brand names. Strategy, in priority order:
     a. Otherwise crop the logo bounding box and OCR the crop; the recognized
        text is matched against the brand catalog aliases.
     b. Else CLIP retrieval against the per-brand reference bank (icon-only /
        stylized marks with no readable text).
     c. Else a brand-named class label above the confidence gate, but ONLY if
        an independent signal corroborates it (fail-closed — see
        class_require_corroboration).
     d. Otherwise the detection stays unresolved and is NOT reported as a brand
        product (avoids "TEXT LOGO" showing up as a brand name).

    Two precision gates live here, same trust-class as the CLIP margin guard:
      * OCR screen-content gate — a phone video shows the DEVICE with its own
        on-screen UI (app drawer, Messages/2FA screen, camera watermark); YOLO
        flags those regions as logos and OCR reads the app names ("Play Store /
        Gaming Hub / Instagram") -> a "brand" (META/GOOGLE/SUPREME) that is
        screen content, not a brand appearance. Suppressed fail-closed.
      * class-label corroboration gate — YOLO-World's zero-shot brand class
        labels ('GUCCI logo', 'SUPREME logo') are the fabrication path: a tiny
        spurious box at confidence just above the gate, no readable text, no
        retrieval match, still resolves to a brand -> a fabricated DIRECT
        recommendation. Asserted only when an independent signal corroborates.

2. TemporalBrandSmoother — a static on-screen overlay (wordmark held for many
   frames) gets region-proposal confidence that flaps around the cut-off, so
   identical pixels resolve a brand one frame and 'text logo'/nothing the next.
   A brand that resolves solidly and persistently on a stable spatial region
   back-fills the unresolved neighbour boxes; never weakens a resolved brand.

3. build_brand_timeline / brand_evidence_from_timeline — temporal memory /
   cross-scene reasoning (prompt §5). Every resolved brand entity accumulates a
   running memory of appearances (frame index, timestamp, modality source). A
   speech mention of a brand also visually established is flagged cross_scene.

The crop-OCR step reuses the already-loaded PaddleOCR extractor, so it adds no
new models — only a bounded amount of per-logo inference.
"""

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.brand_catalog import match_brand

logger = logging.getLogger(__name__)

# Generic labels that carry no brand identity and require OCR resolution.
GENERIC_LOGO_LABELS = {
    "brand logo", "company logo", "text logo", "product logo",
    "label", "wordmark", "logo", "brand",
}

# Screen/app-UI content markers used by the OCR-context brand gate.
#
# A creator phone video shows the DEVICE (a Samsung foldable) with its own on
# screen text — the app drawer ("Play Store / Gaming Hub / Instagram"), a
# Messages/2FA screen, or a camera watermark ("Creative Studio") — and YOLO-World
# flags those screen regions as logos while OCR reads the app names off them.
# That yields a brand attribution ("META"/"GOOGLE"/"SUPREME") that is screen
# CONTENT, not a brand appearance. This is the same trust class as a spurious
# CLIP retrieval: a confident-looking wrong answer is worse than no answer, so
# we suppress it (fail-closed) rather than report it as a detected brand.
#
# Tokens are deliberately app-store / app-drawer / camera-UI lexicon — written on
# a phone's home screen, never in a legitimate brand wordmark title card. They
# are checked ONLY against a crop whose OCR already produced a brand match, so a
# generic word like "camera"/"phone"/"store" alone is never enough to suppress;
# an app-drawer read (e.g. "Play Store" + "Instagram" + "Gaming Hub") is.
SCREEN_UI_TOKENS = frozenset([
    "play store", "app store", "gaming hub", "instalive", "instagram",
    "facebook", "youtube", "whatsapp", "snapchat", "tiktok", "threads",
    "voice recorder", "voice record", "messages", "q messages", "stories",
    "screenshot", "screen recorded", "creative studio", "play services",
    "settings", "gallery", "recent", "home screen", "app drawer", "notifications",
])

# Reserved canonical name for a logo that could not be resolved to a real brand
# ('UNKNOWN BRAND' grouping). It is NOT a catalog brand: it must never be emitted
# by brand_evidence_from_timeline() nor recommended by Layer 3. Unresolved logo
# detections are grouped by spatial region (see group_unknown_logo_regions) so a
# persistent-but-unnameable mark reads as ONE unknown region, not N noisy boxes.
UNKNOWN_BRAND = "__UNKNOWN__"


class BrandResolver:
    """Resolve logo detections to canonical brand names."""

    # A per-brand class label ('SAMSUNG logo', 'Nike logo') from the detector is
    # only trusted as brand evidence above this confidence. Below it, the label
    # is treated as unconfirmed and we fall through to crop-OCR. This suppresses
    # YOLO-World's low-confidence spurious brand hits (e.g. 'SUPREME logo' on a
    # Samsung/Sony/Apple logo) that otherwise resolve to the wrong brand.
    DEFAULT_CLASS_CONFIDENCE = 0.40

    # Crop upscale factor applied before OCR so small/low-res logo text is more
    # likely to be read (the vanilla crop on a phone video can be tiny).
    DEFAULT_CROP_SCALE = 2.0

    def __init__(
        self,
        ocr_extractor=None,
        class_confidence=DEFAULT_CLASS_CONFIDENCE,
        crop_scale=DEFAULT_CROP_SCALE,
        retrieval_index=None,
        retrieval_min_similarity: float = 0.22,
        retrieval_min_margin: float = 0.10,
        screen_content_filter: bool = True,
        class_require_corroboration: bool = True,
        max_logo_area_fraction: float = 0.50,
        superset_margin_ratio: float = 0.45,
        product_resolver=None,
    ):
        self.ocr = ocr_extractor
        self.class_confidence = float(class_confidence)
        self.crop_scale = float(crop_scale)
        # Multiline-card OCR superset (see _crop_superset). The primary (tight)
        # crop carries the same small margin as before; only a re-OCR attempt on
        # a padded superset uses this wider margin.
        self.superset_margin_ratio = float(superset_margin_ratio)
        # Optional CLIP logo-retrieval index (Phase 1/2). When provided it is
        # used as the icon-only fallback: if neither the class label nor OCR
        # names a brand, the crop is matched by CLIP retrieval against a
        # per-brand reference bank. Retrieval is the PRIMARY classifier for
        # icon-only / stylized marks where there is no readable text.
        self.retrieval_index = retrieval_index
        self.retrieval_min_similarity = float(retrieval_min_similarity)
        # Minimum top-1 vs top-2 (distinct-brand) similarity gap to treat the
        # top retrieval as a genuine match rather than noise. An absolute floor
        # alone is insufficient: on a clean wordmark, random brand hits cluster
        # around 0.7-0.9 (no gap between "confident" and noise), so a blanket
        # floor raise lets them through AND would kill real matches that sit in
        # the same band. The margin is the reliable discriminator — a genuine
        # match (e.g. SAMSUNG 0.9 / others 0.3) has a wide gap, whereas noise
        # (0.9 vs 0.89 vs 0.88) has ~none.
        self.retrieval_min_margin = float(retrieval_min_margin)
        # OCR-context screen-content gate. When an OCR-resolved crop also reads
        # phone-screen/app-drawer/camera-UI text (see SCREEN_UI_TOKENS), the
        # "brand" is on-screen content, not a brand appearance — suppress it so
        # it never becomes brand evidence / a direct recommendation.
        self.screen_content_filter = bool(screen_content_filter)
        # Class-label corroboration gate. YOLO-World's zero-shot brand class
        # labels (e.g. 'GUCCI logo', 'SUPREME logo') are the fabrication-prone
        # path: a tiny spurious box at confidence just above the gate, with no
        # readable text (OCR empty) and no retrieval match, still resolved to a
        # brand -> a fabricated DIRECT recommendation. When enabled, a
        # class-label brand is asserted ONLY if an independent signal
        # corroborates it (a clean CLIP-retrieval match naming the same brand —
        # OCR already ran and would have caught text), else the detection stays
        # unresolved (fail-closed, no fabrication).
        self.class_require_corroboration = bool(class_require_corroboration)
        # Full-frame editorial-box guard. A genuine brand wordmark/logo occupies
        # a compact region of the frame. A detection box covering more than this
        # fraction of frame area (a title card, a full-screen editorial overlay,
        # a screen recording of a whole page) is NOT a brand appearance — OCR on
        # the box reads the surrounding scene text (e.g. a news headline whose
        # body mentions "Google"), and resolving that to a brand yields a false
        # DIRECT recommendation (observed: a near-full-frame 0.58-area box whose
        # title text resolved to GOOGLE). When the box is oversized it is
        # suppressed fail-closed (brand stays None) and tagged for audit.
        self.max_logo_area_fraction = float(max_logo_area_fraction)
        # Optional tiered product→brand resolver (Layer 2b). When OCR reads a
        # product name but no catalog brand (e.g. "Mac Mini"), this fallback
        # resolves the product to its parent brand via Wikidata / learned
        # memory (Tier 2/4). None disables the fallback (fail-closed). The
        # resolver returns provenance (resolution_tier/source) so every such
        # resolution is auditable and never silently asserted.
        self.product_resolver = product_resolver

    @staticmethod
    def _crop(frame: np.ndarray, bbox) -> Optional[np.ndarray]:
        """Crop a bounding box from a frame with a small margin."""
        if frame is None or frame.size == 0 or bbox is None:
            return None
        try:
            x1, y1, x2, y2 = (int(v) for v in bbox)
        except (TypeError, ValueError):
            return None
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        pad_x = int((x2 - x1) * 0.1) + 1
        pad_y = int((y2 - y1) * 0.1) + 1
        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        return frame[y1:y2, x1:x2]

    def _crop_superset(self, frame: np.ndarray, bbox) -> Optional[np.ndarray]:
        """Crop a padded superset of a proposal box, biased upward.

        Target: multi-line brand/product cards (e.g. a Snapdragon 8 Elite Gen 5
        chip card) where the detector's tight proposal wraps only one line (the
        lower 'Gen 5' sub-label) and OCR reads that fragment alone -> no brand.
        The missing wordmark line usually sits ABOVE the proposed box, so the
        margin is biased to extend further upward than down/sideways.

        The expansion is bounded: each side grows by at most
        `superset_margin_ratio * bbox_dimension`, and the resulting crop is
        clamped to the frame. This can never turn a compact box into an
        editorial-scale crop — the frame area gate (max_logo_area_fraction) on
        the ORIGINAL bbox still governs whether the box is a real wordmark.
        """
        if frame is None or frame.size == 0 or bbox is None:
            return None
        try:
            x1, y1, x2, y2 = (int(v) for v in bbox)
        except (TypeError, ValueError):
            return None
        h, w = frame.shape[:2]
        bh = max(1, y2 - y1)
        bw = max(1, x2 - x1)
        r = self.superset_margin_ratio
        # Bias heavily toward extending the top (where the split wordmark sits).
        top_pad = int(bh * r * 1.5)
        bottom_pad = int(bh * r * 0.6)
        side_pad = int(bw * r * 0.8)
        nx1 = max(0, x1 - side_pad)
        ny1 = max(0, y1 - top_pad)
        nx2 = min(w, x2 + side_pad)
        ny2 = min(h, y2 + bottom_pad)
        if nx2 - nx1 < 4 or ny2 - ny1 < 4:
            return None
        return frame[ny1:ny2, nx1:nx2]

    @staticmethod
    def _upscale(crop: np.ndarray, scale: float) -> np.ndarray:
        """Upscale a crop (bicubic) to aid small-text OCR.

        Only meaningful (and cheap) for genuinely small crops. Large crops —
        e.g. a detection box covering most of a frame — are already high-res
        and upscaling them just pushes PaddleOCR past its side limit and burns
        inference time. So the scale is applied only when the crop's max
        dimension is under a small threshold (SMALL_MAX), and the result is
        additionally capped so it never exceeds LARGE_MAX.
        """
        if scale <= 1.0:
            return crop
        import cv2

        SMALL_MAX = 192
        LARGE_MAX = 1200
        h, w = crop.shape[:2]
        if max(h, w) > SMALL_MAX:
            return crop
        nh, nw = int(h * scale), int(w * scale)
        if max(nh, nw) > LARGE_MAX:
            ratio = LARGE_MAX / max(nh, nw)
            nh, nw = int(nh * ratio), int(nw * ratio)
        return cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_CUBIC)

    def _ocr_crop_texts(self, crop: np.ndarray) -> List[str]:
        """OCR a crop and return the recognized text list.

        Tries the upscaled crop first (best for small logo text), and falls back
        to the original if upscaling produced nothing. Dedupes while preserving
        order.
        """
        if self.ocr is None or crop is None:
            return []
        candidates = []
        scaled = self._upscale(crop, self.crop_scale)
        candidates.append(scaled)
        if scaled is not crop:
            candidates.append(crop)
        seen = set()
        texts: List[str] = []
        for img in candidates:
            try:
                results = self.ocr.extract_text(img)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Crop-OCR failed for logo box: %s", exc)
                continue
            for r in results:
                t = r.get("text", "")
                if t and t not in seen:
                    seen.add(t)
                    texts.append(t)
            if texts:
                break
        return texts

    def _resolve_detection(self, det: dict, frame: np.ndarray) -> dict:
        """Resolve a single logo detection to a canonical brand name."""
        out = dict(det)
        out["brand"] = None
        class_name = str(det.get("class_name") or "")
        confidence = float(det.get("confidence", 0.0))
        class_brand = match_brand(class_name)

        # 0. Full-frame editorial-box gate (fail-closed). A detection box that
        #    covers most of the frame is a title card / full-screen editorial /
        #    screen recording, not a compact brand wordmark. OCR over such a box
        #    reads the surrounding scene text, and resolving it to a brand yields
        #    a false DIRECT recommendation (observed: a ~0.58-frame-area box whose
        #    headline text resolved to GOOGLE). Suppress oversized boxes entirely
        #    and tag them for audit so they never become brand evidence.
        bbox = det.get("bbox")
        frame_h, frame_w = frame.shape[:2]
        frac = 0.0
        if bbox and frame_h > 0 and frame_w > 0:
            try:
                bx1, by1, bx2, by2 = (int(v) for v in bbox)
            except (TypeError, ValueError):
                bx1 = by1 = bx2 = by2 = 0
            bh = max(0, min(frame_h, by2) - max(0, by1))
            bw = max(0, min(frame_w, bx2) - max(0, bx1))
            frac = (bh * bw) / (frame_h * frame_w)
        if frac > self.max_logo_area_fraction:
            out["editorial_box"] = True
            out["editorial_box_reason"] = (
                f"bbox {round(frac * 100, 1)}% of frame (> "
                f"{self.max_logo_area_fraction * 100:.0f}%) — full-frame title/"
                f"editorial, not a compact wordmark"
            )
            out["ocr_text"] = ""
            logger.info(
                "Brand SUPPRESSED as full-frame editorial: class='%s' (conf=%.2f) "
                "bbox=%.1f%% of frame (>\u00a0%.0f%%) — not a brand wordmark",
                class_name, confidence, frac * 100, self.max_logo_area_fraction * 100,
            )
            return out

        # 1. Gather signals ONCE from the logo crop, reused by every path:
        #      * OCR  — reads the real on-screen wordmark (ground-truth-adjacent)
        #      * CLIP — image retrieval against the per-brand reference bank
        #      * class — YOLO-World's zero-shot brand guess (LEAST trustworthy:
        #        it confidently mislabels icon/stylized marks, e.g. "SUPREME logo"
        #        on a Samsung foldable at 0.45-0.60).
        #    Phase 3 visibility: every resolution records all three signals so a
        #    disagreement (the SUPREME-style failure mode) is never silent.
        crop = self._crop(frame, det.get("bbox"))
        ocr_brand, joined_ocr = None, ""
        retrieval_candidates: List[Tuple[str, float]] = []
        retrieval_brand, retrieval_sim = None, 0.0
        superset_crop = None
        if crop is not None:
            if self.ocr is not None:
                texts = self._ocr_crop_texts(crop)
                joined_ocr = " ".join(texts)
                if texts:
                    ocr_brand = match_brand(joined_ocr)
                if texts and not ocr_brand:
                    self._log_near_misses(joined_ocr, class_name, confidence)
                # Multiline-card remediation: a tight proposal box often wraps only
                # ONE line of a multi-line card (e.g. the lower 'Gen 5' fragment of a
                # 'Snapdragon / 8 Elite / Gen 5' chip card), so OCR reads a fragment
                # that matches no catalog brand. Retry OCR on a padded SUPERSET --
                # biased upward, where the missing wordmark line usually sits -- so
                # a full-multiline read ('Snapdragon ...') can still resolve the
                # brand. Only fires when the tight crop failed to name a brand.
                if not ocr_brand:
                    superset_crop = self._crop_superset(frame, det.get("bbox"))
                    if superset_crop is not None:
                        superset_texts = self._ocr_crop_texts(superset_crop)
                        superset_joined = " ".join(superset_texts)
                        superset_brand = (
                            match_brand(superset_joined) if superset_texts else None
                        )
                        # Product-name resolution also benefits from the superset
                        # read: a fragment ("Gen 5") resolves to nothing, but the
                        # full card ("Snapdragon 8 Elite Gen 5") can. When either
                        # the catalog or the product resolver finds a brand in the
                        # superset, adopt the superset text so the same (brand OR
                        # product) path resolves the real name.
                        superset_product = None
                        if (not superset_brand and superset_texts
                                and self.product_resolver is not None):
                            sup_res = self.product_resolver.resolve(superset_joined)
                            if sup_res:
                                superset_product = max(
                                    sup_res, key=lambda r: r.get("confidence", 0.0))
                        if superset_brand or superset_product:
                            logger.info(
                                "OCR superset extended the read: tight OCR=%r "
                                "-> superset OCR=%r (brand=%s product=%s)",
                                joined_ocr[:40], superset_joined[:40],
                                superset_brand, bool(superset_product),
                            )
                            joined_ocr = superset_joined
                            out["superset_ocr"] = True
            if self.retrieval_index is not None and not self.retrieval_index.is_empty:
                retrieval_candidates = self._retrieval_query(crop)
                if retrieval_candidates:
                    retrieval_brand, retrieval_sim = retrieval_candidates[0]
                    retrieval_candidates = retrieval_candidates[:3]

        # 2. Fusion priority (plan §11): OCR wins when text is present and names
        #    a brand; else CLIP retrieval (icon-only / corrects spurious class
        #    labels); else the class label above the confidence gate; else keep
        #    the detection unresolved.
        #
        #    OCR-context gate: a crop whose OCR names a brand but ALSO reads
        #    phone-screen/app-drawer UI text (e.g. "Play Store / Gaming Hub /
        #    Instagram") is the device's screen CONTENT — not a brand appearance.
        #    We suppress it fail-closed (brand stays None) so it never becomes
        #    brand evidence / a direct recommendation, and tag it for audit.
        ocr_screen_content = (
            ocr_brand and self._is_screen_content_text(joined_ocr)
        )
        if ocr_brand and not ocr_screen_content:
            out["brand"] = ocr_brand
            out["class_name"] = ocr_brand
            out["class_confirmed"] = False
            out["resolution_source"] = "ocr"
            # Resolution-quality (Layer 2b): a brand read directly off the
            # on-screen wordmark is the strongest, ground-truth-adjacent signal.
            # This keys `logo_detected` evidence (and the per-chip display) on
            # resolution trust, not the zero-shot detector's raw box confidence.
            out["resolution_quality"] = 0.90
            out["ocr_text"] = joined_ocr[:40]
            out["retrieval_top3"] = retrieval_candidates
            if class_brand and class_brand != ocr_brand:
                out["resolved_vs_class"] = {
                    "class_brand": class_brand,
                    "class_confidence": round(confidence, 3),
                }
            logger.info(
                "Brand resolved via OCR: class='%s' (conf=%.2f) ocr_text=%r -> %s",
                class_name, confidence, joined_ocr[:40], ocr_brand,
            )
            return out

        if ocr_screen_content:
            # On-screen app/UI text masquerading as a brand — suppress. Keep the
            # raw class + reason for audit; never promote to a brand appearance.
            out["screen_content"] = True
            out["screen_content_reason"] = "ocr_text_reads_phone_ui"
            out["ocr_text"] = joined_ocr[:40]
            out["retrieval_top3"] = retrieval_candidates
            if class_brand and class_brand != ocr_brand:
                out["resolved_vs_class"] = {
                    "class_brand": class_brand,
                    "class_confidence": round(confidence, 3),
                }
            logger.info(
                "OCR brand SUPPRESSED as screen content: class='%s' (conf=%.2f) "
                "ocr_text=%r (matched %s — phone UI, not a brand appearance)",
                class_name, confidence, joined_ocr[:40], ocr_brand,
            )
            return out

        # Tiered product→brand fallback (Layer 2b). OCR read a PRODUCT name but
        # no catalog brand (e.g. "Mac Mini", "AirPods Pro"). Resolve it to its
        # parent brand via Wikidata / learned memory (Tier 2/4). Provenance
        # (resolution_tier/resolution_source) is recorded so the resolution is
        # auditable and never silently asserted. Fails closed when unresolved.
        if self.product_resolver is not None and joined_ocr and not ocr_brand:
            pres = self.product_resolver.resolve(joined_ocr)
            if pres:
                best = max(pres, key=lambda r: r.get("confidence", 0.0))
                resolved_brand = best.get("brand")
                if resolved_brand:
                    out["brand"] = resolved_brand
                    out["class_name"] = resolved_brand
                    out["class_confirmed"] = False
                    out["resolution_source"] = ("product_" + str(
                        best.get("resolution_tier", "tier")))
                    out["resolution_quality"] = best.get(
                        "resolution_quality", 0.5)
                    out["resolution_tier"] = best.get("resolution_tier")
                    out["product_span"] = best.get("product_span")
                    out["ocr_text"] = joined_ocr[:40]
                    out["retrieval_top3"] = retrieval_candidates
                    if class_brand and class_brand != resolved_brand:
                        out["resolved_vs_class"] = {
                            "class_brand": class_brand,
                            "class_confidence": round(confidence, 3),
                        }
                    logger.info(
                        "Brand resolved via product resolver: class='%s' "
                        "(conf=%.2f) ocr_text=%r -> %s (tier=%s)",
                        class_name, confidence, joined_ocr[:40], resolved_brand,
                        best.get("resolution_tier"),
                    )
                    return out

        if retrieval_brand:
            out["brand"] = retrieval_brand
            out["class_name"] = retrieval_brand
            out["class_confirmed"] = False
            out["resolution_source"] = "clip_retrieval"
            # Icon-only match via CLIP retrieval (with the margin guard) — strong
            # but below a direct wordmark read.
            out["resolution_quality"] = 0.70
            out["retrieval_top3"] = retrieval_candidates
            out["retrieval_similarity"] = retrieval_sim
            if class_brand and class_brand != retrieval_brand:
                out["resolved_vs_class"] = {
                    "class_brand": class_brand,
                    "class_confidence": round(confidence, 3),
                }
            logger.info(
                "Brand resolved via CLIP retrieval: class='%s' (conf=%.2f) "
                "crop->%s sim=%.3f candidates=%s",
                class_name, confidence, retrieval_brand, retrieval_sim,
                retrieval_candidates,
            )
            return out

        if class_brand and confidence >= self.class_confidence:
            # Class-label corroboration gate. Reaching here means neither OCR nor
            # CLIP retrieval named a brand (both earlier branches would have
            # returned), so the only signal is YOLO-World's zero-shot class guess.
            # Those guesses are the fabrication path (tiny spurious 'GUCCI logo' /
            # 'REEBOK logo' / 'SUPREME logo' boxes at confidence just above the
            # gate, no readable text) -> a fabricated DIRECT recommendation. With
            # corroboration required, an uncorroborated class label is NOT
            # asserted as a brand; it is tagged for audit and left unresolved.
            if self.class_require_corroboration:
                out["class_unconfirmed"] = True
                out["class_unconfirmed_reason"] = "no_ocr_no_retrieval_corroboration"
                if retrieval_candidates:
                    out["retrieval_top3"] = retrieval_candidates
                logger.info(
                    "Class-label brand UNCONFIRMED (corroboration required, "
                    "gate=%.2f): class='%s' conf=%.2f -> not asserted as a brand",
                    self.class_confidence, class_name, confidence,
                )
                return out
            out["brand"] = class_brand
            out["class_name"] = class_brand
            out["class_confirmed"] = True
            # Zero-shot detector class only (no OCR, no retrieval) — weakest
            # resolution that still asserts a brand, hence lowest quality.
            out["resolution_quality"] = 0.50
            if retrieval_candidates:
                out["retrieval_top3"] = retrieval_candidates
            logger.debug(
                "Resolver: class '%s' -> %s (conf=%.2f, gate %.2f)",
                class_name, class_brand, confidence, self.class_confidence,
            )
            return out

        # 3. Unresolved — keep the raw label, brand stays None so it is
        #    excluded from brand products and recommendations.
        return out

    def _is_screen_content_text(self, joined_ocr: str) -> bool:
        """True if OCR text reads phone-screen/app-drawer/camera-UI content.

        Checks the lowercase normalized text for SCREEN_UI_TOKENS. Only invoked
        on a crop that already produced a brand match, as the contextual
        discriminator: a region whose crop OCR reads app-store/app-drawer UI is
        the device's screen content (a phone displaying apps), not a brand mark.
        """
        if not self.screen_content_filter or not joined_ocr:
            return False
        low = " " + joined_ocr.lower() + " "
        for tok in SCREEN_UI_TOKENS:
            if " " + tok + " " in low:
                return True
        return False

    def _retrieval_query(self, crop: np.ndarray) -> List[Tuple[str, float]]:
        """Top retrieval candidates for a crop, filtered by similarity+margin.

        Returns [(brand, similarity), ...] desc by similarity, truncated to top-k.
        A candidate is accepted ONLY when BOTH hold:
          1. absolute floor  — top-1 similarity >= retrieval_min_similarity
          2. margin          — top-1 minus top-2 (distinct-brand) similarity
                               >= retrieval_min_margin

        The margin is the primary noise guard: YOLO-World crops that aren't a
        real logo produce a stack of near-tied brand similarities (e.g. ADIDAS
        0.9 / NEW BALANCE 0.89 / ASICS 0.88) — noise dressed as confidence that
        passes any absolute floor. A genuine logo match is unambiguous (large
        gap to every other brand). Every rejected candidate is logged with the
        reason so the thresholds can be tuned on real data rather than guessed.
        """
        if self.retrieval_index is None or self.retrieval_index.is_empty:
            return []
        candidates = self.retrieval_index.query(crop)
        candidates = [
            (b, float(s)) for (b, s) in candidates
            if float(s) >= self.retrieval_min_similarity
        ]
        if not candidates:
            return []
        top1_brand, top1_sim = candidates[0]
        # Distinct-brand top-2 (skip any duplicate brand, e.g. two aliases of the
        # same canonical brand ranked 1st and 2nd).
        top2_sim: Optional[float] = None
        seen_first = False
        for b, s in candidates[1:]:
            if b == top1_brand and not seen_first:
                seen_first = True
                continue
            top2_sim = s
            break
        # margin = top-1 minus top-2 (distinct-brand) gap. None when only ONE
        # brand cleared the absolute floor — i.e. every other brand in the whole
        # reference bank fell below min_similarity. That lone-clearance is itself
        # a strong genuine-match signal (a noise crop like the live ADIDAS 0.9/
        # NEW BALANCE 0.89/ASICS 0.88 stack always drags 2+ brands over the
        # floor), so a sole candidate is trusted rather than spuriously rejected.
        margin = (
            top1_sim - top2_sim
            if top2_sim is not None
            else None
        )
        floor_ok = top1_sim >= self.retrieval_min_similarity
        margin_ok = (
            margin is None  # single unique brand above floor == genuine match
            or margin >= self.retrieval_min_margin
        )
        top3 = [(b, round(s, 3)) for b, s in candidates[:3]]
        if floor_ok and margin_ok:
            return candidates
        # Rejected — log why so the next failure mode is caught pre-ship.
        reason = "below_similarity_floor" if not floor_ok else "small_margin"
        logger.info(
            "Retrieval REJECTED (reason=%s, min_sim=%.2f, min_margin=%.2f): "
            "top1=%s sim=%.3f top2_sim=%s margin=%s top3=%s",
            reason, self.retrieval_min_similarity, self.retrieval_min_margin,
            top1_brand, top1_sim,
            ("%.3f" % top2_sim) if top2_sim is not None else "None",
            ("%.3f" % margin) if margin is not None else "None",
            top3,
        )
        return []

    def _log_near_misses(self, joined: str, class_name: str, confidence: float) -> None:
        """Log OCR/class text that nearly matches a catalog brand (diagnostics)."""
        from src.brand_catalog import normalize_text, _fuzzy_token_match

        norm = normalize_text(joined)
        near = []
        for token in set(norm.split()):
            if not token:
                continue
            hit = _fuzzy_token_match(token, 1)
            if hit:
                near.append({"token": token, "brand": hit[0], "dist": hit[1]})
        if near:
            logger.info(
                "Resolver near-miss: class='%s' (conf=%.2f) ocr_text=%r nearly "
                "matches %s",
                class_name, confidence, joined[:48],
                [(n["brand"], n["dist"]) for n in near],
            )

    def resolve(
        self,
        logo_detections: List[List[dict]],
        frames: List[np.ndarray],
    ) -> List[List[dict]]:
        """Resolve all logo detections across frames.

        Args:
            logo_detections: per-frame lists of logo detections
            frames: full list of RGB frames (same indexing)

        Returns:
            New per-frame detection lists, each detection carrying a `brand`
            key (None when unresolved) and `class_name` set to the canonical
            brand when resolved.
        """
        resolved = []
        n_resolved = 0
        n_total = 0
        class_outcomes: Dict[str, int] = {}
        resolved_by_class: Dict[str, str] = {}
        for frame, frame_dets in zip(frames, logo_detections):
            out = []
            for det in frame_dets:
                n_total += 1
                r = self._resolve_detection(det, frame)
                cls = str(det.get("class_name") or "?")
                class_outcomes[cls] = class_outcomes.get(cls, 0) + 1
                if r.get("brand"):
                    n_resolved += 1
                    resolved_by_class.setdefault(cls, r["brand"])
                out.append(r)
            resolved.append(out)
        logger.info(
            "Brand resolution: %d/%d logo detections mapped to a brand (gate=%.2f)",
            n_resolved, n_total, self.class_confidence,
        )
        detail = "; ".join(
            f"{cls}x{n}->{resolved_by_class.get(cls, 'UNRESOLVED')}"
            for cls, n in sorted(class_outcomes.items())
        )
        if detail:
            logger.info("Brand resolution per input class: %s", detail)
        return resolved


class TemporalBrandSmoother:
    """Stabilize logo brand resolution across adjacent near-identical frames.

    Root cause this targets (Phase 1 instability): a creator's static on-screen
    overlay — e.g. a wordmark or chip card held for many consecutive frames —
    gets region-proposal confidence that flaps around the cut-off, so one frame
    resolves "SAMSUNG" (0.50) and the next sees the SAME pixels as a generic
    "text logo" (0.13) or nothing at all (zero boxes). The content never
    changes; only the proposal confidence flickers.

    Rather than re-resolve noisy single frames, we track logo boxes across
    adjacent frames by spatial overlap (label-agnostic, like the COCO
    TemporalObjectSmoother) and let a brand that resolves SOLIDLY and PERSISTENTLY
    on a stable region back-fill the gaps:

      * A logo box on frame t that overlaps (by IoU or center proximity) a box
        on a nearby frame t+k carrying a *resolved* brand inherits that brand,
        PROVIDED the inherited brand is consistent across >= `min_votes`
        resolved neighbours (no majority -> keep unresolved, no guessing).
      * A resolved brand is never overwritten by an equal-or-different one here;
        only boxes whose brand is None are candidates for back-fill, and the
        highest-confidence resolved candidate wins.
      * `smoothed_*` audit keys are added so the raw signal and the source of
        the propagation are never lost.

    Input/output are per-frame logo detection lists (the BrandResolver output,
    each detection carrying `brand`, `bbox`, `confidence`). Input is not
    mutated; detections are copied on edit.
    """

    def __init__(
        self,
        window: int = 2,
        min_iou: float = 0.3,
        min_votes: int = 2,
    ):
        self.window = int(window)
        self.min_iou = float(min_iou)
        self.min_votes = int(min_votes)

    def smooth(
        self,
        resolved_logos: Sequence[Sequence[dict]],
    ) -> Sequence[Sequence[dict]]:
        # First pass: give every unresolved logo box a brand, using resolved
        # neighbours in adjacent frames that overlap it.
        out: List[List[dict]] = [list(fd) for fd in resolved_logos]
        n = len(resolved_logos)
        for t, frame_dets in enumerate(resolved_logos):
            for i, det in enumerate(frame_dets):
                if det.get("brand"):
                    continue  # already resolved — never weaken it
                bbox = det.get("bbox")
                if not bbox:
                    continue
                candidates: Dict[str, List[float]] = {}
                for k in range(max(0, t - self.window), min(n, t + self.window + 1)):
                    if k == t:
                        continue
                    for cand in resolved_logos[k]:
                        cb = cand.get("bbox")
                        if not cb:
                            continue
                        overlap = _box_iou(bbox, cb)
                        if overlap < self.min_iou:
                            continue
                        brand = cand.get("brand")
                        if not brand:
                            continue
                        candidates.setdefault(brand, []).append(
                            float(cand.get("confidence", 0.0))
                        )
                if not candidates:
                    continue
                # Best brand = the one with >= min_votes supporting frames,
                # chosen by mean confidence among those.
                viable = {
                    b: confs for b, confs in candidates.items()
                    if len(confs) >= self.min_votes
                }
                if not viable:
                    continue
                best_brand = max(
                    viable, key=lambda b: sum(viable[b]) / len(viable[b])
                )
                best_confs = viable[best_brand]
                nv = dict(out[t][i])
                nv["brand"] = best_brand
                nv["class_name"] = best_brand
                nv["resolution_source"] = "temporal_smoothing"
                # Back-filled from a resolved, persistent neighbour — trust the
                # resolved source's quality, not the flickering proposal score.
                nv["resolution_quality"] = 0.70
                nv["smoothed_votes"] = len(best_confs)
                nv["smoothed_from"] = best_brand
                out[t][i] = nv
        return out


def build_brand_timeline(
    resolved_logos: List[List[dict]],
    brand_mentions: List[dict],
    video_fps: float = 0.0,
    transcript: Optional[str] = None,
    transcript_duration: Optional[float] = None,
) -> Dict[str, dict]:
    """Build a temporal memory of brand appearances (Layer 2c).

    Each brand accumulates a memory bank of appearances with modality source
    ('logo' from the visual track, 'speech' from ASR mentions). Cross-scene
    resolution: brands established both visually and verbally are flagged
    cross_scene=True, matching the prompt's "this phone at minute 9 → the
    Apple logo shown at minute 1" scenario.

    Returns a dict keyed by canonical brand:
        {
            "brand": str,
            "appearance_count": int,
            "first_seen": float|None, "last_seen": float|None,
            "modalities": [...], "cross_scene": bool,
            "appearances": [{frame_index, timestamp, modality, confidence,
                             resolution_source, resolution_quality}]
        }
    """
    timeline: Dict[str, dict] = {}

    def _entry(brand: str) -> dict:
        if brand not in timeline:
            timeline[brand] = {
                "appearances": [],
                "modalities": set(),
            }
        return timeline[brand]

    for idx, frame_dets in enumerate(resolved_logos):
        for det in frame_dets:
            brand = det.get("brand")
            if not brand:
                continue
            entry = _entry(brand)
            entry["appearances"].append({
                "frame_index": idx,
                "timestamp": round(idx / video_fps, 1) if video_fps else idx,
                "modality": "logo",
                "confidence": round(float(det.get("confidence", 0.0)), 3),
                "resolution_source": det.get("resolution_source"),
                "resolution_quality": det.get("resolution_quality"),
            })
            entry["modalities"].add("logo")

    for mention in brand_mentions:
        brand = mention.get("brand")
        if not brand:
            continue
        entry = _entry(brand)
        # Real STT segment timestamps win; the proportional transcript-length
        # estimate is the legacy fallback when the ASR backend gave no timing.
        start = mention.get("start_time")
        end = mention.get("end_time")
        if start is None and transcript and transcript_duration and len(transcript) > 0:
            start = transcript_duration * (
                mention.get("position", 0) / len(transcript)
            )
        entry["appearances"].append({
            "frame_index": None,
            "timestamp": round(float(start), 1) if start is not None else None,
            "start_time": start,
            "end_time": end,
            "modality": "speech",
            "confidence": 1.0,
        })
        entry["modalities"].add("speech")

    out: Dict[str, dict] = {}
    for brand, entry in timeline.items():
        apps = entry["appearances"]
        modalities = sorted(entry["modalities"])
        timestamps = [a["timestamp"] for a in apps if a.get("timestamp") is not None]
        out[brand] = {
            "brand": brand,
            "appearance_count": len(apps),
            "first_seen": timestamps[0] if timestamps else None,
            "last_seen": timestamps[-1] if timestamps else None,
            "modalities": modalities,
            "cross_scene": "logo" in modalities and "speech" in modalities,
            "appearances": apps,
        }
    return out


def group_unknown_logo_regions(
    resolved_logos: List[List[dict]],
    merge_iou: float = 0.3,
    max_frame_gap: int = 2,
) -> List[dict]:
    """Group unresolved logo detections into distinct 'UNKNOWN BRAND' regions.

    A logo that fails to resolve to a real brand (crop-OCR miss, no retrieval
    match, generic 'text logo') is dropped from the brand timeline. If the SAME
    physical mark persists across many frames we would normally show one noisy
    red box per frame with no name. This is the 'UNKNOWN BRAND merge step': it
    clusters unresolved detections by spatial overlap (IoU) within a small frame
    gap, so a persistent-but-unnameable mark collapses into a single unknown
    region with its full spatial extent and frame span.

    The region brand is `UNKNOWN_BRAND` (a reserved sentinel, never a catalog
    brand). Callers must keep it out of brand_evidence_from_timeline() and Layer 3.

    Args:
        resolved_logos: per-frame resolved logo detections (as produced by
            BrandResolver.resolve()).
        merge_iou: min IoU between a detection and an existing region to merge.
        max_frame_gap: allow a det to merge a region even if a couple of frames
            are skipped (region-proposal suppression flickers frame to frame).

    Returns:
        List of region dicts:
            {brand, bbox:[x1,y1,x2,y2] (union), frames:[frame_idx],
             appearance_count, first_frame, last_frame}
    """
    regions: List[dict] = []
    for idx, frame_dets in enumerate(resolved_logos):
        for det in frame_dets:
            if det.get("brand"):
                continue  # resolved brands are already attributed; not unknown
            bbox = det.get("bbox")
            if not bbox:
                continue
            bbox = tuple(float(v) for v in bbox)
            chosen = None
            for r in regions:
                if idx - r["last_frame"] > max_frame_gap:
                    continue
                if _box_iou(r["bbox"], bbox) >= merge_iou:
                    chosen = r
                    break
            if chosen is None:
                regions.append({
                    "brand": UNKNOWN_BRAND,
                    "bbox": list(bbox),
                    "frames": [idx],
                    "appearance_count": 1,
                    "first_frame": idx,
                    "last_frame": idx,
                })
            else:
                x1, y1, x2, y2 = chosen["bbox"]
                bx1, by1, bx2, by2 = bbox
                chosen["bbox"] = [min(x1, bx1), min(y1, by1),
                                  max(x2, bx2), max(y2, by2)]
                chosen["frames"].append(idx)
                chosen["appearance_count"] += 1
                chosen["last_frame"] = idx
    return regions


def brand_evidence_from_timeline(
    timeline: Dict[str, dict],
) -> Dict[str, float]:
    """Aggregate a per-brand evidence strength (0-1) from the timeline."""
    evidence: Dict[str, float] = {}
    for brand, entry in timeline.items():
        if brand == UNKNOWN_BRAND:
            continue  # reserved sentinel: never becomes evidence/recommendation
        logo_confs = [
            a["confidence"] for a in entry["appearances"]
            if a["modality"] == "logo" and a.get("confidence") is not None
        ]
        base = float(np.mean(logo_confs)) if logo_confs else 0.0
        if "speech" in entry["modalities"]:
            base = max(base, 0.6)
        evidence[brand] = round(min(1.0, base), 3)
    return evidence


def _box_center(bbox) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _box_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    union = a_area + b_area - inter
    if union <= 0:
        return 0.0
    return inter / union
