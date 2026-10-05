"""Diagnose job K02ESBCO-U1K27 Defects 1.1 / 1.2 / 1.3 on the real video.

Reproduces the pipeline's logo-detection + resolution path (same keyframe
sampling, same resolver components) so the before/after is a real comparison
with the archived job, not an approximation:

  * raw YOLO-World proposal inventory for the Apple-logo frame (D1.1):
    every box at conf=0.01 — is the Apple region proposed AT ALL?
  * full-frame OCR string for the Mac-Mini frames (D1.2): did OCR read
    "MAC MINI" and was nothing done with it?
  * per-scene UNKNOWN BRAND box counts + resolver_acceptance under the
    archived detection threshold (0.10) vs the detector default (0.30):
    how many of the Defect-1.3 boxes vanish, and what happens to the
    confirmed SAMSUNG true positive (raw box conf 0.101)?

Read-only: never writes to the store, never changes config. Run from repo root:
    python3 scripts/diag_job_defects.py
"""

import logging
logging.basicConfig(level=logging.WARNING)
logging.getLogger("src.layer2.brand_resolver").setLevel(logging.INFO)
logging.getLogger("src.pipeline").setLevel(logging.WARNING)

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from src.pipeline import (
    Phase1Pipeline,
    VideoProcessor,
    temporal_corroboration,
    resolver_acceptance_canary,
)
from src.layer1.logo_detector import create_logo_detector
from src.layer1.ocr import OCRExtractor
from src.layer2.brand_resolver import (
    BrandResolver,
    TemporalBrandSmoother,
    group_unknown_logo_regions,
)

VIDEO = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
    "DIAG_VIDEO",
    "/var/folders/dj/8w9p_2v54hzdjz60kb2ylq280000gn/T/adscene_uploads/"
    "K02ESBCO-U1K27_vidssave.com Last Chance to Buy a Smartphone _ 360P.mp4",
)
CFG = yaml.safe_load(open("config/config.yaml"))


def _full_queries(ld_cfg: dict) -> list:
    from src.brand_catalog import build_text_queries
    queries = list(ld_cfg.get("text_queries") or [])
    for q in build_text_queries():
        if q not in queries:
            queries.append(q)
    return queries


def main():
    print(f"video: {VIDEO}")
    ld_cfg = CFG["layer1"]["logo_detection"]
    brush_cfg = ld_cfg.get("temporal_smoothing", {})

    # ---- 1. Frames exactly as the archived job sampled them ----
    frames, video_fps, total_frames, stride = VideoProcessor.load_video(
        VIDEO,
        frame_rate=CFG["evaluation"]["video_frame_rate"],
        max_frames=CFG["evaluation"]["max_frames"],
    )
    print(f"sampled {len(frames)} frames (total {total_frames}, fps {video_fps}, stride {stride})")

    def src_frame(idx: int) -> int:
        return idx * stride

    # ---- 2. Detector + resolver components (CLIP retrieval omitted: OCR path
    # ----    alone drove the archived SAMSUNG resolution; retrieval is a pure
    # ----    additive signal and would only raise acceptance, not lower it.)
    detector = create_logo_detector(
        backend=ld_cfg.get("backend", "yolo_world"),
        model_name=ld_cfg.get("model", "yolov8s-worldv2.pt"),
        confidence_threshold=ld_cfg.get("confidence_threshold", 0.30),
        device="mps",
        text_queries=_full_queries(ld_cfg),
    )
    ocr = OCRExtractor(
        lang=CFG["layer1"]["ocr"]["lang"],
        use_angle_cls=CFG["layer1"]["ocr"]["use_angle_cls"],
        det_db_thresh=CFG["layer1"]["ocr"]["det_db_thresh"],
        rec_batch_num=CFG["layer1"]["ocr"].get("rec_batch_num", 6),
    )

    # ---- 3. D1.1 + D1.2: raw proposal inventory + full-frame OCR on the
    # ----    Apple-logo / Mac-Mini frames (scene 10 frame_index=11, 11→14, 12→15)
    target_idxs = (11, 14, 15, 41)  # 41 = frame where SAMSUNG visually resolved
    print("\n=== D1.1/D1.2 raw evidence for target frames ===")
    for idx in target_idxs:
        if idx >= len(frames):
            continue
        frame = frames[idx]
        ts = src_frame(idx) / video_fps
        detector.confidence_threshold = 0.01  # enumerate EVERY proposal
        raw = detector.detect(frame)
        detector.confidence_threshold = 0.10  # archived pipeline threshold
        gated = detector.detect(frame)
        ocr_texts = [t.get("text", "") for t in ocr.extract_text_batch([frame])[0]]
        print(f"\n  sampled#{idx} (src 0~{src_frame(idx)} ts~{ts:.1f}s)")
        print(f"    RAW proposals @conf=0.01: "
              f"{[(d['class_name'], round(d['confidence'], 3), [round(b) for b in d['bbox']]) for d in raw]}")
        print(f"    GATED @conf=0.10: {[(d['class_name'], round(d['confidence'], 3)) for d in gated]}")
        print(f"    FULL-FRAME OCR: {ocr_texts}")

    # ---- 4. D1.3: unresolved/UNKNOWN-box inventory under 0.10 vs 0.30 ----
    # (scene UI shows `det.get("brand") or "UNKNOWN BRAND"`, so unreolved == any
    #  det without a truthy brand — the exact quantity the dashboard surfaces.)
    print("\n=== D1.3 UNKNOWN-brand inventory: detection threshold 0.10 vs 0.30 ===")
    logo_indices = Phase1Pipeline._select_keyframes(
        frames, max_frames=ld_cfg.get("max_logo_candidates", 30)
    )
    print(f"logo keyframes: {len(logo_indices)}")

    def run_at(conf: float):
        detector.confidence_threshold = conf
        all_logo_detections = [[] for _ in frames]
        for idx, dets in zip(logo_indices, detector.detect_batch([frames[i] for i in logo_indices])):
            all_logo_detections[idx] = dets
        resolver = BrandResolver(
            ocr_extractor=ocr,
            class_confidence=ld_cfg.get("class_confidence", 0.40),
            crop_scale=ld_cfg.get("crop_scale", 2.0),
            retrieval_index=None,
            retrieval_min_similarity=CFG["layer1"]["logo_retrieval"].get("min_similarity", 0.22),
            retrieval_min_margin=CFG["layer1"]["logo_retrieval"].get("min_margin", 0.10),
            screen_content_filter=ld_cfg.get("screen_content_filter", {}).get("enabled", True),
            class_require_corroboration=ld_cfg.get("class_require_corroboration", True),
            max_logo_area_fraction=ld_cfg.get("max_logo_area_fraction", 0.50),
            superset_margin_ratio=ld_cfg.get("superset_margin_ratio", 0.45),
        )
        resolved = resolver.resolve(all_logo_detections, frames)
        resolved_pre = sum(1 for dets in resolved for det in dets if det.get("brand"))
        if brush_cfg.get("enabled", True):
            resolved = TemporalBrandSmoother(
                window=int(brush_cfg.get("window", 2)),
                min_iou=float(brush_cfg.get("min_iou", 0.3)),
                min_votes=int(brush_cfg.get("min_votes", 2)),
            ).smooth(resolved)
        resolved_post = sum(1 for dets in resolved for det in dets if det.get("brand"))
        unknown_count = 0
        unknown_by_frame = {}
        for i, dets in enumerate(resolved):
            n = sum(1 for d in dets if not d.get("brand"))
            if n:
                unknown_by_frame[str(i)] = n
            unknown_count += n
        regions = group_unknown_logo_regions(resolved)
        return {
            "conf": conf,
            "total_boxes": sum(len(d) for d in resolved),
            "unknown_boxes": unknown_count,
            "unknown_per_keyframe": unknown_by_frame,
            "n_regions": len(regions),
            "resolver_acceptance": resolver_acceptance_canary(resolved),
            "corroboration": temporal_corroboration(resolved_pre, resolved_post),
            "resolved_brands": sorted(
                {det.get("brand") for dets in resolved for det in dets if det.get("brand")}
            ),
        }

    a = run_at(0.10)
    b = run_at(0.30)
    for r in (a, b):
        print(f"\n  conf>= {r['conf']}: total_boxes={r['total_boxes']} "
              f"UNKNOWN_boxes={r['unknown_boxes']} regions={r['n_regions']} "
              f"acceptance={r['resolver_acceptance']} corroboration={r['corroboration']} "
              f"brands={r['resolved_brands']}")
        print(f"    unknown per keyframe idx: {r['unknown_per_keyframe']}")
    vanished = a["unknown_boxes"] - b["unknown_boxes"]
    print(f"\n  UNKNOWN boxes that vanish at 0.30: {vanished} "
          f"({round(100 * vanished / max(1, a['unknown_boxes']), 1)}%)")
    keep = set(a["resolved_brands"]) & set(b["resolved_brands"])
    print(f"  resolved brands kept across both: {sorted(keep)}")

    # ---- 5. Samsung TP at frame 41: raw box confidence under the hood ----
    detector.confidence_threshold = 0.01
    raw41 = detector.detect(frames[41])
    samsung_box = [(d['class_name'], d['confidence']) for d in raw41
                   if d['class_name'] in ("brand logo", "company logo", "text logo", "product logo")]
    print("\n=== D1.3 sanity: SAMSUNG true positive raw box @ frame 41 ===")
    print(f"  generic-brand raw boxes @conf=0.01: {[(c, round(v,3)) for c,v in samsung_box]}")
    print("  (archived SAMSUNG TP came from OCR 'Samsung Galaxy M07...' text on this frame,")
    print(f"   resolved via crop-OCR; its own YOLO conf is {samsung_box[0][1]:.3f} if present)" if samsung_box else
          "  no generic-brand box on frame 41; SAMSUNG was resolved purely from OCR text.")


if __name__ == "__main__":
    main()