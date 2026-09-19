import logging, sys
logging.basicConfig(level=logging.WARNING)
logging.getLogger("src.layer2.brand_resolver").setLevel(logging.INFO)
logging.getLogger("src.pipeline").setLevel(logging.INFO)

from src.pipeline import Phase1Pipeline

VID = "/var/folders/dj/8w9p_2v54hzdjz60kb2ylq280000gn/T/adscene_uploads/03X3FAI6-AWWSI_vidssave.com Last Chance to Buy a Smartphone _ 360P.mp4"

p = Phase1Pipeline("config/config.yaml")
ld = p.logo_detector
ocr = p.ocr
ri = p.logo_retrieval
print("device:", p.device)
print("logo_retrieval empty:", ri.is_empty if ri else "n/a")
print("retrieval min_sim:", p.cfg["layer1"]["logo_retrieval"].get("min_similarity"),
      "min_margin:", p.cfg["layer1"]["logo_retrieval"].get("min_margin"))

import cv2
cap = cv2.VideoCapture(VID)
vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
step = max(1, int(nframes / 40))
frames = []
pos = 0
while pos < nframes and len(frames) < 40:
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    ok, fr = cap.read()
    if not ok:
        break
    frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    pos += step
cap.release()
print("loaded frames:", len(frames), "fps:", vfps, "step:", step)

from src.layer2.brand_resolver import BrandResolver
r = BrandResolver(
    ocr_extractor=ocr,
    class_confidence=p.cfg["layer1"]["logo_detection"].get("class_confidence", 0.40),
    crop_scale=p.cfg["layer1"]["logo_detection"].get("crop_scale", 2.0),
    retrieval_index=ri,
    retrieval_min_similarity=p.cfg["layer1"]["logo_retrieval"].get("min_similarity", 0.22),
    retrieval_min_margin=p.cfg["layer1"]["logo_retrieval"].get("min_margin", 0.10),
    screen_content_filter=True,
    class_require_corroboration=p.cfg["layer1"]["logo_detection"].get("class_require_corroboration", True),
    max_logo_area_fraction=p.cfg["layer1"]["logo_detection"].get("max_logo_area_fraction", 0.50),
    superset_margin_ratio=p.cfg["layer1"]["logo_detection"].get("superset_margin_ratio", 0.45),
)

dets_batch = ld.detect_batch(frames)
print("\n=== DETECTION + RESOLUTION PER FRAME ===")
import numpy as np
for idx, dets in enumerate(dets_batch):
    if not dets:
        continue
    for det in dets:
        cls = det.get("class_name")
        conf = det.get("confidence")
        bbox = det.get("bbox")
        res = r._resolve_detection(det, frames[idx])
        brand = res.get("brand")
        src_ = res.get("resolution_source")
        ocrtxt = res.get("ocr_text","")
        top3 = res.get("retrieval_top3", [])
        extra = ""
        if res.get("screen_content"): extra = " [SCREEN_CONTENT]"
        if res.get("editorial_box"): extra = " [EDITORIAL]"
        if res.get("class_unconfirmed"): extra = " [CLASS_UNCONFIRMED]"
        print(f"frame{idx:02d} class={cls!r:18} conf={conf:.3f} -> brand={brand} src={src_} "
              f"ocr={ocrtxt!r} retr={[(b,round(s,3)) for b,s in top3]}{extra}")
