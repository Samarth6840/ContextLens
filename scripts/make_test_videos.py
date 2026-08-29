"""Build real test videos from real LogoDet-3K test-split images.

Video A — genuinely 3 seconds: one real photo with a real logo, pan/zoom.
Video B — ~60s @ 20fps: 60 real photos (1s each), to reproduce the
          "SCENE 059 / 3 SEC" dashboard state caused by the duration unit bug.

Both videos carry a real audio track (generated tone) so the pipeline's
audio path is exercised; ASR/BEATs will still be skipped when the track is
silence-free noise below their thresholds. Real data only.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PARQUET = Path("/tmp/adscene_bench/test-00000-of-00002.parquet")
CLASSES = Path("/tmp/adscene_bench/logodet3k_classes.json")
OUT = Path("/tmp/adscene_bench/videos")
OUT.mkdir(parents=True, exist_ok=True)


def load_images(limit: int):
    import cv2
    import pandas as pd

    from src.brand_catalog import match_brand

    classes = json.loads(CLASSES.read_text())
    df = pd.read_parquet(PARQUET)

    def _path(v):
        return v.get("path") if isinstance(v, dict) else v

    df["_path"] = df["image_path"].apply(_path)
    images = []
    for path, group in df.groupby("_path"):
        raw = group.iloc[0]["image_path"]
        if isinstance(raw, dict) and "bytes" in raw:
            raw = raw["bytes"]
        img = None
        if isinstance(raw, (bytes, bytearray)):
            buf = np.frombuffer(bytes(raw), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            continue
        gt = []
        for _, row in group.iterrows():
            name = classes.get(str(int(row["company_name"])))
            brand = match_brand(name or "")
            if not brand:
                continue
            bbox = row["bbox"]
            if bbox is None or len(bbox) < 4:
                continue
            box = [float(v) for v in np.asarray(bbox).flatten()[:4]]
            gt.append({"bbox": box, "brand": brand})
        if not gt:
            continue
        images.append({"image": img, "gt": gt, "id": path})
        if len(images) >= limit:
            break
    return images


def scale_to(img, w=640, h=360):
    import cv2

    hh, ww = img.shape[:2]
    scale = max(w / ww, h / hh)
    resized = cv2.resize(img, (int(ww * scale), int(hh * scale)))
    top = (resized.shape[0] - h) // 2
    left = (resized.shape[1] - w) // 2
    return resized[top:top + h, left:left + w]


def zoom_pan(image, t_frac: float, w=640, h=360):
    """t_frac in [0,1): slow zoom-out + left-to-right pan over one image."""
    import cv2

    src_h, src_w = image.shape[:2]
    # crop window grows from 70% to 100% of the image while panning
    scale = 0.70 + 0.30 * t_frac
    cw, ch = int(src_w * scale), int(src_h * scale)
    max_x = src_w - cw
    x = int(max_x * t_frac)
    y = (src_h - ch) // 2
    crop = image[y:y + ch, x:x + cw]
    return scale_to(crop, w, h)


def write_video(path: Path, frames: list, fps: float, with_tone: bool):
    import cv2

    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    if with_tone:
        import subprocess

        n_sec = len(frames) / fps
        tone = OUT / "tone.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=0.1",
             "-af", f"atrim=0:{n_sec},asetpts=PTS-STARTPTS",
             str(tone)],
            capture_output=True,
        )
        muxed = path.with_name(path.stem + "_tone.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(path), "-i", str(tone),
             "-c:v", "copy", "-c:a", "aac", "-shortest", str(muxed)],
            capture_output=True,
        )
        if muxed.is_file():
            muxed.replace(path)


def main():
    images = load_images(limit=70)
    print(f"loaded {len(images)} real images")
    brands = {}
    for i in images:
        for g in i["gt"]:
            brands[g["brand"]] = brands.get(g["brand"], 0) + 1
    print("brand distribution:", dict(sorted(brands.items(), key=lambda kv: -kv[1])[:20]))

    # Video A: genuinely 3 seconds, 30 fps, 90 frames, one real image
    base = scale_to(images[0]["image"])
    frames_a = [zoom_pan(base, i / 89) for i in range(90)]
    write_video(OUT / "videoA_genuine_3s.mp4", frames_a, 30.0, with_tone=False)
    print("wrote video A:", len(frames_a), "frames")

    # Video B: ~60s @ 20fps = 1200 frames, 60 real images cycled
    cycle = images[:60]
    frames_b = []
    for img in cycle:
        scaled = scale_to(img["image"])
        frames_b.extend([scaled] * 20)
    write_video(OUT / "videoB_60s_20fps.mp4", frames_b, 20.0, with_tone=False)
    print("wrote video B:", len(frames_b), "frames")


if __name__ == "__main__":
    main()
