"""Dataset schema + synthetic ground-truth video builder.

Entry schema (JSON-serializable), one per video:

    {
        "video_id": str,
        "gt_brand": str | None,        # None => negative entry (no brand claimable)
        "difficulty": "easy|small|blur|conflict|negative",
        "visible_brand": bool,         # is any branded object actually on screen
        "logo": bool, "ocr": bool, "speech": bool, "product": bool,
        "audio_event": bool,           # which modalities carry GT evidence
        "timestamps": [[start_s, end_s], ...],
        "video_path": str
    }

Synthetic builder composes reference logos (DINOv2's own reference set) onto
moving gradient backgrounds so detections/OCR/product paths actually fire; the
hard-versus-easy axis (scale / blur / conflict / negative) feeds the failure
taxonomy.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional

import cv2
import numpy as np

FPS = 25
FRAMES = 25          # 1 second of video per entry
W, H = 720, 480


def _noise_bg(rng: np.random.Generator) -> np.ndarray:
    base = np.full((H, W, 3), (18, 18, 22), np.uint8)
    for _ in range(60):  # soft gradient blobs so OCR/logo boxes have context
        x, y = rng.integers(0, W), rng.integers(0, H)
        r = rng.integers(40, 140)
        color = rng.integers(40, 180, size=3).tolist()
        cv2.circle(base, (int(x), int(y)), int(r), color, -1)
    return cv2.GaussianBlur(base, (0, 0), 18)


def _logo_path(bank_root: str, brand: str) -> Optional[str]:
    d = os.path.join(bank_root, brand.upper())
    if not os.path.isdir(d):
        return None
    for f in sorted(os.listdir(d)):  # deterministic pick
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            return os.path.join(d, f)
    return None


def _compose(bg: np.ndarray, logo: np.ndarray, scale: float, blur: int) -> np.ndarray:
    lh, lw = logo.shape[:2]
    s = min(W * scale / lw, H * scale / lh)
    res = cv2.resize(logo, (max(1, int(lw * s)), max(1, int(lh * s))))
    if blur:
        res = cv2.GaussianBlur(res, (0, 0), blur)
    h, w = res.shape[:2]
    x, y = (W - w) // 2, (H - h) // 2 - 20
    out = bg.copy()
    alpha = res[:, :, 3].astype(np.float32) / 255.0 if res.shape[2] == 4 else np.ones((h, w), np.float32)
    fg = res[:, :, :3].astype(np.float32)
    out[y:y + h, x:x + w] = (fg * alpha[..., None] + out[y:y + h, x:x + w].astype(np.float32) * (1 - alpha[..., None])).astype(np.uint8)
    return out


def _synthesize(entry: dict, bank_root: str, out_dir: str) -> str:
    rng = np.random.default_rng(abs(hash(entry["video_id"])) % 2**32)
    bg = _noise_bg(rng)
    frames: List[np.ndarray] = []
    if entry["difficulty"] == "negative":
        d = [bg.copy() for _ in range(FRAMES)]           # translate bg for motion
        for i, f in enumerate(d):
            dx = int(6 * np.sin(i / FRAMES * np.pi * 2))
            frames.append(cv2.warpAffine(f, np.float32([[1, 0, dx], [0, 1, 0]]), (W, H)))
    else:
        logo = cv2.imread(_logo_path(bank_root, entry["gt_brand"]), cv2.IMREAD_UNCHANGED)
        for i in range(FRAMES):
            scale = {"easy": 0.55, "small": 0.18, "blur": 0.55, "conflict": 0.55}[entry["difficulty"]]
            blur = {"blur": 7}.get(entry["difficulty"], 0)
            frame = _compose(bg, logo, scale, blur)
            if entry["difficulty"] == "conflict":
                other = entry.get("conflict_brand")
                if other:
                    l2 = cv2.imread(_logo_path(bank_root, other), cv2.IMREAD_UNCHANGED)
                    frame = _compose(frame, l2, 0.30 * scale, 0)
            dx = int(6 * np.sin(i / FRAMES * np.pi * 2))
            frames.append(cv2.warpAffine(frame, np.float32([[1, 0, dx], [0, 1, 0]]), (W, H)))
    path = os.path.join(out_dir, entry["video_id"] + ".mp4")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for f in frames:
        vw.write(f)
    vw.release()
    return path


def build_dataset(bank_root: str, out_dir: str, budget: int = 8, seed: int = 7) -> List[dict]:
    """Build a small synthetic ground-truth dataset. budget caps total entries."""
    brands = sorted(
        b for b in os.listdir(bank_root)
        if os.path.isdir(os.path.join(bank_root, b)) and _logo_path(bank_root, b)
    )
    entries: List[dict] = []
    seen = set()
    rng = np.random.default_rng(seed)

    def add(diff: str, gt: Optional[str], visible: bool, conflict: Optional[str] = None):
        vid = f"{diff}_{gt or 'NEG'}"
        if conflict:
            vid += f"_vs_{conflict}"
        if vid in seen:
            return
        seen.add(vid)
        entries.append({
            "video_id": vid,
            "gt_brand": gt,
            "difficulty": diff,
            "visible_brand": visible,
            "logo": gt is not None, "ocr": gt is not None,
            "speech": False, "product": gt is not None, "audio_event": False,
            "timestamps": [[0.0, FRAMES / FPS]],
            "video_path": "",
        })

    if len(brands) >= 2:  # one unambiguous + one near-tie conflict
        for b in brands[: max(1, budget // 6)]:
            add("easy", b, True)
    rng.shuffle(brands)
    for i, b in enumerate(brands[: max(0, budget // 6)]):
        add("easy", b, True)
        if i == 0:
            add("small", b, True)
            add("blur", b, True)
    if len(brands) >= 2:
        add("conflict", brands[0], True, conflict=brands[1])
    add("negative", None, False)

    for e in entries:
        e["video_path"] = _synthesize(e, bank_root, out_dir)
    gt_path = os.path.join(out_dir, "ground_truth.json")
    json.dump(entries, open(gt_path, "w"), indent=2)
    return entries


def load_dataset(path: str) -> List[dict]:
    return json.load(open(path))