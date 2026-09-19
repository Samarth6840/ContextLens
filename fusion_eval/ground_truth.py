"""Ground-truth table from dataset entries: the labels the metrics compare against."""

from __future__ import annotations

from typing import List

from .dataset import load_dataset


def ground_truth(entries: List[dict]) -> List[dict]:
    """Return one label row per entry (already present in the schema)."""
    rows = []
    for e in entries:
        rows.append({
            "video_id": e["video_id"],
            "gt_brand": e.get("gt_brand"),
            "difficulty": e.get("difficulty", "easy"),
            "visible_brand": bool(e.get("visible_brand")),
            "modalities": {
                "logo": bool(e.get("logo")),
                "ocr": bool(e.get("ocr")),
                "speech": bool(e.get("speech")),
                "product": bool(e.get("product")),
                "audio_event": bool(e.get("audio_event")),
            },
        })
    return rows


def load_ground_truth(gt_json: str) -> List[dict]:
    return ground_truth(load_dataset(gt_json))