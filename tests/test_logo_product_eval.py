import numpy as np
from pathlib import Path

from scripts.build_logo_product_eval_manifest import build_manifest
from scripts.benchmark_open_set_providers import crop_gold_box, score_result_titles


def test_product_eval_manifest_is_heldout_and_one_example_per_brand():
    rows = build_manifest(
        Path("benchmark/realvideo_v1/labels.jsonl"),
        Path("benchmark/realvideo_v1/splits.json"),
    )
    assert rows
    assert len({row["brand"] for row in rows}) == len(rows)
    assert {row["split"] for row in rows} <= {"val", "test"}
    assert all(row["label_provenance"] == "human" for row in rows)


def test_provider_benchmark_crops_only_gold_box():
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    crop = crop_gold_box(frame, [20, 20, 30, 20], padding=0)
    assert crop.shape == (20, 30, 3)


def test_provider_results_score_first_and_top_five_titles():
    from src.openset import ReverseImageResult
    results = [
        ReverseImageResult("Unknown item", "https://example.com/", "test"),
        ReverseImageResult("Samsung Galaxy", "https://samsung.com/", "test"),
    ]
    score = score_result_titles(results, "SAMSUNG")
    assert score["top1_hit"] is False
    assert score["top5_hit"] is True
