import json

import pytest

from fusion_eval.ablations import rerun_fusion
from fusion_eval.calibration import best_temperature
from fusion_eval.dataset import load_dataset
from fusion_eval.metrics import compute_metrics, split_by_modality
from fusion_eval.reliability import reliability_curve


def test_ece_rejects_probs_outside_unit_interval():
    """Out-of-range probs match no bin, so they inflate the denominator
    `p.size` without ever reaching the numerator — silently deflating ECE.
    A perfect set reported ECE 0.4545 because of one stray 1.4."""
    for bad in (1.4, -0.5, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            reliability_curve([0.5, bad], [True, True])
    with pytest.raises(ValueError):
        reliability_curve([0.5], [True, False])


def test_ece_is_binned_not_mean_absolute_error():
    """ECE must measure the confidence/accuracy GAP WITHIN each bin.

    The previous implementation returned mean(abs(p - y)), which scored a
    perfectly calibrated model at 0.18 and could not tell a calibrated model
    from an inverted one.
    """
    # 0.9-bin is 90% correct, 0.1-bin is 10% correct -> calibrated.
    probs = [0.9] * 10 + [0.1] * 10
    good = [True] * 9 + [False] + [True] + [False] * 9
    _, ece, _ = reliability_curve(probs, good)
    assert ece == 0.0, ece

    # Same confidences, accuracy inverted -> must blow up.
    bad = [False] * 9 + [True] + [False] + [True] * 9
    _, ece_bad, _ = reliability_curve(probs, bad)
    assert ece_bad > 0.7, ece_bad


def test_best_temperature_reports_heldout_ece():
    """T must be scored on rows it was not fitted on."""
    fit = [_pred(f"f{i}", "A", "A", "confident", prob=0.9) for i in range(4)]
    test = [_pred(f"t{i}", "A", "A", "confident", prob=0.8) for i in range(3)]
    cal = best_temperature(fit, test_rows=test)
    assert cal["n_fit"] == 4 and cal["n_test"] == 3
    # Held-out numbers exist and are measured on the other rows, so they must
    # not simply echo the fit-split objective the grid search minimised.
    assert "heldout_ece" in cal and "heldout_brier" in cal
    assert cal["heldout_ece"] != cal["ece"]


def _pred(video_id, gt, winner, verdict, prob=0.9, mods=None, diff="easy"):
    return {
        "video_id": video_id,
        "gt_brand": gt,
        "visible": gt is not None,
        "difficulty": diff,
        "winner": winner,
        "prob": prob,
        "margin": 0.2,
        "verdict": verdict,
        "top3": [winner] if winner else [],
        "latency_layer1": 0.1,
        "gpu_mem_mb": 0.0,
        "modalities": mods,
    }


def test_split_by_modality_groups_by_available_signals():
    preds = [
        _pred("a", "NIKE", "NIKE", "confident", mods={"logo": True, "ocr": True}),
        _pred("b", "NIKE", "NIKE", "confident", mods={"logo": True, "ocr": True}),
        _pred("c", "ADIDAS", None, "abstain", 0.0, mods={"speech": True}),
    ]
    out = split_by_modality(preds)
    assert "logo+ocr" in out and "speech" in out
    assert out["logo+ocr"]["n"] == 2.0


def test_split_by_modality_tolerates_missing_field():
    out = split_by_modality([_pred("a", "NIKE", "NIKE", "confident")])
    assert "unknown" in out


def test_compute_metrics_top1_and_abstention():
    preds = [
        _pred("a", "NIKE", "NIKE", "confident"),
        _pred("b", "ADIDAS", "NIKE", "confident"),   # wrong accept
        _pred("c", "SAMSUNG", None, "abstain", 0.0),
        _pred("d", None, None, "abstain", 0.0),      # negative correctly abstained
    ]
    m = compute_metrics(preds)
    # correct NIKE + correctly-abstained negative = 2/4; wrong accept + missed
    # positive are the two errors.
    assert m["top1_accuracy"] == 0.5
    assert m["abstention_rate"] == 0.5
    assert m["false_positive_rate"] == 0.0


def test_rerun_fusion_disable_family_changes_verdict():
    pred = {
        "video_id": "v",
        "gt_brand": "NIKE",
        "visible": True,
        "winner": "NIKE",
        "prob": 0.9,
        "verdict": "confident",
        "top3": ["NIKE"],
        "_ledger": {
            "NIKE": [{"family": "logo", "strength": 0.9, "quality": 1.0, "timestamp": 0.0}],
        },
    }
    ablated = rerun_fusion(pred, ["logo"])
    assert ablated["winner"] is None
    assert ablated["verdict"] == "abstain"
    # original untouched
    assert pred["winner"] == "NIKE"


def test_load_dataset_round_trip(tmp_path):
    rows = [{"video_id": "v1", "gt_brand": "NIKE", "difficulty": "easy",
             "visible_brand": True, "logo": True, "ocr": False, "speech": False,
             "product": False, "audio_event": False, "timestamps": [[0, 1]],
             "video_path": "/tmp/v1.mp4"}]
    p = tmp_path / "gt.json"
    p.write_text(json.dumps(rows))
    assert load_dataset(str(p))[0]["gt_brand"] == "NIKE"
