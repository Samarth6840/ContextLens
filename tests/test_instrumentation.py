import pytest

from src.layer2.brand_memory import BrandMemoryBank
from src.layer2.brand_resolver import build_brand_timeline
from src.pipeline import _count_resolved_logos, temporal_corroboration
from src.store import prune_for_store
from scripts.offline_analysis import (
    job_summary,
    modality_ablation,
    parse_labels,
    precision_recall_report,
)


# --------------------------------------------------------------------------- #
# Brand-memory provenance (metrics-only; must never affect matching)
# --------------------------------------------------------------------------- #
def test_memory_provenance_recorded_without_tracking_score():
    bank = BrandMemoryBank()
    bank.record("Apple", confidence=0.8, resolution_source="ocr",
                embedding_version="dinoV2:vitl14")
    bank.record("apple", confidence=0.95, resolution_source="clip_retrieval",
                embedding_version="dinoV2:vitl14")
    ent = bank.get("APPLE")
    assert ent["sources"] == {"ocr": 1, "clip_retrieval": 1}
    assert ent["max_confidence_source"] == "clip_retrieval"
    assert ent["max_confidence"] == pytest.approx(0.95)
    assert ent["embedding_model_version"] == "dinoV2:vitl14"


def test_memory_provenance_does_not_drive_resolution():
    bank = BrandMemoryBank(recency_decay=1.0)
    bank.record("NIKE", confidence=0.9, resolution_source="ocr", wall_time=0)
    bank.record("ADIDAS", confidence=0.7, resolution_source="clip_retrieval",
                wall_time=1)
    # Provenance is ignored by ranking: newer+lower-confidence ADIDAS must NOT
    # leapfrog NIKE purely because of a 'resolution_source' tag.
    resolved = bank.resolve_reference("these shoes", ref_clock=10)
    assert resolved["brand"] == "NIKE"


def test_memory_provenance_serialization_roundtrip():
    bank = BrandMemoryBank()
    bank.record("APPLE", confidence=0.9, resolution_source="ocr",
                embedding_version="dinoV2:vitl14")
    data = bank.to_dict()
    loaded = BrandMemoryBank.from_dict(data)
    ent = loaded.get("APPLE")
    assert ent["sources"] == {"ocr": 1}
    assert ent["embedding_model_version"] == "dinoV2:vitl14"


def test_memory_record_backward_compatible_defaults():
    bank = BrandMemoryBank()
    bank.record("NIKE", confidence=0.8)  # no provenance kwargs at all
    ent = bank.get("NIKE")
    assert ent["sources"] == {}
    assert ent["max_confidence_source"] is None
    assert ent["embedding_model_version"] is None


# --------------------------------------------------------------------------- #
# Brand-timeline provenance
# --------------------------------------------------------------------------- #
def test_build_brand_timeline_attaches_resolution_source():
    resolved = [[
        {"brand": "APPLE", "confidence": 0.7,
         "resolution_source": "ocr", "resolution_quality": 0.90},
    ]]
    timeline = build_brand_timeline(resolved, [], video_fps=30.0)
    app = timeline["APPLE"]["appearances"][0]
    assert app["resolution_source"] == "ocr"
    assert app["resolution_quality"] == 0.90


# --------------------------------------------------------------------------- #
# Temporal-corroboration metric
# --------------------------------------------------------------------------- #
def test_temporal_corroboration_counts_agreement_backfill():
    m = temporal_corroboration(3, 5)
    assert m["needed_multi_frame_agreement"] == 2
    assert m["corroboration_rate"] == pytest.approx(0.4)
    assert m["resolved_without_agreement"] == 3


def test_temporal_corroboration_zero_division_safe():
    m = temporal_corroboration(0, 0)
    assert m["needed_multi_frame_agreement"] == 0
    assert m["corroboration_rate"] == 0.0


def test_count_resolved_logos_ignores_unbranded_detections():
    dets = [[{"brand": "APPLE"}], [{}], [{"brand": "NIKE"}, {"brand": None}]]
    assert _count_resolved_logos(dets) == 2


# --------------------------------------------------------------------------- #
# prune_for_store retention
# --------------------------------------------------------------------------- #
def test_prune_retains_temporal_corroboration():
    pruned = prune_for_store({
        "layer2c": {"temporal_corroboration": {"corroboration_rate": 0.4}},
    })
    assert pruned["temporal_corroboration"]["corroboration_rate"] == 0.4


# --------------------------------------------------------------------------- #
# Offline analysis functions
# --------------------------------------------------------------------------- #
def _sample_job(vid, timeline, created_at="2026-01-01T00:00:00+00:00"):
    return {
        "video_path": vid,
        "created_at": created_at,
        "resolver_acceptance": {"resolved": 3, "acceptance_rate": 0.6},
        "temporal_corroboration": {"corroboration_rate": 0.2},
        "layer2c": {
            "brand_timeline": timeline,
            "indirect_resolutions": [],  # computed below from outputs
        },
    }


def test_parse_labels_skips_comments_and_empty_lines():
    text = "# header comment\nvideo_id,brand,present\n_v1.mp4,APPLE,1\n\n_v1.mp4,NIKE,0\n"
    labels = parse_labels(text)
    assert labels["_v1.mp4"]["present"] == {"APPLE"}
    assert labels["_v1.mp4"]["absent"] == {"NIKE"}


def test_parse_labels_unknown_control():
    labels = parse_labels("video_id,brand,present\n_v1.mp4,__UNKNOWN__,1\n")
    assert labels["_v1.mp4"]["has_unknown"] is True


def test_job_summary_aggregates_sources_and_modalities():
    job = _sample_job("_v1.mp4", {
        "APPLE": {
            "appearances": [
                {"modality": "logo", "confidence": 0.6,
                 "resolution_source": "ocr"},
                {"modality": "logo", "confidence": 0.9,
                 "resolution_source": "ocr"},
                {"modality": "speech", "confidence": 1.0},
            ],
            "modalities": ["logo", "speech"],
        },
    })
    s = job_summary(job)
    assert s["sources"] == {"ocr": 2}
    assert s["max_conf_visual"] == pytest.approx(0.9)
    assert s["max_conf_speech"] == pytest.approx(1.0)
    assert s["cross_scene"] == 1
    assert s["acceptance_rate"] == 0.6


def test_modality_ablation_partitions_by_evidence():
    job = _sample_job("_v1.mp4", {
        "APPLE": {"modalities": ["logo", "speech"],
                 "appearances": [{"modality": "logo", "confidence": 0.9},
                                 {"modality": "speech", "confidence": 1.0}]},
        "NIKE": {"modalities": ["logo"],
                 "appearances": [{"modality": "logo", "confidence": 0.8}]},
        "SPEECHY": {"modalities": ["speech"],
                    "appearances": [{"modality": "speech", "confidence": 1.0}]},
    })
    ab = modality_ablation([job])
    assert ab["cross_scene"]["n"] == 1
    assert ab["visual_only"]["n"] == 1
    assert ab["speech_only"]["n"] == 1


def test_precision_recall_report_counts_tp_fp_fn_and_unknown_errors():
    jobs = [
        _sample_job("a.mp4", {"APPLE": {
            "appearances": [{"modality": "logo", "confidence": 0.9,
                             "resolution_quality": 0.9}],
            "modalities": ["logo"]}}),
        _sample_job("b.mp4", {"NIKE": {
            "appearances": [{"modality": "logo", "confidence": 0.5,
                             "resolution_quality": 0.5}],
            "modalities": ["logo"]}}),
        _sample_job("c.mp4", {"ADIDAS": {
            "appearances": [{"modality": "logo", "confidence": 0.7,
                             "resolution_quality": 0.7}],
            "modalities": ["logo"]}}),
    ]
    text = (
        "video_id,brand,present\n"
        "a.mp4,APPLE,1\n"
        "b.mp4,NIKE,1\n"
        "b.mp4,APPLE,0\n"      # APPLE resolved in b? no -> not counted
        "c.mp4,ADIDAS,1\n"
        "c.mp4,__UNKNOWN__,1\n"  # UNKNOWN control: c resolved ADIDAS -> precision error
    )
    report = precision_recall_report(jobs, parse_labels(text))
    assert report["tp"] == 3
    assert report["fn"] == 0
    assert report["fp"] == 0
    assert report["unresolved_precision_errors"] == 1
    assert report["precision"] == pytest.approx(1.0)
    assert report["ece"] >= 0.0


def test_precision_recall_report_counts_false_positive():
    jobs = [_sample_job("a.mp4", {"APPLE": {
        "appearances": [{"modality": "logo", "confidence": 0.9,
                         "resolution_quality": 0.9}],
        "modalities": ["logo"]}})]
    text = "video_id,brand,present\na.mp4,NIKE,1\na.mp4,APPLE,0\n"
    report = precision_recall_report(jobs, parse_labels(text))
    assert report["fn"] == 1
    assert report["fp"] == 1
    assert report["precision"] == 0.0
    assert report["recall"] == 0.0