"""
Tests for the restored dashboard products aggregation.

The dashboard `products` table was previously hard-gated to NOT_AVAILABLE
("0% brand accuracy"). It now aggregates RESOLVED on-screen brands from the
logo pipeline. These tests pin that behavior:
  * a resolved brand in logo_detections -> products_status AVAILABLE + a row
  * only resolved (brand-set) detections become products
  * unresolved "UNKNOWN BRAND" detections never become products
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from server import _build_dashboard  # noqa: E402


def _make_result(logo_detections, brand_mentions=(), num_frames=4):
    # Pad logo detections to num_frames sublists (one per sampled frame).
    framed_logos = list(logo_detections) + [[]] * (num_frames - len(logo_detections))
    return {
        "num_frames": num_frames,
        "video_fps": 25.0,
        "video_total_frames": 100,
        "video_stride": 1,
        "has_audio": True,
        "layer1": {
            "logo_detections": framed_logos,
            "ocr_results": [[{"text": "SAMSUNG", "confidence": 0.9}] for _ in range(num_frames)],
            "scene_object_detections": [[] for _ in range(num_frames)],
            "transcript": "hello samsung phone",
            "audio_events": [],
            "brand_mentions": list(brand_mentions),
        },
        "layer2b": {
            "confidence": 0.9,
            "is_confident": True,
            "status": "confident",
            "evidence_breakdown": {},
        },
        "layer2c": {"unknown_brand_regions": []},
        "layer3": {"recommendations": []},
    }


def _logo(brand, conf=0.9, resolution_quality=0.9):
    return {
        "class_name": brand,
        "brand": brand,
        "confidence": conf,
        "resolution_quality": resolution_quality,
        "ocr_text": brand,
    }


def test_resolved_logo_produces_available_product():
    result = _make_result([[_logo("SAMSUNG"), _logo("GOOGLE")]])
    job = {"job_id": "J1", "title": "t", "creator": "c", "filename": "f.mp4"}
    dash = _build_dashboard(result, job)
    assert dash["products_status"] == "AVAILABLE"
    brands = {p["brand"] for p in dash["products"]}
    assert "SAMSUNG" in brands
    assert "GOOGLE" in brands
    assert all(p["appearances"] for p in dash["products"])


def test_unresolved_logo_not_a_product():
    # brand None / class 'UNKNOWN BRAND' must NOT become a product row.
    unresolved = {
        "class_name": "UNKNOWN BRAND",
        "brand": None,
        "confidence": 0.13,
        "resolution_quality": None,
    }
    result = _make_result([[unresolved]])
    job = {"job_id": "J2", "title": "t", "creator": "c", "filename": "f.mp4"}
    dash = _build_dashboard(result, job)
    assert dash["products_status"] == "NO_PRODUCTS"
    assert dash["products"] == []


def test_class_name_only_logo_not_a_product():
    # A detector box classed "ROLEX logo" but NOT resolved to a brand (brand is
    # None) must not become a product. Fixes the regression where logodev /
    # yolo-world class names ("<Brand> logo") leaked into the products table as
    # NON-catalog brands with category=GENERAL.
    class_only = {
        "class_name": "ROLEX logo",
        "brand": None,
        "confidence": 0.2,
        "resolution_quality": None,
        "ocr_text": "ROLEX logo",
    }
    result = _make_result([[class_only]])
    job = {"job_id": "J2b", "title": "t", "creator": "c", "filename": "f.mp4"}
    dash = _build_dashboard(result, job)
    assert dash["products_status"] == "NO_PRODUCTS"
    assert dash["products"] == []


def test_no_logo_no_products_status():
    result = _make_result([[]])
    job = {"job_id": "J3", "title": "t", "creator": "c", "filename": "f.mp4"}
    dash = _build_dashboard(result, job)
    assert dash["products_status"] == "NO_PRODUCTS"
    assert dash["products"] == []
