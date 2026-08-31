"""Tests for the DB seed script (scripts/seed_db.py)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.brand_catalog import BRAND_CATALOG

from scripts import seed_db


def test_all_demo_brands_are_catalog_keys():
    for _creator, _handle, _f, _e, tallies in seed_db.DEMO_CREATORS:
        for brand, _n in tallies:
            assert brand in BRAND_CATALOG, f"{brand} not a catalog key"


def test_build_job_produces_persistable_snapshot():
    job = seed_db.build_job(
        "DEMO-TEST-1", "demo title", "videos/test.mp4",
        "tester", "tester.handle", 1000, 100,
        [("NIKE", 2), ("ADIDAS", 1)],
    )
    assert job["job_id"] == "DEMO-TEST-1"
    assert job["status"] == "done"
    assert job["result"]["layer2d"]["creator_profile"]["brand_tallies"] == {
        "NIKE": 2, "ADIDAS": 1,
    }
    recs = job["result"]["layer3"]["recommendations"]
    assert [r["brand"] for r in recs] == ["NIKE", "ADIDAS"]
    assert all(r["type"] == "DIRECT" for r in recs)
    assert "dashboard" in job and job["dashboard"]["creator"] == "tester.handle"


def test_build_job_creator_profile_tallies_categories():
    job = seed_db.build_job(
        "DEMO-TEST-2", "t", "v", "c", "h", 0, 0, [("NIKE", 1), ("ADIDAS", 1)],
    )
    cats = job["result"]["layer2d"]["creator_profile"]["categories"]
    # NIKE/ADIDAS both carry APPAREL and FOOTWEAR
    assert cats.get("APPAREL", 0) == 2
    assert cats.get("FOOTWEAR", 0) == 2
