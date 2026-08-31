"""Tests for spec/creator attribute extraction (Phase 3 content-type split)."""

from src.layer1.spec_extractor import extract_creator, extract_specs


def test_screen_size_extraction():
    specs = extract_specs("8 Inches Inner Display")
    assert any(s["field"] == "screen_size" and s["value"] == 8.0
               and s["unit"] == "in" for s in specs)


def test_thickness_and_battery():
    specs = extract_specs("4.1mm Thickness When Opened · 5000mAh Battery")
    d = {s["field"]: s for s in specs}
    assert d["thickness"]["value"] == 4.1 and d["thickness"]["unit"] == "mm"
    assert d["battery"]["value"] == 5000.0 and d["battery"]["unit"] == "mAh"


def test_material_extraction():
    specs = extract_specs("Titanium Alloy Film")
    assert any(s["field"] == "material" and "Titanium" in s["value"] for s in specs)


def test_processor_extraction():
    specs = extract_specs("Snapdragon 8 Elite Gen 5")
    proc = [s for s in specs if s["field"] == "processor"]
    assert proc and "Snapdragon" in proc[0]["value"]


def test_fails_closed_on_unrelated_text():
    assert extract_specs("Hello world, this is a friendly greeting") == []


def test_creator_handle_and_followers():
    c = extract_creator("TechBurner · @techburner · 4.3m followers")
    assert c["handle"] == "techburner"
    assert c["followers"] == 4_300_000


def test_creator_fails_closed():
    assert extract_creator("Welcome to the channel") == {}


def test_creator_no_handle_but_followers():
    c = extract_creator("1.2M followers and counting")
    assert c.get("handle") is None
    assert c["followers"] == 1_200_000
