"""A frame the detector could not resolve must not render as 131 identical chips.

A real job stored 1,572 logo chips across 277 scenes, 1,567 of which were the
same string "UNKNOWN BRAND" — 131 of them on a single frame at p50 confidence
0.15. sceneChips capped objects at OBJECT_CHIP_CAP but printed every logo box,
and it printed one chip per box instead of one per distinct label.

These tests render the real sceneChips through node and assert on the HTML, so
they fail on a wrong value as well as a deleted line: a first pass used static
string checks, and setting LOGO_CHIP_CAP to 0 left all of them green.
"""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
JS = (ROOT / "static/app.js").read_text()
CSS = (ROOT / "static/styles.css").read_text()

NODE = shutil.which("node")

# sceneChips renders through these; the rest of app.js is browser-only wiring.
HELPERS = ["escapeHtml", "pct", "chip", "metaChip", "groupLogos"]
CAPS = ["OBJECT_CHIP_CAP", "LOGO_CHIP_CAP", "LOGO_OVERLOAD"]


def _extract(pattern, name):
    m = re.search(pattern, JS, re.S | re.M)
    assert m, f"{name} was removed from app.js"
    return m.group(0)


def render_chips(scene):
    """Return the actual chip HTML sceneChips builds for this scene."""
    parts = [_extract(rf"^const {c} = [^;]+;", c) for c in CAPS]
    parts += [_extract(rf"^function {h}\([\s\S]*?\n\}}", h) for h in HELPERS]
    parts.append(_extract(r"^function sceneChips\(s\) \{[\s\S]*?\n(?=\n)", "sceneChips"))
    body = "\n\n".join(parts)
    script = f"{body}\nconsole.log(sceneChips({json.dumps(scene)}));"
    out = subprocess.run([NODE, "-e", script], capture_output=True, text=True, check=True)
    return out.stdout


def unknown(n, conf=0.15):
    return [{"class_name": "UNKNOWN BRAND", "confidence": conf,
             "confidence_metric": "detector_box_confidence"}] * n


pytestmark = pytest.mark.skipif(NODE is None, reason="node not installed")


def test_swarm_collapses_to_one_counted_chip():
    html = render_chips({"objects": [], "logos": unknown(131)})
    assert html.count("UNKNOWN BRAND") == 1, (
        "identical labels must not each get a chip:\n" + html[:400]
    )
    assert "×131" in html, f"the count is the information, not the repetition:\n{html[:400]}"


def test_swarm_is_flagged_as_detector_overload():
    html = render_chips({"objects": [], "logos": unknown(131)})
    assert "chip--overload" in html, "an implausible box count must be flagged"
    assert "131 logo boxes on this frame" in html, (
        "the overload chip must state the real count, not hide it behind a cap:\n" + html[:400]
    )


def test_normal_frame_is_not_flagged():
    html = render_chips({"objects": [], "logos": unknown(3)})
    assert "chip--overload" not in html, "3 boxes on a frame is normal, not an overload"
    assert html.count("UNKNOWN BRAND") == 1


def test_cap_bounds_distinct_labels():
    # 9 distinct labels, cap is 6: grouped into 6, the rest become a ghost chip.
    logos = [{"class_name": f"BRAND{i}", "confidence": 0.5} for i in range(9)]
    html = render_chips({"objects": [], "logos": logos})
    shown = sum(1 for i in range(9) if f"BRAND{i}" in html)
    assert shown == 6, f"expected the cap to keep 6 labels, kept {shown}:\n{html[:400]}"
    assert "chip-ghost" in html, "truncated labels must be summarised, not dropped silently"


def test_distinct_brands_survive_grouping():
    logos = unknown(2, conf=0.11) + [{"class_name": "GOOGLE", "confidence": 0.90,
                                     "confidence_metric": "resolution_quality"}]
    html = render_chips({"objects": [], "logos": logos})
    assert "GOOGLE" in html, "a real brand must not be swallowed by the cap"
    assert "UNKNOWN BRAND ×2" in html
    assert "90%" in html, "grouped confidence should be the strongest instance"


def test_empty_scene_still_renders_placeholder():
    assert "No objects detected" in render_chips({"objects": [], "logos": []})


def test_overload_chip_is_styled_as_an_error():
    assert ".chip--overload" in CSS, "the overload chip needs its own styling"
    block = re.search(r"\.chip--overload \{[^}]*\}", CSS, re.S)
    assert block and "var(--error)" in block.group(0), (
        "the overload chip must use the error colour, not a neutral one — "
        "the scene is not fine"
    )


# ── the same failure at video scope ──────────────────────────────────────
# Job KH2G70N9-PNFAA stored 62 logo boxes, 0 candidates, 57 unresolved. Read
# alone that is a resolver that cannot find 57 brands. It is not: the detector
# was proposing ~112 boxes per frame, and its confidence distribution on
# frames with a logo (p50 0.148) is indistinguishable from frames without one
# (p50 0.143), so the 30% gate rejects a flat noise floor. The funnel of zeros
# is a detector reading. These assert the panel says so.


def _extract_pattern(pattern, name):
    m = re.search(pattern, JS, re.S | re.M)
    assert m, f"{name} was removed from app.js"
    return m.group(0)


def render_panel(dashboard):
    """Return the actual HTML openSetPanel builds."""
    parts = [_extract_pattern(rf"^function {h}\([\s\S]*?\n\}}", h) for h in
             ("escapeHtml", "escapeHtmlNum", "chip", "metaChip", "groupLogos")]
    parts.append(_extract_pattern(
        r"^function openSetPanel\(d\) \{[\s\S]*?\n\}(?=\n\nfunction )", "openSetPanel"))
    body = "\n\n".join(parts)
    script = f"{body}\nconsole.log(openSetPanel({json.dumps(dashboard)}));"
    out = subprocess.run([NODE, "-e", script], capture_output=True, text=True, check=True)
    return out.stdout


SATURATED = {
    "open_set": {
        "available": True,
        "backend": "gemini_grounded",
        "min_confidence": 0.30,
        "min_crop_area": 3000,
        "max_crop_aspect": 3.0,
        "candidates": [],
        "rejected": [],
        "skipped_counts": {"below_min_confidence": 62, "duplicate_hash": 0},
        "resolved": 0,
        "detector_diagnostics": {
            "proposals_total": 10596,
            "frames_with_proposals": 98,
            "max_proposals_in_a_frame": 187,
            "median_proposals_in_a_frame": 109,
            "saturated_proposals_per_frame": 30,
            "saturated_frames": [0, 1, 2, 3, 4, 5],
            "saturated_frame_count": 98,
        },
    }
}


def test_saturated_detector_is_flagged_in_the_open_set_panel():
    html = render_panel(SATURATED)
    assert "Detector saturated on 98 frames" in html, (
        "a saturated detector must be named, not left to read as a resolver "
        f"failure:\n{html[:600]}"
    )
    assert "187" in html and "109" in html, (
        "the warning must carry the real counts, since the whole point is "
        f"that they were hidden:\n{html[:600]}"
    )


def test_saturation_warning_says_it_is_not_a_verdict_on_brands():
    # The misreading to prevent: "0 candidates" -> "no brands here". The
    # warning has to contradict that explicitly.
    html = render_panel(SATURATED)
    assert "not a verdict on the brands" in html, (
        f"the zero funnel must be reframed as a detector reading:\n{html[:600]}"
    )


def test_clean_detector_shows_no_saturation_warning():
    clean = json.loads(json.dumps(SATURATED))
    dd = clean["open_set"]["detector_diagnostics"]
    dd.update({"saturated_frame_count": 0, "saturated_frames": [],
               "max_proposals_in_a_frame": 3, "median_proposals_in_a_frame": 2})
    html = render_panel(clean)
    assert "Detector saturated" not in html, (
        "a normal frame count must not cry saturation:\n" + html[:600])


def test_missing_diagnostics_do_not_break_older_jobs():
    # Jobs stored before this report have no detector_diagnostics key. They
    # must still render rather than throw on undefined.
    old = json.loads(json.dumps(SATURATED))
    del old["open_set"]["detector_diagnostics"]
    html = render_panel(old)
    assert "Detector saturated" not in html
    assert "62" in html, "the existing funnel must still render its counts"


def test_saturation_warning_is_styled_as_a_warning():
    assert ".os-warn" in CSS, "the saturation warning needs its own styling"
    assert "os-warn" in JS, "the class must be what the panel actually renders"
