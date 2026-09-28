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
