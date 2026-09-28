"""The gates must measure the right thing, or they are decoration.

`--freeze` on benchmark/eval_video/labels.jsonl blocked with
"597 labelled keys vs 0 frames on disk" while the pool plainly had 739 frames.
The cause was `glob("*.jpg")` against a pool that stores one subdirectory per
video (images/a, images/b, images/c). The frozen set is flat, so the bug only
appeared once a nested pool met the gate - and it reported 0 frames, so every
downstream number (box sizes, the skip threshold, the key/frame ratio) was
derived from that zero rather than from the data.

These tests pin the two behaviours that matter:

  * nested frames are discovered, so a nested pool is not silently zero
  * the judged-percentage rule applies to --freeze and NOT to --pool

The second is the semantic split. A frozen set is measured once and cannot
grow, so an unjudged frame in it is a test case that will never run, and the
rule is right. A working pool is reviewed incrementally and legitimately
carries skips; failing it for that would mean deleting unreviewed frames to
make a number look tidy. One rule, two meanings, was the bug.
"""

import importlib.util as ilu
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
_spec = ilu.spec_from_file_location(
    "gate", ROOT / "scripts/check_label_store.py")
gate = ilu.module_from_spec(_spec)
_spec.loader.exec_module(gate)


def _row(file, verdict, boxes=()):
    return {"file": file, "video": "v", "frame_index": 0,
            "verdict": verdict, "boxes": list(boxes), "human_drawn": bool(boxes)}


def _box(x=10, y=10, w=20, h=20, src="human"):
    return {"x": x, "y": y, "w": w, "h": h, "src": src, "on": True}


def _pool(tmp_path, layout, rows):
    """`layout` maps relative path -> True to write a frame. Nested on purpose
    for the discovery test."""
    for rel in layout:
        p = tmp_path / "images" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), np.full((200, 200, 3), 255, np.uint8))
    lab = tmp_path / "labels.jsonl"
    lab.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return lab


def test_frame_discovery_is_recursive(tmp_path):
    """The bug. A pool with frames in per-video subdirectories reported zero
    frames, which is how a 597-row label set failed against 739 real frames."""
    lab = _pool(tmp_path, ["a/aaa_000000.jpg", "b/bbb_000000.jpg",
                           "c/ccc_000000.jpg"],
                [_row("aaa_000000.jpg", "free")])
    found = gate.frame_index(lab)
    assert len(found) == 3, f"nested frames not discovered: {sorted(found)}"


def test_frame_discovery_still_works_flat(tmp_path):
    lab = _pool(tmp_path, ["t_000000.jpg"], [_row("t_000000.jpg", "free")])
    assert len(gate.frame_index(lab)) == 1


def test_pool_gate_passes_with_nested_frames_and_skips(tmp_path):
    """Working pool: nested frames, some skipped, some frames not reviewed.
    This is the exact shape that used to fail, for reasons unrelated to the
    labels."""
    lab = _pool(
        tmp_path,
        [f"a/v_{i:06d}.jpg" for i in range(6)],
        [_row(f"v_{i:06d}.jpg", "free" if i % 2 else "logo",
              [_box()] if i % 2 == 0 else ())
         for i in range(4)] + [_row(f"v_{i:06d}.jpg", "skip") for i in (4,)])
    assert gate.pool_gate(lab) == 0


def test_freeze_gate_blocks_the_same_pool_for_being_unfinished(tmp_path):
    """Same shape, strict gate. 200 frames with 5 skipped: the allowance is
    2%, so 5 skips means the set is not fully judged - and a frozen set is
    measured once, so those 5 frames would never be tested. This is the rule
    that must NOT leak into --pool."""
    n = 200
    lab = _pool(
        tmp_path,
        [f"a/v_{i:06d}.jpg" for i in range(n)],
        [_row(f"v_{i:06d}.jpg", "free" if i % 2 else "logo",
              [_box()] if i % 2 == 0 else ())
         for i in range(n - 5)] +
        [_row(f"v_{i:06d}.jpg", "skip") for i in range(n - 5, n)])
    assert gate.pool_gate(lab) == 0, "pool must tolerate an unfinished review"
    assert gate.freeze_gate(lab) == 1, "freeze must not"


def test_pool_gate_rejects_a_label_with_no_frame(tmp_path):
    """Structurally broken, so both gates must fail it: a label for a missing
    frame grades nothing and the real frame is silently unjudged."""
    lab = _pool(tmp_path, ["a/v_000000.jpg"],
                [_row("v_000000.jpg", "logo", [_box()]),
                 _row("v_999999.jpg", "free")])
    assert gate.pool_gate(lab) == 1


def test_pool_gate_rejects_model_drawn_boxes(tmp_path):
    """The whole loop grades a detector against human ground truth. A box
    with src != human is the detector's own output, and accepting it makes
    every number downstream self-graded."""
    lab = _pool(tmp_path, ["v_000000.jpg"],
                [_row("v_000000.jpg", "logo", [_box(src="model")])])
    assert gate.pool_gate(lab) == 1


def test_pool_gate_rejects_boxes_on_a_free_verdict(tmp_path):
    lab = _pool(tmp_path, ["v_000000.jpg"],
                [_row("v_000000.jpg", "free", [_box()])])
    assert gate.pool_gate(lab) == 1


def test_pool_gate_rejects_duplicate_keys(tmp_path):
    lab = _pool(tmp_path, ["v_000000.jpg"],
                [_row("v_000000.jpg", "free"), _row("v_000000.jpg", "logo", [_box()])])
    assert gate.pool_gate(lab) == 1


def test_pool_gate_rejects_unknown_verdict(tmp_path):
    lab = _pool(tmp_path, ["v_000000.jpg"], [_row("v_000000.jpg", "maybe")])
    assert gate.pool_gate(lab) == 1


def test_both_gates_are_reachable_and_mutually_exclusive(tmp_path):
    lab = _pool(tmp_path, ["v_000000.jpg"],
                [_row("v_000000.jpg", "logo", [_box()]),
                 _row("v_000001.jpg", "free")][:1])
    import subprocess
    import sys
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts/check_label_store.py"),
         "--pool", str(lab), "--freeze", str(lab)],
        capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0
    assert "pick one" in (r.stdout + r.stderr)
