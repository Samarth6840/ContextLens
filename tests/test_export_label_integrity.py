"""A YOLO label file may never drop an individual box.

Every unlabelled region is trained as background, so dropping one box from an
image's label list teaches the model that a real annotated logo is background.
export_yolo previously subsampled dense images to 50 boxes with np.linspace,
pushing 13.9x more box-area into the background class than the positive class.
The only safe operations are: keep every box, or drop the whole image.
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import importlib.util as ilu
from pathlib import Path

_spec = ilu.spec_from_file_location(
    "tld", Path(__file__).resolve().parent.parent / "scripts" / "train_logo_detector.py"
)
tld = ilu.module_from_spec(_spec)
_spec.loader.exec_module(tld)


def _png_bytes(h=200, w=200):
    import cv2
    arr = np.full((h, w, 3), 128, np.uint8)
    ok, buf = cv2.imencode(".png", arr)
    assert ok
    return buf.tobytes()


def _shard(path, imgs):
    """imgs: [(image_name, [boxes_xyxy])] -> one parquet row per box, which is
    how LogoDet-3K is actually laid out (158,654 rows for 310 images)."""
    ipath, bboxes = [], []
    for name, boxes in imgs:
        for b in boxes:
            ipath.append({"bytes": _png_bytes(), "path": f"x/{name}.png"})
            bboxes.append([int(v) for v in b])
    table = pa.table({
        "image_path": pa.array(
            ipath,
            type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]),
        ),
        "company_name": pa.array([0] * len(bboxes), type=pa.int64()),
        "bbox": pa.array(bboxes, type=pa.list_(pa.int64())),
    })
    pq.write_table(table, path, compression=None)
    import json
    pq.write_table(
        pq.read_table(path).replace_schema_metadata({
            b"huggingface": json.dumps(
                {"info": {"features": {"company_name": {"names": ["a", "b"]}}}}
            ).encode()}),
        path, compression=None,
    )


def test_export_keeps_every_box_or_drops_the_image(tmp_path):
    """Three in-range boxes must all survive; one out-of-range box must take
    the whole image with it, never be silently dropped."""
    src = tmp_path / "pq"
    src.mkdir()
    good = [[10, 10, 100, 100], [20, 20, 120, 120], [30, 30, 150, 150]]
    _shard(src / "a.parquet", [
        ("good", good),
        # 190x190 on a 200x200 frame = 0.9025 area, over max_area 0.6
        ("bad", [[5, 5, 195, 195]]),
    ])
    out = tmp_path / "out"
    tld.export_yolo(src, out, 0.15, 0.25, 0.0, 0.6, 0, max_boxes_per_image=0)

    labels = list((out / "labels").rglob("*.txt"))
    assert len(labels) == 1, "image with an unwritable box must be dropped entirely"
    lines = [l for l in labels[0].read_text().splitlines() if l.strip()]
    assert len(lines) == 3, f"all 3 in-range boxes must be labelled, got {len(lines)}"
    assert all(l.split()[0] == "0" for l in lines)


def test_export_drops_dense_images_instead_of_subsampling(tmp_path):
    """A 60-box image under a 50 cap must be dropped, not truncated to 50."""
    src = tmp_path / "pq"
    src.mkdir()
    dense = [[i, i, i + 20, i + 20] for i in range(0, 180, 3)]
    _shard(src / "a.parquet", [("dense", dense)])
    out = tmp_path / "out"
    tld.export_yolo(src, out, 0.15, 0.25, 0.0, 1.0, 0, max_boxes_per_image=50)
    assert not list((out / "labels").rglob("*.txt")), \
        "dense image must be dropped, never written with a truncated box list"
