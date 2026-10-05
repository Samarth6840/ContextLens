from src.layer2.evidence_tracking import attach_track_ids, build_tracks


def _det(bbox, brand=None, conf=0.5):
    return {"bbox": bbox, "brand": brand, "confidence": conf}


def test_single_object_forms_one_track():
    frames = [
        [_det((100, 100, 160, 140), "NIKE")],
        [_det((103, 101, 163, 141), "NIKE")],
        [_det((106, 102, 166, 142), "NIKE")],
    ]
    tracks = build_tracks(frames)
    assert len(tracks) == 1
    assert tracks[0].length == 3
    assert tracks[0].dominant_brand() == "NIKE"


def test_distinct_objects_are_separate_tracks():
    frames = [
        [_det((0, 0, 50, 50), "NIKE"), _det((400, 300, 460, 350), "ADIDAS")],
        [_det((1, 0, 51, 50), "NIKE"), _det((401, 300, 461, 350), "ADIDAS")],
    ]
    tracks = build_tracks(frames)
    assert len(tracks) == 2
    assert {t.dominant_brand() for t in tracks} == {"NIKE", "ADIDAS"}


def test_gap_rejoins_within_max_gap():
    frames = [
        [_det((100, 100, 160, 140))],
        [],  # one-frame drop-out
        [_det((101, 100, 161, 140))],
    ]
    tracks = build_tracks(frames, max_gap=2)
    assert len(tracks) == 1
    assert tracks[0].length == 2


def test_gap_beyond_max_gap_starts_new_track():
    frames = [
        [_det((100, 100, 160, 140))],
        [], [], [],
        [_det((100, 100, 160, 140))],
    ]
    tracks = build_tracks(frames, max_gap=1)
    assert len(tracks) == 2


def test_attach_track_ids_marks_each_detection():
    frames = [
        [_det((100, 100, 160, 140), "NIKE")],
        [_det((102, 100, 162, 140), "NIKE")],
    ]
    out = attach_track_ids(frames)
    ids = [d["track_id"] for frame in out for d in frame]
    assert ids[0] == ids[1]
    persist = [d["track_persistence"] for frame in out for d in frame]
    assert persist == [2, 2]


def test_detections_without_bbox_are_ignored():
    frames = [[{"brand": "NIKE"}], [{"brand": "NIKE", "bbox": None}]]
    assert build_tracks(frames) == []
