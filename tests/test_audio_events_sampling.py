"""BEATs chunk sampling caps the audio-events wall time without losing coverage.

Uncapped BEATs on a 50-minute audio track = ~100 chunks x ~12 s = 20+ minutes
of silent wall time in one supplementary stage (job BNVRDTNO-X9J53 measured).
max_chunks caps the pass with UNIFORM sampling: real inference, real
timestamps, first/last chunk always included.
"""
import numpy as np
import pytest
import torch

from src.layer1.audio import AudioEventDetector

SR = 16000
CHUNK_S = 30


class _StubModel:
    """Records chunk start times; returns a dim-3 feature (fallback branch)."""

    def __init__(self):
        self.starts = []

    def extract_features(self, chunk_tensor):
        n = chunk_tensor.shape[1]
        self.starts.append(round(n / SR, 3))
        return torch.zeros(1, 10, 4), None


def _detector_with(seconds: float) -> tuple:
    det = AudioEventDetector.__new__(AudioEventDetector)
    det.device = "cpu"
    det.sample_rate = SR
    det.model = _StubModel()
    audio = np.zeros(int(SR * seconds), dtype=np.float32)
    return det, audio


import re

_T_START = re.compile(r"chunk \d+/\d+(?: of \d+)? \(([\d.]+)\u2013")


def _chunk_starts(det, audio, **kwargs):
    # Chunk offsets are read from the progress lines (they carry the REAL
    # start second computed from the offset). The stub model only sees chunk
    # tensors, which carry no position — with sampling, chunks are not
    # contiguous, so offsets cannot be reconstructed from lengths.
    msgs = []
    kwargs.setdefault("progress", msgs.append)
    det.detect_events(audio, max_chunk_seconds=CHUNK_S, **kwargs)
    return [float(_T_START.search(m).group(1)) for m in msgs]


def test_no_cap_processes_every_chunk():
    det, audio = _detector_with(300.0)  # 10 full chunks
    starts = _chunk_starts(det, audio)
    assert len(starts) == 10
    assert starts == [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 240.0, 270.0]


def test_cap_samples_uniformly_and_keeps_first_last():
    det, audio = _detector_with(3000.0)  # 100 chunks
    starts = _chunk_starts(det, audio, max_chunks=20)
    assert len(starts) == 20
    assert starts[0] == 0.0, "first chunk must always be analysed"
    assert starts[-1] == 2970.0, "last full chunk must always be analysed"
    # uniform coverage: gaps between consecutive picks are near-equal
    gaps = np.diff(starts)
    assert gaps.max() - gaps.min() <= CHUNK_S, f"gaps not uniform: {gaps}"
    # spread spans the full duration, not just the opening
    assert starts[-1] >= 0.9 * 2970.0


def test_cap_above_chunk_count_is_a_noop():
    det, audio = _detector_with(300.0)  # 10 chunks
    starts = _chunk_starts(det, audio, max_chunks=50)
    assert len(starts) == 10


def test_cap_none_is_legacy_full_pass():
    det, audio = _detector_with(300.0)
    starts = _chunk_starts(det, audio, max_chunks=None)
    assert len(starts) == 10


def test_progress_line_reflects_sampling():
    det, audio = _detector_with(3000.0)  # 100 chunks -> 20 picked
    msgs = []
    det.detect_events(audio, max_chunk_seconds=CHUNK_S, max_chunks=20,
                      progress=msgs.append)
    assert len(msgs) == 20
    assert msgs[0].startswith("Audio events — chunk 1/20 of 100 ")
    assert "0.0–30.0 s of audio" in msgs[0]
    assert msgs[-1].startswith("Audio events — chunk 20/20 of 100 ")


def test_short_tail_chunk_is_excluded_from_total():
    # 75 s = chunks [0-30, 30-60, 60-75]; the 15 s tail is >= 0.5 s so it IS
    # processed (the 0.5 s floor only drops sub-half-second slivers).
    det, audio = _detector_with(75.0)
    msgs = []
    det.detect_events(audio, max_chunk_seconds=CHUNK_S, progress=msgs.append)
    assert len(msgs) == 3
    assert msgs[-1].startswith("Audio events — chunk 3/3 ")


def test_progress_callback_failure_does_not_kill_detection():
    det, audio = _detector_with(75.0)

    def _boom(_msg):
        raise RuntimeError("callback exploded")

    events = det.detect_events(audio, max_chunk_seconds=CHUNK_S, progress=_boom)
    assert isinstance(events, list)  # ran to completion


def test_empty_audio_returns_no_events():
    det, audio = _detector_with(0.0)
    assert det.detect_events(audio, max_chunks=5) == []
