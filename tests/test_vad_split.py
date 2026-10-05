"""VAD segmentation must actually close segments.

Regression: `silence_len` was computed as `frame_start - (i-1)*hop_len - frame_len`,
which is always `-hop_len`, so the "close the segment" branch was unreachable and
every run collapsed into one segment chopped at max_segment_s. The anti-
hallucination measure (fresh decoder state per segment) silently never fired.
"""

import numpy as np
import pytest

from src.layer1.audio import SpeechToText

SR = 16000


def _tone(seconds: float, rng, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * 220 * t) + rng.normal(0, 0.01, int(SR * seconds))).astype(np.float32)


def _floor(seconds: float, rng) -> np.ndarray:
    return rng.normal(0, 0.01, int(SR * seconds)).astype(np.float32)


def test_separated_bursts_produce_separate_segments():
    rng = np.random.default_rng(0)
    audio = np.concatenate([_tone(1.0, rng), _floor(4.0, rng), _tone(1.0, rng), _floor(2.0, rng)])

    segments = SpeechToText._vad_split(audio, SR)

    assert len(segments) == 2, [(s / SR, e / SR) for s, e in segments]


def test_short_pause_does_not_split():
    rng = np.random.default_rng(0)
    audio = np.concatenate([_tone(1.0, rng), _floor(0.1, rng), _tone(1.0, rng)])

    segments = SpeechToText._vad_split(audio, SR)

    assert len(segments) == 1


def test_long_run_is_capped_at_max_segment_s():
    rng = np.random.default_rng(0)
    audio = np.concatenate([_floor(8.0, rng), _tone(40.0, rng), _floor(2.0, rng)])

    segments = SpeechToText._vad_split(audio, SR)

    assert len(segments) > 1
    assert all(e - s <= SR * 30.0 for s, e in segments)


@pytest.mark.parametrize("n", [0, 1, 100])
def test_short_or_empty_audio_does_not_raise(n):
    rng = np.random.default_rng(0)
    audio = np.zeros(n, dtype=np.float32) if n == 0 else _tone(n / SR, rng)

    assert all(s < e for s, e in SpeechToText._vad_split(audio, SR))
