from types import SimpleNamespace

import numpy as np

from src.pipeline import _run_speech_to_text


def test_transcription_returns_timed_text_and_backend_status():
    stt = SimpleNamespace(
        _backend="openai",
        transcribe_segments=lambda _audio: [
            {"text": "Hello Samsung", "start": 1.2, "end": 2.4},
        ],
    )

    transcript, segments, status = _run_speech_to_text(lambda: stt, np.zeros(32000))

    assert transcript == "Hello Samsung"
    assert segments[0]["start"] == 1.2
    assert status["status"] == "transcribed"
    assert status["backend"] == "openai"


def test_transcription_failure_is_returned_as_visible_status():
    def broken_stt():
        raise RuntimeError("Whisper checkpoint is unavailable")

    transcript, segments, status = _run_speech_to_text(broken_stt, np.zeros(32000))

    assert transcript == ""
    assert segments == []
    assert status["status"] == "failed"
    assert status["message"] == "Whisper checkpoint is unavailable"


def test_transcription_skips_model_when_audio_is_missing():
    def unexpected_model_load():
        raise AssertionError("model should not load")

    transcript, segments, status = _run_speech_to_text(unexpected_model_load, None)

    assert transcript == ""
    assert segments == []
    assert status["status"] == "no_audio"
