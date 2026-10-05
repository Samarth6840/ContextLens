"""_locked_call must record wall time into the dict it is GIVEN.

Regression: the timing dict was a bare local name (`_model_wall_times = {}`)
in process_video, and _locked_call — a different method, running on a worker
thread — tried to read that name from its own scope. Every locked inference
call raised `NameError: name '_model_wall_times' is not defined`.

The per-modality `except Exception` in process_video swallowed it, so the run
continued with logo_detection, detection, embeddings and open_vocab ALL
silently empty, then died at the embedding guard with an unrelated
"Embedding extraction returned no results" error. No unit test caught it
because none exercised _locked_call.

This pins both halves: the call completes, and the timing actually lands in
the caller-owned dict.
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.pipeline import Phase1Pipeline


def test_locked_call_records_wall_time_and_returns_the_result():
    pipe = Phase1Pipeline()
    lock = threading.Lock()
    wall_times = {}

    got = pipe._locked_call(
        lock, "detector", lambda x: x * 2, wall_times, 21
    )

    assert got == 42, "locked call did not return fn's result"
    assert "detector" in wall_times, (
        "wall time never reached the caller's dict — the recording is "
        "decorative, so the parallelism diagnostic silently reports nothing"
    )
    assert wall_times["detector"] >= 0.0


def test_locked_submit_passes_the_dict_through_the_thread_boundary():
    """The bug lived on the worker-thread path. Going through submit() is what
    a real run does, and it is what failed in production."""
    pipe = Phase1Pipeline()
    wall_times = {}

    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = pipe._locked_submit(
            ex, "stt", lambda: "transcript", wall_times=wall_times
        )
        assert fut.result() == "transcript"

    assert "stt" in wall_times, (
        "submit() lost the wall_times dict on the way to the worker thread; "
        "every Layer 1 modality raises NameError instead of returning results"
    )


def test_wall_times_stays_per_job_not_shared():
    """Two concurrent jobs must not write into one dict, or the diagnostic
    attributes one job's model time to the other."""
    pipe = Phase1Pipeline()
    job_a, job_b = {}, {}

    with ThreadPoolExecutor(max_workers=2) as ex:
        fa = pipe._locked_submit(ex, "detector", lambda: 1, wall_times=job_a)
        fb = pipe._locked_submit(ex, "stt", lambda: 2, wall_times=job_b)
        assert (fa.result(), fb.result()) == (1, 2)

    assert "detector" in job_a and "stt" not in job_a
    assert "stt" in job_b and "detector" not in job_b


def test_locked_call_still_works_with_no_dict():
    """The synchronous product-resolve path passes None (it runs outside
    process_video). It must not crash."""
    pipe = Phase1Pipeline()
    assert pipe._locked_call(
        threading.Lock(), "central_vision", lambda: "ok", None
    ) == "ok"
