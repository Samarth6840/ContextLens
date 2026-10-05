"""Long analyses must stay observable in the browser, not freeze at 15 minutes.

The original pollJob stopped after 600 attempts (600 x 1.5s = 15 min). Long
videos legitimately run 30-90+ minutes (uncapped Whisper + per-chunk BEATs on
MPS), and the server keeps the job running after the tab stops polling — so the
page froze mid-analysis and the result site "never loaded". These assertions
are static on purpose (house style: tests/test_ui_auth_wiring.py): pollJob is
pure string/DOM work in app.js, so a source contract catches the regression
without needing a browser.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
JS = (ROOT / "static/app.js").read_text()
HTML = (ROOT / "static/index.html").read_text()

POLL_BLOCK_START = JS.index("function pollJob(")
# End the block at the next top-level declaration — `async function` does NOT
# match "\nfunction ", so slicing on that alone would swallow unrelated code
# and let needles pass even if pollJob itself lost them.
_next = min(
    i for needle in ("\nfunction ", "\nasync function ", "\nconst ", "\nlet ", "\n/* ")
    for i in [JS.find(needle, POLL_BLOCK_START + 1)] if i != -1
)
POLL_BLOCK = JS[POLL_BLOCK_START:_next]


def test_poll_job_block_exists():
    assert "async () =>" in POLL_BLOCK or "async (" in POLL_BLOCK, (
        "pollJob body was replaced with a non-async stub"
    )


@pytest.mark.parametrize("needle,why", [
    ("setInterval(tick, POLL_MS)", "polling must be a repeating interval, not a bounded loop"),
    ("MAX_CONSECUTIVE_FAILURES", "giving up must key on server reachability, not elapsed time"),
    ("fmtElapsed", "elapsed time must be computed for the live timer"),
    ("Elapsed:", "the elapsed line must be rendered"),
    ("Date.now() - startedAt", "elapsed must derive from a captured start time"),
    ("procFeed(job.feed)", "the live feed must keep updating while running"),
    ("setStatus(", "the status line must keep updating while running"),
    ("Done in ", "completion should report how long the job took"),
    ("reload the page to check", "the give-up message must tell the user the job may still finish"),
])
def test_poll_survives_long_jobs(needle, why):
    assert needle in POLL_BLOCK, f"pollJob lost {needle} ({why})"


def test_no_attempt_cap_regression():
    # The old bug: `if (attempts < 600) setTimeout(tick, 1500)` silently
    # abandoned every job longer than 15 minutes.
    assert "attempts < 600" not in POLL_BLOCK, (
        "pollJob regressed to the 600-attempt (~15 min) cap that froze long jobs"
    )
    assert "setTimeout(tick" not in POLL_BLOCK, (
        "pollJob must not re-arm via stacked setTimeout (use the single setInterval)"
    )


def test_cache_buster_bumped_past_v12():
    # Browsers cache app.js aggressively; if index.html pins ?v=12 or lower,
    # users keep running the freezing pollJob no matter what app.js contains.
    # (v12 was the shipping version when the 15-min cap existed.)
    version = HTML.split("app.js?v=")[1].split('"')[0]
    assert int(version) >= 13, (
        f"index.html still serves app.js?v={version}; the polling fix shipped as v=13"
    )


def test_audio_chunk_progress_reaches_the_feed():
    # The audio-events stage was the longest silent stage (uncapped BEATs);
    # its per-chunk lines are what the elapsed timer + feed exist to show.
    audio = (ROOT / "src/layer1/audio.py").read_text()
    pipeline = (ROOT / "src/pipeline.py").read_text()
    assert "Audio events — chunk" in audio, (
        "BEATs per-chunk progress line was removed from detect_events"
    )
    assert 'progress=_beats_progress' in pipeline, (
        "pipeline no longer wires the job feed into the audio-events pass"
    )
