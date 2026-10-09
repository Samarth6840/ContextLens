"""Production gunicorn configuration (gunicorn wsgi:application -c gunicorn.conf.py).

Most knobs are environment-driven so the same config serves dev and prod. The
one exception is `workers`, which is hard-pinned to 1 because the job registry
is process-local — see the comment at its definition.

The Flask app is thread-safe (per-model threading locks in the pipeline), so the
`gthread` worker class lets one worker host several concurrent analyse jobs —
appropriate given the heavy single-worker memory footprint of the ML models.
Concurrency comes from CONTEXTLENS_THREADS, not from workers.
"""

import os


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# Platforms (Railway/Heroku/Fly) inject PORT; CONTEXTLENS_PORT is the local/dev
# override. PORT wins so a managed deploy binds where the platform expects.
bind = f"0.0.0.0:{os.environ.get('PORT') or os.environ.get('CONTEXTLENS_PORT', '5000')}"

# The ML models (YOLO, DINOv2, Whisper, BEATs) are large; keep worker count low
# to bound RAM, but allow threads to serve interleaved requests.
#
# HARD-PINNED at 1, not configurable. server.JOBS is a process-local dict and
# the pipeline is started as a daemon thread inside the worker that received
# the request, so a second worker cannot see jobs submitted to the first:
# /api/analyse returns an id the other worker 404s on, and the job's progress
# and results are simply gone. Scaling out needs a shared job store (Redis or a
# DB table) first, not a worker count.
_requested_workers = _env_int("CONTEXTLENS_WORKERS", 1)
if _requested_workers != 1:
    raise RuntimeError(
        f"CONTEXTLENS_WORKERS={_requested_workers} is not supported: server.JOBS is "
        f"process-local and daemon analysis threads are invisible to other "
        f"workers, so submitted jobs 404 and their results are lost. Fixing "
        f"this requires a shared job store (Redis/DB), not more workers. "
        f"Run with CONTEXTLENS_WORKERS=1 and raise CONTEXTLENS_THREADS for concurrency."
    )
workers = 1
threads = _env_int("CONTEXTLENS_THREADS", 4)
worker_class = "gthread"

# Long-running analyse jobs can exceed a naive 30s timeout; give them headroom.
timeout = _env_int("CONTEXTLENS_TIMEOUT", 300)
graceful_timeout = _env_int("CONTEXTLENS_GRACEFUL_TIMEOUT", 60)

# NOTE: deliberately NO max_requests/max_requests_jitter. The dashboard polls
# /api/analyse/<id> every ~1.5s, so one open tab exhausts a 200-request budget in
# ~5 minutes. Recycling the worker kills the daemon analysis thread and discards
# the in-memory JOBS dict, taking every running job with it. Memory is bounded by
# workers=1 plus the model working set instead.

accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("CONTEXTLENS_LOG_LEVEL", "info")
