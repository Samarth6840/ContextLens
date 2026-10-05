"""Production gunicorn configuration (gunicorn wsgi:application -c gunicorn.conf.py).

Every knob is driven by environment so the same config serves dev and prod.
The Flask app is thread-safe (per-model threading locks in the pipeline), so the
`gthread` worker class lets one worker host several concurrent analyse jobs —
appropriate given the heavy single-worker memory footprint of the ML models.
"""

import os


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


bind = f"0.0.0.0:{os.environ.get('ADSCENE_PORT', '5000')}"

# The ML models (YOLO, DINOv2, Whisper, BEATs) are large; keep worker count low
# to bound RAM, but allow threads to serve interleaved requests.
workers = _env_int("ADSCENE_WORKERS", 1)
threads = _env_int("ADSCENE_THREADS", 4)
worker_class = "gthread"

# Long-running analyse jobs can exceed a naive 30s timeout; give them headroom.
timeout = _env_int("ADSCENE_TIMEOUT", 300)
graceful_timeout = _env_int("ADSCENE_GRACEFUL_TIMEOUT", 60)

# NOTE: deliberately NO max_requests/max_requests_jitter. The dashboard polls
# /api/analyse/<id> every ~1.5s, so one open tab exhausts a 200-request budget in
# ~5 minutes. Recycling the worker kills the daemon analysis thread and discards
# the in-memory JOBS dict, taking every running job with it. Memory is bounded by
# workers=1 plus the model working set instead.

accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("ADSCENE_LOG_LEVEL", "info")
