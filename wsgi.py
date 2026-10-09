"""
WSGI entrypoint for production serving.

Run with:  gunicorn wsgi:application -c gunicorn.conf.py
The heavy ML models load lazily on first job. gunicorn.conf.py pins workers=1
(server.JOBS is process-local — see its comment), so one worker owns the single
pipeline copy; concurrency comes from its gthread threads.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from server import app as application  # noqa: E402

__all__ = ["application"]
