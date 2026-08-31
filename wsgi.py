"""
WSGI entrypoint for production serving.

Run with:  gunicorn wsgi:app -c gunicorn.conf.py
(The heavy ML models load lazily inside each worker on first job, so multiple
workers each own an independent pipeline copy — see server.get_pipeline.)
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from server import app as application  # noqa: E402

__all__ = ["application"]
