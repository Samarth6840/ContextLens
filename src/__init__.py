"""ContextLens package bootstrap."""

import os
import tempfile
from pathlib import Path


def _configure_numba_cache() -> None:
    """Give Numba a writable cache when dependencies live in read-only site-packages."""
    if os.environ.get("NUMBA_CACHE_DIR"):
        return

    cache_dir = Path.home() / ".cache" / "contextlens" / "numba"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        cache_dir = Path(tempfile.gettempdir()) / "contextlens-numba"
        cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ["NUMBA_CACHE_DIR"] = str(cache_dir)


_configure_numba_cache()
