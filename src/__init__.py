"""ContextLens package bootstrap."""

import os
import tempfile
from pathlib import Path


def _configure_numba_cache() -> None:
    """Give Numba a writable cache when dependencies live in read-only site-packages."""
    configured = os.environ.get("NUMBA_CACHE_DIR")
    candidates = ([Path(configured)] if configured else []) + [
        Path.home() / ".cache" / "contextlens" / "numba",
        Path(tempfile.gettempdir()) / "contextlens-numba",
    ]
    for cache_dir in candidates:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryFile(dir=cache_dir):
                pass
        except OSError:
            continue
        os.environ["NUMBA_CACHE_DIR"] = str(cache_dir)
        return

    # Imports remain usable on locked-down systems even when no disk cache can
    # be created. Numba then runs the decorated function without compiling it.
    os.environ["NUMBA_DISABLE_JIT"] = "1"


_configure_numba_cache()
