import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_src_bootstrap_sets_writable_numba_cache_before_librosa_import():
    env = os.environ.copy()
    env.pop("NUMBA_CACHE_DIR", None)
    env["PYTHONPATH"] = str(ROOT)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os, pathlib, src; "
            "p = pathlib.Path(os.environ['NUMBA_CACHE_DIR']); "
            "assert p.is_dir(); "
            "import librosa.core.notation; "
            "print(p)",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
    )

    assert result.returncode == 0, result.stderr
