import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_wsgi_application_exposes_flask_health():
    import wsgi

    assert callable(wsgi.application)
    client = wsgi.application.test_client()
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.get_json()["ok"] is True


def test_gunicorn_config_loads_and_binds():
    ns = {}
    with open("gunicorn.conf.py") as fh:
        code = compile(fh.read(), "gunicorn.conf.py", "exec")
        exec(code, ns)
    assert ns["worker_class"] == "gthread"
    # Hard-pinned, not merely defaulted: see gunicorn.conf.py.
    assert ns["workers"] == 1
    assert ns["threads"] >= 1
    assert ns["timeout"] >= 30
    assert "0.0.0.0" in ns["bind"]
    # No worker recycling. The dashboard polls ~1.5s, so any request budget is
    # exhausted in minutes and recycling kills the analysis thread plus the
    # in-memory JOBS dict — taking every running job with it.
    assert "max_requests" not in ns
    assert "max_requests_jitter" not in ns


def test_docker_and_procfile_present():
    assert Path("Dockerfile").is_file()
    assert Path("Procfile").is_file()
    assert Path(".dockerignore").is_file()


def test_dockerignore_excludes_secrets_and_models():
    text = Path(".dockerignore").read_text()
    for needle in (".env", "*.pt", "var/", "weights/"):
        assert needle in text, f".dockerignore missing '{needle}'"


def _exec_gunicorn_conf(env=None):
    """Run gunicorn.conf.py under a given environment, returning its namespace."""
    import os
    from unittest import mock
    ns = {}
    with open("gunicorn.conf.py") as fh:
        code = compile(fh.read(), "gunicorn.conf.py", "exec")
    with mock.patch.dict(os.environ, env or {}, clear=False):
        exec(code, ns)
    return ns


def test_gunicorn_refuses_more_than_one_worker():
    """server.JOBS is process-local and the pipeline runs as a daemon thread in
    the worker that got the request. A second worker 404s every submitted job
    and the results are unrecoverable, so booting with >1 must fail loudly
    rather than silently dropping jobs."""
    assert _exec_gunicorn_conf()["workers"] == 1
    assert _exec_gunicorn_conf({"ADSCENE_WORKERS": "1"})["workers"] == 1
    # Concurrency still available through threads.
    assert _exec_gunicorn_conf({"ADSCENE_THREADS": "8"})["threads"] == 8

    for bad in ("2", "4", "16"):
        try:
            _exec_gunicorn_conf({"ADSCENE_WORKERS": bad})
        except RuntimeError as exc:
            assert "process-local" in str(exc)
            assert "ADSCENE_THREADS" in str(exc)
        else:
            raise AssertionError(
                f"ADSCENE_WORKERS={bad} booted instead of refusing: jobs submitted "
                f"to one worker would 404 on another")
