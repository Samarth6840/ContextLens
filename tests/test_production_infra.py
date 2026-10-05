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
    assert ns["workers"] >= 1
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
