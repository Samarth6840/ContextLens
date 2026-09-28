"""The auth flow must exist in the client, not just the server.

tests/test_server_auth.py proves the server gates correctly when AUTH_ENABLED is
forced on. It cannot see the UI, and this deployment ships AUTH_ENABLED = False,
so `login_required` is a pass-through and the whole server-side suite is blind to
a missing browser login path. That is exactly how view-login, nav-auth and the
bearer header were once dropped from a UI rewrite with every test still green.

These assertions are static on purpose: the auth flow is pure string work in
app.js, so a source contract catches the deletion without needing a DOM.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
JS = (ROOT / "static/app.js").read_text()
HTML = (ROOT / "static/index.html").read_text()
CSS = (ROOT / "static/styles.css").read_text()


@pytest.mark.parametrize("needle,why", [
    ("TOKEN_KEY", "token storage key"),
    ("getToken", "reads the stored token"),
    ("setToken", "writes and clears the stored token"),
    ("Authorization", "sends the bearer header"),
    ("Bearer", "bearer scheme"),
    ("AUTH REQUIRED", "recognises the server's 401 payload"),
    ("#/login", "redirects an expired session to the login view"),
    ("refreshNavAuth", "flips the nav link between Log in and Log out"),
    ("renderLogin", "renders the sign-in form"),
    ("/api/login", "exchanges credentials for a token"),
    ("/api/logout", "revokes the token"),
    ("/api/me", "reports who is signed in"),
])
def test_auth_flow_present_in_client(needle, why):
    assert needle in JS, f"client auth flow lost {needle} ({why})"


def test_login_view_present_in_markup():
    assert 'id="view-login"' in HTML, "the login view was dropped from index.html"
    assert 'id="login-root"' in HTML, "renderLogin has no mount point to write into"
    assert 'id="nav-auth"' in HTML, "no nav link to reach the login view"


def test_login_route_is_routed():
    assert "page === 'login'" in JS, "#/login is not handled by the hash router"


def test_el_map_includes_login():
    # showView() toggles .is-active via el[name]; a view missing from this map
    # never becomes visible even though the section exists in the markup.
    block = JS[JS.index("const el = {"):JS.index("};", JS.index("const el = {"))]
    assert "login:" in block, "showView('login') cannot find the view element"


def test_auth_state_has_a_visual_treatment():
    # is-authed is toggled on the nav link; with no rule the signed-in state is
    # indistinguishable from signed-out.
    assert ".nav-item.is-authed" in CSS, "signed-in nav state has no styling"
    assert ".auth-card" in CSS, "the login card is unstyled"
    assert ".auth-actions" in CSS, "the signed-in action row is unstyled"
