#!/usr/bin/env python3
"""Static UI contract check.

Guards the two failure modes a stylesheet rewrite actually produces: a class the
app emits that no rule mentions (silent unstyled markup) and a custom property
the app reads that no longer exists (silent dropped styling).

This is coverage, not selector integrity — a rule renamed while another selector
still mentions the class will pass.

    python3 scripts/check_ui_contract.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
css = (ROOT / "static/styles.css").read_text()
html = (ROOT / "static/index.html").read_text()
js = (ROOT / "static/app.js").read_text()

# Classes built at runtime and toggled by script, never present in a literal
# class="..." attribute.
DYNAMIC = {
    "is-active", "is-done", "is-checked", "has-file", "is-dragover",
    "error", "blink", "forwarded", "scene-highlight", "scene-flash",
    "scene-row--has-unresolved", "chip--brand", "chip--object", "chip--unknown",
    "chip--audio-real", "chip--audio-fallback", "chip-accent", "chip-ghost",
    "filter-item--brand", "filter-item--object",
}

failures = []


def classes(markup: str) -> set[str]:
    found = set()
    for attr in re.findall(r'class="([^"]*)"', markup):
        # Drop ${...} template expressions, whose bare identifiers are JS, not
        # class names: chip${extra ? ' chip--x' : ''} yields chip and chip--x.
        for name in re.sub(r"\$\{[^}]*\}", " ", attr).split():
            if re.fullmatch(r"[a-zA-Z][\w-]*", name):
                found.add(name)
    return found


used = classes(html) | classes(js) | DYNAMIC
defined = set(re.findall(r"\.([a-zA-Z][\w-]*)", css))

for name in sorted(used - defined):
    failures.append(f"class .{name} is used but has no rule in styles.css")

needed = set(re.findall(r"var\((--[\w-]+)", html + js))
available = set(re.findall(r"(--[\w-]+)\s*:", css))
for name in sorted(needed - available):
    failures.append(f"{name} is read by the app but not defined in styles.css")

if css.count("{") != css.count("}"):
    failures.append("styles.css has unbalanced braces")

if failures:
    print("\n".join(failures))
    sys.exit(1)
print(f"ok — {len(used)} classes styled, {len(needed)} custom properties defined")
