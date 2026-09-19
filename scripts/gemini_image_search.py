#!/usr/bin/env python3
"""Reverse-image search / image identification via Gemini (google_search grounding).

Usage:
    python3 scripts/gemini_image_search.py <image_url>
    python3 scripts/gemini_image_search.py /local/file.png
    MODEL=gemini-2.0-flash python3 scripts/gemini_image_search.py <image_url> "extra question"

Key: $GEMINI_API_KEY or ~/.gemini_key (chmod 600)
"""

import base64
import json
import os
import sys
import tempfile

import requests

API = "https://generativelanguage.googleapis.com/v1beta"


def get_key():
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key.strip()
    p = os.path.expanduser("~/.gemini_key")
    if os.path.exists(p):
        with open(p) as f:
            return f.read().strip()
    sys.exit("No API key. Set GEMINI_API_KEY or write key to ~/.gemini_key")


def load_image(src):
    if src.startswith(("http://", "https://")):
        r = requests.get(src, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        return r.content, r.headers.get("Content-Type", "image/jpeg").split(";")[0]
    with open(src, "rb") as f:
        return f.read(), "image/jpeg"


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    src = sys.argv[1]
    extra = sys.argv[2] if len(sys.argv) > 2 else ""
    model = os.environ.get("MODEL", "gemini-2.0-flash")
    key = get_key()

    data, mime = load_image(src)
    print(f"Image loaded: {len(data)} bytes ({mime})", file=sys.stderr)

    prompt = (
        "This is a reverse-image search task. Identify this image precisely: "
        "what it depicts, the source/creator if recognizable, and any notable "
        "objects, logos, text, or landmarks. Then use the google_search tool to "
        "look up and verify your findings, and report the best sources you found."
        + (f"\nAdditional user question: {extra}" if extra else "")
    )

    payload = {
        "contents": [{
            "role": "user",
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": mime, "data": base64.b64encode(data).decode()}},
            ],
        }],
        "tools": [{"google_search": {}}],
    }

    resp = requests.post(
        f"{API}/models/{model}:generateContent",
        params={"key": key},
        json=payload,
        timeout=120,
    )
    if resp.status_code != 200:
        sys.exit(f"API error {resp.status_code}: {resp.text[:500]}")

    out = resp.json()
    parts = out["candidates"][0]["content"]["parts"]
    for p in parts:
        if "text" in p:
            print(p["text"])
        if "grounding_metadata" in p or "groundingMetadata" in out.get("candidates", [{}])[0]:
            gm = out["candidates"][0].get("groundingMetadata", {})
            for cs in gm.get("groundingChunks", []):
                web = cs.get("web", {})
                print(f"- {web.get('title', '')}: {web.get('uri', '')}")
        if "thought" in p:
            print(f"\n[thought] {p['thought']}")


if __name__ == "__main__":
    main()