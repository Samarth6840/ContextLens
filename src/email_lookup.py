"""
Grounded brand-email lookup via Google Gemini (google_search tool).

Real need: the brand catalog's `contact_email` values are editorial placeholders.
For a DETECTED brand we want the brand's official public contact and HR/careers
addresses as they appear on real, citable web pages — not an LLM inventing or
guessing an address (the same fabrication rule that governs open-set).

Flow (same evidence contract as GeminiGroundedBackend in openset.py):
1. POST a prompt to ``generateContent`` with ``tools=[{"google_search": {}}]``.
   The prompt demands a compact JSON array of ``{email, type, source}`` and
   explicitly forbids inventing addresses.
2. The API returns the answer plus ``groundingMetadata`` (groundingChunks with
   real URIs + groundingSupports linking claims to those pages).
3. Only addresses that pass the deterministic validator AND appear alongside
   real grounding evidence are kept. The source URL for each address defaults to
   a grounding chunk URI (or is taken from the model's own `source` field when
   it matches a chunk URI).

Fail-closed contract:
  * no GEMINI_API_KEY (env, .env, or ~/.gemini_key) => unavailable, no emails.
  * no grounding evidence in the response => no emails (ungrounded output is
    treated as unverified guesswork, even if it parses).
  * any request/JSON/validation error => empty result (caller keeps the
    catalog placeholder or an explicit "no contact" state).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import requests

from src.openset import GEMINI_API_URL, _env

logger = __import__("logging").getLogger(__name__)

_SUPPORTED_TYPES = ("press", "pr", "contact", "info", "legal", "hr")

_PROMPT = (
    "You are an email-address finder. Find the OFFICIAL public contact email "
    "address(es) for the brand {brand} — press/pr/contact/info/legal addresses "
    "as well as any HR or careers contact. Use the google_search tool and base "
    "every address on a real page you retrieve. NEVER invent, guess, or "
    "synthesize an email address. "
    'Return ONLY a compact JSON object in exactly this shape, with no '
    'commentary or markdown fence:\n'
    '{{"emails": [{{"email": "<address>", "type": "press|pr|contact|info|legal|hr", '
    '"source": "<real published URL containing the address>"}}]}}\n'
    'Only include an address that literally appears on a page you retrieved in '
    'this search. If you find none, return {{"emails": []}}.'
)

# Validator + placeholder blacklist.
_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,63}$")
_PLACEHOLDER_PATTERNS = (
    re.compile(r"^(email|your|sender|name|user)@", re.I),
    re.compile(r"@(example|domain|email|yourbrand|yourcompany|brand)\.", re.I),
    re.compile(r"(\.invalid|\.test|\.example|\.local)$", re.I),
    re.compile(r"(^|@)([0-9]{1,3}\.){3}[0-9]{1,3}$"),
)

# Per-brand module cache (lives for the process — same discipline as the
# logo.dev validation cache).
_EMAIL_CACHE: Dict[str, dict] = {}


def available() -> bool:
    return bool(_api_key())


def _api_key() -> Optional[str]:
    key = _env("GEMINI_API_KEY")
    if not key:
        key_path = Path.home() / ".gemini_key"
        if key_path.is_file():
            try:
                key = key_path.read_text(encoding="utf-8").strip() or None
            except OSError:
                key = None
    return key


# Consumer webmail and mail providers. A brand's official contact is published
# on the brand's own domain, so an address at one of these cannot be one — yet
# every one of them passes _EMAIL_RE, and a grounded search surfaces them often
# (agency/aggregator pages list a contact's personal gmail as if it were the
# brand's). Rejecting them here is what stops a plausible-looking but wrong
# address reaching an outreach draft.
_PERSONAL_DOMAINS = frozenset({
    "gmail.com", "googlemail.com", "yahoo.com", "yahoo.co.uk", "ymail.com",
    "hotmail.com", "hotmail.co.uk", "outlook.com", "live.com", "msn.com",
    "aol.com", "icloud.com", "me.com", "mac.com", "gmx.com", "gmx.de",
    "mail.com", "zoho.com", "protonmail.com", "proton.me", "pm.me",
    "tutanota.com", "fastmail.com", "yandex.com", "qq.com", "163.com",
    "126.com", "rediffmail.com", "sify.com", "inbox.com", "hey.com",
})

# Subdomain of a personal provider ("mail.yahoo.com") is still that provider.
_PERSONAL_DOMAIN_SUFFIX = (".gmail.com", ".googlemail.com", ".yahoo.com",
                           ".hotmail.com", ".outlook.com", ".live.com")


def _is_personal_domain(domain: str) -> bool:
    d = (domain or "").strip().lower().rstrip(".")
    return d in _PERSONAL_DOMAINS or d.endswith(_PERSONAL_DOMAIN_SUFFIX)


def _is_valid_email(address: str, brand_domain: str = "") -> bool:
    address = (address or "").strip()
    if not address or not _EMAIL_RE.match(address):
        return False
    if ".." in address:
        return False
    if any(pat.search(address) for pat in _PLACEHOLDER_PATTERNS):
        return False
    if _is_personal_domain(address.rsplit("@", 1)[-1]):
        return False
    # When we know the brand's official domain, require a matching registrable
    # label. This is deliberately label-based rather than exact-domain: mail
    # often lives on a sibling TLD or host (sony.co.jp, sony.net, mail.sony.com),
    # and there is no shipped public-suffix list to tell a real ccTLD from a
    # crafted one. So the guarantee is "same second-level label", which rejects
    # notsony.com and sony.com.evil.net while accepting any sony.* TLD. It is a
    # brake, not proof of ownership.
    if brand_domain:
        want = _registrable_label(brand_domain)
        got = _registrable_label(address.rsplit("@", 1)[-1])
        if want and got and want != got:
            return False
    return True


def _brand_domain(brand: str) -> str:
    """Official domain for a catalog brand, or '' when unknown.

    A non-catalog brand has no curated domain, so the brand-domain check is
    skipped and only the personal-provider check applies. That is the right
    direction to fail: we still refuse recruiter@gmail.com, but we do not
    reject a legitimate address just because we never curated a domain.
    """
    try:
        from src.brand_catalog import contact_for
        site = (contact_for(brand) or {}).get("website") or ""
    except Exception:
        return ""
    m = re.search(r"https?://([^/]+)", site)
    return m.group(1) if m else ""


def _registrable_label(host: str) -> str:
    """Rough eTLD+1: 'mail.nike.co.uk' -> 'nike'."""
    parts = [p for p in (host or "").strip().lower().replace("www.", "").split(".")
             if p and p != "www"]
    if len(parts) < 2:
        return (parts[0] if parts else "")
    # Two-part public suffixes we care about; anything else takes the last two.
    two_part = {"co.uk", "com.au", "co.in", "co.jp", "com.br", "co.nz"}
    if len(parts) >= 3 and ".".join(parts[-2:]) in two_part:
        return parts[-3]
    return parts[-2]


def _classify_type(label: str, address: str) -> str:
    label = (label or "").strip().lower()
    blob = f"{label} {address}".lower()
    if re.search(r"\b(hr|careers|career|jobs|job|talent|recruit|hiring)\b", blob):
        return "hr"
    for t in _SUPPORTED_TYPES:
        if t in label:
            return t
    if "press" in label or ("press" in address):
        return "press"
    return "contact"


def _extract_json_emails(text: str) -> Optional[List[dict]]:
    """Pull the first balanced JSON object that has an \"emails\" list."""
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    data = json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    break
                if isinstance(data.get("emails"), list):
                    return data.get("emails")
    return None


def _grounding(urls: List[str], supports: List[dict]) -> Set[str]:
    """Collect real retrieved URLs from groundingMetadata."""
    found = set(urls)
    for s in supports or []:
        uris = (s.get("groundingChunks") or [])
        for chunk in uris or []:
            uri = ((chunk or {}).get("web") or {}).get("uri") or ""
            if uri:
                found.add(uri)
    return {u for u in found if u.startswith("http")}


def parse_brand_emails(text: str, urls: List[str], supports: List[dict],
                       brand_domain: str = "") -> List[dict]:
    """Deterministic, offline-testable parser for a Gemini grounded answer.

    Keeps only validated addresses that cite a real grounding URI. An answer
    with zero grounding URLs yields no addresses (ungrounded).

    No raw-text fallback: a scanned address has no per-address source, and
    attributing it to whichever page sorted first fabricates provenance for an
    address that may not appear on that page at all.
    """
    grounded = _grounding(urls, supports)

    items = _extract_json_emails(text)
    if items is None:
        return []

    seen: Set[str] = set()
    out: List[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        address = (item.get("email") or "").strip()
        if not _is_valid_email(address, brand_domain) or address in seen:
            continue
        # `source` comes from the model's own JSON, so it is untrusted input.
        # It is honoured only when it is a URI the API actually returned in
        # groundingMetadata. Matching a *prefix* is not enough: a fabricated
        # "https://vertexaisearch.cloud.google.com/grounding-api-redirect/xyz"
        # would otherwise pass and give a fabricated address a plausible-looking
        # provenance. Real redirect URLs are present in `urls` when Google emits
        # them, so exact membership still covers that citation format.
        source = (item.get("source") or "").strip()
        if not source or source not in grounded:
            continue
        seen.add(address)
        out.append({
            "email": address,
            "type": _classify_type(item.get("type"), address),
            "source": source,
        })
    return out


def lookup_brand_emails(brand: str, timeout: float = 120.0, model: Optional[str] = None) -> dict:
    """Run the grounded lookup for one brand. Fails closed on any problem."""
    key = (brand or "").strip().upper()
    if not key:
        return {"status": "error", "reason": "no brand"}
    if key in _EMAIL_CACHE:
        return _EMAIL_CACHE[key]

    api_key = _api_key()
    if not api_key:
        result = {"status": "unavailable", "reason": "no GEMINI_API_KEY", "emails": []}
        _EMAIL_CACHE[key] = result
        return result

    model = model or _env("GEMINI_MODEL") or "gemini-2.0-flash"
    payload = {
        "contents": [{
            "role": "user",
            "parts": [{"text": _PROMPT.format(brand=brand)}],
        }],
        "tools": [{"google_search": {}}],
        "generationConfig": {"temperature": 0.1},
    }
    resp = None
    last_exc = None
    for attempt in range(2):
        try:
            resp = requests.post(
                GEMINI_API_URL.format(model=model),
                params={"key": api_key},
                json=payload,
                timeout=timeout,
            )
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_exc = exc
            continue
        break
    if resp is None:
        # Transient (timeout/connection). Not cached: caching it makes a brand
        # permanently email-less for the life of the process after one network
        # blip, which is indistinguishable from a real "this brand has none".
        return {"status": "error", "reason": f"Gemini request failed: {last_exc}",
                "emails": [], "transient": True}
    if resp.status_code != 200:
        result = {"status": "error", "reason": f"Gemini HTTP {resp.status_code}", "emails": []}
        # 429 and 5xx are the server's problem, not the brand's — retry later
        # rather than memoising the failure. 4xx is a real config/permission
        # error and stays cached.
        if resp.status_code == 429 or resp.status_code >= 500:
            result["transient"] = True
            return result
        _EMAIL_CACHE[key] = result
        return result
    try:
        data = resp.json()
        cand = (data.get("candidates") or [{}])[0]
        parts = ((cand.get("content") or {}).get("parts")) or []
        text = " ".join(
            p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text")
        )
        metadata = cand.get("groundingMetadata") or {}
        urls = [
            ((chunk or {}).get("web") or {}).get("uri", "")
            for chunk in metadata.get("groundingChunks") or []
        ]
        supports = metadata.get("groundingSupports") or []

        emails = parse_brand_emails(text, urls, supports,
                                    brand_domain=_brand_domain(key))
        evidence_urls = {
            u for u in urls if u.startswith("http")
        } | {e["source"] for e in emails}
        result = {
            "status": "ok" if emails else "no_emails_found",
            "brand": key,
            "emails": emails,
            "hr_emails": [e["email"] for e in emails if e["type"] == "hr"],
            "evidence": [{"url": u} for u in sorted(evidence_urls)[:8]],
        }
    except Exception as exc:  # noqa: BLE001 — fail closed, never raise
        result = {"status": "error", "reason": str(exc)[:200], "emails": [],
                  "transient": True}
    _EMAIL_CACHE[key] = result
    return result


def clear_cache() -> None:
    _EMAIL_CACHE.clear()


def _set_env(key: str, value: str) -> None:
    """Test seam: inject an API key without touching the real environment."""
    os.environ[key] = value