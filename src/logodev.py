"""
Logo.dev client — B2a reference logo bank + B2b brand validation.

Two independent responsibilities:

B2a — Reference logo bank. Official brand logos (fetched from logo.dev) act as
the visual reference set for region-proposal + CLIP retrieval logo detection
(see scripts/logo_detection_benchmark.py). Only catalog brands are fetched, so
the reference bank is bounded and reproducible.

B2b — Brand validation as a fabrication safeguard. Before a detected/suggested
brand is presented as real collaboration evidence, we ask logo.dev whether the
brand actually exists (has an official logo + domain). A brand that logo.dev
does not know about is flagged as UNVERIFIED and must never be presented as a
detected appearance. This closes the fabrication path identified in the Part A
incident: fabricated brand names can no longer reach an outreach email without
an external, authoritative existence check.

API key hygiene (hard rule):
  * Keys are read from the environment or a local .env file — NEVER committed.
  * .env is gitignored (see .gitignore). The client never logs, prints, or
    persists keys.
  * If no key is configured the client is "unavailable": validation returns a
    neutral UNVERIFIED status and the reference bank simply cannot be built.
    An unconfigured client must not be treated as "validated" — fail closed.

Endpoints used:
  * Search:   GET https://api.logo.dev/search?q=<query>   (Bearer secret key)
  * Logo img: GET https://img.logo.dev/<domain>?token=<publishable>&size=256
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

API_SEARCH_URL = "https://api.logo.dev/search"
IMG_BASE_URL = "https://img.logo.dev"

ENV_SECRET = "LOGO_DEV_SECRET_KEY"
ENV_PUBLISHABLE = "LOGO_DEV_PUBLISHABLE_TOKEN"


def _load_env_file(project_root: Optional[Path] = None) -> None:
    """Load .env into os.environ if present (keys stay process-local)."""
    root = project_root or Path(__file__).resolve().parent.parent
    env_path = root / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


def _get_secret_key() -> Optional[str]:
    _load_env_file()
    return os.environ.get(ENV_SECRET) or None


def _get_publishable_token() -> Optional[str]:
    _load_env_file()
    return os.environ.get(ENV_PUBLISHABLE) or None


class LogoDevClient:
    """Minimal logo.dev API client with fail-closed semantics."""

    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout
        self._secret = _get_secret_key()
        self._publishable = _get_publishable_token()

    # ── availability ────────────────────────────────────────────
    @property
    def available(self) -> bool:
        """True when a secret key is configured (search/validation usable)."""
        return bool(self._secret)

    @property
    def can_fetch_images(self) -> bool:
        """True when a publishable token is configured (image downloads)."""
        return bool(self._publishable)

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self._secret}"}

    # ── B2a: reference bank ─────────────────────────────────────
    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Search logo.dev for brands matching `query`.

        Returns raw result dicts (id/name/logo/domain fields). Raises
        RuntimeError when no secret key is configured (callers should gate on
        `available`), or when the API errors — the caller decides whether a
        failure means "unverified" (fail closed).
        """
        if not self.available:
            raise RuntimeError("LogoDevClient: no LOGO_DEV_SECRET_KEY configured")
        import requests

        resp = requests.get(
            API_SEARCH_URL,
            params={"q": query},
            headers=self._headers(),
            timeout=self.timeout,
        )
        if resp.status_code == 429:
            raise RuntimeError("Logo.dev rate limit (429) — no key or quota exhausted")
        resp.raise_for_status()
        return resp.json()

    def fetch_logo_bytes(self, domain: str, size: int = 256) -> Optional[bytes]:
        """
        Download an official logo image for `domain` from the CDN.

        Returns raw PNG bytes, or None when no publishable token is configured
        or the domain has no logo.
        """
        if not self.can_fetch_images:
            logger.warning("No LOGO_DEV_PUBLISHABLE_TOKEN — cannot fetch logo images")
            return None
        import requests

        url = f"{IMG_BASE_URL}/{domain}"
        resp = requests.get(
            url, params={"token": self._publishable, "size": size}, timeout=self.timeout
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        if not resp.content:
            return None
        return resp.content

    def build_reference_bank(
        self,
        brand_names: List[str],
        output_dir: Path,
        size: int = 256,
        overwrite: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch one official logo per brand into `output_dir/<BRAND>.png`.

        Args:
            brand_names: canonical brand names (must be resolvable to a domain
                         by logo.dev search; catalog brands map via search).
            output_dir: where logos are written (created if missing).
            size: pixel size requested from the CDN.
            overwrite: re-fetch even if a file already exists.

        Returns a dict brand -> {file, domain, source: 'logo.dev'} for brands
        that resolved; brands with no logo.dev entry are omitted (the caller
        can log them as UNVERIFIED).

        This method never persists secrets: only logo bytes + metadata go to
        disk.
        """
        if not self.available:
            raise RuntimeError("LogoDevClient: no LOGO_DEV_SECRET_KEY configured")
        output_dir.mkdir(parents=True, exist_ok=True)
        bank: Dict[str, Dict[str, Any]] = {}
        for brand in brand_names:
            target = output_dir / f"{brand}.png"
            if target.is_file() and not overwrite:
                bank[brand] = {
                    "file": str(target),
                    "domain": brand.lower().replace(" ", "-"),
                    "source": "logo.dev",
                }
                continue
            try:
                results = self.search(brand)
            except Exception as exc:  # noqa: BLE001
                logger.warning("logo.dev search failed for %s: %s", brand, exc)
                continue
            if not results:
                logger.warning("logo.dev has no entry for %s — UNVERIFIED", brand)
                continue
            # Prefer the top result whose name matches, else the first hit.
            top = results[0]
            domain = top.get("domain") or top.get("id") or ""
            if not domain:
                logger.warning("logo.dev result for %s has no domain — UNVERIFIED", brand)
                continue
            data = self.fetch_logo_bytes(domain, size=size)
            if data is None:
                logger.warning(
                    "logo.dev logo fetch failed for %s (%s) — UNVERIFIED", brand, domain
                )
                continue
            target.write_bytes(data)
            bank[brand] = {
                "file": str(target),
                "domain": domain,
                "source": "logo.dev",
            }
            logger.info("Reference bank: %s -> %s (%d bytes)", brand, domain, len(data))
        return bank

    # ── B2b: brand validation (fabrication safeguard) ──────────
    def validate_brand(self, brand: str) -> Dict[str, Any]:
        """
        Check whether `brand` exists in logo.dev (authoritative existence).

        Returns one of:
          {"status": "verified", "brand": <logo.dev name>, "domain": <domain>}
          {"status": "unverified", ...}   # logo.dev has no entry for the name
          {"status": "unavailable", ...}  # no secret key / API error (fail closed)

        Fabrication rule for callers: only status "verified" may be presented
        as a real brand. "unavailable" MUST NOT be promoted to "verified".
        """
        if not self.available:
            return {"status": "unavailable", "brand": brand, "domain": None}
        try:
            results = self.search(brand)
        except Exception as exc:  # noqa: BLE001
            logger.warning("logo.dev validation failed for %s: %s", brand, exc)
            return {"status": "unavailable", "brand": brand, "domain": None}
        if not results:
            return {"status": "unverified", "brand": brand, "domain": None}
        top = results[0]
        return {
            "status": "verified",
            "brand": top.get("name") or brand,
            "domain": top.get("domain"),
        }
