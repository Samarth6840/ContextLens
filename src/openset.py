"""
Open-set brand identification for brands outside the fixed catalogue.

Real need: startup / DTC / regional brands won't exist in LogoDet-3K, the
knowledge graph, or any fixed catalogue. This module builds an evidence-based
path for identifying them — NOT an LLM guessing a plausible brand name, which
would reintroduce exactly the fabrication problem this remediation exists to
fix.

Flow (escalation Task 3):
1. A real detector (YOLO-World) finds a logo-shaped region with real confidence
   >= min_logo_confidence. Brand labels from the detector are NOT trusted.
2. The region is treated as an "unknown brand candidate": never silently
   dropped, never labeled with a guessed name.
3. The cropped region is sent to a REAL reverse-image-search / grounded
   web-search backend (Google Cloud Vision Logo/Web Detection, SerpApi Google
   Lens reverse image search, Google Gemini with google_search grounding, or a
   browser-driven grounded search). The
   search must return real, citable source URLs — never a free-text guess.
4. The candidate name is cross-validated against logo.dev's search endpoint
   (the Part A B2b safeguard). A name is only "surfaced as a detection" when
   logo.dev resolves it to a real registered brand/domain.
5. Every surfaced candidate carries its evidence trail: the cropped image,
   the search result(s) with source links, and the logo.dev validation result.

Cost guard (Task 3.6): reverse-image search and logo.dev calls cost money per
call, so this only fires on logo-shaped regions the primary detector found
with real confidence above a documented threshold, deduplicated across
near-identical crops, capped per video, and cached per crop hash. Fail closed
without keys — no key means no names are surfaced, ever.
"""

from __future__ import annotations

import base64
from contextlib import contextmanager
import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.brand_catalog import match_brand_wordmark

logger = logging.getLogger(__name__)


def _safe_external_error(error: object) -> str:
    """Redact query-string credentials from third-party exception text."""
    message = str(error)
    return re.sub(
        r"([?&](?:key|api_key|token)=)[^&\s]+",
        r"\1[REDACTED]",
        message,
        flags=re.IGNORECASE,
    )[:600]

# Backend -> env var that unlocks it (None = no key required).
# This is a mapping of real backend identifiers to the real env vars that
# carry their API keys — a routing table, not a mock and not tunable data.
BACKEND_ENV = {
    "google_vision_web": "GOOGLE_CLOUD_VISION_API_KEY",
    "bing_visual": "BING_VISUAL_SEARCH_KEY",
    "serpapi": "SERPAPI_KEY",
    "browser_grounded": None,
    "gemini_grounded": "GEMINI_API_KEY",
}

GOOGLE_VISION_URL = "https://vision.googleapis.com/v1/images:annotate"
SERPAPI_URL = "https://serpapi.com/search.json"
SERPAPI_IMAGE_URL = "https://serpapi.com/image"
GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)


class SerpApiBudgetExceeded(RuntimeError):
    """Raised before a Google Lens query would exceed the configured monthly cap."""


class OpenSetSearchStore:
    """Persistent, cross-process cache and monthly SerpApi query ledger."""

    def __init__(self, path, cache_ttl_days: int = 180):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_ttl_days = max(1, int(cache_ttl_days))
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute(
                "CREATE TABLE IF NOT EXISTS reverse_image_cache ("
                "cache_key TEXT PRIMARY KEY, results_json TEXT NOT NULL, "
                "expires_at REAL NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS serpapi_monthly_usage ("
                "month TEXT PRIMARY KEY, used INTEGER NOT NULL CHECK (used >= 0))"
            )

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(str(self.path), timeout=15, isolation_level=None)
        db.execute("PRAGMA busy_timeout=15000")
        try:
            yield db
        finally:
            db.close()

    @staticmethod
    def _month(now=None) -> str:
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        return current.astimezone(timezone.utc).strftime("%Y-%m")

    def serpapi_status(self, monthly_limit: int, now=None) -> Dict[str, Any]:
        limit = max(0, int(monthly_limit))
        month = self._month(now)
        with self._connect() as db:
            row = db.execute(
                "SELECT used FROM serpapi_monthly_usage WHERE month = ?", (month,)
            ).fetchone()
        used = int(row[0]) if row else 0
        return {
            "period": month,
            "used": used,
            "limit": limit,
            "remaining": max(0, limit - used),
            "exhausted": used >= limit,
        }

    def reserve_serpapi_search(self, monthly_limit: int, now=None) -> Dict[str, Any]:
        """Atomically claim one search slot across threads and server processes."""
        limit = max(0, int(monthly_limit))
        month = self._month(now)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT used FROM serpapi_monthly_usage WHERE month = ?", (month,)
            ).fetchone()
            used = int(row[0]) if row else 0
            if used >= limit:
                db.execute("COMMIT")
                return {
                    "allowed": False, "period": month, "used": used,
                    "limit": limit, "remaining": 0, "exhausted": True,
                }
            used += 1
            db.execute(
                "INSERT INTO serpapi_monthly_usage(month, used) VALUES (?, ?) "
                "ON CONFLICT(month) DO UPDATE SET used = excluded.used",
                (month, used),
            )
            db.execute("COMMIT")
        return {
            "allowed": True, "period": month, "used": used,
            "limit": limit, "remaining": max(0, limit - used),
            "exhausted": used >= limit,
        }

    def get_results(self, cache_key: str) -> Optional[List["ReverseImageResult"]]:
        now = time.time()
        with self._connect() as db:
            row = db.execute(
                "SELECT results_json, expires_at FROM reverse_image_cache "
                "WHERE cache_key = ?", (cache_key,),
            ).fetchone()
            if row and float(row[1]) <= now:
                db.execute("DELETE FROM reverse_image_cache WHERE cache_key = ?", (cache_key,))
                row = None
        if row is None:
            return None
        try:
            return [ReverseImageResult(**item) for item in json.loads(row[0])]
        except (TypeError, ValueError, KeyError, json.JSONDecodeError):
            with self._connect() as db:
                db.execute("DELETE FROM reverse_image_cache WHERE cache_key = ?", (cache_key,))
            return None

    def put_results(self, cache_key: str, results: List["ReverseImageResult"]) -> None:
        payload = json.dumps([result.to_dict() for result in results], ensure_ascii=False)
        expires_at = time.time() + self.cache_ttl_days * 24 * 60 * 60
        with self._connect() as db:
            db.execute(
                "INSERT INTO reverse_image_cache(cache_key, results_json, expires_at) "
                "VALUES (?, ?, ?) ON CONFLICT(cache_key) DO UPDATE SET "
                "results_json=excluded.results_json, expires_at=excluded.expires_at",
                (cache_key, payload, expires_at),
            )


class ReverseImageResult:
    """One real, citable search result from a reverse-image-search backend."""

    __slots__ = ("title", "url", "source", "thumbnail_url")

    def __init__(self, title: str, url: str, source: str, thumbnail_url: str = ""):
        self.title = title
        self.url = url
        self.source = source
        self.thumbnail_url = thumbnail_url

    def to_dict(self) -> Dict[str, str]:
        return {
            "title": self.title,
            "url": self.url,
            "source": self.source,
            "thumbnail_url": self.thumbnail_url,
        }


class ReverseImageSearchBackend(ABC):
    """Abstract reverse-image-search backend (real API or grounded browser)."""

    name = "abstract"

    @property
    def available(self) -> bool:
        return True

    @abstractmethod
    def search_crop(self, crop: np.ndarray, context_text: str = "") -> List[ReverseImageResult]:
        """Return real search results for a cropped logo image (bytes upload)."""
        raise NotImplementedError


class FallbackReverseImageSearchBackend(ReverseImageSearchBackend):
    """Try configured grounded search providers in order until one responds."""

    name = "auto"

    def __init__(self, backends: List[ReverseImageSearchBackend]):
        self.backends = [backend for backend in backends if backend.available]

    @property
    def available(self) -> bool:
        return any(backend.available for backend in self.backends)

    def provider_usage(self) -> Dict[str, Dict[str, Any]]:
        usage = {}
        for backend in self.backends:
            status = getattr(backend, "usage_status", None)
            if callable(status):
                usage[backend.name] = status()
        return usage

    def search_crop(self, crop: np.ndarray, context_text: str = "") -> List[ReverseImageResult]:
        errors = []
        for backend in self.backends:
            if not backend.available:
                continue
            try:
                if context_text and isinstance(backend, GeminiGroundedBackend):
                    results = backend.search_crop(crop, context_text=context_text)
                else:
                    results = backend.search_crop(crop)
                # A model's bare answer/description is not grounding. A direct
                # Google logo annotation is independent visual evidence; all
                # other label-only replies need a citable page URL.
                direct_logo_evidence = (
                    backend.name == "google_vision_web"
                    and any(r.source == "google_vision_logo_tags" for r in results)
                )
                if results and (any(r.url for r in results) or direct_logo_evidence):
                    logger.info("Open-set search used %s", backend.name)
                    return results
                if results:
                    errors.append(f"{backend.name}: response had no citable source URL")
            except Exception as exc:  # noqa: BLE001 — try the next real provider
                safe_error = _safe_external_error(exc)
                errors.append(f"{backend.name}: {safe_error}")
                logger.warning("Open-set provider %s failed: %s", backend.name, safe_error)
        if errors:
            raise RuntimeError("All grounded search providers failed: " + "; ".join(errors))
        return []

    def search_text(self, text: str) -> List[ReverseImageResult]:
        """Use text-grounding providers first when OCR recovered a wordmark."""
        errors = []
        for backend in self.backends:
            if not backend.available:
                continue
            search_text = getattr(backend, "search_text", None)
            if not callable(search_text):
                continue
            try:
                results = search_text(text)
                if results and any(r.url for r in results):
                    logger.info("Open-set text search used %s", backend.name)
                    return results
                if results:
                    errors.append(f"{backend.name}: response had no citable source URL")
            except Exception as exc:  # noqa: BLE001 — try the next text provider
                safe_error = _safe_external_error(exc)
                errors.append(f"{backend.name}: {safe_error}")
        if errors:
            logger.info("Grounded text search produced no sources: %s", "; ".join(errors))
        return []


_ENV_LOADED = False


def _load_env_file(project_root: Optional[Path] = None) -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
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


def _env(key: str) -> Optional[str]:
    _load_env_file()
    return os.environ.get(key) or None


def _img_bytes(crop: np.ndarray) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".png", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("could not encode crop image")
    return buf.tobytes()


class GoogleVisionWebBackend(ReverseImageSearchBackend):
    """Google Cloud Vision Web Detection (real API, needs API key)."""

    name = "google_vision_web"

    def __init__(self, timeout: float = 20.0):
        self.timeout = timeout
        self._key = _env(BACKEND_ENV[self.name])
        self._disabled = False

    @property
    def available(self) -> bool:
        return bool(self._key) and not self._disabled

    def search_crop(self, crop: np.ndarray, context_text: str = "") -> List[ReverseImageResult]:
        if not self.available:
            raise RuntimeError("Google Cloud Vision: no GOOGLE_CLOUD_VISION_API_KEY")
        import requests

        b64 = base64.b64encode(_img_bytes(crop)).decode("ascii")
        try:
            resp = requests.post(
                GOOGLE_VISION_URL,
                params={"key": self._key},
                json={
                    "requests": [{
                        "image": {"content": b64},
                    "features": [
                        {"type": "LOGO_DETECTION", "maxResults": 8},
                        {"type": "WEB_DETECTION", "maxResults": 6},
                    ],
                    }]
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except Exception as exc:
            if getattr(getattr(exc, "response", None), "status_code", None) in {401, 403, 404}:
                self._disabled = True
            raise
        data = resp.json()
        web = (((data.get("responses") or [{}])[0]).get("webDetection")) or {}
        image_response = ((data.get("responses") or [{}])[0])
        results: List[ReverseImageResult] = []
        for logo in image_response.get("logoAnnotations") or []:
            label = str(logo.get("description") or "").strip()
            if label:
                results.append(ReverseImageResult(
                    title=label,
                    url="",
                    source="google_vision_logo_tags",
                ))
        for page in web.get("pagesWithMatchingImages") or []:
            results.append(ReverseImageResult(
                title=(page.get("pageTitle") or page.get("url") or "page"),
                url=page.get("url") or "",
                source="google_vision_web_pages",
                thumbnail_url=(page.get("image") or {}).get("url") or "",
            ))
        for ent in web.get("webEntities") or []:
            desc = ent.get("description") or ""
            if not desc:
                continue
            results.append(ReverseImageResult(
                title=desc,
                url=ent.get("uri") or "",
                source="google_vision_web_tags",
                thumbnail_url="",
            ))
        return results


class BingVisualSearchBackend(ReverseImageSearchBackend):
    """Retired Bing Search API adapter, retained for a clear migration error."""

    name = "bing_visual"

    def __init__(self, timeout: float = 20.0):
        self.timeout = timeout
        self._key = _env(BACKEND_ENV[self.name])

    @property
    def available(self) -> bool:
        return False

    def search_crop(self, crop: np.ndarray) -> List[ReverseImageResult]:
        raise RuntimeError(
            "Bing Visual Search API retired on 2025-08-11; configure SerpApi "
            "Google Lens, Google Cloud Vision, or Gemini grounding instead."
        )


class SerpApiReverseBackend(ReverseImageSearchBackend):
    """Google Lens image search through SerpApi's temporary image-upload API."""

    name = "serpapi"

    MAX_UPLOAD_BYTES = 480 * 1024

    def __init__(
        self,
        image_base_url: str = "",
        timeout: float = 30.0,
        monthly_limit: int = 250,
        usage_store: Optional[OpenSetSearchStore] = None,
    ):
        self.timeout = timeout
        self._key = _env(BACKEND_ENV[self.name])
        self.monthly_limit = max(0, int(monthly_limit))
        self.usage_store = usage_store or OpenSetSearchStore(
            Path(__file__).resolve().parents[1] / "data" / "open_set_search.sqlite3"
        )

    @property
    def available(self) -> bool:
        return bool(self._key)

    def usage_status(self) -> Dict[str, Any]:
        return self.usage_store.serpapi_status(self.monthly_limit)

    @classmethod
    def _jpeg_bytes(cls, crop: np.ndarray) -> bytes:
        """Encode a bounded JPEG accepted by SerpApi's 500 KB upload limit."""
        import cv2

        if crop is None or crop.size == 0:
            raise ValueError("empty logo crop")
        h, w = crop.shape[:2]
        scale = min(1.0, 900.0 / max(h, w))
        resized = cv2.resize(
            crop,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
        ) if scale < 1 else crop
        bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
        for quality in (90, 82, 72, 60, 48):
            ok, encoded = cv2.imencode(
                ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality]
            )
            if ok and encoded.nbytes <= cls.MAX_UPLOAD_BYTES:
                return encoded.tobytes()
        raise ValueError("logo crop exceeds SerpApi's 500 KB upload limit")

    def search_crop(self, crop: np.ndarray) -> List[ReverseImageResult]:
        if not self.available:
            raise RuntimeError("SerpApi Google Lens: missing SERPAPI_KEY")
        status = self.usage_status()
        if status["exhausted"]:
            raise SerpApiBudgetExceeded(
                "SerpApi monthly Google Lens limit reached "
                f"({status['used']}/{status['limit']} for {status['period']})"
            )
        import requests

        image = self._jpeg_bytes(crop)
        upload = requests.post(
            SERPAPI_IMAGE_URL,
            data={"api_key": self._key},
            files={"image": ("logo-crop.jpg", image, "image/jpeg")},
            timeout=self.timeout,
        )
        upload.raise_for_status()
        upload_data = upload.json()
        if upload_data.get("error"):
            raise RuntimeError(f"SerpApi image upload failed: {upload_data['error']}")
        image_id = str(upload_data.get("image_id") or "").strip()
        if not image_id:
            raise RuntimeError("SerpApi image upload returned no image_id")

        # Reserve immediately before the billable search call. Image uploads
        # do not count as searches, and a rejected reservation prevents the
        # Google Lens query even if another worker used the final slot while
        # this crop was uploading.
        reservation = self.usage_store.reserve_serpapi_search(self.monthly_limit)
        if not reservation["allowed"]:
            raise SerpApiBudgetExceeded(
                "SerpApi monthly Google Lens limit reached "
                f"({reservation['used']}/{reservation['limit']} for "
                f"{reservation['period']})"
            )

        response = requests.get(
            SERPAPI_URL,
            params={
                "engine": "google_lens",
                "image_id": image_id,
                "type": "all",
                "hl": "en",
                "api_key": self._key,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("error"):
            raise RuntimeError(f"SerpApi Google Lens failed: {data['error']}")
        results = []
        for match in data.get("visual_matches") or []:
            url = str(match.get("link") or "").strip()
            if not re.match(r"^https?://", url, re.IGNORECASE):
                continue
            results.append(ReverseImageResult(
                title=match.get("title") or match.get("source") or url,
                url=url,
                source="serpapi_google_lens_visual_match",
                thumbnail_url=match.get("thumbnail") or "",
            ))
        for item in data.get("text_results") or []:
            url = str(item.get("link") or "").strip()
            if not re.match(r"^https?://", url, re.IGNORECASE):
                continue
            results.append(ReverseImageResult(
                title=item.get("title") or url,
                url=url,
                source="serpapi_google_lens_text_result",
                thumbnail_url=item.get("thumbnail") or "",
            ))
        return results


class GeminiGroundedBackend(ReverseImageSearchBackend):
    """Free, web-grounded identification via Google Gemini (google_search tool).

    Uses the Google Generative Language API free tier with the built-in
    ``google_search`` tool: Gemini *sees* the crop, then grounds its
    identification in a real Google web search. This needs no paid key and no
    public image host (the crop is uploaded inline as base64), and it returns
    real, citable source URLs via the API's ``groundingMetadata`` — the same
    evidence contract as the other backends (no free-text guess).

    Flow (same fail-closed contract as the other engines):
      1. Encode the crop and POST it to ``generateContent`` with
         ``tools=[{"google_search": {}}]``.
      2. The model returns an identification plus ``groundingChunks``
         (title + URI per real search result) supporting its claims.
      3. Every real chunk becomes a ``ReverseImageResult`` (source
         ``gemini_grounded_web``); the model's compact ``BRAND: <name>`` line
         becomes a tags result (source ``gemini_grounded_tags``) routed
         through the same deterministic ``_derive_candidate_name`` + logo.dev
         gate as every other engine; the full grounded summary is carried as
         source ``gemini_grounded_desc`` for the evidence trail only (never a
         candidate name).

    Fail-closed: no GEMINI_API_KEY (env, .env, or ~/.gemini_key) => unavailable.
    """

    name = "gemini_grounded"

    _PROMPT = (
        "Identify the company or consumer brand whose logo or wordmark is "
        "visibly present in this image crop. Do not infer a brand from a food "
        "flavor, product category, color, shape, or unrelated surrounding "
        "packaging text. Use google_search to find an official brand source "
        "that matches the visible mark. If the mark is not clear, respond "
        "BRAND: UNKNOWN. Return exactly one final line in this format:\n"
        "BRAND: <canonical company/brand name>\n"
    )

    def __init__(self, timeout: float = 60.0, model: Optional[str] = None):
        self.timeout = timeout
        self.model = model or _env("GEMINI_MODEL") or "gemini-2.5-flash"
        self._key = _env("GEMINI_API_KEY")
        self._disabled_until = 0.0
        self._permanently_disabled = False
        if not self._key:
            key_path = Path.home() / ".gemini_key"
            if key_path.is_file():
                try:
                    self._key = key_path.read_text(encoding="utf-8").strip() or None
                except OSError:
                    self._key = None

    @property
    def available(self) -> bool:
        return bool(self._key) and not self._permanently_disabled and time.monotonic() >= self._disabled_until

    def _check_response(self, response) -> None:
        if response.status_code == 200:
            return
        if response.status_code in {401, 403}:
            self._permanently_disabled = True
        elif response.status_code in {429, 503}:
            # A free-tier quota or transient overload must not trigger one more
            # doomed call for each remaining candidate in the same video.
            self._disabled_until = time.monotonic() + 60.0
        raise RuntimeError(
            f"Gemini API error {response.status_code}: {response.text[:300]}"
        )

    def search_crop(self, crop: np.ndarray, context_text: str = "") -> List[ReverseImageResult]:
        if not self.available:
            raise RuntimeError("Gemini: no GEMINI_API_KEY")
        import requests

        b64 = base64.b64encode(_img_bytes(crop)).decode("ascii")
        payload = {
            "contents": [{
                "role": "user",
                "parts": [
                    {"text": self._PROMPT + (
                        " The OCR system read this possible wordmark from the crop: "
                        f"{context_text[:120]!r}. Search the exact wordmark and verify "
                        "the official brand site before identifying it. If web search "
                        "does not return a matching official source, respond BRAND: UNKNOWN."
                        if context_text else ""
                    )},
                    {"inline_data": {"mime_type": "image/png", "data": b64}},
                ],
            }],
            "tools": [{"google_search": {}}],
            "generationConfig": {"temperature": 0.2},
        }
        resp = requests.post(
            GEMINI_API_URL.format(model=self.model),
            params={"key": self._key},
            json=payload,
            timeout=self.timeout,
        )
        self._check_response(resp)
        data = resp.json()
        try:
            cand = data["candidates"][0]
        except (KeyError, IndexError):
            raise RuntimeError("Gemini API returned no candidates")

        parts = ((cand.get("content") or {}).get("parts")) or []
        text = " ".join(
            p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text")
        )
        metadata = cand.get("groundingMetadata") or {}
        chunks = metadata.get("groundingChunks") or []

        results: List[ReverseImageResult] = []
        for chunk in chunks:
            web = (chunk or {}).get("web") or {}
            uri = web.get("uri") or ""
            if not uri:
                continue
            results.append(ReverseImageResult(
                title=(web.get("title") or uri),
                url=uri,
                source="gemini_grounded_web",
                thumbnail_url="",
            ))

        identification = self._identification_line(text)
        if identification:
            results.append(ReverseImageResult(
                title=identification,
                url="",
                source="gemini_grounded_tags",
                thumbnail_url="",
            ))
        if text:
            results.append(ReverseImageResult(
                title=text[:500],
                url="",
                source="gemini_grounded_desc",
                thumbnail_url="",
            ))
        return results

    def search_text(self, text: str) -> List[ReverseImageResult]:
        """Search an OCR wordmark directly, retaining Google grounding sources."""
        if not self.available:
            raise RuntimeError("Gemini: no GEMINI_API_KEY")
        import requests

        wordmark = re.sub(r"\s+", " ", str(text)).strip()[:100]
        if not wordmark:
            return []
        prompt = (
            f'Search the web for the exact visible logo wordmark "{wordmark}". '
            "Identify the company or consumer brand, and verify it against its "
            "official website. Do not infer from a product category or unrelated "
            "text. If sources do not match, respond BRAND: UNKNOWN. Return the "
            "canonical name on one line: BRAND: <name>."
        )
        resp = requests.post(
            GEMINI_API_URL.format(model=self.model),
            params={"key": self._key},
            json={
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "tools": [{"google_search": {}}],
                "generationConfig": {"temperature": 0.1},
            },
            timeout=self.timeout,
        )
        self._check_response(resp)
        data = resp.json()
        try:
            cand = data["candidates"][0]
        except (KeyError, IndexError):
            raise RuntimeError("Gemini API returned no candidates")
        parts = ((cand.get("content") or {}).get("parts")) or []
        answer = " ".join(
            p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text")
        )
        results = []
        metadata = cand.get("groundingMetadata") or {}
        for chunk in metadata.get("groundingChunks") or []:
            web = (chunk or {}).get("web") or {}
            uri = web.get("uri") or ""
            if uri:
                results.append(ReverseImageResult(
                    title=web.get("title") or uri,
                    url=uri,
                    source="gemini_grounded_web",
                ))
        name = self._identification_line(answer)
        if name:
            results.append(ReverseImageResult(
                title=name, url="", source="gemini_grounded_tags"
            ))
        if answer:
            results.append(ReverseImageResult(
                title=answer[:500], url="", source="gemini_grounded_desc"
            ))
        return results

    @staticmethod
    def _identification_line(text: str) -> str:
        """Extract the compact ``BRAND: <name>`` line from model output.

        Returns "" when the model did not emit the requested format. The
        previous fallback returned the first line of whatever came back, which
        is prose in practice: "The image appears to show a television screen
        displaying an advertisement for..." became a ReverseImageResult title
        and then scored as a brand. Its own docstring claimed verbose prose
        "never becomes a candidate name", which was the opposite of what the
        code did.

        The description is not lost — it is still surfaced as a
        ``gemini_grounded_desc`` result for a human to read. Degrading to no
        brand claim is the correct failure mode here: this class exists to stop
        unverified names reaching outreach, and an unparseable answer is
        precisely the case that must not be guessed at.
        """
        if not text:
            return ""
        # \n-delimited, not "."-delimited: truncating on the first period turned
        # "BRAND: Dr. Pepper" into "Dr".
        match = re.search(r"\bBRAND\s*:[ \t]*([^\n]+)", text, re.IGNORECASE)
        if match:
            return match.group(1).strip().rstrip(".").strip()[:120]
        return ""


class BrowserGroundedSearchBackend(ReverseImageSearchBackend):
    """Grounded reverse-image search driven through a real browser.

    Uses the `agent-browser` CLI to upload the cropped logo to a real
    reverse-image-search engine (Yandex CBIR, with Google Lens as fallback) and
    scrape the real, citable matching results. This is a REAL reverse-image
    search with source URLs and engine-read wordmark tags — not a captioning
    LLM guess. It needs no paid API key, only the agent-browser binary.

    Yandex CBIR returns two kinds of real evidence:
      * "Image appears to contain" tags — the engine's own text/object
        recognition on the actual crop (often the wordmark text itself, e.g.
        "BEST EXPRESS" for a courier logo that the detector mislabeled).
      * similar-image results — real pages with real source URLs.

    Google Lens is attempted first only when Yandex is unreachable; Lens is
    bot-gated and often captcha-redirects, so Yandex is the primary engine.
    """

    name = "browser_grounded"

    def __init__(self, timeout: float = 90.0):
        self.timeout = timeout
        self._binary = shutil.which("agent-browser")

    @property
    def available(self) -> bool:
        return bool(self._binary)

    @staticmethod
    def _run(*args: str, timeout: float) -> str:
        proc = subprocess.run(
            [shutil.which("agent-browser"), *args],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"agent-browser failed: {proc.stderr[:500]}")
        return proc.stdout

    def _eval(self, js: str) -> str:
        out = self._run("eval", js, timeout=self.timeout)
        out = (out or "").strip()
        if out.startswith('"') and out.endswith('"'):
            try:
                return json.loads(out)
            except ValueError:
                pass
        return out

    def search_crop(self, crop: np.ndarray) -> List[ReverseImageResult]:
        if not self.available:
            raise RuntimeError("agent-browser binary not found")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            import cv2

            ok, buf = cv2.imencode(".png", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            if not ok:
                raise ValueError("could not encode crop image")
            tmp.write(buf.tobytes())
            crop_path = tmp.name
        try:
            try:
                results = self._search_yandex(crop_path)
            except RuntimeError as exc:
                logger.warning("Yandex CBIR failed, falling back to Lens: %s", exc)
                results = self._search_lens(crop_path)
        finally:
            Path(crop_path).unlink(missing_ok=True)
        return results

    def _open_upload_engine(self, url: str) -> bool:
        self._run("open", url, timeout=self.timeout)
        time.sleep(2)
        # Retry once: first page load can land on a consent/blank shell.
        for _ in range(2):
            try:
                has_input = self._eval(
                    "JSON.stringify(document.querySelectorAll('input[type=file]').length)"
                )
            except RuntimeError:
                has_input = "0"
            if has_input.strip().strip('"') == "1":
                return True
            time.sleep(2)
        return False

    def _upload_crop(self, crop_path: str) -> None:
        self._run(
            "upload", "input[type=file]", crop_path, timeout=self.timeout
        )

    @staticmethod
    def _parse_yandex_tags(raw: str) -> List[str]:
        try:
            tags = json.loads(raw)
        except (ValueError, TypeError):
            return []
        if not isinstance(tags, list):
            return []
        return [str(t).strip() for t in tags if str(t).strip()][:12]

    def _yandex_tags(self) -> List[str]:
        """Engine-read wordmark/object tags under 'Image appears to contain'."""
        js = (
            "(()=>{const h=document.getElementById('cbir-tags-title');"
            "if(!h)return [];"
            "const sec=h.closest('section')||h.parentElement.parentElement||h.parentElement;"
            "const lines=(sec.innerText||'').split(/\\n/).map(s=>s.trim()).filter(Boolean);"
            "if(lines[0]&&/Image appears to contain/i.test(lines[0]))lines.shift();"
            "return lines.slice(0,12);})()"
        )
        raw = self._eval(js)
        return self._parse_yandex_tags(raw)

    def _yandex_anchors(self) -> List[ReverseImageResult]:
        """Real similar-image result pages (title + source URL) via the DOM."""
        js = (
            "(()=>{const h=document.getElementById('cbir-similar-title');"
            "let sec=h?h.closest('section'):null;"
            "const scope=sec||document.body;"
            "const out=[];"
            "for(const a of scope.querySelectorAll('a[href]')){"
            "const h2=a.getAttribute('href');"
            "if(!h2||/^javascript/.test(h2))continue;"
            "let abs=h2;"
            "try{abs=new URL(h2,location.href).href}catch(e){continue}"
            "out.push({t:(a.innerText||'').replace(/\\s+/g,' ').trim().slice(0,300),h:abs});}"
            "return JSON.stringify(out.slice(0,25));})()"
        )
        raw = self._eval(js)
        try:
            anchors = json.loads(raw)
        except (ValueError, TypeError):
            return []
        results: List[ReverseImageResult] = []
        seen = set()
        for a in anchors or []:
            href = a.get("h") or ""
            title = a.get("t") or ""
            if "yandex.com/images/" in href and "/search?" in href:
                continue
            domain = re.match(r"https?://(?:www\.)?([^/]+)", href)
            if not domain:
                continue
            host = domain.group(1).lower()
            if re.search(r"(yandex\.com$|yandex\.ru|^yandex)", host):
                continue
            key = href.split("?")[0]
            if key in seen:
                continue
            seen.add(key)
            results.append(ReverseImageResult(
                title=title or href,
                url=href,
                source="yandex_cbir_similar",
                thumbnail_url="",
            ))
        return results[:12]

    def _search_yandex(self, crop_path: str) -> List[ReverseImageResult]:
        """Real Yandex CBIR reverse-image search for the crop."""
        if not self._open_upload_engine("https://yandex.com/images/"):
            raise RuntimeError("Yandex Images file input not found")
        self._upload_crop(crop_path)
        time.sleep(8)
        try:
            self._run("wait", "--load", "networkidle", timeout=min(30, self.timeout))
        except RuntimeError:
            pass
        if "images/search" not in self._run("get", "url", timeout=self.timeout):
            time.sleep(5)
        results: List[ReverseImageResult] = []
        for tag in self._yandex_tags():
            results.append(ReverseImageResult(
                title=tag, url="", source="yandex_cbir_tags", thumbnail_url=""
            ))
        results.extend(self._yandex_anchors())
        return results

    def _search_lens(self, crop_path: str) -> List[ReverseImageResult]:
        """Google Lens fallback via agent-browser (bot-gated; may captcha)."""
        if not self._open_upload_engine("https://lens.google.com/"):
            raise RuntimeError("Lens file input not found")
        self._upload_crop(crop_path)
        time.sleep(6)
        try:
            self._run("wait", "--load", "networkidle", timeout=min(30, self.timeout))
        except RuntimeError:
            pass
        url = self._run("get", "url", timeout=self.timeout)
        if "sorry" in url:
            raise RuntimeError("Google Lens captcha redirect")
        js = (
            "JSON.stringify(Array.from(document.querySelectorAll('a[href]'))."
            "map(a=>({t:(a.innerText||'').replace(/\\s+/g,' ').trim().slice(0,300),"
            "h:a.href})).filter(x=>x.h&&!/google\\.com/.test(x.h)&&"
            "!/^javascript/.test(x.h)).slice(0,12))"
        )
        raw = self._eval(js)
        try:
            anchors = json.loads(raw)
        except (ValueError, TypeError):
            anchors = []
        results: List[ReverseImageResult] = []
        seen = set()
        for a in anchors or []:
            href = a.get("h") or ""
            if href in seen:
                continue
            seen.add(href)
            results.append(ReverseImageResult(
                title=a.get("t") or href,
                url=href,
                source="google_lens",
                thumbnail_url="",
            ))
        return results


def create_backend(
    name: str,
    *,
    serpapi_monthly_limit: int = 250,
    usage_store: Optional[OpenSetSearchStore] = None,
) -> ReverseImageSearchBackend:
    """Factory: create the configured reverse-image-search backend."""
    if name == "google_vision_web":
        return GoogleVisionWebBackend()
    if name == "bing_visual":
        raise ValueError(
            "bing_visual is retired; configure serpapi, google_vision_web, "
            "gemini_grounded, or browser_grounded"
        )
    if name == "serpapi":
        return SerpApiReverseBackend(
            monthly_limit=serpapi_monthly_limit, usage_store=usage_store
        )
    if name == "browser_grounded":
        return BrowserGroundedSearchBackend()
    if name == "gemini_grounded":
        return GeminiGroundedBackend()
    if name == "auto":
        return FallbackReverseImageSearchBackend([
            GoogleVisionWebBackend(),
            SerpApiReverseBackend(
                monthly_limit=serpapi_monthly_limit, usage_store=usage_store
            ),
            GeminiGroundedBackend(),
            BrowserGroundedSearchBackend(),
        ])
    raise ValueError(f"unknown reverse-image-search backend: {name}")


def _crop_bbox(frame: np.ndarray, bbox, margin: float = 0.1) -> Optional[np.ndarray]:
    """Crop a bounding box from a frame with a small margin (same logic as the
    brand resolver so crops are consistent with detection boxes)."""
    try:
        x1, y1, x2, y2 = (int(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    px = int((x2 - x1) * margin) + 1
    py = int((y2 - y1) * margin) + 1
    x1, y1 = max(0, x1 - px), max(0, y1 - py)
    x2, y2 = min(w, x2 + px), min(h, y2 + py)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return frame[y1:y2, x1:x2]


# Hamming distance (out of 256 bits) at which two crops count as the same
# logo. Measured on rendered wordmarks (tests/test_crop_hash_perceptual.py):
# the same logo across frame-to-frame jitter and sensor noise lands at <=20,
# while the closest two DIFFERENT wordmarks (SONY/BMW) are 105 apart. 32 sits
# in that empty middle: far above same-logo noise, far below any cross-logo
# pair. Re-measure if CROP_HASH_BITS changes — the two must scale together.
CROP_HASH_BITS = 256
CROP_HASH_MAX_DISTANCE = 32

# Grayscale level above which a crop pixel counts as "logo" when trimming to
# the content bounding box. Dark wordmarks on a dark video frame still land
# well above this; compression noise does not.
CROP_TRIM_THRESHOLD = 40


def _crop_hash(crop: np.ndarray) -> str:
    """Perceptual difference hash (dHash) for deduplicating near-identical crops.

    Hashing raw bytes only dedupes byte-identical crops, and two frames of the
    same video never are — the detection box shifts a pixel or two and sensor
    noise changes the values, so exact hashing never fired and every frame of
    a persistent logo spent its own search budget.

    Trimming to the content bounding box FIRST is what makes this work: the box
    shift is the dominant source of variation, and normalising it away makes
    the same logo hash identically (distance 0). Without the trim, dHash on the
    raw crop is not translation-stable at any grid or blur radius I measured —
    and a fixed Hamming threshold cannot be calibrated, because the same-logo
    and different-logo distributions overlap.

    Returns a CROP_HASH_BITS-bit hash as hex; compare with _crop_hash_distance.
    """
    import cv2

    grid = int(CROP_HASH_BITS**0.5)

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    mask = (gray > CROP_TRIM_THRESHOLD).astype(np.uint8)
    if mask.any():
        ys, xs = np.nonzero(mask)
        gray = gray[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]

    small = cv2.resize(gray, (grid + 1, grid), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):0{CROP_HASH_BITS // 4}x}"


def _crop_hash_distance(a: str, b: str) -> int:
    """Hamming distance between two hex dHashes, in bits."""
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def _boxes_overlap(box_a, box_b, min_iou: float) -> bool:
    try:
        ax1, ay1, ax2, ay2 = (float(v) for v in box_a)
        bx1, by1, bx2, by2 = (float(v) for v in box_b)
    except (TypeError, ValueError):
        return False
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return union > 0 and inter / union >= min_iou


def _read_frame(
    video_path: str, frame_index: int, video_stride: int = 1
) -> Optional[np.ndarray]:
    """Seek to the source frame behind an extracted-frame index and return the
    RGB frame (mirrors server._read_video_frame).

    `frame_index` is a position in the pipeline's sampled frame list. Long
    videos are decoded with a stride > 1, so sampled index i corresponds to
    source frame i*stride; seek there so the crop matches the frame the
    analysis actually attributed the logo to.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target = int(round(frame_index * max(1, video_stride)))
    if total > 0:
        target = min(target, total - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, target))
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _is_generic_tag(tag: str, generic_words) -> bool:
    t = tag.lower().strip()
    if t in (generic_words or ()):
        return True
    if len(t) <= 2:
        return True
    return False


def _brand_likeness(tag: str, generic_words) -> float:
    """Deterministic heuristic score for how brand-like an engine tag is.

    Prefers multi-word alphabetic phrases (wordmark text) over single generic
    descriptors. Not an LLM guess — a fixed function of the real engine tags.
    The generic-descriptor vocabulary is data-driven (config
    `open_set.generic_tag_filter`), never hardcoded here.
    """
    t = tag.strip()
    if not t:
        return -1.0
    alpha = sum(ch.isalpha() or ch.isspace() for ch in t)
    if alpha / max(1, len(t)) < 0.7:
        return -1.0
    words = [w for w in re.split(r"[\s_\-]+", t) if w]
    if not words:
        return -1.0
    if t.lower() in (generic_words or ()):  # exact phrase match to the filter
        return 0.5
    if any(_is_generic_tag(w, generic_words) for w in words):
        return 0.5
    return min(1.0, 0.4 + 0.3 * len(words) + 0.02 * len(t))


def _derive_candidate_name(
    results: List[ReverseImageResult],
    generic_words,
    generic_domains,
) -> Optional[str]:
    """Best-effort candidate name derived ONLY from real search results.

    Primary signal: the engine's own wordmark/object tags read from the crop
    (e.g. Yandex "Image appears to contain: BEST EXPRESS"), scoring brand-like
    phrases over generic descriptors. Secondary signal: the most frequent
    non-generic domain among result URLs. This is a deterministic function of
    the real, citable search output — not an LLM guess — and the result is only
    ever a lower-trust *candidate*, never a presented detection, until logo.dev
    validates it. The generic tag/domain vocabularies are data-driven
    (config `open_set.generic_tag_filter` / `open_set.generic_domain_filter`).
    """
    if not results:
        return None
    candidates: List[tuple] = []
    serpapi_brand_hosts: Dict[str, set] = {}
    for r in results:
        if r.source.endswith("_tags") and r.title:
            score = _brand_likeness(r.title, generic_words)
            if score > 0.5:
                candidates.append((score, r.title))
        if r.source.startswith("serpapi_google_lens"):
            # Lens results are pages, not engine-extracted wordmark tags.
            # Do not turn the most frequent shopping/hosting domain (Amazon,
            # eBay, etc.) into the brand. Use catalog matches in page titles,
            # and require agreement from multiple independent result hosts.
            from src.brand_catalog import match_brand_wordmark

            matched = match_brand_wordmark(r.title or "", max_prefix_words=1)
            if matched and r.url:
                host_match = re.match(r"https?://(?:www\.)?([^/]+)", r.url)
                host = host_match.group(1).lower() if host_match else r.url
                serpapi_brand_hosts.setdefault(matched.upper(), set()).add(host)
    if candidates:
        _, best = max(candidates, key=lambda c: c[0])
        best = re.sub(r"\s+", " ", best).strip().upper()
        if _is_void_name(best):
            return None
        return best
    if serpapi_brand_hosts:
        ranked = sorted(
            ((len(hosts), brand) for brand, hosts in serpapi_brand_hosts.items()),
            reverse=True,
        )
        if ranked[0][0] >= 2 and (len(ranked) == 1 or ranked[0][0] > ranked[1][0]):
            return ranked[0][1]
        # Conflicting/weak Lens page-title evidence stays unresolved; never
        # fall through and label a logo with the retailer's domain.
        return None
    domain_counts: Dict[str, int] = {}
    for r in results:
        if not r.url:
            continue
        m = re.match(r"https?://(?:www\.)?([^/]+)", r.url)
        if not m:
            continue
        domain = m.group(1).lower()
        # Substring match is wrong: with pattern "x.com", "netflix.com",
        # "dropbox.com" and "wix.com" ALL contain "x.com". Only the domain
        # itself (or a subdomain of it) is a match.
        if any(domain == pat or domain.endswith("." + pat) for pat in (generic_domains or ())):
            continue
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    if not domain_counts:
        return None
    top_domain = max(domain_counts, key=domain_counts.get)
    if any(r.source.startswith("serpapi_google_lens") for r in results) and domain_counts[top_domain] < 2:
        # An isolated Lens page can be a visually similar false match. An
        # unknown brand inferred from hostnames needs corroborating pages.
        return None
    # brand-ish token = the registrable part before the TLD. For a two-letter
    # ccTLD ("foo.co.uk", "bar.com.au") the label before it is a pseudo-TLD
    # ("co", "com"), so step one more label back.
    parts = top_domain.split(".")
    if len(parts) >= 2:
        tld = parts[-1]
        idx = -2 if len(tld) > 2 else -3
        name = parts[idx].upper() if len(parts) >= -idx else top_domain.upper()
        return None if _is_void_name(name) else name
    name = top_domain.upper()
    return None if _is_void_name(name) else name


_VOID_NAMES = {
    "unknown",
    "unresolved",
    "none",
    "null",
    "nothing",
    "no",
    "no brand",
    "no match",
    "invalid",
    "unidentified",
    "n a",
    "n/a",
    "unknown brand",
}

# A model that answers "not identifiable" is reporting a failed lookup, not a
# brand. Matched as phrases so the whole negation family is caught, not just the
# exact strings above ("NOT IDENTIFIABLE" was reaching logo.dev and coming back
# verified as identifiable.ca). Whole-phrase matching keeps real brands that
# merely start with "not" — Notion, Nottingham — out of the net.
_VOID_PHRASES = (
    "not identifiable",
    "not identified",
    "not recognizable",
    "not recognisable",
    "not a brand",
    "not a logo",
    "cannot identify",
    "can t identify",
    "unable to identify",
    "no identifiable",
    "unidentifiable",
    "no logo detected",
    "no brand detected",
)


def _is_void_name(name: str) -> bool:
    """A candidate name that semantically means "no identification" is treated
    as no name — surfacing "UNKNOWN"/"NONE" as a brand would be a fabricated
    label, not a real identification."""
    if not name:
        return True
    t = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
    if not t or t in _VOID_NAMES:
        return True
    return any(phrase in t for phrase in _VOID_PHRASES)


class OpenSetBrandIdentifier:
    """Cost-gated, evidence-trail-producing unknown-brand identification."""

    def __init__(
        self,
        backend: Optional[ReverseImageSearchBackend],
        consent_upload: bool,
        min_logo_confidence: float,
        min_crop_area: float,
        max_crop_aspect: float,
        max_candidates_per_video: int,
        crop_cache_dir,
        generic_tag_filter,
        generic_domain_filter,
        logodev_timeout: float,
        saturated_proposals_per_frame: int = 30,
        ocr_min_crop_area: float = 600.0,
        max_crop_area_fraction: float = 0.75,
        search_store: Optional[OpenSetSearchStore] = None,
    ):
        """All tunables are explicit and come from config `open_set:`.

        Every parameter is required — the caller (server.py) reads them from
        config/config.yaml. `None`/empty values are NOT silently defaulted
        here: an identifier constructed without a real backend, threshold or
        cache dir is unusable and `identify()` fails closed.
        """
        if backend is None:
            raise ValueError("OpenSetBrandIdentifier requires a backend")
        # A crop is the creator's own image. Uploading it to a third-party
        # search API is an explicit consent decision, so it is required as a
        # real bool rather than defaulted — a caller that forgets to pass it
        # must fail loudly here, not silently start leaking frames.
        if not isinstance(consent_upload, bool):
            raise ValueError("OpenSetBrandIdentifier requires a bool consent_upload")
        if not min_logo_confidence or float(min_logo_confidence) <= 0.0:
            raise ValueError("OpenSetBrandIdentifier requires min_logo_confidence > 0")
        if min_crop_area is None or float(min_crop_area) <= 0.0:
            raise ValueError("OpenSetBrandIdentifier requires min_crop_area > 0")
        if max_crop_aspect is None or float(max_crop_aspect) <= 0.0:
            raise ValueError("OpenSetBrandIdentifier requires max_crop_aspect > 0")
        if not max_candidates_per_video or int(max_candidates_per_video) <= 0:
            raise ValueError("OpenSetBrandIdentifier requires max_candidates_per_video > 0")
        if not crop_cache_dir:
            raise ValueError("OpenSetBrandIdentifier requires a crop_cache_dir")
        if logodev_timeout is None or float(logodev_timeout) <= 0.0:
            raise ValueError("OpenSetBrandIdentifier requires logodev_timeout > 0")

        self.backend = backend
        self.consent_upload = consent_upload
        self.min_logo_confidence = float(min_logo_confidence)
        self.min_crop_area = float(min_crop_area)
        self.ocr_min_crop_area = max(1.0, float(ocr_min_crop_area))
        self.max_crop_area_fraction = min(1.0, max(0.01, float(max_crop_area_fraction)))
        self.max_crop_aspect = float(max_crop_aspect)
        self.max_candidates_per_video = int(max_candidates_per_video)
        # Reporting threshold only. It never drops a candidate: the point is
        # to make a saturated frame visible, and silently discarding proposals
        # to hit a number would hide the defect instead of describing it.
        # 30 is roughly where a frame stops being "a few marks" and starts
        # being the detector failing to discriminate - on the one video
        # measured (7864fa1c52) the p50 was 109 and the minimum on any frame
        # was 62, so every frame clears it and the flag is honest rather than
        # tuned to look rare.
        self.saturated_proposals_per_frame = int(saturated_proposals_per_frame)
        self.max_saturated_frames_listed = 20
        self.crop_cache_dir = Path(crop_cache_dir)
        self.crop_cache_dir.mkdir(parents=True, exist_ok=True)
        self.generic_words = set(str(w).lower().strip() for w in (generic_tag_filter or ()))
        self.generic_domains = set(str(d).lower().strip() for d in (generic_domain_filter or ()))
        self.logodev_timeout = float(logodev_timeout)
        if not saturated_proposals_per_frame or int(saturated_proposals_per_frame) <= 0:
            raise ValueError(
                "OpenSetBrandIdentifier requires saturated_proposals_per_frame > 0")
        self._logodev_client = None  # lazy
        self._result_cache: Dict[str, List[ReverseImageResult]] = {}
        self.search_store = search_store
        self.search_cache_hits = 0

    def _logodev(self):
        from src.logodev import LogoDevClient

        if self._logodev_client is None:
            self._logodev_client = LogoDevClient(timeout=self.logodev_timeout)
        return self._logodev_client

    def _search_cache_key(self, crop_hash: str, context_text: str) -> str:
        context_hash = hashlib.sha256(
            " ".join(str(context_text or "").lower().split()).encode("utf-8")
        ).hexdigest()[:20]
        return f"{self.backend.name}:{crop_hash}:{context_hash}"

    def _unknown_ocr_proposals(self, l1: dict) -> List[List[dict]]:
        """Extract compact, high-confidence OCR wordmarks absent from catalog.

        This is the path that makes an unknown readable brand searchable even
        when the trained logo detector misses its box. Long sentence OCR and
        generic UI labels are deliberately excluded.
        """
        from src.layer2.brand_resolver import SCREEN_UI_TOKENS

        proposals: List[List[dict]] = []
        for regions in l1.get("ocr_results") or []:
            frame_proposals = []
            for region in regions or []:
                text = re.sub(r"\s+", " ", str(region.get("text") or "")).strip()
                try:
                    confidence = float(region.get("confidence", 0.0) or 0.0)
                    points = np.asarray(region.get("bbox"), dtype=float).reshape(-1, 2)
                except (TypeError, ValueError):
                    continue
                if not text or not np.isfinite(confidence) or confidence < 0.88:
                    continue
                padded_text = " " + text.lower() + " "
                if any(" " + token + " " in padded_text for token in SCREEN_UI_TOKENS):
                    continue
                # A catalog match is already handled locally; don't pay for a
                # web lookup when normalization/OCR variations match a known name.
                if match_brand_wordmark(text, max_prefix_words=1):
                    continue
                tokens = re.findall(r"[\w&'’-]+", text, flags=re.UNICODE)
                if not 1 <= len(tokens) <= 3 or len(text) > 40:
                    continue
                letters = [c for c in text if c.isalpha()]
                if len(letters) < 3:
                    continue
                uppercase_ratio = sum(c.isupper() for c in letters) / len(letters)
                titlecase_single = len(tokens) == 1 and text[:1].isupper() and confidence >= 0.96
                if uppercase_ratio < 0.72 and not titlecase_single:
                    continue
                if any(token.lower() in self.generic_words for token in tokens):
                    continue
                if points.shape[0] < 2:
                    continue
                # OCR polygons are clipped in _candidate_crops once the source
                # frame is loaded; reject malformed geometry here only.
                x1, y1 = points.min(axis=0)
                x2, y2 = points.max(axis=0)
                if x2 <= x1 or y2 <= y1:
                    continue
                frame_proposals.append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": confidence,
                    "class_name": "unknown_ocr_wordmark",
                    "ocr_text": text,
                    "detection_source": "ocr_unknown_wordmark",
                })
            proposals.append(sorted(frame_proposals, key=lambda d: d["confidence"], reverse=True))
        return proposals

    def _candidate_crops(self, result: dict, video_path: str) -> tuple:
        """Collect cost-gated candidate crops from real logo detections.

        Gates (editorial, all recorded for the UI's open-set status states):
          - detector confidence >= min_logo_confidence (documented threshold),
          - banner-shape gate: bbox aspect ratio < max_crop_aspect (rejects
            full-width ticker/news-strap crops that flood the top "confidence
            hits" with non-logoographic material), checked pre-read for cost,
          - minimum searchable crop area: crop pixels >= min_crop_area (a
            crop smaller than this is below the resolution reverse-image
            search meaningfully returns on),
          - dedup by crop hash (near-identical crops fire once),
          - capped at max_candidates_per_video,
          - per-crop search results cached.

        Returns ``(accepted, rejected, skipped, diagnostics)`` where ``rejected``
        is a list of {frame_index, det_index, confidence, bbox, width, height,
        aspect, reason} for every gate-skipped det, ``skipped`` counts the
        non-recordable skips, and ``diagnostics`` reports the state of the
        detector's output upstream of every gate here.

        That last part is the point. When the detector is saturated, the
        confidence distribution is flat and the gate rejects everything; read
        alone, that looks like a resolver failure. Measured on 7864fa1c52 with
        its human labels: 112 proposals/frame on frames WITH a logo vs 62 on
        frames WITHOUT one, p50 confidence 0.148 vs 0.143. Confidence carries
        almost no information about whether a logo is present, so a 0.30 gate
        rejects a flat noise floor. The gates are behaving as documented; the
        number of them says nothing on its own about why.
        """
        l1 = result["layer1"]
        video_stride = int(result.get("video_stride", 1) or 1)
        accepted: List[dict] = []
        rejected: List[dict] = []
        skipped_below_conf = 0
        skipped_dup = 0
        seen_hashes: set = set()
        budget = int(self.max_candidates_per_video)
        # Saturation is a property of the detector's raw output, so it is
        # counted before any gate can discard a detection. Measuring it after
        # the confidence gate would report the gate's effect, not the cause.
        proposals_total = 0
        per_frame: List[dict] = []
        detector_rows = l1.get("logo_search_proposals")
        if detector_rows is None:
            detector_rows = l1.get("logo_detections") or []
        ocr_rows = self._unknown_ocr_proposals(l1)
        known_rows = l1.get("logo_detections") or []
        proposal_rows = []
        for frame_idx in range(max(len(detector_rows), len(ocr_rows))):
            for det_idx, det in enumerate(ocr_rows[frame_idx] if frame_idx < len(ocr_rows) else []):
                proposal_rows.append((frame_idx, det_idx, det))
        for frame_idx, dets in enumerate(detector_rows):
            n_dets = len(dets or [])
            proposals_total += n_dets
            if n_dets:
                per_frame.append({"frame_index": frame_idx, "proposals": n_dets})
            for det_idx, det in enumerate(dets or []):
                proposal_rows.append((frame_idx, det_idx, det))
        for frame_idx, det_idx, det in proposal_rows:
            if len(accepted) >= budget:
                break
            source = str(det.get("detection_source") or "logo_detector")
            # Search only brands not already confidently named locally.
            local_known = [
                d for d in (known_rows[frame_idx] if frame_idx < len(known_rows) else [])
                if d.get("brand") and d.get("bbox")
            ]
            if match_brand_wordmark(str(det.get("class_name") or "")):
                continue
            if source != "ocr_unknown_wordmark" and any(
                _boxes_overlap(det.get("bbox"), known.get("bbox"), 0.35)
                for known in local_known
            ):
                continue
            conf = float(det.get("confidence", 0.0))
            if conf < self.min_logo_confidence:
                skipped_below_conf += 1
                continue
            bbox = det.get("bbox")
            if not bbox:
                # Recorded, not skipped silently: a detection that cannot
                # be located is a detector defect, and dropping it without
                # a record is how a stage ends up explaining nothing.
                rejected.append({
                    "frame_index": frame_idx, "det_index": det_idx,
                    "confidence": round(conf, 3), "bbox": None,
                    "reason": "no_bbox",
                })
                continue
            bw = abs(int(bbox[2]) - int(bbox[0]))
            bh = abs(int(bbox[3]) - int(bbox[1]))
            bbox_aspect = bw / max(1, bh)
            if bbox_aspect >= self.max_crop_aspect:
                rejected.append({
                    "frame_index": frame_idx,
                    "det_index": det_idx,
                    "confidence": round(conf, 3),
                    "bbox": [float(v) for v in bbox],
                    "width": bw,
                    "height": bh,
                    "aspect": round(bbox_aspect, 2),
                    "reason": "banner_shape",
                })
                continue
            frame = _read_frame(video_path, frame_idx, video_stride)
            if frame is None:
                rejected.append({
                    "frame_index": frame_idx, "det_index": det_idx,
                    "confidence": round(conf, 3),
                    "bbox": [float(v) for v in bbox],
                    "reason": "unreadable_frame",
                })
                continue
            frame_area = max(1, int(frame.shape[0]) * int(frame.shape[1]))
            box_area_fraction = (bw * bh) / frame_area
            if box_area_fraction > self.max_crop_area_fraction:
                rejected.append({
                    "frame_index": frame_idx,
                    "det_index": det_idx,
                    "confidence": round(conf, 3),
                    "bbox": [float(v) for v in bbox],
                    "width": bw,
                    "height": bh,
                    "aspect": round(bbox_aspect, 2),
                    "area_fraction": round(box_area_fraction, 3),
                    "reason": "editorial_box",
                })
                continue
            crop = _crop_bbox(frame, bbox)
            if crop is None or crop.size == 0:
                rejected.append({
                    "frame_index": frame_idx, "det_index": det_idx,
                    "confidence": round(conf, 3),
                    "bbox": [float(v) for v in bbox],
                    "reason": "invalid_crop",
                })
                continue
            ch, cw = crop.shape[:2]
            crop_area = cw * ch
            min_area = (
                float(getattr(self, "ocr_min_crop_area", 600.0))
                if source == "ocr_unknown_wordmark"
                else self.min_crop_area
            )
            if crop_area < min_area:
                rejected.append({
                    "frame_index": frame_idx,
                    "det_index": det_idx,
                    "confidence": round(conf, 3),
                    "bbox": [float(v) for v in bbox],
                    "width": cw,
                    "height": ch,
                    "aspect": round(cw / max(1, ch), 2),
                    "reason": "too_small",
                })
                continue
            h = _crop_hash(crop)
            if any(
                _crop_hash_distance(h, seen) <= CROP_HASH_MAX_DISTANCE
                for seen in seen_hashes
            ):
                skipped_dup += 1
                continue
            seen_hashes.add(h)
            accepted.append({
                "frame_index": frame_idx,
                "det_index": det_idx,
                "confidence": conf,
                "bbox": [float(v) for v in bbox],
                "crop": crop,
                "crop_hash": h,
                "source": source,
                "ocr_text": det.get("ocr_text"),
            })
        skipped = {"below_min_confidence": skipped_below_conf, "duplicate_hash": skipped_dup}

        # ── per-frame detector diagnostics. Reporting only, no behaviour
        #    change: nothing here gates, reorders or drops a crop. A saturated
        #    frame is flagged so the funnel's zeros are legible as a detector
        #    problem instead of reading as a resolver problem.
        counts = sorted((f["proposals"] for f in per_frame), reverse=True)
        n_frames_with = len(per_frame)
        saturated = [
            f["frame_index"] for f in per_frame
            if f["proposals"] >= self.saturated_proposals_per_frame
        ]
        diagnostics = {
            "proposals_total": proposals_total,
            "frames_with_proposals": n_frames_with,
            "max_proposals_in_a_frame": counts[0] if counts else 0,
            "median_proposals_in_a_frame": (
                counts[len(counts) // 2] if counts else 0),
            "saturated_proposals_per_frame": self.saturated_proposals_per_frame,
            "saturated_frames": saturated[:self.max_saturated_frames_listed],
            "saturated_frame_count": len(saturated),
            "saturation_note": (
                "The detector proposed this many boxes in one frame. A high "
                "proposal count means identity work downstream is unreliable on "
                "that frame: the real logo may not be distinguishable from the "
                "noise around it. This is a detector-quality signal, not a "
                "resolver result."
            ),
        }
        return accepted, rejected, skipped, diagnostics

    def identify(self, result: dict, video_path: str) -> Dict[str, Any]:
        """Run open-set identification for a finished job.

        Returns the dashboard `open_set` block:
          {available, backend, min_confidence, candidates: [...],
           rejected: [...], skipped_counts: {...}}
        Fail-closed: with no runnable backend and no search results, no
        candidate name is ever surfaced. Every gate skip is recorded in
        `rejected` with its reason so the UI never shows a silent blank for a
        candidate slot and never presents a void name as an identification.
        """
        candidates: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        skipped_counts: Dict[str, int] = {}
        backend_available = self.backend.available
        reason = ""
        if not self.consent_upload:
            # Local work still ran: detection, crops, the cache and every
            # diagnostic are all below. Only the outbound call is withheld.
            reason = (
                "OPEN-SET UPLOAD DISABLED — crops were identified locally but "
                "NOT sent to a third-party search API. Set "
                "open_set.consent_upload: true to allow it."
            )
        if not backend_available:
            reason = (
                "OPEN-SET IDENTIFICATION UNAVAILABLE — NO RUNNABLE "
                "REVERSE-IMAGE-SEARCH BACKEND (set SERPAPI_KEY / "
                "GOOGLE_CLOUD_VISION_API_KEY / GEMINI_API_KEY, or "
                "install agent-browser)"
            )

        accepted, rejected, skipped_counts, diagnostics = self._candidate_crops(
            result, video_path)
        self.search_cache_hits = 0
        for cand in accepted:
            crop = cand.pop("crop")
            entry: Dict[str, Any] = {
                "frame_index": cand["frame_index"],
                "det_index": cand["det_index"],
                "confidence": round(cand["confidence"], 3),
                "bbox": cand["bbox"],
                "crop_id": cand["crop_hash"],
                "candidate_name": None,
                "status": "unresolved",
                "search_results": [],
                "logo_dev_validation": None,
                "proposal_source": cand.get("source", "logo_detector"),
                "ocr_text": cand.get("ocr_text"),
            }
            # Save the crop so the UI evidence trail can display it.
            crop_path = self.crop_cache_dir / f"{cand['crop_hash']}.png"
            if not crop_path.is_file():
                import cv2

                cv2.imwrite(
                    str(crop_path),
                    cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
                )
            entry["crop_url"] = f"/api/crop/{cand['crop_hash']}"

            if backend_available and self.consent_upload:
                try:
                    cache_key = self._search_cache_key(
                        cand["crop_hash"], cand.get("ocr_text") or ""
                    )
                    if cache_key in self._result_cache:
                        results = self._result_cache[cache_key]
                    elif self.search_store is not None and (
                        cached_results := self.search_store.get_results(cache_key)
                    ) is not None:
                        results = cached_results
                        self._result_cache[cache_key] = results
                        self.search_cache_hits += 1
                    else:
                        results = []
                        if cand.get("ocr_text"):
                            search_text = getattr(self.backend, "search_text", None)
                            if callable(search_text):
                                results = search_text(cand["ocr_text"])
                        if not results or not any(r.url for r in results):
                            search_with_context = getattr(self.backend, "search_crop")
                            if cand.get("ocr_text") and isinstance(
                                self.backend,
                                (FallbackReverseImageSearchBackend, GeminiGroundedBackend),
                            ):
                                results = search_with_context(
                                    crop, context_text=cand["ocr_text"]
                                )
                            else:
                                results = search_with_context(crop)
                        self._result_cache[cache_key] = results
                        if self.search_store is not None:
                            try:
                                self.search_store.put_results(cache_key, results)
                            except Exception as cache_exc:  # noqa: BLE001
                                logger.warning("reverse-image cache write failed: %s", cache_exc)
                    entry["search_results"] = [r.to_dict() for r in results]
                except Exception as exc:  # noqa: BLE001
                    logger.warning("reverse-image search failed for %s: %s", cand["crop_hash"], exc)
                    entry["search_results"] = []
                    entry["search_error"] = f"Reverse-image search failed: {_safe_external_error(exc)}"

                name = _derive_candidate_name(
                    [ReverseImageResult(**{k: r[k] for k in r}) for r in entry["search_results"]],
                    self.generic_words,
                    self.generic_domains,
                ) if entry["search_results"] else None
                entry["candidate_name"] = name
                if name:
                    validation = self._logodev().validate_brand(name)
                    entry["logo_dev_validation"] = validation
                    if validation.get("status") == "verified":
                        entry["status"] = "candidate_verified"
                    else:
                        entry["status"] = "candidate_unverified"
                elif entry["search_results"]:
                    entry["status"] = "candidate_no_name"
                else:
                    entry["status"] = "unresolved"
            else:
                # "unavailable" means no backend; "consent_withheld" means we
                # had one and chose not to send. Collapsing them would hide a
                # consent decision behind a missing-dependency message.
                entry["status"] = (
                    "consent_withheld" if backend_available else "unavailable"
                )

            candidates.append(entry)

        resolved = len([c for c in candidates if c["status"] == "candidate_verified"])
        provider_usage = {}
        usage_reader = getattr(self.backend, "provider_usage", None)
        try:
            if callable(usage_reader):
                provider_usage = usage_reader()
            else:
                usage_reader = getattr(self.backend, "usage_status", None)
                if callable(usage_reader):
                    provider_usage = {self.backend.name: usage_reader()}
        except Exception as usage_exc:  # noqa: BLE001
            logger.warning("reverse-image usage status unavailable: %s", usage_exc)
        return {
            "available": backend_available,
            "consent_upload": self.consent_upload,
            "backend": self.backend.name,
            "min_confidence": self.min_logo_confidence,
            "min_crop_area": self.min_crop_area,
            "max_crop_aspect": self.max_crop_aspect,
            "reason": reason,
            "candidates": candidates,
            "rejected": rejected,
            "skipped_counts": skipped_counts,
            "resolved": resolved,
            "provider_usage": provider_usage,
            "search_cache_hits": self.search_cache_hits,
            # Detector state upstream of every gate above, so a funnel of
            # zeros reads as a detector diagnosis rather than a resolver one.
            "detector_diagnostics": diagnostics,
            "rejected_reason_counts": dict(Counter(r["reason"] for r in rejected)),
        }
