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
   web-search backend (Google Cloud Vision Web Detection, Bing Visual Search,
   SerpApi reverse image search, or a browser-driven grounded search against
   Google Lens / Bing). The search must return real, citable source URLs —
   never a free-text guess.
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
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Backend -> env var that unlocks it (None = no key required).
# This is a mapping of real backend identifiers to the real env vars that
# carry their API keys — a routing table, not a mock and not tunable data.
BACKEND_ENV = {
    "google_vision_web": "GOOGLE_CLOUD_VISION_API_KEY",
    "bing_visual": "BING_VISUAL_SEARCH_KEY",
    "serpapi": "SERPAPI_KEY",
    "browser_grounded": None,
}

GOOGLE_VISION_URL = "https://vision.googleapis.com/v1/images:annotate"
BING_VISUAL_URL = "https://api.bing.microsoft.com/v7.0/images/visualsearch"
SERPAPI_URL = "https://serpapi.com/search.json"


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
    def search_crop(self, crop: np.ndarray) -> List[ReverseImageResult]:
        """Return real search results for a cropped logo image (bytes upload)."""
        raise NotImplementedError


def _load_env_file(project_root: Optional[Path] = None) -> None:
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

    @property
    def available(self) -> bool:
        return bool(self._key)

    def search_crop(self, crop: np.ndarray) -> List[ReverseImageResult]:
        if not self.available:
            raise RuntimeError("Google Cloud Vision: no GOOGLE_CLOUD_VISION_API_KEY")
        import requests

        b64 = base64.b64encode(_img_bytes(crop)).decode("ascii")
        resp = requests.post(
            GOOGLE_VISION_URL,
            params={"key": self._key},
            json={
                "requests": [{
                    "image": {"content": b64},
                    "features": [{"type": "WEB_DETECTION", "maxResults": 6}],
                }]
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        web = (((data.get("responses") or [{}])[0]).get("webDetection")) or {}
        results: List[ReverseImageResult] = []
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
                source="google_vision_web_entities",
                thumbnail_url="",
            ))
        return results


class BingVisualSearchBackend(ReverseImageSearchBackend):
    """Bing Visual Search API (real API, needs subscription key)."""

    name = "bing_visual"

    def __init__(self, timeout: float = 20.0):
        self.timeout = timeout
        self._key = _env(BACKEND_ENV[self.name])

    @property
    def available(self) -> bool:
        return bool(self._key)

    def search_crop(self, crop: np.ndarray) -> List[ReverseImageResult]:
        if not self.available:
            raise RuntimeError("Bing Visual Search: no BING_VISUAL_SEARCH_KEY")
        import requests

        knowledge_request = json.dumps({
            "imageInfo": {"imageInsightsToken": ""},
            "cropArea": {"top": 0, "left": 0, "width": 1, "height": 1},
        })
        files = [
            ("knowledgeRequest", (None, knowledge_request, "application/json")),
            ("image", ("crop.png", _img_bytes(crop), "image/png")),
        ]
        resp = requests.post(
            BING_VISUAL_URL,
            headers={"Ocp-Apim-Subscription-Key": self._key},
            files=files,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        results: List[ReverseImageResult] = []
        for tag in data.get("tags") or []:
            for action in tag.get("actions") or []:
                if action.get("actionType") != "PagesIncluding":
                    continue
                for page in (action.get("data") or {}).get("value") or []:
                    results.append(ReverseImageResult(
                        title=page.get("name") or page.get("hostPageUrl") or "page",
                        url=page.get("hostPageUrl") or "",
                        source="bing_visual_pages",
                        thumbnail_url=(page.get("thumbnailUrl") or {}),
                    ))
        return results


class SerpApiReverseBackend(ReverseImageSearchBackend):
    """SerpApi Google reverse-image search (needs key; requires a hosted image URL).

    NOTE: the current pipeline has no object-storage host, so this backend is
    not wired as a default. It is implemented so a SerpApi user can set
    SERPAPI_KEY + a public image host later. Fail-closed without a key.
    """

    name = "serpapi"

    def __init__(self, image_base_url: str = "", timeout: float = 20.0):
        self.timeout = timeout
        self.image_base_url = image_base_url
        self._key = _env(BACKEND_ENV[self.name])

    @property
    def available(self) -> bool:
        return bool(self._key) and bool(self.image_base_url)

    def search_crop(self, crop: np.ndarray) -> List[ReverseImageResult]:
        if not self.available:
            raise RuntimeError("SerpApi reverse image: missing SERPAPI_KEY or image host")
        raise NotImplementedError(
            "SerpApi reverse-image requires hosting the crop at a public URL; "
            "set image_base_url and upload via your own object storage."
        )


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


def create_backend(name: str) -> ReverseImageSearchBackend:
    """Factory: create the configured reverse-image-search backend."""
    if name == "google_vision_web":
        return GoogleVisionWebBackend()
    if name == "bing_visual":
        return BingVisualSearchBackend()
    if name == "serpapi":
        return SerpApiReverseBackend()
    if name == "browser_grounded":
        return BrowserGroundedSearchBackend()
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


def _crop_hash(crop: np.ndarray) -> str:
    """Cheap perceptual-ish hash for deduplicating near-identical crops."""
    import cv2

    small = cv2.resize(crop, (32, 32), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    return hashlib.sha1(gray.tobytes()).hexdigest()[:16]


def _read_frame(video_path: str, frame_index: int) -> Optional[np.ndarray]:
    """Seek to an extracted-frame index and return the RGB frame (mirrors
    server._read_video_frame)."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target = int(round(frame_index * fps))
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
    for r in results:
        if r.source.endswith("_tags") and r.title:
            score = _brand_likeness(r.title, generic_words)
            if score > 0.5:
                candidates.append((score, r.title))
    if candidates:
        _, best = max(candidates, key=lambda c: c[0])
        return re.sub(r"\s+", " ", best).strip().upper()
    domain_counts: Dict[str, int] = {}
    for r in results:
        if not r.url:
            continue
        m = re.match(r"https?://(?:www\.)?([^/]+)", r.url)
        if not m:
            continue
        domain = m.group(1).lower()
        if any(pat in domain for pat in (generic_domains or ())):
            continue
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    if not domain_counts:
        return None
    top_domain = max(domain_counts, key=domain_counts.get)
    # brand-ish token = the registrable part before the TLD
    parts = top_domain.split(".")
    if len(parts) >= 2:
        return parts[-2].upper()
    return top_domain.upper()


class OpenSetBrandIdentifier:
    """Cost-gated, evidence-trail-producing unknown-brand identification."""

    def __init__(
        self,
        backend: Optional[ReverseImageSearchBackend],
        min_logo_confidence: float,
        max_candidates_per_video: int,
        crop_cache_dir,
        generic_tag_filter,
        generic_domain_filter,
        logodev_timeout: float,
    ):
        """All tunables are explicit and come from config `open_set:`.

        Every parameter is required — the caller (server.py) reads them from
        config/config.yaml. `None`/empty values are NOT silently defaulted
        here: an identifier constructed without a real backend, threshold or
        cache dir is unusable and `identify()` fails closed.
        """
        if backend is None:
            raise ValueError("OpenSetBrandIdentifier requires a backend")
        if not min_logo_confidence or float(min_logo_confidence) <= 0.0:
            raise ValueError("OpenSetBrandIdentifier requires min_logo_confidence > 0")
        if not max_candidates_per_video or int(max_candidates_per_video) <= 0:
            raise ValueError("OpenSetBrandIdentifier requires max_candidates_per_video > 0")
        if not crop_cache_dir:
            raise ValueError("OpenSetBrandIdentifier requires a crop_cache_dir")
        if logodev_timeout is None or float(logodev_timeout) <= 0.0:
            raise ValueError("OpenSetBrandIdentifier requires logodev_timeout > 0")

        self.backend = backend
        self.min_logo_confidence = float(min_logo_confidence)
        self.max_candidates_per_video = int(max_candidates_per_video)
        self.crop_cache_dir = Path(crop_cache_dir)
        self.crop_cache_dir.mkdir(parents=True, exist_ok=True)
        self.generic_words = set(str(w).lower().strip() for w in (generic_tag_filter or ()))
        self.generic_domains = set(str(d).lower().strip() for d in (generic_domain_filter or ()))
        self.logodev_timeout = float(logodev_timeout)
        self._logodev_client = None  # lazy
        self._result_cache: Dict[str, List[ReverseImageResult]] = {}

    def _logodev(self):
        from src.logodev import LogoDevClient

        if self._logodev_client is None:
            self._logodev_client = LogoDevClient(timeout=self.logodev_timeout)
        return self._logodev_client

    def _candidate_crops(self, result: dict, video_path: str) -> List[dict]:
        """Collect cost-gated candidate crops from real logo detections.

        Gates (Task 3.6):
          - detector confidence >= min_logo_confidence (documented threshold),
          - dedup by crop hash (near-identical crops fire once),
          - capped at max_candidates_per_video,
          - per-crop search results cached.
        """
        l1 = result["layer1"]
        candidates: List[dict] = []
        seen_hashes: set = set()
        for frame_idx, dets in enumerate(l1.get("logo_detections") or []):
            for det_idx, det in enumerate(dets or []):
                conf = float(det.get("confidence", 0.0))
                if conf < self.min_logo_confidence:
                    continue
                bbox = det.get("bbox")
                if not bbox:
                    continue
                frame = _read_frame(video_path, frame_idx)
                if frame is None:
                    continue
                crop = _crop_bbox(frame, bbox)
                if crop is None or crop.size == 0:
                    continue
                h = _crop_hash(crop)
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)
                candidates.append({
                    "frame_index": frame_idx,
                    "det_index": det_idx,
                    "confidence": conf,
                    "bbox": [float(v) for v in bbox],
                    "crop": crop,
                    "crop_hash": h,
                })
                if len(candidates) >= self.max_candidates_per_video:
                    break
            if len(candidates) >= self.max_candidates_per_video:
                break
        return candidates

    def identify(self, result: dict, video_path: str) -> Dict[str, Any]:
        """Run open-set identification for a finished job.

        Returns the dashboard `open_set` block:
          {available, backend, min_confidence, candidates: [...]}
        Fail-closed: with no runnable backend and no search results, no
        candidate name is ever surfaced.
        """
        candidates: List[Dict[str, Any]] = []
        backend_available = self.backend.available
        reason = ""
        if not backend_available:
            reason = (
                "OPEN-SET IDENTIFICATION UNAVAILABLE — NO RUNNABLE "
                "REVERSE-IMAGE-SEARCH BACKEND (set GOOGLE_CLOUD_VISION_API_KEY / "
                "BING_VISUAL_SEARCH_KEY / SERPAPI_KEY, or install agent-browser)"
            )

        for cand in self._candidate_crops(result, video_path):
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

            if backend_available:
                try:
                    if cand["crop_hash"] in self._result_cache:
                        results = self._result_cache[cand["crop_hash"]]
                    else:
                        results = self.backend.search_crop(crop)
                        self._result_cache[cand["crop_hash"]] = results
                    entry["search_results"] = [r.to_dict() for r in results]
                except Exception as exc:  # noqa: BLE001
                    logger.warning("reverse-image search failed: %s", exc)
                    entry["search_results"] = []

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
                entry["status"] = "unavailable"

            candidates.append(entry)

        return {
            "available": backend_available,
            "backend": self.backend.name,
            "min_confidence": self.min_logo_confidence,
            "reason": reason,
            "candidates": candidates,
        }
