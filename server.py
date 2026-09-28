"""
ADSCENE API server.

Serves the static frontend and wraps the Phase 1 pipeline
(src.pipeline.Phase1Pipeline) behind a job queue so uploads
run in background threads and the UI can poll for progress.

Remediation notes (see MAJOR_REMEDIATION_REPORT.md / REMEDIATION_ESCALATION_REPORT.md):

  * Brand attribution has been re-enabled: the dashboard products table now
    aggregates RESOLVED on-screen brands from the logo pipeline (products_status
    AVAILABLE / NO_PRODUCTS; see _build_dashboard). Unresolved "UNKNOWN BRAND"
    regions are never asserted as a real brand. Scene cards emit "LOGO REGION" +
    real model confidence for unresolved boxes; ads come from real ASR brand
    mentions; recommendations come from the Layer 3 recommender.
    _validate_dashboard_bounds() raises loudly on any SCENE n > num_frames
    instead of silently trimming.
  * Duration is computed from the source video's real codec frame count
    (video_total_frames / video_fps), fixing the sampled-frames/source-fps unit
    bug that previously rendered a 60 s video as "3 SEC".
  * Outreach (DRAFT EMAIL) is gated on config ui.outreach_enabled (fail-closed
    OFF when config is missing) and every draft requires an externally verified
    logo.dev result plus at least one real on-screen appearance.
  * Open-set brand identification (config open_set:) surfaces unknown-brand
    candidates with a full evidence trail; it fails closed (no names) without
    a runnable reverse-image-search backend.

Usage:
    python server.py

Env:
    ADSCENE_PORT    overrides the default port (5000)
"""

import os
import random
import string
import sys
import tempfile
import threading
import uuid
import functools
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Optional

import yaml
from flask import Flask, Response, jsonify, request, send_file, send_from_directory

from src.layer1.spec_extractor import extract_creator, extract_specs
from src.logodev import LogoDevClient
from src.outreach import generate_personalized_outreach

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

PORT = int(os.environ.get("ADSCENE_PORT", "5000"))
MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB
ALLOWED_EXT = {".mp4", ".mov", ".webm", ".avi", ".mkv", ".m4v"}

UPLOAD_DIR = Path(tempfile.gettempdir()) / "adscene_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder="static", static_url_path="")

JOBS: dict = {}
JOBS_LOCK = threading.Lock()
# Cap on in-memory completed jobs. Older finished/errored jobs are pruned from
# the JOBS dict (they remain available via the SQLite archive) so a long-running
# server doesn't accumulate unbounded memory.
MAX_MEMORY_JOBS = 200

CONFIG_PATH = ROOT / "config" / "config.yaml"


def _load_config() -> dict:
    """Load config/config.yaml; fail closed (empty dict) on any error."""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg if isinstance(cfg, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


CFG = _load_config()

# Part A incident containment — outreach is feature-flagged from config and
# defaults OFF (fail closed) when the flag is missing or config fails to load.
_UI_CFG = CFG.get("ui", {}) if isinstance(CFG.get("ui"), dict) else {}
OUTREACH_ENABLED = bool(_UI_CFG.get("outreach_enabled", False))
OUTREACH_REASON = (
    _UI_CFG.get("outreach_reason")
    or "DRAFT EMAIL DISABLED — PENDING DATA-INTEGRITY REVIEW"
)

# Gemini-grounded brand email lookup (official contact + HR addresses). Fails
# closed: ON without a GEMINI_API_KEY yields NO emails (catalog placeholders
# stay), and no ungrounded/fabricated address is ever surfaced.
EMAIL_LOOKUP_ENABLED = bool(_UI_CFG.get("gemini_email_lookup", True))

# Open-set identification config (escalation Task 3) — every tunable comes from
# config; the code fails closed naming any missing key.
_OPEN_SET_CFG = CFG.get("open_set", {}) if isinstance(CFG.get("open_set"), dict) else {}
OPEN_SET_CROP_DIR = ROOT / str(_OPEN_SET_CFG.get("crop_cache_dir", "static/openset_crops"))

# logo.dev brand-validation cache (B2b fabrication safeguard). Only status
# "verified" may ever be presented as a real brand.
_BRAND_VALIDATION_CACHE: dict = {}


# ── Phase 3 SaaS: auth (feature-flagged) + SQLite job store ────────────
from src.auth import SessionManager, hash_password, verify_password
from src.store import JobStore, prune_for_store

_AUTH_CFG = CFG.get("auth", {}) if isinstance(CFG.get("auth"), dict) else {}
AUTH_ENABLED = bool(_AUTH_CFG.get("enabled", False))
SESSION_MANAGER = SessionManager(
    ttl_seconds=int(_AUTH_CFG.get("session_ttl_seconds", 8 * 3600))
)
_DB_PATH = os.environ.get("ADSCENE_DB_PATH")
if not _DB_PATH:
    _store_db = (CFG.get("store") or {}).get("db_path")
    _DB_PATH = str(ROOT / _store_db) if _store_db else str(ROOT / "var" / "contextlens.db")
JOB_STORE = JobStore(_DB_PATH)

_ADMIN_USER = str(
    _AUTH_CFG.get("admin_user") or os.environ.get("ADSCENE_ADMIN_USER", "admin")
)
_ADMIN_PASSWORD = str(
    os.environ.get("ADSCENE_ADMIN_PASSWORD") or _AUTH_CFG.get("admin_password") or ""
)
# PBKDF2-hashed admin credential; None means login is impossible (fail closed).
ADMIN_CREDENTIALS = (
    {"user": _ADMIN_USER, "password_hash": hash_password(_ADMIN_PASSWORD)}
    if _ADMIN_PASSWORD else None
)


# ── Helpers ─────────────────────────────────────────────────


def _silence_ffmpeg():
    """Suppress C-level FFmpeg/libav stderr noise (e.g. H.264 'mmco: unref
    short failure') emitted by OpenCV's VideoCapture during imprecise seeking.

    libav writes these recovery messages directly to file descriptor 2, bypassing
    Python's sys.stderr, so neither logging nor contextlib.redirect_stderr can
    catch them. We redirect the raw fd 2 to os.devnull for the duration of one
    decode call. Returns a no-op context manager.
    """
    import contextlib

    @contextlib.contextmanager
    def _cm():
        _devnull = os.open(os.devnull, os.O_WRONLY)
        _saved = os.dup(2)
        try:
            os.dup2(_devnull, 2)
            yield
        finally:
            os.dup2(_saved, 2)
            os.close(_saved)
            os.close(_devnull)

    return _cm()


def _new_job_id() -> str:
    part_a = "".join(random.choices(string.ascii_uppercase + string.digits, k=8))
    part_b = "".join(random.choices(string.ascii_uppercase + string.digits, k=5))
    return f"{part_a}-{part_b}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _normalize_brand(name: str) -> str:
    return name.strip().upper()


def _catalog_lookup(brand: str):
    from src.brand_catalog import BRAND_CATALOG
    return BRAND_CATALOG.get(_normalize_brand(brand))


def _enrich_recommendations(recs, appearance_counts=None, emails=None) -> list:
    """Add catalog contact/attribution data to each recommendation (additive).

    The dashboard `products` table is deliberately empty (Part A production-
    validation containment), so the outreach UI instead drives off ranked Layer 3
    recommendations. This helper attaches the brand-catalog contact fields so the
    editor can prefill targets — always additive, never overriding the recommender's
    own scores/reasons, and remaining subject to the logo.dev "verified" gate.
    `appearance_counts` (normalized brand -> on-screen count) and `emails`
    (normalized brand -> gemini-grounded email list) are passed in from the caller
    so real pipeline data is used and email lookup stays fail-closed/testable.
    """
    from src.brand_catalog import BRAND_CATALOG

    counts = {(_normalize_brand(b) if isinstance(b, str) else b): c
              for b, c in (appearance_counts or {}).items()}
    out = []
    for rec in recs or []:
        if not isinstance(rec, dict) or not rec.get("brand"):
            out.append(dict(rec) if isinstance(rec, dict) else rec)
            continue
        brand = rec.get("brand")
        key = _normalize_brand(brand)
        info = (BRAND_CATALOG or {}).get(brand) or {}
        enriched = dict(rec)
        enriched.setdefault("contact_email", info.get("contact_email"))
        enriched.setdefault("contact_website", info.get("contact_website"))
        enriched.setdefault("contact_verified", bool(info.get("contact_verified")))
        if key in counts:
            enriched["appearances"] = int(counts.get(key, 0) or 0)
        else:
            enriched.setdefault("appearances", 0)
        found = (emails or {}).get(key)
        if found and found.get("emails"):
            primary = _pick_primary_email(found, brand)
            enriched["contact_email"] = primary or enriched.get("contact_email")
            enriched["contact_email_source"] = "gemini_grounding"
            enriched["contact_verified"] = False
            enriched["gemini_emails"] = found["emails"]
            enriched["hr_emails"] = [e["email"] for e in found["emails"] if e.get("type") == "hr"]
            enriched["email_evidence"] = found.get("evidence") or []
        out.append(enriched)
    return out


def _pick_primary_email(found: dict, brand: str) -> Optional[str]:
    """Choose the best outreach address: brand's own domain first, never HR."""
    emails = (found or {}).get("emails") or []
    nonhr = [e for e in emails if e.get("type") != "hr"] or emails
    slug = str(brand or "").lower()
    if slug:
        for e in nonhr:
            domain = (e.get("email") or "").split("@", 1)[-1].lower()
            if slug in domain:
                return e.get("email")
    return nonhr[0].get("email") if nonhr else None


_EMAIL_LOOKUP_CACHE: Dict[str, dict] = {}


def _lookup_brand_emails(products: list) -> dict:
    """Run the fail-closed Gemini email lookup for resolved on-screen brands.

    Only brands that actually appeared on screen get a lookup (SUGGESTED brands
    from the knowledge graph never trigger network cost). Every call is wrapped:
    any exception (no key, timeout, malformed answer) yields no entry, leaving
    the catalog placeholder untouched. Results are cached per brand for the life
    of the server.
    """
    if not EMAIL_LOOKUP_ENABLED:
        return {}
    from src import email_lookup

    found: Dict[str, dict] = {}
    for p in products or []:
        brand = (p.get("brand") or "").strip()
        if not brand:
            continue
        key = _normalize_brand(brand)
        if key in _EMAIL_LOOKUP_CACHE:
            entry = _EMAIL_LOOKUP_CACHE[key]
        else:
            try:
                entry = email_lookup.lookup_brand_emails(brand)
            except Exception:  # fail closed — never let email lookup break a build
                entry = {"status": "error"}
            _EMAIL_LOOKUP_CACHE[key] = entry
        if entry.get("status") == "ok" and entry.get("emails"):
            found[key] = entry
    return found


def _request_token() -> str:
    """Extract the bearer token from the Authorization header."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[len("Bearer "):].strip()
    return request.headers.get("X-Auth-Token", "").strip()


def login_required(fn):
    """Gate a route behind a valid session token when auth is enabled.

    When AUTH_ENABLED is False this is a pass-through, preserving the current
    standalone/desktop behavior and existing tests.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not AUTH_ENABLED:
            return fn(*args, **kwargs)
        token = _request_token()
        if SESSION_MANAGER.validate(token) is None:
            return jsonify({"error": "AUTH REQUIRED", "status": 401}), 401
        return fn(*args, **kwargs)
    return wrapper



def _job(job_id: str) -> dict:
    with JOBS_LOCK:
        return JOBS.get(job_id)


def _scene_number_map(scene_indexes):
    """Map ordinal position → SCENE 001-style numbering."""
    return {idx: n + 1 for n, idx in enumerate(sorted(scene_indexes))}


def _read_video_frame(video_path: str, frame_index: int, video_stride: int = 1):
    """Seek to the source frame behind an extracted-frame index and return the
    RGB frame (cv2).

    `frame_index` is a position in the pipeline's sampled frame list. Because
    long videos are decoded with a stride > 1, sampled index i corresponds to
    source frame i*stride; we seek to that real source frame so the displayed
    frame matches the timestamp/context the analysis attributed to it.
    """
    import cv2

    with _silence_ffmpeg():
        cap = cv2.VideoCapture(video_path)
        ret = False
        frame = None
        try:
            if not cap.isOpened():
                return None
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            target = int(round(frame_index * max(1, video_stride)))
            if total > 0:
                target = min(target, total - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, target))
            ret, frame = cap.read()
        finally:
            cap.release()
    if not ret or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ── Dashboard bounds (recurrence-proof assertion) ────────────


def _validate_dashboard_bounds(dash: dict, num_frames: int) -> dict:
    """Hard stop for the `SCENE n > num_frames` failure mode.

    Raises RuntimeError("DASHBOARD BOUND VIOLATION — ...") instead of silently
    trimming, so a recurrence is loud and marks the job errored.
    """
    scenes = dash.get("scenes") or []
    if len(scenes) > num_frames:
        raise RuntimeError(
            f"DASHBOARD BOUND VIOLATION — {len(scenes)} scenes exceed num_frames={num_frames}"
        )
    for s in scenes:
        scene_n = s.get("n")
        if scene_n is not None and int(scene_n) > num_frames:
            raise RuntimeError(
                f"DASHBOARD BOUND VIOLATION — SCENE {scene_n} exceeds num_frames={num_frames}"
            )
        frame_index = s.get("frame_index")
        if frame_index is not None and int(frame_index) >= num_frames:
            raise RuntimeError(
                f"DASHBOARD BOUND VIOLATION — scene frame_index {frame_index} "
                f">= num_frames={num_frames}"
            )
    for c in (dash.get("open_set") or {}).get("candidates") or []:
        frame_index = c.get("frame_index")
        if frame_index is not None and int(frame_index) >= num_frames:
            raise RuntimeError(
                f"DASHBOARD BOUND VIOLATION — open-set frame_index {frame_index} "
                f">= num_frames={num_frames}"
            )
    return dash


# ── Open-set identification (escalation Task 3) ──────────────


def _build_open_set(result: dict, video_path: str) -> dict:
    """Build the dashboard `open_set` block from config — fails closed.

    Without a runnable backend, a disabled flag, or an incomplete config the
    block reports `available: false` with the exact reason and zero candidates;
    a candidate name is never surfaced from a guess.
    """
    backend_name = str(_OPEN_SET_CFG.get("backend", "browser_grounded"))
    min_confidence = float(_OPEN_SET_CFG.get("min_logo_confidence", 0.0) or 0.0)

    if not _OPEN_SET_CFG.get("enabled"):
        return {
            "available": False,
            "backend": backend_name,
            "min_confidence": min_confidence,
            "min_crop_area": float(_OPEN_SET_CFG.get("min_crop_area", 0.0) or 0.0),
            "max_crop_aspect": float(_OPEN_SET_CFG.get("max_crop_aspect", 0.0) or 0.0),
            "reason": "OPEN-SET IDENTIFICATION DISABLED (config open_set.enabled: false)",
            "candidates": [],
            "rejected": [],
            "skipped_counts": {},
            "resolved": 0,
        }

    required_keys = [
        "backend",
        "min_logo_confidence",
        "min_crop_area",
        "max_crop_aspect",
        "max_candidates_per_video",
        "crop_cache_dir",
        "generic_tag_filter",
        "generic_domain_filter",
        "logodev_timeout",
    ]
    missing = [k for k in required_keys if k not in _OPEN_SET_CFG]
    if missing:
        return {
            "available": False,
            "backend": backend_name,
            "min_confidence": min_confidence,
            "min_crop_area": float(_OPEN_SET_CFG.get("min_crop_area", 0.0) or 0.0),
            "max_crop_aspect": float(_OPEN_SET_CFG.get("max_crop_aspect", 0.0) or 0.0),
            "reason": (
                "OPEN-SET IDENTIFICATION UNAVAILABLE — CONFIG MISSING KEYS: "
                + ", ".join(missing)
            ),
            "candidates": [],
            "rejected": [],
            "skipped_counts": {},
            "resolved": 0,
        }

    try:
        from src.openset import OpenSetBrandIdentifier, create_backend

        identifier = OpenSetBrandIdentifier(
            backend=create_backend(str(_OPEN_SET_CFG["backend"])),
            min_logo_confidence=float(_OPEN_SET_CFG["min_logo_confidence"]),
            min_crop_area=float(_OPEN_SET_CFG["min_crop_area"]),
            max_crop_aspect=float(_OPEN_SET_CFG["max_crop_aspect"]),
            max_candidates_per_video=int(_OPEN_SET_CFG["max_candidates_per_video"]),
            crop_cache_dir=str(OPEN_SET_CROP_DIR),
            generic_tag_filter=list(_OPEN_SET_CFG["generic_tag_filter"] or []),
            generic_domain_filter=list(_OPEN_SET_CFG["generic_domain_filter"] or []),
            logodev_timeout=float(_OPEN_SET_CFG["logodev_timeout"]),
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "available": False,
            "backend": backend_name,
            "min_confidence": min_confidence,
            "min_crop_area": float(_OPEN_SET_CFG.get("min_crop_area", 0.0) or 0.0),
            "max_crop_aspect": float(_OPEN_SET_CFG.get("max_crop_aspect", 0.0) or 0.0),
            "reason": f"OPEN-SET IDENTIFICATION UNAVAILABLE — {exc}",
            "candidates": [],
            "rejected": [],
            "skipped_counts": {},
            "resolved": 0,
        }

    try:
        return identifier.identify(result, video_path)
    except Exception as exc:  # noqa: BLE001
        return {
            "available": False,
            "backend": backend_name,
            "min_confidence": min_confidence,
            "min_crop_area": float(_OPEN_SET_CFG.get("min_crop_area", 0.0) or 0.0),
            "max_crop_aspect": float(_OPEN_SET_CFG.get("max_crop_aspect", 0.0) or 0.0),
            "reason": f"OPEN-SET IDENTIFICATION FAILED — {exc}",
            "candidates": [],
            "rejected": [],
            "skipped_counts": {},
            "resolved": 0,
        }


# ── Dashboard compilation ────────────────────────────────────


def _scene_caption(objects, logos, ocr_texts,
                   max_objects: int = 6, max_brands: int = 4) -> str:
    """Deterministic, truthful one-line caption synthesized from REAL detections.

    Combines the distinct COCO/open-vocab object labels, the distinct RESOLVED
    brand names, and any OCR text read on the frame — flat text like

        "Person, cell phone, wristwatch · SAMSUNG · on-screen text 'Mac Mini'"

    Empty string when nothing was detected on the frame. Never a model guess:
    every token comes from a real detector box, a resolved brand, or an OCR read
    (anti-fabrication; see ARCHITECTURE_AND_RESEARCH precision-over-recall).
    """
    parts = []
    obj_names = list(dict.fromkeys(
        str(o.get("class_name", "")).strip() for o in (objects or [])
        if o.get("class_name")
    ))
    if obj_names:
        parts.append(", ".join(obj_names[:max_objects]))
    brand_names = list(dict.fromkeys(
        str(o.get("brand", "")).strip()
        for o in (logos or [])
        if o.get("brand") and _normalize_brand(o.get("brand")) != "UNKNOWN BRAND"
    ))
    if brand_names:
        parts.append(" · ".join(brand_names[:max_brands]))
    texts = [str(t).strip() for t in (ocr_texts or []) if str(t).strip()]
    if texts:
        cap = " ".join(texts)[:80].strip()
        parts.append(f"on-screen text '{cap}'")
    return " · ".join(parts)


def _build_product_resolutions(l1, src_timestamp_fn, nearest_frame_fn):
    """Dashboard `product_resolutions` block — timed product->brand resolutions.

    Surfaces Layer 2b resolutions with REAL clip pointers for the UI's
    PLAY-@-time chips. Each entry: brand, product, snippet, resolution
    tier/source, confidence, mode (speech | ocr), a real start_time (speech) or
    a frame-derived timestamp (ocr), and the nearest scene frame for clip
    seeking. Fail-closed: no resolutions => []; an entry without a usable
    timestamp keeps start_time unset so the UI renders NO TIMESTAMP rather than
    fabricating a clip point.
    """
    prs = l1.get("product_resolutions") or []
    out = []
    for r in prs:
        brand = r.get("brand")
        if not brand:
            continue
        ts = r.get("start_time")
        frame_idx = r.get("frame_index")
        if ts is None and frame_idx is not None:
            ts = src_timestamp_fn(frame_idx)
        out.append({
            "brand": brand,
            "product": r.get("product_span") or r.get("span") or brand,
            "snippet": (r.get("span") or "")[:80],
            "resolution_tier": r.get("resolution_tier"),
            "source": r.get("source"),
            "confidence": r.get("confidence"),
            # Product-resolver confidence is the resolution quality of the
            # brand->product attribution (same metric family as the scene and
            # product-table numbers).
            "confidence_metric": "resolution_quality",
            "mode": r.get("mode", "ocr"),
            "start_time": ts,
            "end_time": r.get("end_time"),
            "frame_index": nearest_frame_fn(ts),
        })
    out.sort(key=lambda p: p.get("start_time")
             if p.get("start_time") is not None else float("inf"))
    return out


def _build_dashboard(result: dict, job: dict) -> dict:
    l1 = result["layer1"]
    num_frames = result.get("num_frames", 0)
    video_fps = result.get("video_fps", 0.0)
    video_total_frames = result.get("video_total_frames", 0)
    # Sampled indices in the detection/logo/OCR arrays are positions in the
    # sampled frame list, which is strided by `video_stride` source frames.
    # Convert a sampled index to its real source-frame timestamp via this.
    video_stride = int(result.get("video_stride", 1) or 1)
    video_fps_safe = video_fps or 1.0

    def _src_timestamp(frame_idx: int) -> float:
        """Real source timestamp (sec) for a sampled frame index."""
        return (frame_idx * video_stride) / video_fps_safe

    # Duration from the source video's real codec frame count (fixes the unit
    # bug: sampled-frame count / source fps previously rendered 60 s as 3 s).
    duration_sec = (video_total_frames / video_fps) if video_fps else 0.0

    # ── Scenes ──────────────────────────────────────────────
    # A scene row exists for ANY frame with real signal — object detections,
    # logo detections (resolved or unresolved), on-screen OCR text, or a spoken
    # brand mention — never gated on object detections alone (that was the
    # invisible-scenes bug: frames with only a logo/OCR got no row). Each of
    # these is a deterministic signal, so the render condition matches design
    # principle "nothing renders that isn't real".
    scenes = []
    scene_indexes = set()
    for frame_idx, dets in enumerate(l1["scene_object_detections"]):
        if dets:
            scene_indexes.add(frame_idx)
    for frame_idx, dets in enumerate(l1.get("logo_detections") or ()):
        if dets:
            scene_indexes.add(frame_idx)
    for frame_idx, ocr in enumerate(l1.get("ocr_results") or ()):
        if ocr:
            scene_indexes.add(frame_idx)
    for m in l1.get("brand_mentions") or ():
        ts = m.get("start_time")
        if ts is not None and video_fps:
            idx = int(ts * video_fps / max(1, video_stride))
            scene_indexes.add(max(0, min(num_frames - 1, idx)))
    scene_nums = _scene_number_map(scene_indexes)
    for frame_idx in sorted(scene_indexes):
        objects = l1["scene_object_detections"][frame_idx]
        logos = l1["logo_detections"][frame_idx]
        _ocr_here = (
            (l1.get("ocr_results") or [])[frame_idx]
            if frame_idx < len(l1.get("ocr_results") or []) else []
        )
        _ocr_texts = [t.get("text", "") for t in _ocr_here if t.get("text")]
        scenes.append({
            "n": scene_nums[frame_idx],
            "frame_index": frame_idx,
            "timestamp": round(_src_timestamp(frame_idx), 2),
            "objects": [
                {"class_name": o["class_name"], "confidence": o["confidence"]}
                for o in objects
            ],
            # Brand label: resolved brands print their name; a logo that could
            # not be resolved is grouped as "UNKNOWN BRAND" (limitation #5) so it
            # is explicit rather than a nameless box. Never a fabricated name.
            # Confidence: for RESOLVED brands we show the resolution quality
            # (OCR=0.90 / retrieval & temporal=0.70 / class=0.50) — how much we
            # trust the brand attribution itself — NOT the raw detector box
            # confidence, which was ~13% even for a correct OCR read. UNKNOWN
            # chips keep the raw detector confidence (the only honest number for
            # boxes we failed to resolve).
            "logos": [
                {
                    "class_name": o.get("brand") or "UNKNOWN BRAND",
                    "confidence": (
                        o["resolution_quality"]
                        if o.get("brand") and o.get("resolution_quality") is not None
                        else o["confidence"]
                    ),
                    # Canonical metric name for this number (design principle:
                    # "one number, one meaning"). RESOLVED brands always show
                    # resolution_quality — how much we trust the brand
                    # attribution itself. UNKNOWN boxes keep the raw detector
                    # box confidence (the only honest number for unresolved).
                    "confidence_metric": "resolution_quality"
                    if o.get("brand")
                    else "detector_box_confidence",
                }
                for o in logos
            ],
            # Deterministic, real-detection caption line (objects + resolved
            # brand names + OCR text read this frame). Every token is verifiable
            # against the raw detections — never a generated/model guess.
            "caption": _scene_caption(objects, logos, _ocr_texts),
        })

    # ── Products: aggregate resolved brand appearances ──────
    # Build the per-brand product table from logo detections that the brand
    # resolver named. Only RESOLVED brands (det["brand"] set) become products;
    # unresolved detections ("UNKNOWN BRAND") are excluded so a nameless box
    # never becomes a fabricated brand. Open-set candidates stay separate.
    product_map: Dict[str, dict] = {}
    for frame_idx, dets in enumerate(l1["logo_detections"]):
        if not dets:
            continue
        frame_texts = [
            t.get("text", "")
            for t in (l1.get("ocr_results") or [])[frame_idx]
            if t.get("text")
        ]
        for det in dets:
            # Only a RESOLVED brand becomes a product. `class_name` is the raw
            # detector box label (e.g. "ROLEX logo") and is NOT a confirmed brand
            # attribution; a logo the resolver could not name stays "UNKNOWN
            # BRAND" and must never be asserted as a real brand.
            brand = det.get("brand")
            if not brand or _normalize_brand(brand) == "UNKNOWN BRAND":
                continue
            key = _normalize_brand(brand)
            if key not in product_map:
                info = _catalog_lookup(brand) or {}
                product = info.get("product") or brand
                category = (info.get("categories") or [info.get("category")])[0] \
                    if (info.get("categories") or info.get("category")) else "GENERAL"
                product_map[key] = {
                    "brand": brand,
                    "product": product,
                    "category": category,
                    "appearances": set(),
                    "confidences": [],
                    "frame_texts": [],
                    "contact_email": info.get("contact_email"),
                    "contact_website": info.get("contact_website"),
                    "contact_verified": bool(info.get("contact_verified")),
                }
            entry = product_map[key]
            entry["appearances"].add(frame_idx)
            # Canonical product confidence: the brand-attribution quality
            # (resolution_quality), NOT the raw detector box confidence. The
            # scene chip, product table, and recommendations must all point at
            # the SAME metric so one detection never renders three rival
            # unlabeled numbers (the Samsung bug). Raw box conf falls back only
            # when a resolved brand carries no resolution quality.
            qual = det.get("resolution_quality")
            entry["confidences"].append(
                float(qual) if qual is not None else float(det.get("confidence") or 0.0)
            )
            entry["frame_texts"].extend(frame_texts)
    product_list = []
    for pid, entry in product_map.items():
        entry["appearances"] = sorted(entry["appearances"])
        entry["appearance_count"] = len(entry["appearances"])
        entry["first_frame"] = entry["appearances"][0]
        entry["confidence"] = round(
            (sum(entry["confidences"]) / len(entry["confidences"]))
            if entry["confidences"] else 0.0,
            3,
        )
        entry["confidence_metric"] = "resolution_quality"
        del entry["confidences"]
        product_list.append(entry)
    product_list.sort(key=lambda p: len(p["appearances"]), reverse=True)
    if product_list:
        products_status = "AVAILABLE"
        products_status_reason = (
            f"AGGREGATED FROM {len(product_list)} RESOLVED ON-SCREEN BRAND(S)."
        )
    else:
        products_status = "NO_PRODUCTS"
        products_status_reason = (
            "NO BRAND WAS RESOLVED ON SCREEN — logo detections could not be "
            "attributed to a catalog brand, so no product is asserted."
        )

    # ── Spec callouts + creator attribution (Phase 3) ────────
    # Creator on-screen overlays fall into three buckets; brand wordmarks go to
    # BrandResolver, while spec callouts (screen size, thickness, battery,
    # material, processor) and the creator-attribution card (handle/followers)
    # are structured, high-value, brand-catalog-INDEPENDENT data. They are
    # extracted here from the union of (a) full-frame OCR results and (b) the
    # per-logo crop-OCR text attached by BrandResolver (so overlay text inside a
    # logo region contributes even when full-frame OCR skipped that frame).
    # Fail-closed: any missing/sparse OCR yields empty output, never a guess.
    spec_fields: Dict[str, dict] = {}
    creator_seen = {}
    for frame_idx in range(num_frames):
        ocr_texts = list((l1.get("ocr_results") or [])[frame_idx]) \
            if frame_idx < len(l1.get("ocr_results") or []) else []
        joined = " ".join(
            [t.get("text", "") for t in ocr_texts if t.get("text")]
        )
        logo_crop_texts = [
            det.get("ocr_text", "") for det in l1["logo_detections"][frame_idx]
            if det.get("ocr_text")
        ]
        if logo_crop_texts:
            joined = (joined + " " + " ".join(logo_crop_texts)).strip()
        if not joined:
            continue
        ts = _src_timestamp(frame_idx)
        for s in extract_specs(joined):
            if s["field"] not in spec_fields:
                spec_fields[s["field"]] = {
                    **{k: s[k] for k in ("field", "value", "unit", "raw")},
                    "first_frame": frame_idx,
                    "timestamp": round(ts, 2),
                }
        for k, v in extract_creator(joined).items():
            if k in ("handle", "followers", "followers_label") and k not in creator_seen:
                creator_seen[k] = v
    specs = sorted(spec_fields.values(), key=lambda s: s["timestamp"])
    creator_card = {
        "handle": creator_seen.get("handle"),
        "followers": creator_seen.get("followers"),
        "followers_label": creator_seen.get("followers_label"),
    }

    # ── Ads: real ASR evidence only ─────────────────────────
    # Speech mentions carry REAL STT segment timestamps (when the ASR backend
    # provided them) so the dashboard can render a clip link to the exact
    # second a brand was spoken. `frame_index` maps the mention to the nearest
    # sampled scene frame for the clip-jump target.
    scene_ts = sorted(
        ((s["timestamp"], s["frame_index"]) for s in scenes if s.get("timestamp") is not None),
        key=lambda p: p[0],
    )

    def _nearest_scene_frame(ts: float) -> Optional[int]:
        if ts is None or not scene_ts:
            return None
        best = min(scene_ts, key=lambda p: abs(p[0] - ts))
        return best[1]

    ads = []
    for m in l1.get("brand_mentions", []):
        ts = m.get("start_time")
        ads.append({
            "brand": m.get("brand", "SPOKEN BRAND"),
            "product": m.get("brand", "SPOKEN BRAND"),
            "category": "SPEECH",
            "type": "SPEECH MENTION (ASR)",
            "scenes": [],
            "score": m.get("confidence", 0.0),
            "start_time": ts,
            "end_time": m.get("end_time"),
            "frame_index": _nearest_scene_frame(ts),
        })

    # ── Timed product resolutions (Layer 2b, speech + OCR) ───
    # Product->brand resolutions carry REAL clip pointers (speech STT timestamps
    # propagated from the timed brand mentions, or frame-derived timestamps for
    # on-screen OCR reads) so the UI can PLAY-@-time to each one. Fail-closed:
    # none resolve => empty block, never a guess.
    product_resolutions = _build_product_resolutions(
        l1,
        src_timestamp_fn=_src_timestamp,
        nearest_frame_fn=_nearest_scene_frame,
    )
    if product_resolutions:
        product_resolutions_status = "AVAILABLE"
        product_resolutions_status_reason = (
            f"{len(product_resolutions)} TIMED PRODUCT->BRAND RESOLUTION(S) "
            "(SPEECH ASR + ON-SCREEN OCR)."
        )
    else:
        product_resolutions_status = "NONE"
        product_resolutions_status_reason = (
            "NO PRODUCT NAME WAS RESOLVED TO A CATALOG BRAND FROM SPEECH OR "
            "ON-SCREEN OCR."
        )

    # ── Layer 3 recommendations ─────────────────────────────
    # Additive enrichment with catalog + timeline contact data so the outreach
    # UI can drive drafts off ranked recommendations (the `products` table is
    # deliberately empty per Part A containment). Only adds keys; never mutates
    # the recommenders' scores/reasons. Non-verified placeholder contacts are
    # still subject to the logo.dev "verified" gate before any draft is sent.
    appearance_counts = {
        p["brand"]: p["appearance_count"] for p in product_list
    }
    found_emails = _lookup_brand_emails(product_list)
    for p in product_list:
        entry = found_emails.get(_normalize_brand(p["brand"]))
        if entry and entry.get("emails"):
            primary = _pick_primary_email(entry, p["brand"])
            if primary:
                p["contact_email"] = primary
            p["contact_email_source"] = "gemini_grounding"
            p["contact_verified"] = False
            p["gemini_emails"] = entry["emails"]
            p["hr_emails"] = entry.get("hr_emails") or []
        else:
            p["contact_email_source"] = "catalog"
    recommendations = _enrich_recommendations(
        (result.get("layer3") or {}).get("recommendations", []),
        appearance_counts=appearance_counts,
        emails=found_emails,
    )

    # ── Open-set identification (fail-closed) ───────────────
    open_set = _build_open_set(result, job.get("video_path", ""))

    # ── Confidence summary ──────────────────────────────────
    l2b = result["layer2b"]
    confidence = l2b.get("confidence", 0.0)
    evidence_breakdown = {}
    for src, ev in (l2b.get("evidence_breakdown") or {}).items():
        evidence_breakdown[src] = {
            "strength": ev.get("strength", 0.0),
            "weight": ev.get("modulated_weight", 0.0),
            "contribution": ev.get("contribution", 0.0),
            "status": ev.get("status", "unknown"),
        }

    dash = {
        "title": job.get("title", "UNTITLED"),
        "creator": job.get("creator", "UNKNOWN"),
        "filename": job.get("filename", ""),
        "job_id": job["job_id"],
        "duration_sec": duration_sec or job.get("duration_sec", 0),
        "num_frames": num_frames,
        "video_total_frames": video_total_frames,
        "video_fps": video_fps,
        "has_audio": bool(result.get("has_audio")),
        "confidence": float(confidence),
        "is_confident": bool(l2b.get("is_confident")),
        "confidence_status": l2b.get("status", "unknown"),
        "evidence_breakdown": evidence_breakdown,
        "scenes": scenes,
        "products": product_list,
        "products_status": products_status,
        "products_status_reason": products_status_reason,
        "specs": specs,
        "creator_card": creator_card,
        "ads": ads,
        "product_resolutions": product_resolutions,
        "product_resolutions_status": product_resolutions_status,
        "product_resolutions_status_reason": product_resolutions_status_reason,
        "open_set": open_set,
        "recommendations": recommendations,
        "outreach_enabled": OUTREACH_ENABLED,
        "outreach_reason": OUTREACH_REASON,
        "transcript": l1.get("transcript", ""),
        "audio_events": l1.get("audio_events", [])[:20],
        "unknown_brand_regions": (result.get("layer2c") or {}).get("unknown_brand_regions", []),
        # Canary metric for the brand-attribution re-enable: fraction of logo
        # detections the resolver NAMED a real brand vs. left UNKNOWN. Watch it
        # across threshold/config changes; a sharp jump signals relaxed settings
        # feeding unverified candidates into a pipeline that now surfaces them.
        "resolver_acceptance": (result.get("layer2c") or {}).get(
            "resolver_acceptance"
        ) or {
            "total_logo_detections": 0,
            "resolved": 0,
            "unresolved": 0,
            "acceptance_rate": 0.0,
        },
    }
    return _validate_dashboard_bounds(dash, num_frames)


# ── Pipeline access (lazy singleton) ─────────────────────────

_pipeline = None
_pipeline_lock = threading.Lock()
_warmup_started = False
_warmup_lock = threading.Lock()


def get_pipeline():
    global _pipeline, _warmup_started
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                from src.pipeline import Phase1Pipeline

                _pipeline = Phase1Pipeline(device_override="auto")
                _attach_affinity_model(_pipeline)
    # Fire-and-forget model warmup on the first job so the expensive lazy model
    # loads (YOLO/DINOv2/PaddleOCR/Whisper/BEATs/logo) happen once, in parallel,
    # and are taken out of the first job's input->calculation critical path.
    if not _warmup_started:
        with _warmup_lock:
            if not _warmup_started:
                _warmup_started = True
                threading.Thread(
                    target=_warmup_models, args=(_pipeline,), daemon=True
                ).start()
    return _pipeline


def _attach_affinity_model(pipeline) -> None:
    """Train the LightGCN creator-brand affinity model on stored job history and
    attach it to the pipeline. Best-effort: on any failure (no torch, no
    interactions, training error) the pipeline keeps its cold-start fallback."""
    try:
        from src.layer3.affinity_trainer import train_affinity_from_jobs

        jobs = JOB_STORE.list()
        model, summary = train_affinity_from_jobs(jobs, seed=0)
        if summary.get("status") == "trained":
            pipeline.set_affinity_model(model)
            app.logger.info(
                "Attached trained affinity model: %s",
                {k: summary.get(k) for k in ("creators", "brands", "interactions")},
            )
    except Exception as exc:  # noqa: BLE001
        app.logger.warning("Affinity training skipped (cold start): %s", exc)


def _warmup_models(pipeline) -> None:
    try:
        pipeline.warmup()
    except Exception as exc:  # noqa: BLE001
        app.logger.warning("Model warmup failed (continuing): %s", exc)


def _persist_job(job: dict) -> None:
    """Persist a JSON-serializable snapshot of a job to the SQLite store.

    Keeps the dashboard + a pruned result (recommendations, creator profile,
    brand-memory summary) needed to render the UI and drive personalized
    outreach, dropping heavy embed payloads. Never raises — persistence is
    best-effort so a storage failure doesn't fail a completed analysis.
    """
    try:
        snapshot = {
            "job_id": job.get("job_id"),
            "status": job.get("status"),
            "stage": job.get("stage"),
            "filename": job.get("filename"),
            "title": job.get("title"),
            "creator": job.get("creator"),
            "created_at": job.get("created_at"),
            "finished_at": job.get("finished_at"),
            "dashboard": job.get("dashboard"),
            "forwarded": job.get("forwarded"),
            "result": prune_for_store(job.get("result") or {}),
        }
        JOB_STORE.save(job["job_id"], snapshot)
    except Exception as exc:  # noqa: BLE001
        app.logger.warning("Failed to persist job %s: %s", job.get("job_id"), exc)
    _prune_memory_jobs()


def _prune_memory_jobs() -> None:
    """Drop oldest finished/errored jobs from JOBS once the in-memory cap is
    exceeded (running jobs are never pruned). Persisted snapshots in the SQLite
    store are untouched, so pruned jobs remain viewable via the archive.

    The uploaded source video file for a pruned job is deleted here too: once a
    job leaves live memory its per-frame scene images can no longer be rendered
    (they require the in-memory result + video), so the file is unreachable and
    can be reclaimed. This keeps UPLOAD_DIR bounded at ~MAX_MEMORY_JOBS files
    instead of growing unboundedly. Deletion is best-effort and never raises.
    """
    to_delete = []
    with JOBS_LOCK:
        done = [jid for jid, j in JOBS.items() if j.get("status") in ("done", "error")]
        if len(done) > MAX_MEMORY_JOBS:
            done.sort(key=lambda jid: JOBS[jid].get("finished_at") or "")
            for jid in done[: len(done) - MAX_MEMORY_JOBS]:
                job = JOBS.pop(jid, None)
                if job:
                    to_delete.append(job.get("video_path"))
    for path in to_delete:
        if path:
            try:
                Path(path).unlink(missing_ok=True)
            except Exception as exc:  # noqa: BLE001
                app.logger.warning("Failed to remove uploaded video %s: %s", path, exc)


def _job_feed_push(job: dict) -> Callable[[str], None]:
    """Progress callback that records real pipeline stage messages on a job.

    Bounded to the last 50 entries; never raises — progress is best-effort."""

    def _push(msg: str) -> None:
        feed = job.get("feed")
        if feed is None:
            return
        try:
            feed.append(msg)
        except Exception:  # noqa: BLE001
            return
        del feed[:-50]

    return _push


def _run_job(job_id: str) -> None:
    job = _job(job_id)
    try:
        job["stage"] = "LOADING MODELS"
        pipeline = get_pipeline()
        job["stage"] = "EXTRACTING AUDIO & VISUAL SIGNALS"
        result = pipeline.process_video(
            job["video_path"], frame_rate=1.0,
            progress=_job_feed_push(job),
        )
        if "error" in result:
            raise RuntimeError(result["error"])
        job["stage"] = "COMPILING INTELLIGENCE"
        job["result"] = result
        job["dashboard"] = _build_dashboard(result, job)
        job["status"] = "done"
        job["stage"] = "COMPLETE"
    except Exception as exc:  # noqa: BLE001
        job["status"] = "error"
        job["error"] = str(exc)
        app.logger.exception("Job %s failed", job_id)
    finally:
        job["finished_at"] = _now_iso()
        _persist_job(job)


# ── Routes: static ───────────────────────────────────────────


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


# ── Routes: health / jobs ────────────────────────────────────


@app.get("/api/health")
def health():
    with JOBS_LOCK:
        return jsonify({
            "ok": True,
            "jobs": len(JOBS),
            "outreach_enabled": OUTREACH_ENABLED,
            "outreach_reason": OUTREACH_REASON,
        })


# ── Routes: auth (SaaS dashboard login) ──────────────────────


@app.post("/api/login")
def login():
    """Exchange credentials for a bearer session token.

    Claims: verify_password against the PBKDF2 hash. Fails closed — if auth is
    disabled, or no admin password is configured, or the password is wrong, we
    return 401/403 and never issue a token.
    """
    if not AUTH_ENABLED:
        return jsonify({"error": "AUTH DISABLED"}), 403
    if ADMIN_CREDENTIALS is None:
        return jsonify({"error": "AUTH UNCONFIGURED — NO ADMIN PASSWORD SET"}), 403
    data = request.get_json(silent=True) or {}
    user = (data.get("username") or "").strip()
    password = data.get("password") or ""
    if user != ADMIN_CREDENTIALS["user"]:
        return jsonify({"error": "INVALID CREDENTIALS"}), 401
    if not verify_password(password, ADMIN_CREDENTIALS["password_hash"]):
        return jsonify({"error": "INVALID CREDENTIALS"}), 401
    token = SESSION_MANAGER.create(user)
    return jsonify({"ok": True, "token": token, "user": user})


@app.post("/api/logout")
def logout():
    token = _request_token()
    SESSION_MANAGER.revoke(token)
    return jsonify({"ok": True})


@app.get("/api/me")
def me():
    token = _request_token()
    user = SESSION_MANAGER.validate(token) if AUTH_ENABLED else None
    return jsonify({
        "authenticated": bool(user),
        "user": user,
        "auth_enabled": AUTH_ENABLED,
    })


@app.get("/api/jobs")
@login_required
def list_jobs():
    with JOBS_LOCK:
        jobs = sorted(
            JOBS.values(), key=lambda j: j.get("created_at", ""), reverse=True
        )
        return jsonify({
            "jobs": [{
                "job_id": j["job_id"],
                "title": j.get("title", "UNTITLED"),
                "creator": j.get("creator", "UNKNOWN"),
                "status": j.get("status"),
                "stage": j.get("stage"),
                "created_at": j.get("created_at"),
            } for j in jobs]
        })


@app.get("/api/jobs/archive")
@login_required
def list_archive_jobs():
    """List jobs persisted in the SQLite store (survives server restarts)."""
    return jsonify({
        "jobs": [{
            "job_id": j.get("job_id"),
            "title": j.get("title", "UNTITLED"),
            "creator": j.get("creator"),
            "status": j.get("status"),
            "created_at": j.get("created_at"),
            "finished_at": j.get("finished_at"),
        } for j in JOB_STORE.list()]
    })


@app.get("/api/jobs/archive/<job_id>")
@login_required
def get_archive_job(job_id: str):
    """Return a fully persisted job (dashboard + pruned result) from the store."""
    job = JOB_STORE.load(job_id)
    if job is None:
        return jsonify({"error": "ARCHIVE JOB NOT FOUND"}), 404
    return jsonify(job)


# ── Routes: analyse ──────────────────────────────────────────


@app.post("/api/analyse")
@login_required
def analyse():
    upload = request.files.get("video")
    if upload is None or upload.filename == "":
        return jsonify({"error": "NO VIDEO FILE PROVIDED"}), 400

    ext = Path(upload.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"UNSUPPORTED FORMAT '{ext}'"}), 400

    upload.seek(0, os.SEEK_END)
    size = upload.tell()
    upload.seek(0)
    if size > MAX_UPLOAD_BYTES:
        return jsonify({"error": "FILE EXCEEDS 200 MB LIMIT"}), 413

    title = (request.form.get("title") or "").strip() or Path(upload.filename).stem
    creator = (request.form.get("creator") or "").strip() or "UNKNOWN"
    duration_raw = (request.form.get("duration") or "").strip()
    duration_sec = 0
    if duration_raw:
        try:
            duration_sec = float(duration_raw)
        except ValueError:
            duration_sec = 0

    job_id = _new_job_id()
    safe_name = "".join(ch for ch in upload.filename if ch.isalnum() or ch in ".-_ ")
    video_path = UPLOAD_DIR / f"{job_id}_{safe_name or 'upload'}"
    upload.save(video_path)

    job = {
        "job_id": job_id,
        "title": title,
        "creator": creator,
        "filename": upload.filename,
        "status": "running",
        "stage": "QUEUED",
        "duration_sec": duration_sec,
        "created_at": _now_iso(),
        "finished_at": None,
        "error": None,
        "video_path": str(video_path),
        "result": None,
        "dashboard": None,
        "feed": [],
    }
    with JOBS_LOCK:
        JOBS[job_id] = job

    threading.Thread(target=_run_job, args=(job_id,), daemon=True).start()

    return jsonify({"job_id": job_id, "status": "running"})


@app.get("/api/analyse/<job_id>")
@login_required
def analyse_status(job_id: str):
    job = _job(job_id)
    if job is None:
        return jsonify({"error": "JOB NOT FOUND"}), 404
    return jsonify({
        "job_id": job_id,
        "status": job["status"],
        "stage": job.get("stage"),
        "error": job.get("error"),
        "feed": (job.get("feed") or [])[-8:],
    })


# ── Routes: dashboard ────────────────────────────────────────


def _sanitize_void_candidates(dash: dict) -> dict:
    """Downgrade candidates whose derived name means "no identification".

    `_derive_candidate_name` refuses void names, but a candidate verified
    *before* that guard existed is still sitting in the persisted dashboard
    (e.g. "NOT IDENTIFIABLE" sent to logo.dev, which matched identifiable.ca
    and reported it verified). Correcting at serve time repairs every stored job
    without re-running the paid reverse-image search. Returns a copy; the
    persisted record is left alone.
    """
    from src.openset import _is_void_name

    os_block = dash.get("open_set")
    if not isinstance(os_block, dict):
        return dash
    candidates = os_block.get("candidates")
    if not candidates:
        return dash

    repaired = 0
    fixed = []
    for cand in candidates:
        if _is_void_name(cand.get("candidate_name") or ""):
            fixed.append({**cand, "status": "candidate_void", "logo_dev_validation": None})
            repaired += 1
        else:
            fixed.append(cand)
    if not repaired:
        return dash

    app.logger.info(
        "OPEN-SET REPAIR — %d void candidate(s) downgraded at serve time", repaired
    )
    return {
        **dash,
        "open_set": {
            **os_block,
            "candidates": fixed,
            "resolved": len([c for c in fixed if c.get("status") == "candidate_verified"]),
        },
    }


@app.get("/api/pipeline/<job_id>")
@login_required
def pipeline_dashboard(job_id: str):
    job = _job(job_id)
    if job is None:
        return jsonify({"error": "JOB NOT FOUND"}), 404
    if job.get("status") == "error":
        return jsonify({"error": job.get("error") or "PIPELINE FAILED"}), 500
    if job.get("status") != "done" or job.get("dashboard") is None:
        return jsonify({"error": "JOB NOT COMPLETE", "status": job.get("status")}), 409
    return jsonify(_sanitize_void_candidates(job["dashboard"]))


@app.get("/api/insights/<job_id>")
@login_required
def insights(job_id: str):
    """Phase 2 insights from the raw result: creator profile, cross-video brand
    memory, and indirect-reference resolutions, plus ranked recommendations.

    Falls back to the SQLite store when the job is no longer in live memory
    (e.g. after a server restart), so archived/seed data remains viewable."""
    job = _job(job_id)
    result = None
    if job is not None:
        result = job.get("result")
    else:
        archived = JOB_STORE.load(job_id)
        if archived is not None:
            result = archived.get("result")
    if result is None:
        return jsonify({"error": "JOB RESULT NOT AVAILABLE"}), 409
    l3 = result.get("layer3") or {}
    l2d = result.get("layer2d") or {}
    l2c = result.get("layer2c") or {}
    return jsonify({
        "job_id": job_id,
        "recommendations": l3.get("recommendations") or [],
        "creator_profile": l2d.get("creator_profile"),
        "brand_memory": {
            "size": l2c.get("memory_size"),
            "brands": l2c.get("memory_brands") or [],
            "indirect_resolutions": l2c.get("indirect_resolutions") or [],
        },
    })


# ── Routes: scene / crop images ──────────────────────────────


@app.get("/api/scene/<job_id>/<int:frame_index>")
@login_required
def scene_image(job_id: str, frame_index: int):
    """Annotated JPEG of a sampled frame (drawn boxes, no brand class names)."""
    job = _job(job_id)
    if job is None:
        return jsonify({"error": "JOB NOT FOUND"}), 404
    if job.get("status") != "done" or job.get("result") is None:
        return jsonify({"error": "JOB NOT COMPLETE"}), 409

    result = job["result"]
    num_frames = result.get("num_frames", 0)
    video_stride = int(result.get("video_stride", 1) or 1)
    if frame_index >= num_frames:
        return jsonify({"error": "FRAME OUT OF RANGE"}), 404

    frame = _read_video_frame(job["video_path"], frame_index, video_stride)
    if frame is None:
        return jsonify({"error": "FRAME NOT FOUND"}), 404

    import cv2

    l1 = result["layer1"]
    objects = l1["scene_object_detections"][frame_index] or []
    logos = l1["logo_detections"][frame_index] or []
    annotated = frame.copy()
    for o in objects:
        bbox = o.get("bbox")
        if not bbox:
            continue
        x1, y1, x2, y2 = (int(v) for v in bbox)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
        label = o.get("class_name", "OBJECT")
        cv2.putText(
            annotated, label, (x1, max(10, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1,
        )
    for o in logos:
        bbox = o.get("bbox")
        if not bbox:
            continue
        x1, y1, x2, y2 = (int(v) for v in bbox)
        brand = o.get("brand")
        if brand:
            color, label = (0, 200, 0), brand[:24]
        else:
            # UNKNOWN-BRAND grouping (limitation #5): an unresolved logo shown as
            # a nameless box was visually noisy; label it explicitly instead.
            color, label = (200, 110, 0), "UNKNOWN BRAND"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated, label, (x1, max(10, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
        )
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    if not ok:
        return jsonify({"error": "FRAME ENCODE FAILED"}), 500
    return Response(buf.tobytes(), mimetype="image/jpeg")


@app.get("/api/crop/<crop_hash>")
def crop_image(crop_hash: str):
    """Open-set candidate crop evidence image (static/openset_crops/<hash>.png)."""
    crop_path = OPEN_SET_CROP_DIR / f"{crop_hash}.png"
    if not crop_path.is_file():
        return jsonify({"error": "CROP NOT FOUND"}), 404
    return send_from_directory(str(OPEN_SET_CROP_DIR), f"{crop_hash}.png")


@app.get("/api/video/<job_id>")
@login_required
def source_video(job_id: str):
    """Source video stream for dashboard clip links (seeking = Range requests).

    Serves the analysed video file with conditional/Range support so a <video>
    element can seek to the exact second a brand was spoken or shown. Fails
    closed (404) when the job is incomplete or the source file is gone.
    """
    job = _job(job_id)
    if job is None:
        return jsonify({"error": "JOB NOT FOUND"}), 404
    if job.get("status") != "done" or job.get("result") is None:
        return jsonify({"error": "JOB NOT COMPLETE"}), 409
    video_path = job.get("video_path") or ""
    if not video_path or not os.path.isfile(video_path):
        return jsonify({"error": "SOURCE VIDEO NOT FOUND"}), 404
    return send_file(video_path, conditional=True)


# ── Routes: outreach ─────────────────────────────────────────


def _validate_brand(brand: str) -> dict:
    """External logo.dev existence check, cached per brand (B2b safeguard)."""
    key = _normalize_brand(brand)
    if key in _BRAND_VALIDATION_CACHE:
        return _BRAND_VALIDATION_CACHE[key]
    result = LogoDevClient().validate_brand(brand)
    _BRAND_VALIDATION_CACHE[key] = result
    return result


@app.post("/api/outreach/generate")
@login_required
def outreach_generate():
    if not OUTREACH_ENABLED:
        return jsonify({"error": OUTREACH_REASON or "OUTREACH DISABLED"}), 403

    data = request.get_json(silent=True) or {}
    job_id = data.get("job_id")
    brand = (data.get("brand") or "").strip()
    if not brand:
        return jsonify({"error": "BRAND IS REQUIRED"}), 400

    job = _job(job_id) if job_id else None
    if job is None or job.get("dashboard") is None:
        return jsonify({"error": "JOB NOT FOUND"}), 404

    # External fabrication safeguard: only a logo.dev "verified" brand may be
    # presented as a collaboration opportunity (shared by both paths below).
    validation = _validate_brand(brand)
    if validation.get("status") != "verified":
        return jsonify({
            "error": "BRAND NOT EXTERNALLY VERIFIED",
            "brand_validation": validation,
        }), 400

    # Phase 3 personalized outreach: when the job retained the rich pipeline
    # result (ranked recommendations + creator profile + cross-video brand
    # memory), use the data-driven generator. It fails closed — it refuses to
    # fabricate copy for any brand with no grounded recommendation.
    if job.get("result"):
        res = job["result"]
        if (res.get("layer3") or {}).get("recommendations"):
            out = generate_personalized_outreach(
                res,
                brand,
                target=(data.get("target") or "").strip(),
                tone=(data.get("tone") or "professional").strip(),
                creator_name=(data.get("creator_name") or "").strip(),
            )
            if out.get("status") == "ok":
                out["brand_validation"] = validation
                return jsonify(out)

    dash = job["dashboard"]
    creator = dash.get("creator") or "CHANNEL"
    title = dash.get("title") or "THE CHANNEL"

    product = brand
    count = 0
    scenes = []
    category = "GENERAL"
    source_rows = dash.get("products") or dash.get("recommendations") or []
    for p in source_rows:
        if _normalize_brand(p.get("brand", "")) == _normalize_brand(brand):
            product = p.get("product") or brand
            category = p.get("category") or "GENERAL"
            raw_app = p.get("appearances")
            count = (
                p.get("appearance_count")
                or (raw_app if isinstance(raw_app, int) else len(raw_app or []))
            ) or 0
            scenes = raw_app if isinstance(raw_app, list) else []
            break

    # A draft may only be generated for a brand with real on-screen appearances.
    if count == 0:
        return jsonify({
            "error": (
                "NO REAL ON-SCREEN APPEARANCES — REFUSING TO GENERATE A DRAFT "
                "(brand attribution is not production-validated)"
            )
        }), 400

    target = (data.get("target") or "").strip()
    scene_txt = ", ".join(str(s) for s in scenes) if scenes else "SEVERAL SCENES"
    count_txt = f"{count} SCENES" if count else "SEVERAL SCENES"

    subject = f"PARTNERSHIP OPPORTUNITY — {brand} × {creator}"

    body = (
        f"TO: {target}\n"
        f"FROM: {creator}\n"
        f"CHANNEL: {title}\n"
        f"CATEGORY: {category}\n"
        f"\n"
        f"HELLO {brand} TEAM,\n"
        f"\n"
        f"I RUN {title}, AND I AM REACHING OUT BECAUSE THE ADSCENE\n"
        f"PLATFORM IDENTIFIED A NATURAL PLACEMENT FOR YOUR BRAND.\n"
        f"\n"
        f"WHILE REVIEWING MY ARCHIVES, YOUR {product} APPEARED\n"
        f"ON SCREEN ACROSS {count_txt} ({scene_txt}) — CLEARLY VISIBLE,\n"
        f"UNPROMPTED, AND IN CONTEXT WITH THE CONTENT.\n"
        f"\n"
        f"THIS IS A GENUINE OPPORTUNITY FOR A NATIVE INTEGRATION OR\n"
        f"SPONSORED SEGMENT. I AM HAPPY TO SHARE FULL VIEW COUNTS,\n"
        f"AUDIENCE DEMOGRAPHICS, AND THE COMPLETE SCENE BREAKDOWN.\n"
        f"\n"
        f"WOULD YOU BE OPEN TO A CONVERSATION NEXT WEEK?\n"
        f"\n"
        f"BEST,\n"
        f"{creator}"
    )

    return jsonify({
        "subject": subject,
        "body": body,
        "target": target,
        "brand": brand,
        "product": product,
        "brand_validation": validation,
    })


@app.post("/api/outreach/forward")
@login_required
def outreach_forward():
    if not OUTREACH_ENABLED:
        return jsonify({"error": OUTREACH_REASON or "OUTREACH DISABLED"}), 403

    data = request.get_json(silent=True) or {}
    job_id = data.get("job_id")
    brand = data.get("brand")

    job = _job(job_id) if job_id else None
    if job is None or job.get("dashboard") is None:
        return jsonify({"error": "JOB NOT FOUND"}), 404

    forwarded = job.setdefault("forwarded", [])
    stamped = {
        "brand": brand,
        "at": _now_iso(),
        "request_id": str(uuid.uuid4())[:8].upper(),
    }
    forwarded.append(stamped)
    _persist_job(job)

    return jsonify({
        "status": "forwarded",
        "target": job["dashboard"].get("creator", "CHANNEL"),
        "at": stamped["at"],
    })


if __name__ == "__main__":
    print(f"ADSCENE SERVER — http://127.0.0.1:{PORT}")
    app.run(host="127.0.0.1", port=PORT, debug=False, threaded=True)
