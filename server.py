"""
ADSCENE API server.

Serves the static frontend and wraps the Phase 1 pipeline
(src.pipeline.Phase1Pipeline) behind a job queue so uploads
run in background threads and the UI can poll for progress.

Remediation notes (see MAJOR_REMEDIATION_REPORT.md / REMEDIATION_ESCALATION_REPORT.md):

  * Brand attribution is NOT production-validated. The dashboard products table
    is empty by construction (products_status: NOT_AVAILABLE); scene cards emit
    "LOGO REGION" + real model confidence, never a detector brand class name;
    ads come only from real ASR brand mentions. _validate_dashboard_bounds()
    raises loudly on any SCENE n > num_frames instead of silently trimming.
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
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml
from flask import Flask, Response, jsonify, request, send_from_directory

from src.logodev import LogoDevClient

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

# Open-set identification config (escalation Task 3) — every tunable comes from
# config; the code fails closed naming any missing key.
_OPEN_SET_CFG = CFG.get("open_set", {}) if isinstance(CFG.get("open_set"), dict) else {}
OPEN_SET_CROP_DIR = ROOT / str(_OPEN_SET_CFG.get("crop_cache_dir", "static/openset_crops"))

# logo.dev brand-validation cache (B2b fabrication safeguard). Only status
# "verified" may ever be presented as a real brand.
_BRAND_VALIDATION_CACHE: dict = {}


# ── Helpers ─────────────────────────────────────────────────


def _new_job_id() -> str:
    part_a = "".join(random.choices(string.ascii_uppercase + string.digits, k=8))
    part_b = "".join(random.choices(string.ascii_uppercase + string.digits, k=5))
    return f"{part_a}-{part_b}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _normalize_brand(name: str) -> str:
    return name.strip().upper()


def _job(job_id: str) -> dict:
    with JOBS_LOCK:
        return JOBS.get(job_id)


def _scene_number_map(scene_indexes):
    """Map ordinal position → SCENE 001-style numbering."""
    return {idx: n + 1 for n, idx in enumerate(sorted(scene_indexes))}


def _read_video_frame(video_path: str, frame_index: int):
    """Seek to an extracted-frame index and return the RGB frame (cv2)."""
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
            "reason": "OPEN-SET IDENTIFICATION DISABLED (config open_set.enabled: false)",
            "candidates": [],
        }

    required_keys = [
        "backend",
        "min_logo_confidence",
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
            "reason": (
                "OPEN-SET IDENTIFICATION UNAVAILABLE — CONFIG MISSING KEYS: "
                + ", ".join(missing)
            ),
            "candidates": [],
        }

    try:
        from src.openset import OpenSetBrandIdentifier, create_backend

        identifier = OpenSetBrandIdentifier(
            backend=create_backend(str(_OPEN_SET_CFG["backend"])),
            min_logo_confidence=float(_OPEN_SET_CFG["min_logo_confidence"]),
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
            "reason": f"OPEN-SET IDENTIFICATION UNAVAILABLE — {exc}",
            "candidates": [],
        }

    try:
        return identifier.identify(result, video_path)
    except Exception as exc:  # noqa: BLE001
        return {
            "available": False,
            "backend": backend_name,
            "min_confidence": min_confidence,
            "reason": f"OPEN-SET IDENTIFICATION FAILED — {exc}",
            "candidates": [],
        }


# ── Dashboard compilation ────────────────────────────────────


def _build_dashboard(result: dict, job: dict) -> dict:
    l1 = result["layer1"]
    num_frames = result.get("num_frames", 0)
    video_fps = result.get("video_fps", 0.0)
    video_total_frames = result.get("video_total_frames", 0)

    # Duration from the source video's real codec frame count (fixes the unit
    # bug: sampled-frame count / source fps previously rendered 60 s as 3 s).
    duration_sec = (video_total_frames / video_fps) if video_fps else 0.0

    # ── Scenes ──────────────────────────────────────────────
    scenes = []
    scene_indexes = set()
    for frame_idx, dets in enumerate(l1["scene_object_detections"]):
        if dets:
            scene_indexes.add(frame_idx)
    scene_nums = _scene_number_map(scene_indexes)
    for frame_idx in sorted(scene_indexes):
        objects = l1["scene_object_detections"][frame_idx]
        logos = l1["logo_detections"][frame_idx]
        scenes.append({
            "n": scene_nums[frame_idx],
            "frame_index": frame_idx,
            "timestamp": frame_idx / video_fps if video_fps else 0.0,
            "objects": [
                {"class_name": o["class_name"], "confidence": o["confidence"]}
                for o in objects
            ],
            # Detector brand class labels are NOT trusted attribution — scene
            # cards emit "LOGO REGION" + real model confidence only.
            "logos": [
                {"class_name": "LOGO REGION", "confidence": o["confidence"]}
                for o in logos
            ],
        })

    # ── Products: empty by construction ─────────────────────
    # Brand attribution is not production-validated (see MAJOR_REMEDIATION_REPORT.md
    # Part B + benchmark): no code path may repopulate this table. Open-set
    # candidates are surfaced separately and never merged here.
    products_status = "NOT_AVAILABLE"
    products_status_reason = (
        "BRAND ATTRIBUTION IS NOT PRODUCTION-VALIDATED — the logo-detection path "
        "scores 0% brand accuracy on the held-out benchmark, so no brand is "
        "asserted as an on-screen appearance. See MAJOR_REMEDIATION_REPORT.md Part B."
    )
    product_list = []

    # ── Ads: real ASR evidence only ─────────────────────────
    ads = []
    for m in l1.get("brand_mentions", []):
        ads.append({
            "brand": m.get("brand", "SPOKEN BRAND"),
            "product": m.get("brand", "SPOKEN BRAND"),
            "category": "SPEECH",
            "type": "SPEECH MENTION (ASR)",
            "scenes": [],
            "score": m.get("confidence", 0.0),
        })

    # ── Layer 3 recommendations ─────────────────────────────
    recommendations = (result.get("layer3") or {}).get("recommendations", [])

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
        "ads": ads,
        "open_set": open_set,
        "recommendations": recommendations,
        "outreach_enabled": OUTREACH_ENABLED,
        "outreach_reason": OUTREACH_REASON,
        "transcript": l1.get("transcript", ""),
        "audio_events": l1.get("audio_events", [])[:20],
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


def _warmup_models(pipeline) -> None:
    try:
        pipeline.warmup()
    except Exception as exc:  # noqa: BLE001
        app.logger.warning("Model warmup failed (continuing): %s", exc)


def _run_job(job_id: str) -> None:
    job = _job(job_id)
    try:
        job["stage"] = "LOADING MODELS"
        pipeline = get_pipeline()
        job["stage"] = "EXTRACTING AUDIO & VISUAL SIGNALS"
        result = pipeline.process_video(job["video_path"], frame_rate=1.0)
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


@app.get("/api/jobs")
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


# ── Routes: analyse ──────────────────────────────────────────


@app.post("/api/analyse")
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
    }
    with JOBS_LOCK:
        JOBS[job_id] = job

    threading.Thread(target=_run_job, args=(job_id,), daemon=True).start()

    return jsonify({"job_id": job_id, "status": "running"})


@app.get("/api/analyse/<job_id>")
def analyse_status(job_id: str):
    job = _job(job_id)
    if job is None:
        return jsonify({"error": "JOB NOT FOUND"}), 404
    return jsonify({
        "job_id": job_id,
        "status": job["status"],
        "stage": job.get("stage"),
        "error": job.get("error"),
    })


# ── Routes: dashboard ────────────────────────────────────────


@app.get("/api/pipeline/<job_id>")
def pipeline_dashboard(job_id: str):
    job = _job(job_id)
    if job is None:
        return jsonify({"error": "JOB NOT FOUND"}), 404
    if job.get("status") == "error":
        return jsonify({"error": job.get("error") or "PIPELINE FAILED"}), 500
    if job.get("status") != "done" or job.get("dashboard") is None:
        return jsonify({"error": "JOB NOT COMPLETE", "status": job.get("status")}), 409
    return jsonify(job["dashboard"])


# ── Routes: scene / crop images ──────────────────────────────


@app.get("/api/scene/<job_id>/<int:frame_index>")
def scene_image(job_id: str, frame_index: int):
    """Annotated JPEG of a sampled frame (drawn boxes, no brand class names)."""
    job = _job(job_id)
    if job is None:
        return jsonify({"error": "JOB NOT FOUND"}), 404
    if job.get("status") != "done" or job.get("result") is None:
        return jsonify({"error": "JOB NOT COMPLETE"}), 409

    result = job["result"]
    num_frames = result.get("num_frames", 0)
    if frame_index >= num_frames:
        return jsonify({"error": "FRAME OUT OF RANGE"}), 404

    frame = _read_video_frame(job["video_path"], frame_index)
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
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (220, 40, 40), 2)
        cv2.putText(
            annotated, "LOGO REGION", (x1, max(10, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 40, 40), 1,
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

    dash = job["dashboard"]
    creator = dash.get("creator") or "CHANNEL"
    title = dash.get("title") or "THE CHANNEL"

    product = brand
    count = 0
    scenes = []
    category = "GENERAL"
    for p in dash.get("products", []):
        if _normalize_brand(p.get("brand", "")) == _normalize_brand(brand):
            product = p.get("product") or brand
            category = p.get("category") or "GENERAL"
            count = p.get("appearance_count", 0)
            scenes = p.get("appearances", [])
            break

    # A draft may only be generated for a brand with real on-screen appearances.
    if count == 0:
        return jsonify({
            "error": (
                "NO REAL ON-SCREEN APPEARANCES — REFUSING TO GENERATE A DRAFT "
                "(brand attribution is not production-validated)"
            )
        }), 400

    # External fabrication safeguard: only a logo.dev "verified" brand may be
    # presented as a collaboration opportunity.
    validation = _validate_brand(brand)
    if validation.get("status") != "verified":
        return jsonify({
            "error": "BRAND NOT EXTERNALLY VERIFIED",
            "brand_validation": validation,
        }), 400

    target = (data.get("target") or "").strip()
    scene_txt = ", ".join(scenes) if scenes else "SEVERAL SCENES"
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

    return jsonify({
        "status": "forwarded",
        "target": job["dashboard"].get("creator", "CHANNEL"),
        "at": stamped["at"],
    })


if __name__ == "__main__":
    print(f"ADSCENE SERVER — http://127.0.0.1:{PORT}")
    app.run(host="127.0.0.1", port=PORT, debug=False, threaded=True)
