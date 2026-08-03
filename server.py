"""
ADSCENE API server.

Serves the static frontend and wraps the Phase 1 pipeline
(src.pipeline.Phase1Pipeline) behind a job queue so uploads
run in background threads and the UI can poll for progress.

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

from flask import Flask, jsonify, request, send_from_directory

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

# Brand → (product, category) catalog. Used to enrich detections.
BRAND_CATALOG = {
    "NESCAFÉ": ("NESCAFÉ GOLD", "BEVERAGE"),
    "NESCAFE": ("NESCAFÉ GOLD", "BEVERAGE"),
    "SAMSUNG": ("SAMSUNG GALAXY", "ELECTRONICS"),
    "APPLE": ("APPLE VISION PRO", "ELECTRONICS"),
    "COCA-COLA": ("COCA-COLA ZERO", "BEVERAGE"),
    "COKE": ("COCA-COLA", "BEVERAGE"),
    "PEPSI": ("PEPSI MAX", "BEVERAGE"),
    "RED BULL": ("RED BULL ENERGY", "BEVERAGE"),
    "NIKE": ("NIKE AIR", "APPAREL"),
    "ADIDAS": ("ADIDAS SAMBA", "APPAREL"),
    "SONY": ("SONY WH-1000XM5", "ELECTRONICS"),
    "LG": ("LG OLED", "ELECTRONICS"),
    "STARBUCKS": ("STARBUCKS CUP", "BEVERAGE"),
    "MERCADES": ("MERCEDES C-CLASS", "AUTOMOTIVE"),
    "MERCEDES": ("MERCEDES C-CLASS", "AUTOMOTIVE"),
    "BMW": ("BMW 5 SERIES", "AUTOMOTIVE"),
    "TESLA": ("TESLA MODEL 3", "AUTOMOTIVE"),
    "STANLEY": ("STANLEY MUG", "DRINKWARE"),
    "YETI": ("YETI RAMBLER", "DRINKWARE"),
    "SUPREME": ("SUPREME BOX LOGO", "APPAREL"),
    "GUCCI": ("GUCCI BAG", "APPAREL"),
}


# ── Helpers ─────────────────────────────────────────────────


def _new_job_id() -> str:
    part_a = "".join(random.choices(string.ascii_uppercase + string.digits, k=8))
    part_b = "".join(random.choices(string.ascii_uppercase + string.digits, k=5))
    return f"{part_a}-{part_b}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _normalize_brand(name: str) -> str:
    return name.strip().upper()


def _catalog_lookup(brand: str):
    return BRAND_CATALOG.get(_normalize_brand(brand))


def _job(job_id: str) -> dict:
    with JOBS_LOCK:
        return JOBS.get(job_id)


def _scene_number_map(scene_indexes):
    """Map ordinal position → SCENE 001-style numbering."""
    return {idx: n + 1 for n, idx in enumerate(sorted(scene_indexes))}


# ── Dashboard compilation ────────────────────────────────────


def _build_dashboard(result: dict, job: dict) -> dict:
    l1 = result["layer1"]
    num_frames = result.get("num_frames", 0)
    video_fps = result.get("video_fps", 0.0)
    duration_sec = (num_frames / video_fps) if video_fps else job.get("duration_sec", 0)

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
            "logos": [
                {"class_name": o["class_name"], "confidence": o["confidence"]}
                for o in logos
            ],
        })

    # ── Products (aggregated brand appearances) ─────────────
    products = {}
    for frame_idx, dets in enumerate(l1["logo_detections"]):
        if not dets:
            continue
        frame_texts = [r.get("text", "") for r in l1["ocr_results"][frame_idx]]
        for det in dets:
            brand = det["class_name"]
            key = _normalize_brand(brand)
            if key not in products:
                product, category = _catalog_lookup(brand) or (brand, "GENERAL")
                products[key] = {
                    "brand": brand,
                    "product": product,
                    "category": category,
                    "appearances": set(),
                    "confidences": [],
                    "frame_texts": [],
                }
            entry = products[key]
            entry["appearances"].add(frame_idx)
            entry["confidences"].append(det["confidence"])
            entry["frame_texts"].extend(frame_texts)

    product_list = []
    for entry in products.values():
        appearances = sorted(entry["appearances"])
        # Prefer catalog product; else most frequent OCR text seen near the logo.
        product = entry["product"]
        if _catalog_lookup(entry["brand"]) is None:
            texts = [t for t in entry["frame_texts"] if t.strip()]
            if texts:
                product = max(set(texts), key=texts.count)
        product_list.append({
            "brand": entry["brand"],
            "product": product,
            "category": entry["category"],
            "confidence": float(sum(entry["confidences"]) / len(entry["confidences"])),
            "appearances": [fmt_scene(scene_nums[i]) for i in appearances if i in scene_nums]
            or [f"FRAME {i}" for i in appearances],
            "appearance_count": len(appearances),
        })
    product_list.sort(key=lambda p: p["appearance_count"], reverse=True)

    # ── Ad opportunities ────────────────────────────────────
    ads = []
    for p in product_list:
        ads.append({
            "brand": p["brand"],
            "product": p["product"],
            "category": p["category"],
            "type": "LOGO APPEARANCE",
            "scenes": p["appearances"],
            "score": p["confidence"],
        })
    for m in l1.get("brand_mentions", []):
        ads.append({
            "brand": m.get("brand", "SPOKEN BRAND"),
            "product": m.get("brand", "SPOKEN BRAND"),
            "category": "SPEECH",
            "type": "SPEECH MENTION",
            "scenes": [],
            "score": m.get("confidence", 0.0),
        })

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

    return {
        "title": job.get("title", "UNTITLED"),
        "creator": job.get("creator", "UNKNOWN"),
        "filename": job.get("filename", ""),
        "job_id": job["job_id"],
        "duration_sec": duration_sec or job.get("duration_sec", 0),
        "num_frames": num_frames,
        "video_fps": video_fps,
        "has_audio": bool(result.get("has_audio")),
        "confidence": float(confidence),
        "is_confident": bool(l2b.get("is_confident")),
        "confidence_status": l2b.get("status", "unknown"),
        "evidence_breakdown": evidence_breakdown,
        "scenes": scenes,
        "products": product_list,
        "ads": ads,
        "transcript": l1.get("transcript", ""),
        "audio_events": l1.get("audio_events", [])[:20],
    }


def fmt_scene(n: int) -> str:
    return f"SCENE {n:03d}"


# ── Pipeline access (lazy singleton) ─────────────────────────

_pipeline = None
_pipeline_lock = threading.Lock()


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                from src.pipeline import Phase1Pipeline

                _pipeline = Phase1Pipeline(device_override="auto")
    return _pipeline


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
    safe_name = "".join(ch for ch in upload.filename if ch.isalnum() or ch in "._- ")
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


# ── Routes: outreach ─────────────────────────────────────────


@app.post("/api/outreach/generate")
def outreach_generate():
    data = request.get_json(silent=True) or {}
    job_id = data.get("job_id")
    brand = data.get("brand")
    target = (data.get("target") or "").strip()

    if not target:
        return jsonify({"error": "TARGET IS REQUIRED"}), 400

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
        if _normalize_brand(p["brand"]) == _normalize_brand(brand):
            product = p.get("product") or brand
            category = p.get("category") or "GENERAL"
            count = p.get("appearance_count", 0)
            scenes = p.get("appearances", [])
            break
    brand = next(
        (p["brand"] for p in dash.get("products", [])
         if _normalize_brand(p["brand"]) == _normalize_brand(brand)),
        brand,
    )

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
    })


@app.post("/api/outreach/forward")
def outreach_forward():
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
