"""Run the real pipeline + dashboard on a real video and dump dashboard state.

Used for remediation verification: captures num_frames, video_fps,
duration_sec, scenes (count + max SCENE n), products status, open-set
candidates, and confidence — the exact fields the escalated ADSCENE
screenshot shows, so a "before" and "after" fix can be compared on the
same real inputs.
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from server import _build_dashboard, _new_job_id  # noqa: E402


def run(video_path: str, out_json: str = ""):
    from src.pipeline import Phase1Pipeline

    print(f"=== PIPELINE RUN: {video_path} ===", flush=True)
    t0 = time.monotonic()
    pipe = Phase1Pipeline(device_override="auto")
    result = pipe.process_video(video_path, frame_rate=1.0)
    print(f"pipeline wall time: {time.monotonic() - t0:.1f}s", flush=True)
    if "error" in result:
        print("ERROR:", result["error"])
        return None, None

    job = {
        "job_id": _new_job_id(),
        "title": f"REMEDIATION RE-RUN {Path(video_path).stem}",
        "creator": "LOCAL RE-RUN HARNESS",
        "filename": Path(video_path).name,
        "duration_sec": 0,
        "video_path": video_path,
    }
    dash = _build_dashboard(result, job)

    scenes = dash["scenes"]
    max_scene = max((s["n"] for s in scenes), default=0)
    print("\n--- DASHBOARD KEY FIELDS (POST-FIX) ---")
    print(f"num_frames (sampled)   = {dash['num_frames']}")
    print(f"video_total_frames     = {dash['video_total_frames']}")
    print(f"video_fps              = {dash['video_fps']}")
    print(f"duration_sec (display) = {dash['duration_sec']:.4f}")
    print(f"scenes count           = {len(scenes)}")
    print(f"max SCENE number       = {max_scene}")
    print(f"confidence             = {dash['confidence']:.4f}  ({(dash['confidence']*100):.0f}%)")
    print(f"products count         = {len(dash['products'])}")
    print(f"products_status        = {dash['products_status']}")
    print(f"products_status_reason = {dash['products_status_reason']}")
    print("ads count =", len(dash["ads"]))
    os_ = dash.get("open_set") or {}
    print(f"open_set.available     = {os_.get('available')}")
    print(f"open_set.backend       = {os_.get('backend')}")
    print(f"open_set.reason        = {os_.get('reason')}")
    print(f"open_set.candidates    = {len(os_.get('candidates') or [])}")
    for c in os_.get("candidates") or []:
        print(f"  frame={c['frame_index']} conf={c['confidence']} "
              f"status={c['status']} name={c['candidate_name']!r} "
              f"results={len(c['search_results'])} "
              f"logodev={(c.get('logo_dev_validation') or {}).get('status')}")

    if out_json:
        dump = {
            "video": str(video_path),
            "result_keys": sorted(result.keys()),
            "logo_detections": [
                [
                    {
                        "bbox": d.get("bbox"),
                        "confidence": d.get("confidence"),
                        "class_name": d.get("class_name"),
                        "brand": d.get("brand"),
                    }
                    for d in (frame_dets or [])
                ]
                for frame_dets in result.get("layer1", {}).get("logo_detections", [])
            ],
            "dashboard": {
                "num_frames": dash["num_frames"],
                "video_total_frames": dash["video_total_frames"],
                "video_fps": dash["video_fps"],
                "duration_sec": dash["duration_sec"],
                "scenes_count": len(scenes),
                "max_scene": max_scene,
                "confidence": dash["confidence"],
                "products_count": len(dash["products"]),
                "products_status": dash["products_status"],
                "products_status_reason": dash["products_status_reason"],
                "ads_count": len(dash["ads"]),
                "open_set": os_,
            },
        }
        Path(out_json).write_text(json.dumps(dump, indent=2))
        print(f"\nwrote JSON evidence -> {out_json}")
    return result, dash


if __name__ == "__main__":
    args = sys.argv[1:]
    run(args[0], args[1] if len(args) > 1 else "")
