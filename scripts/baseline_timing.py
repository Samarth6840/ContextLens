#!/usr/bin/env python3
"""
Baseline timing script — measures per-stage latency on the Samsung test video.
Records model device placement and writes outputs/evaluation/baseline_timing.json.
"""

import json
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("baseline_timing")

DEFAULT_VIDEO_PATH = "test_video.mp4"


def get_model_devices(pipeline) -> dict:
    """Check which device each loaded model is on."""
    devices = {}
    if pipeline._detector is not None:
        devices["scene_detector"] = str(getattr(pipeline._detector, "device", "unknown"))
    if pipeline._logo_detector is not None:
        devices["logo_detector"] = str(getattr(pipeline._logo_detector, "device", "unknown"))
    if pipeline._embedding_extractor is not None:
        devices["visual_embeddings"] = str(getattr(pipeline._embedding_extractor, "device", "unknown"))
    if pipeline._ocr is not None:
        devices["ocr"] = "cpu"  # PaddleOCR is always CPU
    if pipeline._stt is not None:
        devices["speech_to_text"] = str(getattr(pipeline._stt, "device", "unknown"))
    if pipeline._audio_events is not None:
        devices["audio_events"] = str(getattr(pipeline._audio_events, "device", "unknown"))
    return devices


def main():
    from src.pipeline import Phase1Pipeline

    video_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_VIDEO_PATH)
    if not video_path.exists():
        print(f"ERROR: Test video not found at {video_path}")
        sys.exit(1)

    print("=== ContextLens Baseline Timing ===")
    print(f"Video: {video_path} ({video_path.stat().st_size / 1e6:.1f} MB)")
    print()

    # Initialize pipeline (models load lazily on first process_video call)
    pipeline = Phase1Pipeline(config_path="config/config.yaml")
    print(f"Pipeline device: {pipeline.device}")
    print()

    # Warm up — trigger all lazy model loads without timing them
    print("Warming up (loading all models)...")

    # Trigger model loads by accessing properties
    _ = pipeline.detector
    _ = pipeline.logo_detector
    _ = pipeline.embedding_extractor
    _ = pipeline.ocr
    _ = pipeline.stt
    _ = pipeline.audio_events
    _ = pipeline.fusion
    _ = pipeline.audio_quality_estimator
    _ = pipeline.video_quality_estimator
    _ = pipeline.confidence_scorer

    devices = get_model_devices(pipeline)
    print(f"Model devices: {json.dumps(devices, indent=2)}")
    print()

    # Run timed inference
    print("Running full pipeline...")
    t0 = time.monotonic()
    result = pipeline.process_video(str(video_path))
    total_time = time.monotonic() - t0

    if "error" in result:
        print(f"ERROR: {result['error']}")
        sys.exit(1)

    # Print timing breakdown
    timings = result["timings"]
    print()
    print("=== Timing Breakdown ===")
    for stage, duration in timings.items():
        print(f"  {stage:25s} {duration:8.3f}s")
    print(f"  {'wall_total':25s} {total_time:8.3f}s")
    print()

    # Print model devices
    print("=== Model Devices ===")
    for name, device in devices.items():
        print(f"  {name:25s} {device}")
    print()

    # Print result summary
    print("=== Result Summary ===")
    l1 = result["layer1"]
    print(f"  Frames: {result['num_frames']}")
    print(f"  Has audio: {result['has_audio']}")
    print(f"  Scene objects detected: {sum(len(d) for d in l1['scene_object_detections'])}")
    print(f"  Logo detections: {sum(len(d) for d in l1['logo_detections'])}")
    print(f"  Embeddings: {l1['num_embeddings']}")
    print(f"  OCR results: {sum(len(r) for r in l1['ocr_results'])}")
    print(f"  Transcript length: {len(l1['transcript'])} chars")
    print(f"  Brand mentions: {len(l1['brand_mentions'])}")
    print(f"  Audio events: {len(l1['audio_events'])}")
    print()

    # Compute percentage breakdown
    layer1_total = timings.get("layer1_visual", 0)
    layer2a = timings.get("layer2a_fusion", 0)
    layer2b = timings.get("layer2b_confidence", 0)
    load_total = timings.get("load_video", 0) + timings.get("extract_audio", 0)
    detection = timings.get("detection", 0)
    logo_det = timings.get("logo_detection", 0)
    print("=== Percentage of Layer 1 Visual ===")
    if layer1_total > 0:
        print(f"  detection (YOLOv8x):    {detection/layer1_total*100:5.1f}%")
        print(f"  logo_detection (WORLD): {logo_det/layer1_total*100:5.1f}%")
        # Embeddings/OCR/STT are inside ThreadPoolExecutor — not individually timed
        # The layer1_visual total captures everything in the ThreadPool
    print()
    print("=== Percentage of Total ===")
    if total_time > 0:
        print(f"  load_video:    {load_total/total_time*100:5.1f}%")
        print(f"  detection:     {detection/total_time*100:5.1f}%")
        print(f"  logo_detection:{logo_det/total_time*100:5.1f}%")
        print(f"  layer1_visual: {layer1_total/total_time*100:5.1f}%")
        print(f"  layer2a_fusion:{layer2a/total_time*100:5.1f}%")
        print(f"  layer2b_conf:  {layer2b/total_time*100:5.1f}%")

    # Save raw JSON
    output_path = Path("outputs/evaluation/baseline_timing.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "timings": timings,
            "wall_total": total_time,
            "devices": devices,
            "summary": {
                "num_frames": result["num_frames"],
                "has_audio": result["has_audio"],
                "scene_objects": sum(len(d) for d in l1["scene_object_detections"]),
                "logo_detections": sum(len(d) for d in l1["logo_detections"]),
                "num_embeddings": l1["num_embeddings"],
                "ocr_results": sum(len(r) for r in l1["ocr_results"]),
                "transcript_length": len(l1["transcript"]),
                "brand_mentions": len(l1["brand_mentions"]),
                "audio_events": len(l1["audio_events"]),
            },
        }, f, indent=2)
    print(f"\nRaw timing saved to {output_path}")


if __name__ == "__main__":
    main()
