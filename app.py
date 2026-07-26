"""
Streamlit app for Phase 1 — Multimodal Brand/Context Understanding.
Upload a video and see the pipeline results in real time.
"""

import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import Phase1Pipeline, VideoProcessor

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ContextLens — Phase 1 Demo",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🎥 ContextLens — Phase 1")
st.markdown(
    "**Multimodal Brand/Context Understanding** — Upload a video to run "
    "Layer 1 (detection, OCR, ASR, audio events) → Layer 2a (quality-aware fusion) "
    "→ Layer 2b (evidence-based confidence)."
)

# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.header("⚙️ Settings")

device_choice = st.sidebar.radio(
    "Device",
    options=["auto", "cpu", "cuda"],
    index=0,
    help="'auto' uses GPU if available, else CPU.",
)

frame_rate = st.sidebar.slider(
    "Frame extraction rate (fps)",
    min_value=0.5,
    max_value=5.0,
    value=1.0,
    step=0.5,
    help="How many frames per second to extract from the video.",
)

show_raw = st.sidebar.checkbox(
    "Show raw JSON output",
    value=False,
    help="Display the full pipeline output dict.",
)

# ── Load pipeline (cached) ───────────────────────────────────────────────────

@st.cache_resource
def load_pipeline(device: str):
    """Load the Phase 1 pipeline (cached across reruns)."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    st.sidebar.info(f"Pipeline device: {device}")
    return Phase1Pipeline(device_override=device)


# ── Upload ───────────────────────────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "Choose a video file",
    type=["mp4", "mov", "avi", "mkv", "webm"],
    help="Upload a short video (a few seconds to a few minutes).",
)

if uploaded_file is not None:
    # Save uploaded file to a temp location
    with tempfile.NamedTemporaryFile(
        suffix=Path(uploaded_file.name).suffix, delete=False
    ) as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name

    st.video(video_path)

    # ── Run pipeline ─────────────────────────────────────────────────────────

    if st.button("🚀 Run Pipeline", type="primary"):
        with st.spinner("Running Phase 1 pipeline... This may take a while."):
            try:
                pipeline = load_pipeline(device_choice)
                result = pipeline.process_video(video_path)
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.stop()

        # ── Display results ──────────────────────────────────────────────────

        if "error" in result:
            st.error(result["error"])
            st.stop()

        st.success("Pipeline completed successfully!")

        # Layout: two columns
        col1, col2 = st.columns(2)

        # ── Column 1: Layer 1 results ────────────────────────────────────────

        with col1:
            st.subheader("📹 Layer 1 — Multimodal Understanding")

            # Video info
            st.markdown(
                f"**Video:** {uploaded_file.name}  "
                f"| **Frames:** {result['num_frames']}  "
                f"| **FPS:** {result['video_fps']:.1f}  "
                f"| **Audio:** {'✅' if result['has_audio'] else '❌'}"
            )

            # Detections
            l1 = result["layer1"]
            total_detections = sum(
                len(dets) for dets in l1["detections"]
            )
            st.metric("Brand detections (total)", total_detections)

            if total_detections > 0:
                st.markdown("**Detected brands per frame:**")
                for i, dets in enumerate(l1["detections"]):
                    if dets:
                        labels = ", ".join(
                            f"{d['class_name']} ({d['confidence']:.2f})"
                            for d in dets
                        )
                        st.markdown(f"  • Frame {i}: {labels}")

            # OCR
            total_ocr = sum(len(ocr) for ocr in l1["ocr_results"])
            st.metric("OCR texts found", total_ocr)

            if total_ocr > 0:
                st.markdown("**OCR results (first 5 frames):**")
                for i, ocrs in enumerate(l1["ocr_results"][:5]):
                    if ocrs:
                        texts = ", ".join(
                            f"'{r['text']}' ({r['confidence']:.2f})"
                            for r in ocrs
                        )
                        st.markdown(f"  • Frame {i}: {texts}")

            # Transcript
            if l1["transcript"]:
                st.markdown("**Transcript:**")
                st.text(l1["transcript"][:500])
            else:
                st.info("No transcript (no audio or ASR not run).")

            # Brand mentions
            if l1["brand_mentions"]:
                st.markdown("**Brand mentions in speech:**")
                for m in l1["brand_mentions"]:
                    st.markdown(f"  • **{m['brand']}** — `...{m['text_snippet']}...`")

            # Audio events
            if l1["audio_events"]:
                st.markdown("**Audio events:**")
                for ev in l1["audio_events"][:10]:
                    st.markdown(
                        f"  • {ev['event']} ({ev['confidence']:.2f})"
                    )

        # ── Column 2: Layer 2 results ────────────────────────────────────────

        with col2:
            st.subheader("🔬 Layer 2 — Brand & Context Intelligence")

            # Quality estimates
            l2a = result["layer2a"]
            st.markdown("**Modality Quality Estimates:**")

            aq = l2a["audio_quality"]
            vq = l2a["video_quality"]

            # Audio quality
            st.markdown("**Audio:**")
            st.markdown(
                f"  • SNR: {aq.get('snr_db', 0):.1f} dB  "
                f"| VAD: {aq.get('vad_confidence', 0):.2f}  "
                f"| Quality: {aq.get('quality_score', 0):.2f}"
            )

            # Video quality
            st.markdown("**Video:**")
            st.markdown(
                f"  • Blur: {vq.get('mean_blur_score', 0):.1f}  "
                f"| Exposure: {vq.get('mean_pixel', 0):.0f}  "
                f"| Stability: {vq.get('detection_stability', 0):.2f}  "
                f"| Quality: {vq.get('quality_score', 0):.2f}"
            )

            # Fusion weights
            st.markdown("**Fusion Weights (Layer 2a):**")
            aw = l2a["fusion_audio_weight"]
            vw = l2a["fusion_video_weight"]
            st.markdown(f"  • Audio weight: {aw:.2f}")
            st.markdown(f"  • Video weight: {vw:.2f}")

            # Visual bar
            st.progress(
                int(aw * 100),
                text=f"Audio weight: {aw:.0%}",
            )
            st.progress(
                int(vw * 100),
                text=f"Video weight: {vw:.0%}",
            )

            # Confidence
            l2b = result["layer2b"]
            st.markdown("---")
            st.markdown("**Evidence-Based Confidence (Layer 2b):**")

            conf = l2b["confidence"]
            status = l2b["status"]
            is_confident = l2b["is_confident"]

            # Big confidence display
            if is_confident:
                st.metric(
                    "Confidence",
                    f"{conf:.1%}",
                    delta="Confident ✅",
                    delta_color="normal",
                )
            else:
                st.metric(
                    "Confidence",
                    f"{conf:.1%}",
                    delta="No confident evidence ⚠️",
                    delta_color="inverse",
                )

            # Evidence breakdown
            st.markdown("**Evidence breakdown:**")
            breakdown = l2b["evidence_breakdown"]
            for ev_type, ev_data in breakdown.items():
                contrib = ev_data["contribution"]
                st.markdown(
                    f"  • **{ev_type}**: "
                    f"strength={ev_data['strength']:.2f}, "
                    f"weight={ev_data['modulated_weight']:.2f}, "
                    f"contribution={contrib:.2f}"
                )

        # ── Raw JSON ─────────────────────────────────────────────────────────

        if show_raw:
            st.subheader("📄 Raw Pipeline Output")
            st.json(result)

        # ── Cleanup ──────────────────────────────────────────────────────────

        os.unlink(video_path)

else:
    st.info("👆 Upload a video file to get started.")

    # Show example of what the pipeline does
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown(
        """
        1. **Layer 1** extracts frames, runs YOLO logo detection, DINOv2 visual
           embeddings, PaddleOCR, Whisper ASR, and BEATs audio events.
        2. **Layer 2a** estimates per-modality quality (SNR, blur, exposure,
           detection stability), then fuses audio + video embeddings via a
           learned gating network and cross-attention transformer.
        3. **Layer 2b** decomposes evidence from all modalities into an
           explainable confidence score with calibration.
        """
    )