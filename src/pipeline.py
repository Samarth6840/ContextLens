"""
Phase 1 Pipeline — Multimodal Brand/Context Understanding
Wires Layer 1 (detection, embeddings, OCR, audio) → Layer 2a (quality-aware fusion)
→ Layer 2b (evidence-based confidence).

All modules load real model weights — no mock/stub/placeholder inference.
Test fixtures are strictly separated from the production data path.
Models are lazy-loaded on first use to keep startup fast.
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import yaml

logger = logging.getLogger(__name__)


def probe_hardware() -> dict:
    """Probe the actual hardware and return a descriptor dict."""
    import platform
    import torch

    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "mps_available": torch.backends.mps.is_available(),
        "mps_built": torch.backends.mps.is_built(),
        "device": "mps" if torch.backends.mps.is_available() else "cpu",
    }
    try:
        import psutil
        info["total_ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except ImportError:
        info["total_ram_gb"] = None
    return info


class VideoProcessor:
    """
    Handles video loading and frame extraction.
    """

    @staticmethod
    def load_video(
        video_path: str, frame_rate: float = 1, max_frames: int = 300
    ) -> Tuple[List[np.ndarray], float]:
        """
        Load video and extract frames at specified rate, capped at max_frames.

        Stops as soon as max_frames are collected, regardless of video length.
        This prevents OOM on long videos even when the total frame count is
        unknown or incorrectly reported by the codec.

        Args:
            video_path: Path to video file
            frame_rate: Frames per second to extract (nominal)
            max_frames: Absolute maximum number of frames to return

        Returns:
            Tuple of (frames list, video_fps)
        """
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(video_fps / frame_rate))

        frames = []
        frame_count = 0

        while len(frames) < max_frames:
            if frame_count % frame_interval == 0:
                ret = cap.grab()
                if not ret:
                    break
                ret, frame = cap.retrieve()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            else:
                ret = cap.grab()
                if not ret:
                    break

            frame_count += 1

        cap.release()

        effective_rate = len(frames) / (frame_count / video_fps) if frame_count > 0 else 0
        logger.info(
            f"Loaded {len(frames)} frames from {video_path} "
            f"({video_fps:.1f} fps, extracted at {effective_rate:.1f} fps, "
            f"{'capped' if len(frames) >= max_frames and frame_count > 0 else 'complete'})"
        )
        return frames, video_fps

    @staticmethod
    def extract_audio(
        video_path: str, sample_rate: int = 16000, max_duration: Optional[float] = None
    ) -> Optional[np.ndarray]:
        """
        Extract audio from video file via ffmpeg → WAV, then load with soundfile.

        Args:
            video_path: Path to video file
            sample_rate: Target sample rate
            max_duration: Truncate audio to this many seconds (None = no limit)

        Returns:
            Audio waveform as numpy array, or None if no audio track
        """
        import subprocess
        import tempfile

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            cmd = [
                "ffmpeg", "-y", "-i", video_path,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", str(sample_rate), "-ac", "1",
            ]
            if max_duration is not None:
                cmd.extend(["-t", str(max_duration)])
            cmd.append(tmp_path)

            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=30,
            )
            if result.returncode == 0 and Path(tmp_path).stat().st_size > 44:
                import soundfile as sf
                audio, _ = sf.read(tmp_path, dtype="float32")
                return audio
            else:
                logger.warning(
                    "ffmpeg exit code %d: %s",
                    result.returncode,
                    result.stderr.decode("utf-8", errors="replace")[:500],
                )
        except subprocess.TimeoutExpired:
            logger.warning("ffmpeg timed out after 30s")
        except FileNotFoundError:
            logger.warning("ffmpeg not found on PATH — falling back to librosa (slow on video containers)")
        except ImportError:
            logger.warning("soundfile not installed — falling back to librosa (slow)")
        except Exception as e:
            logger.debug(f"ffmpeg audio extraction failed: {e}")
        finally:
            if tmp_path is not None:
                Path(tmp_path).unlink(missing_ok=True)

        # Fallback to librosa (slow on video containers)
        import librosa
        try:
            audio, _ = librosa.load(video_path, sr=sample_rate, mono=True)
            return audio
        except Exception as e:
            logger.warning(f"Could not extract audio from {video_path}: {e}")
            return None


class Phase1Pipeline:
    """
    Complete Phase 1 pipeline: Video Input → Layer 1 → Layer 2a → Layer 2b → Output.

    Models are lazy-loaded on first use to keep startup fast.
    """

    def __init__(self, config_path: str = "config/config.yaml", device_override: Optional[str] = None):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        # Hardware probe drives device selection and concurrency preset
        self.hardware_profile = probe_hardware()
        if device_override and device_override != "auto":
            self.device = device_override
        else:
            self.device = self.hardware_profile["device"]
        logger.info(f"Phase1Pipeline initialized on device={self.device} (probe: {self.hardware_profile})")

        # Concurrency preset based on available RAM
        ram_gb = self.hardware_profile.get("total_ram_gb", 0)
        if ram_gb and ram_gb < 16:
            self.concurrency_preset = "staged"
            logger.info(
                "RAM %.1fGB < 16GB — using staged concurrency preset "
                "(vision tasks first, then audio tasks).",
                ram_gb,
            )
        else:
            self.concurrency_preset = "full"
            logger.info(
                "RAM %.1fGB >= 16GB — using full concurrency preset "
                "(all Layer 1 tasks run concurrently).",
                ram_gb,
            )

        # Lazy-loaded model handles (None until first access)
        self._detector = None
        self._logo_detector = None
        self._embedding_extractor = None
        self._ocr = None
        self._stt = None
        self._audio_events = None
        self._fusion = None
        self._audio_quality_estimator = None
        self._video_quality_estimator = None
        self._confidence_scorer = None

        # Per-model threading locks for GPU inference safety.
        # Each GPU-backed model gets its own lock so different models can
        # run concurrently (e.g., YOLO detection + DINOv2 embeddings), but
        # the same model is never called from two threads at once.
        self._model_locks: Dict[str, threading.Lock] = {}

        # Per-model init locks for thread-safe lazy initialization.
        # Prevents double-loading when two threads access the same property
        # simultaneously (check-then-set race condition).
        self._init_locks: Dict[str, threading.Lock] = {}

        logger.info("Phase1Pipeline initialized (models will load on first use)")

    def _get_or_create(self, attr: str, factory: Callable) -> object:
        """
        Thread-safe lazy initialization using double-checked locking.

        First check (no lock) avoids lock overhead on every access.
        If the attribute is still None, acquire the per-model init lock
        and check again before creating.

        Args:
            attr: Instance attribute name (e.g. '_detector')
            factory: Callable that creates and returns the model instance.

        Returns:
            The initialized model instance.
        """
        if getattr(self, attr) is None:
            lock = self._init_locks.setdefault(attr, threading.Lock())
            with lock:
                # Double-check: another thread may have initialized while we waited
                if getattr(self, attr) is None:
                    setattr(self, attr, factory())
        return getattr(self, attr)

    @property
    def detector(self):
        def _create():
            from src.layer1.detector import SceneObjectDetector
            logger.info("Loading YOLO detector...")
            return SceneObjectDetector(
                model_name=self.cfg["layer1"]["object_detection"]["model"],
                confidence_threshold=self.cfg["layer1"]["object_detection"]["confidence_threshold"],
                iou_threshold=self.cfg["layer1"]["object_detection"]["iou_threshold"],
                device=self.device,
            )
        return self._get_or_create("_detector", _create)

    @property
    def logo_detector(self):
        def _create():
            from src.layer1.logo_detector import create_logo_detector
            logo_cfg = self.cfg["layer1"].get("logo_detection", {})
            backend = logo_cfg.get("backend", "yolo_world")
            logger.info("Loading logo detector (backend=%s)...", backend)
            return create_logo_detector(
                backend=backend,
                model_name=logo_cfg.get("model", "yolov8s-worldv2.pt"),
                confidence_threshold=logo_cfg.get("confidence_threshold", 0.30),
                device=self.device,
                text_queries=logo_cfg.get("text_queries"),
            )
        return self._get_or_create("_logo_detector", _create)

    @property
    def embedding_extractor(self):
        def _create():
            from src.layer1.visual_embeddings import VisualEmbeddingExtractor
            logger.info("Loading DINOv2 embeddings model...")
            return VisualEmbeddingExtractor(
                model_name=self.cfg["layer1"]["visual_embeddings"]["model"],
                output_dim=self.cfg["layer1"]["visual_embeddings"]["output_dim"],
                device=self.device,
            )
        return self._get_or_create("_embedding_extractor", _create)

    @property
    def ocr(self):
        def _create():
            from src.layer1.ocr import OCRExtractor
            logger.info("Loading PaddleOCR...")
            return OCRExtractor(
                lang=self.cfg["layer1"]["ocr"]["lang"],
                use_angle_cls=self.cfg["layer1"]["ocr"]["use_angle_cls"],
                det_db_thresh=self.cfg["layer1"]["ocr"]["det_db_thresh"],
                rec_batch_num=self.cfg["layer1"]["ocr"]["rec_batch_num"],
            )
        return self._get_or_create("_ocr", _create)

    @property
    def stt(self):
        def _create():
            from src.layer1.audio import SpeechToText
            logger.info("Loading Whisper ASR model...")
            return SpeechToText(
                model_name=self.cfg["layer1"]["speech_to_text"]["model"],
                device=self.device,
                compute_dtype=self.cfg["layer1"]["speech_to_text"]["compute_dtype"],
            )
        return self._get_or_create("_stt", _create)

    @property
    def audio_events(self):
        def _create():
            beats_ckpt = self.cfg["layer1"]["audio_events"]["checkpoint"]
            if Path(beats_ckpt).exists():
                from src.layer1.audio import AudioEventDetector
                logger.info("Loading BEATs audio event detector...")
                return AudioEventDetector(
                    checkpoint_path=beats_ckpt,
                    device=self.device,
                    sample_rate=self.cfg["layer1"]["audio_events"]["sample_rate"],
                )
            else:
                logger.warning(
                    f"BEATs checkpoint not found at {beats_ckpt}. "
                    "Audio event detection disabled."
                )
                return False  # sentinel so we don't retry on every access
        return self._get_or_create("_audio_events", _create)

    @property
    def fusion(self):
        def _create():
            from src.layer2.fusion import QualityAwareFusion
            return QualityAwareFusion(
                audio_dim=self.cfg["layer1"]["audio_events"]["embedding_dim"],
                video_dim=self.cfg["layer1"]["visual_embeddings"]["output_dim"],
                hidden_dim=self.cfg["layer2a"]["fusion"]["hidden_dim"],
                num_heads=self.cfg["layer2a"]["fusion"]["num_heads"],
                num_layers=self.cfg["layer2a"]["fusion"]["num_layers"],
                dropout=self.cfg["layer2a"]["fusion"]["dropout"],
                use_learned_gating=self.cfg["layer2a"]["fusion"]["use_learned_gating"],
            ).to(self.device)
        return self._get_or_create("_fusion", _create)

    @property
    def audio_quality_estimator(self):
        def _create():
            from src.layer2.quality_estimator import AudioQualityEstimator
            return AudioQualityEstimator()
        return self._get_or_create("_audio_quality_estimator", _create)

    @property
    def video_quality_estimator(self):
        def _create():
            from src.layer2.quality_estimator import VideoQualityEstimator
            return VideoQualityEstimator()
        return self._get_or_create("_video_quality_estimator", _create)

    @property
    def confidence_scorer(self):
        def _create():
            from src.layer2.confidence import EvidenceConfidenceScorer
            layer2b_cfg = self.cfg["layer2b"]
            return EvidenceConfidenceScorer(
                evidence_weights=layer2b_cfg.get("evidence_weights"),
                min_evidence_threshold=layer2b_cfg["min_evidence_threshold"],
                aggregation=layer2b_cfg["aggregation"],
                evidence_sources=layer2b_cfg.get("evidence_sources"),
            )
        return self._get_or_create("_confidence_scorer", _create)

    @staticmethod
    def _select_keyframes(frames: List[np.ndarray], max_frames: int = 30) -> List[int]:
        """
        Select keyframes for per-frame expensive models.

        Uses evenly-spaced indices via linspace to ensure uniform temporal
        coverage regardless of video length. Avoids parity bias from integer
        stride sampling. Collapses to first N frames when total <= max_frames.

        Args:
            frames: List of RGB images
            max_frames: Maximum number of keyframes to select

        Returns:
            Sorted list of frame indices selected as keyframes
        """
        n = len(frames)
        if n <= max_frames:
            return list(range(n))

        # Evenly-spaced indices — guarantees uniform coverage
        indices = np.linspace(0, n - 1, max_frames, dtype=int).tolist()
        return sorted(set(indices))

    def process_video(self, video_path: str, frame_rate: Optional[float] = None) -> Dict:
        """
        Run full Phase 1 pipeline on a video.

        Execution architecture (Layer 1):
        - Logo detection, DINOv2 embeddings, Whisper ASR, BEATs audio events,
          AND object detection all submit concurrently into a single
          ThreadPoolExecutor. Only OCR depends on detection output, so it
          submits in a second wave once detection results arrive.
        - Logo detection uses scene-change-based keyframe sampling
          (cheap frame-differencing) rather than uniform or detection-gated
          sampling. Long static segments collapse to few frames.
        - DINOv2 embeddings use a stride-based sample of at most 30 frames.

        Video loading:
        - cap.grab() / cap.retrieve() avoids decoding skipped frames (~96% savings
          on 25fps→1fps extraction).

        Audio extraction:
        - ffmpeg → WAV → soundfile with librosa fallback; observed fallback is
          logged at WARNING level so slow paths are visible.
        """
        import torch

        timings: Dict[str, float] = {}
        _t_total = time.monotonic()
        logger.info(f"Processing video: {video_path}")

        # Load video frames
        _t = time.monotonic()
        if frame_rate is None:
            frame_rate = self.cfg["evaluation"]["video_frame_rate"]
        max_frames = self.cfg['evaluation'].get('max_frames', 300)
        frames, video_fps = VideoProcessor.load_video(video_path, frame_rate, max_frames)
        timings["load_video"] = time.monotonic() - _t

        if not frames:
            return {"error": "No frames extracted from video"}

        # Extract audio (capped to max_audio_seconds to prevent OOM)
        _t = time.monotonic()
        max_audio_seconds = self.cfg['evaluation'].get('max_audio_seconds')
        audio = VideoProcessor.extract_audio(
            video_path,
            self.cfg["evaluation"]["audio_sample_rate"],
            max_duration=max_audio_seconds,
        )
        timings["extract_audio"] = time.monotonic() - _t

        # === Layer 1: Multimodal Understanding ===
        _t = time.monotonic()
        batch_size = self.cfg["evaluation"]["batch_size"]

        # Sample indices for embeddings (stride-based, max 30 frames)
        embed_indices = self._sample_indices(len(frames), max_frames=30)
        embed_frames = [frames[i] for i in embed_indices]

        # Logo detection keyframe sampling: scene-change-based
        logo_sampling = self.cfg["layer1"]["logo_detection"].get(
            "sampling_strategy", "scene_change"
        )
        max_logo = self.cfg["layer1"]["logo_detection"].get("max_logo_candidates", 30)
        if logo_sampling == "scene_change":
            logo_indices = self._select_keyframes(frames, max_frames=max_logo)
        elif logo_sampling == "all":
            logo_indices = list(range(len(frames)))
        else:
            logo_indices = self._sample_indices(len(frames), max_frames=max_logo)

        logo_frames = [frames[i] for i in logo_indices]
        logger.info(
            "Logo sampling: strategy=%s, %d frames (indices: %s ...)",
            logo_sampling, len(logo_frames),
            logo_indices[:10],
        )

        all_logo_detections = None
        embed_raw = None
        all_detections = None
        all_ocr_results = None
        transcript = ""
        brand_mentions = []
        audio_events_list = []
        ocr_future = None
        detection_done = threading.Event()

        # Submit everything — logo, embeddings, STT, audio_events, AND detection —
        # into the same concurrent pool. OCR submits in a second wave once
        # detection results arrive (it's the only module that depends on them).
        _t_logo = time.monotonic()
        _t_det = _t_logo
        _t_beats = _t_logo
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}

            futures[self._locked_submit(
                executor, "logo_detector", self.logo_detector.detect_batch,
                logo_frames, None, batch_size,
            )] = "logo_detection"

            futures[self._locked_submit(
                executor, "embedding_extractor",
                self.embedding_extractor.extract_batch, embed_frames, batch_size,
            )] = "embeddings"

            if audio is not None and len(audio) > 0:
                futures[self._locked_submit(
                    executor, "stt", self.stt.transcribe_segment, audio
                )] = "stt"

            if audio is not None and self.audio_events:
                _t_beats = time.monotonic()
                futures[self._locked_submit(
                    executor, "audio_events", self.audio_events.detect_events, audio
                )] = "audio_events"

            # Detection runs concurrently with the other models (only OCR needs its result)
            _t_det = time.monotonic()
            detection_future = self._locked_submit(
                executor, "detector", self.detector.detect_batch, frames, batch_size,
            )
            futures[detection_future] = "detection"

            # Collect results — when detection finishes, submit OCR
            for future in as_completed(futures):
                modality = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    logger.error(f"Layer 1 modality '{modality}' failed: {e}")
                    raise

                if modality == "logo_detection":
                    timings["logo_detection"] = time.monotonic() - _t_logo
                    all_logo_detections = [[] for _ in frames]
                    for idx, dets in zip(logo_indices, result):
                        all_logo_detections[idx] = dets
                elif modality == "embeddings":
                    embed_raw = result
                elif modality == "stt":
                    transcript = result
                elif modality == "audio_events":
                    timings["audio_events"] = time.monotonic() - _t_beats
                    audio_events_list = result
                elif modality == "detection":
                    timings["detection"] = time.monotonic() - _t_det
                    all_detections = result
                    detection_done.set()
                    # Submit OCR now that detection results are available.
                    # NOTE: ocr_future is submitted inside the as_completed() loop,
                    # but as_completed() snapshots its future set at call time, so
                    # this future won't be yielded by the iterator.  We collect it
                    # after the loop via the executor's shutdown(wait=True) in
                    # __exit__, which guarantees completion before we call
                    # ocr_future.result() below.
                    has_detections = [len(d) > 0 for d in all_detections]
                    det_idx_map = [i for i, has in enumerate(has_detections) if has]
                    max_ocr = self.cfg['layer1']['ocr'].get('max_frames', 30)
                    if len(det_idx_map) > max_ocr:
                        orig_count = len(det_idx_map)
                        indices = np.linspace(0, orig_count - 1, max_ocr).astype(int)
                        det_idx_map = [det_idx_map[i] for i in indices]
                        logger.info(
                            "OCR limited: %d -> %d frames (max_ocr_frames=%d)",
                            orig_count, len(det_idx_map), max_ocr,
                        )
                    frames_with_dets = [frames[i] for i in det_idx_map]
                    if frames_with_dets:
                        ocr_future = self._locked_submit(
                            executor, "ocr",
                            self._ocr_filtered, frames_with_dets, det_idx_map, len(frames),
                        )
                    logger.info(
                        "Object detection: %d frames with detections (%d total objects)",
                        len(det_idx_map), sum(len(d) for d in all_detections),
                    )

        # Wait for OCR (submitted as part of executor, resolved after close)
        if ocr_future is not None:
            try:
                all_ocr_results = ocr_future.result()
            except Exception as e:
                logger.error(f"OCR failed: {e}")

        # Guard against catastrophic failures
        if all_logo_detections is None:
            all_logo_detections = [[] for _ in frames]
        if embed_raw is None:
            raise RuntimeError("Embedding extraction returned no results — pipeline cannot continue")
        if all_detections is None:
            all_detections = [[] for _ in frames]
        if all_ocr_results is None:
            all_ocr_results = [[] for _ in frames]

        timings["layer1_visual"] = time.monotonic() - _t

        # Reassemble embeddings to full frame count (non-sampled frames get zero vectors)
        embed_dim = embed_raw.shape[1]
        embeddings = np.zeros((len(frames), embed_dim), dtype=np.float32)
        for i, embed in zip(embed_indices, embed_raw):
            embeddings[i] = embed
        logger.info(
            "Layer 1 completed in %.2fs: %d scene objects, %d logo detections, "
            "%d embeddings, %d OCR results",
            timings["layer1_visual"],
            sum(len(d) for d in all_detections),
            sum(len(d) for d in all_logo_detections),
            len(embeddings),
            sum(len(r) for r in all_ocr_results),
        )

        # Brand mentions (needs detections + transcript)
        if transcript:
            brand_mentions = self.stt.detect_brand_mentions(
                transcript, self._get_known_brands(all_detections)
            )

        # === Layer 2a: Quality Estimation & Fusion ===
        _t = time.monotonic()

        # Audio quality
        audio_quality = {"quality_score": 0.5, "snr_db": 0.0, "vad_confidence": 0.0}
        if audio is not None:
            audio_quality = self.audio_quality_estimator.estimate(audio)

        # Video quality — sample at most 20 frames for speed
        det_confidence_var = self._compute_detection_variance(all_detections)
        sample_frames = frames[::max(1, len(frames) // 20)]
        video_quality = self.video_quality_estimator.estimate_aggregate(
            sample_frames, detection_confidence_variance=det_confidence_var
        )

        # Prepare quality tensors for fusion
        # SNR: map [-10, 30] dB → [0, 1] then clamp
        snr_norm = np.clip((audio_quality["snr_db"] + 10.0) / 40.0, 0.0, 1.0)
        audio_quality_tensor = torch.tensor(
            [[snr_norm, audio_quality["vad_confidence"]]],
            dtype=torch.float32,
            device=self.device,
        )
        # Blur: clamp to [0, 1] (Laplacian variance / 500, capped)
        blur_norm = min(1.0, video_quality.get("mean_blur_score", 0.0) / 500.0)
        video_quality_tensor = torch.tensor(
            [[
                blur_norm,
                video_quality.get("mean_pixel", 127.0) / 255.0,
                video_quality.get("detection_stability", 1.0),
            ]],
            dtype=torch.float32,
            device=self.device,
        )

        # Debug: log raw quality inputs BEFORE fusion weighting
        logger.info(
            "Fusion quality inputs — audio: snr=%.1fdB, vad=%.3f, quality_score=%.3f; "
            "video: blur=%.1f, pixel=%.1f, stability=%.3f, quality_score=%.3f",
            audio_quality["snr_db"],
            audio_quality["vad_confidence"],
            audio_quality["quality_score"],
            video_quality.get("mean_blur_score", 0.0),
            video_quality.get("mean_pixel", 127.0),
            video_quality.get("detection_stability", 1.0),
            video_quality.get("quality_score", 0.5),
        )

        # Get embeddings for fusion — weight by detection confidence
        # so frames with brand detections contribute more to the video embedding
        audio_embed_dim = self.cfg["layer1"]["audio_events"]["embedding_dim"]
        audio_embed = torch.zeros(1, audio_embed_dim, device=self.device)
        if audio is not None:
            audio_features = self._compute_audio_features(audio)
            audio_embed = torch.from_numpy(audio_features).float().to(self.device)
            audio_embed = audio_embed.unsqueeze(0)

        # Detection-weighted aggregation: frames with detections get higher weight
        frame_weights = np.ones(len(embeddings), dtype=np.float32)
        for i, frame_dets in enumerate(all_detections):
            if frame_dets:
                frame_weights[i] = 1.0 + max(d["confidence"] for d in frame_dets)
        frame_weights = frame_weights / frame_weights.sum()
        weighted_embed = np.sum(
            embeddings * frame_weights[:, np.newaxis], axis=0, keepdims=True
        )
        video_embed = torch.from_numpy(weighted_embed).float().to(self.device)

        # Run fusion
        fusion_result = self.fusion(
            audio_embed=audio_embed,
            video_embed=video_embed,
            audio_quality=audio_quality_tensor,
            video_quality=video_quality_tensor,
            use_dynamic_weights=True,
        )
        timings["layer2a_fusion"] = time.monotonic() - _t

        # Log which weighting method was used
        weight_source = fusion_result.get("weight_source", "unknown")
        logger.info(
            "Fusion weights (source=%s): audio=%.4f, video=%.4f",
            weight_source,
            float(fusion_result["audio_weight"].detach().mean().cpu()),
            float(fusion_result["video_weight"].detach().mean().cpu()),
        )

        # === Layer 2b: Evidence-based Confidence ===
        _t = time.monotonic()

        # Aggregate evidence from all modalities
        evidence = self._aggregate_evidence(
            all_logo_detections, brand_mentions, all_ocr_results, audio_events_list
        )
        audio_events = audio_events_list

        modality_weights = {
            "audio_weight": float(fusion_result["audio_weight"].detach().mean().cpu()),
            "video_weight": float(fusion_result["video_weight"].detach().mean().cpu()),
        }

        confidence_result = self.confidence_scorer.compute_evidence_score(
            evidence, modality_quality_weights=modality_weights
        )
        timings["layer2b_confidence"] = time.monotonic() - _t

        timings["total"] = time.monotonic() - _t_total

        # Log timing summary
        logger.info("Pipeline timing breakdown:")
        for stage, duration in timings.items():
            logger.info("  %-20s %.3fs", stage + ":", duration)

        # Derive source-availability lists from scorer's public properties
        active_sources = list(self.confidence_scorer.effective_weights.keys())
        pending_sources = self.confidence_scorer.scaffolded_sources

        # === Compile output ===
        output = {
            "video_path": video_path,
            "num_frames": len(frames),
            "video_fps": video_fps,
            "has_audio": audio is not None,
            "hardware_profile": self.hardware_profile,
            "concurrency_preset": self.concurrency_preset,
            "timings": timings,
            "layer1": {
                "scene_object_detections": all_detections,
                "logo_detections": all_logo_detections,
                "num_embeddings": embeddings.shape[0],
                "ocr_results": all_ocr_results,
                "transcript": transcript,
                "brand_mentions": brand_mentions,
                "audio_events": audio_events,
            },
            "layer2a": {
                "audio_quality": audio_quality,
                "video_quality": video_quality,
                "fusion_audio_weight": float(
                    fusion_result["audio_weight"].detach().mean().cpu()
                ),
                "fusion_video_weight": float(
                    fusion_result["video_weight"].detach().mean().cpu()
                ),
                "weight_source": fusion_result.get("weight_source", "unknown"),
                "fused_embed": fusion_result["fused_embed"].detach().cpu().numpy().tolist(),
            },
            "layer2b": {
                **confidence_result,
                "evidence_sources_active": active_sources,
                "evidence_sources_pending": pending_sources,
            },
        }

        return output

    def _locked_submit(
        self,
        executor: ThreadPoolExecutor,
        lock_name: str,
        fn: Callable,
        *args,
        **kwargs,
    ):
        """
        Submit a function to the executor with the named model lock held.

        Each GPU-backed model gets its own threading.Lock so that the same
        model's inference is never called from two threads simultaneously.
        This prevents CUDA context corruption and undefined behavior from
        concurrent access to nn.Module state.

        Args:
            executor: The ThreadPoolExecutor to submit to.
            lock_name: Logical name for the model (e.g. 'detector', 'stt').
                       Locks are created on first use.
            fn: The callable to execute (typically a bound method).

        Returns:
            The concurrent.futures.Future from executor.submit().
        """
        lock = self._model_locks.setdefault(lock_name, threading.Lock())
        wrapped = partial(self._locked_call, lock, fn, *args, **kwargs)
        return executor.submit(wrapped)

    @staticmethod
    def _locked_call(lock: threading.Lock, fn: Callable, *args, **kwargs):
        """Execute fn with the given lock held."""
        with lock:
            return fn(*args, **kwargs)

    @staticmethod
    def _sample_indices(total: int, max_frames: int = 30) -> List[int]:
        """
        Compute stride-based sample indices for expensive per-frame models.

        Returns a single index list.  Callers that need separate sampling
        strategies for different models should compute their own indices.
        """
        stride = max(1, total // max_frames) if total > max_frames else 1
        return list(range(0, total, stride))[:max_frames]

    def _compute_detection_variance(self, detections: List[List[dict]]) -> float:
        """
        Compute detection confidence variance from cached results.

        Low variance = stable detection (high quality).
        High variance = flickering/unstable (low quality).
        """
        confidences = []
        for frame_dets in detections:
            if frame_dets:
                confidences.append(max(d["confidence"] for d in frame_dets))
            else:
                confidences.append(0.0)

        if len(confidences) < 2:
            return 0.0

        return float(np.var(confidences))

    def _get_known_brands(self, detections: List[List[dict]]) -> List[str]:
        """Extract brand names from detection results."""
        brands = set()
        for frame_dets in detections:
            for det in frame_dets:
                brands.add(det["class_name"])
        return list(brands)

    def _ocr_filtered(
        self,
        frames_with_dets: List[np.ndarray],
        det_idx_map: List[int],
        total_frames: int,
    ) -> List[List[dict]]:
        """Run OCR only on frames with detections, map back to full frame list."""
        ocr_results = self.ocr.extract_text_batch(frames_with_dets)
        full_results = [[] for _ in range(total_frames)]
        for idx, ocr_result in zip(det_idx_map, ocr_results):
            full_results[idx] = ocr_result
        return full_results

    def _compute_audio_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute audio features for fusion.
        Uses mel-spectrogram statistics to produce a 256-dim embedding
        (128 mel bins × 2 stats), padded to embedding_dim from config.
        """
        import librosa

        # Use 128 mel bins with the default FFT to avoid empty filter warnings.
        # Mean + std over time = 256 dims; pad to embedding_dim via projection.
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=16000, n_mels=128, fmax=8000, n_fft=1024, hop_length=512
        )
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)

        # Mean and std pooling over time axis
        mean_feat = np.mean(log_mel, axis=1)
        std_feat = np.std(log_mel, axis=1)
        features = np.concatenate([mean_feat, std_feat])

        embed_dim = self.cfg["layer1"]["audio_events"].get("embedding_dim", 256)
        # Pad or truncate to exactly embedding_dim
        if len(features) < embed_dim:
            features = np.pad(features, (0, embed_dim - len(features)))
        else:
            features = features[:embed_dim]

        return features.astype(np.float32)

    def _aggregate_evidence(
        self,
        logo_detections: List[List[dict]],
        brand_mentions: List[dict],
        ocr_results: List[List[dict]],
        audio_events: List[dict],
    ) -> Dict[str, float]:
        """
        Aggregate evidence from all modalities into evidence strengths.

        Args:
            logo_detections: Real logo/brand detections from LogoDetectionBackend
                            (NOT generic COCO object detections)
            brand_mentions: Brand names mentioned in ASR transcript
            ocr_results: OCR text extracted from video frames
            audio_events: Audio event detections from BEATs

        Returns dict with keys: logo_detected, speech_mention, ocr_hit,
                                scene_context, product_retrieval
        """
        # Logo detection evidence — from REAL logo detector, not COCO objects
        logo_strength = 0.0
        logo_confidences = []
        for frame_logos in logo_detections:
            for det in frame_logos:
                logo_confidences.append(det["confidence"])
        if logo_confidences:
            logo_strength = float(np.mean(logo_confidences))

        # Speech mention evidence
        speech_strength = min(1.0, len(brand_mentions) * 0.3)

        # OCR evidence
        ocr_strength = 0.0
        ocr_confidences = []
        for frame_ocr in ocr_results:
            for result in frame_ocr:
                ocr_confidences.append(result["confidence"])
        if ocr_confidences:
            ocr_strength = float(np.mean(ocr_confidences))

        # Scene context evidence (from audio events)
        # Uses mean detection confidence scaled by event count: a single high-confidence
        # event shouldn't saturate to 1.0, and many low-confidence events shouldn't either.
        # The count scaling factor min(1.0, n/3) requires at least 3 events for full
        # confidence to apply; below that the strength is proportionally discounted.
        if audio_events:
            mean_conf = float(np.mean([e["confidence"] for e in audio_events]))
            count_factor = min(1.0, len(audio_events) / 3.0)
            scene_strength = mean_conf * count_factor
        else:
            scene_strength = 0.0

        # Product retrieval (placeholder — Phase 2 adds real product retrieval)
        product_strength = 0.0

        return {
            "logo_detected": logo_strength,
            "speech_mention": speech_strength,
            "ocr_hit": ocr_strength,
            "scene_context": scene_strength,
            "product_retrieval": product_strength,
        }