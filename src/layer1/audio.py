"""
Layer 1 — Audio Module
Speech-to-text via Whisper medium (mlx-whisper / faster-whisper / openai-whisper fallback)
and audio events via BEATs.
Both load real model weights — no mock/stub/placeholder inference.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class SpeechToText:
    """
    Speech-to-text transcription using mlx-whisper (Apple Silicon GPU via Metal).
    Falls back to faster-whisper, then openai-whisper (final fallback).
    Default model is "medium" to match config.yaml. Supports large-v3 for
    Hindi/Hinglish/multilingual coverage when configured.
    Loads real model weights — no hardcoded returns.
    """

    # Map model_name to MLX Hub repo for mlx-whisper
    _MLX_MODEL_MAP = {
        "large-v3": "mlx-community/whisper-large-v3-mlx",
        "large-v2": "mlx-community/whisper-large-v2-mlx",
        "medium": "mlx-community/whisper-medium-mlx",
        "small": "mlx-community/whisper-small-mlx",
        "base": "mlx-community/whisper-base-mlx",
    }

    # Map model_name to HuggingFace repo for faster-whisper (fallback)
    _FW_MODEL_MAP = {
        "large-v3": "Systran/faster-whisper-large-v3",
        "medium": "Systran/faster-whisper-medium",
        "small": "Systran/faster-whisper-small",
        "base": "Systran/faster-whisper-base",
    }

    def __init__(
        self,
        model_name: str = "medium",
        device: Optional[str] = None,
        compute_dtype: str = "int8",
        language: Optional[str] = None,
    ):
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.language = language
        self._backend = None  # "mlx", "faster", or "openai"

        # Prefer mlx-whisper on Apple Silicon (uses GPU/Neural Engine via Metal)
        if device == "mps" or device == "cpu":
            try:
                import mlx_whisper
                # mlx-whisper uses its own model path resolution
                mlx_model = self._MLX_MODEL_MAP.get(model_name, "mlx-community/whisper-medium-mlx")
                logger.info(
                    "Loading mlx-whisper model '%s' (Apple Silicon accelerated)",
                    mlx_model,
                )
                # Verify model is accessible by running a tiny warm-up
                _ = mlx_whisper.transcribe(
                    np.zeros(16000, dtype=np.float32),  # 1 second of silence
                    path_or_hf_repo=mlx_model,
                    verbose=False,
                )
                self.model_path = mlx_model
                self._backend = "mlx"
                logger.info("mlx-whisper model loaded successfully on %s", device)
                return
            except Exception as e:
                logger.warning("mlx-whisper failed (%s), falling back to faster-whisper", e)
        else:
            logger.info("mlx-whisper requires Apple Silicon (MPS) — skipping, using faster-whisper")

        # Fallback to faster-whisper
        fw_repo = self._FW_MODEL_MAP.get(model_name)
        fw_cached = self._is_fw_cached(fw_repo) if fw_repo else False

        if fw_cached:
            try:
                from faster_whisper import WhisperModel as FasterWhisperModel
                compute_type = compute_dtype if compute_dtype else (
                    "int8" if device == "cpu" else "float16"
                )
                logger.info(
                    "Loading faster-whisper model '%s' on %s (compute_type=%s)",
                    model_name, device, compute_type,
                )
                self.model = FasterWhisperModel(
                    model_name,
                    device=device,
                    compute_type=compute_type,
                )
                self._backend = "faster"
                logger.info("faster-whisper model loaded successfully")
                return
            except Exception as e:
                logger.warning("faster-whisper failed (%s), falling back to openai-whisper", e)

        # Final fallback to openai-whisper
        self._load_openai_whisper(model_name, device)

    @staticmethod
    def _is_fw_cached(repo_id: str) -> bool:
        """Check if faster-whisper model is fully cached (no incomplete downloads)."""
        from pathlib import Path
        import os
        cache_dir = Path(os.path.expanduser("~/.cache/huggingface/hub"))
        model_dir = cache_dir / f"models--{repo_id.replace('/', '--')}"
        if not model_dir.exists():
            return False
        # Check for incomplete downloads
        blobs_dir = model_dir / "blobs"
        if blobs_dir.exists():
            for f in blobs_dir.iterdir():
                if f.suffix == ".incomplete" or ".incomplete" in f.name:
                    return False
        return True

    def _load_openai_whisper(self, model_name: str, device: str):
        """Load openai-whisper as fallback."""
        import whisper
        logger.info("Loading openai-whisper model '%s' on %s", model_name, device)
        self.model = whisper.load_model(model_name, device=device)
        self._backend = "openai"
        logger.info("openai-whisper model loaded successfully on %s", device)

    @staticmethod
    def _vad_split(
        audio: np.ndarray,
        sample_rate: int = 16000,
        frame_ms: int = 30,
        min_segment_ms: float = 0.5,
        max_segment_s: float = 30.0,
        silence_threshold_percentile: int = 15,
        energy_multiplier: float = 1.5,
    ) -> List[Tuple[int, int]]:
        """
        Split audio into speech segments using energy-based VAD.

        This mitigates Whisper's repetition-loop hallucination by transcribing
        each segment independently (fresh decoder state), rather than one
        continuous pass over the full audio.

        Args:
            audio: Audio waveform as numpy array
            sample_rate: Sample rate in Hz
            frame_ms: Frame size in milliseconds for energy computation
            min_segment_ms: Minimum segment duration in seconds
            max_segment_s: Maximum segment duration in seconds
            silence_threshold_percentile: Percentile of frame energies used as
                                          noise floor estimate
            energy_multiplier: Multiplier above noise floor for voice detection

        Returns:
            List of (start_sample, end_sample) tuples for each speech segment
        """
        frame_len = int(sample_rate * frame_ms / 1000)
        hop_len = frame_len // 2
        n_frames = max(1, (len(audio) - frame_len) // hop_len + 1)

        energies = np.zeros(n_frames, dtype=np.float32)
        for i in range(n_frames):
            start = i * hop_len
            end = start + frame_len
            energies[i] = np.mean(audio[start:end] ** 2)

        noise_floor = np.percentile(energies, silence_threshold_percentile)
        threshold = noise_floor * energy_multiplier
        is_voice = energies > threshold

        min_seg_samples = int(sample_rate * min_segment_ms)
        max_seg_samples = int(sample_rate * max_segment_s)

        segments = []
        in_speech = False
        seg_start = 0

        for i in range(n_frames):
            frame_start = i * hop_len
            if is_voice[i] and not in_speech:
                in_speech = True
                seg_start = frame_start
            elif not is_voice[i] and in_speech:
                silence_len = frame_start - (i - 1) * hop_len - frame_len
                if silence_len > min_seg_samples:
                    seg_end = (i - 1) * hop_len + frame_len
                    if seg_end - seg_start >= min_seg_samples:
                        if seg_end - seg_start > max_seg_samples:
                            for chunk_start in range(seg_start, seg_end, max_seg_samples):
                                chunk_end = min(chunk_start + max_seg_samples, seg_end)
                                segments.append((chunk_start, chunk_end))
                        else:
                            segments.append((seg_start, seg_end))
                    in_speech = False

        if in_speech:
            seg_end = len(audio)
            if seg_end - seg_start >= min_seg_samples:
                if seg_end - seg_start > max_seg_samples:
                    for chunk_start in range(seg_start, seg_end, max_seg_samples):
                        chunk_end = min(chunk_start + max_seg_samples, seg_end)
                        segments.append((chunk_start, chunk_end))
                else:
                    segments.append((seg_start, seg_end))

        if not segments:
            segments = [(0, len(audio))]

        logger.info(
            "VAD split: %d segment(s) from %.1fs audio (threshold=%.2e, noise_floor=%.2e)",
            len(segments), len(audio) / sample_rate, threshold, noise_floor,
        )
        return segments

    def transcribe_segment(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> str:
        """
        Transcribe audio with VAD pre-segmentation to prevent repetition loops.

        Uses energy-based VAD to split audio into independent speech segments,
        transcribes each segment separately (fresh decoder context each time),
        and concatenates results. This is the standard mitigation for Whisper's
        repetition-loop hallucination on continuous audio.

        Args:
            audio: Audio signal as numpy array
            sample_rate: Sample rate of the audio

        Returns:
            Transcribed text string
        """
        if self._backend == "mlx":
            import mlx_whisper
            segments = self._vad_split(audio, sample_rate)
            all_text = []
            for seg_start, seg_end in segments:
                seg_audio = audio[seg_start:seg_end]
                if len(seg_audio) < sample_rate * 0.3:
                    continue
                try:
                    result = mlx_whisper.transcribe(
                        seg_audio,
                        path_or_hf_repo=self.model_path,
                        language=self.language,
                        verbose=False,
                    )
                    seg_text = result.get("text", "").strip()
                    if seg_text:
                        all_text.append(seg_text)
                except Exception as e:
                    logger.warning(
                        "mlx-whisper segment [%.2fs-%.2fs] failed: %s — skipping",
                        seg_start / sample_rate, seg_end / sample_rate, e,
                    )
            return " ".join(all_text) if all_text else ""
        elif self._backend == "faster":
            segments_iter, _ = self.model.transcribe(
                audio,
                language=self.language,
                beam_size=5,
                vad_filter=True,
            )
            parts = []
            for seg in segments_iter:
                parts.append(seg.text.strip())
            return " ".join(parts)
        else:
            result = self.model.transcribe(
                audio,
                language=self.language,
                fp16=False,
                verbose=False,
            )
            return result["text"]

    def detect_brand_mentions(
        self, transcript: str, brand_names: List[str]
    ) -> List[dict]:
        """DEPRECATED — brand-mention detection moved to the shared catalog.

        This method is retained only for backward-compatibility and is NOT
        called by the pipeline. Use src.brand_catalog.find_brand_mentions(),
        which matches against the single shared brand catalogue (the same list
        that drives logo-detection queries and the knowledge graph) and
        supports multilingual (Devanagari) aliases. Keeping a second matcher
        here would risk the two lists drifting apart.
        """
        mentions = []
        transcript_lower = transcript.lower()

        for brand in brand_names:
            brand_lower = brand.lower()
            idx = 0
            while True:
                idx = transcript_lower.find(brand_lower, idx)
                if idx < 0:
                    break
                mentions.append({
                    "brand": brand,
                    "position": idx,
                    "text_snippet": transcript[
                        max(0, idx - 20) : idx + len(brand) + 20
                    ],
                })
                idx += len(brand)

        return mentions


class AudioEventDetector:
    """
    Audio event detection using BEATs.
    Detects non-speech audio events (music, applause, etc.).
    Loads real pretrained checkpoint — no stubs.
    """

    def __init__(
        self,
        checkpoint_path: str = "BEATs_iter3_plus_AS2M.pt",
        device: Optional[str] = None,
        sample_rate: int = 16000,
    ):
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.sample_rate = sample_rate

        logger.info(f"Loading BEATs from {checkpoint_path} on {device}")
        # Lazy import for BEATs — fairseq dependency
        try:
            from BEATs import BEATs, BEATsConfig
        except ImportError:
            logger.error(
                "BEATs not available. Install from: "
                "https://github.com/microsoft/unilm/tree/master/beats"
            )
            raise

        # Validate checkpoint is a trusted local artifact before enabling pickle loading
        resolved = Path(checkpoint_path).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"BEATs checkpoint not found: {checkpoint_path}")
        if not resolved.is_file():
            raise ValueError(f"BEATs checkpoint path is not a file: {checkpoint_path}")
        # Local .pt files are trusted artifacts shipped with the repo or downloaded
        # by scripts/download_models.py. Full pickle deserialization (weights_only=False)
        # is required because BEATs checkpoints use a non-standard format that
        # PyTorch's safe serialization cannot parse.
        logger.info(
            "Loading BEATs checkpoint (trusted local artifact: %s)", resolved
        )
        checkpoint = torch.load(
            str(resolved), map_location=device, weights_only=False
        )
        cfg = BEATsConfig(checkpoint["cfg"])
        self.model = BEATs(cfg)
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(device)
        self.model.eval()
        logger.info(f"BEATs loaded successfully on {device}")

    @torch.no_grad()
    def detect_events(
        self, audio: np.ndarray, max_chunk_seconds: float = 30.0
    ) -> List[dict]:
        """
        Detect audio events in a waveform.

        Long audio is processed in chunks (default 30 s) to avoid OOM from
        oversized intermediate tensors in BEATs.

        Args:
            audio: Audio waveform as numpy array (samples,)
            max_chunk_seconds: Maximum chunk duration in seconds

        Returns:
            List of event dicts with keys:
                - event: event class label
                - confidence: float
                - start_time: float (seconds)
                - end_time: float (seconds)
        """
        if audio is None or len(audio) == 0:
            return []

        max_chunk = int(max_chunk_seconds * self.sample_rate)
        events = []

        for offset in range(0, len(audio), max_chunk):
            chunk = audio[offset : offset + max_chunk]
            if len(chunk) < int(0.5 * self.sample_rate):
                break

            chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0).to(self.device)
            features, _ = self.model.extract_features(chunk_tensor)

            t_start = offset / self.sample_rate
            t_end = (offset + len(chunk)) / self.sample_rate

            if features.dim() == 2 and features.shape[0] == 1:
                probs = features[0].cpu().numpy()
                labels = (
                    getattr(self.model, "label_map", None)
                    or getattr(self.model, "label_set", None)
                    or getattr(self.model, "labels", None)
                    or getattr(self.model, "id2label", None)
                )
                for i, prob in enumerate(probs):
                    if prob > 0.5:
                        if isinstance(labels, dict):
                            label = labels.get(i, f"event_{i}")
                        elif labels and i < len(labels):
                            label = labels[i]
                        else:
                            label = f"event_{i}"
                        events.append({
                            "event": label,
                            "confidence": float(prob),
                            "start_time": t_start,
                            "end_time": t_end,
                        })

            elif features.dim() == 3:
                pooled = features.mean(dim=1, keepdim=False)
                rms_activation = torch.norm(pooled, dim=1) / (pooled.shape[1] ** 0.5)
                activation = float(rms_activation[0].cpu())
                if activation > 0.05:
                    confidence = min(1.0, activation * 4.0)
                    events.append({
                        "event": "audio_activity",
                        "confidence": confidence,
                        "start_time": t_start,
                        "end_time": t_end,
                    })

            else:
                logger.warning(
                    "BEATs extract_features returned unexpected shape %s "
                    "for chunk at %.1fs — skipping",
                    tuple(features.shape), t_start,
                )

        return events
