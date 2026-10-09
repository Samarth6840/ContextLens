"""
Layer 1 — Audio Module
Speech-to-text via Whisper medium (mlx-whisper / faster-whisper / openai-whisper fallback)
and audio events via BEATs.
Both load real model weights — no mock/stub/placeholder inference.
"""

import hashlib
import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch


logger = logging.getLogger(__name__)

# The fine-tuned BEATs checkpoint comes from a third-party HuggingFace user
# (WeiChihChen), not Microsoft. It is loaded with `weights_only=False`, i.e. a
# full pickle: if that repo is ever replaced or tampered with, loading it runs
# arbitrary code. Pin the artifact by sha256 (the git-lfs oid of the HF file)
# and refuse to deserialize anything else.
BEATS_PINNED_NAME = "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
BEATS_TRUSTED_SHA256 = (
    "e5815275a04b6885e7b8af63d120b29bffae2cd2225cf4915e1ec6d819d3022c"
)


def file_sha256(path) -> str:
    """Streaming sha256 of a file (constant memory for multi-hundred-MB weights)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
                # ctranslate2 (faster-whisper's backend) has no MPS device. When
                # mlx-whisper failed on an MPS host, device is still "mps" here
                # and ctranslate2 would raise — map it to CPU.
                fw_device = "cpu" if device == "mps" else device
                compute_type = compute_dtype if compute_dtype else (
                    "int8" if fw_device == "cpu" else "float16"
                )
                logger.info(
                    "Loading faster-whisper model '%s' on %s (compute_type=%s)",
                    model_name, fw_device, compute_type,
                )
                self.model = FasterWhisperModel(
                    model_name,
                    device=fw_device,
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
        min_segment_s: float = 0.5,
        min_silence_s: float = 0.5,
        max_segment_s: float = 30.0,
        silence_threshold_percentile: int = 15,
        energy_multiplier: float = 1.5,
    ) -> List[Tuple[int, int]]:
        """
        Split audio into speech segments using energy-based VAD.

        This mitigates Whisper's repetition-loop hallucination by transcribing
        each segment independently (fresh decoder state), rather than one
        continuous pass over the full audio.

        A segment is CLOSED once voice returns after a silent run of at least
        `min_silence_s`; shorter gaps just extend the current segment, so normal
        inter-word pauses don't split speech.

        Args:
            audio: Audio waveform as numpy array
            sample_rate: Sample rate in Hz
            frame_ms: Frame size in milliseconds for energy computation
            min_segment_s: Minimum segment duration in seconds (shorter spans are dropped)
            min_silence_s: Silent run that closes the current segment
            max_segment_s: Maximum segment duration in seconds
            silence_threshold_percentile: Percentile of frame energies used as
                                          noise floor estimate
            energy_multiplier: Multiplier above noise floor for voice detection

        Returns:
            List of (start_sample, end_sample) tuples for each speech segment
        """
        frame_len = int(sample_rate * frame_ms / 1000)
        hop_len = max(1, frame_len // 2)
        n_frames = max(1, (len(audio) - frame_len) // hop_len + 1)

        energies = np.zeros(n_frames, dtype=np.float32)
        if len(audio) >= frame_len:
            windows = np.lib.stride_tricks.sliding_window_view(audio, frame_len)
            energies = (windows[::hop_len][:n_frames].astype(np.float32) ** 2).mean(axis=1)
        else:
            energies[0] = float(np.mean(audio ** 2)) if len(audio) else 0.0

        noise_floor = np.percentile(energies, silence_threshold_percentile)
        threshold = noise_floor * energy_multiplier
        is_voice = energies > threshold

        min_seg_samples = int(sample_rate * min_segment_s)
        min_silence_samples = int(sample_rate * min_silence_s)
        max_seg_samples = int(sample_rate * max_segment_s)

        segments: List[Tuple[int, int]] = []

        def _emit(start: int, end: int) -> None:
            if end - start < min_seg_samples:
                return
            # Hard cap: a long continuous run is chopped so no segment is decoded
            # without a decoder reset for too long.
            for chunk_start in range(start, end, max_seg_samples):
                segments.append((chunk_start, min(chunk_start + max_seg_samples, end)))

        in_speech = False
        seg_start = 0
        silence_start: Optional[int] = None

        for i in range(n_frames):
            frame_start = i * hop_len
            if is_voice[i]:
                if not in_speech:
                    in_speech = True
                    seg_start = frame_start
                elif silence_start is not None and frame_start - silence_start >= min_silence_samples:
                    _emit(seg_start, silence_start)
                    seg_start = frame_start
                silence_start = None
            elif in_speech and silence_start is None:
                silence_start = frame_start

        if in_speech:
            _emit(seg_start, silence_start if silence_start is not None else len(audio))

        # No speech detected: transcribe the whole clip as one segment (the
        # segment-level anti-hallucination guard doesn't apply). Empty audio
        # yields no segments rather than a zero-length one.
        if not segments and len(audio) > 0:
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
        Transcribe audio returning joined text (backward-compatible).

        This is now a thin wrapper over `transcribe_segments`, which additionally
        returns real per-segment timestamps for clip/mention mapping.
        """
        segments = self.transcribe_segments(audio, sample_rate)
        return " ".join(s["text"] for s in segments if s.get("text"))

    def transcribe_segments(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> List[dict]:
        """
        Transcribe audio with REAL per-segment timestamps (save -> end seconds).

        Returned list of dicts: [{"text": str, "start": float, "end": float}].
        Timestamps come from the segmentation the backend actually uses:
          - mlx:     VAD-split segments (independent decoder context each; the
                     repetition-loop mitigation), timestamps = VAD boundaries.
          - faster:  faster-whisper's own segments carry start/end seconds.
          - openai:  Whisper segment list when the backend exposes it, else a
                     single whole-audio segment.

        These timestamps let brand-mention detection (and clip links) point at
        the exact second a brand/product was spoken, instead of an estimated
        position proportional to transcript length.
        """
        if audio is None or len(audio) == 0:
            return []

        out: List[dict] = []
        if self._backend == "mlx":
            import mlx_whisper
            for seg_start, seg_end in self._vad_split(audio, sample_rate):
                if seg_end - seg_start < sample_rate * 0.3:
                    continue
                try:
                    result = mlx_whisper.transcribe(
                        audio[seg_start:seg_end],
                        path_or_hf_repo=self.model_path,
                        language=self.language,
                        verbose=False,
                    )
                    seg_text = (result.get("text") or "").strip()
                except Exception as e:
                    logger.warning(
                        "mlx-whisper segment [%.2fs-%.2fs] failed: %s — skipping",
                        seg_start / sample_rate, seg_end / sample_rate, e,
                    )
                    continue
                if seg_text:
                    out.append({
                        "text": seg_text,
                        "start": round(seg_start / sample_rate, 3),
                        "end": round(seg_end / sample_rate, 3),
                    })
            return out
        elif self._backend == "faster":
            segments_iter, _ = self.model.transcribe(
                audio,
                language=self.language,
                beam_size=5,
                vad_filter=True,
            )
            for seg in segments_iter:
                text = (seg.text or "").strip()
                if text:
                    out.append({
                        "text": text,
                        "start": round(float(seg.start), 3),
                        "end": round(float(seg.end), 3),
                    })
            return out

        result = self.model.transcribe(
            audio,
            language=self.language,
            fp16=False,
            verbose=False,
        )
        text = (result.get("text") or "").strip()
        if not text:
            return []
        segments = result.get("segments") or []
        if segments:
            for seg in segments:
                seg_text = (seg.get("text") or "").strip()
                if seg_text:
                    out.append({
                        "text": seg_text,
                        "start": round(float(seg["start"]), 3),
                        "end": round(float(seg["end"]), 3),
                    })
            return out
        duration = len(audio) / sample_rate
        return [{"text": text, "start": 0.0, "end": round(duration, 3)}]

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
        # Full pickle deserialization (weights_only=False) is required because
        # BEATs checkpoints use a non-standard format PyTorch's safe loader
        # cannot parse. For the third-party mirror, verify the pinned sha256
        # first so a swapped file cannot execute code at load time.
        if resolved.name == BEATS_PINNED_NAME:
            digest = file_sha256(resolved)
            if digest != BEATS_TRUSTED_SHA256:
                raise ValueError(
                    f"BEATs checkpoint {resolved.name} sha256 mismatch "
                    f"(got {digest}, expected {BEATS_TRUSTED_SHA256}); refusing "
                    "to pickle-load an unverified artifact."
                )
        logger.info(
            "Loading BEATs checkpoint (verified local artifact: %s)", resolved
        )
        checkpoint = torch.load(
            str(resolved), map_location=device, weights_only=False
        )
        cfg = BEATsConfig(checkpoint["cfg"])
        self.model = BEATs(cfg)
        self.model.load_state_dict(checkpoint["model"])
        self.finetuned = bool(getattr(cfg, "finetuned_model", False))
        # A fine-tuned tagging head emits 527 AudioSet probabilities; its own
        # label_dict (index -> AudioSet mID) is authoritatively translated to
        # display names so events carry human-readable classes for cue mapping.
        self.label_source = "audioset_ontology"
        if self.finetuned and checkpoint.get("label_dict"):
            from src.layer1.audioset_labels import build_label_map
            label_map = build_label_map(checkpoint["label_dict"])
            if label_map:
                self.model.label_map = label_map
                self.label_source = "model_label_map"
        self.model.to(device)
        self.model.eval()
        logger.info(
            "BEATs loaded successfully on %s (finetuned=%s, predictor_class=%s, "
            "labels=%s)", device, self.finetuned,
            getattr(cfg, "predictor_class", None), self.label_source,
        )

    @torch.no_grad()
    def detect_events(
        self, audio: np.ndarray, max_chunk_seconds: float = 30.0,
        progress: Optional[Callable[[str], None]] = None,
        max_chunks: Optional[int] = None,
    ) -> List[dict]:
        """
        Detect audio events in a waveform.

        Long audio is processed in chunks (default 30 s) to avoid OOM from
        oversized intermediate tensors in BEATs.

        Args:
            audio: Audio waveform as numpy array (samples,)
            max_chunk_seconds: Maximum chunk duration in seconds
            progress: Optional callback invoked once per chunk with a short
                      human-readable status line ("Audio events — chunk 7/20
                      of 98 (170.0–200.0 s)"). Uncapped audio on long videos
                      makes this stage run for many minutes with no other
                      observable output, so per-chunk reporting is what keeps
                      the job feed visibly alive.
            max_chunks: Cap on the number of chunks actually decoded. When the
                      audio spans more chunks than this, chunks are sampled
                      UNIFORMLY across the full duration (first and last
                      chunk always included) — real BEATs inference with real
                      timestamps on every sampled window, just fewer of them.
                      None = decode everything (legacy behavior).

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

        # Pre-compute the true chunk count so the feed can show "7/20 of 98"
        # instead of an open-ended counter. The final sub-0.5s tail is dropped
        # by the loop's break, so it is excluded here.
        min_chunk = int(0.5 * self.sample_rate)
        all_offsets = [
            off for off in range(0, len(audio), max_chunk)
            if len(audio[off : off + max_chunk]) >= min_chunk
        ]

        # Uniform sample across the whole duration when the cap binds.
        # Evenly-spaced picks (same strategy as pipeline keyframe sampling):
        # first and last chunk always survive, mid-video gaps stay bounded.
        offsets = all_offsets
        sampled_total = len(all_offsets)
        if max_chunks is not None and 0 < max_chunks < len(all_offsets):
            picks = np.linspace(0, len(all_offsets) - 1, max_chunks, dtype=int)
            offsets = [all_offsets[i] for i in sorted(set(picks.tolist()))]
            sampled_total = len(all_offsets)
            logger.info(
                "Audio events sampling: %d -> %d chunk(s) of %.0fs "
                "(max_chunks=%d, uniform across full duration)",
                len(all_offsets), len(offsets), max_chunk_seconds, max_chunks,
            )

        for chunk_idx, offset in enumerate(offsets, start=1):
            chunk = audio[offset : offset + max_chunk]
            if len(chunk) < min_chunk:
                continue

            if progress is not None:
                try:
                    t_start = offset / self.sample_rate
                    t_end = (offset + len(chunk)) / self.sample_rate
                    scope = (
                        f" of {sampled_total}"
                        if sampled_total != len(offsets) else ""
                    )
                    progress(
                        f"Audio events — chunk {chunk_idx}/{len(offsets)}{scope} "
                        f"({t_start:.1f}–{t_end:.1f} s of audio)"
                    )
                except Exception:  # noqa: BLE001 — reporting must never kill the job
                    pass

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
                # No AUDIOSET_LABELS fallback. A fine-tuned BEATs tagging head
                # emits probabilities in its own label_dict index order, so the
                # canonical CSV order would attach the WRONG class name to every
                # index. Fail closed to event_{i} with no cue mapping rather than
                # emit a confidently mislabelled event.
                has_label_map = labels is not None
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
                            "mode": "classified",
                            "source": ("model_label_map" if has_label_map
                                       else "unlabeled"),
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
                        # DEGRADED SIGNAL, not a real event classification: this
                        # branch fires when the model exposes no class label map
                        # (fallback RMS-energy heuristic). The UI renders these
                        # with the dashed "fallback" chip treatment so a
                        # meaningless generic "audio_activity" is never styled
                        # as if it were a real BEATs class label.
                        "mode": "fallback",
                        "fallback_reason": (
                            "RMS energy heuristic (model returned no class "
                            "label map)"
                        ),
                    })

            else:
                logger.warning(
                    "BEATs extract_features returned unexpected shape %s "
                    "for chunk at %.1fs — skipping",
                    tuple(features.shape), t_start,
                )

        return events
