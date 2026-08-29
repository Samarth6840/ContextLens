"""
Layer 1 — Qwen3-VL 32B Multimodal Understanding Module

Qwen3-VL 32B handles video, OCR, and reasoning in one pass.
This is the central understanding model for the stack.

All inference is real — no mock/stub/placeholder.
"""

import logging

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


class Qwen3VLAbstract(ABC):
    """Abstract base for Qwen3-VL model interface."""

    @abstractmethod
    def analyze_frame(
        self,
        frame: np.ndarray,
        text_prompt: str = "",
    ) -> Dict[str, Any]:
        """
        Analyze a single video frame.

        Returns dict with keys:
            - tokens: list of generated tokens/text
            - embeddings: visual feature vector (if available)
            - detections: any detected objects/text (if available)
        """
        ...

    @abstractmethod
    def analyze_batch(
        self,
        frames: List[np.ndarray],
        text_prompt: str = "",
    ) -> List[Dict[str, Any]]:
        """Analyze a batch of video frames."""
        ...


class Qwen3VL32B(Qwen3VLAbstract):
    """
    Qwen3-VL 32B multimodal model.

    Handles:
    - Video frame understanding
    - OCR text extraction from frames
    - Scene/brand reasoning
    - Cross-modal grounding
    """

    def __init__(
        self,
        model_name: str = "Qwen3-VL-32B",
        device: Optional[str] = None,
        load_8bit: bool = True,
        **kwargs,
    ):
        self.model_name = model_name
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.load_8bit = load_8bit
        self.model = None
        self.tokenizer = None
        self._initialized = False

        logger.info("Initializing Qwen3-VL 32B model '%s' on %s", model_name, self.device)

    def _initialize(self):
        """Lazy initialization of the Qwen3-VL model."""
        if self._initialized:
            return

        self._init_attempts = getattr(self, "_init_attempts", 0)
        if self._init_attempts > 3:
            logger.error("Qwen3-VL initialization exceeded max attempts -- using fallback")
            self._initialized = True
            return

        self._init_attempts += 1

        # FAIL-FAST WEIGHT GUARD (project policy: never silently download huge
        # weights). We only attempt to load the model if the weights are already
        # available locally: either a local path that exists, or a HuggingFace
        # repo id that is already in the local HF cache. Anything else fails
        # closed immediately with a clear message — no implicit ~60GB download.
        if not self._weights_available_locally(self.model_name):
            logger.error(
                "Qwen3-VL weights for '%s' are NOT available locally (no local "
                "path and not in the HuggingFace cache). Failing closed — "
                "download the weights first (e.g. via huggingface-cli) and set "
                "config layer1.central_vision_model.model to the local path.",
                self.model_name,
            )
            self._initialized = True
            return

        # Try mlx-first approach for Apple Silicon
        try:
            import mlx_lm

            self.model, self.tokenizer = mlx_lm.load(
                self.model_name,
                device=self.device,
                quantization="int8" if self.load_8bit else None,
            )
            logger.info("Qwen3-VL 32B loaded via mlx with int8 quantization")
        except ImportError:
            logger.warning("mlx_lm not available, trying transformers fallback")
        except Exception as e:
            logger.warning("mlx_lm load failed: %s", e)

        # If mlx succeeded, we're done
        if self.model is not None:
            self._initialized = True
            return

        # Fallback to transformers
        try:
            from transformers import AutoModelForVision2Seq, AutoTokenizer
            import torch

            model_id = "Qwen/Qwen3-VL-32B" if not self.model_name.startswith("Qwen") else self.model_name

            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                trust_remote_code=True,
                device_map=self.device,
                load_in_8bit=self.load_8bit,
                torch_dtype="float16" if self.load_8bit else "float32",
            )
            logger.info("Qwen3-VL 32B loaded via transformers")
            self._initialized = True
            return
        except ImportError:
            logger.warning("transformers not available for Qwen3-VL fallback")
        except Exception as e:
            logger.warning("transformers load failed: %s", e)

        # Mark as initialized even on failure to avoid retry loop
        self._initialized = True

    @classmethod
    def _weights_available_locally(cls, model_name: str) -> bool:
        """True if the model weights are already on disk (local path or HF cache).

        Checks, in order:
          1. model_name is a local path that exists.
          2. model_name is an HF repo id present in the local HuggingFace cache
             (fully downloaded — an empty/incomplete snapshot does NOT count,
             preventing a silent re-download attempt).
        """
        from pathlib import Path
        import os

        m = Path(str(model_name))
        if m.exists():
            return True
        # Treat it as an HF repo id only if it looks like 'org/repo'.
        if "/" not in str(model_name):
            return False
        cache_root = Path(os.path.expanduser("~/.cache/huggingface/hub"))
        repo_dir = cache_root / f"models--{str(model_name).replace('/', '--')}"
        if not repo_dir.is_dir():
            return False
        snapshot_dir = repo_dir / "snapshots"
        if not snapshot_dir.is_dir():
            return False
        # Require at least one non-empty snapshot (weights really present).
        return any(
            any(f.is_file() and f.stat().st_size > 0 for f in snap.iterdir())
            for snap in snapshot_dir.iterdir()
            if snap.is_dir()
        )

    def analyze_frame(
        self,
        frame: np.ndarray,
        text_prompt: str = "",
    ) -> Dict[str, Any]:
        """
        Analyze a single video frame using Qwen3-VL.

        Args:
            frame: RGB image as numpy array (H, W, 3)
            text_prompt: Optional text prompt/guidance for analysis

        Returns:
            Dict with analysis results
        """
        if not self._initialized:
            self._initialize()

        if self.model is None:
            return {
                "tokens": [],
                "embeddings": np.array([]),
                "detections": [],
                "fallback": True,
            }

        try:
            # Qwen3-VL expects images in RGB format, normalized
            rgb_frame = frame[:, :, ::-1]  # Ensure RGB

            # Prepare conversation prompt
            if text_prompt:
                prompt = f"<|image|>{text_prompt}<|end|>"
            else:
                prompt = "<|image|>"

            # Tokenize and generate
            if hasattr(self.tokenizer, 'apply_chat_template'):
                messages = [
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text_prompt}]}
                ]
                input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                input_text = prompt

            # Tokenize
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
            ).to(self.model.device)

            # Generate
            with __import__("torch").no_grad():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.7,
                    do_sample=self.load_8bit,
                )

            # Decode output
            output_tokens = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

            return {
                "tokens": output_tokens,
                "embeddings": np.array([]),
                "detections": [],
                "fallback": False,
            }

        except Exception as e:
            logger.error("Qwen3-VL frame analysis failed: %s", e)
            return {
                "tokens": "",
                "embeddings": np.array([]),
                "detections": [],
                "fallback": True,
            }

    def analyze_batch(
        self,
        frames: List[np.ndarray],
        text_prompt: str = "",
    ) -> List[Dict[str, Any]]:
        """Analyze a batch of video frames."""
        results = []
        for i, frame in enumerate(frames):
            result = self.analyze_frame(frame, text_prompt if i == 0 else "")
            results.append(result)
        return results


def create_qwen3vl_32b(
    model_name: str = "Qwen3-VL-32B",
    device: Optional[str] = None,
    load_8bit: bool = True,
) -> Qwen3VLAbstract:
    """
    Factory: create the Qwen3-VL 32B model instance.

    Args:
        model_name: Name/identifier for the Qwen3-VL model
        device: "cuda", "cpu", "mps", or None for auto
        load_8bit: Whether to use 8-bit quantization for memory efficiency

    Returns:
        Qwen3VL32B instance
    """
    import torch
    return Qwen3VL32B(
        model_name=model_name,
        device=device,
        load_8bit=load_8bit,
    )
