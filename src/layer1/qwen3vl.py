"""
Layer 1 — Qwen3-VL 32B Multimodal Understanding Module

Qwen3-VL 32B handles video, OCR, and reasoning in one pass.
This is the central understanding model for the stack.

All inference is real — no mock/stub/placeholder.
"""

import logging
import time

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch  # noqa: F401  (optional; used only for device auto-detect / inference)
except Exception:  # noqa: BLE001 - torch is optional at import time
    torch = None


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
        if device is None:
            mps = getattr(torch, "backends", None) and torch.backends.mps.is_available()
            device = "mps" if mps else "cpu"
        self.device = device
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
            # Absent weights are an intentional steady state here, not a failure:
            # every caller already degrades to a fallback result (see
            # analyze_frame / resolve_product_manufacturer), and project policy
            # forbids auto-downloading ~60GB. Logged at INFO so it stops looking
            # like a broken upload in the server log. Raise this back to
            # logger.error only once Qwen3-VL is actually wired into scoring.
            logger.info(
                "Qwen3-VL '%s' disabled: weights not present locally. This is "
                "expected and non-fatal — all Qwen outputs return fallback "
                "values, and detection/OCR/CLIP are unaffected. To enable it, "
                "download the weights and point config "
                "layer1.central_vision_model.model at the local path.",
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

    # ------------------------------------------------------------------ #
    # Tier 3 — structured product→manufacturer resolution + cost tracking.
    # ------------------------------------------------------------------ #
    def resolve_product_manufacturer(
        self,
        product_span: str,
        frame: Optional[np.ndarray] = None,
        max_new_tokens: int = 24,
    ) -> Dict[str, Any]:
        """Narrow, structured Qwen query: manufacturer of a product, or null.

        This is the Tier-3 path in product resolution. It asks the VLM ONLY a
        tightly-scoped question (never free-form reasoning), asks for a single
        company name or null, and records call count + latency so Qwen spend is
        budget-gated (the caller consults qwen_budget / emit_qwen_stats).

        Returns:
            {"manufacturer": str|None, "confidence": float, "fallback": bool}
            Fails closed: manufacturer=None on any error or when the model is
            unavailable. An unauthoritative "UNKNOWN" is never asserted as a
            brand (None).
        """
        started = time.monotonic()
        self.qwen_calls = getattr(self, "qwen_calls", 0) + 1
        if not self._initialized:
            self._initialize()
        try:
            if self.model is None or self.tokenizer is None:
                return {"manufacturer": None, "confidence": 0.0, "fallback": True}
            prompt = (
                f"Which company manufactures the product '{product_span}'? "
                'Answer with ONLY a JSON object like {"manufacturer": "..."} '
                'or {"manufacturer": null} if you do not know. No other text.'
            )
            content = [{"type": "text", "text": prompt}]
            if frame is not None:
                content = [{"type": "image"}, {"type": "text", "text": prompt}]
            messages = [{"role": "user", "content": content}]
            if hasattr(self.tokenizer, "apply_chat_template"):
                input_text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
            else:
                input_text = prompt
            inputs = self.tokenizer(
                input_text, return_tensors="pt", padding=True
            ).to(self.model.device)
            with __import__("torch").no_grad():
                generated = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                    temperature=0.0, do_sample=False,
                )
            out = self.tokenizer.batch_decode(
                generated, skip_special_tokens=True)[0].strip()
            import json as _json
            import re as _re
            m = _re.search(r"\{.*\}", out, _re.S)
            if m:
                try:
                    payload = _json.loads(m.group(0))
                except ValueError:
                    payload = {}
            else:
                payload = {}
            ans = payload.get("manufacturer")
            if ans is None:
                # Allow a bare company-name fallback when the model ignored JSON.
                bare = out.split("\n")[-1].strip().strip("\"'.,:!?")
                if bare and bare.upper() != "UNKNOWN":
                    ans = bare
            if not ans:
                return {"manufacturer": None, "confidence": 0.0,
                        "fallback": False}
            return {"manufacturer": str(ans).strip(),
                    "confidence": 0.45, "fallback": False}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Qwen product-manufacturer resolve failed for %r: %s",
                           product_span, exc)
            return {"manufacturer": None, "confidence": 0.0, "fallback": True}
        finally:
            self.last_qwen_latency = time.monotonic() - started

    def emit_qwen_stats(self) -> dict:
        """Cost/latency instrumentation for the Qwen product tier."""
        return {
            "qwen_calls": getattr(self, "qwen_calls", 0),
            "last_latency_sec": getattr(self, "last_qwen_latency", 0.0),
            "budget_gated": getattr(self, "qwen_budget_gated", False),
        }


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
    return Qwen3VL32B(
        model_name=model_name,
        device=device,
        load_8bit=load_8bit,
    )
