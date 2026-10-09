"""
Layer 1 — OCR Module

Runs PaddleOCR inside a dedicated worker subprocess (src/layer1/ocr_worker.py)
so the paddle runtime is never imported into the torch/MPS server process.
Co-existing paddle + torch in one process intermittently corrupts operator
dispatch ("Tensor holds no memory" inside torchvision NMS — PaddleOCR
#11559/#16199), so OCR is isolated by process, not just by thread.

The public interface (extract_text / extract_text_batch) is unchanged so
callers (pipeline, brand_resolver, product resolver) keep working as-is.
"""

import base64
import json
import logging
import os
import select
import subprocess
import sys
import threading
from pathlib import Path
from typing import List

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_CONFIG_LANG = "en"

# Seconds to wait for one worker reply. OCR on a large batch can genuinely take
# minutes on CPU, so this is generous — it exists to turn a wedged worker into a
# clean retry instead of an indefinite block.
_WORKER_READ_TIMEOUT_SEC = float(
    os.environ.get("CONTEXTLENS_OCR_WORKER_TIMEOUT", "600")
)
# The worker sends its stdout chatter here rather than to DEVNULL, so a crash
# or a Paddle warning is actually diagnosable after the fact.
_WORKER_LOG_DIR = Path(__file__).resolve().parents[2] / "var" / "logs"


def _encode_rgb_jpeg(image: np.ndarray) -> bytes:
    """RGB frame -> JPEG bytes (PaddleOCR expects BGR; the worker decodes from
    JPEG straight into BGR, so the colorspace flip needs no bookkeeping here)."""
    if image.ndim == 3 and image.shape[2] == 3:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image
    _, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return buf.tobytes()


class OCRExtractor:
    """
    OCR text extraction via an isolated PaddleOCR worker subprocess.

    Detects and recognizes text in video frames — real inference, just in a
    separate process so paddle never shares a runtime with torch.
    """

    def __init__(
        self,
        lang: str = "en",
        use_angle_cls: bool = True,
        det_db_thresh: float = 0.3,
        rec_batch_num: int = 6,
        cpu_threads: int = 0,
    ):
        self.lang = lang
        # Forwarded to the worker (as JSON, see _ensure_worker) instead of being
        # dropped: the worker is a separate interpreter, so config that never
        # crosses the process boundary did nothing.
        self.config = {
            "use_angle_cls": use_angle_cls,
            "det_db_thresh": det_db_thresh,
            "rec_batch_num": rec_batch_num,
            "cpu_threads": cpu_threads,
        }
        self._proc: "subprocess.Popen | None" = None
        self._req_id = 0
        self._lock = threading.Lock()
        logger.info("OCRExtractor configured (lang=%s, worker-process mode)", lang)

    # ── Worker subprocess lifecycle ─────────────────────────────

    def _ensure_worker(self):
        if self._proc is not None and self._proc.poll() is None:
            return
        os.environ.setdefault("PYTHONUNBUFFERED", "1")
        # A file, not a PIPE: nobody ever drains a stderr pipe here, so it would
        # fill and block the worker mid-OCR.
        _WORKER_LOG_DIR.mkdir(parents=True, exist_ok=True)
        err_log = open(_WORKER_LOG_DIR / f"ocr_worker-{self.lang}.log", "a",
                       encoding="utf-8")
        self._proc = subprocess.Popen(
            [
                sys.executable, "-u",
                "-m", "src.layer1.ocr_worker", self.lang,
                json.dumps(self.config),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=err_log,
            text=True,
            encoding="utf-8",
            # `-m src.layer1.ocr_worker` is resolved against the cwd, so pin it
            # to the repo root or the worker dies with ModuleNotFoundError.
            cwd=str(Path(__file__).resolve().parents[2]),
        )
        # The child holds its own dup of the fd, so close the parent's copy.
        # Leaving it open leaked one descriptor per worker respawn (a crash
        # loop burned through the process FD limit).
        err_log.close()
        logger.info("Started PaddleOCR worker pid=%s (stderr -> %s)",
                    self._proc.pid, err_log.name)

    def _worker_exec(self, images: List[np.ndarray]) -> List[List[dict]]:
        """Send one batch of RGB frames to the worker and return parsed results.

        The worker is supervised: if it dies mid-request the batch is retried
        once against a freshly spawned worker, so a crash never silently drops
        OCR results (and the dead handle is recycled).
        """
        encoded = [
            base64.b64encode(_encode_rgb_jpeg(f)).decode("ascii")
            for f in images if f is not None and f.size > 0
        ]
        valid = [j for j, f in enumerate(images) if f is not None and f.size > 0]
        with self._lock:
            results = None
            for attempt in range(2):
                results = self._request_once(encoded)
                if results is not None:
                    break
                logger.warning(
                    "OCR worker died mid-request; respawning (attempt %d/2)",
                    attempt + 1,
                )
            if results is None:
                raise RuntimeError("OCR worker crashed repeatedly — giving up")
        full: List[List[dict]] = [[] for _ in images]
        for slot, dets in zip(valid, results):
            full[slot] = dets
        return full

    def _request_once(self, encoded: List[str]) -> "List[List[dict]] | None":
        """One request round-trip. Returns parsed results, or None if the worker
        died (handle recycled for a supervised retry). Raises on worker error.
        Caller must hold self._lock so request/response pairs never interleave
        across threads."""
        self._ensure_worker()
        self._req_id += 1
        payload = json.dumps({"id": self._req_id, "images": encoded})
        try:
            if self._proc.stdin is None or self._proc.stdout is None:
                raise RuntimeError("OCR worker pipe unavailable")
            self._proc.stdin.write(payload + "\n")
            self._proc.stdin.flush()
            # A bare readline() blocks forever if the worker wedges. select()
            # turns that into a clean retry against a fresh worker.
            ready, _, _ = select.select([self._proc.stdout], [], [],
                                        _WORKER_READ_TIMEOUT_SEC)
            if not ready:
                logger.error("OCR worker pid=%s produced no reply within %ss",
                             self._proc.pid, _WORKER_READ_TIMEOUT_SEC)
                self._proc.kill()
                self._proc = None
                return None
            resp_line = self._proc.stdout.readline()
        except (BrokenPipeError, ValueError, OSError):
            self._proc = None
            return None
        if not resp_line:
            self._proc = None
            return None
        try:
            resp = json.loads(resp_line)
        except json.JSONDecodeError:
            # Non-JSON on the protocol pipe. The worker now sends its own
            # library chatter to stderr, so this means a real protocol break:
            # log it and recycle rather than taking the whole stage down.
            logger.error("OCR worker returned non-JSON protocol line: %.200r",
                         resp_line)
            self._proc = None
            return None
        if resp.get("error"):
            raise RuntimeError(f"OCR worker error: {resp['error']}")
        return resp.get("results", [])

    def _shutdown(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            try:
                if self._proc.stdin is not None:
                    self._proc.stdin.write(json.dumps({"id": -1, "method": "exit"}) + "\n")
                    self._proc.stdin.flush()
                    self._proc.stdin.close()
                self._proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                self._proc.terminate()
        self._proc = None

    # ── Public OCR interface (unchanged) ─────────────────────────

    def extract_text(self, image: np.ndarray) -> List[dict]:
        """Extract text from a single RGB image frame."""
        if image is None or image.size == 0:
            return []
        return self._worker_exec([image])[0]

    def extract_text_batch(self, frames: List[np.ndarray]) -> List[List[dict]]:
        """Run OCR on a batch of frames, aligning results to inputs."""
        if not frames:
            return []
        return self._worker_exec(frames)

    def __del__(self):
        try:
            self._shutdown()
        except Exception:  # noqa: BLE001
            pass


def create_ocr_extractor(
    lang: str = "en",
    use_angle_cls: bool = True,
    det_db_thresh: float = 0.3,
    rec_batch_num: int = 6,
    cpu_threads: int = 0,
    **_kwargs,
) -> OCRExtractor:
    """Factory: create the configured OCR extractor (worker-process mode)."""
    if lang != "en":
        logger.info("OCR lang override: %s", lang)
    return OCRExtractor(
        lang=lang,
        use_angle_cls=use_angle_cls,
        det_db_thresh=det_db_thresh,
        rec_batch_num=rec_batch_num,
        cpu_threads=cpu_threads,
    )


if __name__ == "__main__":
    # Self-check: protocol alignment without spawning a subprocess.
    extractor = OCRExtractor.__new__(OCRExtractor)
    extractor._lock = threading.Lock()

    def _fake_worker(images):
        return [[
            {"text": "TEXT", "confidence": 0.9,
             "bbox": [[0, 0], [1, 0], [1, 1], [0, 1]]}
        ] if i % 2 == 0 else [] for i in range(len(images))]

    extractor._worker_exec = _fake_worker
    frames = [np.zeros((4, 4, 3), np.uint8) for _ in range(3)] + [None]
    parsed = extractor.extract_text_batch(frames)
    assert [len(r) for r in parsed] == [1, 0, 1, 0], parsed
    assert parsed[2][0]["text"] == "TEXT"
    print("ocr worker-mode self-check OK")