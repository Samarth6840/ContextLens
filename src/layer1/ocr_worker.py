"""
Standalone PaddleOCR worker process.

Runs PaddleOCR in its OWN interpreter so that the paddle runtime is never
imported into the torch/MPS process. Co-existing paddle + torch in one process
intermittently corrupts operator dispatch, surfacing as Paddle's
"Tensor holds no memory" inside torchvision's NMS (PaddleOCR #11559/#16199).

Protocol: newline-delimited JSON on stdin/stdout.
  req : {"id": int, "images": [base64-encoded JPEG bytes, ...]}
  resp: {"id": int, "results": [[{text, confidence, bbox: [[x,y],...]} ...] ...]}
        or {"id": int, "error": "..."}
  exit: {"id": -1, "method": "exit"}
"""

import base64
import json
import os
import sys

import cv2
import numpy as np


def _init_ocr(lang: str, cfg: dict | None = None):
    import paddle  # noqa: F401  (must load before PaddleOCR; default CPU thread pool)

    from paddleocr import PaddleOCR as _PaddleOCR

    cfg = cfg or {}
    # The parent sends its configured values as JSON (see OCRExtractor); the
    # app keeps the historical config keys and they are mapped here onto the
    # PaddleOCR 3.x argument names (use_textline_orientation / text_det_thresh /
    # text_recognition_batch_size), which replaced the 2.x ones.
    kwargs = {
        "lang": lang,
        "use_textline_orientation": bool(cfg.get("use_angle_cls", True)),
        "text_det_thresh": float(cfg.get("det_db_thresh", 0.3)),
        "text_recognition_batch_size": int(cfg.get("rec_batch_num", 6)),
    }
    cpu_threads = int(cfg.get("cpu_threads", 0) or 0)
    if cpu_threads > 0:
        kwargs["cpu_threads"] = cpu_threads
    return _PaddleOCR(**kwargs)


def _parse_result(result) -> list:
    if result is None:
        return []
    if hasattr(result, "get"):
        rec_texts = result.get("rec_texts", [])
        rec_scores = result.get("rec_scores", [])
        rec_polys = result.get("rec_polys", [])
    else:
        rec_texts = getattr(result, "rec_texts", [])
        rec_scores = getattr(result, "rec_scores", [])
        rec_polys = getattr(result, "rec_polys", [])

    out = []
    for text, score, poly in zip(rec_texts, rec_scores, rec_polys):
        out.append({
            "text": text,
            "confidence": float(score),
            "bbox": [list(pt) for pt in (poly.tolist() if hasattr(poly, "tolist") else poly)],
        })
    return out


def main(lang: str = "en", cfg_json: str = "") -> None:
    # Claim a private duplicate of the real stdout for the protocol, then point
    # fd 1 at stderr. Paddle, cv2 and friends print progress bars and warnings to
    # stdout, and the parent reads this pipe line-by-line and json.loads it — one
    # stray line desynchronises the whole stream and raises. After this, any
    # library print goes to stderr and is logged instead of corrupting replies.
    proto = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr

    ocr = _init_ocr(lang, json.loads(cfg_json) if cfg_json else {})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        rid = req.get("id")
        if rid == -1 and req.get("method") == "exit":
            break
        try:
            results = []
            for b64 in req.get("images", []):
                jpg = base64.b64decode(b64)
                bgr = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                if bgr is None:
                    results.append([])
                    continue
                predicted = ocr.predict(bgr)
                results.append(_parse_result(predicted[0]) if predicted else [])
            proto.write(json.dumps({"id": rid, "results": results}) + "\n")
            proto.flush()
        except Exception as exc:  # noqa: BLE001 — report back to parent per request
            proto.write(json.dumps({"id": rid, "error": str(exc)}) + "\n")
            proto.flush()


if __name__ == "__main__":
    main(
        sys.argv[1] if len(sys.argv) > 1 else "en",
        sys.argv[2] if len(sys.argv) > 2 else "",
    )