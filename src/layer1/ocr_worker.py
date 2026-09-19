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
import sys

import cv2
import numpy as np


def _init_ocr(lang: str):
    import paddle  # noqa: F401  (must load before PaddleOCR; default CPU thread pool)

    from paddleocr import PaddleOCR as _PaddleOCR

    return _PaddleOCR(
        use_angle_cls=True,
        lang=lang,
        det_db_thresh=0.3,
        text_recognition_batch_size=6,
    )


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


def main(lang: str = "en") -> None:
    ocr = _init_ocr(lang)
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
            sys.stdout.write(json.dumps({"id": rid, "results": results}) + "\n")
            sys.stdout.flush()
        except Exception as exc:  # noqa: BLE001 — report back to parent per request
            sys.stdout.write(json.dumps({"id": rid, "error": str(exc)}) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "en")