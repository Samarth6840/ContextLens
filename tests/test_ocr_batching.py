"""OCR worker requests stay bounded and preserve input alignment."""

import numpy as np

from src.layer1.ocr import OCRExtractor


def test_extract_text_batch_splits_large_requests_and_keeps_order():
    extractor = OCRExtractor.__new__(OCRExtractor)
    batch_sizes = []

    def fake_worker_exec(frames):
        batch_sizes.append(len(frames))
        return [[{"text": str(int(frame[0, 0, 0]))}] for frame in frames]

    extractor._worker_exec = fake_worker_exec
    frames = [np.full((2, 2, 3), i, dtype=np.uint8) for i in range(19)]

    result = extractor.extract_text_batch(frames)

    assert batch_sizes == [8, 8, 3]
    assert [row[0]["text"] for row in result] == [str(i) for i in range(19)]
