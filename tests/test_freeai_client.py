"""Free.ai client: response parsing and base_url wiring.

Two bugs, both silent:
  * `ocr()` appended the per-line `results` AND the top-level summary `text`,
    duplicating every recognized string. Lines shown twice inflate any count
    derived from OCR output.
  * the client hardcoded the shipped URL, so `layer1.freeai.base_url` in config
    (and FREE_AI_BASE_URL) did nothing.
"""

import numpy as np

from src.layer1.freeai_client import FreeAIClient


class _Resp:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


class _Session:
    def __init__(self, payload):
        self.payload = payload
        self.posts = []
        self.headers = {}

    def post(self, url, **kwargs):
        self.posts.append(url)
        return _Resp(self.payload)


def _client(payload, base_url="https://example.test/v9"):
    c = FreeAIClient(api_key="k", base_url=base_url)
    c.session = _Session(payload)
    return c


def test_results_preferred_over_summary_no_duplicates():
    c = _client({"text": "AAA\nBBB", "results": [
        {"text": "AAA", "confidence": 0.8},
        {"text": "BBB", "confidence": 0.7},
    ]})
    out = c.ocr(np.zeros((4, 4, 3), np.uint8))
    assert [r["text"] for r in out] == ["AAA", "BBB"], out


def test_summary_used_when_no_results():
    out = _client({"text": "SOLO"}).ocr(np.zeros((4, 4, 3), np.uint8))
    assert [r["text"] for r in out] == ["SOLO"], out


def test_configured_base_url_is_used():
    c = _client({"text": "x"})
    c.ocr(np.zeros((4, 4, 3), np.uint8))
    assert c.session.posts == ["https://example.test/v9/ocr"], c.session.posts
