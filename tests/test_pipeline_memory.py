from src.layer2.brand_memory import BrandMemoryBank
from src.pipeline import Phase1Pipeline


def _minimal_pipeline():
    pipe = Phase1Pipeline.__new__(Phase1Pipeline)
    pipe._brand_memory = BrandMemoryBank(recency_decay=1.0)
    return pipe


def test_add_brand_resolutions_records_timeline():
    pipe = _minimal_pipeline()
    timeline = {
        "APPLE": {"last_frame": 40, "last_timestamp": 12.5,
                  "max_confidence": 0.9, "product": "iphone"},
        "NIKE": {"last_frame": 20, "last_timestamp": 6.0,
                 "max_confidence": 0.7},
    }
    pipe.add_brand_resolutions(timeline, video_id="v1.mp4")
    assert pipe._brand_memory.size() == 2
    assert pipe._brand_memory.get("APPLE")["product"] == "iphone"
    assert pipe._brand_memory.get("NIKE")["video_id"] == "v1.mp4"


def test_resolve_indirect_mention_uses_memory():
    pipe = _minimal_pipeline()
    pipe.add_brand_resolutions(
        {"APPLE": {"last_frame": 0, "last_timestamp": 0.0,
                   "max_confidence": 0.9}},
        video_id="v1.mp4",
    )
    resolutions = pipe._resolve_indirect_mentions([], "This phone is incredible.")
    assert len(resolutions) == 1
    assert resolutions[0]["brand"] == "APPLE"


def test_direct_brand_sentence_skipped():
    pipe = _minimal_pipeline()
    pipe.add_brand_resolutions(
        {"APPLE": {"last_frame": 0, "last_timestamp": 0.0,
                   "max_confidence": 0.9}},
        video_id="v1.mp4",
    )
    # The sentence names APPLE directly -> already handled by the named matcher,
    # so the indirect pass must NOT double-resolve it.
    mentions = [{"brand": "APPLE"}]
    assert pipe._resolve_indirect_mentions(mentions, "This Apple phone is great.") == []


def test_no_resolution_without_device_language():
    pipe = _minimal_pipeline()
    pipe.add_brand_resolutions(
        {"APPLE": {"last_frame": 0, "last_timestamp": 0.0,
                   "max_confidence": 0.9}},
        video_id="v1.mp4",
    )
    assert pipe._resolve_indirect_mentions([], "The weather is nice today.") == []
