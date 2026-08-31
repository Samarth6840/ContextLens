"""
Layer 2c — Cross-scene / cross-video brand entity memory (Phase 2).

TemporalObjectSmoother and SceneConsistencyResolver stabilise labels WITHIN a
single video. BrandMemoryBank adds the second axis of temporal reasoning the
audit asked for: a persistent in-memory entity store of every brand the creator
has featured, which is:

  * accumulated across scenes AND videos (not reset per video), and
  * used to resolve INDIRECT references — ASR anaphora like "this phone" or
    "our foldable" that contain no brand token and therefore never surface in
    the direct brand-mention matcher.

Recall semantics are intentionally recency-biased: a brand that was visually
established seconds ago (high confidence) wins over something mentioned an hour
ago, but a highly confident long-standing memory still competes with weak recent
noise. If an optional visual embedding is supplied for the reference, it is
compared against stored embeddings (cosine similarity) and blended with the
recently/confidence prior.

The memory is JSON-serializable (embeddings kept as lists) so the same bank can
be persisted across processing runs / server restarts and reloaded for the next
creator video.
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    import math

    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (na * nb)


class BrandMemoryBank:
    """Persistent, recency-weighted brand entity memory (Layer 2c).

    Each stored entity aggregates all sightings of a brand:

        {
            "brand": "APPLE",
            "video_id": ...,
            "last_frame": ...,
            "last_timestamp": seconds,
            "first_seen": unix time,
            "last_seen": unix time,
            "max_confidence": ...,
            "best_confidence": recent weighted score,
            "sightings": int,
            "modalities": {modality: count},
            "embedding": List[float] | None,
            "product": Optional[str],
        }
    """

    def __init__(self, recency_decay: float = 0.5):
        self.recency_decay = recency_decay
        self._entities: Dict[str, Dict[str, Any]] = {}
        self._recency_clock = 0.0

    # ------------------------------------------------------------------ #
    # Recording
    # ------------------------------------------------------------------ #
    def record(
        self,
        brand: str,
        video_id: str = "",
        frame: int = 0,
        timestamp: float = 0.0,
        confidence: float = 1.0,
        modality: str = "visual",
        embedding: Optional[Sequence[float]] = None,
        product: Optional[str] = None,
        wall_time: Optional[float] = None,
    ) -> "BrandMemoryBank":
        """Record one sighting of a brand into the memory bank."""
        name = (brand or "").strip().upper()
        if not name:
            return self

        self._clock(wall_time)
        now = wall_time if wall_time is not None else self._recency_clock

        ent = self._entities.get(name)
        if ent is None:
            ent = {
                "brand": name,
                "video_id": video_id,
                "last_frame": frame,
                "last_timestamp": timestamp,
                "first_seen": now,
                "last_seen": now,
                "max_confidence": confidence,
                "sightings": 1,
                "modalities": {modality: 1},
                "embedding": None,
                "product": product,
            }
            if embedding is not None:
                ent["embedding"] = list(embedding)
            self._entities[name] = ent
            return self

        ent["video_id"] = video_id or ent["video_id"]
        ent["last_frame"] = max(ent["last_frame"], frame)
        ent["last_timestamp"] = max(ent["last_timestamp"], timestamp)
        ent["last_seen"] = now
        ent["max_confidence"] = max(ent["max_confidence"], confidence)
        ent["sightings"] += 1
        ent["modalities"][modality] = ent["modalities"].get(modality, 0) + 1
        if embedding is not None:
            ent["embedding"] = list(embedding)
        if product and not ent.get("product"):
            ent["product"] = product
        return self

    def _clock(self, wall_time: Optional[float]) -> None:
        if wall_time is None:
            self._recency_clock += 1.0
        else:
            self._recency_clock = wall_time

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    def brands(self) -> List[str]:
        return sorted(self._entities)

    def size(self) -> int:
        return len(self._entities)

    def get(self, brand: str) -> Optional[Dict[str, Any]]:
        return self._entities.get((brand or "").strip().upper())

    def clear(self) -> None:
        self._entities.clear()
        self._recency_clock = 0.0

    def _score(self, ent: Dict[str, Any], ref_clock: float) -> float:
        """Recency-decayed confidence: older sightings contribute less."""
        age = max(0.0, ref_clock - ent["last_seen"])
        return float(ent["max_confidence"]) * (self.recency_decay ** age)

    def _ranked(self, ref_clock: float) -> List[Dict[str, Any]]:
        scored = [(self._score(e, ref_clock), e) for e in self._entities.values()]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored]

    def most_recent_brand(self) -> Optional[str]:
        if not self._entities:
            return None
        best = max(self._entities.values(), key=lambda e: e["last_seen"])
        return best["brand"]

    def best_brand(self) -> Optional[str]:
        """Highest recency-weighted confidence entity across all memory."""
        if not self._entities:
            return None
        rank = self._ranked(float("inf") if not self._entities else
                            max(e["last_seen"] for e in self._entities.values()))
        return rank[0]["brand"]

    def resolve_reference(
        self,
        reference: str,
        embedding: Optional[Sequence[float]] = None,
        ref_clock: Optional[float] = None,
        returned_products: Sequence[str] = (),
        sim_weight: float = 0.5,
        top_k: int = 3,
    ) -> Optional[Dict[str, Any]]:
        """Resolve an indirect reference to a remembered brand entity.

        `reference` is text (e.g. an ASR mention like "this phone") that did not
        directly match a brand. Returns the best matching stored entity when
        memory has a plausible candidate, else None (no resolution).

        Scoring blends:
          * recency-decayed confidence prior (entirely lexical/embedding-free),
          * optional visual-embedding cosine similarity (when `embedding` given),
          * exclusion of brands already returned (avoid re-recommending a brand
            the outreach pass already returned in this context).

        The returned dict always includes the decision's explanation as
        "reason" so the system is auditable (no silent inference).
        """
        if not reference or not self._entities:
            return None

        ref_embed = list(embedding) if embedding is not None else None
        clock = ref_clock if ref_clock is not None else self._recency_clock
        excluded = {b.strip().upper() for b in returned_products}

        ranked = self._ranked(clock)
        if not ranked:
            return None

        # Start with the best recency-weighted candidate, then optionally
        # re-rank by embedding similarity when available.
        cands: List[Dict[str, Any]] = []
        for ent in ranked:
            score = self._score(ent, clock)
            if ref_embed is not None and ent.get("embedding"):
                sim = _cosine(ref_embed, ent["embedding"])
                score = (1.0 - sim_weight) * score + sim_weight * sim
            cands.append((score, ent))
        cands.sort(key=lambda x: x[0], reverse=True)

        for score, ent in cands[:top_k]:
            if ent["brand"] in excluded:
                continue
            if score <= 0:
                continue
            decision = dict(ent)
            decision["match_score"] = round(float(score), 4)
            decision["reference"] = reference
            decision["reason"] = (
                f"indirect reference '{reference}' resolved to {ent['brand']} "
                f"(recency-decayed confidence {round(score, 3)}"
                + (", embedding sim" if ref_embed is not None else "")
                + ")"
            )
            return decision
        return None

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return {
            "recency_decay": self.recency_decay,
            "entities": list(self._entities.values()),
            "clock": self._recency_clock,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BrandMemoryBank":
        bank = cls(recency_decay=data.get("recency_decay", 0.5))
        bank._recency_clock = data.get("clock", 0.0)
        for ent in data.get("entities", []):
            name = ent["brand"].strip().upper()
            bank._entities[name] = ent
        return bank

    def save(self, path: str) -> str:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "BrandMemoryBank":
        with open(path) as fh:
            return cls.from_dict(json.load(fh))


def load_or_new(path: Optional[str]) -> BrandMemoryBank:
    """Load an existing bank from disk if present, else a fresh empty bank."""
    if path and os.path.exists(path):
        try:
            return BrandMemoryBank.load(path)
        except (OSError, ValueError) as exc:
            logger.warning("Could not load brand memory %s: %s", path, exc)
    return BrandMemoryBank()
