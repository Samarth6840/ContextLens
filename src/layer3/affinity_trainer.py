"""
Layer 3 — Affinity training pathway.

Wires the LightGCN creator-brand affinity model to real historical data so the
learned signal is actually trained (not just constructible). Interactions are
drawn from persisted job history:

  * Primary signal: `layer2d.creator_profile.brand_tallies` — the accumulated
    counts of brands a creator was observed with across their analyzed videos.
    This reflects genuine observed engagement.
  * Fallback: DIRECT recommendations in `layer3.recommendations` (brands with
    on-screen/spoken evidence) when no profile tallies are available.

The resulting model can be attached to a pipeline via `set_affinity_model()`,
or swapped into a server-scoped recommender. Training is seed-deterministic so
two runs over the same history produce identically fitted weights (auditable).
"""

import logging
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


def _job_creator(job: Dict[str, Any]) -> Optional[str]:
    """Best-effort creator id for a stored job snapshot."""
    creator = job.get("creator")
    if isinstance(creator, str) and creator:
        return creator
    profile = (job.get("result") or {}).get("layer2d", {}).get("creator_profile")
    if isinstance(profile, dict) and profile.get("creator_id"):
        return profile["creator_id"]
    return None


def extract_interactions(
    jobs: Sequence[Dict[str, Any]],
) -> List[Tuple[str, str, float]]:
    """Extract (creator_id, brand_id, weight) interactions from job history.

    A job contributes positive interactions for every brand in its creator's
    profile tallies (weight = tally counts, so recurring engagement scores
    higher). Jobs without a usable creator id or with no brand evidence are
    skipped. Deduplicates repeated (creator, brand) pairs by summing weights.
    """
    acc: Dict[Tuple[str, str], float] = {}
    for job in jobs:
        if not isinstance(job, dict):
            continue
        creator = _job_creator(job)
        if not creator:
            continue
        for brand, weight in _job_brand_weights(job):
            key = (creator, brand)
            acc[key] = acc.get(key, 0.0) + float(weight)
    return [(c, b, w) for (c, b), w in sorted(acc.items())]


def _job_brand_weights(job: Dict[str, Any]) -> Iterator[Tuple[str, float]]:
    """Yield (brand, weight) pairs for one job snapshot."""
    result = job.get("result") or {}
    profile = result.get("layer2d", {}).get("creator_profile")
    if isinstance(profile, dict):
        tallies = profile.get("brand_tallies") or {}
        if tallies:
            for brand, n in tallies.items():
                if brand:
                    yield str(brand), float(n)
            return

    recs = result.get("layer3", {}).get("recommendations") or []
    for rec in recs:
        if not isinstance(rec, dict):
            continue
        if rec.get("type") == "DIRECT" and rec.get("brand"):
            yield str(rec["brand"]), 1.0


def train_affinity_from_jobs(
    jobs: Sequence[Dict[str, Any]],
    *,
    embed_dim: int = 64,
    n_layers: int = 3,
    epochs: int = 30,
    lr: float = 1e-3,
    seed: int = 0,
    model_cls=None,
) -> Tuple[Any, Dict[str, Any]]:
    """Fit a CreatorBrandAffinityModel on job history.

    Returns (model, summary). The summary holds `creators`, `brands`, `epochs`,
    `loss`, and `interactions`, plus `status: "trained"` when interactions were
    found (model.is_fitted True) or `"no_interactions"` otherwise (cold start;
    the returned model is unfitted and safe to attach as a no-op).
    """
    from src.layer3.affinity import CreatorBrandAffinityModel

    cls = model_cls or CreatorBrandAffinityModel
    model = cls(embed_dim=embed_dim, n_layers=n_layers, lr=lr, device="cpu")
    model.device = "cpu"

    interactions = extract_interactions(jobs)
    if not interactions:
        logger.info("Affinity training: no creator-brand interactions in history.")
        return model, {
            "status": "no_interactions",
            "interactions": 0,
            "creators": 0,
            "brands": 0,
            "epochs": 0,
            "loss": 0.0,
        }

    ids = [c for c, _b, _w in interactions]
    brand_ids = [b for _c, b, _w in interactions]
    weights = [w for _c, _b, w in interactions]
    summary = model.fit(ids, brand_ids, weights=weights, epochs=epochs, seed=seed)
    summary["status"] = "trained"
    summary["interactions"] = len(interactions)
    return model, summary


def train_affinity_from_store(store, **kwargs) -> Tuple[Any, Dict[str, Any]]:
    """Convenience wrapper: fit affinity from a JobStore's full history.

    `store` must expose `list()`. Returns (model, summary) as documented in
    `train_affinity_from_jobs`.
    """
    jobs = store.list()
    return train_affinity_from_jobs(jobs, **kwargs)
