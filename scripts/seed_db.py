"""Seed the ContextLens SQLite job store with realistic demo data.

Useful for developing the SaaS dashboard, insights API, personalized outreach,
and the LightGCN affinity training pathway without running full video
inference. Every "analysis" is a synthetic but internally-consistent snapshot
matching the shape the real pipeline + `prune_for_store` persist: a creator
profile with brand tallies, a layer3 recommendation list with DIRECT/SUGGESTED
types, and a small dashboard.

Usage:
    python scripts/seed_db.py                 # seed into the configured DB
    python scripts/seed_db.py --clear         # wipe and re-seed
    python scripts/seed_db.py --train-affinity  # also train+attach affinity (needs torch)
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.brand_catalog import BRAND_CATALOG
from src.store import JobStore, _default_db_path, _now_iso

# (creator, handle, followers, engagements, [(brand, tally), ...])
# Brands must be real BRAND_CATALOG keys.
DEMO_CREATORS = [
    ("alice_fit", "alice.fitness", 1_400_000, 220_000, [
        ("NIKE", 4), ("ADIDAS", 3), ("REEBOK", 2), ("UNDER ARMOUR", 1),
    ]),
    ("bob_tech", "bob.techtuber", 820_000, 130_000, [
        ("APPLE", 5), ("SONY", 3), ("SAMSUNG", 2), ("GOOGLE", 2),
    ]),
    ("cara_beauty", "cara.glow", 2_100_000, 310_000, [
        ("GUCCI", 4), ("ROLEX", 3), ("LULULEMON", 2),
    ]),
    ("dave_gaming", "dave.plays", 560_000, 95_000, [
        ("SONY", 3), ("MICROSOFT", 2), ("RED BULL", 2),
    ]),
]


def _creator_profile(creator_id, handle, followers, engagements, tallies):
    categories = {}
    for brand, _n in tallies:
        cats = BRAND_CATALOG[brand].get("categories") or []
        for c in cats:
            categories[c] = categories.get(c, 0) + 1
    return {
        "creator_id": creator_id,
        "handle": handle,
        "followers": followers,
        "engagements": engagements,
        "engagement_rate": round(engagements / followers, 4) if followers else 0.0,
        "categories": categories,
        "dominant_category": max(categories, key=categories.get) if categories else None,
        "brand_tallies": dict(tallies),
        "videos_analyzed": 1,
        "production_quality": 0.78,
    }


def _recommendations(creator_id, tallies):
    recs = []
    for rank, (brand, tally) in enumerate(tallies):
        info = BRAND_CATALOG[brand]
        cats = info.get("categories") or [info.get("category", "GENERAL")]
        recs.append({
            "brand": brand,
            "product": info.get("product", brand),
            "category": cats[0],
            "type": "DIRECT",
            "score": round(max(0.3, min(1.0, 0.6 + 0.05 * tally - 0.02 * rank)), 3),
            "confidence": round(0.7 + 0.05 * rank if rank else 0.9, 3),
            "appearances": tally,
            "reasons": ["LOGO / ON-SCREEN DETECTED", "STRONG EVIDENCE"],
        })
    return recs


def build_job(job_id, title, video_path, creator_id, handle, followers,
              engagements, tallies):
    now = _now_iso()
    profile = _creator_profile(creator_id, handle, followers, engagements, tallies)
    directed = [b for b, _n in tallies]
    recs = _recommendations(creator_id, tallies)
    return {
        "job_id": job_id,
        "status": "done",
        "stage": "COMPLETE",
        "filename": video_path,
        "title": title,
        "creator": creator_id,
        "created_at": now,
        "finished_at": now,
        "dashboard": {
            "creator": handle,
            "confidence": round(0.5 + 0.4 * (len(directed) / 6), 3),
            "title": title,
            "brands": directed,
        },
        "result": {
            "video_path": video_path,
            "num_frames": 30,
            "video_total_frames": 60,
            "video_fps": 1.0,
            "layer2b_confidence": 0.85,
            "layer2d": {"creator_profile": profile},
            "layer2c": {
                "brand_timeline": {b: {"brand": b, "appearance_count": n,
                                       "modalities": ["logo"]}
                                   for b, n in tallies},
                "memory_size": len(tallies),
                "memory_brands": [b for b, _n in tallies],
                "indirect_resolutions": [],
            },
            "layer3": {"recommendations": recs},
        },
    }


def _resolve_db_path():
    import os

    custom = os.environ.get("ADSCENE_DB_PATH")
    if custom:
        return custom
    try:
        import server
        return server._DB_PATH
    except Exception:  # noqa: BLE001 - fall back to default
        return _default_db_path()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clear", action="store_true", help="wipe existing rows first")
    ap.add_argument("--db", default=None, help="explicit db path (default: server default)")
    ap.add_argument("--train-affinity", action="store_true",
                    help="after seeding, train+attach a LightGCN affinity model")
    args = ap.parse_args()

    db_path = args.db or _resolve_db_path()
    store = JobStore(db_path)
    print(f"[seed] store: {store.db_path}")

    if args.clear:
        ids = [job["job_id"] for job in store.list()]
        for jid in ids:
            store.delete(jid)
        print(f"[seed] cleared {len(ids)} existing job(s)")

    existing = {job["job_id"] for job in store.list()}
    created = 0
    for i, (creator, handle, followers, eng, tallies) in enumerate(DEMO_CREATORS):
        for n in range(1, 3):
            job_id = f"DEMO-{creator.upper()}-{n}"
            if job_id in existing:
                continue
            title = f"{handle} — demo video {n}"
            path = f"videos/{creator}_{n}.mp4"
            job = build_job(job_id, title, path, creator, handle,
                            followers, eng, tallies)
            store.save(job_id, job)
            created += 1
    print(f"[seed] wrote {created} new demo job(s); total {store.count()}")

    if args.train_affinity:
        try:
            from src.layer3.affinity_trainer import train_affinity_from_jobs
            model, summary = train_affinity_from_jobs(
                store.list(), embed_dim=16, n_layers=2, lr=1e-2,
                epochs=20, seed=0,
            )
            print(f"[seed] affinity training: {summary.get('status')} "
                  f"(creators={summary.get('creators')}, "
                  f"brands={summary.get('brands')}, "
                  f"interactions={summary.get('interactions')})")
            for creator in ("alice_fit", "bob_tech"):
                scores = model.predict_affinity(creator, ["NIKE", "APPLE", "SONY"])
                if scores is not None:
                    print(f"[seed] affinity({creator}) = "
                          f"{dict(zip(['NIKE', 'APPLE', 'SONY'], [round(s, 3) for s in scores]))}")
        except Exception as exc:  # noqa: BLE001
            print(f"[seed] affinity training unavailable: {exc}")

    store.close()


if __name__ == "__main__":
    main()
