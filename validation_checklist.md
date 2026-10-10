# Validation Checklist — Logo Post-Filter + Snapshot
- pytest tests/test_pipeline.py -q: PASS
- config class_confidence 0.55: PASS
- config retrieval min_similarity 0.65: PASS
- config max_unknown_per_frame present: PASS
- pipeline uses config defaults: PASS
- resolver density suppression wired: PASS
- evaluate.py present and seeded: PASS
- synthetic suppression probe: PASS

Files ready for commit:
- config/config.yaml
- src/pipeline.py
- src/layer2/brand_resolver.py
- scripts/evaluate.py
