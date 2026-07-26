"""
Test fixtures module.

Synthetic test data for unit tests.
Strictly separated from the real training/eval/inference data path.
The production data root is ./data (config.yaml); test fixtures are in ./tests/fixtures.
They can never be accidentally served by the real pipeline.
"""