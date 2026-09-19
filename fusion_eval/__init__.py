"""Evaluation harness for the compiled multimodal evidence pipeline.

Built BEFORE further algorithmic work so every change is measured:

    dataset -> pipeline -> predictions -> metrics -> ablations
                                            -> calibration
                                            -> error taxonomy
                                            -> report

Currently synthesises ground-truth videos from the reference logo bank so the
harness runs today without a labeled video set; swap `build_dataset` for a
real one when labeled data exists.
"""

from .metrics import compute_metrics
from .reports import build_report

__all__ = ["compute_metrics", "build_report"]