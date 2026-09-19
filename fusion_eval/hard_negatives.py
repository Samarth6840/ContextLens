"""Hard negatives: candidate pairs that must not be confused.

Near-tie pairs (same visual style / similar products) are the entries where a
learned discriminator earns its keep; the conflict synthesizer in dataset.py
composes these as two-brand videos expecting an ambiguous/abstain verdict.
"""

from __future__ import annotations

from typing import List, Tuple

# Pairs chosen from the benchmark reference bank where available; entries whose
# brands are absent from the bank are skipped at build time.
HARD_PAIRS: List[Tuple[str, str]] = [
    ("SAMSUNG", "APPLE"),
    ("ADIDAS", "PUMA"),
    ("NIKE", "ADIDAS"),
    ("MICROSOFT", "GOOGLE"),
    ("COCA-COLA", "PEPSI"),
    ("SONY", "SAMSUNG"),
    ("ROLEX", "SONY"),
    ("ZARA", "UNDER ARMOUR"),
]


def available_pairs(bank_root: str) -> List[Tuple[str, str]]:
    import os

    have = {
        b for b in os.listdir(bank_root)
        if os.path.isdir(os.path.join(bank_root, b))
    }
    return [(a, b) for a, b in HARD_PAIRS if a in have and b in have]