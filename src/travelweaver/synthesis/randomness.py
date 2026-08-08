"""Stable random streams derived from one user-facing synthesis seed."""

from __future__ import annotations

import hashlib
import random


def deterministic_rng(seed: int, scope: str, index: int | None = None) -> random.Random:
    """Return an isolated deterministic stream without exposing additional seed knobs."""

    material = f"travelweaver-seed-v1:{seed}:{scope}:{index}".encode()
    derived = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return random.Random(derived)
