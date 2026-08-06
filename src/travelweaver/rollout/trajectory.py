"""Persistence helpers for replayable rollout records."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..paths import project_root


def default_trajectory_path(model: str) -> Path:
    safe_model = re.sub(r"[^a-zA-Z0-9_.-]+", "-", model).strip("-") or "model"
    return project_root() / "data" / "trajectories" / f"{safe_model}.jsonl"


def append_trajectory(path: str | Path, record: dict[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return destination
