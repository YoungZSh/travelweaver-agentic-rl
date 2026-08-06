"""Resolve stored task records into frozen TravelTaskSpec instances."""

from __future__ import annotations

from typing import Any

from ..errors import TaskSpecError
from .chinatravel import ChinaTravelOracleAdapter
from .compiler import build_base_spec
from .models import TravelTaskSpec


class TaskSpecResolver:
    """Dispatch source-specific records before the source-independent reward boundary."""

    def __init__(self, *, world_snapshot_version: str = "chinatravel-pinned-v1") -> None:
        self.world_snapshot_version = world_snapshot_version
        self.chinatravel = ChinaTravelOracleAdapter(
            world_snapshot_version=world_snapshot_version
        )

    def resolve(
        self, public_task: dict[str, Any], oracle: dict[str, Any] | None = None
    ) -> TravelTaskSpec:
        oracle = oracle or {}
        serialized = oracle.get("task_spec")
        if isinstance(serialized, dict):
            return TravelTaskSpec.from_dict(serialized)
        if "hard_logic" in oracle:
            result = self.chinatravel.compile(public_task, oracle)
            if not result.accepted or result.spec is None:
                detail = "; ".join(result.errors[:3]) or "unknown adapter failure"
                raise TaskSpecError(f"Task was quarantined: {detail}")
            return result.spec
        return build_base_spec(
            public_task,
            source="public_task",
            world_snapshot_version=self.world_snapshot_version,
        )
