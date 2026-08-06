"""Public data structures returned by TravelWeaverEnv."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Observation:
    schema_version: str
    environment_version: str
    episode_id: str
    task: dict[str, Any]
    tool_result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    visible_entity_ids: tuple[str, ...] = field(default_factory=tuple)
    visible_route_ids: tuple[str, ...] = field(default_factory=tuple)
    candidates: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    remaining_steps: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StepResult:
    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


@dataclass(frozen=True)
class PlanSnapshot:
    """Environment-owned normalized representation of a submitted plan."""

    schema_version: str
    task_id: str
    people_number: int
    start_city: str
    target_city: str
    days: int
    activities: tuple[dict[str, Any], ...]
    total_cost: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceBundle:
    """Replayable environment evidence used by deterministic reward."""

    schema_version: str
    environment_version: str
    task_id: str
    entities: dict[str, dict[str, Any]]
    routes: dict[str, dict[str, Any]]
    cost_items: tuple[dict[str, Any], ...]
    total_cost: float | None
    quantity_rules_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
