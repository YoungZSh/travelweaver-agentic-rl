"""Internal synthesis records used across composition, witnesses, and polishing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

GENERATOR_VERSION = "travelweaver-synthesis-v4"
PROMPT_VERSION = "travelweaver-zh-polisher-v6"
ARTIFACT_VERSION = "travelweaver-synthesis-artifacts-v6"
WORLD_SNAPSHOT_VERSION = "chinatravel-pinned-v1"


@dataclass(frozen=True)
class PilotSlot:
    index: int
    origin: str
    destination: str
    days: int
    travelers: int
    outbound_mode: str
    return_mode: str
    constraint_count: int
    recipe: tuple[str, ...]
    attractions_per_day: int
    include_meal: bool
    route_mode: str
    transport_strategy: str
    tightness: str
    scenario_profile: str
    surface_style: str
    synthesis_profile: str = "pilot_v2_1"
    task_type: str = "pilot"
    validation_profile: str = "strict"
    persona_context: str | None = None
    metadata_prefix: str | None = None
    preference_kinds: tuple[str, ...] = ()

    @property
    def mixed_transport(self) -> bool:
        return self.outbound_mode != self.return_mode


@dataclass(frozen=True)
class CanonicalTask:
    query: str
    clauses: dict[str, str]
    preference_clauses: dict[str, str]
    protected_literals: tuple[str, ...]


@dataclass(frozen=True)
class WitnessResult:
    public_task: dict[str, Any]
    plan: dict[str, Any]
    plan_snapshot: dict[str, Any]
    evidence_bundle: dict[str, Any]
    reward_detail: dict[str, Any]
    selected: dict[str, Any]
    route_mode: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "public_task": self.public_task,
            "plan": self.plan,
            "plan_snapshot": self.plan_snapshot,
            "evidence_bundle": self.evidence_bundle,
            "reward_detail": self.reward_detail,
            "selected": self.selected,
            "route_mode": self.route_mode,
        }
