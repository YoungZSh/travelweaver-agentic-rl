"""Internal synthesis records used across composition, witnesses, and polishing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

GENERATOR_VERSION = "travelweaver-synthesis-v1"
PROMPT_VERSION = "travelweaver-zh-polisher-v1"
ARTIFACT_VERSION = "travelweaver-synthesis-artifacts-v1"
WORLD_SNAPSHOT_VERSION = "chinatravel-pinned-v1"


@dataclass(frozen=True)
class PilotSlot:
    index: int
    destination: str
    days: int
    travelers: int
    outbound_mode: str
    return_mode: str
    constraint_count: int
    recipe: tuple[str, ...]

    @property
    def mixed_transport(self) -> bool:
        return self.outbound_mode != self.return_mode


@dataclass(frozen=True)
class CanonicalTask:
    query: str
    clauses: dict[str, str]
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
