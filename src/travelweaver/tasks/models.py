"""Versioned, source-independent travel task contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from ..errors import TaskSpecError

SPEC_VERSION = "travelweaver-task-spec-v1"

_CONSTRAINT_KINDS = frozenset(
    {
        "activity_count",
        "category_budget",
        "entity_attribute",
        "entity_category",
        "exclude_entity",
        "include_entity",
        "room_count",
        "room_type",
        "time_window",
        "total_budget",
        "transport_mode",
    }
)
_OPERATORS = frozenset(
    {"contains", "eq", "exclude", "gte", "include", "lte", "not_contains", "not_in"}
)
_SCOPES = frozenset(
    {
        "accommodation",
        "attraction",
        "day",
        "innercity_route",
        "intercity_transport",
        "itinerary",
        "restaurant",
        "trip",
    }
)


def supported_constraint_kinds() -> tuple[str, ...]:
    return tuple(sorted(_CONSTRAINT_KINDS))


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def stable_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TripSpec:
    origin: str
    destinations: tuple[str, ...]
    days: int
    travelers: int
    start_date: str | None = None

    def __post_init__(self) -> None:
        if not self.origin.strip() or not self.destinations:
            raise TaskSpecError("Trip origin and at least one destination are required.")
        if any(not destination.strip() for destination in self.destinations):
            raise TaskSpecError("Trip destinations must be non-empty strings.")
        if self.days <= 0 or self.travelers <= 0:
            raise TaskSpecError("Trip days and travelers must be positive integers.")


@dataclass(frozen=True)
class ConstraintSpec:
    id: str
    kind: str
    operator: str
    value: Any
    scope: str
    hardness: str
    source_text: str
    source_start: int | None = None
    source_end: int | None = None

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise TaskSpecError("Constraint id cannot be empty.")
        if self.kind not in _CONSTRAINT_KINDS:
            raise TaskSpecError(f"Unsupported constraint kind: {self.kind}")
        if self.operator not in _OPERATORS:
            raise TaskSpecError(f"Unsupported constraint operator: {self.operator}")
        if self.scope not in _SCOPES:
            raise TaskSpecError(f"Unsupported constraint scope: {self.scope}")
        if self.hardness not in {"hard", "soft"}:
            raise TaskSpecError("Constraint hardness must be hard or soft.")
        if not self.source_text.strip():
            raise TaskSpecError("Every constraint must retain non-empty source text.")
        if (self.source_start is None) != (self.source_end is None):
            raise TaskSpecError("Constraint source offsets must be both present or both absent.")
        if self.source_start is not None and (
            self.source_start < 0 or self.source_end is None or self.source_end <= self.source_start
        ):
            raise TaskSpecError("Constraint source offsets are invalid.")


@dataclass(frozen=True)
class TravelTaskSpec:
    task_id: str
    public_query: str
    trip: TripSpec
    constraints: tuple[ConstraintSpec, ...]
    unscored_preferences: tuple[str, ...]
    source: str
    compiler_version: str
    input_hash: str
    world_snapshot_version: str
    spec_version: str = SPEC_VERSION

    def __post_init__(self) -> None:
        if self.spec_version != SPEC_VERSION:
            raise TaskSpecError(f"Unsupported TaskSpec version: {self.spec_version}")
        if not self.task_id.strip() or not self.public_query.strip():
            raise TaskSpecError("Task id and public query are required.")
        ids = [constraint.id for constraint in self.constraints]
        if len(ids) != len(set(ids)):
            raise TaskSpecError("Constraint ids must be unique within a task.")
        if not self.source.strip() or not self.compiler_version.strip():
            raise TaskSpecError("Task source and compiler version are required.")
        if not self.input_hash.strip() or not self.world_snapshot_version.strip():
            raise TaskSpecError("Task input hash and world snapshot version are required.")
        for constraint in self.constraints:
            if constraint.source_start is None:
                continue
            assert constraint.source_end is not None
            if self.public_query[constraint.source_start : constraint.source_end] != (
                constraint.source_text
            ):
                raise TaskSpecError(
                    f"Constraint {constraint.id} source offsets do not match public query."
                )

    @property
    def spec_hash(self) -> str:
        payload = asdict(self)
        return stable_hash(payload)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["spec_hash"] = self.spec_hash
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TravelTaskSpec:
        trip_payload = dict(payload["trip"])
        trip_payload["destinations"] = tuple(trip_payload["destinations"])
        constraints = tuple(
            ConstraintSpec(**dict(constraint)) for constraint in payload.get("constraints", [])
        )
        spec = cls(
            task_id=str(payload["task_id"]),
            public_query=str(payload["public_query"]),
            trip=TripSpec(**trip_payload),
            constraints=constraints,
            unscored_preferences=tuple(payload.get("unscored_preferences", [])),
            source=str(payload["source"]),
            compiler_version=str(payload["compiler_version"]),
            input_hash=str(payload["input_hash"]),
            world_snapshot_version=str(payload["world_snapshot_version"]),
            spec_version=str(payload.get("spec_version", SPEC_VERSION)),
        )
        supplied_hash = payload.get("spec_hash")
        if supplied_hash is not None and supplied_hash != spec.spec_hash:
            raise TaskSpecError("TaskSpec hash does not match its contents.")
        return spec
