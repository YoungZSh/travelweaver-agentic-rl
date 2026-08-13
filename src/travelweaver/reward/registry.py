"""Single-owner registry for built-in outcome predicates."""

from __future__ import annotations

from dataclasses import dataclass

from .models import DIMENSION_ARTIFACT, DIMENSION_GOAL, DIMENSION_VALIDITY


@dataclass(frozen=True)
class CheckDefinition:
    check_id: str
    owner_dimension: str
    predicate: str


CHECK_DEFINITIONS = (
    CheckDefinition("terminal_plan", DIMENSION_ARTIFACT, "an evaluable plan object exists"),
    CheckDefinition(
        "plan_structure", DIMENSION_ARTIFACT, "raw plan fields are locally well formed"
    ),
    CheckDefinition(
        "chronology", DIMENSION_VALIDITY, "activity intervals are ordered and disjoint"
    ),
    CheckDefinition(
        "entity_grounding", DIMENSION_VALIDITY, "activity references have saved evidence"
    ),
    CheckDefinition("candidate_usage", DIMENSION_VALIDITY, "saved candidates are used as declared"),
    CheckDefinition("intercity_time", DIMENSION_VALIDITY, "transport times match saved evidence"),
    CheckDefinition(
        "route_grounding", DIMENSION_VALIDITY, "required routes exist and are feasible"
    ),
    CheckDefinition("opening_hours", DIMENSION_VALIDITY, "activities fit evidence opening hours"),
    CheckDefinition("entity_uniqueness", DIMENSION_VALIDITY, "ordinary entities are not repeated"),
    CheckDefinition("meal_commonsense", DIMENSION_VALIDITY, "meal assignments use valid windows"),
    CheckDefinition(
        "quantity_consistency",
        DIMENSION_VALIDITY,
        "physical quantities match the plan's declared party",
    ),
    CheckDefinition("cost_accounting", DIMENSION_VALIDITY, "all costs can be recomputed"),
    CheckDefinition(
        "overnight_coverage", DIMENSION_VALIDITY, "overnight stays cover itinerary nights"
    ),
    CheckDefinition(
        "task_alignment", DIMENSION_GOAL, "user-visible trip metadata matches the task"
    ),
    CheckDefinition(
        "trip_coverage",
        DIMENSION_GOAL,
        "requested trip content, nights, destination, and boundaries are present",
    ),
)

CHECK_OWNER = {definition.check_id: definition.owner_dimension for definition in CHECK_DEFINITIONS}

if len(CHECK_OWNER) != len(CHECK_DEFINITIONS):
    raise RuntimeError("Built-in Reward check ids must be unique.")


def check_owner(check_id: str) -> str:
    try:
        return CHECK_OWNER[check_id]
    except KeyError as error:
        raise ValueError(f"Built-in Reward check is not registered: {check_id}") from error
