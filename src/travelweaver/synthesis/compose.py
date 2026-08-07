"""Derive grounded typed constraints from a strictly valid witness plan."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from ..errors import SynthesisError
from ..reward import TravelReward
from ..tasks import BlueprintConstraint, ConstraintSpec, TaskBlueprint, TravelTaskSpec, TripSpec
from .models import GENERATOR_VERSION, WORLD_SNAPSHOT_VERSION, PilotSlot, WitnessResult


def compose_blueprint(
    slot: PilotSlot,
    witness: WitnessResult,
    *,
    generation_seed: int,
) -> TaskBlueprint:
    constraints = tuple(
        _constraint(index, key, slot, witness)
        for index, key in enumerate(slot.recipe, 1)
    )
    blueprint = TaskBlueprint(
        trip=TripSpec(
            origin=str(witness.public_task["start_city"]),
            destinations=(slot.destination,),
            days=slot.days,
            travelers=slot.travelers,
        ),
        constraints=constraints,
        world_snapshot_version=WORLD_SNAPSHOT_VERSION,
        generator_version=GENERATOR_VERSION,
        generation_seed=generation_seed,
    )
    _verify_constraints(blueprint, witness)
    return blueprint


def _constraint(
    index: int,
    key: str,
    slot: PilotSlot,
    witness: WitnessResult,
) -> BlueprintConstraint:
    constraint_id = f"c{index:03d}"
    selected = witness.selected
    if key == "total_budget":
        amount = _upper_bound(witness.evidence_bundle["total_cost"], quantum=100)
        return _build(constraint_id, "total_budget", "lte", {"amount": amount}, "trip")
    if key in {"all_intercity_mode", "outbound_mode", "return_mode"}:
        leg, mode = {
            "all_intercity_mode": ("all", slot.outbound_mode),
            "outbound_mode": ("outbound", slot.outbound_mode),
            "return_mode": ("return", slot.return_mode),
        }[key]
        return _build(
            constraint_id,
            "transport_mode",
            "eq",
            {"modes": [mode], "leg": leg},
            "intercity_transport",
        )
    if key == "innercity_mode":
        return _build(
            constraint_id,
            "transport_mode",
            "eq",
            {"modes": [witness.route_mode], "leg": "all"},
            "innercity_route",
        )
    if key in {"outbound_time", "return_time"}:
        leg = "outbound" if key == "outbound_time" else "return"
        field = "end_time" if leg == "outbound" else "start_time"
        transport = selected[leg]
        actual = str(transport["arrival_time" if leg == "outbound" else "departure_time"])
        operator = "lte" if leg == "outbound" else "gte"
        boundary = _natural_boundary(actual, operator)
        return _build(
            constraint_id,
            "time_window",
            operator,
            {"leg": leg, "field": field, "time": boundary},
            "intercity_transport",
        )
    if key == "attraction_category":
        category = _required_text(selected["attractions"][0], "category")
        return _build(
            constraint_id,
            "entity_category",
            "contains",
            {"values": [category]},
            "attraction",
        )
    if key == "include_attraction":
        name = _required_text(selected["attractions"][0], "name")
        return _build(
            constraint_id,
            "include_entity",
            "include",
            {"names": [name]},
            "attraction",
        )
    if key == "attraction_count":
        return _build(
            constraint_id,
            "activity_count",
            "eq",
            {"count": len(selected["attractions"]), "activity_type": "attraction"},
            "attraction",
        )
    if key == "restaurant_cuisine":
        cuisine = _required_text(_required_entity(selected, "restaurant"), "cuisine")
        return _build(
            constraint_id,
            "entity_category",
            "contains",
            {"values": [cuisine]},
            "restaurant",
        )
    if key == "restaurant_budget":
        restaurant = _required_entity(selected, "restaurant")
        amount = _upper_bound(restaurant["price"], quantum=10)
        return _build(
            constraint_id,
            "category_budget",
            "lte",
            {"amount": amount, "basis": "per_person_per_activity"},
            "restaurant",
        )
    if key == "include_restaurant":
        name = _required_text(_required_entity(selected, "restaurant"), "name")
        return _build(
            constraint_id,
            "include_entity",
            "include",
            {"names": [name]},
            "restaurant",
        )
    if key == "hotel_attribute":
        attribute = _required_text(_required_entity(selected, "hotel"), "hotel_type")
        return _build(
            constraint_id,
            "entity_attribute",
            "contains",
            {"values": [attribute]},
            "accommodation",
        )
    if key == "hotel_budget":
        actual = _category_actual(witness, "accommodation")
        return _build(
            constraint_id,
            "category_budget",
            "lte",
            {"amount": _upper_bound(actual, quantum=10), "basis": "per_person_per_night"},
            "accommodation",
        )
    if key == "room_type":
        room_type = int(_required_entity(selected, "hotel")["room_type"])
        return _build(
            constraint_id,
            "room_type",
            "eq",
            {"room_type": room_type},
            "accommodation",
        )
    if key == "room_count":
        room_type = int(_required_entity(selected, "hotel")["room_type"])
        rooms = math.ceil(slot.travelers / room_type)
        return _build(
            constraint_id,
            "room_count",
            "eq",
            {"count": rooms},
            "accommodation",
        )
    if key == "include_hotel":
        name = _required_text(_required_entity(selected, "hotel"), "name")
        return _build(
            constraint_id,
            "include_entity",
            "include",
            {"names": [name]},
            "accommodation",
        )
    raise SynthesisError(f"Unknown synthesis recipe key: {key}")


def _build(
    constraint_id: str,
    kind: str,
    operator: str,
    value: dict[str, Any],
    scope: str,
) -> BlueprintConstraint:
    return BlueprintConstraint(
        id=constraint_id,
        kind=kind,
        operator=operator,
        value=value,
        scope=scope,
    )


def _verify_constraints(blueprint: TaskBlueprint, witness: WitnessResult) -> None:
    spec = TravelTaskSpec(
        task_id=str(witness.public_task["uid"]),
        public_query=str(witness.public_task["query"]),
        trip=blueprint.trip,
        constraints=tuple(
            ConstraintSpec(
                id=item.id,
                kind=item.kind,
                operator=item.operator,
                value=item.value,
                scope=item.scope,
                hardness=item.hardness,
                source_text="grounded blueprint",
            )
            for item in blueprint.constraints
        ),
        unscored_preferences=(),
        source="synthetic_blueprint_verification",
        compiler_version=GENERATOR_VERSION,
        input_hash=blueprint.semantic_hash,
        world_snapshot_version=blueprint.world_snapshot_version,
    )
    result = TravelReward().evaluate(
        spec,
        witness.plan_snapshot,
        witness.evidence_bundle,
    )
    if result.reward != 1.0 or not result.all_hard_pass:
        failed = [
            {"id": check.id, "status": check.status, "evidence": check.evidence}
            for check in result.checks
            if check.status != "pass"
        ]
        raise SynthesisError(f"Derived Blueprint does not pass its witness: {failed}")


def _required_entity(selected: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    entity = selected.get(key)
    if not isinstance(entity, Mapping):
        raise SynthesisError(f"Witness is missing required {key} evidence.")
    return entity


def _required_text(entity: Mapping[str, Any], key: str) -> str:
    value = entity.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SynthesisError(f"Witness entity has no usable {key} value.")
    return value.strip()


def _category_actual(witness: WitnessResult, activity_type: str) -> float:
    items = [
        item
        for item in witness.evidence_bundle["cost_items"]
        if item.get("activity_type") == activity_type
    ]
    amounts = [float(item["amount"]) for item in items]
    nights = max(1, int(witness.public_task["days"]) - 1)
    travelers = int(witness.public_task["people_number"])
    return sum(amounts) / travelers / nights


def _upper_bound(value: Any, *, quantum: int) -> int:
    number = float(value)
    return max(quantum, math.ceil(number * 1.1 / quantum) * quantum)


def _natural_boundary(value: str, operator: str) -> str:
    hours, minutes = (int(part) for part in value.split(":", 1))
    actual = hours * 60 + minutes
    if operator == "lte":
        boundary = math.ceil(actual / 30) * 30
        boundary = min(23 * 60 + 59, boundary)
    else:
        boundary = actual // 30 * 30
    return f"{boundary // 60:02d}:{boundary % 60:02d}"
