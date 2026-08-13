"""Derive grounded typed constraints from a strictly valid witness plan."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from ..errors import SynthesisError
from ..reward import TravelReward
from ..tasks import (
    BlueprintConstraint,
    BlueprintPreference,
    ConstraintSpec,
    TaskBlueprint,
    TravelTaskSpec,
    TripSpec,
)
from .models import GENERATOR_VERSION, WORLD_SNAPSHOT_VERSION, PilotSlot, WitnessResult


def compose_blueprint(
    slot: PilotSlot,
    witness: WitnessResult,
    *,
    generation_seed: int,
    world_snapshot_version: str = WORLD_SNAPSHOT_VERSION,
    constraint_witnesses: Sequence[WitnessResult] = (),
) -> TaskBlueprint:
    cohort = tuple(constraint_witnesses) or (witness,)
    constraints = tuple(
        _constraint(index, key, slot, witness, cohort)
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
        world_snapshot_version=world_snapshot_version,
        generator_version=GENERATOR_VERSION,
        generation_seed=generation_seed,
        preferences=_preferences(slot, witness),
        persona_context=slot.persona_context,
        metadata_prefix=slot.metadata_prefix,
    )
    _verify_constraints(blueprint, witness)
    return blueprint


_PREFERENCE_DIRECTIONS = {
    "more_attractions": "maximize",
    "less_innercity_time": "minimize",
    "shorter_meal_transfer": "minimize",
    "higher_dining_share": "maximize",
    "lower_lodging_share": "minimize",
    "near_poi": "minimize",
    "less_walking": "minimize",
    "lower_total_cost": "minimize",
    "relaxed_itinerary": "minimize",
    "higher_attraction_share": "maximize",
    "lower_intercity_share": "minimize",
    "shorter_total_travel_time": "minimize",
}


def _preferences(
    slot: PilotSlot, witness: WitnessResult
) -> tuple[BlueprintPreference, ...]:
    preferences: list[BlueprintPreference] = []
    for index, kind in enumerate(slot.preference_kinds, 1):
        value: Any = None
        if kind == "near_poi":
            anchor = witness.selected.get("preference_anchor")
            if not isinstance(anchor, Mapping):
                attractions = witness.selected.get("attractions", [])
                anchor = attractions[0] if attractions else None
            if not isinstance(anchor, Mapping):
                raise SynthesisError("near_poi preference needs an attraction anchor.")
            value = {"poi_name": _required_text(anchor, "name")}
        preferences.append(
            BlueprintPreference(
                id=f"p{index:03d}",
                kind=kind,
                direction=_PREFERENCE_DIRECTIONS[kind],
                value=value,
            )
        )
    return tuple(preferences)


def _constraint(
    index: int,
    key: str,
    slot: PilotSlot,
    witness: WitnessResult,
    cohort: Sequence[WitnessResult],
) -> BlueprintConstraint:
    constraint_id = f"c{index:03d}"
    selected = witness.selected
    if key == "total_budget":
        amount = _upper_bound(
            max(float(item.evidence_bundle["total_cost"]) for item in cohort),
            quantum=100,
            factor=_budget_factor(slot.tightness),
        )
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
        operator = "lte" if leg == "outbound" else "gte"
        field_name = "arrival_time" if leg == "outbound" else "departure_time"
        actual_values = [str(item.selected[leg][field_name]) for item in cohort]
        actual = (
            max(actual_values, key=_clock_value)
            if operator == "lte"
            else min(actual_values, key=_clock_value)
        )
        boundary = _natural_boundary(
            actual,
            operator,
            slack_minutes=_time_slack(slot.tightness),
        )
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
    if key == "attraction_categories_all":
        categories = list(
            dict.fromkeys(
                _required_text(item, "category")
                for item in selected["attractions"]
            )
        )
        if len(categories) < 2:
            raise SynthesisError(
                "Multi-category attraction constraint needs two witness categories."
            )
        return _build(
            constraint_id,
            "entity_category",
            "contains",
            {"values": categories[:2]},
            "attraction",
        )
    if key == "attraction_categories_any":
        actual = _required_text(selected["attractions"][0], "category")
        alternative = _required_logic_text(witness, "alternative_attraction_category")
        if alternative == actual:
            raise SynthesisError("Attraction category alternatives must differ.")
        return _build(
            constraint_id,
            "entity_category",
            "contains",
            {"any_of": [[actual], [alternative]]},
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
    if key == "exclude_attraction":
        name = _required_logic_text(witness, "excluded_attraction_name")
        return _build(
            constraint_id,
            "exclude_entity",
            "exclude",
            {"names": [name]},
            "attraction",
        )
    if key == "allowed_innercity_modes":
        forbidden = next(
            mode for mode in ("walk", "metro", "taxi") if mode != witness.route_mode
        )
        return _build(
            constraint_id,
            "transport_mode",
            "not_in",
            {"modes": [forbidden], "leg": "all"},
            "innercity_route",
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
        _required_entity(selected, "restaurant")
        actual = _category_actual(witness, "restaurant")
        amount = _upper_bound(
            actual,
            quantum=10,
            factor=_budget_factor(slot.tightness),
        )
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
            {
                "amount": _upper_bound(
                    actual,
                    quantum=10,
                    factor=_budget_factor(slot.tightness),
                ),
                "basis": "per_person_per_night",
            },
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


def _required_logic_text(witness: WitnessResult, key: str) -> str:
    logic = witness.selected.get("logic")
    if not isinstance(logic, Mapping):
        raise SynthesisError("Witness is missing logic-diversity evidence.")
    value = logic.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SynthesisError(f"Witness has no usable logic-diversity value for {key}.")
    return value.strip()


def _category_actual(witness: WitnessResult, activity_type: str) -> float:
    scoped_types = (
        {"breakfast", "lunch", "dinner"}
        if activity_type == "restaurant"
        else {activity_type}
    )
    items = [
        item
        for item in witness.evidence_bundle["cost_items"]
        if item.get("activity_type") in scoped_types
    ]
    amounts = [float(item["amount"]) for item in items]
    travelers = int(witness.public_task["people_number"])
    divisor = (
        len(items)
        if activity_type == "restaurant"
        else max(1, int(witness.public_task["days"]) - 1)
    )
    return sum(amounts) / travelers / divisor


def _upper_bound(value: Any, *, quantum: int, factor: float) -> int:
    number = float(value)
    return max(quantum, math.ceil(number * factor / quantum) * quantum)


def _natural_boundary(value: str, operator: str, *, slack_minutes: int) -> str:
    hours, minutes = (int(part) for part in value.split(":", 1))
    actual = hours * 60 + minutes
    if operator == "lte":
        boundary = min(23 * 60 + 59, actual + slack_minutes)
    else:
        boundary = max(0, actual - slack_minutes)
    boundary = min(23 * 60 + 59, round(boundary / 5) * 5)
    return f"{boundary // 60:02d}:{boundary % 60:02d}"


def _clock_value(value: str) -> int:
    hours, minutes = (int(part) for part in value.split(":", 1))
    return hours * 60 + minutes


def _budget_factor(tightness: str) -> float:
    return {"easy": 1.25, "medium": 1.12, "hard": 1.03}[tightness]


def _time_slack(tightness: str) -> int:
    return {"easy": 60, "medium": 30, "hard": 15}[tightness]
