"""TravelTaskSpec constraint evaluators over normalized plan evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..tasks import ConstraintSpec, TravelTaskSpec
from .models import CHECK_FAIL, CHECK_PASS, CHECK_UNVERIFIABLE, CheckResult

_MEAL_TYPES = {"breakfast", "lunch", "dinner"}


def _result(
    constraint: ConstraintSpec,
    status: str,
    message: str,
    *,
    required: Any = None,
    actual: Any = None,
) -> CheckResult:
    return CheckResult(
        id=constraint.id,
        source="task_spec",
        hardness=constraint.hardness,
        status=status,
        message=message,
        evidence={"required": required, "actual": actual},
    )


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _activities(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = plan.get("activities")
    return [dict(item) for item in value] if isinstance(value, Sequence) else []


def _scope_activities(plan: Mapping[str, Any], scope: str) -> list[dict[str, Any]]:
    activities = _activities(plan)
    if scope == "attraction":
        return [item for item in activities if item.get("activity_type") == "attraction"]
    if scope == "restaurant":
        return [item for item in activities if item.get("activity_type") in _MEAL_TYPES]
    if scope == "accommodation":
        return [item for item in activities if item.get("activity_type") == "accommodation"]
    if scope == "intercity_transport":
        return [
            item for item in activities if item.get("activity_type") in {"train", "airplane"}
        ]
    return activities


def _compare(operator: str, actual: Any, expected: Any) -> bool | None:
    if operator == "eq":
        return actual == expected
    actual_number = _number(actual)
    expected_number = _number(expected)
    if actual_number is None or expected_number is None:
        return None
    if operator == "lte":
        return actual_number <= expected_number
    if operator == "gte":
        return actual_number >= expected_number
    return None


def _required_groups(value: Any, key: str) -> list[set[str]] | None:
    if not isinstance(value, Mapping):
        return None
    any_of = value.get("any_of")
    if isinstance(any_of, list):
        groups = []
        for group in any_of:
            if not isinstance(group, list):
                return None
            groups.append({str(item) for item in group})
        return groups
    values = value.get(key)
    if not isinstance(values, list):
        return None
    return [{str(item) for item in values}]


def _contains_groups(actual_values: set[str], groups: list[set[str]]) -> bool:
    normalized_actual = "\n".join(actual_values)
    return any(
        all(any(required in actual for actual in actual_values) for required in group)
        or all(required in normalized_actual for required in group)
        for group in groups
    )


def _entity_values(
    constraint: ConstraintSpec,
    plan: Mapping[str, Any],
    evidence: Mapping[str, Any],
    field: str,
) -> set[str] | None:
    entities = evidence.get("entities")
    if not isinstance(entities, Mapping):
        return None
    values: set[str] = set()
    for activity in _scope_activities(plan, constraint.scope):
        entity = entities.get(activity.get("candidate_id"))
        if not isinstance(entity, Mapping):
            return None
        value = entity.get(field)
        if value is not None:
            values.add(str(value))
    return values


def evaluate_constraint(
    constraint: ConstraintSpec,
    spec: TravelTaskSpec,
    plan: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> CheckResult:
    evaluator = _EVALUATORS.get(constraint.kind)
    if evaluator is None:
        return _result(
            constraint,
            CHECK_UNVERIFIABLE,
            f"No evaluator for constraint kind {constraint.kind}.",
        )
    return evaluator(constraint, spec, plan, evidence)


def _total_budget(
    constraint: ConstraintSpec,
    spec: TravelTaskSpec,
    plan: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> CheckResult:
    del spec, plan
    required = constraint.value.get("amount") if isinstance(constraint.value, Mapping) else None
    actual = evidence.get("total_cost")
    passed = _compare(constraint.operator, actual, required)
    if passed is None:
        return _result(constraint, CHECK_UNVERIFIABLE, "Budget evidence is incomplete.")
    return _result(
        constraint,
        CHECK_PASS if passed else CHECK_FAIL,
        "Total budget check completed.",
        required=required,
        actual=actual,
    )


def _category_budget(
    constraint: ConstraintSpec,
    spec: TravelTaskSpec,
    plan: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> CheckResult:
    value = constraint.value if isinstance(constraint.value, Mapping) else {}
    required = _number(value.get("amount"))
    basis = value.get("basis")
    cost_items = evidence.get("cost_items")
    if required is None or not isinstance(cost_items, Sequence):
        return _result(constraint, CHECK_UNVERIFIABLE, "Category budget is malformed.")
    activity_types = (
        _MEAL_TYPES if constraint.scope == "restaurant" else {"accommodation"}
    )
    items = [
        item
        for item in cost_items
        if isinstance(item, Mapping) and item.get("activity_type") in activity_types
    ]
    amounts = [_number(item.get("amount")) for item in items]
    if not items or any(amount is None for amount in amounts):
        return _result(constraint, CHECK_UNVERIFIABLE, "Category cost evidence is incomplete.")
    total = sum(amount for amount in amounts if amount is not None)
    if basis == "per_person_per_activity":
        actual = total / spec.trip.travelers / len(items)
    elif basis == "per_person_per_night":
        actual = total / spec.trip.travelers / max(1, spec.trip.days - 1)
    else:
        return _result(constraint, CHECK_UNVERIFIABLE, "Unknown category budget basis.")
    passed = _compare(constraint.operator, actual, required)
    return _result(
        constraint,
        CHECK_PASS if passed else CHECK_FAIL,
        "Category budget check completed.",
        required=required,
        actual=round(actual, 4),
    )


def _transport_mode(
    constraint: ConstraintSpec,
    spec: TravelTaskSpec,
    plan: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> CheckResult:
    value = constraint.value if isinstance(constraint.value, Mapping) else {}
    required = {str(item) for item in value.get("modes", [])}
    leg = str(value.get("leg", "all"))
    if constraint.scope == "innercity_route":
        routes = evidence.get("routes")
        if not isinstance(routes, Mapping):
            return _result(constraint, CHECK_UNVERIFIABLE, "Route modes are unavailable.")
        actual = {str(route.get("mode")) for route in routes.values() if isinstance(route, Mapping)}
    else:
        entities = evidence.get("entities")
        if not isinstance(entities, Mapping):
            return _result(constraint, CHECK_UNVERIFIABLE, "Transport evidence is unavailable.")
        destination = spec.trip.destinations[-1]
        expected_direction = {
            "outbound": (spec.trip.origin, destination),
            "return": (destination, spec.trip.origin),
        }.get(leg)
        actual = set()
        for item in _scope_activities(plan, "intercity_transport"):
            entity = entities.get(item.get("candidate_id"))
            if not isinstance(entity, Mapping):
                return _result(
                    constraint,
                    CHECK_UNVERIFIABLE,
                    "Intercity transport direction evidence is incomplete.",
                )
            direction = (str(entity.get("origin_city")), str(entity.get("destination_city")))
            if expected_direction is None or direction == expected_direction:
                actual.add(str(item.get("activity_type")))
    if constraint.operator == "eq":
        passed = actual == required
    elif constraint.operator in {"include", "contains"}:
        passed = required.issubset(actual)
    elif constraint.operator in {"not_in", "exclude"}:
        passed = actual.isdisjoint(required)
    else:
        return _result(constraint, CHECK_UNVERIFIABLE, "Unsupported mode operator.")
    return _result(
        constraint,
        CHECK_PASS if passed else CHECK_FAIL,
        "Transport mode check completed.",
        required={"leg": leg, "modes": sorted(required)},
        actual=sorted(actual),
    )


def _entity_category(
    constraint: ConstraintSpec,
    spec: TravelTaskSpec,
    plan: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> CheckResult:
    del spec
    field = "category" if constraint.scope == "attraction" else "cuisine"
    actual = _entity_values(constraint, plan, evidence, field)
    groups = _required_groups(constraint.value, "values")
    if actual is None or groups is None:
        return _result(constraint, CHECK_UNVERIFIABLE, "Entity category evidence is incomplete.")
    matched = _contains_groups(actual, groups)
    passed = not matched if constraint.operator == "not_contains" else matched
    return _result(
        constraint,
        CHECK_PASS if passed else CHECK_FAIL,
        "Entity category check completed.",
        required=[sorted(group) for group in groups],
        actual=sorted(actual),
    )


def _entity_attribute(
    constraint: ConstraintSpec,
    spec: TravelTaskSpec,
    plan: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> CheckResult:
    del spec
    actual = _entity_values(constraint, plan, evidence, "hotel_type")
    groups = _required_groups(constraint.value, "values")
    if actual is None or groups is None:
        return _result(constraint, CHECK_UNVERIFIABLE, "Entity attribute evidence is incomplete.")
    matched = _contains_groups(actual, groups)
    passed = not matched if constraint.operator == "not_contains" else matched
    return _result(
        constraint,
        CHECK_PASS if passed else CHECK_FAIL,
        "Entity attribute check completed.",
        required=[sorted(group) for group in groups],
        actual=sorted(actual),
    )


def _entity_name(
    constraint: ConstraintSpec,
    spec: TravelTaskSpec,
    plan: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> CheckResult:
    del spec
    actual = _entity_values(constraint, plan, evidence, "name")
    groups = _required_groups(constraint.value, "names")
    if actual is None or groups is None:
        return _result(constraint, CHECK_UNVERIFIABLE, "Entity name evidence is incomplete.")
    contains = _contains_groups(actual, groups)
    passed = not contains if constraint.kind == "exclude_entity" else contains
    return _result(
        constraint,
        CHECK_PASS if passed else CHECK_FAIL,
        "Entity inclusion check completed.",
        required=[sorted(group) for group in groups],
        actual=sorted(actual),
    )


def _room_value(
    constraint: ConstraintSpec,
    spec: TravelTaskSpec,
    plan: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> CheckResult:
    del spec, evidence
    key = "count" if constraint.kind == "room_count" else "room_type"
    required = constraint.value.get(key) if isinstance(constraint.value, Mapping) else None
    actual = [
        item.get("rooms" if key == "count" else key)
        for item in _scope_activities(plan, "accommodation")
    ]
    if required is None or not actual or any(value is None for value in actual):
        return _result(constraint, CHECK_UNVERIFIABLE, "Room selection evidence is incomplete.")
    outcomes = [_compare(constraint.operator, value, required) for value in actual]
    if any(value is None for value in outcomes):
        return _result(constraint, CHECK_UNVERIFIABLE, "Room comparison is unsupported.")
    return _result(
        constraint,
        CHECK_PASS if all(outcomes) else CHECK_FAIL,
        "Room selection check completed.",
        required=required,
        actual=actual,
    )


def _activity_count(
    constraint: ConstraintSpec,
    spec: TravelTaskSpec,
    plan: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> CheckResult:
    del spec, evidence
    value = constraint.value if isinstance(constraint.value, Mapping) else {}
    required = value.get("count")
    activities = _scope_activities(plan, constraint.scope)
    requested_type = value.get("activity_type")
    if requested_type:
        activities = [item for item in activities if item.get("activity_type") == requested_type]
    passed = _compare(constraint.operator, len(activities), required)
    if passed is None:
        return _result(constraint, CHECK_UNVERIFIABLE, "Activity count is malformed.")
    return _result(
        constraint,
        CHECK_PASS if passed else CHECK_FAIL,
        "Activity count check completed.",
        required=required,
        actual=len(activities),
    )


def _minutes(value: Any) -> int | None:
    if not isinstance(value, str) or ":" not in value:
        return None
    try:
        hours, minutes = (int(part) for part in value.split(":", 1))
    except ValueError:
        return None
    return hours * 60 + minutes if 0 <= hours <= 24 and 0 <= minutes < 60 else None


def _time_window(
    constraint: ConstraintSpec,
    spec: TravelTaskSpec,
    plan: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> CheckResult:
    value = constraint.value if isinstance(constraint.value, Mapping) else {}
    field = value.get("field")
    leg = value.get("leg")
    required = _minutes(value.get("time"))
    candidates = _scope_activities(plan, "intercity_transport")
    selected = []
    for activity in candidates:
        entity_id = activity.get("candidate_id")
        # Direction is available in EvidenceBundle entities, which the wrapper injects below.
        entity = evidence.get("entities", {}).get(entity_id, {})
        if leg == "outbound" and entity.get("origin_city") == spec.trip.origin:
            selected.append(activity)
        elif leg == "return" and entity.get("destination_city") == spec.trip.origin:
            selected.append(activity)
    actual_values = [_minutes(activity.get(field)) for activity in selected]
    if required is None or not selected or any(value is None for value in actual_values):
        return _result(constraint, CHECK_UNVERIFIABLE, "Intercity time evidence is incomplete.")
    outcomes = [_compare(constraint.operator, actual, required) for actual in actual_values]
    return _result(
        constraint,
        CHECK_PASS if all(outcomes) else CHECK_FAIL,
        "Intercity time window check completed.",
        required=value.get("time"),
        actual=[activity.get(field) for activity in selected],
    )


_EVALUATORS = {
    "activity_count": _activity_count,
    "category_budget": _category_budget,
    "entity_attribute": _entity_attribute,
    "entity_category": _entity_category,
    "exclude_entity": _entity_name,
    "include_entity": _entity_name,
    "room_count": _room_value,
    "room_type": _room_value,
    "time_window": _time_window,
    "total_budget": _total_budget,
    "transport_mode": _transport_mode,
}
