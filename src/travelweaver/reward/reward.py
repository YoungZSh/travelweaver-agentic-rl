"""Deterministic unified plan validity and TravelReward v2."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from ..tasks import TravelTaskSpec
from .evaluators import evaluate_constraint
from .models import CHECK_FAIL, CHECK_PASS, CHECK_UNVERIFIABLE, CheckResult, RewardResult

REWARD_VERSION = "travelweaver-reward-v2"

REWARD_GROUPS = (
    "protocol_structure",
    "evidence_grounding",
    "spatiotemporal_commonsense",
    "task_constraints",
    "quantity_cost",
)

_CHECK_GROUPS = {
    "task_alignment": "protocol_structure",
    "plan_structure": "protocol_structure",
    "terminal_plan": "protocol_structure",
    "entity_grounding": "evidence_grounding",
    "route_grounding": "evidence_grounding",
    "opening_hours": "spatiotemporal_commonsense",
    "trip_coverage": "spatiotemporal_commonsense",
    "entity_uniqueness": "spatiotemporal_commonsense",
    "meal_commonsense": "spatiotemporal_commonsense",
    "quantity_consistency": "quantity_cost",
    "cost_accounting": "quantity_cost",
}


def _env_check(
    check_id: str, status: str, message: str, evidence: dict[str, Any]
) -> CheckResult:
    return CheckResult(
        id=check_id,
        source="environment",
        hardness="hard",
        status=status,
        message=message,
        evidence=evidence,
    )


def _minutes(value: Any, *, allow_24: bool = False) -> int | None:
    if value == "24:00" and allow_24:
        return 24 * 60
    if not isinstance(value, str) or ":" not in value:
        return None
    try:
        hours, minutes = (int(part) for part in value.split(":", 1))
    except ValueError:
        return None
    if not 0 <= hours < 24 or not 0 <= minutes < 60:
        return None
    return hours * 60 + minutes


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


class TravelReward:
    """Score one terminal plan from its frozen task contract and evidence."""

    def evaluate(
        self,
        spec: TravelTaskSpec,
        plan_snapshot: Mapping[str, Any],
        evidence_bundle: Mapping[str, Any],
        *,
        termination_reason: str = "plan_submitted",
    ) -> RewardResult:
        checks = self._environment_checks(spec, plan_snapshot, evidence_bundle)
        checks.extend(
            evaluate_constraint(constraint, spec, plan_snapshot, evidence_bundle)
            for constraint in spec.constraints
        )
        return self._score(tuple(checks), termination_reason, spec.spec_hash)

    def no_plan(self, termination_reason: str) -> RewardResult:
        check = _env_check(
            "terminal_plan",
            CHECK_FAIL,
            "Episode ended without an evaluable submitted plan.",
            {"termination_reason": termination_reason},
        )
        return RewardResult(
            reward_version=REWARD_VERSION,
            reward=-1.0,
            reward_type="no_evaluable_plan",
            reward_valid=True,
            termination_reason=termination_reason,
            hard_score=0.0,
            soft_score=1.0,
            all_hard_pass=False,
            checks=(check,),
            task_spec_hash=None,
            group_results={group: False for group in REWARD_GROUPS},
            sft_accepted=False,
            rl_reward=-1.0,
        )

    @staticmethod
    def _score(
        checks: tuple[CheckResult, ...], termination_reason: str, spec_hash: str
    ) -> RewardResult:
        group_results = TravelReward._group_results(checks)
        if any(check.status == CHECK_UNVERIFIABLE for check in checks):
            return RewardResult(
                reward_version=REWARD_VERSION,
                reward=-1.0,
                reward_type="reward_unverifiable",
                reward_valid=False,
                termination_reason=termination_reason,
                hard_score=0.0,
                soft_score=0.0,
                all_hard_pass=False,
                checks=checks,
                task_spec_hash=spec_hash,
                group_results=group_results,
                sft_accepted=False,
                rl_reward=-1.0,
            )
        hard = [check for check in checks if check.hardness == "hard"]
        soft = [check for check in checks if check.hardness == "soft"]
        hard_score = sum(check.status == CHECK_PASS for check in hard) / len(hard) if hard else 1.0
        soft_score = sum(check.status == CHECK_PASS for check in soft) / len(soft) if soft else 1.0
        all_hard_pass = all(check.status == CHECK_PASS for check in hard)
        if all_hard_pass:
            reward = 0.5 + 0.5 * soft_score
            reward_type = "strict_valid_plan" if soft_score == 1.0 else "feasible_plan"
        else:
            reward = -0.5 + 0.1 * sum(group_results.values())
            reward_type = "hard_constraint_failure"
        reward = round(reward, 8)
        sft_accepted = all_hard_pass and soft_score == 1.0
        return RewardResult(
            reward_version=REWARD_VERSION,
            reward=reward,
            reward_type=reward_type,
            reward_valid=True,
            termination_reason=termination_reason,
            hard_score=round(hard_score, 8),
            soft_score=round(soft_score, 8),
            all_hard_pass=all_hard_pass,
            checks=checks,
            task_spec_hash=spec_hash,
            group_results=group_results,
            sft_accepted=sft_accepted,
            rl_reward=reward,
        )

    @staticmethod
    def _group_results(checks: tuple[CheckResult, ...]) -> dict[str, bool]:
        grouped: dict[str, list[CheckResult]] = {group: [] for group in REWARD_GROUPS}
        for check in checks:
            group = (
                "task_constraints"
                if check.source == "task_spec"
                else _CHECK_GROUPS.get(check.id, "protocol_structure")
            )
            if check.hardness == "hard":
                grouped[group].append(check)
        return {
            group: all(check.status == CHECK_PASS for check in group_checks)
            for group, group_checks in grouped.items()
        }

    def _environment_checks(
        self,
        spec: TravelTaskSpec,
        plan: Mapping[str, Any],
        evidence: Mapping[str, Any],
    ) -> list[CheckResult]:
        activities_value = plan.get("activities")
        activities = (
            [dict(item) for item in activities_value]
            if isinstance(activities_value, Sequence)
            else []
        )
        entities = evidence.get("entities")
        routes = evidence.get("routes")
        entities = entities if isinstance(entities, Mapping) else {}
        routes = routes if isinstance(routes, Mapping) else {}
        checks = [self._task_alignment(spec, plan), self._structure(spec, activities)]
        checks.append(self._entity_grounding(activities, entities))
        checks.append(self._route_grounding(activities, routes))
        checks.append(self._opening_hours(activities, entities))
        checks.append(self._trip_coverage(spec, activities, entities))
        checks.append(self._entity_uniqueness(activities, entities))
        checks.append(self._meal_commonsense(activities, entities))
        checks.append(self._quantity_consistency(spec, activities, evidence))
        checks.append(self._cost_accounting(activities, evidence))
        return checks

    @staticmethod
    def _entity_uniqueness(
        activities: list[dict[str, Any]], entities: Mapping[str, Any]
    ) -> CheckResult:
        attraction_ids: list[str] = []
        restaurant_ids: list[str] = []
        for activity in activities:
            candidate_id = activity.get("candidate_id")
            if not isinstance(candidate_id, str):
                continue
            activity_type = activity.get("activity_type")
            entity = entities.get(candidate_id, {})
            entity_type = entity.get("entity_type") if isinstance(entity, Mapping) else None
            if activity_type == "attraction":
                attraction_ids.append(candidate_id)
            elif activity_type in {"breakfast", "lunch", "dinner"} and entity_type == "restaurant":
                restaurant_ids.append(candidate_id)
        repeated_attractions = sorted(
            candidate_id
            for candidate_id in set(attraction_ids)
            if attraction_ids.count(candidate_id) > 1
        )
        repeated_restaurants = sorted(
            candidate_id
            for candidate_id in set(restaurant_ids)
            if restaurant_ids.count(candidate_id) > 1
        )
        passed = not repeated_attractions and not repeated_restaurants
        return _env_check(
            "entity_uniqueness",
            CHECK_PASS if passed else CHECK_FAIL,
            "Attractions and ordinary restaurants are not repeated during the trip.",
            {
                "repeated_attractions": repeated_attractions,
                "repeated_restaurants": repeated_restaurants,
            },
        )

    @staticmethod
    def _meal_commonsense(
        activities: list[dict[str, Any]], entities: Mapping[str, Any]
    ) -> CheckResult:
        windows = {
            "breakfast": (6 * 60, 9 * 60),
            "lunch": (11 * 60, 14 * 60),
            "dinner": (17 * 60, 20 * 60),
        }
        counts: dict[tuple[int, str], int] = defaultdict(int)
        inappropriate: list[dict[str, Any]] = []
        repeated: list[dict[str, Any]] = []
        unverifiable: list[dict[str, Any]] = []
        for activity in activities:
            meal_type = activity.get("activity_type")
            if meal_type not in windows:
                continue
            day = activity.get("day")
            start = _minutes(activity.get("start_time"))
            end = _minutes(activity.get("end_time"), allow_24=True)
            if not isinstance(day, int) or start is None or end is None:
                unverifiable.append({"day": day, "candidate_id": activity.get("candidate_id")})
                continue
            key = (day, str(meal_type))
            counts[key] += 1
            if counts[key] > 1:
                repeated.append({"day": day, "meal_type": meal_type})
            lower, upper = windows[str(meal_type)]
            if start < lower or end > upper:
                inappropriate.append(
                    {
                        "day": day,
                        "meal_type": meal_type,
                        "candidate_id": activity.get("candidate_id"),
                        "start_time": activity.get("start_time"),
                        "end_time": activity.get("end_time"),
                    }
                )
            if meal_type == "breakfast":
                entity = entities.get(activity.get("candidate_id"), {})
                if isinstance(entity, Mapping) and entity.get("entity_type") == "hotel":
                    price = _number(activity.get("unit_price"))
                    amount = _number(activity.get("amount"))
                    if price != 0.0 or amount != 0.0:
                        inappropriate.append(
                            {
                                "day": day,
                                "meal_type": meal_type,
                                "candidate_id": activity.get("candidate_id"),
                                "reason": "hotel_breakfast_must_be_free",
                            }
                        )
        if unverifiable:
            status = CHECK_UNVERIFIABLE
        else:
            status = CHECK_PASS if not inappropriate and not repeated else CHECK_FAIL
        return _env_check(
            "meal_commonsense",
            status,
            "Meal types are unique per day and occur inside official meal windows.",
            {
                "inappropriate": inappropriate,
                "repeated": repeated,
                "unverifiable": unverifiable,
            },
        )

    @staticmethod
    def _task_alignment(spec: TravelTaskSpec, plan: Mapping[str, Any]) -> CheckResult:
        actual = {
            "task_id": plan.get("task_id"),
            "people_number": plan.get("people_number"),
            "start_city": plan.get("start_city"),
            "target_city": plan.get("target_city"),
            "days": plan.get("days"),
        }
        expected = {
            "task_id": spec.task_id,
            "people_number": spec.trip.travelers,
            "start_city": spec.trip.origin,
            "target_city": spec.trip.destinations[-1],
            "days": spec.trip.days,
        }
        return _env_check(
            "task_alignment",
            CHECK_PASS if actual == expected else CHECK_FAIL,
            "Plan snapshot matches the frozen trip metadata.",
            {"required": expected, "actual": actual},
        )

    @staticmethod
    def _structure(spec: TravelTaskSpec, activities: list[dict[str, Any]]) -> CheckResult:
        days: dict[int, list[tuple[int, int]]] = defaultdict(list)
        valid = bool(activities)
        attraction_count = 0
        for activity in activities:
            day = activity.get("day")
            start = activity.get("absolute_start")
            end = activity.get("absolute_end")
            if not isinstance(start, int):
                clock = _minutes(activity.get("start_time"))
                start = (
                    (day - 1) * 24 * 60 + clock
                    if isinstance(day, int) and clock is not None
                    else None
                )
            if not isinstance(end, int):
                clock = _minutes(activity.get("end_time"), allow_24=True)
                end = (
                    (day - 1) * 24 * 60 + clock
                    if isinstance(day, int) and clock is not None
                    else None
                )
            if not isinstance(day, int) or start is None or end is None or end <= start:
                valid = False
                continue
            days[day].append((start, end))
            attraction_count += activity.get("activity_type") == "attraction"
        valid = valid and sorted(days) == list(range(1, spec.trip.days + 1))
        valid = valid and attraction_count > 0
        for intervals in days.values():
            ordered = sorted(intervals)
            valid = valid and all(
                current[0] >= previous[1]
                for previous, current in zip(ordered, ordered[1:], strict=False)
            )
        return _env_check(
            "plan_structure",
            CHECK_PASS if valid else CHECK_FAIL,
            "Plan covers each day with non-overlapping activities.",
            {"days": sorted(days), "attraction_count": attraction_count},
        )

    @staticmethod
    def _entity_grounding(
        activities: list[dict[str, Any]], entities: Mapping[str, Any]
    ) -> CheckResult:
        missing = [
            item.get("candidate_id")
            for item in activities
            if item.get("candidate_id") not in entities
        ]
        return _env_check(
            "entity_grounding",
            CHECK_PASS if not missing else CHECK_FAIL,
            "Every submitted activity is grounded in environment evidence.",
            {"missing_candidate_ids": missing},
        )

    @staticmethod
    def _route_grounding(
        activities: list[dict[str, Any]], routes: Mapping[str, Any]
    ) -> CheckResult:
        missing = []
        mismatched = []
        unverifiable = []
        ordered = sorted(
            activities,
            key=lambda item: (int(item.get("day", 0)), int(item.get("activity_index", 0))),
        )
        previous: dict[str, Any] | None = None
        for activity in ordered:
            current_origin = activity.get("origin_position_id") or activity.get("candidate_id")
            previous_destination = (
                previous.get("destination_position_id") or previous.get("candidate_id")
                if previous is not None
                else None
            )
            requires_route = previous is not None and previous_destination != current_origin
            route_id = activity.get("route_from_previous_id")
            if requires_route:
                route = routes.get(route_id) if isinstance(route_id, str) else None
                if not isinstance(route, Mapping):
                    missing.append(
                        {
                            "origin": previous_destination,
                            "destination": current_origin,
                            "route_id": route_id,
                        }
                    )
                elif (
                    route.get("origin_place_id") != previous_destination
                    or route.get("destination_place_id") != current_origin
                ):
                    mismatched.append(route_id)
                else:
                    segments = route.get("segments")
                    if not isinstance(segments, list) or not segments:
                        unverifiable.append(route_id)
                    else:
                        first = segments[0] if isinstance(segments[0], Mapping) else {}
                        last = segments[-1] if isinstance(segments[-1], Mapping) else {}
                        route_start = _minutes(first.get("start_time"))
                        route_end = _minutes(last.get("end_time"), allow_24=True)
                        previous_end = previous.get("absolute_end")
                        next_start = activity.get("absolute_start")
                        day = activity.get("day")
                        if (
                            route_start is None
                            or route_end is None
                            or not isinstance(previous_end, int)
                            or not isinstance(next_start, int)
                            or not isinstance(day, int)
                        ):
                            unverifiable.append(route_id)
                        else:
                            day_base = (day - 1) * 24 * 60
                            route_start += day_base
                            route_end += day_base
                            if route_end < route_start:
                                route_end += 24 * 60
                            if route_start < previous_end or route_end > next_start:
                                mismatched.append(route_id)
            elif route_id is not None:
                mismatched.append(route_id)
            previous = activity
        if unverifiable:
            status = CHECK_UNVERIFIABLE
        else:
            status = CHECK_PASS if not missing and not mismatched else CHECK_FAIL
        return _env_check(
            "route_grounding",
            status,
            "Every required local connection has matching, feasible route evidence.",
            {"missing": missing, "mismatched": mismatched, "unverifiable": unverifiable},
        )

    @staticmethod
    def _opening_hours(
        activities: list[dict[str, Any]], entities: Mapping[str, Any]
    ) -> CheckResult:
        failures = []
        unverifiable = []
        for activity in activities:
            if activity.get("activity_type") not in {"attraction", "breakfast", "lunch", "dinner"}:
                continue
            entity = entities.get(activity.get("candidate_id"))
            if not isinstance(entity, Mapping):
                unverifiable.append(activity.get("candidate_id"))
                continue
            if (
                activity.get("activity_type") == "breakfast"
                and entity.get("entity_type") == "hotel"
            ):
                continue
            opening = _minutes(entity.get("open_time"))
            closing = _minutes(entity.get("close_time"), allow_24=True)
            start = _minutes(activity.get("start_time"))
            end = _minutes(activity.get("end_time"), allow_24=True)
            if None in {opening, closing, start, end}:
                unverifiable.append(activity.get("candidate_id"))
            elif opening <= closing and not (start >= opening and end <= closing):
                failures.append(activity.get("candidate_id"))
        status = CHECK_UNVERIFIABLE if unverifiable else CHECK_FAIL if failures else CHECK_PASS
        return _env_check(
            "opening_hours",
            status,
            "Attractions and meals fit their recorded opening hours.",
            {"failures": failures, "unverifiable": unverifiable},
        )

    @staticmethod
    def _trip_coverage(
        spec: TravelTaskSpec,
        activities: list[dict[str, Any]],
        entities: Mapping[str, Any],
    ) -> CheckResult:
        accommodations = sum(
            item.get("activity_type") == "accommodation" for item in activities
        )
        actual_directions = set()
        for activity in activities:
            if activity.get("activity_type") not in {"train", "airplane"}:
                continue
            entity = entities.get(activity.get("candidate_id"), {})
            if isinstance(entity, Mapping):
                actual_directions.add(
                    (str(entity.get("origin_city")), str(entity.get("destination_city")))
                )
        destination = spec.trip.destinations[-1]
        required_directions = (
            {(spec.trip.origin, destination), (destination, spec.trip.origin)}
            if spec.trip.origin != destination
            else set()
        )
        valid = accommodations >= max(0, spec.trip.days - 1)
        valid = valid and required_directions.issubset(actual_directions)
        ordered = sorted(
            activities,
            key=lambda item: (int(item.get("day", 0)), int(item.get("activity_index", 0))),
        )
        if required_directions and ordered:
            first_entity = entities.get(ordered[0].get("candidate_id"), {})
            last_entity = entities.get(ordered[-1].get("candidate_id"), {})
            valid = valid and isinstance(first_entity, Mapping) and (
                str(first_entity.get("origin_city")),
                str(first_entity.get("destination_city")),
            ) == (spec.trip.origin, destination)
            valid = valid and isinstance(last_entity, Mapping) and (
                str(last_entity.get("origin_city")),
                str(last_entity.get("destination_city")),
            ) == (destination, spec.trip.origin)
        return _env_check(
            "trip_coverage",
            CHECK_PASS if valid else CHECK_FAIL,
            "Intercity directions and overnight stays cover the trip.",
            {
                "required_directions": sorted(required_directions),
                "actual_directions": sorted(actual_directions),
                "accommodation_nights": accommodations,
                "first_activity": ordered[0].get("candidate_id") if ordered else None,
                "last_activity": ordered[-1].get("candidate_id") if ordered else None,
            },
        )

    @staticmethod
    def _quantity_consistency(
        spec: TravelTaskSpec,
        activities: list[dict[str, Any]],
        evidence: Mapping[str, Any],
    ) -> CheckResult:
        failures = []
        for activity in activities:
            activity_type = activity.get("activity_type")
            if activity_type == "accommodation":
                rooms = activity.get("rooms")
                room_type = activity.get("room_type")
                if not isinstance(rooms, int) or not isinstance(room_type, int):
                    failures.append(activity.get("candidate_id"))
                elif rooms * room_type < spec.trip.travelers:
                    failures.append(activity.get("candidate_id"))
                elif activity.get("derived_quantity") != rooms:
                    failures.append(activity.get("candidate_id"))
            elif activity.get("derived_quantity") != spec.trip.travelers:
                failures.append(activity.get("candidate_id"))
        cost_items = evidence.get("cost_items")
        routes = evidence.get("routes")
        if isinstance(cost_items, Sequence) and isinstance(routes, Mapping):
            for item in cost_items:
                if not isinstance(item, Mapping) or item.get("kind") != "route":
                    continue
                route = routes.get(item.get("route_id"), {})
                mode = route.get("mode") if isinstance(route, Mapping) else None
                expected = (
                    math.ceil(spec.trip.travelers / 4)
                    if mode == "taxi"
                    else spec.trip.travelers
                    if mode == "metro"
                    else 1
                )
                if item.get("quantity") != expected:
                    failures.append(item.get("route_id"))
        return _env_check(
            "quantity_consistency",
            CHECK_PASS if not failures else CHECK_FAIL,
            "Tickets and rooms follow frozen quantity rules.",
            {"failures": failures, "travelers": spec.trip.travelers},
        )

    @staticmethod
    def _cost_accounting(
        activities: list[dict[str, Any]], evidence: Mapping[str, Any]
    ) -> CheckResult:
        items = evidence.get("cost_items")
        total = _number(evidence.get("total_cost"))
        if not isinstance(items, Sequence) or total is None:
            return _env_check(
                "cost_accounting",
                CHECK_UNVERIFIABLE,
                "Environment cost evidence is incomplete.",
                {"total_cost": evidence.get("total_cost")},
            )
        normalized_items = [item for item in items if isinstance(item, Mapping)]
        amounts = [_number(item.get("amount")) for item in normalized_items]
        if len(normalized_items) != len(items) or any(amount is None for amount in amounts):
            return _env_check(
                "cost_accounting",
                CHECK_UNVERIFIABLE,
                "One or more cost items are not verifiable.",
                {"amounts": amounts, "total_cost": total},
            )
        multiplication_failures = []
        for item in normalized_items:
            quantity = _number(item.get("quantity"))
            unit_price = _number(item.get("unit_price"))
            amount = _number(item.get("amount"))
            if None in {quantity, unit_price, amount} or not math.isclose(
                float(amount), float(quantity) * float(unit_price), abs_tol=1e-6
            ):
                multiplication_failures.append(
                    item.get("candidate_id") or item.get("route_id")
                )
        expected_activity_keys = {
            (item.get("day"), item.get("activity_index")) for item in activities
        }
        actual_activity_keys = {
            (item.get("day"), item.get("activity_index"))
            for item in normalized_items
            if item.get("kind") == "activity"
        }
        calculated = round(sum(amount for amount in amounts if amount is not None), 2)
        passed = (
            math.isclose(calculated, total, abs_tol=1e-6)
            and not multiplication_failures
            and actual_activity_keys == expected_activity_keys
        )
        return _env_check(
            "cost_accounting",
            CHECK_PASS if passed else CHECK_FAIL,
            "Environment total equals the sum of evidence-backed costs.",
            {
                "calculated": calculated,
                "total_cost": total,
                "multiplication_failures": multiplication_failures,
                "missing_activity_costs": sorted(expected_activity_keys - actual_activity_keys),
            },
        )
