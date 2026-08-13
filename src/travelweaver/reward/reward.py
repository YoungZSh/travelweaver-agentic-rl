"""Deterministic A/V/G outcome Reward shared by collection and online RL."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from ..tasks import TravelTaskSpec
from .contract import FrozenOutcomeContract
from .evaluators import evaluate_constraint
from .models import (
    CHECK_BLOCKED,
    CHECK_FAIL,
    CHECK_NOT_APPLICABLE,
    CHECK_PASS,
    CHECK_UNVERIFIABLE,
    DIMENSION_ARTIFACT,
    DIMENSION_GOAL,
    DIMENSION_VALIDITY,
    REWARD_DIMENSIONS,
    CheckResult,
    RewardResult,
)
from .registry import check_owner

REWARD_VERSION = "travelweaver-reward-v4"

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
    "candidate_usage": "evidence_grounding",
    "intercity_time": "evidence_grounding",
    "route_grounding": "evidence_grounding",
    "opening_hours": "spatiotemporal_commonsense",
    "trip_coverage": "spatiotemporal_commonsense",
    "entity_uniqueness": "spatiotemporal_commonsense",
    "meal_commonsense": "spatiotemporal_commonsense",
    "quantity_consistency": "quantity_cost",
    "cost_accounting": "quantity_cost",
}


def _env_check(
    check_id: str,
    status: str,
    message: str,
    evidence: dict[str, Any],
    *,
    owner_dimension: str = DIMENSION_VALIDITY,
    score: float | None = None,
    blocked_by: str | None = None,
    affects_success: bool = True,
    affects_shaping: bool = True,
) -> CheckResult:
    registered_owner = check_owner(check_id)
    if owner_dimension != registered_owner:
        raise ValueError(
            f"Reward check {check_id} belongs to {registered_owner}, not {owner_dimension}."
        )
    return CheckResult(
        id=check_id,
        source="environment",
        hardness="hard",
        status=status,
        message=message,
        evidence=evidence,
        owner_dimension=owner_dimension,
        score=score,
        blocked_by=blocked_by,
        affects_success=affects_success,
        affects_shaping=affects_shaping,
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
        contract = FrozenOutcomeContract.compile(spec)
        normalized_plan = self._upgrade_legacy_plan_snapshot(plan_snapshot, evidence_bundle)
        checks = self._environment_checks(spec, normalized_plan, evidence_bundle)
        checks.extend(self._goal_checks(spec, normalized_plan, evidence_bundle, checks))
        return self._score(
            tuple(checks),
            termination_reason,
            spec.spec_hash,
            contract.contract_hash,
            admission_passed=termination_reason == "plan_submitted",
        )

    def evaluate_submission(
        self,
        spec: TravelTaskSpec,
        raw_plan: Mapping[str, Any],
        candidates: Mapping[str, Any],
        routes: Mapping[str, Any],
        *,
        termination_reason: str = "invalid_plan_submitted",
    ) -> RewardResult:
        """Collect all deterministically checkable outcomes from a rejected raw plan."""

        snapshot, evidence = self._materialize_submission(spec, raw_plan, candidates, routes)
        contract = FrozenOutcomeContract.compile(spec)
        checks = self._environment_checks(spec, snapshot, evidence)
        checks.extend(self._goal_checks(spec, snapshot, evidence, checks))
        return self._score(
            tuple(checks),
            termination_reason,
            spec.spec_hash,
            contract.contract_hash,
            admission_passed=False,
        )

    def no_plan(self, termination_reason: str) -> RewardResult:
        check = _env_check(
            "terminal_plan",
            CHECK_FAIL,
            "Episode ended without an evaluable submitted plan.",
            {"termination_reason": termination_reason},
            owner_dimension=DIMENSION_ARTIFACT,
            score=0.0,
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
            dimension_scores={dimension: 0.0 for dimension in REWARD_DIMENSIONS},
            dimension_coverage={
                dimension: {
                    "scored": int(dimension == DIMENSION_ARTIFACT),
                    "blocked": 0,
                    "not_applicable": 0,
                }
                for dimension in REWARD_DIMENSIONS
            },
            outcome_contract_hash=None,
            admission_passed=False,
        )

    @staticmethod
    def _score(
        checks: tuple[CheckResult, ...],
        termination_reason: str,
        spec_hash: str,
        contract_hash: str,
        *,
        admission_passed: bool,
    ) -> RewardResult:
        group_results = TravelReward._group_results(checks)
        dimension_scores, dimension_coverage = TravelReward._dimension_results(checks)
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
                dimension_scores=dimension_scores,
                dimension_coverage=dimension_coverage,
                outcome_contract_hash=contract_hash,
                admission_passed=admission_passed,
            )
        hard = [
            check
            for check in checks
            if check.hardness == "hard" and check.affects_success
        ]
        hard_scored = [
            check for check in hard if check.status in {CHECK_PASS, CHECK_FAIL}
        ]
        soft_scored = [
            check
            for check in checks
            if check.hardness == "soft" and check.status in {CHECK_PASS, CHECK_FAIL}
        ]
        hard_score = (
            sum(check.status == CHECK_PASS for check in hard_scored) / len(hard_scored)
            if hard_scored
            else 1.0
        )
        soft_score = (
            sum(check.status == CHECK_PASS for check in soft_scored) / len(soft_scored)
            if soft_scored
            else 1.0
        )
        all_hard_pass = admission_passed and all(
            check.status == CHECK_PASS for check in hard
        )
        if all_hard_pass:
            reward = 0.5 + 0.5 * soft_score
            reward_type = "strict_valid_plan" if soft_score == 1.0 else "feasible_plan"
        else:
            shaping = sum(dimension_scores.values()) / len(REWARD_DIMENSIONS)
            reward = min(-1.0 + shaping, -1e-8)
            reward_type = (
                "partial_invalid_plan"
                if termination_reason == "invalid_plan_submitted"
                else "hard_constraint_failure"
            )
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
            dimension_scores=dimension_scores,
            dimension_coverage=dimension_coverage,
            outcome_contract_hash=contract_hash,
            admission_passed=admission_passed,
        )

    @staticmethod
    def _dimension_results(
        checks: tuple[CheckResult, ...],
    ) -> tuple[dict[str, float], dict[str, dict[str, int]]]:
        scores: dict[str, float] = {}
        coverage: dict[str, dict[str, int]] = {}
        for dimension in REWARD_DIMENSIONS:
            owned = [check for check in checks if check.owner_dimension == dimension]
            scored = [
                check.score
                for check in owned
                if check.affects_shaping
                and check.status in {CHECK_PASS, CHECK_FAIL}
                and check.score is not None
            ]
            scores[dimension] = round(sum(scored) / len(scored), 8) if scored else 0.0
            coverage[dimension] = {
                "scored": len(scored),
                "blocked": sum(check.status == CHECK_BLOCKED for check in owned),
                "not_applicable": sum(
                    check.status == CHECK_NOT_APPLICABLE for check in owned
                ),
            }
        return scores, coverage

    @staticmethod
    def _group_results(checks: tuple[CheckResult, ...]) -> dict[str, bool]:
        grouped: dict[str, list[CheckResult]] = {group: [] for group in REWARD_GROUPS}
        for check in checks:
            group = (
                "task_constraints"
                if check.source == "task_spec"
                else _CHECK_GROUPS.get(check.id, "protocol_structure")
            )
            if check.hardness == "hard" and check.affects_success:
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
        evidence_schema_version = str(evidence.get("schema_version") or "")
        checks = [self._terminal_plan(plan), self._task_alignment(spec, plan)]
        checks.append(self._structure(spec, activities))
        checks.append(self._chronology(activities))
        checks.append(self._entity_grounding(activities, entities))
        checks.append(self._candidate_usage(activities, entities, evidence))
        checks.append(self._intercity_time(activities, entities))
        checks.append(
            self._route_grounding(
                activities,
                routes,
                evidence_schema_version=evidence_schema_version,
            )
        )
        checks.append(self._opening_hours(activities, entities))
        checks.append(self._trip_coverage(spec, activities, entities))
        checks.append(
            self._entity_uniqueness(
                activities,
                entities,
                affects_success=evidence_schema_version != "travelweaver-evidence-v1",
            )
        )
        checks.append(
            self._meal_commonsense(
                activities,
                entities,
                affects_success=evidence_schema_version != "travelweaver-evidence-v1",
            )
        )
        checks.append(self._quantity_consistency(plan, activities, evidence))
        checks.append(self._cost_accounting(activities, evidence))
        checks.append(self._overnight_coverage(plan, activities))
        return self._resolve_environment_blockers(checks)

    @staticmethod
    def _terminal_plan(plan: Mapping[str, Any]) -> CheckResult:
        return _env_check(
            "terminal_plan",
            CHECK_PASS,
            "A schema-valid terminal plan object is available for evaluation.",
            {"plan_schema_version": plan.get("schema_version")},
            owner_dimension=DIMENSION_ARTIFACT,
            score=1.0,
        )

    @staticmethod
    def _upgrade_legacy_plan_snapshot(
        plan: Mapping[str, Any], evidence: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Derive v2 timing/position fields when reading immutable v1 snapshots."""

        upgraded = dict(plan)
        raw_activities = plan.get("activities")
        if not isinstance(raw_activities, Sequence):
            return upgraded
        entities = evidence.get("entities")
        entities = entities if isinstance(entities, Mapping) else {}
        activities: list[dict[str, Any]] = []
        for raw_activity in raw_activities:
            activity = dict(raw_activity) if isinstance(raw_activity, Mapping) else {}
            day = activity.get("day")
            start = _minutes(activity.get("start_time"))
            end = _minutes(activity.get("end_time"), allow_24=True)
            overnight = (
                activity.get("activity_type") in {"train", "airplane"}
                and start is not None
                and end is not None
                and end <= start
            )
            if "absolute_start" not in activity:
                activity["absolute_start"] = (
                    (day - 1) * 24 * 60 + start
                    if isinstance(day, int) and start is not None
                    else None
                )
            if "absolute_end" not in activity:
                activity["absolute_end"] = (
                    (day - 1 + int(overnight)) * 24 * 60 + end
                    if isinstance(day, int) and end is not None
                    else None
                )
            entity = entities.get(activity.get("candidate_id"), {})
            if isinstance(entity, Mapping) and activity.get("activity_type") in {
                "train",
                "airplane",
            }:
                activity.setdefault("origin_position_id", entity.get("origin_anchor_id"))
                activity.setdefault(
                    "destination_position_id", entity.get("destination_anchor_id")
                )
            else:
                activity.setdefault("origin_position_id", activity.get("candidate_id"))
                activity.setdefault("destination_position_id", activity.get("candidate_id"))
            activities.append(activity)
        upgraded["activities"] = activities
        return upgraded

    @staticmethod
    def _resolve_environment_blockers(checks: list[CheckResult]) -> list[CheckResult]:
        """Distinguish model-caused missing prerequisites from validator failures."""

        blockers = {
            check.id: check
            for check in checks
            if check.status in {CHECK_FAIL, CHECK_BLOCKED}
        }
        dependencies = {
            "candidate_usage": ("entity_grounding",),
            "intercity_time": ("entity_grounding", "plan_structure"),
            "route_grounding": ("entity_grounding", "plan_structure", "chronology"),
            "opening_hours": ("entity_grounding", "plan_structure"),
            "trip_coverage": ("entity_grounding",),
            "meal_commonsense": ("plan_structure", "entity_grounding"),
            "cost_accounting": ("entity_grounding", "route_grounding", "plan_structure"),
        }
        resolved: list[CheckResult] = []
        for check in checks:
            if check.status != CHECK_UNVERIFIABLE:
                resolved.append(check)
                continue
            blocker_id = next(
                (
                    dependency
                    for dependency in dependencies.get(check.id, ())
                    if dependency in blockers
                ),
                None,
            )
            if blocker_id is None:
                resolved.append(check)
                continue
            resolved.append(
                CheckResult(
                    id=check.id,
                    source=check.source,
                    hardness=check.hardness,
                    status=CHECK_BLOCKED,
                    message=f"Environment check is blocked by {blocker_id}.",
                    evidence=check.evidence,
                    owner_dimension=check.owner_dimension,
                    score=None,
                    blocked_by=blocker_id,
                    affects_success=check.affects_success,
                    affects_shaping=check.affects_shaping,
                )
            )
        return resolved

    @staticmethod
    def _goal_checks(
        spec: TravelTaskSpec,
        plan: Mapping[str, Any],
        evidence: Mapping[str, Any],
        environment_checks: Sequence[CheckResult],
    ) -> list[CheckResult]:
        blockers = {
            check.id: check
            for check in environment_checks
            if check.status in {CHECK_FAIL, CHECK_BLOCKED}
        }
        results: list[CheckResult] = []
        for constraint in spec.constraints:
            result = evaluate_constraint(constraint, spec, plan, evidence)
            if result.status != CHECK_UNVERIFIABLE:
                results.append(result)
                continue
            blocker_id = TravelReward._constraint_blocker(constraint.kind, blockers)
            if blocker_id is None:
                results.append(result)
                continue
            results.append(
                CheckResult(
                    id=result.id,
                    source=result.source,
                    hardness=result.hardness,
                    status=CHECK_BLOCKED,
                    message=f"Goal check is blocked by {blocker_id}.",
                    evidence=result.evidence,
                    owner_dimension=DIMENSION_GOAL,
                    score=None,
                    blocked_by=blocker_id,
                )
            )
        return results

    @staticmethod
    def _constraint_blocker(kind: str, blockers: Mapping[str, CheckResult]) -> str | None:
        candidates = {
            "total_budget": ("cost_accounting",),
            "category_budget": ("cost_accounting", "entity_grounding"),
            "transport_mode": ("route_grounding", "entity_grounding"),
            "entity_category": ("entity_grounding",),
            "entity_attribute": ("entity_grounding",),
            "include_entity": ("entity_grounding",),
            "exclude_entity": ("entity_grounding",),
            "room_count": ("plan_structure",),
            "room_type": ("entity_grounding", "plan_structure"),
            "time_window": ("entity_grounding", "chronology"),
            "activity_count": ("plan_structure",),
        }.get(kind, ())
        return next((check_id for check_id in candidates if check_id in blockers), None)

    @staticmethod
    def _materialize_submission(
        spec: TravelTaskSpec,
        raw_plan: Mapping[str, Any],
        candidates: Mapping[str, Any],
        routes: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        declared_people = raw_plan.get("people_number")
        people = declared_people if isinstance(declared_people, int) else 0
        itinerary = raw_plan.get("itinerary")
        days = itinerary if isinstance(itinerary, Sequence) else []
        activities: list[dict[str, Any]] = []
        used_entities: dict[str, dict[str, Any]] = {}
        used_routes: dict[str, dict[str, Any]] = {}
        cost_items: list[dict[str, Any]] = []
        for day_position, raw_day in enumerate(days):
            day_payload = raw_day if isinstance(raw_day, Mapping) else {}
            day = day_payload.get("day")
            raw_activities = day_payload.get("activities")
            day_activities = (
                raw_activities
                if isinstance(raw_activities, Sequence)
                and not isinstance(raw_activities, (str, bytes))
                else []
            )
            for activity_index, raw_activity in enumerate(day_activities):
                activity = raw_activity if isinstance(raw_activity, Mapping) else {}
                candidate_id = activity.get("candidate_id")
                candidate = candidates.get(candidate_id) if isinstance(candidate_id, str) else None
                candidate = candidate if isinstance(candidate, Mapping) else {}
                entity = candidate.get("evidence")
                entity = dict(entity) if isinstance(entity, Mapping) else {}
                if isinstance(candidate_id, str) and entity:
                    used_entities[candidate_id] = entity
                activity_type = activity.get("type")
                start_clock = _minutes(activity.get("start_time"))
                end_clock = _minutes(activity.get("end_time"), allow_24=True)
                overnight = (
                    activity_type in {"train", "airplane"}
                    and start_clock is not None
                    and end_clock is not None
                    and end_clock <= start_clock
                )
                absolute_start = (
                    (day - 1) * 24 * 60 + start_clock
                    if isinstance(day, int) and start_clock is not None
                    else None
                )
                absolute_end = (
                    (day - 1 + int(overnight)) * 24 * 60 + end_clock
                    if isinstance(day, int) and end_clock is not None
                    else None
                )
                if activity_type in {"train", "airplane"}:
                    origin_position_id = entity.get("origin_anchor_id")
                    destination_position_id = entity.get("destination_anchor_id")
                else:
                    origin_position_id = candidate_id
                    destination_position_id = candidate_id
                rooms = activity.get("rooms")
                quantity = rooms if activity_type == "accommodation" else people
                price_key = "cost" if activity_type in {"train", "airplane"} else "price"
                unit_price = _number(entity.get(price_key))
                if activity_type == "breakfast" and entity.get("entity_type") == "hotel":
                    unit_price = 0.0
                amount = (
                    round(float(quantity) * unit_price, 2)
                    if isinstance(quantity, int) and unit_price is not None
                    else None
                )
                normalized = {
                    "day": day if isinstance(day, int) else day_position + 1,
                    "activity_index": activity_index,
                    "candidate_id": candidate_id,
                    "entity_type": entity.get("entity_type"),
                    "activity_type": activity_type,
                    "start_time": activity.get("start_time"),
                    "end_time": activity.get("end_time"),
                    "absolute_start": absolute_start,
                    "absolute_end": absolute_end,
                    "origin_position_id": origin_position_id,
                    "destination_position_id": destination_position_id,
                    "route_from_previous_id": activity.get("route_from_previous_id"),
                    "rooms": rooms,
                    "room_type": activity.get("room_type"),
                    "derived_quantity": quantity,
                    "unit_price": unit_price,
                    "amount": amount,
                }
                activities.append(normalized)
                cost_items.append(
                    {
                        "kind": "activity",
                        "day": normalized["day"],
                        "activity_index": activity_index,
                        "candidate_id": candidate_id,
                        "activity_type": activity_type,
                        "quantity": quantity,
                        "unit_price": unit_price,
                        "amount": amount,
                    }
                )
                route_id = activity.get("route_from_previous_id")
                route = routes.get(route_id) if isinstance(route_id, str) else None
                if isinstance(route_id, str) and isinstance(route, Mapping):
                    used_routes[route_id] = dict(route)
                    route_item = TravelReward._raw_route_cost_item(
                        route, people, normalized["day"], activity_index
                    )
                    cost_items.append(route_item)
        amounts = [_number(item.get("amount")) for item in cost_items]
        total_cost = (
            round(sum(float(amount) for amount in amounts if amount is not None), 2)
            if amounts and all(amount is not None for amount in amounts)
            else None
        )
        snapshot = {
            "schema_version": "travelweaver-plan-snapshot-v2",
            "task_id": spec.task_id,
            "people_number": raw_plan.get("people_number"),
            "start_city": raw_plan.get("start_city"),
            "target_city": raw_plan.get("target_city"),
            "days": len(days),
            "activities": activities,
            "total_cost": total_cost,
        }
        evidence = {
            "schema_version": "travelweaver-evidence-v3",
            "task_id": spec.task_id,
            "entities": used_entities,
            "routes": used_routes,
            "candidate_usages": {
                candidate_id: {
                    "candidate_id": candidate_id,
                    "entity_type": candidate.get("entity_type"),
                    "purpose": candidate.get("purpose"),
                }
                for candidate_id, candidate in candidates.items()
                if candidate_id in used_entities and isinstance(candidate, Mapping)
            },
            "cost_items": cost_items,
            "total_cost": total_cost,
            "quantity_rules_version": "travelweaver-quantity-rules-v1",
        }
        return snapshot, evidence

    @staticmethod
    def _raw_route_cost_item(
        route: Mapping[str, Any], people: int, day: Any, activity_index: int
    ) -> dict[str, Any]:
        mode = str(route.get("mode"))
        quantity = math.ceil(people / 4) if mode == "taxi" else people if mode == "metro" else 1
        segments = route.get("segments")
        segments = segments if isinstance(segments, Sequence) else []
        prices = [
            _number(segment.get("cost"))
            for segment in segments
            if isinstance(segment, Mapping)
        ]
        verifiable = bool(prices) and all(price is not None for price in prices)
        unit_price = round(sum(float(price) for price in prices if price is not None), 2)
        return {
            "kind": "route",
            "day": day,
            "activity_index": activity_index,
            "route_id": route.get("route_id"),
            "mode": mode,
            "quantity": quantity,
            "unit_price": unit_price if verifiable else None,
            "amount": round(quantity * unit_price, 2) if verifiable else None,
        }

    @staticmethod
    def _entity_uniqueness(
        activities: list[dict[str, Any]],
        entities: Mapping[str, Any],
        *,
        affects_success: bool,
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
            affects_success=affects_success,
        )

    @staticmethod
    def _meal_commonsense(
        activities: list[dict[str, Any]],
        entities: Mapping[str, Any],
        *,
        affects_success: bool,
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
            affects_success=affects_success,
        )

    @staticmethod
    def _task_alignment(spec: TravelTaskSpec, plan: Mapping[str, Any]) -> CheckResult:
        actual = {
            "people_number": plan.get("people_number"),
            "start_city": plan.get("start_city"),
            "target_city": plan.get("target_city"),
            "days": plan.get("days"),
        }
        expected = {
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
            owner_dimension=DIMENSION_GOAL,
            score=sum(actual[key] == value for key, value in expected.items()) / len(expected),
        )

    @staticmethod
    def _structure(spec: TravelTaskSpec, activities: list[dict[str, Any]]) -> CheckResult:
        del spec
        days: set[int] = set()
        valid_items = 0
        attraction_count = 0
        for activity in activities:
            day = activity.get("day")
            candidate_id = activity.get("candidate_id")
            activity_type = activity.get("activity_type")
            start = _minutes(activity.get("start_time"))
            end = _minutes(activity.get("end_time"), allow_24=True)
            locally_valid = (
                isinstance(day, int)
                and day > 0
                and isinstance(candidate_id, str)
                and bool(candidate_id)
                and isinstance(activity_type, str)
                and start is not None
                and end is not None
            )
            if locally_valid:
                valid_items += 1
                days.add(day)
            attraction_count += activity.get("activity_type") == "attraction"
        contiguous = bool(days) and sorted(days) == list(range(1, max(days) + 1))
        valid = bool(activities) and valid_items == len(activities) and contiguous
        item_score = valid_items / len(activities) if activities else 0.0
        score = (item_score + float(contiguous)) / 2
        return _env_check(
            "plan_structure",
            CHECK_PASS if valid else CHECK_FAIL,
            "Activities are locally well formed and use contiguous declared day numbers.",
            {"days": sorted(days), "attraction_count": attraction_count},
            owner_dimension=DIMENSION_ARTIFACT,
            score=score,
        )

    @staticmethod
    def _chronology(activities: list[dict[str, Any]]) -> CheckResult:
        ordered = sorted(
            activities,
            key=lambda item: (int(item.get("day", 0)), int(item.get("activity_index", 0))),
        )
        valid_pairs = 0
        total = max(1, len(ordered))
        previous_end: int | None = None
        for activity in ordered:
            start = activity.get("absolute_start")
            end = activity.get("absolute_end")
            own_valid = isinstance(start, int) and isinstance(end, int) and end > start
            if own_valid and (previous_end is None or start >= previous_end):
                valid_pairs += 1
            previous_end = end if isinstance(end, int) else previous_end
        score = valid_pairs / total
        return _env_check(
            "chronology",
            CHECK_PASS if score == 1.0 and bool(ordered) else CHECK_FAIL,
            "Activities have valid, non-overlapping absolute time intervals.",
            {"valid_intervals": valid_pairs, "total_intervals": len(ordered)},
            score=score,
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
        score = 1.0 - len(missing) / len(activities) if activities else 1.0
        return _env_check(
            "entity_grounding",
            CHECK_PASS if not missing else CHECK_FAIL,
            "Every submitted activity is grounded in environment evidence.",
            {"missing_candidate_ids": missing},
            score=score,
        )

    @staticmethod
    def _candidate_usage(
        activities: list[dict[str, Any]],
        entities: Mapping[str, Any],
        evidence: Mapping[str, Any],
    ) -> CheckResult:
        usages = evidence.get("candidate_usages")
        if usages is None:
            return _env_check(
                "candidate_usage",
                CHECK_NOT_APPLICABLE,
                "Legacy evidence has no saved-candidate usage metadata.",
                {"evidence_schema_version": evidence.get("schema_version")},
                affects_success=False,
                affects_shaping=False,
            )
        if not isinstance(usages, Mapping):
            return _env_check(
                "candidate_usage",
                CHECK_UNVERIFIABLE,
                "Candidate usage metadata is malformed.",
                {"candidate_usages_type": type(usages).__name__},
            )

        expected_types: dict[str, str | set[str]] = {
            "attraction": "attraction",
            "breakfast": {"restaurant", "hotel"},
            "lunch": "restaurant",
            "dinner": "restaurant",
            "accommodation": "hotel",
            "train": "train",
            "airplane": "airplane",
        }
        invalid: list[dict[str, Any]] = []
        missing: list[str] = []
        applicable = 0
        valid = 0
        for activity in activities:
            candidate_id = activity.get("candidate_id")
            if candidate_id not in entities:
                continue
            applicable += 1
            usage = usages.get(candidate_id)
            entity = entities.get(candidate_id)
            if not isinstance(usage, Mapping) or not isinstance(entity, Mapping):
                missing.append(str(candidate_id))
                continue
            activity_type = str(activity.get("activity_type"))
            actual_type = usage.get("entity_type")
            expected_type = expected_types.get(activity_type)
            type_ok = (
                actual_type in expected_type
                if isinstance(expected_type, set)
                else expected_type is not None and actual_type == expected_type
            )
            purpose = usage.get("purpose")
            if activity_type == "attraction":
                allowed_purposes = {"attraction"}
            elif activity_type in {"lunch", "dinner"}:
                allowed_purposes = {"meal"}
            elif activity_type == "breakfast":
                allowed_purposes = (
                    {"hotel", "meal"}
                    if entity.get("entity_type") == "hotel"
                    else {"meal"}
                )
            elif activity_type == "accommodation":
                allowed_purposes = {"hotel"}
            elif activity_type in {"train", "airplane"}:
                allowed_purposes = {"outbound_transport", "return_transport"}
            else:
                allowed_purposes = set()
            purpose_ok = purpose in allowed_purposes
            location_ok = (
                bool(entity.get("origin_anchor_id") and entity.get("destination_anchor_id"))
                if activity_type in {"train", "airplane"}
                else isinstance(entity.get("city"), str) and bool(entity.get("city"))
            )
            if type_ok and purpose_ok and location_ok:
                valid += 1
            else:
                invalid.append(
                    {
                        "candidate_id": candidate_id,
                        "activity_type": activity_type,
                        "entity_type": actual_type,
                        "purpose": purpose,
                        "type_ok": type_ok,
                        "purpose_ok": purpose_ok,
                        "location_ok": location_ok,
                    }
                )
        if missing:
            status = CHECK_UNVERIFIABLE
            score = None
        else:
            score = valid / applicable if applicable else 1.0
            status = CHECK_PASS if not invalid else CHECK_FAIL
        return _env_check(
            "candidate_usage",
            status,
            "Saved candidate type, declared purpose, and location metadata match its use.",
            {"invalid": invalid, "missing_usage_metadata": missing},
            score=score,
        )

    @staticmethod
    def _intercity_time(
        activities: list[dict[str, Any]], entities: Mapping[str, Any]
    ) -> CheckResult:
        failures: list[dict[str, Any]] = []
        missing: list[str] = []
        applicable = 0
        for activity in activities:
            if activity.get("activity_type") not in {"train", "airplane"}:
                continue
            applicable += 1
            candidate_id = activity.get("candidate_id")
            entity = entities.get(candidate_id)
            if not isinstance(entity, Mapping):
                missing.append(str(candidate_id))
                continue
            expected_start = entity.get("departure_time")
            expected_end = entity.get("arrival_time")
            start_ok = expected_start is None or activity.get("start_time") == expected_start
            end_ok = expected_end is None or activity.get("end_time") == expected_end
            if not start_ok or not end_ok:
                failures.append(
                    {
                        "candidate_id": candidate_id,
                        "expected_start": expected_start,
                        "actual_start": activity.get("start_time"),
                        "expected_end": expected_end,
                        "actual_end": activity.get("end_time"),
                    }
                )
        if missing:
            status = CHECK_UNVERIFIABLE
            score = None
        else:
            score = 1.0 - len(failures) / max(1, applicable)
            status = CHECK_PASS if not failures else CHECK_FAIL
        return _env_check(
            "intercity_time",
            status,
            "Intercity activity times match frozen candidate evidence.",
            {"failures": failures, "missing_entities": missing},
            score=score,
        )

    @staticmethod
    def _route_grounding(
        activities: list[dict[str, Any]],
        routes: Mapping[str, Any],
        *,
        evidence_schema_version: str,
    ) -> CheckResult:
        if evidence_schema_version == "travelweaver-evidence-v1":
            return _env_check(
                "route_grounding",
                CHECK_NOT_APPLICABLE,
                "Legacy evidence predates explicit route references.",
                {"evidence_schema_version": evidence_schema_version},
                affects_success=False,
                affects_shaping=False,
            )
        missing = []
        mismatched = []
        unverifiable = []
        required_count = 0
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
                required_count += 1
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
        failed = len(missing) + len(mismatched)
        score = max(0.0, 1.0 - failed / max(1, required_count))
        if unverifiable:
            status = CHECK_UNVERIFIABLE
            score = None
        else:
            status = CHECK_PASS if not missing and not mismatched else CHECK_FAIL
        return _env_check(
            "route_grounding",
            status,
            "Every required local connection has matching, feasible route evidence.",
            {"missing": missing, "mismatched": mismatched, "unverifiable": unverifiable},
            score=score,
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
        applicable = sum(
            item.get("activity_type") in {"attraction", "breakfast", "lunch", "dinner"}
            for item in activities
        )
        status = CHECK_UNVERIFIABLE if unverifiable else CHECK_FAIL if failures else CHECK_PASS
        score = None if unverifiable else 1.0 - len(failures) / max(1, applicable)
        return _env_check(
            "opening_hours",
            status,
            "Attractions and meals fit their recorded opening hours.",
            {"failures": failures, "unverifiable": unverifiable},
            score=score,
        )

    @staticmethod
    def _trip_coverage(
        spec: TravelTaskSpec,
        activities: list[dict[str, Any]],
        entities: Mapping[str, Any],
    ) -> CheckResult:
        missing_entities = sorted(
            str(item.get("candidate_id"))
            for item in activities
            if item.get("candidate_id") not in entities
        )
        if missing_entities:
            return _env_check(
                "trip_coverage",
                CHECK_UNVERIFIABLE,
                "Trip-goal geography requires grounded candidate evidence.",
                {"missing_candidate_ids": missing_entities},
                owner_dimension=DIMENSION_GOAL,
                score=None,
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
        attraction_score = float(
            any(item.get("activity_type") == "attraction" for item in activities)
        )
        required_nights = max(0, spec.trip.days - 1)
        accommodations = sum(
            item.get("activity_type") == "accommodation" for item in activities
        )
        night_score = min(accommodations / required_nights, 1.0) if required_nights else 1.0
        direction_score = (
            len(required_directions & actual_directions) / len(required_directions)
            if required_directions
            else 1.0
        )
        local_activities = [
            item
            for item in activities
            if item.get("activity_type") not in {"train", "airplane"}
        ]
        local_destination_matches = sum(
            isinstance((entity := entities.get(item.get("candidate_id"))), Mapping)
            and entity.get("city") == destination
            for item in local_activities
        )
        destination_score = (
            local_destination_matches / len(local_activities) if local_activities else 0.0
        )
        boundary_score = 1.0
        ordered = sorted(
            activities,
            key=lambda item: (int(item.get("day", 0)), int(item.get("activity_index", 0))),
        )
        if required_directions and ordered:
            first_entity = entities.get(ordered[0].get("candidate_id"), {})
            last_entity = entities.get(ordered[-1].get("candidate_id"), {})
            first_ok = isinstance(first_entity, Mapping) and (
                str(first_entity.get("origin_city")),
                str(first_entity.get("destination_city")),
            ) == (spec.trip.origin, destination)
            last_ok = isinstance(last_entity, Mapping) and (
                str(last_entity.get("origin_city")),
                str(last_entity.get("destination_city")),
            ) == (destination, spec.trip.origin)
            boundary_score = (float(first_ok) + float(last_ok)) / 2
        transport_score = min(direction_score, boundary_score)
        valid = all(
            component == 1.0
            for component in (
                attraction_score,
                night_score,
                transport_score,
                destination_score,
            )
        )
        score = (
            attraction_score + night_score + transport_score + destination_score
        ) / 4
        return _env_check(
            "trip_coverage",
            CHECK_PASS if valid else CHECK_FAIL,
            "The plan covers required content, nights, destination, and trip boundaries.",
            {
                "required_directions": sorted(required_directions),
                "actual_directions": sorted(actual_directions),
                "required_nights": required_nights,
                "accommodation_nights": accommodations,
                "local_activity_count": len(local_activities),
                "local_destination_matches": local_destination_matches,
                "has_attraction": bool(attraction_score),
                "first_activity": ordered[0].get("candidate_id") if ordered else None,
                "last_activity": ordered[-1].get("candidate_id") if ordered else None,
            },
            owner_dimension=DIMENSION_GOAL,
            score=score,
        )

    @staticmethod
    def _quantity_consistency(
        plan: Mapping[str, Any],
        activities: list[dict[str, Any]],
        evidence: Mapping[str, Any],
    ) -> CheckResult:
        declared_travelers = plan.get("people_number")
        travelers = (
            declared_travelers
            if isinstance(declared_travelers, int)
            and not isinstance(declared_travelers, bool)
            and declared_travelers > 0
            else 0
        )
        failures = []
        for activity in activities:
            activity_type = activity.get("activity_type")
            if activity_type == "accommodation":
                rooms = activity.get("rooms")
                room_type = activity.get("room_type")
                if not isinstance(rooms, int) or not isinstance(room_type, int):
                    failures.append(activity.get("candidate_id"))
                elif rooms * room_type < travelers:
                    failures.append(activity.get("candidate_id"))
                elif activity.get("derived_quantity") != rooms:
                    failures.append(activity.get("candidate_id"))
            elif activity.get("derived_quantity") != travelers:
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
                    math.ceil(travelers / 4)
                    if mode == "taxi"
                    else travelers
                    if mode == "metro"
                    else 1
                )
                if item.get("quantity") != expected:
                    failures.append(item.get("route_id"))
        applicable = len(activities) + sum(
            isinstance(item, Mapping) and item.get("kind") == "route"
            for item in (cost_items if isinstance(cost_items, Sequence) else [])
        )
        score = max(0.0, 1.0 - len(failures) / max(1, applicable))
        return _env_check(
            "quantity_consistency",
            CHECK_PASS if not failures else CHECK_FAIL,
            "Tickets and rooms are internally consistent with declared travelers.",
            {"failures": failures, "declared_travelers": travelers},
            score=score,
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
        total_matches = math.isclose(calculated, total, abs_tol=1e-6)
        keys_match = actual_activity_keys == expected_activity_keys
        passed = (
            total_matches
            and not multiplication_failures
            and keys_match
        )
        score = (
            float(total_matches)
            + float(not multiplication_failures)
            + float(keys_match)
        ) / 3
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
            score=score,
        )

    @staticmethod
    def _overnight_coverage(
        plan: Mapping[str, Any], activities: list[dict[str, Any]]
    ) -> CheckResult:
        declared_days = plan.get("days")
        days = declared_days if isinstance(declared_days, int) and declared_days > 0 else 0
        counts = {
            day: sum(
                item.get("day") == day and item.get("activity_type") == "accommodation"
                for item in activities
            )
            for day in range(1, days + 1)
        }
        expected = {day: int(day < days) for day in counts}
        matched = sum(counts[day] == expected[day] for day in counts)
        score = matched / len(counts) if counts else 0.0
        return _env_check(
            "overnight_coverage",
            CHECK_PASS if score == 1.0 else CHECK_FAIL,
            "Each non-final day has one overnight stay and the final day has none.",
            {"expected": expected, "actual": counts},
            score=score,
            affects_success=False,
            affects_shaping=True,
        )
