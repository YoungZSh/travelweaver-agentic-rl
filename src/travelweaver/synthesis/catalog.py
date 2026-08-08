"""Deterministic, quota-driven synthesis slots for arbitrary pilot sizes."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import TypeVar

from .models import PilotSlot
from .randomness import deterministic_rng

_T = TypeVar("_T")

_CITIES = ("上海", "北京", "南京", "广州", "成都", "杭州", "武汉", "深圳", "苏州", "重庆")
_MODE_WEIGHTS = {
    ("train", "train"): 0.4,
    ("airplane", "airplane"): 0.2,
    ("train", "airplane"): 0.2,
    ("airplane", "train"): 0.2,
}
_DAY_WEIGHTS = {1: 0.10, 2: 0.30, 3: 0.35, 4: 0.15, 5: 0.10}
_TRAVELER_WEIGHTS = {1: 0.225, 2: 0.225, 3: 0.225, 4: 0.225, 5: 0.05, 6: 0.05}
_CONSTRAINT_WEIGHTS = {1: 0.10, 2: 0.20, 3: 0.30, 4: 0.25, 5: 0.10, 6: 0.05}
_ROUTE_WEIGHTS = {"taxi": 0.4, "metro": 0.35, "walk": 0.25}
_STRATEGY_WEIGHTS = {"early": 0.25, "cheap": 0.25, "short": 0.25, "balanced": 0.25}
_DIFFICULTY_WEIGHTS = {"easy": 0.25, "medium": 0.50, "hard": 0.25}
_SCENARIO_WEIGHTS = {
    "normal": 0.70,
    "poi_closure": 0.08,
    "hotel_unavailable": 0.06,
    "transport_cancellation": 0.08,
    "price_change": 0.08,
}
_SURFACE_STYLE_WEIGHTS = {
    "direct": 0.1,
    "conversational": 0.1,
    "trip_first": 0.1,
    "party_first": 0.1,
    "concise": 0.1,
    "consultant": 0.1,
    "narrative": 0.1,
    "itinerary": 0.1,
    "question": 0.1,
    "compact": 0.1,
}
BLENDED_PROFILE = "chinatravel_blended_v1"
BLENDED_V1_1_PROFILE = "chinatravel_blended_v1_1"
DEFAULT_PROFILE = "pilot_v2_1"
SUPPORTED_PROFILES = (DEFAULT_PROFILE, BLENDED_PROFILE, BLENDED_V1_1_PROFILE)

_BLENDED_TYPE_QUOTAS = {
    "easy_like": 50,
    "medium_like": 70,
    "human_like": 50,
    "preference_like": 20,
    "generalization": 10,
}
_BLENDED_CONSTRAINT_QUOTAS = {
    "easy_like": {1: 28, 2: 15, 3: 7},
    "medium_like": {3: 10, 4: 18, 5: 18, 6: 14, 7: 7, 8: 3},
    "human_like": {0: 5, 1: 10, 2: 13, 3: 12, 4: 6, 5: 3, 6: 1},
    "preference_like": {1: 10, 2: 7, 3: 3},
    "generalization": {1: 1, 2: 2, 3: 3, 4: 2, 5: 1, 6: 1},
}
_BLENDED_SCENARIO_QUOTAS = {
    "normal": 180,
    "poi_closure": 5,
    "hotel_unavailable": 4,
    "transport_cancellation": 5,
    "price_change": 6,
}
_OFFICIAL_PREFERENCES = (
    "more_attractions",
    "less_innercity_time",
    "shorter_meal_transfer",
    "higher_dining_share",
    "lower_lodging_share",
    "near_poi",
)
_EXTENDED_PREFERENCES = (
    "less_walking",
    "lower_total_cost",
    "relaxed_itinerary",
    "higher_attraction_share",
    "lower_intercity_share",
    "shorter_total_travel_time",
)
_HUMAN_PREFERENCES = (
    "relaxed_itinerary",
    "less_walking",
    "lower_total_cost",
    "more_attractions",
    "higher_dining_share",
    "lower_lodging_share",
)
_BASE_KEYS = (
    "total_budget",
    "attraction_category",
    "include_attraction",
    "attraction_count",
    "outbound_time",
    "return_time",
    "restaurant_cuisine",
    "restaurant_budget",
    "include_restaurant",
    "innercity_mode",
    "hotel_attribute",
    "hotel_budget",
    "room_type",
    "room_count",
    "include_hotel",
    "all_intercity_mode",
    "outbound_mode",
    "return_mode",
)
_HOTEL_KEYS = {
    "hotel_attribute",
    "hotel_budget",
    "room_type",
    "room_count",
    "include_hotel",
}
_INTERCITY_MODE_KEYS = {"all_intercity_mode", "outbound_mode", "return_mode"}
_CHINATRAVEL_CONSTRAINT_WEIGHTS = {
    "total_budget": 0.10,
    "attraction_category": 0.07,
    "include_attraction": 0.08,
    "attraction_count": 0.06,
    "outbound_time": 0.06,
    "return_time": 0.06,
    "restaurant_cuisine": 0.05,
    "restaurant_budget": 0.05,
    "include_restaurant": 0.05,
    "innercity_mode": 0.05,
    "hotel_attribute": 0.05,
    "hotel_budget": 0.06,
    "room_type": 0.04,
    "room_count": 0.04,
    "include_hotel": 0.05,
    "all_intercity_mode": 0.03,
    "outbound_mode": 0.05,
    "return_mode": 0.05,
}


def build_pilot_slots(
    count: int,
    seed: int,
    profile: str = DEFAULT_PROFILE,
) -> tuple[PilotSlot, ...]:
    """Build balanced slots using one seed and isolated deterministic random streams."""

    if count <= 0:
        raise ValueError("Synthesis count must be positive.")
    if profile in {BLENDED_PROFILE, BLENDED_V1_1_PROFILE}:
        return _build_blended_slots(count, seed, profile)
    if profile != DEFAULT_PROFILE:
        raise ValueError(f"Unknown synthesis profile: {profile}")

    destinations = _quota_values(
        count,
        {city: 1.0 for city in _CITIES},
        seed=seed,
        scope="destinations",
    )
    origins = _balanced_origins(destinations, seed)
    days = _quota_values(count, _DAY_WEIGHTS, seed=seed, scope="days")
    travelers = _quota_values(count, _TRAVELER_WEIGHTS, seed=seed, scope="travelers")
    constraint_counts = _quota_values(
        count,
        _CONSTRAINT_WEIGHTS,
        seed=seed,
        scope="constraint-counts",
    )
    patterns = _transport_patterns(destinations, seed)
    route_modes = _quota_values(count, _ROUTE_WEIGHTS, seed=seed, scope="route-modes")
    strategies = _quota_values(
        count,
        _STRATEGY_WEIGHTS,
        seed=seed,
        scope="transport-strategies",
    )
    tightnesses = _quota_values(
        count,
        _DIFFICULTY_WEIGHTS,
        seed=seed,
        scope="difficulties",
    )
    scenario_profiles = _quota_values(
        count,
        _SCENARIO_WEIGHTS,
        seed=seed,
        scope="scenario-profiles",
    )
    surface_styles = _quota_values(
        count,
        _SURFACE_STYLE_WEIGHTS,
        seed=seed,
        scope="surface-styles",
    )
    rich_flags = _quota_values(
        count,
        {False: 0.65, True: 0.35},
        seed=seed,
        scope="rich-itineraries",
    )
    meal_flags = _quota_values(
        count,
        {False: 0.55, True: 0.45},
        seed=seed,
        scope="meal-itineraries",
    )

    # Long trips remain below the 35-step rollout budget by using one attraction per day.
    attractions_per_day = [
        2 if rich and day <= 3 else 1
        for rich, day in zip(rich_flags, days, strict=True)
    ]
    for index, profile in enumerate(scenario_profiles):
        if profile == "hotel_unavailable" and days[index] == 1:
            scenario_profiles[index] = "poi_closure"

    recipe_usage: Counter[str] = Counter()
    pair_usage: Counter[tuple[str, str]] = Counter()
    slots: list[PilotSlot] = []
    for index in range(count):
        outbound_mode, return_mode = patterns[index]
        recipe = _recipe(
            index=index,
            seed=seed,
            days=days[index],
            constraint_count=constraint_counts[index],
            outbound_mode=outbound_mode,
            return_mode=return_mode,
            usage=recipe_usage,
            pair_usage=pair_usage,
        )
        recipe_usage.update(recipe)
        for left_index, left in enumerate(sorted(recipe)):
            for right in sorted(recipe)[left_index + 1 :]:
                pair_usage[(left, right)] += 1
        include_meal = meal_flags[index] or any(
            key.startswith("restaurant")
            or key in {"include_restaurant", "innercity_mode"}
            for key in recipe
        )
        slots.append(
            PilotSlot(
                index=index,
                origin=origins[index],
                destination=destinations[index],
                days=days[index],
                travelers=travelers[index],
                outbound_mode=outbound_mode,
                return_mode=return_mode,
                constraint_count=constraint_counts[index],
                recipe=recipe,
                attractions_per_day=attractions_per_day[index],
                include_meal=include_meal,
                route_mode=route_modes[index],
                transport_strategy=strategies[index],
                tightness=tightnesses[index],
                scenario_profile=scenario_profiles[index],
                surface_style=surface_styles[index],
                synthesis_profile=profile,
            )
        )
    _validate_slots(slots)
    return tuple(slots)


def _build_blended_slots(count: int, seed: int, profile: str) -> tuple[PilotSlot, ...]:
    if count != 200:
        raise ValueError(f"{profile} is a frozen 200-task profile.")
    task_types = _exact_values(_BLENDED_TYPE_QUOTAS, seed, "blended-task-types")
    constraint_counts = _values_by_type(
        task_types,
        _BLENDED_CONSTRAINT_QUOTAS,
        seed,
        "blended-constraint-counts",
    )
    destinations = _quota_values(
        count,
        _mixed_prior({city: 1.0 for city in _CITIES}),
        seed=seed,
        scope="blended-destinations",
    )
    origins = _balanced_origins(destinations, seed)
    if profile == BLENDED_V1_1_PROFILE:
        days = _v1_1_days(task_types, seed)
        travelers = _v1_1_travelers(task_types, seed)
    else:
        days = _quota_values(
            count,
            _mixed_prior(_DAY_WEIGHTS),
            seed=seed,
            scope="blended-days",
        )
        travelers = _quota_values(
            count,
            _mixed_prior(_TRAVELER_WEIGHTS),
            seed=seed,
            scope="blended-travelers",
        )
    patterns = _transport_patterns(destinations, seed)
    route_modes = _quota_values(count, _ROUTE_WEIGHTS, seed=seed, scope="blended-routes")
    strategies = _quota_values(
        count, _STRATEGY_WEIGHTS, seed=seed, scope="blended-strategies"
    )
    scenarios = _exact_values(_BLENDED_SCENARIO_QUOTAS, seed, "blended-scenarios")
    _move_hotel_scenarios_to_multiday(scenarios, days)
    tightnesses = _blended_tightnesses(task_types, seed)
    styles = _blended_styles(task_types, seed, profile)
    metadata_flags = _human_metadata_flags(task_types, seed)
    human_preference_counts = _human_preference_counts(task_types, seed)
    preference_like_kinds = _preference_like_kinds(task_types, seed)
    _ensure_preference_day_compatibility(task_types, preference_like_kinds, days)
    _move_hotel_scenarios_to_multiday(scenarios, days)

    recipe_usage: Counter[str] = Counter()
    pair_usage: Counter[tuple[str, str]] = Counter()
    slots: list[PilotSlot] = []
    for index, task_type in enumerate(task_types):
        outbound_mode, return_mode = patterns[index]
        if task_type == "preference_like":
            recipe = _preference_recipe(
                constraint_counts[index],
                seed,
                index,
                preference_like_kinds[index][0],
                profile,
            )
        else:
            recipe = _recipe(
                index=index,
                seed=seed,
                days=days[index],
                constraint_count=constraint_counts[index],
                outbound_mode=outbound_mode,
                return_mode=return_mode,
                usage=recipe_usage,
                pair_usage=pair_usage,
                family_weights=_mixed_prior(_CHINATRAVEL_CONSTRAINT_WEIGHTS),
            )
        recipe_usage.update(recipe)
        for left_index, left in enumerate(sorted(recipe)):
            for right in sorted(recipe)[left_index + 1 :]:
                pair_usage[(left, right)] += 1
        persona = _persona(travelers[index], seed, index) if task_type == "human_like" else None
        metadata_prefix = (
            _metadata_prefix(
                persona,
                seed,
                index,
                profile=profile,
                origin=origins[index],
                destination=destinations[index],
                travelers=travelers[index],
                days=days[index],
            )
            if persona is not None and metadata_flags[index]
            else None
        )
        preferences = preference_like_kinds[index]
        if task_type == "human_like":
            preferences = (
                _pick_human_preferences_v1_1(
                    human_preference_counts[index],
                    seed,
                    index,
                    days=days[index],
                    recipe=recipe,
                    route_mode=route_modes[index],
                )
                if profile == BLENDED_V1_1_PROFILE
                else _pick_human_preferences(human_preference_counts[index], seed, index)
            )
        include_meal = any(
            key.startswith("restaurant")
            or key in {"include_restaurant", "innercity_mode"}
            for key in recipe
        ) or any(
            kind in {"shorter_meal_transfer", "higher_dining_share"}
            for kind in preferences
        )
        slots.append(
            PilotSlot(
                index=index,
                origin=origins[index],
                destination=destinations[index],
                days=days[index],
                travelers=travelers[index],
                outbound_mode=outbound_mode,
                return_mode=return_mode,
                constraint_count=constraint_counts[index],
                recipe=recipe,
                attractions_per_day=1,
                include_meal=include_meal,
                route_mode=route_modes[index],
                transport_strategy=strategies[index],
                tightness=tightnesses[index],
                scenario_profile=scenarios[index],
                surface_style=styles[index],
                synthesis_profile=profile,
                task_type=task_type,
                validation_profile=_validation_profile(task_type, seed, index, profile),
                persona_context=persona,
                metadata_prefix=metadata_prefix,
                preference_kinds=preferences,
            )
        )
    _validate_slots(slots)
    _validate_blended_distribution(slots)
    return tuple(slots)


def _mixed_prior(weights: Mapping[_T, float]) -> dict[_T, float]:
    total = sum(weights.values())
    uniform = 1.0 / len(weights)
    return {
        value: 0.65 * weight / total + 0.35 * uniform
        for value, weight in weights.items()
    }


def _exact_values(quotas: Mapping[_T, int], seed: int, scope: str) -> list[_T]:
    values = [value for value, amount in quotas.items() for _ in range(amount)]
    deterministic_rng(seed, scope).shuffle(values)
    return values


def _values_by_type(
    task_types: Sequence[str],
    quotas: Mapping[str, Mapping[_T, int]],
    seed: int,
    scope: str,
) -> list[_T]:
    pools = {
        task_type: _exact_values(values, seed, f"{scope}-{task_type}")
        for task_type, values in quotas.items()
    }
    offsets: Counter[str] = Counter()
    result: list[_T] = []
    for task_type in task_types:
        result.append(pools[task_type][offsets[task_type]])
        offsets[task_type] += 1
    return result


def _move_hotel_scenarios_to_multiday(scenarios: list[str], days: Sequence[int]) -> None:
    invalid = [
        index
        for index, (scenario, day) in enumerate(zip(scenarios, days, strict=True))
        if scenario == "hotel_unavailable" and day == 1
    ]
    replacements = [
        index
        for index, (scenario, day) in enumerate(zip(scenarios, days, strict=True))
        if scenario == "normal" and day > 1
    ]
    for left, right in zip(invalid, replacements, strict=False):
        scenarios[left], scenarios[right] = scenarios[right], scenarios[left]


def _blended_tightnesses(task_types: Sequence[str], seed: int) -> list[str]:
    weights = {
        "easy_like": {"easy": 0.8, "medium": 0.2},
        "medium_like": {"easy": 0.1, "medium": 0.65, "hard": 0.25},
        "human_like": _DIFFICULTY_WEIGHTS,
        "preference_like": {"easy": 0.4, "medium": 0.6},
        "generalization": {"medium": 0.4, "hard": 0.6},
    }
    pools = {
        task_type: _quota_values(
            task_types.count(task_type),
            values,
            seed=seed,
            scope=f"tightness-{task_type}",
        )
        for task_type, values in weights.items()
    }
    offsets: Counter[str] = Counter()
    result: list[str] = []
    for task_type in task_types:
        result.append(pools[task_type][offsets[task_type]])
        offsets[task_type] += 1
    return result


def _v1_1_days(task_types: Sequence[str], seed: int) -> list[int]:
    weights = {
        "easy_like": {1: 0.18, 2: 0.38, 3: 0.34, 4: 0.06, 5: 0.04},
        "medium_like": {1: 0.05, 2: 0.35, 3: 0.35, 4: 0.18, 5: 0.07},
        "human_like": {1: 0.08, 2: 0.18, 3: 0.35, 4: 0.18, 5: 0.21},
        "preference_like": {1: 0.15, 2: 0.40, 3: 0.30, 4: 0.10, 5: 0.05},
        "generalization": {4: 0.50, 5: 0.50},
    }
    return _dimension_by_type(task_types, weights, seed, "v1-1-days")


def _v1_1_travelers(task_types: Sequence[str], seed: int) -> list[int]:
    weights = {
        "easy_like": {1: 0.35, 2: 0.27, 3: 0.20, 4: 0.12, 5: 0.04, 6: 0.02},
        "medium_like": {1: 0.22, 2: 0.36, 3: 0.20, 4: 0.14, 5: 0.05, 6: 0.03},
        "human_like": {1: 0.25, 2: 0.35, 3: 0.22, 4: 0.12, 5: 0.04, 6: 0.02},
        "preference_like": {1: 0.30, 2: 0.30, 3: 0.20, 4: 0.10, 5: 0.05, 6: 0.05},
        "generalization": {5: 0.50, 6: 0.50},
    }
    return _dimension_by_type(task_types, weights, seed, "v1-1-travelers")


def _dimension_by_type(
    task_types: Sequence[str],
    weights: Mapping[str, Mapping[_T, float]],
    seed: int,
    scope: str,
) -> list[_T]:
    pools = {
        task_type: _quota_values(
            task_types.count(task_type),
            values,
            seed=seed,
            scope=f"{scope}-{task_type}",
        )
        for task_type, values in weights.items()
    }
    offsets: Counter[str] = Counter()
    result: list[_T] = []
    for task_type in task_types:
        result.append(pools[task_type][offsets[task_type]])
        offsets[task_type] += 1
    return result


def _blended_styles(
    task_types: Sequence[str], seed: int, profile: str
) -> list[str]:
    strict_styles = tuple(_SURFACE_STYLE_WEIGHTS)
    result: list[str] = []
    for index, task_type in enumerate(task_types):
        if task_type == "human_like":
            if profile == BLENDED_V1_1_PROFILE:
                result.append(
                    "human_v1_1_metadata" if index % 2 else "human_v1_1_dialogue"
                )
            else:
                result.append("human_metadata" if index % 2 else "human_dialogue")
        else:
            result.append(deterministic_rng(seed, "blended-style", index).choice(strict_styles))
    return result


def _human_metadata_flags(task_types: Sequence[str], seed: int) -> list[bool]:
    human_values = _exact_values({True: 35, False: 15}, seed, "human-metadata")
    offset = 0
    result = [False] * len(task_types)
    for index, task_type in enumerate(task_types):
        if task_type == "human_like":
            result[index] = human_values[offset]
            offset += 1
    return result


def _human_preference_counts(task_types: Sequence[str], seed: int) -> list[int]:
    values = _exact_values({0: 10, 1: 20, 2: 15, 3: 5}, seed, "human-preference-counts")
    offset = 0
    result = [0] * len(task_types)
    for index, task_type in enumerate(task_types):
        if task_type == "human_like":
            result[index] = values[offset]
            offset += 1
    return result


def _preference_like_kinds(task_types: Sequence[str], seed: int) -> list[tuple[str, ...]]:
    official = [kind for kind in _OFFICIAL_PREFERENCES for _ in range(2)]
    extras = list(_OFFICIAL_PREFERENCES)
    deterministic_rng(seed, "official-preference-balance").shuffle(extras)
    official.extend(extras[:2])
    values = [*official, *_EXTENDED_PREFERENCES]
    deterministic_rng(seed, "preference-like-kinds").shuffle(values)
    result: list[tuple[str, ...]] = [()] * len(task_types)
    offset = 0
    for index, task_type in enumerate(task_types):
        if task_type == "preference_like":
            result[index] = (values[offset],)
            offset += 1
    return result


def _ensure_preference_day_compatibility(
    task_types: Sequence[str],
    preference_kinds: Sequence[tuple[str, ...]],
    days: list[int],
) -> None:
    lodging_targets = [
        index
        for index, kinds in enumerate(preference_kinds)
        if kinds
        and kinds[0] in {"lower_lodging_share", "near_poi", "less_walking"}
        and days[index] == 1
    ]
    lodging_replacements = [
        index
        for index, task_type in enumerate(task_types)
        if task_type not in {"preference_like", "generalization"} and days[index] > 1
    ]
    for left, right in zip(lodging_targets, lodging_replacements, strict=False):
        days[left], days[right] = days[right], days[left]
    attraction_targets = [
        index
        for index, kinds in enumerate(preference_kinds)
        if kinds == ("more_attractions",) and days[index] > 3
    ]
    attraction_replacements = [
        index
        for index, task_type in enumerate(task_types)
        if task_type not in {"preference_like", "generalization"} and days[index] <= 3
    ]
    for left, right in zip(attraction_targets, attraction_replacements, strict=False):
        days[left], days[right] = days[right], days[left]


def _pick_human_preferences(count: int, seed: int, index: int) -> tuple[str, ...]:
    values = list(_HUMAN_PREFERENCES)
    deterministic_rng(seed, "human-preferences", index).shuffle(values)
    return tuple(values[:count])


def _pick_human_preferences_v1_1(
    count: int,
    seed: int,
    index: int,
    *,
    days: int,
    recipe: Sequence[str],
    route_mode: str,
) -> tuple[str, ...]:
    excluded: set[str] = set()
    if days == 1:
        excluded.add("lower_lodging_share")
    if route_mode == "walk":
        excluded.add("less_walking")
    if "attraction_count" in recipe:
        excluded.update({"more_attractions", "relaxed_itinerary"})
    values = [kind for kind in _HUMAN_PREFERENCES if kind not in excluded]
    deterministic_rng(seed, "human-preferences-v1-1", index).shuffle(values)
    if len(values) < count:
        raise RuntimeError("Human V1.1 preference eligibility cannot fill its quota.")
    return tuple(values[:count])


def _persona(travelers: int, seed: int, index: int) -> str:
    if travelers == 1:
        return "独自出行"
    if travelers == 2:
        return deterministic_rng(seed, "persona", index).choice(("情侣出行", "朋友出行"))
    return deterministic_rng(seed, "persona", index).choice(("朋友出行", "亲子出行"))


def _metadata_prefix(
    persona: str,
    seed: int,
    index: int,
    *,
    profile: str,
    origin: str,
    destination: str,
    travelers: int,
    days: int,
) -> str:
    if profile == BLENDED_V1_1_PROFILE:
        return (
            f"[当前位置{origin},目标位置{destination},旅行人数{travelers},"
            f"旅行天数{days},出行背景{persona}]"
        )
    labels = ("出行情况", "同行背景", "旅行背景", "人员情况", "这次出游", "同行关系", "出游方式")
    label = deterministic_rng(seed, "metadata-label", index).choice(labels)
    return f"[{label}：{persona}]"


def _preference_recipe(
    count: int,
    seed: int,
    index: int,
    preference_kind: str,
    profile: str,
) -> tuple[str, ...]:
    if profile == BLENDED_V1_1_PROFILE:
        safe = [
            "total_budget",
            "outbound_time",
            "return_time",
            "attraction_count",
            "outbound_mode",
            "return_mode",
            "innercity_mode",
        ]
        if preference_kind in {"more_attractions", "relaxed_itinerary"}:
            safe.remove("attraction_count")
        if preference_kind == "less_walking":
            safe.remove("innercity_mode")
    else:
        safe = ["outbound_mode", "return_mode", "innercity_mode"]
    deterministic_rng(seed, "preference-recipes", index).shuffle(safe)
    return tuple(safe[:count])


def _validation_profile(task_type: str, seed: int, index: int, profile: str) -> str:
    if task_type == "human_like":
        return "human_conservative"
    if profile != BLENDED_V1_1_PROFILE:
        return "strict"
    natural_share = {
        "easy_like": 0.70,
        "medium_like": 0.55,
        "preference_like": 0.65,
        "generalization": 0.50,
    }[task_type]
    return (
        "benchmark_natural"
        if deterministic_rng(seed, "validation-profile-v1-1", index).random()
        < natural_share
        else "strict"
    )


def _validate_blended_distribution(slots: Sequence[PilotSlot]) -> None:
    if Counter(slot.task_type for slot in slots) != Counter(_BLENDED_TYPE_QUOTAS):
        raise RuntimeError("Blended task-type quotas drifted.")
    if Counter(slot.scenario_profile for slot in slots) != Counter(_BLENDED_SCENARIO_QUOTAS):
        raise RuntimeError("Blended scenario quotas drifted.")
    for task_type, quotas in _BLENDED_CONSTRAINT_QUOTAS.items():
        actual = Counter(
            slot.constraint_count for slot in slots if slot.task_type == task_type
        )
        if actual != Counter(quotas):
            raise RuntimeError(f"Blended constraint quotas drifted for {task_type}.")
    humans = [slot for slot in slots if slot.task_type == "human_like"]
    if sum(slot.metadata_prefix is not None for slot in humans) != 35:
        raise RuntimeError("Human metadata quota drifted.")
    if Counter(len(slot.preference_kinds) for slot in humans) != {
        0: 10,
        1: 20,
        2: 15,
        3: 5,
    }:
        raise RuntimeError("Human preference quotas drifted.")


def _quota_values(
    count: int,
    weights: Mapping[_T, float],
    *,
    seed: int,
    scope: str,
) -> list[_T]:
    total = sum(weights.values())
    raw = {value: count * weight / total for value, weight in weights.items()}
    quotas = {value: math.floor(amount) for value, amount in raw.items()}
    remaining = count - sum(quotas.values())
    tie_order = list(weights)
    deterministic_rng(seed, f"{scope}-remainders").shuffle(tie_order)
    tie_break = {value: index for index, value in enumerate(tie_order)}
    ordered = sorted(weights, key=lambda value: (-(raw[value] - quotas[value]), tie_break[value]))
    for value in ordered[:remaining]:
        quotas[value] += 1
    values = [value for value in weights for _ in range(quotas[value])]
    deterministic_rng(seed, scope).shuffle(values)
    return values


def _balanced_origins(destinations: Sequence[str], seed: int) -> list[str]:
    usage: Counter[str] = Counter()
    pair_usage: Counter[tuple[str, str]] = Counter()
    origins: list[str] = []
    for index, destination in enumerate(destinations):
        candidates = [city for city in _CITIES if city != destination]
        rng = deterministic_rng(seed, "origin", index)
        rng.shuffle(candidates)
        candidates.sort(key=lambda city: (pair_usage[(city, destination)], usage[city]))
        origin = candidates[0]
        origins.append(origin)
        usage[origin] += 1
        pair_usage[(origin, destination)] += 1
    return origins


def _transport_patterns(destinations: Sequence[str], seed: int) -> list[tuple[str, str]]:
    count = len(destinations)
    desired = Counter(
        _quota_values(count, _MODE_WEIGHTS, seed=seed, scope="transport-patterns")
    )
    patterns: list[tuple[str, str] | None] = [None] * count
    suzhou_indices = [index for index, city in enumerate(destinations) if city == "苏州"]
    for index in suzhou_indices:
        patterns[index] = ("train", "train")
    desired[("train", "train")] = max(0, desired[("train", "train")] - len(suzhou_indices))

    remaining_indices = [index for index, value in enumerate(patterns) if value is None]
    pool = [pattern for pattern, quota in desired.items() for _ in range(quota)]
    while len(pool) < len(remaining_indices):
        least_used = min(_MODE_WEIGHTS, key=lambda pattern: pool.count(pattern))
        pool.append(least_used)
    pool = pool[: len(remaining_indices)]
    deterministic_rng(seed, "transport-assignment").shuffle(pool)
    for index, pattern in zip(remaining_indices, pool, strict=True):
        patterns[index] = pattern
    return [pattern for pattern in patterns if pattern is not None]


def _recipe(
    *,
    index: int,
    seed: int,
    days: int,
    constraint_count: int,
    outbound_mode: str,
    return_mode: str,
    usage: Counter[str],
    pair_usage: Counter[tuple[str, str]],
    family_weights: Mapping[str, float] | None = None,
) -> tuple[str, ...]:
    rng = deterministic_rng(seed, "recipe", index)
    selected: list[str] = []
    if constraint_count == 0:
        return ()
    if outbound_mode != return_mode and constraint_count >= 2 and rng.random() < 0.6:
        selected.extend(("outbound_mode", "return_mode"))
    elif outbound_mode == return_mode and rng.random() < 0.4:
        selected.append("all_intercity_mode")

    available = [
        key
        for key in _BASE_KEYS
        if key not in _INTERCITY_MODE_KEYS and (days > 1 or key not in _HOTEL_KEYS)
    ]
    rng.shuffle(available)
    while len(selected) < constraint_count:
        candidates = [key for key in available if key not in selected]
        if not candidates:
            raise RuntimeError("Unable to construct a complete constraint recipe.")

        def score(key: str) -> tuple[int, int]:
            pairs = sum(pair_usage[tuple(sorted((key, existing)))] for existing in selected)
            if family_weights is None:
                normalized_usage = usage[key]
            else:
                normalized_usage = round(usage[key] / family_weights[key])
            return normalized_usage, pairs

        best_score = min(score(key) for key in candidates)
        selected.append(next(key for key in candidates if score(key) == best_score))
    return tuple(selected)


def _validate_slots(slots: Sequence[PilotSlot]) -> None:
    for slot in slots:
        if slot.origin == slot.destination:
            raise RuntimeError("Synthesis slot origin and destination must differ.")
        if slot.destination == "苏州" and "airplane" in {
            slot.outbound_mode,
            slot.return_mode,
        }:
            raise RuntimeError("Suzhou synthesis slots cannot require airplane transport.")
        if len(slot.recipe) != slot.constraint_count or len(set(slot.recipe)) != len(slot.recipe):
            raise RuntimeError("Synthesis slot recipe is incomplete or duplicated.")
        if slot.days > 3 and slot.attractions_per_day != 1:
            raise RuntimeError("Long synthesis slots exceed the intended plan complexity.")
