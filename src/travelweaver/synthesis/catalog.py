"""Deterministic pilot slots and constraint recipes."""

from __future__ import annotations

import random
from collections import Counter

from .models import PilotSlot

_CITIES = ("上海", "北京", "南京", "广州", "成都", "杭州", "武汉", "深圳", "苏州", "重庆")
_MODE_QUOTAS = {
    ("train", "train"): 20,
    ("airplane", "airplane"): 10,
    ("train", "airplane"): 10,
    ("airplane", "train"): 10,
}
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


def build_pilot_slots(count: int, seed: int) -> tuple[PilotSlot, ...]:
    if count <= 0:
        raise ValueError("Synthesis count must be positive.")
    if count != 50:
        return _build_scaled_slots(count, seed)
    rng = random.Random(seed)
    destinations = [city for city in _CITIES for _ in range(5)]
    rng.shuffle(destinations)

    # ChinaTravel has no Suzhou flights. Reserve those five train/train slots before
    # shuffling the remaining exact mode quota so later choices cannot starve Suzhou.
    remaining_patterns = [
        pattern
        for pattern, quota in _MODE_QUOTAS.items()
        for _ in range(quota - (5 if pattern == ("train", "train") else 0))
    ]
    rng.shuffle(remaining_patterns)
    pattern_iterator = iter(remaining_patterns)
    patterns = [
        ("train", "train") if destination == "苏州" else next(pattern_iterator)
        for destination in destinations
    ]

    days = [1] * 20 + [2] * 20 + [3] * 10
    counts = [1] * 15 + [2] * 20 + [3] * 10 + [4] * 5
    rng.shuffle(days)
    rng.shuffle(counts)

    mixed_indices = [index for index, pair in enumerate(patterns) if pair[0] != pair[1]]
    rich_indices = [index for index, value in enumerate(counts) if value >= 2]
    for mixed_index, rich_index in zip(mixed_indices[:10], rich_indices, strict=False):
        if counts[mixed_index] >= 2:
            continue
        counts[mixed_index], counts[rich_index] = counts[rich_index], counts[mixed_index]

    slots = []
    recipe_usage: Counter[str] = Counter()
    explicit_mixed = set(mixed_indices[:10])
    for index, (destination, day_count, constraint_count, modes) in enumerate(
        zip(destinations, days, counts, patterns, strict=True)
    ):
        recipe = _recipe(
            index=index,
            days=day_count,
            constraint_count=constraint_count,
            outbound_mode=modes[0],
            return_mode=modes[1],
            explicit_mixed=index in explicit_mixed,
            usage=recipe_usage,
        )
        recipe_usage.update(recipe)
        slots.append(
            PilotSlot(
                index=index,
                destination=destination,
                days=day_count,
                travelers=index % 4 + 1,
                outbound_mode=modes[0],
                return_mode=modes[1],
                constraint_count=constraint_count,
                recipe=recipe,
            )
        )
    _validate_distribution(slots)
    return tuple(slots)


def _build_scaled_slots(count: int, seed: int) -> tuple[PilotSlot, ...]:
    rng = random.Random(seed)
    slots = []
    patterns = tuple(_MODE_QUOTAS)
    for index in range(count):
        destination = _CITIES[index % len(_CITIES)]
        compatible = [
            pattern
            for pattern in patterns
            if destination != "苏州" or "airplane" not in pattern
        ]
        modes = compatible[index % len(compatible)]
        days = index % 3 + 1
        constraint_count = min(4, index % 4 + 1)
        recipe = _recipe(
            index=rng.randrange(10_000),
            days=days,
            constraint_count=constraint_count,
            outbound_mode=modes[0],
            return_mode=modes[1],
            explicit_mixed=modes[0] != modes[1] and constraint_count >= 2,
        )
        slots.append(
            PilotSlot(
                index=index,
                destination=destination,
                days=days,
                travelers=index % 4 + 1,
                outbound_mode=modes[0],
                return_mode=modes[1],
                constraint_count=constraint_count,
                recipe=recipe,
            )
        )
    return tuple(slots)


def _recipe(
    *,
    index: int,
    days: int,
    constraint_count: int,
    outbound_mode: str,
    return_mode: str,
    explicit_mixed: bool,
    usage: Counter[str] | None = None,
) -> tuple[str, ...]:
    selected: list[str] = []
    if explicit_mixed:
        selected.extend(("outbound_mode", "return_mode"))
    elif outbound_mode == return_mode and index % 4 == 0:
        selected.append("all_intercity_mode")

    available = [
        key
        for key in _BASE_KEYS
        if key not in _INTERCITY_MODE_KEYS and (days > 1 or key not in _HOTEL_KEYS)
    ]
    offset = index % len(available)
    rotated = available[offset:] + available[:offset]
    if usage is not None:
        tie_break = {key: position for position, key in enumerate(rotated)}
        rotated.sort(key=lambda key: (usage[key], tie_break[key]))
    for key in rotated:
        if len(selected) >= constraint_count:
            break
        if key in selected:
            continue
        selected.append(key)
    if len(selected) != constraint_count:
        raise RuntimeError("Unable to construct a complete constraint recipe.")
    return tuple(selected)


def _validate_distribution(slots: list[PilotSlot]) -> None:
    if Counter(slot.destination for slot in slots) != Counter({city: 5 for city in _CITIES}):
        raise RuntimeError("Pilot destinations are not balanced.")
    if Counter(slot.days for slot in slots) != Counter({1: 20, 2: 20, 3: 10}):
        raise RuntimeError("Pilot day distribution is invalid.")
    if Counter(slot.constraint_count for slot in slots) != Counter({1: 15, 2: 20, 3: 10, 4: 5}):
        raise RuntimeError("Pilot constraint distribution is invalid.")
    if Counter((slot.outbound_mode, slot.return_mode) for slot in slots) != Counter(_MODE_QUOTAS):
        raise RuntimeError("Pilot transport distribution is invalid.")
    explicit_mixed = sum(
        {"outbound_mode", "return_mode"}.issubset(slot.recipe) for slot in slots
    )
    if explicit_mixed < 10:
        raise RuntimeError("Pilot must contain at least ten explicit mixed-mode tasks.")
    recipe_counts = Counter(key for slot in slots for key in slot.recipe)
    missing_diversity = {
        key: recipe_counts[key]
        for key in _BASE_KEYS
        if key not in _INTERCITY_MODE_KEYS and recipe_counts[key] < 3
    }
    if missing_diversity:
        raise RuntimeError(f"Pilot constraint diversity is too low: {missing_diversity}")
