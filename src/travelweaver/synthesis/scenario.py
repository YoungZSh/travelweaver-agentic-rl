"""Materialize explicit scenario deltas from a synthesis slot."""

from __future__ import annotations

from typing import Any

from ..env import ChinaTravelBackend, ScenarioEffect, ScenarioSpec
from ..errors import SynthesisError
from .models import WORLD_SNAPSHOT_VERSION, PilotSlot
from .randomness import deterministic_rng


def build_scenario(
    backend: ChinaTravelBackend,
    slot: PilotSlot,
    *,
    origin: str,
    seed: int,
    candidate_attempt: int,
) -> ScenarioSpec:
    """Choose effects once; the returned concrete delta is the replay authority."""

    rng = deterministic_rng(seed, f"scenario-{candidate_attempt}", slot.index)
    effects: list[ScenarioEffect] = []
    if slot.scenario_profile == "normal":
        pass
    elif slot.scenario_profile == "poi_closure":
        records = list(backend._records("attraction", slot.destination))
        rng.shuffle(records)
        effects = _unavailable_effects(records[: min(5, len(records))], "attraction")
    elif slot.scenario_profile == "hotel_unavailable":
        records = list(backend._records("hotel", slot.destination))
        rng.shuffle(records)
        effects = _unavailable_effects(records[: min(5, len(records))], "hotel")
    elif slot.scenario_profile == "transport_cancellation":
        effects = _cancellation_effects(backend, slot, origin, rng)
    elif slot.scenario_profile == "price_change":
        effects = _price_effects(backend, slot, rng)
    else:
        raise SynthesisError(f"Unknown scenario profile: {slot.scenario_profile}")
    if slot.scenario_profile != "normal" and not effects:
        raise SynthesisError(f"Scenario {slot.scenario_profile} produced no usable effects.")
    return ScenarioSpec(
        base_world_snapshot_version=WORLD_SNAPSHOT_VERSION,
        profile=slot.scenario_profile,
        effects=tuple(effects),
    )


def _unavailable_effects(
    records: list[dict[str, Any]], target_type: str
) -> list[ScenarioEffect]:
    return [
        ScenarioEffect(
            effect_id=f"effect_{index:03d}",
            kind="unavailable",
            target_type=target_type,
            target_id=str(record["place_id"]),
            field="available",
            before=True,
            after=False,
        )
        for index, record in enumerate(records, 1)
    ]


def _cancellation_effects(
    backend: ChinaTravelBackend,
    slot: PilotSlot,
    origin: str,
    rng: Any,
) -> list[ScenarioEffect]:
    legs = (
        (origin, slot.destination, slot.outbound_mode),
        (slot.destination, origin, slot.return_mode),
    )
    selected: list[dict[str, Any]] = []
    for leg_origin, leg_destination, mode in legs:
        records = backend.search_intercity_transport(
            origin_city=leg_origin,
            destination_city=leg_destination,
            mode=mode,
        )
        if len(records) < 2:
            raise SynthesisError("Transport cancellation needs at least two options per leg.")
        rng.shuffle(records)
        cancellation_count = min(3, max(1, len(records) // 10), len(records) - 1)
        selected.extend(records[:cancellation_count])
    return [
        ScenarioEffect(
            effect_id=f"effect_{index:03d}",
            kind="cancelled",
            target_type="intercity_transport",
            target_id=str(record["transport_id"]),
            field="available",
            before=True,
            after=False,
        )
        for index, record in enumerate(selected, 1)
    ]


def _price_effects(
    backend: ChinaTravelBackend,
    slot: PilotSlot,
    rng: Any,
) -> list[ScenarioEffect]:
    records = [
        dict(record)
        for kind in ("attraction", "restaurant", "hotel")
        for record in backend._records(kind, slot.destination)
        if isinstance(record.get("price"), (int, float))
        and not isinstance(record.get("price"), bool)
    ]
    rng.shuffle(records)
    selected = records[: min(6, len(records))]
    return [
        ScenarioEffect(
            effect_id=f"effect_{index:03d}",
            kind="field_override",
            target_type=str(record["entity_type"]),
            target_id=str(record["place_id"]),
            field="price",
            before=record["price"],
            after=round(float(record["price"]) * 1.25, 2),
        )
        for index, record in enumerate(selected, 1)
    ]
