"""End-to-end witness-first task synthesis pipeline."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..env import ChinaTravelBackend, ScenarioBackend
from ..errors import BackendQueryError, SynthesisError, TravelWeaverError
from ..llm import DeepSeekConfig
from ..reward import TravelReward
from ..tasks import materialize_task_spec
from .artifacts import ArtifactStore, record_bundle
from .catalog import build_pilot_slots
from .compose import compose_blueprint
from .models import (
    GENERATOR_VERSION,
    PROMPT_VERSION,
    WORLD_SNAPSHOT_VERSION,
    PilotSlot,
    WitnessResult,
)
from .polisher import POLISHER_PROMPT_HASH, TaskPolisher
from .randomness import deterministic_rng
from .render import render_canonical
from .scenario import build_scenario
from .witness import WitnessBuilder


@dataclass(frozen=True)
class SynthesisConfig:
    output_dir: Path
    count: int = 100
    seed: int = 20260807
    max_api_calls: int = 300
    profile: str = "pilot_v2_1"
    validation_policy: str = "minimal_semantic"

    def __post_init__(self) -> None:
        if self.count <= 0 or self.max_api_calls <= 0:
            raise ValueError("Synthesis count and API-call budget must be positive.")
        if self.validation_policy not in {"strict", "minimal_semantic"}:
            raise ValueError("Synthesis validation policy is unsupported.")


@dataclass(frozen=True)
class SynthesisReport:
    output_dir: str
    requested: int
    completed: int
    api_calls: int
    quarantine_count: int
    distributions: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "requested": self.requested,
            "completed": self.completed,
            "api_calls": self.api_calls,
            "quarantine_count": self.quarantine_count,
            "distributions": self.distributions,
        }


class SynthesisPipeline:
    def __init__(
        self,
        config: SynthesisConfig,
        llm_config: DeepSeekConfig,
        *,
        backend: ChinaTravelBackend | None = None,
        polisher: TaskPolisher | None = None,
    ) -> None:
        self.config = config
        self.llm_config = llm_config
        self.backend = backend or ChinaTravelBackend()
        self.polisher = polisher or TaskPolisher(
            llm_config,
            validation_policy=config.validation_policy,
        )

    def run(self) -> SynthesisReport:
        slots = build_pilot_slots(
            self.config.count,
            self.config.seed,
            self.config.profile,
        )
        store = ArtifactStore(
            self.config.output_dir,
            {
                "count": self.config.count,
                "seed": self.config.seed,
                "max_api_calls": self.config.max_api_calls,
                "profile": self.config.profile,
                "validation_policy": self.config.validation_policy,
                "model": self.llm_config.model,
                "base_url": self.llm_config.base_url,
                "thinking": "disabled",
                "generator_version": GENERATOR_VERSION,
                "prompt_version": PROMPT_VERSION,
                "prompt_hash": POLISHER_PROMPT_HASH,
                "world_snapshot_version": WORLD_SNAPSHOT_VERSION,
            },
        )
        completed = store.completed_indices()
        existing = store.records()
        blueprint_ids = {row["blueprint"]["blueprint_id"] for row in existing}
        surface_ids = {row["surface"]["surface_id"] for row in existing}
        self._origin_usage = Counter(
            str(row["slot"]["origin"])
            for row in existing
        )
        self._od_usage = Counter(
            (str(row["slot"]["origin"]), str(row["slot"]["destination"]))
            for row in existing
        )
        starting_api_calls = store.api_calls

        for slot in slots:
            if slot.index in completed:
                continue
            uid = f"tw_syn_v2_{self.config.seed}_{slot.index:04d}"
            accepted = False
            errors: list[str] = []
            for candidate_attempt, origin in enumerate(
                self._origins(slot), 1
            ):
                if (
                    self.config.max_api_calls - self._api_calls(starting_api_calls)
                    < self.polisher.max_attempts
                ):
                    raise SynthesisError(
                        "API budget cannot fund another complete polish operation."
                    )
                generation_seed = deterministic_rng(
                    self.config.seed,
                    f"witness-{candidate_attempt}",
                    slot.index,
                ).getrandbits(63)
                try:
                    effective_slot = replace(slot, origin=origin)
                    if (
                        effective_slot.synthesis_profile == "chinatravel_blended_v1_1"
                        and effective_slot.task_type == "human_like"
                        and effective_slot.metadata_prefix is not None
                    ):
                        effective_slot = replace(
                            effective_slot,
                            metadata_prefix=(
                                f"[当前位置{origin},目标位置{effective_slot.destination},"
                                f"旅行人数{effective_slot.travelers},"
                                f"旅行天数{effective_slot.days},"
                                f"出行背景{effective_slot.persona_context}]"
                            ),
                        )
                    scenario = build_scenario(
                        self.backend,
                        effective_slot,
                        origin=origin,
                        seed=self.config.seed,
                        candidate_attempt=candidate_attempt,
                    )
                    scenario_backend = ScenarioBackend(self.backend, scenario)
                    preference_candidates: list[WitnessResult] = []
                    if effective_slot.task_type == "preference_like":
                        preference_candidates = self._preference_witnesses(
                            scenario_backend,
                            effective_slot,
                            origin=origin,
                            uid=uid,
                            generation_seed=generation_seed,
                        )
                        witness, preference_audit = _choose_preference_witness(
                            preference_candidates,
                            effective_slot.preference_kinds[0],
                        )
                    else:
                        witness = WitnessBuilder(
                            scenario_backend,
                            seed=generation_seed,
                        ).build(effective_slot, origin=origin, uid=uid)
                        preference_audit = None
                    if (
                        effective_slot.synthesis_profile == "chinatravel_blended_v1_1"
                        and effective_slot.task_type == "human_like"
                        and "less_walking" in effective_slot.preference_kinds
                        and witness.route_mode == "walk"
                    ):
                        raise SynthesisError(
                            "Human less-walking preference resolved to an all-walking witness."
                        )
                    blueprint = compose_blueprint(
                        effective_slot,
                        witness,
                        generation_seed=generation_seed,
                        world_snapshot_version=scenario.world_snapshot_version,
                        constraint_witnesses=preference_candidates,
                    )
                    if blueprint.blueprint_id in blueprint_ids:
                        raise SynthesisError("Blueprint semantic hash duplicates an accepted task.")
                    canonical = render_canonical(
                        blueprint,
                        style_profile=effective_slot.surface_style,
                        validation_profile=effective_slot.validation_profile,
                    )
                    surface, polish_audit = self.polisher.polish_with_audit(
                        blueprint,
                        canonical,
                        style_profile=effective_slot.surface_style,
                        validation_profile=effective_slot.validation_profile,
                        audit_context={"slot_index": effective_slot.index},
                    )
                    if surface.surface_id in surface_ids:
                        raise SynthesisError("Polished surface duplicates an accepted task.")
                    spec = materialize_task_spec(blueprint, surface, task_id=uid)
                    reward = TravelReward().evaluate(
                        spec,
                        witness.plan_snapshot,
                        witness.evidence_bundle,
                    )
                    if reward.reward != 1.0 or not reward.all_hard_pass:
                        raise SynthesisError(
                            "Final materialized TaskSpec does not pass its stored witness."
                        )
                    witness_payload = witness.to_dict()
                    witness_payload["public_task"] = {
                        **witness_payload["public_task"],
                        "query": surface.public_query,
                    }
                    witness_payload["reward_detail"] = reward.to_dict()
                    if preference_audit is not None:
                        audited_candidates = []
                        for candidate, candidate_row in zip(
                            preference_candidates,
                            preference_audit["candidates"],
                            strict=True,
                        ):
                            candidate_reward = TravelReward().evaluate(
                                spec,
                                candidate.plan_snapshot,
                                candidate.evidence_bundle,
                            )
                            audited_candidates.append(
                                {
                                    **candidate_row,
                                    "hard_reward": candidate_reward.reward,
                                    "all_hard_pass": candidate_reward.all_hard_pass,
                                }
                            )
                        if not all(row["all_hard_pass"] for row in audited_candidates):
                            raise SynthesisError(
                                "Preference audit contains a candidate that fails hard constraints."
                            )
                        preference_audit["candidates"] = audited_candidates
                    record = record_bundle(
                        slot=effective_slot,
                        blueprint=blueprint.to_dict(),
                        surface=surface.to_dict(),
                        task_spec=spec.to_dict(),
                        witness=witness_payload,
                        scenario=scenario.to_dict(),
                        preference_audit=preference_audit,
                        polish_audit=list(polish_audit),
                        candidate_attempt=candidate_attempt,
                    )
                except (TravelWeaverError, ValueError) as error:
                    message = f"{type(error).__name__}: {error}"
                    errors.append(message)
                    store.quarantine(
                        {
                            "slot_index": slot.index,
                            "candidate_attempt": candidate_attempt,
                            "origin": origin,
                            "destination": slot.destination,
                            "scenario_profile": slot.scenario_profile,
                            "stage_error": message,
                        },
                        api_calls=self._api_calls(starting_api_calls),
                    )
                    if self._api_calls(starting_api_calls) >= self.config.max_api_calls:
                        break
                    continue
                api_calls = self._api_calls(starting_api_calls)
                store.save_record(slot.index, record, api_calls=api_calls)
                blueprint_ids.add(blueprint.blueprint_id)
                surface_ids.add(surface.surface_id)
                self._origin_usage[origin] += 1
                self._od_usage[(origin, slot.destination)] += 1
                accepted = True
                break
            if not accepted:
                detail = " | ".join(errors[-3:]) or "no origin candidates"
                raise SynthesisError(f"Unable to fill synthesis slot {slot.index}: {detail}")

        api_calls = self._api_calls(starting_api_calls)
        distributions = store.finalize(slots, api_calls=api_calls)
        return SynthesisReport(
            output_dir=str(self.config.output_dir.resolve()),
            requested=self.config.count,
            completed=len(store.completed_indices()),
            api_calls=api_calls,
            quarantine_count=store.quarantine_count,
            distributions=distributions,
        )

    def _origins(self, slot: PilotSlot) -> tuple[str, ...]:
        cities = [city for city in self.backend.supported_cities if city != slot.destination]
        viable = [city for city in cities if self._has_usable_round_trip(slot, city)]
        deterministic_rng(self.config.seed, "origin-fallbacks", slot.index).shuffle(viable)
        viable.sort(
            key=lambda city: (
                self._od_usage[(city, slot.destination)],
                self._origin_usage[city],
            )
        )
        return tuple(viable)

    def _has_usable_round_trip(self, slot: PilotSlot, origin: str) -> bool:
        try:
            outbound = self.backend.search_intercity_transport(
                origin_city=origin,
                destination_city=slot.destination,
                mode=slot.outbound_mode,
            )
            returning = self.backend.search_intercity_transport(
                origin_city=slot.destination,
                destination_city=origin,
                mode=slot.return_mode,
            )
        except BackendQueryError:
            return False
        return any(_same_day_outbound(item) for item in outbound) and any(
            _same_day_return(item) for item in returning
        )

    def _api_calls(self, starting_api_calls: int) -> int:
        return starting_api_calls + self.polisher.api_calls

    def _preference_witnesses(
        self,
        backend: ScenarioBackend,
        slot: PilotSlot,
        *,
        origin: str,
        uid: str,
        generation_seed: int,
    ) -> list[WitnessResult]:
        anchor_records = sorted(
            backend._records("attraction", slot.destination),
            key=lambda item: str(item.get("place_id", "")),
        )
        anchor = dict(anchor_records[0]) if anchor_records else None
        candidates: list[WitnessResult] = []
        errors: list[str] = []
        preference_kind = slot.preference_kinds[0]
        vary_route = (
            slot.synthesis_profile == "chinatravel_blended_v1_1"
            and "innercity_mode" not in slot.recipe
        )
        route_modes = (slot.route_mode,) if not vary_route else ("taxi", "metro", "walk")
        max_candidates = 12 if slot.synthesis_profile == "chinatravel_blended_v1_1" else 6
        for candidate_index in range(max_candidates):
            attractions_per_day = (
                2
                if preference_kind in {"more_attractions", "relaxed_itinerary"}
                and slot.days <= 3
                and candidate_index % 2
                else 1
            )
            candidate_slot = replace(
                slot,
                attractions_per_day=attractions_per_day,
                include_meal=(
                    slot.include_meal
                    or preference_kind in {"less_innercity_time", "less_walking"}
                ),
                route_mode=route_modes[candidate_index % len(route_modes)],
            )
            candidate_seed = deterministic_rng(
                generation_seed,
                "preference-candidate",
                candidate_index,
            ).getrandbits(63)
            try:
                candidate = WitnessBuilder(backend, seed=candidate_seed).build(
                    candidate_slot,
                    origin=origin,
                    uid=uid,
                )
            except (TravelWeaverError, ValueError) as error:
                errors.append(str(error))
                continue
            if anchor is not None:
                candidate.selected["preference_anchor"] = anchor
            if (
                candidates
                and "innercity_mode" in slot.recipe
                and candidate.route_mode != candidates[0].route_mode
            ):
                continue
            candidates.append(candidate)
            metrics = [_preference_metric(item, preference_kind) for item in candidates]
            if len(candidates) >= 3 and (
                slot.synthesis_profile != "chinatravel_blended_v1_1"
                or len(set(metrics)) >= 2
            ):
                break
        if len(candidates) < 2:
            raise SynthesisError(
                "Preference witness selection needs at least two feasible candidates: "
                + " | ".join(errors[-2:])
            )
        if (
            slot.synthesis_profile == "chinatravel_blended_v1_1"
            and len({_preference_metric(item, preference_kind) for item in candidates}) < 2
        ):
            raise SynthesisError(
                f"Preference metric {preference_kind} did not distinguish feasible candidates."
            )
        return candidates


def _clock_minutes(value: Any) -> int:
    if not isinstance(value, str) or ":" not in value:
        return -1
    try:
        hours, minutes = (int(part) for part in value.split(":", 1))
    except ValueError:
        return -1
    return hours * 60 + minutes


def _same_day_outbound(item: dict[str, Any]) -> bool:
    departure = _clock_minutes(item.get("departure_time"))
    arrival = _clock_minutes(item.get("arrival_time"))
    return departure < arrival <= 14 * 60


def _same_day_return(item: dict[str, Any]) -> bool:
    departure = _clock_minutes(item.get("departure_time"))
    arrival = _clock_minutes(item.get("arrival_time"))
    return 14 * 60 <= departure < arrival


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


def _choose_preference_witness(
    candidates: list[WitnessResult], kind: str
) -> tuple[WitnessResult, dict[str, Any]]:
    direction = _PREFERENCE_DIRECTIONS[kind]
    metrics = [_preference_metric(candidate, kind) for candidate in candidates]
    selected_index = (
        max(range(len(candidates)), key=metrics.__getitem__)
        if direction == "maximize"
        else min(range(len(candidates)), key=metrics.__getitem__)
    )
    audit = {
        "preference_kind": kind,
        "direction": direction,
        "metric": _preference_metric_name(kind),
        "selected_candidate": selected_index,
        "candidates": [
            {
                "candidate_index": index,
                "metric_value": round(metric, 6),
                "selected": index == selected_index,
            }
            for index, metric in enumerate(metrics)
        ],
    }
    return candidates[selected_index], audit


def _preference_metric(witness: WitnessResult, kind: str) -> float:
    evidence = witness.evidence_bundle
    cost_items = list(evidence.get("cost_items", []))
    routes = list(dict(evidence.get("routes", {})).values())
    total_cost = float(evidence.get("total_cost", 0.0))
    if kind == "more_attractions":
        return float(len(witness.selected.get("attractions", [])))
    if kind == "less_innercity_time":
        return float(sum(_route_minutes(route) for route in routes))
    if kind == "shorter_meal_transfer":
        entities = dict(evidence.get("entities", {}))
        return float(
            sum(
                _route_minutes(route)
                for route in routes
                if _entity_type(entities, route.get("destination_place_id")) == "restaurant"
            )
        )
    if kind == "higher_dining_share":
        return _cost_share(cost_items, {"lunch", "dinner", "breakfast"}, total_cost)
    if kind == "lower_lodging_share":
        return _cost_share(cost_items, {"accommodation"}, total_cost)
    if kind == "near_poi":
        return _anchor_distance(witness.selected)
    if kind == "less_walking":
        return float(sum(_walking_minutes(route) for route in routes))
    if kind == "lower_total_cost":
        return total_cost
    if kind == "relaxed_itinerary":
        return float(
            len(witness.selected.get("attractions", []))
            + (1 if witness.selected.get("restaurant") else 0)
        )
    if kind == "higher_attraction_share":
        return _cost_share(cost_items, {"attraction"}, total_cost)
    if kind == "lower_intercity_share":
        return _cost_share(cost_items, {"train", "airplane"}, total_cost)
    if kind == "shorter_total_travel_time":
        selected = witness.selected
        intercity = sum(
            _transport_minutes(selected.get(leg)) for leg in ("outbound", "return")
        )
        return float(intercity + sum(_route_minutes(route) for route in routes))
    raise SynthesisError(f"Unknown preference metric: {kind}")


def _preference_metric_name(kind: str) -> str:
    return {
        "more_attractions": "attraction_count",
        "less_innercity_time": "innercity_route_minutes",
        "shorter_meal_transfer": "meal_transfer_minutes",
        "higher_dining_share": "dining_cost_share",
        "lower_lodging_share": "lodging_cost_share",
        "near_poi": "anchor_distance_squared",
        "less_walking": "walking_minutes",
        "lower_total_cost": "total_cost",
        "relaxed_itinerary": "scheduled_stop_count",
        "higher_attraction_share": "attraction_cost_share",
        "lower_intercity_share": "intercity_cost_share",
        "shorter_total_travel_time": "total_travel_minutes",
    }[kind]


def _route_minutes(route: Mapping[str, Any]) -> int:
    total = 0
    for segment in route.get("segments", []):
        if not isinstance(segment, Mapping):
            continue
        start = _clock_minutes(segment.get("start_time"))
        end = _clock_minutes(segment.get("end_time"))
        if start >= 0 and end >= start:
            total += end - start
    return total


def _walking_minutes(route: Mapping[str, Any]) -> int:
    total = 0
    for segment in route.get("segments", []):
        if not isinstance(segment, Mapping) or segment.get("mode") != "walk":
            continue
        start = _clock_minutes(segment.get("start_time"))
        end = _clock_minutes(segment.get("end_time"))
        if start >= 0 and end >= start:
            total += end - start
    return total


def _transport_minutes(value: Any) -> int:
    if not isinstance(value, Mapping):
        return 0
    start = _clock_minutes(value.get("departure_time"))
    end = _clock_minutes(value.get("arrival_time"))
    return max(0, end - start)


def _entity_type(entities: Mapping[str, Any], entity_id: Any) -> str | None:
    entity = entities.get(str(entity_id))
    return str(entity.get("entity_type")) if isinstance(entity, Mapping) else None


def _cost_share(
    cost_items: list[Any], activity_types: set[str], total_cost: float
) -> float:
    if total_cost <= 0:
        return 0.0
    amount = sum(
        float(item.get("amount", 0.0))
        for item in cost_items
        if isinstance(item, Mapping) and item.get("activity_type") in activity_types
    )
    return amount / total_cost


def _anchor_distance(selected: Mapping[str, Any]) -> float:
    anchor = selected.get("preference_anchor")
    hotel = selected.get("hotel")
    if not isinstance(anchor, Mapping) or not isinstance(hotel, Mapping):
        return float("inf")
    try:
        latitude_delta = float(anchor["latitude"]) - float(hotel["latitude"])
        longitude_delta = float(anchor["longitude"]) - float(hotel["longitude"])
    except (KeyError, TypeError, ValueError):
        return float("inf")
    return latitude_delta**2 + longitude_delta**2
