"""Construct replayable feasible plans before deriving task constraints."""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ..env import ChinaTravelBackend, ScenarioBackend, TravelWeaverEnv
from ..errors import BackendQueryError, SynthesisError
from .models import PilotSlot, WitnessResult
from .trajectory_policy import MAX_CONSECUTIVE_TOOL_CALLS, MAX_WITNESS_VALID_STEPS


class _SingleTaskStore:
    def __init__(self, public_task: dict[str, Any]) -> None:
        self.public_task = public_task

    def choose(self, seed: int | None = None) -> str:
        del seed
        return str(self.public_task["uid"])

    def get_public(self, task_id: str) -> dict[str, Any]:
        if task_id != self.public_task["uid"]:
            raise KeyError(task_id)
        return dict(self.public_task)

    def get_oracle(self, task_id: str) -> dict[str, Any]:
        if task_id != self.public_task["uid"]:
            raise KeyError(task_id)
        return {"uid": task_id}


@dataclass(frozen=True)
class _LocalActivity:
    evidence: dict[str, Any]
    activity_type: str
    start_time: str
    end_time: str
    route: dict[str, Any] | None = None


class WitnessBuilder:
    """Build a plan using only entities that become visible through environment tools."""

    def __init__(
        self,
        backend: ChinaTravelBackend | ScenarioBackend,
        *,
        seed: int,
    ) -> None:
        self.backend = backend
        self.rng = random.Random(seed)
        self._public_page_indices: dict[tuple[object, ...], dict[str, int]] = {}
        self._nearby_first_page_visibility: dict[tuple[str, str], bool] = {}
        self._nearby_first_page_ids: dict[
            tuple[str, str, int], frozenset[str]
        ] = {}

    def build(
        self,
        slot: PilotSlot,
        *,
        origin: str,
        uid: str,
    ) -> WitnessResult:
        # Candidate selection queries many nearby entities.  Cache the public search
        # ordering for this frozen world so selecting a page-two attraction does not
        # turn synthesis into one backend query per candidate.
        self._public_page_indices = {}
        self._nearby_first_page_visibility = {}
        self._nearby_first_page_ids = {}
        public_task = {
            "uid": uid,
            "tag": "synthetic_witness",
            "start_city": origin,
            "target_city": slot.destination,
            "days": slot.days,
            "people_number": slot.travelers,
            "limit_rooms": False,
            "limits_room_type": False,
            "language": "zh",
            "query": (
                f"请规划从{origin}到{slot.destination}的{slot.days}天行程，"
                f"共{slot.travelers}人。"
            ),
        }
        outbound, return_transport = self._select_transports(slot, origin)
        route_modes = self._route_mode_order(slot.route_mode)
        last_error: Exception | None = None
        for route_mode in route_modes:
            attempts = {"metro": 8, "walk": 5, "taxi": 3}[route_mode]
            for _ in range(attempts):
                try:
                    return self._execute(
                        slot,
                        public_task,
                        outbound,
                        return_transport,
                        route_mode,
                    )
                except (BackendQueryError, SynthesisError, ValueError) as error:
                    last_error = error
        raise SynthesisError(
            f"No feasible witness for {origin}->{slot.destination}: {last_error}"
        )

    def _select_transports(
        self, slot: PilotSlot, origin: str
    ) -> tuple[dict[str, Any], dict[str, Any]]:
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
        usable_outbound = [
            item
            for item in outbound
            if _minutes(item["arrival_time"]) > _minutes(item["departure_time"])
            and _minutes(item["arrival_time"]) <= 14 * 60
        ]
        usable_return = [
            item
            for item in returning
            if _minutes(item["arrival_time"]) > _minutes(item["departure_time"])
            and _minutes(item["departure_time"]) >= 14 * 60
        ]
        usable_outbound = [
            item for item in usable_outbound if self._public_page_index(item) == 0
        ]
        usable_return = [
            item for item in usable_return if self._public_page_index(item) == 0
        ]
        if (
            slot.synthesis_profile == "chinatravel_official_hybrid_v2"
            and slot.days == 1
        ):
            outbound_limit = 11 * 60 if slot.include_meal else 13 * 60
            return_limit = 17 * 60 if slot.include_meal else 16 * 60
            usable_outbound = [
                item
                for item in usable_outbound
                if _minutes(item["arrival_time"]) <= outbound_limit
            ]
            usable_return = [
                item
                for item in usable_return
                if _minutes(item["departure_time"]) >= return_limit
            ]
        if not usable_outbound or not usable_return:
            raise SynthesisError(
                f"No same-day {slot.outbound_mode}/{slot.return_mode} transport for "
                f"{origin}<->{slot.destination}."
            )
        return (
            dict(self._choose_transport(usable_outbound, slot.transport_strategy, outbound=True)),
            dict(self._choose_transport(usable_return, slot.transport_strategy, outbound=False)),
        )

    def _choose_transport(
        self,
        records: list[dict[str, Any]],
        strategy: str,
        *,
        outbound: bool,
    ) -> dict[str, Any]:
        def duration(item: Mapping[str, Any]) -> int:
            return _minutes(item["arrival_time"]) - _minutes(item["departure_time"])

        if strategy == "early":
            records.sort(
                key=lambda item: (
                    _minutes(item["arrival_time"] if outbound else item["departure_time"]),
                    _sortable_cost(item),
                )
            )
        elif strategy == "cheap":
            records.sort(key=lambda item: (_sortable_cost(item), duration(item)))
        elif strategy == "short":
            records.sort(key=lambda item: (duration(item), _sortable_cost(item)))
        elif strategy == "balanced":
            records.sort(
                key=lambda item: (_sortable_cost(item) + duration(item) * 0.8, duration(item))
            )
        else:
            raise SynthesisError(f"Unknown transport strategy: {strategy}")
        return self.rng.choice(records[: min(3, len(records))])

    def _execute(
        self,
        slot: PilotSlot,
        public_task: dict[str, Any],
        outbound: dict[str, Any],
        return_transport: dict[str, Any],
        route_mode: str,
    ) -> WitnessResult:
        needs_restaurant = slot.include_meal
        hotel = self._select_hotel(slot, route_mode) if slot.days > 1 else None
        attraction_targets = _attraction_targets(
            slot,
            outbound_arrival=str(outbound["arrival_time"]),
            return_departure=str(return_transport["departure_time"]),
        )
        attractions = self._select_attractions(
            slot,
            outbound_arrival=str(outbound["arrival_time"]),
            return_departure=str(return_transport["departure_time"]),
            needs_restaurant=needs_restaurant,
            hotel=hotel,
            route_mode=route_mode,
            attraction_targets=attraction_targets,
            initial_anchor_ids=(
                str(outbound["destination_anchor_id"]),
                str(return_transport["origin_anchor_id"]),
            ),
        )
        restaurants: list[dict[str, Any]] = []
        if needs_restaurant:
            selected_count = 0
            for target in attraction_targets:
                selected_count += target
                anchor_index = min(selected_count, len(attractions)) - 1
                restaurants.append(
                    self._select_restaurant(
                        slot,
                        anchor=attractions[anchor_index],
                        route_mode=route_mode,
                        excluded={str(item["place_id"]) for item in restaurants},
                        required_cuisine=(
                            str(restaurants[0]["cuisine"]) if restaurants else None
                        ),
                    )
                )
        local_days, return_route = self._schedule_local_days(
            slot,
            attractions,
            hotel,
            restaurants,
            outbound,
            return_transport,
            route_mode,
            attraction_targets,
        )
        self._require_grounded_local_discovery(
            local_days,
            recipe=slot.recipe,
            preference_kinds=slot.preference_kinds,
            required_facets={
                "attraction": str(attractions[0]["category"]),
                **(
                    {"restaurant": str(restaurants[0]["cuisine"])}
                    if restaurants
                    else {}
                ),
            },
            initial_anchor_ids=(
                str(outbound["destination_anchor_id"]),
                str(return_transport["origin_anchor_id"]),
            ),
        )

        env = TravelWeaverEnv(
            self.backend,
            _SingleTaskStore(public_task),  # type: ignore[arg-type]
            page_size=10,
            max_valid_steps=MAX_WITNESS_VALID_STEPS,
        )
        try:
            env.reset(task_id=str(public_task["uid"]))
            self._reveal_and_save_transport(env, outbound, "outbound_transport")
            self._reveal_and_save_transport(env, return_transport, "return_transport")
            places = {
                item.evidence["place_id"]: item.evidence
                for day in local_days
                for item in day
            }
            for place in places.values():
                purpose = {
                    "attraction": "attraction",
                    "restaurant": "meal",
                    "hotel": "hotel",
                }[str(place["entity_type"])]
                self._reveal_and_save_place(env, place, purpose)

            itinerary: list[dict[str, Any]] = []
            for day_index, local_activities in enumerate(local_days, 1):
                activities: list[dict[str, Any]] = []
                if day_index == 1:
                    activities.append(_transport_activity(outbound))
                for local in local_activities:
                    activity: dict[str, Any] = {
                        "candidate_id": local.evidence["place_id"],
                        "type": local.activity_type,
                        "start_time": local.start_time,
                        "end_time": local.end_time,
                    }
                    if local.route is not None:
                        route_step = self._step(
                            env,
                            "get_route",
                            {
                                "origin_place_id": local.route["origin_place_id"],
                                "destination_place_id": local.route["destination_place_id"],
                                "mode": local.route["mode"],
                                "start_time": local.route["segments"][0]["start_time"],
                            },
                        )
                        route = (route_step.observation.tool_result or {})["route"]
                        activity["route_from_previous_id"] = route["route_id"]
                    if local.activity_type == "accommodation":
                        room_type = int(local.evidence["room_type"])
                        activity["room_type"] = room_type
                        activity["rooms"] = math.ceil(slot.travelers / room_type)
                    activities.append(activity)
                if day_index == slot.days:
                    route_step = self._step(
                        env,
                        "get_route",
                        {
                            "origin_place_id": return_route["origin_place_id"],
                            "destination_place_id": return_route["destination_place_id"],
                            "mode": return_route["mode"],
                            "start_time": return_route["segments"][0]["start_time"],
                        },
                    )
                    resolved_return_route = (route_step.observation.tool_result or {})["route"]
                    return_activity = _transport_activity(return_transport)
                    return_activity["route_from_previous_id"] = resolved_return_route["route_id"]
                    activities.append(return_activity)
                itinerary.append({"day": day_index, "activities": activities})

            plan = {
                "people_number": slot.travelers,
                "start_city": public_task["start_city"],
                "target_city": slot.destination,
                "itinerary": itinerary,
            }
            terminal = self._step(env, "submit_plan", {"plan": plan})
            if not terminal.terminated or terminal.reward != 1.0:
                detail = terminal.info.get("reward_detail")
                raise SynthesisError(f"Witness submission did not strictly pass: {detail}")
            result = terminal.observation.tool_result or {}
            selected = {
                "outbound": outbound,
                "return": return_transport,
                "attractions": attractions,
                "restaurant": restaurants[0] if restaurants else None,
                "restaurants": restaurants,
                "hotel": hotel,
                "logic": self._logic_diversity_evidence(slot, attractions),
            }
            return WitnessResult(
                public_task=public_task,
                plan=plan,
                plan_snapshot=dict(result["plan_snapshot"]),
                evidence_bundle=dict(result["evidence_bundle"]),
                reward_detail=dict(terminal.info["reward_detail"]),
                selected=selected,
                route_mode=route_mode,
            )
        finally:
            env.close()

    def _logic_diversity_evidence(
        self,
        slot: PilotSlot,
        attractions: list[dict[str, Any]],
    ) -> dict[str, str]:
        logic: dict[str, str] = {}
        selected_ids = {str(item["place_id"]) for item in attractions}
        records = [
            dict(item)
            for item in self.backend._records("attraction", slot.destination)
            if str(item.get("place_id")) not in selected_ids
        ]
        self.rng.shuffle(records)
        if "attraction_categories_any" in slot.recipe:
            actual = str(attractions[0]["category"])
            alternative = next(
                (
                    str(item["category"])
                    for item in records
                    if str(item.get("category", "")).strip()
                    and str(item["category"]) != actual
                ),
                None,
            )
            if alternative is None:
                raise SynthesisError("No alternative attraction category is available.")
            logic["alternative_attraction_category"] = alternative
        if "exclude_attraction" in slot.recipe:
            excluded = next(
                (str(item["name"]) for item in records if str(item.get("name", "")).strip()),
                None,
            )
            if excluded is None:
                raise SynthesisError("No attraction is available for an exclusion constraint.")
            logic["excluded_attraction_name"] = excluded
        return logic

    def _select_attractions(
        self,
        slot: PilotSlot,
        *,
        outbound_arrival: str,
        return_departure: str,
        needs_restaurant: bool,
        hotel: dict[str, Any] | None,
        route_mode: str,
        attraction_targets: tuple[int, ...],
        initial_anchor_ids: tuple[str, ...] = (),
    ) -> list[dict[str, Any]]:
        records = [
            dict(item)
            for item in self.backend._records("attraction", slot.destination)
            if _valid_price(item.get("price")) and _valid_opening(item)
        ]
        if not records:
            raise SynthesisError("No attraction has complete price and opening-hour evidence.")
        self.rng.shuffle(records)
        required = sum(attraction_targets)
        radius = _attraction_cluster_radius(
            route_mode,
            attraction_count=required,
            needs_restaurant=needs_restaurant,
            has_hotel=hotel is not None,
        )
        anchor = hotel or max(
            records,
            key=lambda candidate: sum(
                _distance_km(candidate, other) <= radius for other in records
            ),
        )
        nearby = [record for record in records if _distance_km(anchor, record) <= radius]
        self.rng.shuffle(nearby)
        if len(nearby) < required:
            raise SynthesisError(
                f"Only {len(nearby)} attractions fit the {route_mode} spatial cluster; "
                f"need {required}."
            )
        selected: list[dict[str, Any]] = []
        for day, target in enumerate(attraction_targets, 1):
            lower = _minutes(outbound_arrival) + 20 if day == 1 else 9 * 60
            upper = _minutes(return_departure) - 20 if day == slot.days else 20 * 60
            if needs_restaurant:
                upper -= 90
            day_matches: list[dict[str, Any]] = []
            while len(day_matches) < target:
                resolved_attractions = len(selected) + len(day_matches)
                allow_named_query = resolved_attractions == 0
                required_facets = (
                    {"attraction": str(selected[0]["category"])} if selected else None
                )
                established_anchor_ids = tuple(
                    dict.fromkeys(
                        (
                            *initial_anchor_ids,
                            *(str(item["place_id"]) for item in selected),
                            *(str(item["place_id"]) for item in day_matches),
                        )
                    )
                )
                candidates = [
                    item
                    for item in nearby
                    if item not in selected
                    and item not in day_matches
                    and _fits_one_hour(item, lower, upper)
                    and not (
                        "attraction_categories_all" in slot.recipe
                        and len(selected) + len(day_matches) == 1
                        and str(item.get("category"))
                        == str((selected + day_matches)[0].get("category"))
                    )
                ]
                if "attraction_count" in slot.recipe and resolved_attractions > 0:
                    # Once the public quantity requirement is visibly unresolved, a
                    # candidate on one of the next three pages is a grounded reason to
                    # continue the same search. Prefer that real cursor path over
                    # silently filling every count-constrained witness from page one.
                    # ``nearby`` was shuffled above, so ties remain seed-derived rather
                    # than imposing a fixed pagination distribution.
                    candidates.sort(
                        key=lambda item: _count_gap_page_priority(
                            self._task_page_index(
                                item,
                                slot.recipe,
                                slot.preference_kinds,
                                required_facets=required_facets,
                                allow_named_query=allow_named_query,
                            )
                        )
                    )
                else:
                    candidates.sort(
                        key=lambda item: (
                            page_index
                            if (
                                page_index := self._task_page_index(
                                    item,
                                    slot.recipe,
                                    slot.preference_kinds,
                                    required_facets=required_facets,
                                    allow_named_query=allow_named_query,
                                )
                            )
                            is not None
                            else MAX_CONSECUTIVE_TOOL_CALLS + 1
                        )
                    )
                if not candidates:
                    break
                item = next(
                    (
                        candidate
                        for candidate in candidates
                        if self._local_candidate_is_grounded(
                            candidate,
                            recipe=slot.recipe,
                            preference_kinds=slot.preference_kinds,
                            required_facets=required_facets,
                            established_anchor_ids=established_anchor_ids,
                            allow_named_query=allow_named_query,
                            resolved_candidate_count=resolved_attractions,
                        )
                    ),
                    None,
                )
                if item is None:
                    break
                page_index = self._task_page_index(
                    item,
                    slot.recipe,
                    slot.preference_kinds,
                    required_facets=required_facets,
                    allow_named_query=allow_named_query,
                )
                if page_index is None:
                    break
                day_matches.append(item)
            if len(day_matches) != target:
                raise SynthesisError(f"Not enough attractions fit day {day} schedule.")
            selected.extend(day_matches)
        return selected

    def _require_grounded_local_discovery(
        self,
        local_days: list[list[_LocalActivity]],
        *,
        recipe: tuple[str, ...] = (),
        preference_kinds: tuple[str, ...] = (),
        required_facets: Mapping[str, str] | None = None,
        initial_anchor_ids: tuple[str, ...] = (),
    ) -> None:
        """Reject hidden local choices that need arbitrary city-wide pagination.

        An unnamed later-page place remains teachable when a public activity-count
        requirement has already been partially resolved and the cursor chain stays
        within the global consecutive-action limit.  Otherwise the actual itinerary
        must supply a visible spatial anchor and expose the place on the first nearby
        page.  Both paths prevent pagination from recovering an arbitrary hidden ID.
        """

        established_anchor_ids = list(dict.fromkeys(initial_anchor_ids))
        resolved_counts: Counter[str] = Counter()
        for local in (item for day in local_days for item in day):
            evidence = local.evidence
            entity_type = str(evidence.get("entity_type"))
            if entity_type not in {"attraction", "restaurant", "hotel"}:
                continue
            allow_named_query = (
                entity_type == "attraction"
                and "include_attraction" in recipe
                and resolved_counts[entity_type] == 0
            ) or (
                entity_type == "restaurant"
                and "include_restaurant" in recipe
                and resolved_counts[entity_type] == 0
            )
            page_index = self._task_page_index(
                evidence,
                recipe,
                preference_kinds,
                required_facets=required_facets,
                allow_named_query=allow_named_query,
            )
            if page_index == 0:
                established_anchor_ids.append(str(evidence["place_id"]))
                resolved_counts[entity_type] += 1
                continue
            if (
                entity_type == "attraction"
                and "attraction_count" in recipe
                and resolved_counts[entity_type] > 0
                and page_index is not None
                and page_index <= MAX_CONSECUTIVE_TOOL_CALLS
            ):
                established_anchor_ids.append(str(evidence["place_id"]))
                resolved_counts[entity_type] += 1
                continue
            if local.route is None:
                raise SynthesisError(
                    "Later-page local witness has no route anchor for grounded discovery."
                )
            candidate_id = str(evidence["place_id"])
            route_anchor_id = str(local.route["origin_place_id"])
            anchor_ids = list(dict.fromkeys((route_anchor_id, *established_anchor_ids)))
            if not self._visible_on_nearby_first_page(
                evidence, anchor_ids=anchor_ids
            ):
                raise SynthesisError(
                    "Later-page local witness is not visible on the first nearby page "
                    f"from an established itinerary anchor: {candidate_id}."
                )
            established_anchor_ids.append(candidate_id)
            resolved_counts[entity_type] += 1

    def _local_candidate_is_grounded(
        self,
        item: Mapping[str, Any],
        *,
        recipe: tuple[str, ...],
        preference_kinds: tuple[str, ...],
        required_facets: Mapping[str, str] | None,
        established_anchor_ids: tuple[str, ...],
        allow_named_query: bool = True,
        resolved_candidate_count: int = 0,
    ) -> bool:
        """Check discovery feasibility before committing a local witness choice."""

        page_index = self._task_page_index(
            item,
            recipe,
            preference_kinds,
            required_facets=required_facets,
            allow_named_query=allow_named_query,
        )
        if page_index == 0:
            return True
        if (
            str(item.get("entity_type")) == "attraction"
            and "attraction_count" in recipe
            and resolved_candidate_count > 0
            and page_index is not None
            and page_index <= MAX_CONSECUTIVE_TOOL_CALLS
        ):
            return True
        return self._visible_on_nearby_first_page(
            item, anchor_ids=established_anchor_ids
        )

    def _visible_on_nearby_first_page(
        self,
        item: Mapping[str, Any],
        *,
        anchor_ids: tuple[str, ...] | list[str],
    ) -> bool:
        candidate_id = str(item["place_id"])
        entity_type = str(item["entity_type"])
        for anchor_id in dict.fromkeys(anchor_ids):
            cache_key = (str(anchor_id), candidate_id)
            cached = self._nearby_first_page_visibility.get(cache_key)
            if cached is True:
                return True
            if cached is False:
                continue
            visible = False
            for radius in (2, 5, 10, 20, 50):
                page_key = (str(anchor_id), entity_type, radius)
                visible_ids = self._nearby_first_page_ids.get(page_key)
                if visible_ids is None:
                    nearby = self.backend.search_nearby(
                        place_id=str(anchor_id),
                        category=entity_type,
                        radius_km=radius,
                        top_k=10,
                    )
                    visible_ids = frozenset(
                        str(candidate.get("place_id")) for candidate in nearby
                    )
                    self._nearby_first_page_ids[page_key] = visible_ids
                if candidate_id in visible_ids:
                    visible = True
                    break
            self._nearby_first_page_visibility[cache_key] = visible
            if visible:
                return True
        return False

    def _task_page_index(
        self,
        item: Mapping[str, Any],
        recipe: tuple[str, ...],
        preference_kinds: tuple[str, ...] = (),
        *,
        required_facets: Mapping[str, str] | None = None,
        allow_named_query: bool = True,
    ) -> int | None:
        """Locate an entity under filters that the resulting task will state."""

        entity_type = str(item.get("entity_type"))
        city = str(item.get("city"))
        if "lower_total_cost" in preference_kinds:
            cache_key = (entity_type, city, "sort_by", "price")
            search = {
                "attraction": self.backend.search_attractions,
                "restaurant": self.backend.search_restaurants,
                "hotel": self.backend.search_hotels,
            }.get(entity_type)
            if search is not None:
                self._cache_public_page_indices(
                    cache_key,
                    lambda: search(city=city, sort_by="price"),
                )
                return self._public_page_indices[cache_key].get(str(item["place_id"]))
        include_key = {
            "attraction": "include_attraction",
            "restaurant": "include_restaurant",
        }.get(entity_type)
        if allow_named_query and include_key is not None and include_key in recipe:
            name = str(item.get("name", ""))
            search = (
                self.backend.search_attractions
                if entity_type == "attraction"
                else self.backend.search_restaurants
            )
            cache_key = (entity_type, city, "query", name)
            self._cache_public_page_indices(
                cache_key,
                lambda: search(city=city, query=name),
            )
            return self._public_page_indices[cache_key].get(str(item["place_id"]))
        if entity_type == "attraction" and any(
            key in recipe
            for key in (
                "attraction_category",
                "attraction_categories_all",
                "attraction_categories_any",
            )
        ):
            category = str((required_facets or {}).get("attraction") or item.get("category"))
            if str(item.get("category")) != category:
                return self._public_page_index(item)
            cache_key = (entity_type, city, "category", category)
            self._cache_public_page_indices(
                cache_key,
                lambda: self.backend.search_attractions(city=city, category=category),
            )
            return self._public_page_indices[cache_key].get(str(item["place_id"]))
        if entity_type == "restaurant" and "restaurant_cuisine" in recipe:
            cuisine = str((required_facets or {}).get("restaurant") or item.get("cuisine"))
            if str(item.get("cuisine")) != cuisine:
                return self._public_page_index(item)
            cache_key = (entity_type, city, "cuisine", cuisine)
            self._cache_public_page_indices(
                cache_key,
                lambda: self.backend.search_restaurants(city=city, cuisine=cuisine),
            )
            return self._public_page_indices[cache_key].get(str(item["place_id"]))
        if entity_type == "hotel" and "hotel_attribute" in recipe:
            hotel_type = str(item.get("hotel_type"))
            cache_key = (entity_type, city, "hotel_type", hotel_type)
            self._cache_public_page_indices(
                cache_key,
                lambda: self.backend.search_hotels(city=city, hotel_type=hotel_type),
            )
            return self._public_page_indices[cache_key].get(str(item["place_id"]))
        if entity_type == "hotel" and "room_type" in recipe:
            room_type = int(item["room_type"])
            cache_key = (entity_type, city, "room_type", room_type)
            self._cache_public_page_indices(
                cache_key,
                lambda: self.backend.search_hotels(city=city, room_type=room_type),
            )
            return self._public_page_indices[cache_key].get(str(item["place_id"]))
        return self._public_page_index(item)

    def _select_hotel(self, slot: PilotSlot, route_mode: str) -> dict[str, Any]:
        records = [
            dict(item)
            for item in self.backend._records("hotel", slot.destination)
            if _valid_price(item.get("price"))
            and isinstance(item.get("room_type"), int)
            and int(item["room_type"]) > 0
            and (
                "hotel_attribute" not in slot.recipe
                or isinstance(item.get("hotel_type"), str)
                and bool(str(item["hotel_type"]).strip())
            )
        ]
        # A trip has only one hotel witness, so later-page lodging normally has no
        # visible unresolved predicate.  A walking cluster is the exception: the
        # selected hotel must be close to a planned local place and the grounded
        # nearby-first-page path is validated after scheduling.
        max_page = 0
        records = [
            item
            for item in records
            if (
                page_index := self._task_page_index(
                    item, slot.recipe, slot.preference_kinds
                )
            )
            is not None
            and (route_mode == "walk" or page_index <= max_page)
        ]
        if not records:
            raise SynthesisError("No hotel has complete price and room evidence.")
        attractions = [
            dict(item)
            for item in self.backend._records("attraction", slot.destination)
            if _valid_price(item.get("price")) and _valid_opening(item)
        ]
        radius = {"walk": 1.0, "metro": 20.0, "taxi": 15.0}[route_mode]
        self.rng.shuffle(records)
        nearby_counts = {
            str(hotel["place_id"]): sum(
                _distance_km(hotel, attraction) <= radius for attraction in attractions
            )
            for hotel in records
        }
        records.sort(key=lambda hotel: -nearby_counts[str(hotel["place_id"])])
        if route_mode == "walk":
            required = slot.days * slot.attractions_per_day
            records = [
                hotel
                for hotel in records
                if nearby_counts[str(hotel["place_id"])] >= required
                and any(
                    self._task_page_index(
                        attraction,
                        slot.recipe,
                        slot.preference_kinds,
                    )
                    == 0
                    for attraction in attractions
                    if _distance_km(hotel, attraction) <= radius
                )
            ]
            if not records:
                raise SynthesisError(
                    "No hotel anchors enough attractions around a first-page walking start."
                )
        return self.rng.choice(records[: min(10, len(records))])

    def _select_restaurant(
        self,
        slot: PilotSlot,
        *,
        anchor: Mapping[str, Any],
        route_mode: str,
        excluded: set[str] | None = None,
        required_cuisine: str | None = None,
    ) -> dict[str, Any]:
        records = [
            dict(item)
            for item in self.backend._records("restaurant", slot.destination)
            if _valid_price(item.get("price"))
            and _valid_opening(item)
            and isinstance(item.get("cuisine"), str)
            and bool(str(item["cuisine"]).strip())
        ]
        radius = {"walk": 2.0, "metro": 20.0, "taxi": 30.0}[route_mode]
        records = [record for record in records if _distance_km(anchor, record) <= radius]
        records = [
            record for record in records if str(record["place_id"]) not in (excluded or set())
        ]
        max_page = MAX_CONSECUTIVE_TOOL_CALLS
        records = [
            record
            for record in records
            if (
                page_index := self._task_page_index(
                    record,
                    slot.recipe,
                    slot.preference_kinds,
                    required_facets=(
                        {"restaurant": required_cuisine}
                        if required_cuisine is not None
                        else None
                    ),
                    allow_named_query=not excluded,
                )
            )
            is not None
            and (route_mode == "walk" or page_index <= max_page)
        ]
        records.sort(key=lambda record: _distance_km(anchor, record))
        if not records:
            raise SynthesisError("No restaurant has complete price, cuisine, and hours.")
        first_page = [
            record
            for record in records
            if self._task_page_index(
                record,
                slot.recipe,
                slot.preference_kinds,
                required_facets=(
                    {"restaurant": required_cuisine}
                    if required_cuisine is not None
                    else None
                ),
                allow_named_query=not excluded,
            )
            == 0
        ]
        pool = first_page or records
        return self.rng.choice(pool[: min(20, len(pool))])

    def _schedule_local_days(
        self,
        slot: PilotSlot,
        attractions: list[dict[str, Any]],
        hotel: dict[str, Any] | None,
        restaurants: list[dict[str, Any]],
        outbound: Mapping[str, Any],
        return_transport: Mapping[str, Any],
        route_mode: str,
        attraction_targets: tuple[int, ...],
    ) -> tuple[list[list[_LocalActivity]], dict[str, Any]]:
        scheduled: list[list[_LocalActivity]] = []
        attractions_by_day: list[list[dict[str, Any]]] = []
        offset = 0
        for target in attraction_targets:
            attractions_by_day.append(attractions[offset : offset + target])
            offset += target
        previous_day_hotel: Mapping[str, Any] | None = None
        final_previous: Mapping[str, Any] | None = None
        final_previous_end: int | None = None
        for day, day_attractions in enumerate(attractions_by_day, 1):
            earliest = _minutes(outbound["arrival_time"]) if day == 1 else 8 * 60
            latest = (
                _minutes(return_transport["departure_time"])
                if day == slot.days
                else 20 * 60
            )
            day_items: list[_LocalActivity] = []
            if day == 1:
                previous = {
                    "place_id": outbound["destination_anchor_id"],
                    "entity_type": "route_anchor",
                }
            else:
                if previous_day_hotel is None:
                    raise SynthesisError("Cross-day route has no preceding hotel anchor.")
                previous = previous_day_hotel
            previous_end = earliest
            for attraction in day_attractions:
                leg_mode = _local_route_mode(route_mode, previous, attraction)
                route = self.backend.get_route(
                    origin_place_id=str(previous["place_id"]),
                    destination_place_id=str(attraction["place_id"]),
                    mode=leg_mode,
                    start_time=_clock(previous_end),
                )
                _validate_route(route, leg_mode)
                previous_end = _route_end(route)
                attraction_start = max(previous_end, _minutes(attraction["open_time"]))
                attraction_end = attraction_start + 60
                if attraction_end > min(latest, _minutes(attraction["close_time"])):
                    raise SynthesisError(f"Attraction does not fit day {day} after routing.")
                day_items.append(
                    _LocalActivity(
                        evidence=attraction,
                        activity_type="attraction",
                        start_time=_clock(attraction_start),
                        end_time=_clock(attraction_end),
                        route=route,
                    )
                )
                previous = attraction
                previous_end = attraction_end
            assert previous is not None
            restaurant = restaurants[day - 1] if day <= len(restaurants) else None
            if restaurant is not None:
                leg_mode = _local_route_mode(route_mode, previous, restaurant)
                route = self.backend.get_route(
                    origin_place_id=str(previous["place_id"]),
                    destination_place_id=str(restaurant["place_id"]),
                    mode=leg_mode,
                    start_time=_clock(previous_end),
                )
                _validate_route(route, leg_mode)
                route_end = _route_end(route)
                restaurant_open = _minutes(restaurant["open_time"])
                if max(route_end, restaurant_open, 11 * 60) + 60 <= 14 * 60:
                    meal_type = "lunch"
                    meal_start = max(route_end, restaurant_open, 11 * 60)
                    meal_deadline = 14 * 60
                else:
                    meal_type = "dinner"
                    meal_start = max(route_end, restaurant_open, 17 * 60)
                    meal_deadline = 20 * 60
                meal_end = meal_start + 60
                if meal_end > min(
                    _minutes(restaurant["close_time"]), latest, meal_deadline
                ):
                    raise SynthesisError("Selected restaurant does not fit the witness schedule.")
                day_items.append(
                    _LocalActivity(
                        evidence=restaurant,
                        activity_type=meal_type,
                        start_time=_clock(meal_start),
                        end_time=_clock(meal_end),
                        route=route,
                    )
                )
                previous = restaurant
                previous_end = meal_end
            if day < slot.days:
                assert hotel is not None
                leg_mode = _local_route_mode(route_mode, previous, hotel)
                route = self.backend.get_route(
                    origin_place_id=str(previous["place_id"]),
                    destination_place_id=str(hotel["place_id"]),
                    mode=leg_mode,
                    start_time=_clock(previous_end),
                )
                _validate_route(route, leg_mode)
                hotel_start = _route_end(route)
                if hotel_start >= 24 * 60:
                    raise SynthesisError("Hotel route ends outside the itinerary day.")
                day_items.append(
                    _LocalActivity(
                        evidence=hotel,
                        activity_type="accommodation",
                        start_time=_clock(hotel_start),
                        end_time="24:00",
                        route=route,
                    )
                )
                previous_day_hotel = hotel
                final_previous = hotel
                final_previous_end = 24 * 60
            elif previous_end > _minutes(return_transport["departure_time"]):
                raise SynthesisError("Local activities overlap the return transport.")
            else:
                final_previous = previous
                final_previous_end = previous_end
            scheduled.append(day_items)
        if final_previous is None or final_previous_end is None:
            raise SynthesisError("Witness has no final local activity for return routing.")
        return_mode = _local_route_mode(
            route_mode,
            final_previous,
            {"entity_type": "route_anchor"},
        )
        return_route = self.backend.get_route(
            origin_place_id=str(final_previous["place_id"]),
            destination_place_id=str(return_transport["origin_anchor_id"]),
            mode=return_mode,
            start_time=_clock(final_previous_end),
        )
        _validate_route(return_route, return_mode)
        if _route_end(return_route) > _minutes(return_transport["departure_time"]):
            raise SynthesisError("Route to the return station overlaps return transport.")
        return scheduled, dict(return_route)

    def _reveal_and_save_transport(
        self, env: TravelWeaverEnv, transport: Mapping[str, Any], purpose: str
        ) -> None:
        step = self._step(
            env,
            "search_intercity_transport",
            {
                "origin_city": transport["origin_city"],
                "destination_city": transport["destination_city"],
                "mode": transport["mode"],
                "earliest_departure": _coarse_departure(str(transport["departure_time"])),
            },
        )
        self._require_first_page_visibility(env, step, str(transport["transport_id"]))
        self._step(
            env,
            "save_candidate",
            {"entity_id": transport["transport_id"], "purpose": purpose},
        )

    def _reveal_and_save_place(
        self, env: TravelWeaverEnv, place: Mapping[str, Any], purpose: str
    ) -> None:
        tool = {
            "attraction": "search_attractions",
            "restaurant": "search_restaurants",
            "hotel": "search_hotels",
        }[str(place["entity_type"])]
        step = self._step(
            env,
            tool,
            {"city": place["city"], "query": place["name"]},
        )
        self._require_first_page_visibility(env, step, str(place["place_id"]))
        self._step(
            env,
            "save_candidate",
            {"entity_id": place["place_id"], "purpose": purpose},
        )

    @staticmethod
    def _require_first_page_visibility(
        env: TravelWeaverEnv, step: Any, entity_id: str
    ) -> None:
        del env
        result = step.observation.tool_result or {}
        visible = {
            item.get("place_id") or item.get("transport_id") for item in result["items"]
        }
        if entity_id not in visible:
            raise SynthesisError(
                "Witness target is not visible on the first public result page: " + entity_id
            )

    def _public_page_index(self, item: Mapping[str, Any]) -> int | None:
        """Locate an entity in the unfiltered public search available to the teacher."""

        entity_type = str(item.get("entity_type") or item.get("mode"))
        if entity_type == "attraction":
            cache_key = (entity_type, str(item["city"]))
            self._cache_public_page_indices(
                cache_key,
                lambda: self.backend.search_attractions(city=str(item["city"])),
            )
            entity_id = str(item["place_id"])
        elif entity_type == "restaurant":
            cache_key = (entity_type, str(item["city"]))
            self._cache_public_page_indices(
                cache_key,
                lambda: self.backend.search_restaurants(city=str(item["city"])),
            )
            entity_id = str(item["place_id"])
        elif entity_type == "hotel":
            cache_key = (entity_type, str(item["city"]))
            self._cache_public_page_indices(
                cache_key,
                lambda: self.backend.search_hotels(city=str(item["city"])),
            )
            entity_id = str(item["place_id"])
        elif entity_type in {"train", "airplane"}:
            earliest_departure = _coarse_departure(str(item["departure_time"]))
            cache_key = (
                entity_type,
                str(item["origin_city"]),
                str(item["destination_city"]),
                earliest_departure,
            )
            self._cache_public_page_indices(
                cache_key,
                lambda: self.backend.search_intercity_transport(
                    origin_city=str(item["origin_city"]),
                    destination_city=str(item["destination_city"]),
                    mode=entity_type,
                    earliest_departure=earliest_departure,
                ),
            )
            entity_id = str(item["transport_id"])
        else:
            return None
        return self._public_page_indices[cache_key].get(entity_id)

    def _cache_public_page_indices(
        self,
        cache_key: tuple[object, ...],
        search: Callable[[], list[dict[str, Any]]],
    ) -> None:
        if cache_key in self._public_page_indices:
            return
        rows = search()
        self._public_page_indices[cache_key] = {
            str(row.get("place_id") or row.get("transport_id")): index // 10
            for index, row in enumerate(rows)
        }

    @staticmethod
    def _step(env: TravelWeaverEnv, tool: str, arguments: dict[str, Any]) -> Any:
        step = env.step({"tool": tool, "arguments": arguments})
        if not step.info.get("valid_action"):
            error = step.observation.error or {}
            raise SynthesisError(f"Witness tool {tool} failed: {error.get('message', error)}")
        if step.truncated:
            raise SynthesisError("Witness exceeded the environment step limit.")
        return step

    @staticmethod
    def _route_mode_order(preferred: str) -> tuple[str, ...]:
        # A taxi fallback erased walking from otherwise valid batches, so walking
        # remains strict and lets the outer pipeline resample the candidate.  Metro
        # retains the historical taxi fallback because network availability can be
        # absent independently of the spatial candidate cluster.
        return (preferred,) if preferred == "walk" else tuple(
            dict.fromkeys((preferred, "taxi"))
        )


def _attraction_targets(
    slot: PilotSlot,
    *,
    outbound_arrival: str,
    return_departure: str,
) -> tuple[int, ...]:
    """Keep complete days dense while respecting the smaller first/last-day envelope."""

    desired = slot.attractions_per_day
    if slot.synthesis_profile != "chinatravel_official_hybrid_v2":
        return (desired,) * slot.days
    targets: list[int] = []
    for day in range(1, slot.days + 1):
        target = desired
        if day == 1 and _minutes(outbound_arrival) > 10 * 60:
            target = min(target, 1)
        if day == slot.days and _minutes(return_departure) < 18 * 60:
            target = min(target, 1)
        targets.append(max(1, target))
    return tuple(targets)


def _count_gap_page_priority(page_index: int | None) -> tuple[int, int]:
    """Rank a real, bounded continuation ahead of page-one count candidates."""

    if page_index is not None and 1 <= page_index <= MAX_CONSECUTIVE_TOOL_CALLS:
        return (0, page_index)
    if page_index == 0:
        return (1, 0)
    return (2, page_index if page_index is not None else 10**9)


def _transport_activity(transport: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": transport["transport_id"],
        "type": transport["mode"],
        "start_time": transport["departure_time"],
        "end_time": transport["arrival_time"],
    }


def _valid_price(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0


def _sortable_cost(item: Mapping[str, Any]) -> float:
    value = item.get("cost")
    return float(value) if isinstance(value, (int, float)) and value is not None else math.inf


def _valid_opening(item: Mapping[str, Any]) -> bool:
    try:
        return _minutes(item["close_time"]) > _minutes(item["open_time"])
    except (KeyError, TypeError, ValueError):
        return False


def _fits_one_hour(item: Mapping[str, Any], lower: int, upper: int) -> bool:
    start = max(lower, _minutes(item["open_time"]))
    return start + 60 <= min(upper, _minutes(item["close_time"]))


def _route_end(route: Mapping[str, Any]) -> int:
    segments = route.get("segments")
    if not isinstance(segments, list) or not segments:
        raise SynthesisError("Route has no usable segments.")
    end_time = segments[-1].get("end_time")
    return _minutes(end_time)


def _local_route_mode(
    preferred_mode: str,
    origin: Mapping[str, Any],
    destination: Mapping[str, Any],
) -> str:
    """Keep terminal transfers outside an otherwise all-walking local itinerary."""

    touches_terminal = any(
        endpoint.get("entity_type") == "route_anchor"
        for endpoint in (origin, destination)
    )
    return "taxi" if preferred_mode == "walk" and touches_terminal else preferred_mode


def _attraction_cluster_radius(
    route_mode: str,
    *,
    attraction_count: int,
    needs_restaurant: bool,
    has_hotel: bool,
) -> float:
    if (
        route_mode == "walk"
        and attraction_count == 1
        and not needs_restaurant
        and not has_hotel
    ):
        # With only one local place, both routes touch a terminal anchor and are
        # exempt transfers; there is no pair of local places that must be walkable.
        return 15.0
    return {"walk": 1.0, "metro": 20.0, "taxi": 15.0}[route_mode]


def _validate_route(route: Mapping[str, Any], mode: str) -> None:
    segments = route.get("segments")
    if not isinstance(segments, list) or not segments:
        raise SynthesisError("Route has no usable segments.")
    distances = [_numeric_distance(segment.get("distance")) for segment in segments]
    total_distance = sum(distances)
    if mode == "walk" and total_distance > 2.0:
        raise SynthesisError(f"Walking leg is implausibly long: {total_distance:.2f} km.")
    # Airport and high-speed rail boundary legs can legitimately cross the urban fringe.
    if mode == "taxi" and total_distance > 80.0:
        raise SynthesisError(f"Taxi leg is implausibly long: {total_distance:.2f} km.")
    if mode == "metro":
        walking_distance = sum(
            distance
            for segment, distance in zip(segments, distances, strict=True)
            if str(segment.get("mode", "")).lower() == "walk"
        )
        if total_distance > 40.0 or walking_distance > 2.0:
            raise SynthesisError(
                "Metro route exceeds the 40 km total or 2 km walking realism bound."
            )


def _numeric_distance(value: Any) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return max(0.0, float(value))
    raise SynthesisError(f"Route segment has no numeric distance: {value!r}")


def _distance_km(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    try:
        latitude_left = math.radians(float(left["latitude"]))
        longitude_left = math.radians(float(left["longitude"]))
        latitude_right = math.radians(float(right["latitude"]))
        longitude_right = math.radians(float(right["longitude"]))
    except (KeyError, TypeError, ValueError) as error:
        raise SynthesisError("Selected place is missing usable coordinates.") from error
    delta_latitude = latitude_right - latitude_left
    delta_longitude = longitude_right - longitude_left
    haversine = (
        math.sin(delta_latitude / 2) ** 2
        + math.cos(latitude_left)
        * math.cos(latitude_right)
        * math.sin(delta_longitude / 2) ** 2
    )
    return 6371.0088 * 2 * math.asin(min(1.0, math.sqrt(haversine)))


def _minutes(value: Any) -> int:
    if not isinstance(value, str) or ":" not in value:
        raise ValueError(f"Invalid clock value: {value!r}")
    hours, minutes = (int(part) for part in value.split(":", 1))
    if not 0 <= hours <= 24 or not 0 <= minutes < 60:
        raise ValueError(f"Invalid clock value: {value!r}")
    return hours * 60 + minutes


def _clock(minutes: int) -> str:
    if not 0 <= minutes <= 24 * 60:
        raise ValueError(f"Minutes outside one itinerary day: {minutes}")
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _coarse_departure(value: str) -> str:
    hour = _minutes(value) // 60
    return f"{hour - hour % 3:02d}:00"
