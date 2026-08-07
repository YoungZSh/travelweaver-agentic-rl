"""Construct replayable feasible plans before deriving task constraints."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..env import ChinaTravelBackend, TravelWeaverEnv
from ..errors import BackendQueryError, SynthesisError
from .models import PilotSlot, WitnessResult


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

    def __init__(self, backend: ChinaTravelBackend, *, seed: int) -> None:
        self.backend = backend
        self.rng = random.Random(seed)

    def build(self, slot: PilotSlot, *, origin: str, uid: str) -> WitnessResult:
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
        route_modes = self._route_mode_order(slot.index)
        last_error: Exception | None = None
        for route_mode in route_modes:
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
        if not usable_outbound or not usable_return:
            raise SynthesisError(
                f"No same-day {slot.outbound_mode}/{slot.return_mode} transport for "
                f"{origin}<->{slot.destination}."
            )
        usable_outbound.sort(
            key=lambda item: (_minutes(item["arrival_time"]), _sortable_cost(item))
        )
        usable_return.sort(
            key=lambda item: (-_minutes(item["departure_time"]), _sortable_cost(item))
        )
        return dict(usable_outbound[0]), dict(usable_return[0])

    def _execute(
        self,
        slot: PilotSlot,
        public_task: dict[str, Any],
        outbound: dict[str, Any],
        return_transport: dict[str, Any],
        route_mode: str,
    ) -> WitnessResult:
        needs_restaurant = "innercity_mode" in slot.recipe or any(
            "restaurant" in key for key in slot.recipe
        )
        attractions = self._select_attractions(
            slot,
            outbound_arrival=str(outbound["arrival_time"]),
            return_departure=str(return_transport["departure_time"]),
            needs_restaurant=needs_restaurant,
        )
        hotel = self._select_hotel(slot) if slot.days > 1 else None
        restaurant = self._select_restaurant(slot.destination) if needs_restaurant else None
        local_days = self._schedule_local_days(
            slot,
            attractions,
            hotel,
            restaurant,
            outbound,
            return_transport,
            route_mode,
        )

        env = TravelWeaverEnv(
            self.backend,
            _SingleTaskStore(public_task),  # type: ignore[arg-type]
            page_size=10,
            max_valid_steps=100,
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
                    activities.append(_transport_activity(return_transport))
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
                "restaurant": restaurant,
                "hotel": hotel,
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

    def _select_attractions(
        self,
        slot: PilotSlot,
        *,
        outbound_arrival: str,
        return_departure: str,
        needs_restaurant: bool,
    ) -> list[dict[str, Any]]:
        records = [
            dict(item)
            for item in self.backend._records("attraction", slot.destination)
            if _valid_price(item.get("price")) and _valid_opening(item)
        ]
        self.rng.shuffle(records)
        selected: list[dict[str, Any]] = []
        for day in range(1, slot.days + 1):
            lower = _minutes(outbound_arrival) + 20 if day == 1 else 9 * 60
            upper = _minutes(return_departure) - 20 if day == slot.days else 20 * 60
            if needs_restaurant and day == 1:
                upper -= 90
            match = next(
                (
                    item
                    for item in records
                    if item not in selected and _fits_one_hour(item, lower, upper)
                ),
                None,
            )
            if match is None:
                raise SynthesisError(f"No attraction fits day {day} schedule.")
            selected.append(match)
        return selected

    def _select_hotel(self, slot: PilotSlot) -> dict[str, Any]:
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
        if not records:
            raise SynthesisError("No hotel has complete price and room evidence.")
        return self.rng.choice(records)

    def _select_restaurant(self, city: str) -> dict[str, Any]:
        records = [
            dict(item)
            for item in self.backend._records("restaurant", city)
            if _valid_price(item.get("price"))
            and _valid_opening(item)
            and isinstance(item.get("cuisine"), str)
            and bool(str(item["cuisine"]).strip())
        ]
        self.rng.shuffle(records)
        if not records:
            raise SynthesisError("No restaurant has complete price, cuisine, and hours.")
        return records[0]

    def _schedule_local_days(
        self,
        slot: PilotSlot,
        attractions: list[dict[str, Any]],
        hotel: dict[str, Any] | None,
        restaurant: dict[str, Any] | None,
        outbound: Mapping[str, Any],
        return_transport: Mapping[str, Any],
        route_mode: str,
    ) -> list[list[_LocalActivity]]:
        scheduled: list[list[_LocalActivity]] = []
        for day, attraction in enumerate(attractions, 1):
            earliest = _minutes(outbound["arrival_time"]) + 20 if day == 1 else 9 * 60
            latest = (
                _minutes(return_transport["departure_time"]) - 20
                if day == slot.days
                else 20 * 60
            )
            attraction_start = max(earliest, _minutes(attraction["open_time"]))
            attraction_end = attraction_start + 60
            if attraction_end > min(latest, _minutes(attraction["close_time"])):
                raise SynthesisError(f"Attraction does not fit day {day} after selection.")
            day_items = [
                _LocalActivity(
                    evidence=attraction,
                    activity_type="attraction",
                    start_time=_clock(attraction_start),
                    end_time=_clock(attraction_end),
                )
            ]
            previous = attraction
            previous_end = attraction_end
            if restaurant is not None and day == 1:
                route = self.backend.get_route(
                    origin_place_id=str(previous["place_id"]),
                    destination_place_id=str(restaurant["place_id"]),
                    mode=route_mode,
                    start_time=_clock(previous_end),
                )
                route_end = _route_end(route)
                meal_start = max(route_end, _minutes(restaurant["open_time"]))
                meal_end = meal_start + 60
                if meal_end > min(_minutes(restaurant["close_time"]), latest):
                    raise SynthesisError("Selected restaurant does not fit the witness schedule.")
                day_items.append(
                    _LocalActivity(
                        evidence=restaurant,
                        activity_type="lunch" if meal_start < 15 * 60 else "dinner",
                        start_time=_clock(meal_start),
                        end_time=_clock(meal_end),
                        route=route,
                    )
                )
                previous = restaurant
                previous_end = meal_end
            if day < slot.days:
                assert hotel is not None
                route = self.backend.get_route(
                    origin_place_id=str(previous["place_id"]),
                    destination_place_id=str(hotel["place_id"]),
                    mode=route_mode,
                    start_time=_clock(previous_end),
                )
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
            elif previous_end > _minutes(return_transport["departure_time"]):
                raise SynthesisError("Local activities overlap the return transport.")
            scheduled.append(day_items)
        return scheduled

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
            },
        )
        self._page_until_visible(env, step, str(transport["transport_id"]))
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
        self._page_until_visible(env, step, str(place["place_id"]))
        self._step(
            env,
            "save_candidate",
            {"entity_id": place["place_id"], "purpose": purpose},
        )

    def _page_until_visible(self, env: TravelWeaverEnv, step: Any, entity_id: str) -> None:
        result = step.observation.tool_result or {}
        visible = {
            item.get("place_id") or item.get("transport_id") for item in result["items"]
        }
        while entity_id not in visible:
            cursor = result["page"]["next_cursor"]
            if cursor is None:
                raise SynthesisError(f"Entity was not exposed by its exact search: {entity_id}")
            step = self._step(env, "next_page", {"cursor": cursor})
            result = step.observation.tool_result or {}
            visible = {
                item.get("place_id") or item.get("transport_id")
                for item in result["items"]
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
    def _route_mode_order(index: int) -> tuple[str, ...]:
        preferred = ("taxi", "metro", "walk")[index % 3]
        return tuple(dict.fromkeys((preferred, "taxi")))


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
