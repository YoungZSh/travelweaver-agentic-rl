"""Episode lifecycle and guarded query-tool execution."""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator

from ..data.tasks import JsonlTaskStore
from ..errors import BackendQueryError, EnvironmentStateError, TravelWeaverError
from .backend import Backend
from .ids import make_route_id
from .models import EvidenceBundle, Observation, PlanSnapshot, StepResult
from .tool_schemas import parameter_schema, tool_schemas

ENVIRONMENT_VERSION = "travelweaver-environment-v0.3"
OBSERVATION_VERSION = "travelweaver-observation-v3"
TOOLS_VERSION = "travelweaver-tools-v2-agent"
PLAN_SNAPSHOT_VERSION = "travelweaver-plan-snapshot-v1"
EVIDENCE_BUNDLE_VERSION = "travelweaver-evidence-v1"
QUANTITY_RULES_VERSION = "travelweaver-quantity-rules-v1"

_SEARCH_TOOLS = {
    "search_attractions",
    "search_restaurants",
    "search_hotels",
    "search_intercity_transport",
    "search_nearby",
}


@dataclass
class _CursorState:
    episode_id: str
    tool: str
    query_hash: str
    items: list[dict[str, Any]]
    offset: int


@dataclass
class _ToolOutcome:
    result: dict[str, Any]
    terminal_reason: str | None = None


class TravelWeaverEnv:
    """Deterministic, in-process environment for one complete agent trajectory."""

    def __init__(
        self,
        backend: Backend,
        task_store: JsonlTaskStore,
        *,
        page_size: int = 10,
        max_valid_steps: int = 35,
        max_consecutive_invalid: int = 3,
    ) -> None:
        if page_size <= 0 or max_valid_steps <= 0 or max_consecutive_invalid <= 0:
            raise ValueError("Episode limits must be positive integers.")
        self.backend = backend
        self.task_store = task_store
        self.page_size = page_size
        self.max_valid_steps = max_valid_steps
        self.max_consecutive_invalid = max_consecutive_invalid
        self._closed = False
        self._episode_id: str | None = None
        self._task: dict[str, Any] | None = None
        self._visible_ids: set[str] = set()
        self._visible_entities: dict[str, dict[str, Any]] = {}
        self._candidates: dict[str, dict[str, Any]] = {}
        self._routes: dict[str, dict[str, Any]] = {}
        self._cursors: dict[str, _CursorState] = {}
        self._valid_steps = 0
        self._invalid_streak = 0
        self._done = False
        self._cursor_nonce = 0

    def reset(self, task_id: str | None = None, seed: int | None = None) -> Observation:
        if self._closed:
            raise EnvironmentStateError("Environment is closed.")
        selected = task_id or self.task_store.choose(seed)
        self._episode_id = uuid.uuid4().hex
        self._task = self.task_store.get_public(selected)
        self._visible_ids.clear()
        self._visible_entities.clear()
        self._candidates.clear()
        self._routes.clear()
        self._cursors.clear()
        self._valid_steps = 0
        self._invalid_streak = 0
        self._done = False
        self._cursor_nonce = 0
        return self._observation()

    def step(self, action: Mapping[str, Any]) -> StepResult:
        self._ensure_active()
        try:
            tool, arguments = self._validate_action(action)
            self._guard_visible_ids(tool, arguments)
            outcome = self._execute(tool, arguments)
        except (BackendQueryError, TravelWeaverError, ValueError, TypeError) as error:
            return self._invalid_step(error)

        self._valid_steps += 1
        self._invalid_streak = 0
        terminated = outcome.terminal_reason is not None
        truncated = not terminated and self._valid_steps >= self.max_valid_steps
        if terminated or truncated:
            self._done = True
        observation = self._observation(tool_result=outcome.result)
        return StepResult(
            observation=observation,
            reward=0.0,
            terminated=terminated,
            truncated=truncated,
            info={
                "valid_action": True,
                "termination_reason": outcome.terminal_reason
                or ("step_limit" if truncated else None),
                "tools_version": TOOLS_VERSION,
            },
        )

    def close(self) -> None:
        self._closed = True
        self._done = True
        self._cursors.clear()
        self._visible_ids.clear()
        self._visible_entities.clear()
        self._candidates.clear()
        self._routes.clear()

    @staticmethod
    def tool_schemas() -> list[dict[str, Any]]:
        return tool_schemas()

    def _validate_action(self, action: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
        if not isinstance(action, Mapping):
            raise ValueError("Action must be an object with tool and arguments.")
        if set(action) != {"tool", "arguments"}:
            raise ValueError("Action must contain exactly: tool, arguments.")
        tool = action.get("tool")
        arguments = action.get("arguments")
        if not isinstance(tool, str):
            raise ValueError("Action.tool must be a string.")
        if not isinstance(arguments, dict):
            raise ValueError("Action.arguments must be an object.")
        schema = parameter_schema(tool)
        if schema is None:
            raise ValueError(f"Unknown tool: {tool}")
        errors = sorted(Draft202012Validator(schema).iter_errors(arguments), key=str)
        if errors:
            error = errors[0]
            location = ".".join(str(part) for part in error.absolute_path) or "arguments"
            raise ValueError(f"Invalid {location}: {error.message}")
        return tool, dict(arguments)

    def _guard_visible_ids(self, tool: str, arguments: Mapping[str, Any]) -> None:
        guarded: tuple[str, ...] = ()
        if tool in {"inspect_place", "search_nearby"}:
            guarded = (str(arguments["place_id"]),)
        elif tool == "save_candidate":
            guarded = (str(arguments["entity_id"]),)
        elif tool == "get_route":
            guarded = (
                str(arguments["origin_place_id"]),
                str(arguments["destination_place_id"]),
            )
        unseen = [place_id for place_id in guarded if place_id not in self._visible_ids]
        if unseen:
            raise ValueError(
                "Only place_id values shown in the current episode may be used; unseen: "
                + ", ".join(unseen)
            )

    def _execute(self, tool: str, arguments: dict[str, Any]) -> _ToolOutcome:
        if tool == "next_page":
            return _ToolOutcome(self._next_page(arguments["cursor"]))
        if tool == "save_candidate":
            return _ToolOutcome(self._save_candidate(**arguments))
        if tool == "list_candidates":
            return _ToolOutcome(self._list_candidates(**arguments))
        if tool == "remove_candidate":
            return _ToolOutcome(self._remove_candidate(**arguments))
        if tool == "submit_plan":
            return _ToolOutcome(self._submit_plan(arguments["plan"]), "plan_submitted")
        if tool == "finish_without_plan":
            return _ToolOutcome(
                {
                    "tool": tool,
                    "status": "finished_without_plan",
                    "reason": arguments["reason"],
                },
                "finished_without_plan",
            )
        if tool == "get_route":
            return _ToolOutcome(self._get_route(**arguments))
        method = getattr(self.backend, tool)
        raw = method(**arguments)
        if tool in _SEARCH_TOOLS:
            if not isinstance(raw, list):
                raise BackendQueryError(f"Backend returned invalid search result for {tool}.")
            return _ToolOutcome(self._first_page(tool, arguments, raw))
        if not isinstance(raw, dict):
            raise BackendQueryError(f"Backend returned invalid result for {tool}.")
        return _ToolOutcome({"tool": tool, "item" if tool == "inspect_place" else "route": raw})

    def _get_route(
        self,
        *,
        origin_place_id: str,
        destination_place_id: str,
        mode: str,
        start_time: str,
    ) -> dict[str, Any]:
        route = self.backend.get_route(
            origin_place_id=origin_place_id,
            destination_place_id=destination_place_id,
            mode=mode,
            start_time=start_time,
        )
        normalized = dict(route)
        route_id = make_route_id(normalized)
        normalized["route_id"] = route_id
        self._routes[route_id] = normalized
        return {"tool": "get_route", "route": dict(normalized)}

    def _first_page(
        self, tool: str, arguments: Mapping[str, Any], items: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return self._page(tool, arguments, items, offset=0)

    def _page(
        self,
        tool: str,
        arguments: Mapping[str, Any],
        items: list[dict[str, Any]],
        *,
        offset: int,
    ) -> dict[str, Any]:
        page_items = [dict(item) for item in items[offset : offset + self.page_size]]
        self._record_visible(page_items)
        next_offset = offset + len(page_items)
        next_cursor: str | None = None
        if next_offset < len(items):
            query_json = json.dumps(
                arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            query_hash = hashlib.sha256(query_json.encode("utf-8")).hexdigest()
            next_cursor = self._make_cursor(tool, query_hash, items, next_offset)
        return {
            "tool": tool,
            "items": page_items,
            "page": {
                "offset": offset,
                "page_size": self.page_size,
                "returned": len(page_items),
                "total": len(items),
                "next_cursor": next_cursor,
            },
        }

    def _make_cursor(
        self, tool: str, query_hash: str, items: list[dict[str, Any]], offset: int
    ) -> str:
        assert self._episode_id is not None
        self._cursor_nonce += 1
        material = f"{self._episode_id}:{tool}:{query_hash}:{offset}:{self._cursor_nonce}"
        cursor = "twc_" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:28]
        self._cursors[cursor] = _CursorState(
            episode_id=self._episode_id,
            tool=tool,
            query_hash=query_hash,
            items=items,
            offset=offset,
        )
        return cursor

    def _next_page(self, cursor: str) -> dict[str, Any]:
        state = self._cursors.pop(cursor, None)
        if state is None or state.episode_id != self._episode_id:
            raise ValueError(
                "Cursor is unknown, expired, already used, or belongs to another episode."
            )
        arguments = {"query_hash": state.query_hash}
        return self._page(state.tool, arguments, state.items, offset=state.offset)

    def _record_visible(self, items: list[dict[str, Any]]) -> None:
        for item in items:
            entity_id = item.get("place_id") or item.get("transport_id")
            if isinstance(entity_id, str):
                self._visible_ids.add(entity_id)
                self._visible_entities[entity_id] = dict(item)

    def _save_candidate(
        self, *, entity_id: str, purpose: str, note: str | None = None
    ) -> dict[str, Any]:
        evidence = dict(self._visible_entities[entity_id])
        if "place_id" in evidence:
            evidence = self.backend.inspect_place(entity_id)
        candidate = {
            "candidate_id": entity_id,
            "entity_id": entity_id,
            "entity_type": evidence.get("entity_type") or evidence.get("mode"),
            "name": evidence.get("name")
            or evidence.get("source_id")
            or evidence.get("transport_id"),
            "purpose": purpose,
            "note": note,
            "evidence": evidence,
        }
        status = "updated" if entity_id in self._candidates else "saved"
        self._candidates[entity_id] = candidate
        return {"tool": "save_candidate", "status": status, "candidate": dict(candidate)}

    def _list_candidates(self, *, purpose: str | None = None) -> dict[str, Any]:
        candidates = [
            dict(candidate)
            for candidate in self._candidates.values()
            if purpose is None or candidate["purpose"] == purpose
        ]
        candidates.sort(key=lambda item: str(item["candidate_id"]))
        return {
            "tool": "list_candidates",
            "items": candidates,
            "count": len(candidates),
        }

    def _remove_candidate(self, *, candidate_id: str) -> dict[str, Any]:
        try:
            removed = self._candidates.pop(candidate_id)
        except KeyError as error:
            raise ValueError(f"Candidate is not saved in this episode: {candidate_id}") from error
        return {
            "tool": "remove_candidate",
            "status": "removed",
            "candidate": self._candidate_summary(removed),
        }

    def _submit_plan(self, plan: dict[str, Any]) -> dict[str, Any]:
        assert self._task is not None
        expected = {
            "people_number": self._task.get("people_number"),
            "start_city": self._task.get("start_city"),
            "target_city": self._task.get("target_city"),
        }
        mismatches = [
            key for key, expected_value in expected.items() if plan.get(key) != expected_value
        ]
        if mismatches:
            raise ValueError(
                "Submitted plan does not match the task fields: " + ", ".join(mismatches)
            )

        expected_days = int(self._task["days"])
        itinerary = plan["itinerary"]
        day_numbers = [day["day"] for day in itinerary]
        if day_numbers != list(range(1, expected_days + 1)):
            raise ValueError(
                f"Itinerary days must be exactly 1..{expected_days} in chronological order."
            )

        activity_types: list[str] = []
        transport_directions: set[tuple[str, str]] = set()
        accommodation_count = 0
        normalized_activities: list[dict[str, Any]] = []
        cost_items: list[dict[str, Any]] = []
        used_entities: dict[str, dict[str, Any]] = {}
        used_routes: dict[str, dict[str, Any]] = {}
        people = int(self._task["people_number"])
        for day in itinerary:
            previous_end = 0
            previous_local_id: str | None = None
            previous_local_end: int | None = None
            for activity_index, activity in enumerate(day["activities"]):
                candidate_id = activity["candidate_id"]
                candidate = self._candidates.get(candidate_id)
                if candidate is None:
                    raise ValueError(
                        f"Plan references a candidate that was not saved: {candidate_id}"
                    )
                start = self._plan_minutes(activity["start_time"], allow_24=False)
                end = self._plan_minutes(activity["end_time"], allow_24=True)
                if end <= start:
                    raise ValueError(
                        f"Activity {candidate_id} must end after it starts within a day."
                    )
                if start < previous_end:
                    raise ValueError(f"Activities overlap on day {day['day']}.")
                previous_end = end

                activity_type = activity["type"]
                self._validate_candidate_type(candidate, activity_type)
                activity_types.append(activity_type)
                evidence = dict(candidate["evidence"])
                used_entities[candidate_id] = evidence
                if activity_type == "accommodation":
                    accommodation_count += 1
                    self._validate_room_selection(activity, evidence, people)
                if activity_type in {"train", "airplane"}:
                    self._validate_intercity_time(activity, evidence)
                    transport_directions.add(
                        (str(evidence.get("origin_city")), str(evidence.get("destination_city")))
                    )
                elif evidence.get("city") != self._task["target_city"]:
                    raise ValueError(
                        f"Destination activity is outside {self._task['target_city']}: "
                        f"{candidate_id}"
                    )

                if activity_type not in {"train", "airplane"}:
                    self._validate_opening_hours(activity, evidence)
                    route_id = activity.get("route_from_previous_id")
                    if previous_local_id is None:
                        if route_id is not None:
                            raise ValueError(
                                f"First local activity on day {day['day']} must not "
                                "reference a route."
                            )
                    else:
                        if not isinstance(route_id, str):
                            raise ValueError(
                                f"Activity {candidate_id} must reference route_from_previous_id."
                            )
                        route = self._validate_route_reference(
                            route_id=route_id,
                            origin_id=previous_local_id,
                            destination_id=candidate_id,
                            previous_end=previous_local_end,
                            next_start=start,
                        )
                        used_routes[route_id] = route
                        route_cost = self._route_cost_item(
                            route, people, day["day"], activity_index
                        )
                        cost_items.append(route_cost)
                    previous_local_id = candidate_id
                    previous_local_end = end

                quantity, unit_price = self._activity_quantity_and_price(
                    activity_type, activity, evidence, people
                )
                amount = (
                    round(quantity * unit_price, 2) if unit_price is not None else None
                )
                cost_item = {
                    "kind": "activity",
                    "day": day["day"],
                    "activity_index": activity_index,
                    "candidate_id": candidate_id,
                    "activity_type": activity_type,
                    "quantity": quantity,
                    "unit_price": unit_price,
                    "amount": amount,
                    "verifiable": unit_price is not None,
                }
                cost_items.append(cost_item)
                normalized_activities.append(
                    {
                        "day": day["day"],
                        "activity_index": activity_index,
                        "candidate_id": candidate_id,
                        "entity_type": candidate["entity_type"],
                        "activity_type": activity_type,
                        "start_time": activity["start_time"],
                        "end_time": activity["end_time"],
                        "route_from_previous_id": activity.get("route_from_previous_id"),
                        "rooms": activity.get("rooms"),
                        "room_type": activity.get("room_type"),
                        "derived_quantity": quantity,
                        "unit_price": unit_price,
                        "amount": amount,
                    }
                )

        if "attraction" not in activity_types:
            raise ValueError("A submitted travel plan must include at least one attraction.")
        if accommodation_count < max(0, expected_days - 1):
            raise ValueError(
                f"A {expected_days}-day plan requires at least "
                f"{expected_days - 1} accommodation activities."
            )
        start_city = str(self._task["start_city"])
        target_city = str(self._task["target_city"])
        if start_city != target_city:
            required_directions = {(start_city, target_city), (target_city, start_city)}
            missing = required_directions.difference(transport_directions)
            if missing:
                formatted = ", ".join(f"{origin}->{destination}" for origin, destination in missing)
                raise ValueError(f"Plan is missing intercity transport directions: {formatted}")

        total_cost = (
            round(sum(float(item["amount"]) for item in cost_items), 2)
            if all(item["amount"] is not None for item in cost_items)
            else None
        )
        task_id = str(self._task["uid"])
        snapshot = PlanSnapshot(
            schema_version=PLAN_SNAPSHOT_VERSION,
            task_id=task_id,
            people_number=people,
            start_city=start_city,
            target_city=target_city,
            days=expected_days,
            activities=tuple(normalized_activities),
            total_cost=total_cost,
        )
        evidence_bundle = EvidenceBundle(
            schema_version=EVIDENCE_BUNDLE_VERSION,
            environment_version=ENVIRONMENT_VERSION,
            task_id=task_id,
            entities=used_entities,
            routes=used_routes,
            cost_items=tuple(cost_items),
            total_cost=total_cost,
            quantity_rules_version=QUANTITY_RULES_VERSION,
        )

        return {
            "tool": "submit_plan",
            "status": "accepted",
            "plan": plan,
            "plan_snapshot": snapshot.to_dict(),
            "evidence_bundle": evidence_bundle.to_dict(),
            "candidate_count": len(self._candidates),
            "validation": {
                "structure": True,
                "task_alignment": True,
                "candidate_grounding": True,
                "chronology": True,
                "route_grounding": True,
                "cost_accounting": total_cost is not None,
                "reward_status": "not_implemented",
            },
        }

    @staticmethod
    def _validate_room_selection(
        activity: Mapping[str, Any], evidence: Mapping[str, Any], people: int
    ) -> None:
        rooms = activity.get("rooms")
        room_type = activity.get("room_type")
        if not isinstance(rooms, int) or isinstance(rooms, bool):
            raise ValueError("Accommodation activities require an integer rooms selection.")
        if not isinstance(room_type, int) or isinstance(room_type, bool):
            raise ValueError("Accommodation activities require an integer room_type selection.")
        available = evidence.get("room_type")
        if available is not None and int(available) != room_type:
            raise ValueError(
                f"Selected room_type {room_type} does not match hotel evidence {available}."
            )
        if rooms * room_type < people:
            raise ValueError(
                f"Selected rooms provide {rooms * room_type} beds for {people} travelers."
            )

    @staticmethod
    def _validate_intercity_time(
        activity: Mapping[str, Any], evidence: Mapping[str, Any]
    ) -> None:
        departure = evidence.get("departure_time")
        arrival = evidence.get("arrival_time")
        if departure and activity["start_time"] != departure:
            raise ValueError(
                f"Intercity start_time must match evidence {departure}, "
                f"got {activity['start_time']}."
            )
        if arrival and activity["end_time"] != arrival:
            raise ValueError(
                f"Intercity end_time must match evidence {arrival}, got {activity['end_time']}."
            )

    def _validate_route_reference(
        self,
        *,
        route_id: str,
        origin_id: str,
        destination_id: str,
        previous_end: int | None,
        next_start: int,
    ) -> dict[str, Any]:
        route = self._routes.get(route_id)
        if route is None:
            raise ValueError(f"Plan references an unknown route_id: {route_id}")
        if route.get("origin_place_id") != origin_id:
            raise ValueError(f"Route {route_id} does not start at {origin_id}.")
        if route.get("destination_place_id") != destination_id:
            raise ValueError(f"Route {route_id} does not end at {destination_id}.")
        segments = route.get("segments")
        if not isinstance(segments, list) or not segments:
            raise ValueError(f"Route {route_id} has no verifiable segments.")
        route_start = self._segment_boundary(segments[0], "start_time")
        route_end = self._segment_boundary(segments[-1], "end_time")
        if previous_end is not None and route_start is not None and route_start < previous_end:
            raise ValueError(f"Route {route_id} starts before the previous activity ends.")
        if route_end is not None and route_end > next_start:
            raise ValueError(f"Route {route_id} ends after the next activity starts.")
        return dict(route)

    @classmethod
    def _segment_boundary(cls, segment: Any, key: str) -> int | None:
        if not isinstance(segment, Mapping):
            return None
        value = segment.get(key)
        if not isinstance(value, str):
            return None
        try:
            return cls._plan_minutes(value, allow_24=key == "end_time")
        except (TypeError, ValueError):
            return None

    @classmethod
    def _validate_opening_hours(
        cls, activity: Mapping[str, Any], evidence: Mapping[str, Any]
    ) -> None:
        opening = evidence.get("open_time")
        closing = evidence.get("close_time")
        if not isinstance(opening, str) or not isinstance(closing, str):
            return
        try:
            open_minute = cls._plan_minutes(opening, allow_24=False)
            close_minute = cls._plan_minutes(closing, allow_24=True)
        except (TypeError, ValueError):
            return
        start = cls._plan_minutes(str(activity["start_time"]), allow_24=False)
        end = cls._plan_minutes(str(activity["end_time"]), allow_24=True)
        if open_minute <= close_minute:
            valid = start >= open_minute and end <= close_minute
        else:
            valid = start >= open_minute or end <= close_minute
        if not valid:
            raise ValueError(
                f"Activity {activity['candidate_id']} is outside opening hours "
                f"{opening}-{closing}."
            )

    @staticmethod
    def _number(value: Any) -> float | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) and number >= 0 else None

    @classmethod
    def _activity_quantity_and_price(
        cls,
        activity_type: str,
        activity: Mapping[str, Any],
        evidence: Mapping[str, Any],
        people: int,
    ) -> tuple[int, float | None]:
        if activity_type == "accommodation":
            quantity = int(activity["rooms"])
        else:
            quantity = people
        price_key = "cost" if activity_type in {"train", "airplane"} else "price"
        return quantity, cls._number(evidence.get(price_key))

    @classmethod
    def _route_cost_item(
        cls, route: Mapping[str, Any], people: int, day: int, activity_index: int
    ) -> dict[str, Any]:
        mode = str(route.get("mode"))
        quantity = math.ceil(people / 4) if mode == "taxi" else people if mode == "metro" else 1
        segments = route.get("segments") or []
        prices = [
            cls._number(segment.get("cost"))
            for segment in segments
            if isinstance(segment, Mapping)
        ]
        unit_price = round(sum(price for price in prices if price is not None), 2)
        verifiable = bool(prices) and all(price is not None for price in prices)
        amount = round(quantity * unit_price, 2) if verifiable else None
        return {
            "kind": "route",
            "day": day,
            "activity_index": activity_index,
            "route_id": route.get("route_id"),
            "mode": mode,
            "quantity": quantity,
            "unit_price": unit_price if verifiable else None,
            "amount": amount,
            "verifiable": verifiable,
        }

    @staticmethod
    def _validate_candidate_type(candidate: Mapping[str, Any], activity_type: str) -> None:
        actual = candidate["entity_type"]
        expected = {
            "attraction": "attraction",
            "breakfast": "restaurant",
            "lunch": "restaurant",
            "dinner": "restaurant",
            "accommodation": "hotel",
            "train": "train",
            "airplane": "airplane",
        }[activity_type]
        if actual != expected:
            raise ValueError(
                f"Candidate {candidate['candidate_id']} has type {actual!r}, "
                f"not {expected!r} required by activity {activity_type!r}."
            )

    @staticmethod
    def _plan_minutes(value: str, *, allow_24: bool) -> int:
        if value == "24:00":
            if allow_24:
                return 24 * 60
            raise ValueError("24:00 is allowed only as an activity end_time.")
        hours, minutes = value.split(":")
        return int(hours) * 60 + int(minutes)

    @staticmethod
    def _candidate_summary(candidate: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: candidate.get(key)
            for key in ("candidate_id", "entity_id", "entity_type", "name", "purpose", "note")
        }

    def _invalid_step(self, error: Exception) -> StepResult:
        self._invalid_streak += 1
        terminated = self._invalid_streak >= self.max_consecutive_invalid
        if terminated:
            self._done = True
        observation = self._observation(error={"code": "invalid_action", "message": str(error)})
        return StepResult(
            observation=observation,
            reward=0.0,
            terminated=terminated,
            truncated=False,
            info={
                "valid_action": False,
                "consecutive_invalid_actions": self._invalid_streak,
                "termination_reason": "invalid_action_limit" if terminated else None,
                "tools_version": TOOLS_VERSION,
            },
        )

    def _observation(
        self,
        *,
        tool_result: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> Observation:
        assert self._episode_id is not None and self._task is not None
        return Observation(
            schema_version=OBSERVATION_VERSION,
            environment_version=ENVIRONMENT_VERSION,
            episode_id=self._episode_id,
            task=dict(self._task),
            tool_result=tool_result,
            error=error,
            visible_entity_ids=tuple(sorted(self._visible_ids)),
            visible_route_ids=tuple(sorted(self._routes)),
            candidates=tuple(
                self._candidate_summary(candidate)
                for candidate in sorted(
                    self._candidates.values(), key=lambda item: str(item["candidate_id"])
                )
            ),
            remaining_steps=max(0, self.max_valid_steps - self._valid_steps),
        )

    def _ensure_active(self) -> None:
        if self._closed:
            raise EnvironmentStateError("Environment is closed.")
        if self._episode_id is None:
            raise EnvironmentStateError("Call reset() before step().")
        if self._done:
            raise EnvironmentStateError("Episode has terminated; call reset() for a new episode.")
