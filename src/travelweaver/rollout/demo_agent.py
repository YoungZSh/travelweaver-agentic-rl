"""A deterministic tool-only agent used to verify the complete environment loop."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ..env.environment import TravelWeaverEnv
from ..env.models import Observation, StepResult


@dataclass(frozen=True)
class AgentRun:
    task_id: str
    success: bool
    termination_reason: str | None
    step_count: int
    final_plan: dict[str, Any] | None
    final_reward: float
    reward_detail: dict[str, Any]
    trajectory: tuple[dict[str, Any], ...]

    def to_dict(self, *, include_trajectory: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "task_id": self.task_id,
            "success": self.success,
            "termination_reason": self.termination_reason,
            "step_count": self.step_count,
            "final_plan": self.final_plan,
            "final_reward": self.final_reward,
            "reward_detail": dict(self.reward_detail),
        }
        if include_trajectory:
            payload["trajectory"] = list(self.trajectory)
        return payload


class DemoTravelAgent:
    """Simple policy that uses only TravelWeaver actions, never backend internals."""

    def __init__(self, env: TravelWeaverEnv) -> None:
        self.env = env
        self._trajectory: list[dict[str, Any]] = []
        self._observation: Observation | None = None

    def run(self, task_id: str | None = None, *, seed: int | None = 0) -> AgentRun:
        self._trajectory = []
        self._observation = self.env.reset(task_id=task_id, seed=seed)
        task = self._observation.task
        self._trajectory.append({"event": "reset", "observation": self._observation.to_dict()})

        start_city = str(task["start_city"])
        target_city = str(task["target_city"])
        outbound: dict[str, Any] | None = None
        returning: dict[str, Any] | None = None
        if start_city != target_city:
            outbound = self._find_transport(start_city, target_city, "06:00")
            if outbound is None:
                return self._finish("没有找到可用的去程火车或航班。")
            self._save(outbound, "outbound_transport", "去程城际交通")

        attraction_items = self._items(
            "search_attractions", {"city": target_city, "sort_by": "price"}
        )
        if not attraction_items:
            return self._finish("目标城市没有可用景点。")
        attraction = attraction_items[0]
        attraction_id = str(attraction["place_id"])
        self._call("inspect_place", {"place_id": attraction_id})
        self._save(attraction, "attraction", "主要游览景点")

        restaurant_items = self._items(
            "search_nearby",
            {"place_id": attraction_id, "category": "restaurant", "radius_km": 10},
        )
        if not restaurant_items:
            restaurant_items = self._items("search_restaurants", {"city": target_city})
        restaurant = restaurant_items[0] if restaurant_items else None
        if restaurant is not None:
            self._save(restaurant, "meal", "午餐候选")

        hotel: dict[str, Any] | None = None
        if int(task["days"]) > 1:
            hotel_items = self._items("search_hotels", {"city": target_city})
            if not hotel_items:
                return self._finish("多日行程没有找到可用住宿。")
            hotel = hotel_items[0]
            self._save(hotel, "hotel", "过夜住宿")

        if start_city != target_city:
            returning = self._find_transport(target_city, start_city, "18:00")
            if returning is None:
                return self._finish("没有找到可用的返程火车或航班。")
            self._save(returning, "return_transport", "返程城际交通")

        route_ids: dict[tuple[str, str], str] = {}
        local_sequence: list[dict[str, Any]] = []
        if outbound is not None:
            local_sequence.append({"place_id": outbound["destination_anchor_id"]})
        local_sequence.append(attraction)
        if restaurant is not None:
            local_sequence.append(restaurant)
        if hotel is not None:
            local_sequence.append(hotel)
        if returning is not None:
            local_sequence.append({"place_id": returning["origin_anchor_id"]})
        route_start_times = [
            str(outbound["arrival_time"]) if outbound is not None else "12:00",
            "12:00",
            "13:30",
            "20:00",
        ]
        for index, (origin, destination) in enumerate(
            zip(local_sequence, local_sequence[1:], strict=False)
        ):
            route = self._call(
                "get_route",
                {
                    "origin_place_id": origin["place_id"],
                    "destination_place_id": destination["place_id"],
                    "mode": "walk",
                    "start_time": route_start_times[index],
                },
                require_valid=False,
            )
            route_payload = (route.observation.tool_result or {}).get("route", {})
            if not route.info.get("valid_action") or not isinstance(
                route_payload.get("route_id"), str
            ):
                return self._finish("没有找到可验证的同城活动衔接路线。")
            route_ids[(origin["place_id"], destination["place_id"])] = route_payload[
                "route_id"
            ]

        self._call("list_candidates", {})
        plan = self._build_plan(
            task=task,
            attraction=attraction,
            restaurant=restaurant,
            hotel=hotel,
            outbound=outbound,
            returning=returning,
            route_ids=route_ids,
        )
        submitted = self._call("submit_plan", {"plan": plan}, require_valid=False)
        if not submitted.info.get("valid_action") or not submitted.terminated:
            message = (submitted.observation.error or {}).get("message", "结构化行程未被环境接受。")
            return self._finish(f"提交失败：{message}")
        return self._result(submitted, plan)

    def _find_transport(
        self, origin_city: str, destination_city: str, earliest_departure: str
    ) -> dict[str, Any] | None:
        for mode in ("train", "airplane"):
            items = self._items(
                "search_intercity_transport",
                {
                    "origin_city": origin_city,
                    "destination_city": destination_city,
                    "mode": mode,
                    "earliest_departure": earliest_departure,
                },
            )
            if items:
                return items[0]
        return None

    def _save(self, item: dict[str, Any], purpose: str, note: str) -> None:
        entity_id = item.get("place_id") or item.get("transport_id")
        if not isinstance(entity_id, str):
            raise RuntimeError("Search result does not contain a stable entity id.")
        result = self._call(
            "save_candidate",
            {"entity_id": entity_id, "purpose": purpose, "note": note},
        )
        if not result.info.get("valid_action"):
            raise RuntimeError(f"Unable to save visible candidate {entity_id}.")

    def _items(self, tool: str, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        result = self._call(tool, arguments, require_valid=False)
        if not result.info.get("valid_action"):
            return []
        return list((result.observation.tool_result or {}).get("items", []))

    def _call(
        self, tool: str, arguments: dict[str, Any], *, require_valid: bool = True
    ) -> StepResult:
        action = {"tool": tool, "arguments": arguments}
        result = self.env.step(action)
        self._observation = result.observation
        self._trajectory.append({"event": "step", "action": action, "result": result.to_dict()})
        if require_valid and not result.info.get("valid_action"):
            raise RuntimeError(f"Demo agent action failed: {tool}: {result.observation.error}")
        return result

    def _finish(self, reason: str) -> AgentRun:
        assert self._observation is not None
        return AgentRun(
            task_id=str(self._observation.task["uid"]),
            success=False,
            termination_reason="no_plan",
            step_count=len(
                [event for event in self._trajectory if event["event"] == "step"]
            ),
            final_plan=None,
            final_reward=-1.0,
            reward_detail={"reason": reason},
            trajectory=tuple(self._trajectory),
        )

    def _result(self, terminal: StepResult, plan: dict[str, Any] | None) -> AgentRun:
        assert self._observation is not None
        reward_detail = dict(terminal.info.get("reward_detail") or {})
        return AgentRun(
            task_id=str(self._observation.task["uid"]),
            success=bool(
                terminal.info.get("termination_reason") == "plan_submitted"
                and reward_detail.get("reward_valid")
                and reward_detail.get("all_hard_pass")
            ),
            termination_reason=terminal.info.get("termination_reason"),
            step_count=len([event for event in self._trajectory if event["event"] == "step"]),
            final_plan=plan,
            final_reward=terminal.reward,
            reward_detail=reward_detail,
            trajectory=tuple(self._trajectory),
        )

    @staticmethod
    def _build_plan(
        *,
        task: dict[str, Any],
        attraction: dict[str, Any],
        restaurant: dict[str, Any] | None,
        hotel: dict[str, Any] | None,
        outbound: dict[str, Any] | None,
        returning: dict[str, Any] | None,
        route_ids: dict[tuple[str, str], str],
    ) -> dict[str, Any]:
        days = int(task["days"])
        itinerary: list[dict[str, Any]] = []
        for day_number in range(1, days + 1):
            activities: list[dict[str, Any]] = []
            if day_number == 1 and outbound is not None:
                activities.append(
                    {
                        "candidate_id": outbound["transport_id"],
                        "type": outbound["mode"],
                        "start_time": outbound["departure_time"],
                        "end_time": outbound["arrival_time"],
                    }
                )
            activities.append(
                {
                    "candidate_id": attraction["place_id"],
                    "type": "attraction",
                    "start_time": "10:00",
                    "end_time": "12:00",
                    **(
                        {
                            "route_from_previous_id": route_ids[
                                (outbound["destination_anchor_id"], attraction["place_id"])
                            ]
                        }
                        if day_number == 1 and outbound is not None
                        else {}
                    ),
                }
            )
            if restaurant is not None:
                activities.append(
                    {
                        "candidate_id": restaurant["place_id"],
                        "type": "lunch",
                        "start_time": "12:30",
                        "end_time": "13:30",
                        "route_from_previous_id": route_ids[
                            (attraction["place_id"], restaurant["place_id"])
                        ],
                    }
                )
            if day_number == days and returning is not None:
                previous = restaurant if restaurant is not None else attraction
                activities.append(
                    {
                        "candidate_id": returning["transport_id"],
                        "type": returning["mode"],
                        "start_time": returning["departure_time"],
                        "end_time": returning["arrival_time"],
                        "route_from_previous_id": route_ids[
                            (previous["place_id"], returning["origin_anchor_id"])
                        ],
                    }
                )
            elif hotel is not None:
                previous = restaurant if restaurant is not None else attraction
                room_type = int(hotel.get("room_type") or 1)
                activities.append(
                    {
                        "candidate_id": hotel["place_id"],
                        "type": "accommodation",
                        "start_time": "20:00",
                        "end_time": "24:00",
                        "route_from_previous_id": route_ids[
                            (previous["place_id"], hotel["place_id"])
                        ],
                        "rooms": math.ceil(int(task["people_number"]) / room_type),
                        "room_type": room_type,
                    }
                )
            itinerary.append({"day": day_number, "activities": activities})
        return {
            "people_number": int(task["people_number"]),
            "start_city": task["start_city"],
            "target_city": task["target_city"],
            "itinerary": itinerary,
        }
