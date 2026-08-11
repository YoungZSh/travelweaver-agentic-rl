"""Explicit, replayable world deltas layered over a deterministic backend."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from ..errors import BackendQueryError
from .backend import Backend, _facet_values, _is_open, _number

SCENARIO_VERSION = "travelweaver-scenario-v1"


@dataclass(frozen=True)
class ScenarioEffect:
    """One materialized change to an entity in the pinned base world."""

    effect_id: str
    kind: str
    target_type: str
    target_id: str
    field: str
    before: Any
    after: Any

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScenarioSpec:
    """A stable collection of explicit effects; no hidden RNG is replayed at runtime."""

    base_world_snapshot_version: str
    profile: str
    effects: tuple[ScenarioEffect, ...]
    version: str = SCENARIO_VERSION

    @property
    def scenario_id(self) -> str:
        payload = {
            "base_world_snapshot_version": self.base_world_snapshot_version,
            "effects": [effect.to_dict() for effect in self.effects],
            "profile": self.profile,
            "version": self.version,
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return f"scenario_{hashlib.sha256(encoded).hexdigest()[:16]}"

    @property
    def world_snapshot_version(self) -> str:
        return f"{self.base_world_snapshot_version}+{self.scenario_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "world_snapshot_version": self.world_snapshot_version,
            "base_world_snapshot_version": self.base_world_snapshot_version,
            "profile": self.profile,
            "effects": [effect.to_dict() for effect in self.effects],
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ScenarioSpec:
        raw_effects = value.get("effects")
        if not isinstance(raw_effects, list):
            raise ValueError("Scenario effects must be a list.")
        scenario = cls(
            base_world_snapshot_version=str(value["base_world_snapshot_version"]),
            profile=str(value["profile"]),
            effects=tuple(ScenarioEffect(**dict(effect)) for effect in raw_effects),
            version=str(value.get("version", SCENARIO_VERSION)),
        )
        expected_id = value.get("scenario_id")
        if expected_id is not None and expected_id != scenario.scenario_id:
            raise ValueError("Scenario id does not match its materialized effects.")
        return scenario


class ScenarioBackend:
    """Apply a materialized ScenarioSpec without exposing its effects to the Agent."""

    def __init__(self, base: Backend, scenario: ScenarioSpec) -> None:
        self.base = base
        self.scenario = scenario
        self._unavailable = {
            effect.target_id
            for effect in scenario.effects
            if effect.kind in {"unavailable", "cancelled"}
        }
        self._overrides = {
            (effect.target_id, effect.field): effect.after
            for effect in scenario.effects
            if effect.kind == "field_override"
        }

    @property
    def supported_cities(self) -> tuple[str, ...]:
        return tuple(str(city) for city in self.base.supported_cities)

    def _records(self, kind: str, city: str) -> list[dict[str, Any]]:
        records = self.base._records(kind, city)
        return [
            adjusted
            for record in records
            if (adjusted := self._adjust_place(record)) is not None
        ]

    def search_attractions(self, **arguments: Any) -> list[dict[str, Any]]:
        return self._search_places("search_attractions", arguments)

    def list_attraction_categories(self, city: str) -> dict[str, Any]:
        return {
            "city": city,
            "categories": _facet_values(self._records("attraction", city), "category"),
        }

    def search_restaurants(self, **arguments: Any) -> list[dict[str, Any]]:
        return self._search_places("search_restaurants", arguments)

    def list_restaurant_cuisines(self, city: str) -> dict[str, Any]:
        return {
            "city": city,
            "cuisines": _facet_values(self._records("restaurant", city), "cuisine"),
        }

    def search_restaurants_by_food(self, **arguments: Any) -> list[dict[str, Any]]:
        return self._search_places("search_restaurants_by_food", arguments)

    def search_hotels(self, **arguments: Any) -> list[dict[str, Any]]:
        return self._search_places("search_hotels", arguments)

    def list_hotel_features(self, city: str) -> dict[str, Any]:
        records = self._records("hotel", city)
        room_types = sorted(
            {
                int(value)
                for record in records
                if (value := _number(record.get("room_type"))) is not None and value >= 1
            }
        )
        return {
            "city": city,
            "features": _facet_values(records, "hotel_type"),
            "room_types": room_types,
        }

    def _search_places(
        self,
        method_name: str,
        arguments: dict[str, Any],
    ) -> list[dict[str, Any]]:
        requested_min_price = arguments.get("min_price")
        requested_max_price = arguments.get("max_price")
        base_arguments = dict(arguments)
        base_arguments["min_price"] = None
        base_arguments["max_price"] = None
        records = getattr(self.base, method_name)(**base_arguments)
        adjusted = [
            item
            for record in records
            if (item := self._adjust_place(record)) is not None
            and _within_price_range(item, requested_min_price, requested_max_price)
        ]
        if arguments.get("sort_by", "name") == "price":
            adjusted.sort(key=_price_sort_key)
        return adjusted

    def search_intercity_transport(self, **arguments: Any) -> list[dict[str, Any]]:
        records = self.base.search_intercity_transport(**arguments)
        adjusted: list[dict[str, Any]] = []
        for source in records:
            transport_id = str(source.get("transport_id", ""))
            if transport_id in self._unavailable:
                continue
            record = dict(source)
            for (target_id, field), value in self._overrides.items():
                if target_id == transport_id:
                    record[field] = value
            adjusted.append(record)
        return adjusted

    def search_nearby(self, **arguments: Any) -> list[dict[str, Any]]:
        self._require_available(str(arguments["place_id"]))
        base_arguments = dict(arguments)
        top_k = base_arguments.pop("top_k", None)
        records = self.base.search_nearby(**base_arguments)
        adjusted = [
            item
            for record in records
            if (item := self._adjust_place(record)) is not None
        ]
        return adjusted[: int(top_k)] if top_k is not None else adjusted

    def inspect_place(self, place_id: str) -> dict[str, Any]:
        self._require_available(place_id)
        adjusted = self._adjust_place(self.base.inspect_place(place_id))
        if adjusted is None:
            raise BackendQueryError(f"场景中地点 {place_id!r} 不可用。")
        return adjusted

    def check_place_open(self, place_id: str, at_time: str) -> dict[str, Any]:
        record = self.inspect_place(place_id)
        if record.get("entity_type") not in {"attraction", "restaurant"}:
            raise BackendQueryError("只有景点和餐厅支持开放时间检查。")
        return {
            "place_id": place_id,
            "name": record.get("name"),
            "at_time": at_time,
            "is_open": _is_open(record, at_time),
            "open_time": record.get("open_time"),
            "close_time": record.get("close_time"),
        }

    def get_route(self, **arguments: Any) -> dict[str, Any]:
        self._require_available(str(arguments["origin_place_id"]))
        self._require_available(str(arguments["destination_place_id"]))
        return self.base.get_route(**arguments)

    def _adjust_place(self, source: dict[str, Any]) -> dict[str, Any] | None:
        place_id = str(source.get("place_id", ""))
        if place_id in self._unavailable:
            return None
        record = dict(source)
        for (target_id, field), value in self._overrides.items():
            if target_id == place_id:
                record[field] = value
        return record

    def _require_available(self, entity_id: str) -> None:
        if entity_id in self._unavailable:
            raise BackendQueryError(f"场景中实体 {entity_id!r} 不可用。")


def _within_price_range(record: dict[str, Any], minimum: Any, maximum: Any) -> bool:
    if minimum is None and maximum is None:
        return True
    price = record.get("price")
    if not isinstance(price, (int, float)) or isinstance(price, bool):
        return False
    return (minimum is None or float(price) >= float(minimum)) and (
        maximum is None or float(price) <= float(maximum)
    )


def _price_sort_key(record: dict[str, Any]) -> tuple[bool, float, str, str]:
    price = record.get("price")
    valid = isinstance(price, (int, float)) and not isinstance(price, bool)
    return (
        not valid,
        float(price) if valid else 0.0,
        str(record.get("name", "")),
        str(record.get("place_id", "")),
    )
