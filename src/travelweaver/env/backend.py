"""Normalized in-memory and ChinaTravel snapshot backends."""

from __future__ import annotations

import math
import re
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from geopy.distance import geodesic

from ..errors import BackendQueryError, DataUnavailableError
from ..paths import project_root
from .ids import make_place_id, make_transport_id, normalize_name


class Backend(Protocol):
    """Minimal backend contract consumed by TravelWeaverEnv."""

    @property
    def supported_cities(self) -> tuple[str, ...]: ...

    def _records(self, kind: str, city: str) -> list[dict[str, Any]]: ...

    def search_attractions(self, **arguments: Any) -> list[dict[str, Any]]: ...

    def search_restaurants(self, **arguments: Any) -> list[dict[str, Any]]: ...

    def search_hotels(self, **arguments: Any) -> list[dict[str, Any]]: ...

    def search_intercity_transport(self, **arguments: Any) -> list[dict[str, Any]]: ...

    def search_nearby(self, **arguments: Any) -> list[dict[str, Any]]: ...

    def inspect_place(self, place_id: str) -> dict[str, Any]: ...

    def get_route(self, **arguments: Any) -> dict[str, Any]: ...


def _json_value(value: Any) -> Any:
    """Convert pandas/numpy values to strict JSON-compatible Python values."""

    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (AttributeError, ValueError):
            pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _first(record: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None and not (isinstance(value, float) and math.isnan(value)):
            return value
    return None


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        match = re.search(r"-?\d+(?:\.\d+)?", str(value))
        return float(match.group(0)) if match else None
    return None if math.isnan(number) or math.isinf(number) else number


def _contains(value: Any, expected: str | None) -> bool:
    if expected is None:
        return True
    return normalize_name(expected) in normalize_name("" if value is None else str(value))


def _minutes(value: str) -> int:
    try:
        parsed = datetime.strptime(value, "%H:%M")
    except (TypeError, ValueError) as error:
        raise BackendQueryError(f"无效时间：{value!r}，应为 HH:MM。") from error
    return parsed.hour * 60 + parsed.minute


def _is_open(record: Mapping[str, Any], at: str | None) -> bool:
    if at is None:
        return True
    open_time = _first(record, "open_time", "opentime", "weekdayopentime")
    close_time = _first(record, "close_time", "endtime", "weekdayclosetime")
    if not open_time or not close_time or "不营业" in {str(open_time), str(close_time)}:
        return False
    try:
        query = _minutes(at)
        opening = _minutes(str(open_time))
        closing = _minutes(str(close_time))
    except BackendQueryError:
        return False
    if opening <= closing:
        return opening <= query <= closing
    return query >= opening or query <= closing


def _add_hours(start_time: str, hours: float) -> str:
    start = datetime.strptime(start_time, "%H:%M")
    end = start + timedelta(hours=hours)
    return end.strftime("%H:%M")


def _taxi_cost(distance_km: float) -> float:
    if distance_km <= 3:
        return 13.0
    return round(13.0 + (distance_km - 3) * 2.3, 2)


class RecordBackend:
    """Backend over normalized records, also used by unit-test fixtures."""

    def __init__(
        self,
        places: Iterable[Mapping[str, Any]],
        transports: Iterable[Mapping[str, Any]] = (),
        *,
        route_provider: Callable[[dict[str, Any], dict[str, Any], str, str], Any] | None = None,
    ) -> None:
        self._places: dict[str, dict[str, Any]] = {}
        self._by_kind_city: dict[tuple[str, str], list[str]] = defaultdict(list)
        for source in places:
            record = _json_value(dict(source))
            required = {"place_id", "entity_type", "city", "name"}
            missing = required.difference(record)
            if missing:
                raise ValueError(f"Place record is missing required fields: {sorted(missing)}")
            place_id = str(record["place_id"])
            if place_id in self._places:
                continue
            self._places[place_id] = record
            self._by_kind_city[(str(record["entity_type"]), str(record["city"]))].append(place_id)

        self._transports = [_json_value(dict(record)) for record in transports]
        self._route_provider = route_provider

    @property
    def supported_cities(self) -> tuple[str, ...]:
        return tuple(sorted({city for _, city in self._by_kind_city}))

    def _records(self, kind: str, city: str) -> list[dict[str, Any]]:
        if city not in self.supported_cities:
            raise BackendQueryError(
                f"不支持城市 {city!r}；可用城市：{', '.join(self.supported_cities)}。"
            )
        return [self._places[place_id] for place_id in self._by_kind_city[(kind, city)]]

    @staticmethod
    def _summary(record: Mapping[str, Any]) -> dict[str, Any]:
        common = ("place_id", "entity_type", "city", "name", "price", "latitude", "longitude")
        by_type = {
            "attraction": (
                "category",
                "open_time",
                "close_time",
                "recommended_min_hours",
                "recommended_max_hours",
            ),
            "restaurant": ("cuisine", "recommended_food", "open_time", "close_time"),
            "hotel": ("hotel_type", "room_type"),
        }
        keys = common + by_type.get(str(record.get("entity_type")), ())
        return {key: record.get(key) for key in keys if key in record}

    def _search(
        self,
        *,
        kind: str,
        city: str,
        query: str | None = None,
        max_price: float | None = None,
        open_at: str | None = None,
        sort_by: str = "name",
        string_filters: Mapping[str, str | None] | None = None,
        exact_filters: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        records = self._records(kind, city)
        filtered: list[dict[str, Any]] = []
        for record in records:
            if not _contains(record.get("name"), query):
                continue
            price = _number(record.get("price"))
            if max_price is not None and (price is None or price > max_price):
                continue
            if not _is_open(record, open_at):
                continue
            if any(
                not _contains(record.get(key), expected)
                for key, expected in (string_filters or {}).items()
            ):
                continue
            if any(record.get(key) != expected for key, expected in (exact_filters or {}).items()):
                continue
            filtered.append(self._summary(record))

        if sort_by == "price":
            filtered.sort(
                key=lambda row: (
                    _number(row.get("price")) is None,
                    _number(row.get("price")) or 0,
                    normalize_name(str(row.get("name", ""))),
                    str(row.get("place_id")),
                )
            )
        else:
            filtered.sort(
                key=lambda row: (
                    normalize_name(str(row.get("name", ""))),
                    str(row.get("place_id")),
                )
            )
        return filtered

    def search_attractions(
        self,
        *,
        city: str,
        query: str | None = None,
        category: str | None = None,
        max_price: float | None = None,
        open_at: str | None = None,
        sort_by: str = "name",
    ) -> list[dict[str, Any]]:
        return self._search(
            kind="attraction",
            city=city,
            query=query,
            max_price=max_price,
            open_at=open_at,
            sort_by=sort_by,
            string_filters={"category": category},
        )

    def search_restaurants(
        self,
        *,
        city: str,
        query: str | None = None,
        cuisine: str | None = None,
        recommended_food: str | None = None,
        max_price: float | None = None,
        open_at: str | None = None,
        sort_by: str = "name",
    ) -> list[dict[str, Any]]:
        return self._search(
            kind="restaurant",
            city=city,
            query=query,
            max_price=max_price,
            open_at=open_at,
            sort_by=sort_by,
            string_filters={"cuisine": cuisine, "recommended_food": recommended_food},
        )

    def search_hotels(
        self,
        *,
        city: str,
        query: str | None = None,
        hotel_type: str | None = None,
        room_type: int | None = None,
        max_price: float | None = None,
        sort_by: str = "name",
    ) -> list[dict[str, Any]]:
        return self._search(
            kind="hotel",
            city=city,
            query=query,
            max_price=max_price,
            sort_by=sort_by,
            string_filters={"hotel_type": hotel_type},
            exact_filters={"room_type": room_type} if room_type is not None else {},
        )

    def search_intercity_transport(
        self,
        *,
        origin_city: str,
        destination_city: str,
        mode: str,
        earliest_departure: str = "00:00",
    ) -> list[dict[str, Any]]:
        earliest = _minutes(earliest_departure)
        filtered = [
            dict(record)
            for record in self._transports
            if record.get("origin_city") == origin_city
            and record.get("destination_city") == destination_city
            and record.get("mode") == mode
            and _minutes(str(record.get("departure_time", "00:00"))) >= earliest
        ]
        filtered.sort(
            key=lambda row: (
                str(row.get("departure_time", "")),
                _number(row.get("cost")) or 0,
                str(row.get("transport_id", "")),
            )
        )
        return filtered

    def search_nearby(
        self,
        *,
        place_id: str,
        category: str,
        radius_km: float = 2.0,
    ) -> list[dict[str, Any]]:
        origin = self._require_place(place_id)
        origin_position = self._position(origin)
        results: list[dict[str, Any]] = []
        for record in self._records(category, str(origin["city"])):
            if record["place_id"] == place_id:
                continue
            try:
                distance = geodesic(origin_position, self._position(record)).km
            except BackendQueryError:
                continue
            if distance <= radius_km:
                item = self._summary(record)
                item["distance_km"] = round(distance, 3)
                results.append(item)
        results.sort(
            key=lambda row: (
                row["distance_km"],
                normalize_name(str(row.get("name", ""))),
                str(row.get("place_id", "")),
            )
        )
        return results

    def inspect_place(self, place_id: str) -> dict[str, Any]:
        return dict(self._require_place(place_id))

    def get_route(
        self,
        *,
        origin_place_id: str,
        destination_place_id: str,
        mode: str,
        start_time: str,
    ) -> dict[str, Any]:
        origin = self._require_place(origin_place_id)
        destination = self._require_place(destination_place_id)
        if origin["city"] != destination["city"]:
            raise BackendQueryError("市内路线的起点和终点必须位于同一城市。")
        _minutes(start_time)
        if self._route_provider is not None:
            raw_route = self._route_provider(origin, destination, mode, start_time)
            if isinstance(raw_route, str) or raw_route is None:
                raise BackendQueryError(f"路线不可用：{raw_route or '无结果'}。")
            return {
                "origin_place_id": origin_place_id,
                "destination_place_id": destination_place_id,
                "city": origin["city"],
                "mode": mode,
                "segments": _json_value(raw_route),
            }

        distance = geodesic(self._position(origin), self._position(destination)).km
        if mode == "metro":
            raise BackendQueryError("内存后端未配置地铁路线引擎。")
        speed = 5.0 if mode == "walk" else 40.0
        duration_hours = distance / speed
        return {
            "origin_place_id": origin_place_id,
            "destination_place_id": destination_place_id,
            "city": origin["city"],
            "mode": mode,
            "segments": [
                {
                    "start": origin["name"],
                    "end": destination["name"],
                    "mode": mode,
                    "start_time": start_time,
                    "end_time": _add_hours(start_time, duration_hours),
                    "cost": 0.0 if mode == "walk" else _taxi_cost(distance),
                    "distance": round(distance, 3),
                }
            ],
        }

    def _require_place(self, place_id: str) -> dict[str, Any]:
        try:
            return self._places[place_id]
        except KeyError as error:
            raise BackendQueryError(f"未知 place_id：{place_id}。") from error

    @staticmethod
    def _position(record: Mapping[str, Any]) -> tuple[float, float]:
        latitude = _number(record.get("latitude"))
        longitude = _number(record.get("longitude"))
        if latitude is None or longitude is None:
            raise BackendQueryError(f"地点 {record.get('name')!r} 缺少坐标。")
        return latitude, longitude


class InMemoryBackend(RecordBackend):
    """Small deterministic backend intended for tests and examples."""


class ChinaTravelBackend(RecordBackend):
    """Direct adapter over ChinaTravel domain classes (never WorldEnv/eval)."""

    def __init__(self, source_root: str | Path | None = None, *, lang: str = "zh") -> None:
        if lang != "zh":
            raise ValueError("The MVP currently supports lang='zh' only.")
        if source_root is None:
            source_root = project_root() / "vendor" / "ChinaTravel"
        self.source_root = Path(source_root).resolve()
        if not (self.source_root / "chinatravel").is_dir():
            raise DataUnavailableError(
                f"ChinaTravel source not found at {self.source_root}. "
                "Run: git submodule update --init --recursive"
            )
        source_text = str(self.source_root)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)

        try:
            from chinatravel.environment.tools.accommodations.apis import Accommodations
            from chinatravel.environment.tools.attractions.apis import Attractions
            from chinatravel.environment.tools.intercity_transport.apis import (
                IntercityTransport,
            )
            from chinatravel.environment.tools.restaurants.apis import Restaurants
            from chinatravel.environment.tools.transportation.apis import Transportation

            self._attractions_api = Attractions(lang=lang)
            self._restaurants_api = Restaurants(lang=lang)
            self._hotels_api = Accommodations(lang=lang)
            self._intercity_api = IntercityTransport(lang=lang)
            self._transportation_api = Transportation(lang=lang)
        except (FileNotFoundError, OSError, KeyError) as error:
            raise DataUnavailableError(
                "ChinaTravel database is unavailable or incomplete. "
                "Run `travelweaver bootstrap chinatravel` first. "
                f"Original error: {error}"
            ) from error

        places: list[dict[str, Any]] = []
        for city, frame in self._attractions_api.data.items():
            places.extend(self._normalize_places("attraction", city, frame.to_dict("records")))
        for city, frame in self._restaurants_api.data.items():
            places.extend(self._normalize_places("restaurant", city, frame.to_dict("records")))
        for city, frame in self._hotels_api.data.items():
            places.extend(self._normalize_places("hotel", city, frame.to_dict("records")))

        super().__init__(places, route_provider=self._china_route)

    @staticmethod
    def _normalize_places(
        kind: str, city: str, rows: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for source in rows:
            row = {str(key): _json_value(value) for key, value in source.items()}
            name = str(_first(row, "name", "hotelname") or "").strip()
            if not name:
                continue
            place: dict[str, Any] = {
                "place_id": make_place_id(
                    entity_type=kind,
                    city=city,
                    name=name,
                    source_id=row.get("id"),
                ),
                "entity_type": kind,
                "city": city,
                "name": name,
                "latitude": _first(row, "lat", "latitude"),
                "longitude": _first(row, "lon", "longitude"),
                "price": _first(row, "price", "cost"),
                "source": "ChinaTravel",
            }
            if kind == "attraction":
                place.update(
                    {
                        "source_id": row.get("id"),
                        "category": row.get("type"),
                        "open_time": row.get("opentime"),
                        "close_time": row.get("endtime"),
                        "date_description": row.get("datedesc"),
                        "recommended_min_hours": row.get("recommendmintime"),
                        "recommended_max_hours": row.get("recommendmaxtime"),
                    }
                )
            elif kind == "restaurant":
                place.update(
                    {
                        "source_id": row.get("id"),
                        "cuisine": row.get("cuisine"),
                        "open_time": _first(row, "opentime", "weekdayopentime"),
                        "close_time": _first(row, "endtime", "weekdayclosetime"),
                        "recommended_food": row.get("recommendedfood"),
                    }
                )
            else:
                room_type = _number(row.get("numbed"))
                place.update(
                    {
                        "hotel_type": row.get("featurehoteltype"),
                        "room_type": int(room_type) if room_type is not None else None,
                    }
                )
            normalized.append(_json_value(place))
        return normalized

    def search_intercity_transport(
        self,
        *,
        origin_city: str,
        destination_city: str,
        mode: str,
        earliest_departure: str = "00:00",
    ) -> list[dict[str, Any]]:
        try:
            frame = self._intercity_api.select(
                origin_city, destination_city, mode, earliest_departure
            )
        except (KeyError, ValueError, TypeError) as error:
            raise BackendQueryError(f"城际交通查询失败：{error}") from error
        if frame is None or isinstance(frame, str):
            return []
        records: list[dict[str, Any]] = []
        for raw in frame.to_dict("records"):
            raw = {str(key): _json_value(value) for key, value in raw.items()}
            records.append(
                {
                    "transport_id": make_transport_id(mode, raw),
                    "mode": mode,
                    "source_id": raw.get("TrainID") or raw.get("FlightID"),
                    "origin_city": origin_city,
                    "destination_city": destination_city,
                    "origin": raw.get("From"),
                    "destination": raw.get("To"),
                    "departure_time": raw.get("BeginTime"),
                    "arrival_time": raw.get("EndTime"),
                    "duration_hours": raw.get("Duration"),
                    "cost": raw.get("Cost"),
                    "train_type": raw.get("TrainType"),
                }
            )
        records.sort(
            key=lambda row: (
                str(row.get("departure_time", "")),
                _number(row.get("cost")) or 0,
                str(row["transport_id"]),
            )
        )
        return records

    def _china_route(
        self,
        origin: dict[str, Any],
        destination: dict[str, Any],
        mode: str,
        start_time: str,
    ) -> Any:
        try:
            return self._transportation_api.goto(
                origin["city"], origin["name"], destination["name"], start_time, mode
            )
        except (KeyError, TypeError, ValueError) as error:
            raise BackendQueryError(f"市内路线查询失败：{error}") from error
