from __future__ import annotations

import json
from pathlib import Path

import pytest

from travelweaver.data import JsonlTaskStore
from travelweaver.env import InMemoryBackend, TravelWeaverEnv
from travelweaver.env.ids import make_place_id, make_transport_id


def _place(
    kind: str,
    city: str,
    name: str,
    source_id: int | None,
    latitude: float,
    longitude: float,
    **extra: object,
) -> dict[str, object]:
    return {
        "place_id": make_place_id(entity_type=kind, city=city, name=name, source_id=source_id),
        "entity_type": kind,
        "city": city,
        "name": name,
        "latitude": latitude,
        "longitude": longitude,
        **extra,
    }


@pytest.fixture
def place_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index in range(15):
        records.append(
            _place(
                "attraction",
                "杭州",
                f"景点{index:02d}",
                index + 1,
                30.250 + index * 0.001,
                120.150 + index * 0.001,
                category="公园" if index % 2 == 0 else "博物馆",
                price=float(index * 10),
                open_time="08:00",
                close_time="18:00",
                recommended_min_hours=1.0,
                recommended_max_hours=2.0,
            )
        )
    records.extend(
        [
            _place(
                "restaurant",
                "杭州",
                "西湖餐厅",
                101,
                30.251,
                120.151,
                cuisine="杭帮菜",
                recommended_food="西湖醋鱼",
                price=88.0,
                open_time="10:00",
                close_time="22:00",
            ),
            _place(
                "hotel",
                "杭州",
                "湖畔酒店",
                None,
                30.252,
                120.152,
                hotel_type="湖景,健身房",
                room_type=2,
                price=399.0,
            ),
            _place(
                "attraction",
                "上海",
                "上海景点",
                1,
                31.230,
                121.470,
                category="地标",
                price=0.0,
                open_time="00:00",
                close_time="23:59",
            ),
        ]
    )
    return records


@pytest.fixture
def backend(place_records: list[dict[str, object]]) -> InMemoryBackend:
    train = {
        "mode": "train",
        "source_id": "G1",
        "origin_city": "上海",
        "destination_city": "杭州",
        "origin": "上海虹桥站",
        "destination": "杭州东站",
        "departure_time": "08:00",
        "arrival_time": "09:00",
        "duration_hours": 1.0,
        "cost": 73.0,
    }
    train["transport_id"] = make_transport_id("train", train)
    return_train = {
        "mode": "train",
        "source_id": "G2",
        "origin_city": "杭州",
        "destination_city": "上海",
        "origin": "杭州东站",
        "destination": "上海虹桥站",
        "departure_time": "18:00",
        "arrival_time": "19:00",
        "duration_hours": 1.0,
        "cost": 73.0,
    }
    return_train["transport_id"] = make_transport_id("train", return_train)
    flight = {
        "mode": "airplane",
        "source_id": "FL1",
        "origin_city": "上海",
        "destination_city": "杭州",
        "origin": "上海虹桥国际机场",
        "destination": "杭州萧山国际机场",
        "departure_time": "12:00",
        "arrival_time": "13:00",
        "duration_hours": 1.0,
        "cost": 500.0,
    }
    flight["transport_id"] = make_transport_id("airplane", flight)
    return InMemoryBackend(place_records, [train, return_train, flight])


@pytest.fixture
def task_store(tmp_path: Path) -> JsonlTaskStore:
    public = tmp_path / "easy.public.jsonl"
    oracle = tmp_path / "easy.oracle.jsonl"
    rows = [
        {
            "uid": "task-hangzhou",
            "tag": "easy",
            "start_city": "上海",
            "target_city": "杭州",
            "days": 1,
            "people_number": 1,
            "limit_rooms": False,
            "limits_room_type": False,
            "language": "zh",
            "query": "从上海去杭州玩一天。",
        },
        {
            "uid": "task-shanghai",
            "tag": "easy",
            "start_city": "杭州",
            "target_city": "上海",
            "days": 1,
            "people_number": 2,
            "limit_rooms": False,
            "limits_room_type": False,
            "language": "zh",
            "query": "从杭州去上海玩一天。",
        },
    ]
    with public.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with oracle.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    {"uid": row["uid"], "hard_logic": ["result=True"]},
                    ensure_ascii=False,
                )
                + "\n"
            )
    return JsonlTaskStore(public, oracle)


@pytest.fixture
def env(backend: InMemoryBackend, task_store: JsonlTaskStore) -> TravelWeaverEnv:
    return TravelWeaverEnv(backend, task_store)
