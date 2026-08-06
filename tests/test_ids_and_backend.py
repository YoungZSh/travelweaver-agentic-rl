from __future__ import annotations

import pytest

from travelweaver_env.errors import BackendQueryError
from travelweaver_env.ids import make_place_id, make_transport_id


def test_stable_place_ids() -> None:
    first = make_place_id(entity_type="hotel", city="杭州", name="  湖畔 酒店 ", source_id=None)
    second = make_place_id(entity_type="hotel", city="杭州", name="湖畔 酒店", source_id=999)
    assert first == second
    assert make_place_id(
        entity_type="attraction", city="杭州", name="任意名称", source_id=1.0
    ).endswith(":1")


def test_stable_transport_id() -> None:
    record = {"TrainID": "G1", "From": "上海", "To": "杭州", "BeginTime": "08:00"}
    assert make_transport_id("train", record) == make_transport_id("train", dict(record))


def test_backend_rejects_unsupported_city_and_cross_city_route(backend) -> None:
    with pytest.raises(BackendQueryError):
        backend.search_attractions(city="西安")
    hangzhou = backend.search_attractions(city="杭州")[0]["place_id"]
    shanghai = backend.search_attractions(city="上海")[0]["place_id"]
    with pytest.raises(BackendQueryError):
        backend.get_route(
            origin_place_id=hangzhou,
            destination_place_id=shanghai,
            mode="walk",
            start_time="09:00",
        )
