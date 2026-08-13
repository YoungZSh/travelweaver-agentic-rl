from __future__ import annotations

from dataclasses import replace

from travelweaver.reward.evaluators import evaluate_constraint
from travelweaver.tasks import ConstraintSpec, build_base_spec


def _spec(constraint: ConstraintSpec):
    return build_base_spec(
        {
            "uid": "logic-test",
            "query": "测试逻辑约束",
            "start_city": "上海",
            "target_city": "北京",
            "days": 2,
            "people_number": 2,
        },
        constraints=(constraint,),
    )


def test_entity_category_supports_multivalue_and_any_of_groups() -> None:
    plan = {
        "activities": [
            {"activity_type": "attraction", "candidate_id": "park"},
            {"activity_type": "attraction", "candidate_id": "museum"},
        ]
    }
    evidence = {
        "entities": {
            "park": {"category": "公园"},
            "museum": {"category": "博物馆"},
        }
    }
    all_constraint = ConstraintSpec(
        id="all",
        kind="entity_category",
        operator="contains",
        value={"values": ["公园", "博物馆"]},
        scope="attraction",
        hardness="hard",
        source_text="至少去一个公园和一个博物馆",
    )
    any_constraint = ConstraintSpec(
        id="any",
        kind="entity_category",
        operator="contains",
        value={"any_of": [["园林"], ["博物馆"]]},
        scope="attraction",
        hardness="hard",
        source_text="园林或博物馆任选其一",
    )

    assert evaluate_constraint(
        all_constraint, _spec(all_constraint), plan, evidence
    ).status == "pass"
    assert evaluate_constraint(
        any_constraint, _spec(any_constraint), plan, evidence
    ).status == "pass"


def test_transport_mode_negative_set_rejects_only_forbidden_modes() -> None:
    constraint = ConstraintSpec(
        id="allowed",
        kind="transport_mode",
        operator="not_in",
        value={"modes": ["walk"], "leg": "all"},
        scope="innercity_route",
        hardness="hard",
        source_text="市内只能坐地铁或出租车，不要步行",
    )
    plan = {"activities": []}

    allowed = evaluate_constraint(
        constraint,
        _spec(constraint),
        plan,
        {
            "entities": {
                "a": {"entity_type": "attraction"},
                "b": {"entity_type": "restaurant"},
                "c": {"entity_type": "hotel"},
            },
            "routes": {
                "r1": {
                    "origin_place_id": "a",
                    "destination_place_id": "b",
                    "mode": "metro",
                },
                "r2": {
                    "origin_place_id": "b",
                    "destination_place_id": "c",
                    "mode": "taxi",
                },
            },
        },
    )
    forbidden = evaluate_constraint(
        constraint,
        _spec(constraint),
        plan,
        {
            "entities": {
                "a": {"entity_type": "attraction"},
                "b": {"entity_type": "restaurant"},
            },
            "routes": {
                "r1": {
                    "origin_place_id": "a",
                    "destination_place_id": "b",
                    "mode": "walk",
                }
            },
        },
    )

    assert allowed.status == "pass"
    assert forbidden.status == "fail"


def test_innercity_mode_v3_excludes_airport_and_station_transfers() -> None:
    constraint = ConstraintSpec(
        id="walk",
        kind="transport_mode",
        operator="eq",
        value={"modes": ["walk"], "leg": "all"},
        scope="innercity_route",
        hardness="hard",
        source_text="市内地点之间都步行",
    )
    spec = _spec(constraint)
    plan = {"activities": []}
    evidence = {
        "entities": {
            "airport": {"entity_type": "route_anchor"},
            "station": {"entity_type": "route_anchor"},
            "attraction": {"entity_type": "attraction"},
            "restaurant": {"entity_type": "restaurant"},
        },
        "routes": {
            "arrival": {
                "origin_place_id": "airport",
                "destination_place_id": "attraction",
                "mode": "taxi",
            },
            "local": {
                "origin_place_id": "attraction",
                "destination_place_id": "restaurant",
                "mode": "walk",
            },
            "departure": {
                "origin_place_id": "restaurant",
                "destination_place_id": "station",
                "mode": "taxi",
            },
        },
    }

    current = evaluate_constraint(constraint, spec, plan, evidence)
    legacy = evaluate_constraint(
        constraint,
        replace(spec, spec_version="travelweaver-task-spec-v2"),
        plan,
        evidence,
    )

    assert current.status == "pass"
    assert current.evidence["actual"] == ["walk"]
    assert legacy.status == "fail"
