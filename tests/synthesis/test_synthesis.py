from __future__ import annotations

import json
from collections import Counter
from types import SimpleNamespace

import pytest

from travelweaver.errors import SynthesisError
from travelweaver.llm import DeepSeekConfig
from travelweaver.synthesis.catalog import build_pilot_slots
from travelweaver.synthesis.polisher import TaskPolisher, validate_surface
from travelweaver.synthesis.render import render_canonical
from travelweaver.tasks import (
    BlueprintConstraint,
    TaskBlueprint,
    TaskSurface,
    TripSpec,
    materialize_task_spec,
)


def _blueprint() -> TaskBlueprint:
    return TaskBlueprint(
        trip=TripSpec(origin="上海", destinations=("北京",), days=2, travelers=2),
        constraints=(
            BlueprintConstraint(
                id="c001",
                kind="transport_mode",
                operator="eq",
                value={"modes": ["train"], "leg": "outbound"},
                scope="intercity_transport",
            ),
            BlueprintConstraint(
                id="c002",
                kind="total_budget",
                operator="lte",
                value={"amount": 3000},
                scope="trip",
            ),
        ),
        world_snapshot_version="snapshot-v1",
        generator_version="generator-v1",
        generation_seed=7,
    )


def _payload(blueprint: TaskBlueprint) -> dict:
    canonical = render_canonical(blueprint)
    return {
        "query": canonical.query,
        "mentions": [
            {"constraint_id": constraint_id, "text": text}
            for constraint_id, text in canonical.clauses.items()
        ],
    }


def test_pilot_catalog_has_the_frozen_50_task_distribution() -> None:
    slots = build_pilot_slots(50, 20260807)

    assert Counter(slot.destination for slot in slots) == {
        "上海": 5,
        "北京": 5,
        "南京": 5,
        "广州": 5,
        "成都": 5,
        "杭州": 5,
        "武汉": 5,
        "深圳": 5,
        "苏州": 5,
        "重庆": 5,
    }
    assert Counter(slot.days for slot in slots) == {1: 20, 2: 20, 3: 10}
    assert Counter(slot.constraint_count for slot in slots) == {1: 15, 2: 20, 3: 10, 4: 5}
    assert Counter((slot.outbound_mode, slot.return_mode) for slot in slots) == {
        ("train", "train"): 20,
        ("airplane", "airplane"): 10,
        ("train", "airplane"): 10,
        ("airplane", "train"): 10,
    }
    assert all(
        (slot.outbound_mode, slot.return_mode) == ("train", "train")
        for slot in slots
        if slot.destination == "苏州"
    )
    assert sum(
        {"outbound_mode", "return_mode"}.issubset(slot.recipe) for slot in slots
    ) >= 10
    recipe_counts = Counter(key for slot in slots for key in slot.recipe)
    assert min(
        count
        for key, count in recipe_counts.items()
        if key not in {"all_intercity_mode", "outbound_mode", "return_mode"}
    ) >= 3


def test_blueprint_surface_round_trip_materializes_exact_source_spans() -> None:
    blueprint = _blueprint()
    canonical = render_canonical(blueprint)
    surface = validate_surface(
        blueprint,
        canonical,
        _payload(blueprint),
        model="canonical-test",
    )

    assert TaskBlueprint.from_dict(blueprint.to_dict()) == blueprint
    assert TaskSurface.from_dict(surface.to_dict()) == surface
    spec = materialize_task_spec(blueprint, surface, task_id="synthetic-1")
    assert spec.task_id == "synthetic-1"
    assert spec.constraints[0].value["leg"] == "outbound"
    assert all(
        spec.public_query[item.source_start : item.source_end] == item.source_text
        for item in spec.constraints
    )


def test_surface_validator_rejects_changed_numeric_semantics() -> None:
    blueprint = _blueprint()
    canonical = render_canonical(blueprint)
    payload = _payload(blueprint)
    payload["query"] = payload["query"].replace("3000元", "5000元")

    with pytest.raises(SynthesisError, match="Protected literal|Numeric literals"):
        validate_surface(blueprint, canonical, payload, model="bad-test")


def test_polisher_uses_one_required_function_call_and_disables_thinking() -> None:
    blueprint = _blueprint()
    payload = _payload(blueprint)
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(
                                name="polish_travel_query",
                                arguments=json.dumps(payload, ensure_ascii=False),
                            )
                        )
                    ]
                )
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        ),
    )

    class _Client:
        def __init__(self) -> None:
            self.requests = []

        def complete(self, messages, tools):
            self.requests.append((messages, tools))
            return response

    client = _Client()
    polisher = TaskPolisher(
        DeepSeekConfig(api_key="not-a-secret", thinking="enabled"),
        client=client,
    )
    surface = polisher.polish(blueprint, render_canonical(blueprint))

    assert surface.usage["total_tokens"] == 150
    assert polisher.api_calls == 1
    assert polisher.config.thinking == "disabled"
    assert polisher.config.tool_choice == "required"
    assert client.requests[0][1][0]["function"]["name"] == "polish_travel_query"
