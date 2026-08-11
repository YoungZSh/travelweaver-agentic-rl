from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from travelweaver.data import JsonlTaskStore
from travelweaver.errors import TaskSpecError
from travelweaver.tasks import (
    ChinaTravelOracleAdapter,
    LLMTaskSpecCompiler,
    TravelTaskSpec,
    build_base_spec,
)


class _Message(dict):
    def model_dump(self, *, exclude_none=True):
        return dict(self)


class _ChatClient:
    def __init__(self, *payloads):
        self.payloads = iter(payloads)
        self.requests = []

    def complete(self, messages, tools):
        self.requests.append({"messages": messages, "tools": tools})
        payload = next(self.payloads)
        message = _Message(
            tool_calls=[
                {
                    "id": "compile-1",
                    "type": "function",
                    "function": {
                        "name": "emit_travel_task_spec",
                        "arguments": json.dumps(payload, ensure_ascii=False),
                    },
                }
            ]
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _task() -> dict:
    return {
        "uid": "free-text-1",
        "query": "两个人从上海去杭州玩两天，总预算不超过3000元，希望行程轻松。",
        "start_city": "上海",
        "target_city": "杭州",
        "days": 2,
        "people_number": 2,
    }


def test_task_spec_round_trip_and_hash_validation() -> None:
    spec = build_base_spec(_task())
    payload = spec.to_dict()
    assert TravelTaskSpec.from_dict(payload) == spec

    payload["trip"]["days"] = 3
    with pytest.raises(TaskSpecError, match="hash"):
        TravelTaskSpec.from_dict(payload)


def test_llm_compiler_uses_function_call_and_retains_source_span() -> None:
    client = _ChatClient(
        {
            "constraints": [
                {
                    "kind": "total_budget",
                    "operator": "lte",
                    "value": {"amount": 3000, "currency": "CNY"},
                    "scope": "trip",
                    "hardness": "hard",
                    "source_text": "总预算不超过3000元",
                }
            ],
            "unscored_preferences": ["希望行程轻松"],
        }
    )
    result = LLMTaskSpecCompiler(client).compile(_task())

    assert result.accepted
    assert result.spec is not None
    constraint = result.spec.constraints[0]
    assert result.spec.public_query[constraint.source_start : constraint.source_end] == (
        constraint.source_text
    )
    assert result.spec.unscored_preferences == ("希望行程轻松",)
    assert client.requests[0]["tools"][0]["function"]["name"] == "emit_travel_task_spec"


def test_llm_compiler_retries_then_quarantines_unsupported_constraints() -> None:
    invalid = {
        "constraints": [
            {
                "kind": "romantic_score",
                "operator": "gte",
                "value": 0.8,
                "scope": "trip",
                "hardness": "hard",
                "source_text": "希望行程轻松",
            }
        ],
        "unscored_preferences": [],
    }
    result = LLMTaskSpecCompiler(_ChatClient(invalid, invalid)).compile(_task())

    assert not result.accepted
    assert result.status == "quarantined"
    assert result.attempts == 2
    assert all("Unsupported constraint kind" in error for error in result.errors)


def test_llm_compiler_rejects_malformed_supported_constraint_values() -> None:
    malformed = {
        "constraints": [
            {
                "kind": "total_budget",
                "operator": "lte",
                "value": {"amount": "很多"},
                "scope": "trip",
                "hardness": "hard",
                "source_text": "总预算不超过3000元",
            }
        ],
        "unscored_preferences": [],
    }
    result = LLMTaskSpecCompiler(_ChatClient(malformed, malformed)).compile(_task())

    assert not result.accepted
    assert all("invalid value payload" in error for error in result.errors)


def test_chinatravel_adapter_maps_all_654_pinned_tasks_without_exec() -> None:
    store = JsonlTaskStore.default(split="benchmark")
    adapter = ChinaTravelOracleAdapter()

    results = [
        adapter.compile(store.get_public(task_id), store.get_oracle(task_id))
        for task_id in store.task_ids
    ]
    assert len(results) == 654
    assert all(result.accepted for result in results)

    unsafe = adapter.compile(
        _task(),
        {"hard_logic": ["import os\nresult=True"]},
    )
    assert not unsafe.accepted
    assert "unsafe AST" in unsafe.errors[0]


def test_chinatravel_adapter_distinguishes_required_and_allowed_innercity_modes() -> None:
    adapter = ChinaTravelOracleAdapter()
    required = adapter.compile(
        _task(),
        {"hard_logic": ["result = ({'metro'} <= innercity_transport_set)"]},
    )
    allowed = adapter.compile(
        _task(),
        {"hard_logic": ["result = (innercity_transport_set <= {'metro', 'walk'})"]},
    )

    assert required.accepted and required.spec is not None
    assert required.spec.constraints[0].operator == "include"
    assert required.spec.constraints[0].value == {"modes": ["metro"]}
    assert allowed.accepted and allowed.spec is not None
    assert allowed.spec.constraints[0].operator == "not_in"
    assert allowed.spec.constraints[0].value == {"modes": ["taxi"]}
