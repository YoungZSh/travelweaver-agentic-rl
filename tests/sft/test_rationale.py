from __future__ import annotations

import json

from travelweaver.llm import DeepSeekConfig
from travelweaver.sft.rationale import TrajectoryRationalePolisher
from travelweaver.sft.rationale_contract import has_visible_price_comparison


class _FakeClient:
    def __init__(self, rationales: list[str]) -> None:
        self.rationales = rationales
        self.requests: list[tuple[list[dict[str, object]], list[dict[str, object]]]] = []

    def complete(
        self, messages: list[dict[str, object]], tools: list[dict[str, object]]
    ) -> dict[str, object]:
        self.requests.append((messages, tools))
        arguments = json.dumps(
            {
                "turns": [
                    {"step_index": index, "rationale": rationale}
                    for index, rationale in enumerate(self.rationales)
                ]
            },
            ensure_ascii=False,
        )
        return {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "polish_trajectory_rationales",
                                    "arguments": arguments,
                                }
                            }
                        ]
                    }
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        }


def _trajectory() -> dict[str, object]:
    return {
        "task_id": "task-a",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "请安排杭州一日游，想去西湖。"},
            {
                "role": "assistant",
                "content": "先查询杭州的西湖，取得景点候选。",
                "tool_calls": [{"function": {"name": "search_attractions"}}],
            },
            {"role": "tool", "content": "result"},
            {
                "role": "assistant",
                "content": "已经找到西湖，将它保存为景点证据。",
                "tool_calls": [{"function": {"name": "save_candidate"}}],
            },
        ],
        "steps": [
            {
                "action": {
                    "tool": "search_attractions",
                    "arguments": {"city": "杭州", "query": "西湖"},
                }
            },
            {
                "action": {
                    "tool": "save_candidate",
                    "arguments": {"entity_id": "place:west-lake", "purpose": "attraction"},
                }
            },
        ],
        "assistant_loss_mask": [True, True],
        "batch_metadata": {
            "source": "travelweaver-programmatic-policy-v3",
            "thinking": "disabled",
        },
    }


def _audit() -> dict[str, object]:
    return {
        "task_id": "task-a",
        "turns": [
            {
                "position": 0,
                "rationale_kind": "search_evidence",
                "protected_literals": ["杭州", "西湖"],
                "template_rationale": "先查询杭州的西湖，取得景点候选。",
            },
            {
                "position": 1,
                "rationale_kind": "save_evidence",
                "protected_literals": ["西湖"],
                "template_rationale": "已经找到西湖，将它保存为景点证据。",
            },
        ],
    }


def test_rationale_polisher_keeps_actions_and_masks_while_rewriting_content() -> None:
    client = _FakeClient(
        [
            "为了核实杭州的西湖是否可用于行程，先查询对应景点候选。",
            "查询结果中已出现西湖，后续计划会将它作为景点，因此保存这项候选证据。",
        ]
    )
    polisher = TrajectoryRationalePolisher(
        DeepSeekConfig(api_key="not-secret", thinking="enabled"), client=client
    )

    trajectory, audit = polisher.polish(_trajectory(), _audit())

    assistant = [m for m in trajectory["messages"] if m["role"] == "assistant"]
    assert assistant[0]["content"].startswith("为了核实杭州")
    assert assistant[1]["content"].startswith("查询结果中已出现西湖")
    assert trajectory["assistant_loss_mask"] == [True, True]
    assert trajectory["steps"] == _trajectory()["steps"]
    assert audit["rationale_polish"]["outcome"] == "accepted"
    assert polisher.config.thinking == "disabled"
    assert polisher.config.tool_choice == "required"
    assert polisher.api_calls == 1
    assert len(client.requests) == 1


def test_rationale_polisher_falls_back_only_invalid_turn() -> None:
    client = _FakeClient(
        [
            "为了核实杭州的西湖是否可用于行程，先查询对应景点候选。",
            "这个地点不错，继续处理下一步。",
        ]
    )
    polisher = TrajectoryRationalePolisher(
        DeepSeekConfig(api_key="not-secret"), client=client
    )

    trajectory, audit = polisher.polish(_trajectory(), _audit())

    assistant = [m for m in trajectory["messages"] if m["role"] == "assistant"]
    assert assistant[0]["content"].startswith("为了核实杭州")
    assert assistant[1]["content"] == "已经找到西湖，将它保存为景点证据。"
    assert audit["rationale_polish"]["outcome"] == "partial_template_fallback"
    assert audit["turns"][0]["rationale_validation_errors"] == []
    assert audit["turns"][1]["rationale_validation_errors"]


def test_remove_rationale_requires_a_visible_price_comparison() -> None:
    assert has_visible_price_comparison("甲标价199元，高于乙的0元，所以将甲移除。")
    assert not has_visible_price_comparison("甲没有优势，所以将甲移除。")
