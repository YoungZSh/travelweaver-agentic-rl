from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch


def _tools() -> list[dict[str, object]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "search_attractions",
                "description": "搜索景点",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "query": {"type": "string"},
                    },
                    "required": ["city"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "submit_plan",
                "description": "提交计划",
                "parameters": {
                    "type": "object",
                    "properties": {"plan": {"type": "object"}},
                    "required": ["plan"],
                    "additionalProperties": False,
                },
            },
        },
    ]


def _load(relative_path: str, name: str):
    path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_json_adapter_preserves_sparse_tool_arguments() -> None:
    module = _load("src/travelweaver_sft_dataset.py", "travelweaver_sft_dataset_test")

    value = module._decode(
        '{"function":{"name":"next_page","arguments":{"cursor":"twc_1"}}}',
        field="messages_json",
    )

    assert value["function"]["arguments"] == {"cursor": "twc_1"}
    with pytest.raises(ValueError, match="invalid JSON"):
        module._decode("{", field="messages_json")


def test_prepare_validator_rejects_reasoning_and_string_arguments() -> None:
    module = _load("scripts/prepare_qwen_sft.py", "prepare_qwen_sft_test")
    row = {
        "format_version": module.FORMAT_VERSION,
        "sample_id": "sample",
        "task_id": "task",
        "tool_response_mode": "delta",
        "model_tool_response_version": module.MODEL_TOOL_RESPONSE_VERSION,
        "user_content_format": module.USER_CONTENT_FORMAT,
        "supervision_mode": "action_only",
        "assistant_loss_mask": [True],
        "enable_thinking": False,
        "tools": [],
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "hidden",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "finish_without_plan", "arguments": "{}"},
                    }
                ],
            },
        ],
    }

    with pytest.raises(ValueError, match="contains reasoning"):
        module._validate_row(row, 1)


def test_prepare_validator_rejects_json_wrapped_human_query() -> None:
    module = _load("scripts/prepare_qwen_sft.py", "prepare_qwen_sft_user_test")
    row = {
        "format_version": module.FORMAT_VERSION,
        "sample_id": "sample",
        "task_id": "task",
        "tool_response_mode": "delta",
        "model_tool_response_version": module.MODEL_TOOL_RESPONSE_VERSION,
        "user_content_format": module.USER_CONTENT_FORMAT,
        "supervision_mode": "action_only",
        "assistant_loss_mask": [True],
        "enable_thinking": False,
        "tools": _tools(),
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": '{"query":"请规划行程"}'},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "finish_without_plan",
                            "arguments": {"reason": "无解"},
                        },
                    }
                ],
            },
        ],
    }

    with pytest.raises(ValueError, match="wraps the human user query in JSON"):
        module._validate_row(row, 1)


def test_prepare_validator_accepts_visible_react_content() -> None:
    module = _load("scripts/prepare_qwen_sft.py", "prepare_qwen_sft_react_test")
    row = {
        "format_version": module.FORMAT_VERSION,
        "sample_id": "sample",
        "task_id": "task",
        "tool_response_mode": "delta",
        "model_tool_response_version": module.MODEL_TOOL_RESPONSE_VERSION,
        "user_content_format": module.USER_CONTENT_FORMAT,
        "supervision_mode": "react",
        "assistant_loss_mask": [True, True],
        "enable_thinking": False,
        "tools": _tools(),
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "请规划行程"},
            {
                "role": "assistant",
                "content": "我先查询景点。",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "search_attractions",
                            "arguments": {"city": "杭州"},
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "content": (
                    '{"response_version":"travelweaver-model-tool-response-v1",'
                    '"valid_action":true,"remaining_steps":34,"tool_result":{}}'
                ),
            },
            {
                "role": "assistant",
                "content": "信息足够，现在提交。",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "submit_plan",
                            "arguments": {"plan": {}},
                        },
                    }
                ],
            },
        ],
    }

    module._validate_row(row, 1)
    row["supervision_mode"] = "action_only"
    with pytest.raises(ValueError, match="action-only content is not empty"):
        module._validate_row(row, 1)


def test_prepare_validator_accepts_recovery_mask_and_rejects_order_drift() -> None:
    module = _load("scripts/prepare_qwen_sft.py", "prepare_qwen_sft_recovery_test")
    row = {
        "format_version": module.FORMAT_VERSION,
        "sample_id": "sample",
        "task_id": "task",
        "tool_response_mode": "delta",
        "model_tool_response_version": module.MODEL_TOOL_RESPONSE_VERSION,
        "user_content_format": module.USER_CONTENT_FORMAT,
        "supervision_mode": "react_recovery",
        "assistant_loss_mask": [False, True],
        "enable_thinking": False,
        "tools": _tools(),
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "请规划行程"},
            {
                "role": "assistant",
                "content": "我先尝试查询。",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "search_attractions",
                            "arguments": {"city": "杭州"},
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "content": (
                    '{"response_version":"travelweaver-model-tool-response-v1",'
                    '"valid_action":false,"remaining_steps":35,"tool_result":null,'
                    '"error":{"code":"invalid_action","message":"bad"}}'
                ),
            },
            {
                "role": "assistant",
                "content": "根据错误修正后提交。",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "submit_plan", "arguments": {"plan": {}}},
                    }
                ],
            },
        ],
    }

    module._validate_row(row, 1)

    unknown_tool_row = deepcopy(row)
    unknown_tool_row["messages"][2]["tool_calls"][0]["function"] = {
        "name": "config",
        "arguments": {"entity_id": "place:unknown"},
    }
    module._validate_row(unknown_tool_row, 1)
    unknown_tool_row["assistant_loss_mask"] = [True, True]
    with pytest.raises(ValueError, match="calls an unknown tool"):
        module._validate_row(unknown_tool_row, 1)

    row["assistant_loss_mask"] = [True, True]
    with pytest.raises(ValueError, match="disagrees with tool validity"):
        module._validate_row(row, 1)

    row["assistant_loss_mask"] = [False, True]
    row["tools"][0]["function"]["parameters"]["properties"] = {
        "query": {"type": "string"},
        "city": {"type": "string"},
    }
    with pytest.raises(ValueError, match="schema is not required-first"):
        module._validate_row(row, 1)


def test_model_json_preserves_argument_order_and_mask_zeroes_only_loss() -> None:
    prepare = _load("scripts/prepare_qwen_sft.py", "prepare_qwen_sft_order_test")
    dataset = _load("src/travelweaver_sft_dataset.py", "travelweaver_sft_dataset_mask_test")
    value = {
        "arguments": {
            "entity_id": "place:1",
            "purpose": "hotel",
            "note": "住宿",
        }
    }
    decoded = json.loads(prepare._compact_json(value))
    assert list(decoded["arguments"]) == ["entity_id", "purpose", "note"]

    result = (
        torch.tensor([1, 2]),
        torch.tensor([1, 1]),
        torch.tensor([1, 1]),
        {},
    )
    masked = dataset._apply_turn_supervision(result, trainable=False)
    assert masked[0].tolist() == [1, 2]
    assert masked[1].tolist() == [0, 0]
    assert masked[2].tolist() == [1, 1]
