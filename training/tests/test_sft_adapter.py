from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


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
        "enable_thinking": False,
        "tools": [],
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
