"""Convert trainer-neutral TravelWeaver SFT JSONL to audited veRL Parquet."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import OmegaConf
from transformers import AutoProcessor

FORMAT_VERSION = "travelweaver-sft-v4"
SUPPORTED_FORMAT_VERSIONS = frozenset(
    {
        "travelweaver-sft-v1",
        "travelweaver-sft-v2",
        "travelweaver-sft-v3",
        FORMAT_VERSION,
    }
)
SUPERVISION_MODES = frozenset({"action_only", "react", "react_recovery"})
MODEL_TOOL_RESPONSE_VERSION = "travelweaver-model-tool-response-v1"
USER_CONTENT_FORMAT = "travelweaver-natural-query-v1"
MODEL_CONTEXT_LIMIT = 262_144


def _load_dataset_class() -> type:
    source = Path(__file__).resolve().parents[1] / "src" / "travelweaver_sft_dataset.py"
    spec = importlib.util.spec_from_file_location("travelweaver_sft_dataset", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load the TravelWeaver veRL adapter from {source}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TravelWeaverMultiTurnSFTDataset


def _read_neutral(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid neutral JSONL at {path}:{line_number}.") from error
            _validate_row(row, line_number)
            rows.append(row)
    if not rows:
        raise ValueError("Neutral SFT input is empty.")
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Neutral SFT input contains duplicate sample ids.")
    return rows


def _validate_row(row: Any, line_number: int) -> None:
    if not isinstance(row, dict) or row.get("format_version") not in SUPPORTED_FORMAT_VERSIONS:
        raise ValueError(f"Unsupported SFT row at line {line_number}.")
    messages = row.get("messages")
    tools = row.get("tools")
    if not isinstance(messages, list) or not isinstance(tools, list):
        raise ValueError(f"SFT row {line_number} has no messages/tools arrays.")
    if row.get("enable_thinking") is not False:
        raise ValueError(f"SFT row {line_number} must disable thinking.")
    response_mode = row.get("tool_response_mode")
    if response_mode not in {"delta", "snapshot"}:
        raise ValueError(f"SFT row {line_number} has no supported tool response mode.")
    if row.get("model_tool_response_version") != MODEL_TOOL_RESPONSE_VERSION:
        raise ValueError(f"SFT row {line_number} has an unsupported tool response version.")
    format_version = row.get("format_version")
    supervision_mode = row.get("supervision_mode")
    if format_version in {"travelweaver-sft-v1", "travelweaver-sft-v2"}:
        supervision_mode = "action_only"
    if supervision_mode not in SUPERVISION_MODES:
        raise ValueError(f"SFT row {line_number} has no supported supervision mode.")
    if not all(isinstance(message, dict) for message in messages):
        raise ValueError(f"SFT row {line_number} contains a non-object message.")
    if len(messages) < 3 or messages[0].get("role") != "system":
        raise ValueError(f"SFT row {line_number} has no leading system message.")
    if messages[1].get("role") != "user" or messages[-1].get("role") != "assistant":
        raise ValueError(f"SFT row {line_number} has invalid conversation boundaries.")
    if format_version in {"travelweaver-sft-v2", FORMAT_VERSION}:
        if row.get("user_content_format") != USER_CONTENT_FORMAT:
            raise ValueError(f"SFT row {line_number} has an unsupported user content format.")
        user_content = messages[1].get("content")
        if not isinstance(user_content, str) or not user_content.strip():
            raise ValueError(f"SFT row {line_number} has no natural-language user query.")
        try:
            parsed_user_content = json.loads(user_content)
        except json.JSONDecodeError:
            parsed_user_content = None
        if isinstance(parsed_user_content, (dict, list)):
            raise ValueError(f"SFT row {line_number} wraps the human user query in JSON.")
    assistant_count = sum(message.get("role") == "assistant" for message in messages)
    assistant_loss_mask = row.get("assistant_loss_mask")
    if format_version == FORMAT_VERSION:
        if (
            not isinstance(assistant_loss_mask, list)
            or len(assistant_loss_mask) != assistant_count
            or any(not isinstance(value, bool) for value in assistant_loss_mask)
        ):
            raise ValueError(f"SFT row {line_number} has an invalid assistant loss mask.")
        if not assistant_loss_mask or assistant_loss_mask[-1] is not True:
            raise ValueError(f"SFT row {line_number} masks its terminal assistant turn.")
        if supervision_mode in {"action_only", "react"} and not all(assistant_loss_mask):
            raise ValueError(f"SFT row {line_number} masks a clean assistant turn.")
        _validate_required_first_tools(tools, line_number)
    visible_assistant_turns = 0
    assistant_turn_index = 0
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or "reasoning_content" in message:
            raise ValueError(f"SFT row {line_number} message {index} contains reasoning.")
        role = message.get("role")
        if role == "assistant":
            content = message.get("content")
            if not isinstance(content, str):
                raise ValueError(f"SFT row {line_number} assistant content is not text.")
            if supervision_mode == "action_only" and content != "":
                raise ValueError(f"SFT row {line_number} action-only content is not empty.")
            visible_assistant_turns += int(bool(content.strip()))
            calls = message.get("tool_calls")
            if not isinstance(calls, list) or len(calls) != 1:
                raise ValueError(f"SFT row {line_number} assistant must call one tool.")
            function = calls[0].get("function") if isinstance(calls[0], dict) else None
            if not isinstance(function, dict) or not isinstance(function.get("arguments"), dict):
                raise ValueError(f"SFT row {line_number} has string or missing arguments.")
            if format_version == FORMAT_VERSION:
                assert isinstance(assistant_loss_mask, list)
                _validate_required_first_arguments(
                    function,
                    tools,
                    line_number,
                    allow_unknown=(
                        supervision_mode == "react_recovery"
                        and assistant_loss_mask[assistant_turn_index] is False
                    ),
                )
            assistant_turn_index += 1
        elif role == "tool":
            try:
                payload = json.loads(message["content"])
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"SFT row {line_number} has invalid tool content.") from error
            if response_mode == "delta":
                required = {
                    "response_version",
                    "valid_action",
                    "remaining_steps",
                    "tool_result",
                }
                if not required <= payload.keys():
                    raise ValueError(f"SFT row {line_number} has an incomplete delta response.")
                if payload["response_version"] != MODEL_TOOL_RESPONSE_VERSION:
                    raise ValueError(f"SFT row {line_number} mixes tool response versions.")
                if {"reward", "info", "observation"} & payload.keys():
                    raise ValueError(f"SFT row {line_number} leaks snapshot state into delta data.")
            else:
                info = payload.get("info")
                detail = info.get("reward_detail") if isinstance(info, dict) else None
                if detail is not None or payload.get("reward") not in {0}:
                    raise ValueError(f"SFT row {line_number} leaks terminal Reward data.")
    expected_roles = ["system", "user"]
    for _index in range(assistant_count - 1):
        expected_roles.extend(["assistant", "tool"])
    expected_roles.append("assistant")
    if [message.get("role") for message in messages] != expected_roles:
        raise ValueError(f"SFT row {line_number} does not alternate assistant/tool turns.")
    if supervision_mode in {"react", "react_recovery"} and visible_assistant_turns == 0:
        raise ValueError(f"SFT row {line_number} ReAct sample has no visible assistant text.")
    if format_version == FORMAT_VERSION:
        assert isinstance(assistant_loss_mask, list)
        for turn_index, message_index in enumerate(range(2, len(messages), 2)):
            has_tool_response = message_index + 1 < len(messages)
            if not has_tool_response:
                continue
            payload = json.loads(messages[message_index + 1]["content"])
            if bool(payload.get("valid_action")) != bool(assistant_loss_mask[turn_index]):
                raise ValueError(
                    f"SFT row {line_number} loss mask disagrees with tool validity."
                )


def _parquet_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": row["sample_id"],
            "task_id": row["task_id"],
            "tool_response_mode": row["tool_response_mode"],
            "model_tool_response_version": row["model_tool_response_version"],
            "user_content_format": row.get("user_content_format"),
            "supervision_mode": row.get("supervision_mode", "action_only"),
            "assistant_loss_mask_json": _compact_json(
                row.get(
                    "assistant_loss_mask",
                    [
                        True
                        for message in row["messages"]
                        if message.get("role") == "assistant"
                    ],
                )
            ),
            "messages_json": _compact_json(row["messages"]),
            "tools_json": _compact_json(row["tools"]),
            "enable_thinking": False,
        }
        for row in rows
    ]


def _audit_parquet(
    parquet_path: Path,
    model_path: Path,
) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    processor = AutoProcessor.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False
    )
    dataset_class = _load_dataset_class()
    config = OmegaConf.create(
        {
            "pad_mode": "no_padding",
            "max_length": MODEL_CONTEXT_LIMIT,
            "truncation": "error",
            "messages_key": "messages_json",
            "tools_key": "tools_json",
            "enable_thinking_key": "enable_thinking",
            "enable_thinking_default": False,
            "apply_chat_template_kwargs": {"enable_thinking": False},
            "ignore_input_ids_mismatch": False,
            "shuffle": False,
        }
    )
    dataset = dataset_class(
        parquet_files=str(parquet_path),
        tokenizer=processor.tokenizer,
        processor=processor,
        config=config,
    )
    lengths: list[int] = []
    loss_lengths: list[int] = []
    previews: list[tuple[str, str]] = []
    think_token_ids = {
        int(processor.tokenizer.convert_tokens_to_ids(token))
        for token in ("<think>", "</think>")
    }
    tool_call_token_id = int(processor.tokenizer.convert_tokens_to_ids("<tool_call>"))
    supervised_tool_call_tokens = 0
    masked_assistant_turns = 0
    supervision_counts = {
        str(key): int(value)
        for key, value in dataset.dataframe["supervision_mode"].value_counts().items()
    }
    for index in range(len(dataset)):
        item = dataset[index]
        lengths.append(int(item["input_ids"].numel()))
        loss_lengths.append(int(item["loss_mask"].sum().item()))
        input_ids = item["input_ids"]
        loss_mask = item["loss_mask"]
        for token_id in think_token_ids:
            if bool(((input_ids == token_id) & (loss_mask != 0)).any()):
                raise ValueError("Qwen thinking scaffold unexpectedly contributes to SFT loss.")
        supervised_tool_call_tokens += int(
            ((input_ids == tool_call_token_id) & (loss_mask != 0)).sum().item()
        )
        row_mask = json.loads(dataset.dataframe.iloc[index]["assistant_loss_mask_json"])
        masked_assistant_turns += sum(not value for value in row_mask)
        if len(previews) < 10:
            row = dataset.dataframe.iloc[index].to_dict()
            messages = json.loads(row["messages_json"])
            tools = json.loads(row["tools_json"])
            rendered = processor.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            previews.append((str(row["task_id"]), rendered))
    return (
        {
            "samples": len(dataset),
            "supervision_modes": dict(sorted(supervision_counts.items())),
            "sequence_tokens": _distribution(lengths),
            "assistant_loss_tokens": _distribution(loss_lengths),
            "assistant_loss_ratio": round(sum(loss_lengths) / sum(lengths), 6),
            "thinking_scaffold_masked": True,
            "supervised_tool_call_open_tokens": supervised_tool_call_tokens,
            "masked_assistant_turns": masked_assistant_turns,
            "input_ids_mismatch_ignored": False,
            "model_context_limit": MODEL_CONTEXT_LIMIT,
        },
        previews,
    )


def _distribution(values: list[int]) -> dict[str, int]:
    ordered = sorted(values)

    def percentile(ratio: float) -> int:
        return ordered[round((len(ordered) - 1) * ratio)]

    return {
        "min": ordered[0],
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
        "total": sum(ordered),
    }


def _write_preview(path: Path, previews: list[tuple[str, str]]) -> None:
    lines = ["# Qwen3.5 rendered SFT preview", ""]
    for task_id, rendered in previews:
        lines.extend([f"## {task_id}", "", "```text", rendered, "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _update_manifest(path: Path, report: dict[str, Any], parquet_path: Path) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    manifest["status"] = "qwen_parquet_complete"
    manifest["qwen_adapter"] = {
        "model": "Qwen3.5-4B",
        "parquet_path": str(parquet_path.resolve()),
        "parquet_sha256": _sha256(parquet_path),
        "messages_column": "messages_json",
            "tools_column": "tools_json",
            "assistant_loss_mask_column": "assistant_loss_mask_json",
        "enable_thinking": False,
        "apply_chat_template_enable_thinking": False,
        **report,
    }
    _atomic_text(path, json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def prepare(input_path: Path, output_path: Path, model_path: Path) -> dict[str, Any]:
    """Write and fully iterate an audited Qwen-compatible veRL Parquet dataset."""

    rows = _read_neutral(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing Parquet: {output_path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".parquet", dir=output_path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        pd.DataFrame(_parquet_rows(rows)).to_parquet(temporary, index=False)
        report, previews = _audit_parquet(temporary, model_path)
        os.replace(temporary, output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    _write_preview(output_path.parent / "qwen-preview.md", previews)
    _update_manifest(output_path.parent / "manifest.json", report, output_path)
    return report


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=False)


def _validate_required_first_tools(tools: list[dict[str, Any]], line_number: int) -> None:
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        parameters = function.get("parameters") if isinstance(function, dict) else None
        if not isinstance(parameters, dict):
            raise ValueError(f"SFT row {line_number} contains an invalid tool schema.")
        _validate_schema_property_order(parameters, line_number)


def _validate_schema_property_order(schema: dict[str, Any], line_number: int) -> None:
    properties = schema.get("properties")
    required = schema.get("required")
    if isinstance(properties, dict):
        required_keys = (
            [key for key in required if isinstance(key, str) and key in properties]
            if isinstance(required, list)
            else []
        )
        actual = list(properties)
        if actual[: len(required_keys)] != required_keys:
            raise ValueError(f"SFT row {line_number} tool schema is not required-first.")
        for property_schema in properties.values():
            if isinstance(property_schema, dict):
                _validate_schema_property_order(property_schema, line_number)
    items = schema.get("items")
    if isinstance(items, dict):
        _validate_schema_property_order(items, line_number)


def _validate_required_first_arguments(
    function: dict[str, Any],
    tools: list[dict[str, Any]],
    line_number: int,
    *,
    allow_unknown: bool = False,
) -> None:
    name = function.get("name")
    arguments = function.get("arguments")
    for tool in tools:
        tool_function = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(tool_function, dict) and tool_function.get("name") == name:
            schema = tool_function.get("parameters")
            if not isinstance(schema, dict) or not isinstance(arguments, dict):
                break
            _validate_value_order(arguments, schema, line_number)
            return
    if allow_unknown:
        return
    raise ValueError(f"SFT row {line_number} calls an unknown tool {name!r}.")


def _validate_value_order(value: Any, schema: dict[str, Any], line_number: int) -> None:
    if isinstance(value, list):
        items = schema.get("items")
        if isinstance(items, dict):
            for item in value:
                _validate_value_order(item, items, line_number)
        return
    if not isinstance(value, dict):
        return
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return
    expected = [key for key in properties if key in value]
    known_actual = [key for key in value if key in properties]
    if known_actual != expected:
        raise ValueError(f"SFT row {line_number} tool arguments are not schema-ordered.")
    for key in expected:
        property_schema = properties[key]
        if isinstance(property_schema, dict):
            _validate_value_order(value[key], property_schema, line_number)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, content: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.input, args.output, args.model), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
