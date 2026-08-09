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

FORMAT_VERSION = "travelweaver-sft-v2"
SUPPORTED_FORMAT_VERSIONS = frozenset({"travelweaver-sft-v1", FORMAT_VERSION})
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
    if not all(isinstance(message, dict) for message in messages):
        raise ValueError(f"SFT row {line_number} contains a non-object message.")
    if len(messages) < 3 or messages[0].get("role") != "system":
        raise ValueError(f"SFT row {line_number} has no leading system message.")
    if messages[1].get("role") != "user" or messages[-1].get("role") != "assistant":
        raise ValueError(f"SFT row {line_number} has invalid conversation boundaries.")
    if row.get("format_version") == FORMAT_VERSION:
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
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or "reasoning_content" in message:
            raise ValueError(f"SFT row {line_number} message {index} contains reasoning.")
        role = message.get("role")
        if role == "assistant":
            if message.get("content") != "":
                raise ValueError(f"SFT row {line_number} assistant content is not empty.")
            calls = message.get("tool_calls")
            if not isinstance(calls, list) or len(calls) != 1:
                raise ValueError(f"SFT row {line_number} assistant must call one tool.")
            function = calls[0].get("function") if isinstance(calls[0], dict) else None
            if not isinstance(function, dict) or not isinstance(function.get("arguments"), dict):
                raise ValueError(f"SFT row {line_number} has string or missing arguments.")
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
    assistant_count = sum(message.get("role") == "assistant" for message in messages)
    for _index in range(assistant_count - 1):
        expected_roles.extend(["assistant", "tool"])
    expected_roles.append("assistant")
    if [message.get("role") for message in messages] != expected_roles:
        raise ValueError(f"SFT row {line_number} does not alternate assistant/tool turns.")


def _parquet_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": row["sample_id"],
            "task_id": row["task_id"],
            "tool_response_mode": row["tool_response_mode"],
            "model_tool_response_version": row["model_tool_response_version"],
            "user_content_format": row.get("user_content_format"),
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
            "sequence_tokens": _distribution(lengths),
            "assistant_loss_tokens": _distribution(loss_lengths),
            "assistant_loss_ratio": round(sum(loss_lengths) / sum(lengths), 6),
            "thinking_scaffold_masked": True,
            "supervised_tool_call_open_tokens": supervised_tool_call_tokens,
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
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


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
