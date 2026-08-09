"""veRL dataset adapter for JSON-preserving TravelWeaver tool trajectories."""

from __future__ import annotations

import json
from typing import Any

import torch
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.tokenizer.chat_template import apply_chat_template


def _decode(value: Any, *, field: str) -> Any:
    if not isinstance(value, str):
        raise TypeError(f"TravelWeaver Parquet column {field!r} must contain JSON strings.")
    try:
        return json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"TravelWeaver Parquet column {field!r} contains invalid JSON.") from error


class TravelWeaverMultiTurnSFTDataset(MultiTurnSFTDataset):
    """Decode JSON columns before using veRL's standard multi-turn masking logic."""

    def _read_files_and_process(self) -> None:
        super()._read_files_and_process()
        self.messages = [
            _decode(value, field=self.messages_key) for value in self.messages
        ]
        if self.tools is not None:
            self.tools = [_decode(value, field=self.tools_key) for value in self.tools]

    def _build_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        parsed = dict(example)
        parsed[self.messages_key] = _decode(
            parsed[self.messages_key], field=self.messages_key
        )
        return super()._build_messages(parsed)

    def _process_single_message(
        self,
        index: int,
        message: dict[str, Any],
        full_message: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        enable_thinking: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if index == 0 and message["role"] == "system":
            if len(full_message) < 2 or full_message[1]["role"] != "user":
                raise ValueError("TravelWeaver conversations require system followed by user.")
            processor = self.processor if self.processor is not None else self.tokenizer
            kwargs = {**self.apply_chat_template_kwargs}
            if enable_thinking is not None:
                kwargs["enable_thinking"] = enable_thinking
            inputs = apply_chat_template(
                processor,
                messages=full_message[:2],
                tools=tools,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                **kwargs,
            )
            values = dict(inputs)
            input_ids = values.pop("input_ids")[0]
            attention_mask = values.pop("attention_mask")[0]
            loss_mask = torch.zeros_like(attention_mask)
            return input_ids, loss_mask, attention_mask, values
        if index == 1 and full_message[0]["role"] == "system":
            empty = torch.empty(0, dtype=torch.long)
            return empty, empty.clone(), empty.clone(), {}
        return super()._process_single_message(
            index=index,
            message=message,
            full_message=full_message,
            tools=tools,
            enable_thinking=enable_thinking,
        )
