"""Versioned model-facing serialization for environment tool results."""

from __future__ import annotations

from copy import deepcopy
from typing import Literal, cast

from ..env.models import StepResult

ToolResponseMode = Literal["delta", "snapshot"]

MODEL_TOOL_RESPONSE_VERSION = "travelweaver-model-tool-response-v1"
DEFAULT_TOOL_RESPONSE_MODE: ToolResponseMode = "delta"
TOOL_RESPONSE_MODES: tuple[ToolResponseMode, ...] = ("delta", "snapshot")


def validate_tool_response_mode(value: str) -> ToolResponseMode:
    """Return a typed model response mode or reject an unknown value."""

    if value not in TOOL_RESPONSE_MODES:
        choices = ", ".join(TOOL_RESPONSE_MODES)
        raise ValueError(f"Unknown tool response mode {value!r}; expected one of: {choices}.")
    return cast(ToolResponseMode, value)


def serialize_model_tool_response(
    result: StepResult,
    *,
    mode: ToolResponseMode = DEFAULT_TOOL_RESPONSE_MODE,
) -> dict[str, object]:
    """Serialize one step for the model without changing the replayable full result.

    ``snapshot`` preserves the legacy behavior. ``delta`` exposes only information
    introduced by this step plus the small amount of control state required for the
    next decision. Reward and deterministic audit details never enter a delta response.
    """

    validate_tool_response_mode(mode)
    if mode == "snapshot":
        return deepcopy(result.to_dict())

    payload: dict[str, object] = {
        "response_version": MODEL_TOOL_RESPONSE_VERSION,
        "valid_action": result.info.get("valid_action") is True,
        "remaining_steps": result.observation.remaining_steps,
        "tool_result": deepcopy(result.observation.tool_result),
    }
    if result.observation.error is not None:
        payload["error"] = deepcopy(result.observation.error)
    if result.terminated:
        payload["terminated"] = True
    if result.truncated:
        payload["truncated"] = True
    termination_reason = result.info.get("termination_reason")
    if termination_reason is not None:
        payload["termination_reason"] = str(termination_reason)
    consecutive_invalid = result.info.get("consecutive_invalid_actions")
    if isinstance(consecutive_invalid, int) and not isinstance(consecutive_invalid, bool):
        payload["consecutive_invalid_actions"] = consecutive_invalid
    return payload
