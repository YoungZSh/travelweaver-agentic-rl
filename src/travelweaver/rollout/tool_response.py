"""Versioned model-facing serialization for environment tool results."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any, Literal, cast

from ..env.models import StepResult

ToolResponseMode = Literal["delta", "snapshot"]

MODEL_TOOL_RESPONSE_VERSION = "travelweaver-model-tool-response-v3"
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
    next decision. V3 additionally removes duplicated coordinates, source metadata,
    nested evidence snapshots, and full route segments from the *model-facing* view.
    Full ``StepResult`` objects remain untouched in trajectories for deterministic
    replay and audit. Reward and deterministic audit details never enter a delta
    response.
    """

    validate_tool_response_mode(mode)
    if mode == "snapshot":
        return deepcopy(result.to_dict())

    payload: dict[str, object] = {
        "response_version": MODEL_TOOL_RESPONSE_VERSION,
        "valid_action": result.info.get("valid_action") is True,
        "remaining_steps": result.observation.remaining_steps,
        "tool_result": _compact_tool_result(result.observation.tool_result),
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


def _compact_tool_result(value: Any) -> Any:
    """Keep decision-relevant evidence while bounding multi-turn context growth.

    Search pages can contain ten full database rows. Latitude/longitude, source
    metadata, and nested saved snapshots are necessary for the environment's full
    evidence bundle, but not for a model to select an already displayed ID. The
    compact response retains each entity ID, name, price, scheduling facts, search
    cursor, and all fields needed by a following specialised tool call.
    """

    if not isinstance(value, Mapping):
        return deepcopy(value)
    tool = value.get("tool")
    if tool in {
        "search_attractions",
        "search_restaurants",
        "search_restaurants_by_food",
        "search_hotels",
        "search_intercity_transport",
        "search_nearby",
    }:
        return _compact_search_result(value)
    if tool == "save_candidate":
        return {
            "tool": tool,
            "status": value.get("status"),
            "candidate": _compact_candidate(value.get("candidate")),
        }
    if tool == "list_candidates":
        items = value.get("items")
        return {
            "tool": tool,
            "items": [
                _compact_candidate(item)
                for item in items
                if isinstance(item, Mapping)
            ]
            if isinstance(items, Sequence) and not isinstance(items, (str, bytes))
            else [],
            "count": value.get("count"),
        }
    if tool == "remove_candidate":
        return {
            "tool": tool,
            "status": value.get("status"),
            "candidate": _compact_candidate(value.get("candidate")),
        }
    if tool == "inspect_place":
        return {"tool": tool, "item": _compact_place(value.get("item"), detailed=True)}
    if tool == "get_route":
        route = value.get("route")
        if not isinstance(route, Mapping):
            return {"tool": tool, "route": None}
        segments = route.get("segments")
        first = segments[0] if isinstance(segments, list) and segments else {}
        last = segments[-1] if isinstance(segments, list) and segments else {}
        return {
            "tool": tool,
            "route": {
                "route_id": route.get("route_id"),
                "origin_place_id": route.get("origin_place_id"),
                "destination_place_id": route.get("destination_place_id"),
                "mode": route.get("mode"),
                "start_time": first.get("start_time") if isinstance(first, Mapping) else None,
                "end_time": last.get("end_time") if isinstance(last, Mapping) else None,
            },
        }
    if tool == "submit_plan":
        # This response is terminal and is never followed by another model turn.
        # Keeping only its status still prevents accidental reward/evaluator leakage.
        return {
            "tool": tool,
            "status": value.get("status"),
            "candidate_count": value.get("candidate_count"),
        }
    return deepcopy(value)


def _compact_search_result(value: Mapping[str, Any]) -> dict[str, Any]:
    items = value.get("items")
    page = value.get("page")
    return {
        "tool": value.get("tool"),
        "items": [
            _compact_transport(item)
            if isinstance(item, Mapping) and isinstance(item.get("transport_id"), str)
            else _compact_place(item)
            for item in items
            if isinstance(item, Mapping)
        ]
        if isinstance(items, Sequence) and not isinstance(items, (str, bytes))
        else [],
        "page": {
            key: page.get(key)
            for key in ("returned", "total", "next_cursor")
            if isinstance(page, Mapping) and key in page
        },
    }


def _compact_place(value: Any, *, detailed: bool = False) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    entity_type = value.get("entity_type")
    keys = ["place_id", "entity_type", "city", "name", "price"]
    if entity_type == "attraction":
        keys.extend(("category", "open_time", "close_time"))
        if detailed:
            keys.extend(("recommended_min_hours", "recommended_max_hours"))
    elif entity_type == "restaurant":
        keys.extend(("cuisine", "open_time", "close_time"))
    elif entity_type == "hotel":
        keys.extend(("hotel_type", "room_type"))
    compact = {key: value[key] for key in keys if key in value}
    if entity_type == "restaurant" and (hint := _food_hint(value.get("recommended_food"))):
        compact["recommended_food_hint"] = hint
    if "distance_km" in value:
        compact["distance_km"] = value["distance_km"]
    return compact


def _food_hint(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return next(
        (part.strip() for part in re.split(r"[,，、|/]", value) if part.strip()),
        None,
    )


def _compact_transport(value: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "transport_id",
        "mode",
        "source_id",
        "origin_city",
        "destination_city",
        "origin",
        "destination",
        "departure_time",
        "arrival_time",
        "cost",
        "train_type",
        "origin_anchor_id",
        "destination_anchor_id",
    )
    return {key: value[key] for key in keys if key in value}


def _compact_candidate(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    keys = ("candidate_id", "entity_id", "entity_type", "name", "purpose", "note")
    compact = {key: value[key] for key in keys if key in value}
    evidence = value.get("evidence")
    if isinstance(evidence, Mapping) and "price" in evidence:
        compact["price"] = evidence["price"]
    return compact
