"""Schema-directed ordering for model-visible tool definitions and arguments."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def order_tool_schemas(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return tool schemas whose object properties put required fields first."""

    return [_order_schema(deepcopy(tool)) for tool in tools]


def order_tool_arguments(
    tool_name: str,
    arguments: dict[str, Any],
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    """Recursively order one argument object according to its function schema."""

    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(function, dict) and function.get("name") == tool_name:
            parameters = function.get("parameters")
            if not isinstance(parameters, dict):
                raise ValueError(f"Tool {tool_name!r} has no parameter schema.")
            ordered = _order_value(arguments, parameters)
            if not isinstance(ordered, dict):
                raise ValueError(f"Tool {tool_name!r} arguments did not remain an object.")
            return ordered
    raise ValueError(f"Unknown tool schema: {tool_name}")


def _order_schema(value: Any) -> Any:
    if isinstance(value, list):
        return [_order_schema(item) for item in value]
    if not isinstance(value, dict):
        return value

    result: dict[str, Any] = {}
    for key, item in value.items():
        if key != "properties" or not isinstance(item, dict):
            result[key] = _order_schema(item)
            continue
        required = value.get("required")
        required_keys = (
            [key for key in required if isinstance(key, str) and key in item]
            if isinstance(required, list)
            else []
        )
        property_order = required_keys + [key for key in item if key not in required_keys]
        result[key] = {key: _order_schema(item[key]) for key in property_order}
    return result


def _order_value(value: Any, schema: dict[str, Any]) -> Any:
    if isinstance(value, list):
        item_schema = schema.get("items")
        if not isinstance(item_schema, dict):
            return deepcopy(value)
        return [_order_value(item, item_schema) for item in value]
    if not isinstance(value, dict):
        return deepcopy(value)

    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return deepcopy(value)
    required = schema.get("required")
    required_keys = (
        [key for key in required if isinstance(key, str) and key in properties]
        if isinstance(required, list)
        else []
    )
    known_order = required_keys + [key for key in properties if key not in required_keys]
    output: dict[str, Any] = {}
    for key in known_order:
        if key in value:
            property_schema = properties[key]
            output[key] = (
                _order_value(value[key], property_schema)
                if isinstance(property_schema, dict)
                else deepcopy(value[key])
            )
    for key in sorted(set(value) - set(properties)):
        output[key] = deepcopy(value[key])
    return output
