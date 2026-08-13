"""JSON schemas for the complete public agent-tool surface."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def _object(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


CITY = {"type": "string", "minLength": 1, "description": "ChinaTravel 支持的城市名称。"}
QUERY = {"type": "string", "minLength": 1, "description": "地点名称中的关键词。"}
PRICE = {"type": "number", "minimum": 0, "description": "单项价格，人民币。"}
TIME = {
    "type": "string",
    "pattern": "^(?:[01]\\d|2[0-3]):[0-5]\\d$",
    "description": "24 小时时间，格式 HH:MM。",
}
DAY_TIME = {
    "type": "string",
    "pattern": "^(?:(?:[01]\\d|2[0-3]):[0-5]\\d|24:00)$",
    "description": "行程内时间，格式 HH:MM；仅结束时间可使用 24:00。",
}
SORT = {
    "type": "string",
    "enum": ["name", "price"],
    "default": "name",
    "description": "确定性排序字段。",
}

CANDIDATE_PURPOSES = [
    "attraction",
    "meal",
    "hotel",
    "outbound_transport",
    "return_transport",
]

ACTIVITY = _object(
    {
        "candidate_id": {"type": "string", "minLength": 1},
        "type": {
            "type": "string",
            "enum": [
                "attraction",
                "breakfast",
                "lunch",
                "dinner",
                "accommodation",
                "train",
                "airplane",
            ],
        },
        "start_time": DAY_TIME,
        "end_time": DAY_TIME,
        "route_from_previous_id": {
            "type": "string",
            "minLength": 1,
            "description": "若与前一活动位置不同，引用衔接两者的 get_route route_id。",
        },
        "rooms": {
            "type": "integer",
            "minimum": 1,
            "description": "住宿活动明确选择的房间数。",
        },
        "room_type": {
            "type": "integer",
            "minimum": 1,
            "description": "住宿活动明确选择的每间房床位数。",
        },
        "note": {"type": "string", "maxLength": 500},
    },
    ["candidate_id", "type", "start_time", "end_time"],
)

PLAN = _object(
    {
        "people_number": {"type": "integer", "minimum": 1},
        "start_city": CITY,
        "target_city": CITY,
        "itinerary": {
            "type": "array",
            "minItems": 1,
            "items": _object(
                {
                    "day": {"type": "integer", "minimum": 1},
                    "activities": {
                        "type": "array",
                        "minItems": 1,
                        "items": ACTIVITY,
                    },
                },
                ["day", "activities"],
            ),
        },
    },
    ["people_number", "start_city", "target_city", "itinerary"],
)


_TOOL_SCHEMAS: tuple[dict[str, Any], ...] = (
    {
        "type": "function",
        "function": {
            "name": "list_attraction_categories",
            "description": "列出指定城市当前可用的景点类别。",
            "parameters": _object({"city": CITY}, ["city"]),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_attractions",
            "description": "按城市、类别、名称、开放时间、票价或推荐游览时长搜索景点。",
            "parameters": _object(
                {
                    "city": CITY,
                    "query": QUERY,
                    "category": {"type": "string", "minLength": 1},
                    "min_price": PRICE,
                    "max_price": PRICE,
                    "min_recommended_hours": {"type": "number", "minimum": 0},
                    "max_recommended_hours": {"type": "number", "minimum": 0},
                    "open_at": TIME,
                    "sort_by": SORT,
                },
                ["city"],
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_restaurant_cuisines",
            "description": "列出指定城市当前可用的餐厅菜系。",
            "parameters": _object({"city": CITY}, ["city"]),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_restaurants",
            "description": "按城市、菜系、推荐菜、名称、开放时间或价格搜索餐厅。",
            "parameters": _object(
                {
                    "city": CITY,
                    "query": QUERY,
                    "cuisine": {"type": "string", "minLength": 1},
                    "recommended_food": {"type": "string", "minLength": 1},
                    "min_price": PRICE,
                    "max_price": PRICE,
                    "open_at": TIME,
                    "sort_by": SORT,
                },
                ["city"],
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_hotel_features",
            "description": "列出指定城市当前可用的酒店特色和房型床位数。",
            "parameters": _object({"city": CITY}, ["city"]),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_hotels",
            "description": "按城市、酒店特色、床位数、名称或价格搜索住宿。",
            "parameters": _object(
                {
                    "city": CITY,
                    "query": QUERY,
                    "hotel_type": {"type": "string", "minLength": 1},
                    "room_type": {"type": "integer", "minimum": 1},
                    "min_price": PRICE,
                    "max_price": PRICE,
                    "sort_by": SORT,
                },
                ["city"],
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_intercity_transport",
            "description": "查询两个城市之间的火车或航班快照。",
            "parameters": _object(
                {
                    "origin_city": CITY,
                    "destination_city": CITY,
                    "mode": {"type": "string", "enum": ["train", "airplane"]},
                    "earliest_departure": TIME,
                },
                ["origin_city", "destination_city", "mode"],
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_nearby",
            "description": "搜索当前轨迹已见地点附近的景点、餐厅或酒店。",
            "parameters": _object(
                {
                    "place_id": {"type": "string", "minLength": 1},
                    "category": {
                        "type": "string",
                        "enum": ["attraction", "restaurant", "hotel"],
                    },
                    "radius_km": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "maximum": 50,
                        "default": 2,
                    },
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 100},
                },
                ["place_id", "category"],
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_place_open",
            "description": "检查当前轨迹已见景点或餐厅在指定时间是否开放。",
            "parameters": _object(
                {
                    "place_id": {"type": "string", "minLength": 1},
                    "at_time": TIME,
                },
                ["place_id", "at_time"],
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_route",
            "description": "查询同城已见地点或车站/机场锚点间路线，返回稳定 route_id。",
            "parameters": _object(
                {
                    "origin_place_id": {"type": "string", "minLength": 1},
                    "destination_place_id": {"type": "string", "minLength": 1},
                    "mode": {"type": "string", "enum": ["walk", "taxi", "metro"]},
                    "start_time": TIME,
                },
                ["origin_place_id", "destination_place_id", "mode", "start_time"],
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "next_page",
            "description": "使用上一个搜索结果返回的 cursor 获取下一页。",
            "parameters": _object({"cursor": {"type": "string", "minLength": 1}}, ["cursor"]),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_candidate",
            "description": "把当前轨迹已经展示的地点或城际交通保存到候选集。",
            "parameters": _object(
                {
                    "entity_id": {"type": "string", "minLength": 1},
                    "purpose": {"type": "string", "enum": CANDIDATE_PURPOSES},
                },
                ["entity_id", "purpose"],
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_candidates",
            "description": "列出当前 episode 保存的候选及其快照证据。",
            "parameters": _object({"purpose": {"type": "string", "enum": CANDIDATE_PURPOSES}}, []),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_candidate",
            "description": "从当前 episode 的候选集中删除一项。",
            "parameters": _object(
                {"candidate_id": {"type": "string", "minLength": 1}}, ["candidate_id"]
            ),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_plan",
            "description": "提交引用已保存候选的结构化行程，并终止当前 episode。",
            "parameters": _object({"plan": PLAN}, ["plan"]),
        },
    },
)

# Hidden compatibility aliases allow archived v4 trajectories to replay without
# advertising redundant tools to newly trained or evaluated agents.
_LEGACY_PARAMETER_SCHEMAS = {
    "search_restaurants_by_food": _object(
        {
            "city": CITY,
            "food": {"type": "string", "minLength": 1},
            "min_price": PRICE,
            "max_price": PRICE,
            "sort_by": SORT,
        },
        ["city", "food"],
    ),
    "inspect_place": _object(
        {"place_id": {"type": "string", "minLength": 1}},
        ["place_id"],
    ),
    "save_candidate": _object(
        {
            "entity_id": {"type": "string", "minLength": 1},
            "purpose": {
                "type": "string",
                "enum": [*CANDIDATE_PURPOSES, "route_anchor", "other"],
            },
            "note": {"type": "string", "maxLength": 500},
        },
        ["entity_id", "purpose"],
    ),
}


def tool_schemas() -> list[dict[str, Any]]:
    """Return a defensive copy suitable for model tool registration."""

    return deepcopy(list(_TOOL_SCHEMAS))


def parameter_schema(name: str) -> dict[str, Any] | None:
    # Execution accepts the wider archived save_candidate shape while the model-visible
    # v5 schema deliberately omits its unused note and legacy purposes.
    if name == "save_candidate":
        return _LEGACY_PARAMETER_SCHEMAS[name]
    for schema in _TOOL_SCHEMAS:
        function = schema["function"]
        if function["name"] == name:
            return function["parameters"]
    return _LEGACY_PARAMETER_SCHEMAS.get(name)
