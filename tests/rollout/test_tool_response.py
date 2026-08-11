from __future__ import annotations

from travelweaver.rollout.tool_response import _compact_tool_result


def test_compact_search_result_keeps_entity_decision_fields_without_coordinates() -> None:
    payload = {
        "tool": "search_restaurants",
        "items": [
            {
                "place_id": "place-1",
                "entity_type": "restaurant",
                "city": "杭州",
                "name": "示例餐厅",
                "price": 88.0,
                "latitude": 30.1,
                "longitude": 120.2,
                "cuisine": "杭帮菜",
                "recommended_food": "西湖醋鱼,龙井虾仁",
                "open_time": "10:00",
                "close_time": "22:00",
            }
        ],
        "page": {
            "offset": 0,
            "page_size": 10,
            "returned": 1,
            "total": 1,
            "next_cursor": None,
        },
    }

    compact = _compact_tool_result(payload)

    assert compact == {
        "tool": "search_restaurants",
        "items": [
            {
                "place_id": "place-1",
                "entity_type": "restaurant",
                "city": "杭州",
                "name": "示例餐厅",
                "price": 88.0,
                "cuisine": "杭帮菜",
                "recommended_food_hint": "西湖醋鱼",
                "open_time": "10:00",
                "close_time": "22:00",
            }
        ],
        "page": {"returned": 1, "total": 1, "next_cursor": None},
    }


def test_compact_candidate_retains_visible_price_for_comparison() -> None:
    compact = _compact_tool_result(
        {
            "tool": "list_candidates",
            "items": [
                {
                    "candidate_id": "place-1",
                    "entity_id": "place-1",
                    "entity_type": "attraction",
                    "name": "示例景点",
                    "purpose": "attraction",
                    "note": None,
                    "evidence": {"price": 60.0, "latitude": 30.1},
                }
            ],
            "count": 1,
        }
    )

    assert compact["items"] == [
        {
            "candidate_id": "place-1",
            "entity_id": "place-1",
            "entity_type": "attraction",
            "name": "示例景点",
            "purpose": "attraction",
            "note": None,
            "price": 60.0,
        }
    ]
