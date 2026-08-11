from __future__ import annotations

from travelweaver.env.tool_schemas import parameter_schema, tool_schemas


def test_tool_schema_names_and_defensive_copy() -> None:
    schemas = tool_schemas()
    assert [item["function"]["name"] for item in schemas] == [
        "list_attraction_categories",
        "search_attractions",
        "list_restaurant_cuisines",
        "search_restaurants",
        "search_restaurants_by_food",
        "list_hotel_features",
        "search_hotels",
        "search_intercity_transport",
        "search_nearby",
        "inspect_place",
        "check_place_open",
        "get_route",
        "next_page",
        "save_candidate",
        "list_candidates",
        "remove_candidate",
        "submit_plan",
    ]
    schemas[0]["function"]["name"] = "changed"
    assert tool_schemas()[0]["function"]["name"] == "list_attraction_categories"
    assert parameter_schema("missing") is None
