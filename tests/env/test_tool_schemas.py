from __future__ import annotations

from travelweaver.env.tool_schemas import parameter_schema, tool_schemas


def test_tool_schema_names_and_defensive_copy() -> None:
    schemas = tool_schemas()
    assert [item["function"]["name"] for item in schemas] == [
        "list_attraction_categories",
        "search_attractions",
        "list_restaurant_cuisines",
        "search_restaurants",
        "list_hotel_features",
        "search_hotels",
        "search_intercity_transport",
        "search_nearby",
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
    public_save = next(
        item for item in tool_schemas() if item["function"]["name"] == "save_candidate"
    )
    assert "note" not in public_save["function"]["parameters"]["properties"]
    assert "note" in parameter_schema("save_candidate")["properties"]
    assert parameter_schema("search_restaurants_by_food") is not None
    assert parameter_schema("inspect_place") is not None
    assert parameter_schema("missing") is None
