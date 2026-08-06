from __future__ import annotations

from travelweaver_env.tool_schemas import parameter_schema, tool_schemas


def test_tool_schema_names_and_defensive_copy() -> None:
    schemas = tool_schemas()
    assert [item["function"]["name"] for item in schemas] == [
        "search_attractions",
        "search_restaurants",
        "search_hotels",
        "search_intercity_transport",
        "search_nearby",
        "inspect_place",
        "get_route",
        "next_page",
        "save_candidate",
        "list_candidates",
        "remove_candidate",
        "submit_plan",
        "finish_without_plan",
    ]
    schemas[0]["function"]["name"] = "changed"
    assert tool_schemas()[0]["function"]["name"] == "search_attractions"
    assert parameter_schema("missing") is None
