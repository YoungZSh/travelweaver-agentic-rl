from __future__ import annotations

import json

import pytest

from travelweaver.env import InMemoryBackend, TravelWeaverEnv
from travelweaver.errors import EnvironmentStateError


def _step(env: TravelWeaverEnv, tool: str, **arguments: object):
    result = env.step({"tool": tool, "arguments": arguments})
    assert result.info["valid_action"] is True
    return result


def test_query_and_evidence_tools_and_pagination(env: TravelWeaverEnv) -> None:
    reset = env.reset("task-hangzhou")
    assert reset.remaining_steps == 50
    assert len(env.tool_schemas()) == 15

    categories = _step(env, "list_attraction_categories", city="杭州")
    assert "公园" in categories.observation.tool_result["categories"]
    cuisines = _step(env, "list_restaurant_cuisines", city="杭州")
    assert "杭帮菜" in cuisines.observation.tool_result["cuisines"]
    hotel_features = _step(env, "list_hotel_features", city="杭州")
    assert 2 in hotel_features.observation.tool_result["room_types"]

    attractions = _step(env, "search_attractions", city="杭州")
    first_page = attractions.observation.tool_result
    assert first_page is not None
    assert len(first_page["items"]) == 10
    assert first_page["page"]["total"] == 15
    cursor = first_page["page"]["next_cursor"]

    second_page = _step(env, "next_page", cursor=cursor)
    assert len(second_page.observation.tool_result["items"]) == 5
    assert second_page.observation.tool_result["page"]["next_cursor"] is None

    restaurants = _step(env, "search_restaurants", city="杭州", cuisine="杭帮菜")
    assert restaurants.observation.tool_result["items"][0]["name"] == "西湖餐厅"
    food = _step(env, "search_restaurants", city="杭州", recommended_food="西湖醋鱼")
    assert food.observation.tool_result["items"][0]["name"] == "西湖餐厅"

    hotels = _step(env, "search_hotels", city="杭州", room_type=2)
    assert hotels.observation.tool_result["items"][0]["name"] == "湖畔酒店"

    transport = _step(
        env,
        "search_intercity_transport",
        origin_city="上海",
        destination_city="杭州",
        mode="train",
    )
    assert transport.observation.tool_result["items"][0]["source_id"] == "G1"

    first_id, second_id = first_page["items"][0]["place_id"], first_page["items"][1]["place_id"]
    open_check = _step(env, "check_place_open", place_id=first_id, at_time="12:00")
    assert open_check.observation.tool_result["is_open"] is True

    nearby = _step(env, "search_nearby", place_id=first_id, category="attraction")
    assert nearby.observation.tool_result["items"]

    route = _step(
        env,
        "get_route",
        origin_place_id=first_id,
        destination_place_id=second_id,
        mode="walk",
        start_time="09:00",
    )
    route_payload = route.observation.tool_result["route"]
    assert route_payload["segments"][0]["mode"] == "walk"
    assert route_payload["route_id"] in route.observation.visible_route_ids
    json.dumps(route.to_dict(), ensure_ascii=False)


def test_filters_are_deterministic_and_no_result_is_valid(env: TravelWeaverEnv) -> None:
    env.reset("task-hangzhou")
    filtered = _step(
        env,
        "search_attractions",
        city="杭州",
        category="公园",
        max_price=45,
        open_at="12:00",
        sort_by="price",
    )
    items = filtered.observation.tool_result["items"]
    assert [item["price"] for item in items] == [0.0, 20.0, 40.0]

    empty = _step(env, "search_hotels", city="杭州", query="不存在")
    assert empty.observation.tool_result["items"] == []
    assert empty.observation.tool_result["page"]["total"] == 0


def test_unseen_ids_and_schema_errors_count_as_invalid(env: TravelWeaverEnv, backend) -> None:
    env.reset("task-hangzhou")
    hidden_id = backend.search_hotels(city="杭州")[0]["place_id"]
    invalid = env.step(
        {"tool": "check_place_open", "arguments": {"place_id": hidden_id, "at_time": "12:00"}}
    )
    assert not invalid.info["valid_action"]
    assert invalid.info["consecutive_invalid_actions"] == 1

    unknown_argument = env.step(
        {"tool": "search_attractions", "arguments": {"city": "杭州", "secret": True}}
    )
    assert not unknown_argument.info["valid_action"]
    terminated = env.step({"tool": "unknown", "arguments": {}})
    assert terminated.terminated
    assert terminated.info["termination_reason"] == "invalid_action_limit"
    with pytest.raises(EnvironmentStateError):
        env.step({"tool": "search_attractions", "arguments": {"city": "杭州"}})


def test_valid_action_resets_invalid_streak(env: TravelWeaverEnv) -> None:
    env.reset("task-hangzhou")
    env.step({"tool": "unknown", "arguments": {}})
    valid = _step(env, "search_attractions", city="杭州")
    assert valid.info["valid_action"]
    invalid = env.step({"tool": "unknown", "arguments": {}})
    assert invalid.info["consecutive_invalid_actions"] == 1


def test_step_limit_truncates(backend, task_store) -> None:
    env = TravelWeaverEnv(backend, task_store, max_valid_steps=2)
    env.reset("task-hangzhou")
    assert not _step(env, "search_attractions", city="杭州").truncated
    final = _step(env, "search_restaurants", city="杭州")
    assert final.truncated
    assert final.observation.remaining_steps == 0
    assert final.info["termination_reason"] == "step_limit"


def test_rejected_submission_is_terminal_on_the_last_valid_step(backend, task_store) -> None:
    env = TravelWeaverEnv(backend, task_store, max_valid_steps=1)
    env.reset("task-hangzhou")

    final = _step(
        env,
        "submit_plan",
        plan={
            "people_number": 1,
            "start_city": "上海",
            "target_city": "杭州",
            "itinerary": [
                {
                    "day": 1,
                    "activities": [
                        {
                            "candidate_id": "not-saved",
                            "type": "attraction",
                            "start_time": "10:00",
                            "end_time": "12:00",
                        }
                    ],
                }
            ],
        },
    )

    assert final.terminated
    assert not final.truncated
    assert final.observation.remaining_steps == 0
    assert final.info["termination_reason"] == "invalid_plan_submitted"


def test_default_step_limit_truncates_after_50_valid_actions(backend, task_store) -> None:
    env = TravelWeaverEnv(backend, task_store)
    reset = env.reset("task-hangzhou")
    assert reset.remaining_steps == 50

    for expected_remaining in range(49, 0, -1):
        step = _step(env, "search_attractions", city="杭州")
        assert not step.truncated
        assert step.observation.remaining_steps == expected_remaining

    final = _step(env, "search_attractions", city="杭州")
    assert final.truncated
    assert final.observation.remaining_steps == 0
    assert final.info["termination_reason"] == "step_limit"


def test_cursor_and_visible_state_are_episode_local(backend, task_store) -> None:
    first = TravelWeaverEnv(backend, task_store)
    second = TravelWeaverEnv(backend, task_store)
    first.reset("task-hangzhou")
    second.reset("task-hangzhou")
    page = _step(first, "search_attractions", city="杭州").observation.tool_result
    cursor = page["page"]["next_cursor"]
    first_visible = page["items"][0]["place_id"]

    foreign_cursor = second.step({"tool": "next_page", "arguments": {"cursor": cursor}})
    assert not foreign_cursor.info["valid_action"]
    foreign_id = second.step(
        {
            "tool": "check_place_open",
            "arguments": {"place_id": first_visible, "at_time": "12:00"},
        }
    )
    assert not foreign_id.info["valid_action"]

    first.reset("task-shanghai")
    expired = first.step({"tool": "next_page", "arguments": {"cursor": cursor}})
    assert not expired.info["valid_action"]
    assert first_visible not in expired.observation.visible_entity_ids


def test_close_and_pre_reset_state_errors(backend: InMemoryBackend, task_store) -> None:
    env = TravelWeaverEnv(backend, task_store)
    with pytest.raises(EnvironmentStateError):
        env.step({"tool": "search_attractions", "arguments": {"city": "杭州"}})
    env.reset("task-hangzhou")
    env.close()
    with pytest.raises(EnvironmentStateError):
        env.reset("task-hangzhou")


def test_candidate_management_and_plan_submission_close_the_loop(env: TravelWeaverEnv) -> None:
    env.reset("task-hangzhou")
    attraction = _step(env, "search_attractions", city="杭州").observation.tool_result["items"][0]
    restaurant = _step(env, "search_restaurants", city="杭州").observation.tool_result["items"][0]
    outbound = _step(
        env,
        "search_intercity_transport",
        origin_city="上海",
        destination_city="杭州",
        mode="train",
    ).observation.tool_result["items"][0]
    returning = _step(
        env,
        "search_intercity_transport",
        origin_city="杭州",
        destination_city="上海",
        mode="train",
    ).observation.tool_result["items"][0]

    route = _step(
        env,
        "get_route",
        origin_place_id=attraction["place_id"],
        destination_place_id=restaurant["place_id"],
        mode="walk",
        start_time="12:00",
    ).observation.tool_result["route"]
    arrival_route = _step(
        env,
        "get_route",
        origin_place_id=outbound["destination_anchor_id"],
        destination_place_id=attraction["place_id"],
        mode="walk",
        start_time="09:00",
    ).observation.tool_result["route"]
    return_route = _step(
        env,
        "get_route",
        origin_place_id=restaurant["place_id"],
        destination_place_id=returning["origin_anchor_id"],
        mode="walk",
        start_time="13:30",
    ).observation.tool_result["route"]

    for item, purpose in (
        (attraction, "attraction"),
        (restaurant, "meal"),
        (outbound, "outbound_transport"),
        (returning, "return_transport"),
    ):
        entity_id = item.get("place_id") or item.get("transport_id")
        saved = _step(env, "save_candidate", entity_id=entity_id, purpose=purpose)
        assert saved.observation.tool_result["status"] == "saved"

    listed = _step(env, "list_candidates")
    assert listed.observation.tool_result["count"] == 4
    assert len(listed.observation.candidates) == 4

    removed = _step(env, "remove_candidate", candidate_id=restaurant["place_id"])
    assert removed.observation.tool_result["status"] == "removed"
    _step(
        env,
        "save_candidate",
        entity_id=restaurant["place_id"],
        purpose="meal",
    )

    plan = {
        "people_number": 1,
        "start_city": "上海",
        "target_city": "杭州",
        "itinerary": [
            {
                "day": 1,
                "activities": [
                    {
                        "candidate_id": outbound["transport_id"],
                        "type": "train",
                        "start_time": "08:00",
                        "end_time": "09:00",
                    },
                    {
                        "candidate_id": attraction["place_id"],
                        "type": "attraction",
                        "start_time": "10:00",
                        "end_time": "12:00",
                        "route_from_previous_id": arrival_route["route_id"],
                    },
                    {
                        "candidate_id": restaurant["place_id"],
                        "type": "lunch",
                        "start_time": "12:30",
                        "end_time": "13:30",
                        "route_from_previous_id": route["route_id"],
                    },
                    {
                        "candidate_id": returning["transport_id"],
                        "type": "train",
                        "start_time": "18:00",
                        "end_time": "19:00",
                        "route_from_previous_id": return_route["route_id"],
                    },
                ],
            }
        ],
    }
    submitted = _step(env, "submit_plan", plan=plan)
    assert submitted.terminated
    assert not submitted.truncated
    assert submitted.info["termination_reason"] == "plan_submitted"
    assert submitted.reward == 1.0
    assert submitted.info["reward_detail"]["all_hard_pass"] is True
    assert submitted.observation.tool_result["status"] == "accepted"
    assert submitted.observation.tool_result["validation"]["candidate_grounding"]
    assert submitted.observation.tool_result["validation"]["route_grounding"]
    assert submitted.observation.tool_result["plan_snapshot"]["total_cost"] == 234.0
    assert set(submitted.observation.tool_result["evidence_bundle"]["routes"]) == {
        arrival_route["route_id"],
        route["route_id"],
        return_route["route_id"],
    }


def test_schema_valid_invalid_submission_terminates_without_retry(env: TravelWeaverEnv) -> None:
    env.reset("task-hangzhou")
    invalid = env.step(
        {
            "tool": "submit_plan",
            "arguments": {
                "plan": {
                    "people_number": 1,
                    "start_city": "上海",
                    "target_city": "杭州",
                    "itinerary": [
                        {
                            "day": 1,
                            "activities": [
                                {
                                    "candidate_id": "not-saved",
                                    "type": "attraction",
                                    "start_time": "10:00",
                                    "end_time": "12:00",
                                }
                            ],
                        }
                    ],
                }
            },
        }
    )
    assert invalid.info["valid_action"]
    assert invalid.terminated
    assert invalid.info["termination_reason"] == "invalid_plan_submitted"
    assert -1.0 < invalid.reward < 0.0
    detail = invalid.info["reward_detail"]
    assert detail["reward_type"] == "partial_invalid_plan"
    assert detail["admission_passed"] is False
    assert detail["reward_valid"] is True


def test_submit_rejects_candidate_saved_for_the_wrong_purpose(
    env: TravelWeaverEnv,
) -> None:
    env.reset("task-hangzhou")
    attraction = _step(env, "search_attractions", city="杭州").observation.tool_result[
        "items"
    ][0]
    _step(
        env,
        "save_candidate",
        entity_id=attraction["place_id"],
        purpose="meal",
    )

    rejected = _step(
        env,
        "submit_plan",
        plan={
            "people_number": 1,
            "start_city": "上海",
            "target_city": "杭州",
            "itinerary": [
                {
                    "day": 1,
                    "activities": [
                        {
                            "candidate_id": attraction["place_id"],
                            "type": "attraction",
                            "start_time": "10:00",
                            "end_time": "12:00",
                        }
                    ],
                }
            ],
        },
    )

    assert rejected.terminated
    assert rejected.info["termination_reason"] == "invalid_plan_submitted"
    assert "saved for purpose 'meal'" in rejected.info["submission_error"]["message"]


@pytest.mark.parametrize(
    ("activity_type", "entity_type", "purpose"),
    [
        ("attraction", "attraction", "attraction"),
        ("breakfast", "restaurant", "meal"),
        ("breakfast", "hotel", "hotel"),
        ("lunch", "restaurant", "meal"),
        ("dinner", "restaurant", "meal"),
        ("accommodation", "hotel", "hotel"),
    ],
)
def test_candidate_purpose_accepts_supported_local_uses(
    activity_type: str,
    entity_type: str,
    purpose: str,
) -> None:
    TravelWeaverEnv._validate_candidate_purpose(
        {"candidate_id": "candidate", "purpose": purpose},
        activity_type,
        {"entity_type": entity_type},
        start_city="上海",
        target_city="杭州",
    )


def test_candidate_purpose_distinguishes_outbound_and_return_transport() -> None:
    TravelWeaverEnv._validate_candidate_purpose(
        {"candidate_id": "outbound", "purpose": "outbound_transport"},
        "train",
        {"origin_city": "上海", "destination_city": "杭州"},
        start_city="上海",
        target_city="杭州",
    )
    with pytest.raises(ValueError, match="return_transport"):
        TravelWeaverEnv._validate_candidate_purpose(
            {"candidate_id": "return", "purpose": "outbound_transport"},
            "train",
            {"origin_city": "杭州", "destination_city": "上海"},
            start_city="上海",
            target_city="杭州",
        )
