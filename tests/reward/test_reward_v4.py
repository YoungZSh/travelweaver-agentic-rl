from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

from travelweaver.reward import CHECK_DEFINITIONS, REWARD_DIMENSIONS, TravelReward
from travelweaver.rollout import DemoTravelAgent
from travelweaver.tasks import ConstraintSpec, build_base_spec


def test_reward_registry_has_one_owner_per_builtin_predicate() -> None:
    ids = [definition.check_id for definition in CHECK_DEFINITIONS]

    assert len(ids) == len(set(ids))
    assert all(definition.owner_dimension in REWARD_DIMENSIONS for definition in CHECK_DEFINITIONS)


def test_rejected_submission_collects_partial_reward_without_invalidating_sample(env) -> None:
    env.reset("task-hangzhou")
    result = env.step(
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

    detail = result.info["reward_detail"]
    dimensions = detail["dimension_scores"]
    expected = min(-1.0 + sum(dimensions.values()) / 3.0, -1e-8)
    assert detail["reward"] == round(expected, 8)
    assert detail["reward_valid"] is True
    assert detail["admission_passed"] is False
    assert detail["all_hard_pass"] is False
    assert detail["reward"] < 0.0
    assert set(dimensions) == set(REWARD_DIMENSIONS)
    checks = {check["id"]: check for check in detail["checks"]}
    assert checks["entity_grounding"]["status"] == "fail"
    assert checks["opening_hours"]["status"] == "blocked"
    assert checks["opening_hours"]["blocked_by"] == "entity_grounding"
    assert checks["trip_coverage"]["status"] == "blocked"
    assert checks["trip_coverage"]["blocked_by"] == "entity_grounding"


def test_no_plan_uses_the_same_v4_scalar_aliases(env) -> None:
    result = env.reward_evaluator.no_plan("step_limit").to_dict()

    assert result["reward_version"] == "travelweaver-reward-v4"
    assert result["reward"] == result["rl_reward"] == result["terminal_utility"] == -1.0
    assert result["dimension_coverage"]["artifact_conformance"]["scored"] == 1


def test_legacy_v1_success_snapshot_is_explicitly_upgraded_without_reinterpretation(env) -> None:
    run = DemoTravelAgent(env).run("task-hangzhou")
    terminal = run.trajectory[-1]["result"]
    tool_result = terminal["observation"]["tool_result"]
    plan = deepcopy(tool_result["plan_snapshot"])
    evidence = deepcopy(tool_result["evidence_bundle"])
    plan["schema_version"] = "travelweaver-plan-snapshot-v1"
    for activity in plan["activities"]:
        for field in (
            "absolute_start",
            "absolute_end",
            "origin_position_id",
            "destination_position_id",
        ):
            activity.pop(field, None)
    evidence["schema_version"] = "travelweaver-evidence-v1"
    evidence["routes"] = {}
    evidence.pop("candidate_usages", None)
    public_task = run.trajectory[0]["observation"]["task"]

    result = TravelReward().evaluate(build_base_spec(public_task), plan, evidence)

    assert result.reward == 1.0
    assert result.all_hard_pass
    route = next(check for check in result.checks if check.id == "route_grounding")
    candidate_usage = next(check for check in result.checks if check.id == "candidate_usage")
    assert route.status == candidate_usage.status == "not_applicable"


def test_missing_goal_content_is_a_valid_failure_not_infrastructure_error(env) -> None:
    reset = env.reset("task-hangzhou")
    constraint = ConstraintSpec(
        id="rooms",
        kind="room_count",
        operator="gte",
        value={"count": 1},
        scope="accommodation",
        hardness="hard",
        source_text=reset.task["query"],
    )
    spec = build_base_spec(reset.task, constraints=(constraint,))
    raw_plan = {
        "people_number": 1,
        "start_city": "上海",
        "target_city": "杭州",
        "itinerary": [{"day": 1, "activities": []}],
    }

    result = TravelReward().evaluate_submission(spec, raw_plan, {}, {})
    room_check = next(check for check in result.checks if check.id == "rooms")

    assert result.reward_valid
    assert result.reward < 0.0
    assert result.dimension_scores["artifact_conformance"] == 0.5
    assert room_check.status == "fail"
    assert room_check.score == 0.0


def test_goal_coverage_owns_destination_and_required_nights(env) -> None:
    run = DemoTravelAgent(env).run("task-hangzhou")
    terminal = run.trajectory[-1]["result"]["observation"]["tool_result"]
    public_task = run.trajectory[0]["observation"]["task"]
    spec = build_base_spec(public_task)

    wrong_city_evidence = deepcopy(terminal["evidence_bundle"])
    attraction = next(
        item
        for item in terminal["plan_snapshot"]["activities"]
        if item["activity_type"] == "attraction"
    )
    wrong_city_evidence["entities"][attraction["candidate_id"]]["city"] = "上海"
    wrong_city = TravelReward().evaluate(
        spec, terminal["plan_snapshot"], wrong_city_evidence
    )

    two_day_spec = replace(spec, trip=replace(spec.trip, days=2))
    missing_night_plan = deepcopy(terminal["plan_snapshot"])
    missing_night_plan["days"] = 2
    missing_night_plan["activities"] = [
        item
        for item in missing_night_plan["activities"]
        if item["activity_type"] != "accommodation"
    ]
    missing_night = TravelReward().evaluate(
        two_day_spec, missing_night_plan, terminal["evidence_bundle"]
    )

    wrong_city_check = next(
        check for check in wrong_city.checks if check.id == "trip_coverage"
    )
    missing_night_check = next(
        check for check in missing_night.checks if check.id == "trip_coverage"
    )
    assert wrong_city_check.status == "fail"
    assert wrong_city_check.evidence["local_destination_matches"] < wrong_city_check.evidence[
        "local_activity_count"
    ]
    assert missing_night_check.status == "fail"
    assert missing_night_check.evidence["accommodation_nights"] == 0


def test_goal_and_validity_failures_do_not_change_each_others_dimension(env) -> None:
    run = DemoTravelAgent(env).run("task-hangzhou")
    terminal = run.trajectory[-1]["result"]["observation"]["tool_result"]
    spec = build_base_spec(run.trajectory[0]["observation"]["task"])
    baseline = TravelReward().evaluate(
        spec, terminal["plan_snapshot"], terminal["evidence_bundle"]
    )

    wrong_goal_plan = deepcopy(terminal["plan_snapshot"])
    wrong_goal_plan["target_city"] = "北京"
    wrong_goal = TravelReward().evaluate(
        spec, wrong_goal_plan, terminal["evidence_bundle"]
    )

    invalid_evidence = deepcopy(terminal["evidence_bundle"])
    attraction = next(
        item
        for item in terminal["plan_snapshot"]["activities"]
        if item["activity_type"] == "attraction"
    )
    invalid_evidence["entities"][attraction["candidate_id"]]["close_time"] = "09:00"
    invalid_world = TravelReward().evaluate(
        spec, terminal["plan_snapshot"], invalid_evidence
    )

    assert wrong_goal.dimension_scores["goal_satisfaction"] < baseline.dimension_scores[
        "goal_satisfaction"
    ]
    assert wrong_goal.dimension_scores["environment_validity"] == baseline.dimension_scores[
        "environment_validity"
    ]
    assert invalid_world.dimension_scores["environment_validity"] < baseline.dimension_scores[
        "environment_validity"
    ]
    assert invalid_world.dimension_scores["goal_satisfaction"] == baseline.dimension_scores[
        "goal_satisfaction"
    ]
