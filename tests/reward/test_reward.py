from __future__ import annotations

from copy import deepcopy

from travelweaver.reward import TravelReward, strict_rft_filter
from travelweaver.rollout import DemoTravelAgent
from travelweaver.tasks import ConstraintSpec, build_base_spec


def _terminal_evidence(env):
    run = DemoTravelAgent(env).run("task-hangzhou")
    terminal = run.trajectory[-1]["result"]
    tool_result = terminal["observation"]["tool_result"]
    public_task = run.trajectory[0]["observation"]["task"]
    return terminal, public_task, tool_result["plan_snapshot"], tool_result["evidence_bundle"]


def test_environment_returns_strict_terminal_reward(env) -> None:
    terminal, _, _, _ = _terminal_evidence(env)

    assert terminal["reward"] == 1.0
    detail = terminal["info"]["reward_detail"]
    assert detail["reward_version"] == "travelweaver-reward-v1"
    assert detail["reward_type"] == "strict_valid_plan"
    assert detail["all_hard_pass"] is True
    assert all(check["status"] == "pass" for check in detail["checks"])


def test_hard_failure_is_negative_and_soft_failure_stays_positive(env) -> None:
    _, public_task, plan, evidence = _terminal_evidence(env)
    hard_budget = ConstraintSpec(
        id="budget",
        kind="total_budget",
        operator="lte",
        value={"amount": 100, "currency": "CNY"},
        scope="trip",
        hardness="hard",
        source_text=public_task["query"],
    )
    hard_spec = build_base_spec(public_task, constraints=(hard_budget,))
    hard_result = TravelReward().evaluate(hard_spec, plan, evidence)

    assert hard_result.reward_valid
    assert -1.0 <= hard_result.reward < 0.0
    assert not hard_result.all_hard_pass

    soft_category = ConstraintSpec(
        id="category",
        kind="entity_category",
        operator="contains",
        value={"values": ["不存在的景点类型"]},
        scope="attraction",
        hardness="soft",
        source_text=public_task["query"],
    )
    soft_spec = build_base_spec(public_task, constraints=(soft_category,))
    soft_result = TravelReward().evaluate(soft_spec, plan, evidence)

    assert soft_result.all_hard_pass
    assert soft_result.soft_score == 0.0
    assert soft_result.reward == 0.5


def test_unverifiable_evidence_is_excluded_from_rft(env) -> None:
    _, public_task, plan, evidence = _terminal_evidence(env)
    broken = deepcopy(evidence)
    broken["total_cost"] = None
    result = TravelReward().evaluate(build_base_spec(public_task), plan, broken)
    decision = strict_rft_filter(result, termination_reason="plan_submitted")

    assert result.reward == 0.0
    assert not result.reward_valid
    assert not decision.accepted
    assert decision.reason == "reward_unverifiable"


def test_reward_independently_rejects_missing_routes_and_tampered_costs(env) -> None:
    _, public_task, plan, evidence = _terminal_evidence(env)
    spec = build_base_spec(public_task)

    missing_route_plan = deepcopy(plan)
    restaurant = next(
        item
        for item in missing_route_plan["activities"]
        if item["activity_type"] == "lunch"
    )
    restaurant["route_from_previous_id"] = None
    missing_route = TravelReward().evaluate(spec, missing_route_plan, evidence)
    assert missing_route.reward < 0
    route_check = next(
        check for check in missing_route.checks if check.id == "route_grounding"
    )
    assert route_check.status == "fail"

    tampered_evidence = deepcopy(evidence)
    tampered_item = tampered_evidence["cost_items"][0]
    tampered_item["amount"] += 1
    tampered_evidence["total_cost"] += 1
    tampered = TravelReward().evaluate(spec, plan, tampered_evidence)
    assert tampered.reward < 0
    assert next(check for check in tampered.checks if check.id == "cost_accounting").status == (
        "fail"
    )


def test_rft_accepts_only_normal_valid_hard_pass(env) -> None:
    terminal, _, _, _ = _terminal_evidence(env)
    detail = terminal["info"]["reward_detail"]
    # Re-evaluate through public objects instead of trusting a serialized success flag.
    _, public_task, plan, evidence = _terminal_evidence(env)
    result = TravelReward().evaluate(build_base_spec(public_task), plan, evidence)

    accepted = strict_rft_filter(result, termination_reason="plan_submitted")
    wrong_terminal = strict_rft_filter(result, termination_reason="step_limit")
    assert detail["reward_valid"] is True
    assert accepted.accepted
    assert not wrong_terminal.accepted
