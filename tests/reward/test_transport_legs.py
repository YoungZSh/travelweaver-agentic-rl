from __future__ import annotations

from copy import deepcopy

from travelweaver.reward import TravelReward
from travelweaver.rollout import DemoTravelAgent
from travelweaver.tasks import ConstraintSpec, build_base_spec


def test_transport_mode_constraints_score_outbound_and_return_independently(env) -> None:
    run = DemoTravelAgent(env).run("task-hangzhou")
    terminal = run.trajectory[-1]["result"]
    tool_result = terminal["observation"]["tool_result"]
    plan = deepcopy(tool_result["plan_snapshot"])
    evidence = deepcopy(tool_result["evidence_bundle"])
    public_task = run.trajectory[0]["observation"]["task"]

    returning = next(
        item
        for item in plan["activities"]
        if evidence["entities"][item["candidate_id"]].get("origin_city") == "杭州"
    )
    returning["activity_type"] = "airplane"
    constraints = (
        ConstraintSpec(
            id="outbound",
            kind="transport_mode",
            operator="eq",
            value={"modes": ["train"], "leg": "outbound"},
            scope="intercity_transport",
            hardness="hard",
            source_text="去程必须乘坐火车",
        ),
        ConstraintSpec(
            id="return",
            kind="transport_mode",
            operator="eq",
            value={"modes": ["airplane"], "leg": "return"},
            scope="intercity_transport",
            hardness="hard",
            source_text="返程必须乘坐飞机",
        ),
    )
    result = TravelReward().evaluate(
        build_base_spec(public_task, constraints=constraints),
        plan,
        evidence,
    )

    task_checks = {check.id: check for check in result.checks if check.source == "task_spec"}
    assert task_checks["outbound"].status == "pass"
    assert task_checks["return"].status == "pass"
    # Changing only the declared activity type does not change frozen candidate evidence.
    # Goal constraints pass independently, while environment validity correctly rejects it.
    assert next(check for check in result.checks if check.id == "candidate_usage").status == (
        "fail"
    )
    assert result.reward < 0.0
