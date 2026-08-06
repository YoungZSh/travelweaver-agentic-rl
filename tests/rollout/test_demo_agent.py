from __future__ import annotations

from travelweaver.rollout import DemoTravelAgent


def test_demo_agent_reaches_plan_submitted(env) -> None:
    run = DemoTravelAgent(env).run("task-hangzhou")
    assert run.success
    assert run.termination_reason == "plan_submitted"
    assert run.final_plan is not None
    assert run.final_plan["itinerary"][0]["activities"]
    actions = [event["action"]["tool"] for event in run.trajectory if event["event"] == "step"]
    assert "save_candidate" in actions
    assert "list_candidates" in actions
    assert actions[-1] == "submit_plan"
