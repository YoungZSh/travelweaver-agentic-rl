from __future__ import annotations

from travelweaver.evaluation import export_official_plan, validate_official_schema
from travelweaver.rollout import DemoTravelAgent


def test_reward_one_plan_exports_to_official_schema(env) -> None:
    run = DemoTravelAgent(env).run("task-hangzhou")
    terminal = run.trajectory[-1]["result"]
    result = terminal["observation"]["tool_result"]

    official = export_official_plan(result["plan_snapshot"], result["evidence_bundle"])

    assert validate_official_schema(official) == []
    activities = official["itinerary"][0]["activities"]
    assert activities[0]["TrainID"] == "G1"
    assert activities[-1]["TrainID"] == "G2"
    assert activities[1]["transports"]
