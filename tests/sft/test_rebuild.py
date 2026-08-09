from __future__ import annotations

import json
from pathlib import Path

from travelweaver.env import ScenarioBackend, ScenarioSpec, TravelWeaverEnv
from travelweaver.rollout import MODEL_TOOL_RESPONSE_VERSION, DemoTravelAgent
from travelweaver.sft.rebuild import SFTSource, _is_accepted, _PreparedTask, _rebuild_one


def test_sft_acceptance_explicitly_supports_v3_and_v4_trajectories() -> None:
    base = {
        "termination_reason": "plan_submitted",
        "final_reward": 1.0,
        "rft_accepted": True,
        "reward_detail": {"reward_valid": True, "all_hard_pass": True},
    }

    assert _is_accepted({**base, "trajectory_version": "travelweaver-trajectory-v3"})
    assert _is_accepted({**base, "trajectory_version": "travelweaver-trajectory-v4"})
    assert _is_accepted({**base, "trajectory_version": "travelweaver-trajectory-v5"})
    assert not _is_accepted({**base, "trajectory_version": "travelweaver-trajectory-v2"})


def test_rebuild_skips_invalid_turn_and_replays_reward_one(backend, task_store) -> None:
    scenario = ScenarioSpec(
        base_world_snapshot_version="test-world",
        profile="normal",
        effects=(),
    )
    source_env = TravelWeaverEnv(ScenarioBackend(backend, scenario), task_store)
    run = DemoTravelAgent(source_env).run("task-hangzhou")
    assert run.success
    reset = run.trajectory[0]["observation"]
    steps = [event for event in run.trajectory if event["event"] == "step"]
    invalid = {
        "index": 1,
        "action": {"tool": "submit_plan", "arguments": {"plan": {}}},
        "result": {
            "observation": {
                "error": {"code": "invalid_action", "message": "missing fields"}
            },
            "info": {"valid_action": False},
        },
    }
    steps.insert(1, invalid)
    public = task_store.get_public("task-hangzhou")
    oracle = {
        **task_store.get_oracle("task-hangzhou"),
        "scenario": scenario.to_dict(),
    }
    row = {
        "episode_id": reset["episode_id"],
        "task_id": "task-hangzhou",
        "model": "fake-model",
        "steps": steps,
        "tools": source_env.tool_schemas(),
    }

    sample, audit = _rebuild_one(
        SFTSource(Path("generated"), Path("rollout.jsonl")),
        row,
        _PreparedTask(public, oracle, None, public["query"]),
        backend,
    )

    assert audit["invalid_actions_removed"] == 1
    assert audit["replay_reward"] == 1.0
    assert sample["enable_thinking"] is False
    assert sample["tool_response_mode"] == "delta"
    assert sample["model_tool_response_version"] == MODEL_TOOL_RESPONSE_VERSION
    assert sample["format_version"] == "travelweaver-sft-v2"
    assert sample["user_content_format"] == "travelweaver-natural-query-v1"
    assert sample["messages"][1] == {
        "role": "user",
        "content": public["query"],
    }
    assert sample["messages"][-1]["role"] == "assistant"
    assert sample["messages"][-1]["tool_calls"][0]["function"]["name"] == "submit_plan"
    assert all("reasoning_content" not in message for message in sample["messages"])
    tool_payloads = [
        json.loads(message["content"])
        for message in sample["messages"]
        if message["role"] == "tool"
    ]
    assert tool_payloads
    assert all(
        payload["response_version"] == MODEL_TOOL_RESPONSE_VERSION
        for payload in tool_payloads
    )
    assert all(
        "reward" not in payload and "observation" not in payload
        for payload in tool_payloads
    )
