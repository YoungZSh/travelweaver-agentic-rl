from __future__ import annotations

import json
from pathlib import Path

from travelweaver.env import ScenarioBackend, ScenarioSpec, TravelWeaverEnv
from travelweaver.rollout import MODEL_TOOL_RESPONSE_VERSION, DemoTravelAgent
from travelweaver.rollout.api_agent import SYSTEM_PROMPT, render_system_prompt
from travelweaver.sft.ordering import order_tool_arguments
from travelweaver.sft.rebuild import (
    SFTSource,
    _is_accepted,
    _mode_exclusion_reason,
    _PreparedTask,
    _rebuild_one,
)


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
    assert _is_accepted({**base, "trajectory_version": "travelweaver-trajectory-v6"})
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
    assert sample["format_version"] == "travelweaver-sft-v4"
    assert sample["supervision_mode"] == "action_only"
    assert sample["assistant_loss_mask"] == [
        True
        for message in sample["messages"]
        if message["role"] == "assistant"
    ]
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


def test_react_rebuild_preserves_visible_text_for_clean_no_thinking_run(
    backend, task_store
) -> None:
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
    public = task_store.get_public("task-hangzhou")
    messages: list[dict[str, object]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": public["query"]},
    ]
    for index, step in enumerate(steps):
        call_id = f"call-{index}"
        action = step["action"]
        step["tool_call"] = {
            "id": call_id,
            "type": "function",
            "function": {
                "name": action["tool"],
                "arguments": json.dumps(action["arguments"], ensure_ascii=False),
            },
        }
        messages.append(
            {
                "role": "assistant",
                "content": f"第 {index + 1} 步先执行 {action['tool']}。",
                "tool_calls": [step["tool_call"]],
            }
        )
    oracle = {
        **task_store.get_oracle("task-hangzhou"),
        "scenario": scenario.to_dict(),
    }
    row = {
        "episode_id": reset["episode_id"],
        "task_id": "task-hangzhou",
        "model": "fake-no-thinking-model",
        "batch_metadata": {"thinking": "disabled"},
        "messages": messages,
        "steps": steps,
        "tools": source_env.tool_schemas(),
    }

    assert _mode_exclusion_reason(row, "react") is None
    sample, audit = _rebuild_one(
        SFTSource(Path("generated"), Path("rollout.jsonl")),
        row,
        _PreparedTask(public, oracle, None, public["query"]),
        backend,
        supervision_mode="react",
    )

    assistants = [message for message in sample["messages"] if message["role"] == "assistant"]
    assert sample["format_version"] == "travelweaver-sft-v4"
    assert sample["supervision_mode"] == "react"
    assert all(message["content"].startswith("第 ") for message in assistants)
    assert all("reasoning_content" not in message for message in assistants)
    assert audit["assistant_turns_with_content"] == len(steps)
    assert audit["invalid_actions_removed"] == 0

    row["steps"][0]["result"]["info"]["valid_action"] = False
    assert _mode_exclusion_reason(row, "react") == "react_invalid_action_not_supported"


def test_react_rebuild_preserves_legacy_35_step_prompt(backend, task_store) -> None:
    scenario = ScenarioSpec(
        base_world_snapshot_version="test-world",
        profile="normal",
        effects=(),
    )
    source_env = TravelWeaverEnv(
        ScenarioBackend(backend, scenario), task_store, max_valid_steps=35
    )
    run = DemoTravelAgent(source_env).run("task-hangzhou")
    assert run.success
    reset = run.trajectory[0]["observation"]
    steps = [event for event in run.trajectory if event["event"] == "step"]
    public = task_store.get_public("task-hangzhou")
    legacy_prompt = render_system_prompt(35)
    messages: list[dict[str, object]] = [
        {"role": "system", "content": legacy_prompt},
        {"role": "user", "content": public["query"]},
    ]
    for index, step in enumerate(steps):
        call_id = f"legacy-call-{index}"
        action = step["action"]
        step["tool_call"] = {
            "id": call_id,
            "type": "function",
            "function": {
                "name": action["tool"],
                "arguments": json.dumps(action["arguments"], ensure_ascii=False),
            },
        }
        messages.append(
            {
                "role": "assistant",
                "content": f"执行 {action['tool']}。",
                "tool_calls": [step["tool_call"]],
            }
        )
    oracle = {
        **task_store.get_oracle("task-hangzhou"),
        "scenario": scenario.to_dict(),
    }
    row = {
        "episode_id": reset["episode_id"],
        "task_id": "task-hangzhou",
        "model": "legacy-no-thinking-model",
        "batch_metadata": {"thinking": "disabled"},
        "messages": messages,
        "steps": steps,
        "tools": source_env.tool_schemas(),
    }

    sample, _ = _rebuild_one(
        SFTSource(Path("generated"), Path("rollout.jsonl")),
        row,
        _PreparedTask(public, oracle, None, public["query"]),
        backend,
        supervision_mode="react",
    )

    assert sample["messages"][0] == {"role": "system", "content": legacy_prompt}
    first_tool = next(message for message in sample["messages"] if message["role"] == "tool")
    assert json.loads(first_tool["content"])["remaining_steps"] == 34


def test_react_recovery_retains_invalid_turn_as_masked_context(backend, task_store) -> None:
    scenario = ScenarioSpec(
        base_world_snapshot_version="test-world",
        profile="normal",
        effects=(),
    )
    source_env = TravelWeaverEnv(ScenarioBackend(backend, scenario), task_store)
    run = DemoTravelAgent(source_env).run("task-hangzhou")
    assert run.success
    reset = run.trajectory[0]["observation"]
    valid_steps = [event for event in run.trajectory if event["event"] == "step"]
    invalid = {
        "index": 1,
        "action": {"tool": "config", "arguments": {"entity_id": "place:unknown"}},
        "result": {
            "observation": {
                "error": {"code": "invalid_action", "message": "missing fields"}
            },
            "info": {"valid_action": False},
        },
    }
    invalid_cursor = {
        "index": 2,
        "action": {"tool": "next_page", "arguments": {"cursor": "unknown-cursor"}},
        "result": {
            "observation": {
                "error": {"code": "invalid_action", "message": "unknown cursor"}
            },
            "info": {"valid_action": False},
        },
    }
    steps = [valid_steps[0], invalid, invalid_cursor, *valid_steps[1:]]
    public = task_store.get_public("task-hangzhou")
    messages: list[dict[str, object]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": public["query"]},
    ]
    for index, step in enumerate(steps):
        call_id = f"call-{index}"
        action = step["action"]
        step["tool_call"] = {
            "id": call_id,
            "type": "function",
            "function": {
                "name": action["tool"],
                "arguments": json.dumps(action["arguments"], ensure_ascii=False),
            },
        }
        messages.append(
            {
                "role": "assistant",
                "content": (
                    "我先尝试提交。" if step is invalid else f"执行 {action['tool']}。"
                ),
                "tool_calls": [step["tool_call"]],
            }
        )
    oracle = {
        **task_store.get_oracle("task-hangzhou"),
        "scenario": scenario.to_dict(),
    }
    row = {
        "episode_id": reset["episode_id"],
        "task_id": "task-hangzhou",
        "model": "fake-no-thinking-model",
        "batch_metadata": {"thinking": "disabled"},
        "messages": messages,
        "steps": steps,
        "tools": source_env.tool_schemas(),
    }

    assert _mode_exclusion_reason(row, "react_recovery") is None
    sample, audit = _rebuild_one(
        SFTSource(Path("generated"), Path("rollout.jsonl")),
        row,
        _PreparedTask(public, oracle, None, public["query"]),
        backend,
        supervision_mode="react_recovery",
    )

    assert sample["supervision_mode"] == "react_recovery"
    assert sample["assistant_loss_mask"].count(False) == 2
    masked_index = sample["assistant_loss_mask"].index(False)
    assistants = [m for m in sample["messages"] if m["role"] == "assistant"]
    assert assistants[masked_index]["content"] == "我先尝试提交。"
    assert assistants[masked_index]["tool_calls"][0]["function"]["name"] == "config"
    assert assistants[masked_index + 1]["tool_calls"][0]["function"]["name"] == "next_page"
    error_payloads = [
        json.loads(message["content"])
        for message in sample["messages"]
        if message["role"] == "tool"
        and json.loads(message["content"])["valid_action"] is False
    ]
    assert len(error_payloads) == 2
    assert audit["invalid_actions_removed"] == 0
    assert audit["invalid_actions_retained"] == 2
    assert audit["masked_assistant_turns"] == 2
    assert audit["replay_reward"] == 1.0


def test_tool_arguments_are_recursively_schema_ordered(env) -> None:
    tools = env.tool_schemas()
    save = order_tool_arguments(
        "save_candidate",
        {"note": "住宿", "purpose": "hotel", "entity_id": "place:hotel"},
        tools,
    )
    assert list(save) == ["entity_id", "purpose", "note"]

    plan = order_tool_arguments(
        "submit_plan",
        {
            "plan": {
                "itinerary": [
                    {
                        "activities": [
                            {
                                "note": "入住",
                                "end_time": "24:00",
                                "candidate_id": "place:hotel",
                                "start_time": "20:00",
                                "type": "accommodation",
                                "rooms": 1,
                            }
                        ],
                        "day": 1,
                    }
                ],
                "target_city": "杭州",
                "people_number": 1,
                "start_city": "上海",
            }
        },
        tools,
    )
    ordered_plan = plan["plan"]
    assert list(ordered_plan) == ["people_number", "start_city", "target_city", "itinerary"]
    assert list(ordered_plan["itinerary"][0]) == ["day", "activities"]
    assert list(ordered_plan["itinerary"][0]["activities"][0]) == [
        "candidate_id",
        "type",
        "start_time",
        "end_time",
        "rooms",
        "note",
    ]
