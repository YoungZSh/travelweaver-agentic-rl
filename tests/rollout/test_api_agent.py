from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from travelweaver.cli.main import build_parser
from travelweaver.llm import DeepSeekConfig, OpenAICompatibleConfig
from travelweaver.rollout import (
    MODEL_TOOL_RESPONSE_VERSION,
    ToolCallingAgent,
    append_trajectory,
)


class _Payload:
    def __init__(self, payload):
        self.payload = payload

    def model_dump(self, *, exclude_none=True):
        if not exclude_none:
            return dict(self.payload)
        return {key: value for key, value in self.payload.items() if value is not None}


class _Message(_Payload):
    def __init__(self, *tool_calls):
        super().__init__(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                    for tool_call in tool_calls
                ],
            }
        )
        self.content = None
        self.tool_calls = list(tool_calls)


class _Completions:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def create(self, **request):
        self.requests.append(request)
        return self.response


class _SequenceCompletions(_Completions):
    def __init__(self, responses):
        super().__init__(None)
        self.responses = iter(responses)

    def create(self, **request):
        self.requests.append(request)
        return next(self.responses)


def _terminal_call(call_id: str):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(
            name="submit_plan",
            arguments=json.dumps(
                {
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
                ensure_ascii=False,
            ),
        ),
    )


def test_api_agent_executes_model_tool_call_and_records_trajectory(env) -> None:
    tool_call = _terminal_call("call-1")
    message = _Message(tool_call)
    response = SimpleNamespace(
        id="response-1",
        choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
        usage=_Payload(
            {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}
        ),
    )
    completions = _Completions(response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    config = DeepSeekConfig(api_key="not-a-real-secret")

    run = ToolCallingAgent(env, config, client=client).run("task-hangzhou")

    assert not run.success
    assert run.termination_reason == "invalid_plan_submitted"
    assert run.step_count == 1
    assert run.usage["total_tokens"] == 110
    assert len(completions.requests[0]["tools"]) == 15
    assert completions.requests[0]["tool_choice"] == "auto"
    assert completions.requests[0]["messages"][1] == {
        "role": "user",
        "content": "从上海去杭州玩一天。",
    }
    system_prompt = completions.requests[0]["messages"][0]["content"]
    assert "最多执行 50 个有效工具动作" in system_prompt
    assert "按任务复杂度尽量减少无效搜索" in system_prompt
    assert "去程到达站至首个地点" in system_prompt
    assert "酒店至次日首个地点" in system_prompt
    assert "15 个动作内完成" not in system_prompt
    assert completions.requests[0]["extra_body"] == {
        "thinking": {"type": "disabled"}
    }
    assert [event["event"] for event in run.trajectory] == ["reset", "assistant", "step"]
    assert [message["role"] for message in run.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert run.messages[-1]["name"] == "submit_plan"
    terminal_payload = json.loads(run.messages[-1]["content"])
    assert terminal_payload["response_version"] == MODEL_TOOL_RESPONSE_VERSION
    assert terminal_payload["terminated"] is True
    assert "reward" not in terminal_payload
    assert "observation" not in terminal_payload
    assert run.steps[0]["tool_call"]["id"] == "call-1"
    assert run.steps[0]["model_tool_response"] == terminal_payload
    assert "observation" in run.steps[0]["result"]
    assert len(run.tools) == 15
    persisted = run.to_dict(include_trajectory=True)
    assert persisted["trajectory_version"] == "travelweaver-trajectory-v10"
    assert persisted["user_content_format"] == "travelweaver-natural-query-v1"
    assert persisted["tool_response_mode"] == "delta"
    assert persisted["model_tool_response_version"] == MODEL_TOOL_RESPONSE_VERSION
    assert -1.0 < persisted["final_reward"] < 0.0
    assert persisted["rft_accepted"] is False
    assert {"messages", "tools", "steps", "trajectory"} <= persisted.keys()
    assert "not-a-real-secret" not in repr(config)


def test_append_trajectory_writes_one_json_object_per_line(tmp_path) -> None:
    path = tmp_path / "runs.jsonl"
    append_trajectory(path, {"task_id": "one"})
    append_trajectory(path, {"task_id": "two"})

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert records == [{"task_id": "one"}, {"task_id": "two"}]


def test_api_agent_executes_only_first_tool_call_per_turn(env) -> None:
    search = SimpleNamespace(
        id="call-search",
        function=SimpleNamespace(
            name="search_attractions",
            arguments=json.dumps({"city": "杭州"}, ensure_ascii=False),
        ),
    )
    skipped = SimpleNamespace(
        id="call-skipped",
        function=SimpleNamespace(
            name="search_restaurants",
            arguments=json.dumps({"city": "杭州"}, ensure_ascii=False),
        ),
    )
    finish = _terminal_call("call-finish")

    def response(response_id, message):
        return SimpleNamespace(
            id=response_id,
            choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
            usage=_Payload({"total_tokens": 10}),
        )

    completions = _SequenceCompletions(
        [
            response("response-1", _Message(search, skipped)),
            response("response-2", _Message(finish)),
        ]
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    run = ToolCallingAgent(
        env,
        DeepSeekConfig(api_key="not-a-real-secret"),
        client=client,
    ).run("task-hangzhou")

    assert run.step_count == 2
    assert run.termination_reason == "invalid_plan_submitted"
    events = [event["event"] for event in run.trajectory]
    assert events == [
        "reset",
        "assistant",
        "tool_call_truncation",
        "step",
        "assistant",
        "step",
    ]
    truncation = run.trajectory[2]
    assert truncation["kept_tool_call_id"] == "call-search"
    assert truncation["dropped_tool_calls"][0]["id"] == "call-skipped"

    second_request_messages = completions.requests[1]["messages"]
    first_assistant = next(
        message for message in second_request_messages if message["role"] == "assistant"
    )
    assert [call["id"] for call in first_assistant["tool_calls"]] == ["call-search"]
    assert not any(
        message.get("tool_call_id") == "call-skipped"
        for message in second_request_messages
    )
    tool_message = next(
        message for message in second_request_messages if message["role"] == "tool"
    )
    delta = json.loads(tool_message["content"])
    assert delta["response_version"] == MODEL_TOOL_RESPONSE_VERSION
    assert delta["valid_action"] is True
    assert delta["tool_result"]["tool"] == "search_attractions"
    assert "observation" not in delta
    assert "candidates" not in delta
    assert "visible_entity_ids" not in delta
    assert "task" not in delta


def test_snapshot_tool_response_mode_reproduces_legacy_full_step(env) -> None:
    search = SimpleNamespace(
        id="call-search",
        function=SimpleNamespace(
            name="search_attractions",
            arguments=json.dumps({"city": "杭州"}, ensure_ascii=False),
        ),
    )
    finish = _terminal_call("call-finish")

    def response(response_id, message):
        return SimpleNamespace(
            id=response_id,
            choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
            usage=_Payload({"total_tokens": 10}),
        )

    completions = _SequenceCompletions(
        [
            response("response-1", _Message(search)),
            response("response-2", _Message(finish)),
        ]
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    run = ToolCallingAgent(
        env,
        DeepSeekConfig(api_key="not-a-real-secret"),
        client=client,
        tool_response_mode="snapshot",
    ).run("task-hangzhou")

    tool_message = next(
        message
        for message in completions.requests[1]["messages"]
        if message["role"] == "tool"
    )
    expected = json.loads(json.dumps(run.steps[0]["result"], ensure_ascii=False))
    assert json.loads(tool_message["content"]) == expected
    assert run.steps[0]["model_tool_response"] == run.steps[0]["result"]
    assert run.tool_response_mode == "snapshot"


def test_delta_tool_response_preserves_invalid_action_recovery_signal(env) -> None:
    invalid = SimpleNamespace(
        id="call-invalid",
        function=SimpleNamespace(
            name="search_attractions",
            arguments="{}",
        ),
    )
    finish = _terminal_call("call-finish")

    def response(response_id, message):
        return SimpleNamespace(
            id=response_id,
            choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
            usage=_Payload({"total_tokens": 10}),
        )

    completions = _SequenceCompletions(
        [
            response("response-1", _Message(invalid)),
            response("response-2", _Message(finish)),
        ]
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    ToolCallingAgent(
        env,
        DeepSeekConfig(api_key="not-a-real-secret"),
        client=client,
    ).run("task-hangzhou")

    tool_message = next(
        message
        for message in completions.requests[1]["messages"]
        if message["role"] == "tool"
    )
    delta = json.loads(tool_message["content"])
    assert delta["valid_action"] is False
    assert delta["consecutive_invalid_actions"] == 1
    assert delta["error"]["code"] == "invalid_action"
    assert delta["tool_result"] is None
    assert "observation" not in delta


def test_malformed_arguments_are_canonicalized_for_the_next_api_turn(env) -> None:
    malformed = SimpleNamespace(
        id="call-malformed",
        function=SimpleNamespace(
            name="search_attractions",
            arguments='{"city":"杭州"',
        ),
    )
    finish = _terminal_call("call-finish")

    def response(response_id, message):
        return SimpleNamespace(
            id=response_id,
            choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
            usage=_Payload({"total_tokens": 10}),
        )

    completions = _SequenceCompletions(
        [
            response("response-1", _Message(malformed)),
            response("response-2", _Message(finish)),
        ]
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    run = ToolCallingAgent(
        env,
        DeepSeekConfig(api_key="not-a-real-secret"),
        client=client,
    ).run("task-hangzhou")

    second_messages = completions.requests[1]["messages"]
    malformed_history = next(
        message
        for message in second_messages
        if message.get("role") == "assistant"
        and message.get("tool_calls", [{}])[0].get("id") == "call-malformed"
    )
    assert malformed_history["tool_calls"][0]["function"]["arguments"] == "{}"
    assert '{"city":"杭州"' not in json.dumps(second_messages, ensure_ascii=False)
    tool_error = next(
        json.loads(message["content"])
        for message in second_messages
        if message.get("role") == "tool"
        and message.get("tool_call_id") == "call-malformed"
    )
    assert tool_error["valid_action"] is False
    assert "required property" in tool_error["error"]["message"]
    assert run.steps[0]["action"]["arguments"] == {}
    assert run.steps[0]["tool_call"]["function"]["arguments"] == "{}"
    assert run.steps[0]["raw_tool_call"]["function"]["arguments"] == '{"city":"杭州"'
    assert any(
        event["event"] == "tool_argument_normalization"
        and event["raw_arguments"] == '{"city":"杭州"'
        for event in run.trajectory
    )


def test_api_agent_rejects_unknown_tool_response_mode(env) -> None:
    with pytest.raises(ValueError, match="Unknown tool response mode"):
        ToolCallingAgent(
            env,
            DeepSeekConfig(api_key="not-a-real-secret"),
            client=SimpleNamespace(),
            tool_response_mode="legacy",
        )


@pytest.mark.parametrize(
    "arguments",
    [
        ["rollout-api"],
        ["rollout-generated", "--input-dir", "generated", "--output", "runs.jsonl"],
        ["rollout-benchmark", "--output", "runs.jsonl"],
        [
            "rebuild-sft",
            "--source",
            "generated",
            "runs.jsonl",
            "--output-dir",
            "sft",
        ],
    ],
)
def test_rollout_cli_defaults_to_delta_tool_responses(arguments) -> None:
    parser = build_parser()

    parsed = parser.parse_args(arguments)
    assert parsed.tool_response_mode == "delta"
    if arguments[0].startswith("rollout-"):
        assert parsed.max_api_turns == 60
        if arguments[0] != "rollout-api":
            assert parsed.concurrency == 256
    if arguments[0] == "rollout-generated":
        assert parsed.limit is None
        limited = parser.parse_args([*arguments, "--limit", "100"])
        assert limited.limit == 100
    if arguments[0] == "rebuild-sft":
        assert parsed.supervision_mode == "action_only"
        react = parser.parse_args([*arguments, "--supervision-mode", "react"])
        assert react.supervision_mode == "react"
    snapshot = parser.parse_args([*arguments, "--tool-response-mode", "snapshot"])
    assert snapshot.tool_response_mode == "snapshot"


def test_generic_agent_does_not_send_provider_specific_thinking_fields(env) -> None:
    tool_call = _terminal_call("call-generic")
    response = SimpleNamespace(
        id="response-generic",
        choices=[SimpleNamespace(message=_Message(tool_call), finish_reason="tool_calls")],
        usage=None,
    )
    completions = _Completions(response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    config = OpenAICompatibleConfig(
        api_key="not-a-real-secret",
        base_url="http://localhost:8000/v1",
        model="qwen-local",
    )

    run = ToolCallingAgent(env, config, client=client).run("task-hangzhou")

    assert run.model == "qwen-local"
    assert "extra_body" not in completions.requests[0]
    assert completions.requests[0]["tool_choice"] == "auto"
