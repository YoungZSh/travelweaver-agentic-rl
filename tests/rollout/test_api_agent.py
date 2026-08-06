from __future__ import annotations

import json
from types import SimpleNamespace

from travelweaver.rollout import (
    DeepSeekConfig,
    DeepSeekToolAgent,
    OpenAICompatibleConfig,
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


def test_api_agent_executes_model_tool_call_and_records_trajectory(env) -> None:
    tool_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(
            name="finish_without_plan",
            arguments=json.dumps({"reason": "测试终止"}, ensure_ascii=False),
        ),
    )
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

    run = DeepSeekToolAgent(env, config, client=client).run("task-hangzhou")

    assert not run.success
    assert run.termination_reason == "finished_without_plan"
    assert run.step_count == 1
    assert run.usage["total_tokens"] == 110
    assert len(completions.requests[0]["tools"]) == 13
    assert completions.requests[0]["tool_choice"] == "auto"
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
    assert run.messages[-1]["name"] == "finish_without_plan"
    assert run.steps[0]["tool_call"]["id"] == "call-1"
    assert len(run.tools) == 13
    persisted = run.to_dict(include_trajectory=True)
    assert persisted["trajectory_version"] == "travelweaver-trajectory-v2"
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
    finish = SimpleNamespace(
        id="call-finish",
        function=SimpleNamespace(
            name="finish_without_plan",
            arguments=json.dumps({"reason": "测试结束"}, ensure_ascii=False),
        ),
    )

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

    run = DeepSeekToolAgent(
        env,
        DeepSeekConfig(api_key="not-a-real-secret"),
        client=client,
    ).run("task-hangzhou")

    assert run.step_count == 2
    assert run.termination_reason == "finished_without_plan"
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


def test_generic_agent_does_not_send_provider_specific_thinking_fields(env) -> None:
    tool_call = SimpleNamespace(
        id="call-generic",
        function=SimpleNamespace(
            name="finish_without_plan",
            arguments=json.dumps({"reason": "generic endpoint"}),
        ),
    )
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
