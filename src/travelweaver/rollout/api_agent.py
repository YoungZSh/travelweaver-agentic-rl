"""OpenAI-compatible tool-calling agent for real model rollouts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from ..env import DEFAULT_MAX_VALID_STEPS, TravelWeaverEnv
from ..errors import ApiRolloutError
from ..llm import (
    OpenAICompatibleChatClient,
    OpenAICompatibleConfig,
)
from .tool_response import (
    DEFAULT_TOOL_RESPONSE_MODE,
    MODEL_TOOL_RESPONSE_VERSION,
    ToolResponseMode,
    serialize_model_tool_response,
    validate_tool_response_mode,
)

__all__ = [
    "ApiAgentRun",
    "DEFAULT_MAX_API_TURNS",
    "OpenAICompatibleConfig",
    "ToolCallingAgent",
    "render_system_prompt",
    "render_task_user_content",
]

TRAJECTORY_VERSION = "travelweaver-trajectory-v6"
SUPPORTED_TRAJECTORY_VERSIONS = frozenset(
    {
        "travelweaver-trajectory-v3",
        "travelweaver-trajectory-v4",
        "travelweaver-trajectory-v5",
        TRAJECTORY_VERSION,
    }
)
USER_CONTENT_FORMAT = "travelweaver-natural-query-v1"
DEFAULT_MAX_API_TURNS = 60

_SYSTEM_PROMPT_TEMPLATE = """\
你是 TravelWeaver 旅行规划 Agent。你只能通过提供的工具观察环境和提交答案。

规则：
1. 每轮只调用一个工具，不得编造地点 ID、交通 ID、路线 ID、价格或时间。
2. 先查询证据。只有确定候选可能进入最终计划时，才使用 save_candidate 保存；
   不要批量保存暂不使用的结果。
3. 起点和目的地不同时，计划必须包含去程和返程城际交通。
4. 多日行程应在第 1 天至倒数第 2 天各安排一次当晚住宿，最后一天不要重复安排住宿。
   住宿必须填写 rooms 和 room_type。
5. 每个完整的中间旅行日应安排至少一个景点或用餐活动，避免出现只有住宿的空白日期。
6. 每天第一个本地活动不得填写 route_from_previous_id。只有同一天相邻的两个本地活动之间
   才调用 get_route；路线出发时间不得早于前一个活动结束时间，且必须在后一个活动开始前到达。
7. 城际交通的起止时间必须与候选证据完全一致。
8. 先满足全部硬约束。存在多个可行方案时，再根据题面偏好比较少量候选；无需穷举。
9. 最多执行 {max_valid_steps} 个有效工具动作。按任务复杂度尽量减少无效搜索、无用保存和未使用路线。
10. 找到可行方案后调用 submit_plan；确认无解时调用 finish_without_plan。
    不要输出普通文本作为最终答案。
11. 对没有额外景点、餐饮或指定地点要求的单日异地任务，选择首个可行的去程、景点和返程后即可提交；
    若题面有额外要求，必须继续查询并落实。
12. API 即使允许多个 tool call，本环境每轮也只执行第一个。
"""


def render_system_prompt(max_valid_steps: int) -> str:
    """Render the model instruction with the environment's actual action budget."""

    if max_valid_steps <= 0:
        raise ValueError("max_valid_steps must be positive.")
    return _SYSTEM_PROMPT_TEMPLATE.format(max_valid_steps=max_valid_steps)


SYSTEM_PROMPT = render_system_prompt(DEFAULT_MAX_VALID_STEPS)


def render_task_user_content(task: Mapping[str, Any]) -> str:
    """Return the natural-language utterance supplied at the human task boundary."""

    query = task.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Travel task query must be a non-empty natural-language string.")
    return query.strip()


@dataclass(frozen=True)
class ApiAgentRun:
    """One model-driven episode and its replayable event trace."""

    episode_id: str
    task_id: str
    model: str
    success: bool
    termination_reason: str
    step_count: int
    api_turn_count: int
    final_plan: dict[str, Any] | None
    final_text: str | None
    final_reward: float
    reward_detail: dict[str, Any]
    rft_accepted: bool
    usage: dict[str, int]
    user_content_format: str
    tool_response_mode: ToolResponseMode
    messages: tuple[dict[str, Any], ...]
    tools: tuple[dict[str, Any], ...]
    steps: tuple[dict[str, Any], ...]
    trajectory: tuple[dict[str, Any], ...]

    def to_dict(self, *, include_trajectory: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "trajectory_version": TRAJECTORY_VERSION,
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "model": self.model,
            "success": self.success,
            "termination_reason": self.termination_reason,
            "step_count": self.step_count,
            "api_turn_count": self.api_turn_count,
            "final_plan": self.final_plan,
            "final_text": self.final_text,
            "final_reward": self.final_reward,
            "reward_detail": dict(self.reward_detail),
            "rft_accepted": self.rft_accepted,
            "usage": dict(self.usage),
            "user_content_format": self.user_content_format,
            "tool_response_mode": self.tool_response_mode,
            "model_tool_response_version": MODEL_TOOL_RESPONSE_VERSION,
        }
        if include_trajectory:
            payload["messages"] = list(self.messages)
            payload["tools"] = list(self.tools)
            payload["steps"] = list(self.steps)
            payload["trajectory"] = list(self.trajectory)
        return payload


class ToolCallingAgent:
    """Drive TravelWeaverEnv with any OpenAI-compatible function-calling model."""

    def __init__(
        self,
        env: TravelWeaverEnv,
        config: OpenAICompatibleConfig,
        *,
        client: Any | None = None,
        chat_client: Any | None = None,
        max_api_turns: int = DEFAULT_MAX_API_TURNS,
        tool_response_mode: str = DEFAULT_TOOL_RESPONSE_MODE,
    ) -> None:
        if max_api_turns <= 0:
            raise ValueError("max_api_turns must be positive.")
        if client is not None and chat_client is not None:
            raise ValueError("Pass either client or chat_client, not both.")
        self.env = env
        self.config = config
        self.max_api_turns = max_api_turns
        self.tool_response_mode = validate_tool_response_mode(tool_response_mode)
        self.chat_client = chat_client or OpenAICompatibleChatClient(config, client=client)

    def run(self, task_id: str | None = None, *, seed: int | None = 0) -> ApiAgentRun:
        observation = self.env.reset(task_id=task_id, seed=seed)
        tools = self.env.tool_schemas()
        initial_observation = observation.to_dict()
        user_content = render_task_user_content(initial_observation["task"])
        system_prompt = render_system_prompt(self.env.max_valid_steps)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        steps: list[dict[str, Any]] = []
        trajectory: list[dict[str, Any]] = [
            {
                "event": "reset",
                "system_prompt": system_prompt,
                "observation": initial_observation,
                "tools": tools,
                "user_content_format": USER_CONTENT_FORMAT,
                "tool_response_mode": self.tool_response_mode,
                "model_tool_response_version": MODEL_TOOL_RESPONSE_VERSION,
            }
        ]
        usage_total: dict[str, int] = {}
        step_count = 0

        for api_turn in range(1, self.max_api_turns + 1):
            response = self._request(messages, tools)
            if not response.choices:
                raise ApiRolloutError("Model API returned no completion choices.")
            choice = response.choices[0]
            message = choice.message
            message_payload = self._model_dump(message)
            raw_tool_calls = message_payload.get("tool_calls") or []
            if not isinstance(raw_tool_calls, list):
                raise ApiRolloutError("Assistant tool_calls must be a list.")
            dropped_tool_calls = deepcopy(raw_tool_calls[1:])
            if dropped_tool_calls:
                # The canonical conversation is serial: later model calls must not see
                # tool-call IDs that were deliberately not executed.
                message_payload = deepcopy(message_payload)
                message_payload["tool_calls"] = [deepcopy(raw_tool_calls[0])]
            usage = self._model_dump(response.usage) if response.usage is not None else {}
            self._merge_usage(usage_total, usage)
            trajectory.append(
                {
                    "event": "assistant",
                    "api_turn": api_turn,
                    "response_id": getattr(response, "id", None),
                    "finish_reason": choice.finish_reason,
                    "message": message_payload,
                    "usage": usage,
                }
            )
            if dropped_tool_calls:
                trajectory.append(
                    {
                        "event": "tool_call_truncation",
                        "api_turn": api_turn,
                        "kept_tool_call_id": raw_tool_calls[0].get("id"),
                        "dropped_tool_calls": dropped_tool_calls,
                    }
                )

            tool_calls = message_payload.get("tool_calls") or []
            if not tool_calls:
                messages.append(deepcopy(message_payload))
                return self._result(
                    observation=observation,
                    messages=messages,
                    tools=tools,
                    steps=steps,
                    trajectory=trajectory,
                    usage=usage_total,
                    step_count=step_count,
                    api_turn_count=api_turn,
                    termination_reason="model_stopped_without_terminal_action",
                    final_text=message_payload.get("content"),
                )

            tool_call = tool_calls[0]
            if not isinstance(tool_call, dict):
                raise ApiRolloutError("Assistant tool call must be an object.")
            function = tool_call.get("function") or {}
            if not isinstance(function, dict):
                raise ApiRolloutError("Assistant tool call function must be an object.")
            tool_name = function.get("name")
            tool_call_id = tool_call.get("id")
            if not isinstance(tool_name, str) or not tool_name:
                raise ApiRolloutError("Assistant tool call is missing function.name.")
            if not isinstance(tool_call_id, str) or not tool_call_id:
                raise ApiRolloutError("Assistant tool call is missing id.")
            raw_arguments = function.get("arguments")
            argument_normalization: dict[str, Any] | None = None
            if isinstance(raw_arguments, dict):
                arguments: Any = deepcopy(raw_arguments)
            else:
                try:
                    arguments = json.loads(raw_arguments)
                except (json.JSONDecodeError, TypeError) as error:
                    arguments = {}
                    argument_normalization = {
                        "reason": type(error).__name__,
                        "raw_arguments": raw_arguments,
                    }
            if not isinstance(arguments, dict):
                argument_normalization = {
                    "reason": "arguments_not_object",
                    "raw_arguments": raw_arguments,
                }
                arguments = {}
            canonical_message = deepcopy(message_payload)
            canonical_tool_call = canonical_message["tool_calls"][0]
            if argument_normalization is not None:
                canonical_tool_call["function"]["arguments"] = "{}"
                trajectory.append(
                    {
                        "event": "tool_argument_normalization",
                        "api_turn": api_turn,
                        "tool_call_id": tool_call_id,
                        **deepcopy(argument_normalization),
                    }
                )
            messages.append(canonical_message)
            action = {"tool": tool_name, "arguments": arguments}
            result = self.env.step(action)
            step_count += 1
            observation = result.observation
            result_payload = result.to_dict()
            model_tool_response = serialize_model_tool_response(
                result,
                mode=self.tool_response_mode,
            )
            step = {
                "index": step_count - 1,
                "api_turn": api_turn,
                "tool_call": deepcopy(canonical_tool_call),
                "action": action,
                "result": result_payload,
                "model_tool_response": deepcopy(model_tool_response),
            }
            if argument_normalization is not None:
                step["raw_tool_call"] = deepcopy(tool_call)
                step["argument_normalization"] = deepcopy(argument_normalization)
            steps.append(step)
            trajectory.append({"event": "step", **deepcopy(step)})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": json.dumps(
                        model_tool_response,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                }
            )

            if result.terminated or result.truncated:
                terminal_reason = str(
                    result.info.get("termination_reason")
                    or ("truncated" if result.truncated else "terminated")
                )
                tool_result = result.observation.tool_result or {}
                plan = tool_result.get("plan") if isinstance(tool_result, dict) else None
                return self._result(
                    observation=observation,
                    messages=messages,
                    tools=tools,
                    steps=steps,
                    trajectory=trajectory,
                    usage=usage_total,
                    step_count=step_count,
                    api_turn_count=api_turn,
                    termination_reason=terminal_reason,
                    final_plan=plan if isinstance(plan, dict) else None,
                    terminal_result=result,
                )

        return self._result(
            observation=observation,
            messages=messages,
            tools=tools,
            steps=steps,
            trajectory=trajectory,
            usage=usage_total,
            step_count=step_count,
            api_turn_count=self.max_api_turns,
            termination_reason="api_turn_limit",
        )

    def _request(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Any:
        return self.chat_client.complete(messages, tools)

    def _result(
        self,
        *,
        observation: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        steps: list[dict[str, Any]],
        trajectory: list[dict[str, Any]],
        usage: dict[str, int],
        step_count: int,
        api_turn_count: int,
        termination_reason: str,
        final_plan: dict[str, Any] | None = None,
        final_text: str | None = None,
        terminal_result: Any | None = None,
    ) -> ApiAgentRun:
        if terminal_result is None:
            reward_result = self.env.reward_evaluator.no_plan(termination_reason)
            final_reward = reward_result.reward
            reward_detail = reward_result.to_dict()
        else:
            final_reward = float(terminal_result.reward)
            reward_detail = dict(terminal_result.info.get("reward_detail") or {})
        rft_accepted = bool(
            termination_reason == "plan_submitted"
            and reward_detail.get("reward_valid")
            and reward_detail.get("all_hard_pass")
        )
        return ApiAgentRun(
            episode_id=str(observation.episode_id),
            task_id=str(observation.task["uid"]),
            model=self.config.model,
            success=rft_accepted,
            termination_reason=termination_reason,
            step_count=step_count,
            api_turn_count=api_turn_count,
            final_plan=final_plan,
            final_text=final_text,
            final_reward=final_reward,
            reward_detail=reward_detail,
            rft_accepted=rft_accepted,
            usage=usage,
            user_content_format=USER_CONTENT_FORMAT,
            tool_response_mode=self.tool_response_mode,
            messages=tuple(deepcopy(messages)),
            tools=tuple(deepcopy(tools)),
            steps=tuple(deepcopy(steps)),
            trajectory=tuple(deepcopy(trajectory)),
        )

    @staticmethod
    def _model_dump(value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return dict(value)
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            return dict(model_dump(exclude_none=True))
        raise ApiRolloutError(f"API returned an unsupported payload: {type(value).__name__}")

    @staticmethod
    def _merge_usage(total: dict[str, int], usage: dict[str, Any]) -> None:
        for key, value in usage.items():
            if isinstance(value, int) and not isinstance(value, bool):
                total[key] = total.get(key, 0) + value
