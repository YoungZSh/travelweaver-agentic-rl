"""OpenAI-compatible tool-calling agent for real model rollouts."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from ..env import TravelWeaverEnv
from ..errors import ApiRolloutError
from .model_client import (
    DeepSeekConfig,
    OpenAICompatibleChatClient,
    OpenAICompatibleConfig,
)

__all__ = [
    "ApiAgentRun",
    "DeepSeekConfig",
    "DeepSeekToolAgent",
    "OpenAICompatibleConfig",
    "ToolCallingAgent",
]

TRAJECTORY_VERSION = "travelweaver-trajectory-v3"

SYSTEM_PROMPT = """\
你是 TravelWeaver 旅行规划 Agent。你只能通过提供的工具观察环境和提交答案。

规则：
1. 每轮只调用一个工具，不要凭空编造地点 ID、交通 ID 或价格。
2. 先查询证据，再用 save_candidate 保存最终计划会引用的实体。
3. 起点和目的地不同时，计划必须包含去程和返程城际交通。
4. 多日计划需要住宿；住宿必须填写 rooms 和 room_type。
5. 找到可行方案后必须调用 submit_plan。确认无解时调用 finish_without_plan。
6. 不要输出普通文本作为最终答案；episode 必须由一个终止工具结束。
7. 总动作上限是 35，目标是在 15 个动作内完成。优先选择首个满足条件的结果，不要穷举景点。
8. 每次搜索后立即保存选中的候选。API 即使允许多个 tool call，本环境每轮也只执行第一个。
9. 相邻同城地点活动之间先调用 get_route，并把返回的 route_id 写入后一个活动的
   route_from_previous_id；城际交通的起止时间必须与候选证据一致。
10. 单日异地任务只需：首个可行去程、首个可行景点、首个可行返程，然后立即 submit_plan；
    餐厅不是必需项，不要搜索。每类最多搜索一次。多日任务才额外查询并保存住宿。
"""


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
        max_api_turns: int = 40,
    ) -> None:
        if max_api_turns <= 0:
            raise ValueError("max_api_turns must be positive.")
        if client is not None and chat_client is not None:
            raise ValueError("Pass either client or chat_client, not both.")
        self.env = env
        self.config = config
        self.max_api_turns = max_api_turns
        self.chat_client = chat_client or OpenAICompatibleChatClient(config, client=client)

    def run(self, task_id: str | None = None, *, seed: int | None = 0) -> ApiAgentRun:
        observation = self.env.reset(task_id=task_id, seed=seed)
        tools = self.env.tool_schemas()
        initial_observation = observation.to_dict()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "instruction": "请根据当前任务规划行程，并用终止工具结束 episode。",
                        "observation": initial_observation,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        ]
        steps: list[dict[str, Any]] = []
        trajectory: list[dict[str, Any]] = [
            {
                "event": "reset",
                "system_prompt": SYSTEM_PROMPT,
                "observation": initial_observation,
                "tools": tools,
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
            messages.append(deepcopy(message_payload))
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
            try:
                arguments: Any = json.loads(raw_arguments)
            except (json.JSONDecodeError, TypeError):
                arguments = raw_arguments
            action = {"tool": tool_name, "arguments": arguments}
            result = self.env.step(action)
            step_count += 1
            observation = result.observation
            result_payload = result.to_dict()
            step = {
                "index": step_count - 1,
                "api_turn": api_turn,
                "tool_call": deepcopy(tool_call),
                "action": action,
                "result": result_payload,
            }
            steps.append(step)
            trajectory.append({"event": "step", **deepcopy(step)})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": json.dumps(
                        result_payload,
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


class DeepSeekToolAgent(ToolCallingAgent):
    """Backward-compatible name for the DeepSeek rollout entry point."""
