"""veRL multi-turn agent loop backed by one deterministic TravelWeaver episode."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from travelweaver.data import JsonlTaskStore
from travelweaver.env import ChinaTravelBackend, ScenarioBackend, ScenarioSpec, TravelWeaverEnv
from travelweaver.rollout.api_agent import render_system_prompt, render_task_user_content
from travelweaver.rollout.tool_response import serialize_model_tool_response
from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from verl.experimental.agent_loop.tool_agent_loop import (
    AgentData,
    AgentState,
    FunctionCall,
    ToolAgentLoop,
)
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse


class _RawToolSchema:
    """Preserve TravelWeaver's nested JSON schemas despite veRL's narrow schema model."""

    def __init__(self, payload: dict[str, Any]):
        self._payload = payload
        parsed = OpenAIFunctionToolSchema.model_validate(payload)
        self.type = parsed.type
        self.function = parsed.function

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return self._payload


class _ToolProxy:
    def __init__(self, payload: dict[str, Any]):
        self.name = str(payload["function"]["name"])
        self.tool_schema = _RawToolSchema(payload)


@lru_cache(maxsize=8)
def _task_store(task_dir: str) -> JsonlTaskStore:
    directory = Path(task_dir)
    return JsonlTaskStore(directory / "tasks.public.jsonl", directory / "tasks.oracle.jsonl")


@lru_cache(maxsize=1)
def _base_backend() -> ChinaTravelBackend:
    return ChinaTravelBackend()


class TravelWeaverAgentLoop(ToolAgentLoop):
    """Run one complete function-calling trajectory and attach Reward v4."""

    def __init__(self, *args: Any, **kwargs: Any):
        self.trajectory_max_tokens = int(kwargs.pop("trajectory_max_tokens", 32768))
        if self.trajectory_max_tokens <= 0:
            raise ValueError("trajectory_max_tokens must be positive.")
        super().__init__(*args, **kwargs)
        schemas = TravelWeaverEnv.tool_schemas()
        proxies = [_ToolProxy(schema) for schema in schemas]
        self.tools = {tool.name: tool for tool in proxies}
        self.tool_schemas = [schema for schema in schemas]
        self._env: TravelWeaverEnv | None = None
        self._terminal_result: Any | None = None
        self._task_id: str | None = None
        self._audit_steps: list[dict[str, Any]] = []
        self._trajectory_length_exceeded = False

    def _mark_trajectory_length_exceeded(self) -> None:
        self._trajectory_length_exceeded = True

    async def _handle_pending_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any]
    ) -> AgentState:
        state = await super()._handle_pending_state(agent_data, sampling_params)
        remaining = trajectory_response_budget(
            initial_prompt_tokens=len(agent_data.prompt_ids),
            configured_response_tokens=self.response_length,
            trajectory_max_tokens=self.trajectory_max_tokens,
        )
        if remaining <= 0:
            self._mark_trajectory_length_exceeded()
            return AgentState.TERMINATED
        self.response_length = remaining
        return state

    async def _handle_generating_state(
        self,
        agent_data: AgentData,
        sampling_params: dict[str, Any],
        ignore_termination: bool = False,
    ) -> AgentState:
        remaining = self.response_length - len(agent_data.response_mask)
        if remaining <= 0:
            self._mark_trajectory_length_exceeded()
            return AgentState.TERMINATED

        per_turn_sampling_params = dict(sampling_params)
        configured_limit = per_turn_sampling_params.pop("max_new_tokens", None)
        if configured_limit is None:
            configured_limit = per_turn_sampling_params.get("max_tokens")
        per_turn_sampling_params["max_tokens"] = min(
            remaining,
            int(configured_limit) if configured_limit is not None else remaining,
        )
        state = await super()._handle_generating_state(
            agent_data,
            per_turn_sampling_params,
            ignore_termination=ignore_termination,
        )
        if (
            not ignore_termination
            and self._terminal_result is None
            and len(agent_data.response_mask) >= self.response_length
        ):
            self._mark_trajectory_length_exceeded()
        return state

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> AgentLoopOutput:
        task_id = str(kwargs["task_id"])
        task_dir = str(kwargs["task_dir"])
        store = _task_store(task_dir)
        oracle = store.get_oracle(task_id)
        raw_scenario = oracle.get("scenario")
        if not isinstance(raw_scenario, dict):
            raise ValueError(f"TravelWeaver GRPO task {task_id} has no frozen Scenario.")
        scenario = ScenarioSpec.from_dict(raw_scenario)
        self._env = TravelWeaverEnv(ScenarioBackend(_base_backend(), scenario), store)
        self._env.reset(task_id=task_id, seed=0)
        self._task_id = task_id
        try:
            output = await super().run(sampling_params, **kwargs)
            sequence_tokens = len(output.prompt_ids) + len(output.response_ids)
            if sequence_tokens > self.trajectory_max_tokens:
                self._mark_trajectory_length_exceeded()
            length_info = {
                "trajectory_valid": float(not self._trajectory_length_exceeded),
                "trajectory_length_exceeded": float(self._trajectory_length_exceeded),
                "sequence_tokens": float(sequence_tokens),
                "trajectory_max_tokens": float(self.trajectory_max_tokens),
            }
            if self._trajectory_length_exceeded:
                output.reward_score = 0.0
                output.extra_fields["reward_extra_info"] = {
                    "travelweaver_reward": 0.0,
                    "reward_valid": 0.0,
                    "all_hard_pass": 0.0,
                    "artifact_score": 0.0,
                    "validity_score": 0.0,
                    "goal_score": 0.0,
                    **length_info,
                }
                output.extra_fields["travelweaver_audit"] = {
                    "task_id": task_id,
                    "termination_reason": "trajectory_length_exceeded",
                    "discard_reason": "trajectory_length_exceeded",
                    "reward_detail": None,
                    "steps": self._audit_steps,
                    "sequence_tokens": sequence_tokens,
                    "trajectory_max_tokens": self.trajectory_max_tokens,
                }
                return output
            if self._terminal_result is None:
                reason = "model_stopped_without_terminal_action"
                reward_result = self._env.reward_evaluator.no_plan(reason)
            else:
                reason = str(
                    self._terminal_result.info.get("termination_reason") or "terminated"
                )
                reward_result = self._terminal_result.info.get("reward_detail")
                if not isinstance(reward_result, dict):
                    raise RuntimeError("Terminal TravelWeaver step has no Reward detail.")
            detail = (
                reward_result.to_dict()
                if hasattr(reward_result, "to_dict")
                else dict(reward_result)
            )
            reward = float(detail["reward"])
            dimensions = detail.get("dimension_scores", {})
            output.reward_score = reward
            output.extra_fields["reward_extra_info"] = {
                "travelweaver_reward": reward,
                "reward_valid": float(detail.get("reward_valid") is True),
                "all_hard_pass": float(detail.get("all_hard_pass") is True),
                "artifact_score": float(dimensions.get("artifact_conformance", 0.0)),
                "validity_score": float(dimensions.get("environment_validity", 0.0)),
                "goal_score": float(dimensions.get("goal_satisfaction", 0.0)),
                **length_info,
            }
            output.extra_fields["travelweaver_audit"] = {
                "task_id": task_id,
                "termination_reason": reason,
                "reward_detail": detail,
                "steps": self._audit_steps,
            }
            return output
        finally:
            self._env.close()

    async def _call_tool(
        self, tool_call: FunctionCall, tools_kwargs: dict[str, Any], agent_data: AgentData
    ) -> tuple[ToolResponse, float, dict[str, Any]]:
        del tools_kwargs, agent_data
        assert self._env is not None
        raw_arguments: Any = tool_call.arguments
        normalization_error: str | None = None
        try:
            arguments = json.loads(raw_arguments)
        except (json.JSONDecodeError, TypeError) as error:
            arguments = {}
            normalization_error = type(error).__name__
        if not isinstance(arguments, dict):
            arguments = {}
            normalization_error = "arguments_not_object"
        action = {"tool": tool_call.name, "arguments": arguments}
        result = self._env.step(action)
        self._audit_steps.append(
            {
                "index": len(self._audit_steps),
                "action": action,
                "raw_arguments": (
                    raw_arguments
                    if normalization_error is not None and isinstance(raw_arguments, str)
                    else repr(raw_arguments)
                    if normalization_error is not None
                    else None
                ),
                "argument_normalization_error": normalization_error,
                "result": result.to_dict(),
            }
        )
        if result.terminated or result.truncated:
            self._terminal_result = result
        payload = serialize_model_tool_response(result, mode="delta")
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return ToolResponse(text=text), 0.0, {}

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        state = await super()._handle_processing_tools_state(agent_data)
        if self._terminal_result is not None:
            return AgentState.TERMINATED
        if state == AgentState.TERMINATED:
            self._mark_trajectory_length_exceeded()
        return state


def trajectory_response_budget(
    *,
    initial_prompt_tokens: int,
    configured_response_tokens: int,
    trajectory_max_tokens: int,
) -> int:
    """Return the response-side budget under a strict total-sequence cap."""

    if min(initial_prompt_tokens, configured_response_tokens, trajectory_max_tokens) < 0:
        raise ValueError("Trajectory token counts must be non-negative.")
    return min(configured_response_tokens, trajectory_max_tokens - initial_prompt_tokens)


def build_prompt(task: dict[str, Any], *, max_valid_steps: int = 50) -> list[dict[str, str]]:
    """Build the exact production prompt used by online GRPO."""

    return [
        {"role": "system", "content": render_system_prompt(max_valid_steps)},
        {"role": "user", "content": render_task_user_content(task)},
    ]
