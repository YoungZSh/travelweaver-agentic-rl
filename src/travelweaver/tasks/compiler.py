"""LLM semantic compilation followed by deterministic TaskSpec validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from ..errors import TaskSpecError
from .models import (
    ConstraintSpec,
    TravelTaskSpec,
    TripSpec,
    stable_hash,
    supported_constraint_kinds,
)

COMPILER_VERSION = "travelweaver-llm-task-compiler-v1"


class ChatClient(Protocol):
    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Any: ...


@dataclass(frozen=True)
class CompileResult:
    status: str
    spec: TravelTaskSpec | None
    errors: tuple[str, ...]
    attempts: int

    @property
    def accepted(self) -> bool:
        return self.status == "accepted" and self.spec is not None


def _trip_from_task(task: dict[str, Any]) -> TripSpec:
    try:
        return TripSpec(
            origin=str(task["start_city"]),
            destinations=(str(task["target_city"]),),
            days=int(task["days"]),
            travelers=int(task["people_number"]),
            start_date=str(task["start_date"]) if task.get("start_date") else None,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise TaskSpecError(f"Public task metadata is invalid: {error}") from error


def build_base_spec(
    task: dict[str, Any],
    *,
    constraints: tuple[ConstraintSpec, ...] = (),
    unscored_preferences: tuple[str, ...] = (),
    source: str = "public_task",
    compiler_version: str = "travelweaver-base-task-builder-v1",
    world_snapshot_version: str = "chinatravel-pinned-v1",
) -> TravelTaskSpec:
    query = str(task.get("query") or "").strip()
    if not query:
        raise TaskSpecError("Public task query is empty.")
    input_material = {
        "uid": task.get("uid"),
        "query": query,
        "trip": {
            "start_city": task.get("start_city"),
            "target_city": task.get("target_city"),
            "days": task.get("days"),
            "people_number": task.get("people_number"),
            "start_date": task.get("start_date"),
        },
    }
    return TravelTaskSpec(
        task_id=str(task.get("uid") or stable_hash(input_material)[:20]),
        public_query=query,
        trip=_trip_from_task(task),
        constraints=constraints,
        unscored_preferences=unscored_preferences,
        source=source,
        compiler_version=compiler_version,
        input_hash=stable_hash(input_material),
        world_snapshot_version=world_snapshot_version,
    )


def task_spec_tool_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "emit_travel_task_spec",
            "description": (
                "Extract only explicit, objectively checkable travel constraints. "
                "Do not repeat origin, destination, days, or travelers from metadata."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "constraints": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "kind": {
                                    "type": "string",
                                    "enum": list(supported_constraint_kinds()),
                                },
                                "operator": {
                                    "type": "string",
                                    "enum": [
                                        "contains",
                                        "eq",
                                        "exclude",
                                        "gte",
                                        "include",
                                        "lte",
                                        "not_contains",
                                        "not_in",
                                    ],
                                },
                                "value": {},
                                "scope": {
                                    "type": "string",
                                    "enum": [
                                        "accommodation",
                                        "attraction",
                                        "day",
                                        "innercity_route",
                                        "intercity_transport",
                                        "itinerary",
                                        "restaurant",
                                        "trip",
                                    ],
                                },
                                "hardness": {"type": "string", "enum": ["hard", "soft"]},
                                "source_text": {"type": "string", "minLength": 1},
                            },
                            "required": [
                                "kind",
                                "operator",
                                "value",
                                "scope",
                                "hardness",
                                "source_text",
                            ],
                            "additionalProperties": False,
                        },
                    },
                    "unscored_preferences": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                    },
                },
                "required": ["constraints", "unscored_preferences"],
                "additionalProperties": False,
            },
        },
    }


class LLMTaskSpecCompiler:
    """Compile free text through one forced function call, then fail closed."""

    def __init__(
        self,
        chat_client: ChatClient,
        *,
        max_attempts: int = 2,
        world_snapshot_version: str = "chinatravel-pinned-v1",
    ) -> None:
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive.")
        self.chat_client = chat_client
        self.max_attempts = max_attempts
        self.world_snapshot_version = world_snapshot_version

    def compile(self, task: dict[str, Any]) -> CompileResult:
        errors: list[str] = []
        try:
            _trip_from_task(task)
        except TaskSpecError as error:
            return CompileResult("quarantined", None, (str(error),), 0)

        for attempt in range(1, self.max_attempts + 1):
            try:
                payload = self._request(task)
                spec = self._materialize(task, payload)
            except (TaskSpecError, TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
                errors.append(str(error))
                continue
            return CompileResult("accepted", spec, (), attempt)
        return CompileResult("quarantined", None, tuple(errors), self.max_attempts)

    def _request(self, task: dict[str, Any]) -> dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": (
                    "你是旅行任务约束编译器。只提取用户明确表达且 TravelEnv 能客观验证的"
                    "要求。必须保留原文中的连续 source_text。不要把主观描述强行变成硬约束；"
                    "无法客观评分的描述写入 unscored_preferences。必须调用唯一提供的函数。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "query": task.get("query"),
                        "authoritative_trip_metadata": {
                            "origin": task.get("start_city"),
                            "destination": task.get("target_city"),
                            "days": task.get("days"),
                            "travelers": task.get("people_number"),
                        },
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        response = self.chat_client.complete(messages, [task_spec_tool_schema()])
        choices = getattr(response, "choices", None)
        if not choices:
            raise TaskSpecError("Task compiler returned no choices.")
        message = choices[0].message
        model_dump = getattr(message, "model_dump", None)
        raw = model_dump(exclude_none=True) if callable(model_dump) else message
        if not isinstance(raw, dict):
            raise TaskSpecError("Task compiler returned an unsupported message payload.")
        calls = raw.get("tool_calls") or []
        if len(calls) != 1:
            raise TaskSpecError("Task compiler must return exactly one function call.")
        function = calls[0].get("function") or {}
        if function.get("name") != "emit_travel_task_spec":
            raise TaskSpecError("Task compiler called an unexpected function.")
        arguments = function.get("arguments")
        payload = json.loads(arguments) if isinstance(arguments, str) else arguments
        if not isinstance(payload, dict):
            raise TaskSpecError("Task compiler arguments must be an object.")
        return payload

    def _materialize(self, task: dict[str, Any], payload: dict[str, Any]) -> TravelTaskSpec:
        if set(payload) != {"constraints", "unscored_preferences"}:
            raise TaskSpecError("Compiler payload contains missing or unexpected fields.")
        raw_constraints = payload["constraints"]
        raw_preferences = payload["unscored_preferences"]
        if not isinstance(raw_constraints, list) or not isinstance(raw_preferences, list):
            raise TaskSpecError("Compiler constraints and preferences must be arrays.")
        query = str(task["query"])
        constraints: list[ConstraintSpec] = []
        for index, raw in enumerate(raw_constraints, 1):
            if not isinstance(raw, dict):
                raise TaskSpecError("Each compiled constraint must be an object.")
            source_text = str(raw.get("source_text") or "")
            source_start = query.find(source_text)
            if source_start < 0:
                raise TaskSpecError(
                    f"Constraint {index} source_text is not an exact query substring."
                )
            constraints.append(
                ConstraintSpec(
                    id=f"c{index:03d}",
                    kind=str(raw["kind"]),
                    operator=str(raw["operator"]),
                    value=raw["value"],
                    scope=str(raw["scope"]),
                    hardness=str(raw["hardness"]),
                    source_text=source_text,
                    source_start=source_start,
                    source_end=source_start + len(source_text),
                )
            )
        preferences = tuple(str(value).strip() for value in raw_preferences if str(value).strip())
        if not constraints and not preferences:
            # A task with authoritative origin/destination/day/person metadata is still scoreable.
            preferences = ()
        return build_base_spec(
            task,
            constraints=tuple(constraints),
            unscored_preferences=preferences,
            source="llm_compiled_free_text",
            compiler_version=COMPILER_VERSION,
            world_snapshot_version=self.world_snapshot_version,
        )
