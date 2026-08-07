"""LLM surface realization with strict rule-based semantic preservation checks."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from typing import Any

from ..errors import SynthesisError
from ..llm import DeepSeekConfig, OpenAICompatibleChatClient
from ..tasks import ConstraintMention, TaskBlueprint, TaskSurface
from .models import PROMPT_VERSION, CanonicalTask

_SUPPORTED_CITIES = ("上海", "北京", "南京", "广州", "成都", "杭州", "武汉", "深圳", "苏州", "重庆")
_NUMBER = re.compile(r"(?<![A-Za-z])(?:\d{1,2}:\d{2}|\d+(?:\.\d+)?)")
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "polish_travel_query",
            "description": "返回仅经语言润色且保持所有硬约束的中文旅行需求。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "mentions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "constraint_id": {"type": "string"},
                                "text": {"type": "string"},
                            },
                            "required": ["constraint_id", "text"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["query", "mentions"],
                "additionalProperties": False,
            },
        },
    }
]
_SYSTEM_PROMPT = """你是中文旅行任务改写器。你的唯一职责是让输入更自然，不得改变语义。
必须通过 polish_travel_query 函数返回结果，并遵守：
1. 保留起点、终点、天数、人数及每条硬约束，不新增偏好、地点、数字或限制。
2. protected_literals 中每一项必须逐字出现在 query 中。
3. mentions 必须覆盖每个 constraint_id 一次；
   text 必须是 query 中唯一、连续、能独立表达该约束的原文片段。
4. 不输出解释、Markdown、JSON 文本或思考过程，只调用函数。
5. 可以调整语序和连接词，但不得把“不超过/之前”改成下限，也不得把“至少/之后”改成上限。
"""
POLISHER_PROMPT_HASH = hashlib.sha256(_SYSTEM_PROMPT.encode("utf-8")).hexdigest()


class TaskPolisher:
    def __init__(
        self,
        config: DeepSeekConfig,
        *,
        client: Any | None = None,
        max_attempts: int = 3,
    ) -> None:
        if max_attempts <= 0:
            raise ValueError("Polisher max_attempts must be positive.")
        self.config = replace(
            config,
            thinking="disabled",
            tool_choice="required",
            max_tokens=min(config.max_tokens, 1024),
            temperature=0.7,
        )
        self.client = client or OpenAICompatibleChatClient(self.config)
        self.max_attempts = max_attempts
        self.api_calls = 0

    def polish(self, blueprint: TaskBlueprint, canonical: CanonicalTask) -> TaskSurface:
        errors: list[str] = []
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for attempt in range(1, self.max_attempts + 1):
            messages = self._messages(blueprint, canonical, errors)
            self.api_calls += 1
            try:
                response = self.client.complete(messages, _TOOLS)
                payload = _tool_payload(response)
                _add_usage(usage, response)
                return validate_surface(
                    blueprint,
                    canonical,
                    payload,
                    model=self.config.model,
                    usage=usage,
                )
            except Exception as error:
                errors.append(f"attempt {attempt}: {type(error).__name__}: {error}")
        raise SynthesisError("LLM polishing failed validation: " + " | ".join(errors))

    @staticmethod
    def _messages(
        blueprint: TaskBlueprint,
        canonical: CanonicalTask,
        prior_errors: list[str],
    ) -> list[dict[str, str]]:
        payload = {
            "canonical_query": canonical.query,
            "constraint_clauses": canonical.clauses,
            "protected_literals": list(canonical.protected_literals),
            "constraint_semantics": [
                {
                    "constraint_id": item.id,
                    "kind": item.kind,
                    "operator": item.operator,
                    "scope": item.scope,
                    "value": item.value,
                }
                for item in blueprint.constraints
            ],
            "previous_validation_errors": prior_errors[-1:],
        }
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, sort_keys=True),
            },
        ]


def validate_surface(
    blueprint: TaskBlueprint,
    canonical: CanonicalTask,
    payload: dict[str, Any],
    *,
    model: str,
    usage: dict[str, int] | None = None,
) -> TaskSurface:
    query = payload.get("query")
    mention_values = payload.get("mentions")
    if not isinstance(query, str) or not 30 <= len(query) <= 500:
        raise SynthesisError("Polished query must be a 30-500 character string.")
    if any(token in query for token in ("```", "{", "}", "constraint_id")):
        raise SynthesisError("Polished query contains serialization or Markdown artifacts.")
    if not isinstance(mention_values, list):
        raise SynthesisError("Polisher did not return a mentions array.")
    for literal in canonical.protected_literals:
        if literal not in query:
            raise SynthesisError(f"Protected literal was changed or removed: {literal}")
    expected_cities = {blueprint.trip.origin, *blueprint.trip.destinations}
    unexpected_cities = {city for city in _SUPPORTED_CITIES if city in query} - expected_cities
    if unexpected_cities:
        raise SynthesisError(f"Polisher introduced cities: {sorted(unexpected_cities)}")
    canonical_numbers = set(_NUMBER.findall(canonical.query))
    output_numbers = set(_NUMBER.findall(query))
    if output_numbers != canonical_numbers:
        raise SynthesisError(
            f"Numeric literals changed: expected {canonical_numbers}, got {output_numbers}"
        )

    expected = {item.id: item for item in blueprint.constraints}
    returned: dict[str, str] = {}
    for item in mention_values:
        if not isinstance(item, dict) or set(item) != {"constraint_id", "text"}:
            raise SynthesisError("Every mention must contain exactly constraint_id and text.")
        constraint_id = item["constraint_id"]
        text = item["text"]
        if not isinstance(constraint_id, str) or not isinstance(text, str) or not text.strip():
            raise SynthesisError("Mention ids and text must be non-empty strings.")
        if constraint_id in returned:
            raise SynthesisError(f"Duplicate mention id: {constraint_id}")
        returned[constraint_id] = text
    if set(returned) != set(expected):
        raise SynthesisError(
            f"Mention coverage mismatch: expected {sorted(expected)}, got {sorted(returned)}"
        )

    mentions: list[ConstraintMention] = []
    for constraint_id, constraint in expected.items():
        text = returned[constraint_id]
        if query.count(text) != 1:
            raise SynthesisError(f"Mention {constraint_id} must occur exactly once in query.")
        _validate_polarity(constraint.operator, text)
        value = constraint.value if isinstance(constraint.value, dict) else {}
        leg = value.get("leg")
        if leg in {"outbound", "return"}:
            marker = "去程" if leg == "outbound" else "返程"
            if marker not in text:
                raise SynthesisError(f"Mention {constraint_id} lost its {marker} scope.")
        start = query.index(text)
        mentions.append(
            ConstraintMention(
                constraint_id=constraint_id,
                text=text,
                start=start,
                end=start + len(text),
            )
        )
    mentions.sort(key=lambda item: item.start)
    for previous, current in zip(mentions, mentions[1:], strict=False):
        if previous.end > current.start:
            raise SynthesisError("Constraint mention spans overlap.")

    scrubbed = query.replace("不超过", "").replace("不高于", "")
    if any(marker in scrubbed for marker in ("无需", "不需要", "禁止", "排除")):
        raise SynthesisError("Polisher introduced an unsupported negative requirement.")
    return TaskSurface(
        blueprint_id=blueprint.blueprint_id,
        public_query=query,
        canonical_query=canonical.query,
        mentions=tuple(mentions),
        language="zh",
        polisher_model=model,
        prompt_version=PROMPT_VERSION,
        usage=usage or {},
    )


def _validate_polarity(operator: str, text: str) -> None:
    if operator == "lte":
        if not any(marker in text for marker in ("不超过", "以内", "至多", "之前", "前", "不高于")):
            raise SynthesisError(f"Upper-bound mention lost its polarity: {text}")
        if any(marker in text for marker in ("至少", "不少于")):
            raise SynthesisError(f"Upper-bound mention gained lower-bound wording: {text}")
    elif operator == "gte":
        if not any(marker in text for marker in ("至少", "不少于", "之后", "后", "不早于")):
            raise SynthesisError(f"Lower-bound mention lost its polarity: {text}")
        if any(marker in text for marker in ("不超过", "至多", "之前")):
            raise SynthesisError(f"Lower-bound mention gained upper-bound wording: {text}")
    elif operator == "eq":
        if not any(marker in text for marker in ("必须", "统一", "恰好", "正好", "只", "均为")):
            raise SynthesisError(f"Equality mention became optional: {text}")
    elif operator in {"contains", "include"}:
        if not any(
            marker in text
            for marker in ("必须", "至少", "包含", "安排", "具备", "游览", "入住", "去")
        ):
            raise SynthesisError(f"Inclusion mention became optional: {text}")


def _tool_payload(response: Any) -> dict[str, Any]:
    choices = _get(response, "choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise SynthesisError("Polisher response must contain exactly one choice.")
    message = _get(choices[0], "message")
    tool_calls = _get(message, "tool_calls")
    if not isinstance(tool_calls, list) or len(tool_calls) != 1:
        raise SynthesisError("Polisher must make exactly one function call.")
    function = _get(tool_calls[0], "function")
    if _get(function, "name") != "polish_travel_query":
        raise SynthesisError("Polisher called an unexpected function.")
    arguments = _get(function, "arguments")
    if not isinstance(arguments, str):
        raise SynthesisError("Polisher function arguments must be serialized JSON.")
    try:
        payload = json.loads(arguments)
    except json.JSONDecodeError as error:
        raise SynthesisError("Polisher returned invalid function argument JSON.") from error
    if not isinstance(payload, dict):
        raise SynthesisError("Polisher function payload must be an object.")
    return payload


def _add_usage(total: dict[str, int], response: Any) -> None:
    usage = _get(response, "usage")
    for key in total:
        value = _get(usage, key)
        if isinstance(value, int):
            total[key] += value


def _get(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)
