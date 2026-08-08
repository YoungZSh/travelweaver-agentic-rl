"""LLM surface realization with strict rule-based semantic preservation checks."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass, replace
from threading import Lock
from typing import Any

from ..errors import SynthesisError
from ..llm import DeepSeekConfig, OpenAICompatibleChatClient
from ..tasks import ConstraintMention, PreferenceMention, TaskBlueprint, TaskSurface
from .models import PROMPT_VERSION, CanonicalTask

_SUPPORTED_CITIES = ("上海", "北京", "南京", "广州", "成都", "杭州", "武汉", "深圳", "苏州", "重庆")
_NUMBER = re.compile(r"(?<![A-Za-z])(?:\d{1,2}:\d{2}|\d+(?:\.\d+)?)")
_ENTITY_LIKE = re.compile(
    r"[\u4e00-\u9fff]{2,18}(?:公园|博物馆|酒店|餐厅|景区|古镇|广场|寺|宫|塔|山|湖)"
)
_VALIDATION_POLICIES = {"strict", "minimal_semantic"}
_MEAL_REQUIREMENT = re.compile(
    r"(?:(?:至少|最少|起码|不少于)\s*(?:要|需|需要|得)?\s*"
    r"(?:安排|吃|享用|包含|有)?|(?:要|需|需要|得|必须)?\s*"
    r"(?:安排|吃|享用|包含|有))\s*(?:一|1)\s*顿(?:饭|餐|用餐)?"
)
_MULTIPLE_INNERCITY_PLACES = re.compile(
    r"(?:至少|最少|起码|不少于)?\s*(?:安排|选择|去|逛|游览|走访|打卡|包含|有)?"
    r"[^。；！？\n]{0,6}(?:两|二|2)\s*(?:个|处|站)?\s*"
    r"(?:市内|本地)?(?:地点|地方|去处|景点|站点|游览点|目的地|点位)"
)
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
                    "preference_mentions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "preference_id": {"type": "string"},
                                "text": {"type": "string"},
                            },
                            "required": ["preference_id", "text"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["query", "mentions", "preference_mentions"],
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
6. 遵守 style_direction 的表达方向，不要把所有输入改写成同一种开头或结尾。
7. validation_profile=human_conservative 时可以自然调整语序、拆合句子与语气，但仍不得
   改动任何事实或把硬边界改成“左右、大概、尽量”等模糊说法。
8. preference_mentions 必须逐一覆盖已分配偏好；偏好可用受控同义表达，但方向不能改变。
9. validation_profile=benchmark_natural 时使用 benchmark 风格的自然确定表达，不要为了表示
   硬约束而机械添加“必须、硬性要求”等词。
10. 餐厅每餐预算约束必须明确要求至少安排一顿用餐；市内交通方式约束必须明确要求至少
    安排两个市内地点。两项前提都可以自然改写，但不得省略。
"""
_STYLE_DIRECTIONS = {
    "direct": "直接清楚地提出规划请求。",
    "conversational": "像真实用户给旅行顾问发消息，语气自然口语化。",
    "trip_first": "先交代行程，再顺畅带出限制。",
    "party_first": "从同行人数和出行安排切入。",
    "concise": "保持简洁，用两到三句表达，不写固定套话结尾。",
    "consultant": "使用旅行顾问委托语气，表达清晰但不生硬。",
    "narrative": "使用‘我们打算’一类自然叙述，但不得新增背景。",
    "itinerary": "围绕行程安排来组织句子，避免逐条模板复述。",
    "question": "以自然问句提出规划请求，再说明硬性条件。",
    "compact": "紧凑表达全部信息，减少冗余连接词。",
    "human_metadata": "保留方括号元数据前缀，其余内容写得像真人发来的旅行咨询。",
    "human_dialogue": "写成纯自然对话，允许拆合句子，避免‘硬性条件’等模板措辞。",
    "human_v1_1_metadata": "保留事实元数据前缀，正文不要重复元数据中的人设，使用自然咨询语气。",
    "human_v1_1_dialogue": "使用纯自然对话和确定表达，不列举‘硬性要求’或机械复述条件。",
}
POLISHER_PROMPT_HASH = hashlib.sha256(_SYSTEM_PROMPT.encode("utf-8")).hexdigest()


class TaskPolisher:
    def __init__(
        self,
        config: DeepSeekConfig,
        *,
        client: Any | None = None,
        max_attempts: int = 2,
        validation_policy: str = "minimal_semantic",
    ) -> None:
        if max_attempts <= 0:
            raise ValueError("Polisher max_attempts must be positive.")
        if validation_policy not in _VALIDATION_POLICIES:
            raise ValueError(f"Unknown validation policy: {validation_policy}")
        self.config = replace(
            config,
            thinking="disabled",
            tool_choice="required",
            max_tokens=min(config.max_tokens, 1024),
            temperature=0.7,
        )
        self.client = client or OpenAICompatibleChatClient(self.config)
        self.max_attempts = max_attempts
        self.validation_policy = validation_policy
        self._api_calls = 0
        self._counter_lock = Lock()

    @property
    def api_calls(self) -> int:
        with self._counter_lock:
            return self._api_calls

    def polish(
        self,
        blueprint: TaskBlueprint,
        canonical: CanonicalTask,
        *,
        style_profile: str = "direct",
        validation_profile: str = "strict",
    ) -> TaskSurface:
        surface, _ = self.polish_with_audit(
            blueprint,
            canonical,
            style_profile=style_profile,
            validation_profile=validation_profile,
        )
        return surface

    def polish_with_audit(
        self,
        blueprint: TaskBlueprint,
        canonical: CanonicalTask,
        *,
        style_profile: str = "direct",
        validation_profile: str = "strict",
        audit_context: Mapping[str, Any] | None = None,
    ) -> tuple[TaskSurface, tuple[dict[str, Any], ...]]:
        errors: list[str] = []
        audit_events: list[dict[str, Any]] = []
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for attempt in range(1, self.max_attempts + 1):
            messages = self._messages(
                blueprint,
                canonical,
                errors,
                style_profile,
                validation_profile,
                self.validation_policy,
            )
            with self._counter_lock:
                self._api_calls += 1
            response: Any | None = None
            payload: dict[str, Any] | None = None
            try:
                response = self.client.complete(messages, _TOOLS)
                payload = _tool_payload(response)
                _add_usage(usage, response)
                surface = validate_surface(
                    blueprint,
                    canonical,
                    payload,
                    model=self.config.model,
                    usage=usage,
                    validation_profile=validation_profile,
                    validation_policy=self.validation_policy,
                )
                accepted_event = _audit_event(
                        blueprint,
                        messages,
                        attempt=attempt,
                        outcome="accepted",
                        response=response,
                        payload=payload,
                        error=None,
                        context=audit_context,
                    )
                accepted_event["validation_warnings"] = list(surface.validation_warnings)
                audit_events.append(accepted_event)
                return surface, tuple(audit_events)
            except Exception as error:
                message = f"attempt {attempt}: {type(error).__name__}: {error}"
                errors.append(message)
                audit_events.append(
                    _audit_event(
                        blueprint,
                        messages,
                        attempt=attempt,
                        outcome="rejected",
                        response=response,
                        payload=payload,
                        error=message,
                        context=audit_context,
                    )
                )
        surface = validate_surface(
            blueprint,
            canonical,
            {
                "query": canonical.query,
                "mentions": [
                    {"constraint_id": constraint_id, "text": text}
                    for constraint_id, text in canonical.clauses.items()
                ],
                "preference_mentions": [
                    {"preference_id": preference_id, "text": text}
                    for preference_id, text in canonical.preference_clauses.items()
                ],
            },
            model=f"{self.config.model}:canonical-fallback",
            usage=usage,
            validation_profile=validation_profile,
            validation_policy=self.validation_policy,
        )
        audit_events.append(
            {
                **dict(audit_context or {}),
                "blueprint_id": blueprint.blueprint_id,
                "attempt": None,
                "outcome": "canonical_fallback",
                "validation_errors": list(errors),
                "request": None,
                "raw_response": None,
                "parsed_payload": None,
                "validation_warnings": list(surface.validation_warnings),
            }
        )
        return surface, tuple(audit_events)

    @staticmethod
    def _messages(
        blueprint: TaskBlueprint,
        canonical: CanonicalTask,
        prior_errors: list[str],
        style_profile: str,
        validation_profile: str,
        validation_policy: str,
    ) -> list[dict[str, str]]:
        try:
            style_direction = _STYLE_DIRECTIONS[style_profile]
        except KeyError as error:
            raise SynthesisError(f"Unknown surface style profile: {style_profile}") from error
        payload = {
            "canonical_query": canonical.query,
            "constraint_clauses": canonical.clauses,
            "preference_clauses": canonical.preference_clauses,
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
            "preference_semantics": [
                {
                    "preference_id": item.id,
                    "kind": item.kind,
                    "direction": item.direction,
                    "value": item.value,
                }
                for item in blueprint.preferences
            ],
            "previous_validation_errors": prior_errors[-1:],
            "style_direction": style_direction,
            "style_profile": style_profile,
            "validation_profile": validation_profile,
            "validation_policy": validation_policy,
            "persona_context": blueprint.persona_context,
            "metadata_prefix": blueprint.metadata_prefix,
        }
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, sort_keys=True),
            },
        ]


def _audit_event(
    blueprint: TaskBlueprint,
    messages: list[dict[str, str]],
    *,
    attempt: int,
    outcome: str,
    response: Any | None,
    payload: dict[str, Any] | None,
    error: str | None,
    context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    request_payload = json.loads(messages[-1]["content"])
    return {
        **dict(context or {}),
        "blueprint_id": blueprint.blueprint_id,
        "attempt": attempt,
        "outcome": outcome,
        "validation_error": error,
        "request": request_payload,
        "raw_response": _jsonable(response),
        "parsed_payload": _jsonable(payload),
    }


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _jsonable(model_dump(mode="json"))
        except TypeError:
            return _jsonable(model_dump())
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        return {
            str(key): _jsonable(item)
            for key, item in attributes.items()
            if not str(key).startswith("_")
        }
    return repr(value)


def validate_surface(
    blueprint: TaskBlueprint,
    canonical: CanonicalTask,
    payload: dict[str, Any],
    *,
    model: str,
    usage: dict[str, int] | None = None,
    validation_profile: str = "strict",
    validation_policy: str = "strict",
) -> TaskSurface:
    if validation_profile not in {"strict", "human_conservative", "benchmark_natural"}:
        raise SynthesisError(f"Unknown validation profile: {validation_profile}")
    if validation_policy not in _VALIDATION_POLICIES:
        raise SynthesisError(f"Unknown validation policy: {validation_policy}")
    minimal = validation_policy == "minimal_semantic"
    warnings: list[str] = []
    query = payload.get("query")
    mention_values = payload.get("mentions")
    if not isinstance(query, str) or not query.strip() or len(query) > 500:
        raise SynthesisError("Polished query must be a non-empty string of at most 500 characters.")
    if len(query) < 30:
        if minimal:
            warnings.append("query_shorter_than_30_characters")
        else:
            raise SynthesisError("Polished query must be a 30-500 character string.")
    if any(token in query for token in ("```", "{", "}", "constraint_id")):
        raise SynthesisError("Polished query contains serialization or Markdown artifacts.")
    if not isinstance(mention_values, list):
        raise SynthesisError("Polisher did not return a mentions array.")
    for literal in canonical.protected_literals:
        if literal not in query:
            if minimal:
                warnings.append(f"protected_literal_changed:{literal}")
            else:
                raise SynthesisError(f"Protected literal was changed or removed: {literal}")
    if (
        blueprint.metadata_prefix
        and blueprint.metadata_prefix.startswith("[当前位置")
        and blueprint.persona_context
        and blueprint.persona_context in query[len(blueprint.metadata_prefix) :]
    ):
        if minimal:
            warnings.append("metadata_persona_repeated_in_body")
        else:
            raise SynthesisError("V1.1 Human body repeats the persona already stored in metadata.")
    canonical_cities = {city for city in _SUPPORTED_CITIES if city in canonical.query}
    unexpected_cities = {city for city in _SUPPORTED_CITIES if city in query} - canonical_cities
    if unexpected_cities:
        raise SynthesisError(f"Polisher introduced cities: {sorted(unexpected_cities)}")
    for city in (blueprint.trip.origin, *blueprint.trip.destinations):
        if city not in query:
            raise SynthesisError(f"Polisher removed trip city: {city}")
    if minimal:
        metadata_preserved = bool(
            blueprint.metadata_prefix and blueprint.metadata_prefix in query
        )
        if not metadata_preserved and not _contains_traveler_count(
            query, blueprint.trip.travelers
        ):
            raise SynthesisError("Polisher changed or removed the traveler count.")
        if not metadata_preserved and not _contains_count(
            query, blueprint.trip.days, ("天", "日")
        ):
            raise SynthesisError("Polisher changed or removed the trip duration.")
    canonical_numbers = Counter(_NUMBER.findall(canonical.query))
    output_numbers = Counter(_NUMBER.findall(query))
    if output_numbers != canonical_numbers:
        if minimal:
            warnings.append("global_numeric_literal_multiset_changed")
        else:
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
    occupied: list[tuple[int, int]] = []
    for constraint_id, constraint in expected.items():
        text = returned[constraint_id]
        occurrence_count = query.count(text)
        if occurrence_count == 0:
            repaired = _repair_constraint_mention(constraint, query) if minimal else None
            if repaired is None:
                raise SynthesisError(f"Mention {constraint_id} does not occur in query.")
            text = repaired
            occurrence_count = query.count(text)
            warnings.append(f"mention_repaired:{constraint_id}")
        if occurrence_count != 1 and not minimal:
            raise SynthesisError(f"Mention {constraint_id} must occur exactly once in query.")
        if occurrence_count > 1:
            warnings.append(f"mention_repeated:{constraint_id}")
        if minimal:
            _validate_minimal_constraint(constraint, text, query)
        else:
            _validate_polarity(
                constraint.kind,
                constraint.operator,
                text,
                validation_profile=validation_profile,
            )
        _validate_non_vacuous_scope(constraint, query)
        value = constraint.value if isinstance(constraint.value, dict) else {}
        leg = value.get("leg")
        if leg in {"outbound", "return"}:
            marker = "去程" if leg == "outbound" else "返程"
            if marker not in text and not (minimal and _marker_in_context(query, text, marker)):
                raise SynthesisError(f"Mention {constraint_id} lost its {marker} scope.")
            if minimal and marker not in text:
                warnings.append(f"scope_marker_outside_mention:{constraint_id}")
        start = (
            query.index(text)
            if minimal
            else _nonoverlapping_start(query, text, occupied)
        )
        if start is None:
            raise SynthesisError(f"Mention {constraint_id} cannot be aligned without overlap.")
        occupied.append((start, start + len(text)))
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
        if not minimal and previous.end > current.start:
            raise SynthesisError("Constraint mention spans overlap.")

    preference_values = payload.get("preference_mentions", [])
    if not isinstance(preference_values, list):
        raise SynthesisError("Polisher did not return a preference_mentions array.")
    expected_preferences = {item.id: item for item in blueprint.preferences}
    returned_preferences: dict[str, str] = {}
    for item in preference_values:
        if not isinstance(item, dict) or set(item) != {"preference_id", "text"}:
            raise SynthesisError(
                "Every preference mention must contain exactly preference_id and text."
            )
        preference_id = item["preference_id"]
        text = item["text"]
        if not isinstance(preference_id, str) or not isinstance(text, str) or not text.strip():
            raise SynthesisError("Preference mention ids and text must be non-empty strings.")
        if preference_id in returned_preferences:
            raise SynthesisError(f"Duplicate preference mention id: {preference_id}")
        returned_preferences[preference_id] = text
    if set(returned_preferences) != set(expected_preferences):
        raise SynthesisError("Preference mention coverage mismatch.")
    preference_mentions: list[PreferenceMention] = []
    for preference_id, preference in expected_preferences.items():
        text = returned_preferences[preference_id]
        occurrence_count = query.count(text)
        if occurrence_count == 0:
            raise SynthesisError(f"Preference mention {preference_id} does not occur in query.")
        if occurrence_count != 1 and not minimal:
            raise SynthesisError(
                f"Preference mention {preference_id} must occur exactly once in query."
            )
        if occurrence_count > 1:
            warnings.append(f"preference_mention_repeated:{preference_id}")
        try:
            _validate_preference(preference.kind, preference.direction, text)
        except SynthesisError:
            if minimal:
                warnings.append(f"preference_direction_unrecognized:{preference_id}")
            else:
                raise
        start = (
            query.index(text)
            if minimal
            else _nonoverlapping_start(query, text, occupied)
        )
        if start is None:
            raise SynthesisError(
                f"Preference mention {preference_id} cannot be aligned without overlap."
            )
        occupied.append((start, start + len(text)))
        preference_mentions.append(
            PreferenceMention(
                preference_id=preference_id,
                text=text,
                start=start,
                end=start + len(text),
            )
        )
    all_spans = sorted([*mentions, *preference_mentions], key=lambda item: item.start)
    for previous, current in zip(all_spans, all_spans[1:], strict=False):
        if not minimal and previous.end > current.start:
            raise SynthesisError("Hard and preference mention spans overlap.")

    canonical_entities = set(_ENTITY_LIKE.findall(canonical.query))
    introduced_entities = {
        entity
        for entity in set(_ENTITY_LIKE.findall(query)) - canonical_entities
        if not any(literal in entity for literal in canonical.protected_literals)
    }
    if introduced_entities:
        if minimal:
            warnings.append(
                "possible_introduced_entities:" + "|".join(sorted(introduced_entities))
            )
        else:
            raise SynthesisError(
                f"Polisher introduced named entities: {sorted(introduced_entities)}"
            )

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
        preference_mentions=tuple(preference_mentions),
        validation_policy=validation_policy,
        validation_warnings=tuple(dict.fromkeys(warnings)),
    )


def _validate_polarity(
    kind: str,
    operator: str,
    text: str,
    *,
    validation_profile: str,
) -> None:
    if any(marker in text for marker in ("如果可以", "最好", "可选", "看情况", "尽量满足")):
        raise SynthesisError(f"Hard constraint became optional: {text}")
    if operator in {"lte", "gte"} and any(
        marker in text for marker in ("左右", "大概", "差不多", "尽量")
    ):
        raise SynthesisError(f"Hard boundary became ambiguous: {text}")
    polarity_text = _strip_scope_prerequisite(kind, text)
    if operator == "lte":
        if not any(
            marker in polarity_text
            for marker in ("不超过", "以内", "至多", "之前", "前", "不高于")
        ):
            raise SynthesisError(f"Upper-bound mention lost its polarity: {text}")
        if any(marker in polarity_text for marker in ("至少", "不少于")):
            raise SynthesisError(f"Upper-bound mention gained lower-bound wording: {text}")
    elif operator == "gte":
        if not any(marker in text for marker in ("至少", "不少于", "之后", "后", "不早于")):
            raise SynthesisError(f"Lower-bound mention lost its polarity: {text}")
        if any(marker in text for marker in ("不超过", "至多", "之前")):
            raise SynthesisError(f"Lower-bound mention gained upper-bound wording: {text}")
    elif operator == "eq":
        strict_markers = ("必须", "统一", "恰好", "正好", "只", "均为")
        has_hard_marker = any(marker in text for marker in strict_markers)
        imperative_by_kind = {
            "activity_count": ("安排",),
            "room_count": ("订", "预订"),
            "room_type": ("选择", "房型"),
            "transport_mode": ("乘坐", "坐", "使用", "步行"),
        }
        has_typed_imperative = validation_profile in {
            "human_conservative",
            "benchmark_natural",
        } and any(
            marker in text for marker in imperative_by_kind.get(kind, ())
        )
        if not has_hard_marker and not has_typed_imperative:
            raise SynthesisError(f"Equality mention became optional: {text}")
    elif operator in {"contains", "include"}:
        if not any(
            marker in text
            for marker in (
                "必须",
                "至少",
                "包含",
                "安排",
                "具备",
                "游览",
                "入住",
                "去",
                "有",
                "住",
            )
        ):
            raise SynthesisError(f"Inclusion mention became optional: {text}")


def _validate_minimal_constraint(constraint: Any, text: str, query: str) -> None:
    optional_markers = ("如果可以", "最好", "可选", "看情况", "尽量满足")
    if any(marker in text for marker in optional_markers):
        raise SynthesisError(f"Hard constraint became optional: {text}")
    anchor = _constraint_anchor(constraint, query)
    if anchor is not None:
        context = _clause_around(query, anchor)
        if any(marker in context for marker in optional_markers):
            raise SynthesisError(f"Hard constraint became optional in context: {context}")
    operator = str(constraint.operator)
    kind = str(constraint.kind)
    polarity_text = _strip_scope_prerequisite(kind, text)
    if operator == "lte":
        if not any(
            marker in polarity_text
            for marker in (
                "不超过",
                "不超",
                "别超过",
                "不能超过",
                "以内",
                "至多",
                "最多",
                "之前",
                "前",
                "不高于",
                "控制在",
            )
        ):
            raise SynthesisError(f"Upper-bound mention lost its polarity: {text}")
        if any(
            marker in polarity_text for marker in ("至少", "不少于", "之后", "不低于")
        ):
            raise SynthesisError(f"Upper-bound mention reversed direction: {text}")
    elif operator == "gte":
        if not any(
            marker in text
            for marker in ("至少", "不少于", "之后", "后", "不早于", "不低于")
        ):
            raise SynthesisError(f"Lower-bound mention lost its polarity: {text}")
        if any(marker in text for marker in ("不超过", "不超", "至多", "之前")):
            raise SynthesisError(f"Lower-bound mention reversed direction: {text}")

    value = constraint.value if isinstance(constraint.value, dict) else {}
    if kind in {"total_budget", "category_budget"}:
        _require_number(text, value.get("amount"), constraint.id)
    elif kind == "time_window":
        if not _contains_time(text, str(value.get("time", ""))):
            raise SynthesisError(f"Time value changed in mention {constraint.id}: {text}")
    elif kind == "transport_mode":
        modes = [str(mode) for mode in value.get("modes", [])]
        if not any(_mode_present(text, mode) for mode in modes):
            raise SynthesisError(f"Transport mode changed in mention {constraint.id}: {text}")
    elif kind == "entity_category":
        category = str(value.get("values", [""])[0])
        if category not in text:
            raise SynthesisError(f"Entity category changed in mention {constraint.id}: {text}")
    elif kind == "entity_attribute":
        attribute = str(value.get("values", [""])[0])
        if attribute not in text:
            raise SynthesisError(f"Entity attribute changed in mention {constraint.id}: {text}")
    elif kind == "include_entity":
        name = str(value.get("names", [""])[0])
        if name not in text:
            raise SynthesisError(f"Required entity changed in mention {constraint.id}: {text}")
    elif kind == "room_type":
        _require_count(text, value.get("room_type"), ("个床位", "张床"), constraint.id)
    elif kind == "room_count":
        _require_count(text, value.get("count"), ("间房", "间客房"), constraint.id)
    elif kind == "activity_count":
        _require_count(text, value.get("count"), ("个景点", "处景点"), constraint.id)


def _strip_scope_prerequisite(kind: str, text: str) -> str:
    if kind == "category_budget":
        return _MEAL_REQUIREMENT.sub("", text)
    return text


def _validate_non_vacuous_scope(constraint: Any, query: str) -> None:
    kind = str(constraint.kind)
    scope = str(constraint.scope)
    if kind == "category_budget" and scope == "restaurant":
        if _MEAL_REQUIREMENT.search(query) is None:
            raise SynthesisError(
                f"Restaurant budget constraint {constraint.id} does not require a meal."
            )
    if kind == "transport_mode" and scope == "innercity_route":
        if _MULTIPLE_INNERCITY_PLACES.search(query) is None:
            raise SynthesisError(
                f"Inner-city route constraint {constraint.id} does not require two places."
            )


def _require_number(text: str, value: Any, constraint_id: str) -> None:
    if not _contains_number(text, value):
        raise SynthesisError(f"Numeric value changed in mention {constraint_id}: {text}")


def _require_count(
    text: str,
    value: Any,
    units: tuple[str, ...],
    constraint_id: str,
) -> None:
    try:
        count = int(value)
    except (TypeError, ValueError) as error:
        raise SynthesisError(f"Constraint {constraint_id} has an invalid count.") from error
    if not _contains_count(text, count, units):
        raise SynthesisError(f"Count changed in mention {constraint_id}: {text}")


def _contains_number(text: str, value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    rendered = str(int(number)) if number.is_integer() else str(number)
    if re.search(rf"(?<!\d){re.escape(rendered)}(?!\d)", text):
        return True
    if number.is_integer() and number >= 0:
        return any(form in text for form in _integer_forms(int(number)))
    return False


def _contains_count(text: str, value: int, units: tuple[str, ...]) -> bool:
    for form in _integer_forms(value):
        for unit in units:
            if f"{form}{unit}" in text:
                return True
            if unit in {"人", "位", "口"} and f"{form}个{unit}" in text:
                return True
    return False


def _contains_traveler_count(text: str, value: int) -> bool:
    if _contains_count(text, value, ("人", "位", "口", "名")):
        return True
    forms = _integer_forms(value)
    if any(
        f"{form}个{noun}" in text
        for form in forms
        for noun in ("朋友", "大人", "孩子", "成人")
    ):
        return True
    if value == 1 and any(marker in text for marker in ("独自", "一个人", "单独")):
        return True
    return value == 2 and any(marker in text for marker in ("我们俩", "两口子"))


def _repair_constraint_mention(constraint: Any, query: str) -> str | None:
    anchor = _constraint_anchor(constraint, query)
    if anchor is None:
        return None
    if str(constraint.kind) in {
        "include_entity",
        "entity_category",
        "entity_attribute",
    }:
        return anchor
    return _clause_around(query, anchor)


def _constraint_anchor(constraint: Any, query: str) -> str | None:
    value = constraint.value if isinstance(constraint.value, dict) else {}
    kind = str(constraint.kind)
    if kind == "include_entity":
        return _first_present(query, (str(value.get("names", [""])[0]),))
    if kind in {"entity_category", "entity_attribute"}:
        return _first_present(query, (str(value.get("values", [""])[0]),))
    if kind in {"total_budget", "category_budget"}:
        return _number_anchor(query, value.get("amount"))
    if kind == "time_window":
        time = str(value.get("time", ""))
        return _first_present(query, _time_forms(time))
    if kind == "transport_mode":
        aliases = tuple(
            alias
            for mode in value.get("modes", [])
            for alias in _mode_aliases(str(mode))
        )
        return _first_present(query, aliases)
    if kind == "room_type":
        return _count_anchor(query, value.get("room_type"), ("个床位", "张床"))
    if kind == "room_count":
        return _count_anchor(query, value.get("count"), ("间房", "间客房"))
    if kind == "activity_count":
        return _count_anchor(query, value.get("count"), ("个景点", "处景点"))
    return None


def _number_anchor(text: str, value: Any) -> str | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    rendered = str(int(number)) if number.is_integer() else str(number)
    candidates = [rendered]
    if number.is_integer() and number >= 0:
        candidates.extend(_integer_forms(int(number)))
    return _first_present(text, tuple(candidates))


def _count_anchor(text: str, value: Any, units: tuple[str, ...]) -> str | None:
    try:
        count = int(value)
    except (TypeError, ValueError):
        return None
    return _first_present(
        text,
        tuple(f"{form}{unit}" for form in _integer_forms(count) for unit in units),
    )


def _first_present(text: str, candidates: tuple[str, ...]) -> str | None:
    present = [candidate for candidate in candidates if candidate and candidate in text]
    return min(present, key=lambda candidate: text.index(candidate)) if present else None


def _clause_around(query: str, anchor: str) -> str:
    start = query.index(anchor)
    left = max(
        query.rfind(separator, 0, start)
        for separator in ("。", "；", ";", "！", "？", "\n")
    )
    right_candidates = [
        position
        for separator in ("。", "；", ";", "！", "？", "\n")
        if (position := query.find(separator, start + len(anchor))) >= 0
    ]
    right = min(right_candidates, default=len(query))
    return query[left + 1 : right].strip()


def _integer_forms(value: int) -> tuple[str, ...]:
    forms = [str(value), _chinese_integer(value)]
    if value == 2:
        forms.append("两")
    elif forms[1].startswith("二"):
        forms.append("两" + forms[1][1:])
    return tuple(dict.fromkeys(forms))


def _chinese_integer(value: int) -> str:
    digits = "零一二三四五六七八九"
    if value < 10:
        return digits[value]
    if value < 20:
        return "十" + (digits[value % 10] if value % 10 else "")
    if value < 100:
        return digits[value // 10] + "十" + (digits[value % 10] if value % 10 else "")
    if value < 1000:
        remainder = value % 100
        suffix = (
            ""
            if remainder == 0
            else ("零" if remainder < 10 else "") + _chinese_integer(remainder)
        )
        return digits[value // 100] + "百" + suffix
    if value < 10000:
        remainder = value % 1000
        suffix = (
            ""
            if remainder == 0
            else ("零" if remainder < 100 else "") + _chinese_integer(remainder)
        )
        return digits[value // 1000] + "千" + suffix
    if value < 100000000:
        remainder = value % 10000
        suffix = (
            ""
            if remainder == 0
            else ("零" if remainder < 1000 else "") + _chinese_integer(remainder)
        )
        return _chinese_integer(value // 10000) + "万" + suffix
    return str(value)


def _contains_time(text: str, value: str) -> bool:
    return any(form in text for form in _time_forms(value))


def _time_forms(value: str) -> tuple[str, ...]:
    alternatives = {value} if value else set()
    match = re.fullmatch(r"(\d{1,2}):(\d{2})", value)
    if match is None:
        return tuple(alternatives)
    hour, minute = int(match.group(1)), int(match.group(2))
    alternatives.update({f"{hour}点{minute:02d}分", f"{hour}时{minute:02d}分"})
    if minute == 0:
        alternatives.update({f"{hour}点", f"{hour}时"})
    return tuple(sorted(alternatives, key=lambda item: (-len(item), item)))


def _mode_present(text: str, mode: str) -> bool:
    return any(alias in text for alias in _mode_aliases(mode))


def _mode_aliases(mode: str) -> tuple[str, ...]:
    aliases = {
        "train": ("火车", "列车"),
        "airplane": ("飞机", "航班", "飞"),
        "taxi": ("出租车", "打车", "的士"),
        "walk": ("步行", "走路", "走着"),
        "metro": ("地铁",),
        "high_speed_rail": ("高铁", "高速铁路"),
    }
    return aliases.get(mode, (mode,))


def _marker_in_context(query: str, text: str, marker: str) -> bool:
    for start in _occurrence_starts(query, text):
        left = max(
            query.rfind(separator, 0, start)
            for separator in ("。", "；", ";", "！", "？", "\n")
        )
        right_candidates = [
            position
            for separator in ("。", "；", ";", "！", "？", "\n")
            if (position := query.find(separator, start + len(text))) >= 0
        ]
        right = min(right_candidates, default=len(query))
        if marker in query[left + 1 : right]:
            return True
    return False


def _nonoverlapping_start(
    query: str,
    text: str,
    occupied: list[tuple[int, int]],
) -> int | None:
    for start in _occurrence_starts(query, text):
        end = start + len(text)
        if all(
            end <= previous_start or start >= previous_end
            for previous_start, previous_end in occupied
        ):
            return start
    return None


def _occurrence_starts(query: str, text: str) -> tuple[int, ...]:
    starts: list[int] = []
    offset = 0
    while (start := query.find(text, offset)) >= 0:
        starts.append(start)
        offset = start + 1
    return tuple(starts)


def _validate_preference(kind: str, direction: str, text: str) -> None:
    markers = {
        "more_attractions": ("多",),
        "less_innercity_time": ("少", "短"),
        "shorter_meal_transfer": ("短", "少"),
        "higher_dining_share": ("高", "多"),
        "lower_lodging_share": ("低", "少"),
        "near_poi": ("近", "靠近"),
        "less_walking": ("少走", "少步行"),
        "lower_total_cost": ("低", "少"),
        "relaxed_itinerary": ("轻松", "宽松", "别太赶", "不赶"),
        "higher_attraction_share": ("高", "多"),
        "lower_intercity_share": ("低", "少"),
        "shorter_total_travel_time": ("短", "少"),
    }
    if not any(marker in text for marker in markers.get(kind, ())):
        raise SynthesisError(f"Preference direction was lost: {text}")
    if direction == "minimize" and any(marker in text for marker in ("更高", "更多")):
        raise SynthesisError(f"Minimization preference reversed direction: {text}")
    if direction == "maximize" and any(marker in text for marker in ("更低", "更少")):
        raise SynthesisError(f"Maximization preference reversed direction: {text}")


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
