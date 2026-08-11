"""Template-first, LLM-polished visible rationales for programmatic ReAct trajectories."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Lock
from typing import Any

from ..errors import SFTRebuildError
from ..llm import DEFAULT_DEEPSEEK_CONCURRENCY, DeepSeekConfig, OpenAICompatibleChatClient
from .rationale_contract import has_visible_price_comparison

RATIONALE_POLISHER_VERSION = "travelweaver-trajectory-rationale-polisher-v3"
RATIONALE_PROMPT_VERSION = "travelweaver-trajectory-rationale-prompt-v2"

_NUMBER = re.compile(r"\d+(?::\d{2})?")
_PREMATURE_RESULT = re.compile(r"(?:已经|已)(?:找到|确认|核实|获得)|结果(?:显示|表明)|符合要求")
_SEMANTIC_LABELS = {
    "火车",
    "飞机",
    "出租车",
    "地铁",
    "步行",
    "景点",
    "餐厅",
    "酒店",
    "去程交通",
    "返程交通",
    "用餐地点",
    "住宿",
    "往返交通",
    "完整路线",
    "餐饮",
}
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "polish_trajectory_rationales",
            "description": "逐轮返回仅经语言润色、与原工具决策完全一致的可见中文决策说明。",
            "parameters": {
                "type": "object",
                "properties": {
                    "turns": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "step_index": {"type": "integer", "minimum": 0},
                                "rationale": {"type": "string", "minLength": 1},
                            },
                            "required": ["step_index", "rationale"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["turns"],
                "additionalProperties": False,
            },
        },
    }
]
_SYSTEM_PROMPT = """你是旅行规划 Agent 的可见决策说明润色器。
输入是一条已经确定、可回放且正确的工具轨迹。你只能逐轮润色 template_rationale，不能重新规划。

必须通过 polish_trajectory_rationales 函数返回，并严格遵守：
1. turns 数量、step_index 和顺序必须与输入完全一致；不得增删、合并或交换回合。
2. 每条 rationale 只解释同一回合为什么要调用该工具，不得改变工具、参数、候选选择或计划。
3. 只能使用 template_rationale 和同回合 tool/arguments 中已有的事实；不得借用后续回合信息，
   不得声称尚未执行的查询已经获得结果，也不得编造价格、时间、路线、营业状态或约束。
4. protected_literals 中的专名必须逐字保留；交通方式和证据类别可以使用明确同义表达。
   阿拉伯数字只能来自题面、template_rationale 或同回合 arguments，不得新增或改变。
5. 表达应自然、有上下文感，可按决策复杂度使用一至数句，不要机械重复固定句式，也不要写成长篇
   思维链、自我身份说明、Markdown、步骤编号或最终答案。
6. injected_loop 是刻意保留且不计 loss 的错误决策，只润色其原意；loop_recovery 必须清楚表达停止
   重复并转向同回合正确工具。所有 rationale 都发生在同回合工具调用之前，描述当前动作时使用
   “我再执行一次相同的查询”这类将要执行的表达，不得写成“我重复执行了查询”等完成时态；
   evidence_ready_submit 必须说明证据完备后提交。
7. 除函数调用外不要输出任何普通文本或隐藏思考。
"""


@dataclass(frozen=True)
class RationalePolishConfig:
    input_path: Path
    input_audit_path: Path
    output_path: Path
    output_audit_path: Path
    work_dir: Path
    llm_concurrency: int = DEFAULT_DEEPSEEK_CONCURRENCY
    max_api_calls: int = 500
    task_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.llm_concurrency <= 0 or self.max_api_calls <= 0:
            raise ValueError("Rationale polish concurrency and API-call budget must be positive.")
        paths = {
            self.input_path.resolve(),
            self.input_audit_path.resolve(),
            self.output_path.resolve(),
            self.output_audit_path.resolve(),
        }
        if len(paths) != 4:
            raise ValueError("Rationale polish input and output paths must be distinct.")
        if len(self.task_ids) != len(set(self.task_ids)):
            raise ValueError("Rationale polish task_ids must be unique.")


@dataclass(frozen=True)
class RationalePolishReport:
    samples: int
    api_calls: int
    resumed: int
    llm_concurrency: int
    outcomes: dict[str, int]
    polished_turns: int
    fallback_turns: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "rationale_polisher_version": RATIONALE_POLISHER_VERSION,
            "rationale_prompt_version": RATIONALE_PROMPT_VERSION,
            "samples": self.samples,
            "api_calls": self.api_calls,
            "resumed": self.resumed,
            "llm_concurrency": self.llm_concurrency,
            "outcomes": dict(sorted(self.outcomes.items())),
            "polished_turns": self.polished_turns,
            "fallback_turns": self.fallback_turns,
        }


@dataclass(frozen=True)
class RationaleRevalidationConfig:
    """Rebuild polished artifacts from retained provider responses without an API call."""

    input_path: Path
    input_audit_path: Path
    source_audit_path: Path
    output_path: Path
    output_audit_path: Path

    def __post_init__(self) -> None:
        paths = {
            self.input_path.resolve(),
            self.input_audit_path.resolve(),
            self.source_audit_path.resolve(),
            self.output_path.resolve(),
            self.output_audit_path.resolve(),
        }
        if len(paths) != 5:
            raise ValueError("Rationale revalidation paths must be distinct.")


@dataclass(frozen=True)
class RationaleRevalidationReport:
    samples: int
    accepted_turns: int
    fallback_turns: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "rationale_polisher_version": RATIONALE_POLISHER_VERSION,
            "operation": "revalidate_saved_responses",
            "api_calls": 0,
            "samples": self.samples,
            "accepted_turns": self.accepted_turns,
            "fallback_turns": self.fallback_turns,
        }


class TrajectoryRationalePolisher:
    """Polish all visible rationales in one trajectory with one structured API call."""

    def __init__(
        self,
        config: DeepSeekConfig,
        *,
        client: Any | None = None,
    ) -> None:
        self.config = replace(
            config,
            thinking="disabled",
            tool_choice="required",
            max_tokens=min(config.max_tokens, 8192),
            temperature=0.8,
        )
        self.client = client or OpenAICompatibleChatClient(self.config)
        self._api_calls = 0
        self._lock = Lock()

    @property
    def api_calls(self) -> int:
        with self._lock:
            return self._api_calls

    def polish(
        self,
        trajectory: Mapping[str, Any],
        audit: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        request_turns = _request_turns(trajectory, audit)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": _question(trajectory),
                        "turns": request_turns,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        ]
        with self._lock:
            self._api_calls += 1
        response: Any | None = None
        try:
            response = self.client.complete(messages, _TOOLS)
            payload = _tool_payload(response)
            proposed = _proposed_rationales(payload, len(request_turns))
            api_error = None
        except Exception as error:
            proposed = [None] * len(request_turns)
            api_error = f"{type(error).__name__}: {error}"

        return self._finalize(
            trajectory,
            audit,
            request_turns=request_turns,
            proposed=proposed,
            response=response,
            api_error=api_error,
        )

    def revalidate(
        self,
        trajectory: Mapping[str, Any],
        audit: Mapping[str, Any],
        response: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reapply current validators to a saved response without making an API call."""

        request_turns = _request_turns(trajectory, audit)
        try:
            payload = _tool_payload(response)
            proposed = _proposed_rationales(payload, len(request_turns))
            api_error = None
        except Exception as error:
            proposed = [None] * len(request_turns)
            api_error = f"{type(error).__name__}: {error}"
        return self._finalize(
            trajectory,
            audit,
            request_turns=request_turns,
            proposed=proposed,
            response=response,
            api_error=api_error,
        )

    def _finalize(
        self,
        trajectory: Mapping[str, Any],
        audit: Mapping[str, Any],
        *,
        request_turns: list[dict[str, Any]],
        proposed: list[Any],
        response: Any,
        api_error: str | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        final: list[str] = []
        validation_errors: list[list[str]] = []
        all_literals = {
            literal
            for turn in request_turns
            for literal in turn["protected_literals"]
            if literal and literal not in _SEMANTIC_LABELS
        }
        for index, request in enumerate(request_turns):
            candidate = proposed[index]
            errors = (
                [api_error]
                if api_error is not None
                else _validate_rationale(candidate, request, all_literals)
            )
            validation_errors.append([str(value) for value in errors if value])
            final.append(
                str(request["template_rationale"])
                if errors
                else str(candidate).strip()
            )

        fallback_count = sum(bool(errors) for errors in validation_errors)
        if fallback_count == 0:
            outcome = "accepted"
        elif fallback_count == len(final):
            outcome = "full_template_fallback"
        else:
            outcome = "partial_template_fallback"
        output_trajectory = deepcopy(dict(trajectory))
        assistant_messages = [
            message
            for message in output_trajectory["messages"]
            if message.get("role") == "assistant"
        ]
        for message, rationale in zip(assistant_messages, final, strict=True):
            message["content"] = rationale
        metadata = dict(output_trajectory.get("batch_metadata", {}))
        metadata.update(
            {
                "source": RATIONALE_POLISHER_VERSION,
                "rationale_source_policy": trajectory.get("batch_metadata", {}).get("source"),
                "rationale_polisher_model": self.config.model,
                "rationale_prompt_version": RATIONALE_PROMPT_VERSION,
                "thinking": "disabled",
            }
        )
        output_trajectory["batch_metadata"] = metadata

        output_audit = deepcopy(dict(audit))
        for index, turn in enumerate(output_audit["turns"]):
            turn["template_rationale"] = request_turns[index]["template_rationale"]
            turn["polished_rationale"] = final[index]
            turn["proposed_rationale"] = proposed[index]
            turn["visible_reflection"] = final[index]
            turn["rationale_validation_errors"] = validation_errors[index]
            turn["rationale_polish_outcome"] = (
                "template_fallback" if validation_errors[index] else "accepted"
            )
        output_audit["rationale_polish"] = {
            "version": RATIONALE_POLISHER_VERSION,
            "prompt_version": RATIONALE_PROMPT_VERSION,
            "model": self.config.model,
            "outcome": outcome,
            "polished_turns": len(final) - fallback_count,
            "fallback_turns": fallback_count,
            "api_error": api_error,
            "usage": _usage(response),
            "raw_response": _response_dict(response),
        }
        return output_trajectory, output_audit


def polish_programmatic_rationales(
    config: RationalePolishConfig,
    llm_config: DeepSeekConfig,
    *,
    polisher: TrajectoryRationalePolisher | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> RationalePolishReport:
    """Polish a full trajectory batch concurrently with per-task recovery checkpoints."""

    if config.output_path.exists() or config.output_audit_path.exists():
        raise SFTRebuildError("Refusing to overwrite polished rationale artifacts.")
    trajectories = _read_jsonl(config.input_path)
    audits = _read_jsonl(config.input_audit_path)
    if not trajectories or len(trajectories) != len(audits):
        raise SFTRebuildError("Rationale polish inputs must be nonempty and have equal counts.")
    for trajectory, audit in zip(trajectories, audits, strict=True):
        if trajectory.get("task_id") != audit.get("task_id"):
            raise SFTRebuildError("Rationale polish trajectory/audit task order does not match.")
    if config.task_ids:
        requested = set(config.task_ids)
        available = {str(row["task_id"]) for row in trajectories}
        missing = sorted(requested - available)
        if missing:
            raise SFTRebuildError(f"Unknown rationale polish task ids: {missing}")
        selected = [
            index
            for index, row in enumerate(trajectories)
            if str(row["task_id"]) in requested
        ]
        trajectories = [trajectories[index] for index in selected]
        audits = [audits[index] for index in selected]
    config.work_dir.mkdir(parents=True, exist_ok=True)
    _validate_work_manifest(config)
    completed = _completed_records(config.work_dir, len(trajectories))
    worker = polisher or TrajectoryRationalePolisher(llm_config)
    # Checkpoints include the provider response, so a stricter local validator can
    # safely be applied on resume without spending another API request.  This keeps
    # the final artifact aligned with the current acceptance policy after a crash or
    # a validator improvement.
    for index, bundle in completed.items():
        saved_audit = bundle.get("audit", {})
        rationale_polish = (
            saved_audit.get("rationale_polish", {}) if isinstance(saved_audit, Mapping) else {}
        )
        response = (
            rationale_polish.get("raw_response")
            if isinstance(rationale_polish, Mapping)
            else None
        )
        trajectory, audit = worker.revalidate(trajectories[index], audits[index], response)
        completed[index] = {"trajectory": trajectory, "audit": audit}
        _atomic_json(_record_path(config.work_dir, index), completed[index])
    remaining = [index for index in range(len(trajectories)) if index not in completed]
    if len(remaining) > config.max_api_calls:
        raise SFTRebuildError(
            f"Rationale polish API budget {config.max_api_calls} cannot cover "
            f"{len(remaining)} remaining trajectories."
        )
    def run(index: int) -> tuple[int, dict[str, Any]]:
        trajectory, audit = worker.polish(trajectories[index], audits[index])
        bundle = {"trajectory": trajectory, "audit": audit}
        _atomic_json(_record_path(config.work_dir, index), bundle)
        return index, bundle

    with ThreadPoolExecutor(max_workers=config.llm_concurrency) as executor:
        futures = {executor.submit(run, index): index for index in remaining}
        for future in as_completed(futures):
            index, bundle = future.result()
            completed[index] = bundle
            if progress is not None:
                progress(
                    {
                        "event": "rationale_polished",
                        "completed": len(completed),
                        "total": len(trajectories),
                        "task_id": trajectories[index]["task_id"],
                        "outcome": bundle["audit"]["rationale_polish"]["outcome"],
                    }
                )
    if len(completed) != len(trajectories):
        raise SFTRebuildError("Rationale polish did not produce every trajectory.")
    ordered = [completed[index] for index in range(len(trajectories))]
    _atomic_jsonl(config.output_path, [bundle["trajectory"] for bundle in ordered])
    _atomic_jsonl(config.output_audit_path, [bundle["audit"] for bundle in ordered])
    outcomes = Counter(
        str(bundle["audit"]["rationale_polish"]["outcome"]) for bundle in ordered
    )
    return RationalePolishReport(
        samples=len(ordered),
        api_calls=worker.api_calls,
        resumed=len(ordered) - len(remaining),
        llm_concurrency=config.llm_concurrency,
        outcomes=dict(outcomes),
        polished_turns=sum(
            int(bundle["audit"]["rationale_polish"]["polished_turns"])
            for bundle in ordered
        ),
        fallback_turns=sum(
            int(bundle["audit"]["rationale_polish"]["fallback_turns"])
            for bundle in ordered
        ),
    )


def revalidate_programmatic_rationales(
    config: RationaleRevalidationConfig,
    llm_config: DeepSeekConfig,
    *,
    polisher: TrajectoryRationalePolisher | None = None,
) -> RationaleRevalidationReport:
    """Apply the current validator to saved structured DeepSeek responses.

    The original template trajectory and audit remain the source of truth.  This
    permits stricter local validation to fall back individual turns while keeping
    all provider-generated turns that still pass, without a second paid request.
    """

    if config.output_path.exists() or config.output_audit_path.exists():
        raise SFTRebuildError("Refusing to overwrite revalidated rationale artifacts.")
    trajectories = _read_jsonl(config.input_path)
    audits = _read_jsonl(config.input_audit_path)
    source_audits = _read_jsonl(config.source_audit_path)
    if not trajectories or not (
        len(trajectories) == len(audits) == len(source_audits)
    ):
        raise SFTRebuildError("Revalidation inputs must be nonempty and have matching counts.")
    worker = polisher or TrajectoryRationalePolisher(llm_config)
    output_trajectories: list[dict[str, Any]] = []
    output_audits: list[dict[str, Any]] = []
    accepted_turns = 0
    fallback_turns = 0
    for trajectory, audit, source_audit in zip(trajectories, audits, source_audits, strict=True):
        task_id = str(trajectory.get("task_id"))
        if task_id != str(audit.get("task_id")) or task_id != str(source_audit.get("task_id")):
            raise SFTRebuildError("Revalidation task order does not match.")
        source_polish = source_audit.get("rationale_polish")
        if not isinstance(source_polish, Mapping):
            raise SFTRebuildError(f"Saved rationale response is missing for {task_id}.")
        result_trajectory, result_audit = worker.revalidate(
            trajectory,
            audit,
            source_polish.get("raw_response"),
        )
        output_trajectories.append(result_trajectory)
        output_audits.append(result_audit)
        for turn in result_audit["turns"]:
            if turn["rationale_polish_outcome"] == "accepted":
                accepted_turns += 1
            else:
                fallback_turns += 1
    _atomic_jsonl(config.output_path, output_trajectories)
    _atomic_jsonl(config.output_audit_path, output_audits)
    return RationaleRevalidationReport(
        samples=len(output_trajectories),
        accepted_turns=accepted_turns,
        fallback_turns=fallback_turns,
    )


def _request_turns(
    trajectory: Mapping[str, Any], audit: Mapping[str, Any]
) -> list[dict[str, Any]]:
    assistant_messages = [
        message for message in trajectory["messages"] if message.get("role") == "assistant"
    ]
    steps = trajectory["steps"]
    audit_turns = audit["turns"]
    if not (len(assistant_messages) == len(steps) == len(audit_turns)):
        raise SFTRebuildError("Rationale polish assistant/step/audit counts do not align.")
    turns: list[dict[str, Any]] = []
    question_numbers = set(_NUMBER.findall(_question(trajectory)))
    for index, (message, step, audit_turn) in enumerate(
        zip(assistant_messages, steps, audit_turns, strict=True)
    ):
        rationale = str(message.get("content", "")).strip()
        if not rationale:
            raise SFTRebuildError(
                f"Template trajectory {trajectory['task_id']} has an empty rationale at {index}."
            )
        turns.append(
            {
                "step_index": index,
                "rationale_kind": str(audit_turn.get("rationale_kind", "tool_decision")),
                "tool": str(step["action"]["tool"]),
                "arguments": deepcopy(step["action"]["arguments"]),
                "template_rationale": rationale,
                "protected_literals": list(audit_turn.get("protected_literals", [])),
                "allowed_numeric_literals": sorted(
                    question_numbers
                    | set(_NUMBER.findall(rationale))
                    | set(
                        _NUMBER.findall(
                            json.dumps(step["action"]["arguments"], ensure_ascii=False)
                        )
                    )
                ),
            }
        )
    return turns


def _validate_rationale(
    candidate: Any,
    request: Mapping[str, Any],
    all_literals: set[str],
) -> list[str]:
    if not isinstance(candidate, str) or not candidate.strip():
        return ["rationale must be a nonempty string"]
    text = candidate.strip()
    errors: list[str] = []
    if len(text) > 500:
        errors.append("rationale exceeds 500 characters")
    for literal in request["protected_literals"]:
        if literal in _SEMANTIC_LABELS:
            continue
        if literal not in text:
            errors.append(f"missing protected literal: {literal}")
    template = str(request["template_rationale"])
    if not set(_NUMBER.findall(text)).issubset(set(request["allowed_numeric_literals"])):
        errors.append("rationale introduced an unsupported numeric literal")
    for literal in all_literals:
        if literal not in template and literal in text:
            errors.append(f"rationale leaked another turn's literal: {literal}")
    kind = str(request["rationale_kind"])
    tool = str(request["tool"])
    arguments = request["arguments"]
    if kind == "search_evidence" and _PREMATURE_RESULT.search(text):
        errors.append("search rationale claims a result before the search")
    if tool == "save_candidate" and not any(word in text for word in ("保存", "候选", "证据")):
        errors.append("save rationale no longer explains preserving the candidate")
    if tool == "save_candidate":
        purpose = arguments.get("purpose")
        alternatives = {
            "outbound_transport": ("去程",),
            "return_transport": ("返程",),
            "attraction": ("景点", "游览"),
            "meal": ("用餐", "餐厅", "餐饮"),
            "hotel": ("住宿", "酒店"),
        }.get(purpose, (str(purpose),))
        if not any(word in text for word in alternatives):
            errors.append("save rationale changed or omitted the candidate purpose")
    if tool == "remove_candidate" and not has_visible_price_comparison(text):
        errors.append("remove rationale omitted the visible price comparison")
    if tool == "get_route" and not any(word in text for word in ("路线", "衔接", "交通")):
        errors.append("route rationale no longer explains the connection")
    if tool == "submit_plan" and not any(word in text for word in ("提交", "递交")):
        errors.append("submit rationale no longer explains submission")
    if tool == "search_intercity_transport":
        mode = arguments.get("mode")
        alternatives = {
            "train": ("火车", "列车", "车次", "高铁", "动车"),
            "airplane": ("飞机", "航班", "航空"),
        }.get(mode, (str(mode),))
        if not any(word in text for word in alternatives):
            errors.append("transport-search rationale changed or omitted the requested mode")
    if tool == "get_route":
        mode = arguments.get("mode")
        alternatives = {
            "taxi": ("出租车", "打车"),
            "metro": ("地铁", "轨道交通"),
            "walk": ("步行", "走路"),
        }.get(mode, (str(mode),))
        if not any(word in text for word in alternatives):
            errors.append("route rationale changed or omitted the requested mode")
    if tool == "submit_plan":
        semantic_requirements = {
            "往返交通": (("往返",), ("去程", "返程")),
            "景点": (("景点",), ("游览",)),
            "完整路线": (("路线",), ("衔接",)),
            "住宿": (("住宿",), ("酒店",)),
            "餐饮": (("餐饮",), ("用餐",), ("餐厅",)),
        }
        for label in request["protected_literals"]:
            alternatives = semantic_requirements.get(label)
            if alternatives and not any(
                all(word in text for word in group) for group in alternatives
            ):
                errors.append(f"submit rationale omitted evidence category: {label}")
    if kind == "loop_recovery" and not any(
        word in text for word in ("重复", "无需再", "不必再", "停止")
    ):
        errors.append("loop recovery no longer explains stopping repetition")
    return errors


def _proposed_rationales(payload: Mapping[str, Any], expected: int) -> list[str]:
    if set(payload) != {"turns"} or not isinstance(payload["turns"], list):
        raise SFTRebuildError("Rationale polisher payload must contain only a turns array.")
    turns = payload["turns"]
    if len(turns) != expected:
        raise SFTRebuildError("Rationale polisher changed the number of turns.")
    proposed: list[str] = []
    for index, turn in enumerate(turns):
        if not isinstance(turn, Mapping) or set(turn) != {"step_index", "rationale"}:
            raise SFTRebuildError("Rationale polisher returned an invalid turn object.")
        if turn["step_index"] != index:
            raise SFTRebuildError("Rationale polisher changed turn indices or ordering.")
        proposed.append(turn["rationale"])
    return proposed


def _tool_payload(response: Any) -> dict[str, Any]:
    choices = _get(response, "choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise SFTRebuildError("Rationale polisher must return exactly one choice.")
    message = _get(choices[0], "message")
    tool_calls = _get(message, "tool_calls")
    if not isinstance(tool_calls, list) or len(tool_calls) != 1:
        raise SFTRebuildError("Rationale polisher must make exactly one function call.")
    function = _get(tool_calls[0], "function")
    if _get(function, "name") != "polish_trajectory_rationales":
        raise SFTRebuildError("Rationale polisher called an unexpected function.")
    arguments = _get(function, "arguments")
    if not isinstance(arguments, str):
        raise SFTRebuildError("Rationale polisher arguments must be serialized JSON.")
    try:
        payload = json.loads(arguments)
    except json.JSONDecodeError as error:
        raise SFTRebuildError("Rationale polisher returned invalid JSON arguments.") from error
    if not isinstance(payload, dict):
        raise SFTRebuildError("Rationale polisher payload must be an object.")
    return payload


def _question(trajectory: Mapping[str, Any]) -> str:
    return next(
        str(message.get("content", ""))
        for message in trajectory["messages"]
        if message.get("role") == "user"
    )


def _usage(response: Any | None) -> dict[str, int]:
    usage = _get(response, "usage")
    return {
        key: value
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        if isinstance((value := _get(usage, key)), int)
    }


def _response_dict(response: Any | None) -> Any:
    if response is None:
        return None
    if isinstance(response, Mapping):
        return deepcopy(dict(response))
    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    return repr(response)


def _validate_work_manifest(config: RationalePolishConfig) -> None:
    path = config.work_dir / "manifest.json"
    expected = {
        "rationale_polisher_version": RATIONALE_POLISHER_VERSION,
        "rationale_prompt_version": RATIONALE_PROMPT_VERSION,
        "input_path": str(config.input_path.resolve()),
        "input_sha256": _sha256(config.input_path),
        "input_audit_path": str(config.input_audit_path.resolve()),
        "input_audit_sha256": _sha256(config.input_audit_path),
        "task_ids": list(config.task_ids),
    }
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        if current != expected:
            raise SFTRebuildError("Rationale polish work directory belongs to different inputs.")
    else:
        _atomic_json(path, expected)


def _completed_records(work_dir: Path, count: int) -> dict[int, dict[str, Any]]:
    completed: dict[int, dict[str, Any]] = {}
    for path in sorted((work_dir / "records").glob("*.json")):
        try:
            index = int(path.stem)
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError, json.JSONDecodeError) as error:
            raise SFTRebuildError(f"Invalid rationale polish checkpoint: {path}") from error
        if index < 0 or index >= count or not isinstance(payload, dict):
            raise SFTRebuildError(f"Out-of-range rationale polish checkpoint: {path}")
        completed[index] = payload
    return completed


def _record_path(work_dir: Path, index: int) -> Path:
    return work_dir / "records" / f"{index:06d}.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise SFTRebuildError(f"Cannot read rationale polish JSONL: {path}") from error


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _get(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)
