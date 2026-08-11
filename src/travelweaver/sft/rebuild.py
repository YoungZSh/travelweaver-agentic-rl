"""Deterministically rebuild accepted rollouts into replay-verified SFT conversations."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

from ..env import (
    DEFAULT_MAX_VALID_STEPS,
    ChinaTravelBackend,
    ScenarioBackend,
    ScenarioSpec,
    TravelWeaverEnv,
)
from ..errors import SFTRebuildError, TaskNotFoundError
from ..rollout.api_agent import (
    SUPPORTED_TRAJECTORY_VERSIONS,
    SYSTEM_PROMPT,
    USER_CONTENT_FORMAT,
    render_system_prompt,
    render_task_user_content,
)
from ..rollout.tool_response import (
    DEFAULT_TOOL_RESPONSE_MODE,
    MODEL_TOOL_RESPONSE_VERSION,
    serialize_model_tool_response,
    validate_tool_response_mode,
)
from ..synthesis.polisher import validate_surface
from ..synthesis.render import render_canonical
from ..tasks import TaskBlueprint, TaskSurface, materialize_task_spec
from .ordering import order_tool_arguments, order_tool_schemas

SFT_FORMAT_VERSION = "travelweaver-sft-v5"
SFTSupervisionMode = Literal[
    "action_only", "react", "react_recovery", "action_selective"
]
SFT_SUPERVISION_MODES: tuple[SFTSupervisionMode, ...] = (
    "action_only",
    "react",
    "react_recovery",
    "action_selective",
)
DEFAULT_SFT_SUPERVISION_MODE: SFTSupervisionMode = "action_only"


def validate_supervision_mode(value: str) -> SFTSupervisionMode:
    """Return a typed supervision mode or reject an unknown value."""

    if value not in SFT_SUPERVISION_MODES:
        choices = ", ".join(SFT_SUPERVISION_MODES)
        raise ValueError(f"Unknown SFT supervision mode {value!r}; expected: {choices}.")
    return cast(SFTSupervisionMode, value)


@dataclass(frozen=True)
class SFTSource:
    """One generated-task snapshot paired with its rollout records."""

    task_dir: Path
    rollout_path: Path


@dataclass(frozen=True)
class SFTRebuildConfig:
    """Inputs and output policy for deterministic SFT reconstruction."""

    sources: tuple[SFTSource, ...]
    output_dir: Path
    repair_surface_semantics: bool = False
    tool_response_mode: str = DEFAULT_TOOL_RESPONSE_MODE
    supervision_mode: str = DEFAULT_SFT_SUPERVISION_MODE

    def __post_init__(self) -> None:
        if not self.sources:
            raise ValueError("At least one SFT source is required.")
        validate_tool_response_mode(self.tool_response_mode)
        mode = validate_supervision_mode(self.supervision_mode)
        if (
            mode in {"react", "react_recovery", "action_selective"}
            and self.repair_surface_semantics
        ):
            raise ValueError("ReAct SFT cannot change the user query after rollout generation.")


@dataclass(frozen=True)
class SFTRebuildReport:
    """Summary of one completed reconstruction."""

    input_rows: int
    reward_accepted_rows: int
    accepted_rows: int
    mode_excluded_rows: int
    samples: int
    replayed_valid_actions: int
    invalid_actions_removed: int
    invalid_actions_retained: int
    masked_assistant_turns: int
    samples_with_invalid_actions: int
    cursor_remaps: int
    surface_repairs: int
    local_surface_repairs: int
    canonical_surface_fallbacks: int
    output_dir: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _PreparedTask:
    public: dict[str, Any]
    oracle: dict[str, Any]
    repair_kind: str | None
    original_query: str


@dataclass(frozen=True)
class _SourceSnapshot:
    public: dict[str, dict[str, Any]]
    oracle: dict[str, dict[str, Any]]
    records: dict[str, dict[str, Any]]


class _SingleTaskStore:
    def __init__(self, public: dict[str, Any], oracle: dict[str, Any]) -> None:
        self.public = deepcopy(public)
        self.oracle = deepcopy(oracle)

    def choose(self, seed: int | None = None) -> str:
        del seed
        return str(self.public["uid"])

    def get_public(self, task_id: str) -> dict[str, Any]:
        if task_id != self.public["uid"]:
            raise TaskNotFoundError(f"Unknown task id: {task_id}")
        return deepcopy(self.public)

    def get_oracle(self, task_id: str) -> dict[str, Any]:
        if task_id != self.oracle["uid"]:
            raise TaskNotFoundError(f"Unknown task id: {task_id}")
        return deepcopy(self.oracle)


def rebuild_sft_dataset(
    config: SFTRebuildConfig,
    *,
    base_backend: Any | None = None,
) -> SFTRebuildReport:
    """Filter, replay, and atomically write trainer-neutral SFT samples and audit."""

    _validate_sources(config.sources)
    output = config.output_dir
    destinations = {
        "neutral": output / "neutral.jsonl",
        "audit": output / "audit.jsonl",
        "exclusions": output / "exclusions.jsonl",
        "manifest": output / "manifest.json",
        "preview": output / "preview.md",
    }
    existing = [str(path) for path in destinations.values() if path.exists()]
    if existing:
        raise SFTRebuildError(f"Refusing to overwrite existing SFT artifacts: {existing}")
    reward_records: list[tuple[SFTSource, dict[str, Any]]] = []
    snapshots: dict[SFTSource, _SourceSnapshot] = {}
    input_rows = 0
    for source in config.sources:
        snapshots[source] = _load_source_snapshot(source.task_dir)
        rows = _read_jsonl(source.rollout_path)
        input_rows += len(rows)
        reward_records.extend((source, row) for row in rows if _is_accepted(row))
    supervision_mode = validate_supervision_mode(config.supervision_mode)
    records: list[tuple[SFTSource, dict[str, Any]]] = []
    exclusions: list[dict[str, Any]] = []
    for source, row in reward_records:
        reason = _mode_exclusion_reason(row, supervision_mode)
        if reason is None:
            records.append((source, row))
            continue
        exclusions.append(
            {
                "format_version": SFT_FORMAT_VERSION,
                "task_id": row.get("task_id"),
                "episode_id": row.get("episode_id"),
                "source_rollout": str(source.rollout_path.resolve()),
                "supervision_mode": supervision_mode,
                "reason": reason,
            }
        )
    records.sort(key=lambda item: (str(item[1]["task_id"]), str(item[1]["episode_id"])))
    exclusions.sort(key=lambda item: (str(item["task_id"]), str(item["episode_id"])))
    task_ids = [str(row["task_id"]) for _, row in records]
    if len(task_ids) != len(set(task_ids)):
        duplicates = sorted(task_id for task_id, count in Counter(task_ids).items() if count > 1)
        raise SFTRebuildError(f"Accepted sources contain duplicate task ids: {duplicates[:5]}")

    backend = base_backend if base_backend is not None else ChinaTravelBackend()
    samples: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    invalid_removed = 0
    repaired = Counter()
    for source, row in records:
        prepared = _prepare_task(
            snapshots[source],
            str(row["task_id"]),
            repair_surface_semantics=config.repair_surface_semantics,
        )
        sample, audit = _rebuild_one(
            source,
            row,
            prepared,
            backend,
            tool_response_mode=config.tool_response_mode,
            supervision_mode=supervision_mode,
        )
        samples.append(sample)
        audits.append(audit)
        invalid_removed += int(audit["invalid_actions_removed"])
        if prepared.repair_kind is not None:
            repaired[prepared.repair_kind] += 1

    output.mkdir(parents=True, exist_ok=True)

    report = SFTRebuildReport(
        input_rows=input_rows,
        reward_accepted_rows=len(reward_records),
        accepted_rows=len(records),
        mode_excluded_rows=len(exclusions),
        samples=len(samples),
        replayed_valid_actions=sum(int(audit["replayed_valid_steps"]) for audit in audits),
        invalid_actions_removed=invalid_removed,
        invalid_actions_retained=sum(
            int(audit["invalid_actions_retained"]) for audit in audits
        ),
        masked_assistant_turns=sum(
            int(audit["masked_assistant_turns"]) for audit in audits
        ),
        samples_with_invalid_actions=sum(
            int(
                int(audit["invalid_actions_removed"]) > 0
                or int(audit["invalid_actions_retained"]) > 0
            )
            for audit in audits
        ),
        cursor_remaps=sum(int(audit["cursor_remaps"]) for audit in audits),
        surface_repairs=sum(repaired.values()),
        local_surface_repairs=repaired["local_clause_patch"],
        canonical_surface_fallbacks=repaired["canonical_fallback"],
        output_dir=str(output.resolve()),
    )
    manifest = {
        "format_version": SFT_FORMAT_VERSION,
        "status": "neutral_complete",
        "config": {
            "repair_surface_semantics": config.repair_surface_semantics,
            "tool_response_mode": config.tool_response_mode,
            "supervision_mode": supervision_mode,
            "model_tool_response_version": MODEL_TOOL_RESPONSE_VERSION,
            "user_content_format": USER_CONTENT_FORMAT,
            "sources": [
                {
                    "task_dir": str(source.task_dir.resolve()),
                    "rollout_path": str(source.rollout_path.resolve()),
                    "rollout_sha256": _sha256_file(source.rollout_path),
                }
                for source in config.sources
            ],
        },
        "report": report.to_dict(),
        "tool_counts": dict(sorted(_tool_counts(samples).items())),
        "neutral_sha256": _jsonl_digest(samples),
        "audit_sha256": _jsonl_digest(audits),
        "exclusions_sha256": _jsonl_digest(exclusions),
    }
    _atomic_jsonl(destinations["neutral"], samples)
    _atomic_jsonl(destinations["audit"], audits)
    _atomic_jsonl(destinations["exclusions"], exclusions)
    _atomic_json(destinations["manifest"], manifest)
    _atomic_text(destinations["preview"], _preview(samples, audits))
    return report


def _validate_sources(sources: tuple[SFTSource, ...]) -> None:
    for source in sources:
        required = (
            source.task_dir / "tasks.public.jsonl",
            source.task_dir / "tasks.oracle.jsonl",
            source.task_dir / "records",
            source.rollout_path,
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise SFTRebuildError(f"SFT source is incomplete: {missing}")


def _is_accepted(row: dict[str, Any]) -> bool:
    detail = row.get("reward_detail")
    return bool(
        row.get("trajectory_version") in SUPPORTED_TRAJECTORY_VERSIONS
        and row.get("termination_reason") == "plan_submitted"
        and row.get("final_reward") == 1.0
        and row.get("rft_accepted") is True
        and isinstance(detail, dict)
        and detail.get("reward_valid") is True
        and detail.get("all_hard_pass") is True
    )


def _mode_exclusion_reason(
    row: dict[str, Any],
    supervision_mode: SFTSupervisionMode,
) -> str | None:
    if supervision_mode == "action_only":
        return None
    if _source_thinking(row) != "disabled":
        return "react_requires_source_thinking_disabled"
    steps = row.get("steps")
    if not isinstance(steps, list) or not steps:
        return "react_requires_nonempty_steps"
    if supervision_mode in {"react", "action_selective"} and any(
        not _step_is_valid(step) for step in steps
    ):
        return "react_invalid_action_not_supported"
    if supervision_mode == "action_selective":
        mask = row.get("assistant_loss_mask")
        if (
            not isinstance(mask, list)
            or len(mask) != len(steps)
            or any(not isinstance(value, bool) for value in mask)
            or not mask
            or mask[-1] is not True
        ):
            return "action_selective_requires_explicit_loss_mask"
    try:
        _react_source_context(row)
        contents = _react_content_by_call_id(row)
    except SFTRebuildError as error:
        return f"react_message_alignment_error:{error}"
    if supervision_mode in {"react", "react_recovery"} and not any(
        content.strip() for content in contents.values()
    ):
        return "react_requires_visible_assistant_content"
    return None


def _step_is_valid(step: Any) -> bool:
    if not isinstance(step, dict):
        return False
    result = step.get("result")
    info = result.get("info") if isinstance(result, dict) else None
    return isinstance(info, dict) and info.get("valid_action") is True


def _source_thinking(row: dict[str, Any]) -> str | None:
    batch_metadata = row.get("batch_metadata")
    value = batch_metadata.get("thinking") if isinstance(batch_metadata, dict) else None
    return str(value) if isinstance(value, str) else None


def _react_source_context(row: dict[str, Any]) -> tuple[str, str, int]:
    task_id = str(row.get("task_id"))
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise SFTRebuildError(f"Trajectory {task_id} has no source prompt context.")
    system, user = messages[:2]
    supported_prompts = {
        render_system_prompt(35): 35,
        SYSTEM_PROMPT: DEFAULT_MAX_VALID_STEPS,
        render_system_prompt(100): 100,
    }
    system_content = system.get("content") if isinstance(system, dict) else None
    if (
        not isinstance(system, dict)
        or system.get("role") != "system"
        or not isinstance(system_content, str)
        or system_content not in supported_prompts
    ):
        raise SFTRebuildError(f"Trajectory {task_id} used a different system prompt.")
    content = user.get("content") if isinstance(user, dict) and user.get("role") == "user" else None
    if not isinstance(content, str) or not content.strip():
        raise SFTRebuildError(f"Trajectory {task_id} has no natural-language source query.")
    return system_content, content, supported_prompts[system_content]


def _react_content_by_call_id(row: dict[str, Any]) -> dict[str, str]:
    """Align visible no-thinking assistant text with each executed source action."""

    task_id = str(row.get("task_id"))
    raw_messages = row.get("messages")
    raw_steps = row.get("steps")
    if not isinstance(raw_messages, list) or not isinstance(raw_steps, list):
        raise SFTRebuildError(f"Trajectory {task_id} has no replayable messages/steps.")
    assistant_by_id: dict[str, tuple[str, str, dict[str, Any]]] = {}
    for message in raw_messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        reasoning_content = message.get("reasoning_content")
        if reasoning_content is not None and reasoning_content != "":
            raise SFTRebuildError(f"Trajectory {task_id} contains hidden reasoning content.")
        calls = message.get("tool_calls")
        if not isinstance(calls, list) or len(calls) != 1 or not isinstance(calls[0], dict):
            raise SFTRebuildError(
                f"Trajectory {task_id} assistant turns must contain exactly one tool call."
            )
        call = calls[0]
        call_id = call.get("id")
        function = call.get("function")
        content = message.get("content")
        if not isinstance(call_id, str) or not isinstance(function, dict):
            raise SFTRebuildError(f"Trajectory {task_id} has an invalid assistant tool call.")
        if content is None:
            content = ""
        if not isinstance(content, str):
            raise SFTRebuildError(f"Trajectory {task_id} has non-text assistant content.")
        raw_arguments = function.get("arguments")
        if isinstance(raw_arguments, str):
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                arguments = {}
        else:
            arguments = raw_arguments
        if not isinstance(arguments, dict) or not isinstance(function.get("name"), str):
            raise SFTRebuildError(f"Trajectory {task_id} has non-object source arguments.")
        if call_id in assistant_by_id:
            raise SFTRebuildError(f"Trajectory {task_id} repeats tool call id {call_id}.")
        assistant_by_id[call_id] = (content, str(function["name"]), arguments)

    contents: dict[str, str] = {}
    for step in raw_steps:
        if not isinstance(step, dict):
            raise SFTRebuildError(f"Trajectory {task_id} contains a non-object step.")
        tool_call = step.get("tool_call")
        action = step.get("action")
        call_id = tool_call.get("id") if isinstance(tool_call, dict) else None
        if not isinstance(call_id, str) or not isinstance(action, dict):
            raise SFTRebuildError(f"Trajectory {task_id} cannot align a source step.")
        try:
            content, tool_name, arguments = assistant_by_id[call_id]
        except KeyError as error:
            raise SFTRebuildError(
                f"Trajectory {task_id} has no assistant message for call {call_id}."
            ) from error
        source_arguments = action.get("arguments")
        legacy_malformed_match = (
            not _step_is_valid(step)
            and not isinstance(source_arguments, dict)
            and arguments == {}
        )
        if tool_name != action.get("tool") or (
            arguments != source_arguments and not legacy_malformed_match
        ):
            raise SFTRebuildError(
                f"Trajectory {task_id} source message/action mismatch for call {call_id}."
            )
        contents[call_id] = content
    if len(contents) != len(assistant_by_id):
        raise SFTRebuildError(f"Trajectory {task_id} has unexecuted assistant tool calls.")
    return contents


def _prepare_task(
    snapshot: _SourceSnapshot,
    task_id: str,
    *,
    repair_surface_semantics: bool,
) -> _PreparedTask:
    try:
        public = deepcopy(snapshot.public[task_id])
        oracle = deepcopy(snapshot.oracle[task_id])
        record = snapshot.records[task_id]
    except KeyError as error:
        raise SFTRebuildError(f"Generated task snapshot is missing {task_id}.") from error
    original_query = str(public["query"])
    if not repair_surface_semantics:
        return _PreparedTask(public, oracle, None, original_query)
    blueprint = TaskBlueprint.from_dict(record["blueprint"])
    has_other_restaurant_constraint = any(
        constraint.scope == "restaurant" and constraint.kind != "category_budget"
        for constraint in blueprint.constraints
    )
    affected = {
        constraint.id
        for constraint in blueprint.constraints
        if (
            constraint.kind == "category_budget"
            and constraint.scope == "restaurant"
            and not has_other_restaurant_constraint
        )
        or (constraint.kind == "transport_mode" and constraint.scope == "innercity_route")
    }
    if not affected:
        return _PreparedTask(public, oracle, None, original_query)

    slot = dict(record["slot"])
    canonical = render_canonical(
        blueprint,
        style_profile=str(slot["surface_style"]),
        validation_profile=str(slot["validation_profile"]),
    )
    old_surface = TaskSurface.from_dict(record["surface"])
    try:
        query, mentions = _patch_surface(old_surface, canonical.clauses, affected)
        surface = validate_surface(
            blueprint,
            canonical,
            {
                "query": query,
                "mentions": mentions,
                "preference_mentions": [
                    {
                        "preference_id": mention.preference_id,
                        "text": mention.text,
                    }
                    for mention in old_surface.preference_mentions
                ],
            },
            model="deterministic-v4-clause-repair",
            validation_profile=str(slot["validation_profile"]),
            validation_policy="minimal_semantic",
        )
        repair_kind = "local_clause_patch"
    except Exception:  # noqa: BLE001 - deterministic full-surface fallback is audited.
        surface = validate_surface(
            blueprint,
            canonical,
            {
                "query": canonical.query,
                "mentions": [
                    {"constraint_id": key, "text": value}
                    for key, value in canonical.clauses.items()
                ],
                "preference_mentions": [
                    {"preference_id": key, "text": value}
                    for key, value in canonical.preference_clauses.items()
                ],
            },
            model="deterministic-v4-canonical-fallback",
            validation_profile=str(slot["validation_profile"]),
            validation_policy="minimal_semantic",
        )
        repair_kind = "canonical_fallback"

    spec = materialize_task_spec(blueprint, surface, task_id=task_id)
    public = {**public, "query": surface.public_query, "surface_id": surface.surface_id}
    oracle = {
        **oracle,
        "surface_id": surface.surface_id,
        "task_spec": spec.to_dict(),
    }
    return _PreparedTask(public, oracle, repair_kind, original_query)


def _patch_surface(
    surface: TaskSurface,
    canonical_clauses: dict[str, str],
    affected: set[str],
) -> tuple[str, list[dict[str, str]]]:
    replacements: list[tuple[int, int, str]] = []
    boundaries = "，；。！？：\n"
    all_spans = [
        (mention.start, mention.end, mention.constraint_id) for mention in surface.mentions
    ]
    for mention in surface.mentions:
        if mention.constraint_id not in affected:
            continue
        left = max(
            (surface.public_query.rfind(marker, 0, mention.start) for marker in boundaries),
            default=-1,
        )
        right_candidates = [
            position
            for marker in boundaries
            if (position := surface.public_query.find(marker, mention.end)) >= 0
        ]
        start = left + 1
        end = min(right_candidates, default=len(surface.public_query))
        overlaps_other = any(
            other_id != mention.constraint_id
            and other_start < end
            and other_end > start
            for other_start, other_end, other_id in all_spans
        )
        if overlaps_other:
            start, end = mention.start, mention.end
        replacements.append((start, end, canonical_clauses[mention.constraint_id]))
    for previous, current in zip(
        sorted(replacements), sorted(replacements)[1:], strict=False
    ):
        if previous[1] > current[0]:
            raise SFTRebuildError("Affected surface clauses overlap.")
    query = surface.public_query
    for start, end, replacement in sorted(replacements, reverse=True):
        query = query[:start] + replacement + query[end:]
    mentions = [
        {
            "constraint_id": mention.constraint_id,
            "text": (
                canonical_clauses[mention.constraint_id]
                if mention.constraint_id in affected
                else mention.text
            ),
        }
        for mention in surface.mentions
    ]
    return query, mentions


def _rebuild_one(
    source: SFTSource,
    row: dict[str, Any],
    task: _PreparedTask,
    base_backend: Any,
    *,
    tool_response_mode: str = DEFAULT_TOOL_RESPONSE_MODE,
    supervision_mode: str = DEFAULT_SFT_SUPERVISION_MODE,
) -> tuple[dict[str, Any], dict[str, Any]]:
    response_mode = validate_tool_response_mode(tool_response_mode)
    selected_supervision = validate_supervision_mode(supervision_mode)
    task_id = str(row["task_id"])
    is_react = selected_supervision in {"react", "react_recovery", "action_selective"}
    react_content = _react_content_by_call_id(row) if is_react else {}
    output_system_prompt = SYSTEM_PROMPT
    max_valid_steps = DEFAULT_MAX_VALID_STEPS
    if is_react:
        output_system_prompt, source_user_content, max_valid_steps = _react_source_context(row)
        rebuilt_user_content = render_task_user_content(task.public)
        if source_user_content != rebuilt_user_content:
            raise SFTRebuildError(
                f"ReAct trajectory {task_id} source and rebuilt user queries differ."
            )
    raw_scenario = task.oracle.get("scenario")
    if not isinstance(raw_scenario, dict):
        raise SFTRebuildError(f"Generated task {task_id} has no scenario.")
    scenario = ScenarioSpec.from_dict(raw_scenario)
    store = _SingleTaskStore(task.public, task.oracle)
    env = TravelWeaverEnv(  # type: ignore[arg-type]
        ScenarioBackend(base_backend, scenario),
        store,
        max_valid_steps=max_valid_steps,
    )
    model_tools = order_tool_schemas(env.tool_schemas())
    source_tools = row.get("tools")
    if not isinstance(source_tools, list) or _canonical_json(source_tools) != _canonical_json(
        model_tools
    ):
        raise SFTRebuildError(f"Trajectory {task_id} tool schemas differ from the environment.")
    source_episode_id = str(row["episode_id"])
    cursor_old_to_new: dict[str, str] = {}
    cursor_new_to_old: dict[str, str] = {}
    messages: list[dict[str, Any]] = []
    assistant_loss_mask: list[bool] = []
    invalid_steps: list[dict[str, Any]] = []
    valid_source_steps: list[dict[str, Any]] = []
    replayed_steps = 0
    argument_normalizations = 0
    try:
        reset = env.reset(task_id=task_id, seed=0).to_dict()
        runtime_episode_id = str(reset["episode_id"])
        reset = _replace_runtime_values(
            reset, runtime_episode_id, source_episode_id, cursor_new_to_old
        )
        messages.extend(
            [
                {"role": "system", "content": output_system_prompt},
                {
                    "role": "user",
                    "content": render_task_user_content(reset["task"]),
                },
            ]
        )
        raw_steps = row.get("steps")
        if not isinstance(raw_steps, list):
            raise SFTRebuildError(f"Trajectory {task_id} has no step list.")
        for position, step in enumerate(raw_steps):
            if not isinstance(step, dict):
                raise SFTRebuildError(f"Trajectory {task_id} contains a non-object step.")
            source_result = step.get("result")
            valid = _step_is_valid(step)
            if not valid:
                if selected_supervision == "react":
                    raise SFTRebuildError(
                        f"ReAct trajectory {task_id} contains an invalid source action."
                    )
                invalid_steps.append(
                    _invalid_audit(step, valid_source_steps, raw_steps[position + 1 :])
                )
                if selected_supervision == "action_only":
                    continue
            else:
                valid_source_steps.append(step)
            action = deepcopy(step.get("action"))
            if not isinstance(action, dict) or not isinstance(action.get("tool"), str):
                raise SFTRebuildError(f"Source step for {task_id} has no canonical action object.")
            if not isinstance(action.get("arguments"), dict):
                if valid or selected_supervision != "react_recovery":
                    raise SFTRebuildError(
                        f"Source step for {task_id} has non-object arguments."
                    )
                action["arguments"] = {}
                argument_normalizations += 1
            tool_name = str(action["tool"])
            has_tool_schema = any(
                isinstance(tool, dict)
                and isinstance(tool.get("function"), dict)
                and tool["function"].get("name") == tool_name
                for tool in model_tools
            )
            if not has_tool_schema and not valid and selected_supervision == "react_recovery":
                # An unknown tool name is itself a useful recoverable model error. It has no
                # schema-directed ordering, and the corresponding assistant turn is masked.
                ordered_arguments = deepcopy(action["arguments"])
            else:
                try:
                    ordered_arguments = order_tool_arguments(
                        tool_name, action["arguments"], model_tools
                    )
                except ValueError as error:
                    raise SFTRebuildError(
                        f"Cannot order source arguments for {task_id}: {error}"
                    ) from error
            output_action = {"tool": str(action["tool"]), "arguments": ordered_arguments}
            internal_action = deepcopy(action)
            if action.get("tool") == "next_page":
                old_cursor = str(action["arguments"].get("cursor", ""))
                if old_cursor in cursor_old_to_new:
                    internal_action["arguments"]["cursor"] = cursor_old_to_new[old_cursor]
                elif valid or selected_supervision != "react_recovery":
                    raise SFTRebuildError(
                        f"Cannot remap cursor {old_cursor!r} for {task_id}."
                    )
                # A source-invalid unknown cursor has no runtime equivalent. Replaying the
                # unchanged opaque value must still produce the same invalid-action class.
            result = env.step(internal_action)
            replayed_steps += 1
            replay_valid = result.info.get("valid_action") is True
            if replay_valid != valid:
                error = result.observation.error or {}
                raise SFTRebuildError(
                    f"Replay changed action validity for {task_id}: {action}; {error}"
                )
            if valid:
                _record_cursor_mapping(
                    source_result,
                    result.to_dict(),
                    cursor_old_to_new,
                    cursor_new_to_old,
                )
            elif _step_error_code(step) != _result_error_code(result.to_dict()):
                raise SFTRebuildError(
                    f"Replay changed invalid-action error code for {task_id}."
                )
            tool_call = step.get("tool_call")
            source_call_id = tool_call.get("id") if isinstance(tool_call, dict) else None
            visible_content = (
                react_content[str(source_call_id)]
                if is_react
                else ""
            )
            selected_loss = (
                bool(row["assistant_loss_mask"][position])
                if selected_supervision == "action_selective"
                else valid
            )
            messages.append(_assistant_message(output_action, content=visible_content))
            assistant_loss_mask.append(selected_loss)
            if not result.terminated and not result.truncated:
                model_tool_response = serialize_model_tool_response(
                    result,
                    mode=response_mode,
                )
                normalized = _replace_runtime_values(
                    model_tool_response,
                    runtime_episode_id,
                    source_episode_id,
                    cursor_new_to_old,
                )
                messages.append({"role": "tool", "content": _model_json(normalized)})
        terminal = result if valid_source_steps else None
        if terminal is None or not terminal.terminated or terminal.truncated:
            raise SFTRebuildError(f"Replayed trajectory {task_id} did not terminate normally.")
        detail = terminal.info.get("reward_detail")
        if (
            terminal.info.get("termination_reason") != "plan_submitted"
            or terminal.reward != 1.0
            or not isinstance(detail, dict)
            or detail.get("reward_valid") is not True
            or detail.get("all_hard_pass") is not True
        ):
            raise SFTRebuildError(f"Replayed trajectory {task_id} did not retain Reward 1.0.")
    finally:
        env.close()

    sample_id = "tw_sft_" + hashlib.sha256(
        (
            f"{SFT_FORMAT_VERSION}:{MODEL_TOOL_RESPONSE_VERSION}:"
            f"{response_mode}:{selected_supervision}:{task_id}:{source_episode_id}"
        ).encode()
    ).hexdigest()[:24]
    sample = {
        "format_version": SFT_FORMAT_VERSION,
        "sample_id": sample_id,
        "task_id": task_id,
        "tool_response_mode": response_mode,
        "model_tool_response_version": MODEL_TOOL_RESPONSE_VERSION,
        "user_content_format": USER_CONTENT_FORMAT,
        "supervision_mode": selected_supervision,
        "assistant_loss_mask": assistant_loss_mask,
        "messages": messages,
        "tools": model_tools,
        "enable_thinking": False,
    }
    audit = {
        "format_version": SFT_FORMAT_VERSION,
        "sample_id": sample_id,
        "task_id": task_id,
        "episode_id": source_episode_id,
        "source_task_dir": str(source.task_dir.resolve()),
        "source_rollout": str(source.rollout_path.resolve()),
        "source_model": row.get("model"),
        "source_trajectory_version": row.get("trajectory_version"),
        "tool_response_mode": response_mode,
        "model_tool_response_version": MODEL_TOOL_RESPONSE_VERSION,
        "user_content_format": USER_CONTENT_FORMAT,
        "supervision_mode": selected_supervision,
        "source_thinking": _source_thinking(row),
        "source_step_count": len(row.get("steps", [])),
        "replayed_valid_steps": len(valid_source_steps),
        "replayed_steps": replayed_steps,
        "invalid_actions_removed": (
            len(invalid_steps) if selected_supervision == "action_only" else 0
        ),
        "invalid_actions_retained": (
            len(invalid_steps) if selected_supervision == "react_recovery" else 0
        ),
        "masked_assistant_turns": sum(not value for value in assistant_loss_mask),
        "legacy_argument_normalizations": argument_normalizations,
        "invalid_actions": invalid_steps,
        "assistant_turns_with_content": sum(
            message["role"] == "assistant" and bool(message["content"].strip())
            for message in messages
        ),
        "surface_repair": task.repair_kind,
        "original_query": task.original_query if task.repair_kind else None,
        "rebuilt_query": task.public["query"] if task.repair_kind else None,
        "cursor_remaps": len(cursor_old_to_new),
        "replay_reward": 1.0,
        "all_hard_pass": True,
        "message_count": len(messages),
    }
    return sample, audit


def _assistant_message(action: dict[str, Any], *, content: str = "") -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": str(action["tool"]),
                    "arguments": deepcopy(action["arguments"]),
                },
            }
        ],
    }


def _invalid_audit(
    step: dict[str, Any],
    valid_steps: list[dict[str, Any]],
    remaining_steps: list[Any],
) -> dict[str, Any]:
    action = step.get("action") if isinstance(step.get("action"), dict) else {}
    result = step.get("result") if isinstance(step.get("result"), dict) else {}
    observation = result.get("observation") if isinstance(result.get("observation"), dict) else {}
    error = observation.get("error") if isinstance(observation.get("error"), dict) else {}
    next_valid_tool = None
    for candidate in remaining_steps:
        if not isinstance(candidate, dict):
            continue
        candidate_result = candidate.get("result")
        candidate_info = (
            candidate_result.get("info") if isinstance(candidate_result, dict) else None
        )
        if isinstance(candidate_info, dict) and candidate_info.get("valid_action") is True:
            candidate_action = candidate.get("action")
            next_valid_tool = (
                candidate_action.get("tool") if isinstance(candidate_action, dict) else None
            )
            break
    return {
        "index": step.get("index"),
        "tool": action.get("tool"),
        "previous_valid_tool": (
            valid_steps[-1].get("action", {}).get("tool") if valid_steps else None
        ),
        "next_valid_tool": next_valid_tool,
        "error_code": error.get("code"),
        "error": error.get("message"),
    }


def _step_error_code(step: dict[str, Any]) -> str | None:
    result = step.get("result")
    observation = result.get("observation") if isinstance(result, dict) else None
    error = observation.get("error") if isinstance(observation, dict) else None
    code = error.get("code") if isinstance(error, dict) else None
    return str(code) if isinstance(code, str) else None


def _result_error_code(result: dict[str, Any]) -> str | None:
    observation = result.get("observation")
    error = observation.get("error") if isinstance(observation, dict) else None
    code = error.get("code") if isinstance(error, dict) else None
    return str(code) if isinstance(code, str) else None


def _record_cursor_mapping(
    source_result: dict[str, Any],
    replay_result: dict[str, Any],
    old_to_new: dict[str, str],
    new_to_old: dict[str, str],
) -> None:
    old = _next_cursor(source_result)
    new = _next_cursor(replay_result)
    if (old is None) != (new is None):
        raise SFTRebuildError("Replay changed pagination availability.")
    if old is not None and new is not None:
        old_to_new[old] = new
        new_to_old[new] = old


def _next_cursor(result: dict[str, Any]) -> str | None:
    observation = result.get("observation")
    tool_result = observation.get("tool_result") if isinstance(observation, dict) else None
    page = tool_result.get("page") if isinstance(tool_result, dict) else None
    cursor = page.get("next_cursor") if isinstance(page, dict) else None
    return str(cursor) if isinstance(cursor, str) else None


def _replace_runtime_values(
    value: Any,
    runtime_episode_id: str,
    source_episode_id: str,
    cursor_new_to_old: dict[str, str],
) -> Any:
    if isinstance(value, dict):
        return {
            key: _replace_runtime_values(
                item, runtime_episode_id, source_episode_id, cursor_new_to_old
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _replace_runtime_values(
                item, runtime_episode_id, source_episode_id, cursor_new_to_old
            )
            for item in value
        ]
    if isinstance(value, tuple):
        return [
            _replace_runtime_values(
                item, runtime_episode_id, source_episode_id, cursor_new_to_old
            )
            for item in value
        ]
    if isinstance(value, str):
        if value == runtime_episode_id:
            return source_episode_id
        return cursor_new_to_old.get(value, value)
    return value


def _load_source_snapshot(task_dir: Path) -> _SourceSnapshot:
    public_rows = _read_jsonl(task_dir / "tasks.public.jsonl")
    oracle_rows = _read_jsonl(task_dir / "tasks.oracle.jsonl")
    public = {str(row.get("uid")): row for row in public_rows}
    oracle = {str(row.get("uid")): row for row in oracle_rows}
    if len(public) != len(public_rows) or len(oracle) != len(oracle_rows):
        raise SFTRebuildError(f"Generated task snapshot has duplicate ids: {task_dir}")
    records: dict[str, dict[str, Any]] = {}
    records_dir = task_dir / "records"
    for path in sorted(records_dir.glob("*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SFTRebuildError(f"Cannot read synthesis record {path}.") from error
        task_id = row.get("task_spec", {}).get("task_id")
        if not isinstance(task_id, str) or task_id in records:
            raise SFTRebuildError(f"Invalid or duplicate synthesis record: {path}")
        records[task_id] = row
    if public.keys() != oracle.keys() or public.keys() != records.keys():
        raise SFTRebuildError(f"Generated snapshot components do not align: {task_dir}")
    return _SourceSnapshot(public=public, oracle=oracle, records=records)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise TypeError("row is not an object")
                except (json.JSONDecodeError, TypeError) as error:
                    raise SFTRebuildError(
                        f"Invalid JSONL at {path}:{line_number}"
                    ) from error
                rows.append(value)
    except OSError as error:
        raise SFTRebuildError(f"Cannot read JSONL: {path}") from error
    return rows


def _tool_counts(samples: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for sample in samples:
        for message in sample["messages"]:
            if message["role"] == "assistant":
                counts[message["tool_calls"][0]["function"]["name"]] += 1
    return counts


def _preview(samples: list[dict[str, Any]], audits: list[dict[str, Any]]) -> str:
    selected: list[int] = []
    for predicate in (
        lambda audit: audit["surface_repair"] is not None,
        lambda audit: (
            audit["invalid_actions_removed"] > 0
            or audit["invalid_actions_retained"] > 0
        ),
        lambda audit: (
            audit["invalid_actions_removed"] == 0
            and audit["invalid_actions_retained"] == 0
        ),
    ):
        selected.extend(
            index
            for index, audit in enumerate(audits)
            if predicate(audit) and index not in selected
        )
    selected = selected[:10]
    lines = ["# TravelWeaver SFT preview", ""]
    for index in selected:
        sample = samples[index]
        audit = audits[index]
        query = sample["messages"][1]["content"]
        tools = [
            message["tool_calls"][0]["function"]["name"]
            for message in sample["messages"]
            if message["role"] == "assistant"
        ]
        visible_contents = [
            message["content"].strip().replace("\n", " ")
            for message in sample["messages"]
            if message["role"] == "assistant" and message["content"].strip()
        ]
        lines.extend(
            [
                f"## {sample['task_id']}",
                "",
                f"- 题面：{query}",
                f"- 监督模式：{sample['supervision_mode']}",
                f"- 有效动作：{len(tools)}",
                f"- 可见思考回合：{len(visible_contents)}",
                f"- 删除 invalid：{audit['invalid_actions_removed']}",
                f"- 保留并 mask invalid：{audit['invalid_actions_retained']}",
                f"- 题面修复：{audit['surface_repair'] or '无'}",
                f"- 工具序列：{' → '.join(tools)}",
                "",
            ]
        )
        if visible_contents:
            lines.extend([f"- 思考预览：{visible_contents[0][:300]}", ""])
    return "\n".join(lines)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _model_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=False)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl_digest(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(_model_json(row).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _atomic_text(path, "".join(_model_json(row) + "\n" for row in rows))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise
