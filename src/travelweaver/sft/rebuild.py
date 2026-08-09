"""Deterministically rebuild accepted rollout actions into action-only SFT conversations."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..env import ChinaTravelBackend, ScenarioBackend, ScenarioSpec, TravelWeaverEnv
from ..errors import SFTRebuildError, TaskNotFoundError
from ..rollout.api_agent import (
    SUPPORTED_TRAJECTORY_VERSIONS,
    SYSTEM_PROMPT,
    USER_CONTENT_FORMAT,
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

SFT_FORMAT_VERSION = "travelweaver-sft-v2"


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

    def __post_init__(self) -> None:
        if not self.sources:
            raise ValueError("At least one SFT source is required.")
        validate_tool_response_mode(self.tool_response_mode)


@dataclass(frozen=True)
class SFTRebuildReport:
    """Summary of one completed reconstruction."""

    input_rows: int
    accepted_rows: int
    samples: int
    replayed_valid_actions: int
    invalid_actions_removed: int
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
        "manifest": output / "manifest.json",
        "preview": output / "preview.md",
    }
    existing = [str(path) for path in destinations.values() if path.exists()]
    if existing:
        raise SFTRebuildError(f"Refusing to overwrite existing SFT artifacts: {existing}")
    records: list[tuple[SFTSource, dict[str, Any]]] = []
    snapshots: dict[SFTSource, _SourceSnapshot] = {}
    input_rows = 0
    for source in config.sources:
        snapshots[source] = _load_source_snapshot(source.task_dir)
        rows = _read_jsonl(source.rollout_path)
        input_rows += len(rows)
        records.extend((source, row) for row in rows if _is_accepted(row))
    records.sort(key=lambda item: (str(item[1]["task_id"]), str(item[1]["episode_id"])))
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
        )
        samples.append(sample)
        audits.append(audit)
        invalid_removed += int(audit["invalid_actions_removed"])
        if prepared.repair_kind is not None:
            repaired[prepared.repair_kind] += 1

    output.mkdir(parents=True, exist_ok=True)

    report = SFTRebuildReport(
        input_rows=input_rows,
        accepted_rows=len(records),
        samples=len(samples),
        replayed_valid_actions=sum(int(audit["replayed_valid_steps"]) for audit in audits),
        invalid_actions_removed=invalid_removed,
        samples_with_invalid_actions=sum(
            int(int(audit["invalid_actions_removed"]) > 0) for audit in audits
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
    }
    _atomic_jsonl(destinations["neutral"], samples)
    _atomic_jsonl(destinations["audit"], audits)
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
) -> tuple[dict[str, Any], dict[str, Any]]:
    response_mode = validate_tool_response_mode(tool_response_mode)
    task_id = str(row["task_id"])
    raw_scenario = task.oracle.get("scenario")
    if not isinstance(raw_scenario, dict):
        raise SFTRebuildError(f"Generated task {task_id} has no scenario.")
    scenario = ScenarioSpec.from_dict(raw_scenario)
    store = _SingleTaskStore(task.public, task.oracle)
    env = TravelWeaverEnv(ScenarioBackend(base_backend, scenario), store)  # type: ignore[arg-type]
    source_episode_id = str(row["episode_id"])
    cursor_old_to_new: dict[str, str] = {}
    cursor_new_to_old: dict[str, str] = {}
    messages: list[dict[str, Any]] = []
    invalid_steps: list[dict[str, Any]] = []
    valid_source_steps: list[dict[str, Any]] = []
    try:
        reset = env.reset(task_id=task_id, seed=0).to_dict()
        runtime_episode_id = str(reset["episode_id"])
        reset = _replace_runtime_values(
            reset, runtime_episode_id, source_episode_id, cursor_new_to_old
        )
        messages.extend(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
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
            valid = bool(
                isinstance(source_result, dict)
                and isinstance(source_result.get("info"), dict)
                and source_result["info"].get("valid_action") is True
            )
            if not valid:
                invalid_steps.append(
                    _invalid_audit(step, valid_source_steps, raw_steps[position + 1 :])
                )
                continue
            valid_source_steps.append(step)
            action = deepcopy(step.get("action"))
            if not isinstance(action, dict) or not isinstance(action.get("arguments"), dict):
                raise SFTRebuildError(f"Valid source step for {task_id} has no action object.")
            output_action = deepcopy(action)
            internal_action = deepcopy(action)
            if action.get("tool") == "next_page":
                old_cursor = str(action["arguments"].get("cursor", ""))
                try:
                    internal_action["arguments"]["cursor"] = cursor_old_to_new[old_cursor]
                except KeyError as error:
                    raise SFTRebuildError(
                        f"Cannot remap cursor {old_cursor!r} for {task_id}."
                    ) from error
            result = env.step(internal_action)
            if result.info.get("valid_action") is not True:
                error = result.observation.error or {}
                raise SFTRebuildError(
                    f"Replayed action became invalid for {task_id}: {action}; {error}"
                )
            _record_cursor_mapping(
                source_result,
                result.to_dict(),
                cursor_old_to_new,
                cursor_new_to_old,
            )
            messages.append(_assistant_message(output_action))
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
                messages.append({"role": "tool", "content": _compact_json(normalized)})
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
            f"{response_mode}:{task_id}:{source_episode_id}"
        ).encode()
    ).hexdigest()[:24]
    sample = {
        "format_version": SFT_FORMAT_VERSION,
        "sample_id": sample_id,
        "task_id": task_id,
        "tool_response_mode": response_mode,
        "model_tool_response_version": MODEL_TOOL_RESPONSE_VERSION,
        "user_content_format": USER_CONTENT_FORMAT,
        "messages": messages,
        "tools": deepcopy(row["tools"]),
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
        "source_step_count": len(row.get("steps", [])),
        "replayed_valid_steps": len(valid_source_steps),
        "invalid_actions_removed": len(invalid_steps),
        "invalid_actions": invalid_steps,
        "surface_repair": task.repair_kind,
        "original_query": task.original_query if task.repair_kind else None,
        "rebuilt_query": task.public["query"] if task.repair_kind else None,
        "cursor_remaps": len(cursor_old_to_new),
        "replay_reward": 1.0,
        "all_hard_pass": True,
        "message_count": len(messages),
    }
    return sample, audit


def _assistant_message(action: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
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
        "error": error.get("message"),
    }


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
        lambda audit: audit["invalid_actions_removed"] > 0,
        lambda audit: audit["invalid_actions_removed"] == 0,
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
        lines.extend(
            [
                f"## {sample['task_id']}",
                "",
                f"- 题面：{query}",
                f"- 有效动作：{len(tools)}",
                f"- 删除 invalid：{audit['invalid_actions_removed']}",
                f"- 题面修复：{audit['surface_repair'] or '无'}",
                f"- 工具序列：{' → '.join(tools)}",
                "",
            ]
        )
    return "\n".join(lines)


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl_digest(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(_compact_json(row).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _atomic_text(path, "".join(_compact_json(row) + "\n" for row in rows))


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
