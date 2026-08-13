"""Build replayable programmatic policy trajectories from accepted synthesis witnesses."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..env import (
    DEFAULT_MAX_VALID_STEPS,
    ChinaTravelBackend,
    ScenarioBackend,
    ScenarioSpec,
    TravelWeaverEnv,
)
from ..errors import SFTRebuildError
from ..rollout.api_agent import (
    TRAJECTORY_VERSION,
    USER_CONTENT_FORMAT,
    render_system_prompt,
    render_task_user_content,
)
from ..rollout.tool_response import MODEL_TOOL_RESPONSE_VERSION, serialize_model_tool_response
from ..synthesis.trajectory_policy import (
    MAX_CONSECUTIVE_TOOL_CALLS,
    MAX_SYNTHESIS_VALID_STEPS,
    TRAJECTORY_POLICY_VERSION,
)
from .ordering import order_tool_arguments, order_tool_schemas
from .rebuild import _SingleTaskStore

PROGRAMMATIC_POLICY_VERSION = "travelweaver-programmatic-policy-v28"
PROGRAMMATIC_ARTIFACT_VERSION = "travelweaver-programmatic-artifacts-v2"
SAMPLE_FAMILIES = ("efficient_success",)
_EVIDENCE_PATH_KINDS = (
    "nearby_discovery",
    "candidate_comparison",
    "opening_verification",
)
_EVIDENCE_PATH_EXTRA_ACTIONS = {
    "nearby_discovery": 0,
    "candidate_comparison": 3,
    "opening_verification": 1,
}
_SPATIAL_PREFERENCE_KINDS = {
    "less_innercity_time",
    "less_walking",
    "near_poi",
    "shorter_meal_transfer",
    "shorter_total_travel_time",
}


def minimum_tool_coverage_samples(count: int) -> int:
    """Return the recommended ten-percent sample coverage for each public tool."""

    if count <= 0:
        raise ValueError("Tool coverage requires a positive batch size.")
    return max(1, (count + 9) // 10)


def _longest_tool_run(tools: list[str]) -> tuple[str | None, int]:
    """Return the tool and length of the longest contiguous run."""

    longest_tool: str | None = None
    longest = 0
    current_tool: str | None = None
    current = 0
    for tool in tools:
        if tool == current_tool:
            current += 1
        else:
            current_tool = tool
            current = 1
        if current > longest:
            longest_tool = tool
            longest = current
    return longest_tool, longest


@dataclass(frozen=True)
class ProgrammaticBuildConfig:
    task_dir: Path
    output_path: Path
    audit_path: Path
    seed: int
    concurrency: int = min(32, os.cpu_count() or 1)
    work_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.concurrency <= 0:
            raise ValueError("Programmatic trajectory concurrency must be positive.")

    @property
    def resolved_work_dir(self) -> Path:
        return self.work_dir or self.output_path.with_name(
            f"{self.output_path.stem}.work"
        )


_PROGRAMMATIC_WORKER_BACKEND: ChinaTravelBackend | None = None


def _initialize_programmatic_worker() -> None:
    global _PROGRAMMATIC_WORKER_BACKEND
    _PROGRAMMATIC_WORKER_BACKEND = ChinaTravelBackend()


def _worker_backend() -> ChinaTravelBackend:
    if _PROGRAMMATIC_WORKER_BACKEND is None:
        raise RuntimeError("Programmatic CPU worker was not initialized.")
    return _PROGRAMMATIC_WORKER_BACKEND


def _compute_capability_in_worker(
    index: int,
    record: dict[str, Any],
    public_task: dict[str, Any],
) -> tuple[int, dict[str, dict[str, Any]]]:
    return index, _evidence_path_capabilities(record, public_task, _worker_backend())


def _build_one_in_worker(
    index: int,
    record: dict[str, Any],
    public_task: dict[str, Any],
    oracle_task: dict[str, Any],
    evidence_paths: tuple[dict[str, Any], ...],
    seed: int,
    source_question_batch: str,
) -> tuple[int, dict[str, Any], dict[str, Any]]:
    return (
        index,
        *_build_one(
            record,
            public_task,
            oracle_task,
            _worker_backend(),
            family="efficient_success",
            evidence_paths=evidence_paths,
            seed=seed,
            source_question_batch=source_question_batch,
        ),
    )


class _ProgrammaticArtifactStore:
    """Main-process-only checkpoints for two-stage CPU trajectory synthesis."""

    def __init__(
        self,
        config: ProgrammaticBuildConfig,
        *,
        source_identity: str,
        total: int,
    ) -> None:
        self.config = config
        self.work_dir = config.resolved_work_dir
        self.capabilities_dir = self.work_dir / "capabilities"
        self.records_dir = self.work_dir / "records"
        self.manifest_path = self.work_dir / "manifest.json"
        self.progress_path = self.work_dir / "progress.jsonl"
        manifest_existed = self.manifest_path.exists()
        if (
            (config.output_path.exists() or config.audit_path.exists())
            and not manifest_existed
        ):
            raise SFTRebuildError(
                "Programmatic aggregate outputs exist without their recovery manifest."
            )
        self.capabilities_dir.mkdir(parents=True, exist_ok=True)
        self.records_dir.mkdir(parents=True, exist_ok=True)
        identity = {
            "artifact_version": PROGRAMMATIC_ARTIFACT_VERSION,
            "programmatic_policy_version": PROGRAMMATIC_POLICY_VERSION,
            "task_dir": str(config.task_dir.resolve()),
            "output_path": str(config.output_path.resolve()),
            "audit_path": str(config.audit_path.resolve()),
            "source_identity": source_identity,
            "seed": config.seed,
            "total": total,
        }
        if manifest_existed:
            manifest = _read_json(self.manifest_path)
            if manifest.get("config") != identity:
                raise SFTRebuildError(
                    "Programmatic work directory belongs to a different configuration."
                )
            self.manifest = manifest
        else:
            self.manifest = {
                "artifact_version": PROGRAMMATIC_ARTIFACT_VERSION,
                "status": "in_progress",
                "config": identity,
                "capabilities_completed": 0,
                "trajectories_completed": 0,
                "created_at": _now(),
                "updated_at": _now(),
            }
            _atomic_json(self.manifest_path, self.manifest)
        self._reconcile_manifest()

    def capability_indices(self) -> set[int]:
        return _numbered_json_indices(self.capabilities_dir)

    def record_indices(self) -> set[int]:
        return _numbered_json_indices(self.records_dir)

    def capabilities(self) -> dict[int, dict[str, dict[str, Any]]]:
        return {
            int(path.stem): dict(_read_json(path)["capabilities"])
            for path in _numbered_json_paths(self.capabilities_dir)
        }

    def records(self) -> list[dict[str, Any]]:
        return [_read_json(path) for path in _numbered_json_paths(self.records_dir)]

    def save_capability(
        self,
        index: int,
        task_id: str,
        capabilities: Mapping[str, Mapping[str, Any]],
    ) -> None:
        path = self.capabilities_dir / f"{index:06d}.json"
        if path.exists():
            raise SFTRebuildError(f"Capability slot {index} is already complete.")
        _atomic_json(
            path,
            {
                "index": index,
                "task_id": task_id,
                "capabilities": capabilities,
            },
        )
        self._reconcile_manifest()

    def save_record(
        self,
        index: int,
        task_id: str,
        trajectory: Mapping[str, Any],
        audit: Mapping[str, Any],
        evidence_paths: tuple[dict[str, Any], ...],
    ) -> None:
        path = self.records_dir / f"{index:06d}.json"
        if path.exists():
            raise SFTRebuildError(f"Programmatic trajectory slot {index} is already complete.")
        _atomic_json(
            path,
            {
                "index": index,
                "task_id": task_id,
                "evidence_paths": evidence_paths,
                "trajectory": trajectory,
                "audit": audit,
            },
        )
        self._reconcile_manifest()

    def progress(self, event: Mapping[str, Any]) -> dict[str, Any]:
        payload = {**event, "timestamp": _now()}
        with self.progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.manifest["last_event"] = payload
        self.manifest["updated_at"] = payload["timestamp"]
        _atomic_json(self.manifest_path, self.manifest)
        return payload

    def finalize(self, report: Mapping[str, Any]) -> None:
        self.manifest.update(
            {
                "status": "complete",
                "report": report,
                "updated_at": _now(),
            }
        )
        _atomic_json(self.manifest_path, self.manifest)

    def mark_failed(self, error: BaseException) -> None:
        self.manifest.update(
            {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "updated_at": _now(),
            }
        )
        _atomic_json(self.manifest_path, self.manifest)

    def _reconcile_manifest(self) -> None:
        self.manifest["capabilities_completed"] = len(self.capability_indices())
        self.manifest["trajectories_completed"] = len(self.record_indices())
        self.manifest["updated_at"] = _now()
        _atomic_json(self.manifest_path, self.manifest)


def build_programmatic_trajectories(
    config: ProgrammaticBuildConfig,
    *,
    base_backend: Any | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Build resumable trajectories with process workers and per-task checkpoints."""

    public = _read_jsonl_index(config.task_dir / "tasks.public.jsonl")
    oracle = _read_jsonl_index(config.task_dir / "tasks.oracle.jsonl")
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((config.task_dir / "records").glob("*.json"))
    ]
    if not records:
        raise SFTRebuildError("Synthesis directory has no records.")
    source_identity = _programmatic_source_identity(records)
    store = _ProgrammaticArtifactStore(
        config,
        source_identity=source_identity,
        total=len(records),
    )

    def emit(**event: Any) -> None:
        payload = store.progress(event)
        if progress is not None:
            progress(payload)

    completed_records = store.record_indices()
    completed_capabilities = store.capability_indices()
    needs_cpu_work = (
        len(completed_capabilities) != len(records)
        or len(completed_records) != len(records)
    )
    start_event = (
        "programmatic_synthesis_resumed"
        if completed_records
        else "programmatic_synthesis_started"
    )
    emit(
        event=start_event,
        total=len(records),
        capabilities_completed=len(completed_capabilities),
        trajectories_completed=len(completed_records),
        pending=len(records) - len(completed_records),
        concurrency=config.concurrency,
        executor=(
            "process"
            if needs_cpu_work and base_backend is None and config.concurrency > 1
            else "inline"
        ),
    )
    executor = (
        ProcessPoolExecutor(
            max_workers=min(config.concurrency, len(records)),
            initializer=_initialize_programmatic_worker,
        )
        if needs_cpu_work and base_backend is None and config.concurrency > 1
        else None
    )
    inline_backend = base_backend
    if needs_cpu_work and executor is None and inline_backend is None:
        inline_backend = ChinaTravelBackend()
    try:
        missing_capabilities = [
            index
            for index in range(len(records))
            if index not in store.capability_indices()
        ]
        capability_futures: dict[
            Future[tuple[int, dict[str, dict[str, Any]]]], int
        ] = {}
        capability_failures: list[str] = []
        if executor is not None:
            for index in missing_capabilities:
                task_id = str(records[index]["task_spec"]["task_id"])
                capability_futures[
                    executor.submit(
                        _compute_capability_in_worker,
                        index,
                        records[index],
                        public[task_id],
                    )
                ] = index
            capability_items: Any = (
                (capability_futures[future], future)
                for future in as_completed(capability_futures)
            )
        else:
            capability_items = ((index, None) for index in missing_capabilities)
        for submitted_index, capability_future in capability_items:
            try:
                if capability_future is None:
                    index = submitted_index
                    assert inline_backend is not None
                    capabilities = _evidence_path_capabilities(
                        records[index],
                        public[str(records[index]["task_spec"]["task_id"])],
                        inline_backend,
                    )
                else:
                    index, capabilities = capability_future.result()
            except Exception as error:  # noqa: BLE001 - persist all other slots.
                task_id = str(records[submitted_index]["task_spec"]["task_id"])
                message = f"{type(error).__name__}: {error}"
                capability_failures.append(f"slot {submitted_index}: {message}")
                emit(
                    event="capability_failed",
                    slot_index=submitted_index,
                    task_id=task_id,
                    error=message,
                    completed=len(store.capability_indices()),
                    total=len(records),
                )
                continue
            task_id = str(records[index]["task_spec"]["task_id"])
            store.save_capability(index, task_id, capabilities)
            emit(
                event="capability_completed",
                slot_index=index,
                task_id=task_id,
                completed=len(store.capability_indices()),
                total=len(records),
            )
        if capability_failures:
            raise SFTRebuildError(
                "Programmatic capability phase failed: "
                + " | ".join(capability_failures[:3])
            )

        capabilities = store.capabilities()
        if len(capabilities) != len(records):
            raise SFTRebuildError("Programmatic capability phase is incomplete.")
        evidence_paths = _select_natural_evidence_paths(records, public, capabilities)
        missing_records = [
            index for index in range(len(records)) if index not in store.record_indices()
        ]
        build_futures: dict[
            Future[tuple[int, dict[str, Any], dict[str, Any]]], int
        ] = {}
        build_failures: list[str] = []
        if executor is not None:
            for index in missing_records:
                task_id = str(records[index]["task_spec"]["task_id"])
                build_futures[
                    executor.submit(
                        _build_one_in_worker,
                        index,
                        records[index],
                        public[task_id],
                        oracle[task_id],
                        evidence_paths.get(index, ()),
                        config.seed,
                        config.task_dir.name,
                    )
                ] = index
            build_items: Any = (
                (build_futures[future], future)
                for future in as_completed(build_futures)
            )
        else:
            build_items = ((index, None) for index in missing_records)
        for submitted_index, build_future in build_items:
            try:
                if build_future is None:
                    index = submitted_index
                    assert inline_backend is not None
                    trajectory, audit = _build_one(
                        records[index],
                        public[str(records[index]["task_spec"]["task_id"])],
                        oracle[str(records[index]["task_spec"]["task_id"])],
                        inline_backend,
                        family="efficient_success",
                        evidence_paths=evidence_paths.get(index, ()),
                        seed=config.seed,
                        source_question_batch=config.task_dir.name,
                    )
                else:
                    index, trajectory, audit = build_future.result()
            except Exception as error:  # noqa: BLE001 - persist all other slots.
                task_id = str(records[submitted_index]["task_spec"]["task_id"])
                message = f"{type(error).__name__}: {error}"
                build_failures.append(f"slot {submitted_index}: {message}")
                emit(
                    event="trajectory_failed",
                    slot_index=submitted_index,
                    task_id=task_id,
                    error=message,
                    completed=len(store.record_indices()),
                    total=len(records),
                )
                continue
            task_id = str(records[index]["task_spec"]["task_id"])
            store.save_record(
                index,
                task_id,
                trajectory,
                audit,
                evidence_paths.get(index, ()),
            )
            emit(
                event="trajectory_completed",
                slot_index=index,
                task_id=task_id,
                completed=len(store.record_indices()),
                total=len(records),
                pending=len(records) - len(store.record_indices()),
            )
        if build_failures:
            raise SFTRebuildError(
                "Programmatic trajectory phase failed: "
                + " | ".join(build_failures[:3])
            )
    except BaseException as error:
        store.mark_failed(error)
        emit(
            event="programmatic_synthesis_failed",
            error=f"{type(error).__name__}: {error}",
            completed=len(store.record_indices()),
            total=len(records),
        )
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    bundles = store.records()
    if len(bundles) != len(records):
        raise SFTRebuildError("Programmatic trajectory phase is incomplete.")
    trajectories = [dict(bundle["trajectory"]) for bundle in bundles]
    audits = [dict(bundle["audit"]) for bundle in bundles]
    family_counts = Counter(row["sample_family"] for row in audits)
    evidence_path_counts = Counter(
        str(path["kind"])
        for row in audits
        for path in row.get("evidence_paths", [])
    )
    evidence_selection_counts = Counter(
        str(path["selection_reason"])
        for row in audits
        for path in row.get("evidence_paths", [])
    )
    tool_counts = Counter(
        turn["tool"] for row in audits for turn in row.get("turns", [])
    )
    tool_sample_counts = _tool_sample_counts(trajectories)
    public_tools = {
        str(tool["function"]["name"])
        for tool in trajectories[0]["tools"]
    }
    minimum_coverage = minimum_tool_coverage_samples(len(records))
    undercovered = {
        tool: tool_sample_counts[tool]
        for tool in sorted(public_tools)
        if tool_sample_counts[tool] < minimum_coverage
    }
    if undercovered:
        emit(
            event="tool_coverage_warning",
            recommended_minimum_samples=minimum_coverage,
            undercovered_tools=undercovered,
        )
    _atomic_jsonl(config.output_path, trajectories)
    _atomic_jsonl(config.audit_path, audits)
    report = {
        "programmatic_policy_version": PROGRAMMATIC_POLICY_VERSION,
        "programmatic_artifact_version": PROGRAMMATIC_ARTIFACT_VERSION,
        "samples": len(trajectories),
        "families": dict(sorted(family_counts.items())),
        "evidence_paths": dict(sorted(evidence_path_counts.items())),
        "evidence_path_selection": dict(sorted(evidence_selection_counts.items())),
        "minimum_tool_coverage_samples": minimum_coverage,
        "tool_coverage_recommendation_met": not undercovered,
        "tool_coverage_warnings": undercovered,
        "tool_calls": dict(sorted(tool_counts.items())),
        "tool_sample_counts": dict(sorted(tool_sample_counts.items())),
        "concurrency": config.concurrency,
        "all_reward_one": all(row["replay_reward"] == 1.0 for row in audits),
        "all_hard_pass": all(row["all_hard_pass"] is True for row in audits),
        "work_dir": str(config.resolved_work_dir.resolve()),
    }
    store.finalize(report)
    emit(
        event="programmatic_synthesis_completed",
        completed=len(records),
        total=len(records),
    )
    return report


def _assign_families(
    records: list[dict[str, Any]],
    seed: int,
    *,
    coverage_plans: Mapping[int, Mapping[str, Any]] | None = None,
) -> dict[int, str]:
    """Compatibility helper: every released trajectory is a clean success family."""

    del seed, coverage_plans
    return {index: "efficient_success" for index in range(len(records))}


def _select_natural_evidence_paths(
    records: list[dict[str, Any]],
    public: Mapping[str, Mapping[str, Any]],
    capabilities: Mapping[int, Mapping[str, Mapping[str, Any]]],
) -> dict[int, tuple[dict[str, Any], ...]]:
    """Select evidence paths justified by task semantics, without a coverage quota."""

    assignments: dict[int, tuple[dict[str, Any], ...]] = {}
    for index, record in enumerate(records):
        task_id = str(record["task_spec"]["task_id"])
        selected: list[dict[str, Any]] = []
        for kind in _natural_evidence_path_kinds(
            record,
            public[task_id],
            capabilities[index],
        ):
            if not _evidence_paths_fit(record, public[task_id], selected, kind):
                continue
            selected.append(
                {
                    "kind": kind,
                    "selection_reason": "natural_main_graph",
                    **capabilities[index][kind],
                }
            )
        if selected:
            assignments[index] = tuple(
                sorted(selected, key=lambda path: str(path["kind"]))
            )
    return assignments


def _natural_evidence_path_kinds(
    record: Mapping[str, Any],
    public: Mapping[str, Any],
    capabilities: Mapping[str, Mapping[str, Any]],
) -> tuple[str, ...]:
    preferences = {
        str(item.get("kind"))
        for item in record.get("blueprint", {}).get("preferences", [])
        if isinstance(item, Mapping)
    }
    selected: list[str] = []
    opening = capabilities.get("opening_verification")
    if opening is not None:
        entity = record["witness"]["evidence_bundle"]["entities"][
            str(opening["candidate_id"])
        ]
        if _entity_name(entity) in str(public["query"]):
            selected.append("opening_verification")
    if "nearby_discovery" in capabilities and (
        preferences & _SPATIAL_PREFERENCE_KINDS or "附近" in str(public["query"])
    ):
        selected.append("nearby_discovery")
    comparison = capabilities.get("candidate_comparison")
    if comparison is not None:
        target = record["witness"]["evidence_bundle"]["entities"][
            str(comparison["candidate_id"])
        ]
        if _entity_type(target) in _comparison_semantic_entity_types(record):
            selected.append("candidate_comparison")
    return tuple(selected)


def _comparison_semantic_entity_types(record: Mapping[str, Any]) -> set[str]:
    """Return local entity types for which the task makes price comparison useful."""

    preferences = {
        str(item.get("kind"))
        for item in record.get("blueprint", {}).get("preferences", [])
        if isinstance(item, Mapping)
    }
    entity_types: set[str] = set()
    if "lower_total_cost" in preferences:
        entity_types.update({"attraction", "restaurant", "hotel"})
    if "lower_lodging_share" in preferences:
        entity_types.add("hotel")
    constraints = record.get("task_spec", {}).get("constraints", [])
    for constraint in constraints if isinstance(constraints, list) else []:
        if not isinstance(constraint, Mapping):
            continue
        kind = str(constraint.get("kind"))
        scope = str(constraint.get("scope"))
        if kind == "total_budget" and scope == "trip":
            entity_types.update({"attraction", "restaurant", "hotel"})
    return entity_types


def _hard_unit_price_limit(
    record: Mapping[str, Any], entity_type: str
) -> float | None:
    """Return a directly comparable per-entity hard price ceiling, if present."""

    expected_scope = {
        "restaurant": "restaurant",
        "hotel": "accommodation",
    }.get(entity_type)
    if expected_scope is None:
        return None
    constraints = record.get("task_spec", {}).get("constraints", [])
    limits: list[float] = []
    for constraint in constraints if isinstance(constraints, list) else []:
        if not isinstance(constraint, Mapping):
            continue
        if (
            constraint.get("kind") != "category_budget"
            or constraint.get("scope") != expected_scope
            or constraint.get("operator") != "lte"
        ):
            continue
        value = constraint.get("value")
        if not isinstance(value, Mapping):
            continue
        amount = _numeric_price(value.get("amount"))
        if amount is not None:
            limits.append(amount)
    return min(limits) if limits else None


def _passes_hard_unit_price_limit(
    record: Mapping[str, Any], entity: Mapping[str, Any], entity_type: str
) -> bool:
    limit = _hard_unit_price_limit(record, entity_type)
    price = _comparable_unit_price(record, entity, entity_type)
    return limit is None or (price is not None and price <= limit)


def _comparable_unit_price(
    record: Mapping[str, Any], entity: Mapping[str, Any], entity_type: str
) -> float | None:
    """Normalize visible prices to the basis used by category-budget checks."""

    price = _numeric_price(entity.get("price"))
    if price is None or entity_type != "hotel":
        return price
    travelers = int(record.get("task_spec", {}).get("trip", {}).get("travelers", 0))
    room_type = entity.get("room_type")
    if travelers <= 0 or not isinstance(room_type, int) or room_type <= 0:
        return None
    constraints = record.get("task_spec", {}).get("constraints", [])
    explicit_rooms = next(
        (
            int(value["count"])
            for constraint in constraints if isinstance(constraints, list)
            if isinstance(constraint, Mapping)
            and constraint.get("kind") == "room_count"
            and constraint.get("scope") == "accommodation"
            and isinstance((value := constraint.get("value")), Mapping)
            and isinstance(value.get("count"), int)
            and int(value["count"]) > 0
        ),
        None,
    )
    rooms = explicit_rooms or (travelers + room_type - 1) // room_type
    return price * rooms / travelers


def _opening_verification_capability(
    local_ids: list[str],
    entities: Mapping[str, Mapping[str, Any]],
    candidates_by_id: Mapping[str, Mapping[str, Any]],
    public_query: str,
    backend: Any,
) -> dict[str, Any] | None:
    """Prefer a named scheduled place when selecting one useful opening check."""

    candidates: list[dict[str, Any]] = []
    for candidate_id in local_ids:
        entity = entities[candidate_id]
        entity_type = _entity_type(entity)
        at_time = str(candidates_by_id[candidate_id]["start_time"])
        if entity_type not in {"attraction", "restaurant"} or at_time == "24:00":
            continue
        try:
            check = backend.check_place_open(candidate_id, at_time)
        except Exception:
            continue
        if check.get("is_open") is True:
            candidates.append({"candidate_id": candidate_id, "at_time": at_time})
    if not candidates:
        return None
    return next(
        (
            item
            for item in candidates
            if _entity_name(entities[str(item["candidate_id"])]) in public_query
        ),
        candidates[0],
    )


def _catalog_requirement(
    record: Mapping[str, Any],
    entity: Mapping[str, Any],
    entity_type: str,
) -> str | None:
    """Return the task-grounded facet that justifies a catalog lookup."""

    constraints = record.get("task_spec", {}).get("constraints", [])
    for constraint in constraints if isinstance(constraints, list) else []:
        if not isinstance(constraint, Mapping):
            continue
        kind = str(constraint.get("kind"))
        scope = str(constraint.get("scope"))
        value = constraint.get("value")
        if not isinstance(value, Mapping):
            continue
        if (
            entity_type == "attraction"
            and kind == "entity_category"
            and scope == "attraction"
        ):
            actual = str(entity.get("category") or "")
            if actual and actual in _constraint_string_values(value, "values"):
                return actual
        if (
            entity_type == "restaurant"
            and kind == "entity_category"
            and scope == "restaurant"
        ):
            actual = str(entity.get("cuisine") or "")
            if actual and actual in _constraint_string_values(value, "values"):
                return actual
        if entity_type != "hotel" or scope != "accommodation":
            continue
        if kind == "entity_attribute":
            actual = str(entity.get("hotel_type") or "")
            for required in _constraint_string_values(value, "values"):
                if required and required in actual:
                    return required
        if kind == "room_type" and entity.get("room_type") == value.get("room_type"):
            return f"每间{value['room_type']}个床位的房型"
    return None


def _constraint_string_values(value: Mapping[str, Any], key: str) -> set[str]:
    any_of = value.get("any_of")
    groups = any_of if isinstance(any_of, list) else [value.get(key, [])]
    return {
        str(item)
        for group in groups
        if isinstance(group, list)
        for item in group
        if str(item).strip()
    }


def _pagination_is_task_grounded(
    *,
    name_grounded: bool,
    search_tool: str = "search_attractions",
    planned_candidate_count: int = 1,
    visible_planned_count: int = 0,
    required_candidate_count: int = 0,
    resolved_candidate_count: int = 0,
) -> bool:
    """Require a visible unresolved predicate before consuming a cursor."""

    if name_grounded:
        return True
    if search_tool == "search_nearby":
        return False
    planned_gap = (
        planned_candidate_count > 1
        and 0 < visible_planned_count < planned_candidate_count
    )
    task_count_gap = (
        required_candidate_count > 1
        and 0 < resolved_candidate_count < required_candidate_count
    )
    return planned_gap or task_count_gap


def _required_activity_count(record: Mapping[str, Any], entity_type: str) -> int:
    """Return a public hard quantity requirement for one searchable entity type."""

    if entity_type != "attraction":
        return 0
    constraints = record.get("task_spec", {}).get("constraints", [])
    required = 0
    for constraint in constraints if isinstance(constraints, list) else []:
        if not isinstance(constraint, Mapping):
            continue
        if (
            constraint.get("kind") != "activity_count"
            or constraint.get("scope") != "attraction"
            or constraint.get("operator") not in {"eq", "gte"}
        ):
            continue
        value = constraint.get("value")
        if (
            isinstance(value, Mapping)
            and value.get("activity_type") == "attraction"
            and isinstance(value.get("count"), int)
            and not isinstance(value.get("count"), bool)
        ):
            required = max(required, int(value["count"]))
    return required


def _evidence_paths_fit(
    record: Mapping[str, Any],
    public: Mapping[str, Any],
    existing: list[dict[str, Any]],
    kind: str,
) -> bool:
    return (
        _teacher_action_upper_bound(record, public)
        + sum(
            _EVIDENCE_PATH_EXTRA_ACTIONS[str(path["kind"])]
            for path in existing
        )
        + _EVIDENCE_PATH_EXTRA_ACTIONS[kind]
        <= MAX_SYNTHESIS_VALID_STEPS
    )


def _tool_sample_counts(trajectories: list[dict[str, Any]]) -> Counter[str]:
    return Counter(
        tool
        for trajectory in trajectories
        for tool in {
            str(step["action"]["tool"])
            for step in trajectory.get("steps", [])
        }
    )


def _evidence_path_capabilities(
    record: Mapping[str, Any],
    public: Mapping[str, Any],
    base_backend: Any,
) -> dict[str, dict[str, Any]]:
    """Find witness-grounded parameters for optional public-tool demonstrations."""

    scenario = ScenarioSpec.from_dict(record["scenario"])
    backend = ScenarioBackend(base_backend, scenario)
    activities = sorted(
        record["witness"]["plan_snapshot"]["activities"],
        key=lambda item: (item["day"], item["activity_index"]),
    )
    entities = record["witness"]["evidence_bundle"]["entities"]
    candidate_order = list(dict.fromkeys(str(item["candidate_id"]) for item in activities))
    planned_ids = set(candidate_order)
    candidates_by_id = {
        str(item["candidate_id"]): item for item in activities
    }
    local_ids = [
        candidate_id
        for candidate_id in candidate_order
        if _entity_type(entities[candidate_id]) in {"attraction", "restaurant", "hotel"}
    ]
    capabilities: dict[str, dict[str, Any]] = {}

    opening = _opening_verification_capability(
        local_ids,
        entities,
        candidates_by_id,
        str(public["query"]),
        backend,
    )
    if opening is not None:
        capabilities["opening_verification"] = opening

    anchors: list[tuple[str, str]] = []
    for candidate_id in candidate_order:
        entity = entities[candidate_id]
        entity_type = _entity_type(entity)
        if (
            entity_type in {"train", "airplane"}
            and entity.get("origin_city") == public["start_city"]
        ):
            anchor_id = entity.get("destination_anchor_id")
            if isinstance(anchor_id, str):
                anchors.append((anchor_id, "route_anchor"))
            continue
        if entity_type not in {"attraction", "restaurant", "hotel"}:
            continue
        for anchor_id, anchor_type in anchors:
            # If the anchor uses the same search family, its own broad discovery
            # may expose the target before the nearby path becomes executable.
            if anchor_type == entity_type:
                continue
            for radius in (2, 5, 10, 20, 50):
                try:
                    nearby = backend.search_nearby(
                        place_id=anchor_id,
                        category=entity_type,
                        radius_km=radius,
                        top_k=40,
                    )
                except Exception:
                    continue
                if any(
                    str(item.get("place_id")) == candidate_id
                    for item in nearby[:10]
                ):
                    capabilities["nearby_discovery"] = {
                        "anchor_id": anchor_id,
                        "candidate_id": candidate_id,
                        "radius_km": radius,
                    }
                    break
            if "nearby_discovery" in capabilities:
                break
        if "nearby_discovery" in capabilities:
            break
        anchors.append((candidate_id, entity_type))

    public_query = str(public["query"])
    comparison_entity_types = _comparison_semantic_entity_types(record)
    for candidate_id in local_ids:
        entity = entities[candidate_id]
        entity_type = _entity_type(entity)
        if entity_type not in comparison_entity_types:
            continue
        if _entity_name(entity) in public_query:
            continue
        target_price = _comparable_unit_price(record, entity, entity_type)
        if target_price is None:
            continue
        action = _task_grounded_search_action(record, entity, entity_type)
        try:
            alternatives = getattr(backend, str(action["tool"]))(**action["arguments"])
        except Exception:
            continue
        alternative_ids = [
            str(item.get("place_id"))
            for item in alternatives[:10]
            if isinstance(item, Mapping)
            and isinstance(item.get("place_id"), str)
            and str(item["place_id"]) not in planned_ids
            and _entity_type(item) == entity_type
            and (_comparable_unit_price(record, item, entity_type) or -1) > target_price
            and _passes_hard_unit_price_limit(record, item, entity_type)
        ]
        if alternative_ids:
            capabilities["candidate_comparison"] = {
                "candidate_id": candidate_id,
                "alternative_id": alternative_ids[0],
            }
            break
    return capabilities


def _teacher_action_upper_bound(record: Mapping[str, Any], public: Mapping[str, Any]) -> int:
    """Conservative clean-teacher length bound used before assigning coverage extras."""

    activities = record["witness"]["plan_snapshot"]["activities"]
    entities = record["witness"]["evidence_bundle"]["entities"]
    candidate_order = list(dict.fromkeys(str(item["candidate_id"]) for item in activities))
    catalog_types = {
        _entity_type(entities[candidate_id])
        for candidate_id in candidate_order
        if _entity_type(entities[candidate_id]) in {"attraction", "restaurant", "hotel"}
        and _entity_name(entities[candidate_id]) not in str(public["query"])
    }
    route_count = len(
        {
            str(item["route_from_previous_id"])
            for item in activities
            if item.get("route_from_previous_id") is not None
        }
    )
    return 2 * len(candidate_order) + len(catalog_types) + route_count + 3 + 1


def _entity_type(entity: Mapping[str, Any]) -> str:
    return str(entity.get("entity_type") or entity.get("mode"))


def _task_grounded_search_action(
    record: Mapping[str, Any],
    entity: Mapping[str, Any],
    entity_type: str,
) -> dict[str, Any]:
    """Build a search using only filters stated by the task, never hidden witness facets."""

    arguments: dict[str, Any] = {"city": entity["city"]}
    preferences = record.get("blueprint", {}).get("preferences", [])
    if any(
        isinstance(preference, Mapping)
        and preference.get("kind") == "lower_total_cost"
        for preference in preferences if isinstance(preferences, list)
    ):
        arguments["sort_by"] = "price"
    constraints = record.get("task_spec", {}).get("constraints", [])
    for constraint in constraints if isinstance(constraints, list) else []:
        if not isinstance(constraint, Mapping):
            continue
        kind = str(constraint.get("kind"))
        scope = str(constraint.get("scope"))
        value = constraint.get("value")
        if not isinstance(value, Mapping):
            continue
        if (
            entity_type == "attraction"
            and kind == "entity_category"
            and scope == "attraction"
            and str(entity.get("category"))
            in _constraint_string_values(value, "values")
        ):
            arguments["category"] = entity["category"]
        elif (
            entity_type == "restaurant"
            and kind == "entity_category"
            and scope == "restaurant"
            and str(entity.get("cuisine"))
            in _constraint_string_values(value, "values")
        ):
            arguments["cuisine"] = entity["cuisine"]
        elif entity_type == "hotel" and scope == "accommodation":
            if kind == "entity_attribute":
                actual = str(entity.get("hotel_type") or "")
                required = next(
                    (
                        str(item)
                        for item in _constraint_string_values(value, "values")
                        if str(item) and str(item) in actual
                    ),
                    None,
                )
                if required is not None:
                    arguments["hotel_type"] = required
            elif kind == "room_type" and entity.get("room_type") == value.get("room_type"):
                arguments["room_type"] = int(value["room_type"])
    return {
        "tool": {
            "attraction": "search_attractions",
            "restaurant": "search_restaurants",
            "hotel": "search_hotels",
        }[entity_type],
        "arguments": arguments,
    }


def _numeric_price(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _format_price(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:.2f}".rstrip("0")


def _build_one(
    record: dict[str, Any],
    public: dict[str, Any],
    oracle: dict[str, Any],
    base_backend: Any,
    *,
    family: str,
    evidence_paths: tuple[Mapping[str, Any], ...] = (),
    seed: int,
    source_question_batch: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if family not in SAMPLE_FAMILIES:
        raise ValueError(f"Unknown sample family: {family}")
    path_by_kind = {str(path.get("kind")): dict(path) for path in evidence_paths}
    if len(path_by_kind) != len(evidence_paths) or not set(path_by_kind) <= set(
        _EVIDENCE_PATH_KINDS
    ):
        raise SFTRebuildError(
            f"Programmatic evidence paths are invalid for {public.get('uid')}: "
            f"{evidence_paths!r}."
        )
    policy = record.get("trajectory_policy")
    if not isinstance(policy, Mapping):
        raise SFTRebuildError(
            "Programmatic short-trajectory teacher requires a trajectory_policy artifact."
        )
    if (
        policy.get("policy_version") != TRAJECTORY_POLICY_VERSION
        or policy.get("max_valid_steps") != DEFAULT_MAX_VALID_STEPS
        or policy.get("max_consecutive_tool_calls") != MAX_CONSECUTIVE_TOOL_CALLS
    ):
        raise SFTRebuildError(
            "Programmatic teacher requires the current 50-step/three-consecutive policy."
        )
    task_id = str(public["uid"])
    scenario = ScenarioSpec.from_dict(record["scenario"])
    runtime_backend = ScenarioBackend(base_backend, scenario)
    env = TravelWeaverEnv(
        runtime_backend,
        _SingleTaskStore(public, oracle),  # type: ignore[arg-type]
        max_valid_steps=DEFAULT_MAX_VALID_STEPS,
    )
    reset = env.reset(task_id=task_id, seed=0)
    tools = order_tool_schemas(env.tool_schemas())
    system_prompt = render_system_prompt(env.max_valid_steps)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": render_task_user_content(reset.task)},
    ]
    steps: list[dict[str, Any]] = []
    masks: list[bool] = []
    mask_reasons: list[str] = []
    rationale_specs: list[dict[str, Any]] = []
    witness = record["witness"]
    plan = deepcopy(witness["plan"])
    evidence = witness["evidence_bundle"]
    entities = evidence["entities"]
    activities = sorted(
        witness["plan_snapshot"]["activities"],
        key=lambda item: (item["day"], item["activity_index"]),
    )
    candidate_order = list(dict.fromkeys(str(item["candidate_id"]) for item in activities))
    nearby_path = path_by_kind.get("nearby_discovery")
    if nearby_path is not None:
        nearby_target = str(nearby_path["candidate_id"])
        nearby_anchor = str(nearby_path["anchor_id"])
        candidate_order.remove(nearby_target)
        if nearby_anchor in candidate_order:
            # Make the selected nearby path the first discovery path for this
            # target. Otherwise an earlier broad search for the same entity type
            # could expose it before the anchor is available.
            candidate_order.remove(nearby_anchor)
            outbound_position = next(
                (
                    index
                    for index, candidate_id in enumerate(candidate_order)
                    if _entity_type(entities[candidate_id]) in {"train", "airplane"}
                    and entities[candidate_id].get("origin_city") == public["start_city"]
                ),
                -1,
            )
            candidate_order.insert(outbound_position + 1, nearby_anchor)
            candidate_order.insert(outbound_position + 2, nearby_target)
            nearby_anchor = ""
        anchor_position = next(
            (
                index
                for index, candidate_id in enumerate(candidate_order)
                if candidate_id == nearby_anchor
                or str(entities[candidate_id].get("destination_anchor_id"))
                == nearby_anchor
            ),
            None,
        )
        if nearby_anchor and anchor_position is None:
            raise SFTRebuildError(
                f"Nearby evidence path has no candidate-visible anchor for {public['uid']}: "
                f"{nearby_anchor}"
            )
        if nearby_anchor:
            assert anchor_position is not None
            candidate_order.insert(anchor_position + 1, nearby_target)
    candidate_ids = set(candidate_order)
    route_anchor_by_candidate = {
        str(item["candidate_id"]): str(
            evidence["routes"][str(item["route_from_previous_id"])]["origin_place_id"]
        )
        for item in activities
        if item.get("route_from_previous_id") is not None
        and str(item["route_from_previous_id"]) in evidence["routes"]
    }
    itinerary_anchor_ids = {
        str(route[key])
        for route in evidence["routes"].values()
        for key in ("origin_place_id", "destination_place_id")
    }
    route_order = list(
        dict.fromkeys(
            str(item["route_from_previous_id"])
            for item in activities
            if item.get("route_from_previous_id") is not None
        )
    )
    catalogued_types: set[str] = set()
    visible_entities: dict[str, dict[str, Any]] = {}
    active_candidates: dict[str, str] = {}
    executed_routes: set[str] = set()
    executed_actions: set[str] = set()
    search_sessions: dict[str, tuple[Any, int]] = {}
    next_page_calls = 0
    completed_paths: set[str] = set()

    def candidate_context() -> tuple[int, tuple[str, ...]]:
        """Summarize evidence that has already been made visible by candidate actions."""

        purpose_order = (
            "outbound_transport",
            "return_transport",
            "attraction",
            "meal",
            "hotel",
        )
        purposes = set(active_candidates.values())
        return (
            len(active_candidates),
            tuple(_purpose_label(purpose) for purpose in purpose_order if purpose in purposes),
        )

    def execute(
        action: dict[str, Any],
        *,
        supervised: bool,
        reason: str,
        content: str,
        rationale_kind: str,
        protected_literals: tuple[str, ...] = (),
    ) -> Any:
        if not content.strip():
            raise SFTRebuildError(
                f"Programmatic ReAct action has empty rationale for {task_id}: {action}"
            )
        ordered_arguments = order_tool_arguments(action["tool"], action["arguments"], tools)
        output_action = {"tool": action["tool"], "arguments": ordered_arguments}
        action_fingerprint = _action_fingerprint(output_action)
        if action_fingerprint in executed_actions:
            raise SFTRebuildError(
                f"Programmatic policy repeated an identical action for {task_id}: "
                f"{output_action}"
            )
        executed_actions.add(action_fingerprint)
        call_id = f"call_programmatic_{len(steps):04d}"
        tool_call = {
            "id": call_id,
            "type": "function",
            "function": {
                "name": output_action["tool"],
                "arguments": json.dumps(
                    output_action["arguments"], ensure_ascii=False, separators=(",", ":")
                ),
            },
        }
        messages.append({"role": "assistant", "content": content, "tool_calls": [tool_call]})
        result = env.step(output_action)
        if result.info.get("valid_action") is not True:
            raise SFTRebuildError(
                f"Programmatic action failed for {task_id}: {output_action}; "
                f"{result.observation.error}"
            )
        if result.truncated:
            raise SFTRebuildError(
                f"Programmatic action budget exhausted for {task_id}: {output_action}"
            )
        model_response = serialize_model_tool_response(result)
        steps.append(
            {
                "index": len(steps),
                "api_turn": len(steps) + 1,
                "tool_call": deepcopy(tool_call),
                "action": deepcopy(output_action),
                "result": result.to_dict(),
                "model_tool_response": deepcopy(model_response),
            }
        )
        masks.append(supervised)
        mask_reasons.append(reason)
        rationale_specs.append(
            {
                "rationale_kind": rationale_kind,
                "protected_literals": list(protected_literals),
                "template_rationale": content,
            }
        )
        if output_action["tool"] == "save_candidate":
            entity_id = str(output_action["arguments"]["entity_id"])
            active_candidates[entity_id] = str(output_action["arguments"]["purpose"])
        elif output_action["tool"] == "remove_candidate":
            active_candidates.pop(str(output_action["arguments"]["candidate_id"]), None)
        tool_result = result.observation.tool_result or {}
        for item in tool_result.get("items", []):
            if not isinstance(item, Mapping):
                continue
            entity_id = item.get("place_id") or item.get("transport_id")
            if isinstance(entity_id, str):
                visible_entities[entity_id] = dict(item)
            for anchor_key in ("origin_anchor", "destination_anchor"):
                anchor = item.get(anchor_key)
                if isinstance(anchor, Mapping) and isinstance(anchor.get("place_id"), str):
                    visible_entities[str(anchor["place_id"])] = dict(anchor)
        inspected = tool_result.get("item")
        if isinstance(inspected, Mapping) and isinstance(inspected.get("place_id"), str):
            visible_entities[str(inspected["place_id"])] = dict(inspected)
        if not result.terminated and not result.truncated:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": output_action["tool"],
                    "content": json.dumps(
                        model_response, ensure_ascii=False, separators=(",", ":")
                    ),
                }
            )
        return result

    def _action_fingerprint(action: Mapping[str, Any]) -> str:
        return json.dumps(action, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def base_supervision() -> tuple[bool, str]:
        return True, "supervised_correct_action"

    def execute_catalog(tool: str, city: str, label: str, required_facet: str) -> None:
        supervised, reason = base_supervision()
        candidate_count, candidate_purposes = candidate_context()
        execute(
            {"tool": tool, "arguments": {"city": city}},
            supervised=supervised,
            reason=reason,
            content=_catalog_rationale(
                seed,
                task_id,
                position=len(steps),
                city=city,
                label=label,
                tool=tool,
                required_facet=required_facet,
                candidate_count=candidate_count,
                candidate_purposes=candidate_purposes,
            ),
            rationale_kind="discover_catalog_facets",
            protected_literals=(city, label, required_facet),
        )

    def maybe_catalog(entity: Mapping[str, Any], entity_type: str) -> bool:
        if entity_type in catalogued_types:
            return True
        required_facet = _catalog_requirement(record, entity, entity_type)
        if required_facet is None:
            return False
        catalogued_types.add(entity_type)
        city = str(entity["city"])
        if entity_type == "attraction":
            execute_catalog(
                "list_attraction_categories", city, "景点类别", required_facet
            )
        elif entity_type == "restaurant":
            execute_catalog(
                "list_restaurant_cuisines", city, "餐厅菜系", required_facet
            )
        elif entity_type == "hotel":
            execute_catalog(
                "list_hotel_features", city, "酒店特色和房型", required_facet
            )
        return True

    def search_strategy(
        candidate_id: str,
        entity: Mapping[str, Any],
        entity_type: str,
    ) -> tuple[dict[str, Any], str, tuple[str, ...], bool]:
        if entity_type in {"train", "airplane"}:
            mode = "火车" if entity_type == "train" else "飞机"
            earliest = _coarse_departure(str(entity["departure_time"]))
            return (
                {
                    "tool": "search_intercity_transport",
                    "arguments": {
                        "origin_city": entity["origin_city"],
                        "destination_city": entity["destination_city"],
                        "mode": entity_type,
                        "earliest_departure": earliest,
                    },
                },
                _search_rationale(
                    seed,
                    task_id,
                    position=len(steps),
                    entity=entity,
                    entity_type=entity_type,
                ),
                (str(entity["origin_city"]), str(entity["destination_city"]), mode),
                False,
            )

        nearby_path = path_by_kind.get("nearby_discovery")
        if nearby_path is not None and candidate_id == nearby_path.get("candidate_id"):
            anchor_id = str(nearby_path["anchor_id"])
            if anchor_id not in visible_entities:
                raise SFTRebuildError(
                    f"Nearby evidence-path anchor is not visible for {task_id}: {anchor_id}"
                )
            anchor_name = _entity_name(visible_entities[anchor_id])
            noun = {"attraction": "景点", "restaurant": "餐厅", "hotel": "酒店"}[
                entity_type
            ]
            radius = nearby_path["radius_km"]
            return (
                {
                    "tool": "search_nearby",
                    "arguments": {
                        "place_id": anchor_id,
                        "category": entity_type,
                        "radius_km": radius,
                        "top_k": 40,
                    },
                },
                (
                    f"已确定{anchor_name}的位置，下一段需要安排附近{noun}；"
                    f"先在{radius}公里内查看可衔接的候选。"
                ),
                (anchor_name, noun, str(radius)),
                False,
            )

        entity_name = _entity_name(entity)
        if entity_name in str(public["query"]):
            action = {
                "tool": {
                    "attraction": "search_attractions",
                    "restaurant": "search_restaurants",
                    "hotel": "search_hotels",
                }[entity_type],
                "arguments": {"city": entity["city"], "query": entity_name},
            }
            return (
                action,
                _search_rationale(
                    seed,
                    task_id,
                    position=len(steps),
                    entity=entity,
                    entity_type=entity_type,
                ),
                _search_literals(entity, entity_type),
                True,
            )

        catalog_seen = maybe_catalog(entity, entity_type)
        broad = _task_grounded_search_action(record, entity, entity_type)
        facet_key = {"attraction": "category", "restaurant": "cuisine"}.get(entity_type)
        if entity_type == "hotel":
            facet = str(broad["arguments"].get("hotel_type") or "")
            if not facet and broad["arguments"].get("room_type") is not None:
                facet = f"每间{broad['arguments']['room_type']}个床位的房型"
        else:
            facet = str(broad["arguments"].get(facet_key) or "")
        hard_price_limit = _hard_unit_price_limit(record, entity_type)
        noun = {"attraction": "景点", "restaurant": "餐厅", "hotel": "酒店"}[
            entity_type
        ]
        return (
            broad,
            (
                _averaged_budget_search_rationale(
                    city=str(entity["city"]),
                    noun=noun,
                    max_price=hard_price_limit,
                    facet=facet,
                )
                if entity_type in {"restaurant", "hotel"}
                and hard_price_limit is not None
                else _facet_search_rationale(
                    seed,
                    task_id,
                    position=len(steps),
                    city=str(entity["city"]),
                    facet=facet,
                    noun=noun,
                    entity_type=entity_type,
                )
                if catalog_seen and facet
                else _cost_search_rationale(
                    city=str(entity["city"]),
                    noun=noun,
                )
                if broad["arguments"].get("sort_by") == "price"
                else _unfiltered_search_rationale(
                    seed,
                    task_id,
                    position=len(steps),
                    city=str(entity["city"]),
                    noun=noun,
                    entity_type=entity_type,
                )
            ),
            tuple(
                item
                for item in (
                    str(entity["city"]),
                    facet,
                    noun,
                    (
                        _format_price(hard_price_limit)
                        if entity_type in {"restaurant", "hotel"}
                        and hard_price_limit is not None
                        else ""
                    ),
                )
                if item
            ),
            False,
        )

    def public_search_scope(action: Mapping[str, Any], entity_type: str) -> str:
        """Describe a query using only parameters available in the current turn."""

        arguments = action["arguments"]
        if action["tool"] == "search_intercity_transport":
            mode = "火车" if arguments.get("mode") == "train" else "飞机"
            return (
                f"从{arguments['origin_city']}到{arguments['destination_city']}的{mode}班次"
            )
        city = str(arguments.get("city", "当地"))
        if isinstance(arguments.get("query"), str):
            return str(arguments["query"])
        facet_key = {"attraction": "category", "restaurant": "cuisine"}.get(entity_type)
        if entity_type == "hotel":
            facet = str(arguments.get("hotel_type") or "")
            if not facet and arguments.get("room_type") is not None:
                facet = f"每间{arguments['room_type']}个床位的房型"
        else:
            facet = str(arguments.get(facet_key, "")) if facet_key is not None else ""
        noun = {"attraction": "景点", "restaurant": "餐厅", "hotel": "酒店"}.get(
            entity_type, "候选"
        )
        price = _numeric_price(arguments.get("max_price"))
        price_text = f"价格不超过{_format_price(price)}元的" if price is not None else ""
        scope = (
            f"{city}的{price_text}{facet}{noun}"
            if facet
            else f"{city}的{price_text}{noun}"
        )
        return f"按价格排序的{scope}" if arguments.get("sort_by") == "price" else scope

    def route_grounded_nearby_search(
        candidate_id: str,
        entity_type: str,
    ) -> tuple[dict[str, Any], str, tuple[str, ...]] | None:
        """Use an established itinerary anchor only when the target is on page one."""

        if entity_type not in {"attraction", "restaurant", "hotel"}:
            return None
        route_anchor_id = route_anchor_by_candidate.get(candidate_id)
        established_ids = [
            entity_id
            for entity_id in visible_entities
            if entity_id in itinerary_anchor_ids
            and (entity_id not in candidate_ids or entity_id in active_candidates)
        ]
        anchor_ids = list(
            dict.fromkeys(
                item
                for item in (route_anchor_id, *established_ids)
                if item is not None and item in visible_entities
            )
        )
        for anchor_id in anchor_ids:
            for radius in (2, 5, 10, 20, 50):
                nearby = runtime_backend.search_nearby(
                    place_id=anchor_id,
                    category=entity_type,
                    radius_km=radius,
                    top_k=40,
                )
                if any(
                    str(item.get("place_id")) == candidate_id
                    for item in nearby[:10]
                ):
                    break
            else:
                continue
            anchor_name = _entity_name(visible_entities[anchor_id])
            noun = {
                "attraction": "景点",
                "restaurant": "餐厅",
                "hotel": "酒店",
            }[entity_type]
            return (
                {
                    "tool": "search_nearby",
                    "arguments": {
                        "place_id": anchor_id,
                        "category": entity_type,
                        "radius_km": radius,
                        "top_k": 40,
                    },
                },
                (
                    f"上一轮城市级{noun}结果没有提供候选相对{anchor_name}的距离，"
                    f"无法据此判断下一段是否便于衔接；当前已确定{anchor_name}，"
                    f"因此改查其{radius}公里内的{noun}。"
                ),
                (anchor_name, str(radius), noun),
            )
        return None

    def complete_evidence_paths_before_save(
        candidate_id: str,
        entity: Mapping[str, Any],
        entity_type: str,
    ) -> None:
        """Execute validation gates that causally precede candidate adoption."""

        del entity, entity_type
        path = path_by_kind.get("opening_verification")
        if path is None or candidate_id != path.get("candidate_id"):
            return
        supervised, reason = base_supervision()
        entity_name = _entity_name(visible_entities[candidate_id])
        at_time = str(path["at_time"])
        opening = execute(
            {
                "tool": "check_place_open",
                "arguments": {"place_id": candidate_id, "at_time": at_time},
            },
            supervised=supervised,
            reason=reason,
            content=f"计划在{at_time}安排{entity_name}，现在核对该时刻是否开放。",
            rationale_kind="verify_scheduled_place_open",
            protected_literals=(entity_name, at_time),
        )
        if (opening.observation.tool_result or {}).get("is_open") is not True:
            raise SFTRebuildError(
                f"Opening evidence path unexpectedly failed for {task_id}: {candidate_id}"
            )
        completed_paths.add("opening_verification")

    def complete_evidence_paths_after_save(
        candidate_id: str,
        purpose: str,
    ) -> None:
        """Execute a preselected, price-grounded candidate-comparison subgraph."""

        path = path_by_kind.get("candidate_comparison")
        if path is None or candidate_id != path.get("candidate_id"):
            return
        target = visible_entities[candidate_id]
        alternative_id = str(path["alternative_id"])
        if alternative_id not in visible_entities:
            entity_type = _entity_type(target)
            noun = {"attraction": "景点", "restaurant": "餐厅", "hotel": "酒店"}[entity_type]
            execute(
                _task_grounded_search_action(record, target, entity_type),
                supervised=True,
                reason="supervised_correct_action",
                content=(
                    f"{_entity_name(target)}的类别和价格已可见，"
                    f"再按同一条件查看{target['city']}的{noun}，收集可比较的备选。"
                ),
                rationale_kind="search_price_comparable_alternatives",
                protected_literals=(_entity_name(target), str(target["city"]), noun),
            )
        if alternative_id not in visible_entities:
            raise SFTRebuildError(
                f"Comparison-path alternative was not visible for {task_id}: {alternative_id}"
            )
        alternative = visible_entities[alternative_id]
        entity_type = _entity_type(target)
        target_price = _comparable_unit_price(record, target, entity_type)
        alternative_price = _comparable_unit_price(record, alternative, entity_type)
        if target_price is None or alternative_price is None or alternative_price <= target_price:
            raise SFTRebuildError(
                f"Comparison path has no valid price ordering for {task_id}."
            )
        supervised, reason = base_supervision()
        target_name = _entity_name(target)
        alternative_name = _entity_name(alternative)
        price_limit = _hard_unit_price_limit(record, entity_type)
        price_basis = "人均可比价格" if entity_type in {"restaurant", "hotel"} else "价格"
        hard_limit_text = (
            f"，且都未超过题面{_format_price(price_limit)}元的硬上限"
            if price_limit is not None
            else ""
        )
        execute(
            {
                "tool": "save_candidate",
                "arguments": {"entity_id": alternative_id, "purpose": purpose},
            },
            supervised=supervised,
            reason=reason,
            content=(
                f"{alternative_name}{price_basis}为{_format_price(alternative_price)}元，"
                f"{target_name}为{_format_price(target_price)}元；两者属于同类候选"
                f"{hard_limit_text}，因此先保存前者作为{_purpose_label(purpose)}备选，"
                "再比较成本。"
            ),
            rationale_kind="save_price_alternative",
            protected_literals=(alternative_name, target_name, _purpose_label(purpose)),
        )
        execute(
            {"tool": "list_candidates", "arguments": {}},
            supervised=supervised,
            reason=reason,
            content=(
                f"现在已保存的{target_name}和{alternative_name}都满足当前硬条件；"
                "调用候选清单核对二者的价格和用途，再决定保留哪一个。"
            ),
            rationale_kind="review_price_alternatives",
            protected_literals=(target_name, alternative_name),
        )
        execute(
            {"tool": "remove_candidate", "arguments": {"candidate_id": alternative_id}},
            supervised=supervised,
            reason=reason,
            content=(
                f"{alternative_name}{price_basis}为{_format_price(alternative_price)}元，"
                f"高于{target_name}的{_format_price(target_price)}元，"
                "同类比较中没有成本优势，因此移除该备选。"
            ),
            rationale_kind="remove_more_expensive_alternative",
            protected_literals=(
                alternative_name,
                target_name,
                _format_price(alternative_price),
                _format_price(target_price),
            ),
        )
        completed_paths.add("candidate_comparison")

    def execute_route(route_id: str) -> None:
        """Query one necessary route as soon as both of its endpoints are usable."""

        route = evidence["routes"][route_id]
        first_segment = route["segments"][0]
        supervised, reason = base_supervision()
        execute(
            {
                "tool": "get_route",
                "arguments": {
                    "origin_place_id": route["origin_place_id"],
                    "destination_place_id": route["destination_place_id"],
                    "mode": route["mode"],
                    "start_time": first_segment["start_time"],
                },
            },
            supervised=supervised,
            reason=reason,
            content=_route_rationale(
                seed,
                task_id,
                position=len(steps),
                origin_name=str(first_segment["start"]),
                destination_name=str(route["segments"][-1]["end"]),
                mode=str(route["mode"]),
                start_time=str(first_segment["start_time"]),
            ),
            rationale_kind="complete_route_evidence",
            protected_literals=(
                str(first_segment["start"]),
                str(route["segments"][-1]["end"]),
                _route_mode_label(str(route["mode"])),
                str(first_segment["start_time"]),
            ),
        )
        executed_routes.add(route_id)

    def route_endpoint_ready(entity_id: str) -> bool:
        if entity_id not in visible_entities:
            return False
        return entity_id not in candidate_ids or entity_id in active_candidates

    def drain_ready_routes() -> int:
        """Interleave newly available routes instead of batching them before submit."""

        executed = 0
        for route_id in route_order:
            if route_id in executed_routes:
                continue
            route = evidence["routes"][route_id]
            if not route_endpoint_ready(str(route["origin_place_id"])):
                continue
            if not route_endpoint_ready(str(route["destination_place_id"])):
                continue
            execute_route(route_id)
            executed += 1
        return executed

    def reveal_and_save(candidate_id: str) -> None:
        """Collect only the evidence needed by the fixed, already-feasible plan."""

        nonlocal next_page_calls
        entity = entities[candidate_id]
        entity_type = str(entity.get("entity_type") or entity.get("mode"))
        if entity_type in {"train", "airplane"}:
            purpose = (
                "outbound_transport"
                if entity["origin_city"] == public["start_city"]
                else "return_transport"
            )
        else:
            purpose = {"attraction": "attraction", "restaurant": "meal", "hotel": "hotel"}[
                entity_type
            ]
        supervised, reason = base_supervision()
        entity_name = _entity_name(entity)
        if candidate_id not in visible_entities:
            search, search_content, search_literals, name_grounded = search_strategy(
                candidate_id, entity, entity_type
            )
            search_key = _action_fingerprint(search)
            session = search_sessions.get(search_key)
            if session is None:
                result = execute(
                    search,
                    supervised=supervised,
                    reason=reason,
                    content=search_content,
                    rationale_kind="search_evidence",
                    protected_literals=search_literals,
                )
                page_number = 1
                search_sessions[search_key] = (result, page_number)
            else:
                result, page_number = session
            if candidate_id not in visible_entities and not name_grounded:
                nearby_fallback = route_grounded_nearby_search(candidate_id, entity_type)
                if nearby_fallback is not None:
                    search, search_content, search_literals = nearby_fallback
                    search_key = _action_fingerprint(search)
                    session = search_sessions.get(search_key)
                    if session is None:
                        result = execute(
                            search,
                            supervised=supervised,
                            reason=reason,
                            content=search_content,
                            rationale_kind="search_route_continuity",
                            protected_literals=search_literals,
                        )
                        page_number = 1
                        search_sessions[search_key] = (result, page_number)
                    else:
                        result, page_number = session
            page_scope = public_search_scope(search, entity_type)
            consecutive_pages = 0
            planned_search_ids = (
                {
                    planned_id
                    for planned_id in candidate_order
                    if _entity_type(entities[planned_id]) == entity_type
                    and _entity_name(entities[planned_id]) not in str(public["query"])
                    and _action_fingerprint(
                        _task_grounded_search_action(
                            record,
                            entities[planned_id],
                            entity_type,
                        )
                    )
                    == search_key
                }
                if entity_type in {"attraction", "restaurant", "hotel"}
                else set()
            )
            visible_planned_count = len(planned_search_ids & visible_entities.keys())
            required_candidate_count = _required_activity_count(record, entity_type)
            resolved_candidate_count = sum(
                _entity_type(visible_entities[entity_id]) == entity_type
                for entity_id in active_candidates
                if entity_id in visible_entities
            )
            if candidate_id not in visible_entities and not _pagination_is_task_grounded(
                name_grounded=name_grounded,
                search_tool=str(search["tool"]),
                planned_candidate_count=len(planned_search_ids),
                visible_planned_count=visible_planned_count,
                required_candidate_count=required_candidate_count,
                resolved_candidate_count=resolved_candidate_count,
            ):
                raise SFTRebuildError(
                    "Task would require pagination without a visible unresolved predicate: "
                    f"{task_id}:{candidate_id}. Regenerate the witness within the global "
                    "first-page/grounded-nearby policy or add a real multi-entity gap."
                )
            while candidate_id not in visible_entities:
                if consecutive_pages >= MAX_CONSECUTIVE_TOOL_CALLS:
                    raise SFTRebuildError(
                        f"Task {task_id} needs more than three consecutive next_page calls."
                    )
                page = (result.observation.tool_result or {}).get("page", {})
                cursor = page.get("next_cursor")
                if not isinstance(cursor, str):
                    raise SFTRebuildError(
                        "Search did not expose witness entity "
                        f"{candidate_id} within the available cursor chain."
                    )
                page_label = entity_name if name_grounded else _entity_type_label(entity_type)
                result = execute(
                    {"tool": "next_page", "arguments": {"cursor": cursor}},
                    supervised=supervised,
                    reason=reason,
                    content=_page_rationale(
                        seed,
                        task_id,
                        position=len(steps),
                        entity_name=page_label,
                        public_scope=page_scope,
                        page_number=page_number + 1,
                        pages_checked=page_number,
                        planned_candidate_count=len(planned_search_ids),
                        visible_planned_count=visible_planned_count,
                        required_candidate_count=required_candidate_count,
                        resolved_candidate_count=resolved_candidate_count,
                    ),
                    rationale_kind="continue_search",
                    protected_literals=(page_label,),
                )
                next_page_calls += 1
                page_number += 1
                consecutive_pages += 1
                visible_planned_count = len(planned_search_ids & visible_entities.keys())
                search_sessions[search_key] = (result, page_number)
            if (
                search["tool"] == "search_nearby"
                and nearby_path is not None
                and candidate_id == str(nearby_path.get("candidate_id"))
            ):
                completed_paths.add("nearby_discovery")
        if candidate_id not in visible_entities:
            raise SFTRebuildError(f"Witness entity was not made visible: {candidate_id}.")
        complete_evidence_paths_before_save(candidate_id, entity, entity_type)
        execute(
            {
                "tool": "save_candidate",
                "arguments": {"entity_id": candidate_id, "purpose": purpose},
            },
            supervised=supervised,
            reason=reason,
            content=_save_rationale(
                seed,
                task_id,
                position=len(steps),
                entity_name=entity_name,
                purpose=purpose,
                decision_facts=_save_decision_facts(
                    record,
                    public,
                    visible_entities[candidate_id],
                    entity_type,
                    purpose,
                ),
            ),
            rationale_kind="save_evidence",
            protected_literals=(entity_name, _purpose_label(purpose)),
        )
        complete_evidence_paths_after_save(candidate_id, purpose)

    try:
        remaining_candidates = list(candidate_order)
        while remaining_candidates:
            # A broad search can expose several later witness entities at once. If the
            # previous turn already saved one, prefer a candidate that still needs a
            # real search so the policy does not learn a long save_candidate burst.
            if steps and steps[-1]["action"]["tool"] == "save_candidate":
                candidate_id = next(
                    (
                        item
                        for item in remaining_candidates
                        if item not in visible_entities
                    ),
                    remaining_candidates[0],
                )
            else:
                candidate_id = remaining_candidates[0]
            reveal_and_save(candidate_id)
            remaining_candidates.remove(candidate_id)
            # Query every route that became ready because of this candidate. This
            # dependency-driven placement avoids a large pre-submit route batch;
            # the number executed here follows the itinerary graph rather than a
            # universal per-tool run cap.
            drain_ready_routes()
        if completed_paths != set(path_by_kind):
            raise SFTRebuildError(
                f"Programmatic evidence paths were not completed for {task_id}: "
                f"expected={sorted(path_by_kind)}, actual={sorted(completed_paths)}."
            )
        drain_ready_routes()
        if executed_routes != set(route_order):
            missing = sorted(set(route_order) - executed_routes)
            raise SFTRebuildError(
                f"Programmatic route dependencies were never satisfied for {task_id}: {missing}"
            )
        longest_tool, longest_run = _longest_tool_run(
            [str(step["action"]["tool"]) for step in steps]
        )
        if longest_run > MAX_CONSECUTIVE_TOOL_CALLS:
            raise SFTRebuildError(
                f"Programmatic policy produced {longest_run} consecutive "
                f"{longest_tool} calls for {task_id}; maximum is "
                f"{MAX_CONSECUTIVE_TOOL_CALLS}."
            )
        kinds = {str(item["activity_type"]) for item in activities}
        evidence_names = ["往返交通", "景点", "完整路线"]
        if "accommodation" in kinds:
            evidence_names.append("住宿")
        if kinds & {"breakfast", "lunch", "dinner"}:
            evidence_names.append("餐饮")
        candidate_count, _candidate_purposes = candidate_context()
        submit_landmarks, submit_literal_names = _submit_landmarks(
            active_candidates,
            visible_entities,
        )
        submit_content = _submit_reflection(
            seed,
            task_id,
            evidence_names,
            candidate_count=candidate_count,
            days=int(public["days"]),
            route_count=len(route_order),
            evidence_landmarks=submit_landmarks,
        )
        terminal = execute(
            {"tool": "submit_plan", "arguments": {"plan": plan}},
            supervised=True,
            reason="supervised_correct_action",
            content=submit_content,
            rationale_kind="submit_evidence_ready_plan",
            protected_literals=tuple(evidence_names) + submit_literal_names,
        )
        detail = terminal.info.get("reward_detail")
        if (
            not terminal.terminated
            or terminal.info.get("termination_reason") != "plan_submitted"
            or terminal.reward != 1.0
            or not isinstance(detail, Mapping)
            or detail.get("all_hard_pass") is not True
        ):
            raise SFTRebuildError(f"Programmatic replay failed for {task_id}: {detail}")
        if len(steps) > DEFAULT_MAX_VALID_STEPS:
            raise SFTRebuildError(
                f"Programmatic trajectory exceeded {DEFAULT_MAX_VALID_STEPS} actions for {task_id}."
            )
    finally:
        env.close()

    row = {
        "trajectory_version": TRAJECTORY_VERSION,
        "episode_id": reset.episode_id,
        "task_id": task_id,
        "model": "deterministic-programmatic-teacher",
        "success": True,
        "termination_reason": "plan_submitted",
        "step_count": len(steps),
        "api_turn_count": len(steps),
        "final_plan": plan,
        "final_text": None,
        "final_reward": 1.0,
        "reward_detail": dict(detail),
        "rft_accepted": True,
        "usage": {},
        "user_content_format": USER_CONTENT_FORMAT,
        "tool_response_mode": "delta",
        "model_tool_response_version": MODEL_TOOL_RESPONSE_VERSION,
        "messages": messages,
        "tools": tools,
        "steps": steps,
        "assistant_loss_mask": masks,
        "mask_reasons": mask_reasons,
        "sample_family": family,
        "batch_metadata": {
            "thinking": "disabled",
            "source": PROGRAMMATIC_POLICY_VERSION,
            "max_valid_steps": DEFAULT_MAX_VALID_STEPS,
            "max_consecutive_tool_calls": MAX_CONSECUTIVE_TOOL_CALLS,
            "evidence_paths": [dict(path) for path in evidence_paths],
        },
    }
    audit = {
        "programmatic_policy_version": PROGRAMMATIC_POLICY_VERSION,
        "task_id": task_id,
        "sample_family": family,
        "question_batch": source_question_batch,
        "blueprint_semantic_hash": record["blueprint"]["blueprint_id"],
        "assistant_loss_mask": masks,
        "mask_reasons": mask_reasons,
        "turns": [
            {
                "position": index,
                "tool": step["action"]["tool"],
                "loss_mask": masks[index],
                "mask_reason": mask_reasons[index],
                "visible_reflection": messages[2 + index * 2].get("content", ""),
                **rationale_specs[index],
            }
            for index, step in enumerate(steps)
        ],
        "max_valid_steps": DEFAULT_MAX_VALID_STEPS,
        "max_consecutive_tool_calls": MAX_CONSECUTIVE_TOOL_CALLS,
        "next_page_calls": next_page_calls,
        "evidence_paths": [dict(path) for path in evidence_paths],
        "action_count": len(steps),
        "termination_reason": "plan_submitted",
        "replay_reward": 1.0,
        "all_hard_pass": True,
        "reward_groups": dict(detail.get("group_results", {})),
    }
    return row, audit


def _entity_name(entity: Mapping[str, Any]) -> str:
    return str(entity.get("name") or entity.get("source_id") or "目标候选")


def _choice(
    templates: tuple[str, ...], seed: int, task_id: str, position: int, scope: str
) -> str:
    digest = hashlib.sha256(f"{seed}:{task_id}:{position}:{scope}".encode()).hexdigest()
    return templates[int(digest, 16) % len(templates)]


def _coarse_departure(value: str) -> str:
    hour = int(value[:2])
    return f"{hour - hour % 3:02d}:00"


def _entity_type_label(entity_type: str) -> str:
    return {
        "attraction": "合适的景点",
        "restaurant": "合适的餐厅",
        "hotel": "合适的酒店",
        "train": "合适的火车班次",
        "airplane": "合适的航班",
    }[entity_type]


def _search_literals(entity: Mapping[str, Any], entity_type: str) -> tuple[str, ...]:
    if entity_type in {"train", "airplane"}:
        mode = "火车" if entity_type == "train" else "飞机"
        return (str(entity["origin_city"]), str(entity["destination_city"]), mode)
    noun = {"attraction": "景点", "restaurant": "餐厅", "hotel": "酒店"}[entity_type]
    return (_entity_name(entity), noun)


def _search_rationale(
    seed: int,
    task_id: str,
    *,
    position: int,
    entity: Mapping[str, Any],
    entity_type: str,
) -> str:
    if entity_type in {"train", "airplane"}:
        mode = "火车" if entity_type == "train" else "飞机"
        template = _choice(
            (
                "先查询从{origin}到{destination}的{mode}班次，核实这段城际交通的可用时间和候选。",
                "当前需要补充{origin}前往{destination}的城际交通证据，先检索可用的{mode}班次。",
                "为了安排{origin}到{destination}这一程，先查询{mode}候选及其准确时刻。",
            ),
            seed,
            task_id,
            position,
            "transport-search",
        )
        return template.format(
            origin=entity["origin_city"], destination=entity["destination_city"], mode=mode
        )
    noun = {"attraction": "景点", "restaurant": "餐厅", "hotel": "酒店"}[entity_type]
    template = _choice(
        (
            "接下来查询{city}的{name}，确认这个{noun}是否能为计划提供可用证据。",
            "行程还需要核实{name}，先在{city}的{noun}候选中检索它。",
            "为了完善本地安排，先查询{city}的{name}并取得准确的{noun}信息。",
        ),
        seed,
        task_id,
        position,
        "place-search",
    )
    return template.format(
        city=entity["city"], name=_entity_name(entity), noun=noun
    )


def _save_rationale(
    seed: int,
    task_id: str,
    *,
    position: int,
    entity_name: str,
    purpose: str,
    decision_facts: tuple[str, ...] = (),
) -> str:
    purpose_text = _purpose_label(purpose)
    if decision_facts:
        return (
            f"{'；'.join(decision_facts)}。这些当前可见事实说明它可用于"
            f"{purpose_text}，后续计划会引用该实体，因此现在保存候选。"
        )
    template = _choice(
        (
            "当前搜索结果已经明确返回{name}，计划仍缺少{purpose}证据；"
            "后续提交会引用该实体，因此现在保存候选。",
            "{name}已经出现在当前工具结果中，可作为计划需要的{purpose}；"
            "为保证最终提交只引用已保存证据，现在保存它。",
        ),
        seed,
        task_id,
        position,
        "save",
    )
    return template.format(name=entity_name, purpose=purpose_text)


def _save_decision_facts(
    record: Mapping[str, Any],
    public: Mapping[str, Any],
    entity: Mapping[str, Any],
    entity_type: str,
    purpose: str,
) -> tuple[str, ...]:
    """Explain adoption using facts visible before ``save_candidate``."""

    name = _entity_name(entity)
    facts: list[str] = []
    if entity_type in {"train", "airplane"}:
        mode = "火车" if entity_type == "train" else "飞机"
        origin = str(entity.get("origin_city") or "起点")
        destination = str(entity.get("destination_city") or "终点")
        departure = entity.get("departure_time")
        arrival = entity.get("arrival_time")
        timing = (
            f"，班次时刻为{departure}–{arrival}"
            if isinstance(departure, str) and isinstance(arrival, str)
            else ""
        )
        facts.append(f"搜索结果显示{name}是{origin}到{destination}的{mode}{timing}")
        facts.append(f"当前计划需要一项{_purpose_label(purpose)}")
        return tuple(facts)

    query = str(public.get("query") or public.get("public_query") or "")
    if name and name in query:
        facts.append(f"题面指定的{name}已经由当前搜索结果确认")
    required_facet = _catalog_requirement(record, entity, entity_type)
    if required_facet is not None:
        facts.append(f"结果字段显示{name}符合题面的{required_facet}要求")
    price = _numeric_price(entity.get("price"))
    price_limit = _hard_unit_price_limit(record, entity_type)
    comparable_price = _comparable_unit_price(record, entity, entity_type)
    if (
        entity_type == "hotel"
        and price is not None
        and comparable_price is not None
        and price_limit is not None
    ):
        travelers = int(record["task_spec"]["trip"]["travelers"])
        rooms = round(comparable_price * travelers / price)
        facts.append(
            f"{name}每间每晚{_format_price(price)}元，按{rooms}间房和{travelers}人计算，"
            f"人均每晚{_format_price(comparable_price)}元，未超过题面"
            f"{_format_price(price_limit)}元的硬上限"
        )
    elif price is not None and comparable_price is not None and price_limit is not None:
        if comparable_price <= price_limit:
            facts.append(
                f"{name}本餐人均{_format_price(comparable_price)}元，不高于题面"
                f"{_format_price(price_limit)}元的每餐平均上限，有助于控制最终平均值"
            )
        else:
            facts.append(
                f"{name}本餐人均{_format_price(comparable_price)}元，高于题面"
                f"{_format_price(price_limit)}元的平均目标；该约束按全部已安排餐次"
                "取平均，后续必须搭配更低价餐并在提交前复核"
            )
    elif price is not None:
        facts.append(
            f"当前结果显示{name}价格为{_format_price(price)}元，题面没有更具体的"
            f"{_purpose_label(purpose)}硬筛选条件"
        )
    if not facts:
        facts.append(
            f"当前搜索结果已经返回{name}的可引用ID，题面没有更具体的"
            f"{_purpose_label(purpose)}筛选条件"
        )
    return tuple(dict.fromkeys(facts))


def _purpose_label(purpose: str) -> str:
    return {
        "outbound_transport": "去程交通",
        "return_transport": "返程交通",
        "attraction": "景点",
        "meal": "用餐地点",
        "hotel": "住宿",
    }[purpose]


def _submit_landmarks(
    active_candidates: Mapping[str, str],
    visible_entities: Mapping[str, Mapping[str, Any]],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Select a few already-saved entities for a grounded submit summary.

    The final reflection should be specific enough to sound like it follows the
    episode, but it must not reveal a witness entity that the model has not
    observed.  ``active_candidates`` only contains save_candidate results that
    remain after any remove_candidate correction, and every returned name came
    from an earlier tool observation.
    """

    labels = (
        ("outbound_transport", "去程"),
        ("attraction", "景点"),
        ("meal", "用餐"),
        ("hotel", "住宿"),
        ("return_transport", "返程"),
    )
    landmarks: list[str] = []
    literal_names: list[str] = []
    for purpose, label in labels:
        candidate_id = next(
            (entity_id for entity_id, saved_purpose in active_candidates.items()
             if saved_purpose == purpose),
            None,
        )
        if candidate_id is None:
            continue
        entity = visible_entities.get(candidate_id)
        if entity is None:
            continue
        name = _entity_name(entity)
        landmarks.append(f"{label}{name}")
        literal_names.append(name)
    return tuple(landmarks[:3]), tuple(literal_names[:3])


def _page_rationale(
    seed: int,
    task_id: str,
    *,
    position: int,
    entity_name: str,
    public_scope: str,
    page_number: int,
    pages_checked: int,
    planned_candidate_count: int = 1,
    visible_planned_count: int = 0,
    required_candidate_count: int = 0,
    resolved_candidate_count: int = 0,
) -> str:
    if (
        required_candidate_count > 1
        and 0 < resolved_candidate_count < required_candidate_count
    ):
        return (
            f"题目要求安排{required_candidate_count}个不同的{entity_name}；"
            f"目前只确定了{resolved_candidate_count}个，还缺"
            f"{required_candidate_count - resolved_candidate_count}个。"
            f"保持{public_scope}的公开搜索条件，继续查看第{page_number}页。"
        )
    if planned_candidate_count > 1 and 0 < visible_planned_count < planned_candidate_count:
        return (
            f"既定行程还需要{planned_candidate_count}个不同的{entity_name}；"
            f"{public_scope}前{pages_checked}页只展示了其中{visible_planned_count}个，尚缺"
            f"{planned_candidate_count - visible_planned_count}个；保持当前公开筛选条件，"
            f"继续查看第{page_number}页。"
        )
    if entity_name.startswith("合适的"):
        raise SFTRebuildError(
            "Unnamed pagination has no visible name, ranking objective, or quantity gap."
        )
    templates = (
        "{scope}当前结果里还没有出现{entity}，继续查看第{page}页候选。",
        "还需在{scope}中定位{entity}，因此翻到第{page}页继续检索。",
        "现有{scope}候选尚未找到{entity}，下一步查看第{page}页。",
        "已核对前{checked}页{scope}结果，{entity}仍未出现；继续进入第{page}页。",
        "前{checked}页结束后还缺{entity}，保持{scope}条件继续查第{page}页。",
        "本次{scope}搜索仍有下一页；为了找到{entity}，继续读取第{page}页。",
        "当前筛选条件不变，前{checked}页{scope}结果暂未定位{entity}，继续检索第{page}页候选。",
        "已完成{scope}前{checked}页的核对，下一页继续寻找{entity}。",
    )
    template = _choice(templates, seed, task_id, position, f"page:{public_scope}")
    return template.format(
        entity=entity_name,
        scope=public_scope,
        page=page_number,
        checked=pages_checked,
    )


def _catalog_rationale(
    seed: int,
    task_id: str,
    *,
    position: int,
    city: str,
    label: str,
    tool: str,
    required_facet: str | None = None,
    candidate_count: int = 0,
    candidate_purposes: tuple[str, ...] = (),
) -> str:
    if required_facet is not None:
        return (
            f"题目明确要求{required_facet}；先核对{city}的{label}目录中存在这一选项，"
            "再按该条件搜索。"
        )
    saved_context = (
        f"目前已保存{candidate_count}项候选（{'、'.join(candidate_purposes)}）"
        if candidate_purposes
        else "当前还没有已保存候选"
    )
    templates = (
        "先查看{city}当前可用的{label}，再据此筛选合适候选。",
        "需要先确定{city}{label}的可选范围，查看目录后再开始检索。",
        "先打开{city}的{label}目录，避免在没有依据的情况下设定筛选条件。",
        "为了让后续搜索有明确条件，先核对{city}提供哪些{label}。",
        "本地安排还没有确定筛选方向，先从{city}的{label}中了解可用选项。",
        "{context}；{label}的筛选方向尚未确定，先查看{city}目录。",
        "在已有证据的基础上，先用{city}的{label}目录确定下一次搜索条件。",
        "选择{city}候选前先获取{label}目录，保证后续筛选只使用已见条件。",
        "先把{city}{label}的可选项放入当前上下文，再按其中一个方向继续检索。",
        "本地候选的条件还不能凭空设定；先列出{city}的{label}再缩小范围。",
    )
    template = _choice(templates, seed, task_id, position, f"catalog:{tool}")
    return template.format(city=city, label=label, context=saved_context)


def _facet_search_rationale(
    seed: int,
    task_id: str,
    *,
    position: int,
    city: str,
    facet: str,
    noun: str,
    entity_type: str,
) -> str:
    templates = (
        "目录中已经确认{facet}可用，现在按这一条件查看{city}的{noun}候选。",
        "刚才的目录包含{facet}，接下来以它为条件检索{city}的{noun}。",
        "先用已看到的{facet}筛选{city}{noun}，从结果中继续判断具体安排。",
        "{city}的目录给出了{facet}这一方向，现在查看对应的{noun}列表。",
        "为了缩小{city}{noun}的范围，沿用目录中的{facet}条件进行查询。",
    )
    template = _choice(templates, seed, task_id, position, f"facet:{entity_type}")
    return template.format(city=city, facet=facet, noun=noun)


def _unfiltered_search_rationale(
    seed: int,
    task_id: str,
    *,
    position: int,
    city: str,
    noun: str,
    entity_type: str,
) -> str:
    template = _choice(
        (
            "题面没有限定具体{noun}类别，先查看{city}的公开候选，再从结果中选择可行安排。",
            "当前只需补充一处可行{noun}，先不添加额外筛选条件，查看{city}候选。",
            "没有更多题面条件可以缩小{noun}范围，先查询{city}的首批公开候选。",
        ),
        seed,
        task_id,
        position,
        f"unfiltered:{entity_type}",
    )
    return template.format(city=city, noun=noun)


def _cost_search_rationale(*, city: str, noun: str) -> str:
    return f"题面希望降低总花费；先按价格从低到高查看{city}的{noun}候选。"


def _averaged_budget_search_rationale(
    *, city: str, noun: str, max_price: float, facet: str = ""
) -> str:
    facet_text = f"，搜索时仍使用题面的{facet}条件" if facet else ""
    basis = "每间房报价" if noun == "酒店" else "单家餐厅的人均报价"
    average = "每人每晚平均" if noun == "酒店" else "所有已安排餐次的每餐平均"
    return (
        f"题面要求{average}不超过{_format_price(max_price)}元；工具返回的是{basis}，"
        f"不能把平均上限直接当作单项max_price{facet_text}。先查询{city}{noun}候选，"
        "随后按题面计价口径核算组合。"
    )


def _route_rationale(
    seed: int,
    task_id: str,
    *,
    position: int,
    origin_name: str,
    destination_name: str,
    mode: str,
    start_time: str,
) -> str:
    mode_text = _route_mode_label(mode)
    template = _choice(
        (
            "计划在{time}从{origin}衔接到{destination}，现在查询{mode}路线并补齐这一段交通证据。",
            "接下来核实{time}从{origin}到{destination}的市内衔接，按计划查询{mode}路线。",
            "为了保证行程地点连续，按{time}出发查询从{origin}前往{destination}的{mode}路线。",
            "前一段安排结束后将于{time}离开{origin}，先确认到{destination}的{mode}路线衔接。",
        ),
        seed,
        task_id,
        position,
        "route",
    )
    return template.format(
        origin=origin_name,
        destination=destination_name,
        mode=mode_text,
        time=start_time,
    )


def _route_mode_label(mode: str) -> str:
    return {"taxi": "出租车", "metro": "地铁", "walk": "步行"}.get(mode, mode)


def _submit_reflection(
    seed: int,
    task_id: str,
    evidence_names: list[str],
    *,
    candidate_count: int | None = None,
    days: int | None = None,
    route_count: int | None = None,
    evidence_landmarks: tuple[str, ...] = (),
) -> str:
    evidence_text = "、".join(evidence_names)
    if candidate_count is None or days is None or route_count is None:
        templates = (
            "{evidence}证据均已齐全，现在可以提交完整计划。",
            "题目所需的{evidence}已经全部核实，可以直接提交方案。",
            "当前已具备完整的{evidence}证据，下一步提交最终计划。",
            "候选和衔接信息已覆盖{evidence}，没有待补的关键证据，可以提交。",
            "已逐项确认{evidence}，现在把这些已保存的证据组织为最终计划并提交。",
            "行程所需的{evidence}都已准备好，提交后方案即可接受验证。",
        )
    elif evidence_landmarks:
        templates = (
            "{landmarks}等关键候选已保存；{days}天行程的{count}项候选和{routes}段路线已覆盖{evidence}，现在提交。",
            "以{landmarks}等已见证据为基础，{days}天方案的{count}项候选和{routes}段路线均已补齐，提交。",
            "当前候选集中已有{landmarks}等安排，{days}天行程的{count}项候选连同{routes}段路线可形成完整方案，提交。",
            "{landmarks}已经分别落实到已保存候选；{days}天安排的{count}项候选和{routes}段路线覆盖{evidence}，现在提交。",
            "已核对{landmarks}等关键选择；{days}天行程的{count}项候选及{routes}段衔接没有缺口，可以提交。",
            "围绕{landmarks}等已保存候选，{days}天安排的{count}项候选和{routes}段路线已经准备完毕，提交最终计划。",
            "{landmarks}等候选均可在当前上下文中引用；{days}天计划的{count}项候选、{routes}段路线和{evidence}已齐全，提交。",
            "现在的{count}项候选包含{landmarks}等关键安排，{days}天行程需要的{routes}段路线与{evidence}已齐，提交。",
        )
    else:
        templates = (
            "{days}天行程已保存{count}项候选并核对{routes}段市内路线，{evidence}证据齐全，可以提交。",
            "围绕这{days}天安排，{count}项候选和{routes}段路线均已核实；现在提交包含{evidence}的完整计划。",
            "当前{count}项已保存候选已覆盖{evidence}，且{routes}段路线已补齐，提交这份{days}天方案。",
            "{evidence}均已落实到{count}项候选和{routes}段衔接中，没有待补的关键证据，提交{days}天行程。",
            "已为{days}天行程逐项确认{evidence}，共保留{count}项候选并完成{routes}段路线，现在提交。",
            "候选和衔接信息已准备完毕：{count}项候选、{routes}段路线覆盖{evidence}，可以提交最终方案。",
        )
    index = int(hashlib.sha256(f"{seed}:{task_id}:submit-text".encode()).hexdigest(), 16)
    return templates[index % len(templates)].format(
        evidence=evidence_text,
        count=candidate_count,
        days=days,
        routes=route_count,
        landmarks="、".join(evidence_landmarks),
    )


def _read_jsonl_index(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row["uid"])] = row
    return rows


def _programmatic_source_identity(records: list[dict[str, Any]]) -> str:
    payload = [
        {
            "task_id": record["task_spec"]["task_id"],
            "blueprint_id": record["blueprint"]["blueprint_id"],
            "surface_id": record["surface"]["surface_id"],
            "scenario_id": record["scenario"]["scenario_id"],
        }
        for record in records
    ]
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _numbered_json_paths(directory: Path) -> list[Path]:
    paths: list[tuple[int, Path]] = []
    for path in directory.glob("*.json"):
        try:
            index = int(path.stem)
        except ValueError:
            continue
        paths.append((index, path))
    return [path for _, path in sorted(paths)]


def _numbered_json_indices(directory: Path) -> set[int]:
    return {int(path.stem) for path in _numbered_json_paths(directory)}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SFTRebuildError(f"Invalid programmatic checkpoint JSON: {path}") from error
    if not isinstance(value, dict):
        raise SFTRebuildError(f"Programmatic checkpoint must be an object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)
