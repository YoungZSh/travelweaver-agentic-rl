"""Resumable concurrent rollouts over a generated task directory."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..data.tasks import JsonlTaskStore
from ..env import ChinaTravelBackend, ScenarioBackend, ScenarioSpec, TravelWeaverEnv
from ..errors import DataUnavailableError
from ..llm import (
    DEFAULT_DEEPSEEK_CONCURRENCY,
    DeepSeekConfig,
    OpenAICompatibleChatClient,
    OpenAICompatibleConfig,
)
from .api_agent import DEFAULT_MAX_API_TURNS, ApiAgentRun, ToolCallingAgent
from .tool_response import DEFAULT_TOOL_RESPONSE_MODE, validate_tool_response_mode
from .trajectory import append_trajectory

DEFAULT_ROLLOUT_CONCURRENCY = DEFAULT_DEEPSEEK_CONCURRENCY


@dataclass(frozen=True)
class BenchmarkRolloutBatchConfig:
    output_path: Path
    error_path: Path
    split: str = "benchmark"
    concurrency: int = DEFAULT_ROLLOUT_CONCURRENCY
    max_api_turns: int = DEFAULT_MAX_API_TURNS
    seed: int = 20260808
    task_id: str | None = None
    tool_response_mode: str = DEFAULT_TOOL_RESPONSE_MODE

    def __post_init__(self) -> None:
        if self.concurrency <= 0 or self.max_api_turns <= 0:
            raise ValueError("Rollout concurrency and API-turn limit must be positive.")
        validate_tool_response_mode(self.tool_response_mode)


@dataclass(frozen=True)
class BenchmarkRolloutBatchReport:
    split: str
    total_tasks: int
    skipped: int
    attempted: int
    accepted: int
    errors: int
    output_path: str
    error_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GeneratedRolloutBatchConfig:
    input_dir: Path
    output_path: Path
    error_path: Path
    concurrency: int = DEFAULT_ROLLOUT_CONCURRENCY
    max_api_turns: int = DEFAULT_MAX_API_TURNS
    seed: int = 20260808
    task_id: str | None = None
    limit: int | None = None
    tool_response_mode: str = DEFAULT_TOOL_RESPONSE_MODE

    def __post_init__(self) -> None:
        if self.concurrency <= 0 or self.max_api_turns <= 0:
            raise ValueError("Rollout concurrency and API-turn limit must be positive.")
        if self.limit is not None and self.limit <= 0:
            raise ValueError("Rollout task limit must be positive.")
        if self.task_id is not None and self.limit is not None:
            raise ValueError("Pass either task_id or limit, not both.")
        validate_tool_response_mode(self.tool_response_mode)


@dataclass(frozen=True)
class GeneratedRolloutBatchReport:
    total_tasks: int
    skipped: int
    attempted: int
    accepted: int
    errors: int
    output_path: str
    error_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ProgressCallback = Callable[[dict[str, Any]], None]
EpisodeRunner = Callable[[TravelWeaverEnv, str], ApiAgentRun]


def run_benchmark_rollout_batch(
    config: BenchmarkRolloutBatchConfig,
    llm_config: OpenAICompatibleConfig,
    *,
    base_backend: Any | None = None,
    chat_client: Any | None = None,
    episode_runner: EpisodeRunner | None = None,
    progress: ProgressCallback | None = None,
) -> BenchmarkRolloutBatchReport:
    """Run a resumable model rollout over one pinned official benchmark split."""

    store = JsonlTaskStore.default(split=config.split)
    attempted_ids = _record_ids(config.output_path) | _record_ids(config.error_path)
    if config.task_id is not None:
        store.get_public(config.task_id)
        pending = [] if config.task_id in attempted_ids else [config.task_id]
    else:
        pending = [task_id for task_id in store.task_ids if task_id not in attempted_ids]

    _emit(
        progress,
        {
            "event": "batch_start",
            "model": llm_config.model,
            "split": config.split,
            "total": len(store.task_ids),
            "already_attempted": len(attempted_ids),
            "pending": len(pending),
            "concurrency": config.concurrency,
        },
    )

    backend = base_backend if base_backend is not None else ChinaTravelBackend()
    if episode_runner is None:
        client = chat_client or OpenAICompatibleChatClient(llm_config)

        def episode_runner(env: TravelWeaverEnv, task_id: str) -> ApiAgentRun:
            return ToolCallingAgent(
                env,
                llm_config,
                chat_client=client,
                max_api_turns=config.max_api_turns,
                tool_response_mode=config.tool_response_mode,
            ).run(task_id=task_id, seed=config.seed)

    def run_one(task_id: str) -> dict[str, Any]:
        env = TravelWeaverEnv(backend, store)
        try:
            run = episode_runner(env, task_id)
            payload = run.to_dict(include_trajectory=True)
            public_task = store.get_public(task_id)
            payload["batch_metadata"] = {
                "split": config.split,
                "tag": public_task.get("tag"),
                "max_tokens": llm_config.max_tokens,
                "max_api_turns": config.max_api_turns,
                "seed": config.seed,
                "task_source": "LAMDA-NeSy/ChinaTravel",
                "rollout_index": 0,
                "tool_response_mode": config.tool_response_mode,
            }
            return payload
        finally:
            env.close()

    accepted = 0
    errors = 0
    finished = 0
    with ThreadPoolExecutor(max_workers=config.concurrency) as executor:
        futures = {executor.submit(run_one, task_id): task_id for task_id in pending}
        for future in as_completed(futures):
            task_id = futures[future]
            finished += 1
            try:
                payload = future.result()
            except Exception as error:  # noqa: BLE001 - persist every benchmark attempt.
                errors += 1
                append_trajectory(
                    config.error_path,
                    {
                        "task_id": task_id,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                )
                status = "error"
                reward = None
                steps = None
            else:
                append_trajectory(config.output_path, payload)
                accepted += int(bool(payload["success"]))
                status = "success" if payload["success"] else "failed"
                reward = payload["final_reward"]
                steps = payload["step_count"]
            _emit(
                progress,
                {
                    "event": "task_complete",
                    "task_id": task_id,
                    "status": status,
                    "reward": reward,
                    "steps": steps,
                    "batch_finished": finished,
                    "batch_pending": len(pending) - finished,
                },
            )

    report = BenchmarkRolloutBatchReport(
        split=config.split,
        total_tasks=len(store.task_ids),
        skipped=len(store.task_ids) - len(pending),
        attempted=len(pending),
        accepted=accepted,
        errors=errors,
        output_path=str(config.output_path.resolve()),
        error_path=str(config.error_path.resolve()),
    )
    _emit(progress, {"event": "batch_complete", **report.to_dict()})
    return report


def run_generated_rollout_batch(
    config: GeneratedRolloutBatchConfig,
    llm_config: DeepSeekConfig,
    *,
    base_backend: Any | None = None,
    chat_client: Any | None = None,
    episode_runner: EpisodeRunner | None = None,
    progress: ProgressCallback | None = None,
) -> GeneratedRolloutBatchReport:
    """Run each not-yet-attempted generated task once and persist full trajectories."""

    store = JsonlTaskStore(
        config.input_dir / "tasks.public.jsonl",
        config.input_dir / "tasks.oracle.jsonl",
    )
    attempted_ids = _record_ids(config.output_path) | _record_ids(config.error_path)
    if config.task_id is not None:
        store.get_public(config.task_id)
        selected = [config.task_id]
    else:
        selected = _select_generated_task_ids(store, limit=config.limit, seed=config.seed)
    selected_ids = set(selected)
    attempted_selected = selected_ids & attempted_ids
    pending = [task_id for task_id in selected if task_id not in attempted_ids]

    _emit(
        progress,
        {
            "event": "batch_start",
            "model": llm_config.model,
            "source_total": len(store.task_ids),
            "total": len(selected),
            "selection_limit": config.limit,
            "selection_seed": config.seed,
            "already_attempted": len(attempted_selected),
            "pending": len(pending),
            "concurrency": config.concurrency,
        },
    )

    backend = base_backend if base_backend is not None else ChinaTravelBackend()
    if episode_runner is None:
        client = chat_client or OpenAICompatibleChatClient(llm_config)

        def episode_runner(env: TravelWeaverEnv, task_id: str) -> ApiAgentRun:
            return ToolCallingAgent(
                env,
                llm_config,
                chat_client=client,
                max_api_turns=config.max_api_turns,
                tool_response_mode=config.tool_response_mode,
            ).run(task_id=task_id, seed=config.seed)

    def run_one(task_id: str) -> dict[str, Any]:
        oracle = store.get_oracle(task_id)
        raw_scenario = oracle.get("scenario")
        if not isinstance(raw_scenario, dict):
            raise DataUnavailableError(f"Generated task {task_id} has no materialized scenario.")
        scenario = ScenarioSpec.from_dict(raw_scenario)
        env = TravelWeaverEnv(ScenarioBackend(backend, scenario), store)
        try:
            run = episode_runner(env, task_id)
            payload = run.to_dict(include_trajectory=True)
            payload["batch_metadata"] = {
                "input_dir": str(config.input_dir.resolve()),
                "thinking": llm_config.thinking,
                "max_tokens": llm_config.max_tokens,
                "timeout_seconds": llm_config.timeout_seconds,
                "scenario_id": scenario.scenario_id,
                "scenario_profile": scenario.profile,
                "task_type": store.get_public(task_id).get("task_type"),
                "rollout_index": 0,
                "tool_response_mode": config.tool_response_mode,
                "selection_limit": config.limit,
                "selection_seed": config.seed,
            }
            return payload
        finally:
            env.close()

    accepted = 0
    errors = 0
    finished = 0
    with ThreadPoolExecutor(max_workers=config.concurrency) as executor:
        futures = {executor.submit(run_one, task_id): task_id for task_id in pending}
        for future in as_completed(futures):
            task_id = futures[future]
            finished += 1
            try:
                payload = future.result()
            except Exception as error:  # noqa: BLE001 - persist each paid batch attempt.
                errors += 1
                append_trajectory(
                    config.error_path,
                    {
                        "task_id": task_id,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                )
                status = "error"
                reward = None
                steps = None
            else:
                append_trajectory(config.output_path, payload)
                accepted += int(bool(payload["success"]))
                status = "success" if payload["success"] else "failed"
                reward = payload["final_reward"]
                steps = payload["step_count"]
            _emit(
                progress,
                {
                    "event": "task_complete",
                    "task_id": task_id,
                    "status": status,
                    "reward": reward,
                    "steps": steps,
                    "batch_finished": finished,
                    "batch_pending": len(pending) - finished,
                },
            )

    report = GeneratedRolloutBatchReport(
        total_tasks=len(selected),
        skipped=len(attempted_selected),
        attempted=len(pending),
        accepted=accepted,
        errors=errors,
        output_path=str(config.output_path.resolve()),
        error_path=str(config.error_path.resolve()),
    )
    _emit(progress, {"event": "batch_complete", **report.to_dict()})
    return report


def _select_generated_task_ids(
    store: JsonlTaskStore,
    *,
    limit: int | None,
    seed: int,
) -> list[str]:
    """Select a stable cohort while preserving generated task-type proportions."""

    values = list(store.task_ids)
    if limit is None or limit >= len(values):
        return values

    def selection_key(task_id: str) -> tuple[bytes, str]:
        digest = hashlib.sha256(f"{seed}:{task_id}".encode()).digest()
        return digest, task_id

    buckets: dict[str, dict[str, list[str]]] = {}
    for task_id in values:
        task_type = str(store.get_public(task_id).get("task_type") or "unknown")
        scenario = store.get_oracle(task_id).get("scenario")
        scenario_profile = (
            str(scenario.get("profile") or "unknown")
            if isinstance(scenario, dict)
            else "unknown"
        )
        buckets.setdefault(task_type, {}).setdefault(scenario_profile, []).append(task_id)
    type_counts = {
        task_type: sum(len(task_ids) for task_ids in scenarios.values())
        for task_type, scenarios in buckets.items()
    }
    type_quotas = _proportional_quotas(type_counts, total=limit)
    selected: list[str] = []
    for task_type, scenarios in buckets.items():
        scenario_quotas = _proportional_quotas(
            {profile: len(task_ids) for profile, task_ids in scenarios.items()},
            total=type_quotas[task_type],
        )
        for profile, task_ids in scenarios.items():
            selected.extend(
                sorted(task_ids, key=selection_key)[: scenario_quotas[profile]]
            )
    return sorted(selected, key=selection_key)


def _proportional_quotas(counts: dict[str, int], *, total: int) -> dict[str, int]:
    population = sum(counts.values())
    exact = {key: total * count / population for key, count in counts.items()}
    quotas = {key: int(value) for key, value in exact.items()}
    remaining = total - sum(quotas.values())
    remainder_order = sorted(
        counts,
        key=lambda key: (-(exact[key] - quotas[key]), key),
    )
    for key in remainder_order[:remaining]:
        quotas[key] += 1
    return quotas


def _record_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    result: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                task_id = row["task_id"]
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise DataUnavailableError(
                    f"Invalid rollout row at {path}:{line_number}"
                ) from error
            if not isinstance(task_id, str):
                raise DataUnavailableError(f"Invalid task id at {path}:{line_number}")
            result.add(task_id)
    return result


def _emit(callback: ProgressCallback | None, payload: dict[str, Any]) -> None:
    if callback is not None:
        callback(payload)
