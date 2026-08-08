"""Resumable concurrent rollouts over a generated task directory."""

from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..data.tasks import JsonlTaskStore
from ..env import ChinaTravelBackend, ScenarioBackend, ScenarioSpec, TravelWeaverEnv
from ..errors import DataUnavailableError
from ..llm import DeepSeekConfig, OpenAICompatibleChatClient
from .api_agent import ApiAgentRun, ToolCallingAgent
from .trajectory import append_trajectory

DEFAULT_ROLLOUT_CONCURRENCY = 256


@dataclass(frozen=True)
class GeneratedRolloutBatchConfig:
    input_dir: Path
    output_path: Path
    error_path: Path
    concurrency: int = DEFAULT_ROLLOUT_CONCURRENCY
    max_api_turns: int = 40
    seed: int = 20260808
    task_id: str | None = None

    def __post_init__(self) -> None:
        if self.concurrency <= 0 or self.max_api_turns <= 0:
            raise ValueError("Rollout concurrency and API-turn limit must be positive.")


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
        pending = [] if config.task_id in attempted_ids else [config.task_id]
    else:
        pending = [task_id for task_id in store.task_ids if task_id not in attempted_ids]

    _emit(
        progress,
        {
            "event": "batch_start",
            "model": llm_config.model,
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
