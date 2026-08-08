from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from travelweaver.env import InMemoryBackend, ScenarioSpec
from travelweaver.llm import DeepSeekConfig
from travelweaver.rollout import GeneratedRolloutBatchConfig, run_generated_rollout_batch


@dataclass(frozen=True)
class _FakeRun:
    task_id: str

    def to_dict(self, *, include_trajectory: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "task_id": self.task_id,
            "success": True,
            "final_reward": 1.0,
            "step_count": 1,
        }
        if include_trajectory:
            payload["trajectory"] = [{"event": "fake"}]
        return payload


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_generated_batch_is_resumable_and_persists_model_configuration(tmp_path) -> None:
    input_dir = tmp_path / "generated"
    output_path = tmp_path / "trajectories.jsonl"
    error_path = tmp_path / "errors.jsonl"
    task_ids = ["task-1", "task-2"]
    _write_jsonl(
        input_dir / "tasks.public.jsonl",
        [
            {
                "uid": task_id,
                "task_type": "easy_like",
                "query": f"query {task_id}",
            }
            for task_id in task_ids
        ],
    )
    scenario = ScenarioSpec(
        base_world_snapshot_version="test-world",
        profile="normal",
        effects=(),
    ).to_dict()
    _write_jsonl(
        input_dir / "tasks.oracle.jsonl",
        [{"uid": task_id, "scenario": scenario} for task_id in task_ids],
    )
    config = GeneratedRolloutBatchConfig(
        input_dir=input_dir,
        output_path=output_path,
        error_path=error_path,
        concurrency=2,
    )
    llm_config = DeepSeekConfig(
        api_key="not-a-secret",
        thinking="enabled",
        max_tokens=16384,
    )
    called: list[str] = []

    def run_episode(_env, task_id: str) -> _FakeRun:
        called.append(task_id)
        return _FakeRun(task_id)

    report = run_generated_rollout_batch(
        config,
        llm_config,
        base_backend=InMemoryBackend([]),
        episode_runner=run_episode,
    )

    assert report.attempted == 2
    assert report.accepted == 2
    assert report.errors == 0
    assert sorted(called) == task_ids
    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert {row["task_id"] for row in records} == set(task_ids)
    assert all(row["batch_metadata"]["thinking"] == "enabled" for row in records)
    assert all(row["batch_metadata"]["max_tokens"] == 16384 for row in records)

    called.clear()
    resumed = run_generated_rollout_batch(
        config,
        llm_config,
        base_backend=InMemoryBackend([]),
        episode_runner=run_episode,
    )

    assert resumed.skipped == 2
    assert resumed.attempted == 0
    assert called == []
