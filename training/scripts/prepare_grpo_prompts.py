"""Build versioned trainer-neutral prompt Parquet for online TravelWeaver GRPO."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from travelweaver.data import JsonlTaskStore

from travelweaver_grpo_agent_loop import build_prompt

GRPO_PROMPT_VERSION = "travelweaver-grpo-prompts-v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def prepare(input_dirs: Sequence[Path], output: Path) -> dict[str, Any]:
    if not input_dirs:
        raise ValueError("At least one generated task directory is required.")
    rows = []
    seen_task_ids: set[str] = set()
    sources = []
    for source_index, input_dir in enumerate(input_dirs):
        public_path = input_dir / "tasks.public.jsonl"
        oracle_path = input_dir / "tasks.oracle.jsonl"
        source_manifest = input_dir / "manifest.json"
        if not (
            public_path.is_file()
            and oracle_path.is_file()
            and source_manifest.is_file()
        ):
            raise ValueError(f"Generated task directory is incomplete: {input_dir}")
        store = JsonlTaskStore(public_path, oracle_path)
        resolved_dir = str(input_dir.resolve())
        source_row_count = 0
        for task_id in store.task_ids:
            if task_id in seen_task_ids:
                raise ValueError(f"Duplicate task_id across generated batches: {task_id}")
            seen_task_ids.add(task_id)
            task = store.get_public(task_id)
            oracle = store.get_oracle(task_id)
            scenario = oracle.get("scenario")
            if not isinstance(scenario, dict):
                raise ValueError(f"Generated task {task_id} has no frozen Scenario.")
            rows.append(
                {
                    "prompt": build_prompt(task),
                    "data_source": "travelweaver_generated",
                    "agent_name": "travelweaver_agent",
                    "task_id": task_id,
                    "task_dir": resolved_dir,
                    "scenario_id": str(scenario.get("scenario_id")),
                    "extra_info": {
                        "index": len(rows),
                        "source_index": source_index,
                    },
                }
            )
            source_row_count += 1
        sources.append(
            {
                "source_dir": resolved_dir,
                "source_manifest_sha256": _sha256(source_manifest),
                "row_count": source_row_count,
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".parquet", dir=output.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        pd.DataFrame(rows).to_parquet(temporary, index=False)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    report = {
        "format_version": GRPO_PROMPT_VERSION,
        "row_count": len(rows),
        "sources": sources,
        "output": str(output.resolve()),
        "output_sha256": _sha256(output),
        "contains_witness": False,
        "contains_reward_labels": False,
    }
    _atomic_json(output.with_name(f"{output.name}.manifest.json"), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True, action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.input_dir, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
