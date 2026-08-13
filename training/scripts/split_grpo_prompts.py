"""Create a deterministic stratified train/validation split for GRPO prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

SPLIT_VERSION = "travelweaver-grpo-prompt-split-v1"
PROMPT_SPLIT_VERSION = "travelweaver-grpo-prompts-split-v1"


@dataclass(frozen=True)
class SplitRecord:
    task_id: str
    task_type: str
    scenario_profile: str
    constraint_count: int
    trip_days: int

    @property
    def inner_stratum(self) -> tuple[str, int, int]:
        return (self.scenario_profile, self.constraint_count, self.trip_days)


def _stable_rank(seed: int, *parts: str) -> str:
    payload = "\x1f".join((str(seed), *parts)).encode()
    return hashlib.sha256(payload).hexdigest()


def _allocate_quotas(
    sizes: dict[Any, int], *, count: int, seed: int, scope: str
) -> dict[Any, int]:
    total = sum(sizes.values())
    raw = {key: size * count / total for key, size in sizes.items()}
    quotas = {key: int(value) for key, value in raw.items()}
    remaining = count - sum(quotas.values())
    ranked = sorted(
        sizes,
        key=lambda key: (
            -(raw[key] - quotas[key]),
            _stable_rank(seed, scope, repr(key)),
        ),
    )
    for key in ranked[:remaining]:
        quotas[key] += 1
    return quotas


def split_records(
    records: list[SplitRecord], *, validation_count: int, seed: int
) -> set[str]:
    if not 0 < validation_count < len(records):
        raise ValueError("validation_count must be between zero and the row count.")
    if len({record.task_id for record in records}) != len(records):
        raise ValueError("GRPO prompt rows must have unique task_id values.")

    by_type: dict[str, list[SplitRecord]] = defaultdict(list)
    for record in records:
        by_type[record.task_type].append(record)
    type_quotas = _allocate_quotas(
        {key: len(group) for key, group in by_type.items()},
        count=validation_count,
        seed=seed,
        scope="task-type",
    )

    selected: set[str] = set()
    for task_type, type_group in sorted(by_type.items()):
        by_scenario: dict[str, list[SplitRecord]] = defaultdict(list)
        for record in type_group:
            by_scenario[record.scenario_profile].append(record)
        scenario_quotas = _allocate_quotas(
            {key: len(group) for key, group in by_scenario.items()},
            count=type_quotas[task_type],
            seed=seed,
            scope=f"scenario-{task_type}",
        )
        for scenario_profile, scenario_group in sorted(by_scenario.items()):
            strata: dict[tuple[int, int], list[SplitRecord]] = defaultdict(list)
            for record in scenario_group:
                strata[(record.constraint_count, record.trip_days)].append(record)
            stratum_quotas = _allocate_quotas(
                {key: len(group) for key, group in strata.items()},
                count=scenario_quotas[scenario_profile],
                seed=seed,
                scope=f"inner-{task_type}-{scenario_profile}",
            )
            for stratum, group in strata.items():
                ranked = sorted(
                    group,
                    key=lambda record: _stable_rank(seed, "task", record.task_id),
                )
                selected.update(
                    record.task_id for record in ranked[: stratum_quotas[stratum]]
                )
    if len(selected) != validation_count:
        raise AssertionError("Stratified split did not satisfy the exact validation count.")
    return selected


def create_split(
    *, input_parquet: Path, output_dir: Path, validation_count: int, seed: int
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite split directory: {output_dir}")
    source_manifest_path = input_parquet.with_name(f"{input_parquet.name}.manifest.json")
    source_manifest = _read_json(source_manifest_path)
    frame = pd.read_parquet(input_parquet)
    required = {"task_id", "task_dir", "prompt", "agent_name", "data_source"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Input GRPO Parquet is missing columns: {sorted(missing)}")
    if frame.empty or frame["task_id"].duplicated().any():
        raise ValueError("Input GRPO Parquet must be non-empty with unique task IDs.")

    metadata = _task_metadata(frame)
    records = [metadata[str(task_id)] for task_id in frame["task_id"]]
    validation_ids = split_records(
        records, validation_count=validation_count, seed=seed
    )
    train_frame = frame[~frame["task_id"].isin(validation_ids)].copy()
    validation_frame = frame[frame["task_id"].isin(validation_ids)].copy()
    if set(train_frame["task_id"]) & set(validation_frame["task_id"]):
        raise AssertionError("GRPO train and validation task IDs overlap.")

    output_dir.mkdir(parents=True, exist_ok=False)
    paths = {
        "train": output_dir / "train.parquet",
        "validation": output_dir / "validation.parquet",
    }
    _atomic_parquet(train_frame, paths["train"])
    _atomic_parquet(validation_frame, paths["validation"])

    assignments = [
        {
            **asdict(record),
            "split": "validation" if record.task_id in validation_ids else "train",
        }
        for record in sorted(records, key=lambda item: item.task_id)
    ]
    _atomic_jsonl(output_dir / "assignments.jsonl", assignments)
    descriptors = {}
    for split, path in paths.items():
        split_frame = train_frame if split == "train" else validation_frame
        descriptor = {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "rows": len(split_frame),
        }
        descriptors[split] = descriptor
        _atomic_json(
            path.with_name(f"{path.name}.manifest.json"),
            {
                "format_version": PROMPT_SPLIT_VERSION,
                "split": split,
                "row_count": len(split_frame),
                "source_parquet": str(input_parquet.resolve()),
                "source_parquet_sha256": _sha256(input_parquet),
                "output": str(path.resolve()),
                "output_sha256": descriptor["sha256"],
                "contains_witness": False,
                "contains_reward_labels": False,
            },
        )

    report = {
        "format_version": SPLIT_VERSION,
        "seed": seed,
        "total_rows": len(frame),
        "train_rows": len(train_frame),
        "validation_rows": len(validation_frame),
        "cross_split_task_overlap": 0,
        "stratification": [
            "task_type",
            "scenario_profile",
            "constraint_count × trip_days",
        ],
        "source": {
            "parquet": str(input_parquet.resolve()),
            "parquet_sha256": _sha256(input_parquet),
            "manifest": str(source_manifest_path.resolve()),
            "manifest_sha256": _sha256(source_manifest_path),
            "format_version": source_manifest.get("format_version"),
        },
        "splits": descriptors,
        "train_distribution": _distribution(assignments, "train"),
        "validation_distribution": _distribution(assignments, "validation"),
    }
    _atomic_json(output_dir / "split-manifest.json", report)
    return report


def _task_metadata(frame: pd.DataFrame) -> dict[str, SplitRecord]:
    by_dir: dict[str, set[str]] = defaultdict(set)
    for row in frame[["task_id", "task_dir"]].itertuples(index=False):
        by_dir[str(row.task_dir)].add(str(row.task_id))
    result: dict[str, SplitRecord] = {}
    for raw_dir, expected_ids in by_dir.items():
        task_dir = Path(raw_dir)
        public = {row["uid"]: row for row in _read_jsonl(task_dir / "tasks.public.jsonl")}
        oracle = {row["uid"]: row for row in _read_jsonl(task_dir / "tasks.oracle.jsonl")}
        for task_id in expected_ids:
            task = public.get(task_id)
            hidden = oracle.get(task_id)
            if not isinstance(task, dict) or not isinstance(hidden, dict):
                raise ValueError(f"Cannot resolve GRPO source metadata for {task_id}.")
            spec = hidden.get("task_spec")
            scenario = hidden.get("scenario")
            if not isinstance(spec, dict) or not isinstance(scenario, dict):
                raise ValueError(f"GRPO task {task_id} has incomplete hidden metadata.")
            trip = spec.get("trip")
            constraints = spec.get("constraints")
            if not isinstance(trip, dict) or not isinstance(constraints, list):
                raise ValueError(f"GRPO task {task_id} has malformed TaskSpec metadata.")
            result[task_id] = SplitRecord(
                task_id=task_id,
                task_type=str(task["task_type"]),
                scenario_profile=str(scenario["profile"]),
                constraint_count=len(constraints),
                trip_days=int(trip["days"]),
            )
    return result


def _distribution(assignments: list[dict[str, Any]], split: str) -> dict[str, Any]:
    rows = [row for row in assignments if row["split"] == split]
    fields = ("task_type", "scenario_profile", "constraint_count", "trip_days")
    return {
        field: dict(sorted(Counter(str(row[field]) for row in rows).items()))
        for field in fields
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _atomic_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
    )


def _atomic_text(path: Path, content: str) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    report = create_split(
        input_parquet=args.input_parquet,
        output_dir=args.output_dir,
        validation_count=args.validation_count,
        seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
