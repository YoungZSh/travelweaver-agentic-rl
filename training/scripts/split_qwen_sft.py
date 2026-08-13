"""Create a deterministic, stratified train/validation split of audited Qwen SFT data."""

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

SPLIT_VERSION = "travelweaver-qwen-sft-split-v2"
HOLDOUT_NAMES = ("validation", "test")
STRATIFICATION_FIELDS = (
    "task_type",
    "scenario_profile",
    "sample_family",
    "trip_days",
)


@dataclass(frozen=True)
class SplitRecord:
    """Metadata used to assign one already-tokenized SFT sample deterministically."""

    sample_id: str
    task_id: str
    semantic_hash: str
    task_type: str
    scenario_profile: str
    sample_family: str
    trip_days: int

    @property
    def stratum(self) -> tuple[str, str, str, int]:
        return (
            self.task_type,
            self.scenario_profile,
            self.sample_family,
            self.trip_days,
        )


def split_records(
    records: Iterable[SplitRecord], *, validation_count: int, seed: int
) -> tuple[set[str], dict[tuple[str, str, str, int], int]]:
    """Return validation sample IDs with proportional integer quotas in every stratum."""

    materialized = list(records)
    if not materialized:
        raise ValueError("Cannot split an empty SFT dataset.")
    if not 0 < validation_count < len(materialized):
        raise ValueError("validation_count must be between zero and the total sample count.")
    sample_ids = [record.sample_id for record in materialized]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Cannot split SFT data with duplicate sample IDs.")

    groups: dict[tuple[str, str, str, int], list[SplitRecord]] = defaultdict(list)
    for record in materialized:
        groups[record.stratum].append(record)

    target_ratio = validation_count / len(materialized)
    raw_quotas = {key: len(group) * target_ratio for key, group in groups.items()}
    quotas = {key: int(value) for key, value in raw_quotas.items()}
    remaining = validation_count - sum(quotas.values())
    def tie_break(key: tuple[str, str, str, int]) -> str:
        return _stable_rank(seed, "quota", *map(str, key))

    for key in sorted(
        groups,
        key=lambda item: (-(raw_quotas[item] - quotas[item]), tie_break(item)),
    )[:remaining]:
        quotas[key] += 1

    selected: set[str] = set()
    for key, group in groups.items():
        ranked = sorted(group, key=lambda record: _stable_rank(seed, "sample", record.sample_id))
        selected.update(record.sample_id for record in ranked[: quotas[key]])
    if len(selected) != validation_count:
        raise AssertionError("Stratified split did not satisfy its exact validation quota.")
    return selected, quotas


def create_split(
    *,
    input_parquet: Path,
    input_audit: Path,
    output_dir: Path,
    validation_count: int,
    seed: int,
    holdout_name: str = "validation",
) -> dict[str, Any]:
    """Write a non-overlapping train/holdout pair and an auditable split manifest."""

    if holdout_name not in HOLDOUT_NAMES:
        raise ValueError(f"holdout_name must be one of {HOLDOUT_NAMES}.")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing split directory: {output_dir}")
    frame = pd.read_parquet(input_parquet)
    required_columns = {"sample_id", "task_id", "messages_json", "tools_json", "enable_thinking"}
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"Input Parquet is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Input Parquet is empty.")
    if frame["sample_id"].duplicated().any() or frame["task_id"].duplicated().any():
        raise ValueError("Input Parquet must have unique sample_id and task_id values.")
    audit_mapping = _audit_sample_task_ids(input_audit)
    parquet_mapping = dict(
        zip(frame["sample_id"].astype(str), frame["task_id"].astype(str), strict=True)
    )
    if audit_mapping != parquet_mapping:
        raise ValueError("Input audit does not exactly match the Parquet sample/task mapping.")

    source_manifest_path = input_parquet.parent / "manifest.json"
    source_manifest = _read_json(source_manifest_path)
    metadata = _source_task_metadata(source_manifest)
    records = _split_records(frame, metadata)
    validation_ids, quotas = split_records(records, validation_count=validation_count, seed=seed)

    semantic_hashes = [record.semantic_hash for record in records]
    if len(semantic_hashes) != len(set(semantic_hashes)):
        raise ValueError("Input SFT data reuses a Blueprint semantic hash; refusing split leakage.")
    train_frame = frame[~frame["sample_id"].isin(validation_ids)].copy()
    validation_frame = frame[frame["sample_id"].isin(validation_ids)].copy()
    if len(train_frame) + len(validation_frame) != len(frame):
        raise AssertionError("Split rows do not cover the input Parquet.")
    if set(train_frame["sample_id"]) & set(validation_frame["sample_id"]):
        raise AssertionError("Train and validation SFT samples overlap.")

    output_dir.mkdir(parents=True, exist_ok=False)
    train_path = output_dir / "train.parquet"
    validation_path = output_dir / f"{holdout_name}.parquet"
    _atomic_parquet(train_frame, train_path)
    _atomic_parquet(validation_frame, validation_path)

    assignments = []
    validation_set = set(validation_ids)
    for record in sorted(records, key=lambda item: item.sample_id):
        assignment = holdout_name if record.sample_id in validation_set else "train"
        assignments.append({**asdict(record), "split": assignment})
    _atomic_jsonl(output_dir / "assignments.jsonl", assignments)

    report = {
        "samples": len(records),
        "train_samples": len(train_frame),
        f"{holdout_name}_samples": len(validation_frame),
        "stratification_fields": list(STRATIFICATION_FIELDS),
        "train_distribution": _distribution(assignments, "train"),
        f"{holdout_name}_distribution": _distribution(assignments, holdout_name),
        "stratum_validation_quotas": {
            "|".join(map(str, key)): value for key, value in sorted(quotas.items())
        },
        "semantic_hashes_unique": True,
        "cross_split_semantic_hash_overlap": 0,
    }
    source_adapter = dict(source_manifest.get("qwen_adapter", {}))
    source_adapter.pop("parquet_path", None)
    source_adapter.pop("parquet_sha256", None)
    source_adapter["parquet_splits"] = {
        "train": _parquet_descriptor(train_path, len(train_frame)),
        holdout_name: _parquet_descriptor(validation_path, len(validation_frame)),
    }
    manifest = {
        "format_version": SPLIT_VERSION,
        "status": "qwen_parquet_split_complete",
        "source": {
            "input_parquet": str(input_parquet.resolve()),
            "input_parquet_sha256": _sha256(input_parquet),
            "input_audit": str(input_audit.resolve()),
            "input_audit_sha256": _sha256(input_audit),
            "input_manifest": str(source_manifest_path.resolve()),
            "input_manifest_sha256": _sha256(source_manifest_path),
        },
        "split": {
            "seed": seed,
            "holdout_count": validation_count,
            "holdout_ratio": round(validation_count / len(frame), 8),
            "holdout_name": holdout_name,
            "stratification_fields": list(STRATIFICATION_FIELDS),
        },
        "qwen_adapter": source_adapter,
        "report": report,
    }
    _atomic_json(output_dir / "manifest.json", manifest)
    return manifest


def _source_task_metadata(source_manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    config = source_manifest.get("config")
    sources = config.get("sources") if isinstance(config, dict) else None
    if not isinstance(sources, list) or not sources:
        raise ValueError("SFT manifest has no source task/trajectory metadata.")
    metadata: dict[str, dict[str, Any]] = {}
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("SFT manifest source is malformed.")
        task_dir = Path(str(source["task_dir"]))
        rollout_path = Path(str(source["rollout_path"]))
        blueprints = {
            str(row["blueprint_id"]): row for row in _read_jsonl(task_dir / "blueprints.jsonl")
        }
        scenarios = {
            int(row["slot_index"]): str(row["profile"])
            for row in _read_jsonl(task_dir / "scenarios.jsonl")
        }
        families = {
            str(row["task_id"]): str(row["sample_family"])
            for row in _read_jsonl(rollout_path)
        }
        for task in _read_jsonl(task_dir / "tasks.public.jsonl"):
            task_id = str(task["uid"])
            blueprint = blueprints.get(str(task["blueprint_id"]))
            if blueprint is None or task_id not in families:
                raise ValueError(f"Cannot find complete source metadata for task {task_id}.")
            slot_index = int(blueprint["slot_index"])
            scenario_profile = scenarios.get(slot_index)
            if scenario_profile is None:
                raise ValueError(f"Task {task_id} references a missing Scenario slot {slot_index}.")
            if task_id in metadata:
                raise ValueError(f"Source task_id appears in multiple SFT sources: {task_id}.")
            trip = blueprint.get("trip")
            if not isinstance(trip, dict) or not isinstance(trip.get("days"), int):
                raise ValueError(f"Task {task_id} Blueprint has no integer trip days.")
            metadata[task_id] = {
                "semantic_hash": str(blueprint["semantic_hash"]),
                "task_type": str(task["task_type"]),
                "scenario_profile": scenario_profile,
                "sample_family": families[task_id],
                "trip_days": int(trip["days"]),
            }
    return metadata


def _split_records(frame: pd.DataFrame, metadata: dict[str, dict[str, Any]]) -> list[SplitRecord]:
    records: list[SplitRecord] = []
    for row in frame[["sample_id", "task_id"]].itertuples(index=False):
        task_id = str(row.task_id)
        item = metadata.get(task_id)
        if item is None:
            raise ValueError(f"SFT Parquet task_id has no source metadata: {task_id}.")
        records.append(
            SplitRecord(
                sample_id=str(row.sample_id),
                task_id=task_id,
                semantic_hash=str(item["semantic_hash"]),
                task_type=str(item["task_type"]),
                scenario_profile=str(item["scenario_profile"]),
                sample_family=str(item["sample_family"]),
                trip_days=int(item["trip_days"]),
            )
        )
    if set(metadata) != {record.task_id for record in records}:
        missing = sorted(set(metadata) - {record.task_id for record in records})
        raise ValueError(f"SFT Parquet is missing source tasks, e.g. {missing[:3]}.")
    return records


def _distribution(assignments: list[dict[str, Any]], split: str) -> dict[str, dict[str, int]]:
    selected = [item for item in assignments if item["split"] == split]
    return {
        field: dict(sorted(Counter(str(item[field]) for item in selected).items()))
        for field in STRATIFICATION_FIELDS
    }


def _parquet_descriptor(path: Path, samples: int) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": _sha256(path), "samples": samples}


def _stable_rank(seed: int, *parts: str) -> str:
    payload = "\x1f".join((str(seed), *parts)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
            rows.append(value)
    return rows


def _audit_sample_task_ids(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for line_number, row in enumerate(_read_jsonl(path), 1):
        sample_id = row.get("sample_id")
        task_id = row.get("task_id")
        if not isinstance(sample_id, str) or not isinstance(task_id, str):
            raise ValueError(f"SFT audit row has no sample_id/task_id: {path}:{line_number}")
        if sample_id in mapping:
            raise ValueError(f"SFT audit has duplicate sample_id: {sample_id}")
        mapping[sample_id] = task_id
    if not mapping:
        raise ValueError(f"SFT audit is empty: {path}")
    return mapping


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".parquet", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    content = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    _atomic_text(path, content)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _atomic_text(path: Path, content: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-parquet", type=Path, required=True)
    parser.add_argument("--input-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--holdout-name",
        choices=HOLDOUT_NAMES,
        default="validation",
        help="Name the held-out split and Parquet file (default: validation).",
    )
    args = parser.parse_args()
    manifest = create_split(
        input_parquet=args.input_parquet,
        input_audit=args.input_audit,
        output_dir=args.output_dir,
        validation_count=args.validation_count,
        seed=args.seed,
        holdout_name=args.holdout_name,
    )
    print(json.dumps(manifest["report"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
