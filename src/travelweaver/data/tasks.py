"""Pinned ChinaTravel task snapshot import and JSONL task store."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import random
import tempfile
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..errors import DataUnavailableError, TaskNotFoundError
from ..paths import project_root

CHINATRAVEL_DATASET = "LAMDA-NeSy/ChinaTravel"
CHINATRAVEL_DATASET_REVISION = "802b18d9844a4a9927bb5750edd155e918c20913"
EASY_CSV_SHA256 = "a74520d152c04c09f47850e1e313c2c285a679ed70ccc200592395085f74bf41"
_SPLITS: dict[str, tuple[int, str]] = {
    "easy": (300, EASY_CSV_SHA256),
    "medium": (150, "9cd49188a8f2a01b900abecc3f10263c257bcb761cf2635e5f0d4fb53d93f77b"),
    "human": (154, "e6cfe692f2ad902ae075d55b005d7f8ff20cfbcd745da4f9c8e79ca69cb65fbe"),
    "preference_base50": (
        50,
        "3c83ea677c3d312e1d5d792b3e379c7d0d0ec90fc91ea1d161877f2f6bae7331",
    ),
}


def _split_url(split: str) -> str:
    return (
        "https://huggingface.co/datasets/LAMDA-NeSy/ChinaTravel/resolve/"
        f"{CHINATRAVEL_DATASET_REVISION}/{split}.csv"
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise DataUnavailableError(
            f"Task snapshot not found: {path}. Run `travelweaver import-tasks --split easy`."
        )
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise DataUnavailableError(f"Invalid JSONL at {path}:{line_number}") from error
            if not isinstance(row, dict) or not isinstance(row.get("uid"), str):
                raise DataUnavailableError(f"Invalid task row at {path}:{line_number}")
            rows.append(row)
    return rows


class JsonlTaskStore:
    """Task store that keeps agent-visible input separate from hidden oracle data."""

    def __init__(self, public_path: str | Path, oracle_path: str | Path) -> None:
        public_rows = _read_jsonl(Path(public_path))
        oracle_rows = _read_jsonl(Path(oracle_path))
        self._public = {row["uid"]: row for row in public_rows}
        self._oracle = {row["uid"]: row for row in oracle_rows}
        if len(self._public) != len(public_rows) or len(self._oracle) != len(oracle_rows):
            raise DataUnavailableError("Task snapshot contains duplicate uid values.")
        if self._public.keys() != self._oracle.keys():
            missing_oracle = sorted(self._public.keys() - self._oracle.keys())
            missing_public = sorted(self._oracle.keys() - self._public.keys())
            raise DataUnavailableError(
                "Public/oracle task ids do not match. "
                f"Missing oracle={missing_oracle[:3]}, missing public={missing_public[:3]}"
            )
        if not self._public:
            raise DataUnavailableError("Task snapshot is empty.")

    @classmethod
    def default(cls, *, split: str = "easy", root: str | Path | None = None) -> JsonlTaskStore:
        base = Path(root) if root is not None else project_root()
        task_dir = base / "data" / "tasks"
        return cls(task_dir / f"{split}.public.jsonl", task_dir / f"{split}.oracle.jsonl")

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._public))

    def get_public(self, task_id: str) -> dict[str, Any]:
        try:
            return dict(self._public[task_id])
        except KeyError as error:
            raise TaskNotFoundError(f"Unknown task id: {task_id}") from error

    def get_oracle(self, task_id: str) -> dict[str, Any]:
        try:
            return dict(self._oracle[task_id])
        except KeyError as error:
            raise TaskNotFoundError(f"Unknown task id: {task_id}") from error

    def choose(self, seed: int | None = None) -> str:
        return random.Random(seed).choice(self.task_ids)


def _parse_hard_logic(value: str) -> list[str]:
    """Parse the list as data; deliberately never execute constraint source."""

    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as error:
        raise DataUnavailableError("hard_logic_py is not a Python string-list literal") from error
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise DataUnavailableError("hard_logic_py must contain a list of strings")
    return parsed


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def import_task_split(
    output_dir: str | Path,
    *,
    split: str,
    source_csv: str | Path | None = None,
) -> dict[str, Any]:
    """Download, verify, separate, and persist one pinned benchmark split."""

    if split not in _SPLITS:
        raise DataUnavailableError(f"Unsupported ChinaTravel task split: {split}")
    expected_count, expected_digest = _SPLITS[split]

    temporary_path: Path | None = None
    if source_csv is None:
        with tempfile.NamedTemporaryFile(
            prefix=f"travelweaver-{split}-", suffix=".csv", delete=False
        ) as handle:
            temporary_path = Path(handle.name)
        try:
            urllib.request.urlretrieve(_split_url(split), temporary_path)
        except Exception as error:
            temporary_path.unlink(missing_ok=True)
            raise DataUnavailableError(
                f"Failed to download pinned {split} split: {error}"
            ) from error
        csv_path = temporary_path
    else:
        csv_path = Path(source_csv)

    try:
        digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        if source_csv is None and digest != expected_digest:
            raise DataUnavailableError(
                f"{split} split checksum mismatch: expected {expected_digest}, got {digest}"
            )
        public_rows: list[dict[str, Any]] = []
        oracle_rows: list[dict[str, Any]] = []
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                uid = row["uid"].strip()
                public_rows.append(
                    {
                        "uid": uid,
                        "tag": row["tag"],
                        "start_city": row["start_city"],
                        "target_city": row["target_city"],
                        "days": int(row["days"]),
                        "people_number": int(row["people_number"]),
                        "limit_rooms": row["limit_rooms"] == "True",
                        "limits_room_type": row["limits_room_type"] == "True",
                        "language": "zh",
                        "query": row["nature_language"],
                    }
                )
                oracle_rows.append(
                    {
                        "uid": uid,
                        "hard_logic": _parse_hard_logic(row["hard_logic_py"]),
                        "dataset": CHINATRAVEL_DATASET,
                        "dataset_revision": CHINATRAVEL_DATASET_REVISION,
                    }
                )
        if source_csv is None and len(public_rows) != expected_count:
            raise DataUnavailableError(
                f"Expected {expected_count} {split} tasks, found {len(public_rows)}"
            )
        output = Path(output_dir)
        public_path = output / f"{split}.public.jsonl"
        oracle_path = output / f"{split}.oracle.jsonl"
        _write_jsonl(public_path, public_rows)
        _write_jsonl(oracle_path, oracle_rows)
        return {
            "split": split,
            "count": len(public_rows),
            "source_sha256": digest,
            "public_path": str(public_path),
            "oracle_path": str(oracle_path),
        }
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
def import_benchmark_tasks(output_dir: str | Path) -> dict[str, Any]:
    """Import and combine all 654 ChinaTravel tasks with executable oracle data."""

    reports = [import_task_split(output_dir, split=split) for split in _SPLITS]
    output = Path(output_dir)
    public_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for split in _SPLITS:
        split_public = _read_jsonl(output / f"{split}.public.jsonl")
        split_oracle = _read_jsonl(output / f"{split}.oracle.jsonl")
        for public, oracle in zip(split_public, split_oracle, strict=True):
            uid = str(public["uid"])
            if oracle["uid"] != uid:
                raise DataUnavailableError(f"{split} public/oracle rows are out of order.")
            if uid in seen_ids:
                namespaced = f"{split}:{uid}"
                public = {**public, "source_uid": uid, "uid": namespaced}
                oracle = {**oracle, "source_uid": uid, "uid": namespaced}
            seen_ids.add(str(public["uid"]))
            public_rows.append(public)
            oracle_rows.append(oracle)
    if len(public_rows) != 654 or len({row["uid"] for row in public_rows}) != 654:
        raise DataUnavailableError("Combined benchmark must contain 654 addressable task rows.")
    public_path = output / "benchmark.public.jsonl"
    oracle_path = output / "benchmark.oracle.jsonl"
    _write_jsonl(public_path, public_rows)
    _write_jsonl(oracle_path, oracle_rows)
    return {
        "split": "benchmark",
        "count": len(public_rows),
        "components": reports,
        "public_path": str(public_path),
        "oracle_path": str(oracle_path),
    }
