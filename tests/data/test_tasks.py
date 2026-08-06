from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from travelweaver.data.tasks import JsonlTaskStore, import_easy_tasks
from travelweaver.errors import DataUnavailableError, TaskNotFoundError

FIELDNAMES = [
    "uid",
    "tag",
    "start_city",
    "target_city",
    "days",
    "people_number",
    "limit_rooms",
    "limits_room_type",
    "hard_logic_py",
    "nature_language",
    "nature_language_en",
]


def _write_csv(path: Path, hard_logic: str = "['result=True']") -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerow(
            {
                "uid": "task-1",
                "tag": "easy",
                "start_city": "上海",
                "target_city": "杭州",
                "days": "1",
                "people_number": "2",
                "limit_rooms": "False",
                "limits_room_type": "True",
                "hard_logic_py": hard_logic,
                "nature_language": "去杭州玩一天。",
                "nature_language_en": "Visit Hangzhou for a day.",
            }
        )


def test_import_separates_public_and_oracle(tmp_path: Path) -> None:
    source = tmp_path / "easy.csv"
    output = tmp_path / "tasks"
    _write_csv(source)
    report = import_easy_tasks(output, source_csv=source)
    assert report["count"] == 1

    public = json.loads((output / "easy.public.jsonl").read_text(encoding="utf-8"))
    oracle = json.loads((output / "easy.oracle.jsonl").read_text(encoding="utf-8"))
    assert "hard_logic" not in public
    assert public["query"] == "去杭州玩一天。"
    assert public["limits_room_type"] is True
    assert oracle["hard_logic"] == ["result=True"]

    store = JsonlTaskStore(output / "easy.public.jsonl", output / "easy.oracle.jsonl")
    assert store.choose(seed=0) == "task-1"
    assert store.get_public("task-1")["language"] == "zh"
    assert store.get_oracle("task-1")["hard_logic"] == ["result=True"]
    with pytest.raises(TaskNotFoundError):
        store.get_public("missing")


def test_hard_logic_is_never_executed(tmp_path: Path) -> None:
    marker = tmp_path / "executed"
    source = tmp_path / "easy.csv"
    expression = f"__import__('pathlib').Path({str(marker)!r}).write_text('bad')"
    _write_csv(source, hard_logic=expression)
    with pytest.raises(DataUnavailableError):
        import_easy_tasks(tmp_path / "tasks", source_csv=source)
    assert not marker.exists()


def test_store_rejects_public_oracle_mismatch(tmp_path: Path) -> None:
    public = tmp_path / "public.jsonl"
    oracle = tmp_path / "oracle.jsonl"
    public.write_text('{"uid":"one"}\n', encoding="utf-8")
    oracle.write_text('{"uid":"two"}\n', encoding="utf-8")
    with pytest.raises(DataUnavailableError):
        JsonlTaskStore(public, oracle)
