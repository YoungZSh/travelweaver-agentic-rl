"""Resumable, inspectable synthesis artifact storage."""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..errors import SynthesisError
from .models import ARTIFACT_VERSION, PROMPT_VERSION, PilotSlot


class ArtifactStore:
    def __init__(self, output_dir: str | Path, config: dict[str, Any]) -> None:
        self.output_dir = Path(output_dir)
        self.records_dir = self.output_dir / "records"
        self.manifest_path = self.output_dir / "manifest.json"
        self.quarantine_path = self.output_dir / "quarantine.jsonl"
        self.config = dict(config)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.records_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = self._load_or_create_manifest()

    @property
    def api_calls(self) -> int:
        return int(self.manifest.get("api_calls", 0))

    @property
    def quarantine_count(self) -> int:
        return int(self.manifest.get("quarantine_count", 0))

    def completed_indices(self) -> set[int]:
        completed: set[int] = set()
        for path in self.records_dir.glob("*.json"):
            try:
                completed.add(int(path.stem))
            except ValueError:
                continue
        return completed

    def records(self) -> list[dict[str, Any]]:
        return [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(self.records_dir.glob("*.json"))
        ]

    def save_record(self, index: int, record: dict[str, Any], *, api_calls: int) -> None:
        path = self.records_dir / f"{index:03d}.json"
        if path.exists():
            raise SynthesisError(f"Synthesis slot {index} already has a completed record.")
        _atomic_json(path, record)
        self.manifest["completed"] = len(self.completed_indices())
        self.manifest["api_calls"] = api_calls
        self.manifest["updated_at"] = _now()
        _atomic_json(self.manifest_path, self.manifest)

    def quarantine(self, row: dict[str, Any], *, api_calls: int) -> None:
        with self.quarantine_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        self.manifest["quarantine_count"] = self.quarantine_count + 1
        self.manifest["api_calls"] = api_calls
        self.manifest["updated_at"] = _now()
        _atomic_json(self.manifest_path, self.manifest)

    def finalize(self, slots: tuple[PilotSlot, ...], *, api_calls: int) -> dict[str, Any]:
        records = self.records()
        if len(records) != len(slots):
            raise SynthesisError(
                f"Cannot finalize {len(records)} records for {len(slots)} requested slots."
            )
        _atomic_jsonl(
            self.output_dir / "blueprints.jsonl",
            (
                {"slot_index": row["slot"]["index"], **row["blueprint"]}
                for row in records
            ),
        )
        _atomic_jsonl(
            self.output_dir / "surfaces.jsonl",
            ({"slot_index": row["slot"]["index"], **row["surface"]} for row in records),
        )
        _atomic_jsonl(
            self.output_dir / "tasks.public.jsonl",
            (_public_row(row) for row in records),
        )
        _atomic_jsonl(
            self.output_dir / "tasks.oracle.jsonl",
            (
                {
                    "uid": row["task_spec"]["task_id"],
                    "blueprint_id": row["blueprint"]["blueprint_id"],
                    "surface_id": row["surface"]["surface_id"],
                    "task_spec": row["task_spec"],
                }
                for row in records
            ),
        )
        _atomic_jsonl(
            self.output_dir / "witnesses.jsonl",
            (
                {
                    "uid": row["task_spec"]["task_id"],
                    "blueprint_id": row["blueprint"]["blueprint_id"],
                    **row["witness"],
                }
                for row in records
            ),
        )
        preview = _preview(records)
        _atomic_text(self.output_dir / "preview.md", preview)
        distributions = _distributions(slots, records)
        self.manifest.update(
            {
                "status": "complete",
                "completed": len(records),
                "api_calls": api_calls,
                "distributions": distributions,
                "usage": _usage(records),
                "updated_at": _now(),
            }
        )
        _atomic_json(self.manifest_path, self.manifest)
        return distributions

    def _load_or_create_manifest(self) -> dict[str, Any]:
        if self.manifest_path.exists():
            try:
                manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise SynthesisError("Existing synthesis manifest is invalid JSON.") from error
            if manifest.get("config") != self.config:
                raise SynthesisError(
                    "Output directory belongs to a different synthesis configuration; "
                    "choose a new directory or restore the matching arguments."
                )
            return manifest
        manifest = {
            "artifact_version": ARTIFACT_VERSION,
            "prompt_version": PROMPT_VERSION,
            "status": "in_progress",
            "config": self.config,
            "completed": 0,
            "api_calls": 0,
            "quarantine_count": 0,
            "created_at": _now(),
            "updated_at": _now(),
        }
        _atomic_json(self.manifest_path, manifest)
        return manifest


def record_bundle(
    *,
    slot: PilotSlot,
    blueprint: dict[str, Any],
    surface: dict[str, Any],
    task_spec: dict[str, Any],
    witness: dict[str, Any],
    candidate_attempt: int,
) -> dict[str, Any]:
    return {
        "slot": asdict(slot),
        "blueprint": blueprint,
        "surface": surface,
        "task_spec": task_spec,
        "witness": witness,
        "candidate_attempt": candidate_attempt,
    }


def _public_row(record: dict[str, Any]) -> dict[str, Any]:
    spec = record["task_spec"]
    trip = spec["trip"]
    return {
        "uid": spec["task_id"],
        "tag": "synthetic_pilot_v1",
        "start_city": trip["origin"],
        "target_city": trip["destinations"][-1],
        "days": trip["days"],
        "people_number": trip["travelers"],
        "limit_rooms": False,
        "limits_room_type": False,
        "language": record["surface"]["language"],
        "query": record["surface"]["public_query"],
        "blueprint_id": record["blueprint"]["blueprint_id"],
        "surface_id": record["surface"]["surface_id"],
    }


def _preview(records: list[dict[str, Any]]) -> str:
    lines = ["# TravelWeaver 50-task synthesis preview", ""]
    for row in records:
        slot = row["slot"]
        lines.extend(
            [
                f"## {slot['index'] + 1:02d}. {slot['destination']} / {slot['days']}天",
                "",
                row["surface"]["public_query"],
                "",
                "- 约束：" + "、".join(slot["recipe"]),
                f"- 交通：{slot['outbound_mode']} → / {slot['return_mode']} ←",
                f"- Blueprint：`{row['blueprint']['blueprint_id']}`",
                "",
            ]
        )
    return "\n".join(lines)


def _distributions(
    slots: tuple[PilotSlot, ...], records: list[dict[str, Any]]
) -> dict[str, Any]:
    recipe_families = Counter(key for slot in slots for key in slot.recipe)
    return {
        "destinations": dict(sorted(Counter(slot.destination for slot in slots).items())),
        "days": dict(sorted(Counter(str(slot.days) for slot in slots).items())),
        "constraint_counts": dict(
            sorted(Counter(str(slot.constraint_count) for slot in slots).items())
        ),
        "transport_patterns": dict(
            sorted(
                Counter(
                    f"{slot.outbound_mode}/{slot.return_mode}" for slot in slots
                ).items()
            )
        ),
        "constraint_recipes": dict(sorted(recipe_families.items())),
        "unique_blueprints": len(
            {row["blueprint"]["blueprint_id"] for row in records}
        ),
        "unique_surfaces": len({row["surface"]["surface_id"] for row in records}),
    }


def _usage(records: list[dict[str, Any]]) -> dict[str, int]:
    keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    return {
        key: sum(int(row["surface"].get("usage", {}).get(key, 0)) for row in records)
        for key in keys
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _atomic_jsonl(path: Path, rows: Any) -> None:
    text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    _atomic_text(path, text)


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
