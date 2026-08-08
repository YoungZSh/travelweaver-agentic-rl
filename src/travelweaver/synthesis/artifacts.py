"""Resumable, inspectable synthesis artifact storage."""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from ..errors import SynthesisError
from ..paths import project_root
from .catalog import (
    BLENDED_PROFILE,
    BLENDED_V1_1_PROFILE,
    blended_scenario_quotas,
    blended_task_type_quotas,
)
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
            self.output_dir / "scenarios.jsonl",
            (
                {"slot_index": row["slot"]["index"], **row["scenario"]}
                for row in records
            ),
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
                    "scenario_id": row["scenario"]["scenario_id"],
                    "scenario": row["scenario"],
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
                    "scenario_id": row["scenario"]["scenario_id"],
                    **row["witness"],
                }
                for row in records
            ),
        )
        _atomic_jsonl(
            self.output_dir / "preference-audit.jsonl",
            (
                {
                    "uid": row["task_spec"]["task_id"],
                    **row["preference_audit"],
                }
                for row in records
                if row.get("preference_audit") is not None
            ),
        )
        _atomic_jsonl(
            self.output_dir / "polish-audit.jsonl",
            (
                {
                    "uid": row["task_spec"]["task_id"],
                    "slot_index": row["slot"]["index"],
                    **event,
                }
                for row in records
                for event in row.get("polish_audit", [])
            ),
        )
        preview = _preview(records)
        _atomic_text(self.output_dir / "preview.md", preview)
        for task_type in sorted({str(row["slot"]["task_type"]) for row in records}):
            _atomic_text(
                self.output_dir / f"preview-{task_type.replace('_', '-')}.md",
                _preview(
                    [row for row in records if row["slot"]["task_type"] == task_type]
                ),
            )
        distributions = _distributions(slots, records)
        _atomic_json(self.output_dir / "diversity.json", distributions)
        alignment = _alignment(records, distributions, self.config)
        _atomic_json(self.output_dir / "alignment.json", alignment)
        if self.config.get("profile") in {
            "chinatravel_blended_v1",
            "chinatravel_blended_v1_1",
        } and not all(alignment["checks"].values()):
            failed = [key for key, value in alignment["checks"].items() if not value]
            raise SynthesisError(f"Blended synthesis acceptance checks failed: {failed}")
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
    scenario: dict[str, Any],
    preference_audit: dict[str, Any] | None,
    polish_audit: list[dict[str, Any]],
    candidate_attempt: int,
) -> dict[str, Any]:
    return {
        "slot": asdict(slot),
        "blueprint": blueprint,
        "surface": surface,
        "task_spec": task_spec,
        "witness": witness,
        "scenario": scenario,
        "preference_audit": preference_audit,
        "polish_audit": polish_audit,
        "candidate_attempt": candidate_attempt,
    }


def _public_row(record: dict[str, Any]) -> dict[str, Any]:
    spec = record["task_spec"]
    trip = spec["trip"]
    return {
        "uid": spec["task_id"],
        "tag": f"synthetic_{record['slot']['synthesis_profile']}",
        "task_type": record["slot"]["task_type"],
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
    lines = [f"# TravelWeaver {len(records)}-task synthesis preview", ""]
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
                (
                    f"- 行程密度：每天 {slot['attractions_per_day']} 个景点；"
                    f"市内优先 {slot['route_mode']}"
                ),
                f"- Blueprint：`{row['blueprint']['blueprint_id']}`",
                "",
            ]
        )
    return "\n".join(lines)


def _distributions(
    slots: tuple[PilotSlot, ...], records: list[dict[str, Any]]
) -> dict[str, Any]:
    recipe_families = Counter(key for slot in slots for key in slot.recipe)
    recipe_pairs = Counter(
        f"{left}+{right}"
        for slot in slots
        for left_index, left in enumerate(sorted(slot.recipe))
        for right in sorted(slot.recipe)[left_index + 1 :]
    )
    return {
        "count": len(slots),
        "task_types": dict(sorted(Counter(slot.task_type for slot in slots).items())),
        "origins": dict(
            sorted(Counter(str(row["slot"]["origin"]) for row in records).items())
        ),
        "destinations": dict(sorted(Counter(slot.destination for slot in slots).items())),
        "directed_od_pairs": len(
            {
                (str(row["slot"]["origin"]), str(row["slot"]["destination"]))
                for row in records
            }
        ),
        "undirected_od_pairs": len(
            {
                tuple(
                    sorted(
                        (
                            str(row["slot"]["origin"]),
                            str(row["slot"]["destination"]),
                        )
                    )
                )
                for row in records
            }
        ),
        "days": dict(sorted(Counter(str(slot.days) for slot in slots).items())),
        "travelers": dict(sorted(Counter(str(slot.travelers) for slot in slots).items())),
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
        "route_modes": dict(sorted(Counter(slot.route_mode for slot in slots).items())),
        "actual_route_modes": dict(
            sorted(Counter(str(row["witness"]["route_mode"]) for row in records).items())
        ),
        "transport_strategies": dict(
            sorted(Counter(slot.transport_strategy for slot in slots).items())
        ),
        "tightness": dict(sorted(Counter(slot.tightness for slot in slots).items())),
        "scenario_profiles": dict(
            sorted(Counter(slot.scenario_profile for slot in slots).items())
        ),
        "surface_styles": dict(
            sorted(Counter(slot.surface_style for slot in slots).items())
        ),
        "attractions_per_day": dict(
            sorted(Counter(str(slot.attractions_per_day) for slot in slots).items())
        ),
        "meal_included": dict(
            sorted(Counter(str(slot.include_meal).lower() for slot in slots).items())
        ),
        "constraint_recipes": dict(sorted(recipe_families.items())),
        "constraint_recipe_pairs": dict(sorted(recipe_pairs.items())),
        "unique_recipe_signatures": len({tuple(sorted(slot.recipe)) for slot in slots}),
        "unique_blueprints": len(
            {row["blueprint"]["blueprint_id"] for row in records}
        ),
        "unique_surfaces": len({row["surface"]["surface_id"] for row in records}),
        "surface_quality": _surface_quality(records),
        "preferences": dict(
            sorted(Counter(kind for slot in slots for kind in slot.preference_kinds).items())
        ),
    }


def _usage(records: list[dict[str, Any]]) -> dict[str, int]:
    keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    return {
        key: sum(int(row["surface"].get("usage", {}).get(key, 0)) for row in records)
        for key in keys
    }


def _surface_quality(records: list[dict[str, Any]]) -> dict[str, Any]:
    queries = [str(row["surface"]["public_query"]) for row in records]
    models = Counter(str(row["surface"]["polisher_model"]) for row in records)
    similarities = [
        SequenceMatcher(None, left, right).ratio()
        for index, left in enumerate(queries)
        for right in queries[index + 1 :]
    ]
    lengths = [len(query) for query in queries]
    human_queries = [
        str(row["surface"]["public_query"])
        for row in records
        if row["slot"].get("task_type") == "human_like"
    ]
    openings = Counter(query[:20] for query in human_queries)
    human_rows = [
        row for row in records if row["slot"].get("task_type") == "human_like"
    ]
    repeated_personas = sum(
        bool(row["slot"].get("metadata_prefix"))
        and str(row["slot"].get("persona_context", ""))
        in str(row["surface"]["public_query"])[
            len(str(row["slot"].get("metadata_prefix", ""))) :
        ]
        for row in human_rows
    )
    template_terms = ("硬性条件", "必须满足以下要求")
    warning_counts = Counter(
        str(warning).split(":", 1)[0]
        for row in records
        for warning in row["surface"].get("validation_warnings", [])
    )
    return {
        "canonical_fallbacks": sum(
            count for model, count in models.items() if model.endswith(":canonical-fallback")
        ),
        "exact_unique_queries": len(set(queries)),
        "max_pairwise_sequence_similarity": round(max(similarities, default=0.0), 4),
        "pairs_at_or_above_0_8_similarity": sum(value >= 0.8 for value in similarities),
        "polisher_models": dict(sorted(models.items())),
        "surfaces_with_warnings": sum(
            bool(row["surface"].get("validation_warnings")) for row in records
        ),
        "validation_warning_counts": dict(sorted(warning_counts.items())),
        "query_length": {
            "min": min(lengths, default=0),
            "mean": round(sum(lengths) / len(lengths), 2) if lengths else 0.0,
            "max": max(lengths, default=0),
        },
        "human_template_term_rate": (
            round(
                sum(any(term in query for term in template_terms) for query in human_queries)
                / len(human_queries),
                4,
            )
            if human_queries
            else 0.0
        ),
        "human_max_opening_share": (
            round(max(openings.values(), default=0) / len(human_queries), 4)
            if human_queries
            else 0.0
        ),
        "human_metadata_persona_repetitions": repeated_personas,
    }


def _alignment(
    records: list[dict[str, Any]],
    distributions: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    requested_count = int(config["count"])
    seed = int(config["seed"])
    queries = [str(row["surface"]["public_query"]) for row in records]
    benchmark_queries = _benchmark_queries()
    benchmark_reuse = sorted(set(queries) & benchmark_queries)
    quality = dict(distributions["surface_quality"])
    hard_passes = sum(
        bool(row["witness"]["reward_detail"].get("all_hard_pass"))
        and float(row["witness"]["reward_detail"].get("reward", 0.0)) == 1.0
        for row in records
    )
    actual_types = dict(distributions["task_types"])
    actual_scenarios = dict(distributions["scenario_profiles"])
    is_blended = config.get("profile") in {BLENDED_PROFILE, BLENDED_V1_1_PROFILE}
    expected_types = (
        {
            key: value
            for key, value in blended_task_type_quotas(requested_count, seed).items()
            if value
        }
        if is_blended
        else None
    )
    expected_scenarios = (
        {
            key: value
            for key, value in blended_scenario_quotas(requested_count, seed).items()
            if value
        }
        if is_blended
        else None
    )
    checks = {
        "requested_count": len(records) == requested_count,
        "hard_reward_100_percent": hard_passes == len(records),
        "unique_queries": len(set(queries)) == len(queries),
        "no_benchmark_exact_reuse": not benchmark_reuse,
        "human_template_free_at_least_70_percent": (
            quality["human_template_term_rate"] <= 0.3
        ),
        "human_same_opening_at_most_10_percent": (
            quality["human_max_opening_share"] <= 0.1
        ),
    }
    if is_blended:
        assert expected_types is not None
        assert expected_scenarios is not None
        checks["task_type_quotas"] = actual_types == expected_types
        checks["scenario_quotas"] = actual_scenarios == expected_scenarios
        checks["preference_audit_count"] = sum(
            row.get("preference_audit") is not None for row in records
        ) == expected_types.get("preference_like", 0)
    if config.get("profile") == BLENDED_V1_1_PROFILE:
        audits = [
            row["preference_audit"]
            for row in records
            if row.get("preference_audit") is not None
        ]
        if config.get("validation_policy") != "minimal_semantic":
            checks["human_metadata_persona_not_repeated"] = (
                quality["human_metadata_persona_repetitions"] == 0
            )
        checks["preference_metrics_discriminate"] = all(
            len({candidate["metric_value"] for candidate in audit["candidates"]}) >= 2
            for audit in audits
        )
        checks["preference_candidates_all_hard_pass"] = all(
            candidate.get("all_hard_pass") is True
            and float(candidate.get("hard_reward", 0.0)) == 1.0
            for audit in audits
            for candidate in audit["candidates"]
        )
    return {
        "profile": config.get("profile", "pilot_v2_1"),
        "expected": {
            "task_types": expected_types,
            "scenario_profiles": expected_scenarios,
            "preference_audit_count": (
                expected_types.get("preference_like", 0)
                if expected_types is not None
                else None
            ),
        },
        "actual": {
            "task_types": actual_types,
            "scenario_profiles": actual_scenarios,
            "hard_passes": hard_passes,
            "benchmark_exact_reuse": benchmark_reuse,
            "surface_quality": quality,
        },
        "checks": checks,
    }


def _benchmark_queries() -> set[str]:
    path = project_root() / "data" / "tasks" / "benchmark.public.jsonl"
    if not path.exists():
        return set()
    queries: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        query = row.get("query")
        if isinstance(query, str):
            queries.add(query)
    return queries


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
