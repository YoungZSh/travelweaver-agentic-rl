"""Aggregate acceptance metrics for a synthesized programmatic SFT batch."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any

from .rationale_contract import has_visible_price_comparison

BATCH_AUDIT_VERSION = "travelweaver-programmatic-batch-audit-v5"

_EXPECTED_TOOLS = (
    "list_attraction_categories",
    "search_attractions",
    "list_restaurant_cuisines",
    "search_restaurants",
    "search_restaurants_by_food",
    "list_hotel_features",
    "search_hotels",
    "search_intercity_transport",
    "search_nearby",
    "inspect_place",
    "check_place_open",
    "get_route",
    "next_page",
    "save_candidate",
    "list_candidates",
    "remove_candidate",
    "submit_plan",
)

def audit_programmatic_batch(
    task_dir: str | Path,
    trajectory_path: str | Path,
    trajectory_audit_path: str | Path,
    *,
    sft_manifest_path: str | Path | None = None,
    output_path: str | Path | None = None,
    require_rationale_polish: bool = False,
) -> dict[str, Any]:
    directory = Path(task_dir)
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((directory / "records").glob("*.json"))
    ]
    trajectories = _read_jsonl(Path(trajectory_path))
    trajectory_audits = _read_jsonl(Path(trajectory_audit_path))
    if not records or len(records) != len(trajectories) or len(records) != len(
        trajectory_audits
    ):
        raise ValueError("Task, trajectory, and trajectory-audit counts must match and be nonzero.")
    synthesis_manifest = _read_json(directory / "manifest.json")
    alignment = _read_json(directory / "alignment.json")
    official_rows = _read_jsonl(directory / "official-audit.jsonl")
    family_counts = Counter(str(row["sample_family"]) for row in trajectories)
    record_by_task = {str(record["task_spec"]["task_id"]): record for record in records}
    tool_counts = Counter(
        str(step["action"]["tool"])
        for row in trajectories
        for step in row["steps"]
    )
    supervised_tool_counts = Counter(
        str(step["action"]["tool"])
        for row in trajectories
        for step, mask in zip(row["steps"], row["assistant_loss_mask"], strict=True)
        if mask
    )
    tool_sample_counts = Counter(
        tool
        for row in trajectories
        for tool in {str(step["action"]["tool"]) for step in row["steps"]}
    )
    supervised_tool_sample_counts = Counter(
        tool
        for row in trajectories
        for tool in {
            str(step["action"]["tool"])
            for step, mask in zip(row["steps"], row["assistant_loss_mask"], strict=True)
            if mask
        }
    )
    minimum_tool_calls = max(1, len(records) // 10)
    tool_coverage = {
        "expected_tools": list(_EXPECTED_TOOLS),
        "minimum_calls_per_tool": minimum_tool_calls,
        "all_tools_present": all(tool_counts[tool] > 0 for tool in _EXPECTED_TOOLS),
        "all_tools_nontrivial": all(
            tool_counts[tool] >= minimum_tool_calls for tool in _EXPECTED_TOOLS
        ),
        "all_tools_have_supervised_targets": all(
            supervised_tool_counts[tool] > 0 for tool in _EXPECTED_TOOLS
        ),
        "all_tools_have_nontrivial_supervised_samples": all(
            supervised_tool_sample_counts[tool] >= minimum_tool_calls
            for tool in _EXPECTED_TOOLS
        ),
        "counts": {tool: tool_counts[tool] for tool in _EXPECTED_TOOLS},
        "supervised_counts": {
            tool: supervised_tool_counts[tool] for tool in _EXPECTED_TOOLS
        },
        "sample_counts": {tool: tool_sample_counts[tool] for tool in _EXPECTED_TOOLS},
        "supervised_sample_counts": {
            tool: supervised_tool_sample_counts[tool] for tool in _EXPECTED_TOOLS
        },
    }
    causal_grounding = _causal_grounding_checks(trajectories, record_by_task)
    action_counts: dict[str, list[int]] = {}
    for row in trajectories:
        action_counts.setdefault(str(row["sample_family"]), []).append(int(row["step_count"]))
    queries = [str(record["surface"]["public_query"]) for record in records]
    query_by_task = {
        str(record["task_spec"]["task_id"]): str(record["surface"]["public_query"])
        for record in records
    }
    warning_counts = Counter(
        warning
        for record in records
        for warning in record["surface"].get("validation_warnings", [])
    )
    tpc_rows = [_tpc_metrics(record) for record in records]
    masks_valid = {
        "efficient_all_supervised": all(
            all(row["assistant_loss_mask"])
            for row in trajectories
            if row["sample_family"] == "efficient_success"
        ),
        "loops_all_masked": all(
            all(
                not mask
                for mask, reason in zip(
                    row["assistant_loss_mask"], row["mask_reasons"], strict=True
                )
                if reason == "injected_loop"
            )
            for row in trajectories
            if row["sample_family"] == "loop_recovery"
        ),
        "loop_recovery_supervised": all(
            any(
                mask and reason == "supervised_loop_exit_reflection"
                for mask, reason in zip(
                    row["assistant_loss_mask"], row["mask_reasons"], strict=True
                )
            )
            for row in trajectories
            if row["sample_family"] == "loop_recovery"
        ),
        "evidence_ready_only_submit_supervised": all(
            sum(row["assistant_loss_mask"]) == 1 and row["assistant_loss_mask"][-1]
            for row in trajectories
            if row["sample_family"] == "evidence_ready_submit"
        ),
    }
    reflections = [
        str(turn["visible_reflection"])
        for row in trajectory_audits
        for turn in row.get("turns", [])
        if turn.get("visible_reflection")
    ]
    assistant_contents = [
        str(message.get("content", ""))
        for row in trajectories
        for message in row["messages"]
        if message.get("role") == "assistant"
    ]
    polish_rows = [
        row["rationale_polish"]
        for row in trajectory_audits
        if isinstance(row.get("rationale_polish"), Mapping)
    ]
    rationale_checks = {
        "all_tool_turns_have_visible_rationale": bool(assistant_contents)
        and all(content.strip() for content in assistant_contents),
        "all_turns_have_template_audit": all(
            all(str(turn.get("template_rationale", "")).strip() for turn in row["turns"])
            for row in trajectory_audits
        ),
        "required_polish_present": not require_rationale_polish
        or len(polish_rows) == len(records),
    }
    report: dict[str, Any] = {
        "audit_version": BATCH_AUDIT_VERSION,
        "task_dir": str(directory.resolve()),
        "count": len(records),
        "api_calls": int(synthesis_manifest.get("api_calls", 0)),
        "family_counts": dict(sorted(family_counts.items())),
        "action_counts": {
            family: _distribution(values) for family, values in sorted(action_counts.items())
        },
        "tool_counts": dict(sorted(tool_counts.items())),
        "tool_coverage": tool_coverage,
        "causal_grounding": causal_grounding,
        "question_characters": _distribution([len(query) for query in queries]),
        "question_uniqueness": {
            "exact": len(set(queries)),
            "alignment_checks": alignment.get("checks", {}),
        },
        "surface_quality": synthesis_manifest.get("distributions", {}).get(
            "surface_quality", {}
        ),
        "warning_counts": dict(sorted(warning_counts.items())),
        "reward": {
            "reward_one": sum(float(row["final_reward"]) == 1.0 for row in trajectories),
            "all_hard_pass": sum(
                row.get("reward_detail", {}).get("all_hard_pass") is True
                for row in trajectories
            ),
            "plan_submitted": sum(
                row.get("termination_reason") == "plan_submitted" for row in trajectories
            ),
        },
        "official": {
            "count": len(official_rows),
            "schema_passes": sum(row.get("schema_passed") is True for row in official_rows),
            "commonsense_passes": sum(
                row.get("commonsense_passed") is True for row in official_rows
            ),
        },
        "mask_checks": masks_valid,
        "visible_reflections": {
            "count": len(reflections),
            "unique": len(set(reflections)),
            "unique_rate": round(len(set(reflections)) / len(reflections), 6)
            if reflections
            else 0.0,
        },
        "rationale": {
            "checks": rationale_checks,
            "turns": len(assistant_contents),
            "unique": len(set(assistant_contents)),
            "unique_rate": round(
                len(set(assistant_contents)) / len(assistant_contents), 6
            )
            if assistant_contents
            else 0.0,
            "polished_samples": len(polish_rows),
            "polish_outcomes": dict(
                sorted(Counter(str(row.get("outcome")) for row in polish_rows).items())
            ),
            "polished_turns": sum(int(row.get("polished_turns", 0)) for row in polish_rows),
            "fallback_turns": sum(int(row.get("fallback_turns", 0)) for row in polish_rows),
        },
        "examples": _family_examples(trajectories, query_by_task),
        "tpc": {
            "dav_raw": round(mean(row["dav"] for row in tpc_rows), 6),
            "dav_score": round(mean(row["dav_score"] for row in tpc_rows), 6),
            "att_minutes": round(mean(row["att"] for row in tpc_rows), 6),
            "att_score": round(mean(row["att_score"] for row in tpc_rows), 6),
            "ddr_raw": round(mean(row["ddr"] for row in tpc_rows), 6),
            "ddr_score": round(mean(row["ddr_score"] for row in tpc_rows), 6),
        },
    }
    if sft_manifest_path is not None:
        sft_manifest = _read_json(Path(sft_manifest_path))
        report["qwen_adapter"] = sft_manifest.get("qwen_adapter")
    report["accepted"] = (
        family_counts
        == Counter(
            {
                "efficient_success": len(records) * 8 // 10,
                "loop_recovery": len(records) // 10,
                "evidence_ready_submit": len(records) // 10,
            }
        )
        and all(masks_valid.values())
        and all(rationale_checks.values())
        and all(
            tool_coverage[key]
            for key in (
                "all_tools_present",
                "all_tools_nontrivial",
                "all_tools_have_supervised_targets",
                "all_tools_have_nontrivial_supervised_samples",
            )
        )
        and all(causal_grounding["checks"].values())
        and all(value == len(records) for value in report["reward"].values())
        and report["official"]["schema_passes"] == len(records)
        and report["official"]["commonsense_passes"] == len(records)
    )
    if output_path is not None:
        _atomic_json(Path(output_path), report)
    return report


def _family_examples(
    trajectories: Sequence[Mapping[str, Any]], query_by_task: Mapping[str, str]
) -> dict[str, Any]:
    examples: dict[str, Any] = {}
    for family in ("efficient_success", "loop_recovery", "evidence_ready_submit"):
        row = next(item for item in trajectories if item["sample_family"] == family)
        assistant_messages = [
            message for message in row["messages"] if message.get("role") == "assistant"
        ]
        examples[family] = {
            "task_id": row["task_id"],
            "question": query_by_task[str(row["task_id"])],
            "actions": [
                {
                    "position": index,
                    "tool": step["action"]["tool"],
                    "arguments": step["action"]["arguments"],
                    "loss_mask": row["assistant_loss_mask"][index],
                    "mask_reason": row["mask_reasons"][index],
                    "visible_reflection": assistant_messages[index].get("content", ""),
                }
                for index, step in enumerate(row["steps"])
            ],
            "final_plan": row["final_plan"],
            "reward": row["final_reward"],
        }
    return examples


def _causal_grounding_checks(
    trajectories: Sequence[Mapping[str, Any]],
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Verify that tool decisions only introduce facts available at that turn.

    Environment replay proves an ID is legal, but it cannot prove that an SFT teacher did not
    smuggle a witness-only name, food, or exact search bound into the policy.  This audit uses the
    user query and prior model-visible tool responses as the causal frontier for each action.
    """

    violations: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    nearby_categories: Counter[str] = Counter()
    remove_positions: Counter[int] = Counter()
    for trajectory in trajectories:
        task_id = str(trajectory["task_id"])
        record = records[task_id]
        query = _query(trajectory)
        assistant_messages = [
            message for message in trajectory["messages"] if message.get("role") == "assistant"
        ]
        steps = trajectory["steps"]
        if len(assistant_messages) != len(steps):
            violations["assistant_step_alignment"] += 1
            continue
        witness_names = {
            str(entity.get("name"))
            for entity in record["witness"]["evidence_bundle"]["entities"].values()
            if isinstance(entity, Mapping)
            and isinstance(entity.get("name"), str)
            and len(str(entity["name"]).strip()) >= 3
        }
        visible_text = query
        visible_ids: set[str] = set()
        saved_ids: set[str] = set()
        catalog_values: dict[str, set[str]] = {}
        for position, (step, assistant) in enumerate(
            zip(steps, assistant_messages, strict=True)
        ):
            action = step["action"]
            tool = str(action["tool"])
            arguments = action["arguments"]
            content = str(assistant.get("content", ""))
            counts[tool] += 1
            if tool in {"search_attractions", "search_restaurants", "search_hotels"}:
                search_name = arguments.get("query")
                if isinstance(search_name, str) and search_name not in visible_text:
                    violations[f"ungrounded_name_query:{tool}"] += 1
            if tool == "search_restaurants_by_food":
                food = str(arguments["food"])
                if food not in visible_text:
                    violations["ungrounded_food_query"] += 1
            if tool == "search_intercity_transport":
                departure = str(arguments.get("earliest_departure", ""))
                if not _is_coarse_departure(departure):
                    violations["noncoarse_transport_departure"] += 1
            guarded_argument = {
                "inspect_place": "place_id",
                "check_place_open": "place_id",
                "search_nearby": "place_id",
                "save_candidate": "entity_id",
            }.get(tool)
            if guarded_argument is not None:
                entity_id = str(arguments[guarded_argument])
                if entity_id not in visible_ids:
                    violations[f"unseen_id:{tool}"] += 1
            if tool == "get_route":
                for key in ("origin_place_id", "destination_place_id"):
                    if str(arguments[key]) not in visible_ids:
                        violations["unseen_id:get_route"] += 1
            if tool == "save_candidate":
                saved_ids.add(str(arguments["entity_id"]))
            if tool == "remove_candidate":
                candidate_id = str(arguments["candidate_id"])
                remove_positions[position] += 1
                if candidate_id not in saved_ids:
                    violations["remove_unsaved_candidate"] += 1
                if position == 0 or steps[position - 1]["action"]["tool"] != "list_candidates":
                    violations["remove_without_immediate_review"] += 1
                if not has_visible_price_comparison(content):
                    violations["remove_without_visible_price_reason"] += 1
            if tool == "search_nearby":
                nearby_categories[str(arguments["category"])] += 1
                radius = arguments.get("radius_km")
                if radius not in {2, 5, 10, 20, 50}:
                    violations["unexpected_nearby_radius"] += 1
                if arguments.get("top_k") != 40:
                    violations["unexpected_nearby_top_k"] += 1
            if tool in _CATALOG_CONSUMERS:
                result = step["result"]["observation"].get("tool_result") or {}
                # The search parameter (for example ``category``) differs from the
                # catalog response key (``categories``).  Read the latter when
                # checking that the immediately following search is grounded in
                # the just-observed catalog response.
                values = result.get(_CATALOG_CONSUMERS[tool][2], [])
                catalog_values[tool] = {str(value) for value in values}
                if position + 1 >= len(steps):
                    violations[f"catalog_not_consumed:{tool}"] += 1
                else:
                    next_action = steps[position + 1]["action"]
                    search_tool, parameter, _ = _CATALOG_CONSUMERS[tool]
                    selected = next_action["arguments"].get(parameter)
                    if (
                        next_action["tool"] != search_tool
                        or not isinstance(selected, str)
                        or not _catalog_selection_is_visible(
                            selected, catalog_values[tool]
                        )
                    ):
                        violations[f"catalog_not_consumed:{tool}"] += 1
            for name in witness_names:
                if name in content and name not in visible_text:
                    violations["hidden_witness_name_in_rationale"] += 1
                    break
            model_response = step.get("model_tool_response")
            model_result = (
                model_response.get("tool_result")
                if isinstance(model_response, Mapping)
                else None
            )
            _record_visible_ids(model_result, visible_ids)
            visible_text += "\n" + json.dumps(
                model_response or {}, ensure_ascii=False, separators=(",", ":")
            )
    checks = {
        "all_name_queries_grounded": not any(
            key.startswith("ungrounded_name_query:") for key in violations
        ),
        "all_food_queries_grounded": violations["ungrounded_food_query"] == 0,
        "all_id_actions_grounded": not any(
            key.startswith("unseen_id:") for key in violations
        ),
        "catalog_results_drive_next_search": not any(
            key.startswith("catalog_not_consumed:") for key in violations
        ),
        "all_transport_searches_use_coarse_time_windows": violations[
            "noncoarse_transport_departure"
        ]
        == 0,
        "candidate_removal_has_visible_comparison": not any(
            key.startswith("remove_") for key in violations
        ),
        "nearby_search_parameters_follow_progressive_policy": not any(
            key.startswith("unexpected_nearby_") for key in violations
        ),
        "rationales_do_not_leak_hidden_witness_names": violations[
            "hidden_witness_name_in_rationale"
        ]
        == 0,
    }
    return {
        "checks": checks,
        "violations": dict(sorted(violations.items())),
        "tool_turns": dict(sorted(counts.items())),
        "nearby_category_counts": dict(sorted(nearby_categories.items())),
        "remove_position_distribution": dict(sorted(remove_positions.items())),
    }


_CATALOG_CONSUMERS = {
    "list_attraction_categories": ("search_attractions", "category", "categories"),
    "list_restaurant_cuisines": ("search_restaurants", "cuisine", "cuisines"),
    "list_hotel_features": ("search_hotels", "hotel_type", "features"),
}


def _query(trajectory: Mapping[str, Any]) -> str:
    return next(
        str(message.get("content", ""))
        for message in trajectory["messages"]
        if message.get("role") == "user"
    )


def _is_coarse_departure(value: str) -> bool:
    if len(value) != 5 or value[2] != ":" or not value[:2].isdigit():
        return False
    hour = int(value[:2])
    return value.endswith(":00") and 0 <= hour <= 21 and hour % 3 == 0


def _catalog_selection_is_visible(selected: str, values: set[str]) -> bool:
    """Accept a raw multifacet value when each listed facet was cataloged.

    ChinaTravel stores some attraction types as values such as
    ``博物馆/纪念馆`` while the catalog deliberately exposes selectable atomic
    facets (``博物馆`` and ``纪念馆``).  Searching with the raw value is valid
    and remains causally grounded if every component was visible.
    """

    if selected in values:
        return True
    parts = [part.strip() for part in re.split(r"[,，、|/]+", selected) if part.strip()]
    return bool(parts) and all(part in values for part in parts)


def _record_visible_ids(tool_result: Any, visible_ids: set[str]) -> None:
    if not isinstance(tool_result, Mapping):
        return
    items: list[Any] = list(tool_result.get("items", []))
    if isinstance(tool_result.get("item"), Mapping):
        items.append(tool_result["item"])
    for item in items:
        if not isinstance(item, Mapping):
            continue
        entity_id = item.get("place_id") or item.get("transport_id")
        if isinstance(entity_id, str):
            visible_ids.add(entity_id)
        for key in ("origin_anchor_id", "destination_anchor_id"):
            anchor_id = item.get(key)
            if isinstance(anchor_id, str):
                visible_ids.add(anchor_id)
        for key in ("origin_anchor", "destination_anchor"):
            anchor = item.get(key)
            if isinstance(anchor, Mapping) and isinstance(anchor.get("place_id"), str):
                visible_ids.add(str(anchor["place_id"]))


def _tpc_metrics(record: Mapping[str, Any]) -> dict[str, float]:
    snapshot = record["witness"]["plan_snapshot"]
    evidence = record["witness"]["evidence_bundle"]
    activities = snapshot["activities"]
    days = int(record["task_spec"]["trip"]["days"])
    attractions = sum(item["activity_type"] == "attraction" for item in activities)
    meals = sum(item["activity_type"] in {"breakfast", "lunch", "dinner"} for item in activities)
    route_minutes: list[float] = []
    for activity in activities:
        route_id = activity.get("route_from_previous_id")
        if route_id is None:
            continue
        route = evidence["routes"][route_id]
        route_minutes.append(
            sum(
                _duration(segment["start_time"], segment["end_time"])
                for segment in route["segments"]
            )
        )
    dav = attractions / days
    ddr = meals / days
    att = mean(route_minutes) if route_minutes else 120.0
    return {
        "dav": dav,
        "dav_score": min(max(dav / 4, 0.0), 1.0),
        "att": att,
        "att_score": min(max((120 - att) / 105, 0.0), 1.0),
        "ddr": ddr,
        "ddr_score": min(max(ddr / 3, 0.0), 1.0),
    }


def _duration(start: str, end: str) -> int:
    start_minutes = int(start[:2]) * 60 + int(start[3:])
    end_minutes = int(end[:2]) * 60 + int(end[3:])
    if end_minutes >= start_minutes:
        return end_minutes - start_minutes
    return end_minutes + 1440 - start_minutes


def _distribution(values: Sequence[int]) -> dict[str, float | int]:
    ordered = sorted(values)

    def percentile(ratio: float) -> int:
        return ordered[round((len(ordered) - 1) * ratio)]

    return {
        "min": ordered[0],
        "mean": round(mean(ordered), 3),
        "p50": percentile(0.5),
        "p90": percentile(0.9),
        "max": ordered[-1],
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)
