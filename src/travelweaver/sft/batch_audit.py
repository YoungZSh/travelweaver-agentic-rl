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

from ..synthesis.trajectory_policy import (
    MAX_CONSECUTIVE_TOOL_CALLS,
    MAX_SYNTHESIS_VALID_STEPS,
    TRAJECTORY_POLICY_VERSION,
)
from .programmatic import SAMPLE_FAMILIES, minimum_tool_coverage_samples
from .rationale_contract import has_visible_price_comparison

BATCH_AUDIT_VERSION = "travelweaver-programmatic-batch-audit-v11"
ROLLOUT_COMPARISON_VERSION = "travelweaver-rollout-comparison-v1"


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
    public_tools = _public_tool_names(trajectories)
    minimum_coverage = minimum_tool_coverage_samples(len(records))
    tool_coverage = {
        "required_tools": list(public_tools),
        "minimum_supervised_samples_per_tool": minimum_coverage,
        "required_tools_present": all(
            tool_counts[tool] > 0 for tool in public_tools
        ),
        "required_tools_supervised": all(
            supervised_tool_counts[tool] > 0 for tool in public_tools
        ),
        "minimum_sample_coverage": all(
            tool_sample_counts[tool] >= minimum_coverage for tool in public_tools
        ),
        "minimum_supervised_sample_coverage": all(
            supervised_tool_sample_counts[tool] >= minimum_coverage for tool in public_tools
        ),
        "coverage_recommendation_only": True,
        "counts": dict(sorted(tool_counts.items())),
        "supervised_counts": dict(sorted(supervised_tool_counts.items())),
        "sample_counts": dict(sorted(tool_sample_counts.items())),
        "supervised_sample_counts": dict(sorted(supervised_tool_sample_counts.items())),
    }
    audit_warnings = _audit_warnings(
        public_tools=public_tools,
        minimum_coverage=minimum_coverage,
        tool_sample_counts=tool_sample_counts,
        supervised_tool_sample_counts=supervised_tool_sample_counts,
        official_rows=official_rows,
        sample_count=len(records),
    )
    trajectory_policy = _short_trajectory_policy_checks(records, trajectories, trajectory_audits)
    action_concentration = trajectory_action_statistics(trajectories)
    causal_grounding = _causal_grounding_checks(trajectories, record_by_task)
    evidence_path_planning = {
        "kinds": dict(
            sorted(
                Counter(
                    str(path.get("kind"))
                    for row in trajectory_audits
                    for path in row.get("evidence_paths", [])
                ).items()
            )
        ),
        "selection_reasons": dict(
            sorted(
                Counter(
                    str(path.get("selection_reason"))
                    for row in trajectory_audits
                    for path in row.get("evidence_paths", [])
                ).items()
            )
        ),
    }
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
        "all_actions_supervised": all(all(row["assistant_loss_mask"]) for row in trajectories),
        "no_masked_recovery_actions": all(
            "injected_loop" not in row["mask_reasons"] for row in trajectories
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
        "trajectory_policy": trajectory_policy,
        "action_concentration": action_concentration,
        "evidence_path_planning": evidence_path_planning,
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
        "audit_warnings": {
            "count": len(audit_warnings),
            "items": audit_warnings,
        },
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
            "schema_required": True,
            "commonsense_recommendation_only": True,
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
    report["accepted"] = _hard_acceptance_passes(
        family_counts=family_counts,
        masks_valid=masks_valid,
        rationale_checks=rationale_checks,
        tool_coverage=tool_coverage,
        trajectory_policy=trajectory_policy,
        action_concentration=action_concentration,
        causal_grounding=causal_grounding,
        reward=report["reward"],
        official=report["official"],
        sample_count=len(records),
    )
    if output_path is not None:
        _atomic_json(Path(output_path), report)
    return report


def _hard_acceptance_passes(
    *,
    family_counts: Mapping[str, int],
    masks_valid: Mapping[str, bool],
    rationale_checks: Mapping[str, bool],
    tool_coverage: Mapping[str, Any],
    trajectory_policy: Mapping[str, Any],
    action_concentration: Mapping[str, Any],
    causal_grounding: Mapping[str, Any],
    reward: Mapping[str, int],
    official: Mapping[str, Any],
    sample_count: int,
) -> bool:
    """Evaluate hard gates; recommendation-only metrics must stay out of this function."""

    return bool(
        set(family_counts) <= set(SAMPLE_FAMILIES)
        and all(masks_valid.values())
        and all(rationale_checks.values())
        and tool_coverage["required_tools_present"]
        and tool_coverage["required_tools_supervised"]
        and all(trajectory_policy["checks"].values())
        and all(action_concentration["checks"].values())
        and all(causal_grounding["checks"].values())
        and all(value == sample_count for value in reward.values())
        and official["schema_passes"] == sample_count
    )


def _audit_warnings(
    *,
    public_tools: Sequence[str],
    minimum_coverage: int,
    tool_sample_counts: Mapping[str, int],
    supervised_tool_sample_counts: Mapping[str, int],
    official_rows: Sequence[Mapping[str, Any]],
    sample_count: int,
) -> list[dict[str, Any]]:
    """Build explicit non-blocking warnings without weakening true hard gates."""

    warnings: list[dict[str, Any]] = []
    undercovered: list[dict[str, Any]] = []
    for tool in public_tools:
        samples = int(tool_sample_counts.get(tool, 0))
        supervised_samples = int(supervised_tool_sample_counts.get(tool, 0))
        if samples >= minimum_coverage and supervised_samples >= minimum_coverage:
            continue
        undercovered.append(
            {
                "tool": tool,
                "recommended_minimum_samples": minimum_coverage,
                "samples": samples,
                "sample_rate": round(samples / sample_count, 6),
                "supervised_samples": supervised_samples,
                "supervised_sample_rate": round(supervised_samples / sample_count, 6),
                "sample_shortfall": max(0, minimum_coverage - samples),
                "supervised_sample_shortfall": max(
                    0, minimum_coverage - supervised_samples
                ),
            }
        )
    if undercovered:
        warnings.append(
            {
                "code": "tool_sample_coverage_below_recommendation",
                "severity": "warning",
                "blocking": False,
                "message": (
                    "Some tools appear in fewer than the recommended 10% of samples; "
                    "this does not block batch acceptance."
                ),
                "details": undercovered,
            }
        )

    commonsense_failures: list[dict[str, Any]] = []
    for row in official_rows:
        if row.get("commonsense_passed") is True:
            continue
        failed_checks = sorted(
            str(check.get("check"))
            for check in row.get("commonsense_checks", [])
            if isinstance(check, Mapping) and check.get("passed") is not True
        )
        commonsense_failures.append(
            {
                "task_id": str(row.get("uid", "")),
                "failed_checks": failed_checks,
            }
        )
    if commonsense_failures:
        warnings.append(
            {
                "code": "official_commonsense_not_all_passed",
                "severity": "warning",
                "blocking": False,
                "message": (
                    "The pinned ChinaTravel static Common Sense audit did not pass every "
                    "sample; this does not block batch acceptance."
                ),
                "passed": len(official_rows) - len(commonsense_failures),
                "failed": len(commonsense_failures),
                "details": commonsense_failures,
            }
        )
    return warnings


def trajectory_action_statistics(
    trajectories: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize tool coverage and action concentration for any rollout batch."""

    total = len(trajectories)
    if total == 0:
        raise ValueError("Trajectory action statistics require a nonempty batch.")
    tool_calls: Counter[str] = Counter()
    tool_samples: Counter[str] = Counter()
    max_per_trajectory: Counter[str] = Counter()
    max_consecutive: Counter[str] = Counter()
    longest_runs: list[int] = []
    duplicate_actions = 0
    samples_with_duplicates = 0
    for trajectory in trajectories:
        actions = [
            step.get("action")
            for step in trajectory.get("steps", [])
            if isinstance(step, Mapping) and isinstance(step.get("action"), Mapping)
        ]
        tools = [str(action.get("tool")) for action in actions]
        counts = Counter(tools)
        tool_calls.update(counts)
        tool_samples.update(counts.keys())
        for tool, count in counts.items():
            max_per_trajectory[tool] = max(max_per_trajectory[tool], count)

        seen: Counter[str] = Counter(
            json.dumps(action, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            for action in actions
        )
        repeated = sum(count - 1 for count in seen.values() if count > 1)
        duplicate_actions += repeated
        samples_with_duplicates += repeated > 0

        current_tool: str | None = None
        current_run = 0
        longest = 0
        for tool in tools:
            if tool == current_tool:
                current_run += 1
            else:
                current_tool = tool
                current_run = 1
            longest = max(longest, current_run)
            max_consecutive[tool] = max(max_consecutive[tool], current_run)
        longest_runs.append(longest)

    tool_statistics = {
        tool: {
            "calls": tool_calls[tool],
            "samples": tool_samples[tool],
            "sample_rate": round(tool_samples[tool] / total, 6),
            "max_per_trajectory": max_per_trajectory[tool],
            "max_consecutive": max_consecutive[tool],
        }
        for tool in sorted(tool_calls)
    }
    return {
        "samples": total,
        "steps": _distribution(
            [
                sum(
                    isinstance(step, Mapping) and isinstance(step.get("action"), Mapping)
                    for step in trajectory.get("steps", [])
                )
                for trajectory in trajectories
            ]
        ),
        "distinct_tools": len(tool_statistics),
        "tools": tool_statistics,
        "longest_consecutive_run": _distribution(longest_runs),
        "identical_action_repeats": duplicate_actions,
        "samples_with_identical_action_repeats": samples_with_duplicates,
        "checks": {
            "no_identical_action_repeats": duplicate_actions == 0,
            "max_three_consecutive_tool_calls": all(
                value <= MAX_CONSECUTIVE_TOOL_CALLS
                for value in max_consecutive.values()
            ),
        },
    }


def compare_rollout_batches(
    programmatic_path: str | Path,
    model_path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compare two rollout strategies on an identical task-id cohort."""

    batches = {
        "programmatic": _read_jsonl(Path(programmatic_path)),
        "model": _read_jsonl(Path(model_path)),
    }
    task_ids = {
        name: {str(row["task_id"]) for row in rows}
        for name, rows in batches.items()
    }
    if not all(batches.values()):
        raise ValueError("Rollout comparison requires two nonempty batches.")
    if task_ids["programmatic"] != task_ids["model"]:
        raise ValueError("Rollout comparison batches must contain identical task ids.")

    summaries = {
        name: _rollout_batch_summary(rows) for name, rows in batches.items()
    }
    all_tools = sorted(
        {
            tool
            for summary in summaries.values()
            for tool in summary["actions"]["tools"]
        }
    )
    tool_comparison = {
        tool: {
            name: summary["actions"]["tools"].get(
                tool,
                {
                    "calls": 0,
                    "samples": 0,
                    "sample_rate": 0.0,
                    "max_per_trajectory": 0,
                    "max_consecutive": 0,
                },
            )
            for name, summary in summaries.items()
        }
        for tool in all_tools
    }
    report = {
        "comparison_version": ROLLOUT_COMPARISON_VERSION,
        "same_task_ids": True,
        "task_count": len(task_ids["programmatic"]),
        "paths": {
            "programmatic": str(Path(programmatic_path).resolve()),
            "model": str(Path(model_path).resolve()),
        },
        "batches": summaries,
        "tool_comparison": tool_comparison,
    }
    if output_path is not None:
        _atomic_json(Path(output_path), report)
    return report


def _rollout_batch_summary(trajectories: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    action_statistics = trajectory_action_statistics(trajectories)
    valid_action_counts: list[int] = []
    invalid_action_counts: list[int] = []
    failure_checks: Counter[str] = Counter()
    for trajectory in trajectories:
        valid = 0
        invalid = 0
        for step in trajectory.get("steps", []):
            if not isinstance(step, Mapping):
                continue
            is_valid = step.get("valid_action")
            if not isinstance(is_valid, bool):
                result = step.get("result")
                if isinstance(result, Mapping):
                    info = result.get("info")
                    if isinstance(info, Mapping):
                        is_valid = info.get("valid_action")
            if is_valid is True:
                valid += 1
            elif is_valid is False:
                invalid += 1
        valid_action_counts.append(valid)
        invalid_action_counts.append(invalid)
        detail = trajectory.get("reward_detail")
        if isinstance(detail, Mapping):
            for check in detail.get("checks", []):
                if isinstance(check, Mapping) and check.get("status") == "fail":
                    failure_checks[str(check.get("id", "unknown"))] += 1

    declared_tools = _public_tool_names(trajectories)
    observed_tools = action_statistics["tools"]
    usage_keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    usage = {
        key: {
            "total": sum(
                int(row.get("usage", {}).get(key, 0))
                for row in trajectories
                if isinstance(row.get("usage"), Mapping)
            ),
            "per_trajectory": _distribution(
                [
                    int(row.get("usage", {}).get(key, 0))
                    if isinstance(row.get("usage"), Mapping)
                    else 0
                    for row in trajectories
                ]
            ),
        }
        for key in usage_keys
    }
    return {
        "samples": len(trajectories),
        "model_names": dict(
            sorted(Counter(str(row.get("model")) for row in trajectories).items())
        ),
        "successes": sum(row.get("success") is True for row in trajectories),
        "success_rate": round(
            sum(row.get("success") is True for row in trajectories) / len(trajectories),
            6,
        ),
        "rft_accepted": sum(row.get("rft_accepted") is True for row in trajectories),
        "reward_one": sum(float(row.get("final_reward", 0.0)) == 1.0 for row in trajectories),
        "termination_reasons": dict(
            sorted(
                Counter(str(row.get("termination_reason")) for row in trajectories).items()
            )
        ),
        "failed_reward_checks": dict(sorted(failure_checks.items())),
        "valid_actions": _distribution(valid_action_counts),
        "invalid_actions": {
            "total": sum(invalid_action_counts),
            "affected_samples": sum(value > 0 for value in invalid_action_counts),
            "distribution": _distribution(invalid_action_counts),
        },
        "samples_reaching_50_valid_actions": sum(
            value >= MAX_SYNTHESIS_VALID_STEPS for value in valid_action_counts
        ),
        "api_turns": _distribution(
            [int(row.get("api_turn_count", 0)) for row in trajectories]
        ),
        "declared_tools": list(declared_tools),
        "observed_declared_tools": sum(tool in observed_tools for tool in declared_tools),
        "unobserved_declared_tools": [
            tool for tool in declared_tools if tool not in observed_tools
        ],
        "actions": action_statistics,
        "usage": usage,
    }


def _public_tool_names(trajectories: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    """Return the complete model-visible tool surface and reject schema drift."""

    declared: set[str] | None = None
    for trajectory in trajectories:
        names = {
            str(tool["function"]["name"])
            for tool in trajectory.get("tools", [])
            if isinstance(tool, Mapping)
            and isinstance(tool.get("function"), Mapping)
            and isinstance(tool["function"].get("name"), str)
        }
        if not names:
            raise ValueError("Programmatic trajectory has no valid public tool schemas.")
        if declared is None:
            declared = names
        elif declared != names:
            raise ValueError("Programmatic trajectory batch has inconsistent tool schemas.")
    if declared is None:
        raise ValueError("Programmatic trajectory batch is empty.")
    return tuple(sorted(declared))


def _short_trajectory_policy_checks(
    records: Sequence[Mapping[str, Any]],
    trajectories: Sequence[Mapping[str, Any]],
    trajectory_audits: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Verify the serialized budget is exact for every new teacher trajectory."""

    policy_by_task = {
        str(record["task_spec"]["task_id"]): record.get("trajectory_policy")
        for record in records
    }
    actual_pages: Counter[int] = Counter()
    policy_present = True
    action_budget_ok = True
    valid_actions_only = True
    consecutive_limit_ok = True
    audit_alignment = True
    for trajectory, audit in zip(trajectories, trajectory_audits, strict=True):
        task_id = str(trajectory["task_id"])
        policy = policy_by_task.get(task_id)
        if not isinstance(policy, Mapping):
            policy_present = False
            continue
        if (
            policy.get("policy_version") != TRAJECTORY_POLICY_VERSION
            or policy.get("max_valid_steps") != MAX_SYNTHESIS_VALID_STEPS
            or policy.get("max_consecutive_tool_calls")
            != MAX_CONSECUTIVE_TOOL_CALLS
        ):
            policy_present = False
            continue
        steps = trajectory["steps"]
        actual = sum(step["action"]["tool"] == "next_page" for step in steps)
        actual_pages[actual] += 1
        action_budget_ok &= len(steps) <= MAX_SYNTHESIS_VALID_STEPS
        valid_actions_only &= all(
            step["result"].get("info", {}).get("valid_action") is True for step in steps
        )
        tools = [str(step["action"]["tool"]) for step in steps]
        longest = 0
        previous: str | None = None
        current = 0
        for tool in tools:
            current = current + 1 if tool == previous else 1
            previous = tool
            longest = max(longest, current)
        consecutive_limit_ok &= longest <= MAX_CONSECUTIVE_TOOL_CALLS
        audit_alignment &= (
            audit.get("max_valid_steps") == MAX_SYNTHESIS_VALID_STEPS
            and audit.get("max_consecutive_tool_calls")
            == MAX_CONSECUTIVE_TOOL_CALLS
            and audit.get("next_page_calls") == actual
            and audit.get("action_count") == len(steps)
        )
    return {
        "actual_next_page_calls": dict(sorted(actual_pages.items())),
        "checks": {
            "policy_present": policy_present,
            "max_50_actions": action_budget_ok,
            "zero_invalid_actions": valid_actions_only,
            "max_three_consecutive_tool_calls": consecutive_limit_ok,
            "audit_alignment": audit_alignment,
        },
    }


def _family_examples(
    trajectories: Sequence[Mapping[str, Any]], query_by_task: Mapping[str, str]
) -> dict[str, Any]:
    examples: dict[str, Any] = {}
    for family in ("efficient_success",):
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
                    visible_values: set[str | int] = set(catalog_values[tool])
                    if (
                        tool == "list_hotel_features"
                        and selected is None
                        and next_action["arguments"].get("room_type") is not None
                    ):
                        parameter = "room_type"
                        selected = next_action["arguments"][parameter]
                        visible_values = {
                            int(value)
                            for value in result.get("room_types", [])
                            if isinstance(value, int) and not isinstance(value, bool)
                        }
                    if (
                        next_action["tool"] != search_tool
                        or (
                            isinstance(selected, str)
                            and not _catalog_selection_is_visible(
                                selected, {str(value) for value in visible_values}
                            )
                        )
                        or (
                            not isinstance(selected, str)
                            and selected not in visible_values
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
