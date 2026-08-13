import json

from travelweaver.sft.batch_audit import (
    _audit_warnings,
    _catalog_selection_is_visible,
    _distribution,
    _duration,
    _hard_acceptance_passes,
    _public_tool_names,
    compare_rollout_batches,
    trajectory_action_statistics,
)
from travelweaver.sft.rationale_contract import has_visible_price_comparison


def test_batch_audit_helpers_handle_percentiles_and_midnight() -> None:
    assert _duration("23:50", "00:20") == 30
    assert _duration("10:00", "10:45") == 45
    assert _distribution([1, 2, 3, 4, 5]) == {
        "min": 1,
        "mean": 3,
        "p50": 3,
        "p90": 5,
        "max": 5,
    }


def test_catalog_grounding_accepts_compound_raw_facets() -> None:
    visible = {"博物馆", "纪念馆", "公园"}

    assert _catalog_selection_is_visible("博物馆/纪念馆", visible)
    assert not _catalog_selection_is_visible("博物馆/艺术馆", visible)


def test_remove_rationale_accepts_natural_price_comparison_synonyms() -> None:
    assert has_visible_price_comparison(
        "清单里甲标价199元，而乙是0元，前者没有成本优势，所以把它移除。"
    )
    assert has_visible_price_comparison("甲人均更贵，因此移除这个候选。")
    assert not has_visible_price_comparison("甲看起来不合适，因此移除这个候选。")


def test_batch_audit_uses_the_complete_model_visible_tool_surface() -> None:
    trajectories = [
        {
            "tools": [
                {"function": {"name": "check_place_open"}},
                {"function": {"name": "search_nearby"}},
            ]
        },
        {
            "tools": [
                {"function": {"name": "check_place_open"}},
                {"function": {"name": "search_nearby"}},
            ]
        },
    ]

    assert _public_tool_names(trajectories) == ("check_place_open", "search_nearby")


def test_batch_audit_reports_coverage_and_commonsense_as_nonblocking_warnings() -> None:
    warnings = _audit_warnings(
        public_tools=("next_page", "submit_plan"),
        minimum_coverage=10,
        tool_sample_counts={"next_page": 8, "submit_plan": 100},
        supervised_tool_sample_counts={"next_page": 8, "submit_plan": 100},
        official_rows=[
            {
                "uid": "task-pass",
                "commonsense_passed": True,
                "commonsense_checks": [],
            },
            {
                "uid": "task-warning",
                "commonsense_passed": False,
                "commonsense_checks": [
                    {"check": "Is_hotels_correct", "passed": False}
                ],
            },
        ],
        sample_count=100,
    )

    assert [warning["code"] for warning in warnings] == [
        "tool_sample_coverage_below_recommendation",
        "official_commonsense_not_all_passed",
    ]
    assert all(warning["severity"] == "warning" for warning in warnings)
    assert all(warning["blocking"] is False for warning in warnings)
    assert warnings[0]["details"] == [
        {
            "tool": "next_page",
            "recommended_minimum_samples": 10,
            "samples": 8,
            "sample_rate": 0.08,
            "supervised_samples": 8,
            "supervised_sample_rate": 0.08,
            "sample_shortfall": 2,
            "supervised_sample_shortfall": 2,
        }
    ]
    assert warnings[1]["passed"] == 1
    assert warnings[1]["failed"] == 1
    assert warnings[1]["details"] == [
        {"task_id": "task-warning", "failed_checks": ["Is_hotels_correct"]}
    ]


def test_hard_acceptance_ignores_coverage_recommendation_and_commonsense() -> None:
    assert _hard_acceptance_passes(
        family_counts={"efficient_success": 100},
        masks_valid={"all_actions_supervised": True},
        rationale_checks={"all_tool_turns_have_visible_rationale": True},
        tool_coverage={
            "required_tools_present": True,
            "required_tools_supervised": True,
            "minimum_sample_coverage": False,
            "minimum_supervised_sample_coverage": False,
        },
        trajectory_policy={"checks": {"max_50_actions": True}},
        action_concentration={"checks": {"max_three_consecutive_tool_calls": True}},
        causal_grounding={"checks": {"all_id_actions_grounded": True}},
        reward={"reward_one": 100, "all_hard_pass": 100, "plan_submitted": 100},
        official={"schema_passes": 100, "commonsense_passes": 98},
        sample_count=100,
    )


def test_action_statistics_reject_repeated_actions_and_long_tool_bursts() -> None:
    repeated = {"tool": "get_route", "arguments": {"route": "same"}}
    trajectories = [
        {
            "steps": [
                {"action": repeated},
                {"action": repeated},
                {"action": repeated},
                {"action": repeated},
                {"action": {"tool": "submit_plan", "arguments": {}}},
            ]
        }
    ]

    statistics = trajectory_action_statistics(trajectories)

    assert statistics["distinct_tools"] == 2
    assert statistics["tools"]["get_route"]["max_consecutive"] == 4
    assert statistics["identical_action_repeats"] == 3
    assert statistics["checks"] == {
        "no_identical_action_repeats": False,
        "max_three_consecutive_tool_calls": False,
    }


def test_rollout_comparison_requires_same_tasks_and_reports_failures(tmp_path) -> None:
    tools = [{"function": {"name": "search_attractions"}}]
    successful = {
        "task_id": "same",
        "model": "teacher",
        "tools": tools,
        "success": True,
        "rft_accepted": True,
        "final_reward": 1.0,
        "termination_reason": "plan_submitted",
        "api_turn_count": 1,
        "usage": {},
        "steps": [
            {
                "action": {"tool": "search_attractions", "arguments": {"city": "北京"}},
                "valid_action": True,
            }
        ],
        "reward_detail": {"checks": []},
    }
    failed = {
        **successful,
        "model": "model",
        "success": False,
        "rft_accepted": False,
        "final_reward": -1.0,
        "termination_reason": "model_stopped_without_terminal_action",
        "steps": [
            {
                "action": {"tool": "search_attractions", "arguments": {}},
                "valid_action": False,
            }
        ],
        "reward_detail": {
            "checks": [{"id": "terminal_plan", "status": "fail"}]
        },
    }
    programmatic_path = tmp_path / "programmatic.jsonl"
    model_path = tmp_path / "model.jsonl"
    programmatic_path.write_text(json.dumps(successful) + "\n", encoding="utf-8")
    model_path.write_text(json.dumps(failed) + "\n", encoding="utf-8")

    report = compare_rollout_batches(programmatic_path, model_path)

    assert report["same_task_ids"] is True
    assert report["batches"]["programmatic"]["success_rate"] == 1.0
    assert report["batches"]["model"]["success_rate"] == 0.0
    assert report["batches"]["model"]["invalid_actions"]["total"] == 1
    assert report["batches"]["model"]["failed_reward_checks"] == {"terminal_plan": 1}
