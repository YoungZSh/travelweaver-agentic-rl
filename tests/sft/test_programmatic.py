from __future__ import annotations

import json
from collections import Counter

import pytest

from travelweaver.errors import SFTRebuildError
from travelweaver.sft.programmatic import (
    SAMPLE_FAMILIES,
    ProgrammaticBuildConfig,
    _assign_families,
    _build_one,
    _catalog_rationale,
    _catalog_requirement,
    _comparison_semantic_entity_types,
    _hard_unit_price_limit,
    _longest_tool_run,
    _natural_evidence_path_kinds,
    _opening_verification_capability,
    _page_rationale,
    _pagination_is_task_grounded,
    _required_activity_count,
    _route_rationale,
    _save_decision_facts,
    _save_rationale,
    _search_rationale,
    _submit_reflection,
    _task_grounded_search_action,
    _unfiltered_search_rationale,
    build_programmatic_trajectories,
    minimum_tool_coverage_samples,
)


def _records(count: int) -> list[dict[str, object]]:
    task_types = ("easy_like", "medium_like", "human_like", "preference_like")
    return [
        {
            "task_spec": {"task_id": f"task-{index:03d}"},
            "slot": {
                "task_type": task_types[index % len(task_types)],
                "days": index % 5 + 1,
                "scenario_profile": "normal" if index % 10 else "price_change",
            },
        }
        for index in range(count)
    ]


def test_family_assignment_is_exact_deterministic_and_stratified() -> None:
    records = _records(500)

    first = _assign_families(records, 20260821)
    second = _assign_families(records, 20260821)

    assert first == second
    assert Counter(first.values()) == {"efficient_success": 500}


def test_longest_tool_run_detects_route_bursts() -> None:
    assert _longest_tool_run(["search_attractions", "save_candidate", "get_route"]) == (
        "search_attractions",
        1,
    )
    assert _longest_tool_run(
        ["save_candidate", "get_route", "get_route", "get_route", "submit_plan"]
    ) == ("get_route", 3)


def test_evidence_paths_do_not_create_separate_sample_families() -> None:
    records = _records(10)
    coverage_plans = {
        0: {"profile": "verification"},
        1: {"profile": "nearby"},
        2: {"profile": "replacement"},
    }

    families = _assign_families(records, 20260821, coverage_plans=coverage_plans)

    assert minimum_tool_coverage_samples(500) == 50
    assert minimum_tool_coverage_samples(9) == 1
    assert minimum_tool_coverage_samples(101) == 11
    assert set(families.values()) <= set(SAMPLE_FAMILIES)
    assert set(families.values()) == {"efficient_success"}


def test_main_graph_selects_only_semantically_relevant_evidence_paths() -> None:
    record = {
        "blueprint": {
            "preferences": [
                {"kind": "near_poi"},
                {"kind": "lower_total_cost"},
            ]
        },
        "witness": {
            "evidence_bundle": {
                "entities": {
                    "opening": {
                        "entity_type": "attraction",
                        "name": "西湖",
                        "open_time": "09:00",
                    },
                    "comparison": {
                        "entity_type": "attraction",
                        "name": "候选景点",
                    },
                }
            }
        },
    }
    capabilities = {
        "opening_verification": {
            "candidate_id": "opening",
            "at_time": "11:00",
        },
        "nearby_discovery": {
            "anchor_id": "anchor",
            "candidate_id": "comparison",
            "radius_km": 2,
        },
        "candidate_comparison": {
            "candidate_id": "comparison",
            "alternative_id": "alternative",
        },
    }

    selected = _natural_evidence_path_kinds(
        record,
        {"query": "必须游览西湖"},
        capabilities,
    )

    assert selected == (
        "opening_verification",
        "nearby_discovery",
        "candidate_comparison",
    )


def test_price_comparison_semantics_include_hard_budget_scopes() -> None:
    record = {
        "blueprint": {"preferences": [{"kind": "lower_lodging_share"}]},
        "task_spec": {
            "constraints": [
                {"kind": "category_budget", "scope": "restaurant"},
                {"kind": "total_budget", "scope": "trip"},
            ]
        },
    }

    assert _comparison_semantic_entity_types(record) == {
        "attraction",
        "restaurant",
        "hotel",
    }


def test_lodging_cost_semantics_do_not_select_non_hotel_comparisons() -> None:
    record = {
        "blueprint": {"preferences": [{"kind": "lower_lodging_share"}]},
        "task_spec": {"constraints": []},
    }

    assert _comparison_semantic_entity_types(record) == {"hotel"}


def test_hard_category_budget_alone_does_not_invent_a_comparison_goal() -> None:
    record = {
        "blueprint": {"preferences": []},
        "task_spec": {
            "constraints": [
                {
                    "kind": "category_budget",
                    "scope": "accommodation",
                    "operator": "lte",
                    "value": {"amount": 470},
                }
            ]
        },
    }

    assert _comparison_semantic_entity_types(record) == set()
    assert _hard_unit_price_limit(record, "hotel") == 470


def test_opening_capability_prefers_a_named_scheduled_place() -> None:
    class FakeBackend:
        @staticmethod
        def check_place_open(candidate_id: str, at_time: str) -> dict[str, bool]:
            assert candidate_id in {"unnamed", "named"}
            assert at_time in {"10:00", "14:00"}
            return {"is_open": True}

    result = _opening_verification_capability(
        ["unnamed", "named"],
        {
            "unnamed": {"entity_type": "attraction", "name": "普通景点"},
            "named": {"entity_type": "restaurant", "name": "指定餐厅"},
        },
        {
            "unnamed": {"start_time": "10:00"},
            "named": {"start_time": "14:00"},
        },
        "这次必须去指定餐厅。",
        FakeBackend(),
    )

    assert result == {"candidate_id": "named", "at_time": "14:00"}


def test_catalog_lookup_requires_a_matching_task_constraint() -> None:
    record = {
        "task_spec": {
            "trip": {"travelers": 3},
            "constraints": [
                {
                    "kind": "entity_category",
                    "scope": "attraction",
                    "value": {"values": ["公园"]},
                },
                {
                    "kind": "room_type",
                    "scope": "accommodation",
                    "value": {"room_type": 2},
                },
            ]
        }
    }

    assert _catalog_requirement(record, {"category": "公园"}, "attraction") == "公园"
    assert _catalog_requirement(record, {"category": "博物馆"}, "attraction") is None
    assert (
        _catalog_requirement(record, {"room_type": 2}, "hotel")
        == "每间2个床位的房型"
    )

    any_of_record = {
        "task_spec": {
            "constraints": [
                {
                    "kind": "entity_category",
                    "scope": "attraction",
                    "value": {"any_of": [["公园"], ["博物馆"]]},
                }
            ]
        }
    }
    assert _catalog_requirement(
        any_of_record, {"category": "博物馆"}, "attraction"
    ) == "博物馆"


def test_search_filters_never_copy_an_unstated_witness_facet() -> None:
    unconstrained = {"task_spec": {"constraints": []}}
    attraction = {"city": "杭州", "category": "公园"}
    constrained = {
        "task_spec": {
            "constraints": [
                {
                    "kind": "entity_category",
                    "scope": "attraction",
                    "value": {"values": ["公园"]},
                }
            ]
        }
    }

    assert _task_grounded_search_action(unconstrained, attraction, "attraction") == {
        "tool": "search_attractions",
        "arguments": {"city": "杭州"},
    }
    assert _task_grounded_search_action(constrained, attraction, "attraction")[
        "arguments"
    ] == {"city": "杭州", "category": "公园"}
    rationale = _unfiltered_search_rationale(
        1,
        "task-a",
        position=0,
        city="杭州",
        noun="景点",
        entity_type="attraction",
    )
    assert "目录" not in rationale
    assert "公园" not in rationale


def test_lower_cost_preference_is_the_only_unnamed_pagination_objective() -> None:
    record = {
        "blueprint": {
            "preferences": [{"kind": "lower_total_cost", "direction": "minimize"}]
        },
        "task_spec": {"constraints": []},
    }

    assert _task_grounded_search_action(
        record,
        {"city": "南京", "entity_type": "hotel"},
        "hotel",
    ) == {
        "tool": "search_hotels",
        "arguments": {"city": "南京", "sort_by": "price"},
    }


def test_averaged_category_budget_is_not_misused_as_a_per_item_search_filter() -> None:
    record = {
        "task_spec": {
            "constraints": [
                {
                    "kind": "category_budget",
                    "scope": "accommodation",
                    "operator": "lte",
                    "value": {"amount": 470},
                }
            ]
        }
    }

    assert _task_grounded_search_action(
        record,
        {"city": "上海", "entity_type": "hotel"},
        "hotel",
    ) == {
        "tool": "search_hotels",
        "arguments": {"city": "上海"},
    }
    restaurant_record = {
        "task_spec": {
            "constraints": [
                {
                    "kind": "category_budget",
                    "scope": "restaurant",
                    "operator": "lte",
                    "value": {"amount": 410},
                }
            ]
        }
    }
    assert _task_grounded_search_action(
        restaurant_record,
        {"city": "重庆", "entity_type": "restaurant"},
        "restaurant",
    ) == {
        "tool": "search_restaurants",
        "arguments": {"city": "重庆"},
    }


def test_pagination_uses_the_global_consecutive_tool_limit() -> None:
    assert _pagination_is_task_grounded(
        name_grounded=True,
    )
    assert not _pagination_is_task_grounded(
        name_grounded=False,
        search_tool="search_nearby",
    )
    assert not _pagination_is_task_grounded(
        name_grounded=False,
    )
    assert _pagination_is_task_grounded(
        name_grounded=False,
        planned_candidate_count=3,
        visible_planned_count=1,
    )
    assert not _pagination_is_task_grounded(
        name_grounded=False,
        planned_candidate_count=3,
        visible_planned_count=0,
    )
    assert _pagination_is_task_grounded(
        name_grounded=False,
        required_candidate_count=2,
        resolved_candidate_count=1,
    )
    assert not _pagination_is_task_grounded(
        name_grounded=False,
        required_candidate_count=2,
        resolved_candidate_count=0,
    )


def test_activity_count_is_a_public_pagination_predicate() -> None:
    record = {
        "task_spec": {
            "constraints": [
                {
                    "kind": "activity_count",
                    "scope": "attraction",
                    "operator": "eq",
                    "value": {"activity_type": "attraction", "count": 2},
                }
            ]
        }
    }

    assert _required_activity_count(record, "attraction") == 2
    assert _required_activity_count(record, "restaurant") == 0
    rationale = _page_rationale(
        1,
        "task-a",
        position=0,
        entity_name="景点",
        public_scope="北京的景点",
        page_number=2,
        pages_checked=1,
        required_candidate_count=2,
        resolved_candidate_count=1,
    )
    assert "要求安排2个不同的景点" in rationale
    assert "还缺1个" in rationale


def test_programmatic_teacher_rejects_records_without_the_50_step_policy() -> None:
    with pytest.raises(SFTRebuildError, match="trajectory_policy"):
        _build_one(
            {},
            {},
            {},
            object(),
            family="efficient_success",
            seed=1,
            source_question_batch="test",
        )


def test_programmatic_build_persists_each_slot_and_resumes(tmp_path, monkeypatch) -> None:
    task_dir = tmp_path / "tasks"
    records_dir = task_dir / "records"
    records_dir.mkdir(parents=True)
    public_rows = []
    oracle_rows = []
    for index in range(3):
        task_id = f"task-{index}"
        public_rows.append({"uid": task_id, "query": f"问题{index}"})
        oracle_rows.append({"uid": task_id})
        (records_dir / f"{index:06d}.json").write_text(
            json.dumps(
                {
                    "task_spec": {"task_id": task_id},
                    "blueprint": {"blueprint_id": f"blueprint-{index}", "preferences": []},
                    "surface": {"surface_id": f"surface-{index}"},
                    "scenario": {"scenario_id": f"scenario-{index}"},
                }
            ),
            encoding="utf-8",
        )
    for path, rows in (
        (task_dir / "tasks.public.jsonl", public_rows),
        (task_dir / "tasks.oracle.jsonl", oracle_rows),
    ):
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )

    capability_calls = []
    monkeypatch.setattr(
        "travelweaver.sft.programmatic._evidence_path_capabilities",
        lambda record, public, backend: capability_calls.append(public["uid"]) or {},
    )
    first_build_calls = []

    def fake_build(record, public, oracle, backend, **kwargs):
        del record, oracle, backend, kwargs
        first_build_calls.append(public["uid"])
        if public["uid"] == "task-1":
            raise SFTRebuildError("simulated slot failure")
        return _fake_programmatic_bundle(public["uid"])

    monkeypatch.setattr("travelweaver.sft.programmatic._build_one", fake_build)
    config = ProgrammaticBuildConfig(
        task_dir=task_dir,
        output_path=tmp_path / "trajectories.jsonl",
        audit_path=tmp_path / "audits.jsonl",
        work_dir=tmp_path / "work",
        seed=7,
        concurrency=1,
    )
    with pytest.raises(SFTRebuildError, match="simulated slot failure"):
        build_programmatic_trajectories(config, base_backend=object())

    assert capability_calls == ["task-0", "task-1", "task-2"]
    assert first_build_calls == ["task-0", "task-1", "task-2"]
    assert {
        int(path.stem) for path in (config.resolved_work_dir / "records").glob("*.json")
    } == {0, 2}
    assert not config.output_path.exists()

    resumed_build_calls = []

    def resumed_build(record, public, oracle, backend, **kwargs):
        del record, oracle, backend, kwargs
        resumed_build_calls.append(public["uid"])
        return _fake_programmatic_bundle(public["uid"])

    monkeypatch.setattr("travelweaver.sft.programmatic._build_one", resumed_build)
    report = build_programmatic_trajectories(config, base_backend=object())

    assert capability_calls == ["task-0", "task-1", "task-2"]
    assert resumed_build_calls == ["task-1"]
    assert report["samples"] == 3
    assert report["all_reward_one"] is True
    assert len(config.output_path.read_text(encoding="utf-8").splitlines()) == 3
    manifest = json.loads(
        (config.resolved_work_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "complete"
    assert manifest["capabilities_completed"] == 3
    assert manifest["trajectories_completed"] == 3


def _fake_programmatic_bundle(task_id: str) -> tuple[dict, dict]:
    trajectory = {
        "task_id": task_id,
        "tools": [{"type": "function", "function": {"name": "submit_plan"}}],
        "steps": [{"action": {"tool": "submit_plan"}}],
    }
    audit = {
        "task_id": task_id,
        "sample_family": "efficient_success",
        "evidence_paths": [],
        "turns": [{"tool": "submit_plan"}],
        "replay_reward": 1.0,
        "all_hard_pass": True,
    }
    return trajectory, audit


def test_programmatic_react_templates_explain_each_tool_decision() -> None:
    search = _search_rationale(
        1,
        "task-a",
        position=0,
        entity={"entity_type": "attraction", "city": "杭州", "name": "西湖"},
        entity_type="attraction",
    )
    save = _save_rationale(
        1,
        "task-a",
        position=1,
        entity_name="西湖",
        purpose="attraction",
    )
    route = _route_rationale(
        1,
        "task-a",
        position=2,
        origin_name="杭州站",
        destination_name="西湖",
        mode="metro",
        start_time="10:00",
    )

    assert "杭州" in search and "西湖" in search
    assert "查询" in search or "检索" in search
    assert "西湖" in save and "保存" in save
    assert "杭州站" in route and "西湖" in route and "路线" in route and "10:00" in route


def test_save_rationale_cites_visible_hard_constraint_facts() -> None:
    record = {
        "task_spec": {
            "trip": {"travelers": 3},
            "constraints": [
                {
                    "kind": "category_budget",
                    "scope": "accommodation",
                    "operator": "lte",
                    "value": {"amount": 470},
                }
            ]
        }
    }
    facts = _save_decision_facts(
        record,
        {"query": "住宿人均每晚不超过470元"},
        {
            "entity_type": "hotel",
            "name": "测试酒店",
            "price": 417,
            "room_type": 1,
        },
        "hotel",
        "hotel",
    )
    rationale = _save_rationale(
        1,
        "task-hotel",
        position=1,
        entity_name="测试酒店",
        purpose="hotel",
        decision_facts=facts,
    )

    assert "417元" in rationale
    assert "470元" in rationale
    assert "硬上限" in rationale
    assert "住宿" in rationale and "保存" in rationale


def test_contextual_templates_use_only_already_visible_decision_facts() -> None:
    submit = _submit_reflection(
        1,
        "task-d",
        ["往返交通", "景点", "完整路线"],
        candidate_count=6,
        days=2,
        route_count=3,
        evidence_landmarks=("去程G1", "景点西湖", "返程G2"),
    )
    catalog = _catalog_rationale(
        1,
        "task-e",
        position=7,
        city="杭州",
        label="景点类别",
        tool="list_attraction_categories",
        candidate_count=1,
        candidate_purposes=("去程交通",),
    )

    assert "2天" in submit and "6项候选" in submit and "3段" in submit
    assert "G1" in submit or "西湖" in submit or "G2" in submit
    assert "杭州" in catalog and "景点类别" in catalog


def test_paging_scope_does_not_duplicate_the_linking_particle() -> None:
    values = {
        _page_rationale(
            1,
            f"task-{position}",
            position=position,
            entity_name="合适的餐厅",
            public_scope="重庆的川菜餐厅",
            page_number=2,
            pages_checked=1,
            planned_candidate_count=3,
            visible_planned_count=1,
        )
        for position in range(24)
    }

    assert all("川菜餐厅的当前" not in value for value in values)
    assert all("同一重庆的" not in value for value in values)
    assert all("重庆的川菜餐厅" in value for value in values)
    assert all("尚缺2个" in value for value in values)
    assert all("当前公开筛选条件" in value for value in values)
