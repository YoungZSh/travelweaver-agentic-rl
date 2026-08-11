from __future__ import annotations

import re
from collections import Counter

from travelweaver.sft.programmatic import (
    _assign_families,
    _candidate_review_rationale,
    _catalog_rationale,
    _loop_action_rationale,
    _loop_reflection,
    _nearby_page_rationale,
    _page_rationale,
    _route_rationale,
    _save_rationale,
    _search_rationale,
    _submit_reflection,
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
    assert Counter(first.values()) == {
        "efficient_success": 400,
        "loop_recovery": 50,
        "evidence_ready_submit": 50,
    }
    for task_type in {str(record["slot"]["task_type"]) for record in records}:
        family_counts = Counter(
            first[index]
            for index, record in enumerate(records)
            if record["slot"]["task_type"] == task_type
        )
        assert family_counts["loop_recovery"] in {12, 13}
        assert family_counts["evidence_ready_submit"] in {12, 13}


def test_visible_reflections_describe_the_actual_recovery_state() -> None:
    paging = _loop_reflection(
        1, "task-a", entity_name="西湖", target_visible=False
    )
    saving = _loop_reflection(
        1, "task-a", entity_name="西湖", target_visible=True
    )
    submitting = _submit_reflection(1, "task-a", ["往返交通", "景点", "完整路线"])

    assert "西湖" in paging and ("翻页" in paging or "后续" in paging or "下一页" in paging)
    assert "西湖" in saving and "保存" in saving
    assert "往返交通" in submitting and "提交" in submitting


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

    loop = _loop_action_rationale(
        1,
        "task-a",
        position=3,
        entity_name="西湖",
        city="杭州",
    )
    assert "西湖" in loop and "杭州" in loop
    assert "执行了" not in loop and "查询了" not in loop

    generic_loop = _loop_action_rationale(
        1,
        "task-generic",
        position=3,
        entity_name="合适的景点",
        city="杭州",
    )
    assert "合适的景点的" not in generic_loop
    assert "景点" in generic_loop and "杭州" in generic_loop


def test_contextual_templates_use_only_already_visible_decision_facts() -> None:
    comparison = _candidate_review_rationale(
        1,
        "task-a",
        position=4,
        review_kind="compare",
        candidate_count=6,
        candidate_purposes=("去程交通", "景点"),
        comparison_names=("西湖", "灵隐寺"),
    )
    coverage = _candidate_review_rationale(
        1,
        "task-b",
        position=5,
        review_kind="coverage",
        candidate_count=6,
        candidate_purposes=("去程交通", "返程交通", "景点"),
    )
    nearby_page = _nearby_page_rationale(
        1,
        "task-c",
        position=6,
        anchor_name="杭州站",
        radius_km=5,
        noun="景点",
        page_number=2,
    )
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

    assert "西湖" in comparison and "灵隐寺" in comparison and "清单" in comparison
    assert "6项" in coverage and "返程交通" in coverage
    assert "杭州站" in nearby_page and "5公里" in nearby_page and "第2页" in nearby_page
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
            collecting_group=grouped,
        )
        for position in range(24)
        for grouped in (False, True)
    }

    assert all("川菜餐厅的当前" not in value for value in values)
    assert all("同一重庆的" not in value for value in values)
    assert all("重庆的川菜餐厅" in value for value in values)
    assert any("重庆的川菜餐厅当前结果" in value for value in values)


def test_paged_and_looped_searches_cycle_visible_language() -> None:
    pages = [
        _page_rationale(
            1,
            "task-page-cycle",
            position=page,
            entity_name="合适的餐厅",
            public_scope="重庆的川菜餐厅",
            page_number=page,
            pages_checked=page - 1,
            collecting_group=False,
        )
        for page in range(1, 9)
    ]
    normalized_pages = {re.sub(r"[0-9]+", "N", page) for page in pages}
    loops = [
        _loop_action_rationale(
            1,
            "task-loop-cycle",
            position=attempt,
            entity_name="合适的餐厅",
            city="重庆",
            attempt=attempt,
        )
        for attempt in range(3)
    ]

    assert len(normalized_pages) == len(pages)
    assert len(set(loops)) == len(loops)
