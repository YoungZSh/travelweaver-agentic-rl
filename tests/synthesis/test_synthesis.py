from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from travelweaver.env import InMemoryBackend, ScenarioBackend, ScenarioEffect, ScenarioSpec
from travelweaver.errors import SynthesisError
from travelweaver.llm import DeepSeekConfig
from travelweaver.synthesis.artifacts import ArtifactStore, _alignment
from travelweaver.synthesis.catalog import build_pilot_slots
from travelweaver.synthesis.compose import _constraint
from travelweaver.synthesis.pipeline import (
    SynthesisConfig,
    SynthesisPipeline,
    _load_task_exclusions,
    _normalize_question,
    _preference_metric,
)
from travelweaver.synthesis.polisher import TaskPolisher, validate_surface
from travelweaver.synthesis.render import render_canonical
from travelweaver.synthesis.trajectory_policy import (
    MAX_CONSECUTIVE_TOOL_CALLS,
    MAX_PROGRAMMATIC_CATALOG_ACTIONS,
    MAX_SYNTHESIS_VALID_STEPS,
    MAX_WITNESS_VALID_STEPS,
    TRAJECTORY_POLICY_VERSION,
    trajectory_policy,
)
from travelweaver.synthesis.witness import (
    WitnessBuilder,
    _attraction_cluster_radius,
    _local_route_mode,
    _LocalActivity,
)
from travelweaver.tasks import (
    BlueprintConstraint,
    BlueprintPreference,
    TaskBlueprint,
    TaskSurface,
    TripSpec,
    materialize_task_spec,
)


def _blueprint() -> TaskBlueprint:
    return TaskBlueprint(
        trip=TripSpec(origin="上海", destinations=("北京",), days=2, travelers=2),
        constraints=(
            BlueprintConstraint(
                id="c001",
                kind="transport_mode",
                operator="eq",
                value={"modes": ["train"], "leg": "outbound"},
                scope="intercity_transport",
            ),
            BlueprintConstraint(
                id="c002",
                kind="total_budget",
                operator="lte",
                value={"amount": 3000},
                scope="trip",
            ),
        ),
        world_snapshot_version="snapshot-v1",
        generator_version="generator-v1",
        generation_seed=7,
    )


def _payload(blueprint: TaskBlueprint) -> dict:
    canonical = render_canonical(blueprint)
    return {
        "query": canonical.query,
        "mentions": [
            {"constraint_id": constraint_id, "text": text}
            for constraint_id, text in canonical.clauses.items()
        ],
        "preference_mentions": [
            {"preference_id": preference_id, "text": text}
            for preference_id, text in canonical.preference_clauses.items()
        ],
    }


def test_walking_witness_uses_taxi_only_for_terminal_transfers() -> None:
    airport = {"entity_type": "route_anchor"}
    attraction = {"entity_type": "attraction"}
    restaurant = {"entity_type": "restaurant"}

    assert _local_route_mode("walk", airport, attraction) == "taxi"
    assert _local_route_mode("walk", attraction, restaurant) == "walk"
    assert _local_route_mode("walk", restaurant, airport) == "taxi"
    assert _local_route_mode("metro", airport, attraction) == "metro"
    assert WitnessBuilder._route_mode_order("walk") == ("walk",)
    assert WitnessBuilder._route_mode_order("metro") == ("metro", "taxi")
    assert (
        _attraction_cluster_radius(
            "walk",
            attraction_count=1,
            needs_restaurant=False,
            has_hotel=False,
        )
        == 15.0
    )
    assert (
        _attraction_cluster_radius(
            "walk",
            attraction_count=1,
            needs_restaurant=True,
            has_hotel=False,
        )
        == 1.0
    )


def test_walking_hotel_selection_requires_enough_nearby_attractions() -> None:
    sparse = {
        "place_id": "hotel-sparse",
        "entity_type": "hotel",
        "city": "测试城",
        "name": "稀疏酒店",
        "price": 100,
        "room_type": 1,
        "latitude": 31.0,
        "longitude": 121.0,
    }
    dense = {
        **sparse,
        "place_id": "hotel-dense",
        "name": "密集酒店",
        "latitude": 30.0,
        "longitude": 120.0,
    }
    attractions = [
        {
            "place_id": f"attraction-{index}",
            "entity_type": "attraction",
            "city": "测试城",
            "price": 0,
            "open_time": "08:00",
            "close_time": "20:00",
            "latitude": 30.0 + index / 10000,
            "longitude": 120.0 + index / 10000,
        }
        for index in range(3)
    ]

    class FakeBackend:
        @staticmethod
        def _records(kind: str, city: str):
            assert city == "测试城"
            return [sparse, dense] if kind == "hotel" else attractions

        @staticmethod
        def search_hotels(**arguments):
            assert arguments == {"city": "测试城"}
            return [sparse, dense]

        @staticmethod
        def search_attractions(**arguments):
            assert arguments == {"city": "测试城"}
            return attractions

    slot = replace(
        build_pilot_slots(1, 20260812)[0],
        destination="测试城",
        days=3,
        attractions_per_day=1,
        route_mode="walk",
        recipe=(),
    )

    selected = WitnessBuilder(FakeBackend(), seed=7)._select_hotel(  # type: ignore[arg-type]
        slot, "walk"
    )

    assert selected["place_id"] == "hotel-dense"


def test_question_normalization_ignores_only_surface_typography() -> None:
    assert _normalize_question("请安排 上海→杭州，2 天！") == _normalize_question(
        "请安排上海杭州2天"
    )
    assert _normalize_question("上海到杭州2天") != _normalize_question("上海到杭州3天")


def test_task_exclusions_load_complete_synthesis_records(tmp_path) -> None:
    task_dir = tmp_path / "batch"
    records_dir = task_dir / "records"
    records_dir.mkdir(parents=True)
    (task_dir / "manifest.json").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8"
    )
    (records_dir / "000.json").write_text(
        json.dumps(
            {
                "task_spec": {"task_id": "task-1"},
                "blueprint": {"blueprint_id": "blueprint-1"},
                "surface": {
                    "surface_id": "surface-1",
                    "public_query": "请安排上海到杭州两天。",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    exclusions = _load_task_exclusions((task_dir,))

    assert exclusions["task_ids"] == {"task-1"}
    assert exclusions["blueprint_ids"] == {"blueprint-1"}
    assert exclusions["surface_ids"] == {"surface-1"}
    assert exclusions["normalized_queries"] == {"请安排上海到杭州两天"}
    assert len(exclusions["sha256"]) == 64


def test_synthesis_defaults_to_256_llm_workers(tmp_path) -> None:
    assert SynthesisConfig(output_dir=tmp_path).llm_concurrency == 256


def test_short_trajectory_policy_uses_one_global_consecutive_limit() -> None:
    assert trajectory_policy() == {
        "policy_version": TRAJECTORY_POLICY_VERSION,
        "max_valid_steps": 50,
        "max_consecutive_tool_calls": 3,
    }


def test_witness_step_reserve_covers_the_clean_teacher_overhead() -> None:
    assert (
        MAX_WITNESS_VALID_STEPS
        + MAX_PROGRAMMATIC_CATALOG_ACTIONS
        + MAX_CONSECUTIVE_TOOL_CALLS
        == MAX_SYNTHESIS_VALID_STEPS
    )


@pytest.mark.parametrize("attraction_targets", [(1,), (2,)])
def test_witness_prefers_attractions_on_first_unfiltered_page(
    attraction_targets: tuple[int, ...],
) -> None:
    records = [
        {
            "place_id": f"attraction-{index:02d}",
            "entity_type": "attraction",
            "city": "测试城",
            "name": f"景点{index:02d}",
            "category": "公园",
            "price": 10.0,
            "open_time": "08:00",
            "close_time": "20:00",
            "latitude": 30.0 + index / 10000,
            "longitude": 120.0 + index / 10000,
        }
        for index in range(30)
    ]

    class FakeBackend:
        @staticmethod
        def _records(kind: str, city: str) -> list[dict[str, object]]:
            assert kind == "attraction" and city == "测试城"
            return records

        @staticmethod
        def search_attractions(**arguments: object) -> list[dict[str, object]]:
            assert arguments == {"city": "测试城"}
            return records

    slot = replace(
        build_pilot_slots(1, 20260812)[0],
        destination="测试城",
        days=1,
        attractions_per_day=1,
        include_meal=False,
        route_mode="taxi",
        recipe=(),
    )
    builder = WitnessBuilder(FakeBackend(), seed=7)  # type: ignore[arg-type]
    selected = builder._select_attractions(
        slot,
        outbound_arrival="08:00",
        return_departure="20:00",
        needs_restaurant=False,
        hotel=None,
        route_mode="taxi",
        attraction_targets=attraction_targets,
    )

    pages = [builder._public_page_index(item) for item in selected]
    assert len(pages) == sum(attraction_targets)
    assert pages == [0] * sum(attraction_targets)


def test_witness_uses_bounded_later_page_to_fill_public_attraction_count() -> None:
    records = [
        {
            "place_id": f"attraction-{index:02d}",
            "entity_type": "attraction",
            "city": "测试城",
            "name": f"景点{index:02d}",
            "category": "公园",
            "price": 10.0,
            "open_time": "08:00",
            "close_time": "20:00",
            "latitude": 30.0 + index / 10000,
            "longitude": 120.0 + index / 10000,
        }
        for index in range(50)
    ]

    class FakeBackend:
        @staticmethod
        def _records(kind: str, city: str) -> list[dict[str, object]]:
            assert kind == "attraction" and city == "测试城"
            return records

        @staticmethod
        def search_attractions(**arguments: object) -> list[dict[str, object]]:
            assert arguments == {"city": "测试城"}
            return records

        @staticmethod
        def search_nearby(**arguments: object) -> list[dict[str, object]]:
            del arguments
            return []

    slot = replace(
        build_pilot_slots(1, 20260812)[0],
        destination="测试城",
        days=1,
        attractions_per_day=2,
        include_meal=False,
        route_mode="taxi",
        recipe=("attraction_count",),
    )
    builder = WitnessBuilder(FakeBackend(), seed=7)  # type: ignore[arg-type]
    selected = builder._select_attractions(
        slot,
        outbound_arrival="08:00",
        return_departure="20:00",
        needs_restaurant=False,
        hotel=None,
        route_mode="taxi",
        attraction_targets=(2,),
    )

    pages = [builder._public_page_index(item) for item in selected]
    assert pages[0] == 0
    assert pages[1] in {1, 2, 3}


def test_witness_beyond_three_continuations_allows_nearby_discovery() -> None:
    records = [
        {
            "place_id": f"attraction-{index:02d}",
            "entity_type": "attraction",
            "city": "测试城",
            "name": f"景点{index:02d}",
        }
        for index in range(42)
    ]

    class FakeBackend:
        @staticmethod
        def search_attractions(**arguments: object) -> list[dict[str, object]]:
            assert arguments == {"city": "测试城"}
            return records

        @staticmethod
        def search_nearby(**arguments: object) -> list[dict[str, object]]:
            assert arguments["place_id"] == "anchor-1"
            assert arguments["category"] == "attraction"
            assert arguments["top_k"] == 10
            return [records[40]]

    activity = _LocalActivity(
        evidence=records[40],
        activity_type="attraction",
        start_time="10:00",
        end_time="11:00",
        route={"origin_place_id": "anchor-1"},
    )
    WitnessBuilder(FakeBackend(), seed=7)._require_grounded_local_discovery(  # type: ignore[arg-type]
        [[activity]], initial_anchor_ids=("anchor-1",)
    )


def test_witness_beyond_three_continuations_rejects_hidden_target() -> None:
    records = [
        {
            "place_id": f"attraction-{index:02d}",
            "entity_type": "attraction",
            "city": "测试城",
            "name": f"景点{index:02d}",
        }
        for index in range(42)
    ]

    class FakeBackend:
        @staticmethod
        def search_attractions(**arguments: object) -> list[dict[str, object]]:
            assert arguments == {"city": "测试城"}
            return records

        @staticmethod
        def search_nearby(**arguments: object) -> list[dict[str, object]]:
            del arguments
            return records[:10]

    activity = _LocalActivity(
        evidence=records[40],
        activity_type="attraction",
        start_time="10:00",
        end_time="11:00",
        route={"origin_place_id": "anchor-1"},
    )
    with pytest.raises(SynthesisError, match="first nearby page"):
        WitnessBuilder(FakeBackend(), seed=7)._require_grounded_local_discovery(  # type: ignore[arg-type]
            [[activity]], initial_anchor_ids=("anchor-1",)
        )


def test_candidate_beyond_three_continuations_is_rejected_before_commit() -> None:
    records = [
        {
            "place_id": f"attraction-{index:02d}",
            "entity_type": "attraction",
            "city": "测试城",
            "name": f"景点{index:02d}",
        }
        for index in range(42)
    ]

    class FakeBackend:
        @staticmethod
        def search_attractions(**arguments: object) -> list[dict[str, object]]:
            assert arguments == {"city": "测试城"}
            return records

        @staticmethod
        def search_nearby(**arguments: object) -> list[dict[str, object]]:
            del arguments
            return records[:10]

    builder = WitnessBuilder(FakeBackend(), seed=7)  # type: ignore[arg-type]
    assert not builder._local_candidate_is_grounded(
        records[40],
        recipe=(),
        preference_kinds=(),
        required_facets=None,
        established_anchor_ids=("anchor-1",),
    )


def test_nearby_visibility_reuses_the_same_public_page_for_multiple_candidates() -> None:
    records = [
        {
            "place_id": f"attraction-{index:02d}",
            "entity_type": "attraction",
            "city": "测试城",
            "name": f"景点{index:02d}",
        }
        for index in range(42)
    ]
    calls = 0

    class FakeBackend:
        @staticmethod
        def search_attractions(**arguments: object) -> list[dict[str, object]]:
            assert arguments == {"city": "测试城"}
            return records

        @staticmethod
        def search_nearby(**arguments: object) -> list[dict[str, object]]:
            nonlocal calls
            calls += 1
            assert arguments["place_id"] == "anchor-1"
            return records[:10]

    builder = WitnessBuilder(FakeBackend(), seed=7)  # type: ignore[arg-type]
    for candidate in records[40:42]:
        assert not builder._local_candidate_is_grounded(
            candidate,
            recipe=(),
            preference_kinds=(),
            required_facets=None,
            established_anchor_ids=("anchor-1",),
        )

    assert calls == 5


def test_unnamed_later_page_candidate_is_not_directly_grounded() -> None:
    records = [
        {
            "place_id": f"attraction-{index:02d}",
            "entity_type": "attraction",
            "city": "测试城",
            "name": f"景点{index:02d}",
        }
        for index in range(40)
    ]

    class FakeBackend:
        @staticmethod
        def search_attractions(**arguments: object) -> list[dict[str, object]]:
            assert arguments == {"city": "测试城"}
            return records

    builder = WitnessBuilder(FakeBackend(), seed=7)  # type: ignore[arg-type]
    assert not builder._local_candidate_is_grounded(
        records[30],
        recipe=(),
        preference_kinds=(),
        required_facets=None,
        established_anchor_ids=(),
    )
    assert builder._local_candidate_is_grounded(
        records[0],
        recipe=(),
        preference_kinds=(),
        required_facets=None,
        established_anchor_ids=(),
    )


def test_partial_activity_count_grounds_up_to_three_cursor_steps() -> None:
    records = [
        {
            "place_id": f"attraction-{index:02d}",
            "entity_type": "attraction",
            "city": "测试城",
            "name": f"景点{index:02d}",
        }
        for index in range(42)
    ]

    class FakeBackend:
        @staticmethod
        def search_attractions(**arguments: object) -> list[dict[str, object]]:
            assert arguments == {"city": "测试城"}
            return records

        @staticmethod
        def search_nearby(**arguments: object) -> list[dict[str, object]]:
            del arguments
            return []

    builder = WitnessBuilder(FakeBackend(), seed=7)  # type: ignore[arg-type]
    assert builder._local_candidate_is_grounded(
        records[30],
        recipe=("attraction_count",),
        preference_kinds=(),
        required_facets=None,
        established_anchor_ids=(),
        allow_named_query=False,
        resolved_candidate_count=1,
    )
    assert not builder._local_candidate_is_grounded(
        records[40],
        recipe=("attraction_count",),
        preference_kinds=(),
        required_facets=None,
        established_anchor_ids=(),
        allow_named_query=False,
        resolved_candidate_count=1,
    )


def test_candidate_beyond_three_continuations_can_use_grounded_nearby() -> None:
    records = [
        {
            "place_id": f"attraction-{index:02d}",
            "entity_type": "attraction",
            "city": "测试城",
            "name": f"景点{index:02d}",
        }
        for index in range(42)
    ]

    class FakeBackend:
        @staticmethod
        def search_attractions(**arguments: object) -> list[dict[str, object]]:
            assert arguments == {"city": "测试城"}
            return records

        @staticmethod
        def search_nearby(**arguments: object) -> list[dict[str, object]]:
            assert arguments["place_id"] == "anchor-1"
            return [records[40]]

    builder = WitnessBuilder(FakeBackend(), seed=7)  # type: ignore[arg-type]
    assert builder._local_candidate_is_grounded(
        records[40],
        recipe=(),
        preference_kinds=(),
        required_facets=None,
        established_anchor_ids=("anchor-1",),
    )


def test_task_page_index_does_not_apply_one_meal_cuisine_to_other_meals() -> None:
    target = {
        "place_id": "restaurant-western",
        "entity_type": "restaurant",
        "city": "测试城",
        "name": "西餐厅",
        "cuisine": "西餐",
    }

    class FakeBackend:
        @staticmethod
        def search_restaurants(**arguments: object) -> list[dict[str, object]]:
            assert arguments == {"city": "测试城"}
            return [target]

    builder = WitnessBuilder(FakeBackend(), seed=7)  # type: ignore[arg-type]

    assert builder._task_page_index(
        target,
        ("restaurant_cuisine",),
        required_facets={"restaurant": "北京菜"},
    ) == 0


def test_named_attraction_uses_the_question_grounded_name_search() -> None:
    target = {
        "place_id": "named-attraction",
        "entity_type": "attraction",
        "city": "测试城",
        "name": "题面指定景点",
    }

    class FakeBackend:
        @staticmethod
        def search_attractions(**arguments: object) -> list[dict[str, object]]:
            assert arguments == {"city": "测试城", "query": "题面指定景点"}
            return [target]

    builder = WitnessBuilder(FakeBackend(), seed=7)  # type: ignore[arg-type]

    assert builder._task_page_index(target, ("include_attraction",)) == 0


def test_additional_attraction_does_not_inherit_named_search() -> None:
    target = {
        "place_id": "additional-attraction",
        "entity_type": "attraction",
        "city": "测试城",
        "name": "未被题面点名的景点",
    }
    records = [
        {
            "place_id": f"other-{index}",
            "entity_type": "attraction",
            "city": "测试城",
            "name": f"其他景点{index}",
        }
        for index in range(10)
    ] + [target]

    class FakeBackend:
        @staticmethod
        def search_attractions(**arguments: object) -> list[dict[str, object]]:
            assert arguments == {"city": "测试城"}
            return records

    builder = WitnessBuilder(FakeBackend(), seed=7)  # type: ignore[arg-type]

    assert builder._task_page_index(
        target,
        ("include_attraction", "attraction_count"),
        allow_named_query=False,
    ) == 1


def test_witness_refill_tries_each_original_origin_before_destination_replacements(
    monkeypatch,
) -> None:
    pipeline = object.__new__(SynthesisPipeline)
    pipeline.config = SimpleNamespace(seed=20260813)
    pipeline.backend = SimpleNamespace(
        supported_cities=("武汉", "广州", "北京", "上海")
    )
    monkeypatch.setattr(pipeline, "_origins", lambda slot: ("武汉", "广州"))
    slot = build_pilot_slots(1, 20260813, "chinatravel_blended_v1_1")[0]

    origins = pipeline._candidate_origins(slot)
    attempts = pipeline._candidate_slot_origins(slot)

    assert origins == ("武汉", "广州")
    assert attempts[:2] == ((slot, "武汉"), (slot, "广州"))
    assert all(candidate.destination != slot.destination for candidate, _ in attempts[2:])
    assert len(attempts) <= 24


def test_witness_candidate_deadline_moves_to_next_deterministic_attempt(
    monkeypatch,
) -> None:
    pipeline = object.__new__(SynthesisPipeline)
    pipeline.config = SimpleNamespace(seed=20260813, profile="chinatravel_blended_v1_1")
    slot = build_pilot_slots(1, 20260813, "chinatravel_blended_v1_1")[0]
    monkeypatch.setattr(
        pipeline,
        "_candidate_slot_origins",
        lambda _slot: ((slot, "武汉"), (slot, "广州")),
    )
    monkeypatch.setattr(
        "travelweaver.synthesis.pipeline.MAX_WITNESS_CANDIDATE_SECONDS",
        0.01,
    )

    def prepare_candidate(
        _slot,
        *,
        uid,
        origin,
        candidate_attempt,
    ):
        del uid, candidate_attempt
        if origin == "武汉":
            time.sleep(1)
        return SimpleNamespace(blueprint=SimpleNamespace(blueprint_id="unique"))

    monkeypatch.setattr(pipeline, "_prepare_candidate", prepare_candidate)

    candidate, quarantine = pipeline._prepare_slot(slot, frozenset())

    assert candidate.blueprint.blueprint_id == "unique"
    assert len(quarantine) == 1
    assert "exceeded 0.01s CPU deadline" in quarantine[0]["stage_error"]


def test_origin_fallbacks_try_catalog_origin_without_round_trip_prescan() -> None:
    class FakeBackend:
        supported_cities = ("北京", "上海", "杭州")

        @staticmethod
        def search_intercity_transport(**arguments):
            raise AssertionError(f"origin ordering must not query transport: {arguments}")

    pipeline = object.__new__(SynthesisPipeline)
    pipeline.backend = FakeBackend()
    pipeline.config = SimpleNamespace(seed=20260813)
    slot = replace(
        build_pilot_slots(1, 20260813, "chinatravel_blended_v1_1")[0],
        origin="北京",
        destination="上海",
    )

    origins = pipeline._origins(slot)

    assert origins[0] == "北京"
    assert set(origins) == {"北京", "杭州"}


def test_artifact_store_migrates_only_missing_llm_concurrency(tmp_path) -> None:
    output_dir = tmp_path / "batch"
    output_dir.mkdir()
    old_config = {"count": 500, "seed": 17}
    (output_dir / "manifest.json").write_text(
        json.dumps({"status": "in_progress", "config": old_config}),
        encoding="utf-8",
    )

    store = ArtifactStore(output_dir, {**old_config, "llm_concurrency": 256})

    assert store.manifest["config"]["llm_concurrency"] == 256


def test_artifact_store_persists_slot_progress_immediately(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "batch", {"count": 2, "seed": 17})
    store.save_record(1000, {"value": "complete"}, api_calls=0)
    store.record_progress(
        {
            "event": "slot_completed",
            "slot_index": 1000,
            "requested": 2,
            "completed": 1,
        }
    )

    assert (tmp_path / "batch" / "records" / "001000.json").is_file()
    manifest = json.loads(
        (tmp_path / "batch" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["completed"] == 1
    assert manifest["last_event"]["event"] == "slot_completed"
    events = [
        json.loads(line)
        for line in (tmp_path / "batch" / "progress.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert events[0]["slot_index"] == 1000


def test_synthesis_resumes_after_the_last_persisted_slot(tmp_path, monkeypatch) -> None:
    slots = build_pilot_slots(2, 20260807)
    monkeypatch.setattr(
        "travelweaver.synthesis.pipeline.build_pilot_slots",
        lambda count, seed, profile: slots,
    )
    monkeypatch.setattr(ArtifactStore, "finalize", lambda self, slots, api_calls: {})
    config = SynthesisConfig(
        output_dir=tmp_path / "streaming-batch",
        count=2,
        seed=20260807,
        canonical_only=True,
    )
    llm_config = DeepSeekConfig(
        api_key="offline-canonical", model="deterministic-canonical"
    )

    def prepared(slot):
        return SimpleNamespace(
            slot=slot,
            uid=f"task-{slot.index}",
            candidate_attempt=1,
            blueprint=SimpleNamespace(blueprint_id=f"blueprint-{slot.index}"),
        )

    def record(candidate):
        index = candidate.slot.index
        return {
            "slot": asdict(candidate.slot),
            "blueprint": {"blueprint_id": f"blueprint-{index}"},
            "surface": {
                "surface_id": f"surface-{index}",
                "public_query": f"第{index}道恢复测试题",
            },
            "task_spec": {"task_id": f"task-{index}"},
            "witness": {},
        }

    first_events = []
    first = SynthesisPipeline(
        config,
        llm_config,
        backend=object(),  # type: ignore[arg-type]
        progress=first_events.append,
    )
    monkeypatch.setattr(
        first,
        "_prepare_slot",
        lambda slot, blocked: (prepared(slot), []),
    )

    def fail_second(candidate):
        if candidate.slot.index == 1:
            raise SynthesisError("simulated interruption")
        return record(candidate)

    monkeypatch.setattr(first, "_materialize_candidate", fail_second)
    with pytest.raises(SynthesisError, match="simulated interruption"):
        first.run()

    assert {
        int(path.stem) for path in (config.output_dir / "records").glob("*.json")
    } == {0}
    assert any(event["event"] == "slot_completed" for event in first_events)

    resumed_events = []
    prepared_on_resume = []
    resumed = SynthesisPipeline(
        config,
        llm_config,
        backend=object(),  # type: ignore[arg-type]
        progress=resumed_events.append,
    )

    def prepare_remaining(slot, blocked):
        del blocked
        prepared_on_resume.append(slot.index)
        return prepared(slot), []

    monkeypatch.setattr(resumed, "_prepare_slot", prepare_remaining)
    monkeypatch.setattr(resumed, "_materialize_candidate", record)
    report = resumed.run()

    assert prepared_on_resume == [1]
    assert report.completed == 2
    assert resumed_events[0]["event"] == "synthesis_resumed"
    assert {
        int(path.stem) for path in (config.output_dir / "records").glob("*.json")
    } == {0, 1}


def test_synthesis_persists_later_slots_after_an_independent_slot_failure(
    tmp_path, monkeypatch
) -> None:
    slots = build_pilot_slots(3, 20260807)
    monkeypatch.setattr(
        "travelweaver.synthesis.pipeline.build_pilot_slots",
        lambda count, seed, profile: slots,
    )
    monkeypatch.setattr(ArtifactStore, "finalize", lambda self, slots, api_calls: {})
    config = SynthesisConfig(
        output_dir=tmp_path / "failure-isolation",
        count=3,
        seed=20260807,
        canonical_only=True,
    )
    pipeline = SynthesisPipeline(
        config,
        DeepSeekConfig(api_key="offline-canonical", model="deterministic-canonical"),
        backend=object(),  # type: ignore[arg-type]
    )

    def prepared(slot, blocked):
        del blocked
        if slot.index == 1:
            raise SynthesisError("structurally infeasible")
        candidate = SimpleNamespace(
            slot=slot,
            uid=f"task-{slot.index}",
            candidate_attempt=1,
            blueprint=SimpleNamespace(blueprint_id=f"blueprint-{slot.index}"),
        )
        return candidate, []

    def materialized(candidate):
        index = candidate.slot.index
        return {
            "slot": asdict(candidate.slot),
            "blueprint": {"blueprint_id": f"blueprint-{index}"},
            "surface": {
                "surface_id": f"surface-{index}",
                "public_query": f"第{index}道失败隔离测试题",
            },
            "task_spec": {"task_id": f"task-{index}"},
            "witness": {},
        }

    monkeypatch.setattr(pipeline, "_prepare_slot", prepared)
    monkeypatch.setattr(pipeline, "_materialize_candidate", materialized)

    with pytest.raises(SynthesisError, match="1 synthesis slot"):
        pipeline.run()

    assert {
        int(path.stem) for path in (config.output_dir / "records").glob("*.json")
    } == {0, 2}


def test_pilot_catalog_has_balanced_100_task_distribution() -> None:
    slots = build_pilot_slots(100, 20260807)

    assert Counter(slot.destination for slot in slots) == {
        "上海": 10,
        "北京": 10,
        "南京": 10,
        "广州": 10,
        "成都": 10,
        "杭州": 10,
        "武汉": 10,
        "深圳": 10,
        "苏州": 10,
        "重庆": 10,
    }
    assert Counter(slot.days for slot in slots) == {1: 10, 2: 30, 3: 35, 4: 15, 5: 10}
    assert Counter(slot.constraint_count for slot in slots) == {
        1: 10,
        2: 20,
        3: 30,
        4: 25,
        5: 10,
        6: 5,
    }
    assert Counter((slot.outbound_mode, slot.return_mode) for slot in slots) == {
        ("train", "train"): 40,
        ("airplane", "airplane"): 20,
        ("train", "airplane"): 20,
        ("airplane", "train"): 20,
    }
    assert Counter(slot.route_mode for slot in slots) == {"taxi": 40, "metro": 35, "walk": 25}
    assert Counter(slot.scenario_profile for slot in slots) == {
        "normal": 70,
        "poi_closure": 8,
        "hotel_unavailable": 6,
        "transport_cancellation": 8,
        "price_change": 8,
    }
    assert Counter(slot.surface_style for slot in slots) == {
        "compact": 10,
        "concise": 10,
        "consultant": 10,
        "conversational": 10,
        "direct": 10,
        "itinerary": 10,
        "narrative": 10,
        "party_first": 10,
        "question": 10,
        "trip_first": 10,
    }
    assert len({(slot.origin, slot.destination) for slot in slots}) == 90
    assert len({tuple(sorted(slot.recipe)) for slot in slots}) >= 85
    assert all(
        (slot.outbound_mode, slot.return_mode) == ("train", "train")
        for slot in slots
        if slot.destination == "苏州"
    )
    recipe_counts = Counter(key for slot in slots for key in slot.recipe)
    assert min(
        count
        for key, count in recipe_counts.items()
        if key not in {"all_intercity_mode", "outbound_mode", "return_mode"}
    ) >= 8


def test_catalog_uses_one_seed_reproducibly_for_arbitrary_counts() -> None:
    assert build_pilot_slots(73, 19) == build_pilot_slots(73, 19)
    assert build_pilot_slots(73, 19) != build_pilot_slots(73, 20)
    for count in range(1, 121):
        slots = build_pilot_slots(count, 7)
        assert len(slots) == count
        assert all(slot.origin != slot.destination for slot in slots)
        assert all(len(slot.recipe) == slot.constraint_count for slot in slots)


def test_chinatravel_blended_profile_preserves_200_task_baseline() -> None:
    slots = build_pilot_slots(200, 20260808, "chinatravel_blended_v1")

    assert Counter(slot.task_type for slot in slots) == {
        "easy_like": 50,
        "medium_like": 70,
        "human_like": 50,
        "preference_like": 20,
        "generalization": 10,
    }
    assert Counter(slot.scenario_profile for slot in slots) == {
        "normal": 180,
        "poi_closure": 5,
        "hotel_unavailable": 4,
        "transport_cancellation": 5,
        "price_change": 6,
    }
    humans = [slot for slot in slots if slot.task_type == "human_like"]
    assert sum(slot.metadata_prefix is not None for slot in humans) == 35
    assert Counter(len(slot.preference_kinds) for slot in humans) == {
        0: 10,
        1: 20,
        2: 15,
        3: 5,
    }
    assert all(slot.validation_profile == "human_conservative" for slot in humans)
    preferences = [slot for slot in slots if slot.task_type == "preference_like"]
    assert all(len(slot.preference_kinds) == 1 for slot in preferences)
    preference_counts = Counter(slot.preference_kinds[0] for slot in preferences)
    official = {
        "more_attractions",
        "less_innercity_time",
        "shorter_meal_transfer",
        "higher_dining_share",
        "lower_lodging_share",
        "near_poi",
    }
    assert all(preference_counts[kind] >= 2 for kind in official)
    assert sum(preference_counts[kind] for kind in official) == 14
    assert len(preference_counts.keys() - official) == 6


@pytest.mark.parametrize(
    ("profile", "expected_digest"),
    [
        (
            "chinatravel_blended_v1",
            "11892b9b0db71e6cd9e61c2f954d27d19aa98c1aee5819579c5b82222bf91f72",
        ),
        (
            "chinatravel_blended_v1_1",
            "315f4e0b704033e41b4b6ec754aeb13664a83b0e3d5fa08780b27a82f1fbb525",
        ),
    ],
)
def test_chinatravel_blended_200_slots_do_not_drift(
    profile: str, expected_digest: str
) -> None:
    slots = build_pilot_slots(200, 20260808, profile)
    payload = json.dumps(
        [asdict(slot) for slot in slots],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    assert hashlib.sha256(payload.encode()).hexdigest() == expected_digest


@pytest.mark.parametrize("count", [1, 37, 199, 500])
@pytest.mark.parametrize(
    "profile", ["chinatravel_blended_v1", "chinatravel_blended_v1_1"]
)
def test_chinatravel_blended_profiles_scale_reproducibly(
    count: int, profile: str
) -> None:
    slots = build_pilot_slots(count, 20260811, profile)

    assert len(slots) == count
    assert slots == build_pilot_slots(count, 20260811, profile)
    assert slots != build_pilot_slots(count, 20260812, profile)


def test_chinatravel_blended_v1_1_scales_to_500_task_quotas() -> None:
    slots = build_pilot_slots(500, 20260811, "chinatravel_blended_v1_1")

    assert Counter(slot.task_type for slot in slots) == {
        "easy_like": 125,
        "medium_like": 175,
        "human_like": 125,
        "preference_like": 50,
        "generalization": 25,
    }
    assert Counter(slot.scenario_profile for slot in slots) == {
        "normal": 450,
        "poi_closure": 12,
        "hotel_unavailable": 10,
        "transport_cancellation": 13,
        "price_change": 15,
    }
    humans = [slot for slot in slots if slot.task_type == "human_like"]
    assert Counter(slot.metadata_prefix is not None for slot in humans) == {
        True: 87,
        False: 38,
    }
    assert Counter(len(slot.preference_kinds) for slot in humans) == {
        0: 25,
        1: 50,
        2: 37,
        3: 13,
    }
    preferences = [
        slot.preference_kinds[0]
        for slot in slots
        if slot.task_type == "preference_like"
    ]
    official = {
        "more_attractions",
        "less_innercity_time",
        "shorter_meal_transfer",
        "higher_dining_share",
        "lower_lodging_share",
        "near_poi",
    }
    assert sum(kind in official for kind in preferences) == 35
    assert max(Counter(preferences).values()) - min(Counter(preferences).values()) <= 4
    logic_keys = {
        "attraction_categories_all",
        "attraction_categories_any",
        "exclude_attraction",
        "allowed_innercity_modes",
    }
    logic_counts = Counter(
        key for slot in slots for key in slot.recipe if key in logic_keys
    )
    assert set(logic_counts) == logic_keys
    assert all(25 <= count <= 50 for count in logic_counts.values())
    assert all(sum(key in logic_keys for key in slot.recipe) <= 1 for slot in slots)
    assert all(
        slot.days > 1
        for slot in slots
        if "attraction_categories_all" in slot.recipe
    )


def test_logic_diversity_constraints_render_and_validate_end_to_end() -> None:
    blueprint = TaskBlueprint(
        trip=TripSpec(origin="上海", destinations=("北京",), days=2, travelers=2),
        constraints=(
            BlueprintConstraint(
                id="c001",
                kind="entity_category",
                operator="contains",
                value={"values": ["公园", "博物馆"]},
                scope="attraction",
            ),
            BlueprintConstraint(
                id="c002",
                kind="entity_category",
                operator="contains",
                value={"any_of": [["美术馆"], ["历史建筑"]]},
                scope="attraction",
            ),
            BlueprintConstraint(
                id="c003",
                kind="exclude_entity",
                operator="exclude",
                value={"names": ["测试景点"]},
                scope="attraction",
            ),
            BlueprintConstraint(
                id="c004",
                kind="transport_mode",
                operator="not_in",
                value={"modes": ["walk"], "leg": "all"},
                scope="innercity_route",
            ),
        ),
        world_snapshot_version="snapshot-v1",
        generator_version="generator-v1",
        generation_seed=7,
    )

    canonical = render_canonical(blueprint)
    assert "公园类景点和博物馆类景点" in canonical.query
    assert "美术馆类景点或历史建筑类景点" in canonical.query
    assert "不要安排测试景点" in canonical.query
    assert "只能使用出租车或地铁，不要步行" in canonical.query
    validate_surface(
        blueprint,
        canonical,
        _payload(blueprint),
        model="logic-diversity-test",
    )


def test_polisher_rejects_swapped_conjunction_and_disjunction() -> None:
    blueprint = TaskBlueprint(
        trip=TripSpec(origin="上海", destinations=("北京",), days=2, travelers=2),
        constraints=(
            BlueprintConstraint(
                id="c001",
                kind="entity_category",
                operator="contains",
                value={"values": ["公园", "博物馆"]},
                scope="attraction",
            ),
            BlueprintConstraint(
                id="c002",
                kind="entity_category",
                operator="contains",
                value={"any_of": [["美术馆"], ["历史建筑"]]},
                scope="attraction",
            ),
        ),
        world_snapshot_version="snapshot-v1",
        generator_version="generator-v1",
        generation_seed=7,
    )
    canonical = render_canonical(blueprint)
    query = canonical.query.replace(
        "公园类景点和博物馆类景点", "公园类景点或博物馆类景点"
    ).replace(
        "美术馆类景点或历史建筑类景点", "美术馆类景点和历史建筑类景点"
    )

    with pytest.raises(SynthesisError, match="Logical conjunction changed"):
        validate_surface(
            blueprint,
            canonical,
            {
                "query": query,
                "mentions": [
                    {
                        "constraint_id": "c001",
                        "text": "至少分别安排一个公园类景点或博物馆类景点",
                    },
                    {
                        "constraint_id": "c002",
                        "text": "至少安排一个美术馆类景点和历史建筑类景点",
                    },
                ],
                "preference_mentions": [],
            },
            model="logic-shape-test",
            validation_policy="minimal_semantic",
        )


def test_polisher_rejects_conjunctive_allowed_transport_set() -> None:
    blueprint = TaskBlueprint(
        trip=TripSpec(origin="上海", destinations=("北京",), days=2, travelers=2),
        constraints=(
            BlueprintConstraint(
                id="c001",
                kind="transport_mode",
                operator="not_in",
                value={"modes": ["walk"], "leg": "all"},
                scope="innercity_route",
            ),
        ),
        world_snapshot_version="snapshot-v1",
        generator_version="generator-v1",
        generation_seed=7,
    )
    canonical = render_canonical(blueprint)
    query = canonical.query.replace("出租车或地铁", "出租车和地铁")

    with pytest.raises(SynthesisError, match="Logical disjunction changed"):
        validate_surface(
            blueprint,
            canonical,
            {
                "query": query,
                "mentions": [
                    {
                        "constraint_id": "c001",
                        "text": "至少安排两个市内地点，地点之间只能使用出租车和地铁，不要步行",
                    }
                ],
                "preference_mentions": [],
            },
            model="allowed-set-test",
            validation_policy="minimal_semantic",
        )


def test_official_hybrid_v2_first_batch_has_locked_quotas() -> None:
    slots = build_pilot_slots(500, 20260821, "chinatravel_official_hybrid_v2")

    assert Counter(slot.task_type for slot in slots) == {
        "easy_like": 184,
        "medium_like": 92,
        "human_like": 94,
        "preference_base": 30,
        "preference_like": 50,
        "generalization": 50,
    }
    assert Counter(slot.scenario_profile for slot in slots) == {
        "normal": 450,
        "price_change": 15,
        "transport_cancellation": 13,
        "poi_closure": 12,
        "hotel_unavailable": 10,
    }
    assert Counter(
        slot.preference_kinds[0] for slot in slots if slot.preference_kinds
    ) == {
        "more_attractions": 9,
        "less_innercity_time": 9,
        "shorter_meal_transfer": 8,
        "higher_dining_share": 8,
        "lower_lodging_share": 8,
        "near_poi": 8,
    }


def test_scaled_alignment_uses_the_same_500_task_targets() -> None:
    slots = build_pilot_slots(500, 20260811, "chinatravel_blended_v1_1")
    records = []
    for index, slot in enumerate(slots):
        preference_audit = None
        if slot.task_type == "preference_like":
            preference_audit = {
                "candidates": [
                    {"metric_value": 0, "all_hard_pass": True, "hard_reward": 1.0},
                    {"metric_value": 1, "all_hard_pass": True, "hard_reward": 1.0},
                ]
            }
        records.append(
            {
                "surface": {"public_query": f"合成任务{index}"},
                "witness": {"reward_detail": {"all_hard_pass": True, "reward": 1.0}},
                "preference_audit": preference_audit,
            }
        )
    distributions = {
        "task_types": dict(Counter(slot.task_type for slot in slots)),
        "scenario_profiles": dict(Counter(slot.scenario_profile for slot in slots)),
        "surface_quality": {
            "human_template_term_rate": 0.0,
            "human_max_opening_share": 0.01,
            "human_metadata_persona_repetitions": 0,
        },
    }

    alignment = _alignment(
        records,
        distributions,
        {
            "count": 500,
            "seed": 20260811,
            "profile": "chinatravel_blended_v1_1",
            "validation_policy": "minimal_semantic",
        },
    )

    assert alignment["expected"]["task_types"]["preference_like"] == 50
    assert alignment["expected"]["preference_audit_count"] == 50
    assert all(alignment["checks"].values())


def test_innercity_preference_candidates_add_a_routable_meal(monkeypatch) -> None:
    slot = build_pilot_slots(500, 20260811, "chinatravel_blended_v1_1")[68]
    captured_include_meal: list[bool] = []

    class FakeBackend:
        @staticmethod
        def _records(entity_type: str, city: str) -> list[dict[str, object]]:
            del entity_type, city
            return []

    class FakeWitnessBuilder:
        def __init__(self, backend, *, seed: int) -> None:
            del backend, seed

        def build(self, candidate_slot, *, origin: str, uid: str):
            del origin, uid
            captured_include_meal.append(candidate_slot.include_meal)
            duration = {"taxi": 10, "metro": 20, "walk": 30}[candidate_slot.route_mode]
            return SimpleNamespace(
                evidence_bundle={
                    "routes": {
                        "meal": {
                            "segments": [
                                {"start_time": "12:00", "end_time": f"12:{duration:02d}"}
                            ]
                        }
                    },
                    "cost_items": [],
                    "total_cost": 0.0,
                },
                route_mode=candidate_slot.route_mode,
                selected={"attractions": [{}]},
            )

    monkeypatch.setattr(
        "travelweaver.synthesis.pipeline.WitnessBuilder", FakeWitnessBuilder
    )
    pipeline = object.__new__(SynthesisPipeline)

    candidates = pipeline._preference_witnesses(
        FakeBackend(),
        slot,
        origin="成都",
        uid="preference-regression",
        generation_seed=7,
    )

    assert all(captured_include_meal)
    assert len({_preference_metric(row, "less_innercity_time") for row in candidates}) >= 2


def test_long_relaxed_preference_varies_meal_stop_instead_of_extra_attractions(
    monkeypatch,
) -> None:
    slot = replace(
        build_pilot_slots(500, 20260838, "chinatravel_blended_v1_1")[69],
        days=4,
        preference_kinds=("relaxed_itinerary",),
        include_meal=False,
    )
    captured: list[tuple[int, bool]] = []

    class FakeBackend:
        @staticmethod
        def _records(entity_type: str, city: str) -> list[dict[str, object]]:
            del entity_type, city
            return []

    class FakeWitnessBuilder:
        def __init__(self, backend, *, seed: int) -> None:
            del backend, seed

        def build(self, candidate_slot, *, origin: str, uid: str):
            del origin, uid
            captured.append(
                (candidate_slot.attractions_per_day, candidate_slot.include_meal)
            )
            return SimpleNamespace(
                evidence_bundle={"routes": {}, "cost_items": [], "total_cost": 0.0},
                route_mode=candidate_slot.route_mode,
                selected={
                    "attractions": [{}] * candidate_slot.days,
                    "restaurant": (
                        {"place_id": "restaurant"}
                        if candidate_slot.include_meal
                        else None
                    ),
                },
            )

    monkeypatch.setattr(
        "travelweaver.synthesis.pipeline.WitnessBuilder", FakeWitnessBuilder
    )
    pipeline = object.__new__(SynthesisPipeline)

    candidates = pipeline._preference_witnesses(
        FakeBackend(),
        slot,
        origin="南京",
        uid="long-relaxed-regression",
        generation_seed=7,
    )

    assert {include_meal for _, include_meal in captured} == {False, True}
    assert {attractions_per_day for attractions_per_day, _ in captured} == {1}
    assert len({_preference_metric(row, "relaxed_itinerary") for row in candidates}) == 2


def test_long_relaxed_preference_varies_meal_when_slot_already_includes_one(
    monkeypatch,
) -> None:
    slot = replace(
        build_pilot_slots(500, 20260839, "chinatravel_blended_v1_1")[386],
        days=4,
        preference_kinds=("relaxed_itinerary",),
        include_meal=True,
    )
    captured_include_meal: list[bool] = []

    class FakeBackend:
        @staticmethod
        def _records(entity_type: str, city: str) -> list[dict[str, object]]:
            del entity_type, city
            return []

    class FakeWitnessBuilder:
        def __init__(self, backend, *, seed: int) -> None:
            del backend, seed

        def build(self, candidate_slot, *, origin: str, uid: str):
            del origin, uid
            captured_include_meal.append(candidate_slot.include_meal)
            return SimpleNamespace(
                evidence_bundle={"routes": {}, "cost_items": [], "total_cost": 0.0},
                route_mode=candidate_slot.route_mode,
                selected={
                    "attractions": [{}] * candidate_slot.days,
                    "restaurant": (
                        {"place_id": "restaurant"}
                        if candidate_slot.include_meal
                        else None
                    ),
                },
            )

    monkeypatch.setattr(
        "travelweaver.synthesis.pipeline.WitnessBuilder", FakeWitnessBuilder
    )
    pipeline = object.__new__(SynthesisPipeline)

    candidates = pipeline._preference_witnesses(
        FakeBackend(),
        slot,
        origin="武汉",
        uid="long-relaxed-initial-meal-regression",
        generation_seed=7,
    )

    assert set(captured_include_meal) == {False, True}
    assert len({_preference_metric(row, "relaxed_itinerary") for row in candidates}) == 2


def test_chinatravel_blended_v1_1_keeps_benchmark_core_and_tail_split() -> None:
    slots = build_pilot_slots(200, 20260808, "chinatravel_blended_v1_1")

    generalization = [slot for slot in slots if slot.task_type == "generalization"]
    assert Counter(slot.days for slot in generalization) == {4: 5, 5: 5}
    assert Counter(slot.travelers for slot in generalization) == {5: 5, 6: 5}
    easy = [slot for slot in slots if slot.task_type == "easy_like"]
    assert sum(slot.days <= 3 for slot in easy) == 45
    assert sum(slot.travelers <= 4 for slot in easy) == 47
    assert any(slot.validation_profile == "benchmark_natural" for slot in easy)
    humans = [slot for slot in slots if slot.task_type == "human_like"]
    assert sum(slot.metadata_prefix is not None for slot in humans) == 35
    assert all(
        slot.metadata_prefix is None or slot.metadata_prefix.startswith("[当前位置")
        for slot in humans
    )
    assert not any(
        (slot.days == 1 and "lower_lodging_share" in slot.preference_kinds)
        or (slot.route_mode == "walk" and "less_walking" in slot.preference_kinds)
        or (
            "attraction_count" in slot.recipe
            and bool(
                set(slot.preference_kinds) & {"more_attractions", "relaxed_itinerary"}
            )
        )
        for slot in humans
    )
    preference_recipes = {
        key
        for slot in slots
        if slot.task_type == "preference_like"
        for key in slot.recipe
    }
    assert preference_recipes & {
        "total_budget",
        "outbound_time",
        "return_time",
        "attraction_count",
    }


def test_scenario_backend_materializes_hidden_availability_and_price_changes() -> None:
    backend = InMemoryBackend(
        [
            {
                "place_id": "place-1",
                "entity_type": "attraction",
                "city": "上海",
                "name": "景点甲",
                "price": 50,
            },
            {
                "place_id": "place-2",
                "entity_type": "attraction",
                "city": "上海",
                "name": "景点乙",
                "price": 100,
            },
        ]
    )
    scenario = ScenarioSpec(
        base_world_snapshot_version="world-v1",
        profile="combined-test",
        effects=(
            ScenarioEffect(
                "effect-1",
                "unavailable",
                "attraction",
                "place-1",
                "available",
                True,
                False,
            ),
            ScenarioEffect(
                "effect-2",
                "field_override",
                "attraction",
                "place-2",
                "price",
                100,
                125,
            ),
        ),
    )
    wrapped = ScenarioBackend(backend, scenario)

    assert wrapped.search_attractions(city="上海", max_price=110) == []
    assert wrapped.inspect_place("place-2")["price"] == 125
    assert ScenarioSpec.from_dict(scenario.to_dict()) == scenario
    assert scenario.scenario_id == ScenarioSpec(
        base_world_snapshot_version="world-v1",
        profile="combined-test",
        effects=scenario.effects,
    ).scenario_id


def test_blueprint_surface_round_trip_materializes_exact_source_spans() -> None:
    blueprint = _blueprint()
    canonical = render_canonical(blueprint)
    surface = validate_surface(
        blueprint,
        canonical,
        _payload(blueprint),
        model="canonical-test",
    )

    assert TaskBlueprint.from_dict(blueprint.to_dict()) == blueprint
    assert TaskSurface.from_dict(surface.to_dict()) == surface
    spec = materialize_task_spec(blueprint, surface, task_id="synthetic-1")
    assert spec.task_id == "synthetic-1"
    assert spec.constraints[0].value["leg"] == "outbound"
    assert all(
        spec.public_query[item.source_start : item.source_end] == item.source_text
        for item in spec.constraints
    )


def test_canonical_styles_change_expression_without_changing_protected_facts() -> None:
    queries = {
        render_canonical(_blueprint(), style_profile=style).query
        for style in (
            "compact",
            "concise",
            "consultant",
            "conversational",
            "direct",
            "itinerary",
            "narrative",
            "party_first",
            "question",
            "trip_first",
        )
    }

    assert len(queries) == 10
    assert all(
        all(literal in query for literal in ("上海", "北京", "2天", "2人", "火车", "3000元"))
        for query in queries
    )


def test_benchmark_natural_canonical_uses_determined_non_template_phrases() -> None:
    blueprint = TaskBlueprint(
        trip=TripSpec(origin="上海", destinations=("杭州",), days=2, travelers=2),
        constraints=(
            BlueprintConstraint(
                "c001",
                "transport_mode",
                "eq",
                {"modes": ["train"], "leg": "all"},
                "intercity_transport",
            ),
            BlueprintConstraint(
                "c002",
                "include_entity",
                "include",
                {"names": ["西湖"]},
                "attraction",
            ),
            BlueprintConstraint(
                "c003",
                "room_count",
                "eq",
                {"count": 2},
                "accommodation",
            ),
        ),
        world_snapshot_version="snapshot-v1",
        generator_version="generator-v1",
        generation_seed=5,
    )
    canonical = render_canonical(blueprint, validation_profile="benchmark_natural")
    payload = {
        "query": canonical.query,
        "mentions": [
            {"constraint_id": constraint_id, "text": text}
            for constraint_id, text in canonical.clauses.items()
        ],
        "preference_mentions": [],
    }

    surface = validate_surface(
        blueprint,
        canonical,
        payload,
        model="benchmark-natural-test",
        validation_profile="benchmark_natural",
    )

    assert "必须" not in surface.public_query
    assert all(
        text in surface.public_query
        for text in ("往返坐火车", "想去西湖", "酒店每晚订2间房")
    )


def test_minimal_semantic_policy_accepts_natural_number_and_equality_forms() -> None:
    blueprint = _blueprint()
    canonical = render_canonical(blueprint)
    payload = _payload(blueprint)
    payload["query"] = payload["query"].replace("2人", "两个人")
    payload["query"] = payload["query"].replace("去程必须乘坐火车", "去程坐火车")
    payload["mentions"][0]["text"] = "去程坐火车"

    surface = validate_surface(
        blueprint,
        canonical,
        payload,
        model="minimal-semantic-test",
        validation_policy="minimal_semantic",
    )

    assert surface.validation_policy == "minimal_semantic"
    assert "protected_literal_changed:2人" in surface.validation_warnings
    assert "global_numeric_literal_multiset_changed" in surface.validation_warnings


def test_minimal_semantic_policy_still_rejects_changed_value_and_optional_hard_rule() -> None:
    blueprint = _blueprint()
    canonical = render_canonical(blueprint)
    changed = _payload(blueprint)
    changed["query"] = changed["query"].replace("3000元", "5000元")
    changed["mentions"][1]["text"] = changed["mentions"][1]["text"].replace(
        "3000元", "5000元"
    )
    with pytest.raises(SynthesisError, match="Numeric value changed"):
        validate_surface(
            blueprint,
            canonical,
            changed,
            model="minimal-changed-value-test",
            validation_policy="minimal_semantic",
        )

    optional = _payload(blueprint)
    optional["query"] = optional["query"].replace(
        "去程必须乘坐火车", "去程最好坐火车"
    )
    optional["mentions"][0]["text"] = "去程最好坐火车"
    with pytest.raises(SynthesisError, match="became optional"):
        validate_surface(
            blueprint,
            canonical,
            optional,
            model="minimal-optional-test",
            validation_policy="minimal_semantic",
        )


def test_minimal_semantic_policy_repairs_mentions_and_allows_shared_clauses() -> None:
    blueprint = TaskBlueprint(
        trip=TripSpec(origin="上海", destinations=("杭州",), days=2, travelers=2),
        constraints=(
            BlueprintConstraint(
                "c001",
                "transport_mode",
                "eq",
                {"modes": ["train"], "leg": "outbound"},
                "intercity_transport",
            ),
            BlueprintConstraint(
                "c002",
                "time_window",
                "lte",
                {"leg": "outbound", "field": "end_time", "time": "10:00"},
                "intercity_transport",
            ),
            BlueprintConstraint(
                "c003",
                "include_entity",
                "include",
                {"names": ["西湖"]},
                "attraction",
            ),
        ),
        world_snapshot_version="snapshot-v1",
        generator_version="generator-v1",
        generation_seed=11,
    )
    canonical = render_canonical(blueprint, validation_profile="benchmark_natural")
    query = "上海到杭州玩2天，共2人。去程坐火车且要在10:00前到达，想打卡西湖。"
    payload = {
        "query": query,
        "mentions": [
            {"constraint_id": "c001", "text": "去程坐火车"},
            {"constraint_id": "c002", "text": "去程坐火车且要在10:00前到达"},
            {"constraint_id": "c003", "text": "想去打卡西湖"},
        ],
        "preference_mentions": [],
    }

    surface = validate_surface(
        blueprint,
        canonical,
        payload,
        model="minimal-repair-test",
        validation_profile="benchmark_natural",
        validation_policy="minimal_semantic",
    )

    assert "mention_repaired:c003" in surface.validation_warnings
    assert next(
        mention.text for mention in surface.mentions if mention.constraint_id == "c003"
    ) == "西湖"

    optional_entity = dict(payload)
    optional_entity["query"] = query.replace("想打卡西湖", "最好去西湖")
    optional_entity["mentions"] = [
        *payload["mentions"][:2],
        {"constraint_id": "c003", "text": "西湖"},
    ]
    with pytest.raises(SynthesisError, match="optional in context"):
        validate_surface(
            blueprint,
            canonical,
            optional_entity,
            model="minimal-context-optional-test",
            validation_profile="benchmark_natural",
            validation_policy="minimal_semantic",
        )


@pytest.mark.parametrize(
    ("validation_profile", "expected"),
    (
        ("strict", "至少安排两个市内地点，地点之间必须步行"),
        ("benchmark_natural", "至少安排两个市内地点，地点之间都步行"),
    ),
)
def test_walking_constraint_uses_natural_verb(
    validation_profile: str,
    expected: str,
) -> None:
    blueprint = TaskBlueprint(
        trip=TripSpec(origin="上海", destinations=("杭州",), days=2, travelers=2),
        constraints=(
            BlueprintConstraint(
                "c001",
                "transport_mode",
                "eq",
                {"modes": ["walk"], "leg": "all"},
                "innercity_route",
            ),
        ),
        world_snapshot_version="snapshot-v1",
        generator_version="generator-v1",
        generation_seed=5,
    )

    canonical = render_canonical(blueprint, validation_profile=validation_profile)

    payload = {
        "query": canonical.query,
        "mentions": [
            {"constraint_id": constraint_id, "text": text}
            for constraint_id, text in canonical.clauses.items()
        ],
        "preference_mentions": [],
    }
    validate_surface(
        blueprint,
        canonical,
        payload,
        model="walking-canonical-test",
        validation_profile=validation_profile,
    )

    assert expected in canonical.query
    assert "坐步行" not in canonical.query
    assert "使用步行" not in canonical.query


def test_restaurant_budget_requires_nonempty_meal_scope_after_polishing() -> None:
    blueprint = TaskBlueprint(
        trip=TripSpec(origin="上海", destinations=("杭州",), days=2, travelers=2),
        constraints=(
            BlueprintConstraint(
                "c001",
                "category_budget",
                "lte",
                {"amount": 100, "basis": "per_person_per_activity"},
                "restaurant",
            ),
        ),
        world_snapshot_version="snapshot-v1",
        generator_version="generator-v1",
        generation_seed=5,
    )
    canonical = render_canonical(blueprint, validation_profile="benchmark_natural")
    assert "至少安排一顿用餐，餐厅人均每餐不超过100元" in canonical.query
    validate_surface(
        blueprint,
        canonical,
        {
            "query": canonical.query,
            "mentions": [{"constraint_id": "c001", "text": canonical.clauses["c001"]}],
            "preference_mentions": [],
        },
        model="canonical-nonempty-meal-test",
        validation_profile="benchmark_natural",
    )
    natural_query = canonical.query.replace("至少安排一顿用餐", "最少吃一顿饭")
    natural_payload = {
        "query": natural_query,
        "mentions": [
            {
                "constraint_id": "c001",
                "text": "最少吃一顿饭，餐厅人均每餐不超过100元",
            }
        ],
        "preference_mentions": [],
    }
    validate_surface(
        blueprint,
        canonical,
        natural_payload,
        model="nonempty-meal-test",
        validation_profile="benchmark_natural",
        validation_policy="minimal_semantic",
    )

    missing = {
        "query": canonical.query.replace("至少安排一顿用餐，", ""),
        "mentions": [{"constraint_id": "c001", "text": "餐厅人均每餐不超过100元"}],
        "preference_mentions": [],
    }
    with pytest.raises(SynthesisError, match="does not require a meal"):
        validate_surface(
            blueprint,
            canonical,
            missing,
            model="empty-meal-scope-test",
            validation_profile="benchmark_natural",
            validation_policy="minimal_semantic",
        )


def test_restaurant_budget_uses_all_planned_meals_not_only_first_price() -> None:
    slot = replace(
        build_pilot_slots(1, 20260812)[0],
        days=2,
        travelers=2,
        tightness="medium",
        recipe=("restaurant_budget",),
    )
    witness = SimpleNamespace(
        selected={"restaurant": {"price": 80}},
        public_task={"days": 2, "people_number": 2},
        evidence_bundle={
            "cost_items": [
                {"activity_type": "lunch", "amount": 160},
                {"activity_type": "dinner", "amount": 400},
            ]
        },
    )

    constraint = _constraint(1, "restaurant_budget", slot, witness, (witness,))

    assert constraint.value == {
        "amount": 160,
        "basis": "per_person_per_activity",
    }


def test_innercity_transport_requires_two_places_after_polishing() -> None:
    blueprint = TaskBlueprint(
        trip=TripSpec(origin="上海", destinations=("杭州",), days=2, travelers=2),
        constraints=(
            BlueprintConstraint(
                "c001",
                "transport_mode",
                "eq",
                {"modes": ["metro"], "leg": "all"},
                "innercity_route",
            ),
        ),
        world_snapshot_version="snapshot-v1",
        generator_version="generator-v1",
        generation_seed=5,
    )
    canonical = render_canonical(blueprint, validation_profile="benchmark_natural")
    assert "至少安排两个市内地点，地点之间统一坐地铁" in canonical.query

    natural_query = canonical.query.replace("至少安排两个市内地点", "市内选择两处地点")
    natural_payload = {
        "query": natural_query,
        "mentions": [
            {"constraint_id": "c001", "text": "市内选择两处地点，地点之间统一坐地铁"}
        ],
        "preference_mentions": [],
    }
    validate_surface(
        blueprint,
        canonical,
        natural_payload,
        model="two-innercity-places-test",
        validation_profile="benchmark_natural",
        validation_policy="minimal_semantic",
    )

    missing = {
        "query": canonical.query.replace("至少安排两个市内地点，", ""),
        "mentions": [{"constraint_id": "c001", "text": "地点之间统一坐地铁"}],
        "preference_mentions": [],
    }
    with pytest.raises(SynthesisError, match="does not require two places"):
        validate_surface(
            blueprint,
            canonical,
            missing,
            model="empty-innercity-scope-test",
            validation_profile="benchmark_natural",
            validation_policy="minimal_semantic",
        )


def test_surface_validator_allows_city_substring_in_protected_entity_name() -> None:
    blueprint = TaskBlueprint(
        trip=TripSpec(origin="上海", destinations=("广州",), days=2, travelers=2),
        constraints=(
            BlueprintConstraint(
                "c001",
                "include_entity",
                "include",
                {"names": ["北京路步行街"]},
                "attraction",
            ),
        ),
        world_snapshot_version="snapshot-v1",
        generator_version="generator-v1",
        generation_seed=5,
    )
    canonical = render_canonical(blueprint, validation_profile="benchmark_natural")
    payload = {
        "query": canonical.query,
        "mentions": [
            {"constraint_id": constraint_id, "text": text}
            for constraint_id, text in canonical.clauses.items()
        ],
        "preference_mentions": [],
    }

    surface = validate_surface(
        blueprint,
        canonical,
        payload,
        model="protected-entity-city-test",
        validation_profile="benchmark_natural",
    )

    assert "北京路步行街" in surface.public_query


def test_v1_1_human_metadata_is_not_exposed_in_the_natural_user_surface() -> None:
    metadata = "[当前位置上海,目标位置杭州,旅行人数2,旅行天数2,出行背景情侣出行]"
    blueprint = TaskBlueprint(
        trip=TripSpec(origin="上海", destinations=("杭州",), days=2, travelers=2),
        constraints=(),
        world_snapshot_version="snapshot-v1",
        generator_version="generator-v1",
        generation_seed=7,
        persona_context="情侣出行",
        metadata_prefix=metadata,
    )
    canonical = render_canonical(
        blueprint,
        style_profile="human_v1_1_metadata",
        validation_profile="human_conservative",
    )
    payload = {"query": canonical.query, "mentions": [], "preference_mentions": []}

    surface = validate_surface(
        blueprint,
        canonical,
        payload,
        model="human-v1-1-test",
        validation_profile="human_conservative",
    )

    assert not surface.public_query.startswith("[")
    assert metadata not in surface.public_query
    assert surface.public_query.count("情侣出行") == 1

    direct = render_canonical(
        blueprint,
        style_profile="direct",
        validation_profile="human_conservative",
    )
    assert not direct.query.startswith("[")
    assert direct.query.count("情侣出行") == 1


def test_surface_validator_rejects_changed_numeric_semantics() -> None:
    blueprint = _blueprint()
    canonical = render_canonical(blueprint)
    payload = _payload(blueprint)
    payload["query"] = payload["query"].replace("3000元", "5000元")

    with pytest.raises(SynthesisError, match="Protected literal|Numeric literals"):
        validate_surface(blueprint, canonical, payload, model="bad-test")


def test_human_surface_validator_accepts_concise_chinese_hard_modality() -> None:
    blueprint = _blueprint()
    canonical = render_canonical(blueprint)
    payload = _payload(blueprint)
    payload["query"] = payload["query"].replace("去程必须乘坐火车", "去程需乘坐火车")
    payload["mentions"][0]["text"] = "去程需乘坐火车"

    surface = validate_surface(
        blueprint,
        canonical,
        payload,
        model="concise-test",
        validation_profile="human_conservative",
    )

    assert surface.mentions[0].text == "去程需乘坐火车"


def test_human_surface_validator_accepts_typed_imperative_without_redundant_modal() -> None:
    blueprint = _blueprint()
    canonical = render_canonical(blueprint)
    payload = _payload(blueprint)
    payload["query"] = payload["query"].replace("去程必须乘坐火车", "去程坐火车")
    payload["mentions"][0]["text"] = "去程坐火车"

    surface = validate_surface(
        blueprint,
        canonical,
        payload,
        model="imperative-test",
        validation_profile="human_conservative",
    )

    assert surface.mentions[0].text == "去程坐火车"


@pytest.mark.parametrize(
    ("constraint", "canonical_text", "natural_text"),
    [
        (
            BlueprintConstraint(
                "c001",
                "transport_mode",
                "eq",
                {"modes": ["high_speed_rail"], "leg": "all"},
                "intercity_transport",
            ),
            "往返城际交通必须乘坐高铁",
            "往返坐高铁",
        ),
        (
            BlueprintConstraint(
                "c001",
                "include_entity",
                "include",
                {"names": ["西湖"]},
                "attraction",
            ),
            "必须游览西湖",
            "想去西湖",
        ),
        (
            BlueprintConstraint(
                "c001",
                "room_count",
                "eq",
                {"count": 2},
                "accommodation",
            ),
            "每晚必须预订2间房",
            "酒店订2间房",
        ),
    ],
)
def test_human_surface_validator_accepts_natural_determined_phrases(
    constraint: BlueprintConstraint,
    canonical_text: str,
    natural_text: str,
) -> None:
    blueprint = TaskBlueprint(
        trip=TripSpec(origin="上海", destinations=("杭州",), days=2, travelers=2),
        constraints=(constraint,),
        world_snapshot_version="snapshot-v1",
        generator_version="generator-v1",
        generation_seed=3,
    )
    canonical = render_canonical(blueprint)
    payload = _payload(blueprint)
    payload["query"] = payload["query"].replace(canonical_text, natural_text)
    payload["mentions"][0]["text"] = natural_text

    surface = validate_surface(
        blueprint,
        canonical,
        payload,
        model="human-natural-test",
        validation_profile="human_conservative",
    )

    assert surface.mentions[0].text == natural_text


def test_strict_surface_validator_does_not_use_human_equality_relaxation() -> None:
    blueprint = _blueprint()
    canonical = render_canonical(blueprint)
    payload = _payload(blueprint)
    payload["query"] = payload["query"].replace("去程必须乘坐火车", "去程坐火车")
    payload["mentions"][0]["text"] = "去程坐火车"

    with pytest.raises(SynthesisError, match="became optional"):
        validate_surface(blueprint, canonical, payload, model="strict-test")


def test_human_surface_validator_rejects_optional_hard_constraint() -> None:
    blueprint = _blueprint()
    canonical = render_canonical(blueprint)
    payload = _payload(blueprint)
    payload["query"] = payload["query"].replace(
        "去程必须乘坐火车", "去程如果可以坐火车"
    )
    payload["mentions"][0]["text"] = "去程如果可以坐火车"

    with pytest.raises(SynthesisError, match="became optional"):
        validate_surface(
            blueprint,
            canonical,
            payload,
            model="optional-test",
            validation_profile="human_conservative",
        )


def test_human_surface_validator_rejects_weakened_boundary_and_new_fact() -> None:
    blueprint = _blueprint()
    canonical = render_canonical(blueprint)
    payload = _payload(blueprint)
    payload["query"] = payload["query"].replace("总预算不超过3000元", "总预算3000元左右")
    payload["mentions"][1]["text"] = "总预算3000元左右"
    with pytest.raises(SynthesisError, match="polarity|ambiguous"):
        validate_surface(
            blueprint,
            canonical,
            payload,
            model="boundary-test",
            validation_profile="human_conservative",
        )

    payload = _payload(blueprint)
    payload["query"] += "顺便去成都。"
    with pytest.raises(SynthesisError, match="introduced cities"):
        validate_surface(
            blueprint,
            canonical,
            payload,
            model="fact-test",
            validation_profile="human_conservative",
        )


def test_human_surface_validator_requires_all_hard_and_preference_mentions() -> None:
    blueprint = TaskBlueprint(
        trip=TripSpec(origin="上海", destinations=("北京",), days=2, travelers=2),
        constraints=_blueprint().constraints,
        world_snapshot_version="snapshot-v1",
        generator_version="generator-v1",
        generation_seed=8,
        preferences=(
            BlueprintPreference(
                id="p001",
                kind="less_walking",
                direction="minimize",
            ),
        ),
        persona_context="情侣出行",
    )
    canonical = render_canonical(blueprint, style_profile="human_dialogue")
    payload = {
        "query": canonical.query,
        "mentions": [
            {"constraint_id": constraint_id, "text": text}
            for constraint_id, text in canonical.clauses.items()
        ],
        "preference_mentions": [
            {"preference_id": preference_id, "text": text}
            for preference_id, text in canonical.preference_clauses.items()
        ],
    }
    payload["mentions"].pop()
    with pytest.raises(SynthesisError, match="Mention coverage"):
        validate_surface(
            blueprint,
            canonical,
            payload,
            model="missing-hard-test",
            validation_profile="human_conservative",
        )

    payload = {
        "query": canonical.query,
        "mentions": [
            {"constraint_id": constraint_id, "text": text}
            for constraint_id, text in canonical.clauses.items()
        ],
        "preference_mentions": [
            {"preference_id": preference_id, "text": text}
            for preference_id, text in canonical.preference_clauses.items()
        ],
    }
    payload["preference_mentions"] = []
    with pytest.raises(SynthesisError, match="Preference mention coverage"):
        validate_surface(
            blueprint,
            canonical,
            payload,
            model="missing-preference-test",
            validation_profile="human_conservative",
        )


def test_preference_mentions_allow_controlled_synonyms_and_materialize() -> None:
    blueprint = TaskBlueprint(
        trip=TripSpec(origin="上海", destinations=("北京",), days=2, travelers=2),
        constraints=_blueprint().constraints,
        world_snapshot_version="snapshot-v1",
        generator_version="generator-v1",
        generation_seed=9,
        preferences=(BlueprintPreference("p001", "relaxed_itinerary", "minimize"),),
    )
    canonical = render_canonical(blueprint)
    payload = _payload(blueprint)
    payload["query"] = payload["query"].replace("希望行程轻松一些", "行程别太赶")
    payload["preference_mentions"][0]["text"] = "行程别太赶"
    surface = validate_surface(
        blueprint,
        canonical,
        payload,
        model="preference-test",
        validation_profile="human_conservative",
    )
    spec = materialize_task_spec(blueprint, surface)

    assert spec.unscored_preferences == ("行程别太赶",)


def test_polisher_uses_one_required_function_call_and_disables_thinking() -> None:
    blueprint = _blueprint()
    payload = _payload(blueprint)
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(
                                name="polish_travel_query",
                                arguments=json.dumps(payload, ensure_ascii=False),
                            )
                        )
                    ]
                )
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        ),
    )

    class _Client:
        def __init__(self) -> None:
            self.requests = []

        def complete(self, messages, tools):
            self.requests.append((messages, tools))
            return response

    client = _Client()
    polisher = TaskPolisher(
        DeepSeekConfig(api_key="not-a-secret", thinking="enabled"),
        client=client,
    )
    surface = polisher.polish(blueprint, render_canonical(blueprint))

    assert surface.usage["total_tokens"] == 150
    assert polisher.api_calls == 1
    assert polisher.config.thinking == "disabled"
    assert polisher.config.tool_choice == "required"
    assert client.requests[0][1][0]["function"]["name"] == "polish_travel_query"


def test_polisher_uses_validated_styled_canonical_as_bounded_fallback() -> None:
    class _FailingClient:
        @staticmethod
        def complete(messages, tools):
            del messages, tools
            raise RuntimeError("model unavailable")

    blueprint = _blueprint()
    canonical = render_canonical(blueprint, style_profile="question")
    polisher = TaskPolisher(
        DeepSeekConfig(api_key="not-a-secret"),
        client=_FailingClient(),
    )

    surface = polisher.polish(
        blueprint,
        canonical,
        style_profile="question",
    )

    assert polisher.api_calls == 2
    assert surface.public_query == canonical.query
    assert surface.polisher_model.endswith(":canonical-fallback")


def test_polisher_audit_preserves_rejected_raw_responses_and_errors() -> None:
    blueprint = _blueprint()
    canonical = render_canonical(blueprint)
    invalid_payload = _payload(blueprint)
    invalid_payload["query"] = invalid_payload["query"].replace("3000元", "5000元")
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(
                                name="polish_travel_query",
                                arguments=json.dumps(invalid_payload, ensure_ascii=False),
                            )
                        )
                    ]
                )
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )

    class _InvalidClient:
        @staticmethod
        def complete(messages, tools):
            del messages, tools
            return response

    polisher = TaskPolisher(
        DeepSeekConfig(api_key="not-a-secret"),
        client=_InvalidClient(),
    )
    surface, audit = polisher.polish_with_audit(
        blueprint,
        canonical,
        audit_context={"slot_index": 12},
    )

    assert surface.polisher_model.endswith(":canonical-fallback")
    assert [event["outcome"] for event in audit] == [
        "rejected",
        "rejected",
        "canonical_fallback",
    ]
    assert audit[0]["slot_index"] == 12
    assert audit[0]["raw_response"]["choices"]
    assert audit[0]["parsed_payload"] == invalid_payload
    assert "does not occur in query" in audit[0]["validation_error"]
