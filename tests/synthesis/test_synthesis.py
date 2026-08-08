from __future__ import annotations

import json
from collections import Counter
from types import SimpleNamespace

import pytest

from travelweaver.env import InMemoryBackend, ScenarioBackend, ScenarioEffect, ScenarioSpec
from travelweaver.errors import SynthesisError
from travelweaver.llm import DeepSeekConfig
from travelweaver.synthesis.catalog import build_pilot_slots
from travelweaver.synthesis.polisher import TaskPolisher, validate_surface
from travelweaver.synthesis.render import render_canonical
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


def test_chinatravel_blended_profile_has_frozen_200_task_quotas() -> None:
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
    with pytest.raises(ValueError, match="frozen 200-task"):
        build_pilot_slots(199, 20260808, "chinatravel_blended_v1")


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
    (("strict", "市内地点之间必须步行"), ("benchmark_natural", "市内地点之间都步行")),
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


def test_v1_1_human_metadata_fallback_does_not_repeat_persona() -> None:
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

    assert surface.public_query.count("情侣出行") == 1
    repeated = dict(payload)
    repeated["query"] += "我们是情侣出行。"
    with pytest.raises(SynthesisError, match="repeats the persona"):
        validate_surface(
            blueprint,
            canonical,
            repeated,
            model="human-v1-1-repeat-test",
            validation_profile="human_conservative",
        )


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
