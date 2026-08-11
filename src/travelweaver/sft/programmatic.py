"""Build replayable programmatic policy trajectories from accepted synthesis witnesses."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..env import ChinaTravelBackend, ScenarioBackend, ScenarioSpec, TravelWeaverEnv
from ..errors import SFTRebuildError
from ..rollout.api_agent import (
    TRAJECTORY_VERSION,
    USER_CONTENT_FORMAT,
    render_system_prompt,
    render_task_user_content,
)
from ..rollout.tool_response import MODEL_TOOL_RESPONSE_VERSION, serialize_model_tool_response
from .ordering import order_tool_arguments, order_tool_schemas
from .rebuild import _SingleTaskStore

PROGRAMMATIC_POLICY_VERSION = "travelweaver-programmatic-policy-v12"
SAMPLE_FAMILIES = ("efficient_success", "loop_recovery", "evidence_ready_submit")


@dataclass(frozen=True)
class ProgrammaticBuildConfig:
    task_dir: Path
    output_path: Path
    audit_path: Path
    seed: int
    concurrency: int = min(32, os.cpu_count() or 1)

    def __post_init__(self) -> None:
        if self.concurrency <= 0:
            raise ValueError("Programmatic trajectory concurrency must be positive.")


def build_programmatic_trajectories(
    config: ProgrammaticBuildConfig,
    *,
    base_backend: Any | None = None,
) -> dict[str, Any]:
    """Build one deterministic policy trajectory for every synthesis record."""

    if config.output_path.exists() or config.audit_path.exists():
        raise SFTRebuildError("Refusing to overwrite programmatic trajectory artifacts.")
    public = _read_jsonl_index(config.task_dir / "tasks.public.jsonl")
    oracle = _read_jsonl_index(config.task_dir / "tasks.oracle.jsonl")
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((config.task_dir / "records").glob("*.json"))
    ]
    if not records:
        raise SFTRebuildError("Synthesis directory has no records.")
    families = _assign_families(records, config.seed)
    backend = base_backend if base_backend is not None else ChinaTravelBackend()

    def build(item: tuple[int, dict[str, Any]]) -> tuple[int, dict[str, Any], dict[str, Any]]:
        index, record = item
        task_id = str(record["task_spec"]["task_id"])
        return (
            index,
            *_build_one(
                record,
                public[task_id],
                oracle[task_id],
                backend,
                family=families[index],
                seed=config.seed,
                source_question_batch=config.task_dir.name,
            ),
        )

    with ThreadPoolExecutor(max_workers=config.concurrency) as executor:
        built = list(executor.map(build, enumerate(records)))
    built.sort(key=lambda item: item[0])
    trajectories = [item[1] for item in built]
    audits = [item[2] for item in built]
    _atomic_jsonl(config.output_path, trajectories)
    _atomic_jsonl(config.audit_path, audits)
    family_counts = Counter(row["sample_family"] for row in audits)
    tool_counts = Counter(
        turn["tool"] for row in audits for turn in row.get("turns", [])
    )
    return {
        "programmatic_policy_version": PROGRAMMATIC_POLICY_VERSION,
        "samples": len(trajectories),
        "families": dict(sorted(family_counts.items())),
        "tool_calls": dict(sorted(tool_counts.items())),
        "concurrency": config.concurrency,
        "all_reward_one": all(row["replay_reward"] == 1.0 for row in audits),
        "all_hard_pass": all(row["all_hard_pass"] is True for row in audits),
    }


def _assign_families(records: list[dict[str, Any]], seed: int) -> dict[int, str]:
    loop_count = len(records) // 10
    submit_count = len(records) // 10
    by_type: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        by_type.setdefault(str(record["slot"]["task_type"]), []).append(index)
    loop_quotas = _proportional_quotas(by_type, loop_count, seed, "loop")
    submit_quotas = _proportional_quotas(by_type, submit_count, seed, "submit")
    assignments = {index: "efficient_success" for index in range(len(records))}
    for task_type, indices in sorted(by_type.items()):
        ordered = sorted(
            indices,
            key=lambda index: hashlib.sha256(
                (
                    f"{seed}:family:{task_type}:"
                    f"{records[index]['slot']['days']}:"
                    f"{records[index]['slot']['scenario_profile']}:"
                    f"{records[index]['task_spec']['task_id']}"
                ).encode()
            ).hexdigest(),
        )
        loop_end = loop_quotas[task_type]
        submit_end = loop_end + submit_quotas[task_type]
        for index in ordered[:loop_end]:
            assignments[index] = "loop_recovery"
        for index in ordered[loop_end:submit_end]:
            assignments[index] = "evidence_ready_submit"
    return assignments


def _proportional_quotas(
    groups: Mapping[str, list[int]], total: int, seed: int, scope: str
) -> dict[str, int]:
    population = sum(len(indices) for indices in groups.values())
    exact = {key: total * len(indices) / population for key, indices in groups.items()}
    quotas = {key: int(value) for key, value in exact.items()}
    remainder = total - sum(quotas.values())
    order = sorted(
        groups,
        key=lambda key: (
            -(exact[key] - quotas[key]),
            hashlib.sha256(f"{seed}:{scope}:{key}".encode()).hexdigest(),
        ),
    )
    for key in order[:remainder]:
        quotas[key] += 1
    return quotas


def _build_one(
    record: dict[str, Any],
    public: dict[str, Any],
    oracle: dict[str, Any],
    base_backend: Any,
    *,
    family: str,
    seed: int,
    source_question_batch: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if family not in SAMPLE_FAMILIES:
        raise ValueError(f"Unknown sample family: {family}")
    task_id = str(public["uid"])
    scenario = ScenarioSpec.from_dict(record["scenario"])
    env = TravelWeaverEnv(
        ScenarioBackend(base_backend, scenario),
        _SingleTaskStore(public, oracle),  # type: ignore[arg-type]
        # Programmatic teachers may need more than the interactive default of
        # 50 steps to collect evidence for a multi-day plan, but must stay
        # within the already-versioned ReAct/SFT 100-step context contract.
        max_valid_steps=100,
    )
    reset = env.reset(task_id=task_id, seed=0)
    tools = order_tool_schemas(env.tool_schemas())
    system_prompt = render_system_prompt(env.max_valid_steps)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": render_task_user_content(reset.task)},
    ]
    steps: list[dict[str, Any]] = []
    masks: list[bool] = []
    mask_reasons: list[str] = []
    rationale_specs: list[dict[str, Any]] = []
    loop_injected = False
    pending_recovery = False
    witness = record["witness"]
    plan = deepcopy(witness["plan"])
    evidence = witness["evidence_bundle"]
    entities = evidence["entities"]
    activities = sorted(
        witness["plan_snapshot"]["activities"],
        key=lambda item: (item["day"], item["activity_index"]),
    )
    candidate_order = list(dict.fromkeys(str(item["candidate_id"]) for item in activities))
    activity_by_candidate = {
        str(item["candidate_id"]): item for item in activities
    }
    loop_candidates = [
        candidate_id
        for candidate_id in candidate_order
        if entities[candidate_id].get("entity_type") in {"attraction", "restaurant", "hotel"}
    ]
    loop_target = (
        loop_candidates[
            int(hashlib.sha256(f"{seed}:{task_id}:loop".encode()).hexdigest(), 16)
            % len(loop_candidates)
        ]
        if family == "loop_recovery" and loop_candidates
        else None
    )
    loop_count = 1 + (
        int(hashlib.sha256(f"{seed}:{task_id}:loop-count".encode()).hexdigest(), 16) % 3
    )
    catalogued_types: set[str] = set()
    visible_entities: dict[str, dict[str, Any]] = {}
    last_visible_place_id: str | None = None
    removed_alternative = False
    food_search_used = False
    nearby_search_used = False
    active_candidates: dict[str, str] = {}

    def candidate_context() -> tuple[int, tuple[str, ...]]:
        """Summarize evidence that has already been made visible by candidate actions."""

        purpose_order = (
            "outbound_transport",
            "return_transport",
            "attraction",
            "meal",
            "hotel",
        )
        purposes = set(active_candidates.values())
        return (
            len(active_candidates),
            tuple(_purpose_label(purpose) for purpose in purpose_order if purpose in purposes),
        )

    def execute(
        action: dict[str, Any],
        *,
        supervised: bool,
        reason: str,
        content: str,
        rationale_kind: str,
        protected_literals: tuple[str, ...] = (),
    ) -> Any:
        if not content.strip():
            raise SFTRebuildError(
                f"Programmatic ReAct action has empty rationale for {task_id}: {action}"
            )
        ordered_arguments = order_tool_arguments(action["tool"], action["arguments"], tools)
        output_action = {"tool": action["tool"], "arguments": ordered_arguments}
        call_id = f"call_programmatic_{len(steps):04d}"
        tool_call = {
            "id": call_id,
            "type": "function",
            "function": {
                "name": output_action["tool"],
                "arguments": json.dumps(
                    output_action["arguments"], ensure_ascii=False, separators=(",", ":")
                ),
            },
        }
        messages.append({"role": "assistant", "content": content, "tool_calls": [tool_call]})
        result = env.step(output_action)
        if result.info.get("valid_action") is not True:
            raise SFTRebuildError(
                f"Programmatic action failed for {task_id}: {output_action}; "
                f"{result.observation.error}"
            )
        if result.truncated:
            raise SFTRebuildError(
                f"Programmatic action budget exhausted for {task_id}: {output_action}"
            )
        model_response = serialize_model_tool_response(result)
        steps.append(
            {
                "index": len(steps),
                "api_turn": len(steps) + 1,
                "tool_call": deepcopy(tool_call),
                "action": deepcopy(output_action),
                "result": result.to_dict(),
                "model_tool_response": deepcopy(model_response),
            }
        )
        masks.append(supervised)
        mask_reasons.append(reason)
        rationale_specs.append(
            {
                "rationale_kind": rationale_kind,
                "protected_literals": list(protected_literals),
                "template_rationale": content,
            }
        )
        if output_action["tool"] == "save_candidate":
            entity_id = str(output_action["arguments"]["entity_id"])
            active_candidates[entity_id] = str(output_action["arguments"]["purpose"])
        elif output_action["tool"] == "remove_candidate":
            active_candidates.pop(str(output_action["arguments"]["candidate_id"]), None)
        tool_result = result.observation.tool_result or {}
        for item in tool_result.get("items", []):
            if not isinstance(item, Mapping):
                continue
            entity_id = item.get("place_id") or item.get("transport_id")
            if isinstance(entity_id, str):
                visible_entities[entity_id] = dict(item)
            for anchor_key in ("origin_anchor", "destination_anchor"):
                anchor = item.get(anchor_key)
                if isinstance(anchor, Mapping) and isinstance(anchor.get("place_id"), str):
                    visible_entities[str(anchor["place_id"])] = dict(anchor)
        inspected = tool_result.get("item")
        if isinstance(inspected, Mapping) and isinstance(inspected.get("place_id"), str):
            visible_entities[str(inspected["place_id"])] = dict(inspected)
        if not result.terminated and not result.truncated:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": output_action["tool"],
                    "content": json.dumps(
                        model_response, ensure_ascii=False, separators=(",", ":")
                    ),
                }
            )
        return result

    def base_supervision() -> tuple[bool, str]:
        if family == "evidence_ready_submit":
            return False, "teacher_forced_evidence_prefix"
        return True, "supervised_correct_action"

    def execute_catalog(tool: str, city: str, label: str) -> None:
        supervised, reason = base_supervision()
        candidate_count, candidate_purposes = candidate_context()
        execute(
            {"tool": tool, "arguments": {"city": city}},
            supervised=supervised,
            reason=reason,
            content=_catalog_rationale(
                seed,
                task_id,
                position=len(steps),
                city=city,
                label=label,
                tool=tool,
                candidate_count=candidate_count,
                candidate_purposes=candidate_purposes,
            ),
            rationale_kind="discover_catalog_facets",
            protected_literals=(city, label),
        )

    def maybe_catalog(entity: Mapping[str, Any], entity_type: str) -> None:
        if entity_type in catalogued_types:
            return
        catalogued_types.add(entity_type)
        city = str(entity["city"])
        if entity_type == "attraction":
            execute_catalog("list_attraction_categories", city, "景点类别")
        elif entity_type == "restaurant":
            execute_catalog("list_restaurant_cuisines", city, "餐厅菜系")
        elif entity_type == "hotel":
            execute_catalog("list_hotel_features", city, "酒店特色和房型")

    def broad_search(entity: Mapping[str, Any], entity_type: str) -> dict[str, Any]:
        arguments: dict[str, Any] = {"city": entity["city"]}
        if entity_type == "attraction" and entity.get("category"):
            arguments["category"] = entity["category"]
        elif entity_type == "restaurant" and entity.get("cuisine"):
            arguments["cuisine"] = entity["cuisine"]
        elif entity_type == "hotel" and entity.get("hotel_type"):
            arguments["hotel_type"] = entity["hotel_type"]
            if entity.get("room_type") is not None:
                arguments["room_type"] = int(entity["room_type"])
        return {
            "tool": {
                "attraction": "search_attractions",
                "restaurant": "search_restaurants",
                "hotel": "search_hotels",
            }[entity_type],
            "arguments": arguments,
        }

    def preview(action: Mapping[str, Any]) -> list[dict[str, Any]]:
        method = getattr(env.backend, str(action["tool"]))
        raw = method(**dict(action["arguments"]))
        return [dict(item) for item in raw] if isinstance(raw, list) else []

    def search_strategy(
        candidate_id: str,
        entity: Mapping[str, Any],
        entity_type: str,
    ) -> tuple[dict[str, Any], str, tuple[str, ...], bool]:
        if entity_type in {"train", "airplane"}:
            mode = "火车" if entity_type == "train" else "飞机"
            earliest = _coarse_departure(str(entity["departure_time"]))
            return (
                {
                    "tool": "search_intercity_transport",
                    "arguments": {
                        "origin_city": entity["origin_city"],
                        "destination_city": entity["destination_city"],
                        "mode": entity_type,
                        "earliest_departure": earliest,
                    },
                },
                _search_rationale(
                    seed,
                    task_id,
                    position=len(steps),
                    entity=entity,
                    entity_type=entity_type,
                ),
                (str(entity["origin_city"]), str(entity["destination_city"]), mode),
                False,
            )

        entity_name = _entity_name(entity)
        if entity_name in str(public["query"]):
            action = {
                "tool": {
                    "attraction": "search_attractions",
                    "restaurant": "search_restaurants",
                    "hotel": "search_hotels",
                }[entity_type],
                "arguments": {"city": entity["city"], "query": entity_name},
            }
            return (
                action,
                _search_rationale(
                    seed,
                    task_id,
                    position=len(steps),
                    entity=entity,
                    entity_type=entity_type,
                ),
                _search_literals(entity, entity_type),
                True,
            )

        maybe_catalog(entity, entity_type)
        broad = broad_search(entity, entity_type)
        facet_key = {
            "attraction": "category",
            "restaurant": "cuisine",
            "hotel": "hotel_type",
        }[entity_type]
        facet = str(broad["arguments"].get(facet_key) or "当前条件")
        noun = {"attraction": "景点", "restaurant": "餐厅", "hotel": "酒店"}[
            entity_type
        ]
        return (
            broad,
            _facet_search_rationale(
                seed,
                task_id,
                position=len(steps),
                city=str(entity["city"]),
                facet=facet,
                noun=noun,
                entity_type=entity_type,
            ),
            (str(entity["city"]), facet, noun),
            False,
        )

    def grouped_broad_search_candidates(
        candidate_id: str,
        entity_type: str,
        action: Mapping[str, Any],
    ) -> set[str]:
        """Return future plan candidates exposed by this already-grounded search.

        A facet catalog is a public source of a broad local search condition.  Once
        that condition has been selected, paging through it again for every later
        itinerary item is both unnatural and needlessly expensive.  The teacher
        therefore keeps collecting the *same visible result stream* until all
        later plan slots with the identical public search condition have appeared.
        No name, ID, food, price, or other hidden witness value is sent to a tool.
        """

        if entity_type not in {"attraction", "restaurant", "hotel"}:
            return {candidate_id}
        if "query" in action["arguments"]:
            return {candidate_id}
        grouped: set[str] = set()
        expected = {
            "tool": str(action["tool"]),
            "arguments": dict(action["arguments"]),
        }
        for other_id in candidate_order:
            if other_id in visible_entities:
                continue
            other = entities[other_id]
            other_type = str(other.get("entity_type") or other.get("mode"))
            if other_type != entity_type or _entity_name(other) in str(public["query"]):
                continue
            other_action = broad_search(other, entity_type)
            if other_action == expected:
                grouped.add(other_id)
        return grouped or {candidate_id}

    def public_search_scope(action: Mapping[str, Any], entity_type: str) -> str:
        """Describe a query using only parameters available in the current turn."""

        arguments = action["arguments"]
        if action["tool"] == "search_intercity_transport":
            mode = "火车" if arguments.get("mode") == "train" else "飞机"
            return (
                f"从{arguments['origin_city']}到{arguments['destination_city']}的{mode}班次"
            )
        city = str(arguments.get("city", "当地"))
        if isinstance(arguments.get("query"), str):
            return str(arguments["query"])
        facet_key = {
            "attraction": "category",
            "restaurant": "cuisine",
            "hotel": "hotel_type",
        }.get(entity_type)
        facet = str(arguments.get(facet_key, "")) if facet_key is not None else ""
        noun = {"attraction": "景点", "restaurant": "餐厅", "hotel": "酒店"}.get(
            entity_type, "候选"
        )
        return f"{city}的{facet}{noun}" if facet else f"{city}的{noun}"

    def try_nearby_discovery(
        candidate_id: str,
        entity: Mapping[str, Any],
        entity_type: str,
    ) -> bool:
        nonlocal nearby_search_used
        if (
            nearby_search_used
            or last_visible_place_id is None
            or candidate_id == loop_target
            or _bucket(seed, task_id, "nearby-discovery", 4) != 0
            or entity_type
            != ("attraction", "restaurant", "hotel")[
                _bucket(seed, task_id, "nearby-category", 3)
            ]
        ):
            return False
        radii = (2, 5, 10, 20, 50)
        eligible_radius: int | None = None
        for radius in radii:
            action = {
                "tool": "search_nearby",
                "arguments": {
                    "place_id": last_visible_place_id,
                    "category": entity_type,
                    "radius_km": radius,
                    "top_k": 40,
                },
            }
            target_index = _item_index(preview(action), candidate_id)
            if 0 <= target_index < 40:
                eligible_radius = radius
                break
        if eligible_radius is None:
            return False
        nearby_search_used = True
        supervised, reason = base_supervision()
        anchor_name = _entity_name(visible_entities[last_visible_place_id])
        noun = {"attraction": "景点", "restaurant": "餐厅", "hotel": "酒店"}[
            entity_type
        ]
        for radius in radii:
            action = {
                "tool": "search_nearby",
                "arguments": {
                    "place_id": last_visible_place_id,
                    "category": entity_type,
                    "radius_km": radius,
                    "top_k": 40,
                },
            }
            result = execute(
                action,
                supervised=supervised,
                reason=reason,
                content=f"从{anchor_name}周边{radius}公里开始查看{noun}候选，优先寻找衔接方便的选择。",
                rationale_kind="search_nearby_evidence",
                protected_literals=(anchor_name, noun, str(radius)),
            )
            nearby_page_number = 1
            while candidate_id not in result.observation.visible_entity_ids:
                cursor = (result.observation.tool_result or {}).get("page", {}).get(
                    "next_cursor"
                )
                if not isinstance(cursor, str):
                    break
                result = execute(
                    {"tool": "next_page", "arguments": {"cursor": cursor}},
                    supervised=supervised,
                    reason=reason,
                    content=_nearby_page_rationale(
                        seed,
                        task_id,
                        position=len(steps),
                        anchor_name=anchor_name,
                        radius_km=radius,
                        noun=noun,
                        page_number=nearby_page_number + 1,
                    ),
                    rationale_kind="continue_nearby_search",
                    protected_literals=(anchor_name, str(radius), noun),
                )
                nearby_page_number += 1
            if candidate_id in result.observation.visible_entity_ids:
                return True
            if radius == eligible_radius:
                break
        return False

    def maybe_food_comparison(candidate_id: str, entity_type: str) -> None:
        nonlocal food_search_used
        if (
            food_search_used
            or entity_type != "restaurant"
            or _bucket(seed, task_id, "food-comparison", 5) != 0
        ):
            return
        visible = visible_entities.get(candidate_id, {})
        food = _first_facet(str(visible.get("recommended_food") or ""))
        if not food:
            return
        supervised, reason = base_supervision()
        execute(
            {
                "tool": "search_restaurants_by_food",
                "arguments": {"city": visible["city"], "food": food},
            },
            supervised=supervised,
            reason=reason,
            content=(
                f"刚才的候选信息显示{_entity_name(visible)}推荐{food}，"
                "再按这道菜查看同城餐厅，比较是否有更合适的选择。"
            ),
            rationale_kind="compare_restaurants_by_visible_food",
            protected_literals=(_entity_name(visible), food, str(visible["city"])),
        )
        food_search_used = True

    def maybe_compare_and_remove(
        candidate_id: str,
        entity_type: str,
        purpose: str,
    ) -> None:
        nonlocal removed_alternative
        if (
            removed_alternative
            or entity_type not in {"attraction", "restaurant", "hotel"}
            or _bucket(seed, task_id, "candidate-comparison", 6) != 0
        ):
            return
        target = visible_entities.get(candidate_id, {})
        target_price = _numeric_price(target.get("price"))
        if target_price is None:
            return
        alternatives = [
            item
            for item_id, item in visible_entities.items()
            if item_id not in candidate_order
            and item.get("entity_type") == entity_type
            and (price := _numeric_price(item.get("price"))) is not None
            and price > target_price
        ]
        if not alternatives:
            return
        alternative = max(
            alternatives,
            key=lambda item: (
                _numeric_price(item.get("price")) or 0,
                str(item.get("place_id", "")),
            ),
        )
        alternative_id = str(alternative["place_id"])
        alternative_name = _entity_name(alternative)
        alternative_price = _numeric_price(alternative["price"])
        assert alternative_price is not None
        supervised, reason = base_supervision()
        execute(
            {"tool": "inspect_place", "arguments": {"place_id": alternative_id}},
            supervised=supervised,
            reason=reason,
            content=f"先查看备选{alternative_name}的完整信息，再决定是否纳入行程。",
            rationale_kind="inspect_alternative",
            protected_literals=(alternative_name,),
        )
        execute(
            {
                "tool": "save_candidate",
                "arguments": {"entity_id": alternative_id, "purpose": purpose},
            },
            supervised=supervised,
            reason=reason,
            content=(
                f"{alternative_name}和已保存的{_entity_name(target)}属于同类候选，"
                f"先作为{_purpose_label(purpose)}备选保存，随后统一比较。"
            ),
            rationale_kind="save_alternative",
            protected_literals=(
                alternative_name,
                _entity_name(target),
                _purpose_label(purpose),
            ),
        )
        candidate_count, candidate_purposes = candidate_context()
        execute(
            {"tool": "list_candidates", "arguments": {}},
            supervised=supervised,
            reason=reason,
            content=_candidate_review_rationale(
                seed,
                task_id,
                position=len(steps),
                review_kind="compare",
                candidate_count=candidate_count,
                candidate_purposes=candidate_purposes,
                comparison_names=(alternative_name, _entity_name(target)),
            ),
            rationale_kind="review_candidates",
            protected_literals=(alternative_name, _entity_name(target)),
        )
        execute(
            {"tool": "remove_candidate", "arguments": {"candidate_id": alternative_id}},
            supervised=supervised,
            reason=reason,
            content=(
                f"清单显示{alternative_name}价格为{_format_price(alternative_price)}元，"
                f"高于{_entity_name(target)}的{_format_price(target_price)}元；"
                "在同类候选中没有成本优势，因此将它移除。"
            ),
            rationale_kind="remove_alternative",
            protected_literals=(
                alternative_name,
                _entity_name(target),
                _format_price(alternative_price),
                _format_price(target_price),
            ),
        )
        removed_alternative = True

    def reveal_and_save(candidate_id: str) -> None:
        nonlocal loop_injected, pending_recovery, last_visible_place_id
        entity = entities[candidate_id]
        entity_type = str(entity.get("entity_type") or entity.get("mode"))
        if entity_type in {"train", "airplane"}:
            purpose = (
                "outbound_transport"
                if entity["origin_city"] == public["start_city"]
                else "return_transport"
            )
        else:
            purpose = {"attraction": "attraction", "restaurant": "meal", "hotel": "hotel"}[
                entity_type
            ]
        supervised, reason = base_supervision()
        entity_name = _entity_name(entity)
        discovered_nearby = False
        if entity_type in {"attraction", "restaurant", "hotel"} and (
            candidate_id not in visible_entities
        ):
            discovered_nearby = try_nearby_discovery(
                candidate_id, entity, entity_type
            )
        if not discovered_nearby and (
            candidate_id not in visible_entities
            or (candidate_id == loop_target and not loop_injected)
        ):
            search, search_content, search_literals, name_grounded = search_strategy(
                candidate_id, entity, entity_type
            )
            result = execute(
                search,
                supervised=supervised,
                reason=reason,
                content=search_content,
                rationale_kind="search_evidence",
                protected_literals=search_literals,
            )
            grouped_ids = grouped_broad_search_candidates(candidate_id, entity_type, search)
            page_scope = public_search_scope(search, entity_type)
            if candidate_id == loop_target and not loop_injected:
                loop_label = (
                    entity_name
                    if candidate_id in result.observation.visible_entity_ids or name_grounded
                    else _entity_type_label(entity_type)
                )
                for attempt in range(loop_count):
                    execute(
                        search,
                        supervised=False,
                        reason="injected_loop",
                        content=_loop_action_rationale(
                            seed,
                            task_id,
                            position=len(steps),
                            entity_name=loop_label,
                            city=str(search["arguments"].get("city", "")),
                            attempt=attempt,
                        ),
                        rationale_kind="injected_loop",
                        protected_literals=(loop_label,),
                    )
                loop_injected = True
                pending_recovery = True
            page_number = 1
            while not grouped_ids.issubset(visible_entities):
                page = (result.observation.tool_result or {}).get("page", {})
                cursor = page.get("next_cursor")
                if not isinstance(cursor, str):
                    raise SFTRebuildError(
                        "Search did not expose all witness entities sharing a public "
                        f"condition: {sorted(grouped_ids - set(visible_entities))}."
                    )
                page_label = entity_name if name_grounded else _entity_type_label(entity_type)
                content = _page_rationale(
                    seed,
                    task_id,
                    position=len(steps),
                    entity_name=page_label,
                    public_scope=page_scope,
                    page_number=page_number + 1,
                    pages_checked=page_number,
                    collecting_group=len(grouped_ids) > 1,
                )
                rationale_kind = "continue_search"
                is_recovery = False
                if pending_recovery:
                    content = _loop_reflection(
                        seed,
                        task_id,
                        entity_name=page_label,
                        target_visible=False,
                        search_scope=page_scope,
                    )
                    pending_recovery = False
                    rationale_kind = "loop_recovery"
                    is_recovery = True
                result = execute(
                    {"tool": "next_page", "arguments": {"cursor": cursor}},
                    supervised=supervised,
                    reason=("supervised_loop_exit_reflection" if is_recovery else reason),
                    content=content,
                    rationale_kind=rationale_kind,
                    protected_literals=(page_label,),
                )
                page_number += 1
        if candidate_id not in visible_entities:
            raise SFTRebuildError(f"Witness entity was not made visible: {candidate_id}.")
        maybe_food_comparison(candidate_id, entity_type)
        save_content = _save_rationale(
            seed,
            task_id,
            position=len(steps),
            entity_name=entity_name,
            purpose=purpose,
        )
        save_kind = "save_evidence"
        action_reason = reason
        if pending_recovery:
            save_content = _loop_reflection(
                seed,
                task_id,
                entity_name=str(entity.get("name", "目标候选")),
                target_visible=True,
            )
            pending_recovery = False
            action_reason = "supervised_loop_exit_reflection"
            save_kind = "loop_recovery"

        def save() -> None:
            execute(
                {
                    "tool": "save_candidate",
                    "arguments": {"entity_id": candidate_id, "purpose": purpose},
                },
                supervised=supervised,
                reason=action_reason,
                content=save_content,
                rationale_kind=save_kind,
                protected_literals=(
                    (entity_name, _purpose_label(purpose))
                    if save_kind == "save_evidence"
                    else (entity_name,)
                ),
            )

        if save_kind == "loop_recovery":
            save()
        if entity_type in {"attraction", "restaurant", "hotel"} and (
            _bucket(seed, task_id, f"inspect:{candidate_id}", 3) == 0
            or candidate_id == loop_target
        ):
            execute(
                {"tool": "inspect_place", "arguments": {"place_id": candidate_id}},
                supervised=supervised,
                reason=reason,
                content=f"在确定安排前查看{entity_name}的完整快照，核对价格、类型和时间信息。",
                rationale_kind="inspect_evidence",
                protected_literals=(entity_name,),
            )
        activity = activity_by_candidate[candidate_id]
        if entity_type in {"attraction", "restaurant"} and _bucket(
            seed, task_id, f"open:{candidate_id}", 2
        ) == 0:
            at_time = str(activity["start_time"])
            open_preview = env.backend.check_place_open(candidate_id, at_time)
            if open_preview.get("is_open") is True:
                execute(
                    {
                        "tool": "check_place_open",
                        "arguments": {"place_id": candidate_id, "at_time": at_time},
                    },
                    supervised=supervised,
                    reason=reason,
                    content=f"计划在{at_time}使用{entity_name}，现在单独核对该时刻是否开放。",
                    rationale_kind="check_open_evidence",
                    protected_literals=(entity_name, at_time),
                )
        if save_kind != "loop_recovery":
            save()
        if entity_type in {"attraction", "restaurant", "hotel"}:
            last_visible_place_id = candidate_id
            maybe_compare_and_remove(candidate_id, entity_type, purpose)
        elif purpose == "outbound_transport":
            transport = visible_entities.get(candidate_id, {})
            anchor_id = transport.get("destination_anchor_id")
            if isinstance(anchor_id, str) and anchor_id in visible_entities:
                last_visible_place_id = anchor_id

    try:
        for candidate_id in candidate_order:
            reveal_and_save(candidate_id)
        if family != "loop_recovery" and _bucket(
            seed, task_id, "candidate-review", 4
        ) == 0:
            supervised, reason = base_supervision()
            candidate_count, candidate_purposes = candidate_context()
            execute(
                {"tool": "list_candidates", "arguments": {}},
                supervised=supervised,
                reason=reason,
                content=_candidate_review_rationale(
                    seed,
                    task_id,
                    position=len(steps),
                    review_kind="coverage",
                    candidate_count=candidate_count,
                    candidate_purposes=candidate_purposes,
                ),
                rationale_kind="review_candidates",
            )
        route_order = list(
            dict.fromkeys(
                str(item["route_from_previous_id"])
                for item in activities
                if item.get("route_from_previous_id") is not None
            )
        )
        for route_id in route_order:
            route = evidence["routes"][route_id]
            first_segment = route["segments"][0]
            supervised, reason = base_supervision()
            execute(
                {
                    "tool": "get_route",
                    "arguments": {
                        "origin_place_id": route["origin_place_id"],
                        "destination_place_id": route["destination_place_id"],
                        "mode": route["mode"],
                        "start_time": first_segment["start_time"],
                    },
                },
                supervised=supervised,
                reason=reason,
                content=_route_rationale(
                    seed,
                    task_id,
                    position=len(steps),
                    origin_name=str(first_segment["start"]),
                    destination_name=str(route["segments"][-1]["end"]),
                    mode=str(route["mode"]),
                    start_time=str(first_segment["start_time"]),
                ),
                rationale_kind="complete_route_evidence",
                protected_literals=(
                    str(first_segment["start"]),
                    str(route["segments"][-1]["end"]),
                    _route_mode_label(str(route["mode"])),
                    str(first_segment["start_time"]),
                ),
            )
        kinds = {str(item["activity_type"]) for item in activities}
        evidence_names = ["往返交通", "景点", "完整路线"]
        if "accommodation" in kinds:
            evidence_names.append("住宿")
        if kinds & {"breakfast", "lunch", "dinner"}:
            evidence_names.append("餐饮")
        candidate_count, _candidate_purposes = candidate_context()
        submit_landmarks, submit_literal_names = _submit_landmarks(
            active_candidates,
            visible_entities,
        )
        submit_content = _submit_reflection(
            seed,
            task_id,
            evidence_names,
            candidate_count=candidate_count,
            days=int(public["days"]),
            route_count=len(route_order),
            evidence_landmarks=submit_landmarks,
        )
        submit_reason = "supervised_correct_action"
        if family == "evidence_ready_submit":
            submit_reason = "supervised_evidence_ready_reflection"
        terminal = execute(
            {"tool": "submit_plan", "arguments": {"plan": plan}},
            supervised=True,
            reason=submit_reason,
            content=submit_content,
            rationale_kind="evidence_ready_submit",
            protected_literals=tuple(evidence_names) + submit_literal_names,
        )
        detail = terminal.info.get("reward_detail")
        if (
            not terminal.terminated
            or terminal.info.get("termination_reason") != "plan_submitted"
            or terminal.reward != 1.0
            or not isinstance(detail, Mapping)
            or detail.get("all_hard_pass") is not True
        ):
            raise SFTRebuildError(f"Programmatic replay failed for {task_id}: {detail}")
    finally:
        env.close()

    row = {
        "trajectory_version": TRAJECTORY_VERSION,
        "episode_id": reset.episode_id,
        "task_id": task_id,
        "model": "deterministic-programmatic-teacher",
        "success": True,
        "termination_reason": "plan_submitted",
        "step_count": len(steps),
        "api_turn_count": len(steps),
        "final_plan": plan,
        "final_text": None,
        "final_reward": 1.0,
        "reward_detail": dict(detail),
        "rft_accepted": True,
        "usage": {},
        "user_content_format": USER_CONTENT_FORMAT,
        "tool_response_mode": "delta",
        "model_tool_response_version": MODEL_TOOL_RESPONSE_VERSION,
        "messages": messages,
        "tools": tools,
        "steps": steps,
        "assistant_loss_mask": masks,
        "mask_reasons": mask_reasons,
        "sample_family": family,
        "batch_metadata": {"thinking": "disabled", "source": PROGRAMMATIC_POLICY_VERSION},
    }
    audit = {
        "programmatic_policy_version": PROGRAMMATIC_POLICY_VERSION,
        "task_id": task_id,
        "sample_family": family,
        "question_batch": source_question_batch,
        "blueprint_semantic_hash": record["blueprint"]["blueprint_id"],
        "assistant_loss_mask": masks,
        "mask_reasons": mask_reasons,
        "turns": [
            {
                "position": index,
                "tool": step["action"]["tool"],
                "loss_mask": masks[index],
                "mask_reason": mask_reasons[index],
                "visible_reflection": messages[2 + index * 2].get("content", ""),
                **rationale_specs[index],
            }
            for index, step in enumerate(steps)
        ],
        "loop_tool": next(
            (
                step["action"]["tool"]
                for step, reason in zip(steps, mask_reasons, strict=True)
                if reason == "injected_loop"
            ),
            None,
        ),
        "loop_positions": [
            index for index, reason in enumerate(mask_reasons) if reason == "injected_loop"
        ],
        "loop_count": sum(reason == "injected_loop" for reason in mask_reasons),
        "alternative_removed": removed_alternative,
        "first_evidence_ready_position": len(steps) - 1,
        "action_count": len(steps),
        "termination_reason": "plan_submitted",
        "replay_reward": 1.0,
        "all_hard_pass": True,
        "reward_groups": dict(detail.get("group_results", {})),
    }
    return row, audit


def _loop_reflection(
    seed: int,
    task_id: str,
    *,
    entity_name: str,
    target_visible: bool,
    search_scope: str = "",
) -> str:
    if target_visible:
        templates = (
            "{name}已经出现在候选中，重复搜索不会补充新证据；现在保存它并继续完善行程。",
            "刚才的查询结果足以确认{name}，无需再查同一页；下一步保存候选。",
            "已经定位到{name}，继续重复检索没有必要；接下来保存该候选并处理剩余证据。",
        )
    else:
        templates = (
            "重复当前搜索没有带来新结果；接下来查看后续候选，继续定位{name}。",
            "同一页结果已经核对过，无需再次查询；现在继续翻页查找{name}。",
            "再次搜索仍是相同候选列表；下一步应查看后续结果并找到{name}。",
            "{scope}这一页已反复返回相同结果，继续查同一页无法补充{name}；现在翻到下一页。",
            "针对{scope}的重复查询没有新增候选，接下来沿用原条件查看后续页面并定位{name}。",
            "已确认{scope}当前结果不会变化；停止重复查询，继续读取下一页寻找{name}。",
        )
    index = int(hashlib.sha256(f"{seed}:{task_id}:loop-text".encode()).hexdigest(), 16)
    return templates[index % len(templates)].format(
        name=entity_name,
        scope=search_scope or "本轮搜索",
    )


def _entity_name(entity: Mapping[str, Any]) -> str:
    return str(entity.get("name") or entity.get("source_id") or "目标候选")


def _choice(
    templates: tuple[str, ...], seed: int, task_id: str, position: int, scope: str
) -> str:
    digest = hashlib.sha256(f"{seed}:{task_id}:{position}:{scope}".encode()).hexdigest()
    return templates[int(digest, 16) % len(templates)]


def _cycle_choice(
    templates: tuple[str, ...],
    seed: int,
    task_id: str,
    *,
    scope: str,
    ordinal: int,
) -> str:
    """Choose a stable template stream without repeating a form in one search.

    ``ordinal`` is local to a single paged/looped operation.  Hashing only its
    initial offset preserves diversity across tasks while cycling subsequent
    pages or loop attempts through different natural phrasings.
    """

    digest = hashlib.sha256(f"{seed}:{task_id}:{scope}".encode()).hexdigest()
    return templates[(int(digest, 16) + ordinal) % len(templates)]


def _bucket(seed: int, task_id: str, scope: str, modulo: int) -> int:
    digest = hashlib.sha256(f"{seed}:{task_id}:{scope}".encode()).hexdigest()
    return int(digest, 16) % modulo


def _item_index(items: list[dict[str, Any]], candidate_id: str) -> int:
    for index, item in enumerate(items):
        entity_id = item.get("place_id") or item.get("transport_id")
        if entity_id == candidate_id:
            return index
    return -1


def _first_facet(value: str) -> str:
    return next(
        (part.strip() for part in re.split(r"[,，、|/]", value) if part.strip()),
        value.strip(),
    )


def _coarse_departure(value: str) -> str:
    hour = int(value[:2])
    return f"{hour - hour % 3:02d}:00"


def _numeric_price(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _format_price(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:.2f}".rstrip("0")


def _entity_type_label(entity_type: str) -> str:
    return {
        "attraction": "合适的景点",
        "restaurant": "合适的餐厅",
        "hotel": "合适的酒店",
        "train": "合适的火车班次",
        "airplane": "合适的航班",
    }[entity_type]


def _search_literals(entity: Mapping[str, Any], entity_type: str) -> tuple[str, ...]:
    if entity_type in {"train", "airplane"}:
        mode = "火车" if entity_type == "train" else "飞机"
        return (str(entity["origin_city"]), str(entity["destination_city"]), mode)
    noun = {"attraction": "景点", "restaurant": "餐厅", "hotel": "酒店"}[entity_type]
    return (_entity_name(entity), noun)


def _search_rationale(
    seed: int,
    task_id: str,
    *,
    position: int,
    entity: Mapping[str, Any],
    entity_type: str,
) -> str:
    if entity_type in {"train", "airplane"}:
        mode = "火车" if entity_type == "train" else "飞机"
        template = _choice(
            (
                "先查询从{origin}到{destination}的{mode}班次，核实这段城际交通的可用时间和候选。",
                "当前需要补充{origin}前往{destination}的城际交通证据，先检索可用的{mode}班次。",
                "为了安排{origin}到{destination}这一程，先查询{mode}候选及其准确时刻。",
            ),
            seed,
            task_id,
            position,
            "transport-search",
        )
        return template.format(
            origin=entity["origin_city"], destination=entity["destination_city"], mode=mode
        )
    noun = {"attraction": "景点", "restaurant": "餐厅", "hotel": "酒店"}[entity_type]
    template = _choice(
        (
            "接下来查询{city}的{name}，确认这个{noun}是否能为计划提供可用证据。",
            "行程还需要核实{name}，先在{city}的{noun}候选中检索它。",
            "为了完善本地安排，先查询{city}的{name}并取得准确的{noun}信息。",
        ),
        seed,
        task_id,
        position,
        "place-search",
    )
    return template.format(
        city=entity["city"], name=_entity_name(entity), noun=noun
    )


def _save_rationale(
    seed: int,
    task_id: str,
    *,
    position: int,
    entity_name: str,
    purpose: str,
) -> str:
    purpose_text = _purpose_label(purpose)
    template = _choice(
        (
            "查询结果中已经找到{name}，它会作为最终计划的{purpose}，现在保存这项候选证据。",
            "{name}符合当前要补充的{purpose}，后续提交会引用它，因此先保存候选。",
            "已经取得{name}的有效信息，接下来把它保存为本次行程的{purpose}证据。",
        ),
        seed,
        task_id,
        position,
        "save",
    )
    return template.format(name=entity_name, purpose=purpose_text)


def _purpose_label(purpose: str) -> str:
    return {
        "outbound_transport": "去程交通",
        "return_transport": "返程交通",
        "attraction": "景点",
        "meal": "用餐地点",
        "hotel": "住宿",
    }[purpose]


def _submit_landmarks(
    active_candidates: Mapping[str, str],
    visible_entities: Mapping[str, Mapping[str, Any]],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Select a few already-saved entities for a grounded submit summary.

    The final reflection should be specific enough to sound like it follows the
    episode, but it must not reveal a witness entity that the model has not
    observed.  ``active_candidates`` only contains save_candidate results that
    remain after any remove_candidate correction, and every returned name came
    from an earlier tool observation.
    """

    labels = (
        ("outbound_transport", "去程"),
        ("attraction", "景点"),
        ("meal", "用餐"),
        ("hotel", "住宿"),
        ("return_transport", "返程"),
    )
    landmarks: list[str] = []
    literal_names: list[str] = []
    for purpose, label in labels:
        candidate_id = next(
            (entity_id for entity_id, saved_purpose in active_candidates.items()
             if saved_purpose == purpose),
            None,
        )
        if candidate_id is None:
            continue
        entity = visible_entities.get(candidate_id)
        if entity is None:
            continue
        name = _entity_name(entity)
        landmarks.append(f"{label}{name}")
        literal_names.append(name)
    return tuple(landmarks[:3]), tuple(literal_names[:3])


def _page_rationale(
    seed: int,
    task_id: str,
    *,
    position: int,
    entity_name: str,
    public_scope: str,
    page_number: int,
    pages_checked: int,
    collecting_group: bool,
) -> str:
    if collecting_group:
        templates = (
            "{scope}当前结果还不够完整，继续查看第{page}页，避免后续重复从头检索。",
            "同一筛选条件下还需补充更多{scope}候选，现在翻到第{page}页继续收集。",
            "为了让后续安排复用这次{scope}检索，继续浏览第{page}页结果。",
            "已查看前{checked}页{scope}结果，继续读取第{page}页以补充同条件候选。",
            "不改变已确认的{scope}筛选，继续展开第{page}页，收集可比较的选择。",
            "当前搜索流仍有下一页，先查看第{page}页{scope}候选，再决定后续安排。",
            "前{checked}页已提供部分{scope}选择，继续核对第{page}页，避免遗漏同条件候选。",
            "继续沿用当前城市和筛选条件，打开第{page}页{scope}结果。",
        )
        scope = "group-page"
    else:
        templates = (
            "{scope}当前结果里还没有出现{entity}，继续查看第{page}页候选。",
            "还需在{scope}中定位{entity}，因此翻到第{page}页继续检索。",
            "现有{scope}候选尚未找到{entity}，下一步查看第{page}页。",
            "已核对前{checked}页{scope}结果，{entity}仍未出现；继续进入第{page}页。",
            "前{checked}页结束后还缺{entity}，保持{scope}条件继续查第{page}页。",
            "本次{scope}搜索仍有下一页；为了找到{entity}，继续读取第{page}页。",
            "当前筛选条件不变，前{checked}页{scope}结果暂未定位{entity}，继续检索第{page}页候选。",
            "已完成{scope}前{checked}页的核对，下一页继续寻找{entity}。",
        )
        scope = "page"
    template = _cycle_choice(
        templates,
        seed,
        task_id,
        scope=f"{scope}:{public_scope}",
        ordinal=page_number - 1,
    )
    return template.format(
        entity=entity_name,
        scope=public_scope,
        page=page_number,
        checked=pages_checked,
    )


def _catalog_rationale(
    seed: int,
    task_id: str,
    *,
    position: int,
    city: str,
    label: str,
    tool: str,
    candidate_count: int = 0,
    candidate_purposes: tuple[str, ...] = (),
) -> str:
    saved_context = (
        f"目前已保存{candidate_count}项候选（{'、'.join(candidate_purposes)}）"
        if candidate_purposes
        else "当前还没有已保存候选"
    )
    templates = (
        "先查看{city}当前可用的{label}，再据此筛选合适候选。",
        "需要先确定{city}{label}的可选范围，查看目录后再开始检索。",
        "先打开{city}的{label}目录，避免在没有依据的情况下设定筛选条件。",
        "为了让后续搜索有明确条件，先核对{city}提供哪些{label}。",
        "本地安排还没有确定筛选方向，先从{city}的{label}中了解可用选项。",
        "{context}；{label}的筛选方向尚未确定，先查看{city}目录。",
        "在已有证据的基础上，先用{city}的{label}目录确定下一次搜索条件。",
        "选择{city}候选前先获取{label}目录，保证后续筛选只使用已见条件。",
        "先把{city}{label}的可选项放入当前上下文，再按其中一个方向继续检索。",
        "本地候选的条件还不能凭空设定；先列出{city}的{label}再缩小范围。",
    )
    template = _choice(templates, seed, task_id, position, f"catalog:{tool}")
    return template.format(city=city, label=label, context=saved_context)


def _facet_search_rationale(
    seed: int,
    task_id: str,
    *,
    position: int,
    city: str,
    facet: str,
    noun: str,
    entity_type: str,
) -> str:
    templates = (
        "目录中已经确认{facet}可用，现在按这一条件查看{city}的{noun}候选。",
        "刚才的目录包含{facet}，接下来以它为条件检索{city}的{noun}。",
        "先用已看到的{facet}筛选{city}{noun}，从结果中继续判断具体安排。",
        "{city}的目录给出了{facet}这一方向，现在查看对应的{noun}列表。",
        "为了缩小{city}{noun}的范围，沿用目录中的{facet}条件进行查询。",
    )
    template = _choice(templates, seed, task_id, position, f"facet:{entity_type}")
    return template.format(city=city, facet=facet, noun=noun)


def _candidate_review_rationale(
    seed: int,
    task_id: str,
    *,
    position: int,
    review_kind: str,
    candidate_count: int | None = None,
    candidate_purposes: tuple[str, ...] = (),
    comparison_names: tuple[str, str] | None = None,
) -> str:
    if review_kind == "compare":
        if comparison_names is None:
            raise ValueError("Comparison candidate review requires two visible candidate names.")
        templates = (
            "{first}和{second}这组同类备选都已保存，先调出候选清单核对差异再取舍。",
            "当前已有{count}项候选；先在清单中比较{first}与{second}的已展示信息。",
            "现在需要在{first}和{second}之间做选择，先查看候选清单里的对应记录。",
            "先回看清单中{first}与{second}这两个同类选项，再依据已展示的信息完成取舍。",
        )
    elif review_kind == "coverage":
        if candidate_count is None or not candidate_purposes:
            raise ValueError("Coverage candidate review requires visible candidate context.")
        templates = (
            "目前已保存{count}项候选，涵盖{purposes}；先复查清单再补齐路线。",
            "在查询路线前回看这{count}项候选，确认{purposes}的证据没有遗漏。",
            "先汇总核对候选清单：已保存的{count}项是否足以覆盖{purposes}等行程环节。",
            "已有{count}项候选可供提交，先检查清单中{purposes}的证据是否完整。",
        )
    else:
        raise ValueError(f"Unknown candidate review kind: {review_kind}")
    return _choice(templates, seed, task_id, position, f"candidate-review:{review_kind}").format(
        count=candidate_count,
        purposes="、".join(candidate_purposes),
        first=comparison_names[0] if comparison_names is not None else "",
        second=comparison_names[1] if comparison_names is not None else "",
    )


def _loop_action_rationale(
    seed: int,
    task_id: str,
    *,
    position: int,
    entity_name: str,
    city: str = "",
    attempt: int = 0,
) -> str:
    query_object = entity_name.removeprefix("合适的")
    location = f"在{city}" if city else ""
    template = _cycle_choice(
        (
            "为了再确认一次{object}的候选信息，我{location}执行一次相同的查询。",
            "我再{location}按相同条件检索一次{object}，核对结果是否有变化。",
            "这里{location}再次查询{object}，尝试重新确认候选列表。",
        ),
        seed,
        task_id,
        scope=f"loop-action:{location}:{query_object}",
        ordinal=attempt,
    )
    return template.format(object=query_object, location=location)


def _nearby_page_rationale(
    seed: int,
    task_id: str,
    *,
    position: int,
    anchor_name: str,
    radius_km: int,
    noun: str,
    page_number: int,
) -> str:
    template = _choice(
        (
            "{anchor}周边{radius}公里的{noun}还需继续查看，翻到该范围的第{page}页。",
            "当前{radius}公里范围内尚未选定合适的{noun}，继续浏览{anchor}附近的第{page}页结果。",
            "沿用{anchor}周边{radius}公里这一条件，继续查看第{page}页{noun}候选。",
            "为了在{anchor}附近完成衔接，继续核对{radius}公里范围内第{page}页的{noun}。",
        ),
        seed,
        task_id,
        position,
        "nearby-page",
    )
    return template.format(
        anchor=anchor_name,
        radius=radius_km,
        noun=noun,
        page=page_number,
    )


def _route_rationale(
    seed: int,
    task_id: str,
    *,
    position: int,
    origin_name: str,
    destination_name: str,
    mode: str,
    start_time: str,
) -> str:
    mode_text = _route_mode_label(mode)
    template = _choice(
        (
            "计划在{time}从{origin}衔接到{destination}，现在查询{mode}路线并补齐这一段交通证据。",
            "接下来核实{time}从{origin}到{destination}的市内衔接，按计划查询{mode}路线。",
            "为了保证行程地点连续，按{time}出发查询从{origin}前往{destination}的{mode}路线。",
            "前一段安排结束后将于{time}离开{origin}，先确认到{destination}的{mode}路线衔接。",
        ),
        seed,
        task_id,
        position,
        "route",
    )
    return template.format(
        origin=origin_name,
        destination=destination_name,
        mode=mode_text,
        time=start_time,
    )


def _route_mode_label(mode: str) -> str:
    return {"taxi": "出租车", "metro": "地铁", "walk": "步行"}.get(mode, mode)


def _submit_reflection(
    seed: int,
    task_id: str,
    evidence_names: list[str],
    *,
    candidate_count: int | None = None,
    days: int | None = None,
    route_count: int | None = None,
    evidence_landmarks: tuple[str, ...] = (),
) -> str:
    evidence_text = "、".join(evidence_names)
    if candidate_count is None or days is None or route_count is None:
        templates = (
            "{evidence}证据均已齐全，现在可以提交完整计划。",
            "题目所需的{evidence}已经全部核实，可以直接提交方案。",
            "当前已具备完整的{evidence}证据，下一步提交最终计划。",
            "候选和衔接信息已覆盖{evidence}，没有待补的关键证据，可以提交。",
            "已逐项确认{evidence}，现在把这些已保存的证据组织为最终计划并提交。",
            "行程所需的{evidence}都已准备好，提交后方案即可接受验证。",
        )
    elif evidence_landmarks:
        templates = (
            "{landmarks}等关键候选已保存；{days}天行程的{count}项候选和{routes}段路线已覆盖{evidence}，现在提交。",
            "以{landmarks}等已见证据为基础，{days}天方案的{count}项候选和{routes}段路线均已补齐，提交。",
            "当前候选集中已有{landmarks}等安排，{days}天行程的{count}项候选连同{routes}段路线可形成完整方案，提交。",
            "{landmarks}已经分别落实到已保存候选；{days}天安排的{count}项候选和{routes}段路线覆盖{evidence}，现在提交。",
            "已核对{landmarks}等关键选择；{days}天行程的{count}项候选及{routes}段衔接没有缺口，可以提交。",
            "围绕{landmarks}等已保存候选，{days}天安排的{count}项候选和{routes}段路线已经准备完毕，提交最终计划。",
            "{landmarks}等候选均可在当前上下文中引用；{days}天计划的{count}项候选、{routes}段路线和{evidence}已齐全，提交。",
            "现在的{count}项候选包含{landmarks}等关键安排，{days}天行程需要的{routes}段路线与{evidence}已齐，提交。",
        )
    else:
        templates = (
            "{days}天行程已保存{count}项候选并核对{routes}段市内路线，{evidence}证据齐全，可以提交。",
            "围绕这{days}天安排，{count}项候选和{routes}段路线均已核实；现在提交包含{evidence}的完整计划。",
            "当前{count}项已保存候选已覆盖{evidence}，且{routes}段路线已补齐，提交这份{days}天方案。",
            "{evidence}均已落实到{count}项候选和{routes}段衔接中，没有待补的关键证据，提交{days}天行程。",
            "已为{days}天行程逐项确认{evidence}，共保留{count}项候选并完成{routes}段路线，现在提交。",
            "候选和衔接信息已准备完毕：{count}项候选、{routes}段路线覆盖{evidence}，可以提交最终方案。",
        )
    index = int(hashlib.sha256(f"{seed}:{task_id}:submit-text".encode()).hexdigest(), 16)
    return templates[index % len(templates)].format(
        evidence=evidence_text,
        count=candidate_count,
        days=days,
        routes=route_count,
        landmarks="、".join(evidence_landmarks),
    )


def _read_jsonl_index(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row["uid"])] = row
    return rows


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)
