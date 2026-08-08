"""Canonical Chinese rendering for grounded typed constraints."""

from __future__ import annotations

from typing import Any

from ..errors import SynthesisError
from ..tasks import TaskBlueprint
from .models import CanonicalTask

_MODE_ZH = {
    "train": "火车",
    "airplane": "飞机",
    "taxi": "出租车",
    "walk": "步行",
    "metro": "地铁",
    "high_speed_rail": "高铁",
}


_STYLE_FRAMES = {
    "direct": (
        "请规划一趟从{origin}到{destination}的旅行，共{days}天，同行{travelers}人。",
        "另外，",
        "请根据这些要求制定可执行的详细行程。",
    ),
    "conversational": (
        "我们共{travelers}人，想从{origin}去{destination}旅行{days}天。",
        "还希望满足这些条件：",
        "麻烦按这些条件安排一份能实际执行的行程。",
    ),
    "trip_first": (
        "从{origin}到{destination}的这趟行程计划安排{days}天，共{travelers}人参加。",
        "行程需要满足：",
        "请据此给出具体可行的安排。",
    ),
    "party_first": (
        "同行一共{travelers}人，准备从{origin}出发前往{destination}，旅行{days}天。",
        "我们的硬性要求是：",
        "请把以上条件落实到行程中。",
    ),
    "concise": (
        "需要一份{origin}到{destination}的{days}天行程，出行人数为{travelers}人。",
        "要求：",
        "请给出可执行的规划。",
    ),
    "consultant": (
        "麻烦设计一份从{origin}前往{destination}的旅行方案，共{days}天、{travelers}人。",
        "方案必须满足：",
        "请按这些条件完成具体安排。",
    ),
    "narrative": (
        "我们打算从{origin}出发去{destination}，一行{travelers}人，行程为{days}天。",
        "安排时请注意：",
        "请帮我们把行程规划得具体可行。",
    ),
    "itinerary": (
        "这次{origin}至{destination}的旅行共有{travelers}人，安排{days}天。",
        "行程条件包括：",
        "请据此设计一份可实际执行的日程。",
    ),
    "question": (
        "{travelers}人从{origin}到{destination}玩{days}天，该怎样安排行程？",
        "同时需要满足：",
        "请给出符合条件的具体方案。",
    ),
    "compact": (
        "请安排{origin}至{destination}、{days}天、{travelers}人的旅行。",
        "硬性条件为：",
        "请据此制定可执行行程。",
    ),
    "human_metadata": (
        "{persona_lead}准备从{origin}去{destination}玩{days}天，一共{travelers}人。",
        "具体想法是：",
        "请帮忙排一个实际可行的行程。",
    ),
    "human_dialogue": (
        "{persona_lead}想从{origin}到{destination}玩{days}天，我们一共{travelers}人。",
        "安排时，",
        "麻烦把行程具体安排一下。",
    ),
    "human_v1_1_metadata": (
        "想从{origin}去{destination}玩{days}天，一共{travelers}人。",
        "另外，",
        "麻烦帮忙规划一下行程。",
    ),
    "human_v1_1_dialogue": (
        "{persona_lead}想从{origin}到{destination}玩{days}天，一共{travelers}人。",
        "另外，",
        "麻烦帮忙规划一下行程。",
    ),
}


def render_canonical(
    blueprint: TaskBlueprint,
    *,
    style_profile: str = "direct",
    validation_profile: str = "strict",
) -> CanonicalTask:
    trip = blueprint.trip
    destination = trip.destinations[-1]
    try:
        core_template, constraint_prefix, closing = _STYLE_FRAMES[style_profile]
    except KeyError as error:
        raise SynthesisError(f"Unknown surface style profile: {style_profile}") from error
    core = core_template.format(
        origin=trip.origin,
        destination=destination,
        days=trip.days,
        travelers=trip.travelers,
        persona_lead=(
            ""
            if blueprint.metadata_prefix
            else f"我们是{blueprint.persona_context}，"
            if blueprint.persona_context
            else ""
        ),
    )
    if style_profile in {"human_metadata", "human_dialogue"}:
        core = _human_core(blueprint, destination)
    elif style_profile in {"human_v1_1_metadata", "human_v1_1_dialogue"}:
        core = _human_v1_1_core(blueprint, destination)
    if blueprint.metadata_prefix:
        core = blueprint.metadata_prefix + core
    clauses: dict[str, str] = {}
    protected = {
        trip.origin,
        destination,
        f"{trip.days}天",
        f"{trip.travelers}人",
    }
    for constraint in blueprint.constraints:
        clause, literals = _constraint_clause(
            constraint.kind,
            constraint.operator,
            constraint.value,
            constraint.scope,
            validation_profile,
        )
        clauses[constraint.id] = clause
        protected.update(literals)
    preference_clauses: dict[str, str] = {}
    for preference in blueprint.preferences:
        clause, literals = _preference_clause(
            preference.kind,
            preference.direction,
            preference.value,
        )
        preference_clauses[preference.id] = clause
        protected.update(literals)
    all_clauses = [*clauses.values(), *preference_clauses.values()]
    joined = constraint_prefix + "；".join(all_clauses) + "。" if all_clauses else ""
    query = core + joined + closing
    if blueprint.persona_context:
        protected.add(blueprint.persona_context)
    if blueprint.metadata_prefix:
        protected.add(blueprint.metadata_prefix)
    return CanonicalTask(
        query=query,
        clauses=clauses,
        preference_clauses=preference_clauses,
        protected_literals=tuple(sorted(protected, key=lambda value: (-len(value), value))),
    )


def _human_core(blueprint: TaskBlueprint, destination: str) -> str:
    trip = blueprint.trip
    persona = blueprint.persona_context or "出行"
    templates = (
        "我们是{persona}，打算从{origin}去{destination}玩{days}天，一共{travelers}人。",
        "这次是{persona}，{travelers}人想从{origin}到{destination}旅行{days}天。",
        "想请你帮{travelers}人安排{origin}到{destination}的{days}天旅行，我们是{persona}。",
        "有个{persona}的行程想咨询：{travelers}人从{origin}去{destination}，玩{days}天。",
        "准备来一次{persona}，共{travelers}人，从{origin}出发去{destination}{days}天。",
        "麻烦看看这个{persona}计划：{origin}到{destination}，{days}天，{travelers}人。",
        "关于{persona}想做个行程，{travelers}人由{origin}前往{destination}玩{days}天。",
        "最近计划{persona}，我们{travelers}人要从{origin}去{destination}待{days}天。",
        "请帮忙规划一次{persona}：从{origin}到{destination}，同行{travelers}人，共{days}天。",
        "行程是{origin}出发、目的地{destination}，{travelers}人的{days}天{persona}。",
    )
    template = templates[blueprint.generation_seed % len(templates)]
    return template.format(
        persona=persona,
        origin=trip.origin,
        destination=destination,
        days=trip.days,
        travelers=trip.travelers,
    )


def _human_v1_1_core(blueprint: TaskBlueprint, destination: str) -> str:
    trip = blueprint.trip
    if blueprint.metadata_prefix:
        templates = (
            "想请你帮忙规划从{origin}到{destination}的{days}天行程，我们一共{travelers}人。",
            "准备从{origin}去{destination}玩{days}天，共{travelers}人，想请你帮忙安排一下。",
            "这趟行程从{origin}出发去{destination}，一共{travelers}人，玩{days}天。",
            "我们{travelers}人计划由{origin}前往{destination}旅行{days}天。",
            "麻烦看看{origin}到{destination}怎么玩比较合适，同行{travelers}人，共{days}天。",
        )
    else:
        templates = (
            "我准备{persona}，从{origin}去{destination}玩{days}天，一共{travelers}人。",
            "这次是{persona}，我们{travelers}人想从{origin}到{destination}待{days}天。",
            "打算来一次{persona}，共{travelers}人，从{origin}出发去{destination}{days}天。",
            "最近计划{persona}，{travelers}人由{origin}前往{destination}旅行{days}天。",
            "想请你安排一次{persona}：从{origin}到{destination}，{travelers}人玩{days}天。",
        )
    template = templates[blueprint.generation_seed % len(templates)]
    return template.format(
        persona=blueprint.persona_context or "出行",
        origin=trip.origin,
        destination=destination,
        days=trip.days,
        travelers=trip.travelers,
    )


def _preference_clause(
    kind: str, direction: str, value: Any
) -> tuple[str, set[str]]:
    expected_direction, text = {
        "more_attractions": ("maximize", "希望景点尽量多一些"),
        "less_innercity_time": ("minimize", "希望少花时间在市内交通上"),
        "shorter_meal_transfer": ("minimize", "希望去吃饭的路程短一些"),
        "higher_dining_share": ("maximize", "希望餐饮支出占比高一些"),
        "lower_lodging_share": ("minimize", "希望住宿支出占比低一些"),
        "less_walking": ("minimize", "希望少走路"),
        "lower_total_cost": ("minimize", "希望总花费低一些"),
        "relaxed_itinerary": ("minimize", "希望行程轻松一些"),
        "higher_attraction_share": ("maximize", "希望景点支出占比高一些"),
        "lower_intercity_share": ("minimize", "希望城际交通支出占比低一些"),
        "shorter_total_travel_time": ("minimize", "希望整体交通时间短一些"),
    }.get(kind, (None, None))
    literals: set[str] = set()
    if kind == "near_poi":
        expected_direction = "minimize"
        if not isinstance(value, dict) or not str(value.get("poi_name", "")).strip():
            raise SynthesisError("near_poi preference has no POI name.")
        poi_name = str(value["poi_name"])
        text = f"希望住宿尽量靠近{poi_name}"
        literals.add(poi_name)
    if expected_direction is None or text is None:
        raise SynthesisError(f"No canonical renderer for preference {kind}.")
    if direction != expected_direction:
        raise SynthesisError(f"Preference {kind} has an inconsistent direction.")
    return text, literals


def _constraint_clause(
    kind: str,
    operator: str,
    value: Any,
    scope: str,
    validation_profile: str,
) -> tuple[str, set[str]]:
    if not isinstance(value, dict):
        raise SynthesisError(f"Constraint {kind} has no renderable value object.")
    if kind == "total_budget":
        amount = _format_number(value["amount"])
        return f"总预算不超过{amount}元", {f"{amount}元"}
    if kind == "transport_mode":
        modes = [_MODE_ZH[str(mode)] for mode in value["modes"]]
        mode_text = "或".join(modes)
        leg = value.get("leg", "all")
        if scope == "innercity_route":
            if mode_text == "步行":
                clause = (
                    "至少安排两个市内地点，地点之间都步行"
                    if validation_profile != "strict"
                    else "至少安排两个市内地点，地点之间必须步行"
                )
            else:
                clause = (
                    f"至少安排两个市内地点，地点之间统一坐{mode_text}"
                    if validation_profile != "strict"
                    else f"至少安排两个市内地点，地点之间必须统一使用{mode_text}"
                )
            return clause, {mode_text}
        prefix = {"outbound": "去程", "return": "返程", "all": "往返城际交通"}[leg]
        scope_literal = "往返" if leg == "all" else prefix
        if validation_profile != "strict":
            natural_prefix = "往返" if leg == "all" else prefix
            return f"{natural_prefix}坐{mode_text}", {scope_literal, mode_text}
        return f"{prefix}必须乘坐{mode_text}", {scope_literal, mode_text}
    if kind == "time_window":
        leg = "去程" if value["leg"] == "outbound" else "返程"
        field = "出发" if value["field"] == "start_time" else "到达"
        time = str(value["time"])
        direction = "前" if operator == "lte" else "后"
        modality = "要" if validation_profile != "strict" else "必须"
        return f"{leg}{modality}在{time}{direction}{field}", {leg, time}
    if kind == "entity_category":
        category = str(value["values"][0])
        if scope == "attraction":
            suffix = "" if category.endswith("景点") else "类景点"
            return f"至少安排一个{category}{suffix}", {category}
        return f"至少安排一顿{category}", {category}
    if kind == "entity_attribute":
        attribute = str(value["values"][0])
        if validation_profile != "strict":
            return f"想住有{attribute}的酒店", {attribute}
        return f"住宿必须具备{attribute}", {attribute}
    if kind == "include_entity":
        name = str(value["names"][0])
        action = (
            {
                "attraction": "想去",
                "restaurant": "想去",
                "accommodation": "想住",
            }[scope]
            if validation_profile != "strict"
            else {
                "attraction": "必须游览",
                "restaurant": "必须去",
                "accommodation": "必须入住",
            }[scope]
        )
        return f"{action}{name}", {name}
    if kind == "category_budget":
        amount = _format_number(value["amount"])
        if scope == "restaurant":
            return f"至少安排一顿用餐，餐厅人均每餐不超过{amount}元", {f"{amount}元"}
        return f"住宿人均每晚不超过{amount}元", {f"{amount}元"}
    if kind == "room_type":
        room_type = str(value["room_type"])
        if validation_profile != "strict":
            return f"酒店想选每间{room_type}个床位的房型", {f"{room_type}个床位"}
        return f"住宿必须选择每间{room_type}个床位的房型", {f"{room_type}个床位"}
    if kind == "room_count":
        count = str(value["count"])
        if validation_profile != "strict":
            return f"酒店每晚订{count}间房", {f"{count}间房"}
        return f"每晚必须预订{count}间房", {f"{count}间房"}
    if kind == "activity_count":
        count = str(value["count"])
        if validation_profile != "strict":
            return f"全程安排{count}个景点", {f"{count}个景点"}
        return f"全程恰好安排{count}个景点", {f"{count}个景点"}
    raise SynthesisError(f"No canonical renderer for constraint kind {kind}.")


def _format_number(value: Any) -> str:
    number = float(value)
    return str(int(number)) if number.is_integer() else str(number)
