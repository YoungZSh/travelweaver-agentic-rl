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
}


def render_canonical(blueprint: TaskBlueprint) -> CanonicalTask:
    trip = blueprint.trip
    destination = trip.destinations[-1]
    core = (
        f"请规划一趟从{trip.origin}到{destination}的旅行，共{trip.days}天，"
        f"同行{trip.travelers}人。"
    )
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
        )
        clauses[constraint.id] = clause
        protected.update(literals)
    joined = "另外，" + "；".join(clauses.values()) + "。"
    query = core + joined + "请根据这些要求制定可执行的详细行程。"
    return CanonicalTask(
        query=query,
        clauses=clauses,
        protected_literals=tuple(sorted(protected, key=lambda value: (-len(value), value))),
    )


def _constraint_clause(
    kind: str, operator: str, value: Any, scope: str
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
            return f"市内地点之间必须统一使用{mode_text}", {mode_text}
        prefix = {"outbound": "去程", "return": "返程", "all": "往返城际交通"}[leg]
        return f"{prefix}必须乘坐{mode_text}", {prefix, mode_text}
    if kind == "time_window":
        leg = "去程" if value["leg"] == "outbound" else "返程"
        field = "出发" if value["field"] == "start_time" else "到达"
        time = str(value["time"])
        direction = "前" if operator == "lte" else "后"
        return f"{leg}必须在{time}{direction}{field}", {leg, time}
    if kind == "entity_category":
        category = str(value["values"][0])
        if scope == "attraction":
            suffix = "" if category.endswith("景点") else "类景点"
            return f"至少安排一个{category}{suffix}", {category}
        return f"至少安排一顿{category}", {category}
    if kind == "entity_attribute":
        attribute = str(value["values"][0])
        return f"住宿必须具备{attribute}", {attribute}
    if kind == "include_entity":
        name = str(value["names"][0])
        action = {
            "attraction": "必须游览",
            "restaurant": "必须去",
            "accommodation": "必须入住",
        }[scope]
        return f"{action}{name}", {name}
    if kind == "category_budget":
        amount = _format_number(value["amount"])
        if scope == "restaurant":
            return f"餐厅人均每餐不超过{amount}元", {f"{amount}元"}
        return f"住宿人均每晚不超过{amount}元", {f"{amount}元"}
    if kind == "room_type":
        room_type = str(value["room_type"])
        return f"住宿必须选择每间{room_type}个床位的房型", {f"{room_type}个床位"}
    if kind == "room_count":
        count = str(value["count"])
        return f"每晚必须预订{count}间房", {f"{count}间房"}
    if kind == "activity_count":
        count = str(value["count"])
        return f"全程恰好安排{count}个景点", {f"{count}个景点"}
    raise SynthesisError(f"No canonical renderer for constraint kind {kind}.")


def _format_number(value: Any) -> str:
    number = float(value)
    return str(int(number)) if number.is_integer() else str(number)
