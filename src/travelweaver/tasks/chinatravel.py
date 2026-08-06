"""Safe ChinaTravel oracle adapter; source is parsed as data and never executed."""

from __future__ import annotations

import ast
import re
from typing import Any

from ..errors import TaskSpecError
from .compiler import CompileResult, build_base_spec
from .models import ConstraintSpec

ADAPTER_VERSION = "travelweaver-chinatravel-adapter-v1"

_BUDGET = re.compile(r"total_cost\s*<=\s*(\d+(?:\.\d+)?)")
_PER_PERSON_BUDGET = re.compile(r"total_cost\s*<=\s*people_number\s*\*\s*(\d+(?:\.\d+)?)")
_FOOD_BUDGET = re.compile(
    r"food_cost\s*/\s*food_count\s*/\s*people_count\(plan\)\s*<=\s*(\d+(?:\.\d+)?)"
)
_HOTEL_BUDGET = re.compile(
    r"hotel_cost\s*/\s*people_count\(plan\)\s*/\s*\(day_count\(plan\)-1\)\s*"
    r"<=\s*(\d+(?:\.\d+)?)"
)
_TIME_LIMIT = re.compile(
    r"result\s*=\s*\((?P<variable>go_start_time|go_end_time|return_start_time|back_end_time)"
    r"\s*(?P<operator><=|>=)\s*['\"](?P<time>\d{2}:\d{2})['\"]\)"
)
_BASE_CHECK = re.compile(r"result\s*=\s*\((?:day_count|people_count)\(plan\)\s*==\s*\d+\)")
_DERIVED_QUANTITY_MARKERS = (
    "activity_tickets(",
    "metro_tickets(",
    "taxi_cars(",
)
_INNER_MODES = {"metro", "taxi", "walk"}


class ChinaTravelOracleAdapter:
    """Map supported legacy oracle forms into the source-independent contract."""

    def __init__(self, *, world_snapshot_version: str = "chinatravel-pinned-v1") -> None:
        self.world_snapshot_version = world_snapshot_version

    def compile(self, public_task: dict[str, Any], oracle: dict[str, Any]) -> CompileResult:
        sources = oracle.get("hard_logic")
        if not isinstance(sources, list) or not all(isinstance(value, str) for value in sources):
            return CompileResult(
                "quarantined", None, ("ChinaTravel oracle hard_logic must be a string array.",), 0
            )
        constraints: list[ConstraintSpec] = []
        unmapped: list[str] = []
        query = str(public_task.get("query") or "")
        for source in sources:
            try:
                tree = ast.parse(source, mode="exec")
            except SyntaxError as error:
                unmapped.append(f"invalid syntax: {error.msg}")
                continue
            if not self._safe_ast(tree):
                unmapped.append("unsafe AST form")
                continue
            budget = _BUDGET.search(source)
            if budget:
                source_text = self._budget_source_text(query, budget.group(1))
                constraints.append(
                    ConstraintSpec(
                        id=f"c{len(constraints) + 1:03d}",
                        kind="total_budget",
                        operator="lte",
                        value={"amount": float(budget.group(1)), "currency": "CNY"},
                        scope="trip",
                        hardness="hard",
                        source_text=source_text,
                    )
                )
                continue
            per_person_budget = _PER_PERSON_BUDGET.search(source)
            if per_person_budget:
                amount = float(per_person_budget.group(1)) * int(public_task["people_number"])
                constraints.append(
                    ConstraintSpec(
                        id=f"c{len(constraints) + 1:03d}",
                        kind="total_budget",
                        operator="lte",
                        value={"amount": amount, "currency": "CNY"},
                        scope="trip",
                        hardness="hard",
                        source_text=self._budget_source_text(query, str(int(amount))),
                    )
                )
                continue
            category_budget = self._category_budget(
                source, query, f"c{len(constraints) + 1:03d}"
            )
            if category_budget is not None:
                constraints.append(category_budget)
                continue
            time_limit = self._time_limit(source, query, f"c{len(constraints) + 1:03d}")
            if time_limit is not None:
                constraints.append(time_limit)
                continue
            compact = re.sub(r"\s+", "", source)
            if compact in {"result=True", "result=(True)"}:
                continue
            if _BASE_CHECK.fullmatch(source.strip()):
                continue
            if any(marker in source for marker in _DERIVED_QUANTITY_MARKERS):
                # These are universal quantity rules owned by TravelEnv, not task constraints.
                continue
            translated = self._translate_structured_constraint(
                tree, source=source, query=query, constraint_id=f"c{len(constraints) + 1:03d}"
            )
            if translated is not None:
                constraints.append(translated)
                continue
            unmapped.append(source[:120])

        if unmapped:
            return CompileResult(
                "quarantined",
                None,
                tuple(f"Unsupported ChinaTravel oracle: {value}" for value in unmapped),
                1,
            )
        try:
            spec = build_base_spec(
                public_task,
                constraints=tuple(constraints),
                source="chinatravel_oracle",
                compiler_version=ADAPTER_VERSION,
                world_snapshot_version=self.world_snapshot_version,
            )
        except TaskSpecError as error:
            return CompileResult("quarantined", None, (str(error),), 1)
        return CompileResult("accepted", spec, (), 1)

    @staticmethod
    def _safe_ast(tree: ast.AST) -> bool:
        forbidden = (
            ast.AsyncFunctionDef,
            ast.Await,
            ast.ClassDef,
            ast.Delete,
            ast.FunctionDef,
            ast.Global,
            ast.Import,
            ast.ImportFrom,
            ast.Lambda,
            ast.Nonlocal,
            ast.Raise,
            ast.Try,
            ast.While,
            ast.With,
            ast.Yield,
        )
        return not any(isinstance(node, forbidden) for node in ast.walk(tree))

    @classmethod
    def _translate_structured_constraint(
        cls, tree: ast.Module, *, source: str, query: str, constraint_id: str
    ) -> ConstraintSpec | None:
        room = cls._room_constraint(tree, source, query, constraint_id)
        if room is not None:
            return room
        alternative = cls._alternative_constraint(tree, source, query, constraint_id)
        if alternative is not None:
            return alternative
        comparison = cls._result_comparison(tree)
        if comparison is None:
            return None
        left, operator, right = comparison.left, comparison.ops[0], comparison.comparators[0]
        left_set = cls._string_set(left)
        right_set = cls._string_set(right)
        variable = right.id if left_set is not None and isinstance(right, ast.Name) else None
        expected = left_set
        if right_set is not None and isinstance(left, ast.Name):
            variable = left.id
            expected = right_set
        if variable is None or expected is None:
            return None
        source_text = cls._source_for_values(query, expected) or source.strip()
        if variable == "intercity_transport_set" and isinstance(operator, ast.Eq):
            return ConstraintSpec(
                id=constraint_id,
                kind="transport_mode",
                operator="eq",
                value={"modes": sorted(expected)},
                scope="intercity_transport",
                hardness="hard",
                source_text=source_text,
            )
        if variable == "innercity_transport_set" and isinstance(operator, ast.LtE):
            return ConstraintSpec(
                id=constraint_id,
                kind="transport_mode",
                operator="not_in",
                value={"modes": sorted(_INNER_MODES.difference(expected))},
                scope="innercity_route",
                hardness="hard",
                source_text=source_text,
            )
        families = {
            "attraction_type_set": ("entity_category", "attraction", "values"),
            "restaurant_type_set": ("entity_category", "restaurant", "values"),
            "accommodation_type_set": ("entity_attribute", "accommodation", "values"),
            "attraction_name_set": ("include_entity", "attraction", "names"),
            "restaurant_name_set": ("include_entity", "restaurant", "names"),
            "accommodation_name_set": ("include_entity", "accommodation", "names"),
        }
        family = families.get(variable)
        if family is None or not isinstance(operator, ast.LtE):
            return None
        kind, scope, value_key = family
        return ConstraintSpec(
            id=constraint_id,
            kind=kind,
            operator="include" if kind == "include_entity" else "contains",
            value={value_key: sorted(expected)},
            scope=scope,
            hardness="hard",
            source_text=source_text,
        )

    @staticmethod
    def _result_comparison(tree: ast.Module) -> ast.Compare | None:
        value = ChinaTravelOracleAdapter._result_value(tree)
        return value if isinstance(value, ast.Compare) else None

    @staticmethod
    def _result_value(tree: ast.Module) -> ast.AST | None:
        for statement in reversed(tree.body):
            if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
                continue
            target = statement.targets[0]
            if isinstance(target, ast.Name) and target.id == "result":
                return statement.value
        return None

    @staticmethod
    def _string_set(node: ast.AST) -> set[str] | None:
        if not isinstance(node, ast.Set):
            return None
        values: set[str] = set()
        for element in node.elts:
            if not isinstance(element, ast.Constant) or not isinstance(element.value, str):
                return None
            values.add(element.value)
        return values

    @classmethod
    def _room_constraint(
        cls, tree: ast.Module, source: str, query: str, constraint_id: str
    ) -> ConstraintSpec | None:
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare) or not isinstance(node.ops[0], ast.NotEq):
                continue
            if not isinstance(node.left, ast.Call) or not isinstance(node.left.func, ast.Name):
                continue
            function = node.left.func.id
            if function not in {"room_count", "room_type"}:
                continue
            expected = node.comparators[0]
            if not isinstance(expected, ast.Constant) or not isinstance(expected.value, int):
                continue
            return ConstraintSpec(
                id=constraint_id,
                kind=function,
                operator="eq",
                value={"count" if function == "room_count" else "room_type": expected.value},
                scope="accommodation",
                hardness="hard",
                source_text=cls._source_for_values(query, {str(expected.value)}) or source.strip(),
            )
        return None

    @classmethod
    def _alternative_constraint(
        cls, tree: ast.Module, source: str, query: str, constraint_id: str
    ) -> ConstraintSpec | None:
        value = cls._result_value(tree)
        if not isinstance(value, ast.BoolOp) or not isinstance(value.op, ast.Or):
            return None
        alternatives: list[list[str]] = []
        variable: str | None = None
        for branch in value.values:
            if not isinstance(branch, ast.Compare) or not isinstance(branch.ops[0], ast.LtE):
                return None
            expected = cls._string_set(branch.left)
            compared = branch.comparators[0]
            if expected is None or not isinstance(compared, ast.Name):
                return None
            if variable is not None and variable != compared.id:
                return None
            variable = compared.id
            alternatives.append(sorted(expected))
        families = {
            "attraction_type_set": ("entity_category", "attraction"),
            "restaurant_type_set": ("entity_category", "restaurant"),
            "accommodation_type_set": ("entity_attribute", "accommodation"),
            "attraction_name_set": ("include_entity", "attraction"),
            "restaurant_name_set": ("include_entity", "restaurant"),
            "accommodation_name_set": ("include_entity", "accommodation"),
        }
        family = families.get(variable or "")
        if family is None:
            return None
        kind, scope = family
        all_values = {item for alternative in alternatives for item in alternative}
        return ConstraintSpec(
            id=constraint_id,
            kind=kind,
            operator="include" if kind == "include_entity" else "contains",
            value={"any_of": alternatives},
            scope=scope,
            hardness="hard",
            source_text=cls._source_for_values(query, all_values) or source.strip(),
        )

    @classmethod
    def _category_budget(
        cls, source: str, query: str, constraint_id: str
    ) -> ConstraintSpec | None:
        match = _FOOD_BUDGET.search(source)
        if match:
            return ConstraintSpec(
                id=constraint_id,
                kind="category_budget",
                operator="lte",
                value={
                    "amount": float(match.group(1)),
                    "basis": "per_person_per_activity",
                    "currency": "CNY",
                },
                scope="restaurant",
                hardness="hard",
                source_text=cls._source_for_values(query, {match.group(1)}) or source.strip(),
            )
        match = _HOTEL_BUDGET.search(source)
        if match:
            return ConstraintSpec(
                id=constraint_id,
                kind="category_budget",
                operator="lte",
                value={
                    "amount": float(match.group(1)),
                    "basis": "per_person_per_night",
                    "currency": "CNY",
                },
                scope="accommodation",
                hardness="hard",
                source_text=cls._source_for_values(query, {match.group(1)}) or source.strip(),
            )
        return None

    @classmethod
    def _time_limit(
        cls, source: str, query: str, constraint_id: str
    ) -> ConstraintSpec | None:
        match = _TIME_LIMIT.search(source)
        if not match:
            return None
        variable = match.group("variable")
        leg = "outbound" if variable.startswith("go_") else "return"
        field = "start_time" if "start_time" in variable else "end_time"
        time = match.group("time")
        return ConstraintSpec(
            id=constraint_id,
            kind="time_window",
            operator="lte" if match.group("operator") == "<=" else "gte",
            value={"leg": leg, "field": field, "time": time},
            scope="intercity_transport",
            hardness="hard",
            source_text=cls._source_for_values(query, {time}) or source.strip(),
        )

    @staticmethod
    def _source_for_values(query: str, values: set[str]) -> str | None:
        for value in sorted(values, key=len, reverse=True):
            position = query.find(value)
            if position < 0:
                continue
            start = max(query.rfind(separator, 0, position) for separator in "，。；") + 1
            ends = [query.find(separator, position) for separator in "，。；"]
            valid_ends = [end for end in ends if end >= 0]
            end = min(valid_ends) if valid_ends else len(query)
            return query[start:end].strip()
        return None

    @staticmethod
    def _budget_source_text(query: str, amount: str) -> str:
        match = re.search(rf"[^，。；]*{re.escape(amount)}[^，。；]*", query)
        return match.group(0).strip() if match else f"预算{amount}人民币"
