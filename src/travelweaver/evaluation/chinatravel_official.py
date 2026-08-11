"""Deterministic ChinaTravel exporter and read-only official parity audit."""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator

from ..paths import project_root

OFFICIAL_EXPORT_VERSION = "travelweaver-chinatravel-export-v1"
OFFICIAL_AUDIT_VERSION = "travelweaver-chinatravel-official-audit-v1"


def export_official_plan(
    plan_snapshot: Mapping[str, Any], evidence_bundle: Mapping[str, Any]
) -> dict[str, Any]:
    """Expand a TravelWeaver plan into the pinned ChinaTravel output schema."""

    entities = evidence_bundle.get("entities")
    routes = evidence_bundle.get("routes")
    if not isinstance(entities, Mapping) or not isinstance(routes, Mapping):
        raise ValueError("EvidenceBundle must contain entity and route mappings.")
    activities = plan_snapshot.get("activities")
    if not isinstance(activities, Sequence):
        raise ValueError("PlanSnapshot activities must be an array.")
    by_day: dict[int, list[dict[str, Any]]] = {}
    people = int(plan_snapshot["people_number"])
    for source in activities:
        if not isinstance(source, Mapping):
            raise ValueError("PlanSnapshot contains a non-object activity.")
        day = source.get("day")
        if not isinstance(day, int):
            raise ValueError("PlanSnapshot activity day must be an integer.")
        candidate_id = source.get("candidate_id")
        entity = entities.get(candidate_id)
        if not isinstance(entity, Mapping):
            raise ValueError(f"Missing entity evidence for {candidate_id}.")
        activity_type = str(source.get("activity_type"))
        row: dict[str, Any] = {
            "type": activity_type,
            "start_time": source.get("start_time"),
            "end_time": source.get("end_time"),
            "price": float(source.get("unit_price") or 0),
            "cost": float(source.get("amount") or 0),
            "transports": _export_route(
                source.get("route_from_previous_id"), routes, people
            ),
        }
        if activity_type in {"train", "airplane"}:
            row["start"] = entity.get("origin")
            row["end"] = entity.get("destination")
            source_id = str(entity.get("source_id") or "")
            row["TrainID" if activity_type == "train" else "FlightID"] = source_id
            row["tickets"] = int(source.get("derived_quantity") or 0)
        else:
            row["position"] = entity.get("name")
            if activity_type == "accommodation":
                row["rooms"] = int(source.get("rooms") or 0)
                row["room_type"] = int(source.get("room_type") or 0)
            else:
                row["tickets"] = int(source.get("derived_quantity") or 0)
        by_day.setdefault(day, []).append(row)
    return {
        "people_number": people,
        "start_city": str(plan_snapshot["start_city"]),
        "target_city": str(plan_snapshot["target_city"]),
        "itinerary": [
            {"day": day, "activities": rows} for day, rows in sorted(by_day.items())
        ],
    }


def validate_official_schema(
    plan: Mapping[str, Any], source_root: str | Path | None = None
) -> list[str]:
    root = (
        Path(source_root)
        if source_root is not None
        else project_root() / "vendor" / "ChinaTravel"
    )
    schema = json.loads(
        (root / "chinatravel" / "evaluation" / "output_schema.json").read_text(
            encoding="utf-8"
        )
    )
    return [error.message for error in Draft7Validator(schema).iter_errors(dict(plan))]


def audit_official_commonsense(
    symbolic_input: Mapping[str, Any],
    plan: Mapping[str, Any],
    source_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run the pinned eight official checks without making them runtime Reward logic."""

    root = (
        Path(source_root)
        if source_root is not None
        else project_root() / "vendor" / "ChinaTravel"
    )
    root_text = str(root.resolve())
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from chinatravel.symbol_verification.commonsense_constraint import (  # noqa: PLC0415
        Is_activity_grounded,
        Is_attractions_correct,
        Is_hotels_correct,
        Is_intercity_transport_correct,
        Is_restaurants_correct,
        Is_space_correct,
        Is_time_correct,
        Is_transport_correct,
    )

    functions = (
        Is_activity_grounded,
        Is_intercity_transport_correct,
        Is_attractions_correct,
        Is_hotels_correct,
        Is_restaurants_correct,
        Is_transport_correct,
        Is_time_correct,
        Is_space_correct,
    )
    results: list[dict[str, Any]] = []
    for function in functions:
        table, errors = function(dict(symbolic_input), dict(plan), verbose=False)
        failures = int(table.iloc[0].sum()) if len(table.index) else 1
        results.append(
            {"check": function.__name__, "passed": failures == 0, "errors": list(errors)}
        )
    return {
        "export_version": OFFICIAL_EXPORT_VERSION,
        "passed": all(result["passed"] for result in results),
        "checks": results,
    }


def audit_synthesis_directory(
    input_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    exports_path: str | Path | None = None,
    concurrency: int = min(32, os.cpu_count() or 1),
) -> dict[str, Any]:
    """Export and run the pinned official schema/eight-check audit for every witness."""

    if concurrency <= 0:
        raise ValueError("Official audit concurrency must be positive.")
    directory = Path(input_dir)
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((directory / "records").glob("*.json"))
    ]
    if not records:
        raise ValueError("Synthesis directory has no records to audit.")

    def inspect(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        task_id = str(record["task_spec"]["task_id"])
        witness = record["witness"]
        exported = export_official_plan(
            witness["plan_snapshot"], witness["evidence_bundle"]
        )
        schema_errors = validate_official_schema(exported)
        trip = record["task_spec"]["trip"]
        symbolic_input = {
            "start_city": trip["origin"],
            "target_city": trip["destinations"][-1],
            "people_number": trip["travelers"],
            "days": trip["days"],
        }
        commonsense = (
            audit_official_commonsense(symbolic_input, exported)
            if not schema_errors
            else {"passed": False, "checks": []}
        )
        row = {
            "audit_version": OFFICIAL_AUDIT_VERSION,
            "uid": task_id,
            "schema_passed": not schema_errors,
            "schema_errors": schema_errors,
            "commonsense_passed": bool(commonsense["passed"]),
            "commonsense_checks": commonsense["checks"],
        }
        return row, {"uid": task_id, "plan": exported}

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        inspected = list(executor.map(inspect, records))
    audit_rows = [item[0] for item in inspected]
    export_rows = [item[1] for item in inspected]
    audit_target = (
        Path(output_path) if output_path is not None else directory / "official-audit.jsonl"
    )
    export_target = (
        Path(exports_path)
        if exports_path is not None
        else directory / "official-plans.jsonl"
    )
    _atomic_jsonl(audit_target, audit_rows)
    _atomic_jsonl(export_target, export_rows)
    check_failures = Counter(
        check["check"]
        for row in audit_rows
        for check in row["commonsense_checks"]
        if not check["passed"]
    )
    return {
        "audit_version": OFFICIAL_AUDIT_VERSION,
        "input_dir": str(directory.resolve()),
        "count": len(audit_rows),
        "schema_passes": sum(row["schema_passed"] for row in audit_rows),
        "commonsense_passes": sum(row["commonsense_passed"] for row in audit_rows),
        "check_failures": dict(sorted(check_failures.items())),
        "concurrency": concurrency,
        "audit_path": str(audit_target.resolve()),
        "exports_path": str(export_target.resolve()),
    }


def _export_route(
    route_id: Any, routes: Mapping[str, Any], people: int
) -> list[dict[str, Any]]:
    if route_id is None:
        return []
    route = routes.get(route_id)
    if not isinstance(route, Mapping):
        raise ValueError(f"Missing route evidence for {route_id}.")
    mode = str(route.get("mode"))
    segments = route.get("segments")
    if not isinstance(segments, Sequence):
        raise ValueError(f"Route {route_id} has no segments.")
    exported = []
    for segment in segments:
        if not isinstance(segment, Mapping):
            raise ValueError(f"Route {route_id} contains a non-object segment.")
        price = _number(segment.get("cost"))
        distance = _number(segment.get("distance"))
        if price is None or distance is None:
            raise ValueError(f"Route {route_id} has incomplete price or distance evidence.")
        segment_mode = str(segment.get("mode") or mode)
        row = {
            "start": segment.get("start"),
            "end": segment.get("end"),
            "mode": segment_mode,
            "start_time": segment.get("start_time"),
            "end_time": segment.get("end_time"),
            "price": price,
            "cost": 0.0 if segment_mode == "walk" else price,
            "distance": distance,
        }
        if segment_mode == "metro":
            row["tickets"] = people
            row["cost"] = round(price * people, 2)
        elif segment_mode == "taxi":
            cars = math.ceil(people / 4)
            row["cars"] = cars
            row["cost"] = round(price * cars, 2)
        exported.append(row)
    return exported


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)
