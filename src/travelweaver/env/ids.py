"""Stable identifiers shared by all TravelWeaver backends."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping
from typing import Any


def normalize_name(value: str) -> str:
    """Normalize a human name without making it language dependent."""

    normalized = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    return re.sub(r"\s+", " ", normalized)


def _clean_component(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip()
    text = re.sub(r"\s+", "-", text)
    return text.replace(":", "-")


def _numeric_source_id(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric.is_integer():
        return str(int(numeric))
    return None


def make_place_id(
    *,
    entity_type: str,
    city: str,
    name: str,
    source_id: Any = None,
    lang: str = "zh",
) -> str:
    """Build a stable place id using upstream ids where they are trustworthy."""

    kind = _clean_component(entity_type)
    city_component = _clean_component(city)
    if kind in {"attraction", "restaurant"}:
        numeric_id = _numeric_source_id(source_id)
        if numeric_id is not None:
            return f"place:{_clean_component(lang)}:{city_component}:{kind}:{numeric_id}"
    digest = hashlib.sha256(normalize_name(name).encode("utf-8")).hexdigest()[:16]
    return f"place:{_clean_component(lang)}:{city_component}:{kind}:{digest}"


def make_transport_id(mode: str, record: Mapping[str, Any]) -> str:
    """Build a stable id for a train or airplane record."""

    source_id = record.get("TrainID") or record.get("FlightID") or record.get("source_id")
    canonical = json.dumps(dict(record), ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    prefix = _clean_component(source_id) if source_id else "unknown"
    return f"transport:{_clean_component(mode)}:{prefix}:{digest}"
