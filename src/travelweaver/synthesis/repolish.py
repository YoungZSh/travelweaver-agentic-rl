"""Concurrent surface-only rewriting for existing grounded synthesis records."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import SynthesisError
from ..llm import DeepSeekConfig
from ..reward import TravelReward
from ..tasks import TaskBlueprint, materialize_task_spec
from .artifacts import ArtifactStore, record_bundle
from .models import (
    GENERATOR_VERSION,
    PROMPT_VERSION,
    WORLD_SNAPSHOT_VERSION,
    PilotSlot,
)
from .polisher import POLISHER_PROMPT_HASH, TaskPolisher
from .render import render_canonical


@dataclass(frozen=True)
class RepolishConfig:
    input_dir: Path
    output_dir: Path
    llm_concurrency: int = 256
    max_api_calls: int = 400
    validation_policy: str = "minimal_semantic"

    def __post_init__(self) -> None:
        if self.llm_concurrency <= 0 or self.max_api_calls <= 0:
            raise ValueError("LLM concurrency and API-call budget must be positive.")
        if self.validation_policy not in {"strict", "minimal_semantic"}:
            raise ValueError("Repolish validation policy is unsupported.")


@dataclass(frozen=True)
class RepolishReport:
    input_dir: str
    output_dir: str
    completed: int
    api_calls: int
    llm_concurrency: int
    canonical_fallbacks: int
    surfaces_with_warnings: int
    distributions: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_dir": self.input_dir,
            "output_dir": self.output_dir,
            "completed": self.completed,
            "api_calls": self.api_calls,
            "llm_concurrency": self.llm_concurrency,
            "canonical_fallbacks": self.canonical_fallbacks,
            "surfaces_with_warnings": self.surfaces_with_warnings,
            "distributions": self.distributions,
        }


class SurfaceRepolishPipeline:
    """Reuse accepted Blueprints and witnesses while replacing only query surfaces."""

    def __init__(
        self,
        config: RepolishConfig,
        llm_config: DeepSeekConfig,
        *,
        polisher: TaskPolisher | None = None,
    ) -> None:
        self.config = config
        self.llm_config = llm_config
        self.polisher = polisher or TaskPolisher(
            llm_config,
            validation_policy=config.validation_policy,
        )

    def run(self) -> RepolishReport:
        source_manifest = _read_json(self.config.input_dir / "manifest.json")
        source_records = _read_records(self.config.input_dir / "records")
        if not source_records:
            raise SynthesisError("Repolish input directory has no synthesis records.")
        if self.config.max_api_calls < len(source_records) * self.polisher.max_attempts:
            raise SynthesisError(
                "API budget must cover every record at the configured maximum polish attempts."
            )
        source_config = dict(source_manifest.get("config", {}))
        count = len(source_records)
        profile = str(source_config.get("profile", "pilot_v2_1"))
        seed = int(source_config.get("seed", 0))
        store = ArtifactStore(
            self.config.output_dir,
            {
                "count": count,
                "seed": seed,
                "max_api_calls": self.config.max_api_calls,
                "profile": profile,
                "model": self.llm_config.model,
                "base_url": self.llm_config.base_url,
                "thinking": "disabled",
                "generator_version": GENERATOR_VERSION,
                "prompt_version": PROMPT_VERSION,
                "prompt_hash": POLISHER_PROMPT_HASH,
                "world_snapshot_version": WORLD_SNAPSHOT_VERSION,
                "operation": "surface_repolish",
                "source_dir": str(self.config.input_dir.resolve()),
                "llm_concurrency": self.config.llm_concurrency,
                "validation_policy": self.config.validation_policy,
            },
        )
        completed = store.completed_indices()
        if completed:
            expected = {int(record["slot"]["index"]) for record in source_records}
            if completed != expected:
                raise SynthesisError(
                    "Repolish output directory contains only a partial record set."
                )
            existing = {int(row["slot"]["index"]): row for row in store.records()}
            return self._finalize(store, existing, api_calls=store.api_calls)

        rewritten: dict[int, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=self.config.llm_concurrency) as executor:
            futures = {
                executor.submit(self._rewrite_record, record): int(record["slot"]["index"])
                for record in source_records
            }
            for future in as_completed(futures):
                index = futures[future]
                rewritten[index] = future.result()

        surface_ids = [row["surface"]["surface_id"] for row in rewritten.values()]
        if len(surface_ids) != len(set(surface_ids)):
            raise SynthesisError("Concurrent repolish produced duplicate surfaces.")
        for index in sorted(rewritten):
            store.save_record(index, rewritten[index], api_calls=self.polisher.api_calls)

        return self._finalize(store, rewritten, api_calls=self.polisher.api_calls)

    def _finalize(
        self,
        store: ArtifactStore,
        rewritten: dict[int, dict[str, Any]],
        *,
        api_calls: int,
    ) -> RepolishReport:
        slots = tuple(
            _slot_from_dict(rewritten[index]["slot"]) for index in sorted(rewritten)
        )
        distributions = store.finalize(slots, api_calls=api_calls)
        fallbacks = int(distributions["surface_quality"]["canonical_fallbacks"])
        warning_count = int(distributions["surface_quality"]["surfaces_with_warnings"])
        return RepolishReport(
            input_dir=str(self.config.input_dir.resolve()),
            output_dir=str(self.config.output_dir.resolve()),
            completed=len(rewritten),
            api_calls=api_calls,
            llm_concurrency=self.config.llm_concurrency,
            canonical_fallbacks=fallbacks,
            surfaces_with_warnings=warning_count,
            distributions=distributions,
        )

    def _rewrite_record(self, source: dict[str, Any]) -> dict[str, Any]:
        slot = _slot_from_dict(source["slot"])
        blueprint = TaskBlueprint.from_dict(source["blueprint"])
        canonical = render_canonical(
            blueprint,
            style_profile=slot.surface_style,
            validation_profile=slot.validation_profile,
        )
        surface, polish_audit = self.polisher.polish_with_audit(
            blueprint,
            canonical,
            style_profile=slot.surface_style,
            validation_profile=slot.validation_profile,
            audit_context={"slot_index": slot.index},
        )
        task_id = str(source["task_spec"]["task_id"])
        spec = materialize_task_spec(blueprint, surface, task_id=task_id)
        witness = deepcopy(source["witness"])
        reward = TravelReward().evaluate(
            spec,
            witness["plan_snapshot"],
            witness["evidence_bundle"],
        )
        if reward.reward != 1.0 or not reward.all_hard_pass:
            raise SynthesisError(f"Repolished task {task_id} no longer passes its witness.")
        witness["public_task"] = {
            **witness["public_task"],
            "query": surface.public_query,
        }
        witness["reward_detail"] = reward.to_dict()
        return record_bundle(
            slot=slot,
            blueprint=blueprint.to_dict(),
            surface=surface.to_dict(),
            task_spec=spec.to_dict(),
            witness=witness,
            scenario=deepcopy(source["scenario"]),
            preference_audit=deepcopy(source.get("preference_audit")),
            polish_audit=list(polish_audit),
            candidate_attempt=int(source.get("candidate_attempt", 1)),
        )


def _read_records(records_dir: Path) -> list[dict[str, Any]]:
    if not records_dir.is_dir():
        raise SynthesisError(f"Repolish records directory does not exist: {records_dir}")
    return [_read_json(path) for path in sorted(records_dir.glob("*.json"))]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SynthesisError(f"Cannot read synthesis JSON: {path}") from error
    if not isinstance(payload, dict):
        raise SynthesisError(f"Synthesis JSON must contain an object: {path}")
    return payload


def _slot_from_dict(payload: dict[str, Any]) -> PilotSlot:
    values = dict(payload)
    values["recipe"] = tuple(values.get("recipe", ()))
    values["preference_kinds"] = tuple(values.get("preference_kinds", ()))
    return PilotSlot(**values)
