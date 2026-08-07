"""End-to-end witness-first task synthesis pipeline."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..env import ChinaTravelBackend
from ..errors import SynthesisError, TravelWeaverError
from ..llm import DeepSeekConfig
from ..reward import TravelReward
from ..tasks import materialize_task_spec
from .artifacts import ArtifactStore, record_bundle
from .catalog import build_pilot_slots
from .compose import compose_blueprint
from .models import (
    GENERATOR_VERSION,
    PROMPT_VERSION,
    WORLD_SNAPSHOT_VERSION,
)
from .polisher import POLISHER_PROMPT_HASH, TaskPolisher
from .render import render_canonical
from .witness import WitnessBuilder


@dataclass(frozen=True)
class SynthesisConfig:
    output_dir: Path
    count: int = 50
    seed: int = 20260807
    max_api_calls: int = 200

    def __post_init__(self) -> None:
        if self.count <= 0 or self.max_api_calls <= 0:
            raise ValueError("Synthesis count and API-call budget must be positive.")


@dataclass(frozen=True)
class SynthesisReport:
    output_dir: str
    requested: int
    completed: int
    api_calls: int
    quarantine_count: int
    distributions: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "requested": self.requested,
            "completed": self.completed,
            "api_calls": self.api_calls,
            "quarantine_count": self.quarantine_count,
            "distributions": self.distributions,
        }


class SynthesisPipeline:
    def __init__(
        self,
        config: SynthesisConfig,
        llm_config: DeepSeekConfig,
        *,
        backend: ChinaTravelBackend | None = None,
        polisher: TaskPolisher | None = None,
    ) -> None:
        self.config = config
        self.llm_config = llm_config
        self.backend = backend or ChinaTravelBackend()
        self.polisher = polisher or TaskPolisher(llm_config)

    def run(self) -> SynthesisReport:
        slots = build_pilot_slots(self.config.count, self.config.seed)
        store = ArtifactStore(
            self.config.output_dir,
            {
                "count": self.config.count,
                "seed": self.config.seed,
                "max_api_calls": self.config.max_api_calls,
                "model": self.llm_config.model,
                "base_url": self.llm_config.base_url,
                "thinking": "disabled",
                "generator_version": GENERATOR_VERSION,
                "prompt_version": PROMPT_VERSION,
                "prompt_hash": POLISHER_PROMPT_HASH,
                "world_snapshot_version": WORLD_SNAPSHOT_VERSION,
            },
        )
        completed = store.completed_indices()
        existing = store.records()
        blueprint_ids = {row["blueprint"]["blueprint_id"] for row in existing}
        surface_ids = {row["surface"]["surface_id"] for row in existing}
        starting_api_calls = store.api_calls

        for slot in slots:
            if slot.index in completed:
                continue
            uid = f"tw_syn_{self.config.seed}_{slot.index:03d}"
            accepted = False
            errors: list[str] = []
            for candidate_attempt, origin in enumerate(
                self._origins(slot.index, slot.destination), 1
            ):
                if (
                    self.config.max_api_calls - self._api_calls(starting_api_calls)
                    < self.polisher.max_attempts
                ):
                    raise SynthesisError(
                        "API budget cannot fund another complete polish operation."
                    )
                generation_seed = (
                    self.config.seed * 1_000_003 + slot.index * 10_007 + candidate_attempt
                )
                try:
                    witness = WitnessBuilder(
                        self.backend,
                        seed=generation_seed,
                    ).build(slot, origin=origin, uid=uid)
                    blueprint = compose_blueprint(
                        slot,
                        witness,
                        generation_seed=generation_seed,
                    )
                    if blueprint.blueprint_id in blueprint_ids:
                        raise SynthesisError("Blueprint semantic hash duplicates an accepted task.")
                    canonical = render_canonical(blueprint)
                    surface = self.polisher.polish(blueprint, canonical)
                    if surface.surface_id in surface_ids:
                        raise SynthesisError("Polished surface duplicates an accepted task.")
                    spec = materialize_task_spec(blueprint, surface, task_id=uid)
                    reward = TravelReward().evaluate(
                        spec,
                        witness.plan_snapshot,
                        witness.evidence_bundle,
                    )
                    if reward.reward != 1.0 or not reward.all_hard_pass:
                        raise SynthesisError(
                            "Final materialized TaskSpec does not pass its stored witness."
                        )
                    witness_payload = witness.to_dict()
                    witness_payload["public_task"] = {
                        **witness_payload["public_task"],
                        "query": surface.public_query,
                    }
                    witness_payload["reward_detail"] = reward.to_dict()
                    record = record_bundle(
                        slot=slot,
                        blueprint=blueprint.to_dict(),
                        surface=surface.to_dict(),
                        task_spec=spec.to_dict(),
                        witness=witness_payload,
                        candidate_attempt=candidate_attempt,
                    )
                except (TravelWeaverError, ValueError) as error:
                    message = f"{type(error).__name__}: {error}"
                    errors.append(message)
                    store.quarantine(
                        {
                            "slot_index": slot.index,
                            "candidate_attempt": candidate_attempt,
                            "origin": origin,
                            "destination": slot.destination,
                            "stage_error": message,
                        },
                        api_calls=self._api_calls(starting_api_calls),
                    )
                    if self._api_calls(starting_api_calls) >= self.config.max_api_calls:
                        break
                    continue
                api_calls = self._api_calls(starting_api_calls)
                store.save_record(slot.index, record, api_calls=api_calls)
                blueprint_ids.add(blueprint.blueprint_id)
                surface_ids.add(surface.surface_id)
                accepted = True
                break
            if not accepted:
                detail = " | ".join(errors[-3:]) or "no origin candidates"
                raise SynthesisError(f"Unable to fill synthesis slot {slot.index}: {detail}")

        api_calls = self._api_calls(starting_api_calls)
        distributions = store.finalize(slots, api_calls=api_calls)
        return SynthesisReport(
            output_dir=str(self.config.output_dir.resolve()),
            requested=self.config.count,
            completed=len(store.completed_indices()),
            api_calls=api_calls,
            quarantine_count=store.quarantine_count,
            distributions=distributions,
        )

    def _origins(self, slot_index: int, destination: str) -> tuple[str, ...]:
        cities = [city for city in self.backend.supported_cities]
        cities.remove(destination)
        random.Random(self.config.seed + slot_index * 65_537).shuffle(cities)
        return tuple(cities)

    def _api_calls(self, starting_api_calls: int) -> int:
        return starting_api_calls + self.polisher.api_calls
