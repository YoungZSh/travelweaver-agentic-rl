"""End-to-end witness-first task synthesis pipeline."""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import time
import unicodedata
from collections import Counter
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..env import ChinaTravelBackend, ScenarioBackend, ScenarioSpec
from ..errors import SynthesisError, TravelWeaverError
from ..llm import DEFAULT_DEEPSEEK_CONCURRENCY, DeepSeekConfig
from ..reward import TravelReward
from ..tasks import TaskBlueprint, materialize_task_spec
from .artifacts import ArtifactStore, record_bundle
from .catalog import BLENDED_V1_1_PROFILE, OFFICIAL_HYBRID_V2_PROFILE, build_pilot_slots
from .compose import compose_blueprint
from .models import (
    GENERATOR_VERSION,
    PROMPT_VERSION,
    WORLD_SNAPSHOT_VERSION,
    CanonicalTask,
    PilotSlot,
    WitnessResult,
)
from .polisher import POLISHER_PROMPT_HASH, TaskPolisher, canonical_surface
from .randomness import deterministic_rng
from .render import render_canonical
from .scenario import build_scenario
from .trajectory_policy import (
    MAX_CONSECUTIVE_TOOL_CALLS,
    MAX_SYNTHESIS_VALID_STEPS,
    trajectory_policy,
)
from .witness import WitnessBuilder

# A slot first tries every viable origin exactly once.  Remaining attempts use
# deterministic destination replacements instead of repeating an impossible
# destination with new random seeds.  This keeps exact task/scenario quotas while
# preventing a few low-feasibility city combinations from dominating batch time.
MAX_WITNESS_CANDIDATE_ATTEMPTS = 24
MAX_WITNESS_CANDIDATE_SECONDS = 30
_PREPARATION_WORKER: Any | None = None


class _CandidatePreparationTimeout(Exception):
    """Internal signal used to abandon one pathological witness candidate."""


def _prepare_candidate_with_deadline(
    pipeline: SynthesisPipeline,
    slot: PilotSlot,
    *,
    uid: str,
    origin: str,
    candidate_attempt: int,
) -> _PreparedCandidate:
    """Prepare one candidate with a process-local CPU deadline.

    Witness preparation normally completes in well under a second, but a rare
    combinatorial route search can otherwise occupy one pool worker indefinitely.
    SIGALRM is local to the worker process and lets the slot continue to its next
    deterministic origin/destination candidate without changing accepted samples.
    """

    def raise_timeout(_signum: int, _frame: Any) -> None:
        raise _CandidatePreparationTimeout

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, raise_timeout)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, MAX_WITNESS_CANDIDATE_SECONDS)
    try:
        return pipeline._prepare_candidate(
            slot,
            uid=uid,
            origin=origin,
            candidate_attempt=candidate_attempt,
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)


@dataclass(frozen=True)
class SynthesisConfig:
    output_dir: Path
    count: int = 100
    seed: int = 20260807
    max_api_calls: int = 300
    profile: str = "pilot_v2_1"
    validation_policy: str = "minimal_semantic"
    llm_concurrency: int = DEFAULT_DEEPSEEK_CONCURRENCY
    witness_concurrency: int = min(32, os.cpu_count() or 1)
    exclude_task_dirs: tuple[Path, ...] = ()
    canonical_only: bool = False

    def __post_init__(self) -> None:
        if (
            self.count <= 0
            or self.max_api_calls < 0
            or (self.max_api_calls == 0 and not self.canonical_only)
            or self.llm_concurrency <= 0
            or self.witness_concurrency <= 0
        ):
            raise ValueError(
                "Synthesis count, API-call budget, and LLM concurrency must be positive."
            )
        if self.validation_policy not in {"strict", "minimal_semantic"}:
            raise ValueError("Synthesis validation policy is unsupported.")


@dataclass(frozen=True)
class SynthesisReport:
    output_dir: str
    requested: int
    completed: int
    api_calls: int
    quarantine_count: int
    llm_concurrency: int
    witness_concurrency: int
    distributions: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "requested": self.requested,
            "completed": self.completed,
            "api_calls": self.api_calls,
            "quarantine_count": self.quarantine_count,
            "llm_concurrency": self.llm_concurrency,
            "witness_concurrency": self.witness_concurrency,
            "distributions": self.distributions,
        }


@dataclass(frozen=True)
class _PreparedCandidate:
    slot: PilotSlot
    uid: str
    candidate_attempt: int
    scenario: ScenarioSpec
    witness: WitnessResult
    preference_candidates: tuple[WitnessResult, ...]
    preference_audit: dict[str, Any] | None
    blueprint: TaskBlueprint
    canonical: CanonicalTask


class SynthesisPipeline:
    def __init__(
        self,
        config: SynthesisConfig,
        llm_config: DeepSeekConfig,
        *,
        backend: ChinaTravelBackend | None = None,
        polisher: TaskPolisher | None = None,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.config = config
        self.llm_config = llm_config
        self._backend_is_default = backend is None
        self.backend = backend or ChinaTravelBackend()
        self.polisher = polisher
        self.progress = progress
        if self.polisher is None and not config.canonical_only:
            self.polisher = TaskPolisher(
                llm_config,
                validation_policy=config.validation_policy,
            )

    def run(self) -> SynthesisReport:
        slots = build_pilot_slots(
            self.config.count,
            self.config.seed,
            self.config.profile,
        )
        exclusions = _load_task_exclusions(self.config.exclude_task_dirs)
        store = ArtifactStore(
            self.config.output_dir,
            {
                "count": self.config.count,
                "seed": self.config.seed,
                "max_api_calls": self.config.max_api_calls,
                "profile": self.config.profile,
                "validation_policy": self.config.validation_policy,
                "model": self.llm_config.model,
                "base_url": self.llm_config.base_url,
                "thinking": "disabled",
                "generator_version": GENERATOR_VERSION,
                "prompt_version": PROMPT_VERSION,
                "prompt_hash": POLISHER_PROMPT_HASH,
                "world_snapshot_version": WORLD_SNAPSHOT_VERSION,
                "llm_concurrency": self.config.llm_concurrency,
                "witness_concurrency": self.config.witness_concurrency,
                "canonical_only": self.config.canonical_only,
                "trajectory_policy": {
                    "max_valid_steps": MAX_SYNTHESIS_VALID_STEPS,
                    "max_consecutive_tool_calls": MAX_CONSECUTIVE_TOOL_CALLS,
                },
                "exclude_task_dirs": [
                    str(path.resolve()) for path in self.config.exclude_task_dirs
                ],
                "exclusion_count": len(exclusions["task_ids"]),
                "exclusion_sha256": exclusions["sha256"],
            },
        )
        completed = store.completed_indices()
        existing = store.records()
        blueprint_ids = set(exclusions["blueprint_ids"])
        blueprint_ids.update(row["blueprint"]["blueprint_id"] for row in existing)
        surface_ids = set(exclusions["surface_ids"])
        surface_ids.update(row["surface"]["surface_id"] for row in existing)
        normalized_queries = set(exclusions["normalized_queries"])
        normalized_queries.update(
            _normalize_question(str(row["surface"]["public_query"])) for row in existing
        )
        task_ids = set(exclusions["task_ids"])
        task_ids.update(str(row["task_spec"]["task_id"]) for row in existing)
        self._origin_usage = Counter(
            str(row["slot"]["origin"])
            for row in existing
        )
        self._od_usage = Counter(
            (str(row["slot"]["origin"]), str(row["slot"]["destination"]))
            for row in existing
        )
        starting_api_calls = store.api_calls
        pending_slots = [slot for slot in slots if slot.index not in completed]
        uid_version = "v3" if self.config.profile == OFFICIAL_HYBRID_V2_PROFILE else "v2"
        duplicate_task_ids = [
            f"tw_syn_{uid_version}_{self.config.seed}_{slot.index:04d}"
            for slot in pending_slots
            if f"tw_syn_{uid_version}_{self.config.seed}_{slot.index:04d}" in task_ids
        ]
        if duplicate_task_ids:
            raise SynthesisError(
                "Task ids duplicate excluded or accepted tasks: "
                + ", ".join(duplicate_task_ids[:3])
            )
        remaining_budget = self.config.max_api_calls - starting_api_calls
        required_budget = (
            0 if self.polisher is None else len(pending_slots) * self.polisher.max_attempts
        )
        if remaining_budget < required_budget:
            raise SynthesisError(
                "API budget cannot fund every pending slot at the maximum polish attempts: "
                f"need {required_budget}, have {remaining_budget}."
            )

        started_at = time.monotonic()
        start_event = "synthesis_resumed" if completed else "synthesis_started"
        self._emit_progress(
            store,
            event=start_event,
            requested=len(slots),
            completed=len(completed),
            pending=len(pending_slots),
        )

        def persist_candidate(
            candidate: _PreparedCandidate, record: dict[str, Any]
        ) -> None:
            surface = record["surface"]
            surface_id = str(surface["surface_id"])
            normalized_query = _normalize_question(str(surface["public_query"]))
            if surface_id in surface_ids:
                raise SynthesisError(
                    "Polished surface duplicates an excluded or accepted task."
                )
            if normalized_query in normalized_queries:
                raise SynthesisError(
                    "Normalized Question duplicates an excluded or accepted task."
                )
            store.save_record(
                candidate.slot.index,
                record,
                api_calls=self._api_calls(starting_api_calls),
            )
            surface_ids.add(surface_id)
            normalized_queries.add(normalized_query)
            task_ids.add(candidate.uid)
            completed_count = len(store.completed_indices())
            self._emit_progress(
                store,
                event="slot_completed",
                slot_index=candidate.slot.index,
                task_id=candidate.uid,
                requested=len(slots),
                completed=completed_count,
                pending=len(slots) - completed_count,
                percent=round(completed_count * 100 / len(slots), 2),
                elapsed_seconds=round(time.monotonic() - started_at, 3),
            )

        def fail_candidate(candidate: _PreparedCandidate, error: Exception) -> None:
            store.quarantine(
                {
                    "slot_index": candidate.slot.index,
                    "candidate_attempt": candidate.candidate_attempt,
                    "origin": candidate.slot.origin,
                    "destination": candidate.slot.destination,
                    "scenario_profile": candidate.slot.scenario_profile,
                    "stage_error": f"{type(error).__name__}: {error}",
                },
                api_calls=self._api_calls(starting_api_calls),
            )
            self._emit_progress(
                store,
                event="slot_failed",
                slot_index=candidate.slot.index,
                task_id=candidate.uid,
                requested=len(slots),
                completed=len(store.completed_indices()),
                error=f"{type(error).__name__}: {error}",
            )

        materialize_futures: dict[Future[dict[str, Any]], _PreparedCandidate] = {}
        executor = (
            ThreadPoolExecutor(max_workers=self.config.llm_concurrency)
            if self.polisher is not None
            else None
        )
        use_parallel_preparation = (
            self._backend_is_default
            and self.config.witness_concurrency > 1
            and len(pending_slots) > 1
        )
        preparation_executor = (
            ProcessPoolExecutor(
                max_workers=min(self.config.witness_concurrency, len(pending_slots)),
                initializer=_initialize_preparation_worker,
                initargs=(self.config,),
            )
            if use_parallel_preparation
            else None
        )
        preparation_futures: dict[
            Future[tuple[_PreparedCandidate, list[dict[str, Any]]]], PilotSlot
        ] = {}
        failures: list[tuple[int, Exception]] = []
        if preparation_executor is not None:
            for slot in pending_slots:
                self._emit_progress(
                    store,
                    event="slot_started",
                    slot_index=slot.index,
                    requested=len(slots),
                    completed=len(store.completed_indices()),
                )
                preparation_futures[
                    preparation_executor.submit(_prepare_slot_in_worker, slot)
                ] = slot
            preparation_items: Any = (
                (preparation_futures[future], future)
                for future in as_completed(tuple(preparation_futures))
            )
        else:
            preparation_items = ((slot, None) for slot in pending_slots)
        try:
            for slot, preparation_future in preparation_items:
                if preparation_future is None:
                    self._emit_progress(
                        store,
                        event="slot_started",
                        slot_index=slot.index,
                        requested=len(slots),
                        completed=len(store.completed_indices()),
                    )
                try:
                    candidate, quarantine_rows = (
                        self._prepare_slot(slot, frozenset(blueprint_ids))
                        if preparation_future is None
                        else preparation_future.result()
                    )
                    for row in quarantine_rows:
                        store.quarantine(
                            row, api_calls=self._api_calls(starting_api_calls)
                        )
                    if candidate.blueprint.blueprint_id in blueprint_ids:
                        candidate, extra_rows = self._prepare_slot(
                            slot, frozenset(blueprint_ids)
                        )
                        for row in extra_rows:
                            store.quarantine(
                                row, api_calls=self._api_calls(starting_api_calls)
                            )
                    if candidate.blueprint.blueprint_id in blueprint_ids:
                        raise SynthesisError(
                            f"Unable to find a unique Blueprint for synthesis slot {slot.index}."
                        )
                except (TravelWeaverError, ValueError) as error:
                    self._emit_progress(
                        store,
                        event="slot_failed",
                        slot_index=slot.index,
                        requested=len(slots),
                        completed=len(store.completed_indices()),
                        error=f"{type(error).__name__}: {error}",
                    )
                    failures.append((slot.index, error))
                    continue
                blueprint_ids.add(candidate.blueprint.blueprint_id)
                self._origin_usage[candidate.slot.origin] += 1
                self._od_usage[
                    (candidate.slot.origin, candidate.slot.destination)
                ] += 1
                if candidate.slot.destination != slot.destination:
                    self._emit_progress(
                        store,
                        event="slot_replaced",
                        slot_index=slot.index,
                        requested_destination=slot.destination,
                        replacement_destination=candidate.slot.destination,
                        candidate_attempt=candidate.candidate_attempt,
                        requested=len(slots),
                        completed=len(store.completed_indices()),
                    )
                self._emit_progress(
                    store,
                    event="slot_prepared",
                    slot_index=slot.index,
                    task_id=candidate.uid,
                    requested=len(slots),
                    completed=len(store.completed_indices()),
                )
                if executor is None:
                    try:
                        persist_candidate(
                            candidate, self._materialize_candidate(candidate)
                        )
                    except (TravelWeaverError, ValueError) as error:
                        fail_candidate(candidate, error)
                        failures.append((slot.index, error))
                        continue
                else:
                    future = executor.submit(self._materialize_candidate, candidate)
                    materialize_futures[future] = candidate
                    for completed_future in tuple(materialize_futures):
                        if not completed_future.done():
                            continue
                        completed_candidate = materialize_futures.pop(completed_future)
                        try:
                            persist_candidate(
                                completed_candidate, completed_future.result()
                            )
                        except (TravelWeaverError, ValueError) as error:
                            fail_candidate(completed_candidate, error)
                            failures.append((completed_candidate.slot.index, error))
            for completed_future in as_completed(tuple(materialize_futures)):
                candidate = materialize_futures[completed_future]
                try:
                    persist_candidate(candidate, completed_future.result())
                except (TravelWeaverError, ValueError) as error:
                    fail_candidate(candidate, error)
                    failures.append((candidate.slot.index, error))
            if failures:
                detail = " | ".join(
                    f"slot {index}: {type(error).__name__}: {error}"
                    for index, error in failures[-5:]
                )
                raise SynthesisError(
                    f"{len(failures)} synthesis slot(s) failed after all independent "
                    f"results were persisted: {detail}"
                )
        except BaseException as error:
            self._emit_progress(
                store,
                event=(
                    "synthesis_interrupted"
                    if isinstance(error, KeyboardInterrupt)
                    else "synthesis_failed"
                ),
                requested=len(slots),
                completed=len(store.completed_indices()),
                pending=len(slots) - len(store.completed_indices()),
                error=f"{type(error).__name__}: {error}",
                elapsed_seconds=round(time.monotonic() - started_at, 3),
            )
            raise
        finally:
            if preparation_executor is not None:
                preparation_executor.shutdown(wait=True, cancel_futures=True)
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)

        api_calls = self._api_calls(starting_api_calls)
        try:
            distributions = store.finalize(slots, api_calls=api_calls)
        except (TravelWeaverError, ValueError) as error:
            self._emit_progress(
                store,
                event="finalization_failed",
                requested=len(slots),
                completed=len(store.completed_indices()),
                pending=len(slots) - len(store.completed_indices()),
                error=f"{type(error).__name__}: {error}",
                elapsed_seconds=round(time.monotonic() - started_at, 3),
            )
            raise
        self._emit_progress(
            store,
            event="synthesis_completed",
            requested=len(slots),
            completed=len(slots),
            pending=0,
            percent=100.0,
            elapsed_seconds=round(time.monotonic() - started_at, 3),
        )
        return SynthesisReport(
            output_dir=str(self.config.output_dir.resolve()),
            requested=self.config.count,
            completed=len(store.completed_indices()),
            api_calls=api_calls,
            quarantine_count=store.quarantine_count,
            llm_concurrency=self.config.llm_concurrency,
            witness_concurrency=self.config.witness_concurrency,
            distributions=distributions,
        )

    def _emit_progress(
        self,
        store: ArtifactStore,
        **event: Any,
    ) -> None:
        store.record_progress(event)
        if self.progress is not None:
            self.progress(dict(event))

    def _prepare_slot(
        self, slot: PilotSlot, blocked_blueprint_ids: frozenset[str]
    ) -> tuple[_PreparedCandidate, list[dict[str, Any]]]:
        uid_version = "v3" if self.config.profile == OFFICIAL_HYBRID_V2_PROFILE else "v2"
        uid = f"tw_syn_{uid_version}_{self.config.seed}_{slot.index:04d}"
        quarantine_rows: list[dict[str, Any]] = []
        errors: list[str] = []
        for candidate_attempt, (candidate_slot, origin) in enumerate(
            self._candidate_slot_origins(slot), 1
        ):
            try:
                candidate = _prepare_candidate_with_deadline(
                    self,
                    candidate_slot,
                    uid=uid,
                    origin=origin,
                    candidate_attempt=candidate_attempt,
                )
                if candidate.blueprint.blueprint_id in blocked_blueprint_ids:
                    raise SynthesisError(
                        "Blueprint semantic hash duplicates an accepted task."
                    )
                return candidate, quarantine_rows
            except _CandidatePreparationTimeout:
                message = (
                    "SynthesisError: witness candidate exceeded "
                    f"{MAX_WITNESS_CANDIDATE_SECONDS}s CPU deadline"
                )
                errors.append(message)
                quarantine_rows.append(
                    {
                        "slot_index": slot.index,
                        "candidate_attempt": candidate_attempt,
                        "origin": origin,
                        "destination": candidate_slot.destination,
                        "requested_destination": slot.destination,
                        "scenario_profile": candidate_slot.scenario_profile,
                        "stage_error": message,
                    }
                )
            except (TravelWeaverError, ValueError) as error:
                message = f"{type(error).__name__}: {error}"
                errors.append(message)
                quarantine_rows.append(
                    {
                        "slot_index": slot.index,
                        "candidate_attempt": candidate_attempt,
                        "origin": origin,
                        "destination": candidate_slot.destination,
                        "requested_destination": slot.destination,
                        "scenario_profile": candidate_slot.scenario_profile,
                        "stage_error": message,
                    }
                )
        detail = " | ".join(errors[-3:]) or "no origin candidates"
        raise SynthesisError(f"Unable to fill synthesis slot {slot.index}: {detail}")

    def _prepare_candidate(
        self,
        slot: PilotSlot,
        *,
        uid: str,
        origin: str,
        candidate_attempt: int,
    ) -> _PreparedCandidate:
        generation_seed = deterministic_rng(
            self.config.seed,
            f"witness-{candidate_attempt}",
            slot.index,
        ).getrandbits(63)
        effective_slot = replace(slot, origin=origin)
        if (
            effective_slot.synthesis_profile
            in {BLENDED_V1_1_PROFILE, OFFICIAL_HYBRID_V2_PROFILE}
            and effective_slot.task_type == "human_like"
            and effective_slot.metadata_prefix is not None
        ):
            effective_slot = replace(
                effective_slot,
                metadata_prefix=(
                    f"[当前位置{origin},目标位置{effective_slot.destination},"
                    f"旅行人数{effective_slot.travelers},"
                    f"旅行天数{effective_slot.days},"
                    f"出行背景{effective_slot.persona_context}]"
                ),
            )
        scenario = build_scenario(
            self.backend,
            effective_slot,
            origin=origin,
            seed=self.config.seed,
            candidate_attempt=candidate_attempt,
        )
        scenario_backend = ScenarioBackend(self.backend, scenario)
        preference_candidates: list[WitnessResult] = []
        if effective_slot.task_type == "preference_like":
            preference_candidates = self._preference_witnesses(
                scenario_backend,
                effective_slot,
                origin=origin,
                uid=uid,
                generation_seed=generation_seed,
            )
            witness, preference_audit = _choose_preference_witness(
                preference_candidates,
                effective_slot.preference_kinds[0],
            )
        else:
            witness = WitnessBuilder(
                scenario_backend,
                seed=generation_seed,
            ).build(
                effective_slot,
                origin=origin,
                uid=uid,
            )
            preference_audit = None
        if effective_slot.synthesis_profile == OFFICIAL_HYBRID_V2_PROFILE:
            selected_ids = {
                str(item.get("place_id") or item.get("transport_id"))
                for value in witness.selected.values()
                for item in (value if isinstance(value, list) else [value])
                if isinstance(item, Mapping)
            }
            changed_selected = sorted(
                effect.target_id
                for effect in scenario.effects
                if effect.target_id in selected_ids
            )
            if changed_selected:
                raise SynthesisError(
                    "Official-hybrid Scenario changed a selected witness entity: "
                    + ", ".join(changed_selected)
                )
        if (
            effective_slot.synthesis_profile
            in {BLENDED_V1_1_PROFILE, OFFICIAL_HYBRID_V2_PROFILE}
            and effective_slot.task_type == "human_like"
            and "less_walking" in effective_slot.preference_kinds
            and witness.route_mode == "walk"
        ):
            raise SynthesisError(
                "Human less-walking preference resolved to an all-walking witness."
            )
        blueprint = compose_blueprint(
            effective_slot,
            witness,
            generation_seed=generation_seed,
            world_snapshot_version=scenario.world_snapshot_version,
            constraint_witnesses=preference_candidates,
        )
        canonical = render_canonical(
            blueprint,
            style_profile=effective_slot.surface_style,
            validation_profile=effective_slot.validation_profile,
        )
        return _PreparedCandidate(
            slot=effective_slot,
            uid=uid,
            candidate_attempt=candidate_attempt,
            scenario=scenario,
            witness=witness,
            preference_candidates=tuple(preference_candidates),
            preference_audit=preference_audit,
            blueprint=blueprint,
            canonical=canonical,
        )

    def _materialize_candidate(self, candidate: _PreparedCandidate) -> dict[str, Any]:
        if self.polisher is None:
            surface, polish_audit = canonical_surface(
                candidate.blueprint,
                candidate.canonical,
                validation_profile=candidate.slot.validation_profile,
                validation_policy=self.config.validation_policy,
                audit_context={"slot_index": candidate.slot.index},
            )
        else:
            surface, polish_audit = self.polisher.polish_with_audit(
                candidate.blueprint,
                candidate.canonical,
                style_profile=candidate.slot.surface_style,
                validation_profile=candidate.slot.validation_profile,
                audit_context={"slot_index": candidate.slot.index},
            )
        spec = materialize_task_spec(
            candidate.blueprint,
            surface,
            task_id=candidate.uid,
        )
        reward = TravelReward().evaluate(
            spec,
            candidate.witness.plan_snapshot,
            candidate.witness.evidence_bundle,
        )
        if reward.reward != 1.0 or not reward.all_hard_pass:
            raise SynthesisError(
                "Final materialized TaskSpec does not pass its stored witness."
            )
        witness_payload = candidate.witness.to_dict()
        witness_payload["public_task"] = {
            **witness_payload["public_task"],
            "query": surface.public_query,
        }
        witness_payload["reward_detail"] = reward.to_dict()
        preference_audit = candidate.preference_audit
        if preference_audit is not None:
            audited_candidates = []
            for preference_candidate, candidate_row in zip(
                candidate.preference_candidates,
                preference_audit["candidates"],
                strict=True,
            ):
                candidate_reward = TravelReward().evaluate(
                    spec,
                    preference_candidate.plan_snapshot,
                    preference_candidate.evidence_bundle,
                )
                audited_candidates.append(
                    {
                        **candidate_row,
                        "hard_reward": candidate_reward.reward,
                        "all_hard_pass": candidate_reward.all_hard_pass,
                    }
                )
            if not all(row["all_hard_pass"] for row in audited_candidates):
                raise SynthesisError(
                    "Preference audit contains a candidate that fails hard constraints."
                )
            preference_audit = {
                **preference_audit,
                "candidates": audited_candidates,
            }
        return record_bundle(
            slot=candidate.slot,
            blueprint=candidate.blueprint.to_dict(),
            surface=surface.to_dict(),
            task_spec=spec.to_dict(),
            witness=witness_payload,
            scenario=candidate.scenario.to_dict(),
            preference_audit=preference_audit,
            polish_audit=list(polish_audit),
            candidate_attempt=candidate.candidate_attempt,
            trajectory_policy=trajectory_policy(),
        )

    def _origins(self, slot: PilotSlot) -> tuple[str, ...]:
        # Catalog assignment already balances origins.  Trying that city first avoids
        # the former O(slots*cities) round-trip pre-scan; failed cities naturally fall
        # through to deterministic alternatives in _prepare_slot.
        alternatives = [
            city
            for city in self.backend.supported_cities
            if city not in {slot.destination, slot.origin}
        ]
        deterministic_rng(self.config.seed, "origin-fallbacks", slot.index).shuffle(
            alternatives
        )
        return (slot.origin, *alternatives)

    def _candidate_origins(self, slot: PilotSlot) -> tuple[str, ...]:
        """Return every viable origin once in deterministic preference order."""

        origins = self._origins(slot)
        return tuple(dict.fromkeys(origin for origin in origins if origin != slot.destination))

    def _candidate_slot_origins(
        self, slot: PilotSlot
    ) -> tuple[tuple[PilotSlot, str], ...]:
        """Try the requested destination once per origin, then deterministic replacements.

        Repeating the same low-feasibility destination with different random seeds created the
        former 497/500 tail.  Destination replacements preserve the slot's task type, Scenario,
        recipe, day count, and transport requirements; the progress audit records every change.
        """

        attempts: list[tuple[PilotSlot, str]] = [
            (slot, origin) for origin in self._candidate_origins(slot)
        ]
        destinations = [
            city for city in self.backend.supported_cities if city != slot.destination
        ]
        deterministic_rng(
            self.config.seed, "destination-fallbacks", slot.index
        ).shuffle(destinations)
        replacement_origins = [
            (
                replace(slot, destination=destination),
                self._candidate_origins(replace(slot, destination=destination)),
            )
            for destination in destinations
        ]
        round_index = 0
        while len(attempts) < MAX_WITNESS_CANDIDATE_ATTEMPTS:
            added = False
            for candidate_slot, origins in replacement_origins:
                if round_index >= len(origins):
                    continue
                attempts.append((candidate_slot, origins[round_index]))
                added = True
                if len(attempts) == MAX_WITNESS_CANDIDATE_ATTEMPTS:
                    break
            if not added:
                break
            round_index += 1
        return tuple(attempts)

    def _api_calls(self, starting_api_calls: int) -> int:
        return starting_api_calls + (0 if self.polisher is None else self.polisher.api_calls)

    def _preference_witnesses(
        self,
        backend: ScenarioBackend,
        slot: PilotSlot,
        *,
        origin: str,
        uid: str,
        generation_seed: int,
    ) -> list[WitnessResult]:
        anchor_records = sorted(
            backend._records("attraction", slot.destination),
            key=lambda item: str(item.get("place_id", "")),
        )
        anchor = dict(anchor_records[0]) if anchor_records else None
        candidates: list[WitnessResult] = []
        errors: list[str] = []
        preference_kind = slot.preference_kinds[0]
        vary_route = (
            slot.synthesis_profile in {BLENDED_V1_1_PROFILE, OFFICIAL_HYBRID_V2_PROFILE}
            and "innercity_mode" not in slot.recipe
        )
        route_modes = (slot.route_mode,) if not vary_route else ("taxi", "metro", "walk")
        max_candidates = (
            12
            if slot.synthesis_profile in {BLENDED_V1_1_PROFILE, OFFICIAL_HYBRID_V2_PROFILE}
            else 6
        )
        for candidate_index in range(max_candidates):
            vary_relaxed_long_trip_meal = (
                preference_kind == "relaxed_itinerary"
                and slot.days > 3
                and bool(candidate_index % 2)
            )
            attractions_per_day = (
                2
                if preference_kind in {"more_attractions", "relaxed_itinerary"}
                and slot.days <= 3
                and candidate_index % 2
                else 1
            )
            candidate_slot = replace(
                slot,
                attractions_per_day=attractions_per_day,
                include_meal=(
                    slot.include_meal
                    or preference_kind in {"less_innercity_time", "less_walking"}
                    or vary_relaxed_long_trip_meal
                ),
                route_mode=route_modes[candidate_index % len(route_modes)],
            )
            candidate_seed = deterministic_rng(
                generation_seed,
                "preference-candidate",
                candidate_index,
            ).getrandbits(63)
            try:
                candidate = WitnessBuilder(backend, seed=candidate_seed).build(
                    candidate_slot,
                    origin=origin,
                    uid=uid,
                )
            except (TravelWeaverError, ValueError) as error:
                errors.append(str(error))
                continue
            if anchor is not None:
                candidate.selected["preference_anchor"] = anchor
            if (
                candidates
                and "innercity_mode" in slot.recipe
                and candidate.route_mode != candidates[0].route_mode
            ):
                continue
            candidates.append(candidate)
            metrics = [_preference_metric(item, preference_kind) for item in candidates]
            if len(candidates) >= 3 and (
                slot.synthesis_profile
                not in {BLENDED_V1_1_PROFILE, OFFICIAL_HYBRID_V2_PROFILE}
                or len(set(metrics)) >= 2
            ):
                break
        if len(candidates) < 2:
            raise SynthesisError(
                "Preference witness selection needs at least two feasible candidates: "
                + " | ".join(errors[-2:])
            )
        if (
            slot.synthesis_profile in {BLENDED_V1_1_PROFILE, OFFICIAL_HYBRID_V2_PROFILE}
            and len({_preference_metric(item, preference_kind) for item in candidates}) < 2
        ):
            raise SynthesisError(
                f"Preference metric {preference_kind} did not distinguish feasible candidates."
            )
        return candidates


def _initialize_preparation_worker(config: SynthesisConfig) -> None:
    """Create one backend per worker and reuse it across independently seeded slots."""

    global _PREPARATION_WORKER
    worker_config = replace(
        config,
        canonical_only=True,
        max_api_calls=0,
    )
    _PREPARATION_WORKER = SynthesisPipeline(
        worker_config,
        DeepSeekConfig(
            api_key="offline-preparation",
            model="deterministic-witness-preparation",
            thinking="disabled",
        ),
        backend=ChinaTravelBackend(),
        polisher=None,
    )


def _prepare_slot_in_worker(
    slot: PilotSlot,
) -> tuple[_PreparedCandidate, list[dict[str, Any]]]:
    if not isinstance(_PREPARATION_WORKER, SynthesisPipeline):
        raise RuntimeError("Synthesis preparation worker was not initialized.")
    return _PREPARATION_WORKER._prepare_slot(slot, frozenset())


def _clock_minutes(value: Any) -> int:
    if not isinstance(value, str) or ":" not in value:
        return -1
    try:
        hours, minutes = (int(part) for part in value.split(":", 1))
    except ValueError:
        return -1
    return hours * 60 + minutes


_PREFERENCE_DIRECTIONS = {
    "more_attractions": "maximize",
    "less_innercity_time": "minimize",
    "shorter_meal_transfer": "minimize",
    "higher_dining_share": "maximize",
    "lower_lodging_share": "minimize",
    "near_poi": "minimize",
    "less_walking": "minimize",
    "lower_total_cost": "minimize",
    "relaxed_itinerary": "minimize",
    "higher_attraction_share": "maximize",
    "lower_intercity_share": "minimize",
    "shorter_total_travel_time": "minimize",
}


def _choose_preference_witness(
    candidates: list[WitnessResult], kind: str
) -> tuple[WitnessResult, dict[str, Any]]:
    direction = _PREFERENCE_DIRECTIONS[kind]
    metrics = [_preference_metric(candidate, kind) for candidate in candidates]
    selected_index = (
        max(range(len(candidates)), key=metrics.__getitem__)
        if direction == "maximize"
        else min(range(len(candidates)), key=metrics.__getitem__)
    )
    audit = {
        "preference_kind": kind,
        "direction": direction,
        "metric": _preference_metric_name(kind),
        "selected_candidate": selected_index,
        "candidates": [
            {
                "candidate_index": index,
                "metric_value": round(metric, 6),
                "selected": index == selected_index,
            }
            for index, metric in enumerate(metrics)
        ],
    }
    return candidates[selected_index], audit


def _preference_metric(witness: WitnessResult, kind: str) -> float:
    evidence = witness.evidence_bundle
    cost_items = list(evidence.get("cost_items", []))
    routes = list(dict(evidence.get("routes", {})).values())
    total_cost = float(evidence.get("total_cost", 0.0))
    if kind == "more_attractions":
        days = max(1, int(witness.public_task.get("days", 1)))
        return float(len(witness.selected.get("attractions", []))) / days
    if kind == "less_innercity_time":
        return _average([float(_route_minutes(route)) for route in routes])
    if kind == "shorter_meal_transfer":
        entities = dict(evidence.get("entities", {}))
        return _average(
            [
                float(_route_minutes(route))
                for route in routes
                if _entity_type(entities, route.get("destination_place_id")) == "restaurant"
            ]
        )
    if kind == "higher_dining_share":
        return _cost_share(cost_items, {"lunch", "dinner", "breakfast"}, total_cost)
    if kind == "lower_lodging_share":
        return sum(
            float(item.get("amount", 0.0))
            for item in cost_items
            if isinstance(item, Mapping) and item.get("activity_type") == "accommodation"
        )
    if kind == "near_poi":
        return _anchor_distance(witness.selected, evidence)
    if kind == "less_walking":
        return float(sum(_walking_minutes(route) for route in routes))
    if kind == "lower_total_cost":
        return total_cost
    if kind == "relaxed_itinerary":
        return float(
            len(witness.selected.get("attractions", []))
            + (1 if witness.selected.get("restaurant") else 0)
        )
    if kind == "higher_attraction_share":
        return _cost_share(cost_items, {"attraction"}, total_cost)
    if kind == "lower_intercity_share":
        return _cost_share(cost_items, {"train", "airplane"}, total_cost)
    if kind == "shorter_total_travel_time":
        selected = witness.selected
        intercity = sum(
            _transport_minutes(selected.get(leg)) for leg in ("outbound", "return")
        )
        return float(intercity + sum(_route_minutes(route) for route in routes))
    raise SynthesisError(f"Unknown preference metric: {kind}")


def _preference_metric_name(kind: str) -> str:
    return {
        "more_attractions": "daily_attraction_count",
        "less_innercity_time": "average_innercity_route_minutes",
        "shorter_meal_transfer": "average_meal_transfer_minutes",
        "higher_dining_share": "dining_cost_share",
        "lower_lodging_share": "absolute_lodging_cost",
        "near_poi": "average_poi_anchor_distance_km",
        "less_walking": "walking_minutes",
        "lower_total_cost": "total_cost",
        "relaxed_itinerary": "scheduled_stop_count",
        "higher_attraction_share": "attraction_cost_share",
        "lower_intercity_share": "intercity_cost_share",
        "shorter_total_travel_time": "total_travel_minutes",
    }[kind]


def _route_minutes(route: Mapping[str, Any]) -> int:
    total = 0
    for segment in route.get("segments", []):
        if not isinstance(segment, Mapping):
            continue
        start = _clock_minutes(segment.get("start_time"))
        end = _clock_minutes(segment.get("end_time"))
        if start >= 0 and end >= start:
            total += end - start
    return total


def _walking_minutes(route: Mapping[str, Any]) -> int:
    total = 0
    for segment in route.get("segments", []):
        if not isinstance(segment, Mapping) or segment.get("mode") != "walk":
            continue
        start = _clock_minutes(segment.get("start_time"))
        end = _clock_minutes(segment.get("end_time"))
        if start >= 0 and end >= start:
            total += end - start
    return total


def _transport_minutes(value: Any) -> int:
    if not isinstance(value, Mapping):
        return 0
    start = _clock_minutes(value.get("departure_time"))
    end = _clock_minutes(value.get("arrival_time"))
    return max(0, end - start)


def _entity_type(entities: Mapping[str, Any], entity_id: Any) -> str | None:
    entity = entities.get(str(entity_id))
    return str(entity.get("entity_type")) if isinstance(entity, Mapping) else None


def _cost_share(
    cost_items: list[Any], activity_types: set[str], total_cost: float
) -> float:
    if total_cost <= 0:
        return 0.0
    amount = sum(
        float(item.get("amount", 0.0))
        for item in cost_items
        if isinstance(item, Mapping) and item.get("activity_type") in activity_types
    )
    return amount / total_cost


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _anchor_distance(
    selected: Mapping[str, Any], evidence: Mapping[str, Any]
) -> float:
    anchor = selected.get("preference_anchor")
    entities = evidence.get("entities")
    if not isinstance(anchor, Mapping) or not isinstance(entities, Mapping):
        return float("inf")
    distances = []
    seen: set[str] = set()
    for entity_id, entity in entities.items():
        if str(entity_id) in seen or not isinstance(entity, Mapping):
            continue
        if entity.get("entity_type") not in {"attraction", "restaurant", "hotel"}:
            continue
        seen.add(str(entity_id))
        try:
            distances.append(_haversine_km(anchor, entity))
        except (KeyError, TypeError, ValueError):
            return float("inf")
    return _average(distances) if distances else float("inf")


def _haversine_km(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    latitude_left = math.radians(float(left["latitude"]))
    longitude_left = math.radians(float(left["longitude"]))
    latitude_right = math.radians(float(right["latitude"]))
    longitude_right = math.radians(float(right["longitude"]))
    latitude_delta = latitude_right - latitude_left
    longitude_delta = longitude_right - longitude_left
    haversine = (
        math.sin(latitude_delta / 2) ** 2
        + math.cos(latitude_left)
        * math.cos(latitude_right)
        * math.sin(longitude_delta / 2) ** 2
    )
    return 6371.0088 * 2 * math.asin(min(1.0, math.sqrt(haversine)))


def _normalize_question(value: str) -> str:
    """Normalize superficial typography without erasing semantic distinctions."""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _load_task_exclusions(paths: tuple[Path, ...]) -> dict[str, Any]:
    """Load stable identifiers from completed batches used as cross-batch exclusions."""

    task_ids: set[str] = set()
    blueprint_ids: set[str] = set()
    surface_ids: set[str] = set()
    normalized_queries: set[str] = set()
    digest_rows: list[dict[str, str]] = []
    for task_dir in paths:
        records_dir = task_dir / "records"
        manifest_path = task_dir / "manifest.json"
        if not records_dir.is_dir() or not manifest_path.is_file():
            raise SynthesisError(f"Excluded task directory is incomplete: {task_dir}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise SynthesisError(
                f"Excluded task manifest is invalid JSON: {manifest_path}"
            ) from error
        if manifest.get("status") != "complete":
            raise SynthesisError(f"Excluded task directory is not complete: {task_dir}")
        for record_path in sorted(records_dir.glob("*.json")):
            try:
                row = json.loads(record_path.read_text(encoding="utf-8"))
                task_id = str(row["task_spec"]["task_id"])
                blueprint_id = str(row["blueprint"]["blueprint_id"])
                surface_id = str(row["surface"]["surface_id"])
                query = str(row["surface"]["public_query"])
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                raise SynthesisError(
                    f"Excluded synthesis record is invalid: {record_path}"
                ) from error
            normalized_query = _normalize_question(query)
            task_ids.add(task_id)
            blueprint_ids.add(blueprint_id)
            surface_ids.add(surface_id)
            normalized_queries.add(normalized_query)
            digest_rows.append(
                {
                    "task_id": task_id,
                    "blueprint_id": blueprint_id,
                    "surface_id": surface_id,
                    "normalized_query": normalized_query,
                }
            )
    digest_rows.sort(key=lambda row: (row["task_id"], row["blueprint_id"]))
    material = json.dumps(
        digest_rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "task_ids": task_ids,
        "blueprint_ids": blueprint_ids,
        "surface_ids": surface_ids,
        "normalized_queries": normalized_queries,
        "sha256": hashlib.sha256(material.encode("utf-8")).hexdigest(),
    }
