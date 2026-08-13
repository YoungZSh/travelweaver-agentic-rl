"""Frozen, hashable outcome contract compiled before a rollout starts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from ..tasks import TravelTaskSpec

OUTCOME_CONTRACT_VERSION = "travelweaver-outcome-contract-v1"


@dataclass(frozen=True)
class FrozenOutcomeContract:
    version: str
    task_spec_hash: str
    task_id: str
    active_constraint_ids: tuple[str, ...]
    contract_hash: str

    @classmethod
    def compile(cls, spec: TravelTaskSpec) -> FrozenOutcomeContract:
        payload = {
            "version": OUTCOME_CONTRACT_VERSION,
            "task_spec_hash": spec.spec_hash,
            "task_id": spec.task_id,
            "active_constraint_ids": [constraint.id for constraint in spec.constraints],
        }
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        return cls(
            version=OUTCOME_CONTRACT_VERSION,
            task_spec_hash=spec.spec_hash,
            task_id=spec.task_id,
            active_constraint_ids=tuple(payload["active_constraint_ids"]),
            contract_hash=hashlib.sha256(encoded).hexdigest(),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
