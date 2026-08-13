"""Reward result records shared by collection, training, and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CHECK_PASS = "pass"
CHECK_FAIL = "fail"
CHECK_BLOCKED = "blocked"
CHECK_UNVERIFIABLE = "unverifiable"
CHECK_NOT_APPLICABLE = "not_applicable"

DIMENSION_ARTIFACT = "artifact_conformance"
DIMENSION_VALIDITY = "environment_validity"
DIMENSION_GOAL = "goal_satisfaction"
REWARD_DIMENSIONS = (DIMENSION_ARTIFACT, DIMENSION_VALIDITY, DIMENSION_GOAL)


@dataclass(frozen=True)
class CheckResult:
    id: str
    source: str
    hardness: str
    status: str
    message: str
    evidence: dict[str, Any]
    owner_dimension: str = DIMENSION_VALIDITY
    score: float | None = None
    blocked_by: str | None = None
    affects_success: bool = True
    affects_shaping: bool = True

    def __post_init__(self) -> None:
        if self.source == "task_spec" and self.owner_dimension == DIMENSION_VALIDITY:
            object.__setattr__(self, "owner_dimension", DIMENSION_GOAL)
        if self.score is None and self.status == CHECK_PASS:
            object.__setattr__(self, "score", 1.0)
        elif self.score is None and self.status == CHECK_FAIL:
            object.__setattr__(self, "score", 0.0)
        if self.source not in {"artifact", "environment", "task_spec"}:
            raise ValueError("Check source must be artifact, environment, or task_spec.")
        if self.hardness not in {"hard", "soft"}:
            raise ValueError("Check hardness must be hard or soft.")
        if self.status not in {
            CHECK_PASS,
            CHECK_FAIL,
            CHECK_BLOCKED,
            CHECK_UNVERIFIABLE,
            CHECK_NOT_APPLICABLE,
        }:
            raise ValueError(f"Unknown check status: {self.status}")
        if self.owner_dimension not in REWARD_DIMENSIONS:
            raise ValueError(f"Unknown Reward dimension: {self.owner_dimension}")
        if self.score is not None and not 0.0 <= self.score <= 1.0:
            raise ValueError("Check score must be inside [0, 1].")
        expected_score = {CHECK_PASS: 1.0, CHECK_BLOCKED: None, CHECK_NOT_APPLICABLE: None}
        if self.status in expected_score and self.score != expected_score[self.status]:
            raise ValueError(
                f"Check status {self.status} requires score {expected_score[self.status]}."
            )
        if self.status == CHECK_FAIL and self.score == 1.0:
            raise ValueError("Failed checks cannot have score 1.")
        if self.status == CHECK_BLOCKED and not self.blocked_by:
            raise ValueError("Blocked checks must identify blocked_by.")
        if self.status != CHECK_BLOCKED and self.blocked_by is not None:
            raise ValueError("Only blocked checks may set blocked_by.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RewardResult:
    reward_version: str
    reward: float
    reward_type: str
    reward_valid: bool
    termination_reason: str
    hard_score: float
    soft_score: float
    all_hard_pass: bool
    checks: tuple[CheckResult, ...]
    task_spec_hash: str | None
    group_results: dict[str, bool]
    sft_accepted: bool
    rl_reward: float
    dimension_scores: dict[str, float]
    dimension_coverage: dict[str, dict[str, int]]
    outcome_contract_hash: str | None
    admission_passed: bool

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["terminal_utility"] = self.reward
        payload["sampling_invalid"] = not self.reward_valid
        payload["checks"] = [check.to_dict() for check in self.checks]
        return payload
