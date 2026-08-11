"""Reward result records shared by collection, training, and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CHECK_PASS = "pass"
CHECK_FAIL = "fail"
CHECK_UNVERIFIABLE = "unverifiable"


@dataclass(frozen=True)
class CheckResult:
    id: str
    source: str
    hardness: str
    status: str
    message: str
    evidence: dict[str, Any]

    def __post_init__(self) -> None:
        if self.source not in {"environment", "task_spec"}:
            raise ValueError("Check source must be environment or task_spec.")
        if self.hardness not in {"hard", "soft"}:
            raise ValueError("Check hardness must be hard or soft.")
        if self.status not in {CHECK_PASS, CHECK_FAIL, CHECK_UNVERIFIABLE}:
            raise ValueError(f"Unknown check status: {self.status}")

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

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["terminal_utility"] = self.reward
        payload["sampling_invalid"] = not self.reward_valid
        payload["checks"] = [check.to_dict() for check in self.checks]
        return payload
