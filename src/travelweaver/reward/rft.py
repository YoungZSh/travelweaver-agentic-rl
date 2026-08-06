"""Strict acceptance filter for closed-model RFT trajectory collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import RewardResult


@dataclass(frozen=True)
class RFTDecision:
    accepted: bool
    reason: str
    reward: float
    reward_version: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "reward": self.reward,
            "reward_version": self.reward_version,
        }


def strict_rft_filter(result: RewardResult, *, termination_reason: str) -> RFTDecision:
    if termination_reason != "plan_submitted":
        return RFTDecision(
            False,
            "trajectory_did_not_submit_plan",
            result.reward,
            result.reward_version,
        )
    if not result.reward_valid:
        return RFTDecision(False, "reward_unverifiable", result.reward, result.reward_version)
    if not result.all_hard_pass:
        return RFTDecision(False, "hard_constraint_failure", result.reward, result.reward_version)
    return RFTDecision(True, "all_hard_constraints_passed", result.reward, result.reward_version)
