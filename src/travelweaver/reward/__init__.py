"""Deterministic, source-independent TravelEnv reward."""

from .contract import OUTCOME_CONTRACT_VERSION, FrozenOutcomeContract
from .models import (
    CHECK_BLOCKED,
    CHECK_FAIL,
    CHECK_NOT_APPLICABLE,
    CHECK_PASS,
    CHECK_UNVERIFIABLE,
    DIMENSION_ARTIFACT,
    DIMENSION_GOAL,
    DIMENSION_VALIDITY,
    REWARD_DIMENSIONS,
    CheckResult,
    RewardResult,
)
from .registry import CHECK_DEFINITIONS, CHECK_OWNER, CheckDefinition
from .reward import REWARD_VERSION, TravelReward
from .rft import RFTDecision, strict_rft_filter

__all__ = [
    "CHECK_BLOCKED",
    "CHECK_DEFINITIONS",
    "CHECK_FAIL",
    "CHECK_NOT_APPLICABLE",
    "CHECK_OWNER",
    "CHECK_PASS",
    "CHECK_UNVERIFIABLE",
    "DIMENSION_ARTIFACT",
    "DIMENSION_GOAL",
    "DIMENSION_VALIDITY",
    "FrozenOutcomeContract",
    "OUTCOME_CONTRACT_VERSION",
    "REWARD_VERSION",
    "REWARD_DIMENSIONS",
    "CheckResult",
    "CheckDefinition",
    "RFTDecision",
    "RewardResult",
    "TravelReward",
    "strict_rft_filter",
]
