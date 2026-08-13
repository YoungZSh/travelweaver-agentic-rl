"""Shared short-trajectory constraints for synthesis and SFT teachers."""

from __future__ import annotations

from typing import Any

TRAJECTORY_POLICY_VERSION = "travelweaver-short-trajectory-policy-v5"
MAX_SYNTHESIS_VALID_STEPS = 50
MAX_CONSECUTIVE_TOOL_CALLS = 3
MAX_PROGRAMMATIC_CATALOG_ACTIONS = 3
# Witnesses leave enough room for task-grounded catalog checks and optional evidence
# subgraphs. Ordinary unnamed witness entities must be visible on the first filtered
# search page; pagination is reserved for an explicit unresolved selection predicate.
MAX_WITNESS_VALID_STEPS = (
    MAX_SYNTHESIS_VALID_STEPS
    - MAX_PROGRAMMATIC_CATALOG_ACTIONS
    - MAX_CONSECUTIVE_TOOL_CALLS
)


def trajectory_policy() -> dict[str, Any]:
    """Return the policy serialized into every newly synthesized task record."""

    return {
        "policy_version": TRAJECTORY_POLICY_VERSION,
        "max_valid_steps": MAX_SYNTHESIS_VALID_STEPS,
        "max_consecutive_tool_calls": MAX_CONSECUTIVE_TOOL_CALLS,
    }
