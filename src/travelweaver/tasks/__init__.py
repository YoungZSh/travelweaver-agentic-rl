"""Source-independent task specifications and compilation helpers."""

from .chinatravel import ChinaTravelOracleAdapter
from .compiler import CompileResult, LLMTaskSpecCompiler, build_base_spec
from .models import (
    SPEC_VERSION,
    ConstraintSpec,
    TravelTaskSpec,
    TripSpec,
    supported_constraint_kinds,
)
from .resolver import TaskSpecResolver

__all__ = [
    "SPEC_VERSION",
    "ChinaTravelOracleAdapter",
    "CompileResult",
    "ConstraintSpec",
    "LLMTaskSpecCompiler",
    "TravelTaskSpec",
    "TaskSpecResolver",
    "TripSpec",
    "build_base_spec",
    "supported_constraint_kinds",
]
