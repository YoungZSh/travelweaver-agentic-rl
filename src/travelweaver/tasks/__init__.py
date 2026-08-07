"""Source-independent task specifications and compilation helpers."""

from .blueprint import (
    BLUEPRINT_VERSION,
    SURFACE_VERSION,
    BlueprintConstraint,
    ConstraintMention,
    TaskBlueprint,
    TaskSurface,
    materialize_task_spec,
)
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
    "BLUEPRINT_VERSION",
    "SURFACE_VERSION",
    "BlueprintConstraint",
    "ChinaTravelOracleAdapter",
    "CompileResult",
    "ConstraintSpec",
    "ConstraintMention",
    "LLMTaskSpecCompiler",
    "TravelTaskSpec",
    "TaskSpecResolver",
    "TaskBlueprint",
    "TaskSurface",
    "TripSpec",
    "build_base_spec",
    "materialize_task_spec",
    "supported_constraint_kinds",
]
