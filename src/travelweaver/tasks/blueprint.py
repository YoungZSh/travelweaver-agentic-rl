"""Query-independent task blueprints and their natural-language surfaces."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from ..errors import TaskSpecError
from .models import ConstraintSpec, TravelTaskSpec, TripSpec, stable_hash

BLUEPRINT_VERSION = "travelweaver-task-blueprint-v2"
LEGACY_BLUEPRINT_VERSION = "travelweaver-task-blueprint-v1"
SURFACE_VERSION = "travelweaver-task-surface-v3"
PREVIOUS_SURFACE_VERSION = "travelweaver-task-surface-v2"
LEGACY_SURFACE_VERSION = "travelweaver-task-surface-v1"


@dataclass(frozen=True)
class BlueprintConstraint:
    id: str
    kind: str
    operator: str
    value: Any
    scope: str
    hardness: str = "hard"

    def __post_init__(self) -> None:
        # ConstraintSpec remains the single source of truth for supported typed DSL forms.
        ConstraintSpec(
            id=self.id,
            kind=self.kind,
            operator=self.operator,
            value=self.value,
            scope=self.scope,
            hardness=self.hardness,
            source_text="blueprint",
        )

    def semantic_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "operator": self.operator,
            "scope": self.scope,
            "hardness": self.hardness,
            "value": self.value,
        }


@dataclass(frozen=True)
class BlueprintPreference:
    id: str
    kind: str
    direction: str
    value: Any = None

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.kind.strip():
            raise TaskSpecError("Blueprint preferences require non-empty ids and kinds.")
        if self.direction not in {"minimize", "maximize"}:
            raise TaskSpecError("Blueprint preference direction must be minimize or maximize.")

    def semantic_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "direction": self.direction, "value": self.value}


@dataclass(frozen=True)
class TaskBlueprint:
    trip: TripSpec
    constraints: tuple[BlueprintConstraint, ...]
    world_snapshot_version: str
    generator_version: str
    generation_seed: int
    preferences: tuple[BlueprintPreference, ...] = ()
    persona_context: str | None = None
    metadata_prefix: str | None = None
    schema_version: str = BLUEPRINT_VERSION

    def __post_init__(self) -> None:
        if self.schema_version not in {LEGACY_BLUEPRINT_VERSION, BLUEPRINT_VERSION}:
            raise TaskSpecError(f"Unsupported Blueprint version: {self.schema_version}")
        if not self.world_snapshot_version.strip() or not self.generator_version.strip():
            raise TaskSpecError("Blueprint provenance fields cannot be empty.")
        ids = [constraint.id for constraint in self.constraints]
        if len(ids) != len(set(ids)) or any(not value.strip() for value in ids):
            raise TaskSpecError("Blueprint constraints must have unique non-empty ids.")
        preference_ids = [preference.id for preference in self.preferences]
        if len(preference_ids) != len(set(preference_ids)):
            raise TaskSpecError("Blueprint preference ids must be unique.")

    @property
    def semantic_hash(self) -> str:
        constraints = sorted(
            (constraint.semantic_dict() for constraint in self.constraints),
            key=lambda value: stable_hash(value),
        )
        if self.schema_version == LEGACY_BLUEPRINT_VERSION:
            return stable_hash({"trip": asdict(self.trip), "constraints": constraints})
        preferences = sorted(
            (preference.semantic_dict() for preference in self.preferences),
            key=lambda value: stable_hash(value),
        )
        return stable_hash(
            {
                "trip": asdict(self.trip),
                "constraints": constraints,
                "preferences": preferences,
                "persona_context": self.persona_context,
                "metadata_prefix": self.metadata_prefix,
            }
        )

    @property
    def blueprint_id(self) -> str:
        return f"twb_{self.semantic_hash[:20]}"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["blueprint_id"] = self.blueprint_id
        payload["semantic_hash"] = self.semantic_hash
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TaskBlueprint:
        trip = dict(payload["trip"])
        trip["destinations"] = tuple(trip["destinations"])
        blueprint = cls(
            trip=TripSpec(**trip),
            constraints=tuple(
                BlueprintConstraint(**dict(value)) for value in payload["constraints"]
            ),
            world_snapshot_version=str(payload["world_snapshot_version"]),
            generator_version=str(payload["generator_version"]),
            generation_seed=int(payload["generation_seed"]),
            preferences=tuple(
                BlueprintPreference(**dict(value))
                for value in payload.get("preferences", [])
            ),
            persona_context=(
                str(payload["persona_context"])
                if payload.get("persona_context") is not None
                else None
            ),
            metadata_prefix=(
                str(payload["metadata_prefix"])
                if payload.get("metadata_prefix") is not None
                else None
            ),
            schema_version=str(payload.get("schema_version", BLUEPRINT_VERSION)),
        )
        if payload.get("semantic_hash", blueprint.semantic_hash) != blueprint.semantic_hash:
            raise TaskSpecError("Blueprint semantic hash does not match its contents.")
        if payload.get("blueprint_id", blueprint.blueprint_id) != blueprint.blueprint_id:
            raise TaskSpecError("Blueprint id does not match its contents.")
        return blueprint


@dataclass(frozen=True)
class ConstraintMention:
    constraint_id: str
    text: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if not self.constraint_id.strip() or not self.text.strip():
            raise TaskSpecError("Constraint mentions require an id and text.")
        if self.start < 0 or self.end <= self.start:
            raise TaskSpecError("Constraint mention offsets are invalid.")


@dataclass(frozen=True)
class PreferenceMention:
    preference_id: str
    text: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if not self.preference_id.strip() or not self.text.strip():
            raise TaskSpecError("Preference mentions require an id and text.")
        if self.start < 0 or self.end <= self.start:
            raise TaskSpecError("Preference mention offsets are invalid.")


@dataclass(frozen=True)
class TaskSurface:
    blueprint_id: str
    public_query: str
    canonical_query: str
    mentions: tuple[ConstraintMention, ...]
    language: str
    polisher_model: str
    prompt_version: str
    usage: Mapping[str, int]
    preference_mentions: tuple[PreferenceMention, ...] = ()
    validation_policy: str = "strict"
    validation_warnings: tuple[str, ...] = ()
    schema_version: str = SURFACE_VERSION

    def __post_init__(self) -> None:
        if self.schema_version not in {
            LEGACY_SURFACE_VERSION,
            PREVIOUS_SURFACE_VERSION,
            SURFACE_VERSION,
        }:
            raise TaskSpecError(f"Unsupported Surface version: {self.schema_version}")
        if not self.blueprint_id.strip() or not self.public_query.strip():
            raise TaskSpecError("Surface blueprint id and query are required.")
        if not self.canonical_query.strip() or not self.language.strip():
            raise TaskSpecError("Surface canonical query and language are required.")
        if not self.polisher_model.strip() or not self.prompt_version.strip():
            raise TaskSpecError("Surface polisher provenance is required.")
        if self.validation_policy not in {"strict", "minimal_semantic"}:
            raise TaskSpecError("Surface validation policy is unsupported.")
        ids = [mention.constraint_id for mention in self.mentions]
        if len(ids) != len(set(ids)):
            raise TaskSpecError("Surface mention constraint ids must be unique.")
        ordered = sorted(self.mentions, key=lambda mention: mention.start)
        for previous, current in zip(ordered, ordered[1:], strict=False):
            if self.validation_policy == "strict" and previous.end > current.start:
                raise TaskSpecError("Surface constraint mentions cannot overlap.")
        for mention in self.mentions:
            if self.public_query[mention.start : mention.end] != mention.text:
                raise TaskSpecError("Surface mention offsets do not match the public query.")
        preference_ids = [mention.preference_id for mention in self.preference_mentions]
        if len(preference_ids) != len(set(preference_ids)):
            raise TaskSpecError("Surface preference mention ids must be unique.")
        all_mentions = [*self.mentions, *self.preference_mentions]
        ordered_all = sorted(all_mentions, key=lambda mention: mention.start)
        for previous, current in zip(ordered_all, ordered_all[1:], strict=False):
            if self.validation_policy == "strict" and previous.end > current.start:
                raise TaskSpecError("Surface hard and preference mentions cannot overlap.")
        for mention in self.preference_mentions:
            if self.public_query[mention.start : mention.end] != mention.text:
                raise TaskSpecError("Surface preference offsets do not match the public query.")

    @property
    def surface_id(self) -> str:
        material = {
            "blueprint_id": self.blueprint_id,
            "language": self.language,
            "public_query": self.public_query,
        }
        return f"tws_{stable_hash(material)[:20]}"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["usage"] = dict(self.usage)
        payload["validation_status"] = (
            "accepted_with_warnings" if self.validation_warnings else "accepted"
        )
        payload["surface_id"] = self.surface_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TaskSurface:
        surface = cls(
            blueprint_id=str(payload["blueprint_id"]),
            public_query=str(payload["public_query"]),
            canonical_query=str(payload["canonical_query"]),
            mentions=tuple(
                ConstraintMention(**dict(value)) for value in payload.get("mentions", [])
            ),
            language=str(payload["language"]),
            polisher_model=str(payload["polisher_model"]),
            prompt_version=str(payload["prompt_version"]),
            usage={str(key): int(value) for key, value in dict(payload.get("usage", {})).items()},
            preference_mentions=tuple(
                PreferenceMention(**dict(value))
                for value in payload.get("preference_mentions", [])
            ),
            validation_policy=str(payload.get("validation_policy", "strict")),
            validation_warnings=tuple(
                str(value) for value in payload.get("validation_warnings", [])
            ),
            schema_version=str(payload.get("schema_version", SURFACE_VERSION)),
        )
        if payload.get("surface_id", surface.surface_id) != surface.surface_id:
            raise TaskSpecError("Surface id does not match its contents.")
        return surface


def materialize_task_spec(
    blueprint: TaskBlueprint,
    surface: TaskSurface,
    *,
    task_id: str | None = None,
) -> TravelTaskSpec:
    if surface.blueprint_id != blueprint.blueprint_id:
        raise TaskSpecError("Surface does not belong to the supplied Blueprint.")
    mentions = {mention.constraint_id: mention for mention in surface.mentions}
    if set(mentions) != {constraint.id for constraint in blueprint.constraints}:
        raise TaskSpecError("Surface mentions must cover every Blueprint constraint exactly once.")
    constraints = tuple(
        ConstraintSpec(
            id=constraint.id,
            kind=constraint.kind,
            operator=constraint.operator,
            value=constraint.value,
            scope=constraint.scope,
            hardness=constraint.hardness,
            source_text=mentions[constraint.id].text,
            source_start=mentions[constraint.id].start,
            source_end=mentions[constraint.id].end,
        )
        for constraint in blueprint.constraints
    )
    input_hash = stable_hash(
        {"blueprint_id": blueprint.blueprint_id, "surface_id": surface.surface_id}
    )
    return TravelTaskSpec(
        task_id=task_id or surface.surface_id,
        public_query=surface.public_query,
        trip=blueprint.trip,
        constraints=constraints,
        unscored_preferences=tuple(
            mention.text for mention in surface.preference_mentions
        ),
        source="synthetic_blueprint",
        compiler_version=surface.prompt_version,
        input_hash=input_hash,
        world_snapshot_version=blueprint.world_snapshot_version,
    )
