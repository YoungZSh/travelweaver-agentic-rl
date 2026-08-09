"""TravelWeaver exception types."""


class TravelWeaverError(Exception):
    """Base exception for the package."""


class DataUnavailableError(TravelWeaverError):
    """Required snapshot data is absent or incomplete."""


class BackendQueryError(TravelWeaverError):
    """A well-formed query cannot be served by the backend."""


class EnvironmentStateError(TravelWeaverError):
    """The requested operation is invalid for the current episode state."""


class TaskNotFoundError(TravelWeaverError):
    """A requested task identifier is not present in the task store."""


class ConfigurationError(TravelWeaverError):
    """Required runtime configuration is absent or invalid."""


class ApiRolloutError(TravelWeaverError):
    """An external model request failed during a rollout."""


class TaskSpecError(TravelWeaverError):
    """A task cannot be converted to a safe, supported TravelTaskSpec."""


class JudgeError(TravelWeaverError):
    """An offline LLM Judge response is absent or violates its blind contract."""


class SynthesisError(TravelWeaverError):
    """A synthetic task cannot be grounded, polished, or validated safely."""


class SFTRebuildError(TravelWeaverError):
    """A rollout cannot be rebuilt into a safe, replay-verified SFT sample."""
