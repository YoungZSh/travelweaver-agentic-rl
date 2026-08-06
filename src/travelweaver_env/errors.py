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
