"""Repository paths shared by lightweight TravelWeaver components."""

from pathlib import Path


def project_root() -> Path:
    """Return the repository root for an editable or source checkout."""

    return Path(__file__).resolve().parents[2]
