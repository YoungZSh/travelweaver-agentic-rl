from __future__ import annotations

import zipfile
from pathlib import Path

from travelweaver_env.bootstrap import (
    expected_database_files,
    install_database,
    validate_database,
)


def _make_database(root: Path) -> Path:
    database = root / "database"
    for relative in expected_database_files():
        path = database / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    return database


def test_database_manifest_reports_missing_and_complete(tmp_path: Path) -> None:
    missing = validate_database(tmp_path / "missing")
    assert not missing.valid
    assert missing.expected_files == 132
    assert missing.present_files == 0

    database = _make_database(tmp_path / "complete")
    complete = validate_database(database)
    assert complete.valid
    assert complete.present_files == 132


def test_install_database_from_archive(tmp_path: Path) -> None:
    source = _make_database(tmp_path / "source")
    archive = tmp_path / "database.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for path in source.rglob("*"):
            if path.is_file():
                handle.write(path, Path("official") / "database" / path.relative_to(source))
    destination = tmp_path / "installed" / "database"
    report = install_database(archive=archive, destination=destination)
    assert report.valid
    assert validate_database(destination).valid
