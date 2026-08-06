"""ChinaTravel database installation and manifest validation."""

from __future__ import annotations

import shutil
import tarfile
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

from ..errors import DataUnavailableError
from ..paths import project_root

GOOGLE_DRIVE_FOLDER = "https://drive.google.com/drive/folders/1bJ7jA5cfExO_NKxKfi9qgcxEbkYeSdAU"
NJU_DRIVE_FOLDER = "https://box.nju.edu.cn/d/dd83e5a4a9e242ed8eb4/"
CITY_SLUGS = (
    "beijing",
    "shanghai",
    "nanjing",
    "suzhou",
    "hangzhou",
    "shenzhen",
    "chengdu",
    "wuhan",
    "guangzhou",
    "chongqing",
)
CITY_NAMES = ("北京", "上海", "南京", "苏州", "杭州", "深圳", "成都", "武汉", "广州", "重庆")


@dataclass(frozen=True)
class DatabaseReport:
    database_path: str
    valid: bool
    expected_files: int
    present_files: int
    missing_files: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def default_database_path(root: str | Path | None = None) -> Path:
    base = Path(root) if root is not None else project_root()
    return base / "vendor" / "ChinaTravel" / "chinatravel" / "environment" / "database"


def expected_database_files() -> tuple[Path, ...]:
    paths: list[Path] = []
    for slug in CITY_SLUGS:
        paths.extend(
            (
                Path("attractions") / slug / "attractions.csv",
                Path("restaurants") / slug / f"restaurants_{slug}.csv",
                Path("accommodations") / slug / "accommodations.csv",
                Path("poi") / slug / "poi.json",
            )
        )
    for origin in CITY_NAMES:
        for destination in CITY_NAMES:
            if origin != destination:
                paths.append(
                    Path("intercity_transport") / "train" / f"from_{origin}_to_{destination}.json"
                )
    paths.extend(
        (
            Path("intercity_transport") / "airplane.jsonl",
            Path("transportation") / "subways.json",
        )
    )
    return tuple(paths)


def validate_database(path: str | Path | None = None) -> DatabaseReport:
    database = Path(path) if path is not None else default_database_path()
    expected = expected_database_files()
    missing = tuple(str(relative) for relative in expected if not (database / relative).is_file())
    return DatabaseReport(
        database_path=str(database),
        valid=not missing,
        expected_files=len(expected),
        present_files=len(expected) - len(missing),
        missing_files=missing,
    )


def _validate_member(destination: Path, member_name: str) -> None:
    target = (destination / member_name).resolve()
    if destination.resolve() not in target.parents and target != destination.resolve():
        raise DataUnavailableError(f"Archive contains unsafe path: {member_name}")


def _extract_archive(archive: Path, destination: Path) -> None:
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as handle:
            for member in handle.namelist():
                _validate_member(destination, member)
            handle.extractall(destination)
        return
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as handle:
            for member in handle.getmembers():
                _validate_member(destination, member.name)
                if member.issym() or member.islnk():
                    raise DataUnavailableError("Database archive may not contain links.")
            handle.extractall(destination)
        return
    raise DataUnavailableError(f"Unsupported archive format: {archive}")


def _find_database_tree(root: Path) -> Path | None:
    candidates = [root] + [path for path in root.rglob("database") if path.is_dir()]
    for candidate in candidates:
        if (candidate / "attractions").is_dir() and (candidate / "intercity_transport").is_dir():
            return candidate
    return None


def _expand_archives(root: Path) -> None:
    for path in list(root.rglob("*")):
        if not path.is_file():
            continue
        if zipfile.is_zipfile(path) or tarfile.is_tarfile(path):
            target = path.parent / f"{path.name}.extracted"
            target.mkdir(parents=True, exist_ok=True)
            _extract_archive(path, target)


def _download_official_folder(destination: Path) -> None:
    try:
        import gdown
    except ImportError as error:
        raise DataUnavailableError(
            "gdown is required for the official Google Drive download."
        ) from error
    try:
        files = gdown.download_folder(
            url=GOOGLE_DRIVE_FOLDER,
            output=str(destination),
            quiet=False,
            use_cookies=False,
            remaining_ok=True,
        )
    except Exception as error:
        raise DataUnavailableError(f"Official Google Drive download failed: {error}") from error
    if not files:
        raise DataUnavailableError("Official Google Drive folder returned no files.")


def install_database(
    *,
    archive: str | Path | None = None,
    destination: str | Path | None = None,
    force: bool = False,
) -> DatabaseReport:
    """Install from a local archive or the official Google Drive folder."""

    target = Path(destination) if destination is not None else default_database_path()
    current = validate_database(target)
    if current.valid and not force:
        return current
    if target.exists() and any(target.iterdir()) and not force:
        raise DataUnavailableError(
            f"Database directory exists but is incomplete: {target}. "
            "Use --force only after confirming it can be replaced."
        )

    with tempfile.TemporaryDirectory(prefix="travelweaver-database-") as temporary:
        staging = Path(temporary)
        if archive is not None:
            source = Path(archive).expanduser().resolve()
            if not source.is_file():
                raise DataUnavailableError(f"Database archive not found: {source}")
            _extract_archive(source, staging)
        else:
            _download_official_folder(staging)
            _expand_archives(staging)
        database_tree = _find_database_tree(staging)
        if database_tree is None:
            raise DataUnavailableError(
                "Downloaded files do not contain a recognizable database tree. "
                f"Manual fallback: download from {NJU_DRIVE_FOLDER}, then rerun with --archive."
            )
        staged_report = validate_database(database_tree)
        if not staged_report.valid:
            raise DataUnavailableError(
                "Database archive is incomplete; first missing files: "
                + ", ".join(staged_report.missing_files[:10])
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(database_tree, target)
    return validate_database(target)
