"""Command-line utilities for data preparation and environment smoke tests."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .agent import DemoTravelAgent
from .backend import ChinaTravelBackend
from .bootstrap import install_database, validate_database
from .environment import TravelWeaverEnv
from .errors import TravelWeaverError
from .tasks import JsonlTaskStore, import_easy_tasks, project_root


def _print(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _bootstrap(args: argparse.Namespace) -> int:
    if args.verify_only:
        report = validate_database(args.destination)
    else:
        report = install_database(
            archive=args.archive,
            destination=args.destination,
            force=args.force,
        )
    _print(report.to_dict())
    return 0 if report.valid else 2


def _import_tasks(args: argparse.Namespace) -> int:
    if args.split != "easy":
        raise TravelWeaverError("The MVP importer currently supports only --split easy.")
    output_dir = Path(args.output_dir) if args.output_dir else project_root() / "data" / "tasks"
    _print(import_easy_tasks(output_dir, source_csv=args.source_csv))
    return 0


def _smoke(args: argparse.Namespace) -> int:
    store = JsonlTaskStore.default(split="easy")
    backend = ChinaTravelBackend()
    env = TravelWeaverEnv(backend, store)
    observation = env.reset(task_id=args.task_id, seed=args.seed)
    _print({"event": "reset", "observation": observation.to_dict()})
    city = str(observation.task["target_city"])
    search = env.step({"tool": "search_attractions", "arguments": {"city": city}})
    _print({"event": "step", "result": search.to_dict()})
    items = (search.observation.tool_result or {}).get("items", [])
    if items and not search.terminated and not search.truncated:
        inspect = env.step(
            {"tool": "inspect_place", "arguments": {"place_id": items[0]["place_id"]}}
        )
        _print({"event": "step", "result": inspect.to_dict()})
    env.close()
    return 0


def _run_agent(args: argparse.Namespace) -> int:
    store = JsonlTaskStore.default(split="easy")
    env = TravelWeaverEnv(ChinaTravelBackend(), store)
    try:
        run = DemoTravelAgent(env).run(task_id=args.task_id, seed=args.seed)
        _print(run.to_dict(include_trajectory=args.verbose))
        return 0 if run.success else 2
    finally:
        env.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="travelweaver", description="TravelWeaverEnv data and smoke-test utilities."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    bootstrap = subparsers.add_parser("bootstrap", help="Install or verify snapshot data.")
    bootstrap.add_argument("target", choices=["chinatravel"])
    bootstrap.add_argument("--archive", help="Local official database zip/tar archive.")
    bootstrap.add_argument("--destination", help="Override database installation directory.")
    bootstrap.add_argument("--force", action="store_true", help="Replace an incomplete database.")
    bootstrap.add_argument(
        "--verify-only", action="store_true", help="Only validate the existing database."
    )
    bootstrap.set_defaults(handler=_bootstrap)

    import_tasks = subparsers.add_parser("import-tasks", help="Import a pinned task split.")
    import_tasks.add_argument("--split", default="easy", choices=["easy"])
    import_tasks.add_argument("--output-dir")
    import_tasks.add_argument("--source-csv", help="Testing/manual source instead of Hub download.")
    import_tasks.set_defaults(handler=_import_tasks)

    smoke = subparsers.add_parser("smoke-env", help="Run a real reset/search/inspect sequence.")
    smoke.add_argument("--task-id")
    smoke.add_argument("--seed", type=int, default=0)
    smoke.set_defaults(handler=_smoke)

    run_agent = subparsers.add_parser(
        "run-agent", help="Run the deterministic agent through query/candidate/submit."
    )
    run_agent.add_argument("--task-id")
    run_agent.add_argument("--seed", type=int, default=0)
    run_agent.add_argument("--verbose", action="store_true", help="Include the full trajectory.")
    run_agent.set_defaults(handler=_run_agent)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except TravelWeaverError as error:
        print(f"travelweaver: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
