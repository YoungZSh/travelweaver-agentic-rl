"""Command-line utilities for data preparation and environment smoke tests."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..data.bootstrap import install_database, validate_database
from ..data.tasks import JsonlTaskStore, import_benchmark_tasks, import_task_split
from ..env import ChinaTravelBackend, TravelWeaverEnv
from ..errors import TravelWeaverError
from ..paths import project_root
from ..rollout import (
    DeepSeekConfig,
    DeepSeekToolAgent,
    DemoTravelAgent,
    append_trajectory,
    default_trajectory_path,
)
from ..synthesis import SynthesisConfig, SynthesisPipeline


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
    output_dir = Path(args.output_dir) if args.output_dir else project_root() / "data" / "tasks"
    if args.split == "benchmark":
        if args.source_csv:
            raise TravelWeaverError("--source-csv cannot be used with --split benchmark.")
        report = import_benchmark_tasks(output_dir)
    else:
        report = import_task_split(output_dir, split=args.split, source_csv=args.source_csv)
    _print(report)
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


def _rollout_api(args: argparse.Namespace) -> int:
    config = DeepSeekConfig.from_env(args.env_file)
    store = JsonlTaskStore.default(split="easy")
    env = TravelWeaverEnv(ChinaTravelBackend(), store)
    try:
        run = DeepSeekToolAgent(
            env,
            config,
            max_api_turns=args.max_api_turns,
        ).run(task_id=args.task_id, seed=args.seed)
    finally:
        env.close()

    output_path = Path(args.output) if args.output else default_trajectory_path(config.model)
    destination = append_trajectory(output_path, run.to_dict(include_trajectory=True))
    payload = run.to_dict(include_trajectory=args.verbose)
    payload["trajectory_path"] = str(destination.resolve())
    _print(payload)
    return 0 if run.success else 2


def _synthesize_tasks(args: argparse.Namespace) -> int:
    llm_config = DeepSeekConfig.from_env(args.env_file)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else project_root() / "data" / "generated" / "pilot-50-v1"
    )
    report = SynthesisPipeline(
        SynthesisConfig(
            output_dir=output_dir,
            count=args.count,
            seed=args.seed,
            max_api_calls=args.max_api_calls,
        ),
        llm_config,
    ).run()
    _print(report.to_dict())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="travelweaver", description="TravelWeaver environment and rollout utilities."
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
    import_tasks.add_argument(
        "--split",
        default="easy",
        choices=["easy", "medium", "human", "preference_base50", "benchmark"],
    )
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

    rollout_api = subparsers.add_parser(
        "rollout-api", help="Run a model-driven episode through the DeepSeek tool API."
    )
    rollout_api.add_argument("--task-id")
    rollout_api.add_argument("--seed", type=int, default=0)
    rollout_api.add_argument("--env-file", default=str(project_root() / ".env"))
    rollout_api.add_argument("--output", help="Append the complete run to this JSONL file.")
    rollout_api.add_argument("--max-api-turns", type=int, default=40)
    rollout_api.add_argument(
        "--verbose", action="store_true", help="Also print the complete trajectory."
    )
    rollout_api.set_defaults(handler=_rollout_api)

    synthesize = subparsers.add_parser(
        "synthesize-tasks",
        help="Build feasible typed tasks and polish their Chinese query surfaces.",
    )
    synthesize.add_argument("--count", type=int, default=50)
    synthesize.add_argument("--seed", type=int, default=20260807)
    synthesize.add_argument("--env-file", default=str(project_root() / ".env"))
    synthesize.add_argument("--output-dir")
    synthesize.add_argument("--max-api-calls", type=int, default=200)
    synthesize.set_defaults(handler=_synthesize_tasks)
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
