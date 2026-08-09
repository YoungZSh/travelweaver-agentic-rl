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
from ..llm import DeepSeekConfig, OpenAICompatibleConfig
from ..paths import project_root
from ..rollout import (
    DEFAULT_ROLLOUT_CONCURRENCY,
    DEFAULT_TOOL_RESPONSE_MODE,
    TOOL_RESPONSE_MODES,
    BenchmarkRolloutBatchConfig,
    DemoTravelAgent,
    GeneratedRolloutBatchConfig,
    ToolCallingAgent,
    append_trajectory,
    default_trajectory_path,
    run_benchmark_rollout_batch,
    run_generated_rollout_batch,
)
from ..sft import SFTRebuildConfig, SFTSource, rebuild_sft_dataset
from ..synthesis import (
    RepolishConfig,
    SurfaceRepolishPipeline,
    SynthesisConfig,
    SynthesisPipeline,
)


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
        run = ToolCallingAgent(
            env,
            config,
            max_api_turns=args.max_api_turns,
            tool_response_mode=args.tool_response_mode,
        ).run(task_id=args.task_id, seed=args.seed)
    finally:
        env.close()

    output_path = Path(args.output) if args.output else default_trajectory_path(config.model)
    destination = append_trajectory(output_path, run.to_dict(include_trajectory=True))
    payload = run.to_dict(include_trajectory=args.verbose)
    payload["trajectory_path"] = str(destination.resolve())
    _print(payload)
    return 0 if run.success else 2


def _rollout_generated(args: argparse.Namespace) -> int:
    llm_config = DeepSeekConfig.from_env(args.env_file)
    output_path = Path(args.output)
    error_path = (
        Path(args.errors)
        if args.errors
        else output_path.with_name(f"{output_path.stem}-errors.jsonl")
    )
    report = run_generated_rollout_batch(
        GeneratedRolloutBatchConfig(
            input_dir=Path(args.input_dir),
            output_path=output_path,
            error_path=error_path,
            concurrency=args.concurrency,
            max_api_turns=args.max_api_turns,
            seed=args.seed,
            task_id=args.task_id,
            tool_response_mode=args.tool_response_mode,
        ),
        llm_config,
        progress=lambda payload: print(
            json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True
        ),
    )
    return 0 if report.errors == 0 else 2


def _rollout_benchmark(args: argparse.Namespace) -> int:
    llm_config = OpenAICompatibleConfig.from_env(args.env_file)
    output_path = Path(args.output)
    error_path = (
        Path(args.errors)
        if args.errors
        else output_path.with_name(f"{output_path.stem}-errors.jsonl")
    )
    report = run_benchmark_rollout_batch(
        BenchmarkRolloutBatchConfig(
            split=args.split,
            output_path=output_path,
            error_path=error_path,
            concurrency=args.concurrency,
            max_api_turns=args.max_api_turns,
            seed=args.seed,
            task_id=args.task_id,
            tool_response_mode=args.tool_response_mode,
        ),
        llm_config,
        progress=lambda payload: print(
            json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True
        ),
    )
    return 0 if report.errors == 0 else 2


def _synthesize_tasks(args: argparse.Namespace) -> int:
    llm_config = DeepSeekConfig.from_env(args.env_file)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else project_root() / "data" / "generated" / "pilot-100-v2.1"
    )
    report = SynthesisPipeline(
        SynthesisConfig(
            output_dir=output_dir,
            count=args.count,
            seed=args.seed,
            max_api_calls=args.max_api_calls,
            profile=args.profile,
            validation_policy=args.validation_policy,
        ),
        llm_config,
    ).run()
    _print(report.to_dict())
    return 0


def _repolish_tasks(args: argparse.Namespace) -> int:
    llm_config = DeepSeekConfig.from_env(args.env_file)
    input_dir = Path(args.input_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else input_dir.with_name(f"{input_dir.name}-repolished")
    )
    report = SurfaceRepolishPipeline(
        RepolishConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            llm_concurrency=args.llm_concurrency,
            max_api_calls=args.max_api_calls,
            validation_policy=args.validation_policy,
        ),
        llm_config,
    ).run()
    _print(report.to_dict())
    return 0


def _rebuild_sft(args: argparse.Namespace) -> int:
    report = rebuild_sft_dataset(
        SFTRebuildConfig(
            sources=tuple(
                SFTSource(task_dir=Path(task_dir), rollout_path=Path(rollout_path))
                for task_dir, rollout_path in args.source
            ),
            output_dir=Path(args.output_dir),
            repair_surface_semantics=args.repair_surface_semantics,
            tool_response_mode=args.tool_response_mode,
        )
    )
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
        "--tool-response-mode",
        choices=TOOL_RESPONSE_MODES,
        default=DEFAULT_TOOL_RESPONSE_MODE,
        help="Send per-step deltas by default; use snapshot to reproduce the legacy payload.",
    )
    rollout_api.add_argument(
        "--verbose", action="store_true", help="Also print the complete trajectory."
    )
    rollout_api.set_defaults(handler=_rollout_api)

    rollout_generated = subparsers.add_parser(
        "rollout-generated",
        help="Run one resumable model trajectory per task in a generated task directory.",
    )
    rollout_generated.add_argument("--input-dir", required=True)
    rollout_generated.add_argument("--output", required=True)
    rollout_generated.add_argument("--errors")
    rollout_generated.add_argument("--task-id")
    rollout_generated.add_argument("--seed", type=int, default=20260808)
    rollout_generated.add_argument("--env-file", default=str(project_root() / ".env"))
    rollout_generated.add_argument("--max-api-turns", type=int, default=40)
    rollout_generated.add_argument(
        "--tool-response-mode",
        choices=TOOL_RESPONSE_MODES,
        default=DEFAULT_TOOL_RESPONSE_MODE,
    )
    rollout_generated.add_argument(
        "--concurrency", type=int, default=DEFAULT_ROLLOUT_CONCURRENCY
    )
    rollout_generated.set_defaults(handler=_rollout_generated)

    rollout_benchmark = subparsers.add_parser(
        "rollout-benchmark",
        help="Run resumable local/API rollouts over a pinned official ChinaTravel split.",
    )
    rollout_benchmark.add_argument(
        "--split", choices=["easy", "medium", "human", "benchmark"], default="benchmark"
    )
    rollout_benchmark.add_argument("--output", required=True)
    rollout_benchmark.add_argument("--errors")
    rollout_benchmark.add_argument("--task-id")
    rollout_benchmark.add_argument("--seed", type=int, default=20260808)
    rollout_benchmark.add_argument("--env-file", default=str(project_root() / ".env"))
    rollout_benchmark.add_argument("--max-api-turns", type=int, default=40)
    rollout_benchmark.add_argument(
        "--tool-response-mode",
        choices=TOOL_RESPONSE_MODES,
        default=DEFAULT_TOOL_RESPONSE_MODE,
    )
    rollout_benchmark.add_argument("--concurrency", type=int, default=16)
    rollout_benchmark.set_defaults(handler=_rollout_benchmark)

    synthesize = subparsers.add_parser(
        "synthesize-tasks",
        help="Build feasible typed tasks and polish their Chinese query surfaces.",
    )
    synthesize.add_argument("--count", type=int, default=100)
    synthesize.add_argument("--seed", type=int, default=20260807)
    synthesize.add_argument("--env-file", default=str(project_root() / ".env"))
    synthesize.add_argument("--output-dir")
    synthesize.add_argument("--max-api-calls", type=int, default=300)
    synthesize.add_argument(
        "--validation-policy",
        choices=["strict", "minimal_semantic"],
        default="minimal_semantic",
    )
    synthesize.add_argument(
        "--profile",
        choices=[
            "pilot_v2_1",
            "chinatravel_blended_v1",
            "chinatravel_blended_v1_1",
        ],
        default="pilot_v2_1",
    )
    synthesize.set_defaults(handler=_synthesize_tasks)

    repolish = subparsers.add_parser(
        "repolish-tasks",
        help="Concurrently rewrite query surfaces while reusing existing grounded records.",
    )
    repolish.add_argument("--input-dir", required=True)
    repolish.add_argument("--output-dir")
    repolish.add_argument("--env-file", default=str(project_root() / ".env"))
    repolish.add_argument("--max-api-calls", type=int, default=400)
    repolish.add_argument("--llm-concurrency", type=int, default=256)
    repolish.add_argument(
        "--validation-policy",
        choices=["strict", "minimal_semantic"],
        default="minimal_semantic",
    )
    repolish.set_defaults(handler=_repolish_tasks)

    rebuild_sft = subparsers.add_parser(
        "rebuild-sft",
        help="Replay accepted rollout actions into action-only SFT conversations.",
    )
    rebuild_sft.add_argument(
        "--source",
        nargs=2,
        action="append",
        required=True,
        metavar=("TASK_DIR", "ROLLOUT_JSONL"),
        help="Repeatable generated-task directory and matching rollout JSONL pair.",
    )
    rebuild_sft.add_argument("--output-dir", required=True)
    rebuild_sft.add_argument("--repair-surface-semantics", action="store_true")
    rebuild_sft.add_argument(
        "--tool-response-mode",
        choices=TOOL_RESPONSE_MODES,
        default=DEFAULT_TOOL_RESPONSE_MODE,
    )
    rebuild_sft.set_defaults(handler=_rebuild_sft)
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
