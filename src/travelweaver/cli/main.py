"""Command-line utilities for data preparation and environment smoke tests."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..data.bootstrap import install_database, validate_database
from ..data.tasks import JsonlTaskStore, import_benchmark_tasks, import_task_split
from ..env import ChinaTravelBackend, TravelWeaverEnv
from ..errors import TravelWeaverError
from ..evaluation import audit_synthesis_directory
from ..llm import (
    DEFAULT_DEEPSEEK_CONCURRENCY,
    DeepSeekConfig,
    OpenAICompatibleConfig,
)
from ..paths import project_root
from ..rollout import (
    DEFAULT_MAX_API_TURNS,
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
from ..sft import (
    DEFAULT_SFT_SUPERVISION_MODE,
    SFT_SUPERVISION_MODES,
    ProgrammaticBuildConfig,
    RationalePolishConfig,
    RationaleRevalidationConfig,
    SFTRebuildConfig,
    SFTSource,
    audit_programmatic_batch,
    build_programmatic_trajectories,
    compare_rollout_batches,
    polish_programmatic_rationales,
    rebuild_sft_dataset,
    revalidate_programmatic_rationales,
)
from ..synthesis import (
    RepolishConfig,
    SurfaceRepolishPipeline,
    SynthesisConfig,
    SynthesisPipeline,
)
from ..synthesis.catalog import BLENDED_V1_1_PROFILE, SUPPORTED_PROFILES


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
        open_check = env.step(
            {
                "tool": "check_place_open",
                "arguments": {"place_id": items[0]["place_id"], "at_time": "12:00"},
            }
        )
        _print({"event": "step", "result": open_check.to_dict()})
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
            limit=args.limit,
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
    llm_config = (
        DeepSeekConfig(api_key="offline-canonical", model="deterministic-canonical")
        if args.canonical_only
        else DeepSeekConfig.from_env(args.env_file)
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else project_root() / "data" / "generated" / "chinatravel-blended"
    )
    report = SynthesisPipeline(
        SynthesisConfig(
            output_dir=output_dir,
            count=args.count,
            seed=args.seed,
            max_api_calls=args.max_api_calls,
            profile=args.profile,
            validation_policy=args.validation_policy,
            llm_concurrency=args.llm_concurrency,
            witness_concurrency=args.witness_concurrency,
            exclude_task_dirs=tuple(Path(value) for value in args.exclude_task_dir),
            canonical_only=args.canonical_only,
        ),
        llm_config,
        progress=lambda payload: print(
            json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True
        ),
    ).run()
    _print(report.to_dict())
    return 0


def _repolish_tasks(args: argparse.Namespace) -> int:
    llm_config = (
        DeepSeekConfig(api_key="offline-canonical", model="deterministic-canonical")
        if args.canonical_only
        else DeepSeekConfig.from_env(args.env_file)
    )
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
            canonical_only=args.canonical_only,
            allow_partial_input=args.allow_partial_input,
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
            supervision_mode=args.supervision_mode,
            require_official_commonsense=args.require_official_commonsense,
        )
    )
    _print(report.to_dict())
    return 0


def _programmatic_sft(args: argparse.Namespace) -> int:
    output_path = Path(args.output)
    audit_path = (
        Path(args.audit)
        if args.audit
        else output_path.with_name(f"{output_path.stem}-audit.jsonl")
    )
    report = build_programmatic_trajectories(
        ProgrammaticBuildConfig(
            task_dir=Path(args.input_dir),
            output_path=output_path,
            audit_path=audit_path,
            seed=args.seed,
            concurrency=args.concurrency,
            work_dir=Path(args.work_dir) if args.work_dir else None,
        ),
        progress=lambda payload: print(
            json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True
        ),
    )
    _print(report)
    return 0


def _polish_programmatic_react(args: argparse.Namespace) -> int:
    llm_config = DeepSeekConfig.from_env(args.env_file)
    output_path = Path(args.output)
    output_audit_path = (
        Path(args.audit)
        if args.audit
        else output_path.with_name(f"{output_path.stem}-audit.jsonl")
    )
    work_dir = (
        Path(args.work_dir)
        if args.work_dir
        else output_path.with_name(f".{output_path.stem}-work")
    )
    report = polish_programmatic_rationales(
        RationalePolishConfig(
            input_path=Path(args.input),
            input_audit_path=Path(args.input_audit),
            output_path=output_path,
            output_audit_path=output_audit_path,
            work_dir=work_dir,
            llm_concurrency=args.llm_concurrency,
            max_api_calls=args.max_api_calls,
            task_ids=tuple(args.task_id),
        ),
        llm_config,
        progress=lambda payload: print(
            json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True
        ),
    )
    _print(report.to_dict())
    return 0


def _revalidate_programmatic_react(args: argparse.Namespace) -> int:
    report = revalidate_programmatic_rationales(
        RationaleRevalidationConfig(
            input_path=Path(args.input),
            input_audit_path=Path(args.input_audit),
            source_audit_path=Path(args.source_audit),
            output_path=Path(args.output),
            output_audit_path=Path(args.audit),
        ),
        DeepSeekConfig.from_env(args.env_file),
    )
    _print(report.to_dict())
    return 0


def _audit_official(args: argparse.Namespace) -> int:
    report = audit_synthesis_directory(
        args.input_dir,
        output_path=args.output,
        exports_path=args.exports,
        concurrency=args.concurrency,
    )
    _print(report)
    return 0 if report["commonsense_passes"] == report["count"] else 2


def _audit_programmatic(args: argparse.Namespace) -> int:
    report = audit_programmatic_batch(
        args.input_dir,
        args.trajectory,
        args.trajectory_audit,
        sft_manifest_path=args.sft_manifest,
        output_path=args.output,
        require_rationale_polish=args.require_rationale_polish,
    )
    _print(report)
    return 0 if report["accepted"] else 2


def _compare_rollouts(args: argparse.Namespace) -> int:
    report = compare_rollout_batches(
        args.programmatic,
        args.model,
        output_path=args.output,
    )
    _print(report)
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
    rollout_api.add_argument(
        "--max-api-turns", type=int, default=DEFAULT_MAX_API_TURNS
    )
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
    rollout_generated.add_argument(
        "--limit",
        type=int,
        help="Deterministically select a fixed, resumable cohort of at most this many tasks.",
    )
    rollout_generated.add_argument("--seed", type=int, default=20260808)
    rollout_generated.add_argument("--env-file", default=str(project_root() / ".env"))
    rollout_generated.add_argument(
        "--max-api-turns", type=int, default=DEFAULT_MAX_API_TURNS
    )
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
    rollout_benchmark.add_argument(
        "--max-api-turns", type=int, default=DEFAULT_MAX_API_TURNS
    )
    rollout_benchmark.add_argument(
        "--tool-response-mode",
        choices=TOOL_RESPONSE_MODES,
        default=DEFAULT_TOOL_RESPONSE_MODE,
    )
    rollout_benchmark.add_argument(
        "--concurrency", type=int, default=DEFAULT_ROLLOUT_CONCURRENCY
    )
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
        "--canonical-only",
        action="store_true",
        help="Build and validate canonical surfaces without making paid API calls.",
    )
    synthesize.add_argument(
        "--llm-concurrency",
        type=int,
        default=DEFAULT_DEEPSEEK_CONCURRENCY,
    )
    synthesize.add_argument(
        "--witness-concurrency",
        type=int,
        default=min(32, os.cpu_count() or 1),
        help="Process concurrency for deterministic witness construction.",
    )
    synthesize.add_argument(
        "--exclude-task-dir",
        action="append",
        default=[],
        help=(
            "Repeatable completed synthesis directory whose task ids, normalized Questions, "
            "Blueprints, and surfaces must not be reused."
        ),
    )
    synthesize.add_argument(
        "--validation-policy",
        choices=["strict", "minimal_semantic"],
        default="minimal_semantic",
    )
    synthesize.add_argument(
        "--profile",
        choices=SUPPORTED_PROFILES,
        default=BLENDED_V1_1_PROFILE,
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
    repolish.add_argument(
        "--canonical-only",
        action="store_true",
        help="Re-render validated natural canonical surfaces without an external API call.",
    )
    repolish.add_argument(
        "--llm-concurrency",
        type=int,
        default=DEFAULT_DEEPSEEK_CONCURRENCY,
    )
    repolish.add_argument(
        "--validation-policy",
        choices=["strict", "minimal_semantic"],
        default="minimal_semantic",
    )
    repolish.add_argument(
        "--allow-partial-input",
        action="store_true",
        help="Explicitly repolish the records currently present in an incomplete source batch.",
    )
    repolish.set_defaults(handler=_repolish_tasks)

    rebuild_sft = subparsers.add_parser(
        "rebuild-sft",
        help="Replay accepted rollouts into action-only, clean ReAct, or recovery ReAct SFT.",
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
        "--require-official-commonsense",
        action="store_true",
        help="Only retain tasks that pass the complete official-audit.jsonl gate.",
    )
    rebuild_sft.add_argument(
        "--supervision-mode",
        choices=SFT_SUPERVISION_MODES,
        default=DEFAULT_SFT_SUPERVISION_MODE,
        help=(
            "ReAct preserves visible text; react_recovery keeps invalid turns as masked context."
        ),
    )
    rebuild_sft.add_argument(
        "--tool-response-mode",
        choices=TOOL_RESPONSE_MODES,
        default=DEFAULT_TOOL_RESPONSE_MODE,
    )
    rebuild_sft.set_defaults(handler=_rebuild_sft)

    programmatic_sft = subparsers.add_parser(
        "generate-programmatic-sft",
        help=(
            "Build deterministic evidence-grounded tool-call graph trajectories from "
            "synthesis witnesses."
        ),
    )
    programmatic_sft.add_argument("--input-dir", required=True)
    programmatic_sft.add_argument("--output", required=True)
    programmatic_sft.add_argument("--audit")
    programmatic_sft.add_argument(
        "--work-dir",
        help="Per-task recovery directory; defaults next to --output.",
    )
    programmatic_sft.add_argument("--seed", type=int, default=20260821)
    programmatic_sft.add_argument(
        "--concurrency", type=int, default=min(32, os.cpu_count() or 1)
    )
    programmatic_sft.add_argument(
        "--allow-undercovered-tools",
        action="store_true",
        help=(
            "Deprecated compatibility flag; tool coverage below 10%% is always reported "
            "as a warning and does not block output."
        ),
    )
    programmatic_sft.set_defaults(handler=_programmatic_sft)

    polish_programmatic_react = subparsers.add_parser(
        "polish-programmatic-react",
        help="Polish template-first visible rationales for programmatic ReAct trajectories.",
    )
    polish_programmatic_react.add_argument("--input", required=True)
    polish_programmatic_react.add_argument("--input-audit", required=True)
    polish_programmatic_react.add_argument("--output", required=True)
    polish_programmatic_react.add_argument("--audit")
    polish_programmatic_react.add_argument("--work-dir")
    polish_programmatic_react.add_argument(
        "--env-file", default=str(project_root() / ".env")
    )
    polish_programmatic_react.add_argument("--max-api-calls", type=int, default=500)
    polish_programmatic_react.add_argument(
        "--task-id",
        action="append",
        default=[],
        help="Repeatable task id filter for a fixed pilot cohort.",
    )
    polish_programmatic_react.add_argument(
        "--llm-concurrency", type=int, default=DEFAULT_DEEPSEEK_CONCURRENCY
    )
    polish_programmatic_react.set_defaults(handler=_polish_programmatic_react)

    revalidate_programmatic_react = subparsers.add_parser(
        "revalidate-programmatic-react",
        help="Revalidate saved ReAct rationale responses without a new API request.",
    )
    revalidate_programmatic_react.add_argument("--input", required=True)
    revalidate_programmatic_react.add_argument("--input-audit", required=True)
    revalidate_programmatic_react.add_argument("--source-audit", required=True)
    revalidate_programmatic_react.add_argument("--output", required=True)
    revalidate_programmatic_react.add_argument("--audit", required=True)
    revalidate_programmatic_react.add_argument(
        "--env-file", default=str(project_root() / ".env")
    )
    revalidate_programmatic_react.set_defaults(handler=_revalidate_programmatic_react)

    official_audit = subparsers.add_parser(
        "audit-chinatravel-official",
        help="Export synthesis witnesses and run the pinned official schema/commonsense checks.",
    )
    official_audit.add_argument("--input-dir", required=True)
    official_audit.add_argument("--output")
    official_audit.add_argument("--exports")
    official_audit.add_argument(
        "--concurrency", type=int, default=min(32, os.cpu_count() or 1)
    )
    official_audit.set_defaults(handler=_audit_official)

    batch_audit = subparsers.add_parser(
        "audit-programmatic-batch",
        help="Aggregate Question, policy, Reward, official, and token acceptance metrics.",
    )
    batch_audit.add_argument("--input-dir", required=True)
    batch_audit.add_argument("--trajectory", required=True)
    batch_audit.add_argument("--trajectory-audit", required=True)
    batch_audit.add_argument("--sft-manifest")
    batch_audit.add_argument("--output")
    batch_audit.add_argument("--require-rationale-polish", action="store_true")
    batch_audit.set_defaults(handler=_audit_programmatic)

    compare_rollouts = subparsers.add_parser(
        "compare-rollout-batches",
        help="Compare tool use and outcomes for two rollout strategies on identical tasks.",
    )
    compare_rollouts.add_argument("--programmatic", required=True)
    compare_rollouts.add_argument("--model", required=True)
    compare_rollouts.add_argument("--output")
    compare_rollouts.set_defaults(handler=_compare_rollouts)
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
