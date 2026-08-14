from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_sampler_module():
    path = Path(__file__).resolve().parents[1] / "src" / "travelweaver_grpo_sampler.py"
    spec = importlib.util.spec_from_file_location("travelweaver_grpo_sampler_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_agent_loop_module():
    training_root = Path(__file__).resolve().parents[1]
    repository_root = training_root.parent
    sys.path.insert(0, str(repository_root / "src"))
    path = training_root / "src" / "travelweaver_grpo_agent_loop.py"
    spec = importlib.util.spec_from_file_location("travelweaver_grpo_agent_loop_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_prompt_preparation_module():
    training_root = Path(__file__).resolve().parents[1]
    repository_root = training_root.parent
    sys.path.insert(0, str(repository_root / "src"))
    sys.path.insert(0, str(training_root / "src"))
    path = training_root / "scripts" / "prepare_grpo_prompts.py"
    spec = importlib.util.spec_from_file_location("prepare_grpo_prompts_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_prompt_split_module():
    training_root = Path(__file__).resolve().parents[1]
    path = training_root / "scripts" / "split_grpo_prompts.py"
    name = "split_grpo_prompts_test"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_prompt_source(path: Path, task_id: str) -> None:
    path.mkdir()
    public = {"uid": task_id, "query": f"请为{task_id}规划一次旅行。"}
    oracle = {
        "uid": task_id,
        "scenario": {"scenario_id": f"scenario-{task_id}"},
    }
    (path / "tasks.public.jsonl").write_text(
        json.dumps(public, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (path / "tasks.oracle.jsonl").write_text(
        json.dumps(oracle, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (path / "manifest.json").write_text("{}\n", encoding="utf-8")


def _reward_info(
    reward: float, *, valid: bool = True, hard_pass: bool | None = None
) -> dict[str, float]:
    if hard_pass is None:
        hard_pass = reward == 1.0
    return {
        "travelweaver_reward": reward,
        "reward_valid": float(valid),
        "all_hard_pass": float(hard_pass),
        "artifact_score": 0.75,
        "validity_score": 0.5,
        "goal_score": 0.25,
        "trajectory_length_exceeded": 0.0,
        "sequence_tokens": 1024.0,
    }


def test_sampling_metrics_keep_initial_quality_separate_from_adaptive_refills() -> None:
    module = _load_sampler_module()
    mastered = [_reward_info(1.0) for _ in range(8)]
    frontier = [
        *[_reward_info(1.0) for _ in range(4)],
        *[_reward_info(0.0) for _ in range(4)],
    ]
    refill_unsolved = [_reward_info(0.0) for _ in range(8)]

    metrics = module.build_sampling_metrics(
        initial_uids={"initial-mastered", "initial-frontier"},
        group_records={
            "initial-mastered": ("zero_variance", mastered),
            "initial-frontier": ("usable", frontier),
            "refill-unsolved": ("zero_variance", refill_unsolved),
        },
        refill_groups=1,
        selected_uids={"initial-frontier", "another-frontier"},
        group_size=8,
    )

    assert metrics["training/policy_quality/raw_initial/success_avg_at_8"] == 0.75
    assert metrics["training/policy_quality/raw_initial/pass_at_8"] == 1.0
    assert metrics["training/policy_quality/raw_initial/unsolved_at_8"] == 0.0
    assert metrics["training/policy_quality/raw_initial/frontier_at_8"] == 0.5
    assert metrics["training/policy_quality/raw_initial/mastered_at_8"] == 0.5
    assert metrics["training/sampler/initial_groups"] == 2
    assert metrics["training/sampler/refill_groups"] == 1
    assert metrics["training/sampler/selected_groups"] == 2
    assert metrics["training/sampler/filtered_unsolved_groups"] == 1
    assert metrics["training/sampler/filtered_mastered_groups"] == 1
    assert metrics["training/sampler/refill_overhead_ratio"] == 0.5


def test_group_filter_rejects_every_constant_reward_level() -> None:
    module = _load_sampler_module()
    classify = module.classify_reward_group

    for reward in (0.0, 1.0, -1.0, -0.375):
        assert classify(
            [_reward_info(reward) for _ in range(8)], group_size=8, tolerance=1e-8
        ) == ("zero_variance", reward)
    assert classify(
        [_reward_info(0.0) for _ in range(7)] + [_reward_info(0.25)],
        group_size=8,
        tolerance=1e-8,
    ) == ("usable", None)
    assert classify(
        [_reward_info(0.0) for _ in range(7)] + [_reward_info(0.0, valid=False)],
        group_size=8,
        tolerance=1e-8,
    ) == ("invalid", None)


def test_group_filter_rejects_whole_group_when_one_trajectory_reaches_length_cap() -> None:
    module = _load_sampler_module()
    infos = [_reward_info(0.0) for _ in range(8)]
    infos[3]["trajectory_length_exceeded"] = 1.0
    infos[3]["sequence_tokens"] = 32768.0

    assert module.classify_reward_group(
        infos, group_size=8, tolerance=1e-8
    ) == ("length_exceeded", None)


def test_length_exceeded_group_does_not_change_no_signal_streak() -> None:
    module = _load_sampler_module()
    sampler = module.TravelWeaverReplayBuffer.__new__(module.TravelWeaverReplayBuffer)
    sampler.consecutive_no_signal = 4
    sampler.no_signal_history = [{"prompt_uid": "prior"}]
    sampler._save_state = lambda: None

    sampler._record_signal_result("overlong", "length_exceeded", None, [], 5)

    assert sampler.consecutive_no_signal == 4
    assert sampler.no_signal_history == [{"prompt_uid": "prior"}]


def test_trajectory_response_budget_applies_strict_total_sequence_cap() -> None:
    module = _load_agent_loop_module()

    assert module.trajectory_response_budget(
        initial_prompt_tokens=922,
        configured_response_tokens=32768,
        trajectory_max_tokens=32768,
    ) == 31846
    assert module.trajectory_response_budget(
        initial_prompt_tokens=922,
        configured_response_tokens=4096,
        trajectory_max_tokens=32768,
    ) == 4096


def test_ten_consecutive_no_signal_groups_stop_and_usable_group_resets() -> None:
    module = _load_sampler_module()
    sampler = module.TravelWeaverReplayBuffer.__new__(module.TravelWeaverReplayBuffer)
    sampler.max_consecutive_no_signal = 10
    sampler.consecutive_no_signal = 0
    sampler.no_signal_history = []
    sampler.group_size = 8
    sampler.zero_variance_tolerance = 1e-8
    sampler._save_state = lambda: None
    infos = [_reward_info(-0.5) for _ in range(8)]

    for index in range(9):
        sampler._record_signal_result(
            f"dry-{index}", "zero_variance", -0.5, infos, index
        )
    sampler._record_signal_result("usable", "usable", None, infos, 9)
    assert sampler.consecutive_no_signal == 0

    for index in range(9):
        sampler._record_signal_result(
            f"dry-again-{index}", "zero_variance", -0.5, infos, index + 10
        )
    with pytest.raises(module.ConsecutiveNoSignalStop) as caught:
        sampler._record_signal_result("stop", "zero_variance", -0.5, infos, 19)

    report = caught.value.report
    assert report["reason"] == "consecutive_zero_variance_groups"
    assert report["consecutive_no_signal"] == 10
    assert report["group_size"] == 8
    assert report["groups"][-1]["dimension_means"] == {
        "artifact_score": 0.75,
        "validity_score": 0.5,
        "goal_score": 0.25,
    }


def test_qwen_parser_view_keeps_parameter_types_while_prompt_keeps_nested_schema() -> None:
    module = _load_agent_loop_module()
    from travelweaver.env import TravelWeaverEnv

    submit = next(
        schema
        for schema in TravelWeaverEnv.tool_schemas()
        if schema["function"]["name"] == "submit_plan"
    )
    wrapped = module._RawToolSchema(submit)

    assert wrapped.function.parameters.properties["plan"].type == "object"
    dumped = wrapped.model_dump()
    itinerary = dumped["function"]["parameters"]["properties"]["plan"]["properties"][
        "itinerary"
    ]
    assert itinerary["type"] == "array"
    assert "items" in itinerary


def test_prompt_preparation_combines_multiple_source_batches(tmp_path: Path) -> None:
    module = _load_prompt_preparation_module()
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_prompt_source(first, "task-1")
    _write_prompt_source(second, "task-2")
    output = tmp_path / "train.parquet"

    report = module.prepare([first, second], output)

    assert report["format_version"] == "travelweaver-grpo-prompts-v2"
    assert report["row_count"] == 2
    assert [source["row_count"] for source in report["sources"]] == [1, 1]
    assert report["contains_witness"] is False
    assert report["contains_reward_labels"] is False
    assert output.is_file()


def test_prompt_preparation_rejects_duplicate_ids_across_batches(tmp_path: Path) -> None:
    module = _load_prompt_preparation_module()
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_prompt_source(first, "duplicate")
    _write_prompt_source(second, "duplicate")

    with pytest.raises(ValueError, match="Duplicate task_id"):
        module.prepare([first, second], tmp_path / "train.parquet")


def test_grpo_prompt_split_is_exact_deterministic_and_stratified_by_task_type() -> None:
    module = _load_prompt_split_module()
    records = [
        module.SplitRecord(
            task_id=f"{task_type}-{index}",
            task_type=task_type,
            scenario_profile=f"scenario-{index % 2}",
            constraint_count=2 + index % 3,
            trip_days=3 + index % 4,
        )
        for task_type in ("easy", "medium")
        for index in range(10)
    ]

    first = module.split_records(records, validation_count=2, seed=20260813)
    second = module.split_records(records, validation_count=2, seed=20260813)

    assert first == second
    assert len(first) == 2
    selected_types = [record.task_type for record in records if record.task_id in first]
    assert selected_types.count("easy") == 1
    assert selected_types.count("medium") == 1


def test_grpo_prompt_split_preserves_rare_scenario_quota() -> None:
    module = _load_prompt_split_module()
    records = [
        module.SplitRecord(
            task_id=f"normal-{index}",
            task_type="medium",
            scenario_profile="normal",
            constraint_count=2 + index % 3,
            trip_days=2 + index % 4,
        )
        for index in range(90)
    ] + [
        module.SplitRecord(
            task_id=f"price-change-{index}",
            task_type="medium",
            scenario_profile="price_change",
            constraint_count=3,
            trip_days=3,
        )
        for index in range(10)
    ]

    selected = module.split_records(records, validation_count=10, seed=20260813)

    selected_scenarios = [
        record.scenario_profile for record in records if record.task_id in selected
    ]
    assert selected_scenarios.count("normal") == 9
    assert selected_scenarios.count("price_change") == 1


def test_four_gpu_launcher_preserves_two_gpu_training_semantics() -> None:
    training_root = Path(__file__).resolve().parents[1]
    wrapper = (
        training_root / "scripts" / "run_qwen3_5_4b_travelweaver_grpo_4gpu.sh"
    ).read_text(encoding="utf-8")
    expected_exports = {
        'export NCCL_IB_DISABLE="1"',
        'export NCCL_SOCKET_IFNAME="lo"',
        'export CUDA_VISIBLE_DEVICES="0,1,2,3"',
        'export NUM_GPUS="4"',
        'export GROUP_SIZE="8"',
        'export TRAIN_BATCH_SIZE="8"',
        'export VAL_BATCH_SIZE="16"',
        'export PPO_MINI_BATCH_SIZE="4"',
        'export PPO_EPOCHS="1"',
        'export TOTAL_STEPS="112"',
        'export SP_SIZE="2"',
        'export ROLLOUT_TP_SIZE="2"',
        'export ROLLOUT_DP_SIZE="1"',
        'export ROLLOUT_MAX_NUM_SEQS="8"',
        'export AGENT_NUM_WORKERS="16"',
        'export DATALOADER_NUM_WORKERS="4"',
        'export MAX_TOKEN_LEN_PER_GPU="24576"',
        'export GPU_MEMORY_UTILIZATION="0.75"',
    }
    assert "omini_sft_gpu_reservation_gpu" in wrapper
    assert "Data_gen_Mass_V3/gpu_occupy.py" in wrapper
    assert "trap restore_holders EXIT HUP INT TERM" in wrapper
    preflight_offset = wrapper.index('SKIP_PREFLIGHT=0 DRY_RUN=1 "${BASE_LAUNCHER}"')
    stop_offset = wrapper.rindex("\nstop_holders\n")
    assert preflight_offset < stop_offset

    assert all(export in wrapper for export in expected_exports)

    base_launcher = (
        training_root / "scripts" / "run_qwen3_5_4b_travelweaver_grpo.sh"
    ).read_text(encoding="utf-8")
    assert "num_replicas = world_size // rollout_world_size" in base_launcher
    assert "actor_rollout_ref.rollout.data_parallel_size=${ROLLOUT_DP_SIZE}" in base_launcher
    assert 'GPU_PROCESS_REPORT="$(selected_gpu_processes)"' in base_launcher
    assert "TRAIN_BATCH_SIZE / PPO_MINI_BATCH_SIZE * PPO_EPOCHS != 2" in base_launcher
    assert 'VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-1}"' in base_launcher
    assert 'data.val_batch_size=${VAL_BATCH_SIZE}' in base_launcher
    assert "trainer.rollout_data_dir=${ROLLOUT_DATA_DIR}" in base_launcher
    assert "trainer.validation_data_dir=${VALIDATION_DATA_DIR}" in base_launcher
    assert 'export TRAVELWEAVER_ROLLOUT_TRACE_DIR="${ALL_ROLLOUT_TRACE_DIR}"' in base_launcher
    assert "runtime_env.env_vars.NCCL_IB_DISABLE" in base_launcher
    assert "runtime_env.env_vars.NCCL_SOCKET_IFNAME" in base_launcher


def test_agent_loop_writes_complete_rollout_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_agent_loop_module()
    monkeypatch.setenv("TRAVELWEAVER_ROLLOUT_TRACE_DIR", str(tmp_path))

    class FakeTokenizer:
        def decode(self, token_ids, *, skip_special_tokens: bool) -> str:
            assert skip_special_tokens is False
            return "|".join(str(token_id) for token_id in token_ids)

    loop = module.TravelWeaverAgentLoop.__new__(module.TravelWeaverAgentLoop)
    loop.tokenizer = FakeTokenizer()
    output = SimpleNamespace(
        prompt_ids=[1, 2],
        response_ids=[3, 4, 5],
        response_mask=[1, 1, 0],
        num_turns=2,
        reward_score=0.75,
        extra_fields={
            "reward_extra_info": {"travelweaver_reward": 0.75},
            "travelweaver_audit": {"steps": [{"action": {"tool": "search"}}]},
        },
    )

    loop._write_rollout_trace(
        output,
        task_id="task-1",
        task_dir="/tasks",
        kwargs={"index": 7, "scenario_id": "scenario-1"},
    )

    trace_files = list(tmp_path.glob("*.json"))
    assert len(trace_files) == 1
    trace = json.loads(trace_files[0].read_text(encoding="utf-8"))
    assert trace["format_version"] == "travelweaver-grpo-rollout-trace-v1"
    assert trace["task_id"] == "task-1"
    assert trace["prompt_text"] == "1|2"
    assert trace["response_text"] == "3|4|5"
    assert trace["response_mask_rle"] == [[1, 2], [0, 1]]
    assert trace["reward_score"] == 0.75
    assert trace["travelweaver_audit"]["steps"][0]["action"]["tool"] == "search"
