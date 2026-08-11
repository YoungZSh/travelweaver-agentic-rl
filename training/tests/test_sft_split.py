from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_split_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "split_qwen_sft.py"
    spec = importlib.util.spec_from_file_location("travelweaver_sft_split_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_split_records_is_exact_deterministic_and_stratified() -> None:
    module = _load_split_module()
    records = [
        module.SplitRecord(
            sample_id=f"a-{index}",
            task_id=f"task-a-{index}",
            semantic_hash=f"hash-a-{index}",
            task_type="easy_like",
            scenario_profile="normal",
            sample_family="efficient_success",
            trip_days=2,
        )
        for index in range(8)
    ]
    records.extend(
        module.SplitRecord(
            sample_id=f"b-{index}",
            task_id=f"task-b-{index}",
            semantic_hash=f"hash-b-{index}",
            task_type="medium_like",
            scenario_profile="price_change",
            sample_family="loop_recovery",
            trip_days=3,
        )
        for index in range(2)
    )

    selected, quotas = module.split_records(records, validation_count=2, seed=20260811)
    repeat, repeat_quotas = module.split_records(records, validation_count=2, seed=20260811)

    assert selected == repeat
    assert quotas == repeat_quotas
    assert len(selected) == 2
    assert sum(quotas.values()) == 2
    assert quotas[("easy_like", "normal", "efficient_success", 2)] == 2
    assert quotas[("medium_like", "price_change", "loop_recovery", 3)] == 0


def test_split_records_rejects_duplicate_samples() -> None:
    module = _load_split_module()
    record = module.SplitRecord(
        sample_id="duplicate",
        task_id="task-1",
        semantic_hash="hash-1",
        task_type="easy_like",
        scenario_profile="normal",
        sample_family="efficient_success",
        trip_days=2,
    )

    try:
        module.split_records([record, record], validation_count=1, seed=7)
    except ValueError as error:
        assert "duplicate sample IDs" in str(error)
    else:
        raise AssertionError("Duplicate sample IDs must be rejected.")
