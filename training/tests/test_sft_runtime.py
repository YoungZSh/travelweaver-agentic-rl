from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


def _load_runtime_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_verl_sft.py"
    spec = importlib.util.spec_from_file_location("run_verl_sft_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_configure_tf32_enables_both_cuda_paths() -> None:
    module = _load_runtime_module()
    original_precision = torch.get_float32_matmul_precision()
    original_matmul = torch.backends.cuda.matmul.allow_tf32
    original_cudnn = torch.backends.cudnn.allow_tf32
    try:
        report = module.configure_tf32()

        assert report == {
            "float32_matmul_precision": "high",
            "cuda_matmul_allow_tf32": True,
            "cudnn_allow_tf32": True,
        }
    finally:
        torch.set_float32_matmul_precision(original_precision)
        torch.backends.cuda.matmul.allow_tf32 = original_matmul
        torch.backends.cudnn.allow_tf32 = original_cudnn


def test_train_sampler_uses_trainer_seed() -> None:
    module = _load_runtime_module()
    sampler_calls = []
    loader_calls = []

    class FakeSampler:
        def __init__(self, dataset, **kwargs) -> None:
            sampler_calls.append((dataset, kwargs))

    class FakeLoader:
        def __init__(self, **kwargs) -> None:
            loader_calls.append(kwargs)

    runtime = SimpleNamespace(
        DistributedSampler=FakeSampler,
        StatefulDataLoader=FakeLoader,
        get_device_name=lambda: "cuda",
    )
    trainer = SimpleNamespace(
        config=SimpleNamespace(
            trainer=SimpleNamespace(seed=20260810),
            data=SimpleNamespace(num_workers=4),
        ),
        engine=SimpleNamespace(
            get_data_parallel_rank=lambda: 1,
            get_data_parallel_size=lambda: 2,
        ),
        train_dataset="dataset",
        train_batch_size_per_dp=4,
        collate_fn="collator",
    )

    report = module.configure_seeded_train_sampler(trainer, runtime)

    assert report == {
        "shuffle": True,
        "seed": 20260810,
        "data_parallel_rank": 1,
        "data_parallel_size": 2,
    }
    assert sampler_calls == [
        (
            "dataset",
            {
                "shuffle": True,
                "num_replicas": 2,
                "rank": 1,
                "drop_last": True,
                "seed": 20260810,
            },
        )
    ]
    assert loader_calls == [
        {
            "dataset": "dataset",
            "batch_size": 4,
            "sampler": trainer.train_sampler,
            "collate_fn": "collator",
            "num_workers": 4,
            "pin_memory": False,
            "drop_last": True,
            "pin_memory_device": "cuda",
        }
    ]
