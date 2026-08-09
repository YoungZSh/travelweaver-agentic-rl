from __future__ import annotations

import importlib.util
from pathlib import Path

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
