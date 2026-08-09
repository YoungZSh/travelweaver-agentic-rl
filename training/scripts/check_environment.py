"""Check the isolated veRL training environment and its CUDA runtime."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

import torch


def package_version(name: str) -> str:
    """Return an installed package version without importing the package."""
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def main() -> None:
    """Print the pinned stack and fail when CUDA is unavailable."""
    for package in (
        "verl",
        "vllm",
        "torch",
        "ray",
        "transformers",
        "datasets",
        "pyarrow",
        "flash-attn",
        "flash-linear-attention",
        "fla-core",
        "causal-conv1d",
        "flashinfer-python",
        "numpy",
    ):
        print(f"{package}: {package_version(package)}")

    import causal_conv1d_cuda  # noqa: PLC0415
    import fla  # noqa: PLC0415
    import flash_attn  # noqa: PLC0415
    import flash_attn_2_cuda  # noqa: PLC0415
    from transformers import AutoConfig, Qwen3_5ForConditionalGeneration  # noqa: PLC0415
    from verl.trainer.main_ppo import main as ppo_main  # noqa: PLC0415
    from verl.workers.engine_workers import ActorRolloutRefWorker  # noqa: PLC0415
    from verl.workers.rollout.vllm_rollout.vllm_rollout import (  # noqa: PLC0415
        ServerAdapter,
    )
    from vllm.model_executor.models import ModelRegistry  # noqa: PLC0415

    print(f"FlashAttention module: {flash_attn.__file__}")
    print(f"FlashAttention CUDA extension: {flash_attn_2_cuda.__file__}")
    print(f"Flash Linear Attention module: {fla.__file__}")
    print(f"causal-conv1d CUDA extension: {causal_conv1d_cuda.__file__}")
    print(f"Qwen3.5 model class: {Qwen3_5ForConditionalGeneration.__name__}")

    config = AutoConfig.from_pretrained("ckpts/Qwen3.5-4B", local_files_only=True)
    print(f"Local checkpoint config: {config.__class__.__name__}")
    architectures = config.architectures or []
    supported_architectures = set(ModelRegistry.get_supported_archs())
    unsupported_architectures = set(architectures) - supported_architectures
    if unsupported_architectures:
        message = f"vLLM does not support checkpoint architectures: {unsupported_architectures}"
        raise SystemExit(message)
    print(f"vLLM checkpoint architectures: {architectures}")
    print(f"PPO entry point: {ppo_main.__module__}.{ppo_main.__name__}")
    print(f"FSDP worker: {ActorRolloutRefWorker.__module__}.{ActorRolloutRefWorker.__name__}")
    print(f"vLLM rollout: {ServerAdapter.__module__}.{ServerAdapter.__name__}")

    print(f"torch CUDA runtime: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in the training environment")

    print(f"CUDA device count: {torch.cuda.device_count()}")
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        print(
            f"CUDA device {index}: {properties.name}, "
            f"compute capability {properties.major}.{properties.minor}"
        )


if __name__ == "__main__":
    main()
