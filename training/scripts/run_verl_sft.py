"""Enable A800 TF32 acceleration before entering veRL's Hydra SFT launcher."""

from __future__ import annotations

import json
import os
from typing import Any

import torch


def configure_tf32() -> dict[str, bool | str]:
    """Enable TF32 for float32 matmul and cuDNN operations in every trainer rank."""

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }


def configure_seeded_train_sampler(trainer: Any, runtime: Any) -> dict[str, int | bool]:
    """Rebuild veRL's train loader with the configured trainer seed."""

    seed = int(trainer.config.trainer.seed)
    dp_rank = trainer.engine.get_data_parallel_rank()
    dp_size = trainer.engine.get_data_parallel_size()
    trainer.train_sampler = runtime.DistributedSampler(
        trainer.train_dataset,
        shuffle=True,
        num_replicas=dp_size,
        rank=dp_rank,
        drop_last=True,
        seed=seed,
    )
    trainer.train_dataloader = runtime.StatefulDataLoader(
        dataset=trainer.train_dataset,
        batch_size=trainer.train_batch_size_per_dp,
        sampler=trainer.train_sampler,
        collate_fn=trainer.collate_fn,
        num_workers=trainer.config.data.num_workers,
        pin_memory=False,
        drop_last=True,
        pin_memory_device=runtime.get_device_name(),
    )
    return {
        "shuffle": True,
        "seed": seed,
        "data_parallel_rank": dp_rank,
        "data_parallel_size": dp_size,
    }


def install_seeded_sft_trainer(runtime: Any) -> None:
    """Install a local trainer override without modifying the pinned veRL package."""

    base_trainer = runtime.SFTTrainer

    class TravelWeaverSFTTrainer(base_trainer):
        def _build_dataloader(self) -> None:
            super()._build_dataloader()
            sampler_report = configure_seeded_train_sampler(self, runtime)
            if self.rank == 0:
                print(
                    json.dumps(
                        {"event": "sft_train_sampler_configured", **sampler_report},
                        sort_keys=True,
                    ),
                    flush=True,
                )

    runtime.SFTTrainer = TravelWeaverSFTTrainer


def main() -> None:
    """Configure process-wide math settings and delegate CLI parsing to veRL."""

    report = configure_tf32()
    if os.environ.get("RANK", "0") == "0":
        print(json.dumps({"event": "tf32_enabled", **report}, sort_keys=True), flush=True)

    from verl.trainer import sft_trainer as runtime  # noqa: PLC0415

    install_seeded_sft_trainer(runtime)
    runtime.main()


if __name__ == "__main__":
    main()
