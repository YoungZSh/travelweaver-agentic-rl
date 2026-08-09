"""Enable A800 TF32 acceleration before entering veRL's Hydra SFT launcher."""

from __future__ import annotations

import json
import os

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


def main() -> None:
    """Configure process-wide math settings and delegate CLI parsing to veRL."""

    report = configure_tf32()
    if os.environ.get("RANK", "0") == "0":
        print(json.dumps({"event": "tf32_enabled", **report}, sort_keys=True), flush=True)

    from verl.trainer.sft_trainer import main as verl_sft_main  # noqa: PLC0415

    verl_sft_main()


if __name__ == "__main__":
    main()
