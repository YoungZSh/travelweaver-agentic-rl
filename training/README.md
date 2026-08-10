# TravelWeaver training

This directory is the isolated Linux/CUDA training project for TravelWeaver. Keep SFT,
GRPO, veRL configuration, launchers, and training-only utilities here; the repository root
environment remains the lightweight deterministic environment and rollout stack.

## Baseline

- Python: `3.10.19`
- veRL: `0.9.0.dev` (`main` commit `4a2cba76f7f605d2b9f56e640faaeaa71c2c7f71`)
- vLLM: `0.19.1`
- PyTorch: `2.10.0`
- Transformers: `5.5.3`
- FlashAttention: `2.8.3`
- Flash Linear Attention: `0.5.1`
- FlashInfer: `0.6.6`
- CUDA runtime wheels: `12.8`
- Training backend: FSDP/FSDP2
- Rollout backend: vLLM

FLA prints a recommendation for Python 3.11 at import time. Python 3.10 remains supported by
veRL, vLLM, Transformers, and the locked FLA release and is intentionally retained here.

The dependency list follows the official veRL custom-environment installation procedure with
`USE_MEGATRON=0` and `USE_SGLANG=0`. The official instructions explicitly allow following the
installer steps manually when the stock environment is incompatible. veRL is locked to an exact
official `main` commit because `0.9.0` is still under development and has no release tag.

The stock installer currently selects vLLM 0.24.0 and PyTorch 2.11.0/CUDA 13. This host has an
NVIDIA 570 driver, so the environment instead pins the newest mutually compatible CUDA 12.8
combination: vLLM 0.19.1, PyTorch 2.10.0, and Transformers 5.5.3. Do not update these packages
independently.

The host provides glibc 2.31, so UV builds FlashAttention 2.8.3 locally with the matching
PyTorch 2.10 build dependency, CUDA 12.6 toolkit, and `TORCH_CUDA_ARCH_LIST=8.0` for the A800
GPUs. Qwen3.5 also uses Gated DeltaNet linear-attention layers, so Flash Linear Attention and
causal-conv1d are installed in addition to regular FlashAttention. Both CUDA extensions are forced
to build locally because upstream binary wheels require a newer glibc than this host. UV caches the
resulting wheels.

## Setup and verification

Run commands from the repository root:

```bash
uv sync --project training --dev
uv run --project training python training/scripts/check_environment.py
```

The training virtual environment is created at `training/.venv` and is ignored by Git.
The lock file is committed so all workers resolve the same Python package versions.

## Qwen3.5 vLLM smoke inference

The offline smoke script validates the local checkpoint and runs text-only tensor-parallel
generation on GPU 0/1:

```bash
CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  uv run --project training python training/scripts/run_qwen3_5_vllm.py
```

Use `--preflight-only` to validate the tokenizer, chat template, model config, and vLLM model
registry without touching a GPU. Use `--disable-thinking --max-tokens 128` for a short direct-answer
smoke test. The generated JSON is written under `training/outputs/`.

For the resumable, non-thinking Qwen3.5-4B evaluation over all 654 pinned ChinaTravel benchmark
tasks, launch the persistent runner in tmux:

```bash
tmux new-window -d -t train -n tw-qwen-eval \
  "exec bash training/scripts/run_qwen3_5_chinatravel_eval.sh"
```

The runner uses the checkpoint's 262,144-token context limit, writes each completed trajectory and
model-side error immediately, and resumes by skipping previously attempted task IDs. On exit it
stops vLLM and replaces itself with the GPU 0/1 hold process.

## Layout

```text
training/
├── configs/       # veRL, SFT, GRPO, model, and distributed-runtime configuration
├── scripts/       # launchers, preprocessing, and environment checks
├── src/           # reusable training-only Python modules
├── tests/         # offline tests for training adapters and preprocessing
├── pyproject.toml # isolated training dependencies
└── uv.lock        # reproducible dependency resolution
```

Generated training artifacts belong under `training/outputs/`; this path is ignored by Git. Each
SFT experiment keeps its logs, resumable checkpoints, and final model together under
`training/outputs/<project>/<experiment>/`. Exchange data with the root environment through
versioned task and trajectory files rather than importing GPU dependencies into the root
environment.

SGLang, Apex, TransformerEngine, and Megatron-LM are intentionally excluded. This project
uses only FSDP for training and vLLM for rollout generation.

## Qwen3.5 action-only and ReAct SFT data

The root environment produces replay-verified `travelweaver-sft-v4` JSONL with an explicit
`supervision_mode`. Action-only samples supervise tool calls, while clean ReAct samples also
supervise visible `assistant.content` from thinking-disabled rollouts. Supplier-private
`reasoning_content` is rejected in both modes. The leading user message is the original
natural-language task query, while machine-generated tool observations remain versioned JSON. Intermediate
tool messages use the versioned `delta` response by default, while full environment snapshots stay
in the source trajectory for replay and audit. Convert the neutral data to
veRL Parquet without loading model weights:

```bash
uv run --project training python training/scripts/prepare_qwen_sft.py \
  --input data/sft/chinatravel-qwen3.5-4b-action-633-sft-v2-natural/neutral.jsonl \
  --output data/sft/chinatravel-qwen3.5-4b-action-633-sft-v2-natural/all.parquet \
  --model ckpts/Qwen3.5-4B
```

The Parquet stores `messages_json` and `tools_json` strings so heterogeneous function arguments
are not widened into nullable Arrow structs. Configure veRL with
`training/configs/qwen3_5_4b_sft_data.yaml` and the custom dataset class in
`training/src/travelweaver_sft_dataset.py`. It decodes those columns and then delegates tokenization,
masking, and full-conversation consistency checks to veRL's `MultiTurnSFTDataset`.
The adapter remains compatible with existing v2 action-only and v3 action-only/clean ReAct
artifacts. Newly rebuilt datasets use v4 so supervision semantics cannot be inferred from message
text.

Recovery ReAct is specified in
[`docs/react-sft-recovery-v1.md`](../docs/react-sft-recovery-v1.md). V4 retains invalid assistant
turns and tool errors as causal context, supplies an explicit per-assistant-turn loss mask, and
supervises only subsequent reflection and valid actions. The adapter reads the separate
`assistant_loss_mask_json` Parquet column; it does not infer masks from error text or add private
fields to messages passed through Qwen's official chat template.

The default two-GPU full-parameter run is restricted to GPU 0/1. It uses FSDP2, two-way Ulysses
sequence parallelism, dynamic token batches, and a 32,768-token sequence limit. The per-GPU token
budget is 16,384, so the current 22,079-token maximum sample fits across SP=2 without packing an
unnecessarily large micro-batch.

This 32,768-token launcher default is lower than Qwen3.5-4B's 262,144-token model limit. The
79-sample clean/recovery ReAct pilot reaches 46,843 tokens (median 28,216 and p90 40,990). Before
training ReAct data, either raise the training limit and token budgets or isolate over-limit
samples; do not truncate early evidence or error-recovery context.

The launcher preflight checks the Parquet manifest and hash, model family, maximum sequence length,
parallelism divisibility, fused AdamW support, veRL's fused linear cross-entropy kernel, and
installed FLA support:

```bash
bash training/scripts/run_qwen3_5_4b_multiturn_sft.sh --dry-run
bash training/scripts/run_qwen3_5_4b_multiturn_sft.sh
```

The launcher defaults to three epochs, a global batch size of 16, learning rate `1e-5`, and
automatic checkpoint resume. It enables PyTorch fused AdamW, TF32 matmul/cuDNN, Liger RMSNorm and
SwiGLU kernels, and veRL's Triton fused linear cross-entropy path. The latter avoids materializing
the full `[tokens, vocab]` logits tensor, which is especially important for Qwen3.5's 248,320-token
vocabulary and long TravelWeaver trajectories. Gradient checkpointing, remove-padding, BF16 mixed
precision, and reshard-after-forward remain enabled; parameter and optimizer CPU offload stay off
to avoid PCIe stalls on the two A800s.

The launcher streams metrics to the `travelweaver-sft` Weights & Biases project under the default
run name `qwen3.5-4b-multiturn-sft-v2-natural-633-a800x2-seed20260809`. Its local W&B sync files
live under the experiment output directory, and a persisted W&B run ID allows checkpoint-based
launcher restarts to resume the same dashboard run.

The launcher saves every 10 optimizer steps and also saves the final step. It retains only the
newest checkpoint after each save completes. Each checkpoint contains resumable model/optimizer
state and a Hugging Face export. Logs live alongside checkpoints under the experiment output
directory, and a successful completed run publishes `final-model` as a stable symlink to the final
Hugging Face export. Runtime settings can be changed without editing the script, for example:

```bash
TRAIN_BATCH_SIZE=8 MAX_TOKEN_LEN_PER_GPU=12288 \
  bash training/scripts/run_qwen3_5_4b_multiturn_sft.sh
```

Checkpoint and tracking settings can likewise be overridden with `SAVE_FREQ`,
`MAX_CKPT_TO_KEEP`, `RUN_DIR`, `PROJECT_NAME`, `EXPERIMENT_NAME`, and standard `WANDB_*`
environment variables.

Do not launch until every selected GPU is actually free; the script deliberately does not stop GPU
holders or other users' processes. Additional veRL Hydra overrides may be appended after the script
name.
