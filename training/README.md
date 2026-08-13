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
- TransferQueue: `0.1.8`
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

The root environment produces replay-verified `travelweaver-sft-v5` JSONL with an explicit
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
The adapter remains compatible with existing v2 action-only, v3 action-only/clean ReAct, and v4
recovery artifacts. Newly rebuilt datasets use v5; its `action_selective` mode can retain legal
teacher-forced context while supervising only selected correct actions and visible reflections.

Recovery ReAct is specified in
[`docs/react-sft-recovery-v1.md`](../docs/react-sft-recovery-v1.md). V4/V5 retain invalid assistant
turns and tool errors as causal context, supplies an explicit per-assistant-turn loss mask, and
supervises only subsequent reflection and valid actions. The adapter reads the separate
`assistant_loss_mask_json` Parquet column; it does not infer masks from error text or add private
fields to messages passed through Qwen's official chat template.

The default two-GPU full-parameter run is restricted to GPU 0/1. It uses FSDP2, two-way Ulysses
sequence parallelism, dynamic token batches, and a 65,536-token sequence limit. The per-GPU token
budget is 32,768, so the complete sequence fits across SP=2 without truncating early evidence.

Qwen3.5-4B supports a 262,144-token context. Keep the launcher at 65,536 by default and raise it
only after checking the dataset manifest's exact sequence distribution and the available GPU memory;
never truncate early evidence or error-recovery context.

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

For a reproducible train/validation split, use the audited splitter instead of randomly partitioning
Parquet rows. It stratifies task type, Scenario profile, trajectory family, and trip length; it also
rejects reused Blueprint semantic hashes and writes the exact assignments beside both Parquets:

```bash
uv run --project training python training/scripts/split_qwen_sft.py \
  --input-parquet data/sft/<batch>/all.parquet \
  --input-audit data/sft/<batch>/audit.jsonl \
  --output-dir data/sft/<batch>-split-v1 \
  --validation-count 300 \
  --seed 20260811
```

Use `--holdout-name test` when the requested in-distribution holdout should be written as
`test.parquet`; the default remains `validation.parquet`. This naming choice does not change the
split algorithm, and neither holdout replaces the pinned ChinaTravel blind benchmark.

Pass the resulting validation file with `VAL_FILE`. The launcher uses veRL's regular
`trainer.test_freq` path (configured by `VALIDATION_FREQ`) and records `val/loss`, perplexity, and
teacher-forced `val/token_accuracy`. The accuracy only covers tokens selected by the explicit SFT
loss mask, so user text, tool observations, masked recovery context, and Qwen thinking scaffolding
cannot inflate it. Validation includes its final partial batch. This is an in-distribution training
diagnostic; the pinned ChinaTravel benchmark remains the only final blind evaluation.

```bash
TRAIN_FILE=data/sft/<batch>-split-v1/train.parquet \
VAL_FILE=data/sft/<batch>-split-v1/validation.parquet \
VALIDATION_FREQ=25 \
GPU_HOLD_HANDOFF=1 \
bash training/scripts/run_qwen3_5_4b_multiturn_sft.sh
```

The local training entry point also passes `trainer.seed` to veRL's shuffled
`DistributedSampler`. Model initialization, stochastic training operations, and the per-epoch data
permutation therefore share the configured experiment seed instead of the PyTorch sampler default.

The launcher saves every 10 optimizer steps and also saves the final step. It retains only the
newest checkpoint after each save completes. Each checkpoint contains resumable model/optimizer
state and a Hugging Face export. Logs live alongside checkpoints under the experiment output
directory, and a successful completed run publishes `final-model` as a stable symlink to the final
Hugging Face export. Runtime settings can be changed without editing the script, for example:

```bash
TRAIN_BATCH_SIZE=8 MAX_TOKEN_LEN_PER_GPU=32768 \
  bash training/scripts/run_qwen3_5_4b_multiturn_sft.sh
```

Checkpoint and tracking settings can likewise be overridden with `SAVE_FREQ`,
`MAX_CKPT_TO_KEEP`, `RUN_DIR`, `PROJECT_NAME`, `EXPERIMENT_NAME`, and standard `WANDB_*`
environment variables.

On hosts where `/data2/yzs/gpu_hold.py` protects GPU 0/1, set `GPU_HOLD_HANDOFF=1`. The launcher
keeps the holder active throughout preflight, stops the single verified holder immediately before
starting `torchrun`, and installs an `EXIT/INT/TERM` trap that `exec`s the same holder after normal
completion or failure. It refuses an ambiguous zero- or multi-holder takeover instead of killing
unrelated GPU processes.

Do not launch until every selected GPU is actually free; the script deliberately does not stop GPU
holders or other users' processes. Additional veRL Hydra overrides may be appended after the script
name.

## Qwen3.5-4B online GRPO

Online RL uses the same deterministic `travelweaver-reward-v4` evaluator as offline rollout and
data construction. Build prompt-only Parquet from one or more generated-task artifact directories:

```bash
PYTHONPATH=training/src:src uv run --project training python \
  training/scripts/prepare_grpo_prompts.py \
  --input-dir data/generated/<batch-1> \
  --input-dir data/generated/<batch-2> \
  --output data/grpo/<combined-batch>/train.parquet
```

The two-A800 baseline launcher fixes `rollout.n=8`, disables GRPO standard-deviation normalization,
and filters every constant-reward group, including constant negative groups. It safely checkpoints
and stops after ten consecutive valid constant-reward groups; any usable group resets the streak.
The strict trajectory cap is 32,768 total tokens, including the initial prompt, assistant output,
and serialized tool observations. The AgentLoop derives each sample's response budget after prompt
tokenization. A trajectory terminated by the cap invalidates its whole eight-rollout group, which is
discarded and refilled without affecting the consecutive no-signal counter. Validation keeps capped
samples in its denominator and reports them as failures plus a separate overflow rate.
The 1,000-task pilot uses a deterministic 900/100 train/validation split, keeps actor/reference
parameter, optimizer, and activation offload disabled. The two-GPU baseline caps colocated vLLM at
65%; 80% cannot remap the KV cache after a GPU-only actor update. With Ulysses SP=2, its actor/ref
dynamic token cap is 16,384 per GPU, giving a 32K effective sequence budget. Liger, veRL's Triton
fused log-prob/cross-entropy path, fused AdamW, and TF32 reduce memory and compute overhead without
enabling CPU offload. Validation runs once before training and again at the final step; both metrics
and ten sampled validation generations are logged to W&B.

The 1K-prompt profile uses eight prompt groups per global step and eight rollouts per group, for
64 real trajectories per step. veRL v1 interprets `ppo_mini_batch_size` in prompt-group units and
internally multiplies it by `rollout.n`; a value of four therefore produces two 32-trajectory actor
updates with `ppo_epochs=1`. The run uses 112 global steps, validates again only at step 112, and
retains at most two actor checkpoints.

```bash
TRAIN_FILE=data/grpo/<batch>/train.parquet \
MODEL_PATH=training/outputs/<sft-run>/final-model \
bash training/scripts/run_qwen3_5_4b_travelweaver_grpo.sh --dry-run

TRAIN_FILE=data/grpo/<batch>/train.parquet \
MODEL_PATH=training/outputs/<sft-run>/final-model \
GPU_HOLD_HANDOFF=1 \
bash training/scripts/run_qwen3_5_4b_travelweaver_grpo.sh
```

On the eight-A800 `g0008` host, the four-GPU profile uses GPUs 0-3. It preserves the baseline's
eight prompt groups, eight rollouts per group, PPO mini-batch of four, two actor updates per global
step, 112 steps, and 7,168 total training trajectories. The extra GPUs provide two colocated TP=2
vLLM replicas with 16 AgentLoop workers; FSDP uses SP=2 and DP=2. Its per-GPU dynamic token cap is
24,576 and its vLLM memory fraction is 75%. A real launch refuses to start if any selected GPU has a
compute process. A CPU-only dry-run remains safe while GPUs are occupied.

```bash
TRAIN_FILE=/ssd/home/zc/travelweaver-agentic-rl/data/grpo/chinatravel-grpo-v4-1000-split90-10-zca800/train.parquet \
VAL_FILE=/ssd/home/zc/travelweaver-agentic-rl/data/grpo/chinatravel-grpo-v4-1000-split90-10-zca800/validation.parquet \
MODEL_PATH=/ssd/home/zc/travelweaver-agentic-rl/training/outputs/travelweaver-sft/chinatravel-deepseek-react-1500-train90-test10-3ep-a800x2-seed20260811/checkpoints/global_step_252/huggingface \
bash training/scripts/run_qwen3_5_4b_travelweaver_grpo_4gpu.sh --dry-run
```

The default profile uses vLLM TP=2, FSDP2 plus Ulysses SP=2, and a strict 32,768-token trajectory cap.
The prompt manifest and launcher preflight reject witness/Reward leakage, hash mismatches, wrong
model families, incompatible veRL hooks, or a group size other than eight. See
[`docs/outcome-reward-shaping-v4.md`](../docs/outcome-reward-shaping-v4.md) for the exact A/V/G
formula, admission behavior, compatibility policy, and pilot metrics.
