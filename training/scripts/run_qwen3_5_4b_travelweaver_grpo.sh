#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

TRAIN_FILE="${TRAIN_FILE:-data/grpo/chinatravel-grpo-v4-1000-split90-10/train.parquet}"
VAL_FILE="${VAL_FILE:-data/grpo/chinatravel-grpo-v4-1000-split90-10/validation.parquet}"
MODEL_PATH="${MODEL_PATH:-training/outputs/travelweaver-sft/latest/final-model}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NUM_GPUS=2
GROUP_SIZE=8
SP_SIZE=2
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
PPO_EPOCHS="${PPO_EPOCHS:-1}"

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
TRAJECTORY_MAX_TOKENS="${TRAJECTORY_MAX_TOKENS:-32768}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-${TRAJECTORY_MAX_TOKENS}}"
# SP=2 turns this per-GPU budget into the same strict 32K total-sequence cap.
# Fused log-prob kernels avoid materializing the full [tokens, vocabulary] logits.
MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-16384}"
TOTAL_STEPS="${TOTAL_STEPS:-112}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_P="${TOP_P:-0.95}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
SEED="${SEED:-20260813}"

PROJECT_NAME="${PROJECT_NAME:-travelweaver-grpo}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3.5-4b-avg-v4-g8-b8-mb4-s112-a800x2-seed${SEED}}"
RUN_DIR="${RUN_DIR:-training/outputs/${PROJECT_NAME}/${EXPERIMENT_NAME}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RUN_DIR}/checkpoints}"
LOG_DIR="${LOG_DIR:-${RUN_DIR}/logs}"
SAMPLER_STATE="${SAMPLER_STATE:-${RUN_DIR}/sampler-state.json}"
SAVE_FREQ="${SAVE_FREQ:-10}"
TEST_FREQ="${TEST_FREQ:-${TOTAL_STEPS}}"
MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-2}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}"
LOG_VAL_GENERATIONS="${LOG_VAL_GENERATIONS:-10}"
RESUME_MODE="${RESUME_MODE:-auto}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"

GPU_HOLD_HANDOFF="${GPU_HOLD_HANDOFF:-1}"
GPU_HOLD_PYTHON="${GPU_HOLD_PYTHON:-/data2/yzs/.conda/envs/glq_sft/bin/python}"
GPU_HOLD_SCRIPT="${GPU_HOLD_SCRIPT:-/data2/yzs/gpu_hold.py}"

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    shift
fi

if [[ ! -x training/.venv/bin/python ]]; then
    echo "Missing training/.venv; run: uv sync --project training --dev" >&2
    exit 1
fi
if [[ ! -f "${TRAIN_FILE}" ]]; then
    echo "GRPO training Parquet does not exist: ${TRAIN_FILE}" >&2
    exit 1
fi
if [[ ! -f "${VAL_FILE}" ]]; then
    echo "GRPO validation Parquet does not exist: ${VAL_FILE}" >&2
    exit 1
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "Qwen checkpoint does not exist: ${MODEL_PATH}" >&2
    exit 1
fi
if [[ "${CUDA_VISIBLE_DEVICES}" != "0,1" ]]; then
    echo "This launcher is restricted to CUDA_VISIBLE_DEVICES=0,1." >&2
    exit 1
fi
if [[ "${GPU_HOLD_HANDOFF}" != "0" && "${GPU_HOLD_HANDOFF}" != "1" ]]; then
    echo "GPU_HOLD_HANDOFF must be 0 or 1." >&2
    exit 1
fi
if [[ "${GPU_MEMORY_UTILIZATION}" != "0.8" ]]; then
    echo "This experiment requires GPU_MEMORY_UTILIZATION=0.8." >&2
    exit 1
fi
for value in \
    "${TRAIN_BATCH_SIZE}" "${PPO_MINI_BATCH_SIZE}" "${PPO_EPOCHS}" \
    "${TOTAL_STEPS}" "${SAVE_FREQ}" "${TEST_FREQ}" "${MAX_ACTOR_CKPT_TO_KEEP}" \
    "${MAX_PROMPT_LENGTH}" "${MAX_RESPONSE_LENGTH}" "${TRAJECTORY_MAX_TOKENS}" \
    "${MAX_TOKEN_LEN_PER_GPU}"; do
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "Batch, step, checkpoint, and validation settings must be positive integers." >&2
        exit 1
    fi
done
if (( MAX_PROMPT_LENGTH >= TRAJECTORY_MAX_TOKENS )); then
    echo "MAX_PROMPT_LENGTH must be smaller than TRAJECTORY_MAX_TOKENS." >&2
    exit 1
fi
if (( MAX_RESPONSE_LENGTH > TRAJECTORY_MAX_TOKENS )); then
    echo "MAX_RESPONSE_LENGTH cannot exceed TRAJECTORY_MAX_TOKENS." >&2
    exit 1
fi
if (( MAX_TOKEN_LEN_PER_GPU * SP_SIZE < TRAJECTORY_MAX_TOKENS )); then
    echo "The actor/ref SP token budget must cover TRAJECTORY_MAX_TOKENS." >&2
    exit 1
fi
if (( TRAIN_BATCH_SIZE % PPO_MINI_BATCH_SIZE != 0 )); then
    echo "TRAIN_BATCH_SIZE must be divisible by PPO_MINI_BATCH_SIZE." >&2
    exit 1
fi
if (( TRAIN_BATCH_SIZE / PPO_MINI_BATCH_SIZE * PPO_EPOCHS != 2 )); then
    echo "This profile requires exactly two actor optimizer updates per global step." >&2
    exit 1
fi
if [[ "${VAL_BEFORE_TRAIN}" != "true" && "${VAL_BEFORE_TRAIN}" != "false" ]]; then
    echo "VAL_BEFORE_TRAIN must be true or false." >&2
    exit 1
fi
if [[ ! "${LOG_VAL_GENERATIONS}" =~ ^[0-9]+$ ]]; then
    echo "LOG_VAL_GENERATIONS must be a non-negative integer." >&2
    exit 1
fi

TRAIN_FILE="$(realpath "${TRAIN_FILE}")"
VAL_FILE="$(realpath "${VAL_FILE}")"
MODEL_PATH="$(realpath "${MODEL_PATH}")"
RUN_DIR="$(realpath -m "${RUN_DIR}")"
CHECKPOINT_DIR="$(realpath -m "${CHECKPOINT_DIR}")"
LOG_DIR="$(realpath -m "${LOG_DIR}")"
SAMPLER_STATE="$(realpath -m "${SAMPLER_STATE}")"
AGENT_CONFIG="${PROJECT_ROOT}/training/configs/travelweaver_agent_loop.yaml"
SAMPLER_PATH="${PROJECT_ROOT}/training/src/travelweaver_grpo_sampler.py"

export CUDA_VISIBLE_DEVICES
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONPATH="${PROJECT_ROOT}/training/src:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_JOB_TYPE="${WANDB_JOB_TYPE:-grpo}"
export TRAVELWEAVER_TRAJECTORY_MAX_TOKENS="${TRAJECTORY_MAX_TOKENS}"

if [[ "${SKIP_PREFLIGHT}" != "1" ]]; then
    training/.venv/bin/python - \
        "${TRAIN_FILE}" "${VAL_FILE}" "${MODEL_PATH}" "${GROUP_SIZE}" \
        "${NUM_GPUS}" "${SP_SIZE}" "${MAX_PROMPT_LENGTH}" "${MAX_RESPONSE_LENGTH}" \
        "${TRAJECTORY_MAX_TOKENS}" "${MAX_TOKEN_LEN_PER_GPU}" \
        "${TRAIN_BATCH_SIZE}" "${PPO_MINI_BATCH_SIZE}" "${PPO_EPOCHS}" <<'PY'
import hashlib
import importlib.metadata
import inspect
import json
import sys
from pathlib import Path

import pandas as pd
from torch.optim import AdamW
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from transformers import AutoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer
from verl.trainer.ppo.v1.trainer_base import PPOTrainer
from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy

from travelweaver.env import TravelWeaverEnv

train_file, val_file, model_path = map(Path, sys.argv[1:4])
(
    group_size,
    num_gpus,
    sp_size,
    max_prompt,
    max_response,
    trajectory_max_tokens,
    max_token_len_per_gpu,
    train_batch_size,
    ppo_mini_batch_size,
    ppo_epochs,
) = map(int, sys.argv[4:])

if group_size != 8:
    raise SystemExit("TravelWeaver GRPO requires exactly eight rollouts per group.")
if num_gpus != 2 or sp_size != 2:
    raise SystemExit("This profile requires two A800 GPUs and sequence parallel size two.")
if (train_batch_size, ppo_mini_batch_size, ppo_epochs) != (8, 4, 1):
    raise SystemExit("This GRPO profile requires train batch 8, PPO mini-batch 4, and one PPO epoch.")
required = {"prompt", "data_source", "agent_name", "task_id", "task_dir", "extra_info"}
frames = {}
for role, path in (("train", train_file), ("validation", val_file)):
    frame = pd.read_parquet(path)
    frames[role] = frame
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"{role} GRPO Parquet is missing columns: {sorted(missing)}")
    if frame.empty:
        raise SystemExit(f"{role} GRPO Parquet is empty.")
    if frame["task_id"].duplicated().any():
        raise SystemExit(f"{role} GRPO Parquet contains duplicate task_id values.")
    if set(frame["agent_name"]) != {"travelweaver_agent"}:
        raise SystemExit(f"{role} GRPO Parquet has an unexpected agent_name.")
    manifest_path = path.with_name(f"{path.name}.manifest.json")
    if not manifest_path.is_file():
        raise SystemExit(f"{role} GRPO prompt manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if manifest.get("output_sha256") != digest:
        raise SystemExit(f"{role} GRPO Parquet SHA-256 does not match its manifest.")
    if manifest.get("contains_witness") is not False or manifest.get("contains_reward_labels") is not False:
        raise SystemExit(f"{role} GRPO prompt manifest does not prove hidden-label isolation.")
train_ids = set(frames["train"]["task_id"])
validation_ids = set(frames["validation"]["task_id"])
if train_ids & validation_ids:
    raise SystemExit("GRPO train and validation task IDs overlap.")
if len(train_ids) != 900 or len(validation_ids) != 100:
    raise SystemExit("This GRPO profile requires an exact 900/100 train-validation split.")

config = AutoConfig.from_pretrained(model_path, local_files_only=True)
if config.model_type != "qwen3_5":
    raise SystemExit(f"Expected qwen3_5 checkpoint, found {config.model_type!r}.")
if trajectory_max_tokens > int(getattr(config, "max_position_embeddings", 65536)):
    raise SystemExit("Trajectory token cap exceeds the checkpoint context window.")
if max_prompt >= trajectory_max_tokens or max_response > trajectory_max_tokens:
    raise SystemExit("Prompt/response configuration is incompatible with the trajectory cap.")
if max_token_len_per_gpu * sp_size < trajectory_max_tokens:
    raise SystemExit("Actor/ref SP token budget is smaller than the trajectory cap.")
text_config = getattr(config, "text_config", config)
if int(text_config.num_attention_heads) % sp_size != 0:
    raise SystemExit("Qwen attention heads must be divisible by the SP size.")
if sp_size > 1 and "cp_context" not in inspect.signature(chunk_gated_delta_rule).parameters:
    raise SystemExit("Installed FLA does not support Qwen3.5 Ulysses sequence parallelism.")
if "fused" not in inspect.signature(AdamW).parameters:
    raise SystemExit("Installed PyTorch AdamW does not expose the fused CUDA implementation.")
if not callable(linear_cross_entropy):
    raise SystemExit("veRL's fused linear cross-entropy kernel is unavailable.")
signature = inspect.signature(core_algos.compute_grpo_outcome_advantage)
if "norm_adv_by_std_in_grpo" not in signature.parameters:
    raise SystemExit("Pinned veRL no longer exposes GRPO std-normalization control.")
if not hasattr(ReplayBuffer, "_dapo_filtered_keys"):
    raise SystemExit("Pinned veRL no longer exposes the custom group-filter hook.")
update_actor_source = inspect.getsource(PPOTrainer._update_actor)
if "ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n" not in update_actor_source:
    raise SystemExit("Pinned veRL changed the v1 PPO mini-batch unit conversion.")
if importlib.metadata.version("TransferQueue") != "0.1.8":
    raise SystemExit("TransferQueue must remain pinned at 0.1.8.")
if len(TravelWeaverEnv.tool_schemas()) != 15:
    raise SystemExit("TravelWeaver online AgentLoop tool schema count changed unexpectedly.")

print(
    json.dumps(
        {
            "event": "grpo_preflight_ok",
            "group_size": group_size,
            "train_batch_size_groups": train_batch_size,
            "ppo_mini_batch_size_groups": ppo_mini_batch_size,
            "ppo_epochs": ppo_epochs,
            "trajectories_per_step": train_batch_size * group_size,
            "trajectories_per_optimizer_update": ppo_mini_batch_size * group_size,
            "optimizer_updates_per_step": train_batch_size // ppo_mini_batch_size * ppo_epochs,
            "num_gpus": num_gpus,
            "sequence_parallel_size": sp_size,
            "train_rows": len(train_ids),
            "validation_rows": len(validation_ids),
            "max_sequence_length": trajectory_max_tokens,
            "max_response_length": max_response,
            "token_budget_per_gpu": max_token_len_per_gpu,
            "fused_adamw": True,
            "fused_linear_cross_entropy": True,
            "liger": True,
            "tf32": True,
            "gpu_memory_utilization": 0.8,
            "training_offload": False,
            "norm_adv_by_std": False,
            "zero_variance_filter": "all_constant_reward_levels",
            "no_signal_stop_streak": 10,
            "reward_version": "travelweaver-reward-v4",
        },
        sort_keys=True,
    ),
    flush=True,
)
PY
fi

OVERRIDES=(
    "data.train_files=${TRAIN_FILE}"
    "data.val_files=${VAL_FILE}"
    "data.prompt_key=prompt"
    "data.train_batch_size=${TRAIN_BATCH_SIZE}"
    "data.gen_batch_size=1"
    "data.val_batch_size=1"
    "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
    "data.max_response_length=${MAX_RESPONSE_LENGTH}"
    "data.filter_overlong_prompts=true"
    "data.truncation=error"
    "data.dataloader_num_workers=2"
    "data.trust_remote_code=true"
    "data.continuous_token.enable=true"
    "data.continuous_token.model_family=qwen35"
    "+data.apply_chat_template_kwargs.enable_thinking=false"
    "actor_rollout_ref.model.path=${MODEL_PATH}"
    "actor_rollout_ref.model.trust_remote_code=true"
    "actor_rollout_ref.model.use_remove_padding=true"
    "actor_rollout_ref.model.enable_gradient_checkpointing=true"
    "actor_rollout_ref.model.enable_activation_offload=false"
    "actor_rollout_ref.model.use_liger=true"
    "actor_rollout_ref.model.use_fused_kernels=true"
    "actor_rollout_ref.model.fused_kernel_options.impl_backend=triton"
    "actor_rollout_ref.actor.strategy=fsdp2"
    "actor_rollout_ref.actor.optim.lr=${ACTOR_LR}"
    "actor_rollout_ref.actor.optim.override_optimizer_config={fused:true}"
    "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
    "actor_rollout_ref.actor.ppo_epochs=${PPO_EPOCHS}"
    "actor_rollout_ref.actor.use_dynamic_bsz=true"
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU}"
    "actor_rollout_ref.actor.use_kl_loss=true"
    "actor_rollout_ref.actor.kl_loss_coef=0.001"
    "actor_rollout_ref.actor.kl_loss_type=low_var_kl"
    "actor_rollout_ref.actor.entropy_coeff=0.0"
    "actor_rollout_ref.actor.use_torch_compile=false"
    "actor_rollout_ref.actor.fsdp_config.param_offload=false"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=false"
    "actor_rollout_ref.actor.fsdp_config.offload_policy=false"
    "actor_rollout_ref.actor.fsdp_config.reshard_after_forward=true"
    "actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=${SP_SIZE}"
    "actor_rollout_ref.ref.strategy=fsdp2"
    "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true"
    "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU}"
    "actor_rollout_ref.ref.fsdp_config.param_offload=false"
    "actor_rollout_ref.ref.fsdp_config.optimizer_offload=false"
    "actor_rollout_ref.ref.fsdp_config.offload_policy=false"
    "actor_rollout_ref.ref.fsdp_config.reshard_after_forward=true"
    "actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size=${SP_SIZE}"
    "actor_rollout_ref.rollout.name=vllm"
    "actor_rollout_ref.rollout.mode=async"
    "actor_rollout_ref.rollout.n=${GROUP_SIZE}"
    "actor_rollout_ref.rollout.temperature=${TEMPERATURE}"
    "actor_rollout_ref.rollout.top_p=${TOP_P}"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=2"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
    "actor_rollout_ref.rollout.max_num_seqs=${GROUP_SIZE}"
    "actor_rollout_ref.rollout.max_num_batched_tokens=8192"
    "actor_rollout_ref.rollout.enable_chunked_prefill=true"
    "actor_rollout_ref.rollout.enable_prefix_caching=true"
    "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true"
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU}"
    "actor_rollout_ref.rollout.multi_turn.enable=true"
    "actor_rollout_ref.rollout.multi_turn.max_assistant_turns=60"
    "actor_rollout_ref.rollout.multi_turn.max_user_turns=60"
    "actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1"
    "actor_rollout_ref.rollout.multi_turn.max_tool_response_length=16384"
    "actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=middle"
    "actor_rollout_ref.rollout.multi_turn.format=qwen3_coder"
    "actor_rollout_ref.rollout.agent.default_agent_loop=travelweaver_agent"
    "actor_rollout_ref.rollout.agent.agent_loop_config_path=${AGENT_CONFIG}"
    "actor_rollout_ref.rollout.agent.num_workers=8"
    "algorithm.adv_estimator=grpo"
    "algorithm.norm_adv_by_std_in_grpo=false"
    "algorithm.filter_groups.enable=false"
    "algorithm.use_kl_in_reward=false"
    "critic.enable=false"
    "reward.reward_model.enable=false"
    "trainer.use_v1=true"
    "trainer.v1.trainer_mode=travelweaver_sync"
    "trainer.v1.sampler.sync_refill_failed_groups=true"
    "trainer.v1.sampler.custom_sampler.path=${SAMPLER_PATH}"
    "trainer.v1.sampler.custom_sampler.name=TravelWeaverReplayBuffer"
    "+trainer.v1.sampler.sampler_kwargs.group_size=${GROUP_SIZE}"
    "+trainer.v1.sampler.sampler_kwargs.train_batch_size=${TRAIN_BATCH_SIZE}"
    "+trainer.v1.sampler.sampler_kwargs.zero_variance_tolerance=1e-8"
    "+trainer.v1.sampler.sampler_kwargs.max_consecutive_no_signal=10"
    "+trainer.v1.sampler.sampler_kwargs.state_path=${SAMPLER_STATE}"
    "trainer.project_name=${PROJECT_NAME}"
    "trainer.experiment_name=${EXPERIMENT_NAME}"
    "trainer.default_local_dir=${CHECKPOINT_DIR}"
    "trainer.total_epochs=30"
    "trainer.total_training_steps=${TOTAL_STEPS}"
    "trainer.n_gpus_per_node=${NUM_GPUS}"
    "trainer.nnodes=1"
    "trainer.save_freq=${SAVE_FREQ}"
    "trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP}"
    "trainer.max_critic_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP}"
    "trainer.test_freq=${TEST_FREQ}"
    "trainer.val_before_train=${VAL_BEFORE_TRAIN}"
    "trainer.log_val_generations=${LOG_VAL_GENERATIONS}"
    "trainer.logger=[console,wandb]"
    "trainer.resume_mode=${RESUME_MODE}"
    "+ray_kwargs.ray_init.runtime_env.env_vars.TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='1'"
)

if [[ "${DRY_RUN}" == "1" ]]; then
    training/.venv/bin/python training/scripts/run_travelweaver_grpo.py \
        --cfg job --resolve "${OVERRIDES[@]}" "$@"
    exit 0
fi

gpu_holder_pids() {
    pgrep -u "$(id -u)" -f -- "${GPU_HOLD_SCRIPT} --gpus 0,1 --force$" || true
}

restore_gpu_holder() {
    local status=$?
    trap - EXIT INT TERM
    if [[ "${GPU_HOLD_HANDOFF}" == "1" ]]; then
        if [[ -n "$(gpu_holder_pids)" ]]; then
            echo "GPU holder for 0/1 is already running; not starting a duplicate."
            exit "${status}"
        fi
        if [[ -f "${GPU_HOLD_SCRIPT}" && -x "${GPU_HOLD_PYTHON}" ]]; then
            echo "Restoring GPU holder on 0/1 after GRPO exit (status=${status})."
            exec "${GPU_HOLD_PYTHON}" -u "${GPU_HOLD_SCRIPT}" --gpus 0,1 --force
        fi
        echo "Cannot restore GPU holder: missing script or Python interpreter." >&2
    fi
    exit "${status}"
}

stop_gpu_holder() {
    local -a holder_pids=()
    mapfile -t holder_pids < <(gpu_holder_pids)
    if (( ${#holder_pids[@]} != 1 )); then
        echo "Expected exactly one GPU holder for 0/1, found ${#holder_pids[@]}." >&2
        return 1
    fi
    echo "Stopping GPU holder PID ${holder_pids[0]} immediately before GRPO startup."
    kill -TERM "${holder_pids[0]}"
    for _ in $(seq 1 30); do
        if ! kill -0 "${holder_pids[0]}" 2>/dev/null; then
            echo "GPU holder stopped; handing GPU 0/1 to GRPO."
            return 0
        fi
        sleep 1
    done
    echo "GPU holder PID ${holder_pids[0]} did not stop within 30 seconds." >&2
    return 1
}

if [[ "${GPU_HOLD_HANDOFF}" == "1" ]]; then
    trap restore_gpu_holder EXIT INT TERM
    stop_gpu_holder
fi

mkdir -p "${CHECKPOINT_DIR}" "${LOG_DIR}"
RUN_LOG="${LOG_DIR}/${EXPERIMENT_NAME}-$(date +%Y%m%d-%H%M%S).log"

echo "Starting TravelWeaver GRPO; W&B: ${PROJECT_NAME}/${EXPERIMENT_NAME}; log: ${RUN_LOG}"
training/.venv/bin/python training/scripts/run_travelweaver_grpo.py \
    "${OVERRIDES[@]}" "$@" 2>&1 | tee "${RUN_LOG}"
