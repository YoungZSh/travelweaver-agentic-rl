#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

TRAIN_FILE="${TRAIN_FILE:-data/sft/chinatravel-qwen3.5-4b-action-633-sft-v2-natural/all.parquet}"
VAL_FILE="${VAL_FILE:-}"
MODEL_PATH="${MODEL_PATH:-ckpts/Qwen3.5-4B}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NUM_GPUS=2
SP_SIZE="${SP_SIZE:-2}"
GPU_HOLD_HANDOFF="${GPU_HOLD_HANDOFF:-0}"
GPU_HOLD_PYTHON="${GPU_HOLD_PYTHON:-/data2/yzs/.conda/envs/glq_sft/bin/python}"
GPU_HOLD_SCRIPT="${GPU_HOLD_SCRIPT:-/data2/yzs/gpu_hold.py}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
MAX_LENGTH="${MAX_LENGTH:-65536}"
MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-32768}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
LR="${LR:-1e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
SEED="${SEED:-20260809}"
FUSED_KERNEL_BACKEND="${FUSED_KERNEL_BACKEND:-triton}"

PROJECT_NAME="${PROJECT_NAME:-travelweaver-sft}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3.5-4b-multiturn-sft-v2-natural-633-a800x2-seed${SEED}}"
RUN_DIR="${RUN_DIR:-training/outputs/${PROJECT_NAME}/${EXPERIMENT_NAME}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RUN_DIR}/checkpoints}"
LOG_DIR="${LOG_DIR:-${RUN_DIR}/logs}"
WANDB_DIR="${WANDB_DIR:-${RUN_DIR}/wandb}"
SAVE_FREQ="${SAVE_FREQ:-10}"
VALIDATION_FREQ="${VALIDATION_FREQ:-25}"
MAX_CKPT_TO_KEEP="${MAX_CKPT_TO_KEEP:-1}"
RESUME_MODE="${RESUME_MODE:-auto}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    shift
fi

if [[ ! -x training/.venv/bin/torchrun ]]; then
    echo "Missing training/.venv; run: uv sync --project training --dev" >&2
    exit 1
fi
if [[ ! -f "${TRAIN_FILE}" ]]; then
    echo "SFT Parquet does not exist: ${TRAIN_FILE}" >&2
    exit 1
fi
if [[ -n "${VAL_FILE}" && ! -f "${VAL_FILE}" ]]; then
    echo "Validation SFT Parquet does not exist: ${VAL_FILE}" >&2
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
if [[ "${FUSED_KERNEL_BACKEND}" != "triton" && "${FUSED_KERNEL_BACKEND}" != "torch" ]]; then
    echo "FUSED_KERNEL_BACKEND must be triton or torch." >&2
    exit 1
fi
if [[ "${GPU_HOLD_HANDOFF}" != "0" && "${GPU_HOLD_HANDOFF}" != "1" ]]; then
    echo "GPU_HOLD_HANDOFF must be 0 or 1." >&2
    exit 1
fi
if (( NUM_GPUS < 1 || SP_SIZE < 1 || NUM_GPUS % SP_SIZE != 0 )); then
    echo "NUM_GPUS=${NUM_GPUS} must be positive and divisible by SP_SIZE=${SP_SIZE}." >&2
    exit 1
fi
if [[ ! "${SAVE_FREQ}" =~ ^[1-9][0-9]*$ ]]; then
    echo "SAVE_FREQ=${SAVE_FREQ} must be a positive integer." >&2
    exit 1
fi
if [[ -n "${VAL_FILE}" && ! "${VALIDATION_FREQ}" =~ ^[1-9][0-9]*$ && "${VALIDATION_FREQ}" != "after_each_epoch" ]]; then
    echo "VALIDATION_FREQ must be a positive integer or after_each_epoch when VAL_FILE is set." >&2
    exit 1
fi
if [[ ! "${MAX_CKPT_TO_KEEP}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_CKPT_TO_KEEP=${MAX_CKPT_TO_KEEP} must be a positive integer." >&2
    exit 1
fi

DP_SIZE=$((NUM_GPUS / SP_SIZE))
if (( TRAIN_BATCH_SIZE % DP_SIZE != 0 )); then
    echo "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} must be divisible by DP_SIZE=${DP_SIZE}." >&2
    exit 1
fi

TRAIN_FILE="$(realpath "${TRAIN_FILE}")"
if [[ -n "${VAL_FILE}" ]]; then
    VAL_FILE="$(realpath "${VAL_FILE}")"
    TEST_FREQ="${VALIDATION_FREQ}"
else
    VAL_FILE="null"
    TEST_FREQ="-1"
fi
MODEL_PATH="$(realpath "${MODEL_PATH}")"
RUN_DIR="$(realpath -m "${RUN_DIR}")"
CHECKPOINT_DIR="$(realpath -m "${CHECKPOINT_DIR}")"
LOG_DIR="$(realpath -m "${LOG_DIR}")"
WANDB_DIR="$(realpath -m "${WANDB_DIR}")"
DATASET_CLASS="${PROJECT_ROOT}/training/src/travelweaver_sft_dataset.py"

export CUDA_VISIBLE_DEVICES
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export WANDB_DIR
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"
export WANDB_JOB_TYPE="${WANDB_JOB_TYPE:-sft}"

if [[ "${SKIP_PREFLIGHT}" != "1" ]]; then
    training/.venv/bin/python - \
        "${TRAIN_FILE}" "${VAL_FILE}" "${MODEL_PATH}" "${MAX_LENGTH}" "${MAX_TOKEN_LEN_PER_GPU}" \
        "${SP_SIZE}" "${NUM_GPUS}" "${TRAIN_BATCH_SIZE}" <<'PY'
import hashlib
import inspect
import json
import sys
from pathlib import Path

import pandas as pd
from torch.optim import AdamW
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from transformers import AutoConfig
from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy

train_file = Path(sys.argv[1])
raw_val_file = sys.argv[2]
val_file = None if raw_val_file == "null" else Path(raw_val_file)
model_path = Path(sys.argv[3])
max_length, token_budget, sp_size, num_gpus, train_batch_size = map(int, sys.argv[4:])

required = {
    "sample_id",
    "task_id",
    "messages_json",
    "tools_json",
    "enable_thinking",
}
def validate_parquet(path: Path, *, role: str) -> tuple[pd.DataFrame, int | None]:
    frame = pd.read_parquet(path)
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"{role} SFT Parquet is missing columns: {sorted(missing)}")
    if frame.empty:
        raise SystemExit(f"{role} SFT Parquet is empty.")
    if frame["sample_id"].duplicated().any():
        raise SystemExit(f"{role} SFT Parquet contains duplicate sample_id values.")
    if bool(frame["enable_thinking"].any()):
        raise SystemExit(f"All {role} Qwen3.5 SFT rows must set enable_thinking=false.")

    manifest_path = path.parent / "manifest.json"
    sequence_max = None
    if not manifest_path.exists():
        return frame, sequence_max
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") == "travelweaver-sft-v4":
        if "assistant_loss_mask_json" not in frame.columns:
            raise SystemExit(f"{role} SFT V4 Parquet is missing assistant_loss_mask_json.")
        if frame["assistant_loss_mask_json"].isna().any():
            raise SystemExit(f"{role} SFT V4 Parquet contains a null assistant loss mask.")
    adapter = manifest.get("qwen_adapter", {})
    expected_digest = adapter.get("parquet_sha256")
    split_descriptors = adapter.get("parquet_splits", {})
    for descriptor in split_descriptors.values() if isinstance(split_descriptors, dict) else []:
        if not isinstance(descriptor, dict):
            continue
        descriptor_path = descriptor.get("path")
        if descriptor_path and Path(str(descriptor_path)).resolve() == path.resolve():
            expected_digest = descriptor.get("sha256")
            break
    if expected_digest:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected_digest:
            raise SystemExit(f"{role} SFT Parquet SHA-256 does not match its manifest.")
    sequence_max = adapter.get("sequence_tokens", {}).get("max")
    if sequence_max is not None and int(sequence_max) > max_length:
        raise SystemExit(
            f"{role} dataset max sequence {sequence_max} exceeds MAX_LENGTH={max_length}."
        )
    if sequence_max is not None and int(sequence_max) > token_budget * sp_size:
        raise SystemExit(
            f"{role} dataset max sequence {sequence_max} exceeds the SP token budget "
            f"{token_budget * sp_size}."
        )
    return frame, sequence_max


train_frame, sequence_max = validate_parquet(train_file, role="Training")
val_frame = None
if val_file is not None:
    val_frame, val_sequence_max = validate_parquet(val_file, role="Validation")
    if set(train_frame["sample_id"]) & set(val_frame["sample_id"]):
        raise SystemExit("Training and validation SFT Parquet files overlap on sample_id.")
    known_sequence_maxima = [value for value in (sequence_max, val_sequence_max) if value is not None]
    sequence_max = max(known_sequence_maxima) if known_sequence_maxima else None

config = AutoConfig.from_pretrained(model_path, local_files_only=True)
if config.model_type != "qwen3_5":
    raise SystemExit(f"Expected qwen3_5 checkpoint, found {config.model_type!r}.")
text_config = getattr(config, "text_config", config)
attention_heads = int(text_config.num_attention_heads)
if attention_heads % sp_size != 0:
    raise SystemExit(
        f"Qwen attention heads ({attention_heads}) must be divisible by SP_SIZE={sp_size}."
    )
if sp_size > 1 and "cp_context" not in inspect.signature(chunk_gated_delta_rule).parameters:
    raise SystemExit("Installed FLA does not support Qwen3.5 Ulysses sequence parallelism.")
if "fused" not in inspect.signature(AdamW).parameters:
    raise SystemExit("Installed PyTorch AdamW does not expose the fused CUDA implementation.")
if not callable(linear_cross_entropy):
    raise SystemExit("veRL's fused linear cross-entropy kernel is unavailable.")

dp_size = num_gpus // sp_size
print(
    json.dumps(
        {
            "event": "sft_preflight_ok",
            "train_samples": len(train_frame),
            "validation_samples": 0 if val_frame is None else len(val_frame),
            "sequence_max": sequence_max,
            "num_gpus": num_gpus,
            "sp_size": sp_size,
            "dp_size": dp_size,
            "global_batch_size": train_batch_size,
            "samples_per_dp_step": train_batch_size // dp_size,
            "max_length": max_length,
            "token_budget_per_gpu": token_budget,
            "fused_adamw": True,
            "fused_linear_cross_entropy": True,
            "tf32": True,
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
    "data.train_batch_size=${TRAIN_BATCH_SIZE}"
    "data.micro_batch_size_per_gpu=1"
    "data.max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU}"
    "data.use_dynamic_bsz=true"
    "data.messages_key=messages_json"
    "data.tools_key=tools_json"
    "data.enable_thinking_key=enable_thinking"
    "data.enable_thinking_default=false"
    "data.pad_mode=no_padding"
    "data.max_length=${MAX_LENGTH}"
    "data.truncation=error"
    "+data.apply_chat_template_kwargs.enable_thinking=false"
    "data.custom_cls.path=${DATASET_CLASS}"
    "data.custom_cls.name=TravelWeaverMultiTurnSFTDataset"
    "data.ignore_input_ids_mismatch=false"
    "data.num_workers=4"
    "model.path=${MODEL_PATH}"
    "model.use_remove_padding=true"
    "model.enable_gradient_checkpointing=true"
    "model.lora_rank=0"
    "model.use_liger=true"
    "model.use_fused_kernels=true"
    "model.fused_kernel_options.impl_backend=${FUSED_KERNEL_BACKEND}"
    "engine=fsdp"
    "engine.strategy=fsdp2"
    "engine.fsdp_size=-1"
    "engine.ulysses_sequence_parallel_size=${SP_SIZE}"
    "engine.reshard_after_forward=true"
    "engine.param_offload=false"
    "engine.optimizer_offload=false"
    "engine.use_torch_compile=false"
    "optim=fsdp"
    "optim.lr=${LR}"
    "optim.lr_warmup_steps_ratio=${WARMUP_RATIO}"
    "optim.weight_decay=0.1"
    "optim.betas=[0.9,0.95]"
    "optim.override_optimizer_config={fused:true}"
    "optim.clip_grad=1.0"
    "optim.lr_scheduler_type=cosine"
    "optim.min_lr_ratio=0.1"
    "trainer.project_name=${PROJECT_NAME}"
    "trainer.experiment_name=${EXPERIMENT_NAME}"
    "trainer.default_local_dir=${CHECKPOINT_DIR}"
    "trainer.total_epochs=${TOTAL_EPOCHS}"
    "trainer.total_training_steps=null"
    "trainer.save_freq=${SAVE_FREQ}"
    "trainer.test_freq=${TEST_FREQ}"
    "trainer.logger=[console,wandb]"
    "trainer.seed=${SEED}"
    "trainer.resume_mode=${RESUME_MODE}"
    "trainer.max_ckpt_to_keep=${MAX_CKPT_TO_KEEP}"
    "trainer.nnodes=1"
    "trainer.n_gpus_per_node=${NUM_GPUS}"
    "checkpoint.save_contents=[model,optimizer,extra,hf_model]"
)

if [[ "${DRY_RUN}" == "1" ]]; then
    training/.venv/bin/python training/scripts/run_verl_sft.py \
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
            echo "Restoring GPU holder on 0/1 after SFT exit (status=${status})."
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
    echo "Stopping GPU holder PID ${holder_pids[0]} immediately before SFT startup."
    kill -TERM "${holder_pids[0]}"
    for _ in $(seq 1 30); do
        if ! kill -0 "${holder_pids[0]}" 2>/dev/null; then
            echo "GPU holder stopped; handing GPU 0/1 to SFT."
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

mkdir -p "${CHECKPOINT_DIR}" "${LOG_DIR}" "${WANDB_DIR}"

# Keep one W&B run across checkpoint-based launcher restarts.
WANDB_RUN_ID_FILE="${RUN_DIR}/wandb-run-id.txt"
if [[ -f "${WANDB_RUN_ID_FILE}" ]]; then
    read -r WANDB_RUN_ID < "${WANDB_RUN_ID_FILE}"
else
    WANDB_RUN_ID="$(training/.venv/bin/python -c 'import wandb; print(wandb.util.generate_id())')"
    printf '%s\n' "${WANDB_RUN_ID}" > "${WANDB_RUN_ID_FILE}.tmp"
    mv "${WANDB_RUN_ID_FILE}.tmp" "${WANDB_RUN_ID_FILE}"
fi
export WANDB_RUN_ID

RUN_LOG="${LOG_DIR}/${EXPERIMENT_NAME}-$(date +%Y%m%d-%H%M%S).log"

echo "Starting TravelWeaver multi-turn SFT; W&B: ${PROJECT_NAME}/${EXPERIMENT_NAME}; log: ${RUN_LOG}"
training/.venv/bin/torchrun --standalone --nnodes=1 --nproc-per-node="${NUM_GPUS}" \
    training/scripts/run_verl_sft.py "${OVERRIDES[@]}" "$@" 2>&1 | tee "${RUN_LOG}"

# The last step is always checkpointed by veRL, even when it is not divisible by SAVE_FREQ.
# Publish a stable path to the final Hugging Face export after successful training.
TRACKER_FILE="${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt"
if [[ -f "${TRACKER_FILE}" ]]; then
    # veRL writes this tracker without a trailing newline. `read` therefore returns status 1 and,
    # under `set -e`, used to skip publishing final-model even after a successful checkpoint.
    FINAL_STEP="$(<"${TRACKER_FILE}")"
    FINAL_HF_DIR="${CHECKPOINT_DIR}/global_step_${FINAL_STEP}/huggingface"
    if [[ -d "${FINAL_HF_DIR}" ]]; then
        ln -sfnT "checkpoints/global_step_${FINAL_STEP}/huggingface" "${RUN_DIR}/final-model"
        echo "Final Hugging Face model: ${RUN_DIR}/final-model"
    fi
fi
