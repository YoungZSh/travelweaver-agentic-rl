#!/usr/bin/env bash

set -Eeuo pipefail

# Four-A800 profile for g0008. Two colocated TP=2 vLLM replicas increase
# rollout concurrency while FSDP uses SP=2 and DP=2 across all four GPUs.
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export NUM_GPUS="4"
export GROUP_SIZE="8"
export SP_SIZE="2"
export ROLLOUT_TP_SIZE="2"
export ROLLOUT_DP_SIZE="1"
export ROLLOUT_MAX_NUM_SEQS="8"
export AGENT_NUM_WORKERS="16"
export DATALOADER_NUM_WORKERS="4"
export TRAIN_BATCH_SIZE="8"
export PPO_MINI_BATCH_SIZE="4"
export PPO_EPOCHS="1"
export TOTAL_STEPS="112"
export GPU_HOLD_HANDOFF="0"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3.5-4b-avg-v4-g8-b8-mb4-s112-vllm65-a800x4-seed${SEED:-20260813}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_qwen3_5_4b_travelweaver_grpo.sh" "$@"
