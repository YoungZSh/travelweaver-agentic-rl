#!/usr/bin/env bash

set -Eeuo pipefail

# Four-A800 profile for g0008. Two colocated TP=2 vLLM replicas increase
# rollout concurrency while FSDP uses SP=2 and DP=2 across all four GPUs.
# This single-node host's ib0 only has a link-local IPv6 address, which stalls
# NCCL bootstrap. Keep collectives on NVLink and use loopback for OOB setup.
export NCCL_IB_DISABLE="1"
export NCCL_SOCKET_IFNAME="lo"
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
export MAX_TOKEN_LEN_PER_GPU="24576"
export GPU_MEMORY_UTILIZATION="0.75"
export PPO_EPOCHS="1"
export TOTAL_STEPS="112"
export GPU_HOLD_HANDOFF="0"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3.5-4b-avg-v4-g8-b8-mb4-s112-vllm75-a800x4-seed${SEED:-20260813}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_LAUNCHER="${SCRIPT_DIR}/run_qwen3_5_4b_travelweaver_grpo.sh"
HOLDER_SESSION_PREFIX="omini_sft_gpu_reservation_gpu"
HOLDER_CWD="/ssd/home/zc/yzs/omini-st-encoder/omini"
HOLDER_PYTHON="/ssd/home/zc/miniconda3/envs/omni_verl/bin/python"
HOLDER_SCRIPT="/ssd/home/zc/yzs/omini-staqa/datagen/omini_synth/Data_gen_Mass_V3/gpu_occupy.py"
declare -a VERIFIED_HOLDER_PIDS=()

holder_session_name() {
    echo "${HOLDER_SESSION_PREFIX}$1"
}

verify_holder() {
    local gpu="$1"
    local session
    local pid
    local actual_user
    local actual_args
    local actual_cuda
    local expected_args
    session="$(holder_session_name "${gpu}")"
    if ! tmux has-session -t "${session}" 2>/dev/null; then
        echo "Missing expected GPU holder tmux session: ${session}" >&2
        return 1
    fi
    if [[ "$(tmux list-panes -t "${session}" -F '#{pane_id}' | wc -l)" != "1" ]]; then
        echo "GPU holder session ${session} must contain exactly one pane." >&2
        return 1
    fi
    pid="$(tmux display-message -p -t "${session}:0.0" '#{pane_pid}')"
    actual_user="$(ps -o user= -p "${pid}" | sed 's/^[[:space:]]*//')"
    actual_args="$(ps -o args= -p "${pid}" | sed 's/^[[:space:]]*//')"
    actual_cuda="$(tr '\0' '\n' <"/proc/${pid}/environ" | awk -F= '$1 == "CUDA_VISIBLE_DEVICES" {print $2}')"
    expected_args="${HOLDER_PYTHON} ${HOLDER_SCRIPT} --gb 72 --touch-sec 30"
    if [[ "${actual_user}" != "$(id -un)" || "${actual_cuda}" != "${gpu}" ]]; then
        echo "GPU holder ${session} has an unexpected owner or CUDA binding." >&2
        return 1
    fi
    if [[ "${actual_args}" != "${expected_args}" ]]; then
        echo "GPU holder ${session} has an unexpected process command." >&2
        return 1
    fi
    VERIFIED_HOLDER_PIDS[${gpu}]="${pid}"
    echo "Verified holder ${session}: GPU ${gpu}, PID ${pid}."
}

start_holder() {
    local gpu="$1"
    local session
    session="$(holder_session_name "${gpu}")"
    tmux new-session -d \
        -s "${session}" \
        -c "${HOLDER_CWD}" \
        "exec env CUDA_VISIBLE_DEVICES=${gpu} ${HOLDER_PYTHON} ${HOLDER_SCRIPT} --gb 72 --touch-sec 30"
}

restore_holders() {
    local training_status=$?
    local restore_failed=0
    local gpu
    local session
    trap - EXIT HUP INT TERM
    echo "Restoring the exact GPU 0-3 holder sessions after GRPO exit (status=${training_status})."
    for gpu in 0 1 2 3; do
        session="$(holder_session_name "${gpu}")"
        if tmux has-session -t "${session}" 2>/dev/null; then
            :
        else
            start_holder "${gpu}" || restore_failed=1
        fi
    done
    for gpu in 0 1 2 3; do
        verify_holder "${gpu}" || restore_failed=1
    done
    if (( restore_failed )); then
        echo "One or more GPU holders could not be restored exactly." >&2
        exit 1
    fi
    echo "GPU 0-3 holder sessions restored."
    exit "${training_status}"
}

stop_holders() {
    local gpu
    local session
    for gpu in 0 1 2 3; do
        verify_holder "${gpu}"
    done
    for gpu in 0 1 2 3; do
        session="$(holder_session_name "${gpu}")"
        echo "Stopping verified holder session ${session} immediately before GRPO startup."
        tmux kill-session -t "${session}"
    done
    for _ in $(seq 1 30); do
        local any_alive=0
        local pid
        for pid in "${VERIFIED_HOLDER_PIDS[@]}"; do
            if kill -0 "${pid}" 2>/dev/null; then
                any_alive=1
            fi
        done
        if (( ! any_alive )); then
            echo "Verified GPU 0-3 holder processes stopped; handing GPUs to GRPO."
            return 0
        fi
        sleep 1
    done
    echo "GPU holder processes did not stop within 30 seconds." >&2
    return 1
}

if [[ "${DRY_RUN:-0}" == "1" || "${1:-}" == "--dry-run" ]]; then
    exec "${BASE_LAUNCHER}" "$@"
fi

# Keep all holders alive during the complete CPU-only validation pass.
SKIP_PREFLIGHT=0 DRY_RUN=1 "${BASE_LAUNCHER}" "$@"

trap restore_holders EXIT HUP INT TERM
stop_holders
SKIP_PREFLIGHT=1 DRY_RUN=0 "${BASE_LAUNCHER}" "$@"
