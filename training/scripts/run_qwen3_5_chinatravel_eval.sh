#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

MODEL_PATH="${MODEL_PATH:-ckpts/Qwen3.5-4B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-4b}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
CONCURRENCY="${CONCURRENCY:-32}"
MAX_API_TURNS="${MAX_API_TURNS:-40}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-8192}"
GPU_HOLD_PYTHON="${GPU_HOLD_PYTHON:-/data2/yzs/.conda/envs/glq_sft/bin/python}"
GPU_HOLD_SCRIPT="${GPU_HOLD_SCRIPT:-/data2/yzs/gpu_hold.py}"

OUTPUT_PATH="training/outputs/chinatravel-official-654-qwen3.5-4b-native-agent.jsonl"
ERROR_PATH="training/outputs/chinatravel-official-654-qwen3.5-4b-native-agent-errors.jsonl"
SERVER_LOG="training/logs/qwen3_5_4b_chinatravel_server-resume.log"
RUNNER_LOG="training/logs/qwen3_5_4b_chinatravel_benchmark.log"

mkdir -p training/logs training/outputs

server_pid=""

restore_gpu_hold() {
    local status=$?
    trap - EXIT INT TERM
    if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
        kill -TERM "${server_pid}" 2>/dev/null || true
        for _ in $(seq 1 30); do
            kill -0 "${server_pid}" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "${server_pid}" 2>/dev/null || true
    fi
    if [[ -f "${GPU_HOLD_SCRIPT}" && -x "${GPU_HOLD_PYTHON}" ]]; then
        exec "${GPU_HOLD_PYTHON}" -u "${GPU_HOLD_SCRIPT}" --gpus 0,1 --force
    fi
    exit "${status}"
}
trap restore_gpu_hold EXIT INT TERM

CUDA_VISIBLE_DEVICES=0,1 training/.venv/bin/vllm serve "${MODEL_PATH}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 262144 \
    --max-num-seqs 32 \
    --max-num-batched-tokens 32768 \
    --language-model-only \
    --no-enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking":false}' \
    --generation-config vllm \
    --override-generation-config \
        '{"temperature":0.7,"top_p":0.8,"top_k":20,"min_p":0.0,"presence_penalty":1.5}' \
    --seed 20260808 >"${SERVER_LOG}" 2>&1 &
server_pid=$!

ready=0
for _ in $(seq 1 120); do
    if curl -fsS "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
        ready=1
        break
    fi
    if ! kill -0 "${server_pid}" 2>/dev/null; then
        break
    fi
    sleep 2
done
if [[ "${ready}" -ne 1 ]]; then
    tail -n 160 "${SERVER_LOG}"
    exit 1
fi

set +e
OPENAI_API_KEY=local \
OPENAI_BASE_URL="http://${HOST}:${PORT}/v1" \
OPENAI_MODEL="${SERVED_MODEL_NAME}" \
OPENAI_TIMEOUT_SECONDS=600 \
OPENAI_MAX_TOKENS="${MAX_COMPLETION_TOKENS}" \
OPENAI_TEMPERATURE=0.7 \
    uv run travelweaver rollout-benchmark \
        --split benchmark \
        --output "${OUTPUT_PATH}" \
        --errors "${ERROR_PATH}" \
        --concurrency "${CONCURRENCY}" \
        --max-api-turns "${MAX_API_TURNS}" 2>&1 | tee -a "${RUNNER_LOG}"
runner_status=${PIPESTATUS[0]}
set -e

uv run python - "${OUTPUT_PATH}" "${ERROR_PATH}" <<'PY'
import collections
import json
import sys
from pathlib import Path

output_path = Path(sys.argv[1])
error_path = Path(sys.argv[2])
rows = [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()]
errors = [json.loads(line) for line in error_path.read_text().splitlines() if line.strip()]
accepted = sum(bool(row.get("success")) for row in rows)
print(
    json.dumps(
        {
            "event": "benchmark_summary",
            "completed_trajectories": len(rows),
            "error_records": len(errors),
            "total_attempts": len(rows) + len(errors),
            "accepted": accepted,
            "accept_rate_all_attempts": accepted / (len(rows) + len(errors)),
            "termination_reasons": dict(
                collections.Counter(str(row.get("termination_reason")) for row in rows)
            ),
            "error_types": dict(
                collections.Counter(str(row.get("error_type")) for row in errors)
            ),
        },
        ensure_ascii=False,
        sort_keys=True,
    ),
    flush=True,
)
PY

exit "${runner_status}"
