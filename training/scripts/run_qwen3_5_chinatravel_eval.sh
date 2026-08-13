#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

MODEL_PATH="${MODEL_PATH:-ckpts/Qwen3.5-4B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-4b}"
BENCHMARK_SPLIT="${BENCHMARK_SPLIT:-benchmark}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
CONCURRENCY="${CONCURRENCY:-32}"
MAX_API_TURNS="${MAX_API_TURNS:-60}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-8192}"
GPU_HOLD_PYTHON="${GPU_HOLD_PYTHON:-/data2/yzs/.conda/envs/glq_sft/bin/python}"
GPU_HOLD_SCRIPT="${GPU_HOLD_SCRIPT:-/data2/yzs/gpu_hold.py}"
GPU_HOLD_HANDOFF="${GPU_HOLD_HANDOFF:-1}"
KEEP_SERVER_RUNNING="${KEEP_SERVER_RUNNING:-0}"

if [[ "${KEEP_SERVER_RUNNING}" != "0" && "${KEEP_SERVER_RUNNING}" != "1" ]]; then
    echo "KEEP_SERVER_RUNNING must be 0 or 1." >&2
    exit 1
fi

case "${BENCHMARK_SPLIT}" in
    easy)
        EXPECTED_TASKS=300
        ;;
    medium)
        EXPECTED_TASKS=150
        ;;
    human)
        EXPECTED_TASKS=154
        ;;
    benchmark)
        EXPECTED_TASKS=654
        ;;
    *)
        echo "Unsupported BENCHMARK_SPLIT=${BENCHMARK_SPLIT}." >&2
        exit 1
        ;;
esac

if [[ "${BENCHMARK_SPLIT}" == "benchmark" ]]; then
    DEFAULT_OUTPUT_PATH="training/outputs/chinatravel-official-654-qwen3.5-4b-native-agent.jsonl"
else
    DEFAULT_OUTPUT_PATH="training/outputs/chinatravel-official-${BENCHMARK_SPLIT}-${EXPECTED_TASKS}-qwen3.5-4b-native-agent.jsonl"
fi
OUTPUT_PATH="${OUTPUT_PATH:-${DEFAULT_OUTPUT_PATH}}"
ERROR_PATH="${ERROR_PATH:-${OUTPUT_PATH%.jsonl}-errors.jsonl}"
SERVER_LOG="${SERVER_LOG:-training/logs/qwen3_5_4b_chinatravel_server-resume.log}"
RUNNER_LOG="${RUNNER_LOG:-training/logs/qwen3_5_4b_chinatravel_benchmark.log}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_PATH%.jsonl}-summary.json}"
MANIFEST_PATH="${MANIFEST_PATH:-${OUTPUT_PATH%.jsonl}-manifest.json}"
BASELINE_OUTPUT_PATH="${BASELINE_OUTPUT_PATH:-training/outputs/chinatravel-official-654-qwen3.5-4b-native-agent.jsonl}"
BASELINE_ERROR_PATH="${BASELINE_ERROR_PATH:-training/outputs/chinatravel-official-654-qwen3.5-4b-native-agent-errors.jsonl}"
COMPARISON_PATH="${COMPARISON_PATH:-${OUTPUT_PATH%.jsonl}-vs-base.json}"

MODEL_PATH="$(realpath "${MODEL_PATH}")"
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "Model checkpoint does not exist: ${MODEL_PATH}" >&2
    exit 1
fi

mkdir -p \
    "$(dirname "${OUTPUT_PATH}")" \
    "$(dirname "${ERROR_PATH}")" \
    "$(dirname "${SERVER_LOG}")" \
    "$(dirname "${RUNNER_LOG}")" \
    "$(dirname "${SUMMARY_PATH}")" \
    "$(dirname "${MANIFEST_PATH}")" \
    "$(dirname "${COMPARISON_PATH}")"

uv run python - \
    "${MANIFEST_PATH}" "${MODEL_PATH}" "${SERVED_MODEL_NAME}" \
    "${OUTPUT_PATH}" "${ERROR_PATH}" "${CONCURRENCY}" \
    "${MAX_API_TURNS}" "${MAX_COMPLETION_TOKENS}" \
    "${BENCHMARK_SPLIT}" "${EXPECTED_TASKS}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    manifest_path,
    model_path,
    served_model_name,
    output_path,
    error_path,
    concurrency,
    max_api_turns,
    max_completion_tokens,
    benchmark_split,
    expected_tasks,
) = sys.argv[1:]
manifest = {
    "schema_version": "travelweaver-benchmark-rollout-run-v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "benchmark": {"split": benchmark_split, "expected_tasks": int(expected_tasks)},
    "model": {"path": model_path, "served_name": served_model_name},
    "rollout": {
        "output_path": output_path,
        "error_path": error_path,
        "concurrency": int(concurrency),
        "max_api_turns": int(max_api_turns),
        "max_completion_tokens": int(max_completion_tokens),
        "enable_thinking": False,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "seed": 20260808,
        "tool_response_mode": "delta",
    },
}
Path(manifest_path).write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

server_pid=""

gpu_holder_pids() {
    pgrep -u "$(id -u)" -f -- "${GPU_HOLD_SCRIPT} --gpus 0,1 --force$" || true
}

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
    if [[ "${GPU_HOLD_HANDOFF}" != "1" ]]; then
        exit "${status}"
    fi
    if [[ -n "$(gpu_holder_pids)" ]]; then
        echo "GPU holder for 0/1 is already running; not starting a duplicate."
        exit "${status}"
    fi
    if [[ -f "${GPU_HOLD_SCRIPT}" && -x "${GPU_HOLD_PYTHON}" ]]; then
        echo "Restoring GPU holder on 0/1 after benchmark exit (status=${status})."
        exec "${GPU_HOLD_PYTHON}" -u "${GPU_HOLD_SCRIPT}" --gpus 0,1 --force
    fi
    echo "Cannot restore GPU holder: missing script or Python interpreter." >&2
    exit "${status}"
}

stop_gpu_holder() {
    local -a holder_pids=()
    mapfile -t holder_pids < <(gpu_holder_pids)
    if (( ${#holder_pids[@]} == 0 )); then
        echo "No active GPU holder for 0/1; proceeding with benchmark."
        return 0
    fi
    if (( ${#holder_pids[@]} != 1 )); then
        echo "Expected at most one GPU holder for 0/1, found ${#holder_pids[@]}." >&2
        return 1
    fi
    echo "Stopping GPU holder PID ${holder_pids[0]} immediately before benchmark startup."
    kill -TERM "${holder_pids[0]}"
    for _ in $(seq 1 30); do
        if ! kill -0 "${holder_pids[0]}" 2>/dev/null; then
            echo "GPU holder stopped; handing GPU 0/1 to benchmark."
            return 0
        fi
        sleep 1
    done
    echo "GPU holder PID ${holder_pids[0]} did not stop within 30 seconds." >&2
    return 1
}

trap restore_gpu_hold EXIT INT TERM

if [[ "${GPU_HOLD_HANDOFF}" == "1" ]]; then
    stop_gpu_holder
fi

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
        --split "${BENCHMARK_SPLIT}" \
        --output "${OUTPUT_PATH}" \
        --errors "${ERROR_PATH}" \
        --concurrency "${CONCURRENCY}" \
        --max-api-turns "${MAX_API_TURNS}" 2>&1 | tee -a "${RUNNER_LOG}"
runner_status=${PIPESTATUS[0]}
set -e

uv run python - "${OUTPUT_PATH}" "${ERROR_PATH}" <<'PY' | tee "${SUMMARY_PATH}"
import collections
import json
import sys
from pathlib import Path

output_path = Path(sys.argv[1])
error_path = Path(sys.argv[2])
rows = [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()]
errors = (
    [json.loads(line) for line in error_path.read_text().splitlines() if line.strip()]
    if error_path.exists()
    else []
)
accepted = sum(bool(row.get("success")) for row in rows)
print(
    json.dumps(
        {
            "event": "benchmark_summary",
            "completed_trajectories": len(rows),
            "error_records": len(errors),
            "total_attempts": len(rows) + len(errors),
            "accepted": accepted,
            "accept_rate_all_attempts": (
                accepted / (len(rows) + len(errors)) if rows or errors else None
            ),
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

if [[ -f "${BASELINE_OUTPUT_PATH}" && -f "${BASELINE_ERROR_PATH}" ]]; then
    uv run python - \
        "${BASELINE_OUTPUT_PATH}" "${BASELINE_ERROR_PATH}" \
        "${OUTPUT_PATH}" "${ERROR_PATH}" "${COMPARISON_PATH}" <<'PY'
import collections
import json
import sys
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(output_path: Path, error_path: Path) -> dict:
    rows = load_jsonl(output_path)
    errors = load_jsonl(error_path)
    attempts = len(rows) + len(errors)
    accepted = sum(bool(row.get("success")) for row in rows)
    return {
        "completed_trajectories": len(rows),
        "error_records": len(errors),
        "total_attempts": attempts,
        "accepted": accepted,
        "accept_rate_all_attempts": accepted / attempts if attempts else None,
        "termination_reasons": dict(
            collections.Counter(str(row.get("termination_reason")) for row in rows)
        ),
        "error_types": dict(
            collections.Counter(str(row.get("error_type")) for row in errors)
        ),
    }


baseline = summarize(Path(sys.argv[1]), Path(sys.argv[2]))
trained = summarize(Path(sys.argv[3]), Path(sys.argv[4]))
baseline_rate = baseline["accept_rate_all_attempts"]
trained_rate = trained["accept_rate_all_attempts"]
comparison = {
    "schema_version": "travelweaver-benchmark-comparison-v1",
    "baseline": baseline,
    "trained": trained,
    "trained_minus_baseline": {
        "accepted": trained["accepted"] - baseline["accepted"],
        "accept_rate_all_attempts": (
            trained_rate - baseline_rate
            if trained_rate is not None and baseline_rate is not None
            else None
        ),
    },
}
text = json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
Path(sys.argv[5]).write_text(text, encoding="utf-8")
print(text, end="")
PY
fi

if [[ "${KEEP_SERVER_RUNNING}" == "1" ]]; then
    echo "Benchmark finished; keeping vLLM server PID ${server_pid} running at http://${HOST}:${PORT}."
    echo "Stopping this wrapper later will stop vLLM and restore the GPU 0/1 holder."
    set +e
    wait "${server_pid}"
    server_status=$?
    set -e
    if [[ "${runner_status}" -eq 0 ]]; then
        runner_status=${server_status}
    fi
fi

exit "${runner_status}"
