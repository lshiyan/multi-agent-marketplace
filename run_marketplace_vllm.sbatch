#!/usr/bin/env bash
#SBATCH --job-name=marketplace-vllm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=marketplace-%j.out

set -euo pipefail

# Usage: sbatch run_marketplace_vllm.sbatch MODEL_NAME NUM_RUNS
# Example: sbatch run_marketplace_vllm.sbatch Qwen3-8B 20
# Optional environment: MAX_MODEL_LEN=4096 VLLM_DTYPE=float16 VLLM_PORT=8000
MODEL_NAME="${1:?Usage: sbatch run_marketplace_vllm.sbatch MODEL_NAME NUM_RUNS}"
RUNS="${2:?Usage: sbatch run_marketplace_vllm.sbatch MODEL_NAME NUM_RUNS}"
[[ "$MODEL_NAME" =~ ^[a-zA-Z0-9._-]+$ ]] || { echo 'Model must be a directory name, not a path' >&2; exit 2; }
[[ "$RUNS" =~ ^[1-9][0-9]*$ ]] || { echo 'NUM_RUNS must be a positive integer' >&2; exit 2; }

REPO="$HOME/repo/multi-agent-marketplace"
MODEL_DIR="$HOME/scratch/models/$MODEL_NAME"
PORT="${VLLM_PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
VLLM_DTYPE="${VLLM_DTYPE:-float16}"

cd "$REPO"
source .venv/bin/activate
[[ -d "$MODEL_DIR" ]] || { echo "Missing model directory: $MODEL_DIR" >&2; exit 1; }
command -v vllm >/dev/null || { echo 'vllm is missing from the activated environment' >&2; exit 1; }
command -v magentic-marketplace >/dev/null || { echo 'magentic-marketplace is missing from the activated environment' >&2; exit 1; }

suffix="${MODEL_NAME//[-.]/_}"
experiment_name="baseline_mexican_100_300_${suffix}"
output_path="assets/$experiment_name"
mkdir -p "$output_path" logs
VLLM_LOG="logs/vllm-${SLURM_JOB_ID:-manual}.log"
VLLM_PID=''
POSTGRES_STARTED=0

cleanup() {
    local status=$?
    trap - EXIT TERM INT
    echo "Cleaning up (exit status: $status)"
    if [[ -n "$VLLM_PID" ]]; then
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
    # Only stop PostgreSQL when this job started it. Avoid simultaneous
    # jobs sharing the same PostgreSQL data directory without coordination.
    if (( POSTGRES_STARTED )); then
        bash end_services.sh || true
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

echo "Slurm job: ${SLURM_JOB_ID:-unknown}; node: $(hostname)"
echo "Model: $MODEL_NAME ($MODEL_DIR), runs: $RUNS"

# Local-only endpoint. Simulation runs in this same Slurm job/node.
vllm serve "$MODEL_DIR" \
    --served-model-name "$MODEL_NAME" \
    --dtype "$VLLM_DTYPE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization 0.90 \
    --host 127.0.0.1 \
    --port "$PORT" \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
echo "vLLM PID $VLLM_PID, log $VLLM_LOG"

# Wait until THIS model alias is actually served, not just the port open.
ready=0
for ((i=0; i<600; i++)); do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo 'vLLM exited during startup' >&2
        tail -60 "$VLLM_LOG" >&2
        exit 1
    fi
    if curl --silent --fail --max-time 2 "http://127.0.0.1:${PORT}/v1/models" | \
       MODEL_NAME="$MODEL_NAME" python -c 'import json,os,sys; print(any(m.get("id")==os.environ["MODEL_NAME"] for m in json.load(sys.stdin).get("data",[])))' \
       2>/dev/null | grep -qx True; then
        ready=1
        break
    fi
    sleep 1
done
if (( !ready )); then
    echo 'vLLM did not become ready' >&2
    tail -60 "$VLLM_LOG" >&2
    exit 1
fi
echo 'vLLM ready.'

# Existing project PostgreSQL helper (uses Singularity, port 5433).
if singularity exec postgres_16.sif pg_isready -h localhost -p 5433 >/dev/null 2>&1; then
    echo 'Reusing PostgreSQL already listening on localhost:5433.'
else
    bash start_services.sh
    POSTGRES_STARTED=1
fi

# Point project's existing OpenAI-compatible client to this local server.
# Set AFTER starting services so these values have the desired precedence.
export LLM_PROVIDER=local
export LLM_MODEL="$MODEL_NAME"
export LOCAL_LLM_BASE_URL="http://127.0.0.1:8000/v1"
export LOCAL_LLM_API_KEY="EMPTY"
export LOCAL_LLM_STRUCTURED_MODE=json_schema
printf '%s\n' "$MODEL_NAME" > "$output_path/model.txt"

echo "Starting $experiment_name; output: $output_path"
# Same experiment parameters as run_model_experiment.sh on centralized.
magentic-marketplace run data/mexican_100_300 \
    --experiment-name "$experiment_name" \
    --postgres-host localhost \
    --postgres-port 5433 \
    --num_runs "$RUNS" \
    --customers_per_run 20 \
    --output_path "$output_path" \
    --override \
    > "$output_path/run.log" 2>&1

echo "Completed $experiment_name. Log: $output_path/run.log"
