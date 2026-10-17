#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: bash $0 <provider> <model>" >&2
    echo "Example: bash $0 openai gpt-4o-mini" >&2
    exit 1
fi

provider="$1"
model="$2"

# Keep experiment names and paths safe for the filesystem.
suffix="${model//[^a-zA-Z0-9_-]/_}"
experiment_name="baseline_mexican_100_300_${suffix}"
output_path="assets/${experiment_name}"

mkdir -p "$output_path"

export LLM_PROVIDER="$provider"
export LLM_MODEL="$model"

nohup magentic-marketplace run data/mexican_100_300 \
    --experiment-name "$experiment_name" \
    --override \
    --postgres-host localhost \
    --postgres-port 5433 \
    --num_runs 20 \
    --customers_per_run 20 \
    --output_path "$output_path" \
    > "$output_path/run.log" 2>&1 &

pid=$!
echo "Started experiment: $experiment_name"
echo "Provider: $provider | Model: $model"
echo "PID: $pid"
echo "Log: $output_path/run.log"
echo "Monitor: tail -f $output_path/run.log"
