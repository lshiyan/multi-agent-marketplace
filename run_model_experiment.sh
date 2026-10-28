#!/usr/bin/env bash
set -euo pipefail

provider="$1"
model="$2"
runs="$3"

# Keep experiment names and paths safe for the filesystem.
suffix="${model//-/_}"
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
    --num_runs "$runs" \
    --customers_per_run 20 \
    --output_path "$output_path" \
    > "$output_path/run.log" 2>&1 &

pid=$!
echo "Started experiment: $experiment_name"
echo "Provider: $provider | Model: $model"
echo "PID: $pid"
echo "Log: $output_path/run.log"
echo "Monitor: tail -f $output_path/run.log"
