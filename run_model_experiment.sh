
#!/usr/bin/env bash
set -euo pipefail

provider="$1"
model="$2"
runs="$3"

suffix="${model//-/_}"
experiment_name="baseline_mexican_100_300_${suffix}"
output_path="assets/${experiment_name}"

mkdir -p "$output_path"

export LLM_PROVIDER="$provider"
export LLM_MODEL="$model"

# Shared coordination directory.
STATE_DIR="$HOME/.marketplace_experiments"
mkdir -p "$STATE_DIR"

LOCK_FILE="$STATE_DIR/lock"
ACTIVE_FILE="$STATE_DIR/active"

# Register this experiment.
{
    flock -x 200

    touch "$ACTIVE_FILE"
    echo "$$" >> "$ACTIVE_FILE"

} 200>"$LOCK_FILE"

cleanup() {
    local should_stop=0

    {
        flock -x 200

        # Remove this experiment from the active list.
        if [[ -f "$ACTIVE_FILE" ]]; then
            grep -vx "$$" "$ACTIVE_FILE" > "$ACTIVE_FILE.tmp" || true
            mv "$ACTIVE_FILE.tmp" "$ACTIVE_FILE"
        fi

        # If no experiments remain, stop PostgreSQL.
        if [[ ! -s "$ACTIVE_FILE" ]]; then
            should_stop=1
            rm -f "$ACTIVE_FILE"
        fi

        if (( should_stop )); then
            echo "Last experiment finished. Stopping PostgreSQL..."
            bash end_services.sh
            echo "PostgreSQL stopped."
        fi

    } 200>"$LOCK_FILE"
}

trap cleanup EXIT

nohup magentic-marketplace run data/mexican_100_300 \
    --experiment-name "$experiment_name" \
    --postgres-host localhost \
    --postgres-port 5433 \
    --num_runs "$runs" \
    --customers_per_run 20 \
    --output_path "$output_path" \
    --override \
    > "$output_path/run.log" 2>&1 &

pid=$!

echo "Started experiment: $experiment_name"
echo "Provider: $provider | Model: $model"
echo "PID: $pid"
echo "Log: $output_path/run.log"
echo "Monitor: tail -f $output_path/run.log"

wait "$pid"
echo "Experiment completed."
