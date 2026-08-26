#!/bin/bash

set -e

CONFIGS=(
    "openai:gpt-5-nano:gpt5nano"
    "openai:gpt-5-mini:gpt5mini"
)

for config in "${CONFIGS[@]}"; do
    IFS=":" read -r provider model suffix <<< "$config"

    experiment_name="mexican_10_30_${suffix}"
    output_path="assets/baseline_${suffix}"

    echo "Starting $provider / $model"

    (
        export LLM_PROVIDER="$provider"
        export LLM_MODEL="$model"

        magentic-marketplace run data/mexican_10_30 \
            --experiment-name "$experiment_name" \
            --override \
            --postgres-host localhost \
            --postgres-port 5433 \
            --num_runs 10 \
            --customers_per_run 10 \
            --output_path "$output_path"
    ) > "assets/${suffix}.log" 2>&1 &
done

wait

echo "All experiments finished."