#!/usr/bin/env bash
set -euo pipefail

PREFIXES=(OM OM-NS OM-LM OM-WM)
SUFFIXES=(fen ascii)

for prefix in "${PREFIXES[@]}"; do
    for suffix in "${SUFFIXES[@]}"; do
        model_name="${prefix}-${suffix}"
        task="optimal_move_${suffix}"

        echo "=========================================="
        echo "Running ${model_name} (task=${task})"
        echo "=========================================="

        python 02_generate_predictions.py \
            --model "/workspace/peiyao/models/${model_name}/" \
            --steps ./data/steps.parquet \
            --output "./outputs/${model_name}_predictions.parquet" \
            --task "${task}" \
            --tp 1 \
        && python 03_score_puzzle_eval.py \
            --steps ./data/steps.parquet \
            --predictions "./outputs/${model_name}_predictions.parquet" \
            --output "./outputs/${model_name}_scores.json"
    done
done

