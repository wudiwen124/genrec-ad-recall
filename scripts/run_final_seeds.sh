#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${TAAC_PROJECT_ROOT:-/data/taac2025}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/venv/bin/python}"
FEATURE_CACHE="${FEATURE_CACHE:-${PROJECT_ROOT}/processed/features/v1}"
export PYTHONPATH="${PROJECT_ROOT}/code:${PROJECT_ROOT}/code/src:${PYTHONPATH:-}"

for seed in 2027 2028; do
  name="FINAL_seed${seed}"
  "${PYTHON_BIN}" -m taac.training.train \
    --data-root "${PROJECT_ROOT}/raw/TencentGR-1M" \
    --samples-root "${PROJECT_ROOT}/processed/click_samples/v1" \
    --output-dir "${PROJECT_ROOT}/checkpoints/experiments/${name}" \
    --experiment-id "${name}" \
    --feature-cache-dir "${FEATURE_CACHE}" \
    --model-type sasrec \
    --loss-type info_nce \
    --negative-mode mixed \
    --logq \
    --hard-negative \
    --use-action \
    --use-time \
    --epochs 2 \
    --batch-size 512 \
    --eval-batch-size 512 \
    --hidden-size 64 \
    --num-layers 2 \
    --num-heads 4 \
    --dropout 0.1 \
    --negatives 128 \
    --max-valid-samples 10000 \
    --early-stop-patience 2 \
    --num-workers 8 \
    --seed "${seed}"
done
