#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${TAAC_PROJECT_ROOT:-/data/taac2025}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/venv/bin/python}"
export PYTHONPATH="${PROJECT_ROOT}/code:${PROJECT_ROOT}/code/src:${PYTHONPATH:-}"

"${PYTHON_BIN}" -m taac.training.train \
  --data-root "${PROJECT_ROOT}/raw/TencentGR-1M" \
  --samples-root "${PROJECT_ROOT}/processed/click_samples/v1" \
  --output-dir "${PROJECT_ROOT}/checkpoints/v1_item_features" \
  --epochs 5 \
  --batch-size 512 \
  --eval-batch-size 128 \
  --hidden-size 64 \
  --num-layers 2 \
  --num-heads 4 \
  --negatives 128 \
  --max-valid-samples 10000 \
  --early-stop-patience 2
