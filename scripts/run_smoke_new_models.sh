#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${TAAC_PROJECT_ROOT:-/data/taac2025}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/venv/bin/python}"
export PYTHONPATH="${PROJECT_ROOT}/code:${PROJECT_ROOT}/code/src:${PYTHONPATH:-}"

"${PYTHON_BIN}" -m taac.training.train \
  --data-root "${PROJECT_ROOT}/raw/TencentGR-1M" \
  --samples-root "${PROJECT_ROOT}/processed/click_samples/v1" \
  --output-dir "${PROJECT_ROOT}/checkpoints/experiments/SMOKE_hstu" \
  --experiment-id SMOKE_hstu \
  --model-type hstu \
  --loss-type info_nce \
  --negative-mode mixed \
  --logq \
  --hard-negative \
  --curriculum \
  --epochs 2 \
  --batch-size 128 \
  --eval-batch-size 128 \
  --hidden-size 32 \
  --num-layers 1 \
  --num-heads 4 \
  --negatives 32 \
  --max-train-samples 2048 \
  --max-valid-samples 1024 \
  --max-valid-batches 4 \
  --num-workers 4
