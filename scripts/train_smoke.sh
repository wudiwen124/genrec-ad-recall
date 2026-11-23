#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${TAAC_PROJECT_ROOT:-/data/taac2025}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/venv/bin/python}"
export PYTHONPATH="${PROJECT_ROOT}/code:${PROJECT_ROOT}/code/src:${PYTHONPATH:-}"

"${PYTHON_BIN}" -m taac.training.train \
  --data-root "${PROJECT_ROOT}/raw/TencentGR-1M" \
  --samples-root "${PROJECT_ROOT}/processed/click_samples/v1" \
  --output-dir "${PROJECT_ROOT}/checkpoints/smoke_id_only" \
  --epochs 1 \
  --batch-size 128 \
  --eval-batch-size 128 \
  --hidden-size 64 \
  --num-layers 1 \
  --num-heads 4 \
  --negatives 64 \
  --max-train-samples 4096 \
  --max-valid-samples 2048 \
  --max-valid-batches 8
