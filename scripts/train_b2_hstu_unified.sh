#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${TAAC_PROJECT_ROOT:-/data/taac2025}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/venv/bin/python}"
export PYTHONPATH="${PROJECT_ROOT}/code:${PROJECT_ROOT}/code/src:${PYTHONPATH:-}"

"${PYTHON_BIN}" -m taac.training.train \
  --data-root "${PROJECT_ROOT}/raw/TencentGR-1M" \
  --samples-root "${PROJECT_ROOT}/processed/click_samples/v1" \
  --output-dir "${PROJECT_ROOT}/checkpoints/experiments/B2_hstu_unified" \
  --experiment-id B2_hstu_unified \
  --model-type hstu \
  --loss-type info_nce \
  --negative-mode uniform \
  --epochs 5 \
  --batch-size 512 \
  --eval-batch-size 512 \
  --hidden-size 64 \
  --num-layers 2 \
  --num-heads 4 \
  --dropout 0.1 \
  --negatives 128 \
  --use-action \
  --use-time \
  --use-relative-bias \
  --max-valid-samples 10000 \
  --early-stop-patience 2 \
  --num-workers 8
