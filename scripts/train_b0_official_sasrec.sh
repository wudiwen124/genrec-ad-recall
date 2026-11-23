#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${TAAC_PROJECT_ROOT:-/data/taac2025}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/venv/bin/python}"
export PYTHONPATH="${PROJECT_ROOT}/code:${PROJECT_ROOT}/code/src:${PYTHONPATH:-}"

"${PYTHON_BIN}" -m taac.training.train \
  --data-root "${PROJECT_ROOT}/raw/TencentGR-1M" \
  --samples-root "${PROJECT_ROOT}/processed/click_samples/v1" \
  --output-dir "${PROJECT_ROOT}/checkpoints/experiments/B0_official_sasrec" \
  --experiment-id B0_official_sasrec \
  --model-type sasrec \
  --loss-type bce \
  --negative-mode uniform \
  --epochs 3 \
  --batch-size 512 \
  --eval-batch-size 512 \
  --hidden-size 32 \
  --num-layers 1 \
  --num-heads 1 \
  --dropout 0.2 \
  --negatives 1 \
  --no-use-action \
  --no-use-time \
  --no-norm-first \
  --max-valid-samples 10000 \
  --early-stop-patience 2 \
  --num-workers 8
