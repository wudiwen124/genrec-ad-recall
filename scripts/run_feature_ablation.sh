#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${TAAC_PROJECT_ROOT:-/data/taac2025}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/venv/bin/python}"
export PYTHONPATH="${PROJECT_ROOT}/code:${PROJECT_ROOT}/code/src:${PYTHONPATH:-}"

run_one() {
  local name="$1"
  shift
  "${PYTHON_BIN}" -m taac.training.train \
    --data-root "${PROJECT_ROOT}/raw/TencentGR-1M" \
    --samples-root "${PROJECT_ROOT}/processed/click_samples/v1" \
    --output-dir "${PROJECT_ROOT}/checkpoints/experiments/${name}" \
    --experiment-id "${name}" \
    --model-type hstu \
    --loss-type info_nce \
    --negative-mode uniform \
    --epochs 4 \
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
    "$@"
}

run_one M0_hstu_base --no-use-action --no-use-time
run_one M1_hstu_action --use-action --no-use-time
run_one M2_hstu_time --no-use-action --use-time
run_one M3_hstu_action_time --use-action --use-time
