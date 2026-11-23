#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${TAAC_PROJECT_ROOT:-/data/taac2025}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/venv/bin/python}"
export PYTHONPATH="${PROJECT_ROOT}/code:${PROJECT_ROOT}/code/src:${PYTHONPATH:-}"
FEATURE_CACHE="${FEATURE_CACHE:-${PROJECT_ROOT}/processed/features/v1}"

run_one() {
  local name="$1"
  shift
  "${PYTHON_BIN}" -m taac.training.train \
    --data-root "${PROJECT_ROOT}/raw/TencentGR-1M" \
    --samples-root "${PROJECT_ROOT}/processed/click_samples/v1" \
    --output-dir "${PROJECT_ROOT}/checkpoints/experiments/${name}" \
    --experiment-id "${name}" \
    --feature-cache-dir "${FEATURE_CACHE}" \
    --model-type sasrec \
    --loss-type info_nce \
    --epochs 4 \
    --batch-size 512 \
    --eval-batch-size 512 \
    --hidden-size 64 \
    --num-layers 2 \
    --num-heads 4 \
    --dropout 0.1 \
    --negatives 128 \
    --use-action \
    --use-time \
    --max-valid-samples 10000 \
    --early-stop-patience 2 \
    --num-workers 8 \
    "$@"
}

run_one N0_uniform --negative-mode uniform
run_one N1_inbatch --negative-mode inbatch
run_one N2_mixed --negative-mode mixed
run_one N3_mixed_logq --negative-mode mixed --logq
run_one N4_mixed_logq_hard --negative-mode mixed --logq --hard-negative
run_one N5_full_curriculum --negative-mode mixed --logq --hard-negative --curriculum
