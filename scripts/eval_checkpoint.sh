#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${TAAC_PROJECT_ROOT:-/data/taac2025}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/venv/bin/python}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/checkpoints/v1_item_features/best.pt}"
SPLIT="${SPLIT:-test}"
OUTPUT="${OUTPUT:-${PROJECT_ROOT}/logs/eval_${SPLIT}.json}"
FEATURE_CACHE="${FEATURE_CACHE:-${PROJECT_ROOT}/processed/features/v1}"

export PYTHONPATH="${PROJECT_ROOT}/code:${PROJECT_ROOT}/code/src:${PYTHONPATH:-}"

"${PYTHON_BIN}" -m taac.evaluation.evaluate_checkpoint \
  --data-root "${PROJECT_ROOT}/raw/TencentGR-1M" \
  --samples-root "${PROJECT_ROOT}/processed/click_samples/v1" \
  --checkpoint "${CHECKPOINT}" \
  --split "${SPLIT}" \
  --output "${OUTPUT}" \
  --batch-size 512 \
  --filter-history \
  --num-workers 8 \
  --feature-cache-dir "${FEATURE_CACHE}"
