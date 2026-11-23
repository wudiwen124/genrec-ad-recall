#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${TAAC_PROJECT_ROOT:-/data/taac2025}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/venv/bin/python}"
export PYTHONPATH="${PROJECT_ROOT}/code:${PROJECT_ROOT}/code/src:${PYTHONPATH:-}"

"${PYTHON_BIN}" -m taac.sid.tokenizer \
  --data-root "${PROJECT_ROOT}/raw/TencentGR-1M" \
  --checkpoint "${PROJECT_ROOT}/checkpoints/experiments/N4_mixed_logq_hard/best.pt" \
  --feature-cache-dir "${PROJECT_ROOT}/processed/features/v1" \
  --output "${PROJECT_ROOT}/processed/sid/sid_kmeans_v1.npz" \
  --levels 4 \
  --clusters 256 \
  --fit-samples 200000
