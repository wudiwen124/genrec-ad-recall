#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${TAAC_PROJECT_ROOT:-/data/taac2025}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/venv/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/sid_generator_v1}"
SEED="${SEED:-2026}"
EPOCHS="${EPOCHS:-3}"
export PYTHONPATH="${PROJECT_ROOT}/code:${PROJECT_ROOT}/code/src:${PYTHONPATH:-}"

"${PYTHON_BIN}" -m taac.sid.train_sid \
  --data-root "${PROJECT_ROOT}/raw/TencentGR-1M" \
  --samples-root "${PROJECT_ROOT}/processed/click_samples/v1" \
  --sid-path "${PROJECT_ROOT}/processed/sid/sid_kmeans_v1.npz" \
  --feature-cache-dir "${PROJECT_ROOT}/processed/features/v1" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs "${EPOCHS}" \
  --batch-size 512 \
  --eval-batch-size 512 \
  --hidden-size 64 \
  --num-layers 2 \
  --num-heads 4 \
  --lr 1e-3 \
  --seed "${SEED}" \
  --max-valid-samples 10000 \
  --num-workers 8
