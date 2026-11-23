#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${TAAC_PROJECT_ROOT:-/data/taac2025}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/venv/bin/python}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/raw/TencentGR-1M}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/processed/click_samples/v1}"

export PYTHONPATH="${PROJECT_ROOT}/code:${PROJECT_ROOT}/code/src:${PYTHONPATH:-}"

"${PYTHON_BIN}" -m taac.data.click_samples \
  --data-root "${DATA_ROOT}" \
  --output "${OUTPUT_DIR}" \
  --max-history 100
