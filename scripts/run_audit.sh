#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${TAAC_PROJECT_ROOT:-/data/taac2025}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/venv/bin/python}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/raw/TencentGR-1M}"
OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_ROOT}/processed/audit/tencentgr_1m_audit.json}"

export PYTHONPATH="${PROJECT_ROOT}/code:${PROJECT_ROOT}/code/src:${PYTHONPATH:-}"

mkdir -p "$(dirname "${OUTPUT_PATH}")"

"${PYTHON_BIN}" -m taac.data.audit \
  --data-root "${DATA_ROOT}" \
  --output "${OUTPUT_PATH}"
