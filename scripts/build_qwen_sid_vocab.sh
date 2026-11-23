#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${TAAC_PROJECT_ROOT:-/data/taac2025}"
V2_WORKSPACE="${V2_WORKSPACE:-/home/ubuntu/v2_workspace}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-${V2_WORKSPACE}/models/Qwen2.5-0.5B}"
OUTPUT_DIR="${OUTPUT_DIR:-${V2_WORKSPACE}/qwen_sid/vocab_v1}"
SID_PATH="${SID_PATH:-${PROJECT_ROOT}/processed/sid/sid_kmeans_v1.npz}"

export PYTHONPATH="/home/ubuntu/v2_deps:${PROJECT_ROOT}/code:${PROJECT_ROOT}/code/src:${PYTHONPATH:-}"

"${PYTHON_BIN}" -m taac.sid.qwen_tokens \
  --model-path "${MODEL_PATH}" \
  --sid-path "${SID_PATH}" \
  --data-root "${PROJECT_ROOT}/raw/TencentGR-1M" \
  --output-dir "${OUTPUT_DIR}"
