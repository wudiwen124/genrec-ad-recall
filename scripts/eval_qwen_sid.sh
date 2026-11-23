#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${TAAC_PROJECT_ROOT:-/data/taac2025}"
V2_WORKSPACE="${V2_WORKSPACE:-/home/ubuntu/v2_workspace}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/venv/bin/python}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"
BEAM_SIZE="${BEAM_SIZE:-10}"
BATCH_SIZE="${BATCH_SIZE:-4}"
TOP_K="${TOP_K:-100}"
LOGIT_ADJUSTMENT_TAU="${LOGIT_ADJUSTMENT_TAU:-0.0}"
MAX_HISTORY="${MAX_HISTORY:-50}"
OUTPUT="${OUTPUT:-${PROJECT_ROOT}/logs/qwen_sid_eval_${MAX_SAMPLES}_beam${BEAM_SIZE}.json}"
WARMUP_CHECKPOINT="${WARMUP_CHECKPOINT:-${V2_WORKSPACE}/qwen_sid/warmup_v2/warmup_heads.pt}"
ADAPTER_PATH="${ADAPTER_PATH:-${V2_WORKSPACE}/qwen_sid/lora_v2}"
VOCAB_DIR="${VOCAB_DIR:-${V2_WORKSPACE}/qwen_sid/vocab_v2}"

export PYTHONPATH="/home/ubuntu/v2_deps:${PROJECT_ROOT}/code:${PROJECT_ROOT}/code/src:${PYTHONPATH:-}"

"${PYTHON_BIN}" -m taac.sid.evaluate_qwen_sid \
  --model-path "${V2_WORKSPACE}/models/Qwen2.5-0.5B" \
  --adapter-path "${ADAPTER_PATH}" \
  --tokenizer-path "${VOCAB_DIR}" \
  --metadata-path "${VOCAB_DIR}/sid_tokens.json" \
  --warmup-checkpoint "${WARMUP_CHECKPOINT}" \
  --sid-path "${PROJECT_ROOT}/processed/sid/sid_kmeans_v1.npz" \
  --data-root "${PROJECT_ROOT}/raw/TencentGR-1M" \
  --samples-root "${PROJECT_ROOT}/processed/click_samples/v1" \
  --feature-cache-dir "${PROJECT_ROOT}/processed/features/v1" \
  --output "${OUTPUT}" \
  --max-samples "${MAX_SAMPLES}" \
  --beam-size "${BEAM_SIZE}" \
  --batch-size "${BATCH_SIZE}" \
  --top-k "${TOP_K}" \
  --logit-adjustment-tau "${LOGIT_ADJUSTMENT_TAU}" \
  --max-history "${MAX_HISTORY}"
