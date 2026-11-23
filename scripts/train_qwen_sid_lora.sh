#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${TAAC_PROJECT_ROOT:-/data/taac2025}"
V2_WORKSPACE="${V2_WORKSPACE:-/home/ubuntu/v2_workspace}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/venv/bin/python}"
MAX_SAMPLES="${MAX_SAMPLES:-50000}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
MAX_HISTORY="${MAX_HISTORY:-100}"
MAX_LENGTH="${MAX_LENGTH:-800}"
SEED="${SEED:-2026}"
SAMPLING_ALPHA="${SAMPLING_ALPHA:-0.0}"
FREEZE_HEAD_ARGS=()
if [[ "${FREEZE_HEADS:-0}" == "1" ]]; then
  FREEZE_HEAD_ARGS=(--freeze-heads)
fi
SID_PATH="${SID_PATH:-${PROJECT_ROOT}/processed/sid/sid_rqvae_v1.npz}"
TOKENIZER_DIR="${TOKENIZER_DIR:-${V2_WORKSPACE}/qwen_sid/vocab_v2}"
OUTPUT_DIR="${OUTPUT_DIR:-${V2_WORKSPACE}/qwen_sid/lora_v2}"
WARMUP_CHECKPOINT="${WARMUP_CHECKPOINT:-${V2_WORKSPACE}/qwen_sid/warmup_v2/warmup_heads.pt}"

export PYTHONPATH="/home/ubuntu/v2_deps:${PROJECT_ROOT}/code:${PROJECT_ROOT}/code/src:${PYTHONPATH:-}"

"${PYTHON_BIN}" -m taac.sid.qwen_sft \
  --mode lora \
  --model-path "${V2_WORKSPACE}/models/Qwen2.5-0.5B" \
  --tokenizer-path "${TOKENIZER_DIR}" \
  --sid-path "${SID_PATH}" \
  --samples-root "${PROJECT_ROOT}/processed/click_samples/v1" \
  --data-root "${PROJECT_ROOT}/raw/TencentGR-1M" \
  --feature-cache-dir "${PROJECT_ROOT}/processed/features/v1" \
  --output-dir "${OUTPUT_DIR}" \
  --warmup-checkpoint "${WARMUP_CHECKPOINT}" \
  --max-samples "${MAX_SAMPLES}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --gradient-accumulation "${GRAD_ACCUM}" \
  --max-history "${MAX_HISTORY}" \
  --max-length "${MAX_LENGTH}" \
  --num-workers 4 \
  "${FREEZE_HEAD_ARGS[@]}" \
  --sampling-alpha "${SAMPLING_ALPHA}" \
  --seed "${SEED}"
