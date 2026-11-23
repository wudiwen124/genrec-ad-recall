#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${TAAC_PROJECT_ROOT:-/data/taac2025}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/venv/bin/python}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"
BEAM_SIZE="${BEAM_SIZE:-10}"
CANDIDATE_K="${CANDIDATE_K:-10}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DECODE_MODE="${DECODE_MODE:-constrained}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SID_CHECKPOINT="${SID_CHECKPOINT:-${PROJECT_ROOT}/checkpoints/sid_generator_v1/best.pt}"
OUTPUT="${OUTPUT:-${PROJECT_ROOT}/logs/sid_eval_${DECODE_MODE}_${MAX_SAMPLES}.json}"
USER_RESULT_ARGS=()
if [[ -n "${SAVE_USER_RESULTS:-}" ]]; then
  USER_RESULT_ARGS=(--save-user-results "${SAVE_USER_RESULTS}")
fi

export PYTHONPATH="${PROJECT_ROOT}/code:${PROJECT_ROOT}/code/src:${PYTHONPATH:-}"

"${PYTHON_BIN}" -m taac.sid.evaluate_sid \
  --data-root "${PROJECT_ROOT}/raw/TencentGR-1M" \
  --samples-root "${PROJECT_ROOT}/processed/click_samples/v1" \
  --vector-checkpoint "${PROJECT_ROOT}/checkpoints/experiments/N4_mixed_logq_hard/best.pt" \
  --sid-checkpoint "${SID_CHECKPOINT}" \
  --sid-path "${PROJECT_ROOT}/processed/sid/sid_kmeans_v1.npz" \
  --feature-cache-dir "${PROJECT_ROOT}/processed/features/v1" \
  --output "${OUTPUT}" \
  --max-samples "${MAX_SAMPLES}" \
  --batch-size "${BATCH_SIZE}" \
  --beam-size "${BEAM_SIZE}" \
  --candidate-k "${CANDIDATE_K}" \
  --decode-mode "${DECODE_MODE}" \
  "${USER_RESULT_ARGS[@]}" \
  --num-workers "${NUM_WORKERS}"
