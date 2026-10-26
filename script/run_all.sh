#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-python}"
ACTION="${1:-full}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
SEED="${SEED:-1024}"
QUERY_TIME="${QUERY_TIME:-on}"
MAX_PARALLEL="${MAX_PARALLEL:-6}"
GPU_IDS_STRING="${GPU_IDS:-0 1 2 3 4 5}"
SHARD_TIME_COUNT="${SHARD_TIME_COUNT:-30}"
CONTEXT_TIMES="${CONTEXT_TIMES:-60}"
MAX_EDGES_PER_SHARD="${MAX_EDGES_PER_SHARD:-5000}"

RAW_ROOT="${PROJECT_ROOT}/dataset"
PROCESSED_ROOT="${PROCESSED_ROOT:-${PROJECT_ROOT}/pine_data/processed}"
RUN_ROOT="${PROJECT_ROOT}/pine_runs"
RUN_TAG="$(date '+%Y%m%d-%H%M%S')"
LOG_DIR="${PROJECT_ROOT}/pine_runs/launcher_logs/pine_${QUERY_TIME}_${RUN_TAG}-$$"

DATASETS=("icews21" "icews22" "ACLED" "ICEWS14" "ICEWS05-15" "GDELT")
SETTINGS=("extrapolation" "extrapolation" "extrapolation" "interpolation" "interpolation" "interpolation")
TIME_RANGES=(365 365 336 365 4017 366)
read -r -a GPU_IDS_ARRAY <<< "${GPU_IDS_STRING}"

if [[ "${ACTION}" != "full" && "${ACTION}" != "preprocess" && "${ACTION}" != "experiment" ]]; then
  echo "Usage: bash script/run_all.sh [full|preprocess|experiment]" >&2
  exit 2
fi
if [[ "${QUERY_TIME}" != "on" && "${QUERY_TIME}" != "off" ]]; then
  echo "QUERY_TIME must be on or off." >&2
  exit 2
fi
if [[ ! "${MAX_PARALLEL}" =~ ^[1-6]$ ]]; then
  echo "MAX_PARALLEL must be an integer from 1 to 6." >&2
  exit 2
fi
if [[ "${#GPU_IDS_ARRAY[@]}" -lt "${MAX_PARALLEL}" ]]; then
  echo "Expected at least ${MAX_PARALLEL} GPU IDs; received: ${GPU_IDS_STRING}" >&2
  exit 2
fi
ACTIVE_GPU_IDS=("${GPU_IDS_ARRAY[@]:0:${MAX_PARALLEL}}")

mkdir -p "${PROCESSED_ROOT}" "${RUN_ROOT}" "${LOG_DIR}"
PROCESSED_ROOT="$(cd "${PROCESSED_ROOT}" && pwd)"
echo "[config] project=${PROJECT_ROOT}"
echo "[config] data=${RAW_ROOT}"
echo "[config] processed=${PROCESSED_ROOT}"
echo "[config] available GPUs=${GPU_IDS_STRING}"
echo "[config] active GPUs=${ACTIVE_GPU_IDS[*]} epochs=${NUM_EPOCHS} seed=${SEED}"
echo "[config] logs=${LOG_DIR}"
echo "[config] query time=${QUERY_TIME} parallel jobs=${MAX_PARALLEL}"

verify_processed() {
  "${PYTHON}" - "${PROJECT_ROOT}" "$1" "$2" "$3" <<'PY'
import sys
import os

sys.path.insert(0, sys.argv[1])
from pine.data import protocol
from pine.cache_validation import validate_processed

protocol.validate_splits(protocol.read_splits(sys.argv[2]))
validate_processed(sys.argv[2], sys.argv[3], sys.argv[4])
print(f"[verified] {sys.argv[3]}")
PY
}

if [[ "${ACTION}" == "full" || "${ACTION}" == "preprocess" ]]; then
  # Preprocessing is CPU/RAM intensive and writes shared manifests, so run it
  # sequentially before launching the six independent GPU jobs.
  for i in "${!DATASETS[@]}"; do
    dataset="${DATASETS[$i]}"
    setting="${SETTINGS[$i]}"
    raw_dir="${RAW_ROOT}/${setting}/${dataset}"
    processed_dir="${PROCESSED_ROOT}/${setting}/${dataset}"
    log_file="${LOG_DIR}/preprocess_${dataset}.log"
    if [[ ! -d "${raw_dir}" ]]; then
      echo "Missing dataset directory: ${raw_dir}" >&2
      exit 1
    fi
    mkdir -p "${processed_dir}"
    echo "[preprocess] ${setting}/${dataset}"
    (cd "${PROJECT_ROOT}" && "${PYTHON}" -m pine.preprocess \
      --dataset "${dataset}" \
      --data-dir "${raw_dir}" \
      --output-dir "${processed_dir}" \
      --setting "${setting}" \
      --shard-time-count "${SHARD_TIME_COUNT}" \
      --context-times "${CONTEXT_TIMES}" \
      --max-edges "${MAX_EDGES_PER_SHARD}" \
      --shard-msg \
      >"${log_file}" 2>&1)
    verify_processed "${raw_dir}" "${processed_dir}" "${setting}"
    echo "[preprocess done] ${dataset}; log=${log_file}"
  done
fi

if [[ "${ACTION}" == "full" || "${ACTION}" == "experiment" ]]; then
  if [[ "${ACTION}" == "experiment" ]]; then
    # Validate every dataset before starting any of the GPU processes.
    for i in "${!DATASETS[@]}"; do
      verify_processed "${RAW_ROOT}/${SETTINGS[$i]}/${DATASETS[$i]}" \
        "${PROCESSED_ROOT}/${SETTINGS[$i]}/${DATASETS[$i]}" "${SETTINGS[$i]}"
    done
  fi
  pids=()
  pending_datasets=()
  failures=0
  wait_batch() {
    local batch_index
    for batch_index in "${!pids[@]}"; do
      if wait "${pids[$batch_index]}"; then
        echo "[done] ${pending_datasets[$batch_index]}"
      else
        echo "[failed] ${pending_datasets[$batch_index]}; see ${LOG_DIR}/train_${pending_datasets[$batch_index]}_gpu${ACTIVE_GPU_IDS[$batch_index]}.log" >&2
        failures=$((failures + 1))
      fi
    done
    pids=()
    pending_datasets=()
  }
  for i in "${!DATASETS[@]}"; do
    dataset="${DATASETS[$i]}"
    setting="${SETTINGS[$i]}"
    time_range="${TIME_RANGES[$i]}"
    gpu="${ACTIVE_GPU_IDS[$((i % MAX_PARALLEL))]}"
    log_file="${LOG_DIR}/train_${dataset}_gpu${gpu}.log"
    launch_dir="${LOG_DIR}/work_${dataset}"
    mkdir -p "${launch_dir}"
    echo "[launch] dataset=${dataset} setting=${setting} gpu=${gpu} epochs=${NUM_EPOCHS}"
    (
      cd "${launch_dir}"
      CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" \
        "${PROJECT_ROOT}/script/train.py" \
        --config "${PROJECT_ROOT}/config/pine.yaml" \
        --dataset "${dataset}" \
        --setting "${setting}" \
        --time-range "${time_range}" \
        --query-time "${QUERY_TIME}" \
        --seed "${SEED}" \
        --num-epoch "${NUM_EPOCHS}" \
        --processed-root "${PROCESSED_ROOT}"
    ) >"${log_file}" 2>&1 &
    pids+=("$!")
    pending_datasets+=("${dataset}")
    if [[ "${#pids[@]}" -eq "${MAX_PARALLEL}" ]]; then wait_batch; fi
  done

  if [[ "${#pids[@]}" -gt 0 ]]; then wait_batch; fi
  if [[ "${failures}" -ne 0 ]]; then
    echo "${failures} of 6 jobs failed" >&2
    exit 1
  fi

  "${PYTHON}" "${PROJECT_ROOT}/script/collect_results.py" \
    --run-root "${RUN_ROOT}" \
    --output "${RUN_ROOT}/six_dataset_results.csv"
  echo "[results] ${RUN_ROOT}/six_dataset_results.csv"
fi
