#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ACTION="${1:-full}"
QUERY_TIME="${QUERY_TIME:-on}"
RUN_TAG="$(date '+%Y%m%d-%H%M%S')"
LOG_DIR="${PROJECT_ROOT}/pine_runs/launcher_logs"
LOG_FILE="${LOG_DIR}/pine_${QUERY_TIME}_six_${RUN_TAG}-$$.nohup.log"
PID_FILE="${LOG_DIR}/pine_${QUERY_TIME}_six_${RUN_TAG}-$$.pid"

mkdir -p "${LOG_DIR}"

nohup bash "${PROJECT_ROOT}/script/run_all.sh" "${ACTION}" \
  >"${LOG_FILE}" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "${pid}" >"${PID_FILE}"

echo "Started six-dataset PINE training."
echo "PID: ${pid}"
echo "Log: ${LOG_FILE}"
echo "PID file: ${PID_FILE}"
echo "Monitor: tail -f ${LOG_FILE}"
