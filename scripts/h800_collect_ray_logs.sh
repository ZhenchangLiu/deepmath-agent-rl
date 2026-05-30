#!/usr/bin/env bash
# Collect the useful tail of Ray/vLLM logs after a failed H800 smoke run.

set -euo pipefail

LOG_DIR=${LOG_DIR:-/tmp/ray/session_latest/logs}
LINES=${LINES:-240}
PID_FILTER=${PID_FILTER:-}

if [[ ! -d "${LOG_DIR}" ]]; then
    echo "Ray log directory not found: ${LOG_DIR}" >&2
    exit 1
fi

echo "== Latest Ray log files =="
find "${LOG_DIR}" -maxdepth 1 -type f -printf '%T@ %p\n' \
    | sort -nr \
    | head -40 \
    | cut -d' ' -f2-

echo
echo "== Candidate worker logs =="
if [[ -n "${PID_FILTER}" ]]; then
    find "${LOG_DIR}" -maxdepth 1 -type f \
        \( -name "*${PID_FILTER}*" -o -name "worker-*.out" -o -name "worker-*.err" -o -name "python-core-worker-*.log" \) \
        -printf '%T@ %p\n'
else
    find "${LOG_DIR}" -maxdepth 1 -type f \
        \( -name "worker-*.out" -o -name "worker-*.err" -o -name "python-core-worker-*.log" \) \
        -printf '%T@ %p\n'
fi \
    | sort -nr \
    | head -30 \
    | cut -d' ' -f2-

echo
echo "== vLLM / engine / CUDA / traceback matches =="
if [[ -n "${PID_FILTER}" ]]; then
    grep -R -n -i \
        -E "traceback|error|exception|engine core|enginecore|failed|cuda|nccl|out of memory|no available memory|vllm|asyncllm|${PID_FILTER}" \
        "${LOG_DIR}" \
        | tail -"${LINES}" || true
else
    grep -R -n -i \
        -E "traceback|error|exception|engine core|enginecore|failed|cuda|nccl|out of memory|no available memory|vllm|asyncllm" \
        "${LOG_DIR}" \
        | tail -"${LINES}" || true
fi

echo
echo "== vLLM worker stdout/stderr tails =="
while IFS= read -r file; do
    echo
    echo "--- ${file} ---"
    tail -"${LINES}" "${file}" || true
done < <(
    if [[ -n "${PID_FILTER}" ]]; then
        find "${LOG_DIR}" -maxdepth 1 -type f \
            \( -name "*${PID_FILTER}*" -o -name 'worker-*.out' -o -name 'worker-*.err' -o -name 'python-core-worker-*.log' \) \
            -printf '%T@ %p\n'
    else
        find "${LOG_DIR}" -maxdepth 1 -type f \
            \( -name 'worker-*.out' -o -name 'worker-*.err' -o -name 'python-core-worker-*.log' \) \
            -printf '%T@ %p\n'
    fi \
        | sort -nr \
        | head -16 \
        | cut -d' ' -f2-
)
