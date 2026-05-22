#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_NAME="${DATADIFF_TMUX_SESSION:-datadiff-live-24h}"

run_child() {
  local run_id="$1"
  local log_file="${ROOT_DIR}/reports/live-24h-${run_id}.log"
  local status_file="${ROOT_DIR}/reports/live-24h-${run_id}.status"

  cd "${ROOT_DIR}"
  mkdir -p reports runs bugs corpus
  printf 'status=running\nrun_id=%s\nstarted_at=%s\nlog_file=%s\n' \
    "${run_id}" "$(date -Is)" "${log_file}" > "${status_file}"

  set +e
  {
    echo "run_id=${run_id}"
    echo "started_at=$(date -Is)"
    echo "root=${ROOT_DIR}"
    echo "session=${SESSION_NAME}"
    echo
    echo "[resource]"
    df -h .
    nproc
    uptime
    free -h
    echo
    echo "[versions]"
    .venv/bin/python - <<'PY'
import importlib
import sqlite3
import sys

print("python", sys.version.split()[0])
print("stdlib_sqlite", sqlite3.sqlite_version)
for name in ["pandas", "polars", "duckdb", "datafusion", "pyarrow", "numpy", "pysqlite3"]:
    try:
        module = importlib.import_module(name)
        print(name, getattr(module, "__version__", "unknown"), getattr(module, "sqlite_version", ""))
    except Exception as exc:
        print(name, "NOT_INSTALLED", type(exc).__name__, exc)
PY
    echo
    echo "[command]"
    printf '%q ' \
      .venv/bin/datadiff experiment \
      --duration 4h \
      --seeds 1,1001,2001 \
      --presets ordered_groupby_sort,topk_resort,join_ordered_agg_topk \
      --target-suites dataframe_lazy,datafusion_cross,arrow_cross,embedded_sql \
      --evidence-mode live \
      --run-theme live-24h-order-risk \
      --paper-notes "36-job live run: 4 suites x 3 presets x 3 seeds; 4h/job with 6-way bounded parallelism for about 24h wall time." \
      --artifact-limit 5 \
      --log-level minimal \
      --jobs 6 \
      --max-parallel-cost 30 \
      --skip-run-reports \
      --skip-paper-journal
    echo
    echo

    .venv/bin/datadiff experiment \
      --duration 4h \
      --seeds 1,1001,2001 \
      --presets ordered_groupby_sort,topk_resort,join_ordered_agg_topk \
      --target-suites dataframe_lazy,datafusion_cross,arrow_cross,embedded_sql \
      --evidence-mode live \
      --run-theme live-24h-order-risk \
      --paper-notes "36-job live run: 4 suites x 3 presets x 3 seeds; 4h/job with 6-way bounded parallelism for about 24h wall time." \
      --artifact-limit 5 \
      --log-level minimal \
      --jobs 6 \
      --max-parallel-cost 30 \
      --skip-run-reports \
      --skip-paper-journal
    cmd_rc=$?
    echo
    echo "finished_at=$(date -Is)"
    echo "datadiff_exit_code=${cmd_rc}"
    exit "${cmd_rc}"
  } 2>&1 | tee -a "${log_file}"

  local rc=${PIPESTATUS[0]}
  set -e
  {
    printf 'status=%s\n' "$([[ "${rc}" == "0" ]] && echo completed || echo failed)"
    printf 'run_id=%s\n' "${run_id}"
    printf 'finished_at=%s\n' "$(date -Is)"
    printf 'exit_code=%s\n' "${rc}"
    printf 'log_file=%s\n' "${log_file}"
  } > "${status_file}"
  exit "${rc}"
}

if [[ "${1:-}" == "--child" ]]; then
  run_child "${2:?missing run id}"
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION_NAME}"
  echo "attach: tmux attach -t ${SESSION_NAME}"
  exit 2
fi

RUN_ID="$(date +%Y%m%dT%H%M%S%z)"
tmux new-session -d -s "${SESSION_NAME}" "${BASH_SOURCE[0]} --child ${RUN_ID}"

echo "started tmux session: ${SESSION_NAME}"
echo "run_id: ${RUN_ID}"
echo "attach: tmux attach -t ${SESSION_NAME}"
echo "log: ${ROOT_DIR}/reports/live-24h-${RUN_ID}.log"
echo "status: ${ROOT_DIR}/reports/live-24h-${RUN_ID}.status"
