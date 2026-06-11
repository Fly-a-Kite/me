#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${DATADIFF_ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ID="${DATADIFF_FINAL_LIVE_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
DURATION="${DATADIFF_FINAL_LIVE_DURATION:-24h}"
BATCH_DURATION="${DATADIFF_FINAL_LIVE_BATCH_DURATION:-10m}"
SEEDS="${DATADIFF_FINAL_LIVE_SEEDS:-1,1001,2001}"
JOBS="${DATADIFF_FINAL_LIVE_JOBS:-1}"
MAX_PARALLEL_COST="${DATADIFF_FINAL_LIVE_MAX_PARALLEL_COST:-}"
ARTIFACT_LIMIT="${DATADIFF_FINAL_LIVE_ARTIFACT_LIMIT:-50}"
LOG_LEVEL="${DATADIFF_FINAL_LIVE_LOG_LEVEL:-minimal}"
STAGGER_SECONDS="${DATADIFF_FINAL_LIVE_STAGGER_SECONDS:-10}"
SESSION_PREFIX="${DATADIFF_FINAL_LIVE_SESSION_PREFIX:-datadiff-final-live}"
LOG_DIR="${DATADIFF_FINAL_LIVE_LOG_DIR:-logs/final-live}"
INDEX_DIR="${DATADIFF_FINAL_LIVE_INDEX_DIR:-reports/final-live-indexes}"
PROVENANCE_DIR="${DATADIFF_FINAL_LIVE_PROVENANCE_DIR:-reports/final-live-provenance}"
STRATEGY_SNAPSHOT="${DATADIFF_FINAL_LIVE_STRATEGY_SNAPSHOT:-reports/strategy-snapshots/final-frozen-strategy-snapshot.json}"
CONTINUAL_LEARNING_LEDGERS="${DATADIFF_FINAL_LIVE_CONTINUAL_LEARNING_LEDGERS:-}"
REQUIRE_CLEAN_WORKTREE="${DATADIFF_FINAL_LIVE_REQUIRE_CLEAN_WORKTREE:-1}"
PYTHON_BIN="${DATADIFF_PYTHON:-${ROOT_DIR}/.venv/bin/python}"
DRY_RUN=0
PRINT_CONFIG=0

DEFAULT_CAMPAIGNS=(
  "datafusion_cross:live_datafusion"
  "datafusion_cross:live_datafusion_fresh"
  "dataframe_lazy:live_polars_lazy"
  "arrow_cross:live_arrow"
  "embedded_sql:live_embedded_sql"
  "latest_all_engines:live_cross_family"
  "latest_no_datafusion:live_cross_family"
  "polars_cross:live_polars_issue_focus"
  "embedded_sql_cross:live_duckdb_issue_focus"
  "arrow_cross:live_arrow_issue_focus"
  "latest_no_datafusion:live_issue_focus"
)

usage() {
  cat <<'EOF'
usage: start_final_live_campaigns_tmux.sh [--dry-run] [--print-config]

Launches the final ICSE live-discovery matrix as one tmux session per
target_suite:preset campaign. Each campaign writes an independent manifest index
so 24h live evidence can be imported into the main final index after completion.

Key environment variables:
  DATADIFF_FINAL_LIVE_CAMPAIGNS=<comma or space separated suite:preset list>
  DATADIFF_FINAL_LIVE_DURATION=24h
  DATADIFF_FINAL_LIVE_JOBS=1
  DATADIFF_FINAL_LIVE_SESSION_PREFIX=datadiff-final-live
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --print-config) PRINT_CONFIG=1 ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

quote() {
  printf '%q' "$1"
}

campaign_slug() {
  local slug="$1"
  slug="${slug//:/-}"
  slug="${slug//_/-}"
  printf '%s\n' "${slug}"
}

selected_campaigns() {
  local raw="${DATADIFF_FINAL_LIVE_CAMPAIGNS:-${DEFAULT_CAMPAIGNS[*]}}"
  raw="${raw//,/ }"
  # shellcheck disable=SC2086
  printf '%s\n' ${raw}
}

print_config() {
  printf 'root=%s\n' "${ROOT_DIR}"
  printf 'run_id=%s\n' "${RUN_ID}"
  printf 'duration=%s\n' "${DURATION}"
  printf 'batch_duration=%s\n' "${BATCH_DURATION}"
  printf 'seeds=%s\n' "${SEEDS}"
  printf 'jobs=%s\n' "${JOBS}"
  printf 'max_parallel_cost=%s\n' "${MAX_PARALLEL_COST}"
  printf 'session_prefix=%s\n' "${SESSION_PREFIX}"
  printf 'log_dir=%s\n' "${LOG_DIR}"
  printf 'index_dir=%s\n' "${INDEX_DIR}"
  printf 'provenance_dir=%s\n' "${PROVENANCE_DIR}"
  printf 'strategy_snapshot=%s\n' "${STRATEGY_SNAPSHOT}"
  printf 'campaigns=%s\n' "$(selected_campaigns | paste -sd, -)"
}

if [[ "${PRINT_CONFIG}" == "1" ]]; then
  print_config
  exit 0
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "missing Python executable: ${PYTHON_BIN}" >&2
  exit 2
fi

if [[ ! -f "${ROOT_DIR}/${STRATEGY_SNAPSHOT}" && ! -f "${STRATEGY_SNAPSHOT}" ]]; then
  echo "missing strategy snapshot: ${STRATEGY_SNAPSHOT}" >&2
  exit 2
fi

if [[ "${REQUIRE_CLEAN_WORKTREE}" == "1" ]]; then
  if [[ -n "$(git -C "${ROOT_DIR}" status --porcelain)" ]]; then
    echo "refusing final live launch from a dirty workspace" >&2
    git -C "${ROOT_DIR}" status --short >&2
    exit 2
  fi
fi

mkdir -p "${ROOT_DIR}/${LOG_DIR}" "${ROOT_DIR}/${INDEX_DIR}" "${ROOT_DIR}/${PROVENANCE_DIR}"

FREEZE_MANIFEST="${ROOT_DIR}/${PROVENANCE_DIR}/final-live-${RUN_ID}-freeze-manifest.json"
PIP_FREEZE="${ROOT_DIR}/${PROVENANCE_DIR}/final-live-${RUN_ID}-pip-freeze.txt"
GIT_STATUS="${ROOT_DIR}/${PROVENANCE_DIR}/final-live-${RUN_ID}-git-status.txt"
GIT_DIFF="${ROOT_DIR}/${PROVENANCE_DIR}/final-live-${RUN_ID}-git-diff.txt"
LAUNCH_ENV="${ROOT_DIR}/${PROVENANCE_DIR}/final-live-${RUN_ID}-launcher-env.txt"
CAMPAIGN_TABLE="${ROOT_DIR}/${PROVENANCE_DIR}/final-live-${RUN_ID}-campaigns.tsv"
CAMPAIGNS_TEXT="$(selected_campaigns)"

"${PYTHON_BIN}" -m pip freeze > "${PIP_FREEZE}"
git -C "${ROOT_DIR}" status --short --branch > "${GIT_STATUS}"
{
  git -C "${ROOT_DIR}" diff --stat
  git -C "${ROOT_DIR}" diff
} > "${GIT_DIFF}"

{
  print_config
  printf 'freeze_manifest=%s\n' "${FREEZE_MANIFEST}"
  printf 'pip_freeze=%s\n' "${PIP_FREEZE}"
  printf 'git_status=%s\n' "${GIT_STATUS}"
  printf 'git_diff=%s\n' "${GIT_DIFF}"
  printf 'launcher_env=%s\n' "${LAUNCH_ENV}"
} > "${LAUNCH_ENV}"

"${PYTHON_BIN}" - \
  "${FREEZE_MANIFEST}" \
  "${ROOT_DIR}" \
  "${RUN_ID}" \
  "${DURATION}" \
  "${BATCH_DURATION}" \
  "${SEEDS}" \
  "${JOBS}" \
  "${CAMPAIGNS_TEXT}" \
  "${STRATEGY_SNAPSHOT}" \
  "${PIP_FREEZE}" \
  "${GIT_STATUS}" \
  "${GIT_DIFF}" \
  "${LAUNCH_ENV}" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

out = Path(sys.argv[1])
root = Path(sys.argv[2])
payload = {
    "schema_version": "final-live-freeze-manifest-v1",
    "run_id": sys.argv[3],
    "root": str(root),
    "git_commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
    "duration": sys.argv[4],
    "batch_duration": sys.argv[5],
    "seeds": sys.argv[6],
    "jobs": sys.argv[7],
    "campaigns": [line.strip() for line in sys.argv[8].splitlines() if line.strip()],
    "strategy_snapshot": str(Path(sys.argv[9])),
    "pip_freeze": sys.argv[10],
    "git_status": sys.argv[11],
    "git_diff": sys.argv[12],
    "launcher_env": sys.argv[13],
}
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

printf 'campaign\tsession\tindex\tlog\tscript\n' > "${CAMPAIGN_TABLE}"

while IFS= read -r campaign; do
  [[ -n "${campaign}" ]] || continue
  slug="$(campaign_slug "${campaign}")"
  session="${SESSION_PREFIX}-${RUN_ID}-${slug}"
  index_file="${ROOT_DIR}/${INDEX_DIR}/final-live-${RUN_ID}-${slug}.json"
  log_file="${ROOT_DIR}/${LOG_DIR}/final-live-${RUN_ID}-${slug}.log"
  script_file="${ROOT_DIR}/${PROVENANCE_DIR}/final-live-${RUN_ID}-${slug}.sh"
  cat > "${script_file}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
set -o pipefail
cd $(quote "${ROOT_DIR}")
export DATADIFF_ROOT_DIR=$(quote "${ROOT_DIR}")
export DATADIFF_RUN_PROVENANCE_AUTHORITY=1
export DATADIFF_RUN_PROVENANCE_FREEZE_INTENT=1
export DATADIFF_RUN_PROVENANCE_LATEST_CODE_CLAIM=1
export DATADIFF_RUN_PROVENANCE_EVIDENCE_ROLE=latest_live_authority_24h
export DATADIFF_RUN_PROVENANCE_LAUNCH_SOURCE=final_live_campaign_tmux
export DATADIFF_RUN_PROVENANCE_SESSION=$(quote "${session}")
export DATADIFF_RUN_PROVENANCE_DURATION=$(quote "${DURATION}")
export DATADIFF_RUN_PROVENANCE_BATCH_DURATION=$(quote "${BATCH_DURATION}")
export DATADIFF_RUN_PROVENANCE_LOG_PREFIX=$(quote "${log_file}")
export DATADIFF_RUN_PROVENANCE_LAUNCH_SCRIPT=$(quote "$(basename "${BASH_SOURCE[0]}")")
export DATADIFF_RUN_PROVENANCE_FREEZE_MANIFEST=$(quote "${FREEZE_MANIFEST}")
export DATADIFF_RUN_PROVENANCE_PIP_FREEZE=$(quote "${PIP_FREEZE}")
export DATADIFF_RUN_PROVENANCE_GIT_STATUS=$(quote "${GIT_STATUS}")
export DATADIFF_RUN_PROVENANCE_GIT_DIFF=$(quote "${GIT_DIFF}")
export DATADIFF_RUN_PROVENANCE_LAUNCH_ENV=$(quote "${LAUNCH_ENV}")
export DATADIFF_RUN_PROVENANCE_STRATEGY_SNAPSHOT=$(quote "${ROOT_DIR}/${STRATEGY_SNAPSHOT}")
COMMAND=(
  $(quote "${PYTHON_BIN}") scripts/run_final_experiments.py
  --track live
  --live-campaign $(quote "${campaign}")
  --duration $(quote "${DURATION}")
  --live-batch-duration $(quote "${BATCH_DURATION}")
  --live-seeds $(quote "${SEEDS}")
  --jobs $(quote "${JOBS}")
  --artifact-limit $(quote "${ARTIFACT_LIMIT}")
  --log-level $(quote "${LOG_LEVEL}")
  --strategy-snapshot $(quote "${STRATEGY_SNAPSHOT}")
  --manifest-index $(quote "${index_file}")
  --reset-manifest-index
  --skip-paper-journal
  --execute
)
MAX_PARALLEL_COST=$(quote "${MAX_PARALLEL_COST}")
CONTINUAL_LEARNING_LEDGERS=$(quote "${CONTINUAL_LEARNING_LEDGERS}")
if [[ -n "\${MAX_PARALLEL_COST}" ]]; then
  COMMAND+=(--max-parallel-cost "\${MAX_PARALLEL_COST}")
fi
if [[ -n "\${CONTINUAL_LEARNING_LEDGERS}" ]]; then
  COMMAND+=(--continual-learning-ledgers "\${CONTINUAL_LEARNING_LEDGERS}")
  export DATADIFF_RUN_PROVENANCE_STRATEGY_LEARNING="\${CONTINUAL_LEARNING_LEDGERS}"
fi
"\${COMMAND[@]}" 2>&1 | tee $(quote "${log_file}")
EOF
  chmod +x "${script_file}"
  printf '%s\t%s\t%s\t%s\t%s\n' "${campaign}" "${session}" "${index_file}" "${log_file}" "${script_file}" >> "${CAMPAIGN_TABLE}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'dry-run: tmux new-session -d -s %s bash %s\n' "${session}" "${script_file}"
    continue
  fi
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "tmux session already exists: ${session}" >&2
    exit 2
  fi
  tmux new-session -d -s "${session}" "bash $(quote "${script_file}")"
  sleep "${STAGGER_SECONDS}"
done <<< "${CAMPAIGNS_TEXT}"

echo "campaign table: ${CAMPAIGN_TABLE}"
echo "freeze manifest: ${FREEZE_MANIFEST}"
