#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${DATADIFF_ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SESSION_NAME="${DATADIFF_DISCOVERY_LONGHAUL_SESSION:-datadiff-discovery-longhaul-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ID="${DATADIFF_DISCOVERY_LONGHAUL_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
DURATION="${DATADIFF_DISCOVERY_LONGHAUL_DURATION:-12h}"
MAX_CONCURRENT="${DATADIFF_DISCOVERY_LONGHAUL_MAX_CONCURRENT:-6}"
LOAD_LIMIT="${DATADIFF_DISCOVERY_LONGHAUL_LOAD_LIMIT:-}"
BATCH_CASES="${DATADIFF_DISCOVERY_LONGHAUL_BATCH_CASES:-220}"
SLEEP_SECONDS="${DATADIFF_DISCOVERY_LONGHAUL_SLEEP_SECONDS:-60}"
STAGGER_SECONDS="${DATADIFF_DISCOVERY_LONGHAUL_STAGGER_SECONDS:-10}"
SEED_START="${DATADIFF_DISCOVERY_LONGHAUL_SEED_START:-7001}"
LOG_LEVEL="${DATADIFF_DISCOVERY_LONGHAUL_LOG_LEVEL:-compact}"
ARTIFACT_LIMIT="${DATADIFF_DISCOVERY_LONGHAUL_ARTIFACT_LIMIT:-20}"
CLASSIFY_LIMIT="${DATADIFF_DISCOVERY_LONGHAUL_CLASSIFY_LIMIT:-5}"
CANDIDATE_RECHECK_COUNT="${DATADIFF_DISCOVERY_LONGHAUL_CANDIDATE_RECHECK_COUNT:-3}"
CANDIDATE_PIPELINE_RECHECK_ATTEMPTS="${DATADIFF_DISCOVERY_LONGHAUL_CANDIDATE_PIPELINE_RECHECK_ATTEMPTS:-3}"
GENERATED_DIR="${DATADIFF_DISCOVERY_LONGHAUL_GENERATED_DIR:-new_issue/generated}"
LOG_DIR="${DATADIFF_DISCOVERY_LONGHAUL_LOG_DIR:-logs}"
STATUS_FILE="${DATADIFF_DISCOVERY_LONGHAUL_STATUS_FILE:-${LOG_DIR}/discovery-longhaul-${RUN_ID}.status}"
AGGREGATE_FILE="${DATADIFF_DISCOVERY_LONGHAUL_AGGREGATE_FILE:-${GENERATED_DIR}/discovery-campaign-longhaul-${RUN_ID}-aggregate.json}"
FOREGROUND=0
PRINT_CONFIG=0
WORKER=0

LANE_GROUPS=(
  "polars_lazy,arrow_probe_stress"
  "embedded_sql,duckdb_storage"
  "datafusion_common_api,datafusion_optimizer"
  "common_api_workflow,cross_family,deep_probe_rotation"
  "arrow_layout,arrow_probe_stress"
  "polars_lazy,polars_streaming"
)

usage() {
  cat <<'EOF'
usage: start_discovery_longhaul_tmux.sh [--foreground] [--worker] [--print-config]

Starts a 12h/24h discovery-campaign longhaul controller. The controller launches
focused discovery-campaign batches only while active campaign count and system
load are below configured limits, then refreshes the aggregate manifest.

Key environment variables:
  DATADIFF_DISCOVERY_LONGHAUL_DURATION=12h
  DATADIFF_DISCOVERY_LONGHAUL_MAX_CONCURRENT=6
  DATADIFF_DISCOVERY_LONGHAUL_LOAD_LIMIT=<defaults to nproc * 1.35>
  DATADIFF_DISCOVERY_LONGHAUL_BATCH_CASES=220
  DATADIFF_DISCOVERY_LONGHAUL_SESSION=<tmux session>
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --foreground) FOREGROUND=1 ;;
    --worker) WORKER=1 ;;
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

duration_to_seconds() {
  local value="$1"
  local amount=""
  local unit=""
  if [[ "${value}" =~ ^([0-9]+)([smhd])$ ]]; then
    amount="${BASH_REMATCH[1]}"
    unit="${BASH_REMATCH[2]}"
  elif [[ "${value}" =~ ^([0-9]+)$ ]]; then
    amount="${BASH_REMATCH[1]}"
    unit="s"
  else
    echo "invalid duration: ${value}" >&2
    exit 2
  fi
  case "${unit}" in
    s) echo "${amount}" ;;
    m) echo $((amount * 60)) ;;
    h) echo $((amount * 3600)) ;;
    d) echo $((amount * 86400)) ;;
  esac
}

cpu_count() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
    return
  fi
  getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1
}

default_load_limit() {
  awk -v cpus="$(cpu_count)" 'BEGIN { printf "%.2f\n", cpus * 1.35 }'
}

effective_load_limit() {
  if [[ -n "${LOAD_LIMIT}" ]]; then
    printf '%s\n' "${LOAD_LIMIT}"
    return
  fi
  default_load_limit
}

current_load() {
  awk '{print $1}' /proc/loadavg 2>/dev/null || echo 0
}

active_discovery_campaigns() {
  local rows
  rows="$(pgrep -af "datadiff.cli discovery-campaign" 2>/dev/null || true)"
  printf '%s\n' "${rows}" | awk '!/start_discovery_longhaul_tmux/ { count += 1 } END { print count + 0 }'
}

load_below_limit() {
  local load="$1"
  local limit="$2"
  awk -v load="${load}" -v limit="${limit}" 'BEGIN { exit !(load < limit) }'
}

write_status() {
  local status="$1"
  local detail="${2:-}"
  local status_dir
  local status_tmp
  status_dir="$(dirname "${STATUS_FILE}")"
  mkdir -p "$(dirname "${STATUS_FILE}")"
  status_tmp="$(mktemp "${status_dir}/.$(basename "${STATUS_FILE}").tmp.XXXXXX")"
  {
    printf 'status=%s\n' "${status}"
    printf 'detail=%s\n' "${detail}"
    printf 'updated_at=%s\n' "$(date -Is)"
    printf 'root=%s\n' "${ROOT_DIR}"
    printf 'session=%s\n' "${SESSION_NAME}"
    printf 'run_id=%s\n' "${RUN_ID}"
    printf 'duration=%s\n' "${DURATION}"
    printf 'max_concurrent=%s\n' "${MAX_CONCURRENT}"
    printf 'load_limit=%s\n' "$(effective_load_limit)"
    printf 'batch_cases=%s\n' "${BATCH_CASES}"
    printf 'aggregate_file=%s\n' "${AGGREGATE_FILE}"
  } > "${status_tmp}"
  mv -f "${status_tmp}" "${STATUS_FILE}"
}

print_config() {
  printf 'root=%s\n' "${ROOT_DIR}"
  printf 'session=%s\n' "${SESSION_NAME}"
  printf 'run_id=%s\n' "${RUN_ID}"
  printf 'duration=%s\n' "${DURATION}"
  printf 'duration_s=%s\n' "$(duration_to_seconds "${DURATION}")"
  printf 'max_concurrent=%s\n' "${MAX_CONCURRENT}"
  printf 'load_limit=%s\n' "$(effective_load_limit)"
  printf 'batch_cases=%s\n' "${BATCH_CASES}"
  printf 'candidate_recheck_count=%s\n' "${CANDIDATE_RECHECK_COUNT}"
  printf 'candidate_pipeline_recheck_attempts=%s\n' "${CANDIDATE_PIPELINE_RECHECK_ATTEMPTS}"
  printf 'aggregate_file=%s\n' "${AGGREGATE_FILE}"
  printf 'lane_groups=%s\n' "$(IFS=';'; echo "${LANE_GROUPS[*]}")"
}

refresh_aggregate() {
  mkdir -p "${ROOT_DIR}/${GENERATED_DIR}"
  (
    cd "${ROOT_DIR}"
    .venv/bin/python -m datadiff.cli discovery-campaign-aggregate \
      --manifests "${GENERATED_DIR}/discovery-campaign*.json" \
      --output "${AGGREGATE_FILE}" \
      --json >/dev/null
  ) || true
}

seed_pair_for_batch() {
  local batch="$1"
  local first=$((SEED_START + (batch * 200)))
  printf '%s,%s\n' "${first}" "$((first + 101))"
}

launch_batch() {
  local batch="$1"
  local lane_index=$((batch % ${#LANE_GROUPS[@]}))
  local lanes="${LANE_GROUPS[${lane_index}]}"
  local seeds
  local manifest
  local log_file
  local child_session
  local command
  seeds="$(seed_pair_for_batch "${batch}")"
  manifest="${GENERATED_DIR}/discovery-campaign-longhaul-${RUN_ID}-batch$(printf '%04d' "${batch}").json"
  log_file="${LOG_DIR}/discovery-campaign-longhaul-${RUN_ID}-batch$(printf '%04d' "${batch}").log"
  child_session="${SESSION_NAME}-b$(printf '%04d' "${batch}")"
  mkdir -p "${ROOT_DIR}/${LOG_DIR}" "${ROOT_DIR}/${GENERATED_DIR}"
  command=(
    .venv/bin/python -m datadiff.cli discovery-campaign
    --skip-bug-audit
    --cases "${BATCH_CASES}"
    --seeds "${seeds}"
    --lanes "${lanes}"
    --candidate-recheck-count "${CANDIDATE_RECHECK_COUNT}"
    --candidate-pipeline-recheck-attempts "${CANDIDATE_PIPELINE_RECHECK_ATTEMPTS}"
    --artifact-limit "${ARTIFACT_LIMIT}"
    --classify-limit "${CLASSIFY_LIMIT}"
    --refresh-classification
    --log-level "${LOG_LEVEL}"
    --output-manifest "${manifest}"
  )
  if command -v tmux >/dev/null 2>&1; then
    tmux new-session -d -s "${child_session}" \
      "cd \"${ROOT_DIR}\" && $(printf '%q ' "${command[@]}") > \"${log_file}\" 2>&1"
  else
    (
      cd "${ROOT_DIR}"
      "${command[@]}" > "${log_file}" 2>&1 &
    )
  fi
  write_status "running" "launched batch=${batch} lanes=${lanes} seeds=${seeds} manifest=${manifest}"
}

run_worker() {
  local duration_s
  local deadline
  local batch=0
  local active
  local load
  local limit
  duration_s="$(duration_to_seconds "${DURATION}")"
  deadline=$(($(date +%s) + duration_s))
  limit="$(effective_load_limit)"
  write_status "running" "controller started"
  trap 'write_status "interrupted" "received signal"; exit 130' INT TERM HUP
  while [[ "$(date +%s)" -lt "${deadline}" ]]; do
    active="$(active_discovery_campaigns)"
    load="$(current_load)"
    if [[ "${active}" -ge "${MAX_CONCURRENT}" ]]; then
      write_status "waiting" "active campaigns ${active} >= max ${MAX_CONCURRENT}; load=${load}"
      refresh_aggregate
      sleep "${SLEEP_SECONDS}"
      continue
    fi
    if ! load_below_limit "${load}" "${limit}"; then
      write_status "waiting" "load ${load} >= limit ${limit}; active=${active}"
      refresh_aggregate
      sleep "${SLEEP_SECONDS}"
      continue
    fi
    launch_batch "${batch}"
    batch=$((batch + 1))
    refresh_aggregate
    sleep "${STAGGER_SECONDS}"
  done
  refresh_aggregate
  write_status "completed" "launch window completed"
}

if [[ "${PRINT_CONFIG}" == "1" ]]; then
  print_config
  exit 0
fi

if [[ "${WORKER}" == "1" || "${FOREGROUND}" == "1" ]]; then
  run_worker
  exit 0
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required for detached mode; use --foreground without tmux" >&2
  exit 2
fi

mkdir -p "${ROOT_DIR}/${LOG_DIR}"
CONTROLLER_ENV=(
  "DATADIFF_ROOT_DIR=${ROOT_DIR}"
  "DATADIFF_DISCOVERY_LONGHAUL_SESSION=${SESSION_NAME}"
  "DATADIFF_DISCOVERY_LONGHAUL_RUN_ID=${RUN_ID}"
  "DATADIFF_DISCOVERY_LONGHAUL_DURATION=${DURATION}"
  "DATADIFF_DISCOVERY_LONGHAUL_MAX_CONCURRENT=${MAX_CONCURRENT}"
  "DATADIFF_DISCOVERY_LONGHAUL_LOAD_LIMIT=${LOAD_LIMIT}"
  "DATADIFF_DISCOVERY_LONGHAUL_BATCH_CASES=${BATCH_CASES}"
  "DATADIFF_DISCOVERY_LONGHAUL_SLEEP_SECONDS=${SLEEP_SECONDS}"
  "DATADIFF_DISCOVERY_LONGHAUL_STAGGER_SECONDS=${STAGGER_SECONDS}"
  "DATADIFF_DISCOVERY_LONGHAUL_SEED_START=${SEED_START}"
  "DATADIFF_DISCOVERY_LONGHAUL_LOG_LEVEL=${LOG_LEVEL}"
  "DATADIFF_DISCOVERY_LONGHAUL_ARTIFACT_LIMIT=${ARTIFACT_LIMIT}"
  "DATADIFF_DISCOVERY_LONGHAUL_CLASSIFY_LIMIT=${CLASSIFY_LIMIT}"
  "DATADIFF_DISCOVERY_LONGHAUL_CANDIDATE_RECHECK_COUNT=${CANDIDATE_RECHECK_COUNT}"
  "DATADIFF_DISCOVERY_LONGHAUL_CANDIDATE_PIPELINE_RECHECK_ATTEMPTS=${CANDIDATE_PIPELINE_RECHECK_ATTEMPTS}"
  "DATADIFF_DISCOVERY_LONGHAUL_GENERATED_DIR=${GENERATED_DIR}"
  "DATADIFF_DISCOVERY_LONGHAUL_LOG_DIR=${LOG_DIR}"
  "DATADIFF_DISCOVERY_LONGHAUL_STATUS_FILE=${STATUS_FILE}"
  "DATADIFF_DISCOVERY_LONGHAUL_AGGREGATE_FILE=${AGGREGATE_FILE}"
)
CONTROLLER_COMMAND="$(printf '%q ' env "${CONTROLLER_ENV[@]}" bash "${BASH_SOURCE[0]}" --worker)"
tmux new-session -d -s "${SESSION_NAME}" \
  "cd \"${ROOT_DIR}\" && ${CONTROLLER_COMMAND} >> \"${LOG_DIR}/discovery-longhaul-${RUN_ID}.controller.log\" 2>&1"

write_status "starting" "tmux controller launched"
printf 'session: %s\n' "${SESSION_NAME}"
printf 'status: %s\n' "${STATUS_FILE}"
printf 'aggregate: %s\n' "${AGGREGATE_FILE}"
