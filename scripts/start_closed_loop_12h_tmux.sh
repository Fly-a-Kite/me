#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=""
PYTHON_CMD=""
PYTHONPATH_VALUE=""
SESSION_NAME=""
DURATION=""
BATCH_DURATION=""
JOBS=""
MAX_PARALLEL_COST=""
SEEDS=""
TARGET_SUITES=""
PRESETS=""
PROFILE_POOL=""
PROFILE_LEARNING_WEIGHT=""
SEMANTIC_OBJECTIVE_LEARNING_WEIGHT=""
ENABLE_METAMORPHIC_ORACLE=""
METAMORPHIC_RELATION_LEARNING_WEIGHT=""
METAMORPHIC_RELATION_ORDER=""
VERSION_PAIR_POOL=""
VERSION_PAIR_LEARNING_WEIGHT=""
EXPLORATION_WEIGHT=""
GROUP_FAIRNESS_WEIGHT=""
MAX_GROUP_PULL_GAP=""
ADAPTIVE_LEARNING_WEIGHT=""
SCHEDULER_ANNEALING_TEMPERATURE=""
SCHEDULER_ANNEALING_DECAY=""
SCHEDULER_ANNEALING_MIN_TEMPERATURE=""
CONTINUAL_LEARNING_LEDGERS=""
LOCAL_SOURCE_EXPLORATION_WEIGHT=""
ARTIFACT_LIMIT=""
LOG_LEVEL=""
RUN_THEME=""
PAPER_NOTES=""
COMMAND_OVERRIDE=""
TMUX_WATCH_INTERVAL=""
LOG_PREFIX=""
ALLOW_EQUIVALENT_CONCURRENT_LAUNCH=""
RUN_PROVENANCE_AUTHORITY=""
RUN_PROVENANCE_FREEZE_INTENT=""
RUN_PROVENANCE_LATEST_CODE_CLAIM=""
RUN_PROVENANCE_EVIDENCE_ROLE=""
RUN_PROVENANCE_LAUNCH_SOURCE=""
RUN_PROVENANCE_LAUNCH_SCRIPT=""
RUN_PROVENANCE_FREEZE_MANIFEST=""
RUN_PROVENANCE_PIP_FREEZE=""
RUN_PROVENANCE_GIT_STATUS=""
RUN_PROVENANCE_GIT_DIFF=""
RUN_PROVENANCE_LAUNCH_ENV=""
RUN_PROVENANCE_STRATEGY_SNAPSHOT=""
RUN_PROVENANCE_STRATEGY_LEARNING=""
REQUIRE_CLEAN_WORKTREE=""
POST_RUN_EVIDENCE_HOOK=""
FINAL_READINESS_MANIFEST_INDEX=""
FINAL_READINESS_EXTRA_MANIFESTS=""
FINAL_READINESS_FAIL_ON_MISSING=""
POST_RUN_STATUS=""
EXPERIMENT_MANIFEST_PATH=""
EXPERIMENT_SUMMARY_MARKDOWN=""
EXPERIMENT_SUMMARY_CSV=""
EXPERIMENT_SUMMARY_AGGREGATE_CSV=""
EXPERIMENT_ANALYSIS_MARKDOWN=""
EXPERIMENT_ANALYSIS_CSV=""
METHODOLOGY_REPORT_MARKDOWN=""
METHODOLOGY_REPORT_JSON=""
FINAL_READINESS_MARKDOWN=""
FINAL_READINESS_JSON=""
CLASSIFY_RUN_DIR=""
CLASSIFY_RUN_COUNT=""
EXPERIMENT_COMMAND=()
FINAL_READINESS_COMMAND=()

refresh_config() {
  ROOT_DIR="${DATADIFF_ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
  PYTHON_CMD="${DATADIFF_PYTHON:-}"
  PYTHONPATH_VALUE="${DATADIFF_PYTHONPATH:-${PYTHONPATH:-}}"
  SESSION_NAME="${DATADIFF_TMUX_SESSION:-datadiff-closed-loop-12h-authority}"
  DURATION="${DATADIFF_DURATION:-12h}"
  BATCH_DURATION="${DATADIFF_BATCH_DURATION:-10m}"
  JOBS="${DATADIFF_JOBS:-1}"
  MAX_PARALLEL_COST="${DATADIFF_MAX_PARALLEL_COST:-8}"
  SEEDS="${DATADIFF_SEEDS:-1,501,1001,1501,2001,2501}"
  TARGET_SUITES="${DATADIFF_TARGET_SUITES:-latest_no_datafusion}"
  PRESETS="${DATADIFF_PRESETS:-live_issue_focus,live_cross_family}"
  PROFILE_POOL="${DATADIFF_PROFILE_POOL:-issue_focus,discovery,discovery_fresh,deep_probe_rotation,partitioned_running_sum,join_null_sort,ordered_groupby_sort,global_null_aggregate,window_avg_rows_frame,path_basename_keyed_pick}"
  PROFILE_LEARNING_WEIGHT="${DATADIFF_PROFILE_LEARNING_WEIGHT:-1.0}"
  SEMANTIC_OBJECTIVE_LEARNING_WEIGHT="${DATADIFF_SEMANTIC_OBJECTIVE_LEARNING_WEIGHT:-1.0}"
  ENABLE_METAMORPHIC_ORACLE="${DATADIFF_ENABLE_METAMORPHIC_ORACLE:-0}"
  METAMORPHIC_RELATION_LEARNING_WEIGHT="${DATADIFF_METAMORPHIC_RELATION_LEARNING_WEIGHT:-0.0}"
  METAMORPHIC_RELATION_ORDER="${DATADIFF_METAMORPHIC_RELATION_ORDER:-}"
  VERSION_PAIR_POOL="${DATADIFF_VERSION_PAIR_POOL:-}"
  VERSION_PAIR_LEARNING_WEIGHT="${DATADIFF_VERSION_PAIR_LEARNING_WEIGHT:-0.0}"
  EXPLORATION_WEIGHT="${DATADIFF_EXPLORATION_WEIGHT:-0.2}"
  GROUP_FAIRNESS_WEIGHT="${DATADIFF_GROUP_FAIRNESS_WEIGHT:-0.4}"
  MAX_GROUP_PULL_GAP="${DATADIFF_MAX_GROUP_PULL_GAP:-3}"
  ADAPTIVE_LEARNING_WEIGHT="${DATADIFF_ADAPTIVE_LEARNING_WEIGHT:-0.75}"
  SCHEDULER_ANNEALING_TEMPERATURE="${DATADIFF_SCHEDULER_ANNEALING_TEMPERATURE:-0.35}"
  SCHEDULER_ANNEALING_DECAY="${DATADIFF_SCHEDULER_ANNEALING_DECAY:-0.985}"
  SCHEDULER_ANNEALING_MIN_TEMPERATURE="${DATADIFF_SCHEDULER_ANNEALING_MIN_TEMPERATURE:-0.02}"
  CONTINUAL_LEARNING_LEDGERS="${DATADIFF_CONTINUAL_LEARNING_LEDGERS:-}"
  LOCAL_SOURCE_EXPLORATION_WEIGHT="${DATADIFF_LOCAL_SOURCE_EXPLORATION_WEIGHT:-0.1}"
  ARTIFACT_LIMIT="${DATADIFF_ARTIFACT_LIMIT:-10}"
  LOG_LEVEL="${DATADIFF_LOG_LEVEL:-minimal}"
  RUN_THEME="${DATADIFF_RUN_THEME:-closed-loop-12h-live-authority}"
  PAPER_NOTES="${DATADIFF_PAPER_NOTES:-12h single-machine authoritative adaptive live run started after throughput, stage-profile, and fresh-family validation across issue_focus and cross_family.}"
  COMMAND_OVERRIDE="${DATADIFF_COMMAND:-}"
  TMUX_WATCH_INTERVAL="${DATADIFF_TMUX_WATCH_INTERVAL:-2}"
  LOG_PREFIX="${DATADIFF_LOG_PREFIX:-closed-loop-adaptive-12h}"
  ALLOW_EQUIVALENT_CONCURRENT_LAUNCH="${DATADIFF_ALLOW_EQUIVALENT_CONCURRENT_LAUNCH:-0}"
  RUN_PROVENANCE_AUTHORITY="${DATADIFF_RUN_PROVENANCE_AUTHORITY:-1}"
  RUN_PROVENANCE_FREEZE_INTENT="${DATADIFF_RUN_PROVENANCE_FREEZE_INTENT:-1}"
  RUN_PROVENANCE_LATEST_CODE_CLAIM="${DATADIFF_RUN_PROVENANCE_LATEST_CODE_CLAIM:-1}"
  RUN_PROVENANCE_EVIDENCE_ROLE="${DATADIFF_RUN_PROVENANCE_EVIDENCE_ROLE:-latest_live_authority_12h}"
  RUN_PROVENANCE_LAUNCH_SOURCE="${DATADIFF_RUN_PROVENANCE_LAUNCH_SOURCE:-closed_loop_tmux}"
  RUN_PROVENANCE_LAUNCH_SCRIPT="${DATADIFF_RUN_PROVENANCE_LAUNCH_SCRIPT:-$(basename "${BASH_SOURCE[0]}")}"
  RUN_PROVENANCE_FREEZE_MANIFEST="${DATADIFF_RUN_PROVENANCE_FREEZE_MANIFEST:-}"
  RUN_PROVENANCE_PIP_FREEZE="${DATADIFF_RUN_PROVENANCE_PIP_FREEZE:-}"
  RUN_PROVENANCE_GIT_STATUS="${DATADIFF_RUN_PROVENANCE_GIT_STATUS:-}"
  RUN_PROVENANCE_GIT_DIFF="${DATADIFF_RUN_PROVENANCE_GIT_DIFF:-}"
  RUN_PROVENANCE_LAUNCH_ENV="${DATADIFF_RUN_PROVENANCE_LAUNCH_ENV:-}"
  RUN_PROVENANCE_STRATEGY_SNAPSHOT="${DATADIFF_RUN_PROVENANCE_STRATEGY_SNAPSHOT:-}"
  RUN_PROVENANCE_STRATEGY_LEARNING="${DATADIFF_RUN_PROVENANCE_STRATEGY_LEARNING:-}"
  REQUIRE_CLEAN_WORKTREE="${DATADIFF_REQUIRE_CLEAN_WORKTREE:-1}"
  POST_RUN_EVIDENCE_HOOK="${DATADIFF_POST_RUN_EVIDENCE_HOOK:-}"
  FINAL_READINESS_MANIFEST_INDEX="${DATADIFF_FINAL_READINESS_MANIFEST_INDEX:-}"
  FINAL_READINESS_EXTRA_MANIFESTS="${DATADIFF_FINAL_READINESS_EXTRA_MANIFESTS:-}"
  FINAL_READINESS_FAIL_ON_MISSING="${DATADIFF_FINAL_READINESS_FAIL_ON_MISSING:-0}"
  if [[ -n "${PYTHONPATH_VALUE}" ]]; then
    export PYTHONPATH="${PYTHONPATH_VALUE}"
  fi
}

write_config_file() {
  local config_file="$1"
  mkdir -p "$(dirname "${config_file}")"
  {
    printf 'DATADIFF_ROOT_DIR=%q\n' "${ROOT_DIR}"
    printf 'DATADIFF_PYTHON=%q\n' "${PYTHON_CMD}"
    printf 'DATADIFF_PYTHONPATH=%q\n' "${PYTHONPATH_VALUE}"
    printf 'DATADIFF_TMUX_SESSION=%q\n' "${SESSION_NAME}"
    printf 'DATADIFF_DURATION=%q\n' "${DURATION}"
    printf 'DATADIFF_BATCH_DURATION=%q\n' "${BATCH_DURATION}"
    printf 'DATADIFF_JOBS=%q\n' "${JOBS}"
    printf 'DATADIFF_MAX_PARALLEL_COST=%q\n' "${MAX_PARALLEL_COST}"
    printf 'DATADIFF_SEEDS=%q\n' "${SEEDS}"
    printf 'DATADIFF_TARGET_SUITES=%q\n' "${TARGET_SUITES}"
    printf 'DATADIFF_PRESETS=%q\n' "${PRESETS}"
    printf 'DATADIFF_PROFILE_POOL=%q\n' "${PROFILE_POOL}"
    printf 'DATADIFF_PROFILE_LEARNING_WEIGHT=%q\n' "${PROFILE_LEARNING_WEIGHT}"
    printf 'DATADIFF_SEMANTIC_OBJECTIVE_LEARNING_WEIGHT=%q\n' "${SEMANTIC_OBJECTIVE_LEARNING_WEIGHT}"
    printf 'DATADIFF_ENABLE_METAMORPHIC_ORACLE=%q\n' "${ENABLE_METAMORPHIC_ORACLE}"
    printf 'DATADIFF_METAMORPHIC_RELATION_LEARNING_WEIGHT=%q\n' "${METAMORPHIC_RELATION_LEARNING_WEIGHT}"
    printf 'DATADIFF_METAMORPHIC_RELATION_ORDER=%q\n' "${METAMORPHIC_RELATION_ORDER}"
    printf 'DATADIFF_VERSION_PAIR_POOL=%q\n' "${VERSION_PAIR_POOL}"
    printf 'DATADIFF_VERSION_PAIR_LEARNING_WEIGHT=%q\n' "${VERSION_PAIR_LEARNING_WEIGHT}"
    printf 'DATADIFF_EXPLORATION_WEIGHT=%q\n' "${EXPLORATION_WEIGHT}"
    printf 'DATADIFF_GROUP_FAIRNESS_WEIGHT=%q\n' "${GROUP_FAIRNESS_WEIGHT}"
    printf 'DATADIFF_MAX_GROUP_PULL_GAP=%q\n' "${MAX_GROUP_PULL_GAP}"
    printf 'DATADIFF_ADAPTIVE_LEARNING_WEIGHT=%q\n' "${ADAPTIVE_LEARNING_WEIGHT}"
    printf 'DATADIFF_SCHEDULER_ANNEALING_TEMPERATURE=%q\n' "${SCHEDULER_ANNEALING_TEMPERATURE}"
    printf 'DATADIFF_SCHEDULER_ANNEALING_DECAY=%q\n' "${SCHEDULER_ANNEALING_DECAY}"
    printf 'DATADIFF_SCHEDULER_ANNEALING_MIN_TEMPERATURE=%q\n' "${SCHEDULER_ANNEALING_MIN_TEMPERATURE}"
    printf 'DATADIFF_CONTINUAL_LEARNING_LEDGERS=%q\n' "${CONTINUAL_LEARNING_LEDGERS}"
    printf 'DATADIFF_LOCAL_SOURCE_EXPLORATION_WEIGHT=%q\n' "${LOCAL_SOURCE_EXPLORATION_WEIGHT}"
    printf 'DATADIFF_ARTIFACT_LIMIT=%q\n' "${ARTIFACT_LIMIT}"
    printf 'DATADIFF_LOG_LEVEL=%q\n' "${LOG_LEVEL}"
    printf 'DATADIFF_RUN_THEME=%q\n' "${RUN_THEME}"
    printf 'DATADIFF_PAPER_NOTES=%q\n' "${PAPER_NOTES}"
    printf 'DATADIFF_COMMAND=%q\n' "${COMMAND_OVERRIDE}"
    printf 'DATADIFF_TMUX_WATCH_INTERVAL=%q\n' "${TMUX_WATCH_INTERVAL}"
    printf 'DATADIFF_LOG_PREFIX=%q\n' "${LOG_PREFIX}"
    printf 'DATADIFF_ALLOW_EQUIVALENT_CONCURRENT_LAUNCH=%q\n' "${ALLOW_EQUIVALENT_CONCURRENT_LAUNCH}"
    printf 'DATADIFF_RUN_PROVENANCE_AUTHORITY=%q\n' "${RUN_PROVENANCE_AUTHORITY}"
    printf 'DATADIFF_RUN_PROVENANCE_FREEZE_INTENT=%q\n' "${RUN_PROVENANCE_FREEZE_INTENT}"
    printf 'DATADIFF_RUN_PROVENANCE_LATEST_CODE_CLAIM=%q\n' "${RUN_PROVENANCE_LATEST_CODE_CLAIM}"
    printf 'DATADIFF_RUN_PROVENANCE_EVIDENCE_ROLE=%q\n' "${RUN_PROVENANCE_EVIDENCE_ROLE}"
    printf 'DATADIFF_RUN_PROVENANCE_LAUNCH_SOURCE=%q\n' "${RUN_PROVENANCE_LAUNCH_SOURCE}"
    printf 'DATADIFF_RUN_PROVENANCE_SESSION=%q\n' "${SESSION_NAME}"
    printf 'DATADIFF_RUN_PROVENANCE_DURATION=%q\n' "${DURATION}"
    printf 'DATADIFF_RUN_PROVENANCE_BATCH_DURATION=%q\n' "${BATCH_DURATION}"
    printf 'DATADIFF_RUN_PROVENANCE_LOG_PREFIX=%q\n' "${LOG_PREFIX}"
    printf 'DATADIFF_RUN_PROVENANCE_LAUNCH_SCRIPT=%q\n' "${RUN_PROVENANCE_LAUNCH_SCRIPT}"
    printf 'DATADIFF_RUN_PROVENANCE_FREEZE_MANIFEST=%q\n' "${RUN_PROVENANCE_FREEZE_MANIFEST}"
    printf 'DATADIFF_RUN_PROVENANCE_PIP_FREEZE=%q\n' "${RUN_PROVENANCE_PIP_FREEZE}"
    printf 'DATADIFF_RUN_PROVENANCE_GIT_STATUS=%q\n' "${RUN_PROVENANCE_GIT_STATUS}"
    printf 'DATADIFF_RUN_PROVENANCE_GIT_DIFF=%q\n' "${RUN_PROVENANCE_GIT_DIFF}"
    printf 'DATADIFF_RUN_PROVENANCE_LAUNCH_ENV=%q\n' "${RUN_PROVENANCE_LAUNCH_ENV}"
    printf 'DATADIFF_RUN_PROVENANCE_STRATEGY_SNAPSHOT=%q\n' "${RUN_PROVENANCE_STRATEGY_SNAPSHOT}"
    printf 'DATADIFF_RUN_PROVENANCE_STRATEGY_LEARNING=%q\n' "${RUN_PROVENANCE_STRATEGY_LEARNING}"
    printf 'DATADIFF_REQUIRE_CLEAN_WORKTREE=%q\n' "${REQUIRE_CLEAN_WORKTREE}"
    printf 'DATADIFF_POST_RUN_EVIDENCE_HOOK=%q\n' "${POST_RUN_EVIDENCE_HOOK}"
    printf 'DATADIFF_FINAL_READINESS_MANIFEST_INDEX=%q\n' "${FINAL_READINESS_MANIFEST_INDEX}"
    printf 'DATADIFF_FINAL_READINESS_EXTRA_MANIFESTS=%q\n' "${FINAL_READINESS_EXTRA_MANIFESTS}"
    printf 'DATADIFF_FINAL_READINESS_FAIL_ON_MISSING=%q\n' "${FINAL_READINESS_FAIL_ON_MISSING}"
  } > "${config_file}"
}

refresh_config

write_status_file() {
  local status_file="$1"
  local status_value="$2"
  local run_id="$3"
  local log_file="$4"
  local timestamp_label="$5"
  local exit_code="${6:-}"
  local status_dir
  local status_tmp
  status_dir="$(dirname "${status_file}")"
  mkdir -p "${status_dir}"
  status_tmp="$(mktemp "${status_dir}/.$(basename "${status_file}").tmp.XXXXXX")"
  {
    printf 'status=%s\n' "${status_value}"
    printf 'run_id=%s\n' "${run_id}"
    printf '%s=%s\n' "${timestamp_label}" "$(date -Is)"
    if [[ -n "${exit_code}" ]]; then
      printf 'exit_code=%s\n' "${exit_code}"
    fi
    printf 'log_file=%s\n' "${log_file}"
    printf 'authority=%s\n' "${RUN_PROVENANCE_AUTHORITY}"
    printf 'freeze_intent=%s\n' "${RUN_PROVENANCE_FREEZE_INTENT}"
    printf 'latest_code_claim=%s\n' "${RUN_PROVENANCE_LATEST_CODE_CLAIM}"
    printf 'launch_source=%s\n' "${RUN_PROVENANCE_LAUNCH_SOURCE}"
    printf 'freeze_manifest=%s\n' "${RUN_PROVENANCE_FREEZE_MANIFEST}"
    printf 'strategy_snapshot=%s\n' "${RUN_PROVENANCE_STRATEGY_SNAPSHOT}"
    printf 'strategy_learning=%s\n' "${RUN_PROVENANCE_STRATEGY_LEARNING}"
    if [[ -n "${POST_RUN_STATUS}" ]]; then
      printf 'post_run_status=%s\n' "${POST_RUN_STATUS}"
    fi
    if [[ -n "${EXPERIMENT_MANIFEST_PATH}" ]]; then
      printf 'experiment_manifest=%s\n' "${EXPERIMENT_MANIFEST_PATH}"
    fi
    if [[ -n "${EXPERIMENT_SUMMARY_MARKDOWN}" ]]; then
      printf 'experiment_summary_markdown=%s\n' "${EXPERIMENT_SUMMARY_MARKDOWN}"
    fi
    if [[ -n "${EXPERIMENT_SUMMARY_CSV}" ]]; then
      printf 'experiment_summary_csv=%s\n' "${EXPERIMENT_SUMMARY_CSV}"
    fi
    if [[ -n "${EXPERIMENT_SUMMARY_AGGREGATE_CSV}" ]]; then
      printf 'experiment_summary_aggregate_csv=%s\n' "${EXPERIMENT_SUMMARY_AGGREGATE_CSV}"
    fi
    if [[ -n "${EXPERIMENT_ANALYSIS_MARKDOWN}" ]]; then
      printf 'experiment_analysis_markdown=%s\n' "${EXPERIMENT_ANALYSIS_MARKDOWN}"
    fi
    if [[ -n "${EXPERIMENT_ANALYSIS_CSV}" ]]; then
      printf 'experiment_analysis_csv=%s\n' "${EXPERIMENT_ANALYSIS_CSV}"
    fi
    if [[ -n "${METHODOLOGY_REPORT_MARKDOWN}" ]]; then
      printf 'methodology_report_markdown=%s\n' "${METHODOLOGY_REPORT_MARKDOWN}"
    fi
    if [[ -n "${METHODOLOGY_REPORT_JSON}" ]]; then
      printf 'methodology_report_json=%s\n' "${METHODOLOGY_REPORT_JSON}"
    fi
    if [[ -n "${FINAL_READINESS_MARKDOWN}" ]]; then
      printf 'final_readiness_markdown=%s\n' "${FINAL_READINESS_MARKDOWN}"
    fi
    if [[ -n "${FINAL_READINESS_JSON}" ]]; then
      printf 'final_readiness_json=%s\n' "${FINAL_READINESS_JSON}"
    fi
    if [[ -n "${FINAL_READINESS_MANIFEST_INDEX}" ]]; then
      printf 'final_readiness_manifest_index=%s\n' "${FINAL_READINESS_MANIFEST_INDEX}"
    fi
    if [[ -n "${FINAL_READINESS_EXTRA_MANIFESTS}" ]]; then
      printf 'final_readiness_extra_manifests=%s\n' "${FINAL_READINESS_EXTRA_MANIFESTS}"
    fi
    printf 'final_readiness_fail_on_missing=%s\n' "${FINAL_READINESS_FAIL_ON_MISSING}"
    if [[ -n "${CLASSIFY_RUN_DIR}" ]]; then
      printf 'classify_run_dir=%s\n' "${CLASSIFY_RUN_DIR}"
    fi
    if [[ -n "${CLASSIFY_RUN_COUNT}" ]]; then
      printf 'classify_run_count=%s\n' "${CLASSIFY_RUN_COUNT}"
    fi
  } > "${status_tmp}"
  mv -f "${status_tmp}" "${status_file}"
}

_git_in_root() {
  git -C "${ROOT_DIR}" "$@"
}

_git_repo_available() {
  _git_in_root rev-parse --is-inside-work-tree >/dev/null 2>&1
}

_git_commit() {
  _git_in_root rev-parse HEAD 2>/dev/null || true
}

_git_branch() {
  _git_in_root symbolic-ref --quiet --short HEAD 2>/dev/null || true
}

_workspace_dirty() {
  [[ -n "$(_git_in_root status --porcelain 2>/dev/null || true)" ]]
}

_workspace_dirty_text() {
  if _workspace_dirty; then
    printf 'true\n'
  else
    printf 'false\n'
  fi
}

_require_clean_authority_workspace() {
  if [[ "${RUN_PROVENANCE_AUTHORITY}" != "1" || "${REQUIRE_CLEAN_WORKTREE}" != "1" ]]; then
    return 0
  fi
  if ! _git_repo_available; then
    echo "authority long-run requires a git worktree at ROOT_DIR=${ROOT_DIR}"
    return 4
  fi
  if _workspace_dirty; then
    echo "authority long-run requires a clean git worktree at ROOT_DIR=${ROOT_DIR}"
    return 4
  fi
}

_python_cmd() {
  if [[ -n "${PYTHON_CMD}" ]]; then
    printf '%s\n' "${PYTHON_CMD}"
    return
  fi
  if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    printf '%s\n' "${ROOT_DIR}/.venv/bin/python"
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return
  fi
  command -v python
}

_pip_freeze_cmd() {
  _python_cmd
}

_prepare_freeze_snapshot() {
  local run_id="$1"
  local snapshot_dir="${ROOT_DIR}/reports/freeze-snapshots"
  local strategy_snapshot_dir="${ROOT_DIR}/reports/strategy-snapshots"
  local strategy_learning_dir="${ROOT_DIR}/reports/strategy-learning"
  local code_root
  local prefix="${LOG_PREFIX}-${run_id}"
  local python_cmd
  local workspace_dirty_before_snapshot
  mkdir -p "${snapshot_dir}"
  mkdir -p "${strategy_snapshot_dir}"
  mkdir -p "${strategy_learning_dir}"
  RUN_PROVENANCE_FREEZE_MANIFEST="${snapshot_dir}/${prefix}.json"
  RUN_PROVENANCE_PIP_FREEZE="${snapshot_dir}/${prefix}.pip-freeze.txt"
  RUN_PROVENANCE_GIT_STATUS="${snapshot_dir}/${prefix}.git-status.txt"
  RUN_PROVENANCE_GIT_DIFF="${snapshot_dir}/${prefix}.git-diff.patch"
  RUN_PROVENANCE_LAUNCH_ENV="${snapshot_dir}/${prefix}.launcher-env.txt"
  RUN_PROVENANCE_STRATEGY_SNAPSHOT="${strategy_snapshot_dir}/${prefix}.strategy-snapshot.json"
  RUN_PROVENANCE_STRATEGY_LEARNING="${strategy_learning_dir}/${prefix}.strategy-learning.json"
  workspace_dirty_before_snapshot="$(_workspace_dirty_text)"

  if _git_repo_available; then
    _git_in_root status --short --branch > "${RUN_PROVENANCE_GIT_STATUS}"
    _git_in_root diff --no-ext-diff > "${RUN_PROVENANCE_GIT_DIFF}"
  else
    : > "${RUN_PROVENANCE_GIT_STATUS}"
    : > "${RUN_PROVENANCE_GIT_DIFF}"
  fi

  python_cmd="$(_pip_freeze_cmd)"
  code_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  "${python_cmd}" -m pip freeze > "${RUN_PROVENANCE_PIP_FREEZE}" || true
  FREEZE_CODE_ROOT="${code_root}" \
  FREEZE_STRATEGY_SNAPSHOT_PATH="${RUN_PROVENANCE_STRATEGY_SNAPSHOT}" \
  "${python_cmd}" - <<'PY'
from pathlib import Path
import os
import sys

code_root = Path(os.environ["FREEZE_CODE_ROOT"])
sys.path.insert(0, str(code_root / "src"))

from datadiff.classification_oracle import documented_semantic_rule_records
from datadiff.classification_oracle import semantic_boundary_rule_records
from datadiff.dynamic_strategy import write_strategy_snapshot
from datadiff.triage import standalone_reproducer_rule_records

target = Path(os.environ["FREEZE_STRATEGY_SNAPSHOT_PATH"])
write_strategy_snapshot(
    classification_documented_rules=list(documented_semantic_rule_records()),
    classification_boundary_rules=list(semantic_boundary_rule_records()),
    reproducer_rules=list(standalone_reproducer_rule_records()),
    metadata={"generated_by": "start_closed_loop_12h_tmux.sh"},
    output_dir=target.parent,
    snapshot_id=target.stem,
)
PY
  if [[ ! -f "${RUN_PROVENANCE_STRATEGY_SNAPSHOT}" ]]; then
    printf '{\"schema_version\": \"dynamic-strategy-snapshot-v1\", \"generated_at\": \"\", \"snapshot_id\": \"launcher-fallback\", \"classification_documented_rules\": [], \"classification_boundary_rules\": [], \"reproducer_rules\": [], \"learning_summary\": {}, \"metadata\": {\"generated_by\": \"start_closed_loop_12h_tmux.sh\"}}\n' > "${RUN_PROVENANCE_STRATEGY_SNAPSHOT}"
  fi
  printf '{\"schema_version\": \"dynamic-strategy-learning-v1\", \"events\": [], \"generated_by\": \"start_closed_loop_12h_tmux.sh\"}\n' > "${RUN_PROVENANCE_STRATEGY_LEARNING}"

  {
    printf 'root=%s\n' "${ROOT_DIR}"
    printf 'session=%s\n' "${SESSION_NAME}"
    printf 'duration=%s\n' "${DURATION}"
    printf 'batch_duration=%s\n' "${BATCH_DURATION}"
    printf 'target_suites=%s\n' "${TARGET_SUITES}"
    printf 'presets=%s\n' "${PRESETS}"
    printf 'profile_pool=%s\n' "${PROFILE_POOL}"
    printf 'profile_learning_weight=%s\n' "${PROFILE_LEARNING_WEIGHT}"
    printf 'semantic_objective_learning_weight=%s\n' "${SEMANTIC_OBJECTIVE_LEARNING_WEIGHT}"
    printf 'enable_metamorphic_oracle=%s\n' "${ENABLE_METAMORPHIC_ORACLE}"
    printf 'metamorphic_relation_learning_weight=%s\n' "${METAMORPHIC_RELATION_LEARNING_WEIGHT}"
    printf 'metamorphic_relation_order=%s\n' "${METAMORPHIC_RELATION_ORDER}"
    printf 'version_pair_pool=%s\n' "${VERSION_PAIR_POOL}"
    printf 'version_pair_learning_weight=%s\n' "${VERSION_PAIR_LEARNING_WEIGHT}"
    printf 'seeds=%s\n' "${SEEDS}"
    printf 'adaptive_learning_weight=%s\n' "${ADAPTIVE_LEARNING_WEIGHT}"
    printf 'scheduler_annealing_temperature=%s\n' "${SCHEDULER_ANNEALING_TEMPERATURE}"
    printf 'scheduler_annealing_decay=%s\n' "${SCHEDULER_ANNEALING_DECAY}"
    printf 'scheduler_annealing_min_temperature=%s\n' "${SCHEDULER_ANNEALING_MIN_TEMPERATURE}"
    printf 'continual_learning_ledgers=%s\n' "${CONTINUAL_LEARNING_LEDGERS}"
    printf 'persist_closed_loop_state=1\n'
    printf 'log_level=%s\n' "${LOG_LEVEL}"
    printf 'authority=%s\n' "${RUN_PROVENANCE_AUTHORITY}"
    printf 'freeze_intent=%s\n' "${RUN_PROVENANCE_FREEZE_INTENT}"
    printf 'latest_code_claim=%s\n' "${RUN_PROVENANCE_LATEST_CODE_CLAIM}"
    printf 'launch_source=%s\n' "${RUN_PROVENANCE_LAUNCH_SOURCE}"
    printf 'launch_script=%s\n' "${RUN_PROVENANCE_LAUNCH_SCRIPT}"
    printf 'git_commit=%s\n' "$(_git_commit)"
    printf 'git_branch=%s\n' "$(_git_branch)"
    printf 'workspace_dirty=%s\n' "${workspace_dirty_before_snapshot}"
    printf 'strategy_snapshot=%s\n' "${RUN_PROVENANCE_STRATEGY_SNAPSHOT}"
    printf 'strategy_learning=%s\n' "${RUN_PROVENANCE_STRATEGY_LEARNING}"
    printf 'final_readiness_manifest_index=%s\n' "${FINAL_READINESS_MANIFEST_INDEX}"
    printf 'final_readiness_extra_manifests=%s\n' "${FINAL_READINESS_EXTRA_MANIFESTS}"
    printf 'final_readiness_fail_on_missing=%s\n' "${FINAL_READINESS_FAIL_ON_MISSING}"
  } > "${RUN_PROVENANCE_LAUNCH_ENV}"
  FREEZE_MANIFEST_PATH="${RUN_PROVENANCE_FREEZE_MANIFEST}" \
  FREEZE_ROOT_DIR="${ROOT_DIR}" \
  FREEZE_RUN_ID="${run_id}" \
  FREEZE_SESSION="${SESSION_NAME}" \
  FREEZE_DURATION="${DURATION}" \
  FREEZE_BATCH_DURATION="${BATCH_DURATION}" \
  FREEZE_TARGET_SUITES="${TARGET_SUITES}" \
  FREEZE_PRESETS="${PRESETS}" \
  FREEZE_PROFILE_POOL="${PROFILE_POOL}" \
  FREEZE_PROFILE_LEARNING_WEIGHT="${PROFILE_LEARNING_WEIGHT}" \
  FREEZE_SEMANTIC_OBJECTIVE_LEARNING_WEIGHT="${SEMANTIC_OBJECTIVE_LEARNING_WEIGHT}" \
  FREEZE_ENABLE_METAMORPHIC_ORACLE="${ENABLE_METAMORPHIC_ORACLE}" \
  FREEZE_METAMORPHIC_RELATION_LEARNING_WEIGHT="${METAMORPHIC_RELATION_LEARNING_WEIGHT}" \
  FREEZE_METAMORPHIC_RELATION_ORDER="${METAMORPHIC_RELATION_ORDER}" \
  FREEZE_VERSION_PAIR_POOL="${VERSION_PAIR_POOL}" \
  FREEZE_VERSION_PAIR_LEARNING_WEIGHT="${VERSION_PAIR_LEARNING_WEIGHT}" \
  FREEZE_SEEDS="${SEEDS}" \
  FREEZE_ADAPTIVE_LEARNING_WEIGHT="${ADAPTIVE_LEARNING_WEIGHT}" \
  FREEZE_SCHEDULER_ANNEALING_TEMPERATURE="${SCHEDULER_ANNEALING_TEMPERATURE}" \
  FREEZE_SCHEDULER_ANNEALING_DECAY="${SCHEDULER_ANNEALING_DECAY}" \
  FREEZE_SCHEDULER_ANNEALING_MIN_TEMPERATURE="${SCHEDULER_ANNEALING_MIN_TEMPERATURE}" \
  FREEZE_CONTINUAL_LEARNING_LEDGERS="${CONTINUAL_LEARNING_LEDGERS}" \
  FREEZE_PERSIST_CLOSED_LOOP_STATE="1" \
  FREEZE_LOG_LEVEL="${LOG_LEVEL}" \
  FREEZE_AUTHORITY="${RUN_PROVENANCE_AUTHORITY}" \
  FREEZE_FREEZE_INTENT="${RUN_PROVENANCE_FREEZE_INTENT}" \
  FREEZE_LATEST_CODE_CLAIM="${RUN_PROVENANCE_LATEST_CODE_CLAIM}" \
  FREEZE_LAUNCH_SOURCE="${RUN_PROVENANCE_LAUNCH_SOURCE}" \
  FREEZE_LAUNCH_SCRIPT="${RUN_PROVENANCE_LAUNCH_SCRIPT}" \
  FREEZE_GIT_COMMIT="$(_git_commit)" \
  FREEZE_GIT_BRANCH="$(_git_branch)" \
  FREEZE_WORKSPACE_DIRTY="${workspace_dirty_before_snapshot}" \
  FREEZE_PIP_FREEZE="${RUN_PROVENANCE_PIP_FREEZE}" \
  FREEZE_GIT_STATUS="${RUN_PROVENANCE_GIT_STATUS}" \
  FREEZE_GIT_DIFF="${RUN_PROVENANCE_GIT_DIFF}" \
  FREEZE_LAUNCH_ENV="${RUN_PROVENANCE_LAUNCH_ENV}" \
  FREEZE_STRATEGY_SNAPSHOT="${RUN_PROVENANCE_STRATEGY_SNAPSHOT}" \
  FREEZE_STRATEGY_LEARNING="${RUN_PROVENANCE_STRATEGY_LEARNING}" \
  FREEZE_FINAL_READINESS_MANIFEST_INDEX="${FINAL_READINESS_MANIFEST_INDEX}" \
  FREEZE_FINAL_READINESS_EXTRA_MANIFESTS="${FINAL_READINESS_EXTRA_MANIFESTS}" \
  FREEZE_FINAL_READINESS_FAIL_ON_MISSING="${FINAL_READINESS_FAIL_ON_MISSING}" \
  "${python_cmd}" - <<'PY'
import json
import os
from pathlib import Path

def env(name: str) -> str:
    return str(os.environ.get(name, "") or "")

payload = {
    "root_dir": env("FREEZE_ROOT_DIR"),
    "run_id": env("FREEZE_RUN_ID"),
    "session": env("FREEZE_SESSION"),
    "duration": env("FREEZE_DURATION"),
    "batch_duration": env("FREEZE_BATCH_DURATION"),
    "target_suites": env("FREEZE_TARGET_SUITES"),
    "presets": env("FREEZE_PRESETS"),
    "seeds": env("FREEZE_SEEDS"),
    "log_level": env("FREEZE_LOG_LEVEL"),
    "adaptive_config": {
        "profile_pool": env("FREEZE_PROFILE_POOL"),
        "profile_learning_weight": env("FREEZE_PROFILE_LEARNING_WEIGHT"),
        "semantic_objective_learning_weight": env("FREEZE_SEMANTIC_OBJECTIVE_LEARNING_WEIGHT"),
        "enable_metamorphic_oracle": env("FREEZE_ENABLE_METAMORPHIC_ORACLE") == "1",
        "metamorphic_relation_learning_weight": env("FREEZE_METAMORPHIC_RELATION_LEARNING_WEIGHT"),
        "metamorphic_relation_order": env("FREEZE_METAMORPHIC_RELATION_ORDER"),
        "version_pair_pool": env("FREEZE_VERSION_PAIR_POOL"),
        "version_pair_learning_weight": env("FREEZE_VERSION_PAIR_LEARNING_WEIGHT"),
        "adaptive_learning_weight": env("FREEZE_ADAPTIVE_LEARNING_WEIGHT"),
        "scheduler_annealing_temperature": env("FREEZE_SCHEDULER_ANNEALING_TEMPERATURE"),
        "scheduler_annealing_decay": env("FREEZE_SCHEDULER_ANNEALING_DECAY"),
        "scheduler_annealing_min_temperature": env("FREEZE_SCHEDULER_ANNEALING_MIN_TEMPERATURE"),
        "continual_learning_ledgers": env("FREEZE_CONTINUAL_LEARNING_LEDGERS"),
        "persist_closed_loop_state": env("FREEZE_PERSIST_CLOSED_LOOP_STATE") == "1",
    },
    "authority": env("FREEZE_AUTHORITY") == "1",
    "freeze_intent": env("FREEZE_FREEZE_INTENT") == "1",
    "latest_code_claim": env("FREEZE_LATEST_CODE_CLAIM") == "1",
    "launch_source": env("FREEZE_LAUNCH_SOURCE"),
    "launch_script": env("FREEZE_LAUNCH_SCRIPT"),
    "git_commit": env("FREEZE_GIT_COMMIT"),
    "git_branch": env("FREEZE_GIT_BRANCH"),
    "workspace_dirty": env("FREEZE_WORKSPACE_DIRTY") == "true",
    "artifacts": {
        "pip_freeze": env("FREEZE_PIP_FREEZE"),
        "git_status": env("FREEZE_GIT_STATUS"),
        "git_diff": env("FREEZE_GIT_DIFF"),
        "launcher_env": env("FREEZE_LAUNCH_ENV"),
        "strategy_snapshot": env("FREEZE_STRATEGY_SNAPSHOT"),
        "strategy_learning": env("FREEZE_STRATEGY_LEARNING"),
    },
    "post_run_readiness_config": {
        "manifest_index": env("FREEZE_FINAL_READINESS_MANIFEST_INDEX"),
        "extra_manifests": env("FREEZE_FINAL_READINESS_EXTRA_MANIFESTS"),
        "fail_on_missing": env("FREEZE_FINAL_READINESS_FAIL_ON_MISSING") == "1",
    },
}
Path(env("FREEZE_MANIFEST_PATH")).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

_build_experiment_command() {
  local python_cmd
  python_cmd="$(_python_cmd)"
  EXPERIMENT_COMMAND=(
    "${python_cmd}" -m datadiff.cli experiment
    --target-suites "${TARGET_SUITES}"
    --presets "${PRESETS}"
    --profile-pool "${PROFILE_POOL}"
    --profile-learning-weight "${PROFILE_LEARNING_WEIGHT}"
    --semantic-objective-learning-weight "${SEMANTIC_OBJECTIVE_LEARNING_WEIGHT}"
    --metamorphic-relation-learning-weight "${METAMORPHIC_RELATION_LEARNING_WEIGHT}"
    --version-pair-pool "${VERSION_PAIR_POOL}"
    --version-pair-learning-weight "${VERSION_PAIR_LEARNING_WEIGHT}"
    --seeds "${SEEDS}"
    --duration "${DURATION}"
    --schedule adaptive
    --batch-duration "${BATCH_DURATION}"
    --exploration-weight "${EXPLORATION_WEIGHT}"
    --group-fairness-weight "${GROUP_FAIRNESS_WEIGHT}"
    --max-group-pull-gap "${MAX_GROUP_PULL_GAP}"
    --adaptive-learning-weight "${ADAPTIVE_LEARNING_WEIGHT}"
    --scheduler-annealing-temperature "${SCHEDULER_ANNEALING_TEMPERATURE}"
    --scheduler-annealing-decay "${SCHEDULER_ANNEALING_DECAY}"
    --scheduler-annealing-min-temperature "${SCHEDULER_ANNEALING_MIN_TEMPERATURE}"
    --jobs "${JOBS}"
    --max-parallel-cost "${MAX_PARALLEL_COST}"
    --local-source-exploration-weight "${LOCAL_SOURCE_EXPLORATION_WEIGHT}"
    --artifact-limit "${ARTIFACT_LIMIT}"
    --log-level "${LOG_LEVEL}"
    --strategy-snapshot "${RUN_PROVENANCE_STRATEGY_SNAPSHOT}"
    --strategy-learning "${RUN_PROVENANCE_STRATEGY_LEARNING}"
    --freeze-strategy-snapshot
    --run-theme "${RUN_THEME}"
    --paper-notes "${PAPER_NOTES}"
    --persist-closed-loop-state
    --skip-run-reports
  )
  if [[ "${ENABLE_METAMORPHIC_ORACLE}" == "1" ]]; then
    EXPERIMENT_COMMAND+=(--enable-metamorphic-oracle)
  fi
  if [[ -n "${METAMORPHIC_RELATION_ORDER}" ]]; then
    EXPERIMENT_COMMAND+=(--metamorphic-relation-order "${METAMORPHIC_RELATION_ORDER}")
  fi
  if [[ -n "${CONTINUAL_LEARNING_LEDGERS}" ]]; then
    EXPERIMENT_COMMAND+=(--continual-learning-ledgers "${CONTINUAL_LEARNING_LEDGERS}")
  fi
}

_print_command() {
  if [[ -n "${COMMAND_OVERRIDE}" ]]; then
    printf '%q ' bash -lc "${COMMAND_OVERRIDE}"
    return
  fi
  _build_experiment_command
  printf '%q ' "${EXPERIMENT_COMMAND[@]}"
}

_start_command() {
  if [[ -n "${COMMAND_OVERRIDE}" ]]; then
    if command -v setsid >/dev/null 2>&1; then
      setsid bash -lc "${COMMAND_OVERRIDE}" &
    else
      bash -lc "${COMMAND_OVERRIDE}" &
    fi
    return
  fi
  _build_experiment_command
  if command -v setsid >/dev/null 2>&1; then
    setsid "${EXPERIMENT_COMMAND[@]}" &
  else
    "${EXPERIMENT_COMMAND[@]}" &
  fi
}

_terminate_process_group() {
  local pgid="${1:-}"
  local signal_name="${2:-TERM}"
  if [[ -z "${pgid}" ]]; then
    return
  fi
  kill "-${signal_name}" -- "-${pgid}" 2>/dev/null || true
}

_process_group_alive() {
  local pgid="${1:-}"
  if [[ -z "${pgid}" ]]; then
    return 1
  fi
  kill -0 -- "-${pgid}" 2>/dev/null
}

_watch_tmux_session() {
  local session_name="$1"
  local parent_pid="$2"
  local child_pid="$3"
  local interval="$4"
  while kill -0 "${child_pid}" 2>/dev/null; do
    if ! tmux has-session -t "${session_name}" 2>/dev/null; then
      kill -HUP "${parent_pid}" 2>/dev/null || true
      return
    fi
    sleep "${interval}"
  done
}

_launch_signature() {
  printf 'root=%s\n' "${ROOT_DIR}"
  if [[ -n "${COMMAND_OVERRIDE}" ]]; then
    printf 'mode=command_override\n'
    printf 'command=%s\n' "${COMMAND_OVERRIDE}"
    return
  fi
  printf 'mode=adaptive_experiment\n'
  printf 'target_suites=%s\n' "${TARGET_SUITES}"
  printf 'presets=%s\n' "${PRESETS}"
  printf 'profile_pool=%s\n' "${PROFILE_POOL}"
  printf 'profile_learning_weight=%s\n' "${PROFILE_LEARNING_WEIGHT}"
  printf 'semantic_objective_learning_weight=%s\n' "${SEMANTIC_OBJECTIVE_LEARNING_WEIGHT}"
  printf 'enable_metamorphic_oracle=%s\n' "${ENABLE_METAMORPHIC_ORACLE}"
  printf 'metamorphic_relation_learning_weight=%s\n' "${METAMORPHIC_RELATION_LEARNING_WEIGHT}"
  printf 'metamorphic_relation_order=%s\n' "${METAMORPHIC_RELATION_ORDER}"
  printf 'version_pair_pool=%s\n' "${VERSION_PAIR_POOL}"
  printf 'version_pair_learning_weight=%s\n' "${VERSION_PAIR_LEARNING_WEIGHT}"
  printf 'seeds=%s\n' "${SEEDS}"
  printf 'batch_duration=%s\n' "${BATCH_DURATION}"
  printf 'jobs=%s\n' "${JOBS}"
  printf 'max_parallel_cost=%s\n' "${MAX_PARALLEL_COST}"
  printf 'exploration_weight=%s\n' "${EXPLORATION_WEIGHT}"
  printf 'group_fairness_weight=%s\n' "${GROUP_FAIRNESS_WEIGHT}"
  printf 'max_group_pull_gap=%s\n' "${MAX_GROUP_PULL_GAP}"
  printf 'adaptive_learning_weight=%s\n' "${ADAPTIVE_LEARNING_WEIGHT}"
  printf 'scheduler_annealing_temperature=%s\n' "${SCHEDULER_ANNEALING_TEMPERATURE}"
  printf 'scheduler_annealing_decay=%s\n' "${SCHEDULER_ANNEALING_DECAY}"
  printf 'scheduler_annealing_min_temperature=%s\n' "${SCHEDULER_ANNEALING_MIN_TEMPERATURE}"
  printf 'continual_learning_ledgers=%s\n' "${CONTINUAL_LEARNING_LEDGERS}"
  printf 'persist_closed_loop_state=1\n'
  printf 'local_source_exploration_weight=%s\n' "${LOCAL_SOURCE_EXPLORATION_WEIGHT}"
  printf 'artifact_limit=%s\n' "${ARTIFACT_LIMIT}"
  printf 'log_level=%s\n' "${LOG_LEVEL}"
}

_extract_output_field() {
  local text="$1"
  local prefix="$2"
  local line
  while IFS= read -r line; do
    if [[ "${line}" == "${prefix}"* ]]; then
      printf '%s\n' "${line#${prefix}}"
      return 0
    fi
  done <<< "${text}"
  return 1
}

_extract_experiment_manifest_from_log() {
  local log_file="$1"
  local line=""
  if [[ ! -f "${log_file}" ]]; then
    return 1
  fi
  while IFS= read -r line; do
    if [[ "${line}" == "experiment manifest: "* ]]; then
      EXPERIMENT_MANIFEST_PATH="${line#experiment manifest: }"
    fi
  done < "${log_file}"
  [[ -n "${EXPERIMENT_MANIFEST_PATH}" ]]
}

_jsonl_log_stem() {
  local path="$1"
  local name
  name="$(basename "${path}")"
  if [[ "${name}" == *.jsonl.gz ]]; then
    printf '%s\n' "${name%.jsonl.gz}"
    return
  fi
  if [[ "${name}" == *.jsonl ]]; then
    printf '%s\n' "${name%.jsonl}"
    return
  fi
  printf '%s\n' "${name%.*}"
}

_manifest_run_files() {
  local manifest_path="$1"
  local python_cmd
  python_cmd="$(_python_cmd)"
  "${python_cmd}" - "${manifest_path}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
payload = json.loads(manifest_path.read_text(encoding="utf-8"))
for run in payload.get("runs", []):
    run_file = str(run.get("run_file", "") or "").strip()
    if run_file:
        print(run_file)
PY
}

_ensure_final_readiness_manifest_index() {
  local manifest_path="$1"
  local index_path="${FINAL_READINESS_MANIFEST_INDEX%%,*}"
  local python_cmd
  if [[ -z "${index_path}" ]]; then
    return 0
  fi
  python_cmd="$(_python_cmd)"
  INDEX_PATH="${index_path}" \
  MANIFEST_PATH="${manifest_path}" \
  RUN_ID_VALUE="${run_id}" \
  LOG_FILE_VALUE="${log_file}" \
  SESSION_VALUE="${SESSION_NAME}" \
  DURATION_VALUE="${DURATION}" \
  BATCH_DURATION_VALUE="${BATCH_DURATION}" \
  EVIDENCE_ROLE_VALUE="${RUN_PROVENANCE_EVIDENCE_ROLE}" \
  LAUNCH_SCRIPT_VALUE="${RUN_PROVENANCE_LAUNCH_SCRIPT}" \
  "${python_cmd}" - <<'PY'
import json
import os
from pathlib import Path
from datetime import datetime, timezone

def env(name: str) -> str:
    return str(os.environ.get(name, "") or "")

def unique(values):
    seen = set()
    out = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out

index_path = Path(env("INDEX_PATH"))
manifest_path = env("MANIFEST_PATH")
now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
if index_path.exists():
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        payload = {}
else:
    payload = {}
payload.setdefault("schema_version", "final-experiment-manifest-index-v1")
payload.setdefault("created_at", now)
payload["updated_at"] = now
payload["manifest_files"] = unique([*payload.get("manifest_files", []), manifest_path])
payload["extra_manifest_files"] = unique(payload.get("extra_manifest_files", []))
commands = payload.setdefault("commands", [])
if not isinstance(commands, list):
    commands = []
    payload["commands"] = commands
shell = f"closed-loop-launcher {env('LAUNCH_SCRIPT_VALUE')} run_id={env('RUN_ID_VALUE')}"
if not any(str(item.get("shell", "")) == shell for item in commands if isinstance(item, dict)):
    commands.append(
        {
            "track": "live",
            "name": "closed_loop_authority_run",
            "status": "completed",
            "returncode": 0,
            "manifest_files": [manifest_path],
            "extra_manifest_files": [],
            "final_readiness_files": [],
            "shell": shell,
            "launcher": {
                "run_id": env("RUN_ID_VALUE"),
                "session": env("SESSION_VALUE"),
                "duration": env("DURATION_VALUE"),
                "batch_duration": env("BATCH_DURATION_VALUE"),
                "log_file": env("LOG_FILE_VALUE"),
                "evidence_role": env("EVIDENCE_ROLE_VALUE"),
            },
        }
    )
index_path.parent.mkdir(parents=True, exist_ok=True)
index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

_append_csv_flags() {
  local -n command_ref="$1"
  local flag="$2"
  local csv_value="$3"
  local item
  IFS=',' read -ra items <<< "${csv_value}"
  for item in "${items[@]}"; do
    item="${item#"${item%%[![:space:]]*}"}"
    item="${item%"${item##*[![:space:]]}"}"
    if [[ -n "${item}" ]]; then
      command_ref+=("${flag}" "${item}")
    fi
  done
}

_final_readiness_command() {
  local manifest_path="$1"
  local python_cmd
  python_cmd="$(_python_cmd)"
  FINAL_READINESS_COMMAND=("${python_cmd}" -m datadiff.cli final-readiness)
  if [[ -n "${FINAL_READINESS_MANIFEST_INDEX}" ]]; then
    _append_csv_flags FINAL_READINESS_COMMAND "--manifest-index" "${FINAL_READINESS_MANIFEST_INDEX}"
  else
    FINAL_READINESS_COMMAND+=(--manifest "${manifest_path}")
  fi
  if [[ -n "${FINAL_READINESS_EXTRA_MANIFESTS}" ]]; then
    _append_csv_flags FINAL_READINESS_COMMAND "--extra-manifest" "${FINAL_READINESS_EXTRA_MANIFESTS}"
  fi
  if [[ "${FINAL_READINESS_FAIL_ON_MISSING}" == "1" ]]; then
    FINAL_READINESS_COMMAND+=(--fail-on-missing)
  fi
}

_apply_post_run_hook_output() {
  local hook_output="$1"
  local line
  while IFS= read -r line; do
    case "${line}" in
      post_run_status=*) POST_RUN_STATUS="${line#post_run_status=}" ;;
      experiment_manifest=*) EXPERIMENT_MANIFEST_PATH="${line#experiment_manifest=}" ;;
      experiment_summary_markdown=*) EXPERIMENT_SUMMARY_MARKDOWN="${line#experiment_summary_markdown=}" ;;
      experiment_summary_csv=*) EXPERIMENT_SUMMARY_CSV="${line#experiment_summary_csv=}" ;;
      experiment_summary_aggregate_csv=*) EXPERIMENT_SUMMARY_AGGREGATE_CSV="${line#experiment_summary_aggregate_csv=}" ;;
      experiment_analysis_markdown=*) EXPERIMENT_ANALYSIS_MARKDOWN="${line#experiment_analysis_markdown=}" ;;
      experiment_analysis_csv=*) EXPERIMENT_ANALYSIS_CSV="${line#experiment_analysis_csv=}" ;;
      methodology_report_markdown=*) METHODOLOGY_REPORT_MARKDOWN="${line#methodology_report_markdown=}" ;;
      methodology_report_json=*) METHODOLOGY_REPORT_JSON="${line#methodology_report_json=}" ;;
      final_readiness_markdown=*) FINAL_READINESS_MARKDOWN="${line#final_readiness_markdown=}" ;;
      final_readiness_json=*) FINAL_READINESS_JSON="${line#final_readiness_json=}" ;;
      classify_run_dir=*) CLASSIFY_RUN_DIR="${line#classify_run_dir=}" ;;
      classify_run_count=*) CLASSIFY_RUN_COUNT="${line#classify_run_count=}" ;;
    esac
  done <<< "${hook_output}"
}

_refresh_paper_evidence() {
  local manifest_path="$1"
  local python_cmd=""
  local summary_output=""
  local analysis_output=""
  local methodology_output=""
  local readiness_output=""
  local classify_output=""
  local run_file=""
  local run_stem=""
  local classify_target=""
  local hook_output=""
  local hook_rc=0
  local run_count=0

  EXPERIMENT_MANIFEST_PATH="${manifest_path}"

  if [[ -n "${POST_RUN_EVIDENCE_HOOK}" ]]; then
    export DATADIFF_POST_RUN_MANIFEST="${manifest_path}"
    export DATADIFF_POST_RUN_ROOT_DIR="${ROOT_DIR}"
    export DATADIFF_POST_RUN_RUN_ID="${run_id}"
    export DATADIFF_POST_RUN_LOG_FILE="${log_file}"
    set +e
    hook_output="$(bash -lc "${POST_RUN_EVIDENCE_HOOK}" 2>&1)"
    hook_rc=$?
    set -e
    printf '%s\n' "${hook_output}"
    if [[ "${hook_rc}" != "0" ]]; then
      POST_RUN_STATUS="failed"
      return "${hook_rc}"
    fi
    _apply_post_run_hook_output "${hook_output}"
    if [[ -z "${POST_RUN_STATUS}" ]]; then
      POST_RUN_STATUS="ok"
    fi
    return 0
  fi

  python_cmd="$(_python_cmd)"
  if [[ -z "${python_cmd}" ]]; then
    echo "post-run evidence refresh requires a Python executable"
    POST_RUN_STATUS="failed"
    return 5
  fi

  summary_output="$("${python_cmd}" -m datadiff.cli experiment-summary --manifest "${manifest_path}" --refresh)"
  printf '%s\n' "${summary_output}"
  EXPERIMENT_SUMMARY_MARKDOWN="$(_extract_output_field "${summary_output}" "markdown summary: " || true)"
  EXPERIMENT_SUMMARY_CSV="$(_extract_output_field "${summary_output}" "csv summary:      " || true)"
  EXPERIMENT_SUMMARY_AGGREGATE_CSV="$(_extract_output_field "${summary_output}" "aggregate csv:    " || true)"

  analysis_output="$("${python_cmd}" -m datadiff.cli analyze-experiment --manifest "${manifest_path}" --refresh)"
  printf '%s\n' "${analysis_output}"
  EXPERIMENT_ANALYSIS_MARKDOWN="$(_extract_output_field "${analysis_output}" "analysis markdown: " || true)"
  EXPERIMENT_ANALYSIS_CSV="$(_extract_output_field "${analysis_output}" "analysis csv:      " || true)"

  methodology_output="$("${python_cmd}" -m datadiff.cli methodology-report --manifest "${manifest_path}" --refresh)"
  printf '%s\n' "${methodology_output}"
  METHODOLOGY_REPORT_MARKDOWN="$(_extract_output_field "${methodology_output}" "methodology report markdown: " || true)"
  METHODOLOGY_REPORT_JSON="$(_extract_output_field "${methodology_output}" "methodology report json:     " || true)"

  _ensure_final_readiness_manifest_index "${manifest_path}"
  _final_readiness_command "${manifest_path}"
  readiness_output="$("${FINAL_READINESS_COMMAND[@]}")"
  printf '%s\n' "${readiness_output}"
  FINAL_READINESS_MARKDOWN="$(_extract_output_field "${readiness_output}" "final readiness markdown: " || true)"
  FINAL_READINESS_JSON="$(_extract_output_field "${readiness_output}" "final readiness json:     " || true)"

  CLASSIFY_RUN_DIR="${ROOT_DIR}/reports/classify-run-$(basename "${manifest_path%.*}")"
  mkdir -p "${CLASSIFY_RUN_DIR}"
  while IFS= read -r run_file; do
    if [[ -z "${run_file}" ]]; then
      continue
    fi
    run_stem="$(_jsonl_log_stem "${run_file}")"
    classify_target="${CLASSIFY_RUN_DIR}/${run_stem}.json"
    classify_output="$("${python_cmd}" -m datadiff.cli classify-run --run-file "${run_file}" --json)"
    printf '%s\n' "${classify_output}" > "${classify_target}"
    run_count=$((run_count + 1))
  done < <(_manifest_run_files "${manifest_path}")
  CLASSIFY_RUN_COUNT="${run_count}"
  POST_RUN_STATUS="ok"
}

_launch_signature_from_config_file() {
  local config_file="$1"
  (
    # shellcheck disable=SC1090
    source "${config_file}"
    refresh_config
    _launch_signature
  )
}

_session_name_from_config_file() {
  local config_file="$1"
  (
    # shellcheck disable=SC1090
    source "${config_file}"
    refresh_config
    printf '%s\n' "${SESSION_NAME}"
  )
}

_find_equivalent_running_launch() {
  local current_signature="$1"
  local logs_dir="${ROOT_DIR}/logs"
  local env_file
  local other_signature
  local other_session
  if [[ ! -d "${logs_dir}" ]]; then
    return 1
  fi
  shopt -s nullglob
  for env_file in "${logs_dir}"/*.env; do
    other_signature="$(_launch_signature_from_config_file "${env_file}" 2>/dev/null || true)"
    if [[ -z "${other_signature}" || "${other_signature}" != "${current_signature}" ]]; then
      continue
    fi
    other_session="$(_session_name_from_config_file "${env_file}" 2>/dev/null || true)"
    if [[ -z "${other_session}" ]]; then
      continue
    fi
    if tmux has-session -t "${other_session}" 2>/dev/null; then
      printf '%s|%s\n' "${other_session}" "${env_file}"
      shopt -u nullglob
      return 0
    fi
  done
  shopt -u nullglob
  return 1
}

run_child() {
  local run_id="$1"
  local config_file="${2:-}"
  if [[ -n "${config_file}" && -f "${config_file}" ]]; then
    # Load the launch configuration explicitly so multiple sessions started from
    # the same tmux server do not inherit stale environment values.
    # shellcheck disable=SC1090
    source "${config_file}"
    refresh_config
  fi
  export DATADIFF_RUN_PROVENANCE_AUTHORITY="${RUN_PROVENANCE_AUTHORITY}"
  export DATADIFF_RUN_PROVENANCE_FREEZE_INTENT="${RUN_PROVENANCE_FREEZE_INTENT}"
  export DATADIFF_RUN_PROVENANCE_LATEST_CODE_CLAIM="${RUN_PROVENANCE_LATEST_CODE_CLAIM}"
  export DATADIFF_RUN_PROVENANCE_EVIDENCE_ROLE="${RUN_PROVENANCE_EVIDENCE_ROLE}"
  export DATADIFF_RUN_PROVENANCE_LAUNCH_SOURCE="${RUN_PROVENANCE_LAUNCH_SOURCE}"
  export DATADIFF_RUN_PROVENANCE_SESSION="${SESSION_NAME}"
  export DATADIFF_RUN_PROVENANCE_DURATION="${DURATION}"
  export DATADIFF_RUN_PROVENANCE_BATCH_DURATION="${BATCH_DURATION}"
  export DATADIFF_RUN_PROVENANCE_LOG_PREFIX="${LOG_PREFIX}"
  export DATADIFF_RUN_PROVENANCE_LAUNCH_SCRIPT="${RUN_PROVENANCE_LAUNCH_SCRIPT}"
  export DATADIFF_RUN_PROVENANCE_FREEZE_MANIFEST="${RUN_PROVENANCE_FREEZE_MANIFEST}"
  export DATADIFF_RUN_PROVENANCE_PIP_FREEZE="${RUN_PROVENANCE_PIP_FREEZE}"
  export DATADIFF_RUN_PROVENANCE_GIT_STATUS="${RUN_PROVENANCE_GIT_STATUS}"
  export DATADIFF_RUN_PROVENANCE_GIT_DIFF="${RUN_PROVENANCE_GIT_DIFF}"
  export DATADIFF_RUN_PROVENANCE_LAUNCH_ENV="${RUN_PROVENANCE_LAUNCH_ENV}"
  export DATADIFF_RUN_PROVENANCE_STRATEGY_SNAPSHOT="${RUN_PROVENANCE_STRATEGY_SNAPSHOT}"
  export DATADIFF_RUN_PROVENANCE_STRATEGY_LEARNING="${RUN_PROVENANCE_STRATEGY_LEARNING}"
  local log_file="${ROOT_DIR}/logs/${LOG_PREFIX}-${run_id}.log"
  local status_file="${ROOT_DIR}/logs/${LOG_PREFIX}-${run_id}.status"
  local session_label="${SESSION_NAME}"
  local interrupted=0
  local interrupted_signal=""
  local cmd_rc=0
  local cmd_pid=""
  local cmd_pgid=""
  local watch_pid=""
  local postprocess_failed=0
  if command -v tmux >/dev/null 2>&1; then
    session_label="$(tmux display-message -p '#S' 2>/dev/null || printf '%s' "${SESSION_NAME}")"
  fi

  cd "${ROOT_DIR}"
  mkdir -p logs runs reports bugs corpus
  write_status_file "${status_file}" "running" "${run_id}" "${log_file}" "started_at"
  exec > >(tee -a "${log_file}") 2>&1
  on_signal() {
    interrupted=1
    interrupted_signal="$1"
    cmd_rc="$2"
    _terminate_process_group "${cmd_pgid:-$cmd_pid}" TERM
  }
  cleanup() {
    local final_status="completed"
    local final_code="${cmd_rc}"
    if [[ -n "${watch_pid}" ]]; then
      kill "${watch_pid}" 2>/dev/null || true
      wait "${watch_pid}" 2>/dev/null || true
    fi
    if _process_group_alive "${cmd_pgid:-$cmd_pid}"; then
      _terminate_process_group "${cmd_pgid:-$cmd_pid}" TERM
      sleep 1
      if _process_group_alive "${cmd_pgid:-$cmd_pid}"; then
        _terminate_process_group "${cmd_pgid:-$cmd_pid}" KILL
      fi
    fi
    if [[ -n "${cmd_pid}" ]]; then
      wait "${cmd_pid}" 2>/dev/null || true
    fi
    if [[ "${interrupted}" == "1" ]]; then
      final_status="interrupted_${interrupted_signal}"
      if [[ "${final_code}" == "0" ]]; then
        final_code="130"
      fi
    elif [[ "${postprocess_failed}" == "1" ]]; then
      final_status="postprocess_failed"
    elif [[ "${final_code}" != "0" ]]; then
      final_status="failed"
    fi
    write_status_file "${status_file}" "${final_status}" "${run_id}" "${log_file}" "finished_at" "${final_code}"
    if [[ -n "${config_file}" && -f "${config_file}" ]]; then
      rm -f "${config_file}"
    fi
  }
  trap 'on_signal "sigint" 130' INT
  trap 'on_signal "sigterm" 143' TERM
  trap 'on_signal "sighup" 129' HUP
  trap cleanup EXIT

  echo "run_id=${run_id}"
  echo "started_at=$(date -Is)"
  echo "root=${ROOT_DIR}"
  echo "session=${session_label}"
  echo
  echo "[command]"
  _print_command
  echo
  echo

  set +e
  _start_command
  cmd_pid=$!
  cmd_pgid="$(ps -o pgid= -p "${cmd_pid}" 2>/dev/null | tr -d '[:space:]')"
  if [[ -z "${cmd_pgid}" ]]; then
    cmd_pgid="${cmd_pid}"
  fi
  if [[ -n "${TMUX:-}" ]] && command -v tmux >/dev/null 2>&1; then
    _watch_tmux_session "${SESSION_NAME}" "$$" "${cmd_pid}" "${TMUX_WATCH_INTERVAL}" &
    watch_pid=$!
  fi
  wait "${cmd_pid}"
  cmd_rc=$?
  set -e
  if [[ "${cmd_rc}" == "0" && "${interrupted}" == "0" ]]; then
    if _extract_experiment_manifest_from_log "${log_file}"; then
      echo
      echo "[post-run-evidence]"
      echo "experiment_manifest=${EXPERIMENT_MANIFEST_PATH}"
      if ! _refresh_paper_evidence "${EXPERIMENT_MANIFEST_PATH}"; then
        cmd_rc=$?
        postprocess_failed=1
      fi
    else
      POST_RUN_STATUS="skipped"
    fi
  elif [[ -z "${POST_RUN_STATUS}" ]]; then
    POST_RUN_STATUS="skipped"
  fi
  echo
  echo "finished_at=$(date -Is)"
  echo "datadiff_exit_code=${cmd_rc}"
  exit "${cmd_rc}"
}

if [[ "${1:-}" == "--child" ]]; then
  run_child "${2:?missing run id}" "${3:-}"
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION_NAME}"
  echo "attach: tmux attach -t ${SESSION_NAME}"
  exit 2
fi

if [[ "${ALLOW_EQUIVALENT_CONCURRENT_LAUNCH}" != "1" ]]; then
  current_signature="$(_launch_signature)"
  if running_match="$(_find_equivalent_running_launch "${current_signature}")"; then
    IFS='|' read -r existing_session existing_config <<< "${running_match}"
    echo "equivalent long-run already active: ${existing_session}"
    echo "config: ${existing_config}"
    echo "override by setting DATADIFF_ALLOW_EQUIVALENT_CONCURRENT_LAUNCH=1 if you truly want concurrent duplicate coverage"
    exit 3
  fi
fi

RUN_ID="$(date +%Y%m%dT%H%M%S%z)-$$-$RANDOM"
_require_clean_authority_workspace || exit $?
_prepare_freeze_snapshot "${RUN_ID}"
CONFIG_FILE="${ROOT_DIR}/logs/${LOG_PREFIX}-${RUN_ID}.env"
write_config_file "${CONFIG_FILE}"
printf -v child_cmd 'bash %q --child %q %q' "${BASH_SOURCE[0]}" "${RUN_ID}" "${CONFIG_FILE}"
tmux new-session -d -s "${SESSION_NAME}" "${child_cmd}"

echo "started tmux session: ${SESSION_NAME}"
echo "run_id: ${RUN_ID}"
echo "log: ${ROOT_DIR}/logs/${LOG_PREFIX}-${RUN_ID}.log"
echo "status: ${ROOT_DIR}/logs/${LOG_PREFIX}-${RUN_ID}.status"
