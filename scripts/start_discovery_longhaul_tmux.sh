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
SEED_START="${DATADIFF_DISCOVERY_LONGHAUL_SEED_START:-31000001}"
LOG_LEVEL="${DATADIFF_DISCOVERY_LONGHAUL_LOG_LEVEL:-compact}"
ARTIFACT_LIMIT="${DATADIFF_DISCOVERY_LONGHAUL_ARTIFACT_LIMIT:-20}"
CLASSIFY_LIMIT="${DATADIFF_DISCOVERY_LONGHAUL_CLASSIFY_LIMIT:-5}"
CANDIDATE_RECHECK_COUNT="${DATADIFF_DISCOVERY_LONGHAUL_CANDIDATE_RECHECK_COUNT:-3}"
CANDIDATE_PIPELINE_RECHECK_ATTEMPTS="${DATADIFF_DISCOVERY_LONGHAUL_CANDIDATE_PIPELINE_RECHECK_ATTEMPTS:-3}"
GENERATED_DIR="${DATADIFF_DISCOVERY_LONGHAUL_GENERATED_DIR:-new_issue/generated}"
LOG_DIR="${DATADIFF_DISCOVERY_LONGHAUL_LOG_DIR:-logs}"
STATUS_FILE="${DATADIFF_DISCOVERY_LONGHAUL_STATUS_FILE:-${LOG_DIR}/discovery-longhaul-${RUN_ID}.status}"
AGGREGATE_FILE="${DATADIFF_DISCOVERY_LONGHAUL_AGGREGATE_FILE:-${GENERATED_DIR}/discovery-campaign-longhaul-${RUN_ID}-aggregate.json}"
AGGREGATE_MANIFESTS="${DATADIFF_DISCOVERY_LONGHAUL_AGGREGATE_MANIFESTS:-${GENERATED_DIR}/discovery-campaign-longhaul-${RUN_ID}-batch*.json}"
PYTHON_BIN="${DATADIFF_PYTHON:-${ROOT_DIR}/.venv/bin/python}"
REQUIRE_LATEST_TARGETS="${DATADIFF_DISCOVERY_LONGHAUL_REQUIRE_LATEST_TARGETS:-1}"
TARGET_VERSION_AUDIT_FILE="${DATADIFF_DISCOVERY_LONGHAUL_TARGET_VERSION_AUDIT_FILE:-reports/target-version-audits/discovery-longhaul-${RUN_ID}-target-version-audit.json}"
TARGET_VERSION_AUDIT_SOURCE="${DATADIFF_DISCOVERY_LONGHAUL_TARGET_VERSION_AUDIT_SOURCE:-}"
PROVENANCE_DIR="${DATADIFF_DISCOVERY_LONGHAUL_PROVENANCE_DIR:-reports/discovery-longhaul-provenance}"
FREEZE_MANIFEST="${DATADIFF_DISCOVERY_LONGHAUL_FREEZE_MANIFEST:-${PROVENANCE_DIR}/discovery-longhaul-${RUN_ID}-freeze-manifest.json}"
PIP_FREEZE="${DATADIFF_DISCOVERY_LONGHAUL_PIP_FREEZE:-${PROVENANCE_DIR}/discovery-longhaul-${RUN_ID}-pip-freeze.txt}"
GIT_STATUS="${DATADIFF_DISCOVERY_LONGHAUL_GIT_STATUS:-${PROVENANCE_DIR}/discovery-longhaul-${RUN_ID}-git-status.txt}"
GIT_DIFF="${DATADIFF_DISCOVERY_LONGHAUL_GIT_DIFF:-${PROVENANCE_DIR}/discovery-longhaul-${RUN_ID}-git-diff.patch}"
AUTHORITY="${DATADIFF_DISCOVERY_LONGHAUL_AUTHORITY:-0}"
REQUIRE_CLEAN_WORKTREE="${DATADIFF_DISCOVERY_LONGHAUL_REQUIRE_CLEAN_WORKTREE:-${AUTHORITY}}"
LANE_GROUPS_RAW="${DATADIFF_DISCOVERY_LONGHAUL_LANE_GROUPS:-pandas_targeted_boundaries;polars_targeted_boundaries;datafusion_targeted_boundaries;chdb_targeted_boundaries;orthogonal_stress;arrow_layout,polars_streaming;embedded_sql,duckdb_storage;common_api_workflow,cross_family}"
FOREGROUND=0
PRINT_CONFIG=0
WORKER=0
DRY_RUN=0

IFS=';' read -r -a LANE_GROUPS <<< "${LANE_GROUPS_RAW}"

usage() {
  cat <<'EOF'
usage: start_discovery_longhaul_tmux.sh [--foreground] [--worker] [--dry-run] [--print-config]

Starts a 12h/24h discovery-campaign longhaul controller. The controller launches
focused discovery-campaign batches only while active campaign count and system
load are below configured limits, then refreshes the aggregate manifest.

Key environment variables:
  DATADIFF_DISCOVERY_LONGHAUL_DURATION=12h
  DATADIFF_DISCOVERY_LONGHAUL_MAX_CONCURRENT=6
  DATADIFF_DISCOVERY_LONGHAUL_LOAD_LIMIT=<defaults to nproc * 1.35>
  DATADIFF_DISCOVERY_LONGHAUL_BATCH_CASES=220
  DATADIFF_DISCOVERY_LONGHAUL_SESSION=<tmux session>
  DATADIFF_DISCOVERY_LONGHAUL_AUTHORITY=1
  DATADIFF_DISCOVERY_LONGHAUL_REQUIRE_CLEAN_WORKTREE=1
  DATADIFF_DISCOVERY_LONGHAUL_LANE_GROUPS=<semicolon-separated lane groups>
  DATADIFF_DISCOVERY_LONGHAUL_TARGET_VERSION_AUDIT_SOURCE=<validated prior audit>
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --foreground) FOREGROUND=1 ;;
    --worker) WORKER=1 ;;
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
  printf '%s\n' "${rows}" | awk '
    {
      sub(/^[0-9]+[[:space:]]+/, "")
      if ($0 ~ /^([^[:space:]]*\/)?python[0-9.]*[[:space:]]+-m[[:space:]]+datadiff[.]cli[[:space:]]+discovery-campaign/) {
        count += 1
      }
    }
    END { print count + 0 }
  '
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
    printf 'aggregate_manifests=%s\n' "${AGGREGATE_MANIFESTS}"
    printf 'target_version_audit=%s\n' "${TARGET_VERSION_AUDIT_FILE}"
    printf 'freeze_manifest=%s\n' "${FREEZE_MANIFEST}"
    printf 'authority=%s\n' "${AUTHORITY}"
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
  printf 'seed_start=%s\n' "${SEED_START}"
  printf 'candidate_recheck_count=%s\n' "${CANDIDATE_RECHECK_COUNT}"
  printf 'candidate_pipeline_recheck_attempts=%s\n' "${CANDIDATE_PIPELINE_RECHECK_ATTEMPTS}"
  printf 'aggregate_file=%s\n' "${AGGREGATE_FILE}"
  printf 'aggregate_manifests=%s\n' "${AGGREGATE_MANIFESTS}"
  printf 'python=%s\n' "${PYTHON_BIN}"
  printf 'require_latest_targets=%s\n' "${REQUIRE_LATEST_TARGETS}"
  printf 'target_version_audit=%s\n' "${TARGET_VERSION_AUDIT_FILE}"
  printf 'target_version_audit_source=%s\n' "${TARGET_VERSION_AUDIT_SOURCE}"
  printf 'provenance_dir=%s\n' "${PROVENANCE_DIR}"
  printf 'freeze_manifest=%s\n' "${FREEZE_MANIFEST}"
  printf 'authority=%s\n' "${AUTHORITY}"
  printf 'require_clean_worktree=%s\n' "${REQUIRE_CLEAN_WORKTREE}"
  printf 'lane_groups=%s\n' "$(IFS=';'; echo "${LANE_GROUPS[*]}")"
}

run_latest_target_audit() {
  if [[ "${REQUIRE_LATEST_TARGETS}" != "1" ]]; then
    return
  fi
  mkdir -p "${ROOT_DIR}/$(dirname "${TARGET_VERSION_AUDIT_FILE}")"
  (
    cd "${ROOT_DIR}"
    if [[ -n "${TARGET_VERSION_AUDIT_SOURCE}" ]]; then
      "${PYTHON_BIN}" - "${TARGET_VERSION_AUDIT_SOURCE}" "${TARGET_VERSION_AUDIT_FILE}" <<'PY'
import sys
from pathlib import Path

root = Path.cwd()
source = Path(sys.argv[1])
target = Path(sys.argv[2])
if not source.is_absolute():
    source = root / source
if not target.is_absolute():
    target = root / target
if not source.is_file():
    raise SystemExit(f"target version audit source does not exist: {source}")
target.parent.mkdir(parents=True, exist_ok=True)
if source.resolve() != target.resolve():
    target.write_bytes(source.read_bytes())
print(f"reused target version audit: {source} -> {target}")
PY
    else
      "${PYTHON_BIN}" -m datadiff.cli target-version-audit \
        --output "${TARGET_VERSION_AUDIT_FILE}" \
        --json > "${TARGET_VERSION_AUDIT_FILE}.stdout.json"
    fi
    "${PYTHON_BIN}" - "${TARGET_VERSION_AUDIT_FILE}" <<'PY'
from importlib.metadata import PackageNotFoundError, version
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
summary = payload.get("summary", {})
failed = summary.get("all_target_packages_up_to_date") is not True
for row in payload.get("target_packages", []):
    package = str(row.get("package", ""))
    audited_installed = str(row.get("installed_version", ""))
    try:
        runtime_installed = version(package)
    except PackageNotFoundError:
        runtime_installed = ""
    if row.get("up_to_date") is not True or runtime_installed != audited_installed:
        failed = True
        print(
            "package={package} audited_installed={audited} runtime_installed={runtime} "
            "latest={latest} up_to_date={up_to_date} source={source}".format(
                package=package,
                audited=audited_installed or "?",
                runtime=runtime_installed or "?",
                latest=row.get("latest_version") or "?",
                up_to_date=row.get("up_to_date"),
                source=row.get("latest_source") or "?",
            ),
            file=sys.stderr,
        )
if failed:
    print(f"latest target audit failed or no longer matches the runtime: {path}", file=sys.stderr)
    raise SystemExit(2)
print(f"latest target audit passed and matches runtime: {path}")
PY
  )
}

prepare_freeze_manifest() {
  local root_freeze_manifest="${ROOT_DIR}/${FREEZE_MANIFEST}"
  local root_pip_freeze="${ROOT_DIR}/${PIP_FREEZE}"
  local root_git_status="${ROOT_DIR}/${GIT_STATUS}"
  local root_git_diff="${ROOT_DIR}/${GIT_DIFF}"
  if [[ "${REQUIRE_CLEAN_WORKTREE}" == "1" ]] && [[ -n "$(git -C "${ROOT_DIR}" status --porcelain)" ]]; then
    echo "refusing authority discovery launch from a dirty workspace" >&2
    git -C "${ROOT_DIR}" status --short >&2
    exit 2
  fi
  mkdir -p "${ROOT_DIR}/${PROVENANCE_DIR}"
  "${PYTHON_BIN}" -m pip freeze > "${root_pip_freeze}"
  git -C "${ROOT_DIR}" status --short --branch > "${root_git_status}"
  git -C "${ROOT_DIR}" diff --binary -- src scripts pyproject.toml requirements-final.lock \
    > "${root_git_diff}"
  (
    cd "${ROOT_DIR}"
    "${PYTHON_BIN}" - \
      "${root_freeze_manifest}" \
      "${RUN_ID}" \
      "${DURATION}" \
      "${BATCH_CASES}" \
      "${MAX_CONCURRENT}" \
      "${SEED_START}" \
      "${CANDIDATE_RECHECK_COUNT}" \
      "${CANDIDATE_PIPELINE_RECHECK_ATTEMPTS}" \
      "${LANE_GROUPS_RAW}" \
      "${AUTHORITY}" \
      "${root_pip_freeze}" \
      "${root_git_status}" \
      "${root_git_diff}" \
      "${TARGET_VERSION_AUDIT_FILE}" <<'PY'
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from datadiff.env import collect_environment
from datadiff.strategy_registry import discovery_lane_catalog


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""


def canonical_sha256(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


root = Path.cwd()
out = Path(sys.argv[1])
lane_groups = [group for group in sys.argv[9].split(";") if group]
lane_ids = [lane for group in lane_groups for lane in group.split(",") if lane]
catalog = discovery_lane_catalog()
unknown = sorted(set(lane_ids) - set(catalog))
if unknown:
    raise SystemExit(f"unknown frozen discovery lanes: {','.join(unknown)}")
git_status = Path(sys.argv[12])
git_diff = Path(sys.argv[13])
canonical_manifest = root / "experiments/canonical_confirmed_bug_corpus/v2/manifest.json"
canonical = json.loads(canonical_manifest.read_text(encoding="utf-8"))
confirmed_roots = len(canonical.get("confirmed_roots", []))
payload = {
    "schema_version": "final-bug-discovery-freeze-v1",
    "run_id": sys.argv[2],
    "frozen_at": datetime.now(timezone.utc).isoformat(),
    "objective": {
        "target_confirmed_root_count": 30,
        "confirmed_root_count_at_freeze": confirmed_roots,
        "remaining_root_gap_at_freeze": max(0, 30 - confirmed_roots),
        "primary_metric": "independently_confirmed_unique_root_causes",
    },
    "counting_policy": {
        "fresh_latest_version_only": True,
        "require_stable_recheck": True,
        "require_native_reproducer": True,
        "require_upstream_dedup": True,
        "require_independent_upstream_confirmation_for_confirmed_count": True,
        "known_submitted_saturated_replay_and_issue_inspired_do_not_count_as_fresh": True,
        "candidate_rows_and_signatures_do_not_count_as_roots": True,
    },
    "design": {
        "duration": sys.argv[3],
        "batch_cases_per_lane_seed": int(sys.argv[4]),
        "max_concurrent_campaigns": int(sys.argv[5]),
        "seed_start": int(sys.argv[6]),
        "seed_schedule": "batch b uses seed_start + 200*b and seed_start + 200*b + 101",
        "candidate_recheck_count": int(sys.argv[7]),
        "candidate_pipeline_recheck_attempts": int(sys.argv[8]),
        "lane_groups": lane_groups,
        "lane_specs": {lane: catalog[lane] for lane in lane_ids},
        "no_favorable_early_stopping": True,
        "health_stops_only": True,
        "replay_bug_enabled": False,
        "formal_seed_range_excludes": [
            {
                "first": 2026071800,
                "last": 2026071805,
                "reason": "pre-freeze chDB compatibility smoke",
            },
            {
                "first": 30000001,
                "last": 30000113,
                "reason": "pre-freeze five-lane validation smoke",
            },
            {
                "first": 30500001,
                "last": 30500102,
                "reason": "candidate-pipeline performance diagnosis and yield gate",
            },
            {
                "first": 30600001,
                "last": 30600102,
                "reason": "duckdb persistent cost/session compatibility validation",
            },
        ],
    },
    "authority": sys.argv[10] == "1",
    "environment": collect_environment(),
    "provenance": {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
        "workspace_dirty": bool(git_status.read_text(encoding="utf-8").splitlines()[1:]),
        "source_tree_sha256": collect_environment()["source_tree_sha256"],
        "launcher_sha256": sha256_file(root / "scripts/start_discovery_longhaul_tmux.sh"),
        "requirements_lock_sha256": sha256_file(root / "requirements-final.lock"),
        "strategy_snapshot_sha256": sha256_file(
            root / "reports/strategy-snapshots/final-frozen-strategy-snapshot.json"
        ),
        "canonical_manifest_sha256": sha256_file(canonical_manifest),
        "latest_confirmations_sha256": sha256_file(root / "experiments/latest_confirmations.json"),
        "pip_freeze": str(Path(sys.argv[11]).relative_to(root)),
        "git_status": str(git_status.relative_to(root)),
        "git_diff": str(git_diff.relative_to(root)),
        "git_diff_sha256": sha256_file(git_diff),
        "target_version_audit": sys.argv[14],
        "target_version_audit_sha256": sha256_file(root / sys.argv[14]),
    },
    "validation_gates": {
        "latest_target_audit_required": True,
        "full_repository_tests_required": True,
        "all_frozen_lanes_smoke_required": True,
        "preflight_fallback_count_max": 0,
        "candidate_pipeline_process_failures_max": 0,
        "non_ok_backend_results_max": 0,
        "source_digest_must_match_at_launch": True,
    },
    "reporting": {
        "report_all_lanes_and_seeds": True,
        "report_zero_yield": True,
        "per_run_reports_deferred_from_hot_path": True,
        "metrics": [
            "executed_cases",
            "process_cpu_hours",
            "wall_hours",
            "fresh_candidate_families",
            "strict_recheck_survivors",
            "native_reproducer_success",
            "issue_ready_unique_roots",
            "independently_confirmed_unique_roots",
            "time_to_first_issue_ready_root",
            "false_positive_reasons",
        ],
    },
}
payload["protocol_sha256"] = canonical_sha256(payload)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"discovery freeze manifest: {out}")
PY
  )
}

refresh_aggregate() {
  mkdir -p "${ROOT_DIR}/${GENERATED_DIR}"
  (
    cd "${ROOT_DIR}"
    "${PYTHON_BIN}" -m datadiff.cli discovery-campaign-aggregate \
      --manifests "${AGGREGATE_MANIFESTS}" \
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
  local child_env
  seeds="$(seed_pair_for_batch "${batch}")"
  manifest="${GENERATED_DIR}/discovery-campaign-longhaul-${RUN_ID}-batch$(printf '%04d' "${batch}").json"
  log_file="${LOG_DIR}/discovery-campaign-longhaul-${RUN_ID}-batch$(printf '%04d' "${batch}").log"
  child_session="${SESSION_NAME}-b$(printf '%04d' "${batch}")"
  mkdir -p "${ROOT_DIR}/${LOG_DIR}" "${ROOT_DIR}/${GENERATED_DIR}"
  command=(
    "${PYTHON_BIN}" -m datadiff.cli discovery-campaign
    --skip-bug-audit
    --cases "${BATCH_CASES}"
    --seeds "${seeds}"
    --lanes "${lanes}"
    --candidate-recheck-count "${CANDIDATE_RECHECK_COUNT}"
    --candidate-pipeline-recheck-attempts "${CANDIDATE_PIPELINE_RECHECK_ATTEMPTS}"
    --artifact-limit "${ARTIFACT_LIMIT}"
    --classify-limit "${CLASSIFY_LIMIT}"
    --refresh-classification
    --skip-run-report
    --log-level "${LOG_LEVEL}"
    --output-manifest "${manifest}"
  )
  child_env=(
    env
    "DATADIFF_RUN_PROVENANCE_AUTHORITY=${AUTHORITY}"
    "DATADIFF_RUN_PROVENANCE_FREEZE_INTENT=1"
    "DATADIFF_RUN_PROVENANCE_LATEST_CODE_CLAIM=1"
    "DATADIFF_RUN_PROVENANCE_EVIDENCE_ROLE=latest_bug_discovery_longhaul"
    "DATADIFF_RUN_PROVENANCE_LAUNCH_SOURCE=discovery_longhaul_tmux"
    "DATADIFF_RUN_PROVENANCE_SESSION=${child_session}"
    "DATADIFF_RUN_PROVENANCE_DURATION=${DURATION}"
    "DATADIFF_RUN_PROVENANCE_LAUNCH_SCRIPT=scripts/start_discovery_longhaul_tmux.sh"
    "DATADIFF_RUN_PROVENANCE_FREEZE_MANIFEST=${ROOT_DIR}/${FREEZE_MANIFEST}"
    "DATADIFF_RUN_PROVENANCE_PIP_FREEZE=${ROOT_DIR}/${PIP_FREEZE}"
    "DATADIFF_RUN_PROVENANCE_GIT_STATUS=${ROOT_DIR}/${GIT_STATUS}"
    "DATADIFF_RUN_PROVENANCE_GIT_DIFF=${ROOT_DIR}/${GIT_DIFF}"
    "DATADIFF_RUN_PROVENANCE_TARGET_VERSION_AUDIT=${ROOT_DIR}/${TARGET_VERSION_AUDIT_FILE}"
  )
  if command -v tmux >/dev/null 2>&1; then
    tmux new-session -d -s "${child_session}" \
      "cd \"${ROOT_DIR}\" && $(printf '%q ' "${child_env[@]}" "${command[@]}") > \"${log_file}\" 2>&1"
  else
    (
      cd "${ROOT_DIR}"
      "${child_env[@]}" "${command[@]}" > "${log_file}" 2>&1 &
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
  run_latest_target_audit
  prepare_freeze_manifest
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

if [[ "${DRY_RUN}" == "1" ]]; then
  run_latest_target_audit
  prepare_freeze_manifest
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
  "DATADIFF_DISCOVERY_LONGHAUL_AGGREGATE_MANIFESTS=${AGGREGATE_MANIFESTS}"
  "DATADIFF_PYTHON=${PYTHON_BIN}"
  "DATADIFF_DISCOVERY_LONGHAUL_REQUIRE_LATEST_TARGETS=${REQUIRE_LATEST_TARGETS}"
  "DATADIFF_DISCOVERY_LONGHAUL_TARGET_VERSION_AUDIT_FILE=${TARGET_VERSION_AUDIT_FILE}"
  "DATADIFF_DISCOVERY_LONGHAUL_TARGET_VERSION_AUDIT_SOURCE=${TARGET_VERSION_AUDIT_SOURCE}"
  "DATADIFF_DISCOVERY_LONGHAUL_PROVENANCE_DIR=${PROVENANCE_DIR}"
  "DATADIFF_DISCOVERY_LONGHAUL_FREEZE_MANIFEST=${FREEZE_MANIFEST}"
  "DATADIFF_DISCOVERY_LONGHAUL_PIP_FREEZE=${PIP_FREEZE}"
  "DATADIFF_DISCOVERY_LONGHAUL_GIT_STATUS=${GIT_STATUS}"
  "DATADIFF_DISCOVERY_LONGHAUL_GIT_DIFF=${GIT_DIFF}"
  "DATADIFF_DISCOVERY_LONGHAUL_AUTHORITY=${AUTHORITY}"
  "DATADIFF_DISCOVERY_LONGHAUL_REQUIRE_CLEAN_WORKTREE=${REQUIRE_CLEAN_WORKTREE}"
  "DATADIFF_DISCOVERY_LONGHAUL_LANE_GROUPS=${LANE_GROUPS_RAW}"
)
CONTROLLER_COMMAND="$(printf '%q ' env "${CONTROLLER_ENV[@]}" bash "${BASH_SOURCE[0]}" --worker)"
tmux new-session -d -s "${SESSION_NAME}" \
  "cd \"${ROOT_DIR}\" && ${CONTROLLER_COMMAND} >> \"${LOG_DIR}/discovery-longhaul-${RUN_ID}.controller.log\" 2>&1"

write_status "starting" "tmux controller launched"
printf 'session: %s\n' "${SESSION_NAME}"
printf 'status: %s\n' "${STATUS_FILE}"
printf 'aggregate: %s\n' "${AGGREGATE_FILE}"
