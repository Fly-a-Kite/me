from __future__ import annotations

import os
import subprocess
from typing import Any

from datadiff.util import PROJECT_ROOT

RUN_PROVENANCE_SCHEMA_VERSION = "run-provenance-v1"


def current_workspace_git_commit() -> str:
    return _git_stdout("rev-parse", "HEAD")


def collect_run_provenance() -> dict[str, Any]:
    git_commit = current_workspace_git_commit()
    workspace_dirty = _workspace_dirty()
    return {
        "schema_version": RUN_PROVENANCE_SCHEMA_VERSION,
        "vcs": {
            "git_commit": git_commit,
            "git_commit_short": git_commit[:12] if git_commit else "",
            "git_branch": _git_stdout("symbolic-ref", "--quiet", "--short", "HEAD"),
            "workspace_dirty": workspace_dirty,
        },
        "launch": {
            "source": _env_text("DATADIFF_RUN_PROVENANCE_LAUNCH_SOURCE"),
            "session": _env_text("DATADIFF_RUN_PROVENANCE_SESSION"),
            "duration": _env_text("DATADIFF_RUN_PROVENANCE_DURATION"),
            "batch_duration": _env_text("DATADIFF_RUN_PROVENANCE_BATCH_DURATION"),
            "log_prefix": _env_text("DATADIFF_RUN_PROVENANCE_LOG_PREFIX"),
            "launch_script": _env_text("DATADIFF_RUN_PROVENANCE_LAUNCH_SCRIPT"),
        },
        "harness": {
            "authority": _env_flag("DATADIFF_RUN_PROVENANCE_AUTHORITY"),
            "freeze_intent": _env_flag("DATADIFF_RUN_PROVENANCE_FREEZE_INTENT"),
            "latest_code_claim": _env_flag("DATADIFF_RUN_PROVENANCE_LATEST_CODE_CLAIM"),
            "evidence_role": _env_text("DATADIFF_RUN_PROVENANCE_EVIDENCE_ROLE"),
        },
        "freeze_artifacts": {
            "manifest": _env_text("DATADIFF_RUN_PROVENANCE_FREEZE_MANIFEST"),
            "pip_freeze": _env_text("DATADIFF_RUN_PROVENANCE_PIP_FREEZE"),
            "git_status": _env_text("DATADIFF_RUN_PROVENANCE_GIT_STATUS"),
            "git_diff": _env_text("DATADIFF_RUN_PROVENANCE_GIT_DIFF"),
            "launcher_env": _env_text("DATADIFF_RUN_PROVENANCE_LAUNCH_ENV"),
            "strategy_snapshot": _env_text("DATADIFF_RUN_PROVENANCE_STRATEGY_SNAPSHOT"),
            "strategy_learning": _env_text("DATADIFF_RUN_PROVENANCE_STRATEGY_LEARNING"),
        },
    }


def _workspace_dirty() -> bool | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(proc.stdout.strip())


def _git_stdout(*args: str) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return proc.stdout.strip()


def _env_text(name: str) -> str:
    return str(os.environ.get(name, "") or "").strip()


def _env_flag(name: str) -> bool:
    return _env_text(name).lower() in {"1", "true", "yes", "on"}
