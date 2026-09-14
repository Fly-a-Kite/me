"""Execute a case in a pinned version environment and compare the normalized results.

This is the execution-layer counterpart to the config/scheduler ``version_pair``
dimension, which previously only labelled runs. It runs the *same* backend code
from this repository against the *different* installed packages of a
:class:`~datadiff.version_environments.VersionEnvironment`.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

from datadiff.version_environments import (
    VersionEnvironment,
    repository_root,
)

SCHEMA_VERSION = "datadiff-version-subprocess-v1"
_WORKER_MODULE = "datadiff.version_worker"


def _worker_environment(repository_root_path: Path) -> dict[str, str]:
    env = dict(os.environ)
    src = str(repository_root_path / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src + (os.pathsep + existing if existing else "")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    return env


def run_case_in_environment(
    environment: VersionEnvironment,
    case: Mapping[str, Any],
    backend: str,
    *,
    timeout_s: float = 30.0,
    enable_normalizer: bool = True,
    repository_root_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run ``case`` on ``backend`` inside ``environment`` and return the worker payload."""

    root = Path(repository_root_path or environment.repository_root or repository_root())
    interpreter = environment.require_interpreter()
    request = {
        "case": dict(case),
        "backend": str(backend),
        "timeout_s": float(timeout_s),
        "enable_normalizer": bool(enable_normalizer),
    }
    hard_timeout = float(timeout_s) + 30.0
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [str(interpreter), "-m", _WORKER_MODULE],
            input=json.dumps(request, default=str).encode("utf-8"),
            capture_output=True,
            timeout=hard_timeout,
            env=_worker_environment(root),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "schema_version": SCHEMA_VERSION,
            "env_id": environment.env_id,
            "backend": str(backend),
            "status": "timeout",
            "error_type": "TimeoutExpired",
            "error": f"version worker exceeded {hard_timeout:.1f}s",
            "wall_ms": (time.perf_counter() - started) * 1000.0,
        }
    wall_ms = (time.perf_counter() - started) * 1000.0
    if completed.returncode != 0:
        return {
            "schema_version": SCHEMA_VERSION,
            "env_id": environment.env_id,
            "backend": str(backend),
            "status": "worker_error",
            "error_type": "WorkerNonZeroExit",
            "error": completed.stderr.decode("utf-8", "replace")[-2000:],
            "returncode": completed.returncode,
            "wall_ms": wall_ms,
        }
    try:
        payload = json.loads(completed.stdout.decode("utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "env_id": environment.env_id,
            "backend": str(backend),
            "status": "worker_error",
            "error_type": "WorkerInvalidJSON",
            "error": f"{exc}: {completed.stdout[:400]!r}",
            "wall_ms": wall_ms,
        }
    payload["env_id"] = environment.env_id
    payload["wall_ms"] = wall_ms
    return payload


def _comparison_key(payload: Mapping[str, Any]) -> str:
    normalized = payload.get("normalized")
    if isinstance(normalized, Mapping):
        return str(normalized.get("comparison_key", "") or "")
    return ""


def compare_normalized(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    """Compare two worker payloads on their canonical normalized result."""

    left_status = str(left.get("status", ""))
    right_status = str(right.get("status", ""))
    left_key = _comparison_key(left)
    right_key = _comparison_key(right)

    if left_status != "ok" or right_status != "ok":
        match = left_status == right_status
        return {
            "match": match,
            "reason": "status" if not match else "status_equal_non_ok",
            "left_status": left_status,
            "right_status": right_status,
            "left_comparison_key": left_key,
            "right_comparison_key": right_key,
        }
    if not left_key or not right_key:
        return {
            "match": False,
            "reason": "missing_comparison_key",
            "left_status": left_status,
            "right_status": right_status,
            "left_comparison_key": left_key,
            "right_comparison_key": right_key,
        }
    match = left_key == right_key
    return {
        "match": match,
        "reason": "result" if not match else "equal",
        "left_status": left_status,
        "right_status": right_status,
        "left_comparison_key": left_key,
        "right_comparison_key": right_key,
    }


def cross_version_compare(
    left_environment: VersionEnvironment,
    right_environment: VersionEnvironment,
    case: Mapping[str, Any],
    backend: str,
    *,
    timeout_s: float = 30.0,
    enable_normalizer: bool = True,
    repository_root_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run ``case`` in both environments and return the differential outcome."""

    left = run_case_in_environment(
        left_environment,
        case,
        backend,
        timeout_s=timeout_s,
        enable_normalizer=enable_normalizer,
        repository_root_path=repository_root_path,
    )
    right = run_case_in_environment(
        right_environment,
        case,
        backend,
        timeout_s=timeout_s,
        enable_normalizer=enable_normalizer,
        repository_root_path=repository_root_path,
    )
    comparison = compare_normalized(left, right)
    return {
        "schema_version": "datadiff-cross-version-comparison-v1",
        "backend": str(backend),
        "left_env_id": left_environment.env_id,
        "right_env_id": right_environment.env_id,
        "left_packages": dict(sorted(left_environment.packages.items())),
        "right_packages": dict(sorted(right_environment.packages.items())),
        "left": left,
        "right": right,
        "comparison": comparison,
    }
