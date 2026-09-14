"""Tests for the cross-version environment registry and subprocess executor."""

from __future__ import annotations

import importlib.metadata
import json
import sys
from pathlib import Path

import pytest

from datadiff.backends.version_subprocess import (
    compare_normalized,
    cross_version_compare,
    run_case_in_environment,
)
from datadiff.version_environments import (
    VersionEnvironment,
    VersionEnvironmentError,
    environment_from_interpreter,
    load_registry,
    probe_package_versions,
    write_registry,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
POLARS_CASE = (
    REPO_ROOT
    / "experiments/canonical_confirmed_bug_corpus/v1/cases/polars_reflected_arithmetic_operand_order.json"
)


def _current_environment(env_id: str = "current") -> VersionEnvironment:
    return VersionEnvironment(
        env_id=env_id,
        interpreter=sys.executable,
        packages=dict(probe_package_versions(sys.executable, ("pandas", "polars", "duckdb"))),
        label="current interpreter",
        repository_root=str(REPO_ROOT),
    )


def test_registry_roundtrip(tmp_path: Path) -> None:
    environments = [
        VersionEnvironment("a", "/usr/bin/python3", {"pandas": "1.0"}, label="A"),
        VersionEnvironment("b", "/usr/bin/python3", {"pandas": "2.0"}, label="B"),
    ]
    target = write_registry(environments, tmp_path / "envs.json")
    loaded = load_registry(target)
    assert set(loaded) == {"a", "b"}
    assert loaded["a"].packages["pandas"] == "1.0"
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "datadiff-version-environments-v1"


def test_registry_rejects_duplicate_ids(tmp_path: Path) -> None:
    target = tmp_path / "envs.json"
    target.write_text(
        json.dumps(
            {
                "environments": [
                    {"env_id": "x", "interpreter": "/usr/bin/python3"},
                    {"env_id": "x", "interpreter": "/usr/bin/python3"},
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(VersionEnvironmentError, match="duplicate"):
        load_registry(target)


def test_probe_current_interpreter_matches_importlib() -> None:
    versions = probe_package_versions(sys.executable, ("pandas",))
    assert versions["pandas"] == importlib.metadata.version("pandas")


def test_environment_from_interpreter() -> None:
    environment = environment_from_interpreter("here", sys.executable, packages=("pandas",))
    assert environment.env_id == "here"
    assert environment.packages["pandas"]


def test_missing_interpreter_raises() -> None:
    environment = VersionEnvironment("ghost", "/nonexistent/python")
    with pytest.raises(VersionEnvironmentError, match="interpreter is missing"):
        environment.require_interpreter()


def test_worker_executes_a_corpus_case() -> None:
    case = json.loads(POLARS_CASE.read_text(encoding="utf-8"))
    payload = run_case_in_environment(
        _current_environment(), case, "polars", timeout_s=30.0
    )
    assert payload["status"] == "ok", payload.get("error")
    normalized = payload["normalized"]
    assert normalized["columns"]
    assert normalized["comparison_key"]
    assert payload["env_id"] == "current"


def test_cross_version_compare_same_interpreter_matches() -> None:
    case = json.loads(POLARS_CASE.read_text(encoding="utf-8"))
    left = _current_environment("left")
    right = _current_environment("right")
    outcome = cross_version_compare(left, right, case, "polars", timeout_s=30.0)
    assert outcome["comparison"]["match"] is True
    assert outcome["comparison"]["reason"] == "equal"
    assert outcome["left_env_id"] == "left"
    assert outcome["right_env_id"] == "right"


def test_compare_normalized_detects_result_and_status_differences() -> None:
    base = {"status": "ok", "normalized": {"comparison_key": "k1"}}
    same = {"status": "ok", "normalized": {"comparison_key": "k1"}}
    different = {"status": "ok", "normalized": {"comparison_key": "k2"}}
    errored = {"status": "error", "normalized": {"comparison_key": "k1"}}

    assert compare_normalized(base, same)["match"] is True
    diff = compare_normalized(base, different)
    assert diff["match"] is False and diff["reason"] == "result"
    status = compare_normalized(base, errored)
    assert status["match"] is False and status["reason"] == "status"
    missing = compare_normalized(
        {"status": "ok", "normalized": {}}, {"status": "ok", "normalized": {"comparison_key": "k"}}
    )
    assert missing["match"] is False and missing["reason"] == "missing_comparison_key"
