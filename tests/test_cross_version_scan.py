"""Tests for the cross-version discovery lane engine and CLI registration."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from datadiff.cli import build_parser
from datadiff.cross_version_scan import scan_cross_version, write_scan
from datadiff.version_environments import VersionEnvironment, probe_package_versions

REPO_ROOT = Path(__file__).resolve().parents[1]
POLARS_CASE = (
    REPO_ROOT
    / "experiments/canonical_confirmed_bug_corpus/v1/cases/polars_reflected_arithmetic_operand_order.json"
)


def _environment(env_id: str) -> VersionEnvironment:
    return VersionEnvironment(
        env_id=env_id,
        interpreter=sys.executable,
        packages=dict(probe_package_versions(sys.executable, ("pandas", "polars"))),
        label=env_id,
        repository_root=str(REPO_ROOT),
    )


def test_scan_same_interpreter_has_no_findings() -> None:
    registry = {"a": _environment("a"), "b": _environment("b")}
    payload = scan_cross_version(
        registry, [("a", "b")], cases=[POLARS_CASE], backends=["polars"], timeout_s=30.0
    )
    assert payload["result_count"] == 1
    assert payload["finding_count"] == 0
    assert payload["results"][0]["match"] is True


def test_scan_rejects_unknown_environment() -> None:
    registry = {"a": _environment("a")}
    with pytest.raises(ValueError, match="unknown version environment"):
        scan_cross_version(
            registry, [("a", "ghost")], cases=[POLARS_CASE], backends=["polars"], timeout_s=30.0
        )


def test_scan_skips_backends_absent_from_an_environment() -> None:
    registry = {
        "a": VersionEnvironment("a", sys.executable, {"polars": "1"}, repository_root=str(REPO_ROOT)),
        "b": VersionEnvironment("b", sys.executable, {"pandas": "1"}, repository_root=str(REPO_ROOT)),
    }
    payload = scan_cross_version(
        registry, [("a", "b")], cases=[POLARS_CASE], backends=["polars", "pandas"], timeout_s=30.0
    )
    assert payload["result_count"] == 0
    assert payload["finding_count"] == 0


def test_write_scan_emits_json_and_markdown(tmp_path: Path) -> None:
    registry = {"a": _environment("a"), "b": _environment("b")}
    payload = scan_cross_version(
        registry, [("a", "b")], cases=[POLARS_CASE], backends=["polars"], timeout_s=30.0
    )
    json_path, md_path = write_scan(payload, tmp_path)
    assert json_path.is_file() and md_path.is_file()
    assert "Cross-version scan" in md_path.read_text(encoding="utf-8")


def test_scan_with_generated_seeds() -> None:
    registry = {"a": _environment("a"), "b": _environment("b")}
    payload = scan_cross_version(
        registry, [("a", "b")], cases=(), seeds=[30700001], backends=["polars"], timeout_s=30.0
    )
    assert payload["result_count"] >= 1
    assert payload["finding_count"] == 0  # same interpreter, so statuses and results agree
    assert payload["cases"][0].startswith("seed-")


def test_cli_registers_cross_version_scan() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["cross-version-scan", "--env-pair", "a->b", "--seeds", "1,2", "--json"]
    )
    assert callable(args.func)
    assert args.env_pair == ["a->b"]
    assert args.seeds == "1,2"
