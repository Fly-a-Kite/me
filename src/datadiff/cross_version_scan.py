"""Cross-version discovery lane engine.

Runs a set of DSL cases on the same backend across two or more pinned version
environments and reports every canonical-result divergence as a cross-version
finding. This is the execution-layer counterpart of the config-level
``version_pair`` dimension (see :mod:`datadiff.version_environments`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from datadiff.backends.version_subprocess import cross_version_compare
from datadiff.version_environments import VersionEnvironment

SCHEMA_VERSION = "datadiff-cross-version-scan-v1"

DEFAULT_BACKENDS: tuple[str, ...] = (
    "datafusion",
    "polars",
    "duckdb",
    "pyarrow",
    "pandas",
)


def _case_files(cases: str | Path | Iterable[str | Path]) -> list[Path]:
    items = [cases] if isinstance(cases, (str, Path)) else list(cases)
    files: list[Path] = []
    for item in items:
        path = Path(item)
        if path.is_dir():
            files.extend(sorted(path.glob("*.json")))
        elif path.is_file():
            files.append(path)
    return files


def _comparable_backends(
    left: VersionEnvironment,
    right: VersionEnvironment,
    requested: Sequence[str],
) -> list[str]:
    backends: list[str] = []
    for backend in requested:
        if left.packages.get(backend) and right.packages.get(backend):
            backends.append(backend)
    return backends


def _load_cases(case_files: Sequence[Path]) -> list[tuple[str, dict[str, Any]]]:
    items: list[tuple[str, dict[str, Any]]] = []
    for path in case_files:
        try:
            items.append((path.name, json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, json.JSONDecodeError):
            items.append((path.name, {}))
    return items


def _generated_cases(seeds: Sequence[int], profile: str) -> list[tuple[str, dict[str, Any]]]:
    if not seeds:
        return []
    from datadiff.datagen import generate_case

    return [
        (f"seed-{seed}", generate_case(int(seed), profile=profile).to_dict()) for seed in seeds
    ]


def scan_cross_version(
    environments: Mapping[str, VersionEnvironment],
    env_pairs: Sequence[tuple[str, str]],
    *,
    cases: str | Path | Iterable[str | Path] = (),
    backends: Sequence[str] = DEFAULT_BACKENDS,
    seeds: Sequence[int] = (),
    profile: str = "common",
    timeout_s: float = 30.0,
    repository_root_path: str | Path | None = None,
) -> dict[str, Any]:
    items = _load_cases(_case_files(cases)) + _generated_cases(seeds, profile)
    results: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []

    for case_name, case in items:
        if not case:
            results.append({"case": case_name, "status": "case_load_error"})
            continue
        case_id = str(case.get("case_id", case_name.rsplit(".", 1)[0]))
        for left_id, right_id in env_pairs:
            if left_id not in environments or right_id not in environments:
                raise ValueError(f"unknown version environment in pair {left_id!r}->{right_id!r}")
            left = environments[left_id]
            right = environments[right_id]
            comparable = _comparable_backends(left, right, backends)
            for backend in comparable:
                outcome = cross_version_compare(
                    left,
                    right,
                    case,
                    backend,
                    timeout_s=timeout_s,
                    repository_root_path=repository_root_path,
                )
                comparison = outcome["comparison"]
                record = {
                    "case": case_name,
                    "case_id": case_id,
                    "backend": backend,
                    "left_env_id": left_id,
                    "right_env_id": right_id,
                    "left_status": outcome["left"].get("status"),
                    "right_status": outcome["right"].get("status"),
                    "match": comparison["match"],
                    "reason": comparison["reason"],
                    "left_rows": (outcome["left"].get("normalized") or {}).get("rows"),
                    "right_rows": (outcome["right"].get("normalized") or {}).get("rows"),
                    "left_comparison_key": comparison.get("left_comparison_key"),
                    "right_comparison_key": comparison.get("right_comparison_key"),
                }
                results.append(record)
                if not comparison["match"]:
                    findings.append(record)

    return {
        "schema_version": SCHEMA_VERSION,
        "env_pairs": [list(pair) for pair in env_pairs],
        "backends": list(backends),
        "cases": [name for name, _ in items],
        "result_count": len(results),
        "finding_count": len(findings),
        "results": results,
        "findings": findings,
    }


def write_scan(payload: Mapping[str, Any], out_dir: str | Path) -> tuple[Path, Path]:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "cross_version_scan.json"
    md_path = target / "cross_version_scan.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Cross-version scan",
        "",
        f"- env pairs: {payload.get('env_pairs')}",
        f"- backends: {payload.get('backends')}",
        f"- cases: {len(payload.get('cases', []))}",
        f"- results: {payload.get('result_count')}",
        f"- findings: {payload.get('finding_count')}",
        "",
        "| case | backend | left | right | match | reason |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for record in payload.get("results", []):
        lines.append(
            f"| {record.get('case')} | {record.get('backend')} | {record.get('left_status')} | "
            f"{record.get('right_status')} | {record.get('match')} | {record.get('reason')} |"
        )
    for finding in payload.get("findings", []):
        lines += [
            "",
            f"## Finding: {finding.get('case')} / {finding.get('backend')}",
            "",
            f"- left ({finding.get('left_env_id')}): `{json.dumps(finding.get('left_rows'), ensure_ascii=False)}`",
            f"- right ({finding.get('right_env_id')}): `{json.dumps(finding.get('right_rows'), ensure_ascii=False)}`",
        ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path
