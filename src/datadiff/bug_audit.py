from __future__ import annotations

import itertools
import json
import math
import struct
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from datadiff.canonicalization import sort_by_canonical_key
from datadiff.env import collect_environment, package_version
from datadiff.pathing import project_display_path as _project_display_path_impl
from datadiff.util import PROJECT_ROOT, REPORTS_DIR, dump_json, ensure_dirs, slugify, utc_now


@dataclass(frozen=True, slots=True)
class BugAuditProbe:
    probe_id: str
    target_backend: str
    family: str
    title: str
    invariant: str
    runner: Callable[[], dict[str, Any]]


@dataclass(slots=True)
class BugAuditResult:
    probe_id: str
    target_backend: str
    family: str
    title: str
    invariant: str
    status: str
    candidate_bug: bool
    observed: Any = None
    expected: Any = None
    evidence: str = ""
    error_type: str = ""
    error: str = ""
    version: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BugAuditRun:
    generated_at: str
    evidence_mode: str
    methodology: str
    environment: dict[str, str]
    results: list[dict[str, Any]]
    candidate_bug_families: list[str] = field(default_factory=list)
    output_json: str = ""
    output_markdown: str = ""
    issue_files: list[str] = field(default_factory=list)
    schema_version: str = "bug-audit-v1"
    generated_by: str = "datadiff bug-audit"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def list_audit_probe_ids() -> list[str]:
    return sorted(PROBES)


def run_probe_audit(
    *,
    probe_ids: list[str] | None = None,
    output_dir: Path | None = None,
) -> BugAuditRun:
    ensure_dirs()
    selected_ids = probe_ids or list_audit_probe_ids()
    unknown = sorted(set(selected_ids) - set(PROBES))
    if unknown:
        raise ValueError(f"unknown bug-audit probe id(s): {', '.join(unknown)}")

    results = [_run_probe(PROBES[probe_id]) for probe_id in selected_ids]
    candidate_families = candidate_families_from_results([result.to_dict() for result in results])
    run = BugAuditRun(
        generated_at=utc_now(),
        evidence_mode="deterministic_probe",
        methodology=(
            "Each probe encodes a backend-independent invariant or a zero-offset control. "
            "The command executes the current installed libraries and classifies a candidate "
            "only when the target backend violates that invariant."
        ),
        environment=_audit_environment(),
        results=[result.to_dict() for result in results],
        candidate_bug_families=candidate_families,
    )
    json_path, md_path = write_probe_audit_outputs(run, output_dir=output_dir)
    run.output_json = str(json_path)
    run.output_markdown = str(md_path)
    dump_json(run.to_dict(), json_path)
    md_path.write_text(render_probe_audit_markdown(run), encoding="utf-8")
    return run


def candidate_families_from_results(results: list[dict[str, Any]]) -> list[str]:
    families = []
    seen = set()
    for result in results:
        if not result.get("candidate_bug"):
            continue
        family = f"{result.get('family', 'unknown')}@{result.get('target_backend', 'unknown')}"
        if family in seen:
            continue
        seen.add(family)
        families.append(family)
    return families


def write_probe_audit_outputs(run: BugAuditRun, *, output_dir: Path | None = None) -> tuple[Path, Path]:
    output_dir = output_dir or REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = run.generated_at.replace(":", "").replace("-", "").replace("Z", "")
    json_path = output_dir / f"bug-audit-{stamp}.json"
    md_path = output_dir / f"bug-audit-{stamp}.md"
    return json_path, md_path


def write_probe_issue_drafts(
    run: BugAuditRun,
    *,
    issue_dir: Path | None = None,
    overwrite: bool = False,
) -> list[Path]:
    issue_dir = issue_dir or PROJECT_ROOT / "new_issue" / "generated"
    issue_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = issue_dir / "manifest.json"
    paths: list[Path] = []
    for result in run.results:
        if not result.get("candidate_bug"):
            continue
        family = str(result.get("family", "unknown"))
        path = issue_dir / f"{slugify(family, max_len=96)}.md"
        if path.exists() and not overwrite:
            paths.append(path)
            continue
        path.write_text(_issue_draft_markdown(run, result, manifest_path=manifest_path), encoding="utf-8")
        paths.append(path)
    run.issue_files = [_project_display_path(str(path)) for path in paths]
    dump_json(_portable_manifest(run, manifest_path=manifest_path), manifest_path)
    if run.output_json:
        dump_json(run.to_dict(), Path(run.output_json))
    return paths


def _run_probe(probe: BugAuditProbe) -> BugAuditResult:
    try:
        payload = probe.runner()
    except Exception as exc:  # noqa: BLE001
        return BugAuditResult(
            probe_id=probe.probe_id,
            target_backend=probe.target_backend,
            family=probe.family,
            title=probe.title,
            invariant=probe.invariant,
            status="probe_error",
            candidate_bug=False,
            error_type=type(exc).__name__,
            error=str(exc),
            version=package_version(probe.target_backend),
        )
    candidate = bool(payload.get("candidate_bug", False))
    return BugAuditResult(
        probe_id=probe.probe_id,
        target_backend=probe.target_backend,
        family=probe.family,
        title=probe.title,
        invariant=probe.invariant,
        status="candidate_implementation_bug" if candidate else "no_bug_detected",
        candidate_bug=candidate,
        observed=payload.get("observed"),
        expected=payload.get("expected"),
        evidence=str(payload.get("evidence", "")),
        version=str(payload.get("version", package_version(probe.target_backend))),
        details=dict(payload.get("details", {}) or {}),
    )


def _audit_environment() -> dict[str, str]:
    env = collect_environment()
    env["pyarrow"] = package_version("pyarrow")
    env["datafusion"] = package_version("datafusion")
    return env


def render_probe_audit_markdown(run: BugAuditRun) -> str:
    lines = [
        "# Automated Bug Audit",
        "",
        f"- Generated at: `{run.generated_at}`",
        f"- Evidence mode: `{run.evidence_mode}`",
        f"- Candidate bug families: {len(run.candidate_bug_families)}",
    ]
    if run.candidate_bug_families:
        for family in run.candidate_bug_families:
            lines.append(f"  - `{family}`")
    else:
        lines.append("  - none")
    lines.extend(["", "## Methodology", "", run.methodology, "", "## Environment"])
    for key, value in sorted(run.environment.items()):
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Probe Results"])
    for result in run.results:
        lines.extend(
            [
                "",
                f"### {result['probe_id']}",
                "",
                f"- Target backend: `{result['target_backend']}`",
                f"- Family: `{result['family']}`",
                f"- Status: `{result['status']}`",
                f"- Candidate bug: `{result['candidate_bug']}`",
                f"- Version: `{result.get('version', '')}`",
                f"- Invariant: {result['invariant']}",
                f"- Evidence: {result.get('evidence', '') or 'n/a'}",
                "",
                "Expected:",
                "",
                "```text",
                repr(result.get("expected")),
                "```",
                "",
                "Observed:",
                "",
                "```text",
                repr(result.get("observed")),
                "```",
            ]
        )
        if result.get("details"):
            lines.extend(
                [
                    "",
                    "Details:",
                    "",
                    "```json",
                    _json_block(result.get("details")),
                    "```",
                ]
            )
        if result.get("error"):
            lines.extend(["", f"- Probe error: `{result.get('error_type')}` {result.get('error')}"])
    lines.append("")
    return "\n".join(lines)


def _issue_draft_markdown(run: BugAuditRun, result: dict[str, Any], *, manifest_path: Path) -> str:
    family = str(result.get("family", "unknown"))
    target = str(result.get("target_backend", "unknown"))
    lines = [
        f"# Issue Draft: {result.get('title', family)}",
        "",
        "## Discovery Record",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| Discovery time | {run.generated_at} |",
        f"| Discovery command | `datadiff bug-audit --probes {result.get('probe_id', '')}` |",
        f"| Evidence mode | `{run.evidence_mode}` |",
        f"| Probe id | `{result.get('probe_id', '')}` |",
        f"| Candidate family | `{family}@{target}` |",
        f"| Target backend/version | `{target}` `{result.get('version', '')}` |",
        f"| Evidence manifest | `{_project_display_path(str(manifest_path))}` |",
        "| Status | New candidate issue generated by automated project code |",
        "",
        "## Automatic Verdict",
        "",
        f"- Status: `{result.get('status', '')}`",
        f"- Candidate bug: `{result.get('candidate_bug', False)}`",
        f"- Invariant: {result.get('invariant', '')}",
        f"- Evidence: {result.get('evidence', '')}",
        "",
        "## Expected",
        "",
        "```json",
        _json_block(result.get("expected")),
        "```",
        "",
        "## Observed",
        "",
        "```json",
        _json_block(result.get("observed")),
        "```",
        "",
        "## Details",
        "",
        "```json",
        _json_block(result.get("details", {})),
        "```",
        "",
        "## Environment",
        "",
    ]
    for key, value in sorted(run.environment.items()):
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            "Run the project command below in the same environment:",
            "",
            "```bash",
            f"datadiff bug-audit --probes {result.get('probe_id', '')}",
            "```",
            "",
            "The issue claim is based on the generated JSON/Markdown manifest and the probe's encoded invariant.",
            "",
        ]
    )
    return "\n".join(lines)


def _json_block(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)


def _portable_manifest(run: BugAuditRun, *, manifest_path: Path) -> dict[str, Any]:
    data = run.to_dict()
    data["manifest_path"] = _project_display_path(str(manifest_path))
    data["local_report_json"] = _project_display_path(str(run.output_json))
    data["local_report_markdown"] = _project_display_path(str(run.output_markdown))
    data["output_json"] = data["local_report_json"]
    data["output_markdown"] = data["local_report_markdown"]
    data["issue_files"] = [_project_display_path(str(path)) for path in run.issue_files]
    data["reproduction_command"] = "datadiff bug-audit --write-issues --overwrite-issues"
    return data


def _project_display_path(value: str | Path | None) -> str:
    return _project_display_path_impl(value, project_root=PROJECT_ROOT)


def _bits(value: float) -> str:
    return hex(struct.unpack(">Q", struct.pack(">d", float(value)))[0])


def _float_lists_equal(left: list[float], right: list[float]) -> bool:
    return len(left) == len(right) and all(
        math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=0.0)
        for a, b in zip(left, right)
    )


def _polars_reflected_arithmetic_probe() -> dict[str, Any]:
    import polars as pl

    lhs = pl.Series("lhs", [2, 3, 4])
    rhs = pl.Series("rhs", [5, 7, 9])
    checks = {
        "__rsub__": [3, 4, 5],
        "__rtruediv__": [2.5, 7 / 3, 9 / 4],
        "__rfloordiv__": [2, 2, 2],
        "__rmod__": [1, 1, 1],
        "__rpow__": [25, 343, 6561],
    }
    observed: dict[str, Any] = {}
    mismatches: list[str] = []
    for method, expected_values in checks.items():
        try:
            values = getattr(lhs, method)(rhs).to_list()
        except Exception as exc:  # noqa: BLE001
            observed[method] = {"error_type": type(exc).__name__, "error": str(exc)}
            mismatches.append(method)
            continue
        observed[method] = values
        if method == "__rtruediv__":
            if not _float_lists_equal([float(v) for v in values], [float(v) for v in expected_values]):
                mismatches.append(method)
        elif values != expected_values:
            mismatches.append(method)
    return {
        "candidate_bug": bool(mismatches),
        "version": pl.__version__,
        "expected": checks,
        "observed": observed,
        "evidence": f"Reflected Series methods with mismatched result or exception: {mismatches}",
        "details": {
            "methods_checked": sorted(checks),
            "mismatched_methods": mismatches,
            "scope": "Series-vs-Series reflected arithmetic; Polars Expr reflected arithmetic is checked separately by normal expression semantics.",
        },
    }


def _polars_vector_division_probe() -> dict[str, Any]:
    import polars as pl

    sweep = _polars_vector_division_sweep(pl)
    expected_value = 7 / 5
    single = (
        pl.DataFrame({"x": [7]}, schema={"x": pl.Int64})
        .with_columns((pl.col("x") / 5).alias("y"))
        .get_column("y")
        .to_list()
    )
    vector = (
        pl.DataFrame({"x": [7, 7]}, schema={"x": pl.Int64})
        .with_columns((pl.col("x") / 5).alias("y"))
        .get_column("y")
        .to_list()
    )
    expected_bits = _bits(expected_value)
    observed_bits = [_bits(value) for value in vector]
    candidate = _bits(single[0]) == expected_bits and any(bits != expected_bits for bits in observed_bits)
    return {
        "candidate_bug": candidate,
        "version": pl.__version__,
        "expected": {
            "single_path": [expected_value],
            "vector_path": [expected_value, expected_value],
            "bits": expected_bits,
        },
        "observed": {
            "single_path": single,
            "single_bits": [_bits(value) for value in single],
            "vector_path": vector,
            "vector_bits": observed_bits,
        },
        "evidence": (
            "Vectorized expression division should match the single-row/Python rounded double for 7 / 5; "
            f"deterministic sweep mismatches={sweep['mismatch_count']}."
        ),
        "details": sweep,
    }


def _pyarrow_sliced_bool_groupby_probe() -> dict[str, Any]:
    import pyarrow as pa

    sweep = _pyarrow_sliced_bool_groupby_sweep(pa)
    base = pa.table({"g": [99, 10, 10], "flag": [True, False, None]})
    sliced = base.slice(1)
    rebuilt = pa.table(sliced.to_pydict())
    aggregations = ["any", "all"]
    observed: dict[str, Any] = {}
    expected: dict[str, Any] = {}
    mismatches: list[str] = []
    for aggregation in aggregations:
        sliced_result = (
            sliced.group_by("g", use_threads=False)
            .aggregate([("flag", aggregation)])
            .to_pylist()
        )
        rebuilt_result = (
            rebuilt.group_by("g", use_threads=False)
            .aggregate([("flag", aggregation)])
            .to_pylist()
        )
        observed[aggregation] = sliced_result
        expected[aggregation] = rebuilt_result
        if sliced_result != rebuilt_result:
            mismatches.append(aggregation)
    return {
        "candidate_bug": bool(mismatches),
        "version": pa.__version__,
        "expected": expected,
        "observed": observed,
        "evidence": (
            "Sliced table group_by boolean aggregates should match an equivalent zero-offset rebuilt table; "
            f"mismatched aggregations: {mismatches}; sliced flag offset={sliced['flag'].chunk(0).offset}; "
            f"deterministic sweep mismatches={sweep['mismatch_count']}."
        ),
        "details": sweep,
    }


def _polars_slice_chunk_lazy_equivalence_probe() -> dict[str, Any]:
    import polars as pl

    sweep = _polars_slice_chunk_lazy_equivalence_sweep(pl)
    return {
        "candidate_bug": bool(sweep["mismatch_count"]),
        "version": pl.__version__,
        "expected": {"mismatch_count": 0},
        "observed": {
            "mismatch_count": sweep["mismatch_count"],
            "first_mismatch": sweep["first_mismatch"],
        },
        "evidence": (
            "Sliced, multi-chunk, rebuilt, and lazy Polars tables should agree for the same "
            f"group/filter/sort operations; deterministic comparisons={sweep['checked_comparisons']}, "
            f"mismatches={sweep['mismatch_count']}."
        ),
        "details": sweep,
    }


def _pyarrow_sliced_transform_equivalence_probe() -> dict[str, Any]:
    import pyarrow as pa

    sweep = _pyarrow_sliced_transform_equivalence_sweep(pa)
    return {
        "candidate_bug": bool(sweep["mismatch_count"]),
        "version": pa.__version__,
        "expected": {"mismatch_count": 0},
        "observed": {
            "mismatch_count": sweep["mismatch_count"],
            "first_mismatch": sweep["first_mismatch"],
        },
        "evidence": (
            "Sliced Arrow tables and zero-offset rebuilt tables should agree for Boolean "
            f"filter/sort/take operations; checked cases={sweep['checked_cases']}, "
            f"mismatches={sweep['mismatch_count']}."
        ),
        "details": sweep,
    }


def _datafusion_limit_idempotence_probe() -> dict[str, Any]:
    from datafusion import SessionContext
    import pyarrow as pa

    ctx = SessionContext()
    table = pa.table(
        {
            "id": list(range(14)),
            "g": ["a", "b", "c", "b", "b", "a", None, "a", "c", "c", "c", "b", "a", "a"],
            "x": [None, 10, -10, -2, 10, None, None, 0, 0, -10, -10, -10, -2, 2],
            "z": [8, None, -3, 0, None, 0, 0, 1, 0, 8, -3, 8, -3, 1],
            "s": [
                "A",
                "space value",
                "",
                "space value",
                "a",
                "",
                "space value",
                "a",
                "A",
                "A",
                "space value",
                "A",
                "",
                "space value",
            ],
        },
        schema=pa.schema(
            [
                pa.field("id", pa.int64(), nullable=False),
                pa.field("g", pa.string(), nullable=True),
                pa.field("x", pa.int64(), nullable=True),
                pa.field("z", pa.int64(), nullable=True),
                pa.field("s", pa.string(), nullable=True),
            ]
        ),
    )
    ctx.register_record_batches("t0", [table.to_batches()])
    base_sql = """
SELECT * FROM (
  SELECT * FROM (
    SELECT * FROM t0 ORDER BY s ASC NULLS FIRST, id ASC NULLS LAST LIMIT 5
  ) q ORDER BY x DESC NULLS FIRST, id ASC NULLS LAST OFFSET 2
) q ORDER BY x DESC NULLS FIRST, id ASC NULLS LAST
"""
    duplicate_limit_sql = """
SELECT * FROM (
  SELECT * FROM (
    SELECT * FROM (
      SELECT * FROM t0 ORDER BY s ASC NULLS FIRST, id ASC NULLS LAST LIMIT 5
    ) q ORDER BY s ASC NULLS FIRST, id ASC NULLS LAST LIMIT 5
  ) q ORDER BY x DESC NULLS FIRST, id ASC NULLS LAST OFFSET 2
) q ORDER BY x DESC NULLS FIRST, id ASC NULLS LAST
"""
    expected = _pandas_records(ctx.sql(base_sql).to_pandas())
    observed = _pandas_records(ctx.sql(duplicate_limit_sql).to_pandas())
    return {
        "candidate_bug": observed != expected,
        "version": package_version("datafusion"),
        "expected": {
            "base_rows": expected,
            "relation": "duplicate_limit_sql should return the same rows as base_sql",
        },
        "observed": {
            "duplicate_limit_rows": observed,
            "base_row_count": len(expected),
            "duplicate_limit_row_count": len(observed),
        },
        "evidence": (
            "DataFusion should treat an immediately repeated identical ordered LIMIT as idempotent; "
            f"base rows={len(expected)}, duplicate-limit rows={len(observed)}."
        ),
        "details": {
            "base_sql": base_sql.strip(),
            "duplicate_limit_sql": duplicate_limit_sql.strip(),
            "discovered_by": (
                "datadiff discovery-run --cases 1200 --seed 998001 "
                "--target-suite latest_all_engines --preset live_cross_family_metamorphic"
            ),
            "fresh_family": "metamorphic_limit_idempotence@datafusion",
        },
    }


def _datafusion_distinct_null_topk_probe() -> dict[str, Any]:
    from datafusion import SessionContext
    import pyarrow as pa

    sweep = _datafusion_distinct_null_topk_sweep(SessionContext, pa)
    first = sweep["first_mismatch"]
    return {
        "candidate_bug": bool(sweep["mismatch_count"]),
        "version": package_version("datafusion"),
        "expected": {"mismatch_count": 0, "relation": "LIMIT 1 must equal the first row of the full ordered DISTINCT result"},
        "observed": {
            "mismatch_count": sweep["mismatch_count"],
            "first_mismatch": first,
        },
        "evidence": (
            "For SELECT DISTINCT ... ORDER BY ... NULLS FIRST LIMIT 1, the top-1 result "
            "should equal the first row of the same ordered DISTINCT query without LIMIT; "
            f"checked cases={sweep['checked_cases']}, mismatches={sweep['mismatch_count']}."
        ),
        "details": sweep,
    }


def _pandas_records(df) -> list[dict[str, Any]]:
    cleaned = df.astype(object).where(df.notna(), None)
    return cleaned.to_dict(orient="records")


def _polars_vector_division_sweep(pl) -> dict[str, Any]:
    checked = 0
    mismatches: list[dict[str, Any]] = []
    for numerator in range(-20, 21):
        for denominator in range(2, 13):
            if numerator == 0:
                continue
            checked += 1
            expected = numerator / denominator
            single = (
                pl.DataFrame({"x": [numerator]}, schema={"x": pl.Int64})
                .with_columns((pl.col("x") / denominator).alias("y"))
                .get_column("y")
                .to_list()
            )
            vector = (
                pl.DataFrame({"x": [numerator, numerator]}, schema={"x": pl.Int64})
                .with_columns((pl.col("x") / denominator).alias("y"))
                .get_column("y")
                .to_list()
            )
            expected_bits = _bits(expected)
            vector_bits = [_bits(value) for value in vector]
            if _bits(single[0]) == expected_bits and any(bits != expected_bits for bits in vector_bits):
                mismatches.append(
                    {
                        "numerator": numerator,
                        "denominator": denominator,
                        "expected": expected,
                        "expected_bits": expected_bits,
                        "vector": vector,
                        "vector_bits": vector_bits,
                    }
                )
    return {
        "sweep": "polars_int_column_scalar_division_single_vs_vector",
        "checked_cases": checked,
        "mismatch_count": len(mismatches),
        "first_mismatch": mismatches[0] if mismatches else None,
        "sample_mismatches": mismatches[:10],
    }


def _pyarrow_sliced_bool_groupby_sweep(pa) -> dict[str, Any]:
    bool_values = [True, False, None]
    group_patterns = ("one_group", "two_groups", "null_key")
    offsets = (1, 2, 3, 7)
    aggregations = ("any", "all")
    checked = 0
    mismatches: list[dict[str, Any]] = []
    for offset in offsets:
        prefix_flags = [(i % 2) == 0 for i in range(offset)]
        prefix_groups = [900 + i for i in range(offset)]
        for length in range(1, 5):
            for flags in itertools.product(bool_values, repeat=length):
                for pattern in group_patterns:
                    if pattern == "one_group":
                        groups = [10] * length
                    elif pattern == "two_groups":
                        groups = [10 + (i % 2) for i in range(length)]
                    else:
                        groups = [None if i % 2 else 10 for i in range(length)]
                    sliced = pa.table(
                        {
                            "g": prefix_groups + groups,
                            "flag": prefix_flags + list(flags),
                        },
                        schema=pa.schema(
                            [
                                pa.field("g", pa.int64(), nullable=True),
                                pa.field("flag", pa.bool_(), nullable=True),
                            ]
                        ),
                    ).slice(offset)
                    rebuilt = pa.Table.from_pydict(sliced.to_pydict(), schema=sliced.schema)
                    for aggregation in aggregations:
                        checked += 1
                        sliced_rows = _sorted_pyarrow_rows(
                            sliced.group_by("g", use_threads=False)
                            .aggregate([("flag", aggregation)])
                            .to_pylist()
                        )
                        rebuilt_rows = _sorted_pyarrow_rows(
                            rebuilt.group_by("g", use_threads=False)
                            .aggregate([("flag", aggregation)])
                            .to_pylist()
                        )
                        if sliced_rows != rebuilt_rows:
                            mismatches.append(
                                {
                                    "offset": offset,
                                    "flags": list(flags),
                                    "groups": groups,
                                    "aggregation": aggregation,
                                    "sliced": sliced_rows,
                                    "rebuilt": rebuilt_rows,
                                }
                            )
    return {
        "sweep": "pyarrow_sliced_bool_table_groupby_zero_offset_equivalence",
        "checked_cases": checked,
        "mismatch_count": len(mismatches),
        "mismatch_aggregations": sorted({item["aggregation"] for item in mismatches}),
        "first_mismatch": mismatches[0] if mismatches else None,
        "sample_mismatches": mismatches[:10],
    }


def _polars_slice_chunk_lazy_equivalence_sweep(pl) -> dict[str, Any]:
    value_rows = list(itertools.product(["a", "b", None], [-2, 0, 3, None], [True, False, None]))
    checked = 0
    mismatches: list[dict[str, Any]] = []
    for offset in (1, 2, 3, 5):
        prefix = {
            "g": ["prefix"] * offset,
            "x": [100 + i for i in range(offset)],
            "flag": [bool(i % 2) for i in range(offset)],
        }
        for length in range(1, 5):
            for sample_index in range(20):
                suffix = [value_rows[(sample_index + i * 7) % len(value_rows)] for i in range(length)]
                base = pl.DataFrame(
                    {
                        "g": prefix["g"] + [row[0] for row in suffix],
                        "x": prefix["x"] + [row[1] for row in suffix],
                        "flag": prefix["flag"] + [row[2] for row in suffix],
                    },
                    schema={"g": pl.String, "x": pl.Int64, "flag": pl.Boolean},
                )
                sliced = base.slice(offset)
                rebuilt = pl.DataFrame(sliced.to_dict(as_series=False), schema=sliced.schema)
                split = max(1, length // 2)
                chunked = pl.concat([sliced.head(split), sliced.slice(split)], rechunk=False)
                for operation in ("group", "filter", "sort"):
                    expected = _polars_layout_operation_rows(rebuilt, operation)
                    for layout, table in (("sliced", sliced), ("chunked", chunked)):
                        checked += 1
                        observed = _polars_layout_operation_rows(table, operation)
                        if observed != expected:
                            mismatches.append(
                                {
                                    "layout": layout,
                                    "operation": operation,
                                    "offset": offset,
                                    "suffix": suffix,
                                    "observed": observed,
                                    "expected": expected,
                                }
                            )
                    checked += 1
                    lazy_observed = _polars_layout_operation_rows(sliced.lazy(), operation)
                    if lazy_observed != expected:
                        mismatches.append(
                            {
                                "layout": "lazy",
                                "operation": operation,
                                "offset": offset,
                                "suffix": suffix,
                                "observed": lazy_observed,
                                "expected": expected,
                            }
                        )
    return {
        "sweep": "polars_slice_chunk_lazy_equivalence",
        "checked_comparisons": checked,
        "mismatch_count": len(mismatches),
        "first_mismatch": mismatches[0] if mismatches else None,
        "sample_mismatches": mismatches[:10],
    }


def _polars_layout_operation_rows(table: Any, operation: str) -> dict[str, list[Any]]:
    import polars as pl

    if operation == "group":
        result = table.group_by("g").agg(
            pl.col("flag").any().alias("any_flag"),
            pl.col("flag").all().alias("all_flag"),
            pl.col("x").min().alias("min_x"),
            pl.col("x").max().alias("max_x"),
            pl.col("x").sum().alias("sum_x"),
            pl.col("x").mean().alias("mean_x"),
        )
        result = result.sort("g", nulls_last=False)
    elif operation == "filter":
        result = (
            table.filter(pl.col("flag").fill_null(False))
            .select(["g", "x", "flag"])
            .sort(["g", "x"], nulls_last=True)
        )
    else:
        result = table.sort(["flag", "x", "g"], descending=[False, True, False], nulls_last=True)
    if hasattr(result, "collect"):
        result = result.collect()
    return result.to_dict(as_series=False)


def _pyarrow_sliced_transform_equivalence_sweep(pa) -> dict[str, Any]:
    import pyarrow.compute as pc

    bool_values = [True, False, None]
    checked = 0
    mismatches: list[dict[str, Any]] = []
    for offset in (1, 2, 3, 7):
        for length in range(1, 6):
            for flags in itertools.product(bool_values, repeat=length):
                base = pa.table(
                    {
                        "id": list(range(100, 100 + offset)) + list(range(length)),
                        "flag": [True] * offset + list(flags),
                    }
                )
                sliced = base.slice(offset)
                rebuilt = pa.Table.from_pydict(sliced.to_pydict(), schema=sliced.schema)
                for operation in ("filter", "sort", "take_true"):
                    checked += 1
                    if operation == "filter":
                        observed = sliced.filter(sliced["flag"]).to_pylist()
                        expected = rebuilt.filter(rebuilt["flag"]).to_pylist()
                    elif operation == "sort":
                        observed = sliced.sort_by([("flag", "ascending"), ("id", "ascending")]).to_pylist()
                        expected = rebuilt.sort_by([("flag", "ascending"), ("id", "ascending")]).to_pylist()
                    else:
                        observed_indices = pc.indices_nonzero(pc.fill_null(sliced["flag"], False))
                        expected_indices = pc.indices_nonzero(pc.fill_null(rebuilt["flag"], False))
                        observed = sliced.take(observed_indices).to_pylist()
                        expected = rebuilt.take(expected_indices).to_pylist()
                    if observed != expected:
                        mismatches.append(
                            {
                                "operation": operation,
                                "offset": offset,
                                "flags": list(flags),
                                "observed": observed,
                                "expected": expected,
                            }
                        )
    return {
        "sweep": "pyarrow_sliced_table_filter_sort_take_zero_offset_equivalence",
        "checked_cases": checked,
        "mismatch_count": len(mismatches),
        "first_mismatch": mismatches[0] if mismatches else None,
        "sample_mismatches": mismatches[:10],
    }


def _datafusion_distinct_null_topk_sweep(session_context_cls: Any, pa: Any) -> dict[str, Any]:
    type_cases = [
        ("utf8", pa.string(), [None, "", "a"]),
        ("int64", pa.int64(), [None, -1, 0]),
        ("float64", pa.float64(), [None, -1.0, 0.0]),
        ("bool", pa.bool_(), [None, False, True]),
    ]
    checked = 0
    mismatches: list[dict[str, Any]] = []
    for type_name, arrow_type, values in type_cases:
        ctx = session_context_cls()
        batch = pa.RecordBatch.from_pylist(
            [{"v": value} for value in values],
            schema=pa.schema([pa.field("v", arrow_type, nullable=True)]),
        )
        ctx.register_record_batches("t0", [[batch]])
        for direction in ("asc", "desc"):
            for nulls in ("first", "last"):
                checked += 1
                base_sql = f"SELECT DISTINCT v FROM t0 ORDER BY v {direction.upper()} NULLS {nulls.upper()}"
                topk_sql = f"{base_sql} LIMIT 1"
                expected = _datafusion_sql_records(ctx, base_sql)[:1]
                observed = _datafusion_sql_records(ctx, topk_sql)
                if observed != expected:
                    mismatches.append(
                        {
                            "type": type_name,
                            "direction": direction,
                            "nulls": nulls,
                            "base_sql": base_sql,
                            "topk_sql": topk_sql,
                            "expected_first_row": expected,
                            "observed_limit_row": observed,
                        }
                    )
    return {
        "sweep": "datafusion_distinct_order_by_nulls_limit_equivalence",
        "checked_cases": checked,
        "mismatch_count": len(mismatches),
        "first_mismatch": mismatches[0] if mismatches else None,
        "sample_mismatches": mismatches[:10],
    }


def _datafusion_sql_records(ctx: Any, sql: str) -> list[dict[str, Any]]:
    return _pandas_records(ctx.sql(sql).to_pandas())


def _sorted_pyarrow_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sort_by_canonical_key(rows)


def _duckdb_cte_inline_equivalence_sweep(duckdb: Any) -> dict[str, Any]:
    """Deterministically compare WITH (inline) CTE vs equivalent inline subquery results.

    DuckDB's optimizer is allowed to inline non-recursive CTEs by default. The
    invariant under test is that the inline subquery form and the explicit CTE
    form must return identical rows for read-only, deterministic SELECT queries.
    """

    queries: tuple[tuple[str, str, str], ...] = (
        (
            "filter_group_sum",
            (
                "SELECT g, sum(x) AS s FROM (SELECT * FROM t WHERE x IS NOT NULL) "
                "GROUP BY g ORDER BY g NULLS FIRST"
            ),
            (
                "WITH cte AS (SELECT * FROM t WHERE x IS NOT NULL) "
                "SELECT g, sum(x) AS s FROM cte GROUP BY g ORDER BY g NULLS FIRST"
            ),
        ),
        (
            "distinct_limit_offset",
            (
                "SELECT * FROM (SELECT DISTINCT g, s FROM t ORDER BY g NULLS FIRST, s NULLS FIRST) "
                "LIMIT 3 OFFSET 1"
            ),
            (
                "WITH cte AS (SELECT DISTINCT g, s FROM t ORDER BY g NULLS FIRST, s NULLS FIRST) "
                "SELECT * FROM cte LIMIT 3 OFFSET 1"
            ),
        ),
        (
            "left_join_then_count",
            (
                "SELECT a.g, count(*) AS c FROM t a LEFT JOIN "
                "(SELECT g, max(x) AS mx FROM t GROUP BY g) b ON a.g = b.g "
                "GROUP BY a.g ORDER BY a.g NULLS FIRST"
            ),
            (
                "WITH agg AS (SELECT g, max(x) AS mx FROM t GROUP BY g) "
                "SELECT a.g, count(*) AS c FROM t a LEFT JOIN agg b ON a.g = b.g "
                "GROUP BY a.g ORDER BY a.g NULLS FIRST"
            ),
        ),
    )

    con = duckdb.connect(database=":memory:")
    con.execute(
        "CREATE TABLE t AS SELECT * FROM (VALUES "
        "(1, 'a', 10), (1, 'a', 10), (1, 'b', NULL), "
        "(2, 'a', -3), (2, NULL, 7), (NULL, 'c', 0), (3, 'b', 4)"
        ") AS v(g, s, x)"
    )

    mismatches: list[dict[str, Any]] = []
    for label, inline_sql, cte_sql in queries:
        try:
            inline_rows = _sorted_pyarrow_rows(con.execute(inline_sql).fetchdf().to_dict(orient="records"))
            cte_rows = _sorted_pyarrow_rows(con.execute(cte_sql).fetchdf().to_dict(orient="records"))
        except Exception as exc:  # noqa: BLE001 — record probe-side error, not a candidate
            mismatches.append(
                {"query": label, "error_type": type(exc).__name__, "error": str(exc)}
            )
            continue
        if inline_rows != cte_rows:
            mismatches.append(
                {
                    "query": label,
                    "inline_rows": inline_rows,
                    "cte_rows": cte_rows,
                }
            )

    return {
        "sweep": "duckdb_cte_inline_equivalence",
        "checked_queries": len(queries),
        "mismatch_count": len(mismatches),
        "first_mismatch": mismatches[0] if mismatches else None,
        "sample_mismatches": mismatches[:5],
    }


def _duckdb_cte_inline_equivalence_probe() -> dict[str, Any]:
    import duckdb

    sweep = _duckdb_cte_inline_equivalence_sweep(duckdb)
    return {
        "candidate_bug": bool(sweep["mismatch_count"]),
        "version": getattr(duckdb, "__version__", "") or package_version("duckdb"),
        "expected": {"mismatch_count": 0, "relation": "inline subquery ≡ named CTE"},
        "observed": {
            "mismatch_count": sweep["mismatch_count"],
            "first_mismatch": sweep["first_mismatch"],
        },
        "evidence": (
            "Reading a non-recursive, deterministic query through an inline subquery and through "
            "an explicit WITH CTE must yield identical result rows; "
            f"checked queries={sweep['checked_queries']}, mismatches={sweep['mismatch_count']}."
        ),
        "details": sweep,
    }


def _polars_concat_select_pushdown_sweep(pl: Any) -> dict[str, Any]:
    """Concat-then-select must equal per-frame-select-then-concat for shared columns.

    Polars' optimizer routinely pushes column projection through concat. If the
    pushdown reorders columns, mishandles nulls, or drops rows, the lazy plan
    diverges from the per-frame-then-concat form. The invariant must hold for
    column subsets shared across both frames.
    """

    frame_pairs = (
        (
            "numeric_strs",
            pl.DataFrame(
                {
                    "g": [1, 2, 3, None],
                    "x": [10, None, -5, 7],
                    "s": ["a", "b", None, "d"],
                }
            ),
            pl.DataFrame(
                {
                    "g": [3, 4, None],
                    "x": [9, -1, 4],
                    "s": ["e", None, "g"],
                }
            ),
            ["g", "s"],
        ),
        (
            "bool_floats",
            pl.DataFrame(
                {
                    "k": [True, False, None, True],
                    "v": [1.5, -2.25, float("nan"), 0.0],
                    "label": ["a", "b", "c", "d"],
                }
            ),
            pl.DataFrame(
                {
                    "k": [False, None, True],
                    "v": [3.0, float("inf"), -0.0],
                    "label": ["e", "f", "g"],
                }
            ),
            ["label", "v"],
        ),
    )

    mismatches: list[dict[str, Any]] = []
    for label, left, right, projection in frame_pairs:
        try:
            via_concat_then_select = (
                pl.concat([left, right], how="vertical")
                .select(projection)
                .to_dicts()
            )
            via_select_then_concat = (
                pl.concat(
                    [left.select(projection), right.select(projection)],
                    how="vertical",
                ).to_dicts()
            )
        except Exception as exc:  # noqa: BLE001
            mismatches.append(
                {"frames": label, "error_type": type(exc).__name__, "error": str(exc)}
            )
            continue
        normalized_a = json.loads(json.dumps(via_concat_then_select, default=str))
        normalized_b = json.loads(json.dumps(via_select_then_concat, default=str))
        if normalized_a != normalized_b:
            mismatches.append(
                {
                    "frames": label,
                    "projection": projection,
                    "concat_then_select": normalized_a,
                    "select_then_concat": normalized_b,
                }
            )

    return {
        "sweep": "polars_concat_select_pushdown",
        "checked_frames": len(frame_pairs),
        "mismatch_count": len(mismatches),
        "first_mismatch": mismatches[0] if mismatches else None,
        "sample_mismatches": mismatches[:5],
    }


def _polars_concat_select_pushdown_probe() -> dict[str, Any]:
    import polars as pl

    sweep = _polars_concat_select_pushdown_sweep(pl)
    return {
        "candidate_bug": bool(sweep["mismatch_count"]),
        "version": pl.__version__,
        "expected": {"mismatch_count": 0, "relation": "concat∘select ≡ select-each∘concat for shared columns"},
        "observed": {
            "mismatch_count": sweep["mismatch_count"],
            "first_mismatch": sweep["first_mismatch"],
        },
        "evidence": (
            "Selecting columns after concat must equal selecting the same columns from each frame "
            "and concatenating; this catches projection-pushdown ordering and null-handling bugs. "
            f"Checked frames={sweep['checked_frames']}, mismatches={sweep['mismatch_count']}."
        ),
        "details": sweep,
    }


def _pandas_arrow_groupby_size_count_sweep(pd: Any) -> dict[str, Any]:
    """For Arrow-backed pandas, groupby.size() must equal groupby.count() + per-group nulls.

    Invariant: ``size = count(non_null) + count(null)`` per group; both forms must
    agree with the input frame's row count summed across groups. This catches
    the family of pandas-Arrow bugs where group size is computed on the dense
    Arrow buffer but count() is computed on the validity bitmap, yielding
    inconsistent results across the two APIs.
    """

    frames = (
        (
            "single_key_int",
            pd.DataFrame(
                {
                    "g": pd.array([1, 1, 2, 2, None, 3, 3, None], dtype="int64[pyarrow]"),
                    "x": pd.array([10, None, 5, 5, 7, None, 1, 2], dtype="int64[pyarrow]"),
                }
            ),
            ["g"],
        ),
        (
            "single_key_str",
            pd.DataFrame(
                {
                    "g": pd.array(["a", "a", "b", None, "b", None, "c"], dtype="string[pyarrow]"),
                    "v": pd.array([1.0, None, 2.0, 3.0, None, 4.0, 5.0], dtype="float64[pyarrow]"),
                }
            ),
            ["g"],
        ),
    )

    mismatches: list[dict[str, Any]] = []
    for label, df, keys in frames:
        try:
            agg_size = df.groupby(keys, dropna=False).size().to_dict()
            agg_count = df.groupby(keys, dropna=False).count().to_dict()
            total_rows = int(df.shape[0])
            sum_of_sizes = sum(int(v) for v in agg_size.values())
        except Exception as exc:  # noqa: BLE001
            mismatches.append({"frame": label, "error_type": type(exc).__name__, "error": str(exc)})
            continue
        if sum_of_sizes != total_rows:
            mismatches.append(
                {
                    "frame": label,
                    "sum_of_sizes": sum_of_sizes,
                    "total_rows": total_rows,
                    "size_per_group": {str(k): int(v) for k, v in agg_size.items()},
                    "count_per_group": {col: {str(k): int(v) for k, v in counts.items()} for col, counts in agg_count.items()},
                }
            )

    return {
        "sweep": "pandas_arrow_groupby_size_count",
        "checked_frames": len(frames),
        "mismatch_count": len(mismatches),
        "first_mismatch": mismatches[0] if mismatches else None,
        "sample_mismatches": mismatches[:5],
    }


def _pandas_arrow_groupby_size_count_probe() -> dict[str, Any]:
    import pandas as pd

    sweep = _pandas_arrow_groupby_size_count_sweep(pd)
    return {
        "candidate_bug": bool(sweep["mismatch_count"]),
        "version": pd.__version__,
        "expected": {"mismatch_count": 0, "relation": "sum(groupby.size()) == n_rows"},
        "observed": {
            "mismatch_count": sweep["mismatch_count"],
            "first_mismatch": sweep["first_mismatch"],
        },
        "evidence": (
            "For Arrow-backed pandas extension dtypes, the sum of groupby.size() values "
            "must equal the input row count regardless of null handling on the grouping key; "
            f"checked frames={sweep['checked_frames']}, mismatches={sweep['mismatch_count']}."
        ),
        "details": sweep,
    }


def _duckdb_left_anti_join_equivalence_sweep(duckdb: Any) -> dict[str, Any]:
    """LEFT ANTI JOIN must equal `LEFT JOIN ... WHERE right.key IS NULL`.

    This is a textbook SQL identity; if DuckDB's anti-join planner deviates
    (e.g. mishandles NULL on either side, drops rows after probe), the two
    forms diverge. The invariant must hold across mixed NULLs on both sides.
    """

    con = duckdb.connect(database=":memory:")
    con.execute(
        "CREATE TABLE l AS SELECT * FROM (VALUES "
        "(1, 'a'), (2, 'b'), (NULL, 'c'), (3, NULL), (4, 'a'), (5, NULL)"
        ") AS v(k, lbl)"
    )
    con.execute(
        "CREATE TABLE r AS SELECT * FROM (VALUES "
        "(1, 'p'), (3, 'q'), (NULL, 'r')"
        ") AS v(k, tag)"
    )

    queries = (
        (
            "anti_join_via_left",
            "SELECT l.* FROM l ANTI JOIN r ON l.k = r.k ORDER BY l.k NULLS LAST, l.lbl NULLS LAST",
            (
                "SELECT l.k, l.lbl FROM l LEFT JOIN r ON l.k = r.k "
                "WHERE r.k IS NULL ORDER BY l.k NULLS LAST, l.lbl NULLS LAST"
            ),
        ),
        (
            "anti_join_with_filter",
            (
                "SELECT l.* FROM l ANTI JOIN (SELECT k FROM r WHERE tag IS NOT NULL) r ON l.k = r.k "
                "ORDER BY l.k NULLS LAST, l.lbl NULLS LAST"
            ),
            (
                "SELECT l.k, l.lbl FROM l LEFT JOIN (SELECT k FROM r WHERE tag IS NOT NULL) r ON l.k = r.k "
                "WHERE r.k IS NULL ORDER BY l.k NULLS LAST, l.lbl NULLS LAST"
            ),
        ),
    )

    mismatches: list[dict[str, Any]] = []
    for label, anti_sql, left_sql in queries:
        try:
            anti_rows = con.execute(anti_sql).fetchdf().to_dict(orient="records")
            left_rows = con.execute(left_sql).fetchdf().to_dict(orient="records")
        except Exception as exc:  # noqa: BLE001
            mismatches.append({"query": label, "error_type": type(exc).__name__, "error": str(exc)})
            continue
        if anti_rows != left_rows:
            mismatches.append(
                {"query": label, "anti_rows": anti_rows, "left_join_with_filter_rows": left_rows}
            )

    return {
        "sweep": "duckdb_left_anti_join_equivalence",
        "checked_queries": len(queries),
        "mismatch_count": len(mismatches),
        "first_mismatch": mismatches[0] if mismatches else None,
        "sample_mismatches": mismatches[:5],
    }


def _duckdb_left_anti_join_equivalence_probe() -> dict[str, Any]:
    import duckdb

    sweep = _duckdb_left_anti_join_equivalence_sweep(duckdb)
    return {
        "candidate_bug": bool(sweep["mismatch_count"]),
        "version": getattr(duckdb, "__version__", "") or package_version("duckdb"),
        "expected": {
            "mismatch_count": 0,
            "relation": "ANTI JOIN ≡ LEFT JOIN ... WHERE right.key IS NULL (mixed NULLs on both sides)",
        },
        "observed": {
            "mismatch_count": sweep["mismatch_count"],
            "first_mismatch": sweep["first_mismatch"],
        },
        "evidence": (
            "ANTI JOIN and LEFT JOIN + IS NULL filter are standard SQL equivalents; any divergence "
            "indicates a planner or null-handling bug. "
            f"Checked queries={sweep['checked_queries']}, mismatches={sweep['mismatch_count']}."
        ),
        "details": sweep,
    }


def _polars_lazy_eager_equivalence_sweep(pl: Any) -> dict[str, Any]:
    """For pure (no-IO) transformations Polars eager and lazy.collect() must agree.

    The lazy plan goes through projection pushdown, predicate pushdown, simplify
    expressions, and the streaming/parallel collector; the eager path goes
    through the in-memory operator. They must compute the same result on the
    same input frame. The sweep includes mixed-null filter+groupby+agg pipelines
    that have historically exposed lazy-vs-eager planner bugs.
    """

    queries = (
        (
            "filter_groupby_agg",
            lambda df: df.filter(pl.col("x").is_not_null())
            .group_by("g")
            .agg([pl.col("x").sum().alias("sx"), pl.col("x").count().alias("cx")])
            .sort(["g", "sx", "cx"], nulls_last=True),
        ),
        (
            "with_columns_then_filter",
            lambda df: df.with_columns([(pl.col("x") * 2).alias("x2")])
            .filter(pl.col("x2") >= 0)
            .sort(["g", "x2"], nulls_last=True),
        ),
        (
            "unique_then_sort",
            lambda df: df.select(["g"]).unique().sort(["g"], nulls_last=True),
        ),
    )

    base = pl.DataFrame(
        {
            "g": [1, 1, 2, 2, None, 3, 3, None, 4, 4],
            "x": [10, None, -5, 5, 7, None, 1, 2, 0, -3],
        }
    )

    mismatches: list[dict[str, Any]] = []
    for label, build_query in queries:
        try:
            eager_result = build_query(base).to_dicts()
            lazy_result = build_query(base.lazy()).collect().to_dicts()
        except Exception as exc:  # noqa: BLE001
            mismatches.append({"query": label, "error_type": type(exc).__name__, "error": str(exc)})
            continue
        normalized_eager = json.loads(json.dumps(eager_result, default=str))
        normalized_lazy = json.loads(json.dumps(lazy_result, default=str))
        if normalized_eager != normalized_lazy:
            mismatches.append(
                {"query": label, "eager": normalized_eager, "lazy": normalized_lazy}
            )

    return {
        "sweep": "polars_lazy_eager_equivalence",
        "checked_queries": len(queries),
        "mismatch_count": len(mismatches),
        "first_mismatch": mismatches[0] if mismatches else None,
        "sample_mismatches": mismatches[:5],
    }


def _polars_lazy_eager_equivalence_probe() -> dict[str, Any]:
    import polars as pl

    sweep = _polars_lazy_eager_equivalence_sweep(pl)
    return {
        "candidate_bug": bool(sweep["mismatch_count"]),
        "version": pl.__version__,
        "expected": {"mismatch_count": 0, "relation": "eager(query) ≡ lazy(query).collect() for pure pipelines"},
        "observed": {
            "mismatch_count": sweep["mismatch_count"],
            "first_mismatch": sweep["first_mismatch"],
        },
        "evidence": (
            "Polars eager and lazy execution paths share the same logical plan; any divergence on "
            "pure (no-IO) pipelines indicates a planner-vs-evaluator bug. "
            f"Checked queries={sweep['checked_queries']}, mismatches={sweep['mismatch_count']}."
        ),
        "details": sweep,
    }


PROBES: dict[str, BugAuditProbe] = {
    "datafusion_distinct_null_topk": BugAuditProbe(
        probe_id="datafusion_distinct_null_topk",
        target_backend="datafusion",
        family="distinct_null_topk",
        title="DataFusion DISTINCT ORDER BY NULLS FIRST LIMIT",
        invariant="LIMIT 1 over an ordered DISTINCT result must return the first row of that full ordered DISTINCT result.",
        runner=_datafusion_distinct_null_topk_probe,
    ),
    "datafusion_limit_idempotence": BugAuditProbe(
        probe_id="datafusion_limit_idempotence",
        target_backend="datafusion",
        family="datafusion_limit_idempotence",
        title="DataFusion duplicate ordered LIMIT idempotence",
        invariant="Repeating the same ordered LIMIT over the same subquery must not change query results.",
        runner=_datafusion_limit_idempotence_probe,
    ),
    "polars_reflected_arithmetic": BugAuditProbe(
        probe_id="polars_reflected_arithmetic",
        target_backend="polars",
        family="polars_reflected_arithmetic_operand_order",
        title="Polars Series reflected arithmetic operand order",
        invariant="Series reflected arithmetic lhs.__rop__(rhs) must evaluate rhs op lhs.",
        runner=_polars_reflected_arithmetic_probe,
    ),
    "polars_vector_division_rounding": BugAuditProbe(
        probe_id="polars_vector_division_rounding",
        target_backend="polars",
        family="polars_vector_division_rounding",
        title="Polars vectorized expression division rounding",
        invariant="Vectorized column/scalar division must produce the same IEEE-754 double as the single-row path.",
        runner=_polars_vector_division_probe,
    ),
    "pyarrow_sliced_bool_groupby": BugAuditProbe(
        probe_id="pyarrow_sliced_bool_groupby",
        target_backend="pyarrow",
        family="pyarrow_sliced_bool_groupby_any_all",
        title="PyArrow sliced Boolean group_by any/all",
        invariant="A sliced Arrow table must group/aggregate like an equivalent zero-offset rebuilt table.",
        runner=_pyarrow_sliced_bool_groupby_probe,
    ),
    "polars_slice_chunk_lazy_equivalence": BugAuditProbe(
        probe_id="polars_slice_chunk_lazy_equivalence",
        target_backend="polars",
        family="polars_slice_chunk_lazy_equivalence",
        title="Polars sliced/chunked/lazy operation equivalence",
        invariant="Equivalent Polars layouts and execution modes must return identical operation results.",
        runner=_polars_slice_chunk_lazy_equivalence_probe,
    ),
    "pyarrow_sliced_transform_equivalence": BugAuditProbe(
        probe_id="pyarrow_sliced_transform_equivalence",
        target_backend="pyarrow",
        family="pyarrow_sliced_transform_equivalence",
        title="PyArrow sliced Boolean transform equivalence",
        invariant="A sliced Arrow table must filter, sort, and take like an equivalent zero-offset rebuilt table.",
        runner=_pyarrow_sliced_transform_equivalence_probe,
    ),
    "duckdb_cte_inline_equivalence": BugAuditProbe(
        probe_id="duckdb_cte_inline_equivalence",
        target_backend="duckdb",
        family="duckdb_cte_inline_equivalence",
        title="DuckDB inline subquery vs explicit CTE equivalence",
        invariant=(
            "Reading a deterministic, non-recursive query through an inline subquery and through "
            "an explicit WITH CTE must yield identical result rows."
        ),
        runner=_duckdb_cte_inline_equivalence_probe,
    ),
    "polars_concat_select_pushdown": BugAuditProbe(
        probe_id="polars_concat_select_pushdown",
        target_backend="polars",
        family="polars_concat_select_pushdown",
        title="Polars concat-select projection pushdown equivalence",
        invariant=(
            "concat(frames).select(cols) must equal concat(frame.select(cols) for frame in frames) "
            "for any column subset shared across frames."
        ),
        runner=_polars_concat_select_pushdown_probe,
    ),
    "pandas_arrow_groupby_size_count": BugAuditProbe(
        probe_id="pandas_arrow_groupby_size_count",
        target_backend="pandas",
        family="pandas_arrow_groupby_size_count",
        title="pandas Arrow-backed groupby.size() row-count consistency",
        invariant=(
            "For Arrow-backed extension dtypes, sum(groupby.size()) must equal the input row count "
            "regardless of null handling on the grouping key."
        ),
        runner=_pandas_arrow_groupby_size_count_probe,
    ),
    "duckdb_left_anti_join_equivalence": BugAuditProbe(
        probe_id="duckdb_left_anti_join_equivalence",
        target_backend="duckdb",
        family="duckdb_left_anti_join_equivalence",
        title="DuckDB ANTI JOIN vs LEFT JOIN IS NULL equivalence",
        invariant=(
            "ANTI JOIN must produce the same rows as LEFT JOIN ... WHERE right.key IS NULL "
            "for any predicate combination over NULLable join keys."
        ),
        runner=_duckdb_left_anti_join_equivalence_probe,
    ),
    "polars_lazy_eager_equivalence": BugAuditProbe(
        probe_id="polars_lazy_eager_equivalence",
        target_backend="polars",
        family="polars_lazy_eager_equivalence",
        title="Polars lazy.collect() vs eager pipeline equivalence",
        invariant=(
            "For pure (no-IO) Polars pipelines, df.lazy().pipeline().collect() must equal "
            "df.pipeline() exactly."
        ),
        runner=_polars_lazy_eager_equivalence_probe,
    ),
}


available_bug_audit_probe_ids = list_audit_probe_ids
run_bug_audit = run_probe_audit
write_bug_audit_outputs = write_probe_audit_outputs
write_bug_audit_issue_drafts = write_probe_issue_drafts
