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


def _project_display_path(value: str) -> str:
    if not value:
        return ""
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


available_bug_audit_probe_ids = list_audit_probe_ids
run_bug_audit = run_probe_audit
write_bug_audit_outputs = write_probe_audit_outputs
write_bug_audit_issue_drafts = write_probe_issue_drafts
_bug_audit_markdown = render_probe_audit_markdown


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
}
