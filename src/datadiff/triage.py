from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from datadiff.dsl import Case
from datadiff.util import dump_json


def build_triage_report(
    case: Case,
    original_findings: list[dict[str, Any]],
    reproduced_findings: list[dict[str, Any]],
    config: dict[str, Any],
    backends: list[str],
) -> dict[str, Any]:
    original_kinds = sorted({f.get("kind", "") for f in original_findings if f.get("kind")})
    reproduced_kinds = sorted({f.get("kind", "") for f in reproduced_findings if f.get("kind")})
    reproduced_roots = sorted({f.get("root_cause", "unknown") for f in reproduced_findings})
    suspicious_backends = sorted({b for f in reproduced_findings for b in f.get("suspicious_backends", [])})

    reproduced = bool(set(original_kinds) & set(reproduced_kinds))
    features = _case_features(case)
    generator_profile = config.get("generator_profile", "unknown")
    semantic_boundary = _is_semantic_boundary(features, reproduced_roots, generator_profile)
    documented_divergence = _is_documented_polars_nan_semantics(features, reproduced_findings)
    preclassified = _preclassified_verdict(reproduced_findings)

    if not reproduced:
        verdict = "not_reproduced"
        paper_status = "not_usable_until_reproduced"
        confidence = "low"
    elif preclassified is not None:
        verdict = preclassified["verdict"]
        paper_status = preclassified["paper_status"]
        confidence = preclassified["confidence"]
    elif documented_divergence:
        verdict = "documented_semantic_divergence"
        paper_status = "valid_finding_not_bug"
        confidence = "high"
    elif semantic_boundary:
        verdict = "expected_semantic_divergence"
        paper_status = "valid_finding_not_bug"
        confidence = "medium"
    elif _has_clear_minority_backend(reproduced_findings, backends):
        verdict = "candidate_implementation_bug"
        paper_status = "candidate_bug_needs_external_confirmation"
        confidence = "high"
    else:
        verdict = "needs_manual_confirmation"
        paper_status = "valid_finding_needs_triage"
        confidence = "medium"

    return {
        "case_id": case.case_id,
        "seed": case.seed,
        "verdict": verdict,
        "paper_status": paper_status,
        "triage_confidence": confidence,
        "generator_profile": generator_profile,
        "backends": backends,
        "features": features,
        "original_kinds": original_kinds,
        "reproduced_kinds": reproduced_kinds,
        "reproduced_roots": reproduced_roots,
        "suspicious_backends": suspicious_backends,
        "false_positive_reasons": sorted({f.get("false_positive_reason", "") for f in reproduced_findings if f.get("false_positive_reason")}),
        "documentation_refs": _documentation_refs(verdict),
        "recommendation": _recommendation(verdict),
    }


def _preclassified_verdict(findings: list[dict[str, Any]]) -> dict[str, str] | None:
    priority = [
        "generator_false_positive",
        "normalizer_false_positive",
        "documented_semantic_divergence",
        "expected_semantic_divergence",
        "semantic_divergence_needs_confirmation",
        "candidate_implementation_bug",
    ]
    by_verdict = {f.get("triage_verdict"): f for f in findings if f.get("triage_verdict") and f.get("triage_verdict") != "unclassified"}
    for verdict in priority:
        finding = by_verdict.get(verdict)
        if finding is not None:
            return {
                "verdict": verdict,
                "paper_status": finding.get("paper_status", _paper_status_for_verdict(verdict)),
                "confidence": finding.get("triage_confidence", "medium"),
            }
    return None


def _paper_status_for_verdict(verdict: str) -> str:
    if verdict == "generator_false_positive":
        return "exclude_generator_invalid_case"
    if verdict == "normalizer_false_positive":
        return "exclude_normalizer_failure"
    if verdict == "documented_semantic_divergence":
        return "valid_finding_not_bug"
    if verdict == "expected_semantic_divergence":
        return "valid_finding_not_bug"
    if verdict == "semantic_divergence_needs_confirmation":
        return "valid_finding_not_confirmed_bug"
    if verdict == "candidate_implementation_bug":
        return "candidate_bug_needs_external_confirmation"
    return "valid_finding_needs_triage"


def write_triage_artifact(bug_dir: Path, report: dict[str, Any]) -> tuple[Path, Path]:
    json_path = bug_dir / "triage.json"
    md_path = bug_dir / "triage.md"
    dump_json(report, json_path)
    md_path.write_text(_triage_markdown(report), encoding="utf-8")
    return json_path, md_path


def write_standalone_reproducer(bug_dir: Path, report: dict[str, Any] | None = None) -> Path:
    roots = set((report or {}).get("reproduced_roots", []))
    if "grouped_topk_null_sort_key" in roots:
        path = bug_dir / "standalone_datafusion_groupby_null_sortkey_limit.py"
        content = _standalone_datafusion_groupby_null_sortkey_reproducer()
    else:
        path = bug_dir / "standalone_edge_float_reproducer.py"
        content = _standalone_edge_float_reproducer()
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)
    return path


def supports_standalone_reproducer(report: dict[str, Any]) -> bool:
    features = report.get("features", {})
    roots = set(report.get("reproduced_roots", []))
    return (
        bool(roots & {"grouped_topk_null_sort_key"})
        or report.get("generator_profile") == "edge_float"
        or bool(features.get("contains_nan"))
        or bool(features.get("contains_inf"))
        or bool(roots & {"nan_inf_semantics"})
    )


def _case_features(case: Case) -> dict[str, Any]:
    ops = case.program.op_sequence()
    return {
        "contains_null": _case_contains_null(case),
        "contains_nan": _case_contains_nan(case),
        "contains_inf": _case_contains_inf(case),
        "contains_non_ascii_string": _case_contains_non_ascii_string(case),
        "uses_filter": "filter" in ops,
        "uses_mutate": "mutate" in ops,
        "uses_modulo": _case_uses_modulo(case),
        "uses_string_lower": _case_uses_string_lower(case),
        "uses_groupby": "groupby" in ops,
        "uses_sort": "sort" in ops,
        "uses_limit": "limit" in ops,
        "uses_offset": "offset" in ops,
        "operation_sequence": ops,
    }


def _is_semantic_boundary(features: dict[str, Any], roots: list[str], generator_profile: str) -> bool:
    if features["contains_nan"] or features["contains_inf"]:
        return True
    if features["uses_modulo"]:
        return True
    if features["uses_string_lower"] and features["contains_non_ascii_string"]:
        return True
    return any(root in {"nan_inf_semantics", "null_semantics", "ordering_or_limit"} for root in roots)


def _has_clear_minority_backend(findings: list[dict[str, Any]], backends: list[str]) -> bool:
    backend_count = len(backends)
    for finding in findings:
        suspicious = finding.get("suspicious_backends", [])
        if 0 < len(suspicious) < backend_count and finding.get("confidence") == "high":
            return True
    return False


def _is_documented_polars_nan_semantics(
    features: dict[str, Any],
    reproduced_findings: list[dict[str, Any]],
) -> bool:
    if not features["contains_nan"]:
        return False
    return any(
        "polars" in finding.get("suspicious_backends", [])
        and finding.get("root_cause") == "nan_inf_semantics"
        for finding in reproduced_findings
    )


def _recommendation(verdict: str) -> list[str]:
    if verdict == "not_reproduced":
        return [
            "Do not count this artifact as a bug until reproduction succeeds.",
            "Check dependency versions and backend list.",
        ]
    if verdict == "semantic_divergence_needs_confirmation":
        return [
            "Report as a reproducible semantic divergence, not a confirmed implementation bug.",
            "Minimize the case and compare against each backend's documented NaN/NULL/order semantics.",
            "Submit an upstream issue only after the target backend's documented behavior is contradicted.",
        ]
    if verdict == "documented_semantic_divergence":
        return [
            "Do not count this as a confirmed implementation bug.",
            "Use it as a documented semantic-divergence finding for the edge_float profile.",
            "Keep it in the benchmark suite to test whether the fuzzer separates boundary semantics from common-subset bugs.",
        ]
    if verdict == "expected_semantic_divergence":
        return [
            "Do not count this as a confirmed implementation bug.",
            "Report it as an expected semantic-divergence finding for a boundary operation.",
            "Keep it separate from common-subset implementation-bug counts.",
        ]
    if verdict == "generator_false_positive":
        return [
            "Do not count this as a backend bug.",
            "Add a generator regression test and fix the DSL repair logic.",
            "Rerun after the generated program is valid across all configured backends.",
        ]
    if verdict == "normalizer_false_positive":
        return [
            "Do not count this as a backend bug.",
            "Fix normalization or inspect raw backend outputs before claiming a semantic mismatch.",
            "Rerun with the corrected normalizer and compare signatures again.",
        ]
    if verdict == "candidate_implementation_bug":
        return [
            "Minimize the artifact and create a backend-specific reproduction script.",
            "Check documentation and release notes, then submit an upstream issue.",
            "Count as confirmed only after maintainer acknowledgement, fix, or clear spec violation.",
        ]
    return [
        "Keep as a valid finding, but perform manual triage before claiming a bug.",
        "Add a smaller reproducer and classify the expected semantics.",
    ]


def _documentation_refs(verdict: str) -> list[dict[str, str]]:
    if verdict != "documented_semantic_divergence":
        return []
    return [
        {
            "title": "Polars user guide: floating point numbers",
            "url": "https://docs.pola.rs/user-guide/concepts/data-types-and-structures/#floating-point-numbers",
            "note": "Polars documents that NaN is considered larger than any non-NaN floating-point value.",
        },
        {
            "title": "Polars user guide: missing data",
            "url": "https://docs.pola.rs/user-guide/expressions/missing-data/#not-a-number-or-nan-values",
            "note": "Polars documents that NaN values are not missing values; null is the missing-data representation.",
        },
    ]


def _triage_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Triage Report: {report['case_id']}",
        "",
        f"- Verdict: `{report['verdict']}`",
        f"- Paper status: `{report['paper_status']}`",
        f"- Confidence: `{report['triage_confidence']}`",
        f"- Generator profile: `{report['generator_profile']}`",
        f"- Backends: {', '.join(report['backends'])}",
        f"- Rows: {report.get('rows', 'n/a')}",
        f"- Operations: {report.get('operations', 'n/a')}",
        f"- Reduced: {report.get('reduced', False)}",
        f"- Original kinds: {', '.join(report['original_kinds']) or 'none'}",
        f"- Reproduced kinds: {', '.join(report['reproduced_kinds']) or 'none'}",
        f"- Reproduced roots: {', '.join(report['reproduced_roots']) or 'none'}",
        f"- Suspicious backends: {', '.join(report['suspicious_backends']) or 'none'}",
        "",
    ]
    if report.get("documentation_refs"):
        lines.extend(["## Documentation References", ""])
        for ref in report["documentation_refs"]:
            lines.append(f"- [{ref['title']}]({ref['url']}): {ref['note']}")
        lines.append("")
    lines.extend(["## Features", ""])
    for key, value in report["features"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Recommendation", ""])
    for item in report["recommendation"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def _case_contains_null(case: Case) -> bool:
    return any(value is None for table in case.tables for row in table.rows for value in row.values())


def _case_contains_nan(case: Case) -> bool:
    return any(
        isinstance(value, float) and math.isnan(value)
        for table in case.tables
        for row in table.rows
        for value in row.values()
    )


def _case_contains_inf(case: Case) -> bool:
    return any(
        isinstance(value, float) and math.isinf(value)
        for table in case.tables
        for row in table.rows
        for value in row.values()
    )


def _case_contains_non_ascii_string(case: Case) -> bool:
    return any(
        isinstance(value, str) and any(ord(ch) > 127 for ch in value)
        for table in case.tables
        for row in table.rows
        for value in row.values()
    )


def _case_uses_modulo(case: Case) -> bool:
    return any(
        op.get("op") == "mutate"
        and op.get("expr", {}).get("kind") == "arith_const"
        and op.get("expr", {}).get("op") == "mod"
        for op in case.program.operations
    )


def _case_uses_string_lower(case: Case) -> bool:
    return any(
        op.get("op") == "mutate" and op.get("expr", {}).get("kind") == "string_lower"
        for op in case.program.operations
    )


def _standalone_datafusion_groupby_null_sortkey_reproducer() -> str:
    return '''#!/usr/bin/env python3
"""Standalone reproduction for DataFusion grouped top-k NULL sort-key loss.

This script does not import DataDiffFuzz. It constructs one grouped aggregate
whose sort key is NULL, then compares the control query with the top-k query.
"""

from __future__ import annotations

import datafusion
import pyarrow as pa
from datafusion import SessionContext


def _register(rows: list[tuple[str, int | None]]) -> SessionContext:
    ctx = SessionContext()
    batch = pa.RecordBatch.from_arrays(
        [
            pa.array([row[0] for row in rows], type=pa.string()),
            pa.array([row[1] for row in rows], type=pa.int64()),
        ],
        schema=pa.schema(
            [
                pa.field("g", pa.string(), nullable=True),
                pa.field("x", pa.int64(), nullable=True),
            ]
        ),
    )
    ctx.register_record_batches("t0", [[batch]])
    return ctx


def _row_count(ctx: SessionContext, sql: str) -> int:
    return sum(batch.num_rows for batch in ctx.sql(sql).collect())


def main() -> None:
    base = "SELECT g, MIN(x) AS min_x FROM t0 GROUP BY g"
    control_query = f"SELECT min_x FROM ({base}) q LIMIT 20"
    failing_query = f"SELECT min_x FROM ({base}) q ORDER BY min_x ASC NULLS LAST LIMIT 20"

    print(f"datafusion={getattr(datafusion, '__version__', 'unknown')}")
    print(f"pyarrow={pa.__version__}")

    ctx = _register([("a", None)])
    control = ctx.sql(control_query).to_pandas()
    print("control:")
    print(control)

    ctx = _register([("a", None)])
    failing = ctx.sql(failing_query).to_pandas()
    print("top-k:")
    print(failing)

    ctx = _register([("a", None)])
    failing_rows = _row_count(ctx, failing_query)
    print(f"top-k record-batch rows={failing_rows}")

    assert len(control) == 1, "control query should return the grouped NULL aggregate"
    assert failing_rows == 1, "DataFusion dropped the group whose aggregate sort key is NULL"


if __name__ == "__main__":
    main()
'''


def _standalone_edge_float_reproducer() -> str:
    return '''#!/usr/bin/env python3
"""Standalone reproduction for the edge_float semantic divergence.

This script does not import DataDiffFuzz. It shows the core behavior directly
in Polars and compares it with the other tested engines when installed.
"""

from __future__ import annotations

import sqlite3


def run_polars() -> None:
    import polars as pl

    df = pl.DataFrame({"y": [float("nan")]})
    observed = df.with_columns(
        (pl.col("y") + 10).alias("m_0"),
        ((pl.col("y") + 10) > 10.0).alias("predicate"),
        (pl.col("y") + 10).is_nan().alias("m0_is_nan"),
        (pl.col("y") + 10).is_null().alias("m0_is_null"),
    )
    filtered = df.with_columns((pl.col("y") + 10).alias("m_0")).filter(pl.col("m_0") > 10.0)
    print("polars diagnostic")
    print(observed)
    print("polars filtered rows:", filtered.height)


def run_pandas() -> None:
    import pandas as pd

    df = pd.DataFrame({"y": [float("nan")]})
    df["m_0"] = df["y"] + 10
    filtered = df[df["m_0"] > 10.0]
    print("pandas filtered rows:", len(filtered))


def run_duckdb() -> None:
    import duckdb
    import pandas as pd

    df = pd.DataFrame({"y": [float("nan")]})
    con = duckdb.connect(database=":memory:")
    con.register("t0", df)
    out = con.execute("SELECT y, y + 10 AS m_0 FROM t0 WHERE y + 10 > 10.0").fetchall()
    print("duckdb filtered rows:", len(out))


def run_sqlite() -> None:
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE t0 (y REAL)")
    con.execute("INSERT INTO t0 VALUES (?)", (float("nan"),))
    out = con.execute("SELECT y, y + 10 AS m_0 FROM t0 WHERE y + 10 > 10.0").fetchall()
    print("sqlite filtered rows:", len(out))


if __name__ == "__main__":
    run_polars()
    run_pandas()
    run_duckdb()
    run_sqlite()
'''
