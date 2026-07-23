from __future__ import annotations

from collections import Counter
from typing import Any

from datadiff.canonicalization import short_canonical_hash
from datadiff.operation_combo import describe_operation_combo
from datadiff.operation_semantics import operation_names
from datadiff.finding_outcomes import offline_finding_bucket


def behavior_signature(row: dict[str, Any]) -> str:
    payload = {
        "case_ops": row["case"]["program"]["operations"],
        "backend_status": {
            backend: result["status"]
            for backend, result in sorted(row["normalized"].items())
        },
        "normalized_shape": {
            backend: {
                "columns": result.get("columns", []),
                "rows": len(result.get("rows", [])),
                "sample": result.get("rows", [])[:3],
                "error_type": result.get("error_type", ""),
            }
            for backend, result in sorted(row["normalized"].items())
        },
        "finding_kinds": sorted(finding["kind"] for finding in row.get("findings", [])),
        "finding_roots": sorted(
            finding.get("root_cause", "unknown") for finding in row.get("findings", [])
        ),
    }
    return short_canonical_hash(payload, 16)


def row_count_bucket(row_count: int) -> str:
    if row_count <= 0:
        return "0"
    if row_count == 1:
        return "1"
    if row_count <= 3:
        return "2-3"
    if row_count <= 7:
        return "4-7"
    if row_count <= 15:
        return "8-15"
    if row_count <= 31:
        return "16-31"
    return "32+"


def discovery_signature(row: dict[str, Any]) -> str:
    operations = operation_names(row["case"]["program"]["operations"], default="unknown")
    op_histogram = sorted(Counter(operations).items())
    combo = describe_operation_combo(row["case"]["program"]["operations"])
    findings = row.get("findings") or []
    payload = {
        "combo_template": combo.get("template", ""),
        "operation_histogram": op_histogram,
        "backend_outcomes": {
            backend: {
                "status": result.get("status", "unknown"),
                "error_type": result.get("error_type", ""),
                "column_count": len(result.get("columns", [])),
                "row_count_bucket": row_count_bucket(len(result.get("rows", []))),
            }
            for backend, result in sorted((row.get("normalized") or {}).items())
        },
        "finding_kinds": sorted(str(finding.get("kind", "")) for finding in findings),
        "finding_roots": sorted(
            str(finding.get("root_cause", "unknown"))
            for finding in findings
            if not bool(finding.get("false_positive"))
        ),
    }
    return short_canonical_hash(payload, 16)


def coverage_discovery_signature(row: dict[str, Any]) -> str:
    """Compare sampled and full sweeps without treating backend-set rotation as novelty."""
    operations = operation_names(row["case"]["program"]["operations"], default="unknown")
    combo = describe_operation_combo(row["case"]["program"]["operations"])
    findings = row.get("findings") or []
    outcome_classes = {
        (
            str(result.get("status", "unknown")),
            str(result.get("error_type", "")),
            len(result.get("columns", [])),
            row_count_bucket(len(result.get("rows", []))),
        )
        for result in (row.get("normalized") or {}).values()
    }
    payload = {
        "combo_template": combo.get("template", ""),
        "operation_histogram": sorted(Counter(operations).items()),
        "outcome_classes": sorted(outcome_classes),
        "finding_kinds": sorted(str(finding.get("kind", "")) for finding in findings),
        "finding_roots": sorted(
            str(finding.get("root_cause", "unknown"))
            for finding in findings
            if not bool(finding.get("false_positive"))
        ),
        "suspicious_backends": sorted(
            {
                str(backend)
                for finding in findings
                if not bool(finding.get("false_positive"))
                for backend in finding.get("suspicious_backends", []) or []
            }
        ),
    }
    return short_canonical_hash(payload, 16)


def signal_signature(row: dict[str, Any]) -> str:
    operations = sorted(set(operation_names(row["case"]["program"]["operations"], default="unknown")))
    combo = describe_operation_combo(row["case"]["program"]["operations"])
    findings = row.get("findings") or []
    payload = {
        "combo_template": combo.get("template", ""),
        "operation_set": operations,
        "backend_outcomes": {
            backend: {
                "status": result.get("status", "unknown"),
                "error_type": result.get("error_type", ""),
            }
            for backend, result in sorted((row.get("normalized") or {}).items())
        },
        "finding_buckets": sorted(
            {
                offline_finding_bucket(finding)
                for finding in findings
                if not bool(finding.get("false_positive"))
            }
        ),
        "suspicious_backends": sorted(
            {
                str(backend)
                for finding in findings
                if not bool(finding.get("false_positive"))
                for backend in finding.get("suspicious_backends", []) or []
            }
        ),
    }
    return short_canonical_hash(payload, 16)


def coverage_signal_signature(row: dict[str, Any]) -> str:
    operations = sorted(set(operation_names(row["case"]["program"]["operations"], default="unknown")))
    combo = describe_operation_combo(row["case"]["program"]["operations"])
    findings = row.get("findings") or []
    payload = {
        "combo_template": combo.get("template", ""),
        "operation_set": operations,
        "backend_outcome_classes": sorted(
            {
                (
                    str(result.get("status", "unknown")),
                    str(result.get("error_type", "")),
                )
                for result in (row.get("normalized") or {}).values()
            }
        ),
        "finding_buckets": sorted(
            {
                offline_finding_bucket(finding)
                for finding in findings
                if not bool(finding.get("false_positive"))
            }
        ),
        "suspicious_backends": sorted(
            {
                str(backend)
                for finding in findings
                if not bool(finding.get("false_positive"))
                for backend in finding.get("suspicious_backends", []) or []
            }
        ),
    }
    return short_canonical_hash(payload, 16)


_row_count_bucket = row_count_bucket
