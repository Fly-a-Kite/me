from datadiff.oracle import Finding
from datadiff.oracle_complex import cross_validate_oracle_findings


def _finding(
    finding_id: str,
    *,
    oracle: str,
    backend: str,
    root: str,
    kind: str = "semantic_output_mismatch",
) -> Finding:
    return Finding(
        finding_id=finding_id,
        kind=kind,
        severity="high",
        suspicious_backends=[backend],
        evidence="e",
        signature=finding_id,
        root_cause=root,
        oracle=oracle,
        confidence="medium",
    )


def test_cross_validation_marks_differential_finding_corroborated_by_metamorphic():
    differential = _finding("d1", oracle="differential", backend="duckdb", root="join_semantics")
    metamorphic = _finding(
        "m1",
        oracle="metamorphic",
        backend="duckdb",
        root="metamorphic_join_semantics",
        kind="metamorphic_join_semantics_violation",
    )

    summary = cross_validate_oracle_findings([differential], [metamorphic]).to_dict()

    assert summary["cross_validated_count"] == 1
    assert summary["metamorphic_only_count"] == 0
    assert summary["corroborated_backends"] == ["duckdb"]
    assert differential.confidence == "high"
    assert differential.adjudication["metamorphic_support"] == "corroborated"
    assert differential.adjudication["oracle_complex"]["cross_validated"] is True
    assert metamorphic.adjudication["metamorphic_support"] == "corroborates_differential"


def test_cross_validation_marks_metamorphic_only_and_differential_recheck():
    differential = _finding("d1", oracle="differential", backend="duckdb", root="groupby_aggregation")
    metamorphic = _finding(
        "m1",
        oracle="metamorphic",
        backend="polars",
        root="metamorphic_limit_idempotence",
        kind="metamorphic_limit_idempotence_violation",
    )

    summary = cross_validate_oracle_findings([differential], [metamorphic]).to_dict()

    assert summary["cross_validated_count"] == 0
    assert summary["differential_needs_recheck_count"] == 1
    assert summary["metamorphic_only_count"] == 1
    assert differential.adjudication["metamorphic_support"] == "not_observed"
    assert differential.adjudication["oracle_complex"]["needs_recheck"] is True
    assert metamorphic.discovery_origin == "metamorphic_only"
    assert metamorphic.adjudication["metamorphic_support"] == "metamorphic_only"
    assert metamorphic.adjudication["recheck_status"] == "needed"
