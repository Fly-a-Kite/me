import json

from datadiff.bug_audit import (
    BugAuditRun,
    available_bug_audit_probe_ids,
    candidate_families_from_results,
    list_audit_probe_ids,
    write_probe_audit_outputs,
    write_probe_issue_drafts,
    write_bug_audit_issue_drafts,
    write_bug_audit_outputs,
)


def test_bug_audit_registry_contains_latest_fresh_probes():
    assert set(available_bug_audit_probe_ids()) == {
        "datafusion_distinct_null_topk",
        "datafusion_limit_idempotence",
        "duckdb_cte_inline_equivalence",
        "duckdb_left_anti_join_equivalence",
        "pandas_arrow_groupby_size_count",
        "polars_concat_select_pushdown",
        "polars_lazy_eager_equivalence",
        "polars_reflected_arithmetic",
        "polars_slice_chunk_lazy_equivalence",
        "polars_vector_division_rounding",
        "pyarrow_sliced_bool_groupby",
        "pyarrow_sliced_transform_equivalence",
    }
    assert list_audit_probe_ids() == available_bug_audit_probe_ids()


def test_candidate_families_from_results_deduplicates_by_backend_family():
    results = [
        {"candidate_bug": True, "family": "family_a", "target_backend": "polars"},
        {"candidate_bug": True, "family": "family_a", "target_backend": "polars"},
        {"candidate_bug": False, "family": "family_b", "target_backend": "pyarrow"},
        {"candidate_bug": True, "family": "family_c", "target_backend": "pyarrow"},
    ]

    assert candidate_families_from_results(results) == [
        "family_a@polars",
        "family_c@pyarrow",
    ]


def test_write_bug_audit_outputs_uses_reports_style_names(tmp_path):
    run = BugAuditRun(
        generated_at="2026-05-26T12:00:00Z",
        evidence_mode="deterministic_probe",
        methodology="test",
        environment={},
        results=[],
    )

    json_path, md_path = write_bug_audit_outputs(run, output_dir=tmp_path)

    assert json_path.name == "bug-audit-20260526T120000.json"
    assert md_path.name == "bug-audit-20260526T120000.md"
    assert write_probe_audit_outputs(run, output_dir=tmp_path) == (json_path, md_path)


def test_write_bug_audit_issue_drafts_writes_candidate_only(tmp_path):
    run = BugAuditRun(
        generated_at="2026-05-26T12:00:00Z",
        evidence_mode="deterministic_probe",
        methodology="test",
        environment={"polars": "1.41.0"},
        results=[
            {
                "probe_id": "polars_reflected_arithmetic",
                "target_backend": "polars",
                "family": "polars_reflected_arithmetic_operand_order",
                "title": "Polars reflected arithmetic",
                "invariant": "lhs.__rop__(rhs) must evaluate rhs op lhs",
                "status": "candidate_implementation_bug",
                "candidate_bug": True,
                "version": "1.41.0",
                "expected": {"x": [1]},
                "observed": {"x": [2]},
                "evidence": "mismatch",
            },
            {
                "probe_id": "clean_probe",
                "target_backend": "pyarrow",
                "family": "clean_family",
                "title": "Clean probe",
                "invariant": "clean",
                "status": "no_bug_detected",
                "candidate_bug": False,
            },
        ],
    )

    paths = write_bug_audit_issue_drafts(run, issue_dir=tmp_path)

    assert [path.name for path in paths] == ["polars_reflected_arithmetic_operand_order.md"]
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "bug-audit-v1"
    assert manifest["generated_by"] == "datadiff bug-audit"
    assert manifest["reproduction_command"] == "datadiff bug-audit --write-issues --overwrite-issues"
    text = paths[0].read_text(encoding="utf-8")
    assert "datadiff bug-audit --probes polars_reflected_arithmetic" in text
    assert "`polars_reflected_arithmetic_operand_order@polars`" in text
    assert write_probe_issue_drafts(run, issue_dir=tmp_path) == paths
