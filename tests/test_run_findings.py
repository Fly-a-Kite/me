from datadiff import runner as runner_module
from datadiff.adjudication import build_adjudication
from datadiff.config import ExperimentConfig
from datadiff.oracle import Finding
from datadiff.run_findings import (
    _artifact_budget_available,
    _countable_finding_objects,
    _countable_row_findings,
    _finding_recheck_key,
    _format_recheck_key,
    _is_countable_finding_dict,
    _mark_finding_non_reproducible,
)


def _finding(**overrides):
    payload = {
        "finding_id": "finding-1",
        "kind": "semantic_output_mismatch",
        "severity": "critical",
        "suspicious_backends": ["right"],
        "evidence": "mismatch",
        "signature": "sig-1",
        "root_cause": "filter_predicate",
        "mismatch_class": "row_count",
    }
    payload.update(overrides)
    return Finding(**payload)


def test_recheck_key_canonicalizes_backend_order_and_formats_stably():
    key = _finding_recheck_key(
        {
            "kind": "semantic_output_mismatch",
            "root_cause": "filter_predicate",
            "suspicious_backends": ["right", "left"],
            "mismatch_class": "row_count",
        }
    )

    assert key == (
        "semantic_output_mismatch",
        "filter_predicate",
        ("left", "right"),
        "row_count",
    )
    assert _format_recheck_key(key) == "semantic_output_mismatch:filter_predicate@left,right:row_count"


def test_format_recheck_key_handles_empty_backend_and_mismatch():
    key = _finding_recheck_key({"kind": "schema_mismatch", "root_cause": "projection"})

    assert key == ("schema_mismatch", "projection", (), "")
    assert _format_recheck_key(key) == "schema_mismatch:projection@unknown"


def test_mark_finding_non_reproducible_sets_exclusion_adjudication():
    finding = _finding()

    _mark_finding_non_reproducible(finding, attempts=2)

    assert finding.triage_verdict == "non_reproducible_candidate"
    assert finding.paper_status == "exclude_unreproducible_candidate"
    assert finding.triage_confidence == "high"
    assert finding.false_positive is True
    assert finding.false_positive_reason == "candidate_not_reproduced_on_immediate_recheck"
    assert "2 immediate fresh recheck run(s)" in finding.triage_evidence
    assert finding.adjudication["validity_gate"] == "recheck_failed"
    assert finding.adjudication["attribution_gate"] == "reproduction_failed"
    assert finding.adjudication["recheck_status"] == "failed"
    assert finding.adjudication["countable_as_bug_evidence"] is False
    assert finding.adjudication["needs_manual_review"] is False


def test_countable_finding_filters_follow_adjudication_policy():
    countable = {
        "finding_id": "countable",
        "triage_verdict": "candidate_implementation_bug",
        "adjudication": build_adjudication("candidate_implementation_bug"),
    }
    false_positive = {
        "finding_id": "false-positive",
        "triage_verdict": "candidate_implementation_bug",
        "false_positive": True,
        "adjudication": build_adjudication("candidate_implementation_bug"),
    }
    documented_boundary = {
        "finding_id": "documented",
        "triage_verdict": "documented_semantic_divergence",
        "adjudication": build_adjudication("documented_semantic_divergence"),
    }

    assert _is_countable_finding_dict(countable) is True
    assert _is_countable_finding_dict(false_positive) is False
    assert _is_countable_finding_dict(documented_boundary) is False
    assert _countable_row_findings({"findings": [countable, false_positive, documented_boundary]}) == [countable]


def test_countable_finding_objects_use_same_policy_as_rows():
    countable = _finding(
        finding_id="countable",
        triage_verdict="candidate_implementation_bug",
        adjudication=build_adjudication("candidate_implementation_bug"),
    )
    false_positive = _finding(
        finding_id="false-positive",
        triage_verdict="candidate_implementation_bug",
        false_positive=True,
        adjudication=build_adjudication("candidate_implementation_bug"),
    )

    assert _countable_finding_objects([countable, false_positive]) == [countable]


def test_artifact_budget_available_respects_disabled_unlimited_and_limit():
    assert _artifact_budget_available(ExperimentConfig(enable_artifact=False), saved_count=0) is False
    assert _artifact_budget_available(
        ExperimentConfig(enable_artifact=True, artifact_limit=None),
        saved_count=100,
    ) is True
    assert _artifact_budget_available(
        ExperimentConfig(enable_artifact=True, artifact_limit=2),
        saved_count=1,
    ) is True
    assert _artifact_budget_available(
        ExperimentConfig(enable_artifact=True, artifact_limit=2),
        saved_count=2,
    ) is False
    assert _artifact_budget_available(
        ExperimentConfig(enable_artifact=True, artifact_limit=-1),
        saved_count=0,
    ) is False


def test_runner_reexports_run_findings_helpers_for_compatibility():
    assert runner_module._finding_recheck_key is _finding_recheck_key
    assert runner_module._format_recheck_key is _format_recheck_key
    assert runner_module._mark_finding_non_reproducible is _mark_finding_non_reproducible
    assert runner_module._countable_row_findings is _countable_row_findings
    assert runner_module._countable_finding_objects is _countable_finding_objects
    assert runner_module._is_countable_finding_dict is _is_countable_finding_dict
    assert runner_module._artifact_budget_available is _artifact_budget_available
