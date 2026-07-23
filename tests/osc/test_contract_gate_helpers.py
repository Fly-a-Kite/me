from __future__ import annotations

from datadiff_osc.contract_engine.gates import (
    build_contract_gate_report,
    contract_performance_gate,
    false_positive_precision_gate,
    root_recall_gate,
    staged_exact_parity_gate,
)


def test_root_gate_cannot_be_relaxed_below_nine():
    result = root_recall_gate(
        ("root-1",),
        ("root-1",),
        required_roots=1,
        evidence_digests=("raw-root-evidence",),
    )
    assert result.required_count == 9
    assert not result.passed


def test_parity_gate_rejects_small_sample_even_if_threshold_argument_is_lowered():
    result = staged_exact_parity_gate(
        (("group-1", "SATISFIED", "SATISFIED"),),
        minimum_groups=1,
        evidence_digests=("raw-parity-evidence",),
    )
    assert result.required_count == 100_000
    assert not result.passed
    assert "parity_sample_below_100k_gate" in result.reasons


def test_full_100k_zero_discrepancy_parity_input_passes():
    records = (
        (f"group-{index}", "SATISFIED", "SATISFIED")
        for index in range(100_000)
    )
    result = staged_exact_parity_gate(
        records, evidence_digests=("raw-parity-evidence",)
    )
    assert result.passed
    assert dict(result.metrics)["mismatch_count"] == 0


def test_false_positive_gate_forbids_global_axis_suppression():
    result = false_positive_precision_gate(
        ("fp-1",),
        ("fp-1",),
        globally_ignored_axes=("order",),
        evidence_digests=("raw-fp-evidence",),
    )
    assert not result.passed
    assert "global_axis_suppression_forbidden" in result.reasons


def test_performance_gate_needs_raw_evidence_and_minimum_sample_count():
    insufficient = contract_performance_gate(
        [0.1] * 10,
        baseline_throughput=100,
        current_throughput=100,
        evidence_digests=("raw-perf-evidence",),
        minimum_samples=1,
    )
    assert insufficient.required_count == 1_000
    assert not insufficient.passed
    passing = contract_performance_gate(
        [0.1] * 1_000,
        baseline_throughput=100,
        current_throughput=90,
        evidence_digests=("raw-perf-evidence",),
    )
    assert passing.passed


def test_gate_report_never_claims_24h_authority():
    result = root_recall_gate(
        tuple(f"root-{index}" for index in range(9)),
        tuple(f"root-{index}" for index in range(9)),
        evidence_digests=("raw-root-evidence",),
    )
    report = build_contract_gate_report((result,), raw_phase6_evidence=True)
    assert report.passed
    assert report.authorization_state == "phase6_contract_gates_passed_pending_root_authority"
    no_raw = build_contract_gate_report((result,), raw_phase6_evidence=False)
    assert no_raw.authorization_state == "not_authorized_for_24h"

