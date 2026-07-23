"""Pure gate calculations; this module never manufactures run evidence."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

from datadiff_osc._canonical import stable_digest, to_primitive
from datadiff_osc.contract_engine.mutations import MutationAuditResult


REQUIRED_CONFIRMED_ROOTS = 9
REQUIRED_PARITY_GROUPS = 100_000
MINIMUM_PERFORMANCE_SAMPLES = 1_000


@dataclass(frozen=True, slots=True)
class GateResult:
    gate_id: str
    passed: bool
    observed_count: int
    required_count: int
    reasons: tuple[str, ...]
    metrics: tuple[tuple[str, Any], ...] = ()
    evidence_digests: tuple[str, ...] = ()
    schema_version: str = "osc-contract-gate-result-v1"

    def __post_init__(self) -> None:
        if not self.gate_id or self.required_count < 1 or self.observed_count < 0:
            raise ValueError("gate identity and counts are invalid")
        if len(self.reasons) != len(set(self.reasons)):
            raise ValueError("gate reasons must be unique")
        if len(self.evidence_digests) != len(set(self.evidence_digests)) or any(
            not item for item in self.evidence_digests
        ):
            raise ValueError("gate evidence digests must be non-empty and unique")
        if self.passed and (
            self.reasons
            or self.observed_count < self.required_count
            or not self.evidence_digests
        ):
            raise ValueError("a passing gate requires complete counts and raw evidence")

    @property
    def digest(self) -> str:
        return stable_digest("osc-contract-gate-result", self)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


@dataclass(frozen=True, slots=True)
class ContractGateReport:
    results: tuple[GateResult, ...]
    raw_phase6_evidence: bool
    schema_version: str = "osc-contract-gate-report-v1"

    def __post_init__(self) -> None:
        if len({item.gate_id for item in self.results}) != len(self.results):
            raise ValueError("contract gate result IDs must be unique")

    @property
    def passed(self) -> bool:
        return (
            self.raw_phase6_evidence
            and bool(self.results)
            and all(item.passed for item in self.results)
        )

    @property
    def authorization_state(self) -> str:
        # This helper can reject authorization but cannot itself authorize 24h.
        return (
            "phase6_contract_gates_passed_pending_root_authority"
            if self.passed
            else "not_authorized_for_24h"
        )

    @property
    def digest(self) -> str:
        return stable_digest("osc-contract-gate-report", self)


def root_recall_gate(
    expected_root_ids: Iterable[str],
    recalled_root_ids: Iterable[str],
    *,
    evidence_digests: tuple[str, ...] = (),
    required_roots: int = REQUIRED_CONFIRMED_ROOTS,
) -> GateResult:
    required_roots = max(required_roots, REQUIRED_CONFIRMED_ROOTS)
    expected = tuple(expected_root_ids)
    recalled = tuple(recalled_root_ids)
    reasons: list[str] = []
    if len(expected) != len(set(expected)) or any(not item for item in expected):
        reasons.append("expected_root_ids_invalid")
    if len(recalled) != len(set(recalled)) or any(not item for item in recalled):
        reasons.append("recalled_root_ids_invalid")
    if len(expected) != required_roots:
        reasons.append("frozen_root_denominator_incomplete")
    missing = tuple(sorted(set(expected) - set(recalled)))
    unexpected = tuple(sorted(set(recalled) - set(expected)))
    if missing:
        reasons.append("confirmed_roots_missing")
    if unexpected:
        reasons.append("unregistered_roots_cannot_fill_recall")
    if not evidence_digests:
        reasons.append("raw_recall_evidence_missing")
    return GateResult(
        "confirmed_root_exact_recall",
        not reasons,
        len(set(expected) & set(recalled)),
        required_roots,
        tuple(reasons),
        metrics=(("missing", missing), ("unexpected", unexpected)),
        evidence_digests=evidence_digests,
    )


def false_positive_precision_gate(
    expected_fix_ids: Iterable[str],
    suppressed_fix_ids: Iterable[str],
    *,
    globally_ignored_axes: tuple[str, ...] = (),
    evidence_digests: tuple[str, ...] = (),
) -> GateResult:
    expected = tuple(expected_fix_ids)
    suppressed = tuple(suppressed_fix_ids)
    reasons: list[str] = []
    if not expected or len(expected) != len(set(expected)) or any(not item for item in expected):
        reasons.append("false_positive_denominator_invalid")
    if len(suppressed) != len(set(suppressed)):
        reasons.append("duplicate_precision_fix_result")
    missing = tuple(sorted(set(expected) - set(suppressed)))
    unexpected = tuple(sorted(set(suppressed) - set(expected)))
    if missing:
        reasons.append("precision_fixes_missing")
    if unexpected:
        reasons.append("unknown_cases_cannot_fill_precision_gate")
    if globally_ignored_axes:
        reasons.append("global_axis_suppression_forbidden")
    if not evidence_digests:
        reasons.append("raw_precision_evidence_missing")
    return GateResult(
        "false_positive_precision_fixes",
        not reasons,
        len(set(expected) & set(suppressed)),
        len(expected),
        tuple(reasons),
        metrics=(
            ("missing", missing),
            ("unexpected", unexpected),
            ("globally_ignored_axes", globally_ignored_axes),
        ),
        evidence_digests=evidence_digests,
    )


def fault_sensitivity_gate(
    audits: tuple[MutationAuditResult, ...],
    *,
    evidence_digests: tuple[str, ...] = (),
) -> GateResult:
    mutant_ids = tuple(item for audit in audits for item in audit.mutant_ids)
    killed_ids = tuple(item for audit in audits for item in audit.killed_ids)
    survived_ids = tuple(item for audit in audits for item in audit.survived_ids)
    reasons: list[str] = []
    if not audits or not mutant_ids:
        reasons.append("fault_audit_evidence_empty")
    if len(mutant_ids) != len(set(mutant_ids)):
        reasons.append("fault_mutant_ids_overlap")
    if survived_ids:
        reasons.append("high_risk_mutants_survived")
    if set(killed_ids) != set(mutant_ids):
        reasons.append("fault_kill_evidence_incomplete")
    if not evidence_digests:
        reasons.append("raw_fault_evidence_missing")
    return GateResult(
        "high_risk_fault_sensitivity",
        not reasons,
        len(set(killed_ids)),
        len(set(mutant_ids)),
        tuple(reasons),
        metrics=(("survived_ids", tuple(sorted(survived_ids))),),
        evidence_digests=evidence_digests,
    )


def staged_exact_parity_gate(
    comparisons: Iterable[tuple[str, str, str]],
    *,
    evidence_digests: tuple[str, ...] = (),
    minimum_groups: int = REQUIRED_PARITY_GROUPS,
) -> GateResult:
    """Consume ``(group_id, staged_verdict, exact_verdict)`` records."""

    minimum_groups = max(minimum_groups, REQUIRED_PARITY_GROUPS)
    records = tuple(comparisons)
    malformed = tuple(index for index, item in enumerate(records) if len(item) != 3)
    valid_records = tuple(item for item in records if len(item) == 3)
    group_ids = tuple(item[0] for item in valid_records)
    mismatches = tuple(item[0] for item in valid_records if item[1] != item[2])
    reasons: list[str] = []
    if len(records) < minimum_groups:
        reasons.append("parity_sample_below_100k_gate")
    if malformed:
        reasons.append("parity_record_malformed")
    if len(group_ids) != len(set(group_ids)) or any(not item for item in group_ids):
        reasons.append("parity_group_ids_invalid")
    if mismatches:
        reasons.append("staged_exact_verdict_discrepancy")
    if not evidence_digests:
        reasons.append("raw_parity_evidence_missing")
    return GateResult(
        "staged_exact_100k_parity",
        not reasons,
        len(records),
        minimum_groups,
        tuple(reasons),
        metrics=(("mismatch_count", len(mismatches)), ("mismatch_ids", mismatches[:32])),
        evidence_digests=evidence_digests,
    )


def contract_performance_gate(
    compile_match_ms: Iterable[float],
    *,
    baseline_throughput: float,
    current_throughput: float,
    case_wall_ms: Iterable[float] = (),
    evidence_digests: tuple[str, ...] = (),
    minimum_samples: int = MINIMUM_PERFORMANCE_SAMPLES,
) -> GateResult:
    minimum_samples = max(minimum_samples, MINIMUM_PERFORMANCE_SAMPLES)
    samples = tuple(float(item) for item in compile_match_ms)
    wall_samples = tuple(float(item) for item in case_wall_ms)
    reasons: list[str] = []
    if len(samples) < minimum_samples:
        reasons.append("performance_sample_too_small")
    if any(not math.isfinite(item) or item < 0 for item in samples):
        reasons.append("compile_match_sample_invalid")
    if wall_samples and (
        len(wall_samples) != len(samples)
        or any(not math.isfinite(item) or item <= 0 for item in wall_samples)
    ):
        reasons.append("case_wall_sample_invalid")
    p95_ms = _nearest_rank(samples, 0.95) if samples else math.inf
    wall_p95_ms = _nearest_rank(wall_samples, 0.95) if wall_samples else math.inf
    latency_pass = p95_ms <= 2.0 or (
        bool(wall_samples) and p95_ms <= wall_p95_ms * 0.05
    )
    if not latency_pass:
        reasons.append("compile_match_p95_exceeds_gate")
    throughput_ratio = (
        current_throughput / baseline_throughput
        if baseline_throughput > 0
        and math.isfinite(baseline_throughput)
        and math.isfinite(current_throughput)
        else -math.inf
    )
    if throughput_ratio < 0.9:
        reasons.append("paired_throughput_regression_exceeds_10_percent")
    if not evidence_digests:
        reasons.append("raw_performance_evidence_missing")
    return GateResult(
        "contract_compile_match_performance",
        not reasons,
        len(samples),
        minimum_samples,
        tuple(reasons),
        metrics=(
            ("p95_ms", p95_ms),
            ("case_wall_p95_ms", wall_p95_ms),
            ("throughput_ratio", throughput_ratio),
        ),
        evidence_digests=evidence_digests,
    )


def build_contract_gate_report(
    results: tuple[GateResult, ...],
    *,
    raw_phase6_evidence: bool,
) -> ContractGateReport:
    if len({item.gate_id for item in results}) != len(results):
        raise ValueError("contract gate result IDs must be unique")
    return ContractGateReport(results, raw_phase6_evidence)


def _nearest_rank(values: tuple[float, ...], quantile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]
