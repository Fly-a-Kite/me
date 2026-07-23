"""Fail-closed machine-readable pre-24h gate calculations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from numbers import Real
from typing import Any, Iterable, Mapping

from datadiff_osc._canonical import stable_digest, to_primitive
from datadiff_osc.runtime.gate_artifacts import (
    MetricReducer,
    VerifiedGateAuthorityPlan,
    VerifiedGateEvidence,
    recompute_gate_metrics,
)


class GateOperator(str, Enum):
    EQ = "eq"
    GE = "ge"
    LE = "le"
    ANY_LE = "any_le"


@dataclass(frozen=True, slots=True)
class GateSpec:
    gate_id: str
    numerator_key: str
    denominator_key: str
    operator: GateOperator
    threshold: float
    normalize: bool = True
    minimum_denominator: float = 1.0
    allow_zero_denominator: bool = False
    required_denominator: float | None = None
    alternative_numerator_key: str = ""
    alternative_threshold: float | None = None

    def __post_init__(self) -> None:
        if not self.gate_id or not self.numerator_key or not self.denominator_key:
            raise ValueError("gate specification fields must be non-empty")
        if self.minimum_denominator < 0:
            raise ValueError("minimum gate denominator must be non-negative")
        if self.required_denominator is not None and self.required_denominator < 0:
            raise ValueError("required gate denominator must be non-negative")
        if self.operator is GateOperator.ANY_LE:
            if not self.alternative_numerator_key or self.alternative_threshold is None:
                raise ValueError("alternative gate requires its metric and threshold")
        elif self.alternative_numerator_key or self.alternative_threshold is not None:
            raise ValueError("only an alternative gate may define an alternative metric")


@dataclass(frozen=True, slots=True)
class GateResult:
    gate_id: str
    numerator: int | float | None
    denominator: int | float | None
    threshold: float
    observed_value: float | None
    alternative_numerator: int | float | None
    alternative_threshold: float | None
    operator: GateOperator
    passed: bool
    evidence: tuple[str, ...]
    reason: str
    schema_version: str = "osc-runtime-gate-result-v1"

    @property
    def digest(self) -> str:
        return stable_digest("osc-runtime-gate-result", self)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


@dataclass(frozen=True, slots=True)
class GateReport:
    source_digest: str
    expected_authority_plan_sha256: str
    authority_plan_sha256: str
    authority_plan_digest: str
    artifact_manifest_sha256: str
    verified_artifact_digests: tuple[str, ...]
    frozen_spec_set_digest: str
    complete_frozen_spec_set: bool
    verification_errors: tuple[str, ...]
    results: tuple[GateResult, ...]
    all_gates_pass: bool
    authority_eligible: bool
    twenty_four_hour_run_authorized: bool = False
    schema_version: str = "osc-pre24-gate-report-v2"

    def __post_init__(self) -> None:
        if not self.source_digest:
            raise ValueError("gate report requires a source digest")
        for digest in (
            self.expected_authority_plan_sha256,
            self.authority_plan_sha256,
            self.artifact_manifest_sha256,
        ):
            if len(digest) != 64:
                raise ValueError("gate report requires SHA-256 bindings")
        if not self.authority_plan_digest or not self.frozen_spec_set_digest:
            raise ValueError("gate report requires authority and spec-set bindings")
        if len(self.verified_artifact_digests) != len(
            set(self.verified_artifact_digests)
        ):
            raise ValueError("verified gate artifact digests must be unique")
        if not self.results:
            raise ValueError("gate report requires gate results")
        expected_ids = tuple(item.gate_id for item in default_pre24_gate_specs())
        if tuple(item.gate_id for item in self.results) != expected_ids:
            raise ValueError("authoritative gate report requires the complete frozen gate set")
        if not self.complete_frozen_spec_set:
            raise ValueError("authoritative gate report cannot be a subset evaluation")
        if self.frozen_spec_set_digest != pre24_gate_spec_set_digest():
            raise ValueError("authoritative gate report spec-set digest mismatch")
        if self.all_gates_pass != all(item.passed for item in self.results):
            raise ValueError("gate report aggregate does not match gate results")
        if self.all_gates_pass and (
            self.verification_errors or not self.verified_artifact_digests
        ):
            raise ValueError("a passing gate report requires verified raw artifacts")
        expected_eligibility = bool(
            self.all_gates_pass
            and not self.verification_errors
            and self.verified_artifact_digests
            and self.complete_frozen_spec_set
        )
        if self.authority_eligible != expected_eligibility:
            raise ValueError("gate report authority eligibility is inconsistent")
        if self.twenty_four_hour_run_authorized:
            raise ValueError("runtime gate reports cannot authorize a 24h run")

    @property
    def digest(self) -> str:
        return stable_digest("osc-pre24-gate-report", self)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


@dataclass(frozen=True, slots=True)
class GateEvaluationReport:
    source_digest: str
    expected_authority_plan_sha256: str
    authority_plan_sha256: str
    authority_plan_digest: str
    artifact_manifest_sha256: str
    evaluated_spec_set_digest: str
    verification_errors: tuple[str, ...]
    results: tuple[GateResult, ...]
    all_evaluated_gates_pass: bool
    evaluation_only: bool = True
    authority_eligible: bool = False
    twenty_four_hour_run_authorized: bool = False
    schema_version: str = "osc-gate-evaluation-report-v1"

    def __post_init__(self) -> None:
        if not self.source_digest or not self.authority_plan_digest:
            raise ValueError("gate evaluation requires source and authority bindings")
        for digest in (
            self.expected_authority_plan_sha256,
            self.authority_plan_sha256,
            self.artifact_manifest_sha256,
        ):
            if len(digest) != 64:
                raise ValueError("gate evaluation requires SHA-256 bindings")
        if not self.evaluated_spec_set_digest or not self.results:
            raise ValueError("gate evaluation requires specs and results")
        gate_ids = tuple(item.gate_id for item in self.results)
        if len(gate_ids) != len(set(gate_ids)):
            raise ValueError("gate evaluation result IDs must be unique")
        if self.all_evaluated_gates_pass != all(item.passed for item in self.results):
            raise ValueError("gate evaluation aggregate does not match results")
        if not self.evaluation_only or self.authority_eligible:
            raise ValueError("custom gate evaluation can never become authority")
        if self.twenty_four_hour_run_authorized:
            raise ValueError("gate evaluations cannot authorize a 24h run")

    @property
    def digest(self) -> str:
        return stable_digest("osc-gate-evaluation-report", self)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


def _ratio(
    gate_id: str,
    numerator: str,
    denominator: str,
    *,
    minimum: int,
    exact_denominator: bool = False,
) -> GateSpec:
    return GateSpec(
        gate_id,
        numerator,
        denominator,
        GateOperator.EQ,
        1.0,
        minimum_denominator=float(minimum),
        required_denominator=(float(minimum) if exact_denominator else None),
    )


def default_pre24_gate_specs() -> tuple[GateSpec, ...]:
    """Frozen-threshold calculator inputs; declarations remain external evidence."""

    return (
        _ratio("contract.confirmed_root_recall", "confirmed_roots_recalled", "confirmed_roots_total", minimum=9, exact_denominator=True),
        _ratio("contract.false_positive_precision", "false_positive_fixes_verified", "false_positive_fixes_total", minimum=1),
        _ratio("contract.mutants_killed", "high_risk_mutants_killed", "high_risk_mutants_total", minimum=1),
        GateSpec("contract.staged_exact_discrepancies", "staged_exact_discrepancies", "staged_exact_groups", GateOperator.EQ, 0.0, normalize=False, minimum_denominator=100_000),
        GateSpec("contract.compile_match_budget", "contract_compile_match_p95_ms", "unit_denominator", GateOperator.ANY_LE, 2.0, normalize=False, alternative_numerator_key="contract_compile_match_wall_share", alternative_threshold=0.05),
        GateSpec("contract.throughput_regression", "paired_throughput_regression", "unit_denominator", GateOperator.LE, 0.10, normalize=False),
        _ratio("coverage.fresh_families", "fresh_families_observed", "fresh_families_declared", minimum=16, exact_denominator=True),
        _ratio("coverage.fresh_cells", "fresh_cells_observed", "fresh_cells_declared", minimum=232, exact_denominator=True),
        _ratio("coverage.regression_families", "regression_families_observed", "regression_families_declared", minimum=9, exact_denominator=True),
        _ratio("coverage.regression_cells", "regression_cells_observed", "regression_cells_declared", minimum=144, exact_denominator=True),
        _ratio("coverage.contrast_edges", "contrast_edges_observed", "contrast_edges_declared", minimum=384, exact_denominator=True),
        _ratio("coverage.backend_pair_obligations", "backend_pair_obligations_observed_or_unsupported", "backend_pair_obligations_declared", minimum=502, exact_denominator=True),
        _ratio("reachability.cells_constructed_activated", "cells_constructed_activated", "cells_activation_denominator", minimum=232, exact_denominator=True),
        _ratio("reachability.edges_constructed_activated", "edges_constructed_activated", "edges_activation_denominator", minimum=384, exact_denominator=True),
        GateSpec("reachability.scheduled_activation", "scheduled_targets_activated", "scheduled_targets_total", GateOperator.GE, 0.95, minimum_denominator=1),
        GateSpec("reachability.mutation_preservation", "mutations_preserved", "mutations_attempted", GateOperator.GE, 0.90, minimum_denominator=1),
        _ratio("reachability.fresh_cells_two_activations", "fresh_cells_with_two_activations", "fresh_cells_reachability_denominator", minimum=232, exact_denominator=True),
        _ratio("reachability.fresh_families_two_seed_blocks", "fresh_families_with_two_observed_seed_blocks", "fresh_family_reachability_denominator", minimum=16, exact_denominator=True),
        _ratio("reachability.edges_observed", "edges_fully_observed", "edge_observation_denominator", minimum=384, exact_denominator=True),
        _ratio("reachability.scheduled_activation_worst_family", "families_meeting_scheduled_activation_90", "scheduled_activation_family_denominator", minimum=16, exact_denominator=True),
        _ratio("reachability.mutation_worst_family", "families_meeting_mutation_preservation_80", "mutation_family_denominator", minimum=16, exact_denominator=True),
        _ratio("focus.signals", "focus_signals_meeting_minimum", "focus_signals_total", minimum=1),
        _ratio("focus.lanes", "lanes_meeting_hit_rate", "formal_lanes", minimum=11, exact_denominator=True),
        _ratio("dimensions.operations", "operations_observed", "operations_declared", minimum=21, exact_denominator=True),
        _ratio("dimensions.expressions", "expressions_observed", "expressions_declared", minimum=21, exact_denominator=True),
        _ratio("dimensions.aggregates", "aggregates_observed", "aggregates_declared", minimum=8, exact_denominator=True),
        _ratio("dimensions.risk_classes", "risk_classes_observed", "risk_classes_declared", minimum=7, exact_denominator=True),
        _ratio("dimensions.risk_pipelines", "risk_pipelines_observed", "risk_pipelines_declared", minimum=18, exact_denominator=True),
        _ratio("parallel.worker_invariance", "parallel_invariance_checks_passed", "parallel_invariance_checks_total", minimum=1),
        _ratio("parallel.exact_escalation_faults", "exact_escalation_faults_detected", "exact_escalation_faults_injected", minimum=1),
        GateSpec("parallel.six_worker_efficiency", "six_worker_parallel_efficiency", "unit_denominator", GateOperator.GE, 0.70, normalize=False),
        _ratio("v3.completed_runs", "v3_runs_completed", "v3_runs_planned", minimum=22, exact_denominator=True),
        _ratio("v3.executed_cases", "v3_cases_executed", "v3_cases_planned", minimum=2200, exact_denominator=True),
        GateSpec("v3.iteration_failures", "v3_iteration_failures", "v3_cases_executed", GateOperator.EQ, 0.0, normalize=False, minimum_denominator=2200, required_denominator=2200),
        GateSpec("v3.pipeline_errors", "v3_pipeline_errors", "v3_cases_executed", GateOperator.EQ, 0.0, normalize=False, minimum_denominator=2200, required_denominator=2200),
        GateSpec("v3.non_ok_backend_results", "v3_non_ok_backend_results", "v3_backend_results", GateOperator.EQ, 0.0, normalize=False, minimum_denominator=1),
        GateSpec("v3.campaign_rechecks", "campaign_rechecks_completed", "campaign_rechecks_required", GateOperator.EQ, 1.0, allow_zero_denominator=True),
        GateSpec("v3.pipeline_rechecks", "pipeline_rechecks_completed", "pipeline_rechecks_required", GateOperator.EQ, 1.0, allow_zero_denominator=True),
        GateSpec("repository.test_failures", "repository_test_failures", "repository_tests_run", GateOperator.EQ, 0.0, normalize=False, minimum_denominator=1),
        GateSpec("environment.latest_target_mismatches", "latest_target_mismatches", "latest_target_packages", GateOperator.EQ, 0.0, normalize=False, minimum_denominator=1),
        GateSpec("classification.premature_bug_claims", "premature_bug_claims", "candidate_finding_records", GateOperator.EQ, 0.0, normalize=False, minimum_denominator=0, allow_zero_denominator=True),
    )


def pre24_gate_spec_set_digest() -> str:
    return stable_digest("osc-pre24-gate-spec-set", default_pre24_gate_specs())


def _reducers_for_specs(specs: tuple[GateSpec, ...]) -> dict[str, MetricReducer]:
    metric_keys = {
        key
        for spec in specs
        for key in (
            spec.numerator_key,
            spec.denominator_key,
            *((spec.alternative_numerator_key,) if spec.operator is GateOperator.ANY_LE else ()),
        )
    }
    scalar_reducers = {
        "contract_compile_match_p95_ms": MetricReducer.NEAREST_RANK_P95,
        "contract_compile_match_wall_share": MetricReducer.NEAREST_RANK_P95,
        "paired_throughput_regression": MetricReducer.MAXIMUM,
        "six_worker_parallel_efficiency": MetricReducer.MINIMUM,
    }
    return {
        key: scalar_reducers.get(key, MetricReducer.COUNT_UNIQUE)
        for key in metric_keys
    }


def _denominator_keys(specs: tuple[GateSpec, ...]) -> tuple[str, ...]:
    return tuple(sorted({spec.denominator_key for spec in specs}))


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        converted = float(value)
    except OverflowError:
        return None
    if converted != converted or converted in {float("inf"), float("-inf")}:
        return None
    return value


def _compare(value: float, operator: GateOperator, threshold: float) -> bool:
    if operator is GateOperator.EQ:
        return value == threshold
    if operator is GateOperator.GE:
        return value >= threshold
    if operator is GateOperator.LE:
        return value <= threshold
    raise ValueError("alternative gate comparison requires both observed values")


def _calculate_one(
    spec: GateSpec,
    metrics: Mapping[str, Any],
    evidence_by_metric: Mapping[str, Iterable[str]],
    record_ids: Mapping[str, tuple[str, ...]],
    reducers: Mapping[str, MetricReducer],
    verification_errors: tuple[str, ...],
) -> GateResult:
    numerator = _number(metrics.get(spec.numerator_key))
    denominator = _number(metrics.get(spec.denominator_key))
    metric_keys = tuple(
        dict.fromkeys(
            (
                spec.numerator_key,
                spec.denominator_key,
                *(
                    (spec.alternative_numerator_key,)
                    if spec.operator is GateOperator.ANY_LE
                    else ()
                ),
            )
        )
    )
    evidence_items = tuple(
        sorted(
            {
                item
                for key in metric_keys
                for item in evidence_by_metric.get(key, ())
            }
        )
    )
    alternative = (
        _number(metrics.get(spec.alternative_numerator_key))
        if spec.operator is GateOperator.ANY_LE
        else None
    )
    reason = ""
    observed: float | None = None
    if verification_errors:
        reason = "raw artifact verification failed: " + verification_errors[0]
    elif numerator is None or denominator is None or (
        spec.operator is GateOperator.ANY_LE and alternative is None
    ):
        reason = "missing or non-finite recomputed numerator/denominator"
    elif not evidence_items or any(not item for item in evidence_items):
        reason = "missing verified raw gate evidence"
    elif (
        reducers.get(spec.numerator_key) is MetricReducer.COUNT_UNIQUE
        and reducers.get(spec.denominator_key) is MetricReducer.COUNT_UNIQUE
        and not set(record_ids.get(spec.numerator_key, ()))
        <= set(record_ids.get(spec.denominator_key, ()))
    ):
        reason = "recomputed numerator identities are outside the denominator universe"
    elif float(denominator) == 0.0:
        if spec.allow_zero_denominator and float(numerator) == 0.0:
            observed = 1.0 if spec.normalize else 0.0
        else:
            reason = "zero denominator is not admissible"
    elif (
        spec.required_denominator is not None
        and float(denominator) != spec.required_denominator
    ):
        reason = (
            f"denominator {denominator} does not equal frozen denominator "
            f"{spec.required_denominator:g}"
        )
    elif float(denominator) < spec.minimum_denominator:
        reason = (
            f"denominator {denominator} is below required minimum "
            f"{spec.minimum_denominator:g}"
        )
    else:
        observed = (
            float(numerator) / float(denominator)
            if spec.normalize
            else float(numerator)
        )
    if observed is None or reason:
        passed = False
    elif spec.operator is GateOperator.ANY_LE:
        assert alternative is not None and spec.alternative_threshold is not None
        passed = bool(
            observed <= spec.threshold
            or float(alternative) <= spec.alternative_threshold
        )
    else:
        passed = _compare(observed, spec.operator, spec.threshold)
    if not reason:
        reason = "threshold satisfied" if passed else "threshold not satisfied"
    return GateResult(
        gate_id=spec.gate_id,
        numerator=numerator,
        denominator=denominator,
        threshold=spec.threshold,
        observed_value=observed,
        alternative_numerator=alternative,
        alternative_threshold=spec.alternative_threshold,
        operator=spec.operator,
        passed=passed,
        evidence=evidence_items,
        reason=reason,
    )


def calculate_gate_report(
    evidence: VerifiedGateEvidence,
    *,
    expected_source_digest: str,
    expected_authority_plan_sha256: str,
    authority_plan: VerifiedGateAuthorityPlan,
) -> GateReport:
    if not isinstance(evidence, VerifiedGateEvidence):
        raise TypeError(
            "gate reports require typed, hash-verified raw artifact evidence"
        )
    if not isinstance(authority_plan, VerifiedGateAuthorityPlan):
        raise TypeError("gate reports require a verified authority plan")
    selected = default_pre24_gate_specs()
    reducers = _reducers_for_specs(selected)
    recomputed = recompute_gate_metrics(
        evidence,
        reducers,
        expected_source_digest=expected_source_digest,
        expected_authority_plan_sha256=expected_authority_plan_sha256,
        authority_plan=authority_plan,
        required_denominator_keys=_denominator_keys(selected),
        required_gate_spec_set_digest=pre24_gate_spec_set_digest(),
        require_exact_universe_set=True,
    )
    metrics = recomputed.metric_map()
    evidence_by_metric = recomputed.evidence_map()
    record_ids = recomputed.record_id_map()
    results = tuple(
        _calculate_one(
            spec,
            metrics,
            evidence_by_metric,
            record_ids,
            reducers,
            recomputed.errors,
        )
        for spec in selected
    )
    all_gates_pass = all(item.passed for item in results)
    return GateReport(
        source_digest=expected_source_digest,
        expected_authority_plan_sha256=expected_authority_plan_sha256,
        authority_plan_sha256=authority_plan.byte_sha256,
        authority_plan_digest=authority_plan.digest,
        artifact_manifest_sha256=evidence.manifest_sha256,
        verified_artifact_digests=tuple(
            sorted(item.digest for item in evidence.artifacts)
        ),
        frozen_spec_set_digest=pre24_gate_spec_set_digest(),
        complete_frozen_spec_set=True,
        verification_errors=recomputed.errors,
        results=results,
        all_gates_pass=all_gates_pass,
        authority_eligible=bool(
            all_gates_pass
            and not recomputed.errors
            and evidence.artifacts
        ),
        twenty_four_hour_run_authorized=False,
    )


def evaluate_gate_specs(
    evidence: VerifiedGateEvidence,
    *,
    specs: Iterable[GateSpec],
    expected_source_digest: str,
    expected_authority_plan_sha256: str,
    authority_plan: VerifiedGateAuthorityPlan,
) -> GateEvaluationReport:
    """Evaluate custom specs without ever producing authority."""

    if not isinstance(evidence, VerifiedGateEvidence):
        raise TypeError("gate evaluation requires verified gate evidence")
    if not isinstance(authority_plan, VerifiedGateAuthorityPlan):
        raise TypeError("gate evaluation requires a verified authority plan")
    selected = tuple(specs)
    if not selected or len({item.gate_id for item in selected}) != len(selected):
        raise ValueError("gate specifications must be non-empty and uniquely identified")
    reducers = _reducers_for_specs(selected)
    recomputed = recompute_gate_metrics(
        evidence,
        reducers,
        expected_source_digest=expected_source_digest,
        expected_authority_plan_sha256=expected_authority_plan_sha256,
        authority_plan=authority_plan,
        required_denominator_keys=_denominator_keys(selected),
        require_exact_universe_set=False,
        allow_unselected_claims=True,
    )
    metrics = recomputed.metric_map()
    evidence_by_metric = recomputed.evidence_map()
    record_ids = recomputed.record_id_map()
    results = tuple(
        _calculate_one(
            spec,
            metrics,
            evidence_by_metric,
            record_ids,
            reducers,
            recomputed.errors,
        )
        for spec in selected
    )
    return GateEvaluationReport(
        source_digest=expected_source_digest,
        expected_authority_plan_sha256=expected_authority_plan_sha256,
        authority_plan_sha256=authority_plan.byte_sha256,
        authority_plan_digest=authority_plan.digest,
        artifact_manifest_sha256=evidence.manifest_sha256,
        evaluated_spec_set_digest=stable_digest(
            "osc-gate-evaluation-spec-set", selected
        ),
        verification_errors=recomputed.errors,
        results=results,
        all_evaluated_gates_pass=all(item.passed for item in results),
        evaluation_only=True,
        authority_eligible=False,
        twenty_four_hour_run_authorized=False,
    )
