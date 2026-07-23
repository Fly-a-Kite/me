from __future__ import annotations

import hashlib
import json

import pytest

from datadiff_osc.runtime.gate_artifacts import (
    GATE_ARTIFACT_MANIFEST_SCHEMA_VERSION,
    GATE_AUTHORITY_PLAN_SCHEMA_VERSION,
    RAW_GATE_ARTIFACT_SCHEMA_VERSION,
    MetricReducer,
    identity_universe_digest,
    metric_series_payload,
    verify_gate_artifact_manifest,
    verify_gate_authority_plan,
)
from datadiff_osc.runtime.gates import (
    GateOperator,
    GateSpec,
    calculate_gate_report,
    default_pre24_gate_specs,
    evaluate_gate_specs,
    pre24_gate_spec_set_digest,
)


SOURCE = "source-snapshot-1"
_SCALAR_REDUCERS = {
    "contract_compile_match_p95_ms": MetricReducer.NEAREST_RANK_P95,
    "contract_compile_match_wall_share": MetricReducer.NEAREST_RANK_P95,
    "paired_throughput_regression": MetricReducer.MAXIMUM,
    "six_worker_parallel_efficiency": MetricReducer.MINIMUM,
}


def _json_bytes(payload) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _default_denominator_keys():
    return {spec.denominator_key for spec in default_pre24_gate_specs()}


def _write_evidence(
    tmp_path,
    label,
    metrics,
    *,
    universe_keys=None,
    trusted_universes=None,
    raw_record_ids=None,
    source_digest=SOURCE,
    manifest_source_digest=None,
    artifact_source_digest=None,
    plan_spec_set_digest=None,
    authority_artifact_hash=None,
):
    directory = tmp_path / label
    directory.mkdir()
    series = []
    record_ids = {}
    raw_id_overrides = dict(raw_record_ids or {})
    for key, value in sorted(metrics.items()):
        reducer = _SCALAR_REDUCERS.get(key, MetricReducer.COUNT_UNIQUE)
        if reducer is MetricReducer.COUNT_UNIQUE:
            assert float(value).is_integer() and int(value) >= 0
            ids = tuple(
                raw_id_overrides.get(
                    key,
                    tuple(f"record-{index:06d}" for index in range(int(value))),
                )
            )
            assert len(ids) == int(value)
            records = tuple((record_id, 1) for record_id in ids)
            record_ids[key] = ids
        else:
            records = ((f"{key}:sample-000000", value),)
        series.append(metric_series_payload(key, reducer, records))
    raw = _json_bytes(
        {
            "schema_version": RAW_GATE_ARTIFACT_SCHEMA_VERSION,
            "artifact_id": "raw-gates",
            "source_digest": artifact_source_digest or source_digest,
            "series": series,
        }
    )
    artifact_path = directory / "raw.json"
    artifact_path.write_bytes(raw)
    artifact_hash = hashlib.sha256(raw).hexdigest()
    manifest_path = directory / "manifest.json"
    manifest_path.write_bytes(
        _json_bytes(
            {
                "schema_version": GATE_ARTIFACT_MANIFEST_SCHEMA_VERSION,
                "source_digest": manifest_source_digest or source_digest,
                "claimed_metrics": metrics,
                "artifacts": [
                    {
                        "artifact_id": "raw-gates",
                        "path": "raw.json",
                        "sha256": artifact_hash,
                        "schema_version": RAW_GATE_ARTIFACT_SCHEMA_VERSION,
                    }
                ],
            }
        )
    )

    if universe_keys is None:
        universe_keys = tuple(sorted(_default_denominator_keys() & set(metrics)))
    trusted = dict(trusted_universes or {})
    authority_payload = {
        "schema_version": GATE_AUTHORITY_PLAN_SCHEMA_VERSION,
        "source_digest": source_digest,
        "gate_spec_set_digest": plan_spec_set_digest or pre24_gate_spec_set_digest(),
        "artifacts": [
            {
                "artifact_id": "raw-gates",
                "sha256": authority_artifact_hash or artifact_hash,
                "schema_version": RAW_GATE_ARTIFACT_SCHEMA_VERSION,
                "provenance_digest": "root-authority-artifact-index-1",
            }
        ],
        "denominator_universes": [
            {
                "metric_key": key,
                "cardinality": len(trusted.get(key, record_ids[key])),
                "identity_digest": identity_universe_digest(
                    trusted.get(key, record_ids[key])
                ),
                "provenance_digest": f"root-compiled-universe-{key}",
            }
            for key in sorted(universe_keys)
        ],
    }
    authority_path = directory / "authority.json"
    authority_bytes = _json_bytes(authority_payload)
    authority_path.write_bytes(authority_bytes)
    authority = verify_gate_authority_plan(
        authority_path,
        expected_sha256=hashlib.sha256(authority_bytes).hexdigest(),
        expected_source_digest=source_digest,
    )
    evidence = verify_gate_artifact_manifest(
        manifest_path,
        expected_source_digest=source_digest,
        expected_authority_plan_sha256=authority.expected_plan_sha256,
        authority_plan=authority,
    )
    return authority, evidence, artifact_path


def _passing_metrics():
    specs = default_pre24_gate_specs()
    metrics = {}
    for spec in specs:
        if spec.allow_zero_denominator:
            metrics.setdefault(spec.denominator_key, 0)
        else:
            metrics[spec.denominator_key] = max(
                float(metrics.get(spec.denominator_key, 0)),
                spec.minimum_denominator,
                1.0,
            )
    for spec in specs:
        denominator = metrics[spec.denominator_key]
        if spec.allow_zero_denominator and denominator == 0:
            metrics[spec.numerator_key] = 0
        elif spec.normalize:
            metrics[spec.numerator_key] = denominator
        else:
            metrics[spec.numerator_key] = spec.threshold
        if spec.operator is GateOperator.ANY_LE:
            metrics[spec.alternative_numerator_key] = spec.alternative_threshold
    return metrics


def _calculate(evidence, authority):
    return calculate_gate_report(
        evidence,
        expected_source_digest=SOURCE,
        expected_authority_plan_sha256=authority.expected_plan_sha256,
        authority_plan=authority,
    )


def _evaluate(evidence, authority, *specs):
    return evaluate_gate_specs(
        evidence,
        specs=specs,
        expected_source_digest=SOURCE,
        expected_authority_plan_sha256=authority.expected_plan_sha256,
        authority_plan=authority,
    )


def test_complete_authority_report_recomputes_all_41_and_never_authorizes_24h(
    tmp_path,
):
    authority, evidence, _ = _write_evidence(
        tmp_path, "all", _passing_metrics()
    )
    report = _calculate(evidence, authority)
    assert len(report.results) == len(default_pre24_gate_specs()) == 41
    assert report.all_gates_pass is True
    assert report.authority_eligible is True
    assert report.complete_frozen_spec_set is True
    assert report.frozen_spec_set_digest == pre24_gate_spec_set_digest()
    assert report.twenty_four_hour_run_authorized is False
    assert report.verification_errors == ()
    assert all(item.evidence for item in report.results)
    assert all(item.reason == "threshold satisfied" for item in report.results)


def test_self_asserted_source_and_232_fake_cell_counterexample_fails(tmp_path):
    fake_ids = tuple(f"not-a-real-cell-{index}" for index in range(232))
    metrics = {"fresh_cells_observed": 232, "fresh_cells_declared": 232}
    authority, evidence, _ = _write_evidence(
        tmp_path,
        "attacker-source",
        metrics,
        universe_keys=("fresh_cells_declared",),
        trusted_universes={"fresh_cells_declared": fake_ids},
        raw_record_ids={
            "fresh_cells_observed": fake_ids,
            "fresh_cells_declared": fake_ids,
        },
        manifest_source_digest="attacker-self-asserted-source",
        artifact_source_digest="attacker-self-asserted-source",
    )
    spec = next(
        item
        for item in default_pre24_gate_specs()
        if item.gate_id == "coverage.fresh_cells"
    )
    evaluation = _evaluate(evidence, authority, spec)
    assert evidence.valid is False
    assert evaluation.all_evaluated_gates_pass is False
    assert "gate_manifest_expected_source_mismatch" in evaluation.verification_errors
    assert evaluation.authority_eligible is False
    assert evaluation.twenty_four_hour_run_authorized is False


def test_232_fake_ids_fail_against_root_compiled_exact_universe(tmp_path):
    real_ids = tuple(f"fresh-cell-{index}" for index in range(232))
    metrics = {"fresh_cells_observed": 232, "fresh_cells_declared": 232}
    authority, evidence, _ = _write_evidence(
        tmp_path,
        "attacker-identities",
        metrics,
        universe_keys=("fresh_cells_declared",),
        trusted_universes={"fresh_cells_declared": real_ids},
        raw_record_ids={
            "fresh_cells_observed": tuple(
                f"not-a-real-cell-{index}" for index in range(232)
            ),
            "fresh_cells_declared": tuple(
                f"not-a-real-cell-{index}" for index in range(232)
            ),
        },
    )
    spec = next(
        item
        for item in default_pre24_gate_specs()
        if item.gate_id == "coverage.fresh_cells"
    )
    evaluation = _evaluate(evidence, authority, spec)
    assert evidence.valid is False
    assert "trusted_denominator_universe_mismatch:fresh_cells_declared" in (
        evaluation.verification_errors
    )
    assert evaluation.all_evaluated_gates_pass is False


def test_custom_or_subset_specs_only_produce_permanently_non_authoritative_evaluation(
    tmp_path,
):
    authority, evidence, _ = _write_evidence(
        tmp_path, "full-for-subset", _passing_metrics()
    )
    spec = next(
        item
        for item in default_pre24_gate_specs()
        if item.gate_id == "coverage.fresh_cells"
    )
    evaluation = _evaluate(evidence, authority, spec)
    assert evaluation.all_evaluated_gates_pass is True
    assert evaluation.evaluation_only is True
    assert evaluation.authority_eligible is False
    assert evaluation.twenty_four_hour_run_authorized is False
    assert not hasattr(evaluation, "all_gates_pass")
    with pytest.raises(TypeError, match="unexpected keyword argument 'specs'"):
        calculate_gate_report(
            evidence,
            specs=(spec,),
            expected_source_digest=SOURCE,
            expected_authority_plan_sha256=authority.expected_plan_sha256,
            authority_plan=authority,
        )


def test_incomplete_universe_or_wrong_full_spec_digest_blocks_authority(tmp_path):
    metrics = _passing_metrics()
    all_denominators = _default_denominator_keys()
    missing_key = "fresh_cells_declared"
    incomplete, incomplete_evidence, _ = _write_evidence(
        tmp_path,
        "incomplete-plan",
        metrics,
        universe_keys=tuple(sorted(all_denominators - {missing_key})),
    )
    wrong_spec, wrong_spec_evidence, _ = _write_evidence(
        tmp_path,
        "wrong-spec",
        metrics,
        plan_spec_set_digest="attacker-subset-spec-set",
    )
    incomplete_report = _calculate(incomplete_evidence, incomplete)
    wrong_spec_report = _calculate(wrong_spec_evidence, wrong_spec)
    assert incomplete_report.all_gates_pass is False
    assert incomplete_report.authority_eligible is False
    assert any(
        item.startswith("authority_denominator_universes_missing:")
        and missing_key in item
        for item in incomplete_report.verification_errors
    )
    assert wrong_spec_report.all_gates_pass is False
    assert "authority_gate_spec_set_digest_mismatch" in (
        wrong_spec_report.verification_errors
    )


def test_hard_counts_and_thresholds_are_not_relaxed():
    specs = {item.gate_id: item for item in default_pre24_gate_specs()}
    assert specs["contract.confirmed_root_recall"].required_denominator == 9
    assert specs["contract.staged_exact_discrepancies"].minimum_denominator == 100_000
    assert specs["contract.compile_match_budget"].threshold == 2.0
    assert specs["contract.compile_match_budget"].alternative_threshold == 0.05
    assert specs["coverage.fresh_cells"].required_denominator == 232
    assert specs["coverage.contrast_edges"].required_denominator == 384
    assert specs["coverage.backend_pair_obligations"].required_denominator == 502
    assert specs["parallel.six_worker_efficiency"].threshold == 0.70
    assert specs["v3.completed_runs"].minimum_denominator == 22
    assert specs["v3.executed_cases"].required_denominator == 2200


def test_missing_metric_and_small_raw_denominator_fail_closed_in_evaluation(
    tmp_path,
):
    spec = GateSpec(
        "gate",
        "numerator",
        "denominator",
        GateOperator.EQ,
        1.0,
        minimum_denominator=10,
    )
    missing_authority, missing_evidence, _ = _write_evidence(
        tmp_path,
        "missing",
        {"denominator": 10},
        universe_keys=("denominator",),
    )
    small_authority, small_evidence, _ = _write_evidence(
        tmp_path,
        "small",
        {"numerator": 1, "denominator": 1},
        universe_keys=("denominator",),
    )
    missing = _evaluate(missing_evidence, missing_authority, spec).results[0]
    too_small = _evaluate(small_evidence, small_authority, spec).results[0]
    assert missing.passed is False and "missing_metric_series:numerator" in missing.reason
    assert too_small.passed is False and "below required minimum" in too_small.reason


def test_frozen_denominator_cannot_be_inflated_or_shrunk(tmp_path):
    spec = next(
        item
        for item in default_pre24_gate_specs()
        if item.gate_id == "coverage.fresh_cells"
    )
    inflated_authority, inflated, _ = _write_evidence(
        tmp_path,
        "inflated",
        {spec.numerator_key: 233, spec.denominator_key: 233},
        universe_keys=(spec.denominator_key,),
    )
    shrunk_authority, shrunk, _ = _write_evidence(
        tmp_path,
        "shrunk",
        {spec.numerator_key: 231, spec.denominator_key: 231},
        universe_keys=(spec.denominator_key,),
    )
    inflated_result = _evaluate(inflated, inflated_authority, spec).results[0]
    shrunk_result = _evaluate(shrunk, shrunk_authority, spec).results[0]
    assert inflated_result.passed is False and "frozen denominator 232" in inflated_result.reason
    assert shrunk_result.passed is False and "frozen denominator 232" in shrunk_result.reason


def test_zero_candidate_recheck_denominator_only_allows_zero_zero(tmp_path):
    spec = next(
        item
        for item in default_pre24_gate_specs()
        if item.gate_id == "v3.campaign_rechecks"
    )
    zero_authority, zero, _ = _write_evidence(
        tmp_path,
        "zero",
        {spec.numerator_key: 0, spec.denominator_key: 0},
        universe_keys=(spec.denominator_key,),
    )
    bad_authority, bad, _ = _write_evidence(
        tmp_path,
        "bad-zero",
        {spec.numerator_key: 1, spec.denominator_key: 0},
        universe_keys=(spec.denominator_key,),
    )
    assert _evaluate(zero, zero_authority, spec).results[0].passed
    assert _evaluate(bad, bad_authority, spec).results[0].passed is False


@pytest.mark.parametrize(
    ("p95_ms", "wall_share", "passed"),
    [(3.0, 0.04, True), (1.5, 0.08, True), (3.0, 0.08, False)],
)
def test_compile_match_budget_uses_raw_two_ms_or_five_percent_rule(
    tmp_path, p95_ms, wall_share, passed
):
    spec = next(
        item
        for item in default_pre24_gate_specs()
        if item.gate_id == "contract.compile_match_budget"
    )
    authority, evidence, _ = _write_evidence(
        tmp_path,
        f"budget-{p95_ms}-{wall_share}",
        {
            "contract_compile_match_p95_ms": p95_ms,
            "contract_compile_match_wall_share": wall_share,
            "unit_denominator": 1,
        },
        universe_keys=("unit_denominator",),
    )
    assert _evaluate(evidence, authority, spec).results[0].passed is passed


def test_count_numerator_must_belong_to_trusted_denominator_universe(tmp_path):
    metrics = {"numerator": 1, "denominator": 1}
    authority, evidence, _ = _write_evidence(
        tmp_path,
        "outside",
        metrics,
        universe_keys=("denominator",),
        raw_record_ids={"denominator": ("inside",), "numerator": ("outside",)},
    )
    spec = GateSpec("gate", "numerator", "denominator", GateOperator.EQ, 1.0)
    result = _evaluate(evidence, authority, spec).results[0]
    assert result.passed is False
    assert "outside the denominator universe" in result.reason


def test_arbitrary_mapping_and_post_verification_mutation_cannot_pass(tmp_path):
    authority, evidence, artifact_path = _write_evidence(
        tmp_path, "mutated", _passing_metrics()
    )
    with pytest.raises(TypeError, match="typed, hash-verified"):
        calculate_gate_report(
            {"numerator": 1},
            expected_source_digest=SOURCE,
            expected_authority_plan_sha256=authority.expected_plan_sha256,
            authority_plan=authority,
        )
    artifact_path.write_text("{}", encoding="utf-8")
    report = _calculate(evidence, authority)
    assert report.all_gates_pass is False
    assert report.authority_eligible is False
    assert "verified_evidence_snapshot_mismatch" in report.verification_errors
    assert report.twenty_four_hour_run_authorized is False
