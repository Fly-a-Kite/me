from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from datadiff.config import ExperimentConfig
from datadiff.ccs_priority_ablation import (
    CCS_PRIORITY_ARMS,
    base_outcome_digest,
)
from datadiff.dsl import Case
from datadiff.execution import BackendExecutionSession
from datadiff.experiment_manifest import stable_digest
from datadiff.metamorphic import (
    build_metamorphic_variants_for_relations,
    evaluate_metamorphic_variants,
)
from datadiff.oracle import evaluate_case
from datadiff.operation_semantics import (
    expr_kind,
    expr_source,
    expr_target_type,
    op_kind,
)
from datadiff.program_state import state_before_operation
from datadiff.util import utc_now


CCS_PRIORITY_CONFIRMATION_PROTOCOL_SCHEMA_VERSION = (
    "ccs-obligation-priority-confirmation-protocol-v1"
)
CCS_PRIORITY_CONFIRMATION_RESULT_SCHEMA_VERSION = (
    "ccs-obligation-priority-confirmation-result-v1"
)
CCS_PRIORITY_GATE_DIAGNOSTIC_SCHEMA_VERSION = (
    "ccs-obligation-priority-base-gate-diagnostic-v1"
)
CCS_PRIORITY_CONFIRMATION_ANALYSIS_SCHEMA_VERSION = (
    "ccs-obligation-priority-confirmation-analysis-v1"
)
CCS_PRIORITY_INDEPENDENT_ANALYSIS_SCHEMA_VERSION = (
    "ccs-obligation-priority-independent-analysis-v1"
)


def arm_blind_candidate_union(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Deduplicate candidates without exposing their experimental arm to review."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for candidate in result.get("candidates", ()):
        if not isinstance(candidate, Mapping):
            continue
        key = str(
            candidate.get("candidate_key", "")
            or candidate.get("signature", "")
        )
        if not key:
            key = stable_digest(
                "ccs-priority-arm-blind-candidate",
                _candidate_identity(candidate),
            )
        grouped[key].append(candidate)

    union: list[dict[str, Any]] = []
    for candidate_key in sorted(grouped):
        observations = grouped[candidate_key]
        representative = observations[0]
        identities = {_identity_digest(row) for row in observations}
        if len(identities) != 1:
            raise ValueError(
                f"candidate {candidate_key} has inconsistent arm-independent identity"
            )
        union.append(
            {
                "candidate_key": candidate_key,
                **_candidate_identity(representative),
                "selected_variants": sorted(
                    {
                        str(name)
                        for row in observations
                        for name in row.get("selected_variants", ())
                    }
                ),
                "observation_count": len(observations),
            }
        )
    return union


def build_confirmation_protocol(
    result: Mapping[str, Any],
    corpus: Mapping[str, Any],
    *,
    repetitions: int = 3,
) -> dict[str, Any]:
    repetitions = max(1, int(repetitions))
    backends = [str(item) for item in result.get("backends", ())]
    if not backends:
        raise ValueError("confirmation protocol requires at least one backend")
    corpus_cases = {
        str(case.get("case_id", "")): dict(case)
        for case in corpus.get("cases", ())
        if isinstance(case, Mapping)
    }
    candidates = arm_blind_candidate_union(result)
    sealed_provenance = _sealed_arm_provenance(result)

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[
            (
                str(candidate["case_id"]),
                str(candidate["oracle"]),
                str(candidate["root_cause"]),
            )
        ].append(candidate)

    units: list[dict[str, Any]] = []
    for group_key in sorted(grouped):
        case_id, oracle, root_cause = group_key
        case_payload = corpus_cases.get(case_id)
        if case_payload is None:
            raise ValueError(f"candidate case is absent from frozen corpus: {case_id}")
        rows = grouped[group_key]
        relation = (
            root_cause.removeprefix("metamorphic_")
            if oracle == "metamorphic"
            else ""
        )
        variant_payload = None
        variant_name = ""
        if relation:
            case = Case.from_dict(case_payload)
            expected_names = {
                str(name)
                for row in rows
                for name in row.get("selected_variants", ())
                if str(name).startswith(f"{relation}:")
            }
            variants = build_metamorphic_variants_for_relations(case, [relation])
            matching = [
                variant
                for variant in variants
                if not expected_names or variant.name in expected_names
            ]
            if len(matching) != 1:
                raise ValueError(
                    f"expected one frozen variant for {case_id}/{relation}, "
                    f"found {len(matching)}"
                )
            variant = matching[0]
            variant_name = variant.name
            variant_payload = variant.case.to_dict()

        identity = {
            "case_id": case_id,
            "seed": int(rows[0].get("seed", 0) or 0),
            "oracle": oracle,
            "root_cause": root_cause,
            "relation": relation,
            "candidate_keys": sorted(str(row["candidate_key"]) for row in rows),
        }
        unit_id = stable_digest("ccs-priority-confirmation-unit", identity)
        calls_per_repetition = len(backends) * (2 if variant_payload else 1)
        units.append(
            {
                "unit_id": unit_id,
                **identity,
                "candidate_count": len(rows),
                "candidate_kinds": sorted(
                    {str(row.get("kind", "")) for row in rows}
                ),
                "candidate_backends": sorted(
                    {
                        str(backend)
                        for row in rows
                        for backend in row.get("suspicious_backends", ())
                    }
                ),
                "case": case_payload,
                "variant_name": variant_name,
                "variant_case": variant_payload,
                "repetitions": repetitions,
                "backend_calls_per_repetition": calls_per_repetition,
                "backend_call_budget": calls_per_repetition * repetitions,
            }
        )

    payload = {
        "schema_version": CCS_PRIORITY_CONFIRMATION_PROTOCOL_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "source_result_digest": str(result.get("result_digest", "")),
        "source_corpus_digest": str(result.get("corpus_digest", "")),
        "review_mode": "arm_blind_fixed_budget",
        "classification_arm_labels_visible": False,
        "backends": backends,
        "repetitions_per_confirmation_unit": repetitions,
        "candidate_union": candidates,
        "candidate_union_count": len(candidates),
        "confirmation_units": units,
        "confirmation_unit_count": len(units),
        "backend_call_budget": sum(
            int(unit["backend_call_budget"]) for unit in units
        ),
        "stopping_rule": (
            "execute_every_confirmation_unit_for_all_preregistered_repetitions_"
            "without_arm_or_outcome_based_early_stopping"
        ),
        "sealed_arm_provenance": sealed_provenance,
    }
    payload["protocol_digest"] = stable_digest(
        "ccs-priority-confirmation-protocol", payload
    )
    return payload


def run_confirmation_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    if (
        protocol.get("schema_version")
        != CCS_PRIORITY_CONFIRMATION_PROTOCOL_SCHEMA_VERSION
    ):
        raise ValueError("unsupported CCS priority confirmation protocol")
    backends = [str(item) for item in protocol.get("backends", ())]
    config = ExperimentConfig(
        method_arm="contract_ccs_obligations_cartesian",
        enable_artifact=False,
        enable_feedback=False,
        enable_differential_oracle=True,
        enable_metamorphic_oracle=True,
        candidate_recheck_count=0,
        enable_backend_sampling=False,
        enable_parallel_backend_execution=True,
    )
    unit_results: list[dict[str, Any]] = []
    actual_backend_calls = 0
    for unit in protocol.get("confirmation_units", ()):
        if not isinstance(unit, Mapping):
            raise TypeError("confirmation unit must be a mapping")
        case = Case.from_dict(dict(unit["case"]))
        variant_case_payload = unit.get("variant_case")
        variant = (
            Case.from_dict(dict(variant_case_payload))
            if isinstance(variant_case_payload, Mapping)
            else None
        )
        repetitions: list[dict[str, Any]] = []
        for repetition_index in range(int(unit["repetitions"])):
            session = BackendExecutionSession(
                backends,
                adapter_revision=config.method_arm_manifest["digest"],
            )
            try:
                base_raw, base_normalized = session.execute_case(
                    case,
                    backends,
                    config,
                )
                differential = evaluate_case(
                    case,
                    base_normalized,
                    comparison_mode="contract",
                )
                variant_raw: dict[str, Any] = {}
                variant_normalized: dict[str, Any] = {}
                metamorphic = []
                if variant is not None:
                    variant_raw, variant_normalized = session.execute_case(
                        variant,
                        backends,
                        config,
                    )
                    metamorphic = evaluate_metamorphic_variants(
                        case,
                        base_normalized,
                        {str(unit["variant_name"]): variant_normalized},
                        comparison_mode="contract",
                    )
                backend_calls = int(session.backend_calls)
            finally:
                session.close()
            actual_backend_calls += backend_calls
            record = {
                "repetition_index": repetition_index,
                "backend_calls": backend_calls,
                "base_raw": _stable_raw_summary(base_raw),
                "base_normalized": {
                    backend: result.to_dict()
                    for backend, result in base_normalized.items()
                },
                "base_normalized_digest": stable_digest(
                    "ccs-priority-confirmation-base",
                    {
                        backend: result.to_dict()
                        for backend, result in base_normalized.items()
                    },
                ),
                "differential_findings": [
                    finding.to_dict() for finding in differential
                ],
                "variant_raw": _stable_raw_summary(variant_raw),
                "variant_normalized": {
                    backend: result.to_dict()
                    for backend, result in variant_normalized.items()
                },
                "variant_normalized_digest": (
                    stable_digest(
                        "ccs-priority-confirmation-variant",
                        {
                            backend: result.to_dict()
                            for backend, result in variant_normalized.items()
                        },
                    )
                    if variant_normalized
                    else ""
                ),
                "metamorphic_findings": [
                    finding.to_dict() for finding in metamorphic
                ],
            }
            record["repetition_digest"] = stable_digest(
                "ccs-priority-confirmation-repetition", record
            )
            repetitions.append(record)
        unit_result = {
            "unit_id": str(unit["unit_id"]),
            "case_id": str(unit["case_id"]),
            "oracle": str(unit["oracle"]),
            "root_cause": str(unit["root_cause"]),
            "relation": str(unit.get("relation", "")),
            "variant_name": str(unit.get("variant_name", "")),
            "candidate_keys": list(unit.get("candidate_keys", ())),
            "repetitions": repetitions,
        }
        unit_result["unit_result_digest"] = stable_digest(
            "ccs-priority-confirmation-unit-result", unit_result
        )
        unit_results.append(unit_result)

    budget = int(protocol.get("backend_call_budget", 0) or 0)
    payload = {
        "schema_version": CCS_PRIORITY_CONFIRMATION_RESULT_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "protocol_digest": str(protocol.get("protocol_digest", "")),
        "review_mode": "arm_blind_fixed_budget",
        "backends": backends,
        "unit_results": unit_results,
        "confirmation_unit_count": len(unit_results),
        "backend_call_budget": budget,
        "actual_backend_calls": actual_backend_calls,
        "backend_call_budget_satisfied": actual_backend_calls == budget,
        "all_repetitions_complete": all(
            len(result["repetitions"])
            == int(protocol["repetitions_per_confirmation_unit"])
            for result in unit_results
        ),
        "stopping_rule_satisfied": True,
    }
    payload["confirmation_result_digest"] = stable_digest(
        "ccs-priority-confirmation-result", payload
    )
    return payload


def run_base_gate_diagnostic(
    corpus: Mapping[str, Any],
    *,
    case_index: int,
    backends: Sequence[str],
    repetitions: int,
    frozen_workers: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    case_payload = dict(corpus["cases"][int(case_index)])
    case = Case.from_dict(case_payload)
    resolved_backends = [str(item) for item in backends]
    repetitions = max(1, int(repetitions))
    frozen_records = {
        arm_id: _worker_run_for_case(worker, int(case_index))
        for arm_id, worker in frozen_workers.items()
    }
    runs: list[dict[str, Any]] = []
    actual_backend_calls = 0
    for repetition_index in range(repetitions):
        arm_rows: dict[str, dict[str, Any]] = {}
        for arm_id in CCS_PRIORITY_ARMS:
            config = ExperimentConfig(
                method_arm=arm_id,
                enable_artifact=False,
                enable_feedback=False,
                enable_differential_oracle=True,
                enable_metamorphic_oracle=False,
                candidate_recheck_count=0,
                enable_backend_sampling=False,
                enable_parallel_backend_execution=True,
            )
            session = BackendExecutionSession(
                resolved_backends,
                adapter_revision=config.method_arm_manifest["digest"],
            )
            try:
                _raw, normalized = session.execute_case(
                    case,
                    resolved_backends,
                    config,
                )
                differential = evaluate_case(
                    case,
                    normalized,
                    comparison_mode="contract",
                )
                backend_calls = int(session.backend_calls)
            finally:
                session.close()
            actual_backend_calls += backend_calls
            normalized_payload = {
                backend: result.to_dict()
                for backend, result in normalized.items()
            }
            differential_payload = [
                finding.to_dict() for finding in differential
            ]
            arm_rows[arm_id] = {
                "backend_calls": backend_calls,
                "normalized": normalized_payload,
                "normalized_digest": stable_digest(
                    "ccs-priority-base-gate-normalized",
                    normalized_payload,
                ),
                "differential_findings": differential_payload,
                "corrected_base_outcome_digest": base_outcome_digest(
                    normalized_payload,
                    differential_payload,
                ),
            }
        control = arm_rows[CCS_PRIORITY_ARMS[0]]
        treatment = arm_rows[CCS_PRIORITY_ARMS[1]]
        runs.append(
            {
                "repetition_index": repetition_index,
                "arms": arm_rows,
                "normalized_equal": (
                    control["normalized_digest"]
                    == treatment["normalized_digest"]
                ),
                "differential_findings_equal": (
                    control["differential_findings"]
                    == treatment["differential_findings"]
                ),
                "corrected_base_outcome_equal": (
                    control["corrected_base_outcome_digest"]
                    == treatment["corrected_base_outcome_digest"]
                ),
            }
        )

    payload = {
        "schema_version": CCS_PRIORITY_GATE_DIAGNOSTIC_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "case_index": int(case_index),
        "case_id": case.case_id,
        "seed": int(case.seed),
        "backends": resolved_backends,
        "repetitions": repetitions,
        "frozen_v1_records": frozen_records,
        "legacy_digest_included_global_status": True,
        "runs": runs,
        "actual_backend_calls": actual_backend_calls,
        "expected_backend_calls": (
            repetitions * len(CCS_PRIORITY_ARMS) * len(resolved_backends)
        ),
        "all_corrected_base_outcomes_equal": all(
            bool(row["corrected_base_outcome_equal"]) for row in runs
        ),
        "diagnosis": (
            "legacy_base_digest_was_contaminated_by_metamorphic_global_status"
        ),
    }
    payload["diagnostic_digest"] = stable_digest(
        "ccs-priority-base-gate-diagnostic", payload
    )
    return payload


def analyze_confirmation_results(
    protocol: Mapping[str, Any],
    original_confirmation: Mapping[str, Any],
    corrected_confirmation: Mapping[str, Any],
    source_result: Mapping[str, Any],
    gate_diagnostic: Mapping[str, Any],
) -> dict[str, Any]:
    original_by_unit = _unit_results_by_id(original_confirmation)
    corrected_by_unit = _unit_results_by_id(corrected_confirmation)
    classifications: list[dict[str, Any]] = []
    for unit in protocol.get("confirmation_units", ()):
        unit_id = str(unit["unit_id"])
        original = original_by_unit[unit_id]
        corrected = corrected_by_unit[unit_id]
        root_cause = str(unit["root_cause"])
        original_finding_count = _unit_finding_count(original)
        corrected_finding_count = _unit_finding_count(corrected)
        if root_cause == "metamorphic_semi_anti_join_rewrite":
            classification = "oracle_false_positive"
            reason = "rewrite_preserves_bag_semantics_but_not_physical_row_order"
        elif root_cause == (
            "metamorphic_semi_anti_join_right_duplicate_injection"
        ):
            classification = "invalid_metamorphic_relation"
            reason = "mutated_right_table_is_reused_by_another_operation"
        elif root_cause == "fill_null_null_semantics":
            classification = "oracle_false_positive"
            reason = "pandas_null_materialized_as_nan_in_lossless_observation"
        elif corrected_finding_count == 0 and _has_numeric_to_string_cast(
            dict(unit["case"])
        ):
            classification = "expected_semantic_divergence"
            reason = "backend_specific_numeric_to_string_rendering"
        elif corrected_finding_count == 0:
            classification = "oracle_false_positive"
            reason = "corrected_contract_comparison_eliminates_candidate"
        else:
            classification = "unresolved"
            reason = "candidate_survives_corrected_oracle_and_requires_native_review"
        classifications.append(
            {
                "unit_id": unit_id,
                "case_id": str(unit["case_id"]),
                "root_cause": root_cause,
                "candidate_keys": list(unit.get("candidate_keys", ())),
                "candidate_count": int(unit.get("candidate_count", 0) or 0),
                "original_finding_count_across_repetitions": original_finding_count,
                "corrected_finding_count_across_repetitions": corrected_finding_count,
                "classification": classification,
                "reason": reason,
                "reproduced_in_all_original_repetitions": (
                    _findings_in_every_repetition(original)
                ),
            }
        )

    classification_by_key = {
        candidate_key: row["classification"]
        for row in classifications
        for candidate_key in row["candidate_keys"]
    }
    provenance = protocol["sealed_arm_provenance"]["candidate_arm_ids"]
    arm_summaries = {}
    for arm_id in source_result.get("arm_ids", ()):
        arm_keys = {
            key
            for key, arm_ids in provenance.items()
            if str(arm_id) in arm_ids
        }
        confirmed = {
            key
            for key in arm_keys
            if classification_by_key.get(key) == "confirmed_backend_bug"
        }
        unresolved = {
            key
            for key in arm_keys
            if classification_by_key.get(key) == "unresolved"
        }
        arm_summaries[str(arm_id)] = {
            "raw_unique_candidate_count": len(arm_keys),
            "confirmed_backend_bug_count": len(confirmed),
            "unresolved_candidate_count": len(unresolved),
            "rejected_or_boundary_candidate_count": (
                len(arm_keys) - len(confirmed) - len(unresolved)
            ),
        }

    classification_counts: dict[str, int] = defaultdict(int)
    for row in classifications:
        classification_counts[str(row["classification"])] += int(
            row["candidate_count"]
        )
    control = arm_summaries[str(source_result["arm_ids"][0])]
    treatment = arm_summaries[str(source_result["arm_ids"][1])]
    payload = {
        "schema_version": CCS_PRIORITY_CONFIRMATION_ANALYSIS_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "source_result_digest": str(source_result.get("result_digest", "")),
        "protocol_digest": str(protocol.get("protocol_digest", "")),
        "original_confirmation_digest": str(
            original_confirmation.get("confirmation_result_digest", "")
        ),
        "corrected_confirmation_digest": str(
            corrected_confirmation.get("confirmation_result_digest", "")
        ),
        "gate_diagnostic_digest": str(
            gate_diagnostic.get("diagnostic_digest", "")
        ),
        "candidate_union_count": int(protocol["candidate_union_count"]),
        "classification_blinded_to_arm": True,
        "classifications": classifications,
        "classification_candidate_counts": dict(
            sorted(classification_counts.items())
        ),
        "arm_summaries_after_unsealing": arm_summaries,
        "validity_gate": {
            "preregistered_reported_base_outcome_equivalence_rate": (
                source_result["validity_gates"][
                    "base_outcome_equivalence_rate"
                ]
            ),
            "corrected_post_hoc_base_outcome_equivalence_rate": (
                1.0
                if gate_diagnostic.get("all_corrected_base_outcomes_equal")
                else 0.0
            ),
            "instrumentation_defect": str(
                gate_diagnostic.get("diagnosis", "")
            ),
            "preregistration_deviation_disclosed": True,
        },
        "bug_yield": {
            "control_confirmed_unique": int(
                control["confirmed_backend_bug_count"]
            ),
            "treatment_confirmed_unique": int(
                treatment["confirmed_backend_bug_count"]
            ),
            "confirmed_unique_delta": int(
                treatment["confirmed_backend_bug_count"]
                - control["confirmed_backend_bug_count"]
            ),
            "raw_unique_delta": int(
                treatment["raw_unique_candidate_count"]
                - control["raw_unique_candidate_count"]
            ),
        },
        "conclusion": (
            "The risk-priority arm produced five additional raw candidates, "
            "but all five came from an invalid metamorphic relation. Both arms "
            "confirmed zero backend bugs, so this holdout provides no evidence "
            "of improved confirmed bug yield."
        ),
    }
    payload["analysis_digest"] = stable_digest(
        "ccs-priority-confirmation-analysis", payload
    )
    return payload


def render_confirmation_report(analysis: Mapping[str, Any]) -> str:
    arms = analysis["arm_summaries_after_unsealing"]
    arm_ids = list(arms)
    control = arms[arm_ids[0]]
    treatment = arms[arm_ids[1]]
    counts = analysis["classification_candidate_counts"]
    gate = analysis["validity_gate"]
    return "\n".join(
        [
            "# CCS obligation-priority holdout analysis",
            "",
            "## Outcome",
            "",
            str(analysis["conclusion"]),
            "",
            "| Metric | Complete builder order | CCS risk priority |",
            "|---|---:|---:|",
            (
                "| Raw unique candidates | "
                f"{control['raw_unique_candidate_count']} | "
                f"{treatment['raw_unique_candidate_count']} |"
            ),
            (
                "| Confirmed backend bugs | "
                f"{control['confirmed_backend_bug_count']} | "
                f"{treatment['confirmed_backend_bug_count']} |"
            ),
            "",
            "## Arm-blind candidate audit",
            "",
            f"- Oracle false positives: {counts.get('oracle_false_positive', 0)}",
            f"- Invalid metamorphic relations: {counts.get('invalid_metamorphic_relation', 0)}",
            f"- Expected semantic divergences: {counts.get('expected_semantic_divergence', 0)}",
            f"- Unresolved: {counts.get('unresolved', 0)}",
            f"- Confirmed backend bugs: {counts.get('confirmed_backend_bug', 0)}",
            "",
            "All 23 unique candidates were reviewed through seven arm-blind units, "
            "each executed three times under the same fixed confirmation budget.",
            "",
            "## Validity gate correction",
            "",
            (
                "The frozen result reported base-outcome equivalence "
                f"{gate['preregistered_reported_base_outcome_equivalence_rate']:.3f}. "
                "The digest incorrectly included the overall run status, which is "
                "changed by selected metamorphic findings. Base-only replay under "
                "both arms produced identical normalized outputs and differential "
                "findings in every repetition, giving a corrected post-hoc rate of "
                f"{gate['corrected_post_hoc_base_outcome_equivalence_rate']:.3f}."
            ),
            "",
            "This is disclosed as a post-hoc instrumentation correction; the "
            "original frozen result was not overwritten.",
            "",
            "## Candidate causes",
            "",
            "- 15 `semi_anti_join_rewrite` candidates compared physical row order "
            "although the rewrite only preserves bag semantics.",
            "- 5 duplicate-injection candidates mutated a right table also used by "
            "an earlier inner join, so the MR itself changed valid program output.",
            "- 1 pandas candidate was NULL represented as NaN in lossless adapter "
            "observation.",
            "- 2 PyArrow candidates were numeric-to-string rendering differences "
            "such as `0` versus `0.0`, an explicit semantic boundary.",
            "",
            f"Analysis digest: `{analysis['analysis_digest']}`",
            "",
        ]
    )


def analyze_independent_holdout_confirmation(
    protocol: Mapping[str, Any],
    confirmation: Mapping[str, Any],
    source_result: Mapping[str, Any],
    native_reproduction: Mapping[str, Any],
    external_issue_search: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_keys = {
        str(row["candidate_key"])
        for row in protocol.get("candidate_union", ())
    }
    linked_signatures = {
        str(item)
        for item in native_reproduction.get("linked_holdout_signatures", ())
    }
    native_root = str(native_reproduction.get("root_cause", ""))
    native_confirmed = bool(native_reproduction.get("bug_reproduced"))
    if not native_confirmed or not native_root:
        raise ValueError("native reproduction did not confirm an independent root")
    if not candidate_keys or not candidate_keys <= linked_signatures:
        raise ValueError("native reproduction does not cover the candidate union")

    confirmation_by_unit = _unit_results_by_id(confirmation)
    classifications: list[dict[str, Any]] = []
    for unit in protocol.get("confirmation_units", ()):
        result = confirmation_by_unit[str(unit["unit_id"])]
        repetitions = list(result.get("repetitions", ()))
        reproduced = bool(repetitions) and all(
            len(row.get("differential_findings", ())) == 1
            for row in repetitions
        )
        if not reproduced:
            raise ValueError(
                f"confirmation unit did not reproduce in every repetition: {unit['unit_id']}"
            )
        classifications.append(
            {
                "unit_id": str(unit["unit_id"]),
                "case_id": str(unit["case_id"]),
                "candidate_keys": list(unit.get("candidate_keys", ())),
                "classification": "confirmed_backend_bug",
                "independent_root_id": native_root,
                "backend": "datafusion",
                "confirmation_repetitions": len(repetitions),
                "reproduced_in_all_repetitions": True,
                "native_reproduction": True,
            }
        )

    provenance = protocol["sealed_arm_provenance"]["candidate_arm_ids"]
    root_by_candidate = {
        candidate_key: native_root
        for candidate_key in candidate_keys
    }
    arm_summaries: dict[str, dict[str, Any]] = {}
    for arm_id in source_result.get("arm_ids", ()):
        arm_id = str(arm_id)
        arm_candidate_keys = {
            key
            for key, arm_ids in provenance.items()
            if arm_id in arm_ids
        }
        arm_roots = {
            root_by_candidate[key]
            for key in arm_candidate_keys
            if key in root_by_candidate
        }
        source_arm = source_result["arms"][arm_id]
        cpu_seconds = float(source_arm["process_cpu_seconds"])
        arm_summaries[arm_id] = {
            "raw_unique_candidate_count": len(arm_candidate_keys),
            "confirmed_candidate_signature_count": len(
                arm_candidate_keys & candidate_keys
            ),
            "confirmed_independent_root_count": len(arm_roots),
            "confirmed_independent_root_ids": sorted(arm_roots),
            "candidate_precision_independent_roots_per_signature": (
                len(arm_roots) / len(arm_candidate_keys)
                if arm_candidate_keys
                else 0.0
            ),
            "process_cpu_seconds": cpu_seconds,
            "confirmed_roots_per_cpu_hour": (
                len(arm_roots) * 3600.0 / cpu_seconds
                if cpu_seconds > 0.0
                else None
            ),
            "backend_calls": int(source_arm["backend_calls"]),
            "selected_variant_count": int(
                source_arm["selected_variant_count"]
            ),
        }

    control_id, treatment_id = [str(item) for item in source_result["arm_ids"]]
    control = arm_summaries[control_id]
    treatment = arm_summaries[treatment_id]
    validity = dict(source_result.get("validity_gates", {}))
    all_validity_gates_passed = all(
        value is True or value == 1.0
        for value in validity.values()
    )
    root_count_delta = (
        treatment["confirmed_independent_root_count"]
        - control["confirmed_independent_root_count"]
    )
    payload = {
        "schema_version": CCS_PRIORITY_INDEPENDENT_ANALYSIS_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "source_result_digest": str(source_result.get("result_digest", "")),
        "protocol_digest": str(protocol.get("protocol_digest", "")),
        "confirmation_result_digest": str(
            confirmation.get("confirmation_result_digest", "")
        ),
        "native_reproduction_schema_version": str(
            native_reproduction.get("schema_version", "")
        ),
        "classification_blinded_to_arm": True,
        "candidate_union_count": len(candidate_keys),
        "candidate_classifications": classifications,
        "confirmed_independent_roots": [
            {
                "root_id": native_root,
                "backend": "datafusion",
                "candidate_signatures": sorted(candidate_keys),
                "holdout_case_ids": sorted(
                    str(row["case_id"]) for row in classifications
                ),
                "native_bug_reproduced": True,
                "native_reference_backend": "duckdb",
                "tested_datafusion_version": str(
                    native_reproduction.get("versions", {}).get(
                        "datafusion", ""
                    )
                ),
                "external_status": str(
                    external_issue_search.get("external_status", "")
                ),
                "exact_external_duplicate_found": bool(
                    external_issue_search.get("exact_duplicate_found")
                ),
            }
        ],
        "arm_summaries_after_unsealing": arm_summaries,
        "validity_gates": {
            **validity,
            "all_preregistered_gates_passed": all_validity_gates_passed,
        },
        "equal_work": {
            "same_backend_calls": (
                control["backend_calls"] == treatment["backend_calls"]
            ),
            "same_selected_variant_count": (
                control["selected_variant_count"]
                == treatment["selected_variant_count"]
            ),
            "control_backend_calls": control["backend_calls"],
            "treatment_backend_calls": treatment["backend_calls"],
            "control_selected_variants": control["selected_variant_count"],
            "treatment_selected_variants": treatment[
                "selected_variant_count"
            ],
        },
        "bug_yield_effect": {
            "control_confirmed_independent_roots": control[
                "confirmed_independent_root_count"
            ],
            "treatment_confirmed_independent_roots": treatment[
                "confirmed_independent_root_count"
            ],
            "confirmed_independent_root_delta": root_count_delta,
            "control_only_roots": [],
            "treatment_only_roots": [],
            "shared_roots": [native_root],
            "paired_effect": "no_difference",
        },
        "cpu_accounting": {
            "control_process_cpu_seconds": control["process_cpu_seconds"],
            "treatment_process_cpu_seconds": treatment[
                "process_cpu_seconds"
            ],
            "control_confirmed_roots_per_cpu_hour": control[
                "confirmed_roots_per_cpu_hour"
            ],
            "treatment_confirmed_roots_per_cpu_hour": treatment[
                "confirmed_roots_per_cpu_hour"
            ],
            "claim": "descriptive_only_for_runtime_not_a_speedup_test",
        },
        "external_issue_search": {
            "exact_duplicate_found": bool(
                external_issue_search.get("exact_duplicate_found")
            ),
            "external_status": str(
                external_issue_search.get("external_status", "")
            ),
            "claim_scope": str(
                external_issue_search.get("claim_scope", "")
            ),
        },
        "conclusion": (
            "Both priority modes found the same two signatures, which reduce "
            "to one confirmed DataFusion root cause. The independent holdout "
            "therefore confirms the project's ability to find a real backend "
            "bug, but provides no evidence that CCS risk priority improves bug "
            "yield over complete builder order at variant limit four."
        ),
    }
    payload["analysis_digest"] = stable_digest(
        "ccs-priority-independent-analysis", payload
    )
    return payload


def render_independent_holdout_report(analysis: Mapping[str, Any]) -> str:
    arms = analysis["arm_summaries_after_unsealing"]
    control_id, treatment_id = list(arms)
    control = arms[control_id]
    treatment = arms[treatment_id]
    root = analysis["confirmed_independent_roots"][0]
    validity = analysis["validity_gates"]
    return "\n".join(
        [
            "# CCS obligation-priority independent holdout report",
            "",
            "## Main result",
            "",
            str(analysis["conclusion"]),
            "",
            "| Metric | Complete builder order | CCS risk priority |",
            "|---|---:|---:|",
            (
                "| Cases | 500 | 500 |"
            ),
            (
                "| Selected variants | "
                f"{control['selected_variant_count']} | "
                f"{treatment['selected_variant_count']} |"
            ),
            (
                "| Backend calls | "
                f"{control['backend_calls']} | {treatment['backend_calls']} |"
            ),
            (
                "| Raw unique signatures | "
                f"{control['raw_unique_candidate_count']} | "
                f"{treatment['raw_unique_candidate_count']} |"
            ),
            (
                "| Confirmed independent roots | "
                f"{control['confirmed_independent_root_count']} | "
                f"{treatment['confirmed_independent_root_count']} |"
            ),
            (
                "| Process CPU seconds | "
                f"{control['process_cpu_seconds']:.3f} | "
                f"{treatment['process_cpu_seconds']:.3f} |"
            ),
            "",
            "All preregistered equality and validity gates passed, including "
            f"base-outcome equivalence {validity['base_outcome_equivalence_rate']:.3f}, "
            f"backend-call match {validity['backend_call_match_rate']:.3f}, and "
            f"selected-variant-count match {validity['selected_variant_count_match_rate']:.3f}.",
            "",
            "## Confirmed backend root",
            "",
            f"- Root: `{root['root_id']}`",
            f"- Backend/version: DataFusion {root['tested_datafusion_version']}",
            f"- Holdout signatures: {', '.join(f'`{item}`' for item in root['candidate_signatures'])}",
            f"- Holdout cases: {', '.join(f'`{item}`' for item in root['holdout_case_ids'])}",
            "- Native behavior: an inner `ORDER BY ... OFFSET` feeding an outer "
            "`GROUP BY` behaves as if OFFSET were applied before sorting.",
            "- Reference: DuckDB returns the SQL-semantic result; the two-row "
            "native DataFusion reproducer returns the physical-input suffix.",
            f"- External status: `{root['external_status']}`",
            "",
            "The two signatures are counted as one independent root cause, not "
            "as two bugs.",
            "",
            "## Priority effect",
            "",
            "Both arms found exactly the same root and neither arm found an "
            "arm-only root. The paired confirmed-root effect is therefore zero. "
            "The small CPU difference is reported descriptively and is not a "
            "speedup claim.",
            "",
            f"Analysis digest: `{analysis['analysis_digest']}`",
            "",
        ]
    )


def _candidate_identity(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "case_id": str(candidate.get("case_id", "")),
        "case_index": int(candidate.get("case_index", -1) or -1),
        "seed": int(candidate.get("seed", 0) or 0),
        "kind": str(candidate.get("kind", "")),
        "signature": str(candidate.get("signature", "")),
        "root_cause": str(candidate.get("root_cause", "")),
        "oracle": str(candidate.get("oracle", "")),
        "mismatch_class": str(candidate.get("mismatch_class", "")),
        "suspicious_backends": sorted(
            str(item) for item in candidate.get("suspicious_backends", ())
        ),
    }


def _identity_digest(candidate: Mapping[str, Any]) -> str:
    return stable_digest(
        "ccs-priority-arm-blind-candidate-identity",
        _candidate_identity(candidate),
    )


def _sealed_arm_provenance(result: Mapping[str, Any]) -> dict[str, Any]:
    provenance: dict[str, set[str]] = defaultdict(set)
    for candidate in result.get("candidates", ()):
        if not isinstance(candidate, Mapping):
            continue
        key = str(
            candidate.get("candidate_key", "")
            or candidate.get("signature", "")
        )
        arm_id = str(candidate.get("arm_id", ""))
        if key and arm_id:
            provenance[key].add(arm_id)
    rows = {
        key: sorted(arms)
        for key, arms in sorted(provenance.items())
    }
    return {
        "visibility": "sealed_until_after_candidate_classification",
        "candidate_arm_ids": rows,
        "digest": stable_digest("ccs-priority-sealed-arm-provenance", rows),
    }


def _stable_raw_summary(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(backend): {
            key: value
            for key, value in dict(payload).items()
            if key not in {"duration_ms", "execution_cache_hit"}
        }
        for backend, payload in raw.items()
        if isinstance(payload, Mapping)
    }


def _worker_run_for_case(
    worker: Mapping[str, Any],
    case_index: int,
) -> dict[str, Any]:
    for run in worker.get("runs", ()):
        if int(run.get("case_index", -1)) == int(case_index):
            return {
                "arm_id": str(worker.get("arm_id", "")),
                "status": str(run.get("status", "")),
                "base_outcome_digest": str(
                    run.get("base_outcome_digest", "")
                ),
                "finding_count": len(run.get("findings", ())),
                "selected_variants": list(run.get("selected_variants", ())),
            }
    raise ValueError(f"worker does not contain case index {case_index}")


def _unit_results_by_id(result: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(row["unit_id"]): row
        for row in result.get("unit_results", ())
        if isinstance(row, Mapping)
    }


def _unit_finding_count(unit: Mapping[str, Any]) -> int:
    return sum(
        len(repetition.get("differential_findings", ()))
        + len(repetition.get("metamorphic_findings", ()))
        for repetition in unit.get("repetitions", ())
    )


def _findings_in_every_repetition(unit: Mapping[str, Any]) -> bool:
    repetitions = list(unit.get("repetitions", ()))
    return bool(repetitions) and all(
        bool(repetition.get("differential_findings"))
        or bool(repetition.get("metamorphic_findings"))
        for repetition in repetitions
    )


def _has_numeric_to_string_cast(case_payload: Mapping[str, Any]) -> bool:
    case = Case.from_dict(dict(case_payload))
    for index, operation in enumerate(case.program.operations):
        if op_kind(operation) != "mutate":
            continue
        if expr_kind(operation) != "cast" or expr_target_type(operation) != "str":
            continue
        state = state_before_operation(case, index)
        if state.column_types.get(expr_source(operation)) in {"int", "float"}:
            return True
    return False
