"""Root-owned, fail-closed control plane for Phase-6 gate authority.

This private module deliberately separates preregistered identities from raw
gate manifests.  It can describe and verify a complete authority bundle, but it
does not run Phase 6 and can never authorize a 24-hour campaign.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Iterable

from datadiff.semantic_family_universe_v3 import pipeline_evidence_specs
from datadiff.target_version_audit import DEFAULT_TARGET_PACKAGES

from datadiff_osc._canonical import (
    canonical_json,
    decode_canonical_envelope,
    stable_digest,
    to_primitive,
)
from datadiff_osc.contract_engine._phase6_comparison_receipts import (
    ContractComparisonReceipt,
)
from datadiff_osc.contract_engine.mutations import (
    AXIS_FAULTS,
    COMPARATOR_WEAKENING_MUTANTS,
    HYPEREDGE_MUTANTS,
)
from datadiff_osc.contract_engine.replay import (
    _reconstruct_contract_comparison_receipt_payload,
    replay_contract_admission,
)
from datadiff_osc.parallel.invariance import InvarianceReport
from datadiff_osc.runtime._phase6_contract_performance_receipts import (
    BASELINE_STRATEGY,
    CONTRACT_PERFORMANCE_EVIDENCE_RECEIPT_SCHEMA_VERSION,
    ContractPerformanceEvidenceReceipt,
    TREATMENT_STRATEGY,
)
from datadiff_osc.runtime._phase6_contract_performance_provenance import (
    CONTRACT_PERFORMANCE_PRODUCTION_PROVENANCE_ENVELOPE_TYPE,
    CONTRACT_PERFORMANCE_PRODUCTION_PROVENANCE_SCHEMA_VERSION,
    reconstruct_contract_performance_production_provenance_payload,
)
from datadiff_osc.runtime._phase6_contract_performance_execution_provenance import (
    CONTRACT_PERFORMANCE_EXECUTION_PROVENANCE_ENVELOPE_TYPE,
    CONTRACT_PERFORMANCE_EXECUTION_PROVENANCE_SCHEMA_VERSION,
    reconstruct_contract_performance_execution_provenance_payload,
)
from datadiff_osc.runtime._phase6_parallel_scaling_receipts import (
    PARALLEL_SCALING_EVIDENCE_RECEIPT_SCHEMA_VERSION,
    ParallelScalingEvidenceReceipt,
)
from datadiff_osc.runtime._phase6_parallel_scaling_provenance import (
    PARALLEL_SCALING_PRODUCTION_PROVENANCE_ENVELOPE_TYPE,
    PARALLEL_SCALING_PRODUCTION_PROVENANCE_SCHEMA_VERSION,
    reconstruct_parallel_scaling_production_provenance_payload,
)
from datadiff_osc.runtime._semantic_replay import (
    _reconstruct_contract_performance_evidence_receipt_payload,
    _reconstruct_parallel_scaling_evidence_receipt_payload,
    replay_runtime_admission,
)
from datadiff_osc.runtime.gate_artifacts import (
    GATE_AUTHORITY_PLAN_SCHEMA_VERSION,
    RAW_GATE_ARTIFACT_SCHEMA_VERSION,
    MetricReducer,
    RawMetricRecord,
    RawMetricSeries,
    VerifiedGateAuthorityPlan,
    identity_universe_digest,
    verify_gate_artifact_manifest,
    verify_gate_authority_plan,
)
from datadiff_osc.runtime.gates import (
    calculate_gate_report,
    default_pre24_gate_specs,
    pre24_gate_spec_set_digest,
)
from datadiff_osc.semantic_targets.compiler import compile_target_universe
from datadiff_osc.semantic_targets.model import (
    CompiledTargetUniverse,
    ProvenanceClass,
    TargetTemplate,
)
from datadiff_osc.search.semantic_replay import replay_search_admission


SOURCE_SNAPSHOT_SCHEMA_VERSION = "osc-root-source-snapshot-v1"
DYNAMIC_PLAN_SCHEMA_VERSION = "osc-root-phase6-dynamic-plan-v3"
ARTIFACT_RECEIPT_SCHEMA_VERSION = "osc-root-phase6-artifact-receipts-v1"
TYPED_PRODUCER_RECEIPT_SCHEMA_VERSION = (
    "osc-root-phase6-typed-producer-receipt-v1"
)
BUILD_RESULT_SCHEMA_VERSION = "osc-root-gate-build-result-v1"
PREFLIGHT_REPORT_SCHEMA_VERSION = "osc-root-gate-preflight-report-v1"
EXPECTED_PUBLIC_FREEZE_SHA256 = (
    "cb505dbe665e517e53314ddae8ba9302d05ac794d326e9be1e792f8bd4e22131"
)
EXPECTED_COMPILED_UNIVERSE_DIGEST = (
    "osc-compiled-target-universe-"
    "c822459944b38c1ae8abc6c271725bece0509457046c16e1088d455e2ac3982e"
)
EXPECTED_GATE_SPEC_SET_DIGEST = (
    "osc-pre24-gate-spec-set-"
    "d9792996cacb6fc6e47bce602d5f7ab6d6382c62ba9e1ac6bc1f8003ecbec552"
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_DIGEST_RE = re.compile(r"(?:[A-Za-z0-9_.:-]+-)?[0-9a-f]{64}")

_REQUIRED_SOURCE_PATHS = frozenset(
    {
        "pyproject.toml",
        "requirements-final.lock",
        "experiments/final_bug_discovery_protocol_v3/TODO.md",
        "src/datadiff/semantic_family_universe_v3.py",
        "src/datadiff/target_version_audit.py",
        "src/datadiff_osc/_canonical.py",
        "src/datadiff_osc/_phase6_gate_authority.py",
        "src/datadiff_osc/public_api_freeze.json",
        "src/datadiff_osc/schemas.py",
        "src/datadiff_osc/runtime/gate_artifacts.py",
        "src/datadiff_osc/runtime/gates.py",
        "src/datadiff_osc/semantic_targets/declarations.py",
        "src/datadiff_osc/semantic_targets/compiler.py",
        "src/datadiff_osc/semantic_targets/model.py",
        "src/datadiff_osc/semantic_targets/taxonomy.py",
        "src/datadiff_osc/generation/construction.py",
        "src/datadiff_osc/generation/extraction.py",
        "src/datadiff_osc/generation/mutation.py",
        "src/datadiff_osc/search/ledger.py",
        "src/datadiff_osc/contract_engine/mutations.py",
        "src/datadiff_osc/contract_engine/model.py",
        "src/datadiff_osc/contract_engine/planner.py",
        "src/datadiff_osc/contract_engine/applicability.py",
        "src/datadiff_osc/contract_engine/evidence.py",
        "src/datadiff_osc/parallel/invariance.py",
        "src/datadiff_osc/runtime/protocols.py",
        "scripts/osc/build_phase6_gate_authority_bundle.py",
        "scripts/osc/preflight_phase6_gate_authority_bundle.py",
    }
)


def _required_source_paths(repo_root: Path) -> frozenset[str]:
    """Return the independently discovered executable source surface."""

    root = repo_root.resolve()
    paths = set(_REQUIRED_SOURCE_PATHS)
    for relative_root in ("src/datadiff", "src/datadiff_osc", "scripts/osc"):
        directory = root / relative_root
        if not directory.is_dir():
            continue
        for candidate in directory.rglob("*"):
            if (
                not candidate.is_file()
                or "__pycache__" in candidate.parts
                or candidate.suffix in {".pyc", ".pyo"}
            ):
                continue
            paths.add(candidate.relative_to(root).as_posix())
    return frozenset(paths)

_DYNAMIC_RECORD_FIELDS: dict[str, tuple[frozenset[str], str]] = {
    "false_positive_fixes": (
        frozenset(
            {
                "fix_id",
                "case_id",
                "adjudication_digest",
                "expected_verdict",
                "producer_receipt_digest",
            }
        ),
        "fix_id",
    ),
    "contract_comparison_decisions": (
        frozenset(
            {
                "comparison_decision_id",
                "source_snapshot_digest",
                "source_case_digest",
                "result_group_id",
                "result_group_digest",
                "contract_fingerprint_digest",
                "endpoint_set_digest",
                "fingerprint_task_id",
                "fingerprint_result_digest",
                "exact_task_id",
                "exact_result_digest",
                "fingerprint_partition_digest",
                "canonical_partition_digest",
                "producer_receipt_digest",
            }
        ),
        "comparison_decision_id",
    ),
    "scheduled_target_attempts": (
        frozenset(
            {
                "attempt_id",
                "target_id",
                "family_id",
                "assignment_digest",
                "task_id",
                "seed_lineage_digest",
                "producer_receipt_digest",
            }
        ),
        "attempt_id",
    ),
    "mutation_attempts": (
        frozenset(
            {
                "attempt_id",
                "cell_id",
                "family_id",
                "assignment_digest",
                "operator_id",
                "operator_version",
                "task_id",
                "seed_lineage_digest",
                "producer_receipt_digest",
            }
        ),
        "attempt_id",
    ),
    "focus_signals": (
        frozenset(
            {
                "obligation_id",
                "lane_id",
                "signal_id",
                "planned_case_ids",
                "producer_receipt_digest",
            }
        ),
        "obligation_id",
    ),
    "formal_lanes": (
        frozenset(
            {
                "lane_id",
                "planned_case_ids",
                "producer_receipt_digest",
            }
        ),
        "lane_id",
    ),
    "escalation_faults": (
        frozenset(
            {
                "scenario_id",
                "fault_id",
                "fault_version",
                "request_digest",
                "task_id",
                "producer_receipt_digest",
            }
        ),
        "scenario_id",
    ),
    "v3_runs": (
        frozenset(
            {
                "run_id",
                "lane_id",
                "seed",
                "producer_receipt_digest",
            }
        ),
        "run_id",
    ),
    "v3_cases": (
        frozenset(
            {
                "case_id",
                "run_id",
                "case_index",
                "seed_lineage_digest",
                "producer_receipt_digest",
            }
        ),
        "case_id",
    ),
    "v3_executed_cases": (
        frozenset(
            {
                "case_id",
                "run_id",
                "task_id",
                "result_digest",
                "pipeline_digest",
                "execution_outcome_digest",
                "producer_receipt_digest",
            }
        ),
        "case_id",
    ),
    "v3_backend_tasks": (
        frozenset(
            {
                "task_id",
                "case_id",
                "endpoint_digest",
                "producer_receipt_digest",
            }
        ),
        "task_id",
    ),
    "campaign_rechecks": (
        frozenset(
            {
                "recheck_id",
                "candidate_id",
                "ordinal",
                "task_id",
                "seed_lineage_digest",
                "producer_receipt_digest",
            }
        ),
        "recheck_id",
    ),
    "pipeline_rechecks": (
        frozenset(
            {
                "recheck_id",
                "candidate_id",
                "ordinal",
                "task_id",
                "seed_lineage_digest",
                "producer_receipt_digest",
            }
        ),
        "recheck_id",
    ),
    "repository_tests": (
        frozenset(
            {
                "node_id",
                "collection_digest",
                "config_digest",
                "environment_digest",
                "producer_receipt_digest",
            }
        ),
        "node_id",
    ),
    "target_packages": (
        frozenset(
            {
                "package_id",
                "distribution_name",
                "import_name",
                "expected_version",
                "version_source_digest",
                "producer_receipt_digest",
            }
        ),
        "package_id",
    ),
    "candidate_records": (
        frozenset(
            {
                "finding_record_id",
                "source_case_id",
                "result_group_digest",
                "evidence_envelope_digest",
                "classification_digest",
                "bug_claimed",
                "producer_receipt_digest",
            }
        ),
        "finding_record_id",
    ),
    "contract_performance_samples": (
        frozenset(
            {
                "sample_id",
                "source_snapshot_digest",
                "source_case_digest",
                "comparison_decision_id",
                "contract_comparison_producer_receipt_digest",
                "result_group_id",
                "result_group_digest",
                "contract_fingerprint_digest",
                "endpoint_set_digest",
                "fingerprint_task_id",
                "fingerprint_result_digest",
                "exact_task_id",
                "exact_result_digest",
                "fingerprint_partition_digest",
                "canonical_partition_digest",
                "comparison_algorithm_digest",
                "benchmark_plan_digest",
                "environment_digest",
                "baseline_config_digest",
                "treatment_config_digest",
                "baseline_task_id",
                "baseline_task_spec_digest",
                "baseline_result_digest",
                "baseline_seed_lineage_digest",
                "baseline_execution_outcome_digest",
                "baseline_evidence_envelope_digest",
                "treatment_task_id",
                "treatment_task_spec_digest",
                "treatment_result_digest",
                "treatment_seed_lineage_digest",
                "treatment_execution_outcome_digest",
                "treatment_evidence_envelope_digest",
                "baseline_materialized_bytes",
                "treatment_materialized_bytes",
                "baseline_backend_pair_comparisons",
                "treatment_backend_pair_comparisons",
                "baseline_comparison_cpu_ns",
                "treatment_comparison_cpu_ns",
                "baseline_strategy",
                "treatment_strategy",
                "producer_receipt_digest",
            }
        ),
        "sample_id",
    ),
    "parallel_scaling_samples": (
        frozenset(
            {
                "sample_id",
                "workload_id",
                "one_worker_task_set_digest",
                "six_worker_task_set_digest",
                "one_worker_result_digest",
                "six_worker_result_digest",
                "producer_receipt_digest",
            }
        ),
        "sample_id",
    ),
}

_DYNAMIC_POSITIVE_INTEGER_FIELDS = frozenset(
    {
        "baseline_materialized_bytes",
        "baseline_backend_pair_comparisons",
        "baseline_comparison_cpu_ns",
    }
)
_DYNAMIC_NONNEGATIVE_INTEGER_FIELDS = frozenset(
    {
        "treatment_materialized_bytes",
        "treatment_backend_pair_comparisons",
        "treatment_comparison_cpu_ns",
    }
)

_DYNAMIC_UNIVERSE_KEYS = {
    "false_positive_fixes_total": "false_positive_fixes",
    "staged_exact_groups": "contract_comparison_decisions",
    "scheduled_targets_total": "scheduled_target_attempts",
    "mutations_attempted": "mutation_attempts",
    "focus_signals_total": "focus_signals",
    "formal_lanes": "formal_lanes",
    "exact_escalation_faults_injected": "escalation_faults",
    "v3_runs_planned": "v3_runs",
    "v3_cases_planned": "v3_cases",
    "v3_cases_executed": "v3_executed_cases",
    "v3_backend_results": "v3_backend_tasks",
    "campaign_rechecks_required": "campaign_rechecks",
    "pipeline_rechecks_required": "pipeline_rechecks",
    "repository_tests_run": "repository_tests",
    "latest_target_packages": "target_packages",
    "candidate_finding_records": "candidate_records",
}

_REQUIRED_PRODUCER_KINDS = frozenset(
    {
        "contract_exact_replay",
        "coverage_admission_replay",
        "reachability_admission_replay",
        "focus_admission_replay",
        "parallel_authority_replay",
        "v3_typed_run_replay",
        "repository_test_replay",
        "target_version_replay",
        "candidate_classification_replay",
        "performance_paired_replay",
    }
)

_DYNAMIC_SECTION_PRODUCER_KIND = {
    "false_positive_fixes": "contract_exact_replay",
    "contract_comparison_decisions": "contract_exact_replay",
    "scheduled_target_attempts": "reachability_admission_replay",
    "mutation_attempts": "reachability_admission_replay",
    "focus_signals": "focus_admission_replay",
    "formal_lanes": "focus_admission_replay",
    "escalation_faults": "parallel_authority_replay",
    "v3_runs": "v3_typed_run_replay",
    "v3_cases": "v3_typed_run_replay",
    "v3_executed_cases": "v3_typed_run_replay",
    "v3_backend_tasks": "v3_typed_run_replay",
    "campaign_rechecks": "candidate_classification_replay",
    "pipeline_rechecks": "candidate_classification_replay",
    "repository_tests": "repository_test_replay",
    "target_packages": "target_version_replay",
    "candidate_records": "candidate_classification_replay",
    "contract_performance_samples": "performance_paired_replay",
    "parallel_scaling_samples": "performance_paired_replay",
}

_DYNAMIC_SECTION_TYPED_SUBJECT_KIND = {
    "contract_comparison_decisions": "contract_comparison_decision_ids",
}

_EXPECTED_TARGET_PACKAGES = tuple(
    sorted(item.package for item in DEFAULT_TARGET_PACKAGES)
)

_ALLOWED_TYPED_ENVELOPES = {
    "contract_exact_replay": frozenset(
        {
            ("HyperContract", "osc-hypercontract-v1"),
            ("Endpoint", "osc-hypercontract-v1"),
            ("ResultGroup", "osc-result-group-v1"),
            ("StructuredExecutionOutcome", "osc-structured-execution-outcome-v1"),
            ("EvidenceEnvelope", "osc-evidence-envelope-v1"),
            ("ExactEscalationCertificate", "osc-exact-escalation-v1"),
            (
                "ContractComparisonReceipt",
                "osc-phase6-contract-comparison-receipt-v1",
            ),
        }
    ),
    "coverage_admission_replay": frozenset(
        {
            ("ConstructionOutcome", "osc-root-construction-outcome-v1"),
            ("TargetAssignment", "osc-root-target-assignment-v1"),
            ("ExtractionResult", "osc-atom-extraction-v1"),
            ("ContrastExtraction", "osc-contrast-extraction-v1"),
            ("ActivationCertificate", "osc-activation-certificate-v1"),
            ("ApplicabilityCertificate", "osc-applicability-certificate-v1"),
            ("ObservationCertificate", "osc-observation-certificate-v1"),
            ("StructuredExecutionOutcome", "osc-structured-execution-outcome-v1"),
            ("EvidenceEnvelope", "osc-evidence-envelope-v1"),
        }
    ),
    "reachability_admission_replay": frozenset(
        {
            ("ConstructionOutcome", "osc-root-construction-outcome-v1"),
            ("MutationOutcome", "osc-root-mutation-outcome-v1"),
            ("TargetAssignment", "osc-root-target-assignment-v1"),
            ("ExtractionResult", "osc-atom-extraction-v1"),
            ("ContrastExtraction", "osc-contrast-extraction-v1"),
            ("ActivationCertificate", "osc-activation-certificate-v1"),
            ("ObservationCertificate", "osc-observation-certificate-v1"),
            ("StructuredExecutionOutcome", "osc-structured-execution-outcome-v1"),
        }
    ),
    "focus_admission_replay": frozenset(
        {
            ("ActivationCertificate", "osc-activation-certificate-v1"),
            ("ObservationCertificate", "osc-observation-certificate-v1"),
            ("EvidenceEnvelope", "osc-evidence-envelope-v1"),
            ("FormalLanePlan", "osc-root-formal-lane-plan-v1"),
        }
    ),
    "parallel_authority_replay": frozenset(
        {
            ("InvarianceReport", "osc-parallel-invariance-report-v2"),
            ("AuthorityEvidenceSnapshot", "osc-authority-evidence-snapshot-v2"),
            ("TaskSpec", "osc-task-spec-v1"),
            ("StructuredExecutionOutcome", "osc-structured-execution-outcome-v1"),
            ("ExactEscalationCertificate", "osc-exact-escalation-v1"),
        }
    ),
    "v3_typed_run_replay": frozenset(
        {
            ("TaskSpec", "osc-task-spec-v1"),
            ("TaskIdentity", "osc-task-v1"),
            ("SeedLineage", "osc-seed-lineage-v1"),
            ("StructuredExecutionOutcome", "osc-structured-execution-outcome-v1"),
            ("EvidenceEnvelope", "osc-evidence-envelope-v1"),
            ("V3RunReceipt", "osc-root-v3-run-receipt-v1"),
        }
    ),
    "repository_test_replay": frozenset(
        {("RepositoryTestReceipt", "osc-root-repository-test-receipt-v1")}
    ),
    "target_version_replay": frozenset(
        {("TargetVersionReceipt", "osc-root-target-version-receipt-v1")}
    ),
    "candidate_classification_replay": frozenset(
        {
            ("EvidenceEnvelope", "osc-evidence-envelope-v1"),
            ("CandidateClassificationReceipt", "osc-root-candidate-classification-v1"),
            ("StructuredExecutionOutcome", "osc-structured-execution-outcome-v1"),
        }
    ),
    "performance_paired_replay": frozenset(
        {
            (
                "ContractPerformanceEvidenceReceipt",
                CONTRACT_PERFORMANCE_EVIDENCE_RECEIPT_SCHEMA_VERSION,
            ),
            (
                "ParallelScalingEvidenceReceipt",
                PARALLEL_SCALING_EVIDENCE_RECEIPT_SCHEMA_VERSION,
            ),
        }
    ),
}

_CONTRACT_REPLAY_TYPES = frozenset(
    {
        ("HyperContract", "osc-hypercontract-v1"),
        ("Endpoint", "osc-hypercontract-v1"),
        ("ResultGroup", "osc-result-group-v1"),
        ("ApplicabilityCertificate", "osc-applicability-certificate-v1"),
        ("ObservationCertificate", "osc-observation-certificate-v1"),
        ("ExactEscalationCertificate", "osc-exact-escalation-v1"),
        (
            "ContractComparisonReceipt",
            "osc-phase6-contract-comparison-receipt-v1",
        ),
    }
)
_SEARCH_REPLAY_TYPES = frozenset(
    {
        ("ConstructionOutcome", "osc-root-construction-outcome-v1"),
        ("MutationOutcome", "osc-root-mutation-outcome-v1"),
        ("TargetAssignment", "osc-root-target-assignment-v1"),
        ("ExtractionResult", "osc-atom-extraction-v1"),
        ("ContrastExtraction", "osc-contrast-extraction-v1"),
        ("ActivationCertificate", "osc-activation-certificate-v1"),
        ("FormalLanePlan", "osc-root-formal-lane-plan-v1"),
    }
)
_RUNTIME_REPLAY_TYPES = frozenset(
    {
        ("StructuredExecutionOutcome", "osc-structured-execution-outcome-v1"),
        ("EvidenceEnvelope", "osc-evidence-envelope-v1"),
        ("TaskSpec", "osc-task-spec-v1"),
        ("TaskIdentity", "osc-task-v1"),
        ("SeedLineage", "osc-seed-lineage-v1"),
        ("InvarianceReport", "osc-parallel-invariance-report-v2"),
        ("AuthorityEvidenceSnapshot", "osc-authority-evidence-snapshot-v2"),
        ("V3RunReceipt", "osc-root-v3-run-receipt-v1"),
        ("RepositoryTestReceipt", "osc-root-repository-test-receipt-v1"),
        ("TargetVersionReceipt", "osc-root-target-version-receipt-v1"),
        (
            "CandidateClassificationReceipt",
            "osc-root-candidate-classification-v1",
        ),
        (
            "ContractPerformanceEvidenceReceipt",
            CONTRACT_PERFORMANCE_EVIDENCE_RECEIPT_SCHEMA_VERSION,
        ),
        (
            "ParallelScalingEvidenceReceipt",
            PARALLEL_SCALING_EVIDENCE_RECEIPT_SCHEMA_VERSION,
        ),
    }
)
_TYPED_SEMANTIC_REPLAY_DOMAINS = (
    ("contract", _CONTRACT_REPLAY_TYPES, replay_contract_admission),
    ("search", _SEARCH_REPLAY_TYPES, replay_search_admission),
    ("runtime", _RUNTIME_REPLAY_TYPES, replay_runtime_admission),
)
_TYPED_SEMANTIC_REPLAYERS = {}
for _domain_name, _domain_types, _domain_replayer in _TYPED_SEMANTIC_REPLAY_DOMAINS:
    for _type_key in _domain_types:
        if _type_key in _TYPED_SEMANTIC_REPLAYERS:
            raise RuntimeError(f"duplicate typed semantic replay owner: {_type_key}")
        _TYPED_SEMANTIC_REPLAYERS[_type_key] = (
            _domain_name,
            _domain_replayer,
        )
_ALLOWED_TYPED_ENVELOPE_PAIRS = frozenset(
    pair for pairs in _ALLOWED_TYPED_ENVELOPES.values() for pair in pairs
)
if frozenset(_TYPED_SEMANTIC_REPLAYERS) != _ALLOWED_TYPED_ENVELOPE_PAIRS:
    raise RuntimeError("typed semantic replay ownership does not match allowed types")

_ROOT_CONTEXT_PENDING_SUBJECTS = {
    # The legacy ResultGroup-only staged-parity subject has no full canonical
    # receipt chain. R2 binds the stronger private comparison receipt instead;
    # this legacy subject remains fail-closed pending CR-OSC-6-002.
    "staged_parity_groups": "CR-OSC-6-002",
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _strict_json_loads(raw: bytes) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field: {key}")
            result[key] = value
        return result

    return json.loads(raw, object_pairs_hook=reject_duplicates)


def _safe_relative(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("source paths must be non-empty relative strings")
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("source paths must remain inside the declared root")
    normalized = candidate.as_posix()
    if normalized in {"", "."}:
        raise ValueError("source path is empty")
    return normalized


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a SHA-256 or namespaced stable digest")
    return value


def _current_git_state(repo_root: Path) -> tuple[str, str] | None:
    try:
        head = subprocess.run(
            ("git", "-C", str(repo_root), "rev-parse", "HEAD"),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.decode().strip()
        status = subprocess.run(
            (
                "git",
                "-C",
                str(repo_root),
                "status",
                "--porcelain=v1",
                "-z",
                "--untracked-files=all",
                "--",
                "pyproject.toml",
                "requirements-final.lock",
                "experiments/final_bug_discovery_protocol_v3/TODO.md",
                "src/datadiff",
                "src/datadiff_osc",
                "scripts/osc",
            ),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except (OSError, subprocess.SubprocessError, UnicodeDecodeError):
        return None
    if not head:
        return None
    return head, stable_digest("osc-git-status-porcelain-v1", status)


@dataclass(frozen=True, slots=True)
class SourceFileBinding:
    relative_path: str
    sha256: str

    def __post_init__(self) -> None:
        _safe_relative(self.relative_path)
        _require_sha256(self.sha256, "source file SHA-256")


@dataclass(frozen=True, slots=True)
class VerifiedSourceSnapshot:
    repo_root: str
    snapshot_path: str
    expected_snapshot_sha256: str
    byte_sha256: str
    source_digest: str
    git_head: str
    git_status_digest: str
    public_api_freeze_sha256: str
    dependency_lock_sha256: str
    protocol_digest: str
    file_bindings: tuple[SourceFileBinding, ...]
    verification_errors: tuple[str, ...] = ()
    schema_version: str = SOURCE_SNAPSHOT_SCHEMA_VERSION

    @property
    def valid(self) -> bool:
        if self.verification_errors:
            return False
        return verify_source_snapshot(
            repo_root=Path(self.repo_root),
            snapshot_path=Path(self.snapshot_path),
            expected_snapshot_sha256=self.expected_snapshot_sha256,
        ) == self

    @property
    def digest(self) -> str:
        return stable_digest(
            "osc-verified-source-snapshot",
            {
                "source_digest": self.source_digest,
                "byte_sha256": self.byte_sha256,
                "files": self.file_bindings,
            },
        )


def source_snapshot_payload(
    *,
    repo_root: Path,
    relative_paths: Iterable[str],
    git_head: str,
    git_status_digest: str,
) -> dict[str, Any]:
    """Capture source bytes into a payload; callers still freeze its external SHA."""

    root = repo_root.resolve()
    git_state = _current_git_state(root)
    if git_state is None:
        raise ValueError("source snapshot requires a readable Git worktree")
    actual_head, actual_status_digest = git_state
    if git_head != actual_head:
        raise ValueError("git_head does not match the current worktree")
    if git_status_digest != actual_status_digest:
        raise ValueError("git_status_digest does not match the current worktree")
    paths = tuple(sorted({_safe_relative(item) for item in relative_paths}))
    required_paths = _required_source_paths(root)
    if not required_paths <= set(paths):
        missing = sorted(required_paths - set(paths))
        raise ValueError("source snapshot paths missing: " + ",".join(missing))
    bindings = tuple(
        SourceFileBinding(item, _sha256_path(root / item)) for item in paths
    )
    public_hash = dict(
        (item.relative_path, item.sha256) for item in bindings
    )["src/datadiff_osc/public_api_freeze.json"]
    dependency_hash = dict(
        (item.relative_path, item.sha256) for item in bindings
    )["requirements-final.lock"]
    protocol_hash = dict(
        (item.relative_path, item.sha256) for item in bindings
    )["experiments/final_bug_discovery_protocol_v3/TODO.md"]
    body = {
        "schema_version": SOURCE_SNAPSHOT_SCHEMA_VERSION,
        "git_head": _string(git_head, "git_head"),
        "git_status_digest": _string(git_status_digest, "git_status_digest"),
        "public_api_freeze_sha256": public_hash,
        "dependency_lock_sha256": dependency_hash,
        "protocol_digest": protocol_hash,
        "files": [
            {"path": item.relative_path, "sha256": item.sha256}
            for item in bindings
        ],
    }
    body["source_digest"] = stable_digest("osc-phase6-source-snapshot", body)
    return body


def verify_source_snapshot(
    *,
    repo_root: Path,
    snapshot_path: Path,
    expected_snapshot_sha256: str,
) -> VerifiedSourceSnapshot:
    _require_sha256(expected_snapshot_sha256, "expected source snapshot SHA-256")
    root = repo_root.resolve()
    resolved = snapshot_path.resolve()
    errors: list[str] = []
    try:
        raw = resolved.read_bytes()
    except OSError as exc:
        raw = b""
        errors.append(f"source_snapshot_load_failed:{type(exc).__name__}:{exc}")
    byte_sha = _sha256_bytes(raw)
    if byte_sha != expected_snapshot_sha256:
        errors.append("source_snapshot_hash_mismatch")
    try:
        payload = _strict_json_loads(raw) if raw else None
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        payload = None
        errors.append(f"source_snapshot_invalid_json:{exc}")

    source_digest = "unverified-source"
    git_head = "unverified-git-head"
    git_status_digest = "unverified-git-status"
    public_hash = "0" * 64
    dependency_hash = "0" * 64
    protocol_digest = "0" * 64
    bindings: list[SourceFileBinding] = []
    expected_fields = {
        "schema_version",
        "source_digest",
        "git_head",
        "git_status_digest",
        "public_api_freeze_sha256",
        "dependency_lock_sha256",
        "protocol_digest",
        "files",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        errors.append("source_snapshot_schema_mismatch")
    else:
        if payload["schema_version"] != SOURCE_SNAPSHOT_SCHEMA_VERSION:
            errors.append("source_snapshot_schema_version_mismatch")
        try:
            source_digest = _string(payload["source_digest"], "source_digest")
            git_head = _string(payload["git_head"], "git_head")
            git_status_digest = _string(
                payload["git_status_digest"], "git_status_digest"
            )
            public_hash = _require_sha256(
                payload["public_api_freeze_sha256"], "public freeze SHA-256"
            )
            dependency_hash = _require_sha256(
                payload["dependency_lock_sha256"], "dependency lock SHA-256"
            )
            protocol_digest = _require_sha256(
                payload["protocol_digest"], "protocol digest"
            )
        except ValueError as exc:
            errors.append(f"source_snapshot_invalid_binding:{exc}")
        values = payload["files"]
        if not isinstance(values, list) or not values:
            errors.append("source_snapshot_files_missing")
        else:
            for index, item in enumerate(values):
                try:
                    if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
                        raise ValueError("fields do not match schema")
                    binding = SourceFileBinding(
                        _safe_relative(item["path"]),
                        _require_sha256(item["sha256"], "source file SHA-256"),
                    )
                    bindings.append(binding)
                except (TypeError, ValueError) as exc:
                    errors.append(f"source_snapshot_file[{index}]:invalid:{exc}")

        paths = tuple(item.relative_path for item in bindings)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            errors.append("source_snapshot_files_not_unique_sorted")
        missing = sorted(_required_source_paths(root) - set(paths))
        if missing:
            errors.append("source_snapshot_required_files_missing:" + ",".join(missing))
        by_path = {item.relative_path: item.sha256 for item in bindings}
        if by_path.get("src/datadiff_osc/public_api_freeze.json") != public_hash:
            errors.append("source_snapshot_public_freeze_binding_mismatch")
        if public_hash != EXPECTED_PUBLIC_FREEZE_SHA256:
            errors.append("source_snapshot_public_freeze_not_frozen")
        if by_path.get("requirements-final.lock") != dependency_hash:
            errors.append("source_snapshot_dependency_binding_mismatch")
        if (
            by_path.get("experiments/final_bug_discovery_protocol_v3/TODO.md")
            != protocol_digest
        ):
            errors.append("source_snapshot_protocol_binding_mismatch")
        for binding in bindings:
            candidate = (root / binding.relative_path).resolve()
            try:
                candidate.relative_to(root)
            except ValueError:
                errors.append(
                    f"source_snapshot_path_escape:{binding.relative_path}"
                )
                continue
            try:
                actual = _sha256_path(candidate)
            except OSError as exc:
                errors.append(
                    f"source_snapshot_file_load_failed:{binding.relative_path}:"
                    f"{type(exc).__name__}:{exc}"
                )
                continue
            if actual != binding.sha256:
                errors.append(f"source_snapshot_file_hash_mismatch:{binding.relative_path}")

        git_state = _current_git_state(root)
        if git_state is None:
            errors.append("source_snapshot_git_state_unavailable")
        else:
            actual_head, actual_status_digest = git_state
            if git_head != actual_head:
                errors.append("source_snapshot_git_head_mismatch")
            if git_status_digest != actual_status_digest:
                errors.append("source_snapshot_git_status_mismatch")

        body = dict(payload)
        body.pop("source_digest", None)
        recomputed = stable_digest("osc-phase6-source-snapshot", body)
        if source_digest != recomputed:
            errors.append("source_snapshot_digest_mismatch")

    return VerifiedSourceSnapshot(
        repo_root=str(root),
        snapshot_path=str(resolved),
        expected_snapshot_sha256=expected_snapshot_sha256,
        byte_sha256=byte_sha,
        source_digest=source_digest,
        git_head=git_head,
        git_status_digest=git_status_digest,
        public_api_freeze_sha256=public_hash,
        dependency_lock_sha256=dependency_hash,
        protocol_digest=protocol_digest,
        file_bindings=tuple(bindings),
        verification_errors=tuple(dict.fromkeys(errors)),
    )


@dataclass(frozen=True, slots=True)
class IdentityUniverse:
    metric_key: str
    record_ids: tuple[str, ...]
    provenance_digest: str

    def __post_init__(self) -> None:
        _string(self.metric_key, "metric_key")
        _string(self.provenance_digest, "provenance_digest")
        if any(not isinstance(item, str) or not item for item in self.record_ids):
            raise ValueError("identity universe values must be non-empty strings")
        if tuple(sorted(self.record_ids)) != self.record_ids:
            raise ValueError("identity universe values must be sorted")
        if len(self.record_ids) != len(set(self.record_ids)):
            raise ValueError("identity universe values must be unique")

    @property
    def identity_digest(self) -> str:
        return identity_universe_digest(self.record_ids)

    def authority_payload(self) -> dict[str, Any]:
        return {
            "metric_key": self.metric_key,
            "cardinality": len(self.record_ids),
            "identity_digest": self.identity_digest,
            "provenance_digest": self.provenance_digest,
        }


@dataclass(frozen=True, slots=True)
class StaticGateUniverses:
    source_digest: str
    compiled_universe: CompiledTargetUniverse
    universes: tuple[IdentityUniverse, ...]
    verification_errors: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.verification_errors

    @property
    def digest(self) -> str:
        return stable_digest(
            "osc-root-static-gate-universes",
            {
                "source_digest": self.source_digest,
                "compiled_universe_digest": self.compiled_universe.digest,
                "universes": self.universes,
            },
        )

    def universe_map(self) -> dict[str, IdentityUniverse]:
        return {item.metric_key: item for item in self.universes}


def _universe(
    metric_key: str,
    record_ids: Iterable[str],
    *,
    source_digest: str,
    producer_digest: str,
) -> IdentityUniverse:
    identities = tuple(sorted(tuple(record_ids)))
    return IdentityUniverse(
        metric_key,
        identities,
        stable_digest(
            "osc-root-gate-universe-provenance",
            {
                "metric_key": metric_key,
                "source_digest": source_digest,
                "producer_digest": producer_digest,
                "record_ids": identities,
            },
        ),
    )


def derive_static_gate_universes(
    *,
    source: VerifiedSourceSnapshot,
    templates: tuple[TargetTemplate, ...],
) -> StaticGateUniverses:
    errors: list[str] = []
    source_digest = (
        source.source_digest
        if isinstance(source, VerifiedSourceSnapshot)
        else "unverified-source"
    )
    source_valid = False
    if isinstance(source, VerifiedSourceSnapshot):
        try:
            source_valid = source.valid
        except Exception as exc:
            errors.append(
                f"source_snapshot_reverification_failed:{type(exc).__name__}:{exc}"
            )
    if not source_valid:
        errors.append("source_snapshot_not_verified")
    compiled = compile_target_universe(templates)
    if compiled.digest != EXPECTED_COMPILED_UNIVERSE_DIGEST:
        errors.append("compiled_target_universe_digest_mismatch")
    if pre24_gate_spec_set_digest() != EXPECTED_GATE_SPEC_SET_DIGEST:
        errors.append("frozen_gate_spec_set_digest_mismatch")

    fresh_family_ids = tuple(
        sorted({item.test_family_id for item in compiled.fresh_cells})
    )
    regression_family_ids = tuple(
        sorted({item.test_family_id for item in compiled.regression_cells})
    )
    fresh_cell_ids = tuple(item.target_cell_id for item in compiled.fresh_cells)
    regression_cell_ids = tuple(
        item.target_cell_id for item in compiled.regression_cells
    )
    edge_ids = tuple(item.contrast_edge_id for item in compiled.fresh_edges)
    pair_ids = tuple(
        item.obligation_id for item in compiled.fresh_backend_pair_obligations
    )
    regression_root_ids = tuple(
        sorted(
            {
                item.provenance_evidence_id
                for item in templates
                if item.provenance_class is ProvenanceClass.KNOWN_REGRESSION
            }
        )
    )
    pipeline_specs = pipeline_evidence_specs()
    operation_ids = tuple(
        sorted({value for item in pipeline_specs for value in item.required_operations})
    )
    expression_ids = tuple(
        sorted({value for item in pipeline_specs for value in item.required_expressions})
    )
    aggregate_ids = tuple(
        sorted({value for item in pipeline_specs for value in item.required_aggregates})
    )
    risk_class_ids = tuple(sorted({item.risk_class for item in pipeline_specs}))
    risk_pipeline_ids = tuple(sorted(item.pipeline_id for item in pipeline_specs))
    mutant_ids = tuple(
        sorted(
            {
                *(f"comparator:{item}" for item in COMPARATOR_WEAKENING_MUTANTS),
                *(f"axis:{item}" for item in AXIS_FAULTS),
                *(f"hyperedge:{item}" for item in HYPEREDGE_MUTANTS),
            }
        )
    )
    invariant_ids = tuple(
        sorted(
            f"parallel:{item.name}"
            for item in fields(InvarianceReport)
            if item.name
            in {
                "assignment_invariant",
                "epoch_invariant",
                "task_multiset_invariant",
                "task_set_invariant",
                "seed_lineage_invariant",
                "result_order_invariant",
                "outcome_invariant",
                "retry_trace_invariant",
                "authority_verdict_invariant",
                "coverage_bitmap_invariant",
                "ledger_invariant",
                "certificate_invariant",
                "authority_evidence_complete",
            }
        )
    )
    observed_counts = (
        len(fresh_family_ids),
        len(fresh_cell_ids),
        len(regression_family_ids),
        len(regression_cell_ids),
        len(edge_ids),
        len(pair_ids),
        len(operation_ids),
        len(expression_ids),
        len(aggregate_ids),
        len(risk_class_ids),
        len(risk_pipeline_ids),
        len(regression_root_ids),
        len(mutant_ids),
        len(invariant_ids),
    )
    expected_counts = (16, 232, 9, 144, 384, 502, 21, 21, 8, 7, 18, 9, 32, 13)
    if observed_counts != expected_counts:
        errors.append(
            "static_gate_universe_count_mismatch:"
            + ",".join(map(str, observed_counts))
        )

    producer = compiled.digest
    groups = {
        "aggregates_declared": aggregate_ids,
        "backend_pair_obligations_declared": pair_ids,
        "cells_activation_denominator": fresh_cell_ids,
        "confirmed_roots_total": regression_root_ids,
        "contrast_edges_declared": edge_ids,
        "edge_observation_denominator": edge_ids,
        "edges_activation_denominator": edge_ids,
        "expressions_declared": expression_ids,
        "fresh_cells_declared": fresh_cell_ids,
        "fresh_cells_reachability_denominator": fresh_cell_ids,
        "fresh_families_declared": fresh_family_ids,
        "fresh_family_reachability_denominator": fresh_family_ids,
        "high_risk_mutants_total": mutant_ids,
        "mutation_family_denominator": fresh_family_ids,
        "operations_declared": operation_ids,
        "parallel_invariance_checks_total": invariant_ids,
        "regression_cells_declared": regression_cell_ids,
        "regression_families_declared": regression_family_ids,
        "risk_classes_declared": risk_class_ids,
        "risk_pipelines_declared": risk_pipeline_ids,
        "scheduled_activation_family_denominator": fresh_family_ids,
    }
    universes = tuple(
        _universe(
            key,
            values,
            source_digest=source_digest,
            producer_digest=producer,
        )
        for key, values in sorted(groups.items())
    )
    return StaticGateUniverses(
        source_digest,
        compiled,
        universes,
        tuple(dict.fromkeys(errors)),
    )


@dataclass(frozen=True, slots=True)
class VerifiedDynamicGatePlan:
    plan_path: str
    expected_plan_sha256: str
    byte_sha256: str
    expected_source_digest: str
    source_digest: str
    producer_receipts_digest: str
    section_identities: tuple[tuple[str, tuple[str, ...]], ...]
    section_records: tuple[tuple[str, tuple[str, ...]], ...]
    verification_errors: tuple[str, ...] = ()
    schema_version: str = DYNAMIC_PLAN_SCHEMA_VERSION

    @property
    def valid(self) -> bool:
        if self.verification_errors:
            return False
        return verify_dynamic_gate_plan(
            plan_path=Path(self.plan_path),
            expected_plan_sha256=self.expected_plan_sha256,
            expected_source_digest=self.expected_source_digest,
        ) == self

    @property
    def digest(self) -> str:
        return stable_digest(
            "osc-verified-dynamic-gate-plan",
            {
                "byte_sha256": self.byte_sha256,
                "source_digest": self.source_digest,
                "sections": self.section_identities,
                "records": self.section_records,
            },
        )

    def identity_map(self) -> dict[str, tuple[str, ...]]:
        return dict(self.section_identities)

    def record_map(self) -> dict[str, tuple[dict[str, Any], ...]]:
        return {
            section: tuple(_strict_json_loads(item.encode()) for item in records)
            for section, records in self.section_records
        }


def _validate_dynamic_records(
    section: str,
    value: object,
    errors: list[str],
) -> tuple[str, ...]:
    expected_fields, identity_field = _DYNAMIC_RECORD_FIELDS[section]
    if not isinstance(value, list):
        errors.append(f"dynamic_plan_section_not_list:{section}")
        return ()
    identities: list[str] = []
    for index, record in enumerate(value):
        if not isinstance(record, dict) or set(record) != expected_fields:
            errors.append(f"dynamic_plan_record_schema:{section}:{index}")
            continue
        for name, item in record.items():
            if name in _DYNAMIC_POSITIVE_INTEGER_FIELDS:
                if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
                    errors.append(
                        f"dynamic_plan_record_positive_integer:{section}:{index}:{name}"
                    )
            elif name in _DYNAMIC_NONNEGATIVE_INTEGER_FIELDS:
                if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                    errors.append(
                        f"dynamic_plan_record_nonnegative_integer:{section}:{index}:{name}"
                    )
            elif name in {"seed", "case_index", "ordinal"}:
                if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                    errors.append(
                        f"dynamic_plan_record_integer:{section}:{index}:{name}"
                    )
            elif name == "bug_claimed":
                if not isinstance(item, bool):
                    errors.append(
                        f"dynamic_plan_record_boolean:{section}:{index}:{name}"
                    )
            elif name == "planned_case_ids":
                if (
                    not isinstance(item, list)
                    or not item
                    or any(not isinstance(case, str) or not case for case in item)
                    or len(item) != len(set(item))
                ):
                    errors.append(
                        f"dynamic_plan_record_case_ids:{section}:{index}"
                    )
            elif name.endswith("_digest"):
                try:
                    _digest(item, name)
                except ValueError:
                    errors.append(
                        f"dynamic_plan_record_digest:{section}:{index}:{name}"
                    )
            elif not isinstance(item, str) or not item:
                errors.append(f"dynamic_plan_record_text:{section}:{index}:{name}")
        identity = record.get(identity_field)
        if isinstance(identity, str) and identity:
            identities.append(identity)
        if section == "candidate_records" and record.get("bug_claimed") is not False:
            errors.append(f"candidate_record_premature_bug_claim:{index}")
    if len(identities) != len(set(identities)):
        errors.append(f"dynamic_plan_duplicate_identity:{section}")
    if (
        section in {"contract_comparison_decisions", "contract_performance_samples"}
        and identities != sorted(identities)
    ):
        errors.append(
            "dynamic_plan_record_identities_not_sorted:"
            f"{section}"
        )
    return tuple(sorted(set(identities)))


def verify_dynamic_gate_plan(
    *,
    plan_path: Path,
    expected_plan_sha256: str,
    expected_source_digest: str,
) -> VerifiedDynamicGatePlan:
    _require_sha256(expected_plan_sha256, "expected dynamic plan SHA-256")
    _string(expected_source_digest, "expected source digest")
    resolved = plan_path.resolve()
    errors: list[str] = []
    try:
        raw = resolved.read_bytes()
    except OSError as exc:
        raw = b""
        errors.append(f"dynamic_plan_load_failed:{type(exc).__name__}:{exc}")
    byte_sha = _sha256_bytes(raw)
    if byte_sha != expected_plan_sha256:
        errors.append("dynamic_plan_hash_mismatch")
    try:
        payload = _strict_json_loads(raw) if raw else None
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        payload = None
        errors.append(f"dynamic_plan_invalid_json:{exc}")
    source_digest = expected_source_digest
    receipts_digest = "unverified-producer-receipts"
    sections: list[tuple[str, tuple[str, ...]]] = []
    serialized_records: list[tuple[str, tuple[str, ...]]] = []
    expected_fields = {
        "schema_version",
        "source_digest",
        "producer_receipts_digest",
        *_DYNAMIC_RECORD_FIELDS,
    }
    records_by_section: dict[str, list[dict[str, Any]]] = {}
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        errors.append("dynamic_plan_schema_mismatch")
    else:
        if payload["schema_version"] != DYNAMIC_PLAN_SCHEMA_VERSION:
            errors.append("dynamic_plan_schema_version_mismatch")
        try:
            source_digest = _string(payload["source_digest"], "source_digest")
            receipts_digest = _string(
                payload["producer_receipts_digest"], "producer_receipts_digest"
            )
            _digest(receipts_digest, "producer_receipts_digest")
        except ValueError as exc:
            errors.append(f"dynamic_plan_invalid_binding:{exc}")
        if source_digest != expected_source_digest:
            errors.append("dynamic_plan_source_mismatch")
        for section in _DYNAMIC_RECORD_FIELDS:
            value = payload[section]
            identities = _validate_dynamic_records(section, value, errors)
            sections.append((section, identities))
            records_by_section[section] = value if isinstance(value, list) else []
            serialized_records.append(
                (
                    section,
                    tuple(
                        canonical_json(record)
                        for record in value
                        if isinstance(record, dict)
                    ),
                )
            )

        if len(dict(sections)["contract_comparison_decisions"]) < 100_000:
            errors.append("contract_comparison_decision_floor_not_met")
        if len(dict(sections)["formal_lanes"]) != 11:
            errors.append("formal_lane_plan_not_exactly_11")
        if not dict(sections)["false_positive_fixes"]:
            errors.append("false_positive_fix_plan_missing")
        if not dict(sections)["scheduled_target_attempts"]:
            errors.append("scheduled_target_plan_missing")
        if not dict(sections)["mutation_attempts"]:
            errors.append("mutation_attempt_plan_missing")
        if not dict(sections)["focus_signals"]:
            errors.append("focus_signal_plan_missing")
        if not dict(sections)["escalation_faults"]:
            errors.append("escalation_fault_plan_missing")
        if not dict(sections)["repository_tests"]:
            errors.append("repository_test_plan_missing")
        if len(dict(sections)["target_packages"]) != 7:
            errors.append("target_package_plan_not_exactly_7")
        package_ids = tuple(
            sorted(
                record.get("package_id")
                for record in records_by_section["target_packages"]
                if isinstance(record.get("package_id"), str)
            )
        )
        if package_ids != _EXPECTED_TARGET_PACKAGES:
            errors.append("target_package_identity_set_mismatch")

        performance_records = records_by_section["contract_performance_samples"]
        if len(dict(sections)["contract_performance_samples"]) < 1_000:
            errors.append("contract_performance_sample_floor_not_met")
        if any(
            record.get("baseline_task_id") == record.get("treatment_task_id")
            for record in performance_records
        ):
            errors.append("contract_performance_pair_not_distinct")
        if any(
            record.get("baseline_config_digest")
            == record.get("treatment_config_digest")
            for record in performance_records
        ):
            errors.append("contract_performance_config_not_distinct")
        if any(
            record.get("baseline_result_digest")
            == record.get("treatment_result_digest")
            for record in performance_records
        ):
            errors.append("contract_performance_result_not_distinct")
        if any(
            record.get("baseline_seed_lineage_digest")
            != record.get("treatment_seed_lineage_digest")
            for record in performance_records
        ):
            errors.append("contract_performance_seed_lineage_mismatch")
        if any(
            record.get("baseline_strategy") != BASELINE_STRATEGY
            or record.get("treatment_strategy") != TREATMENT_STRATEGY
            for record in performance_records
        ):
            errors.append("contract_performance_strategy_mismatch")
        scaling_records = records_by_section["parallel_scaling_samples"]
        if not dict(sections)["parallel_scaling_samples"]:
            errors.append("parallel_scaling_sample_plan_missing")
        if any(
            record.get("one_worker_task_set_digest")
            != record.get("six_worker_task_set_digest")
            for record in scaling_records
        ):
            errors.append("parallel_scaling_task_set_mismatch")

        run_ids = set(dict(sections)["v3_runs"])
        case_ids = set(dict(sections)["v3_cases"])
        executed_case_ids = set(dict(sections)["v3_executed_cases"])
        if len(run_ids) != 22:
            errors.append("v3_run_plan_not_exactly_22")
        if len(case_ids) != 2200:
            errors.append("v3_case_plan_not_exactly_2200")
        if executed_case_ids != case_ids:
            errors.append("v3_executed_case_set_not_exactly_planned_cases")
        lane_ids = set(dict(sections)["formal_lanes"])
        run_records = records_by_section["v3_runs"]
        runs_by_lane: dict[str, list[dict[str, Any]]] = {}
        seeds_to_lanes: dict[int, set[str]] = {}
        for record in run_records:
            lane = record.get("lane_id")
            seed = record.get("seed")
            if isinstance(lane, str):
                runs_by_lane.setdefault(lane, []).append(record)
            if isinstance(lane, str) and isinstance(seed, int):
                seeds_to_lanes.setdefault(seed, set()).add(lane)
        if set(runs_by_lane) != lane_ids or any(
            len(records) != 2 for records in runs_by_lane.values()
        ):
            errors.append("v3_run_plan_not_11_lanes_by_2_runs")
        if len(seeds_to_lanes) != 2 or any(
            lanes != lane_ids for lanes in seeds_to_lanes.values()
        ):
            errors.append("v3_seed_plan_not_two_blocks_across_all_lanes")
        case_records = records_by_section["v3_cases"]
        if any(record.get("run_id") not in run_ids for record in case_records):
            errors.append("v3_case_references_unknown_run")
        if any(
            record.get("run_id") not in run_ids
            for record in records_by_section["v3_executed_cases"]
        ):
            errors.append("v3_executed_case_references_unknown_run")
        cases_by_run: dict[str, list[dict[str, Any]]] = {}
        for record in case_records:
            run_id = record.get("run_id")
            if isinstance(run_id, str):
                cases_by_run.setdefault(run_id, []).append(record)
        if set(cases_by_run) != run_ids or any(
            len(records) != 100
            or {record.get("case_index") for record in records} != set(range(100))
            for records in cases_by_run.values()
        ):
            errors.append("v3_case_plan_not_100_unique_indices_per_run")
        run_lane = {
            record.get("run_id"): record.get("lane_id")
            for record in run_records
            if isinstance(record.get("run_id"), str)
        }
        cases_by_lane: dict[str, set[str]] = {lane: set() for lane in lane_ids}
        case_lane: dict[str, str] = {}
        for record in case_records:
            case_id = record.get("case_id")
            lane = run_lane.get(record.get("run_id"))
            if isinstance(case_id, str) and isinstance(lane, str):
                cases_by_lane.setdefault(lane, set()).add(case_id)
                case_lane[case_id] = lane
        formal_records = records_by_section["formal_lanes"]
        for record in formal_records:
            lane = record.get("lane_id")
            planned = record.get("planned_case_ids")
            if isinstance(lane, str) and isinstance(planned, list):
                if set(planned) != cases_by_lane.get(lane, set()):
                    errors.append(f"formal_lane_case_plan_mismatch:{lane}")
        for record in records_by_section["focus_signals"]:
            lane = record.get("lane_id")
            planned = record.get("planned_case_ids")
            if not isinstance(planned, list) or len(planned) < 5:
                errors.append("focus_signal_case_floor_not_met")
                continue
            if not isinstance(lane, str) or any(
                case_lane.get(case_id) != lane for case_id in planned
            ):
                errors.append("focus_signal_case_lane_mismatch")
        backend_records = records_by_section["v3_backend_tasks"]
        if any(record.get("case_id") not in case_ids for record in backend_records):
            errors.append("v3_backend_task_references_unknown_case")
        if case_ids and {
            record.get("case_id") for record in backend_records
        } != case_ids:
            errors.append("v3_backend_task_case_coverage_incomplete")
        backend_endpoints: dict[str, set[str]] = {}
        for record in backend_records:
            case_id = record.get("case_id")
            endpoint = record.get("endpoint_digest")
            if isinstance(case_id, str) and isinstance(endpoint, str):
                if endpoint in backend_endpoints.setdefault(case_id, set()):
                    errors.append(f"v3_backend_endpoint_duplicate:{case_id}")
                backend_endpoints[case_id].add(endpoint)

        candidate_ids = set(dict(sections)["candidate_records"])
        for section in ("campaign_rechecks", "pipeline_rechecks"):
            grouped: dict[str, set[int]] = {}
            for record in records_by_section[section]:
                candidate = record.get("candidate_id")
                ordinal = record.get("ordinal")
                if isinstance(candidate, str) and isinstance(ordinal, int):
                    grouped.setdefault(candidate, set()).add(ordinal)
            if set(grouped) != candidate_ids:
                errors.append(f"{section}_candidate_set_mismatch")
            if any(values != {1, 2, 3} for values in grouped.values()):
                errors.append(f"{section}_ordinals_not_exactly_1_2_3")
            counts: dict[str, int] = {}
            for record in records_by_section[section]:
                candidate = record.get("candidate_id")
                if isinstance(candidate, str):
                    counts[candidate] = counts.get(candidate, 0) + 1
            if any(count != 3 for count in counts.values()):
                errors.append(f"{section}_record_count_not_exactly_3")

    return VerifiedDynamicGatePlan(
        plan_path=str(resolved),
        expected_plan_sha256=expected_plan_sha256,
        byte_sha256=byte_sha,
        expected_source_digest=expected_source_digest,
        source_digest=source_digest,
        producer_receipts_digest=receipts_digest,
        section_identities=tuple(sorted(sections)),
        section_records=tuple(sorted(serialized_records)),
        verification_errors=tuple(dict.fromkeys(errors)),
    )


@dataclass(frozen=True, slots=True)
class VerifiedTypedAdmission:
    admission_id: str
    subject_kind: str
    subject_ids: tuple[str, ...]
    envelope_path: str
    byte_sha256: str
    envelope_type: str
    envelope_schema_version: str
    envelope_digest: str


@dataclass(frozen=True, slots=True)
class VerifiedParallelScalingProvenance:
    """Root's compact, re-verified binding for one parallel admission."""

    admission_id: str
    provenance_path: str
    byte_sha256: str
    provenance_digest: str
    provenance_id: str
    receipt_digest: str
    sample_id: str
    source_snapshot_digest: str
    producer_module_sha256: str
    bridge_module_sha256: str


@dataclass(frozen=True, slots=True)
class VerifiedContractPerformanceProvenance:
    """Root's compact, re-verified binding for one Contract raw-pair admission."""

    admission_id: str
    provenance_path: str
    byte_sha256: str
    provenance_digest: str
    provenance_id: str
    receipt_digest: str
    sample_id: str
    source_snapshot_digest: str
    producer_module_sha256: str
    bridge_module_sha256: str


@dataclass(frozen=True, slots=True)
class VerifiedContractPerformanceExecutionProvenance:
    """Root's compact, re-verified binding for one private raw artifact."""

    artifact_id: str
    admission_id: str
    provenance_path: str
    byte_sha256: str
    provenance_digest: str
    provenance_id: str
    receipt_digest: str
    sample_id: str
    source_snapshot_digest: str
    artifact_sha256: str
    output_sha256: str
    execution_module_sha256: str


@dataclass(frozen=True, slots=True)
class VerifiedProducerReceipt:
    receipt_id: str
    producer_kind: str
    artifact_id: str
    admissions: tuple[VerifiedTypedAdmission, ...]
    parallel_scaling_provenance: tuple[VerifiedParallelScalingProvenance, ...] = ()
    contract_performance_provenance: tuple[
        VerifiedContractPerformanceProvenance, ...
    ] = ()
    schema_version: str = TYPED_PRODUCER_RECEIPT_SCHEMA_VERSION

    @property
    def digest(self) -> str:
        return stable_digest("osc-root-typed-producer-receipt", self)

    def subjects(self, subject_kind: str) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    subject
                    for admission in self.admissions
                    if admission.subject_kind == subject_kind
                    for subject in admission.subject_ids
                }
            )
        )


@dataclass(frozen=True, slots=True)
class VerifiedArtifactReceipt:
    artifact_id: str
    path: str
    byte_sha256: str
    provenance_digest: str
    producer_kind: str
    producer_receipt_digest: str
    series: tuple[RawMetricSeries, ...]
    contract_performance_execution_provenance: (
        VerifiedContractPerformanceExecutionProvenance | None
    ) = None

    @property
    def digest(self) -> str:
        return stable_digest("osc-root-artifact-receipt", self)


@dataclass(frozen=True, slots=True)
class VerifiedArtifactReceiptIndex:
    index_path: str
    expected_index_sha256: str
    byte_sha256: str
    expected_source_digest: str
    source_digest: str
    expected_dynamic_plan_sha256: str
    dynamic_plan_sha256: str
    producer_receipts: tuple[VerifiedProducerReceipt, ...]
    artifacts: tuple[VerifiedArtifactReceipt, ...]
    verification_errors: tuple[str, ...] = ()
    schema_version: str = ARTIFACT_RECEIPT_SCHEMA_VERSION

    @property
    def valid(self) -> bool:
        if self.verification_errors:
            return False
        return verify_artifact_receipt_index(
            index_path=Path(self.index_path),
            expected_index_sha256=self.expected_index_sha256,
            expected_source_digest=self.expected_source_digest,
            expected_dynamic_plan_sha256=self.expected_dynamic_plan_sha256,
        ) == self

    @property
    def digest(self) -> str:
        return stable_digest(
            "osc-verified-artifact-receipt-index",
            {
                "byte_sha256": self.byte_sha256,
                "source_digest": self.source_digest,
                "dynamic_plan_sha256": self.dynamic_plan_sha256,
                "producer_receipts": self.producer_receipts,
                "artifacts": self.artifacts,
            },
        )

    @property
    def producer_receipts_digest(self) -> str:
        return stable_digest(
            "osc-root-producer-receipt-set",
            tuple(item.digest for item in self.producer_receipts),
        )

    def producer_map(self) -> dict[str, VerifiedProducerReceipt]:
        return {item.producer_kind: item for item in self.producer_receipts}


def _parse_raw_metric_series(
    value: object,
    *,
    artifact_id: str,
    errors: list[str],
) -> tuple[RawMetricSeries, ...]:
    if not isinstance(value, list) or not value:
        errors.append(f"artifact_receipt_series_empty:{artifact_id}")
        return ()
    result: list[RawMetricSeries] = []
    for series_index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != {
            "metric_key",
            "reducer",
            "records",
        }:
            errors.append(
                f"artifact_receipt_series_schema:{artifact_id}:{series_index}"
            )
            continue
        try:
            reducer = MetricReducer(item["reducer"])
            if not isinstance(item["records"], list):
                raise ValueError("records must be a list")
            records = tuple(
                RawMetricRecord(record["record_id"], record["value"])
                for record in item["records"]
                if isinstance(record, dict) and set(record) == {"record_id", "value"}
            )
            if len(records) != len(item["records"]):
                raise ValueError("record fields do not match schema")
            result.append(RawMetricSeries(item["metric_key"], reducer, records))
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(
                f"artifact_receipt_series_invalid:{artifact_id}:{series_index}:{exc}"
            )
    keys = tuple(item.metric_key for item in result)
    if len(keys) != len(set(keys)):
        errors.append(f"artifact_receipt_duplicate_metric:{artifact_id}")
    return tuple(result)


def _verify_typed_admission(
    *,
    root: Path,
    producer_kind: str,
    value: object,
    index: int,
    errors: list[str],
    allow_parallel_scaling_provenance: bool = False,
    allow_contract_performance_provenance: bool = False,
) -> VerifiedTypedAdmission | None:
    expected_fields = {
        "admission_id",
        "subject_kind",
        "subject_ids",
        "envelope_path",
        "envelope_sha256",
        "envelope_type",
        "envelope_schema_version",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        errors.append(f"typed_admission_schema:{producer_kind}:{index}")
        return None
    try:
        admission_id = _string(value["admission_id"], "admission_id")
        subject_kind = _string(value["subject_kind"], "subject_kind")
        relative = _safe_relative(value["envelope_path"])
        expected_sha = _require_sha256(
            value["envelope_sha256"], "typed admission envelope SHA-256"
        )
        envelope_type = _string(value["envelope_type"], "envelope_type")
        envelope_version = _string(
            value["envelope_schema_version"], "envelope_schema_version"
        )
        subjects_value = value["subject_ids"]
        if (
            not isinstance(subjects_value, list)
            or not subjects_value
            or any(not isinstance(item, str) or not item for item in subjects_value)
        ):
            raise ValueError("subject_ids must be a non-empty string list")
        subject_ids = tuple(subjects_value)
        if subject_ids != tuple(sorted(subject_ids)) or len(subject_ids) != len(
            set(subject_ids)
        ):
            raise ValueError("subject_ids must be uniquely sorted")
    except ValueError as exc:
        errors.append(f"typed_admission_invalid:{producer_kind}:{index}:{exc}")
        return None
    admission_errors: list[str] = []
    if (envelope_type, envelope_version) not in _ALLOWED_TYPED_ENVELOPES.get(
        producer_kind, frozenset()
    ):
        admission_errors.append(
            f"typed_admission_type_not_allowed:{producer_kind}:"
            f"{envelope_type}:{envelope_version}"
        )
    envelope_path = (root / relative).resolve()
    try:
        envelope_path.relative_to(root)
    except ValueError:
        admission_errors.append(f"typed_admission_path_escape:{admission_id}")
        errors.extend(admission_errors)
        return None
    try:
        raw = envelope_path.read_bytes()
    except OSError as exc:
        admission_errors.append(
            f"typed_admission_load_failed:{admission_id}:{type(exc).__name__}:{exc}"
        )
        raw = b""
    actual_sha = _sha256_bytes(raw)
    if actual_sha != expected_sha:
        admission_errors.append(f"typed_admission_hash_mismatch:{admission_id}")
    if b"synthetic" in raw.lower():
        admission_errors.append(f"typed_admission_synthetic_rejected:{admission_id}")
    decoded: dict[str, Any] | None = None
    try:
        decoded = decode_canonical_envelope(raw.decode()) if raw else None
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        admission_errors.append(
            f"typed_admission_invalid_envelope:{admission_id}:{exc}"
        )
    if decoded is not None:
        if decoded["type"] != envelope_type:
            admission_errors.append(f"typed_admission_type_mismatch:{admission_id}")
        if decoded["schema_version"] != envelope_version:
            admission_errors.append(
                f"typed_admission_version_mismatch:{admission_id}"
            )
    if decoded is not None and not admission_errors:
        owner_and_replayer = _TYPED_SEMANTIC_REPLAYERS.get(
            (envelope_type, envelope_version)
        )
        if owner_and_replayer is None:
            admission_errors.append(
                f"typed_admission_semantic_replay_owner_missing:{admission_id}"
            )
        else:
            replay_owner, replayer = owner_and_replayer
            try:
                replay_errors = replayer(
                    envelope_type=envelope_type,
                    schema_version=envelope_version,
                    payload=decoded["payload"],
                    subject_kind=subject_kind,
                    subject_ids=subject_ids,
                )
            except Exception as exc:
                replay_errors = (
                    f"{replay_owner}_replay_unhandled:{type(exc).__name__}",
                )
            if not isinstance(replay_errors, tuple) or any(
                not isinstance(item, str) or not item for item in replay_errors
            ):
                replay_errors = (
                    f"{replay_owner}_replay_invalid_error_contract",
                )
            admission_errors.extend(
                f"typed_admission_semantic_replay_failed:{admission_id}:{item}"
                for item in replay_errors
            )
            pending_change_request = _ROOT_CONTEXT_PENDING_SUBJECTS.get(
                subject_kind
            )
            if not replay_errors and pending_change_request is not None:
                admission_errors.append(
                    f"typed_admission_root_context_pending_phase6:{admission_id}:"
                    f"{subject_kind}:{pending_change_request}"
                )
            elif not replay_errors and replay_owner == "runtime":
                # Runtime object shape and intrinsic identity are not execution
                # provenance. The sole provisional exception stays inside the
                # artifact-index verifier and is retained only after its
                # separately hashed R7 parallel provenance record replays.
                is_parallel_bridge_candidate = (
                    allow_parallel_scaling_provenance
                    and producer_kind == "performance_paired_replay"
                    and envelope_type == "ParallelScalingEvidenceReceipt"
                    and envelope_version
                    == PARALLEL_SCALING_EVIDENCE_RECEIPT_SCHEMA_VERSION
                    and subject_kind == "parallel_scaling_samples"
                )
                is_contract_performance_bridge_candidate = (
                    allow_contract_performance_provenance
                    and producer_kind == "performance_paired_replay"
                    and envelope_type == "ContractPerformanceEvidenceReceipt"
                    and envelope_version
                    == CONTRACT_PERFORMANCE_EVIDENCE_RECEIPT_SCHEMA_VERSION
                    and subject_kind == "contract_performance_samples"
                )
                if not (
                    is_parallel_bridge_candidate
                    or is_contract_performance_bridge_candidate
                ):
                    admission_errors.append(
                        f"typed_admission_root_provenance_pending_phase6:{admission_id}"
                    )
    if admission_errors:
        errors.extend(admission_errors)
        return None
    if decoded is None:
        errors.append(f"typed_admission_semantic_replay_unreachable:{admission_id}")
        return None
    envelope_digest = stable_digest(
        "osc-root-typed-admission-envelope",
        decoded,
    )
    return VerifiedTypedAdmission(
        admission_id,
        subject_kind,
        subject_ids,
        str(envelope_path),
        actual_sha,
        envelope_type,
        envelope_version,
        envelope_digest,
    )


def _verify_parallel_scaling_provenance_binding(
    *,
    root: Path,
    value: object,
    index: int,
    admissions_by_id: dict[str, VerifiedTypedAdmission],
    expected_source_digest: str,
    errors: list[str],
) -> VerifiedParallelScalingProvenance | None:
    """Retain a parallel Runtime admission only after exact R7 re-reading."""

    expected_fields = {
        "admission_id",
        "provenance_path",
        "provenance_sha256",
        "provenance_type",
        "provenance_schema_version",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        errors.append(f"parallel_scaling_provenance_schema:{index}")
        return None
    try:
        admission_id = _string(value["admission_id"], "provenance admission ID")
        relative = _safe_relative(value["provenance_path"])
        expected_sha = _require_sha256(
            value["provenance_sha256"], "parallel provenance SHA-256"
        )
        provenance_type = _string(value["provenance_type"], "provenance type")
        provenance_schema_version = _string(
            value["provenance_schema_version"], "provenance schema version"
        )
    except ValueError as exc:
        errors.append(f"parallel_scaling_provenance_invalid:{index}:{exc}")
        return None
    if (
        provenance_type != PARALLEL_SCALING_PRODUCTION_PROVENANCE_ENVELOPE_TYPE
        or provenance_schema_version
        != PARALLEL_SCALING_PRODUCTION_PROVENANCE_SCHEMA_VERSION
    ):
        errors.append(f"parallel_scaling_provenance_type_mismatch:{admission_id}")
        return None
    admission = admissions_by_id.get(admission_id)
    if admission is None:
        errors.append(f"parallel_scaling_provenance_admission_missing:{admission_id}")
        return None
    if (
        admission.envelope_type != "ParallelScalingEvidenceReceipt"
        or admission.envelope_schema_version
        != PARALLEL_SCALING_EVIDENCE_RECEIPT_SCHEMA_VERSION
        or admission.subject_kind != "parallel_scaling_samples"
    ):
        errors.append(
            f"parallel_scaling_provenance_admission_shape_mismatch:{admission_id}"
        )
        return None
    if "synthetic" in "\0".join(
        (admission_id, relative, provenance_type, provenance_schema_version)
    ).lower():
        errors.append(f"parallel_scaling_provenance_synthetic_rejected:{admission_id}")
        return None
    provenance_path = (root / relative).resolve()
    try:
        provenance_path.relative_to(root)
    except ValueError:
        errors.append(f"parallel_scaling_provenance_path_escape:{admission_id}")
        return None
    try:
        raw = provenance_path.read_bytes()
    except OSError as exc:
        errors.append(
            f"parallel_scaling_provenance_load_failed:{admission_id}:"
            f"{type(exc).__name__}:{exc}"
        )
        return None
    actual_sha = _sha256_bytes(raw)
    if actual_sha != expected_sha:
        errors.append(f"parallel_scaling_provenance_hash_mismatch:{admission_id}")
        return None
    if b"synthetic" in raw.lower():
        errors.append(f"parallel_scaling_provenance_synthetic_rejected:{admission_id}")
        return None
    try:
        decoded = decode_canonical_envelope(raw.decode())
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        errors.append(
            f"parallel_scaling_provenance_envelope_invalid:{admission_id}:"
            f"{type(exc).__name__}:{exc}"
        )
        return None
    if (
        decoded.get("type") != provenance_type
        or decoded.get("schema_version") != provenance_schema_version
    ):
        errors.append(f"parallel_scaling_provenance_envelope_mismatch:{admission_id}")
        return None
    try:
        provenance = reconstruct_parallel_scaling_production_provenance_payload(
            decoded["payload"]
        )
    except Exception as exc:  # Root must reject malformed private evidence.
        errors.append(
            f"parallel_scaling_provenance_reconstruction_failed:{admission_id}:"
            f"{type(exc).__name__}"
        )
        return None
    if provenance.receipt.source_snapshot_digest != expected_source_digest:
        errors.append(f"parallel_scaling_provenance_source_mismatch:{admission_id}")
        return None
    admitted = _reconstruct_parallel_scaling_admission(admission, errors=errors)
    if admitted is None:
        return None
    if admitted != provenance.receipt:
        errors.append(f"parallel_scaling_provenance_receipt_mismatch:{admission_id}")
        return None
    return VerifiedParallelScalingProvenance(
        admission_id=admission_id,
        provenance_path=str(provenance_path),
        byte_sha256=actual_sha,
        provenance_digest=provenance.digest,
        provenance_id=provenance.provenance_id,
        receipt_digest=provenance.receipt.digest,
        sample_id=provenance.receipt.sample_id,
        source_snapshot_digest=provenance.receipt.source_snapshot_digest,
        producer_module_sha256=provenance.producer_module_sha256,
        bridge_module_sha256=provenance.bridge_module_sha256,
    )


def _verify_contract_performance_provenance_binding(
    *,
    root: Path,
    value: object,
    index: int,
    admissions_by_id: dict[str, VerifiedTypedAdmission],
    expected_source_digest: str,
    errors: list[str],
) -> VerifiedContractPerformanceProvenance | None:
    """Retain a Contract raw-pair only after exact private proof re-reading."""

    expected_fields = {
        "admission_id",
        "provenance_path",
        "provenance_sha256",
        "provenance_type",
        "provenance_schema_version",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        errors.append(f"contract_performance_provenance_schema:{index}")
        return None
    try:
        admission_id = _string(value["admission_id"], "provenance admission ID")
        relative = _safe_relative(value["provenance_path"])
        expected_sha = _require_sha256(
            value["provenance_sha256"], "contract performance provenance SHA-256"
        )
        provenance_type = _string(value["provenance_type"], "provenance type")
        provenance_schema_version = _string(
            value["provenance_schema_version"], "provenance schema version"
        )
    except ValueError as exc:
        errors.append(f"contract_performance_provenance_invalid:{index}:{exc}")
        return None
    if (
        provenance_type != CONTRACT_PERFORMANCE_PRODUCTION_PROVENANCE_ENVELOPE_TYPE
        or provenance_schema_version
        != CONTRACT_PERFORMANCE_PRODUCTION_PROVENANCE_SCHEMA_VERSION
    ):
        errors.append(f"contract_performance_provenance_type_mismatch:{admission_id}")
        return None
    admission = admissions_by_id.get(admission_id)
    if admission is None:
        errors.append(
            f"contract_performance_provenance_admission_missing:{admission_id}"
        )
        return None
    if (
        admission.envelope_type != "ContractPerformanceEvidenceReceipt"
        or admission.envelope_schema_version
        != CONTRACT_PERFORMANCE_EVIDENCE_RECEIPT_SCHEMA_VERSION
        or admission.subject_kind != "contract_performance_samples"
    ):
        errors.append(
            "contract_performance_provenance_admission_shape_mismatch:"
            f"{admission_id}"
        )
        return None
    if "synthetic" in "\0".join(
        (admission_id, relative, provenance_type, provenance_schema_version)
    ).lower():
        errors.append(
            f"contract_performance_provenance_synthetic_rejected:{admission_id}"
        )
        return None
    provenance_path = (root / relative).resolve()
    try:
        provenance_path.relative_to(root)
    except ValueError:
        errors.append(f"contract_performance_provenance_path_escape:{admission_id}")
        return None
    try:
        raw = provenance_path.read_bytes()
    except OSError as exc:
        errors.append(
            "contract_performance_provenance_load_failed:"
            f"{admission_id}:{type(exc).__name__}:{exc}"
        )
        return None
    actual_sha = _sha256_bytes(raw)
    if actual_sha != expected_sha:
        errors.append(f"contract_performance_provenance_hash_mismatch:{admission_id}")
        return None
    if b"synthetic" in raw.lower():
        errors.append(
            f"contract_performance_provenance_synthetic_rejected:{admission_id}"
        )
        return None
    try:
        decoded = decode_canonical_envelope(raw.decode())
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        errors.append(
            "contract_performance_provenance_envelope_invalid:"
            f"{admission_id}:{type(exc).__name__}:{exc}"
        )
        return None
    if (
        decoded.get("type") != provenance_type
        or decoded.get("schema_version") != provenance_schema_version
    ):
        errors.append(
            f"contract_performance_provenance_envelope_mismatch:{admission_id}"
        )
        return None
    try:
        provenance = reconstruct_contract_performance_production_provenance_payload(
            decoded["payload"]
        )
    except Exception as exc:  # Root must reject malformed private evidence.
        errors.append(
            "contract_performance_provenance_reconstruction_failed:"
            f"{admission_id}:{type(exc).__name__}"
        )
        return None
    if provenance.receipt.source_snapshot_digest != expected_source_digest:
        errors.append(
            f"contract_performance_provenance_source_mismatch:{admission_id}"
        )
        return None
    admitted = _reconstruct_contract_performance_admission(admission, errors=errors)
    if admitted is None:
        return None
    if admitted != provenance.receipt:
        errors.append(
            f"contract_performance_provenance_receipt_mismatch:{admission_id}"
        )
        return None
    return VerifiedContractPerformanceProvenance(
        admission_id=admission_id,
        provenance_path=str(provenance_path),
        byte_sha256=actual_sha,
        provenance_digest=provenance.digest,
        provenance_id=provenance.provenance_id,
        receipt_digest=provenance.receipt.digest,
        sample_id=provenance.receipt.sample_id,
        source_snapshot_digest=provenance.receipt.source_snapshot_digest,
        producer_module_sha256=provenance.producer_module_sha256,
        bridge_module_sha256=provenance.bridge_module_sha256,
    )


def _verify_contract_performance_execution_provenance_binding(
    *,
    root: Path,
    value: object,
    index: int,
    artifact_id: str,
    artifact_raw: bytes,
    artifact_sha256: str,
    declared_provenance_digest: str,
    producer_kind: str,
    producer_receipt: VerifiedProducerReceipt | None,
    expected_source_digest: str,
    errors: list[str],
) -> VerifiedContractPerformanceExecutionProvenance | None:
    """Retain one Contract raw artifact only after exact proof re-reading."""

    expected_fields = {
        "admission_id",
        "provenance_path",
        "provenance_sha256",
        "provenance_type",
        "provenance_schema_version",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        errors.append(
            "contract_performance_execution_provenance_schema:"
            f"{artifact_id}:{index}"
        )
        return None
    try:
        admission_id = _string(value["admission_id"], "execution provenance admission ID")
        relative = _safe_relative(value["provenance_path"])
        expected_sha = _require_sha256(
            value["provenance_sha256"], "execution provenance SHA-256"
        )
        provenance_type = _string(value["provenance_type"], "execution provenance type")
        provenance_schema_version = _string(
            value["provenance_schema_version"], "execution provenance schema version"
        )
    except ValueError as exc:
        errors.append(
            "contract_performance_execution_provenance_invalid:"
            f"{artifact_id}:{index}:{exc}"
        )
        return None
    if producer_kind != "performance_paired_replay":
        errors.append(
            "contract_performance_execution_provenance_producer_kind_mismatch:"
            f"{artifact_id}"
        )
        return None
    if (
        provenance_type
        != CONTRACT_PERFORMANCE_EXECUTION_PROVENANCE_ENVELOPE_TYPE
        or provenance_schema_version
        != CONTRACT_PERFORMANCE_EXECUTION_PROVENANCE_SCHEMA_VERSION
    ):
        errors.append(
            "contract_performance_execution_provenance_type_mismatch:"
            f"{artifact_id}"
        )
        return None
    if producer_receipt is None:
        errors.append(
            "contract_performance_execution_provenance_producer_receipt_missing:"
            f"{artifact_id}"
        )
        return None
    r9_bindings = tuple(
        item
        for item in producer_receipt.contract_performance_provenance
        if item.admission_id == admission_id
    )
    if len(r9_bindings) != 1:
        errors.append(
            "contract_performance_execution_provenance_r9_binding_missing:"
            f"{artifact_id}:{admission_id}"
        )
        return None
    r9_binding = r9_bindings[0]
    if "synthetic" in "\0".join(
        (artifact_id, admission_id, relative, provenance_type, provenance_schema_version)
    ).lower():
        errors.append(
            "contract_performance_execution_provenance_synthetic_rejected:"
            f"{artifact_id}"
        )
        return None
    provenance_path = (root / relative).resolve()
    try:
        provenance_path.relative_to(root)
    except ValueError:
        errors.append(
            "contract_performance_execution_provenance_path_escape:"
            f"{artifact_id}"
        )
        return None
    try:
        raw = provenance_path.read_bytes()
    except OSError as exc:
        errors.append(
            "contract_performance_execution_provenance_load_failed:"
            f"{artifact_id}:{type(exc).__name__}:{exc}"
        )
        return None
    actual_sha = _sha256_bytes(raw)
    if actual_sha != expected_sha:
        errors.append(
            "contract_performance_execution_provenance_hash_mismatch:"
            f"{artifact_id}"
        )
        return None
    if b"synthetic" in raw.lower():
        errors.append(
            "contract_performance_execution_provenance_synthetic_rejected:"
            f"{artifact_id}"
        )
        return None
    try:
        decoded = decode_canonical_envelope(raw.decode())
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        errors.append(
            "contract_performance_execution_provenance_envelope_invalid:"
            f"{artifact_id}:{type(exc).__name__}:{exc}"
        )
        return None
    if (
        decoded.get("type") != provenance_type
        or decoded.get("schema_version") != provenance_schema_version
    ):
        errors.append(
            "contract_performance_execution_provenance_envelope_mismatch:"
            f"{artifact_id}"
        )
        return None
    try:
        provenance = reconstruct_contract_performance_execution_provenance_payload(
            decoded["payload"]
        )
    except Exception as exc:  # Root must reject malformed private evidence.
        errors.append(
            "contract_performance_execution_provenance_reconstruction_failed:"
            f"{artifact_id}:{type(exc).__name__}"
        )
        return None
    if provenance.source_snapshot_digest != expected_source_digest:
        errors.append(
            "contract_performance_execution_provenance_source_mismatch:"
            f"{artifact_id}"
        )
        return None
    if provenance.production_provenance.digest != r9_binding.provenance_digest:
        errors.append(
            "contract_performance_execution_provenance_r9_proof_mismatch:"
            f"{artifact_id}"
        )
        return None
    if provenance.production_provenance.provenance_id != r9_binding.provenance_id:
        errors.append(
            "contract_performance_execution_provenance_r9_identity_mismatch:"
            f"{artifact_id}"
        )
        return None
    if provenance.receipt.digest != r9_binding.receipt_digest:
        errors.append(
            "contract_performance_execution_provenance_receipt_mismatch:"
            f"{artifact_id}"
        )
        return None
    if provenance.receipt.sample_id != r9_binding.sample_id:
        errors.append(
            "contract_performance_execution_provenance_sample_mismatch:"
            f"{artifact_id}"
        )
        return None
    if provenance.artifact_id != artifact_id:
        errors.append(
            "contract_performance_execution_provenance_artifact_id_mismatch:"
            f"{artifact_id}"
        )
        return None
    if provenance.artifact_sha256 != artifact_sha256:
        errors.append(
            "contract_performance_execution_provenance_artifact_sha_mismatch:"
            f"{artifact_id}"
        )
        return None
    if provenance.artifact_bytes != artifact_raw:
        errors.append(
            "contract_performance_execution_provenance_artifact_bytes_mismatch:"
            f"{artifact_id}"
        )
        return None
    if provenance.transcript.stdout != artifact_raw:
        errors.append(
            "contract_performance_execution_provenance_output_mismatch:"
            f"{artifact_id}"
        )
        return None
    if provenance.transcript.output_sha256 != artifact_sha256:
        errors.append(
            "contract_performance_execution_provenance_output_sha_mismatch:"
            f"{artifact_id}"
        )
        return None
    if provenance.transcript.exit_code != 0:
        errors.append(
            "contract_performance_execution_provenance_exit_mismatch:"
            f"{artifact_id}"
        )
        return None
    if provenance.digest != declared_provenance_digest:
        errors.append(
            "contract_performance_execution_provenance_declared_digest_mismatch:"
            f"{artifact_id}"
        )
        return None
    return VerifiedContractPerformanceExecutionProvenance(
        artifact_id=artifact_id,
        admission_id=admission_id,
        provenance_path=str(provenance_path),
        byte_sha256=actual_sha,
        provenance_digest=provenance.digest,
        provenance_id=provenance.provenance_id,
        receipt_digest=provenance.receipt.digest,
        sample_id=provenance.receipt.sample_id,
        source_snapshot_digest=provenance.source_snapshot_digest,
        artifact_sha256=provenance.artifact_sha256,
        output_sha256=provenance.transcript.output_sha256,
        execution_module_sha256=provenance.execution_module_sha256,
    )


def verify_artifact_receipt_index(
    *,
    index_path: Path,
    expected_index_sha256: str,
    expected_source_digest: str,
    expected_dynamic_plan_sha256: str,
) -> VerifiedArtifactReceiptIndex:
    _require_sha256(expected_index_sha256, "expected receipt-index SHA-256")
    _string(expected_source_digest, "expected source digest")
    _require_sha256(
        expected_dynamic_plan_sha256, "expected dynamic plan SHA-256"
    )
    resolved = index_path.resolve()
    root = resolved.parent
    errors: list[str] = []
    try:
        raw = resolved.read_bytes()
    except OSError as exc:
        raw = b""
        errors.append(f"artifact_receipt_index_load_failed:{type(exc).__name__}:{exc}")
    byte_sha = _sha256_bytes(raw)
    if byte_sha != expected_index_sha256:
        errors.append("artifact_receipt_index_hash_mismatch")
    try:
        payload = _strict_json_loads(raw) if raw else None
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        payload = None
        errors.append(f"artifact_receipt_index_invalid_json:{exc}")
    source_digest = expected_source_digest
    dynamic_plan_sha = expected_dynamic_plan_sha256
    producer_receipts: list[VerifiedProducerReceipt] = []
    artifacts: list[VerifiedArtifactReceipt] = []
    expected_top_fields = {
        "schema_version",
        "source_digest",
        "dynamic_plan_sha256",
        "producer_receipts",
        "artifacts",
    }
    if not isinstance(payload, dict) or set(payload) != expected_top_fields:
        errors.append("artifact_receipt_index_schema_mismatch")
    else:
        if payload["schema_version"] != ARTIFACT_RECEIPT_SCHEMA_VERSION:
            errors.append("artifact_receipt_index_schema_version_mismatch")
        try:
            source_digest = _string(payload["source_digest"], "source_digest")
            dynamic_plan_sha = _require_sha256(
                payload["dynamic_plan_sha256"], "dynamic plan SHA-256"
            )
        except ValueError as exc:
            errors.append(f"artifact_receipt_index_invalid_binding:{exc}")
        if source_digest != expected_source_digest:
            errors.append("artifact_receipt_index_source_mismatch")
        if dynamic_plan_sha != expected_dynamic_plan_sha256:
            errors.append("artifact_receipt_index_dynamic_plan_mismatch")

        receipt_values = payload["producer_receipts"]
        if not isinstance(receipt_values, list) or not receipt_values:
            errors.append("typed_producer_receipts_missing")
            receipt_values = []
        for receipt_index, item in enumerate(receipt_values):
            receipt_fields = {
                "schema_version",
                "receipt_id",
                "producer_kind",
                "artifact_id",
                "admissions",
            }
            optional_provenance_fields = {
                "parallel_scaling_provenance",
                "contract_performance_provenance",
            }
            if (
                not isinstance(item, dict)
                or not receipt_fields <= set(item)
                or not set(item) - receipt_fields <= optional_provenance_fields
            ):
                errors.append(f"typed_producer_receipt_schema:{receipt_index}")
                continue
            try:
                if item["schema_version"] != TYPED_PRODUCER_RECEIPT_SCHEMA_VERSION:
                    raise ValueError("schema version mismatch")
                receipt_id = _string(item["receipt_id"], "receipt_id")
                producer_kind = _string(item["producer_kind"], "producer_kind")
                artifact_id = _string(item["artifact_id"], "artifact_id")
            except ValueError as exc:
                errors.append(f"typed_producer_receipt_invalid:{receipt_index}:{exc}")
                continue
            if producer_kind not in _REQUIRED_PRODUCER_KINDS:
                errors.append(f"typed_producer_receipt_untrusted:{producer_kind}")
            has_parallel_scaling_provenance = (
                "parallel_scaling_provenance" in item
            )
            has_contract_performance_provenance = (
                "contract_performance_provenance" in item
            )
            if (
                has_parallel_scaling_provenance
                and producer_kind != "performance_paired_replay"
            ):
                errors.append(
                    "parallel_scaling_provenance_producer_kind_mismatch:"
                    f"{producer_kind}"
                )
            if (
                has_contract_performance_provenance
                and producer_kind != "performance_paired_replay"
            ):
                errors.append(
                    "contract_performance_provenance_producer_kind_mismatch:"
                    f"{producer_kind}"
                )
            admission_values = item["admissions"]
            if not isinstance(admission_values, list) or not admission_values:
                errors.append(f"typed_producer_receipt_admissions_missing:{producer_kind}")
                admission_values = []
            admissions = tuple(
                admission
                for admission_index, value in enumerate(admission_values)
                if (
                    admission := _verify_typed_admission(
                        root=root,
                        producer_kind=producer_kind,
                        value=value,
                        index=admission_index,
                        errors=errors,
                        allow_parallel_scaling_provenance=(
                            has_parallel_scaling_provenance
                            and producer_kind == "performance_paired_replay"
                        ),
                        allow_contract_performance_provenance=(
                            has_contract_performance_provenance
                            and producer_kind == "performance_paired_replay"
                        ),
                    )
                )
                is not None
            )
            admission_ids = tuple(item.admission_id for item in admissions)
            if admission_ids != tuple(sorted(admission_ids)) or len(
                admission_ids
            ) != len(set(admission_ids)):
                errors.append(
                    f"typed_producer_receipt_admissions_not_unique_sorted:{producer_kind}"
                )
            parallel_provenance: tuple[VerifiedParallelScalingProvenance, ...] = ()
            parallel_admissions = tuple(
                admission
                for admission in admissions
                if (
                    admission.envelope_type == "ParallelScalingEvidenceReceipt"
                    and admission.envelope_schema_version
                    == PARALLEL_SCALING_EVIDENCE_RECEIPT_SCHEMA_VERSION
                    and admission.subject_kind == "parallel_scaling_samples"
                )
            )
            provenance_error_count = len(errors)
            if has_parallel_scaling_provenance:
                provenance_values = item["parallel_scaling_provenance"]
                if not isinstance(provenance_values, list) or not provenance_values:
                    errors.append(
                        "parallel_scaling_provenance_bindings_missing:"
                        f"{producer_kind}"
                    )
                    provenance_values = []
                parsed_provenance = tuple(
                    binding
                    for provenance_index, provenance_value in enumerate(
                        provenance_values
                    )
                    if (
                        binding := _verify_parallel_scaling_provenance_binding(
                            root=root,
                            value=provenance_value,
                            index=provenance_index,
                            admissions_by_id={
                                admission.admission_id: admission
                                for admission in admissions
                            },
                            expected_source_digest=expected_source_digest,
                            errors=errors,
                        )
                    )
                    is not None
                )
                provenance_ids = tuple(
                    binding.admission_id for binding in parsed_provenance
                )
                if provenance_ids != tuple(sorted(provenance_ids)) or len(
                    provenance_ids
                ) != len(set(provenance_ids)):
                    errors.append(
                        "parallel_scaling_provenance_bindings_not_unique_sorted:"
                        f"{producer_kind}"
                    )
                if len(errors) == provenance_error_count:
                    parallel_provenance = parsed_provenance
            if parallel_admissions and not has_parallel_scaling_provenance:
                errors.append(
                    "parallel_scaling_provenance_bindings_missing:"
                    f"{producer_kind}"
                )
            if has_parallel_scaling_provenance and not parallel_admissions:
                errors.append(
                    "parallel_scaling_provenance_bindings_unexpected:"
                    f"{producer_kind}"
                )
            expected_parallel_ids = tuple(
                admission.admission_id for admission in parallel_admissions
            )
            observed_parallel_ids = tuple(
                binding.admission_id for binding in parallel_provenance
            )
            if parallel_admissions and observed_parallel_ids != expected_parallel_ids:
                if has_parallel_scaling_provenance:
                    errors.append(
                        "parallel_scaling_provenance_bijection_mismatch:"
                        f"{producer_kind}"
                    )
                parallel_provenance = ()
            verified_parallel_ids = {
                binding.admission_id for binding in parallel_provenance
            }
            for admission in parallel_admissions:
                if admission.admission_id not in verified_parallel_ids:
                    errors.append(
                        "parallel_scaling_provenance_not_verified:"
                        f"{admission.admission_id}"
                    )
            if parallel_admissions:
                admissions = tuple(
                    admission
                    for admission in admissions
                    if (
                        admission.envelope_type != "ParallelScalingEvidenceReceipt"
                        or admission.envelope_schema_version
                        != PARALLEL_SCALING_EVIDENCE_RECEIPT_SCHEMA_VERSION
                        or admission.subject_kind != "parallel_scaling_samples"
                        or admission.admission_id in verified_parallel_ids
                    )
                )
            contract_performance_provenance: tuple[
                VerifiedContractPerformanceProvenance, ...
            ] = ()
            contract_performance_admissions = tuple(
                admission
                for admission in admissions
                if (
                    admission.envelope_type == "ContractPerformanceEvidenceReceipt"
                    and admission.envelope_schema_version
                    == CONTRACT_PERFORMANCE_EVIDENCE_RECEIPT_SCHEMA_VERSION
                    and admission.subject_kind == "contract_performance_samples"
                )
            )
            contract_provenance_error_count = len(errors)
            if has_contract_performance_provenance:
                provenance_values = item["contract_performance_provenance"]
                if not isinstance(provenance_values, list) or not provenance_values:
                    errors.append(
                        "contract_performance_provenance_bindings_missing:"
                        f"{producer_kind}"
                    )
                    provenance_values = []
                parsed_provenance = tuple(
                    binding
                    for provenance_index, provenance_value in enumerate(
                        provenance_values
                    )
                    if (
                        binding := _verify_contract_performance_provenance_binding(
                            root=root,
                            value=provenance_value,
                            index=provenance_index,
                            admissions_by_id={
                                admission.admission_id: admission
                                for admission in admissions
                            },
                            expected_source_digest=expected_source_digest,
                            errors=errors,
                        )
                    )
                    is not None
                )
                provenance_ids = tuple(
                    binding.admission_id for binding in parsed_provenance
                )
                if provenance_ids != tuple(sorted(provenance_ids)) or len(
                    provenance_ids
                ) != len(set(provenance_ids)):
                    errors.append(
                        "contract_performance_provenance_bindings_not_unique_sorted:"
                        f"{producer_kind}"
                    )
                if len(errors) == contract_provenance_error_count:
                    contract_performance_provenance = parsed_provenance
            if (
                contract_performance_admissions
                and not has_contract_performance_provenance
            ):
                errors.append(
                    "contract_performance_provenance_bindings_missing:"
                    f"{producer_kind}"
                )
            if (
                has_contract_performance_provenance
                and not contract_performance_admissions
            ):
                errors.append(
                    "contract_performance_provenance_bindings_unexpected:"
                    f"{producer_kind}"
                )
            expected_contract_ids = tuple(
                admission.admission_id for admission in contract_performance_admissions
            )
            observed_contract_ids = tuple(
                binding.admission_id
                for binding in contract_performance_provenance
            )
            if (
                contract_performance_admissions
                and observed_contract_ids != expected_contract_ids
            ):
                if has_contract_performance_provenance:
                    errors.append(
                        "contract_performance_provenance_bijection_mismatch:"
                        f"{producer_kind}"
                    )
                contract_performance_provenance = ()
            verified_contract_ids = {
                binding.admission_id
                for binding in contract_performance_provenance
            }
            for admission in contract_performance_admissions:
                if admission.admission_id not in verified_contract_ids:
                    errors.append(
                        "contract_performance_provenance_not_verified:"
                        f"{admission.admission_id}"
                    )
            if contract_performance_admissions:
                admissions = tuple(
                    admission
                    for admission in admissions
                    if (
                        admission.envelope_type
                        != "ContractPerformanceEvidenceReceipt"
                        or admission.envelope_schema_version
                        != CONTRACT_PERFORMANCE_EVIDENCE_RECEIPT_SCHEMA_VERSION
                        or admission.subject_kind != "contract_performance_samples"
                        or admission.admission_id in verified_contract_ids
                    )
                )
            observed_types = {
                (admission.envelope_type, admission.envelope_schema_version)
                for admission in admissions
            }
            required_types = _ALLOWED_TYPED_ENVELOPES.get(
                producer_kind, frozenset()
            )
            missing_types = sorted(required_types - observed_types)
            if missing_types:
                errors.append(
                    f"typed_producer_receipt_payload_types_missing:{producer_kind}:"
                    + ",".join(f"{name}@{version}" for name, version in missing_types)
                )
            producer_receipts.append(
                VerifiedProducerReceipt(
                    receipt_id,
                    producer_kind,
                    artifact_id,
                    admissions,
                    parallel_provenance,
                    contract_performance_provenance,
                )
            )

        receipt_by_kind = {item.producer_kind: item for item in producer_receipts}
        artifact_values = payload["artifacts"]
        if not isinstance(artifact_values, list) or not artifact_values:
            errors.append("artifact_receipt_index_empty")
            artifact_values = []
        required_artifact_fields = {
            "artifact_id",
            "path",
            "sha256",
            "provenance_digest",
            "producer_kind",
            "producer_receipt_digest",
        }
        optional_artifact_provenance_fields = {
            "contract_performance_execution_provenance",
        }
        for artifact_index, item in enumerate(artifact_values):
            if (
                not isinstance(item, dict)
                or not required_artifact_fields <= set(item)
                or not set(item) - required_artifact_fields
                <= optional_artifact_provenance_fields
            ):
                errors.append(f"artifact_receipt_schema:{artifact_index}")
                continue
            try:
                artifact_id = _string(item["artifact_id"], "artifact_id")
                relative = _safe_relative(item["path"])
                expected_sha = _require_sha256(item["sha256"], "artifact SHA-256")
                provenance = _digest(item["provenance_digest"], "provenance_digest")
                producer_kind = _string(item["producer_kind"], "producer_kind")
                receipt_digest = _digest(
                    item["producer_receipt_digest"], "producer_receipt_digest"
                )
            except ValueError as exc:
                errors.append(f"artifact_receipt_invalid:{artifact_index}:{exc}")
                continue
            if producer_kind not in _REQUIRED_PRODUCER_KINDS:
                errors.append(f"artifact_receipt_untrusted_producer:{producer_kind}")
            has_contract_performance_execution_provenance = (
                "contract_performance_execution_provenance" in item
            )
            if (
                has_contract_performance_execution_provenance
                and producer_kind != "performance_paired_replay"
            ):
                errors.append(
                    "contract_performance_execution_provenance_producer_kind_mismatch:"
                    f"{artifact_id}"
                )
            if "synthetic" in "\0".join(
                (artifact_id, relative, provenance, producer_kind, receipt_digest)
            ).lower():
                errors.append(f"artifact_receipt_synthetic_rejected:{artifact_id}")
            artifact_path = (root / relative).resolve()
            try:
                artifact_path.relative_to(root)
            except ValueError:
                errors.append(f"artifact_receipt_path_escape:{artifact_id}")
                continue
            try:
                artifact_raw = artifact_path.read_bytes()
            except OSError as exc:
                artifact_raw = b""
                errors.append(
                    f"artifact_receipt_load_failed:{artifact_id}:"
                    f"{type(exc).__name__}:{exc}"
                )
            actual_sha = _sha256_bytes(artifact_raw)
            if actual_sha != expected_sha:
                errors.append(f"artifact_receipt_hash_mismatch:{artifact_id}")
            try:
                artifact_payload = _strict_json_loads(artifact_raw) if artifact_raw else None
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                artifact_payload = None
                errors.append(f"artifact_receipt_invalid_json:{artifact_id}:{exc}")
            series: tuple[RawMetricSeries, ...] = ()
            if not isinstance(artifact_payload, dict) or set(artifact_payload) != {
                "schema_version",
                "artifact_id",
                "source_digest",
                "series",
            }:
                errors.append(f"artifact_receipt_raw_schema:{artifact_id}")
            else:
                if artifact_payload["schema_version"] != RAW_GATE_ARTIFACT_SCHEMA_VERSION:
                    errors.append(f"artifact_receipt_raw_version:{artifact_id}")
                if artifact_payload["artifact_id"] != artifact_id:
                    errors.append(f"artifact_receipt_id_mismatch:{artifact_id}")
                if artifact_payload["source_digest"] != expected_source_digest:
                    errors.append(f"artifact_receipt_source_mismatch:{artifact_id}")
                series = _parse_raw_metric_series(
                    artifact_payload["series"], artifact_id=artifact_id, errors=errors
                )
            contract_performance_execution_provenance: (
                VerifiedContractPerformanceExecutionProvenance | None
            ) = None
            producer_receipt = receipt_by_kind.get(producer_kind)
            if has_contract_performance_execution_provenance:
                binding_values = item["contract_performance_execution_provenance"]
                if not isinstance(binding_values, list) or len(binding_values) != 1:
                    errors.append(
                        "contract_performance_execution_provenance_bindings_cardinality:"
                        f"{artifact_id}"
                    )
                elif producer_kind == "performance_paired_replay":
                    if producer_receipt is None:
                        errors.append(
                            "contract_performance_execution_provenance_producer_receipt_missing:"
                            f"{artifact_id}"
                        )
                    elif (
                        producer_receipt.artifact_id != artifact_id
                        or producer_receipt.digest != receipt_digest
                    ):
                        errors.append(
                            "contract_performance_execution_provenance_producer_receipt_mismatch:"
                            f"{artifact_id}"
                        )
                    else:
                        contract_performance_execution_provenance = (
                            _verify_contract_performance_execution_provenance_binding(
                                root=root,
                                value=binding_values[0],
                                index=artifact_index,
                                artifact_id=artifact_id,
                                artifact_raw=artifact_raw,
                                artifact_sha256=actual_sha,
                                declared_provenance_digest=provenance,
                                producer_kind=producer_kind,
                                producer_receipt=producer_receipt,
                                expected_source_digest=expected_source_digest,
                                errors=errors,
                            )
                        )
            elif (
                producer_kind == "performance_paired_replay"
                and producer_receipt is not None
                and producer_receipt.contract_performance_provenance
            ):
                errors.append(
                    "contract_performance_execution_provenance_bindings_missing:"
                    f"{artifact_id}"
                )
            artifacts.append(
                VerifiedArtifactReceipt(
                    artifact_id,
                    str(artifact_path),
                    actual_sha,
                    provenance,
                    producer_kind,
                    receipt_digest,
                    series,
                    contract_performance_execution_provenance,
                )
            )

        receipt_kinds = tuple(item.producer_kind for item in producer_receipts)
        artifact_kinds = tuple(item.producer_kind for item in artifacts)
        for label, kinds in (
            ("typed_producer_receipt", receipt_kinds),
            ("artifact_receipt", artifact_kinds),
        ):
            if len(kinds) != len(set(kinds)):
                errors.append(f"{label}_duplicate_producer_kind")
            missing = sorted(_REQUIRED_PRODUCER_KINDS - set(kinds))
            extra = sorted(set(kinds) - _REQUIRED_PRODUCER_KINDS)
            if missing:
                errors.append(f"{label}_producers_missing:" + ",".join(missing))
            if extra:
                errors.append(f"{label}_producers_extra:" + ",".join(extra))
        receipt_by_kind = {item.producer_kind: item for item in producer_receipts}
        for artifact in artifacts:
            receipt = receipt_by_kind.get(artifact.producer_kind)
            if receipt is None:
                continue
            if receipt.artifact_id != artifact.artifact_id:
                errors.append(
                    f"producer_receipt_artifact_mismatch:{artifact.producer_kind}"
                )
            if receipt.digest != artifact.producer_receipt_digest:
                errors.append(
                    f"producer_receipt_digest_mismatch:{artifact.producer_kind}"
                )

    return VerifiedArtifactReceiptIndex(
        index_path=str(resolved),
        expected_index_sha256=expected_index_sha256,
        byte_sha256=byte_sha,
        expected_source_digest=expected_source_digest,
        source_digest=source_digest,
        expected_dynamic_plan_sha256=expected_dynamic_plan_sha256,
        dynamic_plan_sha256=dynamic_plan_sha,
        producer_receipts=tuple(
            sorted(producer_receipts, key=lambda item: item.producer_kind)
        ),
        artifacts=tuple(sorted(artifacts, key=lambda item: item.artifact_id)),
        verification_errors=tuple(dict.fromkeys(errors)),
    )


def _comparison_partition_digest(
    *,
    kind: str,
    partition: tuple[tuple[str, ...], ...],
) -> str:
    return stable_digest(
        "osc-phase6-contract-comparison-partition-binding-v1",
        {"kind": kind, "partition": partition},
    )


def _contract_comparison_dynamic_record(
    receipt: ContractComparisonReceipt,
    *,
    producer_receipt_digest: str,
) -> dict[str, str]:
    """Derive the only dynamic record shape Root can bind for a receipt."""

    if not isinstance(receipt, ContractComparisonReceipt):
        raise TypeError("comparison dynamic record requires ContractComparisonReceipt")
    _digest(producer_receipt_digest, "producer_receipt_digest")
    return {
        "comparison_decision_id": receipt.comparison_decision_id,
        "source_snapshot_digest": receipt.source_snapshot_digest,
        "source_case_digest": receipt.source_case_digest,
        "result_group_id": receipt.result_group.result_group_id,
        "result_group_digest": receipt.result_group.digest,
        "contract_fingerprint_digest": receipt.result_group.contract_fingerprint.digest,
        "endpoint_set_digest": receipt.endpoint_set_digest,
        "fingerprint_task_id": receipt.fingerprint_task.identity.task_id,
        "fingerprint_result_digest": receipt.fingerprint_result.digest,
        "exact_task_id": receipt.exact_task.identity.task_id,
        "exact_result_digest": receipt.exact_result.digest,
        "fingerprint_partition_digest": _comparison_partition_digest(
            kind="fingerprint",
            partition=receipt.fingerprint_partition,
        ),
        "canonical_partition_digest": _comparison_partition_digest(
            kind="canonical",
            partition=receipt.pairwise_canonical_partition,
        ),
        "producer_receipt_digest": producer_receipt_digest,
    }


def _contract_performance_dynamic_record(
    receipt: ContractPerformanceEvidenceReceipt,
    *,
    producer_receipt_digest: str,
) -> dict[str, object]:
    """Derive the only raw-pair record shape Root can bind for a receipt."""

    if not isinstance(receipt, ContractPerformanceEvidenceReceipt):
        raise TypeError(
            "contract performance dynamic record requires "
            "ContractPerformanceEvidenceReceipt"
        )
    _digest(producer_receipt_digest, "producer_receipt_digest")
    return {
        "sample_id": receipt.sample_id,
        "source_snapshot_digest": receipt.source_snapshot_digest,
        "source_case_digest": receipt.source_case_digest,
        "comparison_decision_id": receipt.comparison_decision_id,
        "contract_comparison_producer_receipt_digest": (
            receipt.contract_comparison_producer_receipt_digest
        ),
        "result_group_id": receipt.result_group_id,
        "result_group_digest": receipt.result_group_digest,
        "contract_fingerprint_digest": receipt.contract_fingerprint_digest,
        "endpoint_set_digest": receipt.endpoint_set_digest,
        "fingerprint_task_id": receipt.fingerprint_task_id,
        "fingerprint_result_digest": receipt.fingerprint_result_digest,
        "exact_task_id": receipt.exact_task_id,
        "exact_result_digest": receipt.exact_result_digest,
        "fingerprint_partition_digest": receipt.fingerprint_partition_digest,
        "canonical_partition_digest": receipt.canonical_partition_digest,
        "comparison_algorithm_digest": receipt.comparison_algorithm_digest,
        "benchmark_plan_digest": receipt.benchmark_plan_digest,
        "environment_digest": receipt.environment_digest,
        "baseline_config_digest": receipt.baseline_config_digest,
        "treatment_config_digest": receipt.treatment_config_digest,
        "baseline_task_id": receipt.baseline_task_id,
        "baseline_task_spec_digest": receipt.baseline_task_spec_digest,
        "baseline_result_digest": receipt.baseline_result_digest,
        "baseline_seed_lineage_digest": receipt.baseline_seed_lineage_digest,
        "baseline_execution_outcome_digest": (
            receipt.baseline_execution_outcome_digest
        ),
        "baseline_evidence_envelope_digest": (
            receipt.baseline_evidence_envelope_digest
        ),
        "treatment_task_id": receipt.treatment_task_id,
        "treatment_task_spec_digest": receipt.treatment_task_spec_digest,
        "treatment_result_digest": receipt.treatment_result_digest,
        "treatment_seed_lineage_digest": receipt.treatment_seed_lineage_digest,
        "treatment_execution_outcome_digest": (
            receipt.treatment_execution_outcome_digest
        ),
        "treatment_evidence_envelope_digest": (
            receipt.treatment_evidence_envelope_digest
        ),
        "baseline_materialized_bytes": receipt.baseline_materialized_bytes,
        "treatment_materialized_bytes": receipt.treatment_materialized_bytes,
        "baseline_backend_pair_comparisons": (
            receipt.baseline_backend_pair_comparisons
        ),
        "treatment_backend_pair_comparisons": (
            receipt.treatment_backend_pair_comparisons
        ),
        "baseline_comparison_cpu_ns": receipt.baseline_comparison_cpu_ns,
        "treatment_comparison_cpu_ns": receipt.treatment_comparison_cpu_ns,
        "baseline_strategy": receipt.baseline_strategy,
        "treatment_strategy": receipt.treatment_strategy,
        "producer_receipt_digest": producer_receipt_digest,
    }


def _parallel_scaling_dynamic_record(
    receipt: ParallelScalingEvidenceReceipt,
    *,
    producer_receipt_digest: str,
) -> dict[str, str]:
    """Derive the only parallel record shape Root can bind for a receipt."""

    if not isinstance(receipt, ParallelScalingEvidenceReceipt):
        raise TypeError(
            "parallel scaling dynamic record requires ParallelScalingEvidenceReceipt"
        )
    _digest(producer_receipt_digest, "producer_receipt_digest")
    return {
        "sample_id": receipt.sample_id,
        "workload_id": receipt.workload_id,
        "one_worker_task_set_digest": receipt.one_worker_task_set_digest,
        "six_worker_task_set_digest": receipt.six_worker_task_set_digest,
        "one_worker_result_digest": receipt.one_worker_result_digest,
        "six_worker_result_digest": receipt.six_worker_result_digest,
        "producer_receipt_digest": producer_receipt_digest,
    }


def _reconstruct_contract_comparison_admission(
    admission: VerifiedTypedAdmission,
    *,
    errors: list[str],
) -> ContractComparisonReceipt | None:
    """Re-read one retained admission before Root uses its comparison context."""

    admission_id = admission.admission_id
    if (
        admission.envelope_type != "ContractComparisonReceipt"
        or admission.envelope_schema_version
        != "osc-phase6-contract-comparison-receipt-v1"
        or admission.subject_kind != "contract_comparison_decision_ids"
    ):
        errors.append(f"contract_comparison_admission_shape_mismatch:{admission_id}")
        return None
    try:
        raw = Path(admission.envelope_path).read_bytes()
    except OSError as exc:
        errors.append(
            f"contract_comparison_admission_load_failed:{admission_id}:"
            f"{type(exc).__name__}:{exc}"
        )
        return None
    if _sha256_bytes(raw) != admission.byte_sha256:
        errors.append(f"contract_comparison_admission_hash_mismatch:{admission_id}")
        return None
    try:
        decoded = decode_canonical_envelope(raw.decode())
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        errors.append(
            f"contract_comparison_admission_envelope_invalid:{admission_id}:"
            f"{type(exc).__name__}:{exc}"
        )
        return None
    if (
        decoded.get("type") != admission.envelope_type
        or decoded.get("schema_version") != admission.envelope_schema_version
    ):
        errors.append(f"contract_comparison_admission_envelope_mismatch:{admission_id}")
        return None
    if (
        stable_digest("osc-root-typed-admission-envelope", decoded)
        != admission.envelope_digest
    ):
        errors.append(f"contract_comparison_admission_digest_mismatch:{admission_id}")
        return None
    try:
        receipt = _reconstruct_contract_comparison_receipt_payload(decoded["payload"])
    except Exception as exc:  # Root must fail closed, not propagate bad evidence.
        errors.append(
            f"contract_comparison_admission_reconstruction_failed:{admission_id}:"
            f"{type(exc).__name__}"
        )
        return None
    if admission.subject_ids != (receipt.comparison_decision_id,):
        errors.append(f"contract_comparison_admission_subject_mismatch:{admission_id}")
        return None
    return receipt


def _reconstruct_contract_performance_admission(
    admission: VerifiedTypedAdmission,
    *,
    errors: list[str],
) -> ContractPerformanceEvidenceReceipt | None:
    """Re-read one retained raw-pair admission before Root uses its context."""

    admission_id = admission.admission_id
    if (
        admission.envelope_type != "ContractPerformanceEvidenceReceipt"
        or admission.envelope_schema_version
        != CONTRACT_PERFORMANCE_EVIDENCE_RECEIPT_SCHEMA_VERSION
        or admission.subject_kind != "contract_performance_samples"
    ):
        errors.append(f"contract_performance_admission_shape_mismatch:{admission_id}")
        return None
    try:
        raw = Path(admission.envelope_path).read_bytes()
    except OSError as exc:
        errors.append(
            f"contract_performance_admission_load_failed:{admission_id}:"
            f"{type(exc).__name__}:{exc}"
        )
        return None
    if _sha256_bytes(raw) != admission.byte_sha256:
        errors.append(f"contract_performance_admission_hash_mismatch:{admission_id}")
        return None
    try:
        decoded = decode_canonical_envelope(raw.decode())
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        errors.append(
            f"contract_performance_admission_envelope_invalid:{admission_id}:"
            f"{type(exc).__name__}:{exc}"
        )
        return None
    if (
        decoded.get("type") != admission.envelope_type
        or decoded.get("schema_version") != admission.envelope_schema_version
    ):
        errors.append(f"contract_performance_admission_envelope_mismatch:{admission_id}")
        return None
    if (
        stable_digest("osc-root-typed-admission-envelope", decoded)
        != admission.envelope_digest
    ):
        errors.append(f"contract_performance_admission_digest_mismatch:{admission_id}")
        return None
    try:
        receipt = _reconstruct_contract_performance_evidence_receipt_payload(
            decoded["payload"]
        )
    except Exception as exc:  # Root must fail closed, not propagate bad evidence.
        errors.append(
            f"contract_performance_admission_reconstruction_failed:{admission_id}:"
            f"{type(exc).__name__}"
        )
        return None
    if admission.subject_ids != (receipt.sample_id,):
        errors.append(f"contract_performance_admission_subject_mismatch:{admission_id}")
        return None
    return receipt


def _reconstruct_parallel_scaling_admission(
    admission: VerifiedTypedAdmission,
    *,
    errors: list[str],
) -> ParallelScalingEvidenceReceipt | None:
    """Re-read one retained parallel receipt before Root uses its context."""

    admission_id = admission.admission_id
    if (
        admission.envelope_type != "ParallelScalingEvidenceReceipt"
        or admission.envelope_schema_version
        != PARALLEL_SCALING_EVIDENCE_RECEIPT_SCHEMA_VERSION
        or admission.subject_kind != "parallel_scaling_samples"
    ):
        errors.append(f"parallel_scaling_admission_shape_mismatch:{admission_id}")
        return None
    try:
        raw = Path(admission.envelope_path).read_bytes()
    except OSError as exc:
        errors.append(
            f"parallel_scaling_admission_load_failed:{admission_id}:"
            f"{type(exc).__name__}:{exc}"
        )
        return None
    if _sha256_bytes(raw) != admission.byte_sha256:
        errors.append(f"parallel_scaling_admission_hash_mismatch:{admission_id}")
        return None
    try:
        decoded = decode_canonical_envelope(raw.decode())
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        errors.append(
            f"parallel_scaling_admission_envelope_invalid:{admission_id}:"
            f"{type(exc).__name__}:{exc}"
        )
        return None
    if (
        decoded.get("type") != admission.envelope_type
        or decoded.get("schema_version") != admission.envelope_schema_version
    ):
        errors.append(f"parallel_scaling_admission_envelope_mismatch:{admission_id}")
        return None
    if (
        stable_digest("osc-root-typed-admission-envelope", decoded)
        != admission.envelope_digest
    ):
        errors.append(f"parallel_scaling_admission_digest_mismatch:{admission_id}")
        return None
    try:
        receipt = _reconstruct_parallel_scaling_evidence_receipt_payload(
            decoded["payload"]
        )
    except Exception as exc:  # Root must fail closed, not propagate bad evidence.
        errors.append(
            f"parallel_scaling_admission_reconstruction_failed:{admission_id}:"
            f"{type(exc).__name__}"
        )
        return None
    if admission.subject_ids != (receipt.sample_id,):
        errors.append(f"parallel_scaling_admission_subject_mismatch:{admission_id}")
        return None
    return receipt


def _verify_contract_comparison_dynamic_bindings(
    *,
    source_digest: str,
    records: tuple[dict[str, Any], ...],
    producer_receipt: VerifiedProducerReceipt | None,
    errors: list[str],
) -> None:
    """Bind the private receipt to one exact dynamic decision record per ID."""

    if producer_receipt is None:
        errors.append("contract_comparison_producer_receipt_missing")
        return
    if producer_receipt.producer_kind != "contract_exact_replay":
        errors.append("contract_comparison_producer_kind_mismatch")
        return

    records_by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        decision_id = record.get("comparison_decision_id")
        if not isinstance(decision_id, str) or not decision_id:
            errors.append("contract_comparison_record_identity_invalid")
            continue
        if decision_id in records_by_id:
            errors.append(f"contract_comparison_duplicate_dynamic_record:{decision_id}")
            continue
        records_by_id[decision_id] = record

    receipts_by_id: dict[str, ContractComparisonReceipt] = {}
    for admission in producer_receipt.admissions:
        if admission.subject_kind != "contract_comparison_decision_ids":
            continue
        receipt = _reconstruct_contract_comparison_admission(admission, errors=errors)
        if receipt is None:
            continue
        decision_id = receipt.comparison_decision_id
        if decision_id in receipts_by_id:
            errors.append(
                f"contract_comparison_duplicate_decision_admission:{decision_id}"
            )
            continue
        receipts_by_id[decision_id] = receipt

    for decision_id in sorted(set(records_by_id) - set(receipts_by_id)):
        errors.append(f"contract_comparison_record_not_admitted:{decision_id}")
    for decision_id in sorted(set(receipts_by_id) - set(records_by_id)):
        errors.append(f"contract_comparison_admission_not_planned:{decision_id}")

    for decision_id in sorted(set(records_by_id) & set(receipts_by_id)):
        receipt = receipts_by_id[decision_id]
        record = records_by_id[decision_id]
        if receipt.source_snapshot_digest != source_digest:
            errors.append(
                f"contract_comparison_receipt_source_snapshot_mismatch:{decision_id}"
            )
            continue
        expected = _contract_comparison_dynamic_record(
            receipt,
            producer_receipt_digest=producer_receipt.digest,
        )
        for field in (
            "comparison_decision_id",
            "source_snapshot_digest",
            "source_case_digest",
            "result_group_id",
            "result_group_digest",
            "contract_fingerprint_digest",
            "endpoint_set_digest",
            "fingerprint_task_id",
            "fingerprint_result_digest",
            "exact_task_id",
            "exact_result_digest",
            "fingerprint_partition_digest",
            "canonical_partition_digest",
            "producer_receipt_digest",
        ):
            if record.get(field) != expected[field]:
                errors.append(
                    f"contract_comparison_record_binding_mismatch:{decision_id}:{field}"
                )
                break


def _verify_contract_performance_dynamic_bindings(
    *,
    source_digest: str,
    records: tuple[dict[str, Any], ...],
    comparison_records: tuple[dict[str, Any], ...],
    comparison_producer_receipt: VerifiedProducerReceipt | None,
    performance_producer_receipt: VerifiedProducerReceipt | None,
    errors: list[str],
) -> None:
    """Bind every raw pair to exact Runtime and R2 comparison contexts."""

    if performance_producer_receipt is None:
        errors.append("contract_performance_producer_receipt_missing")
        return
    if performance_producer_receipt.producer_kind != "performance_paired_replay":
        errors.append("contract_performance_producer_kind_mismatch")
        return
    if comparison_producer_receipt is None:
        errors.append("contract_performance_comparison_producer_receipt_missing")
        return
    if comparison_producer_receipt.producer_kind != "contract_exact_replay":
        errors.append("contract_performance_comparison_producer_kind_mismatch")
        return

    records_by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            errors.append("contract_performance_record_identity_invalid")
            continue
        if sample_id in records_by_id:
            errors.append(f"contract_performance_duplicate_dynamic_record:{sample_id}")
            continue
        records_by_id[sample_id] = record

    comparison_records_by_id: dict[str, dict[str, Any]] = {}
    for record in comparison_records:
        decision_id = record.get("comparison_decision_id")
        if not isinstance(decision_id, str) or not decision_id:
            errors.append("contract_performance_comparison_record_identity_invalid")
            continue
        if decision_id in comparison_records_by_id:
            errors.append(
                f"contract_performance_duplicate_comparison_record:{decision_id}"
            )
            continue
        comparison_records_by_id[decision_id] = record

    receipts_by_id: dict[str, ContractPerformanceEvidenceReceipt] = {}
    decision_ids: set[str] = set()
    for admission in performance_producer_receipt.admissions:
        if admission.subject_kind != "contract_performance_samples":
            continue
        receipt = _reconstruct_contract_performance_admission(admission, errors=errors)
        if receipt is None:
            continue
        sample_id = receipt.sample_id
        if sample_id in receipts_by_id:
            errors.append(
                f"contract_performance_duplicate_sample_admission:{sample_id}"
            )
            continue
        if receipt.comparison_decision_id in decision_ids:
            errors.append(
                "contract_performance_duplicate_comparison_decision:"
                f"{receipt.comparison_decision_id}"
            )
            continue
        receipts_by_id[sample_id] = receipt
        decision_ids.add(receipt.comparison_decision_id)

    for sample_id in sorted(set(records_by_id) - set(receipts_by_id)):
        errors.append(f"contract_performance_record_not_admitted:{sample_id}")
    for sample_id in sorted(set(receipts_by_id) - set(records_by_id)):
        errors.append(f"contract_performance_admission_not_planned:{sample_id}")

    comparison_fields = (
        "comparison_decision_id",
        "source_snapshot_digest",
        "source_case_digest",
        "result_group_id",
        "result_group_digest",
        "contract_fingerprint_digest",
        "endpoint_set_digest",
        "fingerprint_task_id",
        "fingerprint_result_digest",
        "exact_task_id",
        "exact_result_digest",
        "fingerprint_partition_digest",
        "canonical_partition_digest",
    )
    for sample_id in sorted(set(records_by_id) & set(receipts_by_id)):
        receipt = receipts_by_id[sample_id]
        record = records_by_id[sample_id]
        if receipt.source_snapshot_digest != source_digest:
            errors.append(
                "contract_performance_receipt_source_snapshot_mismatch:"
                f"{sample_id}"
            )
            continue
        if (
            receipt.contract_comparison_producer_receipt_digest
            != comparison_producer_receipt.digest
        ):
            errors.append(
                "contract_performance_comparison_producer_digest_mismatch:"
                f"{sample_id}"
            )
            continue
        comparison_record = comparison_records_by_id.get(
            receipt.comparison_decision_id
        )
        if comparison_record is None:
            errors.append(
                "contract_performance_comparison_record_missing:"
                f"{receipt.comparison_decision_id}"
            )
            continue
        if (
            comparison_record.get("producer_receipt_digest")
            != comparison_producer_receipt.digest
        ):
            errors.append(
                "contract_performance_comparison_record_producer_mismatch:"
                f"{receipt.comparison_decision_id}"
            )
            continue
        for field in comparison_fields:
            if getattr(receipt, field) != comparison_record.get(field):
                errors.append(
                    "contract_performance_comparison_binding_mismatch:"
                    f"{sample_id}:{field}"
                )
                break
        else:
            expected = _contract_performance_dynamic_record(
                receipt,
                producer_receipt_digest=performance_producer_receipt.digest,
            )
            for field, value in expected.items():
                if record.get(field) != value:
                    errors.append(
                        "contract_performance_record_binding_mismatch:"
                        f"{sample_id}:{field}"
                    )
                    break


def _verify_parallel_scaling_dynamic_bindings(
    *,
    source_digest: str,
    records: tuple[dict[str, Any], ...],
    performance_producer_receipt: VerifiedProducerReceipt | None,
    errors: list[str],
) -> None:
    """Bind every 1-versus-6 receipt to one exact planned dynamic record."""

    if performance_producer_receipt is None:
        errors.append("parallel_scaling_producer_receipt_missing")
        return
    if performance_producer_receipt.producer_kind != "performance_paired_replay":
        errors.append("parallel_scaling_producer_kind_mismatch")
        return

    records_by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            errors.append("parallel_scaling_record_identity_invalid")
            continue
        if sample_id in records_by_id:
            errors.append(f"parallel_scaling_duplicate_dynamic_record:{sample_id}")
            continue
        records_by_id[sample_id] = record

    receipts_by_id: dict[str, ParallelScalingEvidenceReceipt] = {}
    for admission in performance_producer_receipt.admissions:
        if admission.subject_kind != "parallel_scaling_samples":
            continue
        receipt = _reconstruct_parallel_scaling_admission(admission, errors=errors)
        if receipt is None:
            continue
        sample_id = receipt.sample_id
        if sample_id in receipts_by_id:
            errors.append(
                f"parallel_scaling_duplicate_sample_admission:{sample_id}"
            )
            continue
        receipts_by_id[sample_id] = receipt

    for sample_id in sorted(set(records_by_id) - set(receipts_by_id)):
        errors.append(f"parallel_scaling_record_not_admitted:{sample_id}")
    for sample_id in sorted(set(receipts_by_id) - set(records_by_id)):
        errors.append(f"parallel_scaling_admission_not_planned:{sample_id}")

    for sample_id in sorted(set(records_by_id) & set(receipts_by_id)):
        receipt = receipts_by_id[sample_id]
        record = records_by_id[sample_id]
        if receipt.source_snapshot_digest != source_digest:
            errors.append(
                "parallel_scaling_receipt_source_snapshot_mismatch:"
                f"{sample_id}"
            )
            continue
        expected = _parallel_scaling_dynamic_record(
            receipt,
            producer_receipt_digest=performance_producer_receipt.digest,
        )
        for field, value in expected.items():
            if record.get(field) != value:
                errors.append(
                    "parallel_scaling_record_binding_mismatch:"
                    f"{sample_id}:{field}"
                )
                break


@dataclass(frozen=True, slots=True)
class RootGateBuildResult:
    source_digest: str
    compiled_universe_digest: str
    frozen_spec_set_digest: str
    denominator_universes: tuple[IdentityUniverse, ...]
    artifact_receipts: tuple[VerifiedArtifactReceipt, ...]
    authority_plan_json: str
    errors: tuple[str, ...]
    publishable: bool
    twenty_four_hour_run_authorized: bool = False
    schema_version: str = BUILD_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        minimally_publishable = bool(
            not self.errors
            and self.authority_plan_json
            and self.denominator_universes
            and self.artifact_receipts
        )
        if self.publishable != minimally_publishable:
            raise ValueError("Root gate build publication state is inconsistent")
        if self.publishable:
            keys = tuple(item.metric_key for item in self.denominator_universes)
            kinds = tuple(item.producer_kind for item in self.artifact_receipts)
            if (
                len(keys) != 38
                or keys != tuple(sorted(keys))
                or len(keys) != len(set(keys))
                or set(kinds) != _REQUIRED_PRODUCER_KINDS
                or len(kinds) != len(_REQUIRED_PRODUCER_KINDS)
                or self.compiled_universe_digest != EXPECTED_COMPILED_UNIVERSE_DIGEST
                or self.frozen_spec_set_digest != EXPECTED_GATE_SPEC_SET_DIGEST
            ):
                raise ValueError("Root gate build lacks the frozen 41/38 authority set")
            try:
                payload = _strict_json_loads(self.authority_plan_json.encode())
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                raise ValueError("Root gate build authority JSON is invalid") from exc
            if not isinstance(payload, dict) or len(payload.get("denominator_universes", ())) != 38:
                raise ValueError("Root gate build authority JSON is incomplete")
        if self.twenty_four_hour_run_authorized:
            raise ValueError("Root gate builds cannot authorize a 24h run")

    @property
    def digest(self) -> str:
        return stable_digest("osc-root-gate-build-result", self)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


def build_phase6_gate_authority_bundle(
    *,
    source: VerifiedSourceSnapshot,
    templates: tuple[TargetTemplate, ...],
    dynamic_plan: VerifiedDynamicGatePlan,
    artifact_receipts: VerifiedArtifactReceiptIndex,
) -> RootGateBuildResult:
    """Build an unpublished authority payload without accepting a raw manifest."""

    errors: list[str] = []
    source_valid = False
    if isinstance(source, VerifiedSourceSnapshot):
        try:
            source_valid = source.valid
        except Exception as exc:
            errors.append(
                f"trusted_source_snapshot_verification_failed:"
                f"{type(exc).__name__}:{exc}"
            )
    source_digest = (
        source.source_digest if isinstance(source, VerifiedSourceSnapshot) else "unverified-source"
    )
    if not source_valid:
        errors.append("trusted_source_snapshot_invalid")
    try:
        static = derive_static_gate_universes(source=source, templates=templates)
    except (TypeError, ValueError) as exc:
        # Template/compiler failures are trust failures, never reasons to throw a
        # partially constructed authority result at a caller.
        errors.append(f"static_gate_universe_derivation_failed:{type(exc).__name__}:{exc}")
        static = derive_static_gate_universes(source=source, templates=())
    errors.extend(static.verification_errors)

    dynamic_valid = False
    if isinstance(dynamic_plan, VerifiedDynamicGatePlan):
        try:
            dynamic_valid = dynamic_plan.valid
        except Exception as exc:
            errors.append(
                f"trusted_dynamic_plan_verification_failed:"
                f"{type(exc).__name__}:{exc}"
            )
    if not dynamic_valid:
        errors.append("trusted_dynamic_plan_invalid")
        if isinstance(dynamic_plan, VerifiedDynamicGatePlan):
            errors.extend(dynamic_plan.verification_errors)
    receipt_valid = False
    if isinstance(artifact_receipts, VerifiedArtifactReceiptIndex):
        try:
            receipt_valid = artifact_receipts.valid
        except Exception as exc:
            errors.append(
                f"trusted_artifact_receipts_verification_failed:"
                f"{type(exc).__name__}:{exc}"
            )
    if not receipt_valid:
        errors.append("trusted_artifact_receipts_invalid")
        if isinstance(artifact_receipts, VerifiedArtifactReceiptIndex):
            errors.extend(artifact_receipts.verification_errors)

    dynamic_source = (
        dynamic_plan.source_digest
        if isinstance(dynamic_plan, VerifiedDynamicGatePlan)
        else "unverified-dynamic-source"
    )
    dynamic_digest = (
        dynamic_plan.digest
        if isinstance(dynamic_plan, VerifiedDynamicGatePlan)
        else "unverified-dynamic-plan"
    )
    dynamic_ids = (
        dynamic_plan.identity_map()
        if isinstance(dynamic_plan, VerifiedDynamicGatePlan)
        else {}
    )
    dynamic_records = (
        dynamic_plan.record_map()
        if isinstance(dynamic_plan, VerifiedDynamicGatePlan)
        else {}
    )
    receipt_source = (
        artifact_receipts.source_digest
        if isinstance(artifact_receipts, VerifiedArtifactReceiptIndex)
        else "unverified-receipt-source"
    )
    receipt_digest = (
        artifact_receipts.digest
        if isinstance(artifact_receipts, VerifiedArtifactReceiptIndex)
        else "unverified-artifact-receipts"
    )
    receipt_artifacts = (
        artifact_receipts.artifacts
        if isinstance(artifact_receipts, VerifiedArtifactReceiptIndex)
        else ()
    )
    producer_map = (
        artifact_receipts.producer_map()
        if isinstance(artifact_receipts, VerifiedArtifactReceiptIndex)
        else {}
    )
    if dynamic_source != source_digest:
        errors.append("dynamic_plan_rebound_to_source")
    if receipt_source != source_digest:
        errors.append("artifact_receipts_rebound_to_source")
    if isinstance(dynamic_plan, VerifiedDynamicGatePlan) and isinstance(
        artifact_receipts, VerifiedArtifactReceiptIndex
    ):
        if artifact_receipts.dynamic_plan_sha256 != dynamic_plan.byte_sha256:
            errors.append("artifact_receipts_rebound_to_dynamic_plan")
        if (
            dynamic_plan.producer_receipts_digest
            != artifact_receipts.producer_receipts_digest
        ):
            errors.append("dynamic_plan_producer_receipt_set_mismatch")
        for section, producer_kind in sorted(_DYNAMIC_SECTION_PRODUCER_KIND.items()):
            receipt = producer_map.get(producer_kind)
            records = dynamic_records.get(section, ())
            if receipt is None:
                errors.append(f"dynamic_section_receipt_missing:{section}")
                continue
            if any(
                record.get("producer_receipt_digest") != receipt.digest
                for record in records
            ):
                errors.append(f"dynamic_section_receipt_digest_mismatch:{section}")
            planned = dynamic_ids.get(section, ())
            admitted = receipt.subjects(
                _DYNAMIC_SECTION_TYPED_SUBJECT_KIND.get(section, section)
            )
            if section == "candidate_records" and not planned:
                # A zero-candidate claim still requires classification receipts
                # for every executed V3 case; absence is not evidence of zero.
                admitted = receipt.subjects("classified_v3_cases")
                planned = dynamic_ids.get("v3_executed_cases", ())
            if admitted != planned:
                errors.append(f"dynamic_section_typed_subject_mismatch:{section}")
        _verify_contract_comparison_dynamic_bindings(
            source_digest=source_digest,
            records=dynamic_records.get("contract_comparison_decisions", ()),
            producer_receipt=producer_map.get("contract_exact_replay"),
            errors=errors,
        )
        _verify_contract_performance_dynamic_bindings(
            source_digest=source_digest,
            records=dynamic_records.get("contract_performance_samples", ()),
            comparison_records=dynamic_records.get(
                "contract_comparison_decisions", ()
            ),
            comparison_producer_receipt=producer_map.get("contract_exact_replay"),
            performance_producer_receipt=producer_map.get(
                "performance_paired_replay"
            ),
            errors=errors,
        )
        _verify_parallel_scaling_dynamic_bindings(
            source_digest=source_digest,
            records=dynamic_records.get("parallel_scaling_samples", ()),
            performance_producer_receipt=producer_map.get(
                "performance_paired_replay"
            ),
            errors=errors,
        )

    universes = static.universe_map()
    static_subject_producers = {
        "confirmed_roots_total": "contract_exact_replay",
        "high_risk_mutants_total": "contract_exact_replay",
        "parallel_invariance_checks_total": "parallel_authority_replay",
        "cells_activation_denominator": "reachability_admission_replay",
        "edges_activation_denominator": "reachability_admission_replay",
        "fresh_cells_reachability_denominator": "reachability_admission_replay",
        "fresh_family_reachability_denominator": "reachability_admission_replay",
        "edge_observation_denominator": "reachability_admission_replay",
        "scheduled_activation_family_denominator": "reachability_admission_replay",
        "mutation_family_denominator": "reachability_admission_replay",
    }
    for metric_key, universe in sorted(universes.items()):
        producer_kind = static_subject_producers.get(
            metric_key, "coverage_admission_replay"
        )
        receipt = producer_map.get(producer_kind)
        admitted = receipt.subjects(metric_key) if receipt is not None else ()
        if admitted != universe.record_ids:
            errors.append(f"static_universe_typed_subject_mismatch:{metric_key}")
    for metric_key, section in sorted(_DYNAMIC_UNIVERSE_KEYS.items()):
        universes[metric_key] = _universe(
            metric_key,
            dynamic_ids.get(section, ()),
            source_digest=source_digest,
            producer_digest=dynamic_digest,
        )
    unit_id = stable_digest(
        "osc-root-unit-denominator-id",
        {
            "source_digest": source_digest,
            "dynamic_plan_digest": dynamic_digest,
            "artifact_receipt_index_digest": receipt_digest,
            "gate_spec_set_digest": EXPECTED_GATE_SPEC_SET_DIGEST,
        },
    )
    universes["unit_denominator"] = _universe(
        "unit_denominator",
        (unit_id,),
        source_digest=source_digest,
        producer_digest=dynamic_digest,
    )

    expected_keys = {
        item.denominator_key for item in default_pre24_gate_specs()
    }
    available_keys = set(universes)
    missing = sorted(expected_keys - available_keys)
    extra = sorted(available_keys - expected_keys)
    if missing:
        errors.append("root_denominator_universes_missing:" + ",".join(missing))
    if extra:
        errors.append("root_denominator_universes_extra:" + ",".join(extra))
    if len(expected_keys) != 38:
        errors.append("frozen_denominator_key_count_changed")
    if pre24_gate_spec_set_digest() != EXPECTED_GATE_SPEC_SET_DIGEST:
        errors.append("root_gate_spec_set_digest_changed")

    ordered_universes = tuple(universes[key] for key in sorted(expected_keys & available_keys))
    # Independently bind every raw metric series to the complete frozen gate set.
    expected_metric_keys = {
        key
        for spec in default_pre24_gate_specs()
        for key in (
            spec.numerator_key,
            spec.denominator_key,
            *((spec.alternative_numerator_key,) if spec.alternative_numerator_key else ()),
        )
    }
    series_by_key: dict[str, list[RawMetricSeries]] = {}
    for artifact in receipt_artifacts:
        for series in artifact.series:
            series_by_key.setdefault(series.metric_key, []).append(series)
    missing_metrics = sorted(expected_metric_keys - set(series_by_key))
    extra_metrics = sorted(set(series_by_key) - expected_metric_keys)
    duplicate_metrics = sorted(
        key for key, values in series_by_key.items() if len(values) != 1
    )
    if missing_metrics:
        errors.append("root_raw_metrics_missing:" + ",".join(missing_metrics))
    if extra_metrics:
        errors.append("root_raw_metrics_extra:" + ",".join(extra_metrics))
    if duplicate_metrics:
        errors.append("root_raw_metrics_duplicate:" + ",".join(duplicate_metrics))
    scalar_reducers = {
        "contract_compile_match_p95_ms": MetricReducer.NEAREST_RANK_P95,
        "contract_compile_match_wall_share": MetricReducer.NEAREST_RANK_P95,
        "paired_throughput_regression": MetricReducer.MAXIMUM,
        "six_worker_parallel_efficiency": MetricReducer.MINIMUM,
    }
    for key, values in series_by_key.items():
        expected_reducer = scalar_reducers.get(key, MetricReducer.COUNT_UNIQUE)
        if any(series.reducer is not expected_reducer for series in values):
            errors.append(f"root_raw_metric_reducer_mismatch:{key}")
    for key, universe in universes.items():
        series_values = series_by_key.get(key, ())
        record_ids = tuple(
            record.record_id for series in series_values for record in series.records
        )
        if (
            len(record_ids) != len(set(record_ids))
            or identity_universe_digest(record_ids) != universe.identity_digest
        ):
            errors.append(f"root_raw_denominator_identity_mismatch:{key}")

    # Plan identities that refer to compiler-owned targets must be real cells
    # with the matching family, never count-shaped placeholders.
    cell_family = {
        item.target_cell_id: item.test_family_id
        for item in (
            *static.compiled_universe.fresh_cells,
            *static.compiled_universe.regression_cells,
        )
    }
    for section in ("scheduled_target_attempts", "mutation_attempts"):
        for record in dynamic_records.get(section, ()):
            target_id = record.get("target_id", record.get("cell_id"))
            family_id = record.get("family_id")
            if cell_family.get(target_id) != family_id:
                errors.append(f"dynamic_plan_target_family_mismatch:{section}")

    artifact_values = [
        {
            "artifact_id": item.artifact_id,
            "sha256": item.byte_sha256,
            "schema_version": RAW_GATE_ARTIFACT_SCHEMA_VERSION,
            "provenance_digest": stable_digest(
                "osc-root-artifact-provenance",
                {
                    "source_digest": source_digest,
                    "producer_kind": item.producer_kind,
                    "producer_receipt_digest": item.producer_receipt_digest,
                    "declared_provenance_digest": item.provenance_digest,
                    "artifact_sha256": item.byte_sha256,
                },
            ),
        }
        for item in receipt_artifacts
    ]
    payload = {
        "schema_version": GATE_AUTHORITY_PLAN_SCHEMA_VERSION,
        "source_digest": source_digest,
        "gate_spec_set_digest": EXPECTED_GATE_SPEC_SET_DIGEST,
        "artifacts": artifact_values,
        "denominator_universes": [
            item.authority_payload() for item in ordered_universes
        ],
    }
    authority_json = canonical_json(payload)
    unique_errors = tuple(dict.fromkeys(errors))
    publishable = bool(
        not unique_errors
        and len(ordered_universes) == 38
        and artifact_values
    )
    return RootGateBuildResult(
        source_digest=source_digest,
        compiled_universe_digest=static.compiled_universe.digest,
        frozen_spec_set_digest=EXPECTED_GATE_SPEC_SET_DIGEST,
        denominator_universes=ordered_universes,
        artifact_receipts=receipt_artifacts,
        authority_plan_json=authority_json,
        errors=unique_errors,
        publishable=publishable,
        twenty_four_hour_run_authorized=False,
    )


@dataclass(frozen=True, slots=True)
class PublishedGateAuthorityBundle:
    authority_plan_path: str
    authority_plan_sha256: str
    authority_plan: VerifiedGateAuthorityPlan
    build_result_digest: str
    twenty_four_hour_run_authorized: bool = False

    def __post_init__(self) -> None:
        if self.twenty_four_hour_run_authorized:
            raise ValueError("published gate bundles cannot authorize a 24h run")


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_phase6_gate_authority_bundle(
    *,
    result: RootGateBuildResult,
    source: VerifiedSourceSnapshot,
    templates: tuple[TargetTemplate, ...],
    dynamic_plan: VerifiedDynamicGatePlan,
    artifact_receipts: VerifiedArtifactReceiptIndex,
    output_path: Path,
) -> PublishedGateAuthorityBundle:
    if not isinstance(result, RootGateBuildResult):
        raise TypeError("Root gate publication requires a build result")
    fresh = build_phase6_gate_authority_bundle(
        source=source,
        templates=templates,
        dynamic_plan=dynamic_plan,
        artifact_receipts=artifact_receipts,
    )
    if fresh != result or not fresh.publishable:
        raise ValueError("non-publishable Root gate build cannot create authority")
    requested_destination = Path(output_path)
    destination = requested_destination.parent.resolve() / requested_destination.name
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"authority plan already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (result.authority_plan_json + "\n").encode()
    plan_sha = _sha256_bytes(payload)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".osc-phase6-authority-",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    temporary_identity: tuple[int, int] | None = None
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_stat = temporary.stat()
        temporary_identity = (temporary_stat.st_dev, temporary_stat.st_ino)
        verified = verify_gate_authority_plan(
            temporary,
            expected_sha256=plan_sha,
            expected_source_digest=result.source_digest,
        )
        if not verified.valid:
            raise ValueError(
                "new Root authority plan failed Runtime verification: "
                + ";".join(verified.verification_errors)
            )
        os.link(temporary, destination)
        _fsync_directory(destination.parent)
        published = verify_gate_authority_plan(
            destination,
            expected_sha256=plan_sha,
            expected_source_digest=fresh.source_digest,
        )
        if not published.valid or published.byte_sha256 != plan_sha:
            raise ValueError("published authority plan did not survive revalidation")
        temporary.unlink()
        return PublishedGateAuthorityBundle(
            str(destination),
            plan_sha,
            published,
            fresh.digest,
            False,
        )
    except BaseException as exc:
        cleanup_errors: list[str] = []
        owns_destination = False
        if temporary_identity is not None and not isinstance(exc, FileExistsError):
            try:
                destination_stat = destination.lstat()
                owns_destination = owns_destination or (
                    not destination.is_symlink()
                    and
                    (destination_stat.st_dev, destination_stat.st_ino)
                    == temporary_identity
                )
            except FileNotFoundError:
                pass
            except OSError as ownership_exc:
                cleanup_errors.append(
                    f"destination_ownership_check_failed:"
                    f"{type(ownership_exc).__name__}:{ownership_exc}"
                )
        if owns_destination:
            try:
                destination.unlink()
                _fsync_directory(destination.parent)
            except OSError as cleanup_exc:
                cleanup_errors.append(
                    f"destination_rollback_failed:{type(cleanup_exc).__name__}:"
                    f"{cleanup_exc}"
                )
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        except OSError as cleanup_exc:
            cleanup_errors.append(
                f"temporary_cleanup_failed:{type(cleanup_exc).__name__}:{cleanup_exc}"
            )
        if cleanup_errors:
            raise RuntimeError(";".join(cleanup_errors)) from exc
        raise


@dataclass(frozen=True, slots=True)
class RootGatePreflightReport:
    source_digest: str
    build_result_digest: str
    authority_plan_sha256: str
    artifact_manifest_sha256: str
    gate_report_digest: str
    verification_errors: tuple[str, ...]
    full_gate_count: int
    all_gates_pass: bool
    phase6_gate_eligible: bool
    twenty_four_hour_run_authorized: bool = False
    schema_version: str = PREFLIGHT_REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.twenty_four_hour_run_authorized:
            raise ValueError("Root preflight cannot authorize a 24h run")
        if self.phase6_gate_eligible and (
            self.verification_errors
            or not self.all_gates_pass
            or self.full_gate_count != 41
        ):
            raise ValueError("Phase6 preflight eligibility is inconsistent")

    @property
    def digest(self) -> str:
        return stable_digest("osc-root-gate-preflight", self)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


def preflight_phase6_gate_authority_bundle(
    *,
    source: VerifiedSourceSnapshot,
    templates: tuple[TargetTemplate, ...],
    dynamic_plan: VerifiedDynamicGatePlan,
    artifact_receipts: VerifiedArtifactReceiptIndex,
    authority_plan_path: Path,
    expected_authority_plan_sha256: str,
    artifact_manifest_path: Path,
    expected_artifact_manifest_sha256: str,
) -> RootGatePreflightReport:
    """Recompute trust inputs before invoking the complete Runtime gate chain."""

    _require_sha256(
        expected_authority_plan_sha256, "expected authority plan SHA-256"
    )
    _require_sha256(
        expected_artifact_manifest_sha256, "expected artifact manifest SHA-256"
    )
    errors: list[str] = []
    source_digest = (
        source.source_digest
        if isinstance(source, VerifiedSourceSnapshot)
        else "unverified-source"
    )
    build = build_phase6_gate_authority_bundle(
        source=source,
        templates=templates,
        dynamic_plan=dynamic_plan,
        artifact_receipts=artifact_receipts,
    )
    errors.extend(build.errors)
    if not build.publishable:
        errors.append("root_gate_build_not_publishable")
    expected_build_bytes = (build.authority_plan_json + "\n").encode()
    expected_build_sha = _sha256_bytes(expected_build_bytes)
    if expected_build_sha != expected_authority_plan_sha256:
        errors.append("authority_plan_sha_not_equal_to_root_recomputation")
    authority = None
    try:
        authority = verify_gate_authority_plan(
            authority_plan_path,
            expected_sha256=expected_authority_plan_sha256,
            expected_source_digest=source_digest,
        )
        errors.extend(authority.verification_errors)
        if authority.byte_sha256 != expected_build_sha:
            errors.append("authority_plan_snapshot_not_equal_to_root_recomputation")
    except Exception as exc:
        errors.append(f"authority_plan_verification_failed:{type(exc).__name__}:{exc}")

    manifest_path = None
    try:
        manifest_path = artifact_manifest_path.resolve()
        manifest_sha = _sha256_path(manifest_path)
    except (OSError, RuntimeError) as exc:
        manifest_sha = "0" * 64
        errors.append(f"artifact_manifest_load_failed:{type(exc).__name__}:{exc}")
    if manifest_sha != expected_artifact_manifest_sha256:
        errors.append("artifact_manifest_external_hash_mismatch")
    evidence = None
    if authority is not None and manifest_path is not None:
        try:
            evidence = verify_gate_artifact_manifest(
                manifest_path,
                expected_source_digest=source_digest,
                expected_authority_plan_sha256=expected_authority_plan_sha256,
                authority_plan=authority,
            )
            errors.extend(evidence.verification_errors)
            if evidence.manifest_sha256 != expected_artifact_manifest_sha256:
                errors.append("runtime_evidence_manifest_sha_mismatch")
        except Exception as exc:  # fail closed across Runtime filesystem races
            errors.append(f"runtime_evidence_verification_failed:{type(exc).__name__}:{exc}")
    gate_report = None
    if authority is not None and evidence is not None:
        try:
            gate_report = calculate_gate_report(
                evidence,
                expected_source_digest=source_digest,
                expected_authority_plan_sha256=expected_authority_plan_sha256,
                authority_plan=authority,
            )
            errors.extend(gate_report.verification_errors)
            if gate_report.artifact_manifest_sha256 != expected_artifact_manifest_sha256:
                errors.append("runtime_gate_report_manifest_sha_mismatch")
            if gate_report.authority_plan_sha256 != expected_authority_plan_sha256:
                errors.append("runtime_gate_report_authority_sha_mismatch")
        except Exception as exc:  # fail closed across Runtime filesystem races
            errors.append(f"runtime_gate_calculation_failed:{type(exc).__name__}:{exc}")

    for value, expected_type, label in (
        (source, VerifiedSourceSnapshot, "source_snapshot"),
        (dynamic_plan, VerifiedDynamicGatePlan, "dynamic_plan"),
        (artifact_receipts, VerifiedArtifactReceiptIndex, "artifact_receipts"),
    ):
        valid = False
        if isinstance(value, expected_type):
            try:
                valid = value.valid
            except Exception as exc:
                errors.append(
                    f"{label}_reverification_failed:{type(exc).__name__}:{exc}"
                )
        if not valid:
            errors.append(f"{label}_changed_during_preflight")
    if authority is not None:
        try:
            if _sha256_path(Path(authority.plan_path)) != expected_authority_plan_sha256:
                errors.append("authority_plan_changed_during_preflight")
        except (OSError, RuntimeError):
            errors.append("authority_plan_missing_after_preflight")
    if manifest_path is not None:
        try:
            if _sha256_path(manifest_path) != expected_artifact_manifest_sha256:
                errors.append("artifact_manifest_changed_during_preflight")
        except OSError:
            errors.append("artifact_manifest_missing_after_preflight")
    receipt_artifacts = (
        artifact_receipts.artifacts
        if isinstance(artifact_receipts, VerifiedArtifactReceiptIndex)
        else ()
    )
    for receipt in receipt_artifacts:
        try:
            if _sha256_path(Path(receipt.path)) != receipt.byte_sha256:
                errors.append(
                    f"raw_artifact_changed_during_preflight:{receipt.artifact_id}"
                )
        except OSError:
            errors.append(f"raw_artifact_missing_after_preflight:{receipt.artifact_id}")
    if isinstance(artifact_receipts, VerifiedArtifactReceiptIndex):
        for receipt in artifact_receipts.producer_receipts:
            for admission in receipt.admissions:
                try:
                    if _sha256_path(Path(admission.envelope_path)) != admission.byte_sha256:
                        errors.append(
                            f"typed_admission_changed_during_preflight:"
                            f"{admission.admission_id}"
                        )
                except OSError:
                    errors.append(
                        f"typed_admission_missing_after_preflight:{admission.admission_id}"
                    )

    unique_errors = tuple(dict.fromkeys(errors))
    eligible = bool(
        not unique_errors
        and build.publishable
        and authority is not None
        and authority.valid
        and evidence is not None
        and not evidence.verification_errors
        and evidence.artifacts
        and evidence.manifest_sha256 == expected_artifact_manifest_sha256
        and gate_report is not None
        and gate_report.authority_eligible
        and gate_report.artifact_manifest_sha256
        == expected_artifact_manifest_sha256
        and len(gate_report.results) == 41
    )
    gate_report_digest = (
        gate_report.digest
        if gate_report is not None
        else stable_digest("osc-root-missing-gate-report", unique_errors)
    )
    full_gate_count = len(gate_report.results) if gate_report is not None else 0
    all_gates_pass = gate_report.all_gates_pass if gate_report is not None else False
    runtime_manifest_sha = (
        evidence.manifest_sha256 if evidence is not None else manifest_sha
    )
    return RootGatePreflightReport(
        source_digest=source_digest,
        build_result_digest=build.digest,
        authority_plan_sha256=expected_authority_plan_sha256,
        artifact_manifest_sha256=runtime_manifest_sha,
        gate_report_digest=gate_report_digest,
        verification_errors=unique_errors,
        full_gate_count=full_gate_count,
        all_gates_pass=all_gates_pass,
        phase6_gate_eligible=eligible,
        twenty_four_hour_run_authorized=False,
    )


__all__ = [
    "ARTIFACT_RECEIPT_SCHEMA_VERSION",
    "BUILD_RESULT_SCHEMA_VERSION",
    "DYNAMIC_PLAN_SCHEMA_VERSION",
    "EXPECTED_COMPILED_UNIVERSE_DIGEST",
    "EXPECTED_GATE_SPEC_SET_DIGEST",
    "EXPECTED_PUBLIC_FREEZE_SHA256",
    "IdentityUniverse",
    "PREFLIGHT_REPORT_SCHEMA_VERSION",
    "RootGateBuildResult",
    "RootGatePreflightReport",
    "PublishedGateAuthorityBundle",
    "SOURCE_SNAPSHOT_SCHEMA_VERSION",
    "SourceFileBinding",
    "StaticGateUniverses",
    "TYPED_PRODUCER_RECEIPT_SCHEMA_VERSION",
    "VerifiedArtifactReceiptIndex",
    "VerifiedDynamicGatePlan",
    "VerifiedProducerReceipt",
    "VerifiedSourceSnapshot",
    "VerifiedTypedAdmission",
    "build_phase6_gate_authority_bundle",
    "derive_static_gate_universes",
    "preflight_phase6_gate_authority_bundle",
    "publish_phase6_gate_authority_bundle",
    "source_snapshot_payload",
    "verify_artifact_receipt_index",
    "verify_dynamic_gate_plan",
    "verify_source_snapshot",
]
