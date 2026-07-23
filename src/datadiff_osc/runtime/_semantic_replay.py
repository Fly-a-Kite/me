"""Strict, private semantic replay for Runtime-owned typed admissions.

An empty error tuple proves only canonical reconstruction and intrinsic
identity agreement.  It is intentionally not producer-provenance evidence:
Root must still distinguish real execution from diagnostic synthetic input.
"""

from __future__ import annotations

from collections.abc import Callable

from datadiff_osc._canonical import CANONICAL_SCHEMA_VERSION
from datadiff_osc._replay_support import (
    ReplayValidationError,
    assert_payload_roundtrip,
    replay_enum,
    replay_immutable_value,
    require_bool,
    require_exact_mapping,
    require_finite_number,
    require_nonnegative_int,
    require_text,
    require_tuple,
)
from datadiff_osc.parallel.invariance import (
    AuthorityEvidenceSnapshot,
    InvarianceReport,
    validate_worker_count_design,
)
from datadiff_osc.runtime._private_receipts import (
    CANDIDATE_CLASSIFICATION_RECEIPT_SCHEMA_VERSION,
    PAIRED_BENCHMARK_RECEIPT_SCHEMA_VERSION,
    REPOSITORY_TEST_RECEIPT_SCHEMA_VERSION,
    TARGET_VERSION_RECEIPT_SCHEMA_VERSION,
    V3_RUN_RECEIPT_SCHEMA_VERSION,
    CandidateClassificationReceipt,
    CandidateRecheckBinding,
    ContractPerformanceSampleBinding,
    PairedBenchmarkReceipt,
    ParallelScalingSampleBinding,
    RepositoryTestNodeBinding,
    RepositoryTestReceipt,
    TargetPackageVersionBinding,
    TargetVersionReceipt,
    V3BackendTaskBinding,
    V3CaseBinding,
    V3RunReceipt,
)
from datadiff_osc.runtime._phase6_contract_performance_receipts import (
    CONTRACT_PERFORMANCE_EVIDENCE_RECEIPT_SCHEMA_VERSION,
    ContractPerformanceEvidenceReceipt,
)
from datadiff_osc.runtime._phase6_parallel_scaling_receipts import (
    PARALLEL_SCALING_EVIDENCE_RECEIPT_SCHEMA_VERSION,
    ParallelScalingEvidenceReceipt,
)
from datadiff_osc.schemas import (
    EVIDENCE_SCHEMA_VERSION,
    RESOURCE_TOKEN_SCHEMA_VERSION,
    SEED_LINEAGE_SCHEMA_VERSION,
    TASK_SCHEMA_VERSION,
    ContractFingerprint,
    EvidenceEnvelope,
    EvidenceState,
    ExecutionStatus,
    FailureKind,
    ResourceTokens,
    SeedLineage,
    SeedStage,
    StructuredExecutionOutcome,
    TargetFingerprint,
    TaskIdentity,
    TaskKind,
    TaskSpec,
    VerdictKind,
)


RUNTIME_REPLAY_REQUIRES_ROOT_PROVENANCE = True

_OUTCOME_SCHEMA_VERSION = "osc-structured-execution-outcome-v1"
_TASK_SPEC_SCHEMA_VERSION = "osc-task-spec-v1"
_AUTHORITY_SNAPSHOT_SCHEMA_VERSION = "osc-authority-evidence-snapshot-v2"
_INVARIANCE_REPORT_SCHEMA_VERSION = "osc-parallel-invariance-report-v2"
_CONTRACT_FINGERPRINT_SCHEMA_VERSION = "osc-contract-fingerprint-v1"
_TARGET_FINGERPRINT_SCHEMA_VERSION = "osc-target-fingerprint-v1"

_PARALLEL_INVARIANT_SUBJECTS = tuple(
    sorted(
        f"parallel:{name}"
        for name in (
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
        )
    )
)


def _optional_text(value: object, *, path: str) -> str:
    if not isinstance(value, str):
        raise ReplayValidationError(f"{path}: expected a string")
    return value


def _schema(value: object, *, expected: str, path: str) -> str:
    actual = require_text(value, path=path)
    if actual != expected:
        raise ReplayValidationError(f"{path}: schema version mismatch")
    return actual


def _finite(value: object, path: str) -> int | float:
    reconstructed = replay_immutable_value(value, path)
    return require_finite_number(reconstructed, path=path)


def _text_item(value: object, path: str) -> str:
    return require_text(value, path=path)


def _nonnegative_int_item(value: object, path: str) -> int:
    return require_nonnegative_int(value, path=path)


def _replay_contract_fingerprint(
    value: object, path: str
) -> ContractFingerprint:
    data = require_exact_mapping(
        value,
        fields=(
            "contract_digest",
            "registry_digest",
            "canonical_schema_version",
            "schema_version",
        ),
        path=path,
    )
    return ContractFingerprint(
        contract_digest=require_text(
            data["contract_digest"], path=f"{path}.contract_digest"
        ),
        registry_digest=require_text(
            data["registry_digest"], path=f"{path}.registry_digest"
        ),
        canonical_schema_version=_schema(
            data["canonical_schema_version"],
            expected=CANONICAL_SCHEMA_VERSION,
            path=f"{path}.canonical_schema_version",
        ),
        schema_version=_schema(
            data["schema_version"],
            expected=_CONTRACT_FINGERPRINT_SCHEMA_VERSION,
            path=f"{path}.schema_version",
        ),
    )


def _replay_target_fingerprint(value: object, path: str) -> TargetFingerprint:
    data = require_exact_mapping(
        value,
        fields=(
            "universe_digest",
            "taxonomy_digest",
            "template_digest",
            "canonical_schema_version",
            "schema_version",
        ),
        path=path,
    )
    return TargetFingerprint(
        universe_digest=require_text(
            data["universe_digest"], path=f"{path}.universe_digest"
        ),
        taxonomy_digest=require_text(
            data["taxonomy_digest"], path=f"{path}.taxonomy_digest"
        ),
        template_digest=require_text(
            data["template_digest"], path=f"{path}.template_digest"
        ),
        canonical_schema_version=_schema(
            data["canonical_schema_version"],
            expected=CANONICAL_SCHEMA_VERSION,
            path=f"{path}.canonical_schema_version",
        ),
        schema_version=_schema(
            data["schema_version"],
            expected=_TARGET_FINGERPRINT_SCHEMA_VERSION,
            path=f"{path}.schema_version",
        ),
    )


def _replay_outcome(value: object, path: str) -> StructuredExecutionOutcome:
    data = require_exact_mapping(
        value,
        fields=(
            "endpoint_id",
            "status",
            "failure_kind",
            "reason",
            "unsupported_evidence_digest",
            "schema_version",
        ),
        path=path,
    )
    return StructuredExecutionOutcome(
        endpoint_id=require_text(data["endpoint_id"], path=f"{path}.endpoint_id"),
        status=replay_enum(
            ExecutionStatus, data["status"], path=f"{path}.status"
        ),
        failure_kind=replay_enum(
            FailureKind, data["failure_kind"], path=f"{path}.failure_kind"
        ),
        reason=_optional_text(data["reason"], path=f"{path}.reason"),
        unsupported_evidence_digest=_optional_text(
            data["unsupported_evidence_digest"],
            path=f"{path}.unsupported_evidence_digest",
        ),
        schema_version=_schema(
            data["schema_version"],
            expected=_OUTCOME_SCHEMA_VERSION,
            path=f"{path}.schema_version",
        ),
    )


def _replay_seed_lineage(value: object, path: str) -> SeedLineage:
    data = require_exact_mapping(
        value,
        fields=(
            "protocol_digest",
            "master_seed",
            "lane_id",
            "case_index",
            "stage_name",
            "counter",
            "parent_digest",
            "schema_version",
        ),
        path=path,
    )
    return SeedLineage(
        protocol_digest=require_text(
            data["protocol_digest"], path=f"{path}.protocol_digest"
        ),
        master_seed=require_nonnegative_int(
            data["master_seed"], path=f"{path}.master_seed"
        ),
        lane_id=require_text(data["lane_id"], path=f"{path}.lane_id"),
        case_index=require_nonnegative_int(
            data["case_index"], path=f"{path}.case_index"
        ),
        stage_name=replay_enum(
            SeedStage, data["stage_name"], path=f"{path}.stage_name"
        ),
        counter=require_nonnegative_int(data["counter"], path=f"{path}.counter"),
        parent_digest=_optional_text(
            data["parent_digest"], path=f"{path}.parent_digest"
        ),
        schema_version=_schema(
            data["schema_version"],
            expected=SEED_LINEAGE_SCHEMA_VERSION,
            path=f"{path}.schema_version",
        ),
    )


def _replay_task_identity(value: object, path: str) -> TaskIdentity:
    data = require_exact_mapping(
        value,
        fields=(
            "protocol_digest",
            "task_kind",
            "epoch_index",
            "decision_index",
            "seed_lineage_digest",
            "contrast_set_id",
            "endpoint_id",
            "backend",
            "attempt",
            "schema_version",
        ),
        path=path,
    )
    return TaskIdentity(
        protocol_digest=require_text(
            data["protocol_digest"], path=f"{path}.protocol_digest"
        ),
        task_kind=replay_enum(
            TaskKind, data["task_kind"], path=f"{path}.task_kind"
        ),
        epoch_index=require_nonnegative_int(
            data["epoch_index"], path=f"{path}.epoch_index"
        ),
        decision_index=require_nonnegative_int(
            data["decision_index"], path=f"{path}.decision_index"
        ),
        seed_lineage_digest=require_text(
            data["seed_lineage_digest"], path=f"{path}.seed_lineage_digest"
        ),
        contrast_set_id=_optional_text(
            data["contrast_set_id"], path=f"{path}.contrast_set_id"
        ),
        endpoint_id=_optional_text(
            data["endpoint_id"], path=f"{path}.endpoint_id"
        ),
        backend=_optional_text(data["backend"], path=f"{path}.backend"),
        attempt=require_nonnegative_int(data["attempt"], path=f"{path}.attempt"),
        schema_version=_schema(
            data["schema_version"],
            expected=TASK_SCHEMA_VERSION,
            path=f"{path}.schema_version",
        ),
    )


def _replay_resources(value: object, path: str) -> ResourceTokens:
    data = require_exact_mapping(
        value,
        fields=(
            "cpu_tokens",
            "rss_bytes",
            "io_class",
            "backend_internal_threads",
            "exclusive_state",
            "schema_version",
        ),
        path=path,
    )
    return ResourceTokens(
        cpu_tokens=require_nonnegative_int(
            data["cpu_tokens"], path=f"{path}.cpu_tokens"
        ),
        rss_bytes=require_nonnegative_int(
            data["rss_bytes"], path=f"{path}.rss_bytes"
        ),
        io_class=require_text(data["io_class"], path=f"{path}.io_class"),
        backend_internal_threads=require_nonnegative_int(
            data["backend_internal_threads"],
            path=f"{path}.backend_internal_threads",
        ),
        exclusive_state=_optional_text(
            data["exclusive_state"], path=f"{path}.exclusive_state"
        ),
        schema_version=_schema(
            data["schema_version"],
            expected=RESOURCE_TOKEN_SCHEMA_VERSION,
            path=f"{path}.schema_version",
        ),
    )


def _replay_task_spec(value: object, path: str) -> TaskSpec:
    data = require_exact_mapping(
        value,
        fields=(
            "identity",
            "dependency_task_ids",
            "resources",
            "payload_digest",
            "schema_version",
        ),
        path=path,
    )
    dependencies = require_tuple(
        data["dependency_task_ids"],
        path=f"{path}.dependency_task_ids",
        item_replayer=_text_item,
        unique=True,
    )
    return TaskSpec(
        identity=_replay_task_identity(data["identity"], f"{path}.identity"),
        dependency_task_ids=dependencies,
        resources=_replay_resources(data["resources"], f"{path}.resources"),
        payload_digest=require_text(
            data["payload_digest"], path=f"{path}.payload_digest"
        ),
        schema_version=_schema(
            data["schema_version"],
            expected=_TASK_SPEC_SCHEMA_VERSION,
            path=f"{path}.schema_version",
        ),
    )


def _replay_metadata_pair(value: object, path: str) -> tuple[str, object]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ReplayValidationError(f"{path}: metadata entry must contain two items")
    key = require_text(value[0], path=f"{path}[0]")
    return key, replay_immutable_value(value[1], f"{path}[1]")


def _replay_evidence(value: object, path: str) -> EvidenceEnvelope:
    data = require_exact_mapping(
        value,
        fields=(
            "evidence_id",
            "state",
            "result_group_digest",
            "task_id",
            "seed_lineage_digest",
            "contract_fingerprint",
            "target_fingerprint",
            "derivation_certificate_digest",
            "applicability_certificate_digest",
            "activation_certificate_digest",
            "observation_certificate_digest",
            "execution_outcomes",
            "verdict_kind",
            "artifact_refs",
            "metadata",
            "schema_version",
        ),
        path=path,
    )
    target_value = data["target_fingerprint"]
    target = (
        None
        if target_value is None
        else _replay_target_fingerprint(target_value, f"{path}.target_fingerprint")
    )
    outcomes = require_tuple(
        data["execution_outcomes"],
        path=f"{path}.execution_outcomes",
        item_replayer=_replay_outcome,
        min_length=1,
    )
    artifacts = require_tuple(
        data["artifact_refs"],
        path=f"{path}.artifact_refs",
        item_replayer=_text_item,
        unique=True,
    )
    metadata = require_tuple(
        data["metadata"],
        path=f"{path}.metadata",
        item_replayer=_replay_metadata_pair,
    )
    metadata_keys = tuple(item[0] for item in metadata)
    if metadata_keys != tuple(sorted(metadata_keys)) or len(metadata_keys) != len(
        set(metadata_keys)
    ):
        raise ReplayValidationError(
            f"{path}.metadata: keys must be uniquely sorted"
        )
    return EvidenceEnvelope(
        evidence_id=require_text(data["evidence_id"], path=f"{path}.evidence_id"),
        state=replay_enum(EvidenceState, data["state"], path=f"{path}.state"),
        result_group_digest=require_text(
            data["result_group_digest"], path=f"{path}.result_group_digest"
        ),
        task_id=require_text(data["task_id"], path=f"{path}.task_id"),
        seed_lineage_digest=require_text(
            data["seed_lineage_digest"], path=f"{path}.seed_lineage_digest"
        ),
        contract_fingerprint=_replay_contract_fingerprint(
            data["contract_fingerprint"], f"{path}.contract_fingerprint"
        ),
        target_fingerprint=target,
        derivation_certificate_digest=require_text(
            data["derivation_certificate_digest"],
            path=f"{path}.derivation_certificate_digest",
        ),
        applicability_certificate_digest=require_text(
            data["applicability_certificate_digest"],
            path=f"{path}.applicability_certificate_digest",
        ),
        activation_certificate_digest=require_text(
            data["activation_certificate_digest"],
            path=f"{path}.activation_certificate_digest",
        ),
        observation_certificate_digest=require_text(
            data["observation_certificate_digest"],
            path=f"{path}.observation_certificate_digest",
        ),
        execution_outcomes=outcomes,
        verdict_kind=replay_enum(
            VerdictKind, data["verdict_kind"], path=f"{path}.verdict_kind"
        ),
        artifact_refs=artifacts,
        metadata=metadata,
        schema_version=_schema(
            data["schema_version"],
            expected=EVIDENCE_SCHEMA_VERSION,
            path=f"{path}.schema_version",
        ),
    )


def _replay_authority_snapshot(
    value: object, path: str
) -> AuthorityEvidenceSnapshot:
    data = require_exact_mapping(
        value,
        fields=(
            "execution_run_digest",
            "worker_count",
            "assignment_digest",
            "epoch_digest",
            "task_multiset_digest",
            "seed_multiset_digest",
            "authority_verdict_digest",
            "coverage_bitmap_digest",
            "ledger_digest",
            "certificate_digests",
            "schema_version",
        ),
        path=path,
    )
    certificates = require_tuple(
        data["certificate_digests"],
        path=f"{path}.certificate_digests",
        item_replayer=_text_item,
        min_length=1,
        unique=True,
        sorted_values=True,
    )
    return AuthorityEvidenceSnapshot(
        execution_run_digest=require_text(
            data["execution_run_digest"], path=f"{path}.execution_run_digest"
        ),
        worker_count=require_nonnegative_int(
            data["worker_count"], path=f"{path}.worker_count"
        ),
        assignment_digest=require_text(
            data["assignment_digest"], path=f"{path}.assignment_digest"
        ),
        epoch_digest=require_text(
            data["epoch_digest"], path=f"{path}.epoch_digest"
        ),
        task_multiset_digest=require_text(
            data["task_multiset_digest"], path=f"{path}.task_multiset_digest"
        ),
        seed_multiset_digest=require_text(
            data["seed_multiset_digest"], path=f"{path}.seed_multiset_digest"
        ),
        authority_verdict_digest=require_text(
            data["authority_verdict_digest"],
            path=f"{path}.authority_verdict_digest",
        ),
        coverage_bitmap_digest=require_text(
            data["coverage_bitmap_digest"], path=f"{path}.coverage_bitmap_digest"
        ),
        ledger_digest=require_text(
            data["ledger_digest"], path=f"{path}.ledger_digest"
        ),
        certificate_digests=certificates,
        schema_version=_schema(
            data["schema_version"],
            expected=_AUTHORITY_SNAPSHOT_SCHEMA_VERSION,
            path=f"{path}.schema_version",
        ),
    )


def _replay_invariance_report(value: object, path: str) -> InvarianceReport:
    boolean_fields = (
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
    )
    data = require_exact_mapping(
        value,
        fields=(
            "worker_counts",
            *boolean_fields,
            "authority_evidence_errors",
            "run_digests",
            "outcome_digests",
            "authority_evidence_digests",
            "schema_version",
        ),
        path=path,
    )
    worker_counts = require_tuple(
        data["worker_counts"],
        path=f"{path}.worker_counts",
        item_replayer=_nonnegative_int_item,
    )
    validate_worker_count_design(worker_counts)
    errors = require_tuple(
        data["authority_evidence_errors"],
        path=f"{path}.authority_evidence_errors",
        item_replayer=_text_item,
    )
    run_digests = require_tuple(
        data["run_digests"],
        path=f"{path}.run_digests",
        item_replayer=_text_item,
    )
    outcome_digests = require_tuple(
        data["outcome_digests"],
        path=f"{path}.outcome_digests",
        item_replayer=_text_item,
    )
    evidence_digests = require_tuple(
        data["authority_evidence_digests"],
        path=f"{path}.authority_evidence_digests",
        item_replayer=_text_item,
    )
    if len(run_digests) != len(worker_counts) or len(outcome_digests) != len(
        worker_counts
    ):
        raise ReplayValidationError(
            f"{path}: run and outcome bindings must match worker counts"
        )
    replayed_booleans = {
        name: require_bool(data[name], path=f"{path}.{name}")
        for name in boolean_fields
    }
    if replayed_booleans["authority_evidence_complete"] and (
        errors or len(evidence_digests) != len(worker_counts)
    ):
        raise ReplayValidationError(
            f"{path}: complete authority evidence bindings are inconsistent"
        )
    return InvarianceReport(
        worker_counts=worker_counts,
        authority_evidence_errors=errors,
        run_digests=run_digests,
        outcome_digests=outcome_digests,
        authority_evidence_digests=evidence_digests,
        schema_version=_schema(
            data["schema_version"],
            expected=_INVARIANCE_REPORT_SCHEMA_VERSION,
            path=f"{path}.schema_version",
        ),
        **replayed_booleans,
    )


def _replay_v3_backend_task(value: object, path: str) -> V3BackendTaskBinding:
    data = require_exact_mapping(
        value,
        fields=(
            "task_id",
            "endpoint_digest",
            "task_spec_digest",
            "execution_outcome_digest",
            "status",
        ),
        path=path,
    )
    return V3BackendTaskBinding(
        task_id=require_text(data["task_id"], path=f"{path}.task_id"),
        endpoint_digest=require_text(
            data["endpoint_digest"], path=f"{path}.endpoint_digest"
        ),
        task_spec_digest=require_text(
            data["task_spec_digest"], path=f"{path}.task_spec_digest"
        ),
        execution_outcome_digest=require_text(
            data["execution_outcome_digest"],
            path=f"{path}.execution_outcome_digest",
        ),
        status=replay_enum(
            ExecutionStatus, data["status"], path=f"{path}.status"
        ),
    )


def _replay_v3_case(value: object, path: str) -> V3CaseBinding:
    data = require_exact_mapping(
        value,
        fields=(
            "case_id",
            "case_index",
            "seed_lineage_digest",
            "case_task_id",
            "case_task_spec_digest",
            "result_digest",
            "pipeline_digest",
            "execution_outcome_digest",
            "evidence_envelope_digest",
            "executed",
            "iteration_failure",
            "pipeline_error",
            "backend_tasks",
        ),
        path=path,
    )
    backend_tasks = require_tuple(
        data["backend_tasks"],
        path=f"{path}.backend_tasks",
        item_replayer=_replay_v3_backend_task,
        min_length=1,
    )
    return V3CaseBinding(
        case_id=require_text(data["case_id"], path=f"{path}.case_id"),
        case_index=require_nonnegative_int(
            data["case_index"], path=f"{path}.case_index"
        ),
        seed_lineage_digest=require_text(
            data["seed_lineage_digest"], path=f"{path}.seed_lineage_digest"
        ),
        case_task_id=require_text(
            data["case_task_id"], path=f"{path}.case_task_id"
        ),
        case_task_spec_digest=require_text(
            data["case_task_spec_digest"], path=f"{path}.case_task_spec_digest"
        ),
        result_digest=require_text(
            data["result_digest"], path=f"{path}.result_digest"
        ),
        pipeline_digest=require_text(
            data["pipeline_digest"], path=f"{path}.pipeline_digest"
        ),
        execution_outcome_digest=require_text(
            data["execution_outcome_digest"],
            path=f"{path}.execution_outcome_digest",
        ),
        evidence_envelope_digest=require_text(
            data["evidence_envelope_digest"],
            path=f"{path}.evidence_envelope_digest",
        ),
        executed=require_bool(data["executed"], path=f"{path}.executed"),
        iteration_failure=require_bool(
            data["iteration_failure"], path=f"{path}.iteration_failure"
        ),
        pipeline_error=require_bool(
            data["pipeline_error"], path=f"{path}.pipeline_error"
        ),
        backend_tasks=backend_tasks,
    )


def _replay_v3_receipt(value: object, path: str) -> V3RunReceipt:
    data = require_exact_mapping(
        value,
        fields=(
            "source_digest",
            "protocol_digest",
            "v3_gate_plan_digest",
            "run_id",
            "lane_id",
            "seed",
            "execution_run_digest",
            "run_completed",
            "cases",
            "schema_version",
        ),
        path=path,
    )
    cases = require_tuple(
        data["cases"],
        path=f"{path}.cases",
        item_replayer=_replay_v3_case,
        min_length=100,
    )
    return V3RunReceipt(
        source_digest=require_text(
            data["source_digest"], path=f"{path}.source_digest"
        ),
        protocol_digest=require_text(
            data["protocol_digest"], path=f"{path}.protocol_digest"
        ),
        v3_gate_plan_digest=require_text(
            data["v3_gate_plan_digest"], path=f"{path}.v3_gate_plan_digest"
        ),
        run_id=require_text(data["run_id"], path=f"{path}.run_id"),
        lane_id=require_text(data["lane_id"], path=f"{path}.lane_id"),
        seed=require_nonnegative_int(data["seed"], path=f"{path}.seed"),
        execution_run_digest=require_text(
            data["execution_run_digest"], path=f"{path}.execution_run_digest"
        ),
        run_completed=require_bool(
            data["run_completed"], path=f"{path}.run_completed"
        ),
        cases=cases,
        schema_version=_schema(
            data["schema_version"],
            expected=V3_RUN_RECEIPT_SCHEMA_VERSION,
            path=f"{path}.schema_version",
        ),
    )


def _replay_repository_node(
    value: object, path: str
) -> RepositoryTestNodeBinding:
    data = require_exact_mapping(
        value,
        fields=("node_id", "outcome", "result_digest"),
        path=path,
    )
    return RepositoryTestNodeBinding(
        node_id=require_text(data["node_id"], path=f"{path}.node_id"),
        outcome=require_text(data["outcome"], path=f"{path}.outcome"),
        result_digest=require_text(
            data["result_digest"], path=f"{path}.result_digest"
        ),
    )


def _replay_repository_receipt(
    value: object, path: str
) -> RepositoryTestReceipt:
    data = require_exact_mapping(
        value,
        fields=(
            "source_digest",
            "test_plan_digest",
            "collection_digest",
            "config_digest",
            "environment_digest",
            "command_digest",
            "junit_sha256",
            "log_sha256",
            "exit_code",
            "nodes",
            "schema_version",
        ),
        path=path,
    )
    nodes = require_tuple(
        data["nodes"],
        path=f"{path}.nodes",
        item_replayer=_replay_repository_node,
        min_length=1,
    )
    return RepositoryTestReceipt(
        source_digest=require_text(
            data["source_digest"], path=f"{path}.source_digest"
        ),
        test_plan_digest=require_text(
            data["test_plan_digest"], path=f"{path}.test_plan_digest"
        ),
        collection_digest=require_text(
            data["collection_digest"], path=f"{path}.collection_digest"
        ),
        config_digest=require_text(
            data["config_digest"], path=f"{path}.config_digest"
        ),
        environment_digest=require_text(
            data["environment_digest"], path=f"{path}.environment_digest"
        ),
        command_digest=require_text(
            data["command_digest"], path=f"{path}.command_digest"
        ),
        junit_sha256=require_text(
            data["junit_sha256"], path=f"{path}.junit_sha256"
        ),
        log_sha256=require_text(data["log_sha256"], path=f"{path}.log_sha256"),
        exit_code=require_nonnegative_int(
            data["exit_code"], path=f"{path}.exit_code"
        ),
        nodes=nodes,
        schema_version=_schema(
            data["schema_version"],
            expected=REPOSITORY_TEST_RECEIPT_SCHEMA_VERSION,
            path=f"{path}.schema_version",
        ),
    )


def _replay_target_package(
    value: object, path: str
) -> TargetPackageVersionBinding:
    data = require_exact_mapping(
        value,
        fields=(
            "package_id",
            "distribution_name",
            "import_name",
            "installed_version",
            "latest_version",
            "installed_metadata_digest",
            "version_source_kind",
            "version_source_digest",
            "version_source_sha256",
        ),
        path=path,
    )
    return TargetPackageVersionBinding(
        package_id=require_text(data["package_id"], path=f"{path}.package_id"),
        distribution_name=require_text(
            data["distribution_name"], path=f"{path}.distribution_name"
        ),
        import_name=require_text(
            data["import_name"], path=f"{path}.import_name"
        ),
        installed_version=require_text(
            data["installed_version"], path=f"{path}.installed_version"
        ),
        latest_version=require_text(
            data["latest_version"], path=f"{path}.latest_version"
        ),
        installed_metadata_digest=require_text(
            data["installed_metadata_digest"],
            path=f"{path}.installed_metadata_digest",
        ),
        version_source_kind=require_text(
            data["version_source_kind"], path=f"{path}.version_source_kind"
        ),
        version_source_digest=require_text(
            data["version_source_digest"], path=f"{path}.version_source_digest"
        ),
        version_source_sha256=require_text(
            data["version_source_sha256"], path=f"{path}.version_source_sha256"
        ),
    )


def _replay_target_receipt(value: object, path: str) -> TargetVersionReceipt:
    data = require_exact_mapping(
        value,
        fields=(
            "source_digest",
            "audit_plan_digest",
            "environment_digest",
            "python_runtime_digest",
            "packages",
            "schema_version",
        ),
        path=path,
    )
    packages = require_tuple(
        data["packages"],
        path=f"{path}.packages",
        item_replayer=_replay_target_package,
        min_length=1,
    )
    return TargetVersionReceipt(
        source_digest=require_text(
            data["source_digest"], path=f"{path}.source_digest"
        ),
        audit_plan_digest=require_text(
            data["audit_plan_digest"], path=f"{path}.audit_plan_digest"
        ),
        environment_digest=require_text(
            data["environment_digest"], path=f"{path}.environment_digest"
        ),
        python_runtime_digest=require_text(
            data["python_runtime_digest"], path=f"{path}.python_runtime_digest"
        ),
        packages=packages,
        schema_version=_schema(
            data["schema_version"],
            expected=TARGET_VERSION_RECEIPT_SCHEMA_VERSION,
            path=f"{path}.schema_version",
        ),
    )


def _replay_candidate_recheck(
    value: object, path: str
) -> CandidateRecheckBinding:
    data = require_exact_mapping(
        value,
        fields=(
            "recheck_id",
            "ordinal",
            "task_id",
            "seed_lineage_digest",
            "execution_outcome_digest",
            "completed",
            "reproduced",
        ),
        path=path,
    )
    return CandidateRecheckBinding(
        recheck_id=require_text(
            data["recheck_id"], path=f"{path}.recheck_id"
        ),
        ordinal=require_nonnegative_int(
            data["ordinal"], path=f"{path}.ordinal"
        ),
        task_id=require_text(data["task_id"], path=f"{path}.task_id"),
        seed_lineage_digest=require_text(
            data["seed_lineage_digest"], path=f"{path}.seed_lineage_digest"
        ),
        execution_outcome_digest=require_text(
            data["execution_outcome_digest"],
            path=f"{path}.execution_outcome_digest",
        ),
        completed=require_bool(data["completed"], path=f"{path}.completed"),
        reproduced=require_bool(data["reproduced"], path=f"{path}.reproduced"),
    )


def _replay_candidate_receipt(
    value: object, path: str
) -> CandidateClassificationReceipt:
    data = require_exact_mapping(
        value,
        fields=(
            "source_digest",
            "v3_gate_plan_digest",
            "classification_policy_digest",
            "classification_id",
            "source_case_id",
            "result_group_digest",
            "evidence_envelope_digest",
            "evidence_state",
            "finding_record_id",
            "candidate_id",
            "classification_digest",
            "campaign_rechecks",
            "pipeline_rechecks",
            "bug_claimed",
            "schema_version",
        ),
        path=path,
    )
    campaigns = require_tuple(
        data["campaign_rechecks"],
        path=f"{path}.campaign_rechecks",
        item_replayer=_replay_candidate_recheck,
    )
    pipelines = require_tuple(
        data["pipeline_rechecks"],
        path=f"{path}.pipeline_rechecks",
        item_replayer=_replay_candidate_recheck,
    )
    return CandidateClassificationReceipt(
        source_digest=require_text(
            data["source_digest"], path=f"{path}.source_digest"
        ),
        v3_gate_plan_digest=require_text(
            data["v3_gate_plan_digest"], path=f"{path}.v3_gate_plan_digest"
        ),
        classification_policy_digest=require_text(
            data["classification_policy_digest"],
            path=f"{path}.classification_policy_digest",
        ),
        classification_id=require_text(
            data["classification_id"], path=f"{path}.classification_id"
        ),
        source_case_id=require_text(
            data["source_case_id"], path=f"{path}.source_case_id"
        ),
        result_group_digest=require_text(
            data["result_group_digest"], path=f"{path}.result_group_digest"
        ),
        evidence_envelope_digest=require_text(
            data["evidence_envelope_digest"],
            path=f"{path}.evidence_envelope_digest",
        ),
        evidence_state=replay_enum(
            EvidenceState, data["evidence_state"], path=f"{path}.evidence_state"
        ),
        finding_record_id=_optional_text(
            data["finding_record_id"], path=f"{path}.finding_record_id"
        ),
        candidate_id=_optional_text(
            data["candidate_id"], path=f"{path}.candidate_id"
        ),
        classification_digest=require_text(
            data["classification_digest"], path=f"{path}.classification_digest"
        ),
        campaign_rechecks=campaigns,
        pipeline_rechecks=pipelines,
        bug_claimed=require_bool(
            data["bug_claimed"], path=f"{path}.bug_claimed"
        ),
        schema_version=_schema(
            data["schema_version"],
            expected=CANDIDATE_CLASSIFICATION_RECEIPT_SCHEMA_VERSION,
            path=f"{path}.schema_version",
        ),
    )


def _replay_contract_sample(
    value: object, path: str
) -> ContractPerformanceSampleBinding:
    data = require_exact_mapping(
        value,
        fields=(
            "sample_id",
            "case_id",
            "baseline_task_id",
            "treatment_task_id",
            "compile_match_ms",
            "case_wall_ms",
            "baseline_throughput_cases_s",
            "treatment_throughput_cases_s",
        ),
        path=path,
    )
    return ContractPerformanceSampleBinding(
        sample_id=require_text(data["sample_id"], path=f"{path}.sample_id"),
        case_id=require_text(data["case_id"], path=f"{path}.case_id"),
        baseline_task_id=require_text(
            data["baseline_task_id"], path=f"{path}.baseline_task_id"
        ),
        treatment_task_id=require_text(
            data["treatment_task_id"], path=f"{path}.treatment_task_id"
        ),
        compile_match_ms=_finite(
            data["compile_match_ms"], f"{path}.compile_match_ms"
        ),
        case_wall_ms=_finite(data["case_wall_ms"], f"{path}.case_wall_ms"),
        baseline_throughput_cases_s=_finite(
            data["baseline_throughput_cases_s"],
            f"{path}.baseline_throughput_cases_s",
        ),
        treatment_throughput_cases_s=_finite(
            data["treatment_throughput_cases_s"],
            f"{path}.treatment_throughput_cases_s",
        ),
    )


def _replay_parallel_sample(
    value: object, path: str
) -> ParallelScalingSampleBinding:
    data = require_exact_mapping(
        value,
        fields=(
            "sample_id",
            "workload_id",
            "one_worker_task_set_digest",
            "six_worker_task_set_digest",
            "one_worker_result_digest",
            "six_worker_result_digest",
            "one_worker_elapsed_seconds",
            "six_worker_elapsed_seconds",
        ),
        path=path,
    )
    return ParallelScalingSampleBinding(
        sample_id=require_text(data["sample_id"], path=f"{path}.sample_id"),
        workload_id=require_text(
            data["workload_id"], path=f"{path}.workload_id"
        ),
        one_worker_task_set_digest=require_text(
            data["one_worker_task_set_digest"],
            path=f"{path}.one_worker_task_set_digest",
        ),
        six_worker_task_set_digest=require_text(
            data["six_worker_task_set_digest"],
            path=f"{path}.six_worker_task_set_digest",
        ),
        one_worker_result_digest=require_text(
            data["one_worker_result_digest"],
            path=f"{path}.one_worker_result_digest",
        ),
        six_worker_result_digest=require_text(
            data["six_worker_result_digest"],
            path=f"{path}.six_worker_result_digest",
        ),
        one_worker_elapsed_seconds=_finite(
            data["one_worker_elapsed_seconds"],
            f"{path}.one_worker_elapsed_seconds",
        ),
        six_worker_elapsed_seconds=_finite(
            data["six_worker_elapsed_seconds"],
            f"{path}.six_worker_elapsed_seconds",
        ),
    )


def _replay_paired_receipt(value: object, path: str) -> PairedBenchmarkReceipt:
    data = require_exact_mapping(
        value,
        fields=(
            "source_digest",
            "benchmark_plan_digest",
            "environment_digest",
            "baseline_config_digest",
            "treatment_config_digest",
            "contract_samples",
            "parallel_samples",
            "schema_version",
        ),
        path=path,
    )
    contract_samples = require_tuple(
        data["contract_samples"],
        path=f"{path}.contract_samples",
        item_replayer=_replay_contract_sample,
        min_length=1,
    )
    parallel_samples = require_tuple(
        data["parallel_samples"],
        path=f"{path}.parallel_samples",
        item_replayer=_replay_parallel_sample,
        min_length=1,
    )
    return PairedBenchmarkReceipt(
        source_digest=require_text(
            data["source_digest"], path=f"{path}.source_digest"
        ),
        benchmark_plan_digest=require_text(
            data["benchmark_plan_digest"], path=f"{path}.benchmark_plan_digest"
        ),
        environment_digest=require_text(
            data["environment_digest"], path=f"{path}.environment_digest"
        ),
        baseline_config_digest=require_text(
            data["baseline_config_digest"], path=f"{path}.baseline_config_digest"
        ),
        treatment_config_digest=require_text(
            data["treatment_config_digest"], path=f"{path}.treatment_config_digest"
        ),
        contract_samples=contract_samples,
        parallel_samples=parallel_samples,
        schema_version=_schema(
            data["schema_version"],
            expected=PAIRED_BENCHMARK_RECEIPT_SCHEMA_VERSION,
            path=f"{path}.schema_version",
        ),
    )


def _replay_contract_performance_evidence_receipt(
    value: object,
    path: str,
) -> ContractPerformanceEvidenceReceipt:
    data = require_exact_mapping(
        value,
        fields=(
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
            "schema_version",
        ),
        path=path,
    )
    return ContractPerformanceEvidenceReceipt(
        source_snapshot_digest=require_text(
            data["source_snapshot_digest"],
            path=f"{path}.source_snapshot_digest",
        ),
        source_case_digest=require_text(
            data["source_case_digest"], path=f"{path}.source_case_digest"
        ),
        comparison_decision_id=require_text(
            data["comparison_decision_id"],
            path=f"{path}.comparison_decision_id",
        ),
        contract_comparison_producer_receipt_digest=require_text(
            data["contract_comparison_producer_receipt_digest"],
            path=f"{path}.contract_comparison_producer_receipt_digest",
        ),
        result_group_id=require_text(
            data["result_group_id"], path=f"{path}.result_group_id"
        ),
        result_group_digest=require_text(
            data["result_group_digest"], path=f"{path}.result_group_digest"
        ),
        contract_fingerprint_digest=require_text(
            data["contract_fingerprint_digest"],
            path=f"{path}.contract_fingerprint_digest",
        ),
        endpoint_set_digest=require_text(
            data["endpoint_set_digest"], path=f"{path}.endpoint_set_digest"
        ),
        fingerprint_task_id=require_text(
            data["fingerprint_task_id"], path=f"{path}.fingerprint_task_id"
        ),
        fingerprint_result_digest=require_text(
            data["fingerprint_result_digest"],
            path=f"{path}.fingerprint_result_digest",
        ),
        exact_task_id=require_text(
            data["exact_task_id"], path=f"{path}.exact_task_id"
        ),
        exact_result_digest=require_text(
            data["exact_result_digest"], path=f"{path}.exact_result_digest"
        ),
        fingerprint_partition_digest=require_text(
            data["fingerprint_partition_digest"],
            path=f"{path}.fingerprint_partition_digest",
        ),
        canonical_partition_digest=require_text(
            data["canonical_partition_digest"],
            path=f"{path}.canonical_partition_digest",
        ),
        comparison_algorithm_digest=require_text(
            data["comparison_algorithm_digest"],
            path=f"{path}.comparison_algorithm_digest",
        ),
        benchmark_plan_digest=require_text(
            data["benchmark_plan_digest"], path=f"{path}.benchmark_plan_digest"
        ),
        environment_digest=require_text(
            data["environment_digest"], path=f"{path}.environment_digest"
        ),
        baseline_config_digest=require_text(
            data["baseline_config_digest"],
            path=f"{path}.baseline_config_digest",
        ),
        treatment_config_digest=require_text(
            data["treatment_config_digest"],
            path=f"{path}.treatment_config_digest",
        ),
        baseline_task_id=require_text(
            data["baseline_task_id"], path=f"{path}.baseline_task_id"
        ),
        baseline_task_spec_digest=require_text(
            data["baseline_task_spec_digest"],
            path=f"{path}.baseline_task_spec_digest",
        ),
        baseline_result_digest=require_text(
            data["baseline_result_digest"],
            path=f"{path}.baseline_result_digest",
        ),
        baseline_seed_lineage_digest=require_text(
            data["baseline_seed_lineage_digest"],
            path=f"{path}.baseline_seed_lineage_digest",
        ),
        baseline_execution_outcome_digest=require_text(
            data["baseline_execution_outcome_digest"],
            path=f"{path}.baseline_execution_outcome_digest",
        ),
        baseline_evidence_envelope_digest=require_text(
            data["baseline_evidence_envelope_digest"],
            path=f"{path}.baseline_evidence_envelope_digest",
        ),
        treatment_task_id=require_text(
            data["treatment_task_id"], path=f"{path}.treatment_task_id"
        ),
        treatment_task_spec_digest=require_text(
            data["treatment_task_spec_digest"],
            path=f"{path}.treatment_task_spec_digest",
        ),
        treatment_result_digest=require_text(
            data["treatment_result_digest"],
            path=f"{path}.treatment_result_digest",
        ),
        treatment_seed_lineage_digest=require_text(
            data["treatment_seed_lineage_digest"],
            path=f"{path}.treatment_seed_lineage_digest",
        ),
        treatment_execution_outcome_digest=require_text(
            data["treatment_execution_outcome_digest"],
            path=f"{path}.treatment_execution_outcome_digest",
        ),
        treatment_evidence_envelope_digest=require_text(
            data["treatment_evidence_envelope_digest"],
            path=f"{path}.treatment_evidence_envelope_digest",
        ),
        baseline_materialized_bytes=require_nonnegative_int(
            data["baseline_materialized_bytes"],
            path=f"{path}.baseline_materialized_bytes",
        ),
        treatment_materialized_bytes=require_nonnegative_int(
            data["treatment_materialized_bytes"],
            path=f"{path}.treatment_materialized_bytes",
        ),
        baseline_backend_pair_comparisons=require_nonnegative_int(
            data["baseline_backend_pair_comparisons"],
            path=f"{path}.baseline_backend_pair_comparisons",
        ),
        treatment_backend_pair_comparisons=require_nonnegative_int(
            data["treatment_backend_pair_comparisons"],
            path=f"{path}.treatment_backend_pair_comparisons",
        ),
        baseline_comparison_cpu_ns=require_nonnegative_int(
            data["baseline_comparison_cpu_ns"],
            path=f"{path}.baseline_comparison_cpu_ns",
        ),
        treatment_comparison_cpu_ns=require_nonnegative_int(
            data["treatment_comparison_cpu_ns"],
            path=f"{path}.treatment_comparison_cpu_ns",
        ),
        baseline_strategy=require_text(
            data["baseline_strategy"], path=f"{path}.baseline_strategy"
        ),
        treatment_strategy=require_text(
            data["treatment_strategy"], path=f"{path}.treatment_strategy"
        ),
        schema_version=_schema(
            data["schema_version"],
            expected=CONTRACT_PERFORMANCE_EVIDENCE_RECEIPT_SCHEMA_VERSION,
            path=f"{path}.schema_version",
        ),
    )


def _reconstruct_contract_performance_evidence_receipt_payload(
    payload: object,
) -> ContractPerformanceEvidenceReceipt:
    """Reconstruct the private raw pair for Root cross-object binding."""

    receipt = _replay_contract_performance_evidence_receipt(
        payload,
        "payload.ContractPerformanceEvidenceReceipt",
    )
    assert_payload_roundtrip(payload, receipt)
    return receipt


def _replay_parallel_scaling_evidence_receipt(
    value: object,
    path: str,
) -> ParallelScalingEvidenceReceipt:
    data = require_exact_mapping(
        value,
        fields=(
            "source_snapshot_digest",
            "workload_id",
            "benchmark_plan_digest",
            "environment_digest",
            "one_worker_config_digest",
            "six_worker_config_digest",
            "one_worker_task_set_digest",
            "six_worker_task_set_digest",
            "one_worker_result_digest",
            "six_worker_result_digest",
            "one_worker_evidence_digest",
            "six_worker_evidence_digest",
            "one_worker_clock_id",
            "one_worker_start_ns",
            "one_worker_end_ns",
            "one_worker_task_counter_start",
            "one_worker_task_counter_end",
            "six_worker_clock_id",
            "six_worker_start_ns",
            "six_worker_end_ns",
            "six_worker_task_counter_start",
            "six_worker_task_counter_end",
            "one_worker_count",
            "six_worker_count",
            "schema_version",
        ),
        path=path,
    )
    return ParallelScalingEvidenceReceipt(
        source_snapshot_digest=require_text(
            data["source_snapshot_digest"],
            path=f"{path}.source_snapshot_digest",
        ),
        workload_id=require_text(data["workload_id"], path=f"{path}.workload_id"),
        benchmark_plan_digest=require_text(
            data["benchmark_plan_digest"], path=f"{path}.benchmark_plan_digest"
        ),
        environment_digest=require_text(
            data["environment_digest"], path=f"{path}.environment_digest"
        ),
        one_worker_config_digest=require_text(
            data["one_worker_config_digest"],
            path=f"{path}.one_worker_config_digest",
        ),
        six_worker_config_digest=require_text(
            data["six_worker_config_digest"],
            path=f"{path}.six_worker_config_digest",
        ),
        one_worker_task_set_digest=require_text(
            data["one_worker_task_set_digest"],
            path=f"{path}.one_worker_task_set_digest",
        ),
        six_worker_task_set_digest=require_text(
            data["six_worker_task_set_digest"],
            path=f"{path}.six_worker_task_set_digest",
        ),
        one_worker_result_digest=require_text(
            data["one_worker_result_digest"],
            path=f"{path}.one_worker_result_digest",
        ),
        six_worker_result_digest=require_text(
            data["six_worker_result_digest"],
            path=f"{path}.six_worker_result_digest",
        ),
        one_worker_evidence_digest=require_text(
            data["one_worker_evidence_digest"],
            path=f"{path}.one_worker_evidence_digest",
        ),
        six_worker_evidence_digest=require_text(
            data["six_worker_evidence_digest"],
            path=f"{path}.six_worker_evidence_digest",
        ),
        one_worker_clock_id=require_text(
            data["one_worker_clock_id"], path=f"{path}.one_worker_clock_id"
        ),
        one_worker_start_ns=require_nonnegative_int(
            data["one_worker_start_ns"], path=f"{path}.one_worker_start_ns"
        ),
        one_worker_end_ns=require_nonnegative_int(
            data["one_worker_end_ns"], path=f"{path}.one_worker_end_ns"
        ),
        one_worker_task_counter_start=require_nonnegative_int(
            data["one_worker_task_counter_start"],
            path=f"{path}.one_worker_task_counter_start",
        ),
        one_worker_task_counter_end=require_nonnegative_int(
            data["one_worker_task_counter_end"],
            path=f"{path}.one_worker_task_counter_end",
        ),
        six_worker_clock_id=require_text(
            data["six_worker_clock_id"], path=f"{path}.six_worker_clock_id"
        ),
        six_worker_start_ns=require_nonnegative_int(
            data["six_worker_start_ns"], path=f"{path}.six_worker_start_ns"
        ),
        six_worker_end_ns=require_nonnegative_int(
            data["six_worker_end_ns"], path=f"{path}.six_worker_end_ns"
        ),
        six_worker_task_counter_start=require_nonnegative_int(
            data["six_worker_task_counter_start"],
            path=f"{path}.six_worker_task_counter_start",
        ),
        six_worker_task_counter_end=require_nonnegative_int(
            data["six_worker_task_counter_end"],
            path=f"{path}.six_worker_task_counter_end",
        ),
        one_worker_count=require_nonnegative_int(
            data["one_worker_count"], path=f"{path}.one_worker_count"
        ),
        six_worker_count=require_nonnegative_int(
            data["six_worker_count"], path=f"{path}.six_worker_count"
        ),
        schema_version=_schema(
            data["schema_version"],
            expected=PARALLEL_SCALING_EVIDENCE_RECEIPT_SCHEMA_VERSION,
            path=f"{path}.schema_version",
        ),
    )


def _reconstruct_parallel_scaling_evidence_receipt_payload(
    payload: object,
) -> ParallelScalingEvidenceReceipt:
    """Reconstruct the private parallel receipt for Root cross-object binding."""

    receipt = _replay_parallel_scaling_evidence_receipt(
        payload,
        "payload.ParallelScalingEvidenceReceipt",
    )
    assert_payload_roundtrip(payload, receipt)
    return receipt


_REPLAYERS: dict[tuple[str, str], Callable[[object, str], object]] = {
    ("StructuredExecutionOutcome", _OUTCOME_SCHEMA_VERSION): _replay_outcome,
    ("EvidenceEnvelope", EVIDENCE_SCHEMA_VERSION): _replay_evidence,
    ("TaskSpec", _TASK_SPEC_SCHEMA_VERSION): _replay_task_spec,
    ("TaskIdentity", TASK_SCHEMA_VERSION): _replay_task_identity,
    ("SeedLineage", SEED_LINEAGE_SCHEMA_VERSION): _replay_seed_lineage,
    ("InvarianceReport", _INVARIANCE_REPORT_SCHEMA_VERSION): (
        _replay_invariance_report
    ),
    ("AuthorityEvidenceSnapshot", _AUTHORITY_SNAPSHOT_SCHEMA_VERSION): (
        _replay_authority_snapshot
    ),
    ("V3RunReceipt", V3_RUN_RECEIPT_SCHEMA_VERSION): _replay_v3_receipt,
    (
        "RepositoryTestReceipt",
        REPOSITORY_TEST_RECEIPT_SCHEMA_VERSION,
    ): _replay_repository_receipt,
    ("TargetVersionReceipt", TARGET_VERSION_RECEIPT_SCHEMA_VERSION): (
        _replay_target_receipt
    ),
    (
        "CandidateClassificationReceipt",
        CANDIDATE_CLASSIFICATION_RECEIPT_SCHEMA_VERSION,
    ): _replay_candidate_receipt,
    (
        "ContractPerformanceEvidenceReceipt",
        CONTRACT_PERFORMANCE_EVIDENCE_RECEIPT_SCHEMA_VERSION,
    ): _replay_contract_performance_evidence_receipt,
    (
        "ParallelScalingEvidenceReceipt",
        PARALLEL_SCALING_EVIDENCE_RECEIPT_SCHEMA_VERSION,
    ): _replay_parallel_scaling_evidence_receipt,
}


def _intrinsic_subjects(
    envelope_type: str, reconstructed: object
) -> dict[str, tuple[str, ...]]:
    if envelope_type == "StructuredExecutionOutcome":
        outcome = reconstructed
        assert isinstance(outcome, StructuredExecutionOutcome)
        return {"execution_outcomes": (outcome.digest,)}
    if envelope_type == "EvidenceEnvelope":
        evidence = reconstructed
        assert isinstance(evidence, EvidenceEnvelope)
        return {"evidence_envelopes": (evidence.digest,)}
    if envelope_type == "TaskSpec":
        task = reconstructed
        assert isinstance(task, TaskSpec)
        return {"task_specs": (task.digest,)}
    if envelope_type == "TaskIdentity":
        identity = reconstructed
        assert isinstance(identity, TaskIdentity)
        return {"task_identities": (identity.task_id,)}
    if envelope_type == "SeedLineage":
        lineage = reconstructed
        assert isinstance(lineage, SeedLineage)
        return {"seed_lineages": (lineage.digest,)}
    if envelope_type == "InvarianceReport":
        return {"parallel_invariance_checks_total": _PARALLEL_INVARIANT_SUBJECTS}
    if envelope_type == "AuthorityEvidenceSnapshot":
        snapshot = reconstructed
        assert isinstance(snapshot, AuthorityEvidenceSnapshot)
        return {"authority_evidence_snapshots": (snapshot.digest,)}
    if envelope_type == "V3RunReceipt":
        receipt = reconstructed
        assert isinstance(receipt, V3RunReceipt)
        return {
            "v3_runs": (receipt.run_id,),
            "v3_cases": receipt.case_ids,
            "v3_executed_cases": receipt.executed_case_ids,
            "v3_backend_tasks": receipt.backend_task_ids,
        }
    if envelope_type == "RepositoryTestReceipt":
        receipt = reconstructed
        assert isinstance(receipt, RepositoryTestReceipt)
        return {"repository_tests": receipt.node_ids}
    if envelope_type == "TargetVersionReceipt":
        receipt = reconstructed
        assert isinstance(receipt, TargetVersionReceipt)
        return {"target_packages": receipt.package_ids}
    if envelope_type == "CandidateClassificationReceipt":
        receipt = reconstructed
        assert isinstance(receipt, CandidateClassificationReceipt)
        result = {"classified_v3_cases": (receipt.source_case_id,)}
        if receipt.is_candidate:
            result.update(
                {
                    "candidate_records": (receipt.finding_record_id,),
                    "campaign_rechecks": receipt.campaign_recheck_ids,
                    "pipeline_rechecks": receipt.pipeline_recheck_ids,
                }
            )
        return result
    if envelope_type == "ContractPerformanceEvidenceReceipt":
        receipt = reconstructed
        assert isinstance(receipt, ContractPerformanceEvidenceReceipt)
        return {"contract_performance_samples": (receipt.sample_id,)}
    if envelope_type == "ParallelScalingEvidenceReceipt":
        receipt = reconstructed
        assert isinstance(receipt, ParallelScalingEvidenceReceipt)
        return {"parallel_scaling_samples": (receipt.sample_id,)}
    raise AssertionError(f"unhandled Runtime replay type: {envelope_type}")


def replay_runtime_admission(
    *,
    envelope_type: str,
    schema_version: str,
    payload: object,
    subject_kind: str,
    subject_ids: tuple[str, ...],
) -> tuple[str, ...]:
    """Replay one Runtime envelope and compare its intrinsic subject identities.

    The tuple contains deterministic fail-closed diagnostics.  Empty means the
    object and subject claim match exactly; it never means Root provenance was
    established.
    """

    if not isinstance(envelope_type, str) or not envelope_type:
        return ("runtime_replay_invalid_envelope_type",)
    if not isinstance(schema_version, str) or not schema_version:
        return ("runtime_replay_invalid_schema_version",)
    if not isinstance(subject_kind, str) or not subject_kind:
        return ("runtime_replay_invalid_subject_kind",)
    if not isinstance(subject_ids, tuple):
        return ("runtime_replay_subject_ids_not_tuple",)
    if any(not isinstance(item, str) or not item for item in subject_ids):
        return ("runtime_replay_subject_ids_invalid",)
    if subject_ids != tuple(sorted(subject_ids)) or len(subject_ids) != len(
        set(subject_ids)
    ):
        return ("runtime_replay_subject_ids_not_unique_sorted",)
    replayer = _REPLAYERS.get((envelope_type, schema_version))
    if replayer is None:
        return (
            f"runtime_replay_type_not_owned:{envelope_type}@{schema_version}",
        )
    try:
        reconstructed = replayer(payload, "payload")
        assert_payload_roundtrip(payload, reconstructed)
        available = _intrinsic_subjects(envelope_type, reconstructed)
    except (ReplayValidationError, TypeError, ValueError) as exc:
        return (
            f"runtime_replay_invalid_payload:{envelope_type}:{type(exc).__name__}:{exc}",
        )
    expected = available.get(subject_kind)
    if expected is None:
        return (
            f"runtime_replay_subject_not_intrinsic:{envelope_type}:{subject_kind}",
        )
    if subject_ids != expected:
        return (
            f"runtime_replay_subject_mismatch:{envelope_type}:{subject_kind}",
        )
    return ()


__all__ = [
    "RUNTIME_REPLAY_REQUIRES_ROOT_PROVENANCE",
    "replay_runtime_admission",
]
