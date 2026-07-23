"""Private, canonical receipts for Phase-6 runtime evidence.

These records are deliberately not exported from :mod:`datadiff_osc.runtime`.
They bind exact runtime identities and raw observations, but they do not prove
producer provenance.  Root must independently bind a receipt to a real
execution before any gate authority is granted.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Iterable

from datadiff_osc._canonical import stable_digest, to_primitive
from datadiff_osc.schemas import EvidenceState, ExecutionStatus


V3_RUN_RECEIPT_SCHEMA_VERSION = "osc-root-v3-run-receipt-v1"
REPOSITORY_TEST_RECEIPT_SCHEMA_VERSION = (
    "osc-root-repository-test-receipt-v1"
)
TARGET_VERSION_RECEIPT_SCHEMA_VERSION = "osc-root-target-version-receipt-v1"
CANDIDATE_CLASSIFICATION_RECEIPT_SCHEMA_VERSION = (
    "osc-root-candidate-classification-v1"
)
PAIRED_BENCHMARK_RECEIPT_SCHEMA_VERSION = (
    "osc-root-paired-benchmark-receipt-v1"
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_REPOSITORY_OUTCOMES = frozenset(
    {"passed", "failed", "error", "skipped", "xfailed", "xpassed"}
)
_VERSION_SOURCE_KINDS = frozenset(
    {"pypi_json", "root_frozen_public_index"}
)
_CANDIDATE_STATES = frozenset(
    {
        EvidenceState.FINDING,
        EvidenceState.STABLE_SURVIVOR,
        EvidenceState.NATIVE_REPRODUCIBLE,
        EvidenceState.MINIMIZED,
    }
)


def _require_text(name: str, value: object, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        qualifier = "a string" if allow_empty else "a non-empty string"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _require_bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _require_nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _require_finite(
    name: str,
    value: object,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        finite = math.isfinite(float(value))
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not finite:
        raise ValueError(f"{name} must be a finite number")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _require_tuple(name: str, value: object, *, nonempty: bool = False) -> tuple:
    if not isinstance(value, tuple):
        raise ValueError(f"{name} must be an immutable tuple")
    if nonempty and not value:
        raise ValueError(f"{name} must be non-empty")
    return value


def _require_unique_sorted(
    name: str,
    values: tuple[object, ...],
    *,
    key,
) -> None:
    identities = tuple(key(item) for item in values)
    if any(not isinstance(item, str) or not item for item in identities):
        raise ValueError(f"{name} identities must be non-empty strings")
    if len(identities) != len(set(identities)):
        raise ValueError(f"{name} identities must be unique")
    if identities != tuple(sorted(identities)):
        raise ValueError(f"{name} identities must be sorted")


def _require_schema(name: str, actual: str, expected: str) -> None:
    if actual != expected:
        raise ValueError(f"{name} schema version mismatch")


@dataclass(frozen=True, slots=True)
class V3BackendTaskBinding:
    task_id: str
    endpoint_digest: str
    task_spec_digest: str
    execution_outcome_digest: str
    status: ExecutionStatus

    def __post_init__(self) -> None:
        for name in (
            "task_id",
            "endpoint_digest",
            "task_spec_digest",
            "execution_outcome_digest",
        ):
            _require_text(name, getattr(self, name))
        if not isinstance(self.status, ExecutionStatus):
            raise ValueError("backend task status must be an ExecutionStatus")

    @property
    def digest(self) -> str:
        return stable_digest("osc-root-v3-backend-task-binding", self)


@dataclass(frozen=True, slots=True)
class V3CaseBinding:
    case_id: str
    case_index: int
    seed_lineage_digest: str
    case_task_id: str
    case_task_spec_digest: str
    result_digest: str
    pipeline_digest: str
    execution_outcome_digest: str
    evidence_envelope_digest: str
    executed: bool
    iteration_failure: bool
    pipeline_error: bool
    backend_tasks: tuple[V3BackendTaskBinding, ...]

    def __post_init__(self) -> None:
        for name in (
            "case_id",
            "seed_lineage_digest",
            "case_task_id",
            "case_task_spec_digest",
            "result_digest",
            "pipeline_digest",
            "execution_outcome_digest",
            "evidence_envelope_digest",
        ):
            _require_text(name, getattr(self, name))
        _require_nonnegative_int("case_index", self.case_index)
        for name in ("executed", "iteration_failure", "pipeline_error"):
            _require_bool(name, getattr(self, name))
        _require_tuple("backend_tasks", self.backend_tasks, nonempty=True)
        if any(
            not isinstance(item, V3BackendTaskBinding)
            for item in self.backend_tasks
        ):
            raise ValueError("backend_tasks must contain V3BackendTaskBinding values")
        _require_unique_sorted(
            "backend task",
            self.backend_tasks,
            key=lambda item: item.task_id,
        )
        endpoints = tuple(item.endpoint_digest for item in self.backend_tasks)
        if len(endpoints) != len(set(endpoints)):
            raise ValueError("backend endpoint bindings must be unique per case")

    @property
    def digest(self) -> str:
        return stable_digest("osc-root-v3-case-binding", self)


@dataclass(frozen=True, slots=True)
class V3RunReceipt:
    source_digest: str
    protocol_digest: str
    v3_gate_plan_digest: str
    run_id: str
    lane_id: str
    seed: int
    execution_run_digest: str
    run_completed: bool
    cases: tuple[V3CaseBinding, ...]
    schema_version: str = V3_RUN_RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "source_digest",
            "protocol_digest",
            "v3_gate_plan_digest",
            "run_id",
            "lane_id",
            "execution_run_digest",
        ):
            _require_text(name, getattr(self, name))
        _require_nonnegative_int("seed", self.seed)
        _require_bool("run_completed", self.run_completed)
        _require_schema(
            "V3 run receipt", self.schema_version, V3_RUN_RECEIPT_SCHEMA_VERSION
        )
        _require_tuple("cases", self.cases, nonempty=True)
        if len(self.cases) != 100:
            raise ValueError("V3 run receipt requires exactly 100 cases")
        if any(not isinstance(item, V3CaseBinding) for item in self.cases):
            raise ValueError("cases must contain V3CaseBinding values")
        if tuple(item.case_index for item in self.cases) != tuple(range(100)):
            raise ValueError("V3 case indices must be exactly 0 through 99")
        case_ids = tuple(item.case_id for item in self.cases)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("V3 case identities must be unique")
        case_task_ids = tuple(item.case_task_id for item in self.cases)
        if len(case_task_ids) != len(set(case_task_ids)):
            raise ValueError("V3 case task identities must be unique")
        backend_task_ids = tuple(
            task.task_id for case in self.cases for task in case.backend_tasks
        )
        if len(backend_task_ids) != len(set(backend_task_ids)):
            raise ValueError("V3 backend task identities must be globally unique")
        if self.run_completed and any(
            not case.executed or case.iteration_failure or case.pipeline_error
            for case in self.cases
        ):
            raise ValueError(
                "a completed V3 run cannot contain an unexecuted or failed case"
            )

    @property
    def digest(self) -> str:
        return stable_digest("osc-root-v3-run-receipt", self)

    @property
    def case_ids(self) -> tuple[str, ...]:
        return tuple(sorted(item.case_id for item in self.cases))

    @property
    def executed_case_ids(self) -> tuple[str, ...]:
        return tuple(sorted(item.case_id for item in self.cases if item.executed))

    @property
    def backend_task_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(task.task_id for case in self.cases for task in case.backend_tasks)
        )

    def to_dict(self) -> dict[str, object]:
        return to_primitive(self)


@dataclass(frozen=True, slots=True)
class RepositoryTestNodeBinding:
    node_id: str
    outcome: str
    result_digest: str

    def __post_init__(self) -> None:
        _require_text("node_id", self.node_id)
        _require_text("outcome", self.outcome)
        _require_text("result_digest", self.result_digest)
        if self.outcome not in _REPOSITORY_OUTCOMES:
            raise ValueError("repository test outcome is not recognized")

    @property
    def digest(self) -> str:
        return stable_digest("osc-root-repository-test-node", self)


def repository_test_collection_digest(node_ids: Iterable[str]) -> str:
    materialized = tuple(node_ids)
    if not materialized or any(
        not isinstance(item, str) or not item for item in materialized
    ):
        raise ValueError("repository collection requires non-empty node identities")
    if len(materialized) != len(set(materialized)):
        raise ValueError("repository collection node identities must be unique")
    if materialized != tuple(sorted(materialized)):
        raise ValueError("repository collection node identities must be sorted")
    return stable_digest("osc-root-repository-test-collection", materialized)


@dataclass(frozen=True, slots=True)
class RepositoryTestReceipt:
    source_digest: str
    test_plan_digest: str
    collection_digest: str
    config_digest: str
    environment_digest: str
    command_digest: str
    junit_sha256: str
    log_sha256: str
    exit_code: int
    nodes: tuple[RepositoryTestNodeBinding, ...]
    schema_version: str = REPOSITORY_TEST_RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "source_digest",
            "test_plan_digest",
            "collection_digest",
            "config_digest",
            "environment_digest",
            "command_digest",
        ):
            _require_text(name, getattr(self, name))
        _require_sha256("junit_sha256", self.junit_sha256)
        _require_sha256("log_sha256", self.log_sha256)
        _require_nonnegative_int("exit_code", self.exit_code)
        _require_schema(
            "repository test receipt",
            self.schema_version,
            REPOSITORY_TEST_RECEIPT_SCHEMA_VERSION,
        )
        _require_tuple("nodes", self.nodes, nonempty=True)
        if any(not isinstance(item, RepositoryTestNodeBinding) for item in self.nodes):
            raise ValueError("nodes must contain RepositoryTestNodeBinding values")
        _require_unique_sorted("repository test node", self.nodes, key=lambda item: item.node_id)
        expected_collection = repository_test_collection_digest(self.node_ids)
        if self.collection_digest != expected_collection:
            raise ValueError("repository test collection digest mismatch")
        if self.exit_code == 0 and any(
            item.outcome in {"failed", "error"} for item in self.nodes
        ):
            raise ValueError("zero repository test exit cannot contain failures")

    @property
    def digest(self) -> str:
        return stable_digest("osc-root-repository-test-receipt", self)

    @property
    def node_ids(self) -> tuple[str, ...]:
        return tuple(item.node_id for item in self.nodes)

    @property
    def failed_node_ids(self) -> tuple[str, ...]:
        return tuple(
            item.node_id for item in self.nodes if item.outcome in {"failed", "error"}
        )

    def to_dict(self) -> dict[str, object]:
        return to_primitive(self)


@dataclass(frozen=True, slots=True)
class TargetPackageVersionBinding:
    package_id: str
    distribution_name: str
    import_name: str
    installed_version: str
    latest_version: str
    installed_metadata_digest: str
    version_source_kind: str
    version_source_digest: str
    version_source_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "package_id",
            "distribution_name",
            "import_name",
            "installed_version",
            "latest_version",
            "installed_metadata_digest",
            "version_source_kind",
            "version_source_digest",
        ):
            _require_text(name, getattr(self, name))
        _require_sha256("version_source_sha256", self.version_source_sha256)
        if self.version_source_kind not in _VERSION_SOURCE_KINDS:
            raise ValueError("target version source is not authority eligible")

    @property
    def matches_latest(self) -> bool:
        return self.installed_version == self.latest_version

    @property
    def digest(self) -> str:
        return stable_digest("osc-root-target-package-version", self)


@dataclass(frozen=True, slots=True)
class TargetVersionReceipt:
    source_digest: str
    audit_plan_digest: str
    environment_digest: str
    python_runtime_digest: str
    packages: tuple[TargetPackageVersionBinding, ...]
    schema_version: str = TARGET_VERSION_RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "source_digest",
            "audit_plan_digest",
            "environment_digest",
            "python_runtime_digest",
        ):
            _require_text(name, getattr(self, name))
        _require_schema(
            "target version receipt",
            self.schema_version,
            TARGET_VERSION_RECEIPT_SCHEMA_VERSION,
        )
        _require_tuple("packages", self.packages, nonempty=True)
        if any(not isinstance(item, TargetPackageVersionBinding) for item in self.packages):
            raise ValueError("packages must contain TargetPackageVersionBinding values")
        _require_unique_sorted("target package", self.packages, key=lambda item: item.package_id)
        distributions = tuple(item.distribution_name for item in self.packages)
        imports = tuple(item.import_name for item in self.packages)
        if len(distributions) != len(set(distributions)):
            raise ValueError("target package distributions must be unique")
        if len(imports) != len(set(imports)):
            raise ValueError("target package imports must be unique")

    @property
    def digest(self) -> str:
        return stable_digest("osc-root-target-version-receipt", self)

    @property
    def package_ids(self) -> tuple[str, ...]:
        return tuple(item.package_id for item in self.packages)

    @property
    def mismatched_package_ids(self) -> tuple[str, ...]:
        return tuple(item.package_id for item in self.packages if not item.matches_latest)

    @property
    def all_match_latest(self) -> bool:
        return not self.mismatched_package_ids

    def to_dict(self) -> dict[str, object]:
        return to_primitive(self)


@dataclass(frozen=True, slots=True)
class CandidateRecheckBinding:
    recheck_id: str
    ordinal: int
    task_id: str
    seed_lineage_digest: str
    execution_outcome_digest: str
    completed: bool
    reproduced: bool

    def __post_init__(self) -> None:
        for name in (
            "recheck_id",
            "task_id",
            "seed_lineage_digest",
            "execution_outcome_digest",
        ):
            _require_text(name, getattr(self, name))
        _require_nonnegative_int("ordinal", self.ordinal)
        _require_bool("completed", self.completed)
        _require_bool("reproduced", self.reproduced)
        if self.reproduced and not self.completed:
            raise ValueError("an incomplete recheck cannot be reproduced")

    @property
    def digest(self) -> str:
        return stable_digest("osc-root-candidate-recheck", self)


@dataclass(frozen=True, slots=True)
class CandidateClassificationReceipt:
    source_digest: str
    v3_gate_plan_digest: str
    classification_policy_digest: str
    classification_id: str
    source_case_id: str
    result_group_digest: str
    evidence_envelope_digest: str
    evidence_state: EvidenceState
    finding_record_id: str
    candidate_id: str
    classification_digest: str
    campaign_rechecks: tuple[CandidateRecheckBinding, ...]
    pipeline_rechecks: tuple[CandidateRecheckBinding, ...]
    bug_claimed: bool
    schema_version: str = CANDIDATE_CLASSIFICATION_RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "source_digest",
            "v3_gate_plan_digest",
            "classification_policy_digest",
            "classification_id",
            "source_case_id",
            "result_group_digest",
            "evidence_envelope_digest",
            "classification_digest",
        ):
            _require_text(name, getattr(self, name))
        _require_text("finding_record_id", self.finding_record_id, allow_empty=True)
        _require_text("candidate_id", self.candidate_id, allow_empty=True)
        if not isinstance(self.evidence_state, EvidenceState):
            raise ValueError("candidate evidence_state must be an EvidenceState")
        if self.evidence_state not in _CANDIDATE_STATES | {
            EvidenceState.NOT_A_CANDIDATE
        }:
            raise ValueError("confirmed evidence states are forbidden in candidate receipts")
        if self.bug_claimed is not False:
            raise ValueError("candidate receipt bug_claimed must be false")
        _require_schema(
            "candidate classification receipt",
            self.schema_version,
            CANDIDATE_CLASSIFICATION_RECEIPT_SCHEMA_VERSION,
        )
        for name, rechecks in (
            ("campaign_rechecks", self.campaign_rechecks),
            ("pipeline_rechecks", self.pipeline_rechecks),
        ):
            _require_tuple(name, rechecks)
            if any(not isinstance(item, CandidateRecheckBinding) for item in rechecks):
                raise ValueError(f"{name} must contain CandidateRecheckBinding values")
        if self.evidence_state is EvidenceState.NOT_A_CANDIDATE:
            if self.finding_record_id or self.candidate_id:
                raise ValueError("a non-candidate cannot carry candidate identities")
            if self.campaign_rechecks or self.pipeline_rechecks:
                raise ValueError("a non-candidate cannot carry rechecks")
        else:
            if not self.finding_record_id or not self.candidate_id:
                raise ValueError("a candidate requires finding and candidate identities")
            self._validate_rechecks("campaign", self.campaign_rechecks)
            self._validate_rechecks("pipeline", self.pipeline_rechecks)
            all_recheck_ids = tuple(
                item.recheck_id
                for item in self.campaign_rechecks + self.pipeline_rechecks
            )
            if len(all_recheck_ids) != len(set(all_recheck_ids)):
                raise ValueError("candidate recheck identities must be globally unique")

    @staticmethod
    def _validate_rechecks(
        name: str, rechecks: tuple[CandidateRecheckBinding, ...]
    ) -> None:
        if tuple(item.ordinal for item in rechecks) != (1, 2, 3):
            raise ValueError(f"{name} recheck ordinals must be exactly 1, 2, 3")
        identities = tuple(item.recheck_id for item in rechecks)
        if identities != tuple(sorted(identities)):
            raise ValueError(f"{name} recheck identities must be sorted")
        if len(identities) != len(set(identities)):
            raise ValueError(f"{name} recheck identities must be unique")
        task_ids = tuple(item.task_id for item in rechecks)
        if len(task_ids) != len(set(task_ids)):
            raise ValueError(f"{name} recheck task identities must be unique")

    @property
    def is_candidate(self) -> bool:
        return self.evidence_state in _CANDIDATE_STATES

    @property
    def digest(self) -> str:
        return stable_digest("osc-root-candidate-classification", self)

    @property
    def campaign_recheck_ids(self) -> tuple[str, ...]:
        return tuple(item.recheck_id for item in self.campaign_rechecks)

    @property
    def pipeline_recheck_ids(self) -> tuple[str, ...]:
        return tuple(item.recheck_id for item in self.pipeline_rechecks)

    def to_dict(self) -> dict[str, object]:
        return to_primitive(self)


@dataclass(frozen=True, slots=True)
class ContractPerformanceSampleBinding:
    sample_id: str
    case_id: str
    baseline_task_id: str
    treatment_task_id: str
    compile_match_ms: int | float
    case_wall_ms: int | float
    baseline_throughput_cases_s: int | float
    treatment_throughput_cases_s: int | float

    def __post_init__(self) -> None:
        for name in (
            "sample_id",
            "case_id",
            "baseline_task_id",
            "treatment_task_id",
        ):
            _require_text(name, getattr(self, name))
        if self.baseline_task_id == self.treatment_task_id:
            raise ValueError("paired contract sample tasks must be distinct")
        _require_finite("compile_match_ms", self.compile_match_ms, nonnegative=True)
        _require_finite("case_wall_ms", self.case_wall_ms, positive=True)
        _require_finite(
            "baseline_throughput_cases_s",
            self.baseline_throughput_cases_s,
            positive=True,
        )
        _require_finite(
            "treatment_throughput_cases_s",
            self.treatment_throughput_cases_s,
            nonnegative=True,
        )

    @property
    def compile_wall_share(self) -> float:
        return float(self.compile_match_ms) / float(self.case_wall_ms)

    @property
    def throughput_regression(self) -> float:
        return 1.0 - (
            float(self.treatment_throughput_cases_s)
            / float(self.baseline_throughput_cases_s)
        )

    @property
    def compile_sample_digest(self) -> str:
        return stable_digest(
            "osc-root-contract-compile-sample",
            (self.sample_id, self.compile_match_ms),
        )

    @property
    def case_wall_sample_digest(self) -> str:
        return stable_digest(
            "osc-root-contract-case-wall-sample",
            (self.sample_id, self.case_wall_ms),
        )

    @property
    def throughput_pair_digest(self) -> str:
        return stable_digest(
            "osc-root-contract-throughput-pair",
            (
                self.sample_id,
                self.baseline_throughput_cases_s,
                self.treatment_throughput_cases_s,
            ),
        )

    @property
    def digest(self) -> str:
        return stable_digest("osc-root-contract-performance-sample", self)


@dataclass(frozen=True, slots=True)
class ParallelScalingSampleBinding:
    sample_id: str
    workload_id: str
    one_worker_task_set_digest: str
    six_worker_task_set_digest: str
    one_worker_result_digest: str
    six_worker_result_digest: str
    one_worker_elapsed_seconds: int | float
    six_worker_elapsed_seconds: int | float

    def __post_init__(self) -> None:
        for name in (
            "sample_id",
            "workload_id",
            "one_worker_task_set_digest",
            "six_worker_task_set_digest",
            "one_worker_result_digest",
            "six_worker_result_digest",
        ):
            _require_text(name, getattr(self, name))
        _require_finite(
            "one_worker_elapsed_seconds",
            self.one_worker_elapsed_seconds,
            positive=True,
        )
        _require_finite(
            "six_worker_elapsed_seconds",
            self.six_worker_elapsed_seconds,
            positive=True,
        )
        if self.one_worker_task_set_digest != self.six_worker_task_set_digest:
            raise ValueError("parallel scaling task-set identity mismatch")

    @property
    def six_worker_efficiency(self) -> float:
        return float(self.one_worker_elapsed_seconds) / (
            6.0 * float(self.six_worker_elapsed_seconds)
        )

    @property
    def digest(self) -> str:
        return stable_digest("osc-root-parallel-scaling-sample", self)


@dataclass(frozen=True, slots=True)
class PairedBenchmarkReceipt:
    source_digest: str
    benchmark_plan_digest: str
    environment_digest: str
    baseline_config_digest: str
    treatment_config_digest: str
    contract_samples: tuple[ContractPerformanceSampleBinding, ...]
    parallel_samples: tuple[ParallelScalingSampleBinding, ...]
    schema_version: str = PAIRED_BENCHMARK_RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "source_digest",
            "benchmark_plan_digest",
            "environment_digest",
            "baseline_config_digest",
            "treatment_config_digest",
        ):
            _require_text(name, getattr(self, name))
        if self.baseline_config_digest == self.treatment_config_digest:
            raise ValueError("paired benchmark configurations must be distinct")
        _require_schema(
            "paired benchmark receipt",
            self.schema_version,
            PAIRED_BENCHMARK_RECEIPT_SCHEMA_VERSION,
        )
        _require_tuple("contract_samples", self.contract_samples, nonempty=True)
        _require_tuple("parallel_samples", self.parallel_samples, nonempty=True)
        if any(
            not isinstance(item, ContractPerformanceSampleBinding)
            for item in self.contract_samples
        ):
            raise ValueError(
                "contract_samples must contain ContractPerformanceSampleBinding values"
            )
        if any(
            not isinstance(item, ParallelScalingSampleBinding)
            for item in self.parallel_samples
        ):
            raise ValueError(
                "parallel_samples must contain ParallelScalingSampleBinding values"
            )
        _require_unique_sorted(
            "contract performance sample",
            self.contract_samples,
            key=lambda item: item.sample_id,
        )
        _require_unique_sorted(
            "parallel scaling sample",
            self.parallel_samples,
            key=lambda item: item.sample_id,
        )
        contract_ids = {item.sample_id for item in self.contract_samples}
        parallel_ids = {item.sample_id for item in self.parallel_samples}
        if contract_ids & parallel_ids:
            raise ValueError("paired benchmark sample identities must be globally unique")

    @property
    def digest(self) -> str:
        return stable_digest("osc-root-paired-benchmark-receipt", self)

    @property
    def contract_sample_ids(self) -> tuple[str, ...]:
        return tuple(item.sample_id for item in self.contract_samples)

    @property
    def parallel_sample_ids(self) -> tuple[str, ...]:
        return tuple(item.sample_id for item in self.parallel_samples)

    @property
    def compile_match_samples_ms(self) -> tuple[int | float, ...]:
        return tuple(item.compile_match_ms for item in self.contract_samples)

    @property
    def compile_wall_shares(self) -> tuple[float, ...]:
        return tuple(item.compile_wall_share for item in self.contract_samples)

    @property
    def throughput_regressions(self) -> tuple[float, ...]:
        return tuple(item.throughput_regression for item in self.contract_samples)

    @property
    def six_worker_efficiencies(self) -> tuple[float, ...]:
        return tuple(item.six_worker_efficiency for item in self.parallel_samples)

    def to_dict(self) -> dict[str, object]:
        return to_primitive(self)


__all__ = [
    "CANDIDATE_CLASSIFICATION_RECEIPT_SCHEMA_VERSION",
    "PAIRED_BENCHMARK_RECEIPT_SCHEMA_VERSION",
    "REPOSITORY_TEST_RECEIPT_SCHEMA_VERSION",
    "TARGET_VERSION_RECEIPT_SCHEMA_VERSION",
    "V3_RUN_RECEIPT_SCHEMA_VERSION",
    "CandidateClassificationReceipt",
    "CandidateRecheckBinding",
    "ContractPerformanceSampleBinding",
    "PairedBenchmarkReceipt",
    "ParallelScalingSampleBinding",
    "RepositoryTestNodeBinding",
    "RepositoryTestReceipt",
    "TargetPackageVersionBinding",
    "TargetVersionReceipt",
    "V3BackendTaskBinding",
    "V3CaseBinding",
    "V3RunReceipt",
    "repository_test_collection_digest",
]
