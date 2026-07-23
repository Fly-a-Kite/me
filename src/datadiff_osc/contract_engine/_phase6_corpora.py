"""Private, fail-closed Phase-6 corpus foundation.

This module validates candidate corpus material and independent approval
bindings.  It deliberately stops before Root authority: even a structurally
valid independent approval remains context-pending until Root binds it to real
producer evidence in a later, serial change request.

Nothing in this module writes corpus files, assigns false-positive fix IDs,
replays gates, or executes a corpus.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping

from datadiff.dsl import Case

from datadiff_osc._canonical import (
    canonical_json,
    stable_digest,
    to_primitive,
)
from datadiff_osc.schemas import (
    ContractFingerprint,
    EvidenceEnvelope,
    EvidenceState,
    EvidenceTier,
    ExecutionStatus,
    FailureKind,
    ResourceTokens,
    ResultGroup,
    StructuredExecutionOutcome,
    TargetFingerprint,
    TaskIdentity,
    TaskKind,
    TaskSpec,
    VerdictKind,
)
from datadiff_osc.contract_engine.mutations import (
    AXIS_FAULTS,
    COMPARATOR_WEAKENING_MUTANTS,
    HYPEREDGE_MUTANTS,
)


CORPUS_ENTRY_SCHEMA_VERSION = "osc-phase6-corpus-candidate-entry-v2"
CORPUS_MANIFEST_SCHEMA_VERSION = "osc-phase6-corpus-candidate-manifest-v2"
CORPUS_APPROVAL_SCHEMA_VERSION = "osc-phase6-corpus-structural-review-v2"
CORPUS_VALIDATION_SCHEMA_VERSION = "osc-phase6-corpus-validation-v2"
CONFIRMED_ROOT_EXPECTATION_SCHEMA_VERSION = (
    "osc-phase6-confirmed-root-expectation-v1"
)
FALSE_POSITIVE_ADJUDICATION_SCHEMA_VERSION = (
    "osc-phase6-false-positive-adjudication-v1"
)
TYPED_CASE_SOURCE_SCHEMA_VERSION = "osc-phase6-typed-case-source-v1"
EXACT_EXECUTION_RECORD_SCHEMA_VERSION = "osc-phase6-exact-execution-record-v1"
COMPATIBILITY_VERDICT_PAIR_SCHEMA_VERSION = (
    "osc-phase6-compatibility-verdict-pair-v1"
)
MUTANT_KILL_WITNESS_SCHEMA_VERSION = "osc-phase6-mutant-kill-witness-v1"

CONFIRMED_ROOT_MANIFEST_PATH = (
    "experiments/canonical_confirmed_bug_corpus/v2/manifest.json"
)
CONFIRMED_ROOT_MANIFEST_SHA256 = (
    "0203ec26781d32828ed9914cb510efbe2f8c7807d25d4addfaf3652a339b47f4"
)
MUTATION_SOURCE_PATH = "src/datadiff_osc/contract_engine/mutations.py"
MUTATION_SOURCE_SHA256 = (
    "03172519fe17dd4262739e630630a16a1a093fc104fcc8b9f991c0098ee580b8"
)
REQUIRED_CONFIRMED_ROOTS = 9
REQUIRED_PENDING_ROOTS = 1
REQUIRED_FALSE_POSITIVE_MINIMUM = 1

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_CONFIRMED_LEDGER_STATUSES = frozenset(
    {"upstream_labeled_bug", "fixed_upstream"}
)


class CorpusKind(str, Enum):
    CONFIRMED_ROOTS = "confirmed_roots"
    FALSE_POSITIVES = "false_positives"
    V1_V2_COMPATIBILITY = "v1_v2_compatibility"
    MUTANT_KILL_WITNESSES = "mutant_kill_witnesses"


class ExpectedVerdict(str, Enum):
    SATISFIED = "SATISFIED"
    VIOLATED = "VIOLATED"
    INAPPLICABLE = "INAPPLICABLE"
    INCONCLUSIVE = "INCONCLUSIVE"


class KillSemantics(str, Enum):
    COMPARATOR_WEAKENING_EXPOSED = "comparator_weakening_exposed"
    AXIS_FAULT_DETECTED = "axis_fault_detected"
    HYPEREDGE_MUTATION_EXPOSED = "hyperedge_mutation_exposed"


class CorpusValidationState(str, Enum):
    CANDIDATE_MATERIAL = "candidate_material"
    STRUCTURALLY_REVIEWED_CONTEXT_PENDING = (
        "structurally_reviewed_context_pending"
    )
    MISSING_INPUT = "missing_input"
    REJECTED = "rejected"


class CorpusSourceError(ValueError):
    """A frozen source cannot be reconstructed exactly."""


class CorpusDecodeError(ValueError):
    """A private candidate or approval document is not canonical and exact."""


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be non-empty text")
    return value


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _require_unique_sorted_text(
    values: tuple[str, ...], name: str, *, allow_empty: bool = False
) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple")
    if not allow_empty and not values:
        raise ValueError(f"{name} must be non-empty")
    if any(not isinstance(item, str) or not item for item in values):
        raise ValueError(f"{name} must contain non-empty text")
    if values != tuple(sorted(values)) or len(values) != len(set(values)):
        raise ValueError(f"{name} must be unique and sorted")
    return values


def _require_relative_path(value: object, name: str) -> str:
    text = _require_text(value, name)
    if any(ord(character) < 0x20 for character in text):
        raise ValueError(f"{name} must not contain C0 control characters")
    if "\\" in text:
        raise ValueError(f"{name} must use canonical POSIX separators")
    candidate = PurePosixPath(text)
    if (
        candidate.is_absolute()
        or str(candidate) != text
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise ValueError(f"{name} must be a canonical repository-relative path")
    return text


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strict_json_loads(payload: str) -> object:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise CorpusDecodeError(f"duplicate JSON field: {key}")
            result[key] = value
        return result

    try:
        return json.loads(payload, object_pairs_hook=reject_duplicates)
    except json.JSONDecodeError as exc:
        raise CorpusDecodeError(f"invalid JSON: {exc}") from exc


def _exact_mapping(
    value: object, *, fields: frozenset[str], name: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise CorpusDecodeError(f"{name} fields do not match the private schema")
    if any(not isinstance(key, str) for key in value):
        raise CorpusDecodeError(f"{name} fields must be strings")
    return value


def _resolved_bound_path(repo_root: Path, relative: str) -> Path:
    try:
        root = repo_root.resolve()
        resolved = (root / relative).resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise CorpusSourceError(
            f"source path cannot be resolved: {relative}:{type(exc).__name__}"
        ) from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise CorpusSourceError(f"source path escapes repository: {relative}") from exc
    return resolved


@dataclass(frozen=True, slots=True)
class SourceBinding:
    path: str
    sha256: str

    def __post_init__(self) -> None:
        _require_relative_path(self.path, "source path")
        _require_sha256(self.sha256, "source SHA-256")

    @property
    def digest(self) -> str:
        return stable_digest("osc-phase6-corpus-source-binding", self)


@dataclass(frozen=True, slots=True)
class ConfirmedRootExpectation:
    root_id: str
    expected_verdict: ExpectedVerdict
    expected_signature_digest: str
    schema_version: str = CONFIRMED_ROOT_EXPECTATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_text(self.root_id, "confirmed root ID")
        if self.expected_verdict is not ExpectedVerdict.VIOLATED:
            raise ValueError("confirmed root verdict must be frozen as VIOLATED")
        _require_text(
            self.expected_signature_digest,
            "confirmed root expected-signature digest",
        )
        if self.schema_version != CONFIRMED_ROOT_EXPECTATION_SCHEMA_VERSION:
            raise ValueError("confirmed root expectation schema mismatch")

    @property
    def digest(self) -> str:
        return stable_digest("osc-phase6-confirmed-root-expectation", self)


@dataclass(frozen=True, slots=True)
class TypedCaseSource:
    source: SourceBinding
    case_id: str
    case_digest: str
    schema_version: str = TYPED_CASE_SOURCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.source, SourceBinding):
            raise ValueError("typed case requires an exact source binding")
        _require_text(self.case_id, "typed case ID")
        _require_text(self.case_digest, "typed case digest")
        if self.schema_version != TYPED_CASE_SOURCE_SCHEMA_VERSION:
            raise ValueError("typed case source schema mismatch")

    @property
    def digest(self) -> str:
        return stable_digest("osc-phase6-typed-case-source", self)


@dataclass(frozen=True, slots=True)
class FalsePositiveAdjudication:
    finding_id: str
    component_id: str
    case: TypedCaseSource
    adjudication_source: SourceBinding
    observed_verdict: ExpectedVerdict
    adjudicated_verdict: ExpectedVerdict
    component_rule: str
    component_diff: str
    globally_ignored_axes: tuple[str, ...] = ()
    globally_ignored_families: tuple[str, ...] = ()
    schema_version: str = FALSE_POSITIVE_ADJUDICATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_text(self.finding_id, "false-positive finding ID")
        _require_text(self.component_id, "false-positive component ID")
        if not isinstance(self.case, TypedCaseSource):
            raise ValueError("false-positive adjudication requires a typed case")
        if not isinstance(self.adjudication_source, SourceBinding):
            raise ValueError("false-positive adjudication requires an exact source")
        if self.observed_verdict is not ExpectedVerdict.VIOLATED:
            raise ValueError("false-positive observed verdict must be VIOLATED")
        if self.adjudicated_verdict not in {
            ExpectedVerdict.SATISFIED,
            ExpectedVerdict.INAPPLICABLE,
            ExpectedVerdict.INCONCLUSIVE,
        }:
            raise ValueError("false-positive adjudicated verdict is invalid")
        _require_text(self.component_rule, "false-positive component rule")
        _require_text(self.component_diff, "false-positive component diff")
        if self.globally_ignored_axes != ():
            raise ValueError("false-positive evidence cannot globally ignore axes")
        if self.globally_ignored_families != ():
            raise ValueError("false-positive evidence cannot globally ignore families")
        if self.schema_version != FALSE_POSITIVE_ADJUDICATION_SCHEMA_VERSION:
            raise ValueError("false-positive adjudication schema mismatch")

    @property
    def digest(self) -> str:
        return stable_digest("osc-phase6-false-positive-adjudication", self)

    @property
    def adjudication_facts(self) -> dict[str, object]:
        return {
            "finding_id": self.finding_id,
            "component_id": self.component_id,
            "case_digest": self.case.case_digest,
            "observed_verdict": self.observed_verdict.value,
            "adjudicated_verdict": self.adjudicated_verdict.value,
            "component_rule": self.component_rule,
            "component_diff": self.component_diff,
            "globally_ignored_axes": [],
            "globally_ignored_families": [],
        }


@dataclass(frozen=True, slots=True)
class ExactExecutionRecord:
    task_spec: TaskSpec
    result_group: ResultGroup
    evidence: EvidenceEnvelope
    schema_version: str = EXACT_EXECUTION_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.task_spec, TaskSpec):
            raise ValueError("exact execution record requires a typed TaskSpec")
        if not isinstance(self.result_group, ResultGroup):
            raise ValueError("exact execution record requires a typed ResultGroup")
        if not isinstance(self.evidence, EvidenceEnvelope):
            raise ValueError("exact execution record requires typed evidence")
        task_id = self.task_spec.identity.task_id
        if self.result_group.task_ids != (task_id,):
            raise ValueError("result group is not exactly task-bound")
        if self.evidence.task_id != task_id:
            raise ValueError("execution evidence task identity mismatch")
        if (
            self.evidence.seed_lineage_digest
            != self.task_spec.identity.seed_lineage_digest
        ):
            raise ValueError("execution evidence seed lineage mismatch")
        if self.evidence.result_group_digest != self.result_group.digest:
            raise ValueError("execution evidence result identity mismatch")
        if self.evidence.contract_fingerprint != self.result_group.contract_fingerprint:
            raise ValueError("execution evidence contract mismatch")
        if self.evidence.target_fingerprint != self.result_group.target_fingerprint:
            raise ValueError("execution evidence target mismatch")
        outcome_endpoints = tuple(
            sorted(item.endpoint_id for item in self.evidence.execution_outcomes)
        )
        if outcome_endpoints != tuple(sorted(self.result_group.endpoint_ids)):
            raise ValueError("execution outcomes do not cover the exact endpoint set")
        if any(
            item.status is not ExecutionStatus.OK
            for item in self.evidence.execution_outcomes
        ):
            raise ValueError("corpus execution evidence requires OK outcomes")
        if self.schema_version != EXACT_EXECUTION_RECORD_SCHEMA_VERSION:
            raise ValueError("exact execution record schema mismatch")

    @property
    def task_id(self) -> str:
        return self.task_spec.identity.task_id

    @property
    def verdict(self) -> ExpectedVerdict:
        return ExpectedVerdict(self.evidence.verdict_kind.value)

    @property
    def digest(self) -> str:
        return stable_digest("osc-phase6-exact-execution-record", self)


def compatibility_task_payload_digest(
    *,
    case_digest: str,
    endpoint_ids: tuple[str, ...],
    contract_fingerprint: ContractFingerprint,
    implementation_version: str,
) -> str:
    if implementation_version not in {"v1", "v2"}:
        raise ValueError("compatibility implementation version must be v1 or v2")
    return stable_digest(
        "osc-phase6-compatibility-task-payload-v1",
        {
            "case_digest": case_digest,
            "endpoint_ids": endpoint_ids,
            "contract_fingerprint_digest": contract_fingerprint.digest,
            "implementation_version": implementation_version,
        },
    )


def mutant_task_payload_digest(
    *,
    case_digest: str,
    endpoint_ids: tuple[str, ...],
    contract_fingerprint: ContractFingerprint,
    role: str,
    mutant_id: str = "",
    mutant_version: str = "",
    mutant_source_digest: str = "",
) -> str:
    if role not in {"baseline", "mutant"}:
        raise ValueError("mutant task role must be baseline or mutant")
    if role == "mutant" and not all(
        (mutant_id, mutant_version, mutant_source_digest)
    ):
        raise ValueError("mutant task payload requires exact mutant identity")
    if role == "baseline" and any(
        (mutant_id, mutant_version, mutant_source_digest)
    ):
        raise ValueError("baseline task payload cannot carry mutant identity")
    return stable_digest(
        "osc-phase6-mutant-task-payload-v1",
        {
            "case_digest": case_digest,
            "endpoint_ids": endpoint_ids,
            "contract_fingerprint_digest": contract_fingerprint.digest,
            "role": role,
            "mutant_id": mutant_id,
            "mutant_version": mutant_version,
            "mutant_source_digest": mutant_source_digest,
        },
    )


def _validate_execution_context(
    record: ExactExecutionRecord,
    *,
    endpoint_ids: tuple[str, ...],
    contract_fingerprint: ContractFingerprint,
    payload_digest: str,
) -> None:
    if record.result_group.endpoint_ids != endpoint_ids:
        raise ValueError("execution record endpoint set mismatch")
    if record.result_group.contract_fingerprint != contract_fingerprint:
        raise ValueError("execution record contract fingerprint mismatch")
    if record.task_spec.payload_digest != payload_digest:
        raise ValueError("execution record task payload mismatch")
    task_endpoint = record.task_spec.identity.endpoint_id
    if task_endpoint and task_endpoint not in endpoint_ids:
        raise ValueError("execution task endpoint is outside the exact endpoint set")


@dataclass(frozen=True, slots=True)
class CompatibilityVerdictPair:
    pair_id: str
    case: TypedCaseSource
    endpoint_ids: tuple[str, ...]
    contract_fingerprint: ContractFingerprint
    v1: ExactExecutionRecord
    v2: ExactExecutionRecord
    schema_version: str = COMPATIBILITY_VERDICT_PAIR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_text(self.pair_id, "compatibility pair ID")
        if not isinstance(self.case, TypedCaseSource):
            raise ValueError("compatibility pair requires a typed case")
        _require_unique_sorted_text(self.endpoint_ids, "compatibility endpoints")
        if not isinstance(self.contract_fingerprint, ContractFingerprint):
            raise ValueError("compatibility pair requires a contract fingerprint")
        if not isinstance(self.v1, ExactExecutionRecord) or not isinstance(
            self.v2, ExactExecutionRecord
        ):
            raise ValueError("compatibility pair requires exact v1/v2 executions")
        for version, record in (("v1", self.v1), ("v2", self.v2)):
            _validate_execution_context(
                record,
                endpoint_ids=self.endpoint_ids,
                contract_fingerprint=self.contract_fingerprint,
                payload_digest=compatibility_task_payload_digest(
                    case_digest=self.case.case_digest,
                    endpoint_ids=self.endpoint_ids,
                    contract_fingerprint=self.contract_fingerprint,
                    implementation_version=version,
                ),
            )
        if self.v1.task_id == self.v2.task_id:
            raise ValueError("compatibility v1/v2 task identities must be distinct")
        if self.v1.result_group.result_group_id == self.v2.result_group.result_group_id:
            raise ValueError("compatibility v1/v2 result identities must be distinct")
        if (
            self.v1.task_spec.identity.seed_lineage_digest
            != self.v2.task_spec.identity.seed_lineage_digest
        ):
            raise ValueError("compatibility v1/v2 seed lineage mismatch")
        if (
            self.v1.task_spec.identity.protocol_digest
            != self.v2.task_spec.identity.protocol_digest
        ):
            raise ValueError("compatibility v1/v2 protocol mismatch")
        if self.v1.task_spec.identity.task_kind is not self.v2.task_spec.identity.task_kind:
            raise ValueError("compatibility v1/v2 task kind mismatch")
        if self.schema_version != COMPATIBILITY_VERDICT_PAIR_SCHEMA_VERSION:
            raise ValueError("compatibility verdict pair schema mismatch")

    @property
    def digest(self) -> str:
        return stable_digest("osc-phase6-compatibility-verdict-pair", self)


@dataclass(frozen=True, slots=True)
class MutantKillWitness:
    mutant_id: str
    mutant_version: str
    mutant_source: SourceBinding
    case: TypedCaseSource
    endpoint_ids: tuple[str, ...]
    contract_fingerprint: ContractFingerprint
    kill_semantics: KillSemantics
    baseline: ExactExecutionRecord
    mutant: ExactExecutionRecord
    schema_version: str = MUTANT_KILL_WITNESS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_text(self.mutant_id, "mutant witness ID")
        _require_text(self.mutant_version, "mutant witness version")
        if not isinstance(self.mutant_source, SourceBinding):
            raise ValueError("mutant witness requires exact source")
        if not isinstance(self.case, TypedCaseSource):
            raise ValueError("mutant witness requires a typed case")
        _require_unique_sorted_text(self.endpoint_ids, "mutant witness endpoints")
        if not isinstance(self.contract_fingerprint, ContractFingerprint):
            raise ValueError("mutant witness requires a contract fingerprint")
        if not isinstance(self.kill_semantics, KillSemantics):
            raise ValueError("mutant witness requires typed kill semantics")
        try:
            required_semantics = expected_kill_semantics(self.mutant_id)
        except ValueError as exc:
            raise ValueError("mutant witness identity is outside the frozen source") from exc
        if self.kill_semantics is not required_semantics:
            raise ValueError("mutant witness kill predicate does not match its category")
        if not isinstance(self.baseline, ExactExecutionRecord) or not isinstance(
            self.mutant, ExactExecutionRecord
        ):
            raise ValueError("mutant witness requires exact paired executions")
        _validate_execution_context(
            self.baseline,
            endpoint_ids=self.endpoint_ids,
            contract_fingerprint=self.contract_fingerprint,
            payload_digest=mutant_task_payload_digest(
                case_digest=self.case.case_digest,
                endpoint_ids=self.endpoint_ids,
                contract_fingerprint=self.contract_fingerprint,
                role="baseline",
            ),
        )
        _validate_execution_context(
            self.mutant,
            endpoint_ids=self.endpoint_ids,
            contract_fingerprint=self.contract_fingerprint,
            payload_digest=mutant_task_payload_digest(
                case_digest=self.case.case_digest,
                endpoint_ids=self.endpoint_ids,
                contract_fingerprint=self.contract_fingerprint,
                role="mutant",
                mutant_id=self.mutant_id,
                mutant_version=self.mutant_version,
                mutant_source_digest=self.mutant_source.digest,
            ),
        )
        if self.baseline.task_id == self.mutant.task_id:
            raise ValueError("baseline and mutant task identities must be distinct")
        if (
            self.baseline.result_group.result_group_id
            == self.mutant.result_group.result_group_id
        ):
            raise ValueError("baseline and mutant result identities must be distinct")
        if self.baseline.verdict == self.mutant.verdict:
            raise ValueError("mutant witness requires a verdict-changing kill")
        if (
            self.baseline.task_spec.identity.seed_lineage_digest
            != self.mutant.task_spec.identity.seed_lineage_digest
        ):
            raise ValueError("baseline and mutant seed lineage mismatch")
        if (
            self.baseline.task_spec.identity.protocol_digest
            != self.mutant.task_spec.identity.protocol_digest
        ):
            raise ValueError("baseline and mutant protocol mismatch")
        if (
            self.baseline.task_spec.identity.task_kind
            is not self.mutant.task_spec.identity.task_kind
        ):
            raise ValueError("baseline and mutant task kind mismatch")
        if self.schema_version != MUTANT_KILL_WITNESS_SCHEMA_VERSION:
            raise ValueError("mutant kill witness schema mismatch")

    @property
    def pair_identity(self) -> tuple[str, str]:
        return (self.baseline.task_id, self.mutant.task_id)

    @property
    def digest(self) -> str:
        return stable_digest("osc-phase6-mutant-kill-witness", self)


CorpusEntryEvidence = (
    ConfirmedRootExpectation
    | FalsePositiveAdjudication
    | CompatibilityVerdictPair
    | MutantKillWitness
)


@dataclass(frozen=True, slots=True)
class CorpusCandidateEntry:
    corpus_kind: CorpusKind
    entry_id: str
    source: SourceBinding
    evidence: CorpusEntryEvidence
    schema_version: str = CORPUS_ENTRY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.corpus_kind, CorpusKind):
            raise ValueError("corpus entry kind must be a CorpusKind")
        _require_text(self.entry_id, "corpus entry ID")
        if not isinstance(self.source, SourceBinding):
            raise ValueError("corpus entry source must be a SourceBinding")
        expected_type = {
            CorpusKind.CONFIRMED_ROOTS: ConfirmedRootExpectation,
            CorpusKind.FALSE_POSITIVES: FalsePositiveAdjudication,
            CorpusKind.V1_V2_COMPATIBILITY: CompatibilityVerdictPair,
            CorpusKind.MUTANT_KILL_WITNESSES: MutantKillWitness,
        }[self.corpus_kind]
        if not isinstance(self.evidence, expected_type):
            raise ValueError("corpus entry evidence type does not match its kind")
        if isinstance(self.evidence, ConfirmedRootExpectation):
            evidence_id = self.evidence.root_id
        elif isinstance(self.evidence, FalsePositiveAdjudication):
            evidence_id = self.evidence.finding_id
        elif isinstance(self.evidence, CompatibilityVerdictPair):
            evidence_id = self.evidence.pair_id
        else:
            assert isinstance(self.evidence, MutantKillWitness)
            evidence_id = self.evidence.mutant_id
        if self.entry_id != evidence_id:
            raise ValueError("corpus entry identity is not evidence-bound")
        if self.schema_version != CORPUS_ENTRY_SCHEMA_VERSION:
            raise ValueError("corpus entry schema version mismatch")

    @property
    def digest(self) -> str:
        return stable_digest("osc-phase6-corpus-candidate-entry", self)


@dataclass(frozen=True, slots=True)
class CorpusManifestCandidate:
    manifest_id: str
    corpus_kind: CorpusKind
    submitter_id: str
    entries: tuple[CorpusCandidateEntry, ...]
    schema_version: str = CORPUS_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_text(self.manifest_id, "candidate manifest ID")
        _require_text(self.submitter_id, "candidate manifest submitter")
        if not isinstance(self.corpus_kind, CorpusKind):
            raise ValueError("candidate manifest kind must be a CorpusKind")
        if self.schema_version != CORPUS_MANIFEST_SCHEMA_VERSION:
            raise ValueError("candidate manifest schema version mismatch")
        if not isinstance(self.entries, tuple) or not self.entries:
            raise ValueError("candidate manifest entries must be a non-empty tuple")
        if any(not isinstance(item, CorpusCandidateEntry) for item in self.entries):
            raise ValueError("candidate manifest contains an untyped entry")
        if any(item.corpus_kind is not self.corpus_kind for item in self.entries):
            raise ValueError("candidate manifest mixes corpus kinds")
        identities = tuple(item.entry_id for item in self.entries)
        if len(identities) != len(set(identities)):
            raise ValueError("candidate manifest entry identities must be unique")
        if identities != tuple(sorted(identities)):
            raise ValueError("candidate manifest entry identities must be sorted")

    @property
    def entry_ids(self) -> tuple[str, ...]:
        return tuple(item.entry_id for item in self.entries)

    @property
    def digest(self) -> str:
        return stable_digest("osc-phase6-corpus-candidate-manifest", self)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


@dataclass(frozen=True, slots=True)
class IndependentApprovalRecord:
    approval_id: str
    manifest_digest: str
    manifest_source: SourceBinding
    submitter_id: str
    approver_id: str
    reviewed_entry_ids: tuple[str, ...]
    decision: str = "approved"
    schema_version: str = CORPUS_APPROVAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_text(self.approval_id, "approval ID")
        _require_text(self.manifest_digest, "approved manifest digest")
        if not isinstance(self.manifest_source, SourceBinding):
            raise ValueError("approval manifest source must be a SourceBinding")
        _require_text(self.submitter_id, "approval submitter")
        _require_text(self.approver_id, "approval reviewer")
        if self.submitter_id == self.approver_id:
            raise ValueError("a corpus candidate cannot approve itself")
        if (
            not isinstance(self.reviewed_entry_ids, tuple)
            or not self.reviewed_entry_ids
            or any(not isinstance(item, str) or not item for item in self.reviewed_entry_ids)
        ):
            raise ValueError("approval requires non-empty reviewed entry identities")
        if self.reviewed_entry_ids != tuple(sorted(self.reviewed_entry_ids)) or len(
            self.reviewed_entry_ids
        ) != len(set(self.reviewed_entry_ids)):
            raise ValueError("reviewed entry identities must be unique and sorted")
        if self.decision != "approved":
            raise ValueError("independent approval decision must be exactly approved")
        if self.schema_version != CORPUS_APPROVAL_SCHEMA_VERSION:
            raise ValueError("approval schema version mismatch")

    @property
    def digest(self) -> str:
        return stable_digest("osc-phase6-corpus-independent-approval", self)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


@dataclass(frozen=True, slots=True)
class RootSourceEntry:
    root_id: str
    issue_url: str
    ledger_status: str
    case_source: SourceBinding
    expected_signature: str

    def __post_init__(self) -> None:
        for name in ("root_id", "issue_url", "ledger_status", "expected_signature"):
            _require_text(getattr(self, name), name)
        if self.ledger_status not in _CONFIRMED_LEDGER_STATUSES:
            raise ValueError("root source entry is not independently confirmed")

    @property
    def expected_signature_digest(self) -> str:
        return stable_digest(
            "osc-phase6-confirmed-root-expected-signature-v1",
            {
                "root_id": self.root_id,
                "expected_signature": self.expected_signature,
            },
        )


@dataclass(frozen=True, slots=True)
class ConfirmedRootInventory:
    manifest_source: SourceBinding
    confirmed: tuple[RootSourceEntry, ...]
    excluded_pending_ids: tuple[str, ...]

    @property
    def confirmed_ids(self) -> tuple[str, ...]:
        return tuple(item.root_id for item in self.confirmed)

    @property
    def digest(self) -> str:
        return stable_digest("osc-phase6-confirmed-root-source-inventory", self)


@dataclass(frozen=True, slots=True)
class MutantInventory:
    source: SourceBinding
    comparator_ids: tuple[str, ...]
    axis_ids: tuple[str, ...]
    hyperedge_ids: tuple[str, ...]

    @property
    def all_ids(self) -> tuple[str, ...]:
        return tuple(sorted((*self.comparator_ids, *self.axis_ids, *self.hyperedge_ids)))

    @property
    def digest(self) -> str:
        return stable_digest("osc-phase6-mutant-source-inventory", self)


@dataclass(frozen=True, slots=True)
class CorpusValidation:
    corpus_kind: CorpusKind
    state: CorpusValidationState
    entry_ids: tuple[str, ...]
    errors: tuple[str, ...]
    manifest_digest: str = ""
    approval_record_sha256: str = ""
    schema_version: str = CORPUS_VALIDATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.corpus_kind, CorpusKind):
            raise ValueError("validation kind must be a CorpusKind")
        if not isinstance(self.state, CorpusValidationState):
            raise ValueError("validation state must be a CorpusValidationState")
        if self.entry_ids != tuple(sorted(self.entry_ids)) or len(
            self.entry_ids
        ) != len(set(self.entry_ids)):
            raise ValueError("validation entry identities must be unique and sorted")
        if len(self.errors) != len(set(self.errors)) or any(not item for item in self.errors):
            raise ValueError("validation errors must be unique non-empty strings")
        if self.state in {
            CorpusValidationState.CANDIDATE_MATERIAL,
            CorpusValidationState.STRUCTURALLY_REVIEWED_CONTEXT_PENDING,
        } and (self.errors or not self.entry_ids or not self.manifest_digest):
            raise ValueError("successful candidate validation is incomplete")
        if self.state is CorpusValidationState.STRUCTURALLY_REVIEWED_CONTEXT_PENDING:
            _require_sha256(self.approval_record_sha256, "approval record SHA-256")
        elif self.approval_record_sha256:
            raise ValueError("only an approved-context result may bind an approval SHA")
        if self.schema_version != CORPUS_VALIDATION_SCHEMA_VERSION:
            raise ValueError("corpus validation schema version mismatch")

    @property
    def candidate_material_valid(self) -> bool:
        return self.state in {
            CorpusValidationState.CANDIDATE_MATERIAL,
            CorpusValidationState.STRUCTURALLY_REVIEWED_CONTEXT_PENDING,
        }

    @property
    def structurally_reviewed(self) -> bool:
        return self.state is CorpusValidationState.STRUCTURALLY_REVIEWED_CONTEXT_PENDING

    @property
    def independently_approved(self) -> bool:
        """Only a later Root-owned trusted context may establish this."""

        return False

    @property
    def root_authority_eligible(self) -> bool:
        """Foundation validation can never grant Root or gate authority."""

        return False

    @property
    def root_authority_state(self) -> str:
        return (
            "not_evaluated_root_context_required"
            if self.structurally_reviewed
            else "not_evaluated"
        )

    def meets_independently_approved_minimum(self, minimum: int) -> bool:
        if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
            raise ValueError("minimum corpus denominator must be a positive integer")
        return False

    def root_denominator_eligible(self, minimum: int) -> bool:
        if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
            raise ValueError("minimum corpus denominator must be a positive integer")
        return False


def _read_source(repo_root: Path, binding: SourceBinding) -> bytes:
    path = _resolved_bound_path(repo_root, binding.path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise CorpusSourceError(
            f"source is unavailable: {binding.path}:{type(exc).__name__}"
        ) from exc
    observed = _sha256_bytes(raw)
    if observed != binding.sha256:
        raise CorpusSourceError(
            f"source SHA-256 mismatch: {binding.path}:{observed}"
        )
    return raw


def _reconstruct_bound_case(repo_root: Path, binding: SourceBinding) -> Case:
    raw = _read_source(repo_root, binding)
    try:
        decoded = _strict_json_loads(raw.decode("utf-8"))
        if not isinstance(decoded, Mapping):
            raise CorpusDecodeError("typed case source must decode to an object")
        case = Case.from_dict(dict(decoded))
        if canonical_json(case.to_dict()) != canonical_json(decoded):
            raise CorpusDecodeError("typed case source contains unsupported fields")
        _require_text(case.case_id, "typed case ID")
        if isinstance(case.seed, bool) or not isinstance(case.seed, int) or case.seed < 0:
            raise CorpusDecodeError("typed case seed must be a non-negative integer")
    except (
        CorpusDecodeError,
        KeyError,
        TypeError,
        UnicodeDecodeError,
        ValueError,
    ) as exc:
        raise CorpusSourceError(f"typed case source is invalid: {exc}") from exc
    return case


def derive_typed_case_source(
    repo_root: Path, source: SourceBinding
) -> TypedCaseSource:
    if not isinstance(source, SourceBinding):
        raise TypeError("typed case derivation requires a SourceBinding")
    case = _reconstruct_bound_case(repo_root, source)
    return TypedCaseSource(
        source=source,
        case_id=case.case_id,
        case_digest=stable_digest("osc-phase6-typed-case", case.to_dict()),
    )


def _validate_typed_case_source(repo_root: Path, value: TypedCaseSource) -> None:
    if derive_typed_case_source(repo_root, value.source) != value:
        raise CorpusSourceError("typed case identity does not match its source")


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise CorpusSourceError(f"{name} must be an object with string fields")
    return value


def _list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise CorpusSourceError(f"{name} must be a list")
    return value


def derive_confirmed_root_inventory(repo_root: Path) -> ConfirmedRootInventory:
    """Reconstruct the exact 9-root source inventory and exclude pending rows."""

    manifest_source = SourceBinding(
        CONFIRMED_ROOT_MANIFEST_PATH, CONFIRMED_ROOT_MANIFEST_SHA256
    )
    raw = _read_source(repo_root, manifest_source)
    try:
        decoded = _strict_json_loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, CorpusDecodeError) as exc:
        raise CorpusSourceError(f"confirmed-root manifest is invalid: {exc}") from exc
    payload = _mapping(decoded, "confirmed-root manifest")
    if payload.get("schema_version") != "canonical-confirmed-bug-corpus-v1":
        raise CorpusSourceError("confirmed-root manifest schema mismatch")
    target = _mapping(payload.get("target"), "confirmed-root target")
    confirmed_rows = _list(payload.get("confirmed_roots"), "confirmed roots")
    pending_rows = _list(payload.get("pending_roots"), "pending roots")
    if (
        target.get("confirmed_root_count") != REQUIRED_CONFIRMED_ROOTS
        or len(confirmed_rows) != REQUIRED_CONFIRMED_ROOTS
    ):
        raise CorpusSourceError("confirmed-root denominator is not exactly 9")
    if (
        target.get("pending_root_count") != REQUIRED_PENDING_ROOTS
        or len(pending_rows) != REQUIRED_PENDING_ROOTS
    ):
        raise CorpusSourceError("pending-root inventory is not exactly 1")

    confirmed: list[RootSourceEntry] = []
    for index, raw_row in enumerate(confirmed_rows):
        row = _mapping(raw_row, f"confirmed root {index}")
        root_id = _require_text(row.get("root_id"), f"confirmed root {index} ID")
        ledger_status = _require_text(
            row.get("ledger_status"), f"confirmed root {root_id} ledger status"
        )
        case = _mapping(row.get("dsl_case"), f"confirmed root {root_id} case")
        case_source = SourceBinding(
            _require_relative_path(case.get("path"), "confirmed case path"),
            _require_sha256(case.get("sha256"), "confirmed case SHA-256"),
        )
        _read_source(repo_root, case_source)
        confirmed.append(
            RootSourceEntry(
                root_id=root_id,
                issue_url=_require_text(row.get("issue_url"), "confirmed issue URL"),
                ledger_status=ledger_status,
                case_source=case_source,
                expected_signature=_require_text(
                    row.get("expected_signature"), "confirmed expected signature"
                ),
            )
        )
    confirmed.sort(key=lambda item: item.root_id)
    confirmed_ids = tuple(item.root_id for item in confirmed)
    if len(confirmed_ids) != len(set(confirmed_ids)):
        raise CorpusSourceError("confirmed-root identities are duplicated")

    pending_ids: list[str] = []
    for index, raw_row in enumerate(pending_rows):
        row = _mapping(raw_row, f"pending root {index}")
        root_id = _require_text(row.get("root_id"), f"pending root {index} ID")
        if row.get("count_as_confirmed") is not False:
            raise CorpusSourceError("pending root does not fail closed")
        case = _mapping(row.get("dsl_case"), f"pending root {root_id} case")
        pending_source = SourceBinding(
            _require_relative_path(case.get("path"), "pending case path"),
            _require_sha256(case.get("sha256"), "pending case SHA-256"),
        )
        _read_source(repo_root, pending_source)
        pending_ids.append(root_id)
    if len(pending_ids) != len(set(pending_ids)) or set(pending_ids) & set(
        confirmed_ids
    ):
        raise CorpusSourceError("pending-root identities overlap or repeat")

    source = _mapping(payload.get("source"), "confirmed-root source")
    issue_manifest = SourceBinding(
        _require_relative_path(source.get("issue_bundle_manifest"), "issue manifest path"),
        _require_sha256(
            source.get("issue_bundle_manifest_sha256"), "issue manifest SHA-256"
        ),
    )
    _read_source(repo_root, issue_manifest)
    native = _mapping(payload.get("native_reproducer"), "native reproducer")
    native_source = SourceBinding(
        _require_relative_path(native.get("path"), "native reproducer path"),
        _require_sha256(native.get("sha256"), "native reproducer SHA-256"),
    )
    _read_source(repo_root, native_source)
    ledger_path = _require_relative_path(
        source.get("confirmation_ledger"), "confirmation ledger path"
    )
    ledger_expected = _require_sha256(
        source.get("confirmation_ledger_canonical_json_sha256"),
        "confirmation ledger canonical SHA-256",
    )
    ledger_resolved = _resolved_bound_path(repo_root, ledger_path)
    try:
        ledger_value = _strict_json_loads(ledger_resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, CorpusDecodeError) as exc:
        raise CorpusSourceError(f"confirmation ledger is invalid: {exc}") from exc
    ledger_observed = _sha256_bytes(canonical_json(ledger_value).encode("utf-8"))
    if ledger_observed != ledger_expected:
        raise CorpusSourceError("confirmation ledger canonical SHA-256 mismatch")

    return ConfirmedRootInventory(
        manifest_source=manifest_source,
        confirmed=tuple(confirmed),
        excluded_pending_ids=tuple(sorted(pending_ids)),
    )


def derive_mutant_inventory(repo_root: Path) -> MutantInventory:
    """Derive category-separated mutant IDs from the exact frozen source."""

    source = SourceBinding(MUTATION_SOURCE_PATH, MUTATION_SOURCE_SHA256)
    _read_source(repo_root, source)
    comparator = tuple(f"comparator:{item}" for item in COMPARATOR_WEAKENING_MUTANTS)
    axes = tuple(f"axis:{item}" for item in AXIS_FAULTS)
    hyperedges = tuple(f"hyperedge:{item}" for item in HYPEREDGE_MUTANTS)
    if (len(comparator), len(axes), len(hyperedges)) != (12, 13, 7):
        raise CorpusSourceError("mutant category cardinalities are not 12/13/7")
    all_ids = (*comparator, *axes, *hyperedges)
    if len(all_ids) != 32 or len(all_ids) != len(set(all_ids)) or any(
        not item.split(":", 1)[-1] for item in all_ids
    ):
        raise CorpusSourceError("mutant identity universe is not exactly 32")
    return MutantInventory(source, comparator, axes, hyperedges)


def expected_kill_semantics(mutant_id: str) -> KillSemantics:
    if mutant_id.startswith("comparator:"):
        return KillSemantics.COMPARATOR_WEAKENING_EXPOSED
    if mutant_id.startswith("axis:"):
        return KillSemantics.AXIS_FAULT_DETECTED
    if mutant_id.startswith("hyperedge:"):
        return KillSemantics.HYPEREDGE_MUTATION_EXPOSED
    raise ValueError(f"unknown mutant identity: {mutant_id}")


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CorpusDecodeError(f"{name} must be a non-negative integer")
    return value


def _optional_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise CorpusDecodeError(f"{name} must be text")
    return value


def _text_tuple_from_list(
    value: object,
    name: str,
    *,
    allow_empty: bool = False,
    sorted_values: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise CorpusDecodeError(f"{name} must be a list")
    result = tuple(value)
    if not allow_empty and not result:
        raise CorpusDecodeError(f"{name} must be non-empty")
    if any(not isinstance(item, str) or not item for item in result):
        raise CorpusDecodeError(f"{name} must contain non-empty text")
    if len(result) != len(set(result)):
        raise CorpusDecodeError(f"{name} must be unique")
    if sorted_values and result != tuple(sorted(result)):
        raise CorpusDecodeError(f"{name} must be sorted")
    return result


def _enum_value(enum_type: type[Enum], value: object, name: str) -> Enum:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise CorpusDecodeError(f"{name} is invalid: {value!r}") from exc


def _source_binding_from_mapping(value: object, name: str) -> SourceBinding:
    data = _exact_mapping(value, fields=frozenset({"path", "sha256"}), name=name)
    return SourceBinding(
        _require_relative_path(data["path"], f"{name} path"),
        _require_sha256(data["sha256"], f"{name} SHA-256"),
    )


def _typed_case_source_from_mapping(value: object) -> TypedCaseSource:
    data = _exact_mapping(
        value,
        fields=frozenset({"source", "case_id", "case_digest", "schema_version"}),
        name="typed case source",
    )
    return TypedCaseSource(
        source=_source_binding_from_mapping(data["source"], "typed case source"),
        case_id=_require_text(data["case_id"], "typed case ID"),
        case_digest=_require_text(data["case_digest"], "typed case digest"),
        schema_version=_require_text(
            data["schema_version"], "typed case schema version"
        ),
    )


def _contract_fingerprint_from_mapping(value: object) -> ContractFingerprint:
    data = _exact_mapping(
        value,
        fields=frozenset(
            {
                "contract_digest",
                "registry_digest",
                "canonical_schema_version",
                "schema_version",
            }
        ),
        name="contract fingerprint",
    )
    return ContractFingerprint(
        contract_digest=_require_text(
            data["contract_digest"], "contract fingerprint contract digest"
        ),
        registry_digest=_require_text(
            data["registry_digest"], "contract fingerprint registry digest"
        ),
        canonical_schema_version=_require_text(
            data["canonical_schema_version"],
            "contract fingerprint canonical schema",
        ),
        schema_version=_require_text(
            data["schema_version"], "contract fingerprint schema"
        ),
    )


def _target_fingerprint_from_mapping(value: object) -> TargetFingerprint | None:
    if value is None:
        return None
    data = _exact_mapping(
        value,
        fields=frozenset(
            {
                "universe_digest",
                "taxonomy_digest",
                "template_digest",
                "canonical_schema_version",
                "schema_version",
            }
        ),
        name="target fingerprint",
    )
    return TargetFingerprint(
        universe_digest=_require_text(
            data["universe_digest"], "target universe digest"
        ),
        taxonomy_digest=_require_text(
            data["taxonomy_digest"], "target taxonomy digest"
        ),
        template_digest=_require_text(
            data["template_digest"], "target template digest"
        ),
        canonical_schema_version=_require_text(
            data["canonical_schema_version"], "target canonical schema"
        ),
        schema_version=_require_text(data["schema_version"], "target schema"),
    )


def _resource_tokens_from_mapping(value: object) -> ResourceTokens:
    data = _exact_mapping(
        value,
        fields=frozenset(
            {
                "cpu_tokens",
                "rss_bytes",
                "io_class",
                "backend_internal_threads",
                "exclusive_state",
                "schema_version",
            }
        ),
        name="resource tokens",
    )
    return ResourceTokens(
        cpu_tokens=_nonnegative_int(data["cpu_tokens"], "CPU tokens"),
        rss_bytes=_nonnegative_int(data["rss_bytes"], "RSS bytes"),
        io_class=_require_text(data["io_class"], "resource I/O class"),
        backend_internal_threads=_nonnegative_int(
            data["backend_internal_threads"], "backend internal threads"
        ),
        exclusive_state=_optional_text(
            data["exclusive_state"], "resource exclusive state"
        ),
        schema_version=_require_text(
            data["schema_version"], "resource token schema"
        ),
    )


def _task_identity_from_mapping(value: object) -> TaskIdentity:
    data = _exact_mapping(
        value,
        fields=frozenset(
            {
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
            }
        ),
        name="task identity",
    )
    return TaskIdentity(
        protocol_digest=_require_text(
            data["protocol_digest"], "task protocol digest"
        ),
        task_kind=_enum_value(TaskKind, data["task_kind"], "task kind"),
        epoch_index=_nonnegative_int(data["epoch_index"], "task epoch index"),
        decision_index=_nonnegative_int(
            data["decision_index"], "task decision index"
        ),
        seed_lineage_digest=_require_text(
            data["seed_lineage_digest"], "task seed lineage digest"
        ),
        contrast_set_id=_optional_text(
            data["contrast_set_id"], "task contrast-set ID"
        ),
        endpoint_id=_optional_text(data["endpoint_id"], "task endpoint ID"),
        backend=_optional_text(data["backend"], "task backend"),
        attempt=_nonnegative_int(data["attempt"], "task attempt"),
        schema_version=_require_text(data["schema_version"], "task schema"),
    )


def _task_spec_from_mapping(value: object) -> TaskSpec:
    data = _exact_mapping(
        value,
        fields=frozenset(
            {
                "identity",
                "dependency_task_ids",
                "resources",
                "payload_digest",
                "schema_version",
            }
        ),
        name="task specification",
    )
    return TaskSpec(
        identity=_task_identity_from_mapping(data["identity"]),
        dependency_task_ids=_text_tuple_from_list(
            data["dependency_task_ids"],
            "task dependency IDs",
            allow_empty=True,
            sorted_values=True,
        ),
        resources=_resource_tokens_from_mapping(data["resources"]),
        payload_digest=_require_text(data["payload_digest"], "task payload digest"),
        schema_version=_require_text(data["schema_version"], "TaskSpec schema"),
    )


def _result_group_from_mapping(value: object) -> ResultGroup:
    data = _exact_mapping(
        value,
        fields=frozenset(
            {
                "result_group_id",
                "task_ids",
                "endpoint_ids",
                "contract_fingerprint",
                "target_fingerprint",
                "evidence_tier",
                "result_order_key",
                "schema_version",
            }
        ),
        name="result group",
    )
    return ResultGroup(
        result_group_id=_require_text(
            data["result_group_id"], "result group ID"
        ),
        task_ids=_text_tuple_from_list(
            data["task_ids"], "result group task IDs", sorted_values=True
        ),
        endpoint_ids=_text_tuple_from_list(
            data["endpoint_ids"], "result group endpoint IDs", sorted_values=True
        ),
        contract_fingerprint=_contract_fingerprint_from_mapping(
            data["contract_fingerprint"]
        ),
        target_fingerprint=_target_fingerprint_from_mapping(
            data["target_fingerprint"]
        ),
        evidence_tier=_enum_value(
            EvidenceTier, data["evidence_tier"], "result evidence tier"
        ),
        result_order_key=_text_tuple_from_list(
            data["result_order_key"], "result order key"
        ),
        schema_version=_require_text(data["schema_version"], "result group schema"),
    )


def _execution_outcome_from_mapping(value: object) -> StructuredExecutionOutcome:
    data = _exact_mapping(
        value,
        fields=frozenset(
            {
                "endpoint_id",
                "status",
                "failure_kind",
                "reason",
                "unsupported_evidence_digest",
                "schema_version",
            }
        ),
        name="structured execution outcome",
    )
    return StructuredExecutionOutcome(
        endpoint_id=_require_text(data["endpoint_id"], "outcome endpoint ID"),
        status=_enum_value(ExecutionStatus, data["status"], "outcome status"),
        failure_kind=_enum_value(
            FailureKind, data["failure_kind"], "outcome failure kind"
        ),
        reason=_optional_text(data["reason"], "outcome reason"),
        unsupported_evidence_digest=_optional_text(
            data["unsupported_evidence_digest"],
            "outcome unsupported evidence digest",
        ),
        schema_version=_require_text(data["schema_version"], "outcome schema"),
    )


def _evidence_envelope_from_mapping(value: object) -> EvidenceEnvelope:
    data = _exact_mapping(
        value,
        fields=frozenset(
            {
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
            }
        ),
        name="evidence envelope",
    )
    if not isinstance(data["execution_outcomes"], list):
        raise CorpusDecodeError("execution outcomes must be a list")
    if data["metadata"] != []:
        raise CorpusDecodeError("corpus execution metadata must be empty")
    return EvidenceEnvelope(
        evidence_id=_require_text(data["evidence_id"], "evidence ID"),
        state=_enum_value(EvidenceState, data["state"], "evidence state"),
        result_group_digest=_require_text(
            data["result_group_digest"], "evidence result-group digest"
        ),
        task_id=_require_text(data["task_id"], "evidence task ID"),
        seed_lineage_digest=_require_text(
            data["seed_lineage_digest"], "evidence seed lineage digest"
        ),
        contract_fingerprint=_contract_fingerprint_from_mapping(
            data["contract_fingerprint"]
        ),
        target_fingerprint=_target_fingerprint_from_mapping(
            data["target_fingerprint"]
        ),
        derivation_certificate_digest=_require_text(
            data["derivation_certificate_digest"],
            "derivation certificate digest",
        ),
        applicability_certificate_digest=_require_text(
            data["applicability_certificate_digest"],
            "applicability certificate digest",
        ),
        activation_certificate_digest=_require_text(
            data["activation_certificate_digest"],
            "activation certificate digest",
        ),
        observation_certificate_digest=_require_text(
            data["observation_certificate_digest"],
            "observation certificate digest",
        ),
        execution_outcomes=tuple(
            _execution_outcome_from_mapping(item)
            for item in data["execution_outcomes"]
        ),
        verdict_kind=_enum_value(
            VerdictKind, data["verdict_kind"], "evidence verdict"
        ),
        artifact_refs=_text_tuple_from_list(
            data["artifact_refs"],
            "evidence artifact references",
            allow_empty=True,
            sorted_values=True,
        ),
        metadata=(),
        schema_version=_require_text(data["schema_version"], "evidence schema"),
    )


def _exact_execution_record_from_mapping(value: object) -> ExactExecutionRecord:
    data = _exact_mapping(
        value,
        fields=frozenset(
            {"task_spec", "result_group", "evidence", "schema_version"}
        ),
        name="exact execution record",
    )
    return ExactExecutionRecord(
        task_spec=_task_spec_from_mapping(data["task_spec"]),
        result_group=_result_group_from_mapping(data["result_group"]),
        evidence=_evidence_envelope_from_mapping(data["evidence"]),
        schema_version=_require_text(
            data["schema_version"], "exact execution record schema"
        ),
    )


def _confirmed_root_expectation_from_mapping(
    value: object,
) -> ConfirmedRootExpectation:
    data = _exact_mapping(
        value,
        fields=frozenset(
            {
                "root_id",
                "expected_verdict",
                "expected_signature_digest",
                "schema_version",
            }
        ),
        name="confirmed root expectation",
    )
    return ConfirmedRootExpectation(
        root_id=_require_text(data["root_id"], "confirmed root ID"),
        expected_verdict=_enum_value(
            ExpectedVerdict,
            data["expected_verdict"],
            "confirmed root expected verdict",
        ),
        expected_signature_digest=_require_text(
            data["expected_signature_digest"],
            "confirmed root expected-signature digest",
        ),
        schema_version=_require_text(
            data["schema_version"], "confirmed root expectation schema"
        ),
    )


def _false_positive_adjudication_from_mapping(
    value: object,
) -> FalsePositiveAdjudication:
    data = _exact_mapping(
        value,
        fields=frozenset(
            {
                "finding_id",
                "component_id",
                "case",
                "adjudication_source",
                "observed_verdict",
                "adjudicated_verdict",
                "component_rule",
                "component_diff",
                "globally_ignored_axes",
                "globally_ignored_families",
                "schema_version",
            }
        ),
        name="false-positive adjudication",
    )
    return FalsePositiveAdjudication(
        finding_id=_require_text(data["finding_id"], "false-positive finding ID"),
        component_id=_require_text(
            data["component_id"], "false-positive component ID"
        ),
        case=_typed_case_source_from_mapping(data["case"]),
        adjudication_source=_source_binding_from_mapping(
            data["adjudication_source"], "false-positive adjudication source"
        ),
        observed_verdict=_enum_value(
            ExpectedVerdict,
            data["observed_verdict"],
            "false-positive observed verdict",
        ),
        adjudicated_verdict=_enum_value(
            ExpectedVerdict,
            data["adjudicated_verdict"],
            "false-positive adjudicated verdict",
        ),
        component_rule=_require_text(
            data["component_rule"], "false-positive component rule"
        ),
        component_diff=_require_text(
            data["component_diff"], "false-positive component diff"
        ),
        globally_ignored_axes=_text_tuple_from_list(
            data["globally_ignored_axes"],
            "globally ignored axes",
            allow_empty=True,
            sorted_values=True,
        ),
        globally_ignored_families=_text_tuple_from_list(
            data["globally_ignored_families"],
            "globally ignored families",
            allow_empty=True,
            sorted_values=True,
        ),
        schema_version=_require_text(
            data["schema_version"], "false-positive adjudication schema"
        ),
    )


def _compatibility_pair_from_mapping(value: object) -> CompatibilityVerdictPair:
    data = _exact_mapping(
        value,
        fields=frozenset(
            {
                "pair_id",
                "case",
                "endpoint_ids",
                "contract_fingerprint",
                "v1",
                "v2",
                "schema_version",
            }
        ),
        name="compatibility verdict pair",
    )
    return CompatibilityVerdictPair(
        pair_id=_require_text(data["pair_id"], "compatibility pair ID"),
        case=_typed_case_source_from_mapping(data["case"]),
        endpoint_ids=_text_tuple_from_list(
            data["endpoint_ids"],
            "compatibility endpoints",
            sorted_values=True,
        ),
        contract_fingerprint=_contract_fingerprint_from_mapping(
            data["contract_fingerprint"]
        ),
        v1=_exact_execution_record_from_mapping(data["v1"]),
        v2=_exact_execution_record_from_mapping(data["v2"]),
        schema_version=_require_text(
            data["schema_version"], "compatibility pair schema"
        ),
    )


def _mutant_kill_witness_from_mapping(value: object) -> MutantKillWitness:
    data = _exact_mapping(
        value,
        fields=frozenset(
            {
                "mutant_id",
                "mutant_version",
                "mutant_source",
                "case",
                "endpoint_ids",
                "contract_fingerprint",
                "kill_semantics",
                "baseline",
                "mutant",
                "schema_version",
            }
        ),
        name="mutant kill witness",
    )
    return MutantKillWitness(
        mutant_id=_require_text(data["mutant_id"], "mutant witness ID"),
        mutant_version=_require_text(
            data["mutant_version"], "mutant witness version"
        ),
        mutant_source=_source_binding_from_mapping(
            data["mutant_source"], "mutant source"
        ),
        case=_typed_case_source_from_mapping(data["case"]),
        endpoint_ids=_text_tuple_from_list(
            data["endpoint_ids"], "mutant witness endpoints", sorted_values=True
        ),
        contract_fingerprint=_contract_fingerprint_from_mapping(
            data["contract_fingerprint"]
        ),
        kill_semantics=_enum_value(
            KillSemantics, data["kill_semantics"], "mutant kill semantics"
        ),
        baseline=_exact_execution_record_from_mapping(data["baseline"]),
        mutant=_exact_execution_record_from_mapping(data["mutant"]),
        schema_version=_require_text(
            data["schema_version"], "mutant kill witness schema"
        ),
    )


def _entry_evidence_from_mapping(
    corpus_kind: CorpusKind, value: object
) -> CorpusEntryEvidence:
    try:
        return {
            CorpusKind.CONFIRMED_ROOTS: _confirmed_root_expectation_from_mapping,
            CorpusKind.FALSE_POSITIVES: _false_positive_adjudication_from_mapping,
            CorpusKind.V1_V2_COMPATIBILITY: _compatibility_pair_from_mapping,
            CorpusKind.MUTANT_KILL_WITNESSES: _mutant_kill_witness_from_mapping,
        }[corpus_kind](value)
    except CorpusDecodeError:
        raise
    except (TypeError, ValueError) as exc:
        raise CorpusDecodeError(
            f"{corpus_kind.value} evidence is invalid: {exc}"
        ) from exc


def corpus_entry_from_mapping(value: object) -> CorpusCandidateEntry:
    data = _exact_mapping(
        value,
        fields=frozenset(
            {
                "corpus_kind",
                "entry_id",
                "source",
                "evidence",
                "schema_version",
            }
        ),
        name="corpus candidate entry",
    )
    try:
        corpus_kind = CorpusKind(data["corpus_kind"])
    except (TypeError, ValueError) as exc:
        raise CorpusDecodeError(f"corpus candidate entry enum is invalid: {exc}") from exc
    return CorpusCandidateEntry(
        corpus_kind=corpus_kind,
        entry_id=_require_text(data["entry_id"], "corpus entry ID"),
        source=_source_binding_from_mapping(data["source"], "entry source"),
        evidence=_entry_evidence_from_mapping(corpus_kind, data["evidence"]),
        schema_version=_require_text(data["schema_version"], "entry schema version"),
    )


def corpus_manifest_from_mapping(value: object) -> CorpusManifestCandidate:
    data = _exact_mapping(
        value,
        fields=frozenset(
            {"manifest_id", "corpus_kind", "submitter_id", "entries", "schema_version"}
        ),
        name="corpus candidate manifest",
    )
    if not isinstance(data["entries"], list):
        raise CorpusDecodeError("candidate manifest entries must be a list")
    try:
        kind = CorpusKind(data["corpus_kind"])
    except (TypeError, ValueError) as exc:
        raise CorpusDecodeError(f"candidate manifest kind is invalid: {exc}") from exc
    return CorpusManifestCandidate(
        manifest_id=_require_text(data["manifest_id"], "candidate manifest ID"),
        corpus_kind=kind,
        submitter_id=_require_text(data["submitter_id"], "candidate submitter"),
        entries=tuple(corpus_entry_from_mapping(item) for item in data["entries"]),
        schema_version=_require_text(data["schema_version"], "manifest schema version"),
    )


def approval_record_from_mapping(value: object) -> IndependentApprovalRecord:
    data = _exact_mapping(
        value,
        fields=frozenset(
            {
                "approval_id",
                "manifest_digest",
                "manifest_source",
                "submitter_id",
                "approver_id",
                "reviewed_entry_ids",
                "decision",
                "schema_version",
            }
        ),
        name="independent approval record",
    )
    if not isinstance(data["reviewed_entry_ids"], list):
        raise CorpusDecodeError("reviewed entry identities must be a list")
    return IndependentApprovalRecord(
        approval_id=_require_text(data["approval_id"], "approval ID"),
        manifest_digest=_require_text(data["manifest_digest"], "manifest digest"),
        manifest_source=_source_binding_from_mapping(
            data["manifest_source"], "approved manifest source"
        ),
        submitter_id=_require_text(data["submitter_id"], "approval submitter"),
        approver_id=_require_text(data["approver_id"], "approval reviewer"),
        reviewed_entry_ids=tuple(data["reviewed_entry_ids"]),
        decision=_require_text(data["decision"], "approval decision"),
        schema_version=_require_text(data["schema_version"], "approval schema version"),
    )


def decode_candidate_manifest(payload: str) -> CorpusManifestCandidate:
    decoded = _strict_json_loads(payload)
    if canonical_json(decoded) != payload:
        raise CorpusDecodeError("candidate manifest is not canonical JSON")
    return corpus_manifest_from_mapping(decoded)


def decode_approval_record(payload: str) -> IndependentApprovalRecord:
    decoded = _strict_json_loads(payload)
    if canonical_json(decoded) != payload:
        raise CorpusDecodeError("approval record is not canonical JSON")
    return approval_record_from_mapping(decoded)


def _decode_entry_evidence_document(
    corpus_kind: CorpusKind, payload: str
) -> CorpusEntryEvidence:
    decoded = _strict_json_loads(payload)
    if canonical_json(decoded) != payload:
        raise CorpusDecodeError("entry evidence document is not canonical JSON")
    return _entry_evidence_from_mapping(corpus_kind, decoded)


def _validate_specialized_entry_source(
    entry: CorpusCandidateEntry, repo_root: Path
) -> None:
    if entry.corpus_kind is CorpusKind.CONFIRMED_ROOTS:
        return
    raw = _read_source(repo_root, entry.source)
    try:
        reconstructed = _decode_entry_evidence_document(
            entry.corpus_kind, raw.decode("utf-8")
        )
    except (CorpusDecodeError, UnicodeDecodeError, TypeError, ValueError) as exc:
        raise CorpusSourceError(
            f"specialized evidence source is invalid: {type(exc).__name__}:{exc}"
        ) from exc
    if reconstructed != entry.evidence:
        raise CorpusSourceError("specialized evidence source payload mismatch")


def _entry_case(entry: CorpusCandidateEntry) -> TypedCaseSource | None:
    evidence = entry.evidence
    if isinstance(
        evidence,
        (FalsePositiveAdjudication, CompatibilityVerdictPair, MutantKillWitness),
    ):
        return evidence.case
    return None


def validate_candidate_manifest(
    manifest: CorpusManifestCandidate, repo_root: Path
) -> CorpusValidation:
    if not isinstance(manifest, CorpusManifestCandidate):
        raise TypeError("candidate validation requires CorpusManifestCandidate")
    errors: list[str] = []
    for entry in manifest.entries:
        try:
            _read_source(repo_root, entry.source)
        except CorpusSourceError as exc:
            errors.append(f"entry_source_invalid:{entry.entry_id}:{exc}")
            continue
        try:
            _validate_specialized_entry_source(entry, repo_root)
            case = _entry_case(entry)
            if case is not None:
                _validate_typed_case_source(repo_root, case)
            if isinstance(entry.evidence, FalsePositiveAdjudication):
                adjudication_raw = _read_source(
                    repo_root, entry.evidence.adjudication_source
                )
                expected_raw = canonical_json(
                    entry.evidence.adjudication_facts
                ).encode("utf-8")
                if adjudication_raw != expected_raw:
                    raise CorpusSourceError(
                        "false-positive adjudication facts do not match their source"
                    )
        except CorpusSourceError as exc:
            errors.append(f"entry_evidence_invalid:{entry.entry_id}:{exc}")

    if manifest.corpus_kind is CorpusKind.CONFIRMED_ROOTS:
        try:
            inventory = derive_confirmed_root_inventory(repo_root)
        except (CorpusSourceError, ValueError) as exc:
            errors.append(f"confirmed_root_inventory_invalid:{exc}")
        else:
            expected = set(inventory.confirmed_ids)
            observed = set(manifest.entry_ids)
            pending = observed & set(inventory.excluded_pending_ids)
            if pending:
                errors.append("pending_root_forbidden:" + ",".join(sorted(pending)))
            missing = expected - observed
            unknown = observed - expected
            if missing:
                errors.append("confirmed_roots_missing:" + ",".join(sorted(missing)))
            if unknown:
                errors.append("confirmed_roots_unknown:" + ",".join(sorted(unknown)))
            source_by_id = {item.root_id: item.case_source for item in inventory.confirmed}
            root_by_id = {item.root_id: item for item in inventory.confirmed}
            for entry in manifest.entries:
                expected_source = source_by_id.get(entry.entry_id)
                if expected_source is not None and entry.source != expected_source:
                    errors.append(f"confirmed_root_source_mismatch:{entry.entry_id}")
                root = root_by_id.get(entry.entry_id)
                if root is not None:
                    expected_evidence = ConfirmedRootExpectation(
                        root_id=root.root_id,
                        expected_verdict=ExpectedVerdict.VIOLATED,
                        expected_signature_digest=root.expected_signature_digest,
                    )
                    if entry.evidence != expected_evidence:
                        errors.append(
                            f"confirmed_root_expectation_mismatch:{entry.entry_id}"
                        )

    elif manifest.corpus_kind is CorpusKind.FALSE_POSITIVES:
        if any(
            not isinstance(entry.evidence, FalsePositiveAdjudication)
            for entry in manifest.entries
        ):
            errors.append("false_positive_typed_adjudication_required")

    elif manifest.corpus_kind is CorpusKind.V1_V2_COMPATIBILITY:
        if any(
            not isinstance(entry.evidence, CompatibilityVerdictPair)
            for entry in manifest.entries
        ):
            errors.append("compatibility_typed_pair_required")

    elif manifest.corpus_kind is CorpusKind.MUTANT_KILL_WITNESSES:
        try:
            inventory = derive_mutant_inventory(repo_root)
        except CorpusSourceError as exc:
            errors.append(f"mutant_inventory_invalid:{exc}")
        else:
            expected = set(inventory.all_ids)
            observed = set(manifest.entry_ids)
            missing = expected - observed
            unknown = observed - expected
            if missing:
                errors.append("mutant_witnesses_missing:" + ",".join(sorted(missing)))
            if unknown:
                errors.append("mutant_witnesses_unknown:" + ",".join(sorted(unknown)))
            for entry in manifest.entries:
                witness = entry.evidence
                assert isinstance(witness, MutantKillWitness)
                expected_semantics = (
                    expected_kill_semantics(entry.entry_id)
                    if entry.entry_id in expected
                    else None
                )
                if (
                    expected_semantics is not None
                    and witness.kill_semantics is not expected_semantics
                ):
                    errors.append(f"mutant_kill_semantics_mismatch:{entry.entry_id}")
                if witness.mutant_source != inventory.source:
                    errors.append(f"mutant_source_mismatch:{entry.entry_id}")
            pairs = tuple(
                entry.evidence.pair_identity
                for entry in manifest.entries
                if isinstance(entry.evidence, MutantKillWitness)
            )
            baseline_tasks = tuple(item[0] for item in pairs)
            mutant_tasks = tuple(item[1] for item in pairs)
            if len(pairs) != len(set(pairs)):
                errors.append("mutant_execution_pairs_duplicated")
            if len(baseline_tasks) != len(set(baseline_tasks)):
                errors.append("mutant_baseline_tasks_reused")
            if len(mutant_tasks) != len(set(mutant_tasks)):
                errors.append("mutant_tasks_reused")

    state = (
        CorpusValidationState.REJECTED
        if errors
        else CorpusValidationState.CANDIDATE_MATERIAL
    )
    return CorpusValidation(
        corpus_kind=manifest.corpus_kind,
        state=state,
        entry_ids=manifest.entry_ids,
        errors=tuple(dict.fromkeys(errors)),
        manifest_digest=manifest.digest,
    )


def validate_independent_approval(
    manifest: CorpusManifestCandidate,
    approval: IndependentApprovalRecord,
    approval_source: SourceBinding,
    repo_root: Path,
) -> CorpusValidation:
    """Validate structural review bindings without trusting reviewer strings."""

    candidate = validate_candidate_manifest(manifest, repo_root)
    errors = list(candidate.errors)
    if not isinstance(approval, IndependentApprovalRecord):
        raise TypeError("approval validation requires IndependentApprovalRecord")
    if not isinstance(approval_source, SourceBinding):
        raise TypeError("approval validation requires an approval SourceBinding")
    if approval.manifest_digest != manifest.digest:
        errors.append("approval_manifest_digest_mismatch")
    if approval.submitter_id != manifest.submitter_id:
        errors.append("approval_submitter_mismatch")
    if approval.approver_id == manifest.submitter_id:
        errors.append("self_approval_forbidden")
    if approval.reviewed_entry_ids != manifest.entry_ids:
        errors.append("approval_reviewed_entry_set_mismatch")
    try:
        manifest_raw = _read_source(repo_root, approval.manifest_source)
        decoded_manifest = decode_candidate_manifest(manifest_raw.decode("utf-8"))
        if decoded_manifest != manifest:
            errors.append("approval_manifest_payload_mismatch")
    except (UnicodeDecodeError, CorpusSourceError, CorpusDecodeError, ValueError) as exc:
        errors.append(f"approval_manifest_source_invalid:{exc}")
    try:
        approval_raw = _read_source(repo_root, approval_source)
        decoded_approval = decode_approval_record(approval_raw.decode("utf-8"))
        if decoded_approval != approval:
            errors.append("approval_record_payload_mismatch")
    except (UnicodeDecodeError, CorpusSourceError, CorpusDecodeError, ValueError) as exc:
        errors.append(f"approval_record_source_invalid:{exc}")

    state = (
        CorpusValidationState.REJECTED
        if errors
        else CorpusValidationState.STRUCTURALLY_REVIEWED_CONTEXT_PENDING
    )
    return CorpusValidation(
        corpus_kind=manifest.corpus_kind,
        state=state,
        entry_ids=manifest.entry_ids,
        errors=tuple(dict.fromkeys(errors)),
        manifest_digest=manifest.digest,
        approval_record_sha256=(
            approval_source.sha256
            if state is CorpusValidationState.STRUCTURALLY_REVIEWED_CONTEXT_PENDING
            else ""
        ),
    )


def missing_corpus_input(
    corpus_kind: CorpusKind, *, reason: str = "manifest_not_provided"
) -> CorpusValidation:
    if not isinstance(corpus_kind, CorpusKind):
        raise TypeError("missing corpus input requires a CorpusKind")
    _require_text(reason, "missing corpus reason")
    return CorpusValidation(
        corpus_kind=corpus_kind,
        state=CorpusValidationState.MISSING_INPUT,
        entry_ids=(),
        errors=(f"missing_{corpus_kind.value}:{reason}",),
    )


def absent_required_phase6_corpora() -> tuple[CorpusValidation, ...]:
    """Return the three reviewed missing inputs without manufacturing IDs."""

    return (
        missing_corpus_input(CorpusKind.FALSE_POSITIVES),
        missing_corpus_input(CorpusKind.V1_V2_COMPATIBILITY),
        missing_corpus_input(CorpusKind.MUTANT_KILL_WITNESSES),
    )


__all__ = [
    "CONFIRMED_ROOT_MANIFEST_PATH",
    "CONFIRMED_ROOT_MANIFEST_SHA256",
    "CORPUS_APPROVAL_SCHEMA_VERSION",
    "CORPUS_ENTRY_SCHEMA_VERSION",
    "CORPUS_MANIFEST_SCHEMA_VERSION",
    "MUTATION_SOURCE_PATH",
    "MUTATION_SOURCE_SHA256",
    "REQUIRED_FALSE_POSITIVE_MINIMUM",
    "CompatibilityVerdictPair",
    "ConfirmedRootInventory",
    "ConfirmedRootExpectation",
    "CorpusCandidateEntry",
    "CorpusDecodeError",
    "CorpusKind",
    "CorpusManifestCandidate",
    "CorpusSourceError",
    "CorpusValidation",
    "CorpusValidationState",
    "ExpectedVerdict",
    "ExactExecutionRecord",
    "FalsePositiveAdjudication",
    "IndependentApprovalRecord",
    "KillSemantics",
    "MutantKillWitness",
    "MutantInventory",
    "RootSourceEntry",
    "SourceBinding",
    "TypedCaseSource",
    "absent_required_phase6_corpora",
    "approval_record_from_mapping",
    "corpus_entry_from_mapping",
    "corpus_manifest_from_mapping",
    "decode_approval_record",
    "decode_candidate_manifest",
    "derive_confirmed_root_inventory",
    "derive_mutant_inventory",
    "derive_typed_case_source",
    "expected_kill_semantics",
    "compatibility_task_payload_digest",
    "mutant_task_payload_digest",
    "missing_corpus_input",
    "validate_candidate_manifest",
    "validate_independent_approval",
]
