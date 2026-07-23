"""Private, fail-closed producer provenance foundations for Phase 6.

The receipt built here is not authority.  Verification only establishes that
raw immutable records reproduce a receipt and match an independently supplied,
record-by-record policy.  The module is deliberately absent from Runtime's
public exports and has no connection to a launcher or semantic replay.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import hashlib
import math
from pathlib import PurePosixPath
import types
from typing import Any, Callable, Union, get_args, get_origin, get_type_hints

from datadiff_osc._canonical import (
    canonical_envelope,
    decode_canonical_envelope,
    stable_digest,
    to_primitive,
)
from datadiff_osc.runtime._private_receipts import (
    CANDIDATE_CLASSIFICATION_RECEIPT_SCHEMA_VERSION,
    PAIRED_BENCHMARK_RECEIPT_SCHEMA_VERSION,
    REPOSITORY_TEST_RECEIPT_SCHEMA_VERSION,
    TARGET_VERSION_RECEIPT_SCHEMA_VERSION,
    V3_RUN_RECEIPT_SCHEMA_VERSION,
    CandidateClassificationReceipt,
    PairedBenchmarkReceipt,
    RepositoryTestReceipt,
    TargetVersionReceipt,
    V3RunReceipt,
)
from datadiff_osc.runtime._receipt_producers import (
    RepositoryTestObservation,
    build_repository_test_receipt,
)
from datadiff_osc.schemas import (
    EVIDENCE_SCHEMA_VERSION,
    SEED_LINEAGE_SCHEMA_VERSION,
    TASK_SCHEMA_VERSION,
    EvidenceEnvelope,
    ResultGroup,
    SeedLineage,
    StructuredExecutionOutcome,
    TaskSpec,
)


PRODUCER_PROVENANCE_SCHEMA_VERSION = "osc-root-producer-provenance-v3"
REPOSITORY_TEST_RAW_CONTEXT_SCHEMA_VERSION = (
    "osc-root-repository-test-raw-context-v1"
)
REPOSITORY_TEST_JUNIT_SCHEMA_VERSION = "osc-root-repository-junit-xml-v1"
REPOSITORY_TEST_LOG_SCHEMA_VERSION = "osc-root-repository-pytest-log-v1"


class ProducerOriginKind(str, Enum):
    """Typed origin classes; free-form metadata is never an origin."""

    REAL_EXECUTION = "REAL_EXECUTION"
    ROOT_FROZEN_PUBLIC_SOURCE = "ROOT_FROZEN_PUBLIC_SOURCE"
    ROOT_FROZEN_CONTROL_CORPUS = "ROOT_FROZEN_CONTROL_CORPUS"
    DIAGNOSTIC = "DIAGNOSTIC"


class ProducerRecordCategory(str, Enum):
    """The exact role a record has in one producer invocation."""

    IMPLEMENTATION = "IMPLEMENTATION"
    INPUT = "INPUT"
    RAW_OUTPUT = "RAW_OUTPUT"
    TYPED_OUTPUT = "TYPED_OUTPUT"


_FROZEN_ORIGINS = frozenset(
    {
        ProducerOriginKind.ROOT_FROZEN_PUBLIC_SOURCE,
        ProducerOriginKind.ROOT_FROZEN_CONTROL_CORPUS,
    }
)


def _require_text(name: str, value: object, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        qualifier = "a string" if allow_empty else "a non-empty string"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _require_sha256(name: str, value: object, *, allow_empty: bool = False) -> str:
    if allow_empty and value == "":
        return ""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _require_relative_path(name: str, value: object, *, allow_dot: bool = False) -> str:
    text = _require_text(name, value)
    if "\\" in text or "\x00" in text or any(ord(character) < 32 for character in text):
        raise ValueError(f"{name} must be a canonical POSIX path")
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must be a safe relative path")
    normalized = path.as_posix()
    if normalized != text or (normalized == "." and not allow_dot):
        raise ValueError(f"{name} must be a canonical relative path")
    return text


def _require_unique_sorted(name: str, values: tuple[str, ...]) -> None:
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{name} must contain non-empty strings")
    if values != tuple(sorted(values)) or len(values) != len(set(values)):
        raise ValueError(f"{name} must be uniquely sorted")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_string_tuple(name: str, values: tuple[str, ...]) -> None:
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{name} must be a non-empty tuple")
    if any(not isinstance(item, str) or not item for item in values):
        raise ValueError(f"{name} contains an invalid string")


def _require_field_tuple(
    name: str, values: tuple[tuple[str, str], ...]
) -> None:
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{name} must be a non-empty field tuple")
    if any(
        not isinstance(item, tuple)
        or len(item) != 2
        or not isinstance(item[0], str)
        or not item[0]
        or not isinstance(item[1], str)
        for item in values
    ):
        raise ValueError(f"{name} contains an invalid field")
    keys = tuple(item[0] for item in values)
    _require_unique_sorted(f"{name} names", keys)


@dataclass(frozen=True, slots=True)
class SourceSnapshotFileBinding:
    relative_path: str
    sha256: str

    def __post_init__(self) -> None:
        _require_relative_path("source snapshot path", self.relative_path)
        _require_sha256("source snapshot SHA-256", self.sha256)


@dataclass(frozen=True, slots=True)
class SourceSnapshotView:
    """A minimal view extracted from a caller-verified source snapshot."""

    source_digest: str
    files: tuple[SourceSnapshotFileBinding, ...]

    def __post_init__(self) -> None:
        _require_text("source digest", self.source_digest)
        if not isinstance(self.files, tuple) or not self.files:
            raise ValueError("source snapshot files must be a non-empty tuple")
        if any(not isinstance(item, SourceSnapshotFileBinding) for item in self.files):
            raise ValueError("source snapshot files have an invalid type")
        _require_unique_sorted(
            "source snapshot paths", tuple(item.relative_path for item in self.files)
        )

    def file_map(self) -> dict[str, str]:
        return {item.relative_path: item.sha256 for item in self.files}


@dataclass(frozen=True, slots=True)
class RepositoryTestRawContext:
    """Canonical invocation context; JUnit/log remain separate raw parents."""

    source_digest: str
    test_plan: tuple[str, ...]
    config: tuple[tuple[str, str], ...]
    environment: tuple[tuple[str, str], ...]
    command: tuple[str, ...]
    exit_code: int
    schema_version: str = REPOSITORY_TEST_RAW_CONTEXT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_text("repository raw source digest", self.source_digest)
        _require_string_tuple("repository raw test plan", self.test_plan)
        _require_field_tuple("repository raw config", self.config)
        _require_field_tuple("repository raw environment", self.environment)
        _require_string_tuple("repository raw command", self.command)
        if (
            not isinstance(self.exit_code, int)
            or isinstance(self.exit_code, bool)
            or self.exit_code < 0
        ):
            raise ValueError("repository raw exit code must be non-negative")
        if self.schema_version != REPOSITORY_TEST_RAW_CONTEXT_SCHEMA_VERSION:
            raise ValueError("repository raw context schema version mismatch")


@dataclass(frozen=True, slots=True)
class ImmutableFileRecord:
    """Raw immutable bytes from which a file binding is derived."""

    record_id: str
    subject_kind: str
    relative_path: str
    schema_version: str
    origin: ProducerOriginKind
    content: bytes

    def __post_init__(self) -> None:
        for name in ("record_id", "subject_kind", "schema_version"):
            _require_text(name, getattr(self, name))
        _require_relative_path("file record path", self.relative_path)
        if not isinstance(self.origin, ProducerOriginKind):
            raise ValueError("file record origin must be a ProducerOriginKind")
        if not isinstance(self.content, bytes):
            raise ValueError("file record content must be immutable bytes")

    @property
    def sha256(self) -> str:
        return _sha256(self.content)


@dataclass(frozen=True, slots=True)
class RawParentRoleBinding:
    """One semantic role bound to one exact raw-output record ID."""

    role: str
    record_id: str

    def __post_init__(self) -> None:
        _require_text("raw parent role", self.role)
        _require_text("raw parent record ID", self.record_id)


@dataclass(frozen=True, slots=True)
class ImmutableTypedOutputRecord:
    """Canonical typed bytes plus their exact raw-output parents."""

    record_id: str
    subject_kind: str
    subject_ids: tuple[str, ...]
    relative_path: str
    envelope_type: str
    envelope_schema_version: str
    raw_parent_roles: tuple[RawParentRoleBinding, ...]
    origin: ProducerOriginKind
    content: bytes

    def __post_init__(self) -> None:
        for name in (
            "record_id",
            "subject_kind",
            "envelope_type",
            "envelope_schema_version",
        ):
            _require_text(name, getattr(self, name))
        if not isinstance(self.subject_ids, tuple) or not self.subject_ids:
            raise ValueError("typed output subject IDs must be a non-empty tuple")
        _require_unique_sorted("typed output subject IDs", self.subject_ids)
        if not isinstance(self.raw_parent_roles, tuple) or not self.raw_parent_roles:
            raise ValueError("typed output requires ordered raw parent roles")
        if any(
            not isinstance(item, RawParentRoleBinding)
            for item in self.raw_parent_roles
        ):
            raise ValueError("typed output raw parent roles have an invalid type")
        roles = tuple(item.role for item in self.raw_parent_roles)
        record_ids = tuple(item.record_id for item in self.raw_parent_roles)
        if len(roles) != len(set(roles)):
            raise ValueError("typed output raw parent roles must be unique")
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("typed output raw parent record IDs must be unique")
        _require_relative_path("typed output path", self.relative_path)
        if not isinstance(self.origin, ProducerOriginKind):
            raise ValueError("typed output origin must be a ProducerOriginKind")
        if not isinstance(self.content, bytes) or not self.content:
            raise ValueError("typed output content must be non-empty immutable bytes")

    @property
    def sha256(self) -> str:
        return _sha256(self.content)

    @property
    def raw_parent_record_ids(self) -> tuple[str, ...]:
        return tuple(item.record_id for item in self.raw_parent_roles)


@dataclass(frozen=True, slots=True)
class EnvironmentRecord:
    name: str
    value: str

    def __post_init__(self) -> None:
        _require_text("environment name", self.name)
        _require_text("environment value", self.value, allow_empty=True)
        if "=" in self.name or "\x00" in self.name or "\x00" in self.value:
            raise ValueError("environment record is not canonical")


@dataclass(frozen=True, slots=True)
class ProducerFailureRecord:
    failure_id: str
    stage: str
    failure_kind: str
    detail: str

    def __post_init__(self) -> None:
        for name in ("failure_id", "stage", "failure_kind"):
            _require_text(name, getattr(self, name))
        _require_text("failure detail", self.detail, allow_empty=True)


@dataclass(frozen=True, slots=True)
class FileDigestBinding:
    category: ProducerRecordCategory
    record_id: str
    subject_kind: str
    relative_path: str
    schema_version: str
    origin: ProducerOriginKind
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.category, ProducerRecordCategory) or self.category is ProducerRecordCategory.TYPED_OUTPUT:
            raise ValueError("file binding category is invalid")
        for name in ("record_id", "subject_kind", "schema_version"):
            _require_text(name, getattr(self, name))
        _require_relative_path("file binding path", self.relative_path)
        if not isinstance(self.origin, ProducerOriginKind):
            raise ValueError("file binding origin must be a ProducerOriginKind")
        _require_sha256("file binding SHA-256", self.sha256)

    @property
    def digest(self) -> str:
        return stable_digest("osc-root-producer-file-binding", self)


@dataclass(frozen=True, slots=True)
class TypedOutputBinding:
    category: ProducerRecordCategory
    record_id: str
    subject_kind: str
    subject_ids: tuple[str, ...]
    relative_path: str
    envelope_type: str
    envelope_schema_version: str
    raw_parent_roles: tuple[RawParentRoleBinding, ...]
    origin: ProducerOriginKind
    sha256: str

    def __post_init__(self) -> None:
        if self.category is not ProducerRecordCategory.TYPED_OUTPUT:
            raise ValueError("typed binding category must be TYPED_OUTPUT")
        for name in (
            "record_id",
            "subject_kind",
            "envelope_type",
            "envelope_schema_version",
        ):
            _require_text(name, getattr(self, name))
        if not isinstance(self.subject_ids, tuple) or not self.subject_ids:
            raise ValueError("typed binding subject IDs must be a non-empty tuple")
        _require_unique_sorted("typed binding subject IDs", self.subject_ids)
        if not isinstance(self.raw_parent_roles, tuple) or not self.raw_parent_roles:
            raise ValueError("typed binding requires ordered raw parent roles")
        if any(
            not isinstance(item, RawParentRoleBinding)
            for item in self.raw_parent_roles
        ):
            raise ValueError("typed binding raw parent roles have an invalid type")
        roles = tuple(item.role for item in self.raw_parent_roles)
        record_ids = tuple(item.record_id for item in self.raw_parent_roles)
        if len(roles) != len(set(roles)) or len(record_ids) != len(set(record_ids)):
            raise ValueError("typed binding raw parent roles/IDs must be unique")
        _require_relative_path("typed binding path", self.relative_path)
        if not isinstance(self.origin, ProducerOriginKind):
            raise ValueError("typed binding origin must be a ProducerOriginKind")
        _require_sha256("typed binding SHA-256", self.sha256)

    @property
    def digest(self) -> str:
        return stable_digest("osc-root-producer-typed-output-binding", self)

    @property
    def raw_parent_record_ids(self) -> tuple[str, ...]:
        return tuple(item.record_id for item in self.raw_parent_roles)


@dataclass(frozen=True, slots=True)
class EnvironmentBinding:
    name: str
    value_digest: str

    def __post_init__(self) -> None:
        _require_text("environment name", self.name)
        _require_text("environment value digest", self.value_digest)


@dataclass(frozen=True, slots=True)
class StructuredProducerFailure:
    failure_id: str
    stage: str
    failure_kind: str
    detail_digest: str

    def __post_init__(self) -> None:
        for name in ("failure_id", "stage", "failure_kind", "detail_digest"):
            _require_text(name, getattr(self, name))


def _file_binding(
    record: ImmutableFileRecord, category: ProducerRecordCategory
) -> FileDigestBinding:
    return FileDigestBinding(
        category=category,
        record_id=record.record_id,
        subject_kind=record.subject_kind,
        relative_path=record.relative_path,
        schema_version=record.schema_version,
        origin=record.origin,
        sha256=record.sha256,
    )


def _typed_binding(record: ImmutableTypedOutputRecord) -> TypedOutputBinding:
    return TypedOutputBinding(
        category=ProducerRecordCategory.TYPED_OUTPUT,
        record_id=record.record_id,
        subject_kind=record.subject_kind,
        subject_ids=record.subject_ids,
        relative_path=record.relative_path,
        envelope_type=record.envelope_type,
        envelope_schema_version=record.envelope_schema_version,
        raw_parent_roles=record.raw_parent_roles,
        origin=record.origin,
        sha256=record.sha256,
    )


def _environment_binding(record: EnvironmentRecord) -> EnvironmentBinding:
    return EnvironmentBinding(
        name=record.name,
        value_digest=stable_digest(
            "osc-root-producer-environment-value", (record.name, record.value)
        ),
    )


def _failure_binding(record: ProducerFailureRecord) -> StructuredProducerFailure:
    return StructuredProducerFailure(
        failure_id=record.failure_id,
        stage=record.stage,
        failure_kind=record.failure_kind,
        detail_digest=stable_digest(
            "osc-root-producer-failure-detail", (record.failure_id, record.detail)
        ),
    )


def _sorted_file_bindings(
    records: tuple[ImmutableFileRecord, ...],
    *,
    label: str,
    category: ProducerRecordCategory,
) -> tuple[FileDigestBinding, ...]:
    if not isinstance(records, tuple):
        raise ValueError(f"{label} records must be an immutable tuple")
    if any(not isinstance(item, ImmutableFileRecord) for item in records):
        raise ValueError(f"{label} records have an invalid type")
    result = tuple(
        sorted(
            (_file_binding(item, category) for item in records),
            key=lambda item: item.record_id,
        )
    )
    identifiers = tuple(item.record_id for item in result)
    if identifiers:
        _require_unique_sorted(f"{label} record IDs", identifiers)
    return result


@dataclass(frozen=True, slots=True)
class ProducerProvenanceReceipt:
    producer_kind: str
    invocation_id: str
    source_digest: str
    protocol_digest: str
    plan_digest: str
    command: tuple[str, ...]
    command_digest: str
    cwd: str
    environment: tuple[EnvironmentBinding, ...]
    environment_digest: str
    implementation_files: tuple[FileDigestBinding, ...]
    inputs: tuple[FileDigestBinding, ...]
    raw_outputs: tuple[FileDigestBinding, ...]
    typed_outputs: tuple[TypedOutputBinding, ...]
    failures: tuple[StructuredProducerFailure, ...]
    exit_code: int
    schema_version: str = PRODUCER_PROVENANCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "producer_kind",
            "invocation_id",
            "source_digest",
            "protocol_digest",
            "plan_digest",
        ):
            _require_text(name, getattr(self, name))
        if not isinstance(self.command, tuple) or not self.command:
            raise ValueError("producer command must be a non-empty tuple")
        if any(
            not isinstance(item, str) or not item or "\x00" in item
            for item in self.command
        ):
            raise ValueError("producer command contains an invalid argument")
        if self.command_digest != stable_digest(
            "osc-root-producer-command", self.command
        ):
            raise ValueError("producer command digest mismatch")
        _require_relative_path("producer cwd", self.cwd, allow_dot=True)
        if (
            not isinstance(self.exit_code, int)
            or isinstance(self.exit_code, bool)
            or self.exit_code < 0
        ):
            raise ValueError("producer exit code must be a non-negative integer")
        if self.schema_version != PRODUCER_PROVENANCE_SCHEMA_VERSION:
            raise ValueError("producer provenance schema version mismatch")
        tuple_specs = (
            ("environment", self.environment, EnvironmentBinding, lambda item: item.name),
            (
                "implementation files",
                self.implementation_files,
                FileDigestBinding,
                lambda item: item.record_id,
            ),
            ("inputs", self.inputs, FileDigestBinding, lambda item: item.record_id),
            (
                "raw outputs",
                self.raw_outputs,
                FileDigestBinding,
                lambda item: item.record_id,
            ),
            (
                "typed outputs",
                self.typed_outputs,
                TypedOutputBinding,
                lambda item: item.record_id,
            ),
            (
                "failures",
                self.failures,
                StructuredProducerFailure,
                lambda item: item.failure_id,
            ),
        )
        for label, values, expected_type, key in tuple_specs:
            if not isinstance(values, tuple) or any(
                not isinstance(item, expected_type) for item in values
            ):
                raise ValueError(f"producer {label} must be a typed tuple")
            identifiers = tuple(key(item) for item in values)
            if identifiers:
                _require_unique_sorted(f"producer {label}", identifiers)
        if not self.implementation_files:
            raise ValueError("producer provenance requires implementation files")
        if not self.raw_outputs or not self.typed_outputs:
            raise ValueError("producer provenance requires raw and typed outputs")
        expected_categories = (
            (self.implementation_files, ProducerRecordCategory.IMPLEMENTATION),
            (self.inputs, ProducerRecordCategory.INPUT),
            (self.raw_outputs, ProducerRecordCategory.RAW_OUTPUT),
            (self.typed_outputs, ProducerRecordCategory.TYPED_OUTPUT),
        )
        for values, category in expected_categories:
            if any(item.category is not category for item in values):
                raise ValueError("producer record category/list mismatch")
        all_record_ids = tuple(
            item.record_id
            for values in (
                self.implementation_files,
                self.inputs,
                self.raw_outputs,
                self.typed_outputs,
            )
            for item in values
        )
        if len(all_record_ids) != len(set(all_record_ids)):
            raise ValueError("producer record IDs must be globally unique")
        raw_ids = frozenset(item.record_id for item in self.raw_outputs)
        if any(
            not set(item.raw_parent_record_ids).issubset(raw_ids)
            for item in self.typed_outputs
        ):
            raise ValueError("typed output contains an unknown raw parent record ID")
        if self.environment_digest != stable_digest(
            "osc-root-producer-environment", self.environment
        ):
            raise ValueError("producer environment digest mismatch")

    @property
    def digest(self) -> str:
        return stable_digest("osc-root-producer-provenance", self)

    def to_dict(self) -> dict[str, object]:
        return to_primitive(self)


@dataclass(frozen=True, slots=True)
class ExpectedProvenanceRecordPolicy:
    """Independent exact policy for one and only one producer record."""

    category: ProducerRecordCategory
    record_id: str
    subject_kind: str
    subject_ids: tuple[str, ...]
    relative_path: str
    schema_version: str
    envelope_type: str
    raw_parent_roles: tuple[RawParentRoleBinding, ...]
    origin: ProducerOriginKind
    frozen_source_sha256: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.category, ProducerRecordCategory):
            raise ValueError("record policy category is invalid")
        for name in ("record_id", "subject_kind", "schema_version"):
            _require_text(name, getattr(self, name))
        _require_relative_path("record policy path", self.relative_path)
        if not isinstance(self.subject_ids, tuple):
            raise ValueError("record policy subject IDs must be a tuple")
        if self.subject_ids:
            _require_unique_sorted("record policy subject IDs", self.subject_ids)
        if not isinstance(self.raw_parent_roles, tuple):
            raise ValueError("record policy raw parent roles must be a tuple")
        if any(
            not isinstance(item, RawParentRoleBinding)
            for item in self.raw_parent_roles
        ):
            raise ValueError("record policy raw parent roles have an invalid type")
        roles = tuple(item.role for item in self.raw_parent_roles)
        parent_ids = tuple(item.record_id for item in self.raw_parent_roles)
        if len(roles) != len(set(roles)) or len(parent_ids) != len(set(parent_ids)):
            raise ValueError("record policy raw parent roles/IDs must be unique")
        if self.category is ProducerRecordCategory.TYPED_OUTPUT:
            if not self.subject_ids or not self.raw_parent_roles:
                raise ValueError("typed record policy requires subjects and raw parents")
            _require_text("record policy envelope type", self.envelope_type)
        elif self.subject_ids or self.envelope_type or self.raw_parent_roles:
            raise ValueError("file record policy cannot carry typed-only fields")
        if not isinstance(self.origin, ProducerOriginKind):
            raise ValueError("record policy requires one exact origin")
        if self.origin is ProducerOriginKind.DIAGNOSTIC:
            raise ValueError("DIAGNOSTIC cannot satisfy a provenance policy")
        if self.origin in _FROZEN_ORIGINS:
            _require_sha256("record policy frozen source SHA-256", self.frozen_source_sha256)
        else:
            _require_sha256(
                "record policy frozen source SHA-256",
                self.frozen_source_sha256,
                allow_empty=True,
            )
            if self.frozen_source_sha256:
                raise ValueError("non-frozen record policy cannot carry a frozen source SHA")

    @property
    def raw_parent_record_ids(self) -> tuple[str, ...]:
        return tuple(item.record_id for item in self.raw_parent_roles)


@dataclass(frozen=True, slots=True)
class ProducerProvenancePolicy:
    producer_kind: str
    invocation_id: str
    source_digest: str
    protocol_digest: str
    plan_digest: str
    command: tuple[str, ...]
    cwd: str
    environment: tuple[EnvironmentBinding, ...]
    expected_records: tuple[ExpectedProvenanceRecordPolicy, ...]
    allowed_exit_codes: tuple[int, ...] = (0,)

    def __post_init__(self) -> None:
        for name in (
            "producer_kind",
            "invocation_id",
            "source_digest",
            "protocol_digest",
            "plan_digest",
        ):
            _require_text(name, getattr(self, name))
        if not isinstance(self.command, tuple) or not self.command:
            raise ValueError("provenance policy command must be a non-empty tuple")
        if any(not isinstance(item, str) or not item for item in self.command):
            raise ValueError("provenance policy command contains an invalid argument")
        _require_relative_path("provenance policy cwd", self.cwd, allow_dot=True)
        if not isinstance(self.environment, tuple) or any(
            not isinstance(item, EnvironmentBinding) for item in self.environment
        ):
            raise ValueError("provenance policy environment must be typed")
        environment_names = tuple(item.name for item in self.environment)
        if environment_names:
            _require_unique_sorted("policy environment names", environment_names)
        if not isinstance(self.expected_records, tuple) or not self.expected_records:
            raise ValueError("provenance policy requires exact record policies")
        if any(
            not isinstance(item, ExpectedProvenanceRecordPolicy)
            for item in self.expected_records
        ):
            raise ValueError("provenance record policy has an invalid type")
        keys = tuple(
            (item.category.value, item.record_id) for item in self.expected_records
        )
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("provenance record policies must be uniquely sorted")
        record_ids = tuple(item.record_id for item in self.expected_records)
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("provenance policy record IDs must be globally unique")
        if (
            not isinstance(self.allowed_exit_codes, tuple)
            or not self.allowed_exit_codes
            or any(
                not isinstance(item, int) or isinstance(item, bool) or item < 0
                for item in self.allowed_exit_codes
            )
        ):
            raise ValueError("allowed exit codes must be non-negative integers")
        if self.allowed_exit_codes != tuple(sorted(set(self.allowed_exit_codes))):
            raise ValueError("allowed exit codes must be uniquely sorted")


@dataclass(frozen=True, slots=True)
class ProvenanceVerification:
    receipt_digest: str
    errors: tuple[str, ...]

    @property
    def valid(self) -> bool:
        """Structural/policy validity only; never Root authority."""

        return not self.errors


_TYPED_OUTPUT_REGISTRY: dict[str, tuple[type[Any], str]] = {
    "V3RunReceipt": (V3RunReceipt, V3_RUN_RECEIPT_SCHEMA_VERSION),
    "RepositoryTestReceipt": (
        RepositoryTestReceipt,
        REPOSITORY_TEST_RECEIPT_SCHEMA_VERSION,
    ),
    "TargetVersionReceipt": (
        TargetVersionReceipt,
        TARGET_VERSION_RECEIPT_SCHEMA_VERSION,
    ),
    "CandidateClassificationReceipt": (
        CandidateClassificationReceipt,
        CANDIDATE_CLASSIFICATION_RECEIPT_SCHEMA_VERSION,
    ),
    "PairedBenchmarkReceipt": (
        PairedBenchmarkReceipt,
        PAIRED_BENCHMARK_RECEIPT_SCHEMA_VERSION,
    ),
    "SeedLineage": (SeedLineage, SEED_LINEAGE_SCHEMA_VERSION),
    "TaskSpec": (TaskSpec, "osc-task-spec-v1"),
    "StructuredExecutionOutcome": (
        StructuredExecutionOutcome,
        "osc-structured-execution-outcome-v1",
    ),
    "ResultGroup": (ResultGroup, "osc-result-group-v1"),
    "EvidenceEnvelope": (EvidenceEnvelope, EVIDENCE_SCHEMA_VERSION),
}


def _decode_float(value: object, *, path: str) -> float:
    if not isinstance(value, dict) or set(value) != {"$float"}:
        raise ValueError(f"{path} must be a canonical float")
    encoded = value["$float"]
    if encoded == "nan":
        return math.nan
    if encoded == "+inf":
        return math.inf
    if encoded == "-inf":
        return -math.inf
    if encoded == "-0":
        return -0.0
    if not isinstance(encoded, str):
        raise ValueError(f"{path} has an invalid canonical float")
    try:
        return float.fromhex(encoded)
    except ValueError as exc:
        raise ValueError(f"{path} has an invalid canonical float") from exc


def _decode_typed_value(annotation: object, value: object, *, path: str) -> object:
    if annotation is Any:
        return value
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin in (Union, types.UnionType):
        successes: list[object] = []
        for option in arguments:
            try:
                successes.append(_decode_typed_value(option, value, path=path))
            except (TypeError, ValueError):
                continue
        if len(successes) != 1:
            raise ValueError(f"{path} does not match exactly one declared type")
        return successes[0]
    if origin is tuple:
        if not isinstance(value, list):
            raise ValueError(f"{path} must be a canonical tuple")
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return tuple(
                _decode_typed_value(arguments[0], item, path=f"{path}[{index}]")
                for index, item in enumerate(value)
            )
        if len(value) != len(arguments):
            raise ValueError(f"{path} tuple length mismatch")
        return tuple(
            _decode_typed_value(item_type, item, path=f"{path}[{index}]")
            for index, (item_type, item) in enumerate(zip(arguments, value))
        )
    if annotation is type(None):
        if value is not None:
            raise ValueError(f"{path} must be null")
        return None
    if annotation is bool:
        if not isinstance(value, bool):
            raise ValueError(f"{path} must be a boolean")
        return value
    if annotation is int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{path} must be an integer")
        return value
    if annotation is float:
        return _decode_float(value, path=path)
    if annotation is str:
        if not isinstance(value, str):
            raise ValueError(f"{path} must be a string")
        return value
    if annotation is bytes:
        if not isinstance(value, dict) or set(value) != {"$bytes"}:
            raise ValueError(f"{path} must be canonical bytes")
        encoded = value["$bytes"]
        if not isinstance(encoded, str):
            raise ValueError(f"{path} has invalid canonical bytes")
        try:
            return base64.b64decode(encoded, validate=True)
        except (ValueError, base64.binascii.Error) as exc:
            raise ValueError(f"{path} has invalid canonical bytes") from exc
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        try:
            return annotation(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path} has an invalid enum value") from exc
    if isinstance(annotation, type) and is_dataclass(annotation):
        if not isinstance(value, dict):
            raise ValueError(f"{path} must be an object")
        field_defs = tuple(item for item in fields(annotation) if item.init)
        expected = {item.name for item in field_defs}
        if set(value) != expected:
            raise ValueError(f"{path} fields do not match {annotation.__name__}")
        hints = get_type_hints(annotation)
        decoded = {
            item.name: _decode_typed_value(
                hints[item.name], value[item.name], path=f"{path}.{item.name}"
            )
            for item in field_defs
        }
        try:
            return annotation(**decoded)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{path} violates {annotation.__name__} invariants: {exc}"
            ) from exc
    raise TypeError(f"{path} uses an unsupported declared type")


def _typed_subject_ids(value: object) -> tuple[str, ...]:
    if isinstance(value, V3RunReceipt):
        identifiers = (value.run_id, *value.case_ids)
    elif isinstance(value, RepositoryTestReceipt):
        identifiers = value.node_ids
    elif isinstance(value, TargetVersionReceipt):
        identifiers = value.package_ids
    elif isinstance(value, CandidateClassificationReceipt):
        identifiers = (value.candidate_id or value.source_case_id,)
    elif isinstance(value, PairedBenchmarkReceipt):
        identifiers = (*value.contract_sample_ids, *value.parallel_sample_ids)
    elif isinstance(value, TaskSpec):
        identifiers = (value.identity.task_id,)
    elif isinstance(value, SeedLineage):
        identifiers = (value.digest,)
    elif isinstance(value, StructuredExecutionOutcome):
        identifiers = (value.digest,)
    elif isinstance(value, ResultGroup):
        identifiers = (value.result_group_id,)
    elif isinstance(value, EvidenceEnvelope):
        identifiers = (value.evidence_id,)
    else:  # guarded by the closed registry
        raise ValueError("typed output subject identity is unsupported")
    result = tuple(sorted(identifiers))
    _require_unique_sorted("reconstructed typed subject IDs", result)
    return result


def _reconstruct_typed_output(record: ImmutableTypedOutputRecord) -> object:
    registry_entry = _TYPED_OUTPUT_REGISTRY.get(record.envelope_type)
    if registry_entry is None:
        raise ValueError("typed output envelope type is unknown")
    expected_type, expected_schema = registry_entry
    if record.envelope_schema_version != expected_schema:
        raise ValueError("typed output declared schema version is unsupported")
    try:
        text = record.content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("typed output is not UTF-8") from exc
    envelope = decode_canonical_envelope(text)
    if envelope["type"] != record.envelope_type:
        raise ValueError("typed output envelope type mismatch")
    if envelope["schema_version"] != record.envelope_schema_version:
        raise ValueError("typed output envelope schema mismatch")
    reconstructed = _decode_typed_value(
        expected_type, envelope["payload"], path=f"typed:{record.record_id}"
    )
    if type(reconstructed) is not expected_type:
        raise ValueError("typed output reconstruction type mismatch")
    if getattr(reconstructed, "schema_version", None) != expected_schema:
        raise ValueError("typed output reconstructed schema mismatch")
    if to_primitive(reconstructed) != envelope["payload"]:
        raise ValueError("typed output reconstruction is not exact")
    subject_ids = _typed_subject_ids(reconstructed)
    if subject_ids != record.subject_ids:
        raise ValueError("typed output subject identity mismatch")
    return reconstructed


def _decode_repository_test_raw_context(
    record: ImmutableFileRecord,
) -> RepositoryTestRawContext:
    try:
        text = record.content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("repository raw context is not UTF-8") from exc
    envelope = decode_canonical_envelope(text)
    if envelope["type"] != "RepositoryTestRawContext":
        raise ValueError("repository raw context envelope type mismatch")
    if envelope["schema_version"] != REPOSITORY_TEST_RAW_CONTEXT_SCHEMA_VERSION:
        raise ValueError("repository raw context envelope schema mismatch")
    context = _decode_typed_value(
        RepositoryTestRawContext,
        envelope["payload"],
        path="repository_test_raw_context",
    )
    if not isinstance(context, RepositoryTestRawContext):
        raise ValueError("repository raw context reconstruction type mismatch")
    expected = canonical_envelope(
        "RepositoryTestRawContext",
        REPOSITORY_TEST_RAW_CONTEXT_SCHEMA_VERSION,
        context,
    ).encode("utf-8")
    if expected != record.content:
        raise ValueError("repository raw context is not an exact canonical envelope")
    return context


def _rebuild_repository_test_typed_output(
    parents: tuple[ImmutableFileRecord, ...],
) -> RepositoryTestReceipt:
    context = _decode_repository_test_raw_context(parents[0])
    observation = RepositoryTestObservation(
        source_digest=context.source_digest,
        test_plan=context.test_plan,
        config=context.config,
        environment=context.environment,
        command=context.command,
        junit_xml=parents[1].content,
        log=parents[2].content,
        exit_code=context.exit_code,
        nodes=(),
    )
    return build_repository_test_receipt(observation)


@dataclass(frozen=True, slots=True)
class _RawTypedRebuilderSpec:
    roles: tuple[tuple[str, str, str], ...]
    rebuild: Callable[[tuple[ImmutableFileRecord, ...]], object]


_RAW_TYPED_REBUILDERS: dict[tuple[str, str], _RawTypedRebuilderSpec] = {
    (
        "repository_test_replay",
        "RepositoryTestReceipt",
    ): _RawTypedRebuilderSpec(
        roles=(
            (
                "invocation_context",
                "repository_test_context",
                REPOSITORY_TEST_RAW_CONTEXT_SCHEMA_VERSION,
            ),
            (
                "junit_xml",
                "repository_junit",
                REPOSITORY_TEST_JUNIT_SCHEMA_VERSION,
            ),
            (
                "pytest_log",
                "repository_log",
                REPOSITORY_TEST_LOG_SCHEMA_VERSION,
            ),
        ),
        rebuild=_rebuild_repository_test_typed_output,
    ),
}


def _rebuild_typed_output_from_raw(
    record: ImmutableTypedOutputRecord,
    *,
    producer_kind: str,
    provenance_source_digest: str,
    raw_outputs_by_id: dict[str, ImmutableFileRecord],
) -> object:
    """Require a closed producer/type-specific causal rebuild from raw bytes."""

    claimed = _reconstruct_typed_output(record)
    spec = _RAW_TYPED_REBUILDERS.get((producer_kind, record.envelope_type))
    if spec is None:
        raise ValueError("typed output raw rebuilder context is missing")
    declared_roles = tuple(item.role for item in record.raw_parent_roles)
    expected_roles = tuple(item[0] for item in spec.roles)
    if declared_roles != expected_roles:
        raise ValueError("typed output raw parent role/order mismatch")
    parents: list[ImmutableFileRecord] = []
    for binding, (role, subject_kind, schema_version) in zip(
        record.raw_parent_roles, spec.roles
    ):
        if binding.role != role:
            raise ValueError("typed output raw parent role/order mismatch")
        parent = raw_outputs_by_id.get(binding.record_id)
        if parent is None:
            raise ValueError("typed output raw parent context is incomplete")
        if parent.subject_kind != subject_kind:
            raise ValueError(f"typed output raw parent subject mismatch:{role}")
        if parent.schema_version != schema_version:
            raise ValueError(f"typed output raw parent schema mismatch:{role}")
        parents.append(parent)
    rebuilt = spec.rebuild(tuple(parents))
    if type(rebuilt) is not type(claimed):
        raise ValueError("typed output raw rebuild type mismatch")
    if getattr(rebuilt, "source_digest", provenance_source_digest) != provenance_source_digest:
        raise ValueError("typed output raw source/provenance mismatch")
    rebuilt_subject_ids = _typed_subject_ids(rebuilt)
    if rebuilt_subject_ids != record.subject_ids:
        raise ValueError("typed output rebuilt subject identity mismatch")
    rebuilt_content = canonical_envelope(
        record.envelope_type,
        record.envelope_schema_version,
        rebuilt,
    ).encode("utf-8")
    if rebuilt_content != record.content or rebuilt != claimed:
        raise ValueError("typed output does not exactly match raw reconstruction")
    return rebuilt


def build_producer_provenance(
    *,
    producer_kind: str,
    invocation_id: str,
    source_digest: str,
    protocol_digest: str,
    plan_digest: str,
    command: tuple[str, ...],
    cwd: str,
    environment: tuple[EnvironmentRecord, ...],
    implementation_files: tuple[ImmutableFileRecord, ...],
    inputs: tuple[ImmutableFileRecord, ...],
    raw_outputs: tuple[ImmutableFileRecord, ...],
    typed_outputs: tuple[ImmutableTypedOutputRecord, ...],
    failures: tuple[ProducerFailureRecord, ...] = (),
    exit_code: int = 0,
) -> ProducerProvenanceReceipt:
    """Build a canonical receipt while deriving every digest from raw values."""

    if not isinstance(environment, tuple) or any(
        not isinstance(item, EnvironmentRecord) for item in environment
    ):
        raise ValueError("environment records must be an immutable typed tuple")
    environment_bindings = tuple(
        sorted(
            (_environment_binding(item) for item in environment),
            key=lambda item: item.name,
        )
    )
    environment_names = tuple(item.name for item in environment_bindings)
    if environment_names:
        _require_unique_sorted("environment record names", environment_names)
    if not isinstance(typed_outputs, tuple) or any(
        not isinstance(item, ImmutableTypedOutputRecord) for item in typed_outputs
    ):
        raise ValueError("typed output records must be an immutable typed tuple")
    typed_bindings = tuple(
        sorted((_typed_binding(item) for item in typed_outputs), key=lambda item: item.record_id)
    )
    typed_ids = tuple(item.record_id for item in typed_bindings)
    if typed_ids:
        _require_unique_sorted("typed output record IDs", typed_ids)
    if not isinstance(failures, tuple) or any(
        not isinstance(item, ProducerFailureRecord) for item in failures
    ):
        raise ValueError("failure records must be an immutable typed tuple")
    failure_bindings = tuple(
        sorted((_failure_binding(item) for item in failures), key=lambda item: item.failure_id)
    )
    failure_ids = tuple(item.failure_id for item in failure_bindings)
    if failure_ids:
        _require_unique_sorted("failure record IDs", failure_ids)
    return ProducerProvenanceReceipt(
        producer_kind=producer_kind,
        invocation_id=invocation_id,
        source_digest=source_digest,
        protocol_digest=protocol_digest,
        plan_digest=plan_digest,
        command=command,
        command_digest=stable_digest("osc-root-producer-command", command),
        cwd=cwd,
        environment=environment_bindings,
        environment_digest=stable_digest(
            "osc-root-producer-environment", environment_bindings
        ),
        implementation_files=_sorted_file_bindings(
            implementation_files,
            label="implementation",
            category=ProducerRecordCategory.IMPLEMENTATION,
        ),
        inputs=_sorted_file_bindings(
            inputs, label="input", category=ProducerRecordCategory.INPUT
        ),
        raw_outputs=_sorted_file_bindings(
            raw_outputs,
            label="raw output",
            category=ProducerRecordCategory.RAW_OUTPUT,
        ),
        typed_outputs=typed_bindings,
        failures=failure_bindings,
        exit_code=exit_code,
    )


def _policy_projection(
    item: FileDigestBinding | TypedOutputBinding,
) -> tuple[object, ...]:
    if isinstance(item, TypedOutputBinding):
        return (
            item.category,
            item.record_id,
            item.subject_kind,
            item.subject_ids,
            item.relative_path,
            item.envelope_schema_version,
            item.envelope_type,
            item.raw_parent_roles,
        )
    return (
        item.category,
        item.record_id,
        item.subject_kind,
        (),
        item.relative_path,
        item.schema_version,
        "",
        (),
    )


def _expected_projection(item: ExpectedProvenanceRecordPolicy) -> tuple[object, ...]:
    return (
        item.category,
        item.record_id,
        item.subject_kind,
        item.subject_ids,
        item.relative_path,
        item.schema_version,
        item.envelope_type,
        item.raw_parent_roles,
    )


def verify_producer_provenance(
    receipt: ProducerProvenanceReceipt,
    *,
    policy: ProducerProvenancePolicy,
    source_snapshot: SourceSnapshotView,
    environment: tuple[EnvironmentRecord, ...],
    implementation_files: tuple[ImmutableFileRecord, ...],
    inputs: tuple[ImmutableFileRecord, ...],
    raw_outputs: tuple[ImmutableFileRecord, ...],
    typed_outputs: tuple[ImmutableTypedOutputRecord, ...],
    failures: tuple[ProducerFailureRecord, ...] = (),
) -> ProvenanceVerification:
    """Recompute a receipt and compare it with independent exact context."""

    if not isinstance(receipt, ProducerProvenanceReceipt):
        raise TypeError("receipt must be a ProducerProvenanceReceipt")
    if not isinstance(policy, ProducerProvenancePolicy):
        raise TypeError("policy must be a ProducerProvenancePolicy")
    if not isinstance(source_snapshot, SourceSnapshotView):
        raise TypeError("source_snapshot must be a SourceSnapshotView")
    errors: list[str] = []
    try:
        rebuilt = build_producer_provenance(
            producer_kind=receipt.producer_kind,
            invocation_id=receipt.invocation_id,
            source_digest=receipt.source_digest,
            protocol_digest=receipt.protocol_digest,
            plan_digest=receipt.plan_digest,
            command=receipt.command,
            cwd=receipt.cwd,
            environment=environment,
            implementation_files=implementation_files,
            inputs=inputs,
            raw_outputs=raw_outputs,
            typed_outputs=typed_outputs,
            failures=failures,
            exit_code=receipt.exit_code,
        )
    except (TypeError, ValueError) as exc:
        errors.append(f"provenance_rebuild_failed:{type(exc).__name__}:{exc}")
        rebuilt = None
    if rebuilt is not None and rebuilt != receipt:
        errors.append("provenance_raw_binding_mismatch")

    for field_name in (
        "producer_kind",
        "invocation_id",
        "source_digest",
        "protocol_digest",
        "plan_digest",
        "command",
        "cwd",
    ):
        if getattr(receipt, field_name) != getattr(policy, field_name):
            errors.append(f"provenance_policy_mismatch:{field_name}")
    if receipt.environment != policy.environment:
        errors.append("provenance_policy_mismatch:environment")
    if receipt.exit_code not in policy.allowed_exit_codes:
        errors.append("provenance_exit_code_not_allowed")
    if source_snapshot.source_digest != policy.source_digest:
        errors.append("provenance_snapshot_source_mismatch")
    if receipt.failures or failures:
        errors.append("provenance_structured_failure_present")

    observed_records: tuple[FileDigestBinding | TypedOutputBinding, ...] = (
        *receipt.implementation_files,
        *receipt.inputs,
        *receipt.raw_outputs,
        *receipt.typed_outputs,
    )
    expected_by_id = {item.record_id: item for item in policy.expected_records}
    observed_by_id = {item.record_id: item for item in observed_records}
    if set(observed_by_id) != set(expected_by_id):
        errors.append("provenance_policy_mismatch:record_set")
    for record_id in sorted(set(observed_by_id).intersection(expected_by_id)):
        observed = observed_by_id[record_id]
        expected = expected_by_id[record_id]
        if _policy_projection(observed) != _expected_projection(expected):
            errors.append(f"provenance_record_policy_mismatch:{record_id}")
        if observed.origin is not expected.origin:
            errors.append(
                f"provenance_origin_not_allowed:{record_id}:{observed.origin.value}"
            )
        if expected.origin in _FROZEN_ORIGINS:
            if observed.sha256 != expected.frozen_source_sha256:
                errors.append(f"provenance_frozen_source_hash_mismatch:{record_id}")

    snapshot_files = source_snapshot.file_map()
    for item in receipt.implementation_files:
        expected_sha = snapshot_files.get(item.relative_path)
        if expected_sha is None:
            errors.append(
                f"provenance_implementation_missing_from_snapshot:{item.relative_path}"
            )
        elif expected_sha != item.sha256:
            errors.append(
                f"provenance_implementation_snapshot_hash_mismatch:{item.relative_path}"
            )

    raw_ids = frozenset(item.record_id for item in receipt.raw_outputs)
    raw_outputs_by_id = {item.record_id: item for item in raw_outputs}
    if len(raw_outputs_by_id) != len(raw_outputs):
        errors.append("provenance_raw_output_record_ids_not_unique")
    for record in typed_outputs:
        if not set(record.raw_parent_record_ids).issubset(raw_ids):
            errors.append(f"provenance_typed_raw_parent_unknown:{record.record_id}")
        try:
            _rebuild_typed_output_from_raw(
                record,
                producer_kind=receipt.producer_kind,
                provenance_source_digest=receipt.source_digest,
                raw_outputs_by_id=raw_outputs_by_id,
            )
        except (TypeError, ValueError) as exc:
            errors.append(
                f"provenance_typed_raw_rebuild_failed:{record.record_id}:{type(exc).__name__}:{exc}"
            )

    return ProvenanceVerification(
        receipt_digest=receipt.digest,
        errors=tuple(dict.fromkeys(errors)),
    )


__all__ = [
    "PRODUCER_PROVENANCE_SCHEMA_VERSION",
    "EnvironmentBinding",
    "EnvironmentRecord",
    "ExpectedProvenanceRecordPolicy",
    "FileDigestBinding",
    "ImmutableFileRecord",
    "ImmutableTypedOutputRecord",
    "ProducerFailureRecord",
    "ProducerOriginKind",
    "ProducerProvenancePolicy",
    "ProducerProvenanceReceipt",
    "ProducerRecordCategory",
    "ProvenanceVerification",
    "RawParentRoleBinding",
    "RepositoryTestRawContext",
    "SourceSnapshotFileBinding",
    "SourceSnapshotView",
    "StructuredProducerFailure",
    "TypedOutputBinding",
    "build_producer_provenance",
    "verify_producer_provenance",
]
