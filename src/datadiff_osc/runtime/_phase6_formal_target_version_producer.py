"""Private Phase-6 target-version preparation and no-write capability binding.

The module deliberately has no formal execution entry point.  It can prepare
immutable installed-METADATA bytes and frozen public PyPI response bytes, and
it can also inspect an authority-selected target interpreter under a private,
no-write capability boundary.  Neither path creates formal evidence.  A later,
separately authorized task must bind a fresh source snapshot, an exact output
path, and an execution authority before it may persist a formal
``target_version_replay`` record.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from datadiff.target_version_audit import DEFAULT_TARGET_PACKAGES
from datadiff_osc._canonical import (
    assert_deeply_immutable,
    canonical_json,
    canonical_envelope,
    decode_canonical_envelope,
    stable_digest,
)
from datadiff_osc.runtime._private_receipts import TargetVersionReceipt
from datadiff_osc.runtime._receipt_producers import (
    TargetPackageObservation,
    TargetVersionObservation,
    build_target_version_receipt,
)
from datadiff_osc.runtime._semantic_replay import replay_runtime_admission


FORMAL_TARGET_VERSION_PREPARATION_SCHEMA_VERSION = (
    "osc-private-phase6-formal-target-version-preparation-v1"
)
FORMAL_TARGET_VERSION_PREPARATION_MODE = "prepare-only-no-authority-v1"
FORMAL_TARGET_VERSION_EVIDENCE_MANIFEST_SCHEMA_VERSION = (
    "osc-private-phase6-target-version-evidence-manifest-v1"
)
FORMAL_TARGET_VERSION_EVIDENCE_WRITER_MODE = "test-only-staged-no-authority-v1"
FORMAL_TARGET_VERSION_EXECUTION_CAPABILITY_SCHEMA_VERSION = (
    "osc-private-phase6-target-version-execution-capability-v1"
)
FORMAL_TARGET_VERSION_EXECUTION_CAPABILITY_MODE = (
    "test-only-capability-no-durable-output-v1"
)
FORMAL_TARGET_VERSION_METADATA_PROBE_SCHEMA_VERSION = (
    "osc-private-phase6-target-version-metadata-probe-v1"
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_SHA1_OBJECT_ID_RE = re.compile(r"[0-9a-f]{40}")

_EXPECTED_TARGETS = tuple(
    (item.package, item.target) for item in DEFAULT_TARGET_PACKAGES
)
_EXPECTED_PUBLIC_RAW_SHA256 = {
    "pandas": "def430f070766d7d545517b3c2a08afbb59f865c4297281d00fafcc4e25a123f",
    "polars": "f9e79734733ed9c25f2c0a822a9e2f3bf8d7765201dc507293cd594d1e25e23c",
    "duckdb": "d2f9d34a9ba3976cae39e4bb69b464227851c907493e9d1395c91710f3a5a721",
    "pyarrow": "752ee190964d03405cf7e53a2ba55c173c9f69a0f87a53383f19cd760cd80982",
    "datafusion": "27287ace58a4003d8e4b5373d7c688cb0266a8e01ddc45d7f48ec803c515c42c",
    "chdb": "25dfd4a21d3b82d9c83cbbe46c2965f37f9d8f8e89c5c2703dae107afd19a2b9",
    "pysqlite3-binary": "250725f2b802e5d21db15739aa8ae3f18477dd27cee6803eef8f263594ab6d0c",
}

_TARGET_METADATA_PROBE_CODE = """import base64
import hashlib
import importlib.metadata
import json
import sys

TARGETS = (
    (\"pandas\", \"pandas\"),
    (\"polars\", \"polars\"),
    (\"duckdb\", \"duckdb\"),
    (\"pyarrow\", \"pyarrow\"),
    (\"datafusion\", \"datafusion\"),
    (\"chdb\", \"chdb\"),
    (\"pysqlite3-binary\", \"sqlite\"),
)
records = []
for distribution_name, import_name in TARGETS:
    distribution = importlib.metadata.distribution(distribution_name)
    metadata_files = tuple(
        item for item in (distribution.files or ()) if item.name == \"METADATA\"
    )
    if len(metadata_files) != 1:
        raise RuntimeError(\"expected exactly one installed METADATA file\")
    metadata_path = distribution.locate_file(metadata_files[0]).resolve()
    metadata = metadata_path.read_bytes()
    records.append(
        {
            \"distribution_name\": distribution_name,
            \"import_name\": import_name,
            \"metadata_base64\": base64.b64encode(metadata).decode(\"ascii\"),
            \"metadata_path\": str(metadata_path),
            \"metadata_sha256\": hashlib.sha256(metadata).hexdigest(),
        }
    )
payload = {
    \"implementation\": sys.implementation.name,
    \"interpreter\": sys.executable,
    \"records\": records,
    \"schema_version\": \"osc-private-phase6-target-version-metadata-probe-v1\",
    \"version\": sys.version,
}
sys.stdout.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(\",\", \":\"), allow_nan=False))
"""
_TARGET_METADATA_PROBE_COMMAND_SHA256 = hashlib.sha256(
    _TARGET_METADATA_PROBE_CODE.encode("utf-8")
).hexdigest()


def _require_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{name} must be non-empty text without NUL")
    if "synthetic" in value.lower():
        raise ValueError(f"{name} cannot contain a synthetic marker")
    return value


def _require_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _require_git_sha1_object_id(value: object, *, name: str) -> str:
    text = _require_text(value, name=name)
    if _GIT_SHA1_OBJECT_ID_RE.fullmatch(text) is None:
        raise ValueError(f"{name} must be a lowercase 40-hex Git SHA-1 object ID")
    return text


def _require_fields(value: object, *, name: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{name} must be a non-empty immutable tuple")
    fields: list[tuple[str, str]] = []
    for item in value:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError(f"{name} entries must be key/value tuples")
        key = _require_text(item[0], name=f"{name} key")
        item_value = _require_text(item[1], name=f"{name} value")
        fields.append((key, item_value))
    result = tuple(fields)
    if result != tuple(sorted(result)) or len({key for key, _ in result}) != len(result):
        raise ValueError(f"{name} fields must be uniquely sorted")
    return result


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require_absolute_path_text(value: object, *, name: str) -> str:
    text = _require_text(value, name=name)
    path = Path(text)
    if not path.is_absolute() or not path.name:
        raise ValueError(f"{name} must be an absolute non-root path")
    if ".." in path.parts:
        raise ValueError(f"{name} cannot contain a parent traversal")
    return text


@dataclass(frozen=True, slots=True)
class TargetVersionRawMaterial:
    """Immutable raw bytes for one authority-selected target package."""

    distribution_name: str
    import_name: str
    installed_metadata: bytes
    public_version_source: bytes

    def __post_init__(self) -> None:
        _require_text(self.distribution_name, name="target distribution name")
        _require_text(self.import_name, name="target import name")
        if not isinstance(self.installed_metadata, bytes):
            raise TypeError("installed METADATA must be immutable bytes")
        if not isinstance(self.public_version_source, bytes):
            raise TypeError("public version source must be immutable bytes")
        if not self.installed_metadata or not self.public_version_source:
            raise ValueError("target raw material bytes must be non-empty")
        assert_deeply_immutable(self)


@dataclass(frozen=True, slots=True)
class TargetVersionPreparation:
    """An in-memory, explicitly non-authority preparation result."""

    source_digest: str
    source_snapshot_sha256: str
    requirements_lock_sha256: str
    target_revalidation_sha256: str
    environment_audit_sha256: str
    materials: tuple[TargetVersionRawMaterial, ...]
    environment: tuple[tuple[str, str], ...]
    python_runtime: tuple[tuple[str, str], ...]
    receipt: TargetVersionReceipt
    payload_json: str
    schema_version: str = FORMAL_TARGET_VERSION_PREPARATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_text(self.source_digest, name="target source digest")
        for name in (
            "source_snapshot_sha256",
            "requirements_lock_sha256",
            "target_revalidation_sha256",
            "environment_audit_sha256",
        ):
            _require_sha256(getattr(self, name), name=name)
        _require_materials(self.materials)
        _require_fields(self.environment, name="target environment")
        _require_fields(self.python_runtime, name="target Python runtime")
        if not isinstance(self.receipt, TargetVersionReceipt):
            raise TypeError("target preparation requires TargetVersionReceipt")
        if self.receipt.source_digest != self.source_digest:
            raise ValueError("target preparation receipt/source digest mismatch")
        if self.schema_version != FORMAL_TARGET_VERSION_PREPARATION_SCHEMA_VERSION:
            raise ValueError("target preparation schema version mismatch")
        expected = canonical_json(
            _payload(
                source_digest=self.source_digest,
                source_snapshot_sha256=self.source_snapshot_sha256,
                requirements_lock_sha256=self.requirements_lock_sha256,
                target_revalidation_sha256=self.target_revalidation_sha256,
                environment_audit_sha256=self.environment_audit_sha256,
                materials=self.materials,
                receipt=self.receipt,
            )
        )
        if self.payload_json != expected:
            raise ValueError("target preparation payload is not canonical")
        assert_deeply_immutable(self)

    @property
    def preparation_id(self) -> str:
        return stable_digest(
            "osc-private-phase6-target-version-preparation-id",
            (self.source_digest, self.receipt.digest, self.payload_json),
        )

    @property
    def authority_eligible(self) -> bool:
        return False

    @property
    def formal_evidence_created(self) -> bool:
        return False

    @property
    def gate_credit(self) -> bool:
        return False

    @property
    def candidate_confirmed(self) -> bool:
        return False

    @property
    def bug_claimed(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class TargetVersionEvidenceWriteRequest:
    """Exact bindings for one private, test-only target-version output tree."""

    output_root: str
    source_digest: str
    source_snapshot_sha256: str
    requirements_lock_sha256: str
    target_revalidation_sha256: str
    environment_audit_sha256: str
    receipt_digest: str
    schema_version: str = FORMAL_TARGET_VERSION_EVIDENCE_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_output_root_text(self.output_root)
        _require_text(self.source_digest, name="writer source digest")
        for name in (
            "source_snapshot_sha256",
            "requirements_lock_sha256",
            "target_revalidation_sha256",
            "environment_audit_sha256",
        ):
            _require_sha256(getattr(self, name), name=name)
        _require_text(self.receipt_digest, name="writer receipt digest")
        if self.schema_version != FORMAL_TARGET_VERSION_EVIDENCE_MANIFEST_SCHEMA_VERSION:
            raise ValueError("target-version evidence manifest schema version mismatch")
        assert_deeply_immutable(self)

    @property
    def request_id(self) -> str:
        return stable_digest(
            "osc-private-phase6-target-version-evidence-write-request",
            (
                self.output_root,
                self.source_digest,
                self.source_snapshot_sha256,
                self.requirements_lock_sha256,
                self.target_revalidation_sha256,
                self.environment_audit_sha256,
                self.receipt_digest,
                self.schema_version,
            ),
        )


@dataclass(frozen=True, slots=True)
class TargetVersionEvidenceWriteResult:
    """A re-readable private test-output record, never Root authority."""

    output_root: str
    raw_manifest_path: str
    receipt_path: str
    raw_manifest_sha256: str
    receipt_sha256: str
    source_digest: str
    source_snapshot_sha256: str
    receipt_digest: str
    schema_version: str = FORMAL_TARGET_VERSION_EVIDENCE_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "output_root",
            "raw_manifest_path",
            "receipt_path",
            "source_digest",
            "receipt_digest",
        ):
            _require_text(getattr(self, name), name=f"writer result {name}")
        for name in (
            "raw_manifest_sha256",
            "receipt_sha256",
            "source_snapshot_sha256",
        ):
            _require_sha256(getattr(self, name), name=name)
        if self.schema_version != FORMAL_TARGET_VERSION_EVIDENCE_MANIFEST_SCHEMA_VERSION:
            raise ValueError("target-version evidence result schema version mismatch")
        assert_deeply_immutable(self)

    @property
    def evidence_id(self) -> str:
        return stable_digest(
            "osc-private-phase6-target-version-evidence-write-result",
            (
                self.source_digest,
                self.source_snapshot_sha256,
                self.receipt_digest,
                self.raw_manifest_sha256,
                self.receipt_sha256,
            ),
        )

    @property
    def authority_eligible(self) -> bool:
        return False

    @property
    def formal_evidence_created(self) -> bool:
        return False

    @property
    def gate_credit(self) -> bool:
        return False

    @property
    def candidate_confirmed(self) -> bool:
        return False

    @property
    def bug_claimed(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class TargetVersionPublicSourceBinding:
    """One immutable official raw-source path for the private capability."""

    distribution_name: str
    path: str
    sha256: str

    def __post_init__(self) -> None:
        distribution = _require_text(
            self.distribution_name, name="public source distribution name"
        )
        _require_absolute_path_text(self.path, name="public source path")
        _require_sha256(self.sha256, name="public source SHA-256")
        expected = _EXPECTED_PUBLIC_RAW_SHA256.get(distribution)
        if expected is None or self.sha256 != expected:
            raise ValueError("public source binding SHA does not match frozen target")
        if Path(self.path).name != f"pypi-{distribution}.json":
            raise ValueError("public source binding path/name mismatch")
        assert_deeply_immutable(self)


@dataclass(frozen=True, slots=True)
class TargetVersionInstalledMetadataRecord:
    """Metadata bytes independently re-read from the selected target venv."""

    distribution_name: str
    import_name: str
    metadata_path: str
    metadata_sha256: str
    installed_metadata: bytes

    def __post_init__(self) -> None:
        _require_text(self.distribution_name, name="installed metadata distribution name")
        _require_text(self.import_name, name="installed metadata import name")
        _require_absolute_path_text(self.metadata_path, name="installed metadata path")
        _require_sha256(self.metadata_sha256, name="installed metadata SHA-256")
        if not isinstance(self.installed_metadata, bytes) or not self.installed_metadata:
            raise TypeError("installed metadata record requires non-empty immutable bytes")
        if _sha256(self.installed_metadata) != self.metadata_sha256:
            raise ValueError("installed metadata record SHA mismatch")
        assert_deeply_immutable(self)


@dataclass(frozen=True, slots=True)
class TargetVersionFutureExecutionAuthority:
    """Private shape for a later authority; it grants no write in this task."""

    execution_root: str
    future_output_root: str
    execution_command_sha256: str
    source_snapshot_path: str
    source_snapshot_sha256: str
    source_digest: str
    source_git_head: str
    requirements_lock_path: str
    requirements_lock_sha256: str
    target_revalidation_path: str
    target_revalidation_sha256: str
    environment_audit_path: str
    environment_audit_sha256: str
    historical_environment_source_snapshot_sha256: str
    historical_environment_source_digest: str
    target_interpreter: str
    target_interpreter_sha256: str
    public_source_root: str
    public_sources: tuple[TargetVersionPublicSourceBinding, ...]
    metadata_probe_command_sha256: str = _TARGET_METADATA_PROBE_COMMAND_SHA256
    authorization_mode: str = FORMAL_TARGET_VERSION_EXECUTION_CAPABILITY_MODE
    formal_execution_authorized: bool = False
    schema_version: str = FORMAL_TARGET_VERSION_EXECUTION_CAPABILITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_absolute_path_text(self.execution_root, name="execution root")
        output_root = _require_absolute_path_text(
            self.future_output_root, name="future output root"
        )
        if Path(output_root).name != "formal_evidence":
            raise ValueError("future output root must be the exact formal_evidence leaf")
        for name in (
            "source_snapshot_path",
            "requirements_lock_path",
            "target_revalidation_path",
            "environment_audit_path",
            "target_interpreter",
            "public_source_root",
        ):
            _require_absolute_path_text(getattr(self, name), name=name)
        _require_text(self.source_digest, name="execution source digest")
        _require_text(
            self.historical_environment_source_digest,
            name="historical environment source digest",
        )
        for name in (
            "execution_command_sha256",
            "source_snapshot_sha256",
            "requirements_lock_sha256",
            "target_revalidation_sha256",
            "environment_audit_sha256",
            "historical_environment_source_snapshot_sha256",
            "target_interpreter_sha256",
            "metadata_probe_command_sha256",
        ):
            _require_sha256(getattr(self, name), name=name)
        _require_git_sha1_object_id(self.source_git_head, name="source_git_head")
        _require_public_source_bindings(self.public_sources)
        if self.metadata_probe_command_sha256 != _TARGET_METADATA_PROBE_COMMAND_SHA256:
            raise ValueError("target metadata probe command digest mismatch")
        if self.authorization_mode != FORMAL_TARGET_VERSION_EXECUTION_CAPABILITY_MODE:
            raise ValueError("target-version execution capability mode mismatch")
        if self.schema_version != FORMAL_TARGET_VERSION_EXECUTION_CAPABILITY_SCHEMA_VERSION:
            raise ValueError("target-version execution capability schema mismatch")
        if self.formal_execution_authorized is not False:
            raise ValueError("formal target-version execution is not authorized")
        if (
            self.source_snapshot_sha256
            == self.historical_environment_source_snapshot_sha256
            or self.source_digest == self.historical_environment_source_digest
        ):
            raise ValueError("current and historical source provenance must remain distinct")
        assert_deeply_immutable(self)

    @property
    def expected_execution_command_sha256(self) -> str:
        return _execution_command_digest(self)

    @property
    def authority_id(self) -> str:
        return stable_digest(
            "osc-private-phase6-target-version-future-execution-authority",
            (
                self.execution_root,
                self.future_output_root,
                self.execution_command_sha256,
                self.source_snapshot_sha256,
                self.source_digest,
                self.source_git_head,
                self.target_revalidation_sha256,
                self.environment_audit_sha256,
                self.target_interpreter_sha256,
                self.public_source_root,
                tuple(
                    (item.distribution_name, item.path, item.sha256)
                    for item in self.public_sources
                ),
            ),
        )

    @property
    def authority_eligible(self) -> bool:
        return False

    @property
    def formal_evidence_created(self) -> bool:
        return False

    @property
    def gate_credit(self) -> bool:
        return False

    @property
    def candidate_confirmed(self) -> bool:
        return False

    @property
    def bug_claimed(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class TargetVersionExecutionCapability:
    """Collected, no-write target metadata with a permanently false authority bit."""

    authority_id: str
    source_digest: str
    source_snapshot_sha256: str
    target_interpreter: str
    future_output_root: str
    metadata_probe_transcript_sha256: str
    metadata_records: tuple[TargetVersionInstalledMetadataRecord, ...]
    preparation: TargetVersionPreparation
    schema_version: str = FORMAL_TARGET_VERSION_EXECUTION_CAPABILITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_text(self.authority_id, name="execution capability authority id")
        _require_text(self.source_digest, name="execution capability source digest")
        _require_sha256(
            self.source_snapshot_sha256, name="execution capability source snapshot SHA-256"
        )
        _require_absolute_path_text(
            self.target_interpreter, name="execution capability target interpreter"
        )
        _require_absolute_path_text(
            self.future_output_root, name="execution capability future output root"
        )
        _require_sha256(
            self.metadata_probe_transcript_sha256,
            name="execution capability metadata transcript SHA-256",
        )
        _require_metadata_records(self.metadata_records)
        if not isinstance(self.preparation, TargetVersionPreparation):
            raise TypeError("execution capability requires TargetVersionPreparation")
        if (
            self.preparation.source_digest != self.source_digest
            or self.preparation.source_snapshot_sha256 != self.source_snapshot_sha256
        ):
            raise ValueError("execution capability preparation/source binding mismatch")
        if self.schema_version != FORMAL_TARGET_VERSION_EXECUTION_CAPABILITY_SCHEMA_VERSION:
            raise ValueError("execution capability schema mismatch")
        assert_deeply_immutable(self)

    @property
    def formal_execution_authorized(self) -> bool:
        return False

    @property
    def authority_eligible(self) -> bool:
        return False

    @property
    def formal_evidence_created(self) -> bool:
        return False

    @property
    def gate_credit(self) -> bool:
        return False

    @property
    def candidate_confirmed(self) -> bool:
        return False

    @property
    def bug_claimed(self) -> bool:
        return False


def _require_materials(value: object) -> tuple[TargetVersionRawMaterial, ...]:
    if not isinstance(value, tuple) or any(
        not isinstance(item, TargetVersionRawMaterial) for item in value
    ):
        raise TypeError("target materials must be an immutable tuple of raw materials")
    expected_identities = _EXPECTED_TARGETS
    actual_identities = tuple(
        (item.distribution_name, item.import_name) for item in value
    )
    if actual_identities != expected_identities:
        raise ValueError("target materials must match the canonical target sequence")
    if len({item.distribution_name for item in value}) != len(value):
        raise ValueError("target material distributions must be unique")
    return value


def _require_public_source_bindings(
    value: object,
) -> tuple[TargetVersionPublicSourceBinding, ...]:
    if not isinstance(value, tuple) or any(
        not isinstance(item, TargetVersionPublicSourceBinding) for item in value
    ):
        raise TypeError("public source bindings must be an immutable tuple")
    expected = tuple(distribution for distribution, _ in _EXPECTED_TARGETS)
    actual = tuple(item.distribution_name for item in value)
    if actual != expected:
        raise ValueError("public source bindings must match the canonical target sequence")
    if len({item.path for item in value}) != len(value):
        raise ValueError("public source binding paths must be unique")
    return value


def _require_metadata_records(
    value: object,
) -> tuple[TargetVersionInstalledMetadataRecord, ...]:
    if not isinstance(value, tuple) or any(
        not isinstance(item, TargetVersionInstalledMetadataRecord) for item in value
    ):
        raise TypeError("installed metadata records must be an immutable tuple")
    expected = _EXPECTED_TARGETS
    actual = tuple((item.distribution_name, item.import_name) for item in value)
    if actual != expected:
        raise ValueError("installed metadata records must match canonical target identities")
    if len({item.metadata_path for item in value}) != len(value):
        raise ValueError("installed metadata paths must be unique")
    return value


def _execution_command_digest(authority: TargetVersionFutureExecutionAuthority) -> str:
    """Digest the future write shape without exposing an executable write path."""

    payload = {
        "schema_version": FORMAL_TARGET_VERSION_EXECUTION_CAPABILITY_SCHEMA_VERSION,
        "mode": FORMAL_TARGET_VERSION_EXECUTION_CAPABILITY_MODE,
        "execution_root": authority.execution_root,
        "future_output_root": authority.future_output_root,
        "source_snapshot_sha256": authority.source_snapshot_sha256,
        "source_digest": authority.source_digest,
        "source_git_head": authority.source_git_head,
        "requirements_lock_sha256": authority.requirements_lock_sha256,
        "target_revalidation_sha256": authority.target_revalidation_sha256,
        "environment_audit_sha256": authority.environment_audit_sha256,
        "target_interpreter_sha256": authority.target_interpreter_sha256,
        "metadata_probe_command_sha256": authority.metadata_probe_command_sha256,
        "public_source_root": authority.public_source_root,
        "public_sources": [
            {
                "distribution_name": item.distribution_name,
                "path": item.path,
                "sha256": item.sha256,
            }
            for item in authority.public_sources
        ],
    }
    return _sha256(canonical_json(payload).encode("utf-8"))


def _observation(
    *,
    source_digest: str,
    materials: tuple[TargetVersionRawMaterial, ...],
    environment: tuple[tuple[str, str], ...],
    python_runtime: tuple[tuple[str, str], ...],
) -> TargetVersionObservation:
    packages: list[TargetPackageObservation] = []
    for material in materials:
        expected_sha = _EXPECTED_PUBLIC_RAW_SHA256[material.distribution_name]
        actual_sha = _sha256(material.public_version_source)
        if actual_sha != expected_sha:
            raise ValueError(
                "public target version source SHA mismatch: "
                + material.distribution_name
            )
        packages.append(
            TargetPackageObservation(
                distribution_name=material.distribution_name,
                import_name=material.import_name,
                installed_version=_metadata_version(material.installed_metadata),
                latest_version=_public_version(material.public_version_source),
                installed_metadata=material.installed_metadata,
                version_source_kind="pypi_json",
                version_source=material.public_version_source,
            )
        )
    audit_plan = tuple(
        sorted(
            (distribution, _EXPECTED_PUBLIC_RAW_SHA256[distribution])
            for distribution, _ in _EXPECTED_TARGETS
        )
    )
    return TargetVersionObservation(
        source_digest=source_digest,
        audit_plan=audit_plan,
        environment=environment,
        python_runtime=python_runtime,
        packages=tuple(packages),
    )


def _metadata_version(raw: bytes) -> str:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("installed METADATA is not UTF-8") from exc
    header_lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    values = []
    for line in header_lines:
        if not line:
            break
        if line.lower().startswith("version:"):
            values.append(line.split(":", 1)[1].strip())
    if len(values) != 1 or not values[0]:
        raise ValueError("installed METADATA requires exactly one Version")
    return values[0]


def _public_version(raw: bytes) -> str:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("public target version source is not UTF-8") from exc

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("public target version source has a duplicate JSON key")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"public target version source has non-standard JSON constant: {value}")

    try:
        decoded = json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError("public target version source is malformed JSON") from exc
    if not isinstance(decoded, dict) or not isinstance(decoded.get("info"), dict):
        raise ValueError("public target version source requires an info object")
    return _require_text(decoded["info"].get("version"), name="public target version")


def _payload(
    *,
    source_digest: str,
    source_snapshot_sha256: str,
    requirements_lock_sha256: str,
    target_revalidation_sha256: str,
    environment_audit_sha256: str,
    materials: tuple[TargetVersionRawMaterial, ...],
    receipt: TargetVersionReceipt,
) -> dict[str, object]:
    return {
        "schema_version": FORMAL_TARGET_VERSION_PREPARATION_SCHEMA_VERSION,
        "mode": FORMAL_TARGET_VERSION_PREPARATION_MODE,
        "source_digest": source_digest,
        "source_snapshot_sha256": source_snapshot_sha256,
        "requirements_lock_sha256": requirements_lock_sha256,
        "target_revalidation_sha256": target_revalidation_sha256,
        "environment_audit_sha256": environment_audit_sha256,
        "public_raw_sha256": [
            _EXPECTED_PUBLIC_RAW_SHA256[item.distribution_name]
            for item in materials
        ],
        "receipt": receipt.to_dict(),
        "all_targets_match_latest": True,
        "formal_evidence_created": False,
        "authority_eligible": False,
        "gate_credit": False,
        "candidate_confirmed": False,
        "bug_claimed": False,
    }


def prepare_target_version_replay(
    *,
    source_digest: str,
    source_snapshot_sha256: str,
    requirements_lock_sha256: str,
    target_revalidation_sha256: str,
    environment_audit_sha256: str,
    materials: tuple[TargetVersionRawMaterial, ...],
    environment: tuple[tuple[str, str], ...],
    python_runtime: tuple[tuple[str, str], ...],
) -> TargetVersionPreparation:
    """Create a nonpersistent, no-authority target-version preparation."""

    source_text = _require_text(source_digest, name="target source digest")
    _require_materials(materials)
    normalized_environment = _require_fields(environment, name="target environment")
    normalized_runtime = _require_fields(python_runtime, name="target Python runtime")
    for name, value in (
        ("source_snapshot_sha256", source_snapshot_sha256),
        ("requirements_lock_sha256", requirements_lock_sha256),
        ("target_revalidation_sha256", target_revalidation_sha256),
        ("environment_audit_sha256", environment_audit_sha256),
    ):
        _require_sha256(value, name=name)
    receipt = build_target_version_receipt(
        _observation(
            source_digest=source_text,
            materials=materials,
            environment=normalized_environment,
            python_runtime=normalized_runtime,
        )
    )
    if any(item.installed_version != item.latest_version for item in receipt.packages):
        raise ValueError("target latest version mismatch")
    payload_json = canonical_json(
        _payload(
            source_digest=source_text,
            source_snapshot_sha256=source_snapshot_sha256,
            requirements_lock_sha256=requirements_lock_sha256,
            target_revalidation_sha256=target_revalidation_sha256,
            environment_audit_sha256=environment_audit_sha256,
            materials=materials,
            receipt=receipt,
        )
    )
    return TargetVersionPreparation(
        source_digest=source_text,
        source_snapshot_sha256=source_snapshot_sha256,
        requirements_lock_sha256=requirements_lock_sha256,
        target_revalidation_sha256=target_revalidation_sha256,
        environment_audit_sha256=environment_audit_sha256,
        materials=materials,
        environment=normalized_environment,
        python_runtime=normalized_runtime,
        receipt=receipt,
        payload_json=payload_json,
    )


def _require_output_root_text(value: object) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("target-version output root must be non-empty text without NUL")
    if "synthetic" in value.lower():
        raise ValueError("target-version output root cannot contain a synthetic marker")
    path = Path(value)
    if not path.is_absolute() or not path.name:
        raise ValueError("target-version output root must be an absolute directory path")
    if ".." in path.parts:
        raise ValueError("target-version output root cannot contain a parent traversal")
    return value


def _require_no_symlink_components(path: Path) -> None:
    if not path.is_absolute():
        raise ValueError("target-version output path must be absolute")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if os.path.islink(current):
            raise ValueError("target-version output path cannot traverse a symlink")


def _new_output_root(request: TargetVersionEvidenceWriteRequest) -> Path:
    root = Path(_require_output_root_text(request.output_root))
    if os.path.lexists(root):
        raise ValueError("target-version output root already exists")
    parent = root.parent
    _require_no_symlink_components(parent)
    if not parent.is_dir():
        raise ValueError("target-version output parent must be an existing directory")
    return root


def _existing_output_root(request: TargetVersionEvidenceWriteRequest) -> Path:
    root = Path(_require_output_root_text(request.output_root))
    _require_no_symlink_components(root.parent)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("target-version output root must be an existing non-symlink directory")
    return root


def _strict_json(raw: bytes, *, name: str) -> object:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{name} is not UTF-8") from exc

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} has a duplicate JSON key")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"{name} has a non-standard JSON constant: {value}")

    try:
        return json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} is malformed JSON") from exc


def _require_request_matches_preparation(
    preparation: TargetVersionPreparation,
    request: TargetVersionEvidenceWriteRequest,
) -> None:
    if not isinstance(preparation, TargetVersionPreparation):
        raise TypeError("target-version writer requires TargetVersionPreparation")
    if not isinstance(request, TargetVersionEvidenceWriteRequest):
        raise TypeError("target-version writer requires TargetVersionEvidenceWriteRequest")
    bindings = (
        ("source digest", request.source_digest, preparation.source_digest),
        (
            "source snapshot SHA-256",
            request.source_snapshot_sha256,
            preparation.source_snapshot_sha256,
        ),
        (
            "requirements lock SHA-256",
            request.requirements_lock_sha256,
            preparation.requirements_lock_sha256,
        ),
        (
            "target revalidation SHA-256",
            request.target_revalidation_sha256,
            preparation.target_revalidation_sha256,
        ),
        (
            "environment audit SHA-256",
            request.environment_audit_sha256,
            preparation.environment_audit_sha256,
        ),
        ("receipt digest", request.receipt_digest, preparation.receipt.digest),
    )
    for name, actual, expected in bindings:
        if actual != expected:
            raise ValueError(f"target-version writer {name} mismatch")
    rebuilt = build_target_version_receipt(
        _observation(
            source_digest=preparation.source_digest,
            materials=preparation.materials,
            environment=preparation.environment,
            python_runtime=preparation.python_runtime,
        )
    )
    if rebuilt != preparation.receipt or rebuilt.digest != request.receipt_digest:
        raise ValueError("target-version writer receipt cannot be rebuilt from raw material")
    if not preparation.receipt.all_match_latest:
        raise ValueError("target-version writer requires all targets to match latest")


def _material_paths(material: TargetVersionRawMaterial) -> tuple[str, str]:
    if material.distribution_name not in _EXPECTED_PUBLIC_RAW_SHA256:
        raise ValueError("target-version writer has an unsupported distribution")
    raw_root = f"raw/target_version_replay/{material.distribution_name}"
    return f"{raw_root}/installed-METADATA", f"{raw_root}/pypi.json"


def _write_new_bytes(path: Path, raw: bytes) -> None:
    if not isinstance(raw, bytes):
        raise TypeError("target-version writer can only write immutable bytes")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink():
        raise ValueError("target-version writer cannot write through a symlink")
    with path.open("xb") as handle:
        handle.write(raw)


def _expected_manifest(
    *,
    preparation: TargetVersionPreparation,
    receipt_sha256: str,
) -> dict[str, object]:
    package_by_distribution = {
        item.distribution_name: item for item in preparation.receipt.packages
    }
    if tuple(package_by_distribution) != tuple(
        item.distribution_name for item in preparation.receipt.packages
    ):
        raise ValueError("target-version writer receipt distributions must be unique")
    materials: list[dict[str, object]] = []
    for material in preparation.materials:
        installed_path, public_path = _material_paths(material)
        package = package_by_distribution.get(material.distribution_name)
        if package is None or package.import_name != material.import_name:
            raise ValueError("target-version writer receipt/material identity mismatch")
        expected_public_sha = _EXPECTED_PUBLIC_RAW_SHA256[material.distribution_name]
        if _sha256(material.public_version_source) != expected_public_sha:
            raise ValueError(
                "target-version writer public source SHA mismatch: "
                + material.distribution_name
            )
        materials.append(
            {
                "distribution_name": material.distribution_name,
                "import_name": material.import_name,
                "package_id": package.package_id,
                "installed_metadata_path": installed_path,
                "installed_metadata_sha256": _sha256(material.installed_metadata),
                "public_version_source_path": public_path,
                "public_version_source_sha256": expected_public_sha,
            }
        )
    return {
        "schema_version": FORMAL_TARGET_VERSION_EVIDENCE_MANIFEST_SCHEMA_VERSION,
        "mode": FORMAL_TARGET_VERSION_EVIDENCE_WRITER_MODE,
        "source_digest": preparation.source_digest,
        "source_snapshot_sha256": preparation.source_snapshot_sha256,
        "requirements_lock_sha256": preparation.requirements_lock_sha256,
        "target_revalidation_sha256": preparation.target_revalidation_sha256,
        "environment_audit_sha256": preparation.environment_audit_sha256,
        "receipt": {
            "envelope_path": "receipts/target_version_replay.json",
            "envelope_sha256": receipt_sha256,
            "receipt_digest": preparation.receipt.digest,
            "envelope_type": "TargetVersionReceipt",
            "envelope_schema_version": preparation.receipt.schema_version,
        },
        "materials": materials,
        "all_targets_match_latest": True,
        "authority_eligible": False,
        "formal_evidence_created": False,
        "gate_credit": False,
        "candidate_confirmed": False,
        "bug_claimed": False,
    }


def _expected_file_paths(preparation: TargetVersionPreparation) -> tuple[str, ...]:
    paths = {
        "raw/target_version_replay/manifest.json",
        "receipts/target_version_replay.json",
    }
    for material in preparation.materials:
        paths.update(_material_paths(material))
    return tuple(sorted(paths))


def _write_stage(
    root: Path,
    *,
    preparation: TargetVersionPreparation,
) -> None:
    if any(root.iterdir()):
        raise ValueError("target-version staging root must be empty")
    receipt_text = canonical_envelope(
        "TargetVersionReceipt",
        preparation.receipt.schema_version,
        preparation.receipt,
    )
    receipt_raw = receipt_text.encode("utf-8")
    receipt_path = root / "receipts" / "target_version_replay.json"
    _write_new_bytes(receipt_path, receipt_raw)
    for material in preparation.materials:
        installed_path, public_path = _material_paths(material)
        _write_new_bytes(root / installed_path, material.installed_metadata)
        _write_new_bytes(root / public_path, material.public_version_source)
    manifest = _expected_manifest(
        preparation=preparation,
        receipt_sha256=_sha256(receipt_raw),
    )
    _write_new_bytes(
        root / "raw/target_version_replay/manifest.json",
        canonical_json(manifest).encode("utf-8"),
    )


def _result_for_existing_output(
    root: Path,
    *,
    preparation: TargetVersionPreparation,
) -> TargetVersionEvidenceWriteResult:
    manifest_path = root / "raw/target_version_replay/manifest.json"
    receipt_path = root / "receipts/target_version_replay.json"
    return TargetVersionEvidenceWriteResult(
        output_root=str(root),
        raw_manifest_path=str(manifest_path),
        receipt_path=str(receipt_path),
        raw_manifest_sha256=_sha256(manifest_path.read_bytes()),
        receipt_sha256=_sha256(receipt_path.read_bytes()),
        source_digest=preparation.source_digest,
        source_snapshot_sha256=preparation.source_snapshot_sha256,
        receipt_digest=preparation.receipt.digest,
    )


def verify_target_version_replay_evidence(
    *,
    preparation: TargetVersionPreparation,
    request: TargetVersionEvidenceWriteRequest,
) -> TargetVersionEvidenceWriteResult:
    """Re-read a private staged tree without granting it formal authority."""

    _require_request_matches_preparation(preparation, request)
    root = _existing_output_root(request)
    found_paths = tuple(
        sorted(
            item.relative_to(root).as_posix()
            for item in root.rglob("*")
            if item.is_file()
        )
    )
    if found_paths != _expected_file_paths(preparation):
        raise ValueError("target-version evidence file inventory mismatch")
    if any(item.is_symlink() for item in root.rglob("*")):
        raise ValueError("target-version evidence cannot contain a symlink")

    receipt_path = root / "receipts/target_version_replay.json"
    receipt_raw = receipt_path.read_bytes()
    receipt_sha256 = _sha256(receipt_raw)
    try:
        receipt_text = receipt_raw.decode("utf-8")
        envelope = decode_canonical_envelope(receipt_text)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("target-version evidence receipt envelope is invalid") from exc
    if (
        envelope["type"] != "TargetVersionReceipt"
        or envelope["schema_version"] != preparation.receipt.schema_version
    ):
        raise ValueError("target-version evidence receipt envelope type mismatch")
    if receipt_text != canonical_envelope(
        "TargetVersionReceipt",
        preparation.receipt.schema_version,
        preparation.receipt,
    ):
        raise ValueError("target-version evidence receipt bytes mismatch")
    replay_errors = replay_runtime_admission(
        envelope_type="TargetVersionReceipt",
        schema_version=preparation.receipt.schema_version,
        payload=envelope["payload"],
        subject_kind="target_packages",
        subject_ids=preparation.receipt.package_ids,
    )
    if replay_errors:
        raise ValueError(
            "target-version evidence semantic replay failed: " + ",".join(replay_errors)
        )

    expected_manifest = _expected_manifest(
        preparation=preparation,
        receipt_sha256=receipt_sha256,
    )
    manifest_path = root / "raw/target_version_replay/manifest.json"
    manifest_raw = manifest_path.read_bytes()
    manifest_value = _strict_json(manifest_raw, name="target-version evidence manifest")
    if manifest_value != expected_manifest or manifest_raw != canonical_json(
        expected_manifest
    ).encode("utf-8"):
        raise ValueError("target-version evidence manifest is not exact canonical binding")
    for material in preparation.materials:
        installed_path, public_path = _material_paths(material)
        if (root / installed_path).read_bytes() != material.installed_metadata:
            raise ValueError("target-version evidence installed metadata bytes mismatch")
        if (root / public_path).read_bytes() != material.public_version_source:
            raise ValueError("target-version evidence public source bytes mismatch")
    return _result_for_existing_output(root, preparation=preparation)


def write_target_version_replay_evidence(
    *,
    preparation: TargetVersionPreparation,
    request: TargetVersionEvidenceWriteRequest,
) -> TargetVersionEvidenceWriteResult:
    """Stage and publish a test-only tree after all exact bindings validate."""

    _require_request_matches_preparation(preparation, request)
    output_root = _new_output_root(request)
    stage_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.stage-", dir=output_root.parent)
    )
    try:
        _write_stage(stage_root, preparation=preparation)
        verify_target_version_replay_evidence(
            preparation=preparation,
            request=replace(request, output_root=str(stage_root)),
        )
        if os.path.lexists(output_root):
            raise ValueError("target-version output root appeared during staging")
        os.replace(stage_root, output_root)
        stage_root = None  # ownership moved to output_root after successful publish
    finally:
        if stage_root is not None and os.path.lexists(stage_root):
            shutil.rmtree(stage_root)
    return verify_target_version_replay_evidence(
        preparation=preparation,
        request=request,
    )


def _read_exact_regular_file(
    path_text: str,
    *,
    name: str,
    expected_sha256: str,
) -> bytes:
    path = Path(_require_absolute_path_text(path_text, name=name))
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be an existing non-symlink regular file")
    raw = path.read_bytes()
    if _sha256(raw) != expected_sha256:
        raise ValueError(f"{name} SHA-256 mismatch")
    return raw


def _load_exact_canonical_json(
    path_text: str,
    *,
    name: str,
    expected_sha256: str,
) -> dict[str, object]:
    raw = _read_exact_regular_file(
        path_text,
        name=name,
        expected_sha256=expected_sha256,
    )
    value = _strict_json(raw, name=name)
    if not isinstance(value, dict) or raw != canonical_json(value).encode("utf-8"):
        raise ValueError(f"{name} must be exact canonical JSON object bytes")
    return value


def _require_mapping_field(
    value: dict[str, object],
    field: str,
    *,
    name: str,
) -> object:
    if field not in value:
        raise ValueError(f"{name} missing required field: {field}")
    return value[field]


def _verify_source_snapshot_binding(
    authority: TargetVersionFutureExecutionAuthority,
) -> None:
    snapshot = _load_exact_canonical_json(
        authority.source_snapshot_path,
        name="target-version current source snapshot",
        expected_sha256=authority.source_snapshot_sha256,
    )
    if (
        snapshot.get("schema_version") != "osc-root-source-snapshot-v1"
        or snapshot.get("source_digest") != authority.source_digest
        or snapshot.get("git_head") != authority.source_git_head
        or snapshot.get("dependency_lock_sha256") != authority.requirements_lock_sha256
    ):
        raise ValueError("target-version current source snapshot binding mismatch")
    files = _require_mapping_field(snapshot, "files", name="target-version current source snapshot")
    if not isinstance(files, list):
        raise ValueError("target-version current source snapshot files must be a list")
    paths: set[str] = set()
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("target-version current source snapshot file entry is invalid")
        path = item.get("path")
        sha256 = item.get("sha256")
        if not isinstance(path, str) or not isinstance(sha256, str):
            raise ValueError("target-version current source snapshot file entry lacks path/SHA")
        _require_sha256(sha256, name="target-version current source snapshot file SHA")
        paths.add(path)
    required_paths = {
        "pyproject.toml",
        "requirements-final.lock",
        "scripts/osc/run_phase6_formal_target_version_producer.py",
        "src/datadiff_osc/runtime/_phase6_formal_target_version_producer.py",
        "src/datadiff_osc/public_api_freeze.json",
    }
    if not required_paths.issubset(paths):
        raise ValueError("target-version current source snapshot required path omission")


def _verify_target_environment_binding(
    authority: TargetVersionFutureExecutionAuthority,
) -> tuple[dict[str, object], dict[str, object]]:
    lock_raw = _read_exact_regular_file(
        authority.requirements_lock_path,
        name="target-version requirements lock",
        expected_sha256=authority.requirements_lock_sha256,
    )
    if not lock_raw:
        raise ValueError("target-version requirements lock cannot be empty")
    revalidation = _load_exact_canonical_json(
        authority.target_revalidation_path,
        name="target-version revalidation",
        expected_sha256=authority.target_revalidation_sha256,
    )
    audit = _load_exact_canonical_json(
        authority.environment_audit_path,
        name="target-version environment audit",
        expected_sha256=authority.environment_audit_sha256,
    )
    if revalidation.get("schema_version") != "osc-phase6-latest-target-revalidation-r1b":
        raise ValueError("target-version revalidation schema mismatch")
    if audit.get("schema_version") != "osc-phase6-latest-target-environment-audit-r1b":
        raise ValueError("target-version environment audit schema mismatch")
    if revalidation.get("all_targets_match_latest") is not True:
        raise ValueError("target-version revalidation is not all-match-latest")
    for document, name in (
        (revalidation, "target-version revalidation"),
        (audit, "target-version environment audit"),
    ):
        for field in (
            "formal_evidence_created",
            "gate_credit",
            "candidate_confirmed",
            "bug_claimed",
        ):
            if document.get(field) is not False:
                raise ValueError(f"{name} no-authority field mismatch: {field}")
    audit_source = _require_mapping_field(
        audit, "source_snapshot", name="target-version environment audit"
    )
    revalidation_source = _require_mapping_field(
        revalidation, "source_snapshot", name="target-version revalidation"
    )
    if not isinstance(audit_source, dict) or not isinstance(revalidation_source, dict):
        raise ValueError("target-version historical source provenance is invalid")
    for source, name in (
        (audit_source, "target-version environment audit historical source"),
        (revalidation_source, "target-version revalidation historical source"),
    ):
        if (
            source.get("byte_sha256")
            != authority.historical_environment_source_snapshot_sha256
            or source.get("source_digest")
            != authority.historical_environment_source_digest
        ):
            raise ValueError(f"{name} mismatch")
    audit_lock = _require_mapping_field(
        audit, "source_snapshot", name="target-version environment audit"
    )
    if not isinstance(audit_lock, dict) or audit_lock.get(
        "requirements_lock_sha256"
    ) != authority.requirements_lock_sha256:
        raise ValueError("target-version environment audit lock binding mismatch")
    revalidation_audit = _require_mapping_field(
        revalidation, "environment_audit", name="target-version revalidation"
    )
    if not isinstance(revalidation_audit, dict) or (
        revalidation_audit.get("byte_sha256") != authority.environment_audit_sha256
        or revalidation_audit.get("path") != authority.environment_audit_path
        or revalidation_audit.get("requirements_lock_sha256")
        != authority.requirements_lock_sha256
    ):
        raise ValueError("target-version revalidation/environment audit binding mismatch")
    return revalidation, audit


def _verify_target_interpreter_binding(
    authority: TargetVersionFutureExecutionAuthority,
    audit: dict[str, object],
) -> Path:
    interpreter = Path(
        _require_absolute_path_text(
            authority.target_interpreter, name="target-version target interpreter"
        )
    )
    if not interpreter.is_file():
        raise ValueError("target-version target interpreter must be an existing file")
    if _sha256(interpreter.read_bytes()) != authority.target_interpreter_sha256:
        raise ValueError("target-version target interpreter SHA-256 mismatch")
    audit_interpreter = _require_mapping_field(
        audit, "interpreter", name="target-version environment audit"
    )
    if not isinstance(audit_interpreter, dict) or audit_interpreter.get(
        "executable"
    ) != authority.target_interpreter:
        raise ValueError("target-version environment audit interpreter mismatch")
    venv_root = interpreter.parent.parent
    if not venv_root.is_dir() or venv_root.is_symlink():
        raise ValueError("target-version target venv root must be a non-symlink directory")
    return venv_root.resolve(strict=True)


def _verify_public_source_bindings(
    authority: TargetVersionFutureExecutionAuthority,
) -> dict[str, bytes]:
    root = Path(
        _require_absolute_path_text(
            authority.public_source_root, name="target-version public source root"
        )
    )
    if root.is_symlink() or not root.is_dir():
        raise ValueError("target-version public source root must be a non-symlink directory")
    sources: dict[str, bytes] = {}
    for binding in authority.public_sources:
        path = Path(binding.path)
        if path.parent != root:
            raise ValueError("target-version public source binding escapes its root")
        raw = _read_exact_regular_file(
            binding.path,
            name=f"target-version public source {binding.distribution_name}",
            expected_sha256=binding.sha256,
        )
        if _public_version(raw) == "":
            raise ValueError("target-version public source version cannot be empty")
        sources[binding.distribution_name] = raw
    return sources


def _verify_revalidation_records(
    authority: TargetVersionFutureExecutionAuthority,
    revalidation: dict[str, object],
    audit: dict[str, object],
    records: tuple[TargetVersionInstalledMetadataRecord, ...],
    public_sources: dict[str, bytes],
) -> None:
    revalidation_records = _require_mapping_field(
        revalidation, "records", name="target-version revalidation"
    )
    if not isinstance(revalidation_records, list) or len(revalidation_records) != len(
        _EXPECTED_TARGETS
    ):
        raise ValueError("target-version revalidation records are incomplete")
    package_entries = _require_mapping_field(
        audit, "packages", name="target-version environment audit"
    )
    if not isinstance(package_entries, list):
        raise ValueError("target-version environment audit packages are invalid")
    for expected, record, revalidated in zip(
        _EXPECTED_TARGETS, records, revalidation_records, strict=True
    ):
        distribution_name, import_name = expected
        if not isinstance(revalidated, dict):
            raise ValueError("target-version revalidation record is invalid")
        installed_version = _metadata_version(record.installed_metadata)
        latest_version = _public_version(public_sources[distribution_name])
        if (
            record.distribution_name != distribution_name
            or record.import_name != import_name
            or revalidated.get("distribution") != distribution_name
            or revalidated.get("isolated_installed_version") != installed_version
            or revalidated.get("latest_version") != latest_version
            or revalidated.get("matches_latest") is not True
            or revalidated.get("public_raw_sha256")
            != _EXPECTED_PUBLIC_RAW_SHA256[distribution_name]
            or revalidated.get("public_raw_path")
            != str(Path(authority.public_source_root) / f"pypi-{distribution_name}.json")
        ):
            raise ValueError("target-version revalidation record binding mismatch")
        matching_packages = [
            item
            for item in package_entries
            if isinstance(item, dict) and item.get("distribution") == distribution_name
        ]
        if len(matching_packages) != 1 or matching_packages[0].get(
            "installed_version"
        ) != installed_version:
            raise ValueError("target-version environment package binding mismatch")


def _execution_root_for_capability(
    authority: TargetVersionFutureExecutionAuthority,
) -> Path:
    root = Path(_require_absolute_path_text(authority.execution_root, name="execution root"))
    output_root = Path(
        _require_absolute_path_text(authority.future_output_root, name="future output root")
    )
    _require_no_symlink_components(root)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("execution root must be an existing non-symlink directory")
    if output_root.parent != root or os.path.lexists(output_root):
        raise ValueError("future formal output root must be an absent direct child")
    return root


def _run_target_metadata_probe(
    authority: TargetVersionFutureExecutionAuthority,
    *,
    execution_root: Path,
) -> bytes:
    command = [authority.target_interpreter, "-I", "-c", _TARGET_METADATA_PROBE_CODE]
    try:
        completed = subprocess.run(
            command,
            cwd=str(execution_root),
            env={"PYTHONDONTWRITEBYTECODE": "1"},
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("target-version metadata probe failed before collection") from exc
    if completed.returncode != 0:
        raise ValueError("target-version metadata probe returned non-zero")
    if not isinstance(completed.stdout, bytes) or not isinstance(completed.stderr, bytes):
        raise ValueError("target-version metadata probe returned non-byte streams")
    if completed.stderr:
        raise ValueError("target-version metadata probe wrote unexpected stderr")
    if not completed.stdout:
        raise ValueError("target-version metadata probe returned empty stdout")
    return completed.stdout


def _decode_target_metadata_probe(
    raw: bytes,
    *,
    authority: TargetVersionFutureExecutionAuthority,
    venv_root: Path,
    audit: dict[str, object],
) -> tuple[TargetVersionInstalledMetadataRecord, ...]:
    value = _strict_json(raw, name="target-version metadata probe transcript")
    if not isinstance(value, dict) or raw != canonical_json(value).encode("utf-8"):
        raise ValueError("target-version metadata probe transcript is not canonical")
    if set(value) != {
        "implementation",
        "interpreter",
        "records",
        "schema_version",
        "version",
    }:
        raise ValueError("target-version metadata probe transcript field set mismatch")
    audit_interpreter = _require_mapping_field(
        audit, "interpreter", name="target-version environment audit"
    )
    if not isinstance(audit_interpreter, dict) or (
        value.get("schema_version") != FORMAL_TARGET_VERSION_METADATA_PROBE_SCHEMA_VERSION
        or value.get("interpreter") != authority.target_interpreter
        or value.get("implementation") != audit_interpreter.get("implementation")
        or value.get("version") != audit_interpreter.get("version")
    ):
        raise ValueError("target-version metadata probe interpreter binding mismatch")
    raw_records = value.get("records")
    if not isinstance(raw_records, list) or len(raw_records) != len(_EXPECTED_TARGETS):
        raise ValueError("target-version metadata probe records are incomplete")
    records: list[TargetVersionInstalledMetadataRecord] = []
    for expected, raw_record in zip(_EXPECTED_TARGETS, raw_records, strict=True):
        distribution_name, import_name = expected
        if not isinstance(raw_record, dict) or set(raw_record) != {
            "distribution_name",
            "import_name",
            "metadata_base64",
            "metadata_path",
            "metadata_sha256",
        }:
            raise ValueError("target-version metadata probe record field set mismatch")
        if (
            raw_record.get("distribution_name") != distribution_name
            or raw_record.get("import_name") != import_name
        ):
            raise ValueError("target-version metadata probe identity mismatch")
        path_text = _require_absolute_path_text(
            raw_record.get("metadata_path"), name="target-version probe metadata path"
        )
        metadata_path = Path(path_text)
        try:
            resolved_metadata_path = metadata_path.resolve(strict=True)
        except OSError as exc:
            raise ValueError("target-version probe metadata path cannot resolve") from exc
        try:
            resolved_metadata_path.relative_to(venv_root)
        except ValueError as exc:
            raise ValueError("target-version probe metadata path escapes target venv") from exc
        if not resolved_metadata_path.is_file():
            raise ValueError("target-version probe metadata path is not a file")
        encoded = raw_record.get("metadata_base64")
        if not isinstance(encoded, str):
            raise ValueError("target-version probe metadata base64 is invalid")
        try:
            metadata = base64.b64decode(encoded.encode("ascii"), validate=True)
        except (UnicodeEncodeError, ValueError) as exc:
            raise ValueError("target-version probe metadata base64 is invalid") from exc
        sha256 = raw_record.get("metadata_sha256")
        _require_sha256(sha256, name="target-version probe metadata SHA-256")
        if (
            not metadata
            or _sha256(metadata) != sha256
            or resolved_metadata_path.read_bytes() != metadata
        ):
            raise ValueError("target-version probe metadata bytes mismatch")
        records.append(
            TargetVersionInstalledMetadataRecord(
                distribution_name=distribution_name,
                import_name=import_name,
                metadata_path=str(resolved_metadata_path),
                metadata_sha256=sha256,
                installed_metadata=metadata,
            )
        )
    return _require_metadata_records(tuple(records))


def require_formal_target_version_execution_authority(
    authority: TargetVersionFutureExecutionAuthority,
) -> None:
    """Fail closed: this implementation can never grant a durable execution."""

    if not isinstance(authority, TargetVersionFutureExecutionAuthority):
        raise TypeError("formal target-version execution requires an authority input")
    raise PermissionError(
        "formal target-version execution requires a separate successor authority"
    )


def inspect_target_version_execution_capability(
    *,
    authority: TargetVersionFutureExecutionAuthority,
) -> TargetVersionExecutionCapability:
    """Collect the selected target venv only after all no-write bindings hold."""

    if not isinstance(authority, TargetVersionFutureExecutionAuthority):
        raise TypeError("target-version execution capability requires an authority input")
    if authority.execution_command_sha256 != authority.expected_execution_command_sha256:
        raise ValueError("target-version future execution command digest mismatch")
    _verify_source_snapshot_binding(authority)
    revalidation, audit = _verify_target_environment_binding(authority)
    venv_root = _verify_target_interpreter_binding(authority, audit)
    public_sources = _verify_public_source_bindings(authority)
    execution_root = _execution_root_for_capability(authority)
    transcript = _run_target_metadata_probe(authority, execution_root=execution_root)
    records = _decode_target_metadata_probe(
        transcript,
        authority=authority,
        venv_root=venv_root,
        audit=audit,
    )
    _verify_revalidation_records(authority, revalidation, audit, records, public_sources)
    materials = tuple(
        TargetVersionRawMaterial(
            distribution_name=record.distribution_name,
            import_name=record.import_name,
            installed_metadata=record.installed_metadata,
            public_version_source=public_sources[record.distribution_name],
        )
        for record in records
    )
    audit_interpreter = _require_mapping_field(
        audit, "interpreter", name="target-version environment audit"
    )
    if not isinstance(audit_interpreter, dict):
        raise ValueError("target-version environment audit interpreter is invalid")
    preparation = prepare_target_version_replay(
        source_digest=authority.source_digest,
        source_snapshot_sha256=authority.source_snapshot_sha256,
        requirements_lock_sha256=authority.requirements_lock_sha256,
        target_revalidation_sha256=authority.target_revalidation_sha256,
        environment_audit_sha256=authority.environment_audit_sha256,
        materials=materials,
        environment=(
            ("environment_audit_sha256", authority.environment_audit_sha256),
            ("target_interpreter", authority.target_interpreter),
        ),
        python_runtime=(
            ("implementation", _require_text(audit_interpreter.get("implementation"), name="target implementation")),
            ("version", _require_text(audit_interpreter.get("version"), name="target Python version")),
        ),
    )
    return TargetVersionExecutionCapability(
        authority_id=authority.authority_id,
        source_digest=authority.source_digest,
        source_snapshot_sha256=authority.source_snapshot_sha256,
        target_interpreter=authority.target_interpreter,
        future_output_root=authority.future_output_root,
        metadata_probe_transcript_sha256=_sha256(transcript),
        metadata_records=records,
        preparation=preparation,
    )


def target_version_producer_preview() -> dict[str, object]:
    """Return the fixed declaration for the no-output implementation boundary."""

    return {
        "schema_version": FORMAL_TARGET_VERSION_PREPARATION_SCHEMA_VERSION,
        "mode": FORMAL_TARGET_VERSION_PREPARATION_MODE,
        "target_count": len(_EXPECTED_TARGETS),
        "launches_adapter": False,
        "launches_subprocess": False,
        "makes_network_request": False,
        "writes_durable_output": False,
        "test_only_staged_writer_available": True,
        "test_only_execution_capability_available": True,
        "formal_execution_authorized": False,
        "formal_evidence_created": False,
        "authority_eligible": False,
        "gate_credit": False,
        "candidate_confirmed": False,
        "bug_claimed": False,
    }


__all__ = [
    "FORMAL_TARGET_VERSION_EXECUTION_CAPABILITY_MODE",
    "FORMAL_TARGET_VERSION_EXECUTION_CAPABILITY_SCHEMA_VERSION",
    "FORMAL_TARGET_VERSION_EVIDENCE_MANIFEST_SCHEMA_VERSION",
    "FORMAL_TARGET_VERSION_EVIDENCE_WRITER_MODE",
    "FORMAL_TARGET_VERSION_METADATA_PROBE_SCHEMA_VERSION",
    "FORMAL_TARGET_VERSION_PREPARATION_MODE",
    "FORMAL_TARGET_VERSION_PREPARATION_SCHEMA_VERSION",
    "TargetVersionExecutionCapability",
    "TargetVersionEvidenceWriteRequest",
    "TargetVersionEvidenceWriteResult",
    "TargetVersionFutureExecutionAuthority",
    "TargetVersionInstalledMetadataRecord",
    "TargetVersionPreparation",
    "TargetVersionPublicSourceBinding",
    "TargetVersionRawMaterial",
    "inspect_target_version_execution_capability",
    "prepare_target_version_replay",
    "require_formal_target_version_execution_authority",
    "target_version_producer_preview",
    "verify_target_version_replay_evidence",
    "write_target_version_replay_evidence",
]
