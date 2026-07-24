"""Private, preparation-only binding for the Phase-6 target-version stream.

The module deliberately has no persistent-output or execution entry point.
It turns immutable installed METADATA bytes and the frozen public PyPI response
bytes into the already-frozen :class:`TargetVersionReceipt` type, then exposes
only a canonical preparation payload.  A later, separately authorized task
must bind a fresh source snapshot, an exact output path, and a command digest
before it may persist a formal ``target_version_replay`` record.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
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
_SHA256_RE = re.compile(r"[0-9a-f]{64}")

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
        "formal_evidence_created": False,
        "authority_eligible": False,
        "gate_credit": False,
        "candidate_confirmed": False,
        "bug_claimed": False,
    }


__all__ = [
    "FORMAL_TARGET_VERSION_EVIDENCE_MANIFEST_SCHEMA_VERSION",
    "FORMAL_TARGET_VERSION_EVIDENCE_WRITER_MODE",
    "FORMAL_TARGET_VERSION_PREPARATION_MODE",
    "FORMAL_TARGET_VERSION_PREPARATION_SCHEMA_VERSION",
    "TargetVersionEvidenceWriteRequest",
    "TargetVersionEvidenceWriteResult",
    "TargetVersionPreparation",
    "TargetVersionRawMaterial",
    "prepare_target_version_replay",
    "target_version_producer_preview",
    "verify_target_version_replay_evidence",
    "write_target_version_replay_evidence",
]
