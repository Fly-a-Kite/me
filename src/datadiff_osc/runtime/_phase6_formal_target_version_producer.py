"""Private, preparation-only binding for the Phase-6 target-version stream.

The module deliberately has no persistent-output or execution entry point.
It turns immutable installed METADATA bytes and the frozen public PyPI response
bytes into the already-frozen :class:`TargetVersionReceipt` type, then exposes
only a canonical preparation payload.  A later, separately authorized task
must bind a fresh source snapshot, an exact output path, and a command digest
before it may persist a formal ``target_version_replay`` record.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re

from datadiff.target_version_audit import DEFAULT_TARGET_PACKAGES
from datadiff_osc._canonical import (
    assert_deeply_immutable,
    canonical_json,
    stable_digest,
)
from datadiff_osc.runtime._private_receipts import TargetVersionReceipt
from datadiff_osc.runtime._receipt_producers import (
    TargetPackageObservation,
    TargetVersionObservation,
    build_target_version_receipt,
)


FORMAL_TARGET_VERSION_PREPARATION_SCHEMA_VERSION = (
    "osc-private-phase6-formal-target-version-preparation-v1"
)
FORMAL_TARGET_VERSION_PREPARATION_MODE = "prepare-only-no-authority-v1"
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
        "formal_evidence_created": False,
        "authority_eligible": False,
        "gate_credit": False,
        "candidate_confirmed": False,
        "bug_claimed": False,
    }


__all__ = [
    "FORMAL_TARGET_VERSION_PREPARATION_MODE",
    "FORMAL_TARGET_VERSION_PREPARATION_SCHEMA_VERSION",
    "TargetVersionPreparation",
    "TargetVersionRawMaterial",
    "prepare_target_version_replay",
    "target_version_producer_preview",
]
