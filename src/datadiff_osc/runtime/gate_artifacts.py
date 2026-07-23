"""Externally anchored raw artifacts for pre-24h gate recomputation.

Runtime is only a verifier and consumer.  A Root-owned producer independently
derives the frozen source binding, denominator universes, provenance, and raw
artifact hashes, then publishes an authority plan whose SHA-256 is supplied to
this module out of band.  Nothing in the raw manifest can create authority for
itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from datadiff_osc._canonical import stable_digest, to_primitive


RAW_GATE_ARTIFACT_SCHEMA_VERSION = "osc-gate-raw-artifact-v1"
GATE_ARTIFACT_MANIFEST_SCHEMA_VERSION = "osc-gate-artifact-manifest-v1"
GATE_AUTHORITY_PLAN_SCHEMA_VERSION = "osc-gate-authority-plan-v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class MetricReducer(str, Enum):
    COUNT_UNIQUE = "count_unique"
    SINGLE_VALUE = "single_value"
    NEAREST_RANK_P95 = "nearest_rank_p95"
    MAXIMUM = "maximum"
    MINIMUM = "minimum"


def identity_universe_digest(record_ids: Iterable[str]) -> str:
    """Commit to an exact identity set, not merely its cardinality."""

    identities = tuple(record_ids)
    if any(not isinstance(item, str) or not item for item in identities):
        raise ValueError("identity universe values must be non-empty strings")
    if len(identities) != len(set(identities)):
        raise ValueError("identity universe values must be unique")
    return stable_digest("osc-gate-count-identity-universe", tuple(sorted(identities)))


@dataclass(frozen=True, slots=True)
class TrustedArtifactBinding:
    artifact_id: str
    byte_sha256: str
    provenance_digest: str
    schema_version: str = RAW_GATE_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.artifact_id or not self.provenance_digest:
            raise ValueError("trusted artifact bindings must be non-empty")
        if _SHA256_RE.fullmatch(self.byte_sha256) is None:
            raise ValueError("trusted artifact binding requires a SHA-256")
        if self.schema_version != RAW_GATE_ARTIFACT_SCHEMA_VERSION:
            raise ValueError("trusted artifact schema mismatch")


@dataclass(frozen=True, slots=True)
class TrustedIdentityUniverse:
    metric_key: str
    cardinality: int
    identity_digest: str
    provenance_digest: str
    schema_version: str = "osc-gate-identity-universe-v1"

    def __post_init__(self) -> None:
        if not self.metric_key or not self.identity_digest or not self.provenance_digest:
            raise ValueError("trusted identity universe bindings must be non-empty")
        if isinstance(self.cardinality, bool) or not isinstance(self.cardinality, int):
            raise ValueError("trusted identity universe cardinality must be an integer")
        if self.cardinality < 0:
            raise ValueError("trusted identity universe cardinality must be non-negative")

    def matches(self, record_ids: Iterable[str]) -> bool:
        identities = tuple(record_ids)
        return (
            len(identities) == self.cardinality
            and identity_universe_digest(identities) == self.identity_digest
        )


@dataclass(frozen=True, slots=True)
class VerifiedGateAuthorityPlan:
    expected_source_digest: str
    expected_plan_sha256: str
    plan_path: str
    byte_sha256: str
    source_digest: str
    gate_spec_set_digest: str
    artifact_bindings: tuple[TrustedArtifactBinding, ...]
    denominator_universes: tuple[TrustedIdentityUniverse, ...]
    verification_errors: tuple[str, ...] = ()
    schema_version: str = GATE_AUTHORITY_PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for value in (
            self.expected_source_digest,
            self.plan_path,
            self.source_digest,
            self.gate_spec_set_digest,
        ):
            if not isinstance(value, str) or not value:
                raise ValueError("gate authority plan bindings must be non-empty")
        for digest in (self.expected_plan_sha256, self.byte_sha256):
            if _SHA256_RE.fullmatch(digest) is None:
                raise ValueError("gate authority plan requires SHA-256 bindings")
        if self.schema_version != GATE_AUTHORITY_PLAN_SCHEMA_VERSION:
            raise ValueError("gate authority plan schema mismatch")
        artifact_ids = tuple(item.artifact_id for item in self.artifact_bindings)
        if artifact_ids != tuple(sorted(artifact_ids)) or len(artifact_ids) != len(
            set(artifact_ids)
        ):
            raise ValueError("trusted artifact bindings must be uniquely sorted")
        universe_keys = tuple(item.metric_key for item in self.denominator_universes)
        if universe_keys != tuple(sorted(universe_keys)) or len(universe_keys) != len(
            set(universe_keys)
        ):
            raise ValueError("trusted denominator universes must be uniquely sorted")

    @property
    def valid(self) -> bool:
        if self.verification_errors:
            return False
        reverified = verify_gate_authority_plan(
            self.plan_path,
            expected_sha256=self.expected_plan_sha256,
            expected_source_digest=self.expected_source_digest,
        )
        return reverified == self

    @property
    def digest(self) -> str:
        return stable_digest("osc-verified-gate-authority-plan", self)

    def artifact_map(self) -> dict[str, TrustedArtifactBinding]:
        return {item.artifact_id: item for item in self.artifact_bindings}

    def universe_map(self) -> dict[str, TrustedIdentityUniverse]:
        return {item.metric_key: item for item in self.denominator_universes}

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


@dataclass(frozen=True, slots=True)
class RawMetricRecord:
    record_id: str
    value: int | float

    def __post_init__(self) -> None:
        if not isinstance(self.record_id, str) or not self.record_id:
            raise ValueError("raw metric record requires a non-empty identity")
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise ValueError("raw metric record value must be numeric")
        try:
            finite_value = float(self.value)
        except OverflowError as exc:
            raise ValueError("raw metric record value must be finite") from exc
        if not math.isfinite(finite_value):
            raise ValueError("raw metric record value must be finite")


@dataclass(frozen=True, slots=True)
class RawMetricSeries:
    metric_key: str
    reducer: MetricReducer
    records: tuple[RawMetricRecord, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.metric_key, str) or not self.metric_key:
            raise ValueError("raw metric series requires a metric key")
        identities = tuple(item.record_id for item in self.records)
        if len(identities) != len(set(identities)):
            raise ValueError("raw metric record identities must be unique per series")
        if self.reducer is MetricReducer.COUNT_UNIQUE and any(
            float(item.value) != 1.0 for item in self.records
        ):
            raise ValueError("count_unique records must each have value 1")


@dataclass(frozen=True, slots=True)
class VerifiedRawGateArtifact:
    artifact_id: str
    source_digest: str
    byte_sha256: str
    path: str
    series: tuple[RawMetricSeries, ...]
    schema_version: str = RAW_GATE_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.artifact_id or not self.source_digest or not self.path:
            raise ValueError("verified gate artifact bindings must be non-empty")
        if self.schema_version != RAW_GATE_ARTIFACT_SCHEMA_VERSION:
            raise ValueError("verified gate artifact schema mismatch")
        if _SHA256_RE.fullmatch(self.byte_sha256) is None:
            raise ValueError("verified gate artifact requires a SHA-256 digest")
        keys = tuple(item.metric_key for item in self.series)
        if not keys or len(keys) != len(set(keys)):
            raise ValueError("artifact metric series must be non-empty and unique")

    @property
    def digest(self) -> str:
        return stable_digest(
            "osc-verified-gate-artifact",
            {
                "artifact_id": self.artifact_id,
                "source_digest": self.source_digest,
                "byte_sha256": self.byte_sha256,
                "schema_version": self.schema_version,
            },
        )


@dataclass(frozen=True, slots=True)
class VerifiedGateEvidence:
    expected_source_digest: str
    expected_authority_plan_sha256: str
    source_digest: str
    manifest_path: str
    manifest_sha256: str
    authority_plan: VerifiedGateAuthorityPlan
    artifacts: tuple[VerifiedRawGateArtifact, ...]
    claimed_metrics: tuple[tuple[str, int | float], ...]
    verification_errors: tuple[str, ...] = ()
    schema_version: str = GATE_ARTIFACT_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.expected_source_digest or not self.source_digest or not self.manifest_path:
            raise ValueError("gate evidence requires source and manifest bindings")
        for digest in (
            self.expected_authority_plan_sha256,
            self.manifest_sha256,
        ):
            if _SHA256_RE.fullmatch(digest) is None:
                raise ValueError("gate evidence requires SHA-256 bindings")
        if self.schema_version != GATE_ARTIFACT_MANIFEST_SCHEMA_VERSION:
            raise ValueError("gate evidence manifest schema mismatch")
        if not isinstance(self.authority_plan, VerifiedGateAuthorityPlan):
            raise TypeError("gate evidence requires a verified authority plan")
        artifact_ids = tuple(item.artifact_id for item in self.artifacts)
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("gate artifact identities must be unique")
        claim_keys = tuple(key for key, _ in self.claimed_metrics)
        if claim_keys != tuple(sorted(claim_keys)) or len(claim_keys) != len(
            set(claim_keys)
        ):
            raise ValueError("claimed gate metrics must be uniquely sorted")

    @property
    def valid(self) -> bool:
        if self.verification_errors or not self.artifacts:
            return False
        authority = verify_gate_authority_plan(
            self.authority_plan.plan_path,
            expected_sha256=self.expected_authority_plan_sha256,
            expected_source_digest=self.expected_source_digest,
        )
        if authority != self.authority_plan:
            return False
        reverified = verify_gate_artifact_manifest(
            self.manifest_path,
            expected_source_digest=self.expected_source_digest,
            expected_authority_plan_sha256=self.expected_authority_plan_sha256,
            authority_plan=authority,
        )
        return reverified == self

    @property
    def digest(self) -> str:
        return stable_digest("osc-verified-gate-evidence", self)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


@dataclass(frozen=True, slots=True)
class MetricRecomputation:
    metrics: tuple[tuple[str, int | float], ...]
    record_ids: tuple[tuple[str, tuple[str, ...]], ...]
    evidence_by_metric: tuple[tuple[str, tuple[str, ...]], ...]
    errors: tuple[str, ...]

    def metric_map(self) -> dict[str, int | float]:
        return dict(self.metrics)

    def record_id_map(self) -> dict[str, tuple[str, ...]]:
        return dict(self.record_ids)

    def evidence_map(self) -> dict[str, tuple[str, ...]]:
        return dict(self.evidence_by_metric)


def _strict_json_loads(raw: bytes) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field: {key}")
            result[key] = value
        return result

    return json.loads(raw, object_pairs_hook=reject_duplicates)


def _numeric_mapping(value: Any, *, name: str) -> tuple[tuple[str, int | float], ...]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    result: list[tuple[str, int | float]] = []
    for key, item in sorted(value.items()):
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be non-empty strings")
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{name} values must be numeric")
        try:
            finite_value = float(item)
        except OverflowError as exc:
            raise ValueError(f"{name} values must be finite") from exc
        if not math.isfinite(finite_value):
            raise ValueError(f"{name} values must be finite")
        result.append((key, item))
    return tuple(result)


def _parse_series(value: Any) -> tuple[RawMetricSeries, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("raw artifact series must be a non-empty list")
    result: list[RawMetricSeries] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {
            "metric_key",
            "reducer",
            "records",
        }:
            raise ValueError("raw metric series fields do not match the schema")
        records_value = item["records"]
        if not isinstance(records_value, list):
            raise ValueError("raw metric records must be a list")
        records: list[RawMetricRecord] = []
        for record in records_value:
            if not isinstance(record, dict) or set(record) != {"record_id", "value"}:
                raise ValueError("raw metric record fields do not match the schema")
            records.append(RawMetricRecord(record["record_id"], record["value"]))
        try:
            reducer = MetricReducer(item["reducer"])
        except (TypeError, ValueError) as exc:
            raise ValueError("unknown raw metric reducer") from exc
        result.append(RawMetricSeries(item["metric_key"], reducer, tuple(records)))
    keys = tuple(item.metric_key for item in result)
    if len(keys) != len(set(keys)):
        raise ValueError("raw artifact metric keys must be unique")
    return tuple(result)


def verify_gate_authority_plan(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_source_digest: str,
) -> VerifiedGateAuthorityPlan:
    """Verify a Root-produced authority plan against out-of-band bindings."""

    if _SHA256_RE.fullmatch(expected_sha256) is None:
        raise ValueError("expected authority plan SHA-256 is malformed")
    if not isinstance(expected_source_digest, str) or not expected_source_digest:
        raise ValueError("expected source digest must be independently supplied")
    plan_path = Path(path).resolve()
    errors: list[str] = []
    try:
        plan_bytes = plan_path.read_bytes()
    except OSError as exc:
        plan_bytes = b""
        errors.append(f"authority_plan_load_failed:{type(exc).__name__}:{exc}")
    byte_sha256 = hashlib.sha256(plan_bytes).hexdigest()
    if byte_sha256 != expected_sha256:
        errors.append("authority_plan_hash_mismatch")

    payload: Any = None
    if plan_bytes:
        try:
            payload = _strict_json_loads(plan_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"authority_plan_invalid_json:{exc}")
    source_digest = expected_source_digest
    gate_spec_set_digest = "unverified-gate-spec-set"
    artifact_bindings: list[TrustedArtifactBinding] = []
    universes: list[TrustedIdentityUniverse] = []
    expected_fields = {
        "schema_version",
        "source_digest",
        "gate_spec_set_digest",
        "artifacts",
        "denominator_universes",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        errors.append("authority_plan_schema_mismatch")
    else:
        if payload["schema_version"] != GATE_AUTHORITY_PLAN_SCHEMA_VERSION:
            errors.append("authority_plan_schema_version_mismatch")
        candidate_source = payload["source_digest"]
        if not isinstance(candidate_source, str) or not candidate_source:
            errors.append("authority_plan_invalid_source_digest")
        else:
            source_digest = candidate_source
            if source_digest != expected_source_digest:
                errors.append("authority_plan_expected_source_mismatch")
        candidate_spec_digest = payload["gate_spec_set_digest"]
        if not isinstance(candidate_spec_digest, str) or not candidate_spec_digest:
            errors.append("authority_plan_invalid_gate_spec_set_digest")
        else:
            gate_spec_set_digest = candidate_spec_digest

        artifact_values = payload["artifacts"]
        if not isinstance(artifact_values, list) or not artifact_values:
            errors.append("authority_plan_has_no_artifacts")
        else:
            for index, item in enumerate(artifact_values):
                try:
                    if not isinstance(item, dict) or set(item) != {
                        "artifact_id",
                        "sha256",
                        "schema_version",
                        "provenance_digest",
                    }:
                        raise ValueError("fields do not match schema")
                    artifact_bindings.append(
                        TrustedArtifactBinding(
                            artifact_id=item["artifact_id"],
                            byte_sha256=item["sha256"],
                            schema_version=item["schema_version"],
                            provenance_digest=item["provenance_digest"],
                        )
                    )
                except (TypeError, ValueError) as exc:
                    errors.append(f"authority_artifact[{index}]:invalid:{exc}")

        universe_values = payload["denominator_universes"]
        if not isinstance(universe_values, list) or not universe_values:
            errors.append("authority_plan_has_no_denominator_universes")
        else:
            for index, item in enumerate(universe_values):
                try:
                    if not isinstance(item, dict) or set(item) != {
                        "metric_key",
                        "cardinality",
                        "identity_digest",
                        "provenance_digest",
                    }:
                        raise ValueError("fields do not match schema")
                    universes.append(
                        TrustedIdentityUniverse(
                            metric_key=item["metric_key"],
                            cardinality=item["cardinality"],
                            identity_digest=item["identity_digest"],
                            provenance_digest=item["provenance_digest"],
                        )
                    )
                except (TypeError, ValueError) as exc:
                    errors.append(f"authority_universe[{index}]:invalid:{exc}")

    artifact_bindings.sort(key=lambda item: item.artifact_id)
    universes.sort(key=lambda item: item.metric_key)
    artifact_ids = tuple(item.artifact_id for item in artifact_bindings)
    if len(artifact_ids) != len(set(artifact_ids)):
        errors.append("authority_plan_duplicate_artifact_id")
        artifact_bindings = list(dict((item.artifact_id, item) for item in artifact_bindings).values())
    universe_keys = tuple(item.metric_key for item in universes)
    if len(universe_keys) != len(set(universe_keys)):
        errors.append("authority_plan_duplicate_universe_metric")
        universes = list(dict((item.metric_key, item) for item in universes).values())

    return VerifiedGateAuthorityPlan(
        expected_source_digest=expected_source_digest,
        expected_plan_sha256=expected_sha256,
        plan_path=str(plan_path),
        byte_sha256=byte_sha256,
        source_digest=source_digest,
        gate_spec_set_digest=gate_spec_set_digest,
        artifact_bindings=tuple(artifact_bindings),
        denominator_universes=tuple(universes),
        verification_errors=tuple(dict.fromkeys(errors)),
    )


def _invalid_evidence(
    manifest_path: Path,
    manifest_sha256: str,
    expected_source_digest: str,
    expected_authority_plan_sha256: str,
    authority_plan: VerifiedGateAuthorityPlan,
    errors: Iterable[str],
) -> VerifiedGateEvidence:
    return VerifiedGateEvidence(
        expected_source_digest=expected_source_digest,
        expected_authority_plan_sha256=expected_authority_plan_sha256,
        source_digest=expected_source_digest,
        manifest_path=str(manifest_path),
        manifest_sha256=manifest_sha256,
        authority_plan=authority_plan,
        artifacts=(),
        claimed_metrics=(),
        verification_errors=tuple(dict.fromkeys(errors)),
    )


def verify_gate_artifact_manifest(
    path: str | Path,
    *,
    expected_source_digest: str,
    expected_authority_plan_sha256: str,
    authority_plan: VerifiedGateAuthorityPlan,
) -> VerifiedGateEvidence:
    """Verify raw artifacts against an independent Root authority plan."""

    if not isinstance(expected_source_digest, str) or not expected_source_digest:
        raise ValueError("expected source digest must be independently supplied")
    if _SHA256_RE.fullmatch(expected_authority_plan_sha256) is None:
        raise ValueError("expected authority plan SHA-256 must be independently supplied")
    if not isinstance(authority_plan, VerifiedGateAuthorityPlan):
        raise TypeError("gate artifact verification requires an authority plan")
    manifest_path = Path(path).resolve()
    errors: list[str] = []
    fresh_authority = verify_gate_authority_plan(
        authority_plan.plan_path,
        expected_sha256=expected_authority_plan_sha256,
        expected_source_digest=expected_source_digest,
    )
    errors.extend(fresh_authority.verification_errors)
    if fresh_authority != authority_plan:
        errors.append("authority_plan_snapshot_mismatch")
    if authority_plan.expected_source_digest != expected_source_digest:
        errors.append("authority_plan_rebound_to_different_expected_source")
    if authority_plan.expected_plan_sha256 != expected_authority_plan_sha256:
        errors.append("authority_plan_rebound_to_different_external_sha256")

    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as exc:
        manifest_bytes = b""
        errors.append(f"gate_manifest_load_failed:{type(exc).__name__}:{exc}")
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if not manifest_bytes:
        errors.append("gate_manifest_empty")
        return _invalid_evidence(
            manifest_path,
            manifest_sha256,
            expected_source_digest,
            expected_authority_plan_sha256,
            authority_plan,
            errors,
        )
    try:
        payload = _strict_json_loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"gate_manifest_invalid_json:{exc}")
        return _invalid_evidence(
            manifest_path,
            manifest_sha256,
            expected_source_digest,
            expected_authority_plan_sha256,
            authority_plan,
            errors,
        )
    expected_fields = {
        "schema_version",
        "source_digest",
        "claimed_metrics",
        "artifacts",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        errors.append("gate_manifest_schema_mismatch")
        return _invalid_evidence(
            manifest_path,
            manifest_sha256,
            expected_source_digest,
            expected_authority_plan_sha256,
            authority_plan,
            errors,
        )
    if payload["schema_version"] != GATE_ARTIFACT_MANIFEST_SCHEMA_VERSION:
        errors.append("gate_manifest_schema_version_mismatch")
    source_digest = payload["source_digest"]
    if not isinstance(source_digest, str) or not source_digest:
        errors.append("gate_manifest_invalid_source_digest")
        source_digest = expected_source_digest
    if source_digest != expected_source_digest:
        errors.append("gate_manifest_expected_source_mismatch")
    if fresh_authority.source_digest != expected_source_digest:
        errors.append("authority_plan_source_binding_mismatch")
    try:
        claims = _numeric_mapping(payload["claimed_metrics"], name="claimed_metrics")
    except ValueError as exc:
        errors.append(f"gate_manifest_invalid_claims:{exc}")
        claims = ()

    references = payload["artifacts"]
    if not isinstance(references, list) or not references:
        errors.append("manifest_has_no_artifacts")
        references = []
    trusted_artifacts = fresh_authority.artifact_map()
    referenced_ids: set[str] = set()
    artifacts: list[VerifiedRawGateArtifact] = []
    for index, reference in enumerate(references):
        prefix = f"artifact[{index}]"
        if not isinstance(reference, dict) or set(reference) != {
            "artifact_id",
            "path",
            "sha256",
            "schema_version",
        }:
            errors.append(f"{prefix}:reference_schema_mismatch")
            continue
        artifact_id = reference["artifact_id"]
        if not isinstance(artifact_id, str) or not artifact_id:
            errors.append(f"{prefix}:invalid_artifact_id")
            continue
        if artifact_id in referenced_ids:
            errors.append(f"{prefix}:duplicate_artifact_id:{artifact_id}")
            continue
        referenced_ids.add(artifact_id)
        trusted = trusted_artifacts.get(artifact_id)
        if trusted is None:
            errors.append(f"{prefix}:artifact_not_in_authority_plan:{artifact_id}")
            continue
        reference_hash = reference["sha256"]
        if reference_hash != trusted.byte_sha256:
            errors.append(f"{prefix}:authority_artifact_hash_mismatch:{artifact_id}")
        if reference["schema_version"] != trusted.schema_version:
            errors.append(f"{prefix}:authority_artifact_schema_mismatch:{artifact_id}")
        if not isinstance(reference_hash, str) or _SHA256_RE.fullmatch(reference_hash) is None:
            errors.append(f"{prefix}:invalid_sha256")
            continue
        relative_path = reference["path"]
        if not isinstance(relative_path, str) or not relative_path:
            errors.append(f"{prefix}:invalid_path")
            continue
        artifact_path = (manifest_path.parent / relative_path).resolve()
        if not artifact_path.is_file():
            errors.append(f"{prefix}:artifact_missing:{artifact_id}")
            continue
        raw_bytes = artifact_path.read_bytes()
        actual_hash = hashlib.sha256(raw_bytes).hexdigest()
        if actual_hash != reference_hash:
            errors.append(f"{prefix}:artifact_hash_mismatch:{artifact_id}")
            continue
        if actual_hash != trusted.byte_sha256:
            errors.append(f"{prefix}:artifact_not_trusted_by_authority:{artifact_id}")
            continue
        try:
            raw = _strict_json_loads(raw_bytes)
            if not isinstance(raw, dict) or set(raw) != {
                "schema_version",
                "artifact_id",
                "source_digest",
                "series",
            }:
                raise ValueError("raw artifact fields do not match the schema")
            if raw["schema_version"] != trusted.schema_version:
                raise ValueError("raw artifact schema version mismatch")
            if raw["artifact_id"] != artifact_id:
                raise ValueError("raw artifact identity mismatch")
            if raw["source_digest"] != expected_source_digest:
                raise ValueError("raw artifact expected source digest mismatch")
            series = _parse_series(raw["series"])
            artifacts.append(
                VerifiedRawGateArtifact(
                    artifact_id=artifact_id,
                    source_digest=expected_source_digest,
                    byte_sha256=actual_hash,
                    path=str(artifact_path),
                    series=series,
                )
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{prefix}:invalid_raw_artifact:{artifact_id}:{exc}")

    trusted_ids = set(trusted_artifacts)
    if referenced_ids != trusted_ids:
        missing = sorted(trusted_ids - referenced_ids)
        extra = sorted(referenced_ids - trusted_ids)
        if missing:
            errors.append("authority_artifacts_missing:" + ",".join(missing))
        if extra:
            errors.append("authority_artifacts_extra:" + ",".join(extra))

    grouped: dict[str, list[RawMetricSeries]] = {}
    for artifact in artifacts:
        for series in artifact.series:
            grouped.setdefault(series.metric_key, []).append(series)
    for metric_key, universe in fresh_authority.universe_map().items():
        bound = grouped.get(metric_key, [])
        if not bound:
            errors.append(f"trusted_denominator_series_missing:{metric_key}")
            continue
        if any(series.reducer is not MetricReducer.COUNT_UNIQUE for series in bound):
            errors.append(f"trusted_denominator_reducer_mismatch:{metric_key}")
            continue
        identities = tuple(record.record_id for series in bound for record in series.records)
        if len(identities) != len(set(identities)):
            errors.append(f"trusted_denominator_duplicate_identity:{metric_key}")
            continue
        if not universe.matches(identities):
            errors.append(f"trusted_denominator_universe_mismatch:{metric_key}")

    return VerifiedGateEvidence(
        expected_source_digest=expected_source_digest,
        expected_authority_plan_sha256=expected_authority_plan_sha256,
        source_digest=source_digest,
        manifest_path=str(manifest_path),
        manifest_sha256=manifest_sha256,
        authority_plan=authority_plan,
        artifacts=tuple(artifacts),
        claimed_metrics=claims,
        verification_errors=tuple(dict.fromkeys(errors)),
    )


def _reduce(reducer: MetricReducer, records: tuple[RawMetricRecord, ...]) -> int | float:
    values = tuple(item.value for item in records)
    if reducer is MetricReducer.COUNT_UNIQUE:
        return len(records)
    if not values:
        raise ValueError("numeric sample reducer requires at least one record")
    if reducer is MetricReducer.SINGLE_VALUE:
        if len(values) != 1:
            raise ValueError("single_value reducer requires exactly one record")
        return values[0]
    if reducer is MetricReducer.MAXIMUM:
        return max(values)
    if reducer is MetricReducer.MINIMUM:
        return min(values)
    if reducer is MetricReducer.NEAREST_RANK_P95:
        ordered = sorted(float(item) for item in values)
        rank = max(1, math.ceil(0.95 * len(ordered)))
        return ordered[rank - 1]
    raise ValueError(f"unsupported reducer: {reducer.value}")


def recompute_gate_metrics(
    evidence: VerifiedGateEvidence,
    expected_reducers: Mapping[str, MetricReducer],
    *,
    expected_source_digest: str,
    expected_authority_plan_sha256: str,
    authority_plan: VerifiedGateAuthorityPlan,
    required_denominator_keys: Iterable[str],
    required_gate_spec_set_digest: str = "",
    require_exact_universe_set: bool = False,
    allow_unselected_claims: bool = False,
) -> MetricRecomputation:
    """Reverify every trust boundary, then reduce frozen raw records."""

    if not isinstance(evidence, VerifiedGateEvidence):
        raise TypeError("metric recomputation requires verified gate evidence")
    if not isinstance(authority_plan, VerifiedGateAuthorityPlan):
        raise TypeError("metric recomputation requires a verified authority plan")
    errors = list(evidence.verification_errors)
    fresh_authority = verify_gate_authority_plan(
        authority_plan.plan_path,
        expected_sha256=expected_authority_plan_sha256,
        expected_source_digest=expected_source_digest,
    )
    errors.extend(fresh_authority.verification_errors)
    if fresh_authority != authority_plan:
        errors.append("authority_plan_snapshot_mismatch")
    if authority_plan.expected_source_digest != expected_source_digest:
        errors.append("authority_plan_rebound_to_different_expected_source")
    if authority_plan.expected_plan_sha256 != expected_authority_plan_sha256:
        errors.append("authority_plan_rebound_to_different_external_sha256")
    if evidence.authority_plan != authority_plan:
        errors.append("evidence_authority_plan_binding_mismatch")
    reverified = verify_gate_artifact_manifest(
        evidence.manifest_path,
        expected_source_digest=expected_source_digest,
        expected_authority_plan_sha256=expected_authority_plan_sha256,
        authority_plan=fresh_authority,
    )
    errors.extend(reverified.verification_errors)
    if reverified != evidence:
        errors.append("verified_evidence_snapshot_mismatch")
    if evidence.expected_source_digest != expected_source_digest:
        errors.append("evidence_expected_source_rebinding")
    if evidence.expected_authority_plan_sha256 != expected_authority_plan_sha256:
        errors.append("evidence_expected_authority_sha256_rebinding")
    if evidence.source_digest != expected_source_digest:
        errors.append("evidence_source_digest_mismatch")

    if required_gate_spec_set_digest and (
        fresh_authority.gate_spec_set_digest != required_gate_spec_set_digest
    ):
        errors.append("authority_gate_spec_set_digest_mismatch")
    required_universes = set(required_denominator_keys)
    available_universes = set(fresh_authority.universe_map())
    missing_universes = sorted(required_universes - available_universes)
    if missing_universes:
        errors.append("authority_denominator_universes_missing:" + ",".join(missing_universes))
    if require_exact_universe_set:
        extra_universes = sorted(available_universes - required_universes)
        if extra_universes:
            errors.append("authority_denominator_universes_extra:" + ",".join(extra_universes))

    grouped: dict[str, list[tuple[VerifiedRawGateArtifact, RawMetricSeries]]] = {}
    for artifact in evidence.artifacts:
        for series in artifact.series:
            grouped.setdefault(series.metric_key, []).append((artifact, series))

    metrics: dict[str, int | float] = {}
    record_ids: dict[str, tuple[str, ...]] = {}
    evidence_by_metric: dict[str, tuple[str, ...]] = {}
    for metric_key, expected_reducer in sorted(expected_reducers.items()):
        bound = grouped.get(metric_key, [])
        if not bound:
            errors.append(f"missing_metric_series:{metric_key}")
            continue
        if any(series.reducer is not expected_reducer for _, series in bound):
            errors.append(f"metric_reducer_mismatch:{metric_key}")
            continue
        records = tuple(record for _, series in bound for record in series.records)
        identities = tuple(item.record_id for item in records)
        if len(identities) != len(set(identities)):
            errors.append(f"duplicate_metric_record:{metric_key}")
            continue
        try:
            metrics[metric_key] = _reduce(expected_reducer, records)
        except ValueError as exc:
            errors.append(f"metric_reduction_failed:{metric_key}:{exc}")
            continue
        record_ids[metric_key] = tuple(sorted(identities))
        evidence_by_metric[metric_key] = tuple(
            sorted({artifact.digest for artifact, _ in bound})
        )

    claims = dict(evidence.claimed_metrics)
    expected_keys = set(expected_reducers)
    missing_claims = sorted(expected_keys - set(claims))
    if missing_claims:
        errors.append("claimed_metrics_missing:" + ",".join(missing_claims))
    if not allow_unselected_claims:
        extra_claims = sorted(set(claims) - expected_keys)
        if extra_claims:
            errors.append("claimed_metrics_extra:" + ",".join(extra_claims))
    for key in sorted(expected_keys & set(claims) & set(metrics)):
        if float(claims[key]) != float(metrics[key]):
            errors.append(f"claimed_metric_mismatch:{key}")

    return MetricRecomputation(
        metrics=tuple(sorted(metrics.items())),
        record_ids=tuple(sorted(record_ids.items())),
        evidence_by_metric=tuple(sorted(evidence_by_metric.items())),
        errors=tuple(dict.fromkeys(errors)),
    )


def metric_series_payload(
    metric_key: str,
    reducer: MetricReducer,
    records: Iterable[tuple[str, int | float]],
) -> dict[str, Any]:
    """Serialization helper only; it grants no gate authority."""

    return {
        "metric_key": metric_key,
        "reducer": reducer.value,
        "records": [
            {"record_id": record_id, "value": value}
            for record_id, value in records
        ],
    }
