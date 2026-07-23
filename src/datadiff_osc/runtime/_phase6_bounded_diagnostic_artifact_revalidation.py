"""Private, read-only revalidation for one bounded diagnostic artifact.

This module is intentionally narrower than a Phase-6 producer.  It re-reads
one private bounded artifact and rebuilds its fixed declaration-derived typed
context without running adapters or emitting any durable authority material.
The returned value is permanently diagnostic-only and not_evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from datadiff_osc._canonical import stable_digest
from datadiff_osc.runtime._phase6_bounded_real_adapter_artifact import (
    BOUNDED_REAL_ADAPTER_BACKENDS,
    BOUNDED_REAL_ADAPTER_FAMILY_ID,
    BOUNDED_REAL_ADAPTER_MASTER_SEED,
    BOUNDED_REAL_ADAPTER_PROTOCOL_DIGEST,
    BOUNDED_REAL_ADAPTER_SEED_BLOCK_ID,
    BoundedAdapterRecord,
    BoundedRealAdapterArtifact,
    SourceSnapshotBinding,
    build_bounded_real_adapter_binding,
    verify_bounded_real_adapter_artifact,
)
from datadiff_osc.runtime._phase6_real_adapter_execution_sample import (
    DIRECT_SERIAL_UNCACHED_EXECUTION_MODE,
    RealAdapterExecutionRecord,
    RealAdapterExecutionSample,
)


BOUNDED_DIAGNOSTIC_ARTIFACT_REVALIDATION_SCHEMA_VERSION = (
    "osc-private-bounded-diagnostic-artifact-revalidation-v1"
)


def _require_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _reconstruct_sample(
    artifact: BoundedRealAdapterArtifact,
) -> tuple[object, RealAdapterExecutionSample]:
    """Rebuild the exact private context without adapter execution."""

    binding = build_bounded_real_adapter_binding()
    records = tuple(
        RealAdapterExecutionRecord(
            backend=item.backend,
            adapter_type=item.adapter_type,
            raw_summary_json=item.raw_summary_json,
            raw_summary_sha256=item.raw_summary_sha256,
            normalized_result_json=item.normalized_result_json,
            normalized_result_sha256=item.normalized_result_sha256,
        )
        for item in artifact.records
    )
    sample = RealAdapterExecutionSample(
        binding=binding,
        execution_mode=artifact.execution_mode,
        adapter_records=records,
    )
    return binding, sample


def _expected_context(
    *, binding: object, sample: RealAdapterExecutionSample
) -> tuple[tuple[str, object], ...]:
    """Derive every serialized identity from the rebuilt private objects."""

    scheduled = binding.scheduled_attempt
    run, _case = scheduled.plan.case_run_binding(
        scheduled.case_id, scheduled.case_index
    )
    results = binding.results
    result_group = results[0].result_group
    records = tuple(
        BoundedAdapterRecord.from_sample_record(item) for item in sample.adapter_records
    )
    return (
        ("family_id", scheduled.target_family_id),
        ("target_cell_id", scheduled.target_cell_id),
        ("backend_pair_obligation_id", binding.backend_pair.obligation_id),
        ("protocol_digest", run.protocol_digest),
        ("lane_id", run.lane_id),
        ("seed_block_id", run.seed_block_id),
        ("master_seed", run.master_seed),
        ("seed_lineage_digest", results[0].seed_lineage.digest),
        ("scheduled_attempt_id", scheduled.attempt_id),
        ("source_case_id", binding.canonical_case.source_case_id),
        ("result_group_id", result_group.result_group_id),
        ("result_group_digest", result_group.digest),
        ("task_ids", tuple(item.task.identity.task_id for item in results)),
        ("result_ids", tuple(item.result_id for item in results)),
        ("outcome_digests", tuple(item.outcome.digest for item in results)),
        ("evidence_digests", tuple(item.digest for item in binding.evidence)),
        ("execution_mode", sample.execution_mode),
        ("sample_id", sample.sample_id),
        ("sample_digest", sample.digest),
        ("records", records),
    )


def _verify_fixed_authorization(
    *, artifact: BoundedRealAdapterArtifact, binding: object
) -> None:
    """Reject drift from the one private family/pair/seed authorization."""

    scheduled = binding.scheduled_attempt
    run, _case = scheduled.plan.case_run_binding(
        scheduled.case_id, scheduled.case_index
    )
    if (
        scheduled.target_family_id != BOUNDED_REAL_ADAPTER_FAMILY_ID
        or (binding.backend_pair.target_backend, binding.backend_pair.control_backend)
        != BOUNDED_REAL_ADAPTER_BACKENDS
        or run.protocol_digest != BOUNDED_REAL_ADAPTER_PROTOCOL_DIGEST
        or run.seed_block_id != BOUNDED_REAL_ADAPTER_SEED_BLOCK_ID
        or run.master_seed != BOUNDED_REAL_ADAPTER_MASTER_SEED
        or artifact.execution_mode != DIRECT_SERIAL_UNCACHED_EXECUTION_MODE
    ):
        raise ValueError("bounded diagnostic artifact authorization context mismatch")


@dataclass(frozen=True, slots=True)
class BoundedDiagnosticArtifactRevalidation:
    """One re-read private artifact bound to rebuilt typed context.

    The object intentionally has no serialization method and no authority
    promotion path.  It reports only a diagnostic not_evaluated outcome.
    """

    artifact: BoundedRealAdapterArtifact
    artifact_sha256: str
    binding_digest: str
    sample_id: str
    sample_digest: str
    schema_version: str = BOUNDED_DIAGNOSTIC_ARTIFACT_REVALIDATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, BoundedRealAdapterArtifact):
            raise TypeError("revalidation requires BoundedRealAdapterArtifact")
        _require_sha256(self.artifact_sha256, name="revalidation artifact SHA")
        for name, value in (
            ("revalidation binding digest", self.binding_digest),
            ("revalidation sample ID", self.sample_id),
            ("revalidation sample digest", self.sample_digest),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be non-empty text")
        if self.schema_version != BOUNDED_DIAGNOSTIC_ARTIFACT_REVALIDATION_SCHEMA_VERSION:
            raise ValueError("revalidation schema version mismatch")
        if self.sample_id != self.artifact.sample_id:
            raise ValueError("revalidation sample ID/artifact mismatch")
        if self.sample_digest != self.artifact.sample_digest:
            raise ValueError("revalidation sample digest/artifact mismatch")

    @property
    def artifact_id(self) -> str:
        return self.artifact.artifact_id

    @property
    def source_digest(self) -> str:
        return self.artifact.source.source_digest

    @property
    def status(self) -> str:
        return "not_evaluated"

    @property
    def diagnostic_only(self) -> bool:
        return True

    @property
    def authority_eligible(self) -> bool:
        return False

    @property
    def gate_credit(self) -> bool:
        return False

    @property
    def formal_raw_gate_artifact_emitted(self) -> bool:
        return False

    @property
    def typed_producer_receipt_emitted(self) -> bool:
        return False

    @property
    def artifact_receipt_index_emitted(self) -> bool:
        return False

    @property
    def dynamic_plan_emitted(self) -> bool:
        return False

    @property
    def coverage_event_created(self) -> bool:
        return False

    @property
    def authority_bundle_emitted(self) -> bool:
        return False

    @property
    def candidate_confirmed(self) -> bool:
        return False

    @property
    def bug_claimed(self) -> bool:
        return False

    @property
    def digest(self) -> str:
        return stable_digest(
            "osc-private-bounded-diagnostic-artifact-revalidation",
            {
                "artifact_id": self.artifact_id,
                "artifact_sha256": self.artifact_sha256,
                "binding_digest": self.binding_digest,
                "sample_id": self.sample_id,
                "sample_digest": self.sample_digest,
                "schema_version": self.schema_version,
            },
        )


def revalidate_bounded_diagnostic_artifact(
    *,
    artifact_path: Path,
    expected_artifact_sha256: str,
    expected_source: SourceSnapshotBinding,
    expected_runner_sha256: str,
) -> BoundedDiagnosticArtifactRevalidation:
    """Re-read one private artifact and exact-bind it to current typed context.

    This function is read-only.  It never invokes an adapter, subprocess,
    runner, writer, producer receipt, raw-gate artifact, receipt index, or
    dynamic plan.
    """

    artifact = verify_bounded_real_adapter_artifact(
        artifact_path=artifact_path,
        expected_sha256=expected_artifact_sha256,
        expected_source=expected_source,
        expected_runner_sha256=expected_runner_sha256,
    )
    binding, sample = _reconstruct_sample(artifact)
    _verify_fixed_authorization(artifact=artifact, binding=binding)
    for field, expected in _expected_context(binding=binding, sample=sample):
        if getattr(artifact, field) != expected:
            raise ValueError(f"bounded diagnostic artifact exact context mismatch:{field}")
    return BoundedDiagnosticArtifactRevalidation(
        artifact=artifact,
        artifact_sha256=expected_artifact_sha256,
        binding_digest=binding.digest,
        sample_id=sample.sample_id,
        sample_digest=sample.digest,
    )
