from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

import datadiff_osc.runtime as public_runtime
from datadiff_osc._canonical import canonical_json
from datadiff_osc.runtime._phase6_bounded_diagnostic_artifact_revalidation import (
    BoundedDiagnosticArtifactRevalidation,
    revalidate_bounded_diagnostic_artifact,
)
from datadiff_osc.runtime._phase6_bounded_real_adapter_artifact import (
    BOUNDED_REAL_ADAPTER_BACKENDS,
    BoundedRealAdapterArtifact,
    SourceSnapshotBinding,
    build_bounded_real_adapter_artifact,
    build_bounded_real_adapter_binding,
    write_bounded_real_adapter_artifact,
)
from datadiff_osc.runtime._phase6_real_adapter_execution_sample import (
    DIRECT_SERIAL_UNCACHED_EXECUTION_MODE,
    RealAdapterExecutionRecord,
    RealAdapterExecutionSample,
)


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/osc/run_phase6_bounded_real_adapter_artifact.py"
RUNNER_SHA256 = hashlib.sha256(RUNNER.read_bytes()).hexdigest()


def _source() -> SourceSnapshotBinding:
    return SourceSnapshotBinding(
        source_digest="osc-phase6-source-snapshot-test-bounded-revalidation",
        snapshot_byte_sha256="a" * 64,
        git_head="b" * 40,
        git_status_digest="osc-git-status-porcelain-v1-test-bounded-revalidation",
    )


def _record(backend: str) -> RealAdapterExecutionRecord:
    raw = canonical_json(
        {"backend": backend, "error": "", "error_type": "", "status": "ok"}
    )
    normalized = canonical_json(
        {"backend": backend, "error": "", "error_type": "", "status": "ok"}
    )
    return RealAdapterExecutionRecord(
        backend=backend,
        adapter_type="tests:BoundedRevalidationFakeAdapter",
        raw_summary_json=raw,
        raw_summary_sha256=hashlib.sha256(raw.encode()).hexdigest(),
        normalized_result_json=normalized,
        normalized_result_sha256=hashlib.sha256(normalized.encode()).hexdigest(),
    )


def _artifact() -> BoundedRealAdapterArtifact:
    binding = build_bounded_real_adapter_binding()
    sample = RealAdapterExecutionSample(
        binding=binding,
        execution_mode=DIRECT_SERIAL_UNCACHED_EXECUTION_MODE,
        adapter_records=tuple(_record(item) for item in BOUNDED_REAL_ADAPTER_BACKENDS),
    )
    return build_bounded_real_adapter_artifact(
        source=_source(), sample=sample, runner_sha256=RUNNER_SHA256
    )


def _write(tmp_path: Path, artifact: BoundedRealAdapterArtifact, label: str) -> tuple[Path, str]:
    directory = tmp_path / label
    directory.mkdir()
    path = directory / "real-adapter-execution-artifact.json"
    return path, write_bounded_real_adapter_artifact(artifact=artifact, output_path=path)


def _revalidate(path: Path, digest: str) -> BoundedDiagnosticArtifactRevalidation:
    return revalidate_bounded_diagnostic_artifact(
        artifact_path=path,
        expected_artifact_sha256=digest,
        expected_source=_source(),
        expected_runner_sha256=RUNNER_SHA256,
    )


def test_revalidation_rebuilds_exact_private_context_without_authority(tmp_path):
    artifact = _artifact()
    path, digest = _write(tmp_path, artifact, "legal")

    checked = _revalidate(path, digest)

    assert checked.artifact == artifact
    assert checked.artifact_sha256 == digest
    assert checked.binding_digest == build_bounded_real_adapter_binding().digest
    assert checked.sample_id == artifact.sample_id
    assert checked.sample_digest == artifact.sample_digest
    assert checked.status == "not_evaluated"
    assert checked.diagnostic_only is True
    assert checked.authority_eligible is False
    assert checked.gate_credit is False
    assert checked.formal_raw_gate_artifact_emitted is False
    assert checked.typed_producer_receipt_emitted is False
    assert checked.artifact_receipt_index_emitted is False
    assert checked.dynamic_plan_emitted is False
    assert checked.coverage_event_created is False
    assert checked.authority_bundle_emitted is False
    assert checked.candidate_confirmed is False
    assert checked.bug_claimed is False
    assert "BoundedDiagnosticArtifactRevalidation" not in public_runtime.__all__
    assert not hasattr(public_runtime, "BoundedDiagnosticArtifactRevalidation")


@pytest.mark.parametrize(
    ("field", "value", "expected_error"),
    (
        ("task_ids", ("wrong-task-0", "wrong-task-1"), "task_ids"),
        ("result_ids", ("wrong-result-0", "wrong-result-1"), "result_ids"),
        ("seed_lineage_digest", "wrong-seed", "seed_lineage_digest"),
        ("outcome_digests", ("wrong-outcome-0", "wrong-outcome-1"), "outcome_digests"),
        ("evidence_digests", ("wrong-evidence-0", "wrong-evidence-1"), "evidence_digests"),
        ("sample_id", "wrong-sample-id", "sample_id"),
        ("sample_digest", "wrong-sample-digest", "sample_digest"),
    ),
)
def test_revalidation_rejects_exact_context_substitutions(
    tmp_path, field, value, expected_error
):
    path, digest = _write(tmp_path, replace(_artifact(), **{field: value}), field)

    with pytest.raises(
        ValueError,
        match=f"bounded diagnostic artifact exact context mismatch:{expected_error}",
    ):
        _revalidate(path, digest)


def test_revalidation_rejects_record_and_external_binding_substitutions(tmp_path):
    artifact = _artifact()
    first, second = artifact.records
    changed_raw = canonical_json(
        {
            "backend": first.backend,
            "error": "",
            "error_type": "",
            "note": "changed",
            "status": "ok",
        }
    )
    changed_record = replace(
        first,
        raw_summary_json=changed_raw,
        raw_summary_sha256=hashlib.sha256(changed_raw.encode()).hexdigest(),
    )
    path, digest = _write(
        tmp_path, replace(artifact, records=(changed_record, second)), "record"
    )
    with pytest.raises(
        ValueError,
        match="bounded diagnostic artifact exact context mismatch:sample_id",
    ):
        _revalidate(path, digest)

    legal_path, legal_digest = _write(tmp_path, artifact, "external")
    with pytest.raises(ValueError, match="bounded artifact SHA mismatch"):
        _revalidate(legal_path, "0" * 64)
    with pytest.raises(ValueError, match="source binding mismatch"):
        revalidate_bounded_diagnostic_artifact(
            artifact_path=legal_path,
            expected_artifact_sha256=legal_digest,
            expected_source=replace(_source(), source_digest="wrong-source"),
            expected_runner_sha256=RUNNER_SHA256,
        )
    with pytest.raises(ValueError, match="runner binding mismatch"):
        revalidate_bounded_diagnostic_artifact(
            artifact_path=legal_path,
            expected_artifact_sha256=legal_digest,
            expected_source=_source(),
            expected_runner_sha256="d" * 64,
        )
    with pytest.raises(ValueError, match="must be absolute"):
        _revalidate(Path("relative.json"), legal_digest)
    with pytest.raises(ValueError, match="could not be read"):
        _revalidate(tmp_path / "missing.json", legal_digest)


def test_revalidation_module_has_no_adapter_capture_or_durable_output_api():
    import datadiff_osc.runtime._phase6_bounded_diagnostic_artifact_revalidation as module

    assert not hasattr(module, "capture_real_adapter_execution_sample")
    assert not hasattr(module, "write_bounded_real_adapter_artifact")
    assert not hasattr(module, "subprocess")
