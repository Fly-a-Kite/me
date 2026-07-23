from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import os
import subprocess
import sys

import pytest

import datadiff_osc.runtime as public_runtime
from datadiff_osc._canonical import canonical_json
from datadiff_osc.runtime._phase6_bounded_real_adapter_artifact import (
    BOUNDED_REAL_ADAPTER_BACKENDS,
    BoundedRealAdapterArtifact,
    SourceSnapshotBinding,
    artifact_bytes,
    build_bounded_real_adapter_artifact,
    build_bounded_real_adapter_binding,
    verify_bounded_real_adapter_artifact,
    write_bounded_real_adapter_artifact,
)
from datadiff_osc.runtime._phase6_real_adapter_execution_sample import (
    DIRECT_SERIAL_UNCACHED_EXECUTION_MODE,
    RealAdapterExecutionRecord,
    RealAdapterExecutionSample,
)


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/osc/run_phase6_bounded_real_adapter_artifact.py"
RUNNER_SHA256 = hashlib.sha256(SCRIPT.read_bytes()).hexdigest()


def _source() -> SourceSnapshotBinding:
    return SourceSnapshotBinding(
        source_digest="osc-phase6-source-snapshot-test-bounded-artifact",
        snapshot_byte_sha256="a" * 64,
        git_head="b" * 40,
        git_status_digest="osc-git-status-porcelain-v1-test-bounded-artifact",
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
        adapter_type="tests:BoundedFakeAdapter",
        raw_summary_json=raw,
        raw_summary_sha256=hashlib.sha256(raw.encode()).hexdigest(),
        normalized_result_json=normalized,
        normalized_result_sha256=hashlib.sha256(normalized.encode()).hexdigest(),
    )


def _sample() -> RealAdapterExecutionSample:
    binding = build_bounded_real_adapter_binding()
    return RealAdapterExecutionSample(
        binding=binding,
        execution_mode=DIRECT_SERIAL_UNCACHED_EXECUTION_MODE,
        adapter_records=tuple(_record(item) for item in BOUNDED_REAL_ADAPTER_BACKENDS),
    )


def _artifact() -> BoundedRealAdapterArtifact:
    return build_bounded_real_adapter_artifact(
        source=_source(), sample=_sample(), runner_sha256=RUNNER_SHA256
    )


def test_bounded_artifact_is_exact_private_and_non_authoritative():
    artifact = _artifact()

    assert artifact.family_id == "pandas_nullable_bool_reduction"
    assert tuple(record.backend for record in artifact.records) == BOUNDED_REAL_ADAPTER_BACKENDS
    assert artifact.master_seed == 29
    assert artifact.execution_mode == DIRECT_SERIAL_UNCACHED_EXECUTION_MODE
    assert artifact.authority_eligible is False
    assert artifact.gate_credit is False
    assert artifact.dynamic_plan_emitted is False
    assert artifact.typed_receipt_index_emitted is False
    assert artifact.coverage_event_created is False
    assert artifact.candidate_confirmed is False
    assert artifact.bug_claimed is False
    assert BoundedRealAdapterArtifact.from_dict(
        json.loads(artifact_bytes(artifact).decode())
    ) == artifact
    assert "BoundedRealAdapterArtifact" not in public_runtime.__all__
    assert not hasattr(public_runtime, "BoundedRealAdapterArtifact")


def test_writer_and_verifier_bind_exact_bytes_source_and_runner(tmp_path):
    artifact = _artifact()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    output = output_dir / "real-adapter-execution-artifact.json"

    expected_sha = write_bounded_real_adapter_artifact(
        artifact=artifact, output_path=output
    )
    assert expected_sha == hashlib.sha256(output.read_bytes()).hexdigest()
    assert verify_bounded_real_adapter_artifact(
        artifact_path=output,
        expected_sha256=expected_sha,
        expected_source=_source(),
        expected_runner_sha256=RUNNER_SHA256,
    ) == artifact

    with pytest.raises(FileExistsError, match="already exists"):
        write_bounded_real_adapter_artifact(artifact=artifact, output_path=output)
    with pytest.raises(ValueError, match="runner binding mismatch"):
        verify_bounded_real_adapter_artifact(
            artifact_path=output,
            expected_sha256=expected_sha,
            expected_source=_source(),
            expected_runner_sha256="d" * 64,
        )
    with pytest.raises(ValueError, match="source binding mismatch"):
        verify_bounded_real_adapter_artifact(
            artifact_path=output,
            expected_sha256=expected_sha,
            expected_source=replace(_source(), source_digest="other-source"),
            expected_runner_sha256=RUNNER_SHA256,
        )


def test_negative_review_rejects_record_seed_and_json_substitutions(tmp_path):
    sample = _sample()
    first, second = sample.adapter_records
    with pytest.raises(ValueError, match="record order"):
        replace(sample, adapter_records=(second, first))
    with pytest.raises(ValueError, match="raw summary SHA mismatch"):
        replace(first, raw_summary_sha256="0" * 64)

    artifact = _artifact()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    output = output_dir / "real-adapter-execution-artifact.json"
    expected_sha = write_bounded_real_adapter_artifact(
        artifact=artifact, output_path=output
    )
    payload = json.loads(output.read_text())
    payload["master_seed"] = 30
    output.write_text(canonical_json(payload) + "\n")
    with pytest.raises(ValueError, match="duplicate JSON fields"):
        verify_bounded_real_adapter_artifact(
            artifact_path=output,
            expected_sha256=expected_sha,
            expected_source=_source(),
            expected_runner_sha256=RUNNER_SHA256,
        )

    malformed = output_dir / "malformed.json"
    malformed.write_text('{"a":1,"a":2}\n')
    with pytest.raises(ValueError, match="bounded artifact SHA mismatch"):
        verify_bounded_real_adapter_artifact(
            artifact_path=malformed,
            expected_sha256=hashlib.sha256(malformed.read_bytes()).hexdigest(),
            expected_source=_source(),
            expected_runner_sha256=RUNNER_SHA256,
        )


def test_exception_inputs_fail_closed_without_adapter_or_reserved_output(tmp_path):
    artifact = _artifact()
    with pytest.raises(ValueError, match="must be absolute"):
        write_bounded_real_adapter_artifact(
            artifact=artifact,
            output_path=Path("real-adapter-execution-artifact.json"),
        )
    with pytest.raises(ValueError, match="filename is not authorized"):
        write_bounded_real_adapter_artifact(
            artifact=artifact, output_path=tmp_path / "other.json"
        )
    with pytest.raises(ValueError, match="Git head"):
        SourceSnapshotBinding(
            source_digest="source",
            snapshot_byte_sha256="a" * 64,
            git_head="not-a-git-head",
            git_status_digest="status",
        )

    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "--source-snapshot" in completed.stdout
    assert list(tmp_path.iterdir()) == []
