from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from datadiff_osc.runtime._phase6_real_evidence_producer_chain import (
    REAL_EVIDENCE_DIAGNOSTIC_MODE,
    TypedEvidenceBinding,
    capture_diagnostic_process,
    diagnostic_runner_preview,
    stage_diagnostic_capture,
)
from datadiff_osc.runtime._receipt_producers import TypedExecutionResult
from datadiff_osc.schemas import (
    ContractFingerprint,
    EvidenceEnvelope,
    EvidenceState,
    EvidenceTier,
    ExecutionStatus,
    FailureKind,
    ResourceTokens,
    ResultGroup,
    SeedLineage,
    SeedStage,
    StructuredExecutionOutcome,
    TaskIdentity,
    TaskKind,
    TaskSpec,
    VerdictKind,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIGEST = "phase6-real-evidence-source-test"


def _environment() -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (
                ("PATH", os.environ["PATH"]),
                ("PYTHONDONTWRITEBYTECODE", "1"),
            )
        )
    )


def _result_and_evidence() -> tuple[TypedExecutionResult, EvidenceEnvelope]:
    lineage = SeedLineage(
        protocol_digest="phase6-real-evidence-protocol",
        master_seed=17,
        lane_id="diagnostic-lane",
        case_index=0,
        stage_name=SeedStage.BACKEND,
    )
    task = TaskSpec(
        identity=TaskIdentity(
            protocol_digest=lineage.protocol_digest,
            task_kind=TaskKind.BACKEND_EXECUTION,
            epoch_index=0,
            decision_index=0,
            seed_lineage_digest=lineage.digest,
            endpoint_id="endpoint-a",
            backend="backend-a",
        ),
        dependency_task_ids=(),
        resources=ResourceTokens(
            cpu_tokens=1,
            rss_bytes=0,
            io_class="phase6-diagnostic",
            backend_internal_threads=0,
        ),
        payload_digest="phase6-real-evidence-payload",
    )
    group = ResultGroup(
        result_group_id="phase6-real-evidence-result-group",
        task_ids=(task.identity.task_id,),
        endpoint_ids=("endpoint-a",),
        contract_fingerprint=ContractFingerprint(
            contract_digest="phase6-real-evidence-contract",
            registry_digest="phase6-real-evidence-registry",
        ),
        target_fingerprint=None,
        evidence_tier=EvidenceTier.AUDIT,
        result_order_key=("endpoint-a",),
    )
    outcome = StructuredExecutionOutcome(
        endpoint_id="endpoint-a",
        status=ExecutionStatus.OK,
        failure_kind=FailureKind.NONE,
    )
    result = TypedExecutionResult(
        case_id="phase6-real-evidence-case",
        result_group=group,
        task=task,
        seed_lineage=lineage,
        outcome=outcome,
    )
    evidence = EvidenceEnvelope(
        evidence_id="phase6-real-evidence-envelope",
        state=EvidenceState.NOT_A_CANDIDATE,
        result_group_digest=group.digest,
        task_id=task.identity.task_id,
        seed_lineage_digest=lineage.digest,
        contract_fingerprint=group.contract_fingerprint,
        target_fingerprint=None,
        derivation_certificate_digest="phase6-real-evidence-derivation",
        applicability_certificate_digest="phase6-real-evidence-applicability",
        activation_certificate_digest="phase6-real-evidence-activation",
        observation_certificate_digest="phase6-real-evidence-observation",
        execution_outcomes=(outcome,),
        verdict_kind=VerdictKind.SATISFIED,
    )
    return result, evidence


def _binding(output: bytes) -> TypedEvidenceBinding:
    result, evidence = _result_and_evidence()
    return TypedEvidenceBinding(
        source_digest=SOURCE_DIGEST,
        subject_kind="typed_execution_result",
        result=result,
        evidence=evidence,
        expected_result_id=result.result_id,
        expected_result_group_digest=result.result_group.digest,
        expected_task_spec_digest=result.task.digest,
        expected_seed_lineage_digest=result.seed_lineage.digest,
        expected_execution_outcome_digest=result.outcome.digest,
        expected_output_sha256=hashlib.sha256(output).hexdigest(),
    )


def _capture(tmp_path: Path, program: str) -> object:
    return capture_diagnostic_process(
        source_digest=SOURCE_DIGEST,
        source_root=tmp_path,
        cwd=tmp_path,
        command=(sys.executable, "-c", program),
        environment=_environment(),
        timeout_seconds=5,
    )


def test_process_bound_typed_capture_is_in_memory_diagnostic_only(tmp_path):
    output = b'{"observed":"ok"}'
    transcript = _capture(
        tmp_path,
        "import sys; sys.stdout.buffer.write(b'{\"observed\":\"ok\"}')",
    )
    capture = stage_diagnostic_capture(transcript=transcript, binding=_binding(output))

    assert transcript.exit_code == 0
    assert transcript.output_sha256 == hashlib.sha256(output).hexdigest()
    assert capture.status == "not_evaluated"
    assert capture.diagnostic_only is True
    assert capture.authority_eligible is False
    assert capture.gate_credit is False
    assert capture.dynamic_plan_emitted is False
    assert capture.artifact_receipt_index_emitted is False
    assert capture.authority_bundle_emitted is False
    assert capture.candidate_confirmed is False
    assert capture.bug_claimed is False
    payload = json.loads(capture.payload_json)
    assert payload["mode"] == REAL_EVIDENCE_DIAGNOSTIC_MODE
    assert payload["typed_result_id"] == capture.binding.subject_id
    assert payload["output_sha256"] == transcript.output_sha256
    assert list(tmp_path.iterdir()) == []


def test_output_source_and_exact_typed_binding_fail_closed(tmp_path):
    output = b"ok"
    transcript = _capture(tmp_path, "import sys; sys.stdout.buffer.write(b'ok')")
    binding = _binding(output)

    with pytest.raises(ValueError, match="output SHA-256 mismatch"):
        stage_diagnostic_capture(
            transcript=transcript,
            binding=replace(binding, expected_output_sha256="0" * 64),
        )
    with pytest.raises(ValueError, match="source digest mismatch"):
        stage_diagnostic_capture(
            transcript=transcript,
            binding=replace(binding, source_digest="other-source"),
        )
    with pytest.raises(ValueError, match="evidence/task mismatch"):
        TypedEvidenceBinding(
            source_digest=SOURCE_DIGEST,
            subject_kind="typed_execution_result",
            result=binding.result,
            evidence=replace(binding.evidence, task_id="wrong-task"),
            expected_result_id=binding.expected_result_id,
            expected_result_group_digest=binding.expected_result_group_digest,
            expected_task_spec_digest=binding.expected_task_spec_digest,
            expected_seed_lineage_digest=binding.expected_seed_lineage_digest,
            expected_execution_outcome_digest=binding.expected_execution_outcome_digest,
            expected_output_sha256=binding.expected_output_sha256,
        )
    with pytest.raises(ValueError, match="task-spec mismatch"):
        TypedEvidenceBinding(
            source_digest=SOURCE_DIGEST,
            subject_kind="typed_execution_result",
            result=binding.result,
            evidence=binding.evidence,
            expected_result_id=binding.expected_result_id,
            expected_result_group_digest=binding.expected_result_group_digest,
            expected_task_spec_digest="wrong-task-spec",
            expected_seed_lineage_digest=binding.expected_seed_lineage_digest,
            expected_execution_outcome_digest=binding.expected_execution_outcome_digest,
            expected_output_sha256=binding.expected_output_sha256,
        )
    with pytest.raises(ValueError, match="seed-lineage mismatch"):
        replace(binding, expected_seed_lineage_digest="wrong-seed")
    with pytest.raises(ValueError, match="execution-outcome mismatch"):
        replace(binding, expected_execution_outcome_digest="wrong-outcome")


def test_nonzero_fabricated_and_escaped_context_are_rejected(tmp_path):
    transcript = _capture(
        tmp_path,
        "import sys; sys.stdout.buffer.write(b'x'); raise SystemExit(7)",
    )
    with pytest.raises(ValueError, match="nonzero process exit"):
        stage_diagnostic_capture(transcript=transcript, binding=_binding(b"x"))
    result, evidence = _result_and_evidence()
    with pytest.raises(ValueError, match="synthetic marker"):
        TypedEvidenceBinding(
            source_digest="synthetic-source",
            subject_kind="typed_execution_result",
            result=result,
            evidence=evidence,
            expected_result_id=result.result_id,
            expected_result_group_digest=result.result_group.digest,
            expected_task_spec_digest=result.task.digest,
            expected_seed_lineage_digest=result.seed_lineage.digest,
            expected_execution_outcome_digest=result.outcome.digest,
            expected_output_sha256=hashlib.sha256(b"x").hexdigest(),
        )
    with pytest.raises(ValueError, match="escapes the declared source root"):
        capture_diagnostic_process(
            source_digest=SOURCE_DIGEST,
            source_root=tmp_path,
            cwd=tmp_path.parent,
            command=(sys.executable, "-c", "print('no-run')"),
            environment=_environment(),
        )
    with pytest.raises(ValueError, match="not allowlisted"):
        capture_diagnostic_process(
            source_digest=SOURCE_DIGEST,
            source_root=tmp_path,
            cwd=tmp_path,
            command=(sys.executable, "-c", "print('no-run')"),
            environment=(("HOME", "forbidden"),),
        )
    with pytest.raises(ValueError, match="must be non-empty and sorted"):
        capture_diagnostic_process(
            source_digest=SOURCE_DIGEST,
            source_root=tmp_path,
            cwd=tmp_path,
            command=(sys.executable, "-c", "print('no-run')"),
            environment=(
                ("PYTHONDONTWRITEBYTECODE", "1"),
                ("PATH", os.environ["PATH"]),
            ),
        )
    with pytest.raises(ValueError, match="immutable tuple"):
        capture_diagnostic_process(
            source_digest=SOURCE_DIGEST,
            source_root=tmp_path,
            cwd=tmp_path,
            command=(sys.executable, "-c", "print('no-run')"),
            environment=[("PATH", os.environ["PATH"])],  # type: ignore[arg-type]
        )


def test_runner_has_no_execution_or_durable_output_path(tmp_path):
    script = ROOT / "scripts/osc/run_phase6_real_evidence_producer_chain.py"
    completed = subprocess.run(
        (sys.executable, str(script), "--dry-run"),
        cwd=tmp_path,
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode == 0
    assert completed.stderr == ""
    assert json.loads(completed.stdout) == diagnostic_runner_preview()
    assert list(tmp_path.iterdir()) == []

    rejected = subprocess.run(
        (sys.executable, str(script), "--execute"),
        cwd=tmp_path,
        capture_output=True,
        check=False,
        text=True,
    )
    assert rejected.returncode == 2
    assert "unrecognized arguments: --execute" in rejected.stderr
    assert list(tmp_path.iterdir()) == []
