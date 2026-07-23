from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from datadiff_osc.runtime.gate_artifacts import (
    GATE_ARTIFACT_MANIFEST_SCHEMA_VERSION,
    GATE_AUTHORITY_PLAN_SCHEMA_VERSION,
    RAW_GATE_ARTIFACT_SCHEMA_VERSION,
    identity_universe_digest,
)
from datadiff_osc.runtime.gates import (
    default_pre24_gate_specs,
    pre24_gate_spec_set_digest,
)


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = (
    "audit_staged_parity.py",
    "benchmark_clustering.py",
    "audit_parallel_invariance.py",
    "benchmark_parallel_scaling.py",
    "calculate_gate_report.py",
    "build_v3_gate_plan.py",
    "build_2h_pilot_plan.py",
)


def _run(name, *arguments, check=True):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "osc" / name), *map(str, arguments)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=check,
    )


def _json_bytes(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _write_authority_plan(tmp_path, *, artifact_hash, source="source-snapshot-1"):
    denominator_keys = sorted(
        {spec.denominator_key for spec in default_pre24_gate_specs()}
    )
    payload = {
        "schema_version": GATE_AUTHORITY_PLAN_SCHEMA_VERSION,
        "source_digest": source,
        "gate_spec_set_digest": pre24_gate_spec_set_digest(),
        "artifacts": [
            {
                "artifact_id": "raw-gates",
                "sha256": artifact_hash,
                "schema_version": RAW_GATE_ARTIFACT_SCHEMA_VERSION,
                "provenance_digest": "root-artifact-index-1",
            }
        ],
        "denominator_universes": [
            {
                "metric_key": key,
                "cardinality": 0,
                "identity_digest": identity_universe_digest(()),
                "provenance_digest": f"root-universe-{key}",
            }
            for key in denominator_keys
        ],
    }
    path = tmp_path / "authority.json"
    raw = _json_bytes(payload)
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def test_all_runtime_tools_have_read_only_help_entrypoints():
    for script in SCRIPTS:
        result = _run(script, "--help")
        assert "usage:" in result.stdout


def test_bounded_parity_clustering_and_parallel_tools_emit_raw_reports(tmp_path):
    parity = tmp_path / "parity.json"
    clustering = tmp_path / "clustering.json"
    invariance = tmp_path / "invariance.json"
    scaling = tmp_path / "scaling.json"
    _run("audit_staged_parity.py", "--groups", 12, "--output", parity)
    _run("benchmark_clustering.py", "--backend-counts", "2,4", "--repetitions", 2, "--output", clustering)
    _run("audit_parallel_invariance.py", "--tasks", 6, "--workers", "1,2", "--completion-delay-ms", 0, "--output", invariance)
    _run("benchmark_parallel_scaling.py", "--tasks", 6, "--workers", "1,2", "--work-ms", 0, "--output", scaling)
    parity_payload = json.loads(parity.read_text())
    cluster_payload = json.loads(clustering.read_text())
    invariant_payload = json.loads(invariance.read_text())
    scaling_payload = json.loads(scaling.read_text())
    assert parity_payload["report"]["group_count"] == 12
    assert parity_payload["report"]["discrepancy_count"] == 0
    assert parity_payload["report"]["exact_escalation_count"] == 2
    assert parity_payload["phase6_gate_eligible"] is False
    assert cluster_payload["report"]["points"][0]["pairwise_partition_match"] is True
    assert invariant_payload["report"]["task_set_invariant"] is True
    assert invariant_payload["report"]["assignment_invariant"] is True
    assert invariant_payload["report"]["epoch_invariant"] is True
    assert invariant_payload["report"]["task_multiset_invariant"] is True
    assert invariant_payload["report"]["authority_evidence_complete"] is True
    assert scaling_payload["report"]["invariance"]["outcome_invariant"] is True
    assert scaling_payload["report"]["invariance"]["coverage_bitmap_invariant"] is True
    assert invariant_payload["synthetic_authority_evidence"] is True
    assert scaling_payload["synthetic_authority_evidence"] is True
    assert invariant_payload["phase6_gate_eligible"] is False
    assert scaling_payload["phase6_gate_eligible"] is False
    assert all(
        payload["twenty_four_hour_run_authorized"] is False
        for payload in (parity_payload, cluster_payload, invariant_payload, scaling_payload)
    )


@pytest.mark.parametrize(
    "script",
    ["audit_parallel_invariance.py", "benchmark_parallel_scaling.py"],
)
@pytest.mark.parametrize("workers", ["1", "1,1", "2,6"])
def test_parallel_scripts_reject_invalid_worker_designs_without_output(
    tmp_path, script, workers
):
    output = tmp_path / f"{script}-{workers.replace(',', '-')}.json"
    arguments = ["--tasks", "4", "--workers", workers, "--output", output]
    if script == "audit_parallel_invariance.py":
        arguments.extend(("--completion-delay-ms", "0"))
    else:
        arguments.extend(("--work-ms", "0"))
    result = _run(script, *arguments, check=False)
    assert result.returncode != 0
    assert "worker count" in result.stderr
    assert not output.exists()


def test_plan_builders_only_materialize_nonexecuting_plans(tmp_path):
    v3 = tmp_path / "v3.json"
    pilot = tmp_path / "pilot.json"
    lanes = ",".join(f"lane-{index}" for index in range(11))
    seeds = ",".join(str(50000000 + index) for index in range(10))
    _run(
        "build_v3_gate_plan.py",
        "--lanes",
        lanes,
        "--seeds",
        "44000001,44000002",
        "--source-digest",
        "source",
        "--protocol-digest",
        "protocol",
        "--output",
        v3,
    )
    _run(
        "build_2h_pilot_plan.py",
        "--treatment",
        "full-osc",
        "--control",
        "baseline",
        "--seed-blocks",
        seeds,
        "--case-cap-per-arm",
        10000,
        "--cpu-seconds-cap-per-arm",
        7200,
        "--source-digest",
        "source",
        "--environment-digest",
        "environment",
        "--target-digest",
        "target",
        "--contract-digest",
        "contract",
        "--phase6-gate-report-digest",
        "phase6",
        "--output",
        pilot,
    )
    v3_payload = json.loads(v3.read_text())
    pilot_payload = json.loads(pilot.read_text())
    assert v3_payload["planned_runs"] == 22
    assert v3_payload["planned_cases"] == 2200
    assert v3_payload["plan"]["execution_authorized"] is False
    assert v3_payload["plan"]["twenty_four_hour_run_authorized"] is False
    assert pilot_payload["plan"]["duration_seconds_per_arm"] == 7200
    assert pilot_payload["plan"]["execution_authorized"] is False
    assert pilot_payload["plan"]["twenty_four_hour_run_authorized"] is False


def test_gate_cli_returns_nonzero_and_writes_evidence_when_inputs_are_missing(tmp_path):
    input_path = tmp_path / "gate-input.json"
    output_path = tmp_path / "gate-output.json"
    input_path.write_text(json.dumps({"source_digest": "source", "metrics": {}, "evidence": {}}))
    authority_path, authority_sha = _write_authority_plan(
        tmp_path, artifact_hash="0" * 64
    )
    result = _run(
        "calculate_gate_report.py",
        "--input",
        input_path,
        "--authority-plan",
        authority_path,
        "--authority-plan-sha256",
        authority_sha,
        "--expected-source-digest",
        "source-snapshot-1",
        "--output",
        output_path,
        check=False,
    )
    assert result.returncode == 2
    report = json.loads(output_path.read_text())
    assert report["all_gates_pass"] is False
    assert report["authority_eligible"] is False
    assert report["complete_frozen_spec_set"] is True
    assert report["twenty_four_hour_run_authorized"] is False
    assert all(item["passed"] is False for item in report["results"])


def test_gate_cli_rejects_self_asserted_source_and_fake_232_cell_artifact(tmp_path):
    fake_ids = [f"not-a-real-cell-{index}" for index in range(232)]
    raw_payload = {
        "schema_version": RAW_GATE_ARTIFACT_SCHEMA_VERSION,
        "artifact_id": "raw-gates",
        "source_digest": "attacker-self-asserted-source",
        "series": [
            {
                "metric_key": key,
                "reducer": "count_unique",
                "records": [
                    {"record_id": record_id, "value": 1} for record_id in fake_ids
                ],
            }
            for key in ("fresh_cells_observed", "fresh_cells_declared")
        ],
    }
    raw = _json_bytes(raw_payload)
    (tmp_path / "raw.json").write_bytes(raw)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(
        _json_bytes(
            {
                "schema_version": GATE_ARTIFACT_MANIFEST_SCHEMA_VERSION,
                "source_digest": "attacker-self-asserted-source",
                "claimed_metrics": {
                    "fresh_cells_observed": 232,
                    "fresh_cells_declared": 232,
                },
                "artifacts": [
                    {
                        "artifact_id": "raw-gates",
                        "path": "raw.json",
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "schema_version": RAW_GATE_ARTIFACT_SCHEMA_VERSION,
                    }
                ],
            }
        )
    )
    authority_path, authority_sha = _write_authority_plan(
        tmp_path, artifact_hash=hashlib.sha256(raw).hexdigest()
    )
    output_path = tmp_path / "attack-report.json"
    result = _run(
        "calculate_gate_report.py",
        "--input",
        manifest_path,
        "--authority-plan",
        authority_path,
        "--authority-plan-sha256",
        authority_sha,
        "--expected-source-digest",
        "source-snapshot-1",
        "--output",
        output_path,
        check=False,
    )
    report = json.loads(output_path.read_text())
    assert result.returncode == 2
    assert report["all_gates_pass"] is False
    assert report["authority_eligible"] is False
    assert "gate_manifest_expected_source_mismatch" in report["verification_errors"]
    assert report["twenty_four_hour_run_authorized"] is False
