from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from datadiff_osc._canonical import canonical_json, stable_digest
from datadiff_osc.runtime._phase6_formal_repository_test_producer import (
    FORMAL_REPOSITORY_TEST_AUTHORITY_SCHEMA_VERSION,
    FORMAL_REPOSITORY_TEST_OUTPUT_FILES,
    _ExecutableBinding,
    _capture as _capture_with_binding,
    capture_test_only_repository_tests,
    formal_repository_test_producer_preview,
    load_formal_repository_test_execution_authority,
    repository_test_command_digest,
    repository_test_plan_digest,
    verify_repository_test_capture,
)


ROOT = Path(__file__).resolve().parents[2]


def _source_digest(name: str) -> str:
    return stable_digest("osc-phase6-formal-repository-test-test-source", {"name": name})


def _project(tmp_path: Path, *, name: str = "project") -> Path:
    project = tmp_path / name
    project.mkdir()
    (project / "test_example.py").write_text(
        "def test_alpha():\n    assert 2 + 2 == 4\n\n"
        "def test_beta():\n    assert 'osc'.upper() == 'OSC'\n",
        encoding="utf-8",
    )
    return project


def _capture(tmp_path: Path, *, name: str = "project"):
    project = _project(tmp_path, name=name)
    return capture_test_only_repository_tests(
        source_digest=_source_digest(name),
        source_root=ROOT,
        test_root=project,
        test_targets=("test_example.py",),
        output_root=project / "formal-output",
        timeout_seconds=20,
    )


def _authority_environment() -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (
                ("PATH", os.environ["PATH"]),
                ("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1"),
                ("PYTHONDONTWRITEBYTECODE", "1"),
                ("PYTHONPATH", str(ROOT / "src")),
            )
        )
    )


def _write_test_authority(
    tmp_path: Path, *, name: str, interpreter: Path
) -> tuple[Path, str, dict[str, object]]:
    """Create an ephemeral parser fixture only; it never authorizes execution."""

    source_snapshot = tmp_path / f"{name}-source-snapshot.json"
    source_snapshot.write_text(canonical_json({"version": 1}), encoding="utf-8")
    source_digest = _source_digest(f"{name}-authority")
    targets = ("tests",)
    environment = _authority_environment()
    resolved = interpreter.resolve(strict=True)
    target_sha256 = hashlib.sha256(resolved.read_bytes()).hexdigest()
    payload: dict[str, object] = {
        "schema_version": FORMAL_REPOSITORY_TEST_AUTHORITY_SCHEMA_VERSION,
        "authority_id": f"phase6-repository-test-{name}",
        "source_snapshot_path": str(source_snapshot),
        "source_snapshot_sha256": hashlib.sha256(source_snapshot.read_bytes()).hexdigest(),
        "source_digest": source_digest,
        "source_root": str(ROOT),
        "interpreter_path": str(interpreter),
        "interpreter_resolved_path": str(resolved),
        "interpreter_sha256": target_sha256,
        "test_targets": list(targets),
        "test_plan_digest": repository_test_plan_digest(
            source_digest=source_digest, test_targets=targets
        ),
        "environment": [list(item) for item in environment],
        "command_digest": repository_test_command_digest(
            source_root=ROOT,
            interpreter=interpreter,
            test_targets=targets,
            environment=environment,
        ),
        "output_root": str(tmp_path / f"{name}-formal-output"),
        "timeout_seconds": 20.0,
        "formal_execution_authorized": True,
        "gate_credit": False,
        "candidate_confirmed": False,
        "bug_claimed": False,
    }
    authority = tmp_path / f"{name}-authority.json"
    authority.write_text(canonical_json(payload), encoding="utf-8")
    return authority, hashlib.sha256(authority.read_bytes()).hexdigest(), payload


def _replace_authority_payload(path: Path, payload: dict[str, object]) -> str:
    path.write_text(canonical_json(payload), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _replace_authority_raw(path: Path, raw: bytes) -> str:
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def test_real_miniproject_capture_is_test_only_and_self_replayed(tmp_path: Path):
    result = _capture(tmp_path)

    assert result.test_only is True
    assert result.diagnostic_only is True
    assert result.authority_eligible is False
    assert result.formal_evidence_created is False
    assert result.gate_credit is False
    assert result.candidate_confirmed is False
    assert result.bug_claimed is False
    assert result.receipt.exit_code == 0
    assert result.receipt.node_ids == (
        "test_example.py::test_alpha",
        "test_example.py::test_beta",
    )
    assert (
        tuple(sorted(path.name for path in Path(result.output_root).iterdir()))
        == FORMAL_REPOSITORY_TEST_OUTPUT_FILES
    )
    assert verify_repository_test_capture(
        result=result,
        expected_source_digest=_source_digest("project"),
        expected_test_only=True,
    ) == result


def test_interpreter_link_binding_is_exact_and_fails_closed(tmp_path: Path):
    current_launcher = Path(sys.executable)
    assert current_launcher.is_absolute()
    assert current_launcher.is_symlink()
    authority, authority_sha256, payload = _write_test_authority(
        tmp_path, name="current-launcher", interpreter=current_launcher
    )
    loaded = load_formal_repository_test_execution_authority(
        path=authority, expected_sha256=authority_sha256
    )
    assert loaded.interpreter_path == str(current_launcher)
    assert loaded.interpreter_resolved_path == str(current_launcher.resolve(strict=True))
    assert loaded.interpreter_sha256 == hashlib.sha256(
        current_launcher.resolve(strict=True).read_bytes()
    ).hexdigest()
    assert loaded.timeout_seconds == 20.0
    assert not Path(loaded.output_root).exists()

    old_schema = dict(payload)
    old_schema["schema_version"] = (
        "osc-root-phase6-formal-repository-test-execution-authority-v2"
    )
    old_schema_sha256 = _replace_authority_payload(authority, old_schema)
    with pytest.raises(ValueError, match="authority schema version mismatch"):
        load_formal_repository_test_execution_authority(
            path=authority, expected_sha256=old_schema_sha256
        )

    def assert_timeout_rejected(value: object, pattern: str) -> None:
        timeout_payload = dict(payload)
        timeout_payload["timeout_seconds"] = value
        timeout_sha256 = _replace_authority_payload(authority, timeout_payload)
        with pytest.raises(ValueError, match=pattern):
            load_formal_repository_test_execution_authority(
                path=authority, expected_sha256=timeout_sha256
            )

    assert_timeout_rejected(20, "canonical finite-float object")
    assert_timeout_rejected(True, "canonical finite-float object")
    assert_timeout_rejected(None, "canonical finite-float object")
    assert_timeout_rejected([], "canonical finite-float object")
    assert_timeout_rejected({"$float": "nan"}, "finite positive canonical float")
    assert_timeout_rejected({"$float": "+inf"}, "finite positive canonical float")
    assert_timeout_rejected({"$float": "-inf"}, "finite positive canonical float")
    assert_timeout_rejected({"$float": "-0"}, "finite positive canonical float")
    assert_timeout_rejected({"$float": "0x0.0p+0"}, "must be in \\(0, 3600\\]")
    assert_timeout_rejected(
        {"$float": "0x1.0000000000000p+12"}, "must be in \\(0, 3600\\]"
    )
    assert_timeout_rejected({"$float": "not-a-float"}, "canonical float is malformed")
    assert_timeout_rejected(
        {"$float": "0x1.400p+4"}, "canonical float is not normalized"
    )
    assert_timeout_rejected(
        {"$float": "0x1.4000000000000p+4", "extra": "value"},
        "canonical finite-float object",
    )

    native_timeout = dict(payload)
    native_timeout["timeout_seconds"] = 20.0
    native_timeout_raw = json.dumps(
        native_timeout,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    native_timeout_sha256 = _replace_authority_raw(authority, native_timeout_raw)
    with pytest.raises(ValueError, match="document is not canonical JSON"):
        load_formal_repository_test_execution_authority(
            path=authority, expected_sha256=native_timeout_sha256
        )
    whitespace_timeout_sha256 = _replace_authority_raw(
        authority, canonical_json(payload).encode("utf-8") + b"\n"
    )
    with pytest.raises(ValueError, match="document is not canonical JSON"):
        load_formal_repository_test_execution_authority(
            path=authority, expected_sha256=whitespace_timeout_sha256
        )
    assert not Path(loaded.output_root).exists()

    target = tmp_path / "bound-python-target"
    target.write_bytes(b"bound-target-v1")
    link = tmp_path / "bound-python"
    link.symlink_to(target)
    bound_authority, bound_sha256, bound_payload = _write_test_authority(
        tmp_path, name="controlled-link", interpreter=link
    )
    loaded_bound = load_formal_repository_test_execution_authority(
        path=bound_authority, expected_sha256=bound_sha256
    )
    assert loaded_bound.interpreter_path == str(link)
    assert loaded_bound.interpreter_resolved_path == str(target)

    target.write_bytes(b"bound-target-v2")
    with pytest.raises(ValueError, match="interpreter target SHA-256 mismatch"):
        load_formal_repository_test_execution_authority(
            path=bound_authority, expected_sha256=bound_sha256
        )
    target.write_bytes(b"bound-target-v1")

    replacement = tmp_path / "replacement-target"
    replacement.write_bytes(b"replacement-target")
    link.unlink()
    link.symlink_to(replacement)
    with pytest.raises(ValueError, match="interpreter resolved path mismatch"):
        load_formal_repository_test_execution_authority(
            path=bound_authority, expected_sha256=bound_sha256
        )
    assert not Path(loaded_bound.output_root).exists()

    stale_resolved = dict(payload)
    stale_resolved["interpreter_resolved_path"] = str(tmp_path / "other-target")
    stale_resolved_sha256 = _replace_authority_payload(authority, stale_resolved)
    with pytest.raises(ValueError, match="interpreter resolved path mismatch"):
        load_formal_repository_test_execution_authority(
            path=authority, expected_sha256=stale_resolved_sha256
        )

    stale_target = dict(payload)
    stale_target["interpreter_sha256"] = "0" * 64
    stale_target_sha256 = _replace_authority_payload(authority, stale_target)
    with pytest.raises(ValueError, match="interpreter target SHA-256 mismatch"):
        load_formal_repository_test_execution_authority(
            path=authority, expected_sha256=stale_target_sha256
        )

    alternate_current_launcher = tmp_path / "alternate-current-launcher"
    alternate_current_launcher.symlink_to(current_launcher.resolve(strict=True))
    stale_declared = dict(payload)
    stale_declared["interpreter_path"] = str(alternate_current_launcher)
    stale_declared_sha256 = _replace_authority_payload(authority, stale_declared)
    with pytest.raises(ValueError, match="command digest mismatch"):
        load_formal_repository_test_execution_authority(
            path=authority, expected_sha256=stale_declared_sha256
        )

    malformed_sha = dict(payload)
    malformed_sha["interpreter_sha256"] = "not-a-sha"
    malformed_sha256 = _replace_authority_payload(authority, malformed_sha)
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        load_formal_repository_test_execution_authority(
            path=authority, expected_sha256=malformed_sha256
        )

    unordered_environment = dict(payload)
    unordered_environment["environment"] = list(reversed(payload["environment"]))
    unordered_sha256 = _replace_authority_payload(authority, unordered_environment)
    with pytest.raises(ValueError, match="environment values must be sorted"):
        load_formal_repository_test_execution_authority(
            path=authority, expected_sha256=unordered_sha256
        )

    duplicate = tmp_path / "duplicate-authority.json"
    duplicate.write_text(
        '{"schema_version":"one","schema_version":"two"}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="not strict JSON"):
        load_formal_repository_test_execution_authority(
            path=duplicate,
            expected_sha256=hashlib.sha256(duplicate.read_bytes()).hexdigest(),
        )
    assert not Path(loaded.output_root).exists()

    capture_project = _project(tmp_path, name="capture-binding-project")
    capture_target = tmp_path / "capture-target"
    capture_target.write_bytes(b"capture-target-v1")
    capture_link = tmp_path / "capture-link"
    capture_link.symlink_to(capture_target)
    expected_capture_binding = _ExecutableBinding(
        declared_path=str(capture_link),
        resolved_path=str(capture_target),
        target_sha256=hashlib.sha256(capture_target.read_bytes()).hexdigest(),
    )
    capture_replacement = tmp_path / "capture-replacement"
    capture_replacement.write_bytes(b"capture-replacement")
    capture_link.unlink()
    capture_link.symlink_to(capture_replacement)
    capture_output = capture_project / "capture-output"
    with pytest.raises(ValueError, match="capture interpreter resolved path mismatch"):
        _capture_with_binding(
            source_digest=_source_digest("capture-link-replacement"),
            authority_id="phase6-test-only-link-binding",
            source_root=ROOT,
            test_root=capture_project,
            interpreter=capture_link,
            test_targets=("test_example.py",),
            environment=_authority_environment(),
            output_root=capture_output,
            timeout_seconds=20,
            test_only=True,
            expected_interpreter_binding=expected_capture_binding,
        )
    assert not capture_output.exists()

    capture_link.unlink()
    capture_link.symlink_to(capture_target)
    capture_target.write_bytes(b"capture-target-v2")
    with pytest.raises(ValueError, match="capture interpreter target SHA-256 mismatch"):
        _capture_with_binding(
            source_digest=_source_digest("capture-target-replacement"),
            authority_id="phase6-test-only-target-binding",
            source_root=ROOT,
            test_root=capture_project,
            interpreter=capture_link,
            test_targets=("test_example.py",),
            environment=_authority_environment(),
            output_root=capture_output,
            timeout_seconds=20,
            test_only=True,
            expected_interpreter_binding=expected_capture_binding,
        )
    assert not capture_output.exists()

    broken = tmp_path / "broken-python"
    broken.symlink_to(tmp_path / "does-not-exist")
    with pytest.raises(ValueError, match="cannot be resolved"):
        repository_test_command_digest(
            source_root=ROOT,
            interpreter=broken,
            test_targets=("tests",),
            environment=_authority_environment(),
        )
    with pytest.raises(ValueError, match="must be absolute"):
        repository_test_command_digest(
            source_root=ROOT,
            interpreter=Path("relative-python"),
            test_targets=("tests",),
            environment=_authority_environment(),
        )
    loop_a = tmp_path / "loop-a"
    loop_b = tmp_path / "loop-b"
    loop_a.symlink_to(loop_b)
    loop_b.symlink_to(loop_a)
    with pytest.raises(ValueError, match="cannot be resolved"):
        repository_test_command_digest(
            source_root=ROOT,
            interpreter=loop_a,
            test_targets=("tests",),
            environment=_authority_environment(),
        )
    with pytest.raises(ValueError, match="non-empty immutable tuple"):
        repository_test_command_digest(
            source_root=ROOT,
            interpreter=current_launcher,
            test_targets=("tests",),
            environment=list(_authority_environment()),
        )


def test_negative_review_rejects_synthetic_escape_reuse_and_symlink(tmp_path: Path):
    project = _project(tmp_path)
    output = project / "formal-output"
    with pytest.raises(ValueError, match="synthetic marker"):
        capture_test_only_repository_tests(
            source_digest="synthetic-source",
            source_root=ROOT,
            test_root=project,
            test_targets=("test_example.py",),
            output_root=output,
        )
    with pytest.raises(ValueError, match="safe relative pytest target"):
        capture_test_only_repository_tests(
            source_digest=_source_digest("escape"),
            source_root=ROOT,
            test_root=project,
            test_targets=("../test_example.py",),
            output_root=output,
        )
    output.mkdir()
    with pytest.raises(FileExistsError, match="output already exists"):
        capture_test_only_repository_tests(
            source_digest=_source_digest("existing"),
            source_root=ROOT,
            test_root=project,
            test_targets=("test_example.py",),
            output_root=output,
        )
    output.rmdir()
    sink = tmp_path / "sink"
    sink.mkdir()
    os.symlink(sink, output)
    with pytest.raises(FileExistsError, match="output already exists"):
        capture_test_only_repository_tests(
            source_digest=_source_digest("symlink"),
            source_root=ROOT,
            test_root=project,
            test_targets=("test_example.py",),
            output_root=output,
        )


def test_exceptional_input_validation_detects_hash_identity_and_xml_mutations(
    tmp_path: Path,
):
    result = _capture(tmp_path, name="first")
    collection = Path(result.output_root) / "collection.json"
    collection.write_bytes(collection.read_bytes() + b" ")
    with pytest.raises(ValueError, match="collection SHA-256 mismatch"):
        verify_repository_test_capture(
            result=result,
            expected_source_digest=_source_digest("first"),
            expected_test_only=True,
        )

    identity_result = _capture(tmp_path, name="second")
    identity_collection = Path(identity_result.output_root) / "collection.json"
    payload = json.loads(identity_collection.read_text(encoding="utf-8"))
    payload["nodes"] = payload["nodes"][:-1]
    mutated_collection = canonical_json(payload).encode("utf-8")
    identity_collection.write_bytes(mutated_collection)
    with pytest.raises(ValueError, match="collection/result identities disagree"):
        verify_repository_test_capture(
            result=replace(
                identity_result,
                collection_sha256=hashlib.sha256(mutated_collection).hexdigest(),
            ),
            expected_source_digest=_source_digest("second"),
            expected_test_only=True,
        )

    xml_result = _capture(tmp_path, name="third")
    junit = Path(xml_result.output_root) / "junit.xml"
    mutated_xml = b"<testsuites><testsuite>"
    junit.write_bytes(mutated_xml)
    with pytest.raises(ValueError, match="JUnit output is malformed"):
        verify_repository_test_capture(
            result=replace(
                xml_result,
                junit_sha256=hashlib.sha256(mutated_xml).hexdigest(),
            ),
            expected_source_digest=_source_digest("third"),
            expected_test_only=True,
        )


def test_independent_counterexample_and_cli_remain_no_execution(tmp_path: Path):
    preview = formal_repository_test_producer_preview()
    assert preview["default_executes_repository_tests"] is False
    assert preview["writes_durable_output_without_authority"] is False
    assert preview["dynamic_plan_emitted"] is False
    assert preview["artifact_receipt_index_emitted"] is False
    assert preview["gate_credit"] is False
    assert preview["candidate_confirmed"] is False
    assert preview["bug_claimed"] is False

    script = ROOT / "scripts/osc/run_phase6_formal_repository_test_producer.py"
    completed = subprocess.run(
        (sys.executable, str(script)),
        cwd=tmp_path,
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode == 0
    assert completed.stderr == ""
    assert json.loads(completed.stdout) == preview
    rejected = subprocess.run(
        (sys.executable, str(script), "--execute"),
        cwd=tmp_path,
        capture_output=True,
        check=False,
        text=True,
    )
    assert rejected.returncode == 2
    assert "requires --authority and --authority-sha256" in rejected.stderr
    assert list(tmp_path.iterdir()) == []

    malformed = tmp_path / "authority.json"
    malformed.write_text(canonical_json({"schema_version": "wrong"}), encoding="utf-8")
    with pytest.raises(ValueError, match="fields do not match schema"):
        load_formal_repository_test_execution_authority(
            path=malformed,
            expected_sha256=hashlib.sha256(malformed.read_bytes()).hexdigest(),
        )
    with pytest.raises(ValueError, match="authority document SHA-256 mismatch"):
        load_formal_repository_test_execution_authority(
            path=malformed,
            expected_sha256="0" * 64,
        )
