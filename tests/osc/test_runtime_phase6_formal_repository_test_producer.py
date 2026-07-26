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
from datadiff_osc._phase6_gate_authority import SourceFileBinding
import datadiff_osc.runtime._phase6_formal_repository_test_producer as producer
from datadiff_osc.runtime._phase6_formal_repository_test_producer import (
    FORMAL_REPOSITORY_TEST_AUTHORITY_SCHEMA_VERSION,
    FORMAL_REPOSITORY_TEST_OUTPUT_FILES,
    _ExecutableBinding,
    _capture as _capture_with_binding,
    _require_full_test_snapshot_closure,
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


class _ClosureSnapshot:
    def __init__(self, *, source_digest: str, test_paths: tuple[str, ...]) -> None:
        self.source_digest = source_digest
        self.file_bindings = tuple(
            SourceFileBinding(relative_path=path, sha256="0" * 64)
            for path in test_paths
        )
        self.valid = True


def _closure_source_tree(
    tmp_path: Path, *, test_files: dict[str, str]
) -> tuple[Path, _ClosureSnapshot, tuple[str, ...]]:
    source_root = tmp_path / "closure-source"
    (source_root / "src").mkdir(parents=True)
    tests_root = source_root / "tests"
    tests_root.mkdir()
    for relative, contents in test_files.items():
        path = tests_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
    test_paths = tuple(sorted("tests/" + relative for relative in test_files))
    source_digest = _source_digest("closure-source")
    return (
        source_root,
        _ClosureSnapshot(source_digest=source_digest, test_paths=test_paths),
        tuple(path for path in test_paths if Path(path).name.startswith("test_")),
    )


def _write_closure_authority(
    tmp_path: Path,
    *,
    source_root: Path,
    source_digest: str,
    targets: tuple[str, ...],
) -> object:
    tmp_path.mkdir(parents=True)
    source_snapshot = tmp_path / "closure-source-snapshot.json"
    source_snapshot.write_text(canonical_json({"version": 1}), encoding="utf-8")
    environment = tuple(
        sorted(
            (
                ("PATH", os.environ["PATH"]),
                ("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1"),
                ("PYTHONDONTWRITEBYTECODE", "1"),
                ("PYTHONPATH", str(source_root / "src")),
            )
        )
    )
    interpreter = Path(sys.executable)
    resolved = interpreter.resolve(strict=True)
    payload: dict[str, object] = {
        "schema_version": FORMAL_REPOSITORY_TEST_AUTHORITY_SCHEMA_VERSION,
        "authority_id": "phase6-repository-test-closure",
        "source_snapshot_path": str(source_snapshot),
        "source_snapshot_sha256": hashlib.sha256(source_snapshot.read_bytes()).hexdigest(),
        "source_digest": source_digest,
        "source_root": str(source_root),
        "interpreter_path": str(interpreter),
        "interpreter_resolved_path": str(resolved),
        "interpreter_sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
        "test_targets": list(targets),
        "test_plan_digest": repository_test_plan_digest(
            source_digest=source_digest, test_targets=targets
        ),
        "environment": [list(item) for item in environment],
        "command_digest": repository_test_command_digest(
            source_root=source_root,
            interpreter=interpreter,
            test_targets=targets,
            environment=environment,
        ),
        "output_root": str(tmp_path / "closure-formal-output"),
        "timeout_seconds": 20.0,
        "formal_execution_authorized": True,
        "gate_credit": False,
        "candidate_confirmed": False,
        "bug_claimed": False,
    }
    authority_path = tmp_path / "closure-authority.json"
    authority_path.write_text(canonical_json(payload), encoding="utf-8")
    return load_formal_repository_test_execution_authority(
        path=authority_path,
        expected_sha256=hashlib.sha256(authority_path.read_bytes()).hexdigest(),
    )


def test_full_snapshot_closure_rejects_unbound_tree_and_target_variants(
    tmp_path: Path,
):
    source_root, snapshot, targets = _closure_source_tree(
        tmp_path / "baseline",
        test_files={
            "conftest.py": "",
            "test_alpha.py": "def test_alpha():\n    assert True\n",
            "nested/test_beta.py": "def test_beta():\n    assert True\n",
        },
    )
    assert targets == ("tests/nested/test_beta.py", "tests/test_alpha.py")
    _require_full_test_snapshot_closure(
        source_root=source_root, snapshot=snapshot, test_targets=targets
    )
    with pytest.raises(ValueError, match="targets do not exactly cover"):
        _require_full_test_snapshot_closure(
            source_root=source_root,
            snapshot=snapshot,
            test_targets=("tests/test_alpha.py",),
        )
    with pytest.raises(ValueError, match="targets do not exactly cover"):
        _require_full_test_snapshot_closure(
            source_root=source_root,
            snapshot=snapshot,
            test_targets=("tests",),
        )
    with pytest.raises(ValueError, match="targets do not exactly cover"):
        _require_full_test_snapshot_closure(
            source_root=source_root,
            snapshot=snapshot,
            test_targets=targets + ("tests/test_unbound.py",),
        )

    extra_test_root, extra_test_snapshot, extra_test_targets = _closure_source_tree(
        tmp_path / "extra-test",
        test_files={"test_alpha.py": "def test_alpha():\n    assert True\n"},
    )
    (extra_test_root / "tests/test_added.py").write_text(
        "def test_added():\n    assert True\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="test-tree snapshot closure mismatch"):
        _require_full_test_snapshot_closure(
            source_root=extra_test_root,
            snapshot=extra_test_snapshot,
            test_targets=extra_test_targets,
        )

    conftest_root, conftest_snapshot, conftest_targets = _closure_source_tree(
        tmp_path / "extra-conftest",
        test_files={"test_alpha.py": "def test_alpha():\n    assert True\n"},
    )
    (conftest_root / "tests/conftest.py").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="test-tree snapshot closure mismatch"):
        _require_full_test_snapshot_closure(
            source_root=conftest_root,
            snapshot=conftest_snapshot,
            test_targets=conftest_targets,
        )

    deletion_root, deletion_snapshot, deletion_targets = _closure_source_tree(
        tmp_path / "deletion",
        test_files={"test_alpha.py": "def test_alpha():\n    assert True\n"},
    )
    (deletion_root / "tests/test_alpha.py").unlink()
    with pytest.raises(ValueError, match="test tree has no regular files"):
        _require_full_test_snapshot_closure(
            source_root=deletion_root,
            snapshot=deletion_snapshot,
            test_targets=deletion_targets,
        )

    symlink_root, symlink_snapshot, symlink_targets = _closure_source_tree(
        tmp_path / "symlink",
        test_files={"test_alpha.py": "def test_alpha():\n    assert True\n"},
    )
    (symlink_root / "tests/test_link.py").symlink_to(
        symlink_root / "tests/test_alpha.py"
    )
    with pytest.raises(ValueError, match="test tree contains a symlink"):
        _require_full_test_snapshot_closure(
            source_root=symlink_root,
            snapshot=symlink_snapshot,
            test_targets=symlink_targets,
        )


def test_formal_execution_closure_blocks_pre_and_post_capture_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    pre_root, pre_snapshot, pre_targets = _closure_source_tree(
        tmp_path / "pre-capture",
        test_files={"test_alpha.py": "def test_alpha():\n    assert True\n"},
    )
    pre_authority = _write_closure_authority(
        tmp_path / "pre-authority",
        source_root=pre_root,
        source_digest=pre_snapshot.source_digest,
        targets=pre_targets,
    )
    (pre_root / "tests/conftest.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(
        producer, "verify_source_snapshot", lambda **_: pre_snapshot
    )

    def pre_capture_must_not_run(**_: object) -> object:
        raise AssertionError("_capture must not run for pre-capture closure drift")

    monkeypatch.setattr(producer, "_capture", pre_capture_must_not_run)
    with pytest.raises(ValueError, match="test-tree snapshot closure mismatch"):
        producer.execute_formal_repository_test(authority=pre_authority)
    assert not Path(pre_authority.output_root).exists()

    post_root, post_snapshot, post_targets = _closure_source_tree(
        tmp_path / "post-capture",
        test_files={"test_alpha.py": "def test_alpha():\n    assert True\n"},
    )
    post_authority = _write_closure_authority(
        tmp_path / "post-authority",
        source_root=post_root,
        source_digest=post_snapshot.source_digest,
        targets=post_targets,
    )
    calls = {"capture": 0, "verify_result": 0}

    def post_capture_mutation(**_: object) -> object:
        calls["capture"] += 1
        (post_root / "tests/test_added_after_capture.py").write_text(
            "def test_added_after_capture():\n    assert True\n", encoding="utf-8"
        )
        return object()

    def result_must_not_be_admitted(**_: object) -> object:
        calls["verify_result"] += 1
        raise AssertionError("post-capture closure drift reached result admission")

    monkeypatch.setattr(
        producer, "verify_source_snapshot", lambda **_: post_snapshot
    )
    monkeypatch.setattr(producer, "_capture", post_capture_mutation)
    monkeypatch.setattr(producer, "verify_repository_test_capture", result_must_not_be_admitted)
    with pytest.raises(ValueError, match="test-tree snapshot closure mismatch"):
        producer.execute_formal_repository_test(authority=post_authority)
    assert calls == {"capture": 1, "verify_result": 0}
    assert not Path(post_authority.output_root).exists()


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


def test_negative_review_rejects_fabricated_escape_reuse_and_symlink(tmp_path: Path):
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
