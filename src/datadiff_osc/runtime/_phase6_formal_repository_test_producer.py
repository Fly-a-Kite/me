"""Private, authority-bound repository-test receipt production for Phase 6.

The module has two deliberately separate paths.  A pytest ``tmp_path`` may
exercise the complete collection/run/parser path in ``test_only`` mode.  That
result is permanently diagnostic-only.  A durable result requires a separately
written and SHA-bound authority document plus a verified clean source snapshot.
Neither path grants a gate result, candidate confirmation, or bug claim.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ElementTree
from typing import Any

from datadiff_osc._canonical import (
    assert_deeply_immutable,
    canonical_envelope,
    canonical_json,
    decode_canonical_envelope,
    stable_digest,
)
from datadiff_osc._phase6_gate_authority import verify_source_snapshot
from datadiff_osc.runtime._private_receipts import (
    REPOSITORY_TEST_RECEIPT_SCHEMA_VERSION,
    RepositoryTestNodeBinding,
    RepositoryTestReceipt,
    repository_test_collection_digest,
)
from datadiff_osc.runtime._semantic_replay import replay_runtime_admission


FORMAL_REPOSITORY_TEST_PRODUCER_SCHEMA_VERSION = (
    "osc-private-phase6-formal-repository-test-producer-v2"
)
FORMAL_REPOSITORY_TEST_AUTHORITY_SCHEMA_VERSION = (
    "osc-root-phase6-formal-repository-test-execution-authority-v3"
)
FORMAL_REPOSITORY_TEST_REPORTER_SCHEMA_VERSION = (
    "osc-private-phase6-formal-repository-test-reporter-v2"
)
FORMAL_REPOSITORY_TEST_OUTPUT_FILES = (
    "collection.json",
    "junit.xml",
    "repository_test_receipt.json",
    "run.log",
)
_ALLOWED_ENVIRONMENT_KEYS = frozenset(
    {
        "LANG",
        "LC_ALL",
        "PATH",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONPATH",
        "TZ",
    }
)
_OUTCOMES = frozenset(
    {"passed", "failed", "error", "skipped", "xfailed", "xpassed"}
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class _ExecutableBinding:
    """The non-substitutable identity of one interpreter invocation path."""

    declared_path: str
    resolved_path: str
    target_sha256: str

    def __post_init__(self) -> None:
        _require_text(self.declared_path, name="repository-test interpreter declared path")
        _require_text(self.resolved_path, name="repository-test interpreter resolved path")
        _require_sha256(
            self.target_sha256, name="repository-test interpreter target SHA-256"
        )
        assert_deeply_immutable(self)


def _require_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{name} must be non-empty text without NUL")
    return value


def _require_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _require_positive_timeout(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number")
    timeout = float(value)
    if not math.isfinite(timeout) or not 0.0 < timeout <= 3600.0:
        raise ValueError(f"{name} must be in (0, 3600]")
    return timeout


def _resolved_directory(value: object, *, name: str) -> Path:
    path = Path(_require_text(str(value), name=name))
    if not path.is_absolute():
        raise ValueError(f"{name} must be absolute")
    if path.is_symlink():
        raise ValueError(f"{name} must not be a symlink")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{name} cannot be resolved") from exc
    if not resolved.is_dir():
        raise ValueError(f"{name} must resolve to a directory")
    return resolved


def _resolved_regular_file(value: object, *, name: str) -> Path:
    path = Path(_require_text(str(value), name=name))
    if not path.is_absolute():
        raise ValueError(f"{name} must be absolute")
    if path.is_symlink():
        raise ValueError(f"{name} must not be a symlink")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{name} cannot be resolved") from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"{name} must resolve to a regular non-symlink file")
    return resolved


def _executable_binding(value: object, *, name: str) -> _ExecutableBinding:
    """Resolve an executable without discarding its declared-path identity.

    A virtual-environment launcher commonly is a relative symlink chain.  It is
    safe only when the caller binds both the lexical absolute launcher path and
    the final regular-file target plus its content hash.  The caller can then
    compare a fresh binding at every execution boundary instead of treating the
    resolved target as a substitute for the declared launcher.
    """

    raw_value = os.fspath(value) if isinstance(value, os.PathLike) else value
    raw = _require_text(raw_value, name=name)
    declared = Path(raw)
    if not declared.is_absolute():
        raise ValueError(f"{name} must be absolute")
    declared = Path(os.path.abspath(str(declared)))
    if raw != str(declared):
        raise ValueError(f"{name} must be a canonical absolute path")
    try:
        resolved = declared.resolve(strict=True)
        resolved_status = os.lstat(resolved)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"{name} cannot be resolved") from exc
    if not stat.S_ISREG(resolved_status.st_mode):
        raise ValueError(f"{name} must resolve to a regular file")
    try:
        target_sha256 = _sha256(resolved.read_bytes())
    except OSError as exc:
        raise ValueError(f"{name} target cannot be read") from exc
    return _ExecutableBinding(
        declared_path=str(declared),
        resolved_path=str(resolved),
        target_sha256=target_sha256,
    )


def _require_exact_executable_binding(
    *, actual: _ExecutableBinding, expected: _ExecutableBinding, name: str
) -> None:
    if actual.declared_path != expected.declared_path:
        raise ValueError(f"{name} declared path mismatch")
    if actual.resolved_path != expected.resolved_path:
        raise ValueError(f"{name} resolved path mismatch")
    if actual.target_sha256 != expected.target_sha256:
        raise ValueError(f"{name} target SHA-256 mismatch")


def _reject_synthetic(*values: str) -> None:
    if any("synthetic" in value.lower() for value in values):
        raise ValueError("synthetic marker is not admissible for formal evidence")


def _safe_target(value: object, *, name: str) -> str:
    target = _require_text(value, name=name)
    path = PurePosixPath(target)
    if path.is_absolute() or ".." in path.parts or target.startswith("-"):
        raise ValueError(f"{name} must be a safe relative pytest target")
    return target


def _normalized_targets(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{name} must be a non-empty immutable tuple")
    targets = tuple(_safe_target(item, name=name) for item in value)
    if targets != tuple(sorted(targets)) or len(targets) != len(set(targets)):
        raise ValueError(f"{name} must be uniquely sorted")
    return targets


def _is_test_tree_path(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(path.parts) and path.parts[0] == "tests" and len(path.parts) > 1


def _is_test_module_path(value: str) -> bool:
    path = PurePosixPath(value)
    return _is_test_tree_path(value) and path.name.startswith("test_") and path.suffix == ".py"


def _current_test_tree_paths(source_root: Path) -> tuple[str, ...]:
    """Return the exact regular test-tree surface for formal execution checks."""

    root = _resolved_directory(source_root, name="repository-test source root")
    test_root = root / "tests"
    if test_root.is_symlink():
        raise ValueError("repository-test test tree must not be a symlink")
    if not test_root.is_dir():
        raise ValueError("repository-test test tree is unavailable")
    paths: list[str] = []
    for candidate in test_root.rglob("*"):
        if candidate.is_symlink():
            raise ValueError("repository-test test tree contains a symlink")
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(root).as_posix()
        if "__pycache__" in candidate.parts or candidate.suffix in {".pyc", ".pyo"}:
            continue
        paths.append(relative)
    result = tuple(sorted(paths))
    if not result:
        raise ValueError("repository-test test tree has no regular files")
    if len(result) != len(set(result)):
        raise ValueError("repository-test test tree paths are not unique")
    return result


def _snapshot_test_tree_paths(snapshot: object) -> tuple[str, ...]:
    bindings = getattr(snapshot, "file_bindings", None)
    if not isinstance(bindings, tuple):
        raise ValueError("repository-test source snapshot bindings are invalid")
    paths: list[str] = []
    for binding in bindings:
        relative = getattr(binding, "relative_path", None)
        if not isinstance(relative, str):
            raise ValueError("repository-test source snapshot test binding is invalid")
        if _is_test_tree_path(relative):
            paths.append(relative)
    result = tuple(sorted(paths))
    if not result:
        raise ValueError("repository-test source snapshot has no test-tree bindings")
    if len(result) != len(set(result)):
        raise ValueError("repository-test source snapshot test bindings are not unique")
    return result


def _require_full_test_snapshot_closure(
    *, source_root: Path, snapshot: object, test_targets: tuple[str, ...]
) -> None:
    """Fail closed unless the formal plan covers exactly the bound test tree."""

    expected_paths = _snapshot_test_tree_paths(snapshot)
    actual_paths = _current_test_tree_paths(source_root)
    if actual_paths != expected_paths:
        missing = tuple(sorted(set(expected_paths) - set(actual_paths)))
        extra = tuple(sorted(set(actual_paths) - set(expected_paths)))
        raise ValueError(
            "repository-test test-tree snapshot closure mismatch:"
            f"missing={','.join(missing)};extra={','.join(extra)}"
        )
    expected_targets = tuple(
        path for path in expected_paths if _is_test_module_path(path)
    )
    if not expected_targets:
        raise ValueError("repository-test source snapshot has no test-module bindings")
    targets = _normalized_targets(
        test_targets, name="repository-test authority targets"
    )
    if targets != expected_targets:
        missing = tuple(sorted(set(expected_targets) - set(targets)))
        extra = tuple(sorted(set(targets) - set(expected_targets)))
        raise ValueError(
            "repository-test authority targets do not exactly cover the snapshot "
            f"test modules:missing={','.join(missing)};extra={','.join(extra)}"
        )


def _normalized_environment(value: object, *, source_root: Path) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, tuple) or not value:
        raise ValueError("repository-test environment must be a non-empty immutable tuple")
    result: list[tuple[str, str]] = []
    for item in value:
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not isinstance(item[1], str)
        ):
            raise ValueError("repository-test environment values must be text pairs")
        key = _require_text(item[0], name="repository-test environment key")
        field_value = _require_text(item[1], name="repository-test environment value")
        if key not in _ALLOWED_ENVIRONMENT_KEYS:
            raise ValueError(f"repository-test environment key is not allowlisted: {key}")
        result.append((key, field_value))
    normalized = tuple(result)
    if normalized != tuple(sorted(normalized)):
        raise ValueError("repository-test environment values must be sorted")
    if len({key for key, _ in normalized}) != len(normalized):
        raise ValueError("repository-test environment keys must be unique")
    values = dict(normalized)
    if values.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise ValueError("repository-test environment must disable bytecode writes")
    expected_pythonpath = str(source_root / "src")
    if values.get("PYTHONPATH") != expected_pythonpath:
        raise ValueError("repository-test PYTHONPATH must bind exactly to source_root/src")
    if not values.get("PATH"):
        raise ValueError("repository-test environment must contain PATH")
    return normalized


def _source_environment(source_root: Path) -> tuple[tuple[str, str], ...]:
    path = os.environ.get("PATH", "")
    if not path:
        raise ValueError("host PATH is unavailable for repository-test capture")
    return tuple(
        sorted(
            (
                ("PATH", path),
                ("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1"),
                ("PYTHONDONTWRITEBYTECODE", "1"),
                ("PYTHONPATH", str(source_root / "src")),
            )
        )
    )


def _require_no_symlink_components(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.exists() and current.is_symlink():
            raise ValueError(f"repository-test path contains a symlink: {current}")


def _new_output_root(value: object, *, name: str, parent_root: Path | None = None) -> Path:
    path = Path(_require_text(str(value), name=name))
    if not path.is_absolute():
        raise ValueError(f"{name} must be absolute")
    _require_no_symlink_components(path.parent)
    parent = path.parent.resolve(strict=True)
    if not parent.is_dir():
        raise ValueError(f"{name} parent must be a directory")
    destination = parent / path.name
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"repository-test output already exists: {destination}")
    if parent_root is not None:
        try:
            destination.relative_to(parent_root)
        except ValueError as exc:
            raise ValueError(f"{name} escapes its required test-only root") from exc
    return destination


def _strict_json(raw: bytes, *, name: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} has duplicate JSON fields")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"{name} contains non-finite JSON value: {value}")

    try:
        decoded = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{name} is not strict JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{name} must be a JSON object")
    return decoded


def _canonical_authority_timeout(value: object, *, name: str) -> float:
    """Decode the one canonical finite-float representation admitted in authority JSON."""

    if not isinstance(value, dict) or set(value) != {"$float"}:
        raise ValueError(f"{name} must be a canonical finite-float object")
    encoded = value["$float"]
    if not isinstance(encoded, str):
        raise ValueError(f"{name} canonical float must be text")
    if encoded in {"nan", "+inf", "-inf", "-0"}:
        raise ValueError(f"{name} must be a finite positive canonical float")
    try:
        timeout = float.fromhex(encoded)
    except ValueError as exc:
        raise ValueError(f"{name} canonical float is malformed") from exc
    if timeout.hex() != encoded:
        raise ValueError(f"{name} canonical float is not normalized")
    return _require_positive_timeout(timeout, name=name)


def repository_test_plan_digest(
    *, source_digest: str, test_targets: tuple[str, ...]
) -> str:
    """Return the authority-bound identity of one exact repository test plan."""

    source = _require_text(source_digest, name="repository-test source digest")
    _reject_synthetic(source)
    targets = _normalized_targets(test_targets, name="repository-test targets")
    return stable_digest(
        "osc-formal-repository-test-plan-v1",
        {"source_digest": source, "test_targets": targets},
    )


def _repository_test_command_digest_from_binding(
    *,
    source_root: Path,
    executable: _ExecutableBinding,
    test_targets: tuple[str, ...],
    environment: tuple[tuple[str, str], ...],
) -> str:
    return stable_digest(
        "osc-formal-repository-test-command-v2",
        {
            "source_root": str(source_root),
            "interpreter_declared_path": executable.declared_path,
            "interpreter_resolved_path": executable.resolved_path,
            "interpreter_sha256": executable.target_sha256,
            "pytest_arguments": (
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "--disable-warnings",
                "--junitxml=<stage>/junit.xml",
                "-p",
                __name__,
                "--osc-phase6-reporter=<stage>/collection.json",
                *test_targets,
            ),
            "environment": environment,
        },
    )


def repository_test_command_digest(
    *,
    source_root: Path,
    interpreter: Path,
    test_targets: tuple[str, ...],
    environment: tuple[tuple[str, str], ...],
) -> str:
    """Bind the non-temporary command identity used by the producer."""

    root = _resolved_directory(source_root, name="repository-test source root")
    executable = _executable_binding(interpreter, name="repository-test interpreter")
    targets = _normalized_targets(test_targets, name="repository-test targets")
    normalized_environment = _normalized_environment(environment, source_root=root)
    return _repository_test_command_digest_from_binding(
        source_root=root,
        executable=executable,
        test_targets=targets,
        environment=normalized_environment,
    )


@dataclass(frozen=True, slots=True)
class FormalRepositoryTestExecutionAuthority:
    """One external authority document for a future durable repository run."""

    authority_document_path: str
    authority_document_sha256: str
    authority_id: str
    source_snapshot_path: str
    source_snapshot_sha256: str
    source_digest: str
    source_root: str
    interpreter_path: str
    interpreter_resolved_path: str
    interpreter_sha256: str
    test_targets: tuple[str, ...]
    test_plan_digest: str
    environment: tuple[tuple[str, str], ...]
    command_digest: str
    output_root: str
    timeout_seconds: float
    formal_execution_authorized: bool
    gate_credit: bool = False
    candidate_confirmed: bool = False
    bug_claimed: bool = False
    schema_version: str = FORMAL_REPOSITORY_TEST_AUTHORITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        document = _resolved_regular_file(
            self.authority_document_path, name="repository-test authority document"
        )
        _require_sha256(
            self.authority_document_sha256, name="repository-test authority SHA-256"
        )
        if _sha256(document.read_bytes()) != self.authority_document_sha256:
            raise ValueError("repository-test authority document SHA-256 mismatch")
        authority_id = _require_text(self.authority_id, name="repository-test authority ID")
        source_digest = _require_text(
            self.source_digest, name="repository-test authority source digest"
        )
        _reject_synthetic(authority_id, source_digest)
        _resolved_regular_file(
            self.source_snapshot_path, name="repository-test source snapshot"
        )
        _require_sha256(
            self.source_snapshot_sha256, name="repository-test source snapshot SHA-256"
        )
        root = _resolved_directory(
            self.source_root, name="repository-test authority source root"
        )
        executable = _executable_binding(
            self.interpreter_path, name="repository-test authority interpreter"
        )
        resolved_path = _require_text(
            self.interpreter_resolved_path,
            name="repository-test authority interpreter resolved path",
        )
        if resolved_path != executable.resolved_path:
            raise ValueError("repository-test authority interpreter resolved path mismatch")
        _require_sha256(
            self.interpreter_sha256, name="repository-test authority interpreter SHA-256"
        )
        if executable.target_sha256 != self.interpreter_sha256:
            raise ValueError("repository-test authority interpreter target SHA-256 mismatch")
        targets = _normalized_targets(
            self.test_targets, name="repository-test authority targets"
        )
        plan_digest = repository_test_plan_digest(
            source_digest=source_digest, test_targets=targets
        )
        if self.test_plan_digest != plan_digest:
            raise ValueError("repository-test authority test-plan digest mismatch")
        environment = _normalized_environment(self.environment, source_root=root)
        command_digest = repository_test_command_digest(
            source_root=root,
            interpreter=Path(executable.declared_path),
            test_targets=targets,
            environment=environment,
        )
        if self.command_digest != command_digest:
            raise ValueError("repository-test authority command digest mismatch")
        output = Path(_require_text(self.output_root, name="repository-test authority output root"))
        if not output.is_absolute():
            raise ValueError("repository-test authority output root must be absolute")
        _require_no_symlink_components(output.parent)
        _require_positive_timeout(
            self.timeout_seconds, name="repository-test authority timeout"
        )
        if self.schema_version != FORMAL_REPOSITORY_TEST_AUTHORITY_SCHEMA_VERSION:
            raise ValueError("repository-test authority schema version mismatch")
        if self.formal_execution_authorized is not True:
            raise ValueError("repository-test authority must explicitly authorize execution")
        if self.gate_credit or self.candidate_confirmed or self.bug_claimed:
            raise ValueError("repository-test authority cannot grant gate or promotion credit")
        assert_deeply_immutable(self)


def load_formal_repository_test_execution_authority(
    *, path: Path, expected_sha256: str
) -> FormalRepositoryTestExecutionAuthority:
    """Load an exact future authority document without running its workload."""

    document = _resolved_regular_file(path, name="repository-test authority document")
    expected = _require_sha256(
        expected_sha256, name="repository-test authority expected SHA-256"
    )
    raw = document.read_bytes()
    if _sha256(raw) != expected:
        raise ValueError("repository-test authority document SHA-256 mismatch")
    payload = _strict_json(raw, name="repository-test authority document")
    if raw != canonical_json(payload).encode("utf-8"):
        raise ValueError("repository-test authority document is not canonical JSON")
    expected_fields = {
        "schema_version",
        "authority_id",
        "source_snapshot_path",
        "source_snapshot_sha256",
        "source_digest",
        "source_root",
        "interpreter_path",
        "interpreter_resolved_path",
        "interpreter_sha256",
        "test_targets",
        "test_plan_digest",
        "environment",
        "command_digest",
        "output_root",
        "timeout_seconds",
        "formal_execution_authorized",
        "gate_credit",
        "candidate_confirmed",
        "bug_claimed",
    }
    if set(payload) != expected_fields:
        raise ValueError("repository-test authority document fields do not match schema")
    if payload["schema_version"] != FORMAL_REPOSITORY_TEST_AUTHORITY_SCHEMA_VERSION:
        raise ValueError("repository-test authority schema version mismatch")
    targets_value = payload["test_targets"]
    environment_value = payload["environment"]
    if not isinstance(targets_value, list):
        raise ValueError("repository-test authority test_targets must be a list")
    if not isinstance(environment_value, list):
        raise ValueError("repository-test authority environment must be a list")
    try:
        targets = tuple(targets_value)
        environment = tuple(
            tuple(item) if isinstance(item, list) else item for item in environment_value
        )
    except TypeError as exc:
        raise ValueError("repository-test authority contains malformed lists") from exc
    return FormalRepositoryTestExecutionAuthority(
        authority_document_path=str(document),
        authority_document_sha256=expected,
        authority_id=payload["authority_id"],
        source_snapshot_path=payload["source_snapshot_path"],
        source_snapshot_sha256=payload["source_snapshot_sha256"],
        source_digest=payload["source_digest"],
        source_root=payload["source_root"],
        interpreter_path=payload["interpreter_path"],
        interpreter_resolved_path=payload["interpreter_resolved_path"],
        interpreter_sha256=payload["interpreter_sha256"],
        test_targets=targets,
        test_plan_digest=payload["test_plan_digest"],
        environment=environment,
        command_digest=payload["command_digest"],
        output_root=payload["output_root"],
        timeout_seconds=_canonical_authority_timeout(
            payload["timeout_seconds"], name="repository-test authority timeout"
        ),
        formal_execution_authorized=payload["formal_execution_authorized"],
        gate_credit=payload["gate_credit"],
        candidate_confirmed=payload["candidate_confirmed"],
        bug_claimed=payload["bug_claimed"],
        schema_version=payload["schema_version"],
    )


@dataclass(frozen=True, slots=True)
class RepositoryTestCaptureResult:
    """Immutable output of one test-only or authority-bound repository capture."""

    source_digest: str
    authority_id: str
    test_only: bool
    output_root: str
    receipt: RepositoryTestReceipt
    collection_sha256: str
    junit_sha256: str
    log_sha256: str
    receipt_sha256: str
    schema_version: str = FORMAL_REPOSITORY_TEST_PRODUCER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        source = _require_text(self.source_digest, name="repository-test result source digest")
        _reject_synthetic(source)
        _require_text(self.authority_id, name="repository-test result authority ID")
        if not isinstance(self.test_only, bool):
            raise ValueError("repository-test result test_only must be a boolean")
        root = _resolved_directory(self.output_root, name="repository-test result output root")
        if self.receipt.source_digest != source:
            raise ValueError("repository-test result receipt source digest mismatch")
        for name in (
            "collection_sha256",
            "junit_sha256",
            "log_sha256",
            "receipt_sha256",
        ):
            _require_sha256(getattr(self, name), name=f"repository-test result {name}")
        if self.schema_version != FORMAL_REPOSITORY_TEST_PRODUCER_SCHEMA_VERSION:
            raise ValueError("repository-test result schema version mismatch")
        expected_paths = tuple(sorted(item.name for item in root.iterdir()))
        if expected_paths != FORMAL_REPOSITORY_TEST_OUTPUT_FILES:
            raise ValueError("repository-test result output inventory mismatch")
        assert_deeply_immutable(self)

    @property
    def diagnostic_only(self) -> bool:
        return self.test_only

    @property
    def authority_eligible(self) -> bool:
        return not self.test_only

    @property
    def formal_evidence_created(self) -> bool:
        return not self.test_only

    @property
    def gate_credit(self) -> bool:
        return False

    @property
    def candidate_confirmed(self) -> bool:
        return False

    @property
    def bug_claimed(self) -> bool:
        return False

    @property
    def digest(self) -> str:
        return stable_digest("osc-formal-repository-test-capture-result", self)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source_digest": self.source_digest,
            "authority_id": self.authority_id,
            "test_only": self.test_only,
            "output_root": self.output_root,
            "receipt_digest": self.receipt.digest,
            "collection_sha256": self.collection_sha256,
            "junit_sha256": self.junit_sha256,
            "log_sha256": self.log_sha256,
            "receipt_sha256": self.receipt_sha256,
            "authority_eligible": self.authority_eligible,
            "formal_evidence_created": self.formal_evidence_created,
            "gate_credit": False,
            "candidate_confirmed": False,
            "bug_claimed": False,
        }


class _RepositoryTestReporter:
    """A narrow pytest plugin which records actual collection and outcomes."""

    def __init__(self, output_path: Path) -> None:
        self._output_path = output_path
        self._collected: tuple[str, ...] = ()
        self._outcomes: dict[str, str] = {}

    def pytest_collection_finish(self, session: object) -> None:
        items = getattr(session, "items", ())
        node_ids = tuple(getattr(item, "nodeid", None) for item in items)
        if any(not isinstance(item, str) or not item for item in node_ids):
            raise RuntimeError("pytest collection contains an invalid node identity")
        ordered = tuple(sorted(node_ids))
        if len(ordered) != len(set(ordered)):
            raise RuntimeError("pytest collection contains duplicate node identities")
        self._collected = ordered

    def pytest_runtest_logreport(self, report: object) -> None:
        node_id = getattr(report, "nodeid", None)
        when = getattr(report, "when", None)
        outcome = getattr(report, "outcome", None)
        if not isinstance(node_id, str) or not node_id:
            return
        selected: str | None = None
        if when == "call":
            if getattr(report, "wasxfail", None):
                selected = "xpassed" if outcome == "passed" else "xfailed"
            elif outcome in {"passed", "failed", "skipped"}:
                selected = outcome
        elif when in {"setup", "teardown"}:
            if outcome == "failed":
                selected = "error"
            elif outcome == "skipped" and node_id not in self._outcomes:
                selected = "skipped"
        if selected is not None:
            prior = self._outcomes.get(node_id)
            if prior != "error":
                self._outcomes[node_id] = selected

    def pytest_sessionfinish(self, session: object, exitstatus: object) -> None:
        if not self._collected:
            items = getattr(session, "items", ())
            node_ids = tuple(getattr(item, "nodeid", None) for item in items)
            if all(isinstance(item, str) and item for item in node_ids):
                self._collected = tuple(sorted(node_ids))
        nodes = []
        for node_id in self._collected:
            nodes.append(
                {
                    "node_id": node_id,
                    "outcome": self._outcomes.get(node_id, "error"),
                }
            )
        payload = {
            "schema_version": FORMAL_REPOSITORY_TEST_REPORTER_SCHEMA_VERSION,
            "exit_code": int(exitstatus),
            "collected_node_ids": list(self._collected),
            "nodes": nodes,
        }
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        with self._output_path.open("x", encoding="utf-8") as stream:
            stream.write(canonical_json(payload))


def pytest_addoption(parser: object) -> None:
    """Register the private reporter option when pytest loads this module."""

    parser.addoption(
        "--osc-phase6-reporter",
        action="store",
        default="",
        metavar="PATH",
        help="private Phase-6 repository-test reporter path",
    )


def pytest_configure(config: object) -> None:
    reporter_path = config.getoption("--osc-phase6-reporter")
    if not reporter_path:
        return
    path = Path(str(reporter_path))
    config.pluginmanager.register(
        _RepositoryTestReporter(path), "osc-phase6-formal-repository-test-reporter"
    )


def _capture_command(
    *, interpreter: Path, stage: Path, test_targets: tuple[str, ...]
) -> tuple[str, ...]:
    return (
        str(interpreter),
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "--disable-warnings",
        f"--junitxml={stage / 'junit.xml'}",
        "-p",
        __name__,
        f"--osc-phase6-reporter={stage / 'collection.json'}",
        *test_targets,
    )


def _parse_reporter_payload(
    *, raw: bytes, process_exit_code: int
) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
    payload = _strict_json(raw, name="repository-test collection report")
    expected_fields = {
        "schema_version",
        "exit_code",
        "collected_node_ids",
        "nodes",
    }
    if set(payload) != expected_fields:
        raise ValueError("repository-test collection report fields do not match schema")
    if payload["schema_version"] != FORMAL_REPOSITORY_TEST_REPORTER_SCHEMA_VERSION:
        raise ValueError("repository-test collection report schema version mismatch")
    if payload["exit_code"] != process_exit_code:
        raise ValueError("repository-test collection report exit-code mismatch")
    collected_value = payload["collected_node_ids"]
    if not isinstance(collected_value, list) or not collected_value:
        raise ValueError("repository-test collection report has no collected nodes")
    collected = tuple(collected_value)
    if (
        any(not isinstance(item, str) or not item for item in collected)
        or collected != tuple(sorted(collected))
        or len(collected) != len(set(collected))
    ):
        raise ValueError("repository-test collection identities are not uniquely sorted")
    nodes_value = payload["nodes"]
    if not isinstance(nodes_value, list) or not nodes_value:
        raise ValueError("repository-test collection report has no node outcomes")
    nodes: list[tuple[str, str]] = []
    for item in nodes_value:
        if not isinstance(item, dict) or set(item) != {"node_id", "outcome"}:
            raise ValueError("repository-test node outcome fields do not match schema")
        node_id = item["node_id"]
        outcome = item["outcome"]
        if not isinstance(node_id, str) or not node_id or outcome not in _OUTCOMES:
            raise ValueError("repository-test node outcome is invalid")
        nodes.append((node_id, outcome))
    ordered_nodes = tuple(nodes)
    node_ids = tuple(item[0] for item in ordered_nodes)
    if (
        node_ids != tuple(sorted(node_ids))
        or len(node_ids) != len(set(node_ids))
        or node_ids != collected
    ):
        raise ValueError("repository-test collection/result identities disagree")
    return ordered_nodes, collected


def _verify_junit(*, raw: bytes, expected_node_count: int, exit_code: int) -> None:
    if not raw:
        raise ValueError("repository-test JUnit output is empty")
    if b"<!DOCTYPE" in raw.upper() or b"<!ENTITY" in raw.upper():
        raise ValueError("repository-test JUnit output contains prohibited declarations")
    try:
        root = ElementTree.fromstring(raw)
    except ElementTree.ParseError as exc:
        raise ValueError("repository-test JUnit output is malformed") from exc
    testcases = tuple(root.iter("testcase"))
    if len(testcases) != expected_node_count:
        raise ValueError("repository-test JUnit testcase count disagrees with collection")
    if exit_code == 0 and any(
        testcase.find("failure") is not None or testcase.find("error") is not None
        for testcase in testcases
    ):
        raise ValueError("zero repository-test exit contains JUnit failures")


def _replay_receipt(receipt: RepositoryTestReceipt) -> bytes:
    encoded = canonical_envelope(
        "RepositoryTestReceipt", receipt.schema_version, receipt
    ).encode("utf-8")
    decoded = decode_canonical_envelope(encoded.decode("utf-8"))
    errors = replay_runtime_admission(
        envelope_type="RepositoryTestReceipt",
        schema_version=receipt.schema_version,
        payload=decoded["payload"],
        subject_kind="repository_tests",
        subject_ids=receipt.node_ids,
    )
    if errors:
        raise ValueError("repository-test receipt semantic replay failed: " + ";".join(errors))
    return encoded


def _write_new_bytes(path: Path, raw: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"repository-test output already exists: {path}")
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _capture(
    *,
    source_digest: str,
    authority_id: str,
    source_root: Path,
    test_root: Path,
    interpreter: Path,
    test_targets: tuple[str, ...],
    environment: tuple[tuple[str, str], ...],
    output_root: Path,
    timeout_seconds: float,
    test_only: bool,
    expected_interpreter_binding: _ExecutableBinding | None = None,
) -> RepositoryTestCaptureResult:
    source = _require_text(source_digest, name="repository-test capture source digest")
    identity = _require_text(authority_id, name="repository-test capture authority ID")
    _reject_synthetic(source, identity)
    root = _resolved_directory(source_root, name="repository-test capture source root")
    worktree = _resolved_directory(test_root, name="repository-test capture test root")
    targets = _normalized_targets(test_targets, name="repository-test capture targets")
    normalized_environment = _normalized_environment(environment, source_root=root)
    executable = _executable_binding(interpreter, name="repository-test capture interpreter")
    if expected_interpreter_binding is not None:
        if not isinstance(expected_interpreter_binding, _ExecutableBinding):
            raise TypeError("repository-test expected interpreter binding is invalid")
        _require_exact_executable_binding(
            actual=executable,
            expected=expected_interpreter_binding,
            name="repository-test capture interpreter",
        )
    timeout = _require_positive_timeout(
        timeout_seconds, name="repository-test capture timeout"
    )
    if not test_only and worktree != root:
        raise ValueError("formal repository-test execution must run from the source root")
    if test_only:
        destination = _new_output_root(
            output_root, name="repository-test test-only output root", parent_root=worktree
        )
    else:
        destination = _new_output_root(
            output_root, name="repository-test formal output root"
        )
    test_plan_digest = repository_test_plan_digest(
        source_digest=source, test_targets=targets
    )
    command_digest = _repository_test_command_digest_from_binding(
        source_root=root,
        executable=executable,
        test_targets=targets,
        environment=normalized_environment,
    )
    pre_stage_executable = _executable_binding(
        interpreter, name="repository-test capture interpreter"
    )
    _require_exact_executable_binding(
        actual=pre_stage_executable,
        expected=executable,
        name="repository-test capture interpreter",
    )
    if expected_interpreter_binding is not None:
        _require_exact_executable_binding(
            actual=pre_stage_executable,
            expected=expected_interpreter_binding,
            name="repository-test capture interpreter",
        )
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.phase6-repository-stage-",
            dir=destination.parent,
        )
    )
    try:
        pre_execution_executable = _executable_binding(
            interpreter, name="repository-test capture interpreter"
        )
        _require_exact_executable_binding(
            actual=pre_execution_executable,
            expected=executable,
            name="repository-test capture interpreter",
        )
        if expected_interpreter_binding is not None:
            _require_exact_executable_binding(
                actual=pre_execution_executable,
                expected=expected_interpreter_binding,
                name="repository-test capture interpreter",
            )
        command = _capture_command(
            interpreter=Path(pre_execution_executable.declared_path),
            stage=stage,
            test_targets=targets,
        )
        try:
            completed = subprocess.run(
                command,
                cwd=str(worktree),
                env=dict(normalized_environment),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                shell=False,
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ValueError("repository-test subprocess did not complete") from exc
        post_execution_executable = _executable_binding(
            interpreter, name="repository-test capture interpreter"
        )
        _require_exact_executable_binding(
            actual=post_execution_executable,
            expected=executable,
            name="repository-test capture interpreter",
        )
        if expected_interpreter_binding is not None:
            _require_exact_executable_binding(
                actual=post_execution_executable,
                expected=expected_interpreter_binding,
                name="repository-test capture interpreter",
            )
        if not isinstance(completed.stdout, bytes):
            raise ValueError("repository-test subprocess emitted a non-bytes log")
        _write_new_bytes(stage / "run.log", completed.stdout)
        collection_path = stage / "collection.json"
        junit_path = stage / "junit.xml"
        if completed.returncode != 0:
            raise ValueError("repository-test subprocess returned nonzero exit")
        if not collection_path.is_file() or collection_path.is_symlink():
            raise ValueError("repository-test collection report is missing")
        if not junit_path.is_file() or junit_path.is_symlink():
            raise ValueError("repository-test JUnit output is missing")
        collection_raw = collection_path.read_bytes()
        junit_raw = junit_path.read_bytes()
        log_raw = (stage / "run.log").read_bytes()
        nodes, collected = _parse_reporter_payload(
            raw=collection_raw, process_exit_code=completed.returncode
        )
        _verify_junit(
            raw=junit_raw, expected_node_count=len(nodes), exit_code=completed.returncode
        )
        collection_digest = repository_test_collection_digest(collected)
        junit_sha = _sha256(junit_raw)
        log_sha = _sha256(log_raw)
        node_values = tuple(
            RepositoryTestNodeBinding(
                node_id=node_id,
                outcome=outcome,
                result_digest=stable_digest(
                    "osc-formal-repository-test-node-result-v1",
                    {
                        "source_digest": source,
                        "test_plan_digest": test_plan_digest,
                        "command_digest": command_digest,
                        "collection_digest": collection_digest,
                        "junit_sha256": junit_sha,
                        "log_sha256": log_sha,
                        "node_id": node_id,
                        "outcome": outcome,
                    },
                ),
            )
            for node_id, outcome in nodes
        )
        receipt = RepositoryTestReceipt(
            source_digest=source,
            test_plan_digest=test_plan_digest,
            collection_digest=collection_digest,
            config_digest=stable_digest(
                "osc-formal-repository-test-config-v1",
                {
                    "test_targets": targets,
                    "reporter_schema_version": FORMAL_REPOSITORY_TEST_REPORTER_SCHEMA_VERSION,
                    "cache_provider_disabled": True,
                },
            ),
            environment_digest=stable_digest(
                "osc-formal-repository-test-environment-v1", normalized_environment
            ),
            command_digest=command_digest,
            junit_sha256=junit_sha,
            log_sha256=log_sha,
            exit_code=completed.returncode,
            nodes=node_values,
        )
        receipt_raw = _replay_receipt(receipt)
        _write_new_bytes(stage / "repository_test_receipt.json", receipt_raw)
        entries = tuple(sorted(item.name for item in stage.iterdir()))
        if entries != FORMAL_REPOSITORY_TEST_OUTPUT_FILES:
            raise ValueError("repository-test staging output inventory mismatch")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"repository-test output appeared during capture: {destination}")
        os.replace(stage, destination)
        return RepositoryTestCaptureResult(
            source_digest=source,
            authority_id=identity,
            test_only=test_only,
            output_root=str(destination),
            receipt=receipt,
            collection_sha256=_sha256(collection_raw),
            junit_sha256=junit_sha,
            log_sha256=log_sha,
            receipt_sha256=_sha256(receipt_raw),
        )
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def verify_repository_test_capture(
    *,
    result: RepositoryTestCaptureResult,
    expected_source_digest: str,
    expected_test_only: bool,
) -> RepositoryTestCaptureResult:
    """Re-read a capture and fail closed if any output or receipt changed."""

    if not isinstance(result, RepositoryTestCaptureResult):
        raise TypeError("repository-test verification requires a capture result")
    source = _require_text(
        expected_source_digest, name="repository-test expected source digest"
    )
    _reject_synthetic(source)
    if result.source_digest != source:
        raise ValueError("repository-test result source digest mismatch")
    if result.test_only is not expected_test_only:
        raise ValueError("repository-test result test-only mode mismatch")
    root = _resolved_directory(
        result.output_root, name="repository-test verification output root"
    )
    entries = tuple(sorted(item.name for item in root.iterdir()))
    if entries != FORMAL_REPOSITORY_TEST_OUTPUT_FILES:
        raise ValueError("repository-test verification output inventory mismatch")
    collection_raw = (root / "collection.json").read_bytes()
    junit_raw = (root / "junit.xml").read_bytes()
    log_raw = (root / "run.log").read_bytes()
    receipt_raw = (root / "repository_test_receipt.json").read_bytes()
    if _sha256(collection_raw) != result.collection_sha256:
        raise ValueError("repository-test collection SHA-256 mismatch")
    if _sha256(junit_raw) != result.junit_sha256:
        raise ValueError("repository-test JUnit SHA-256 mismatch")
    if _sha256(log_raw) != result.log_sha256:
        raise ValueError("repository-test log SHA-256 mismatch")
    if _sha256(receipt_raw) != result.receipt_sha256:
        raise ValueError("repository-test receipt SHA-256 mismatch")
    nodes, collected = _parse_reporter_payload(
        raw=collection_raw, process_exit_code=result.receipt.exit_code
    )
    _verify_junit(
        raw=junit_raw, expected_node_count=len(nodes), exit_code=result.receipt.exit_code
    )
    if tuple(item[0] for item in nodes) != result.receipt.node_ids:
        raise ValueError("repository-test receipt nodes disagree with collection")
    if repository_test_collection_digest(collected) != result.receipt.collection_digest:
        raise ValueError("repository-test receipt collection digest mismatch")
    if receipt_raw != _replay_receipt(result.receipt):
        raise ValueError("repository-test receipt payload does not match the bound result")
    decoded = decode_canonical_envelope(receipt_raw.decode("utf-8"))
    if (
        decoded["type"] != "RepositoryTestReceipt"
        or decoded["schema_version"] != REPOSITORY_TEST_RECEIPT_SCHEMA_VERSION
    ):
        raise ValueError("repository-test receipt envelope type mismatch")
    errors = replay_runtime_admission(
        envelope_type="RepositoryTestReceipt",
        schema_version=REPOSITORY_TEST_RECEIPT_SCHEMA_VERSION,
        payload=decoded["payload"],
        subject_kind="repository_tests",
        subject_ids=result.receipt.node_ids,
    )
    if errors:
        raise ValueError("repository-test receipt semantic replay failed: " + ";".join(errors))
    return result


def capture_test_only_repository_tests(
    *,
    source_digest: str,
    source_root: Path,
    test_root: Path,
    test_targets: tuple[str, ...],
    output_root: Path,
    timeout_seconds: float = 30.0,
) -> RepositoryTestCaptureResult:
    """Exercise the real subprocess path only inside a pytest-managed temp root."""

    root = _resolved_directory(source_root, name="repository-test test-only source root")
    worktree = _resolved_directory(test_root, name="repository-test test-only root")
    environment = _source_environment(root)
    result = _capture(
        source_digest=source_digest,
        authority_id="phase6-repository-test-only",
        source_root=root,
        test_root=worktree,
        interpreter=Path(sys.executable),
        test_targets=test_targets,
        environment=environment,
        output_root=output_root,
        timeout_seconds=timeout_seconds,
        test_only=True,
    )
    return verify_repository_test_capture(
        result=result, expected_source_digest=source_digest, expected_test_only=True
    )


def execute_formal_repository_test(
    *, authority: FormalRepositoryTestExecutionAuthority
) -> RepositoryTestCaptureResult:
    """Run exactly one future authority-bound repository test workload."""

    if not isinstance(authority, FormalRepositoryTestExecutionAuthority):
        raise TypeError("formal repository-test execution requires an authority")
    if not authority.formal_execution_authorized:
        raise ValueError("formal repository-test execution is not authorized")
    root = _resolved_directory(authority.source_root, name="repository-test source root")
    snapshot = verify_source_snapshot(
        repo_root=root,
        snapshot_path=Path(authority.source_snapshot_path),
        expected_snapshot_sha256=authority.source_snapshot_sha256,
    )
    if not snapshot.valid or snapshot.source_digest != authority.source_digest:
        raise ValueError("repository-test source snapshot binding is not valid")
    _require_full_test_snapshot_closure(
        source_root=root,
        snapshot=snapshot,
        test_targets=authority.test_targets,
    )
    expected_interpreter_binding = _ExecutableBinding(
        declared_path=authority.interpreter_path,
        resolved_path=authority.interpreter_resolved_path,
        target_sha256=authority.interpreter_sha256,
    )
    result = _capture(
        source_digest=authority.source_digest,
        authority_id=authority.authority_id,
        source_root=root,
        test_root=root,
        interpreter=Path(authority.interpreter_path),
        test_targets=authority.test_targets,
        environment=authority.environment,
        output_root=Path(authority.output_root),
        timeout_seconds=authority.timeout_seconds,
        test_only=False,
        expected_interpreter_binding=expected_interpreter_binding,
    )
    post_capture_snapshot = verify_source_snapshot(
        repo_root=root,
        snapshot_path=Path(authority.source_snapshot_path),
        expected_snapshot_sha256=authority.source_snapshot_sha256,
    )
    if (
        not post_capture_snapshot.valid
        or post_capture_snapshot.source_digest != authority.source_digest
    ):
        raise ValueError("repository-test source snapshot binding changed during capture")
    _require_full_test_snapshot_closure(
        source_root=root,
        snapshot=post_capture_snapshot,
        test_targets=authority.test_targets,
    )
    return verify_repository_test_capture(
        result=result,
        expected_source_digest=authority.source_digest,
        expected_test_only=False,
    )


def formal_repository_test_producer_preview() -> dict[str, object]:
    """Return the fixed no-execution declaration used by the private CLI."""

    return {
        "schema_version": FORMAL_REPOSITORY_TEST_PRODUCER_SCHEMA_VERSION,
        "runner": "authority-bound-formal-repository-test-producer",
        "default_executes_repository_tests": False,
        "requires_fresh_source_snapshot": True,
        "requires_external_authority": True,
        "writes_durable_output_without_authority": False,
        "dynamic_plan_emitted": False,
        "artifact_receipt_index_emitted": False,
        "authority_bundle_emitted": False,
        "gate_credit": False,
        "candidate_confirmed": False,
        "bug_claimed": False,
    }


__all__ = [
    "FORMAL_REPOSITORY_TEST_AUTHORITY_SCHEMA_VERSION",
    "FORMAL_REPOSITORY_TEST_OUTPUT_FILES",
    "FORMAL_REPOSITORY_TEST_PRODUCER_SCHEMA_VERSION",
    "FormalRepositoryTestExecutionAuthority",
    "RepositoryTestCaptureResult",
    "capture_test_only_repository_tests",
    "execute_formal_repository_test",
    "formal_repository_test_producer_preview",
    "load_formal_repository_test_execution_authority",
    "repository_test_command_digest",
    "repository_test_plan_digest",
    "verify_repository_test_capture",
]
