"""Private fail-closed eligibility check for a potential source input.

This module does not capture source bytes or create a source snapshot.  It
only records the exact Git top-level, HEAD, and full porcelain status needed to
distinguish a clean input from a dirty one.  The resulting record is always
diagnostic-only and never grants provenance or Phase-6 gate authority.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import subprocess

from datadiff_osc._canonical import stable_digest


SOURCE_SNAPSHOT_ELIGIBILITY_SCHEMA_VERSION = (
    "osc-private-source-snapshot-eligibility-v1"
)
_GIT_HEAD_RE = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_git_head(value: object) -> str:
    if not isinstance(value, str) or _GIT_HEAD_RE.fullmatch(value) is None:
        raise ValueError("Git HEAD must be a lowercase object ID")
    return value


def _require_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _run_git(repo_root: Path, *arguments: str) -> bytes:
    """Run a read-only Git query without shell interpolation."""

    try:
        completed = subprocess.run(
            ("git", "-C", str(repo_root), *arguments),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise ValueError("Git executable is unavailable") from exc
    if completed.returncode != 0:
        raise ValueError("Git query failed")
    if not isinstance(completed.stdout, bytes):
        raise ValueError("Git query stdout is malformed")
    return completed.stdout


@dataclass(frozen=True, slots=True)
class SourceSnapshotEligibility:
    """Exact dirty/clean classification with no source or gate authority."""

    repo_root: str
    git_head: str
    status_porcelain: bytes
    status_sha256: str
    clean_input_eligible: bool
    schema_version: str = SOURCE_SNAPSHOT_ELIGIBILITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        root = Path(self.repo_root)
        if not root.is_absolute() or str(root) in {"", "."}:
            raise ValueError("eligibility repository root must be absolute")
        _require_git_head(self.git_head)
        if not isinstance(self.status_porcelain, bytes):
            raise ValueError("eligibility status must be immutable bytes")
        _require_sha256(self.status_sha256, name="eligibility status SHA")
        if _sha256(self.status_porcelain) != self.status_sha256:
            raise ValueError("eligibility status SHA mismatch")
        if self.status_porcelain and not self.status_porcelain.endswith(b"\0"):
            raise ValueError("eligibility status is malformed")
        if not isinstance(self.clean_input_eligible, bool):
            raise ValueError("eligibility clean state must be boolean")
        if self.clean_input_eligible is not (not self.status_porcelain):
            raise ValueError("eligibility clean state/status mismatch")
        if self.schema_version != SOURCE_SNAPSHOT_ELIGIBILITY_SCHEMA_VERSION:
            raise ValueError("eligibility schema version mismatch")

    @property
    def diagnostic_id(self) -> str:
        return stable_digest(
            "osc-private-source-snapshot-eligibility-id-v1",
            {
                "repo_root": self.repo_root,
                "git_head": self.git_head,
                "status_sha256": self.status_sha256,
                "clean_input_eligible": self.clean_input_eligible,
            },
        )

    @property
    def authority_eligible(self) -> bool:
        return False

    @property
    def gate_credit(self) -> bool:
        return False

    @property
    def source_snapshot_created(self) -> bool:
        return False

    @property
    def raw_artifact_admitted(self) -> bool:
        return False

    @property
    def coverage_event_created(self) -> bool:
        return False

    @property
    def candidate_confirmed(self) -> bool:
        return False

    @property
    def bug_claimed(self) -> bool:
        return False


def inspect_source_snapshot_eligibility(repo_root: Path) -> SourceSnapshotEligibility:
    """Classify one exact Git top-level without creating any artifact.

    `--ignored=matching` deliberately makes ignored files visible too: any
    status bytes mean the caller must not treat this input as clean.
    """

    if not isinstance(repo_root, Path):
        raise TypeError("eligibility repository root must be a Path")
    root = repo_root.resolve()
    reported_root = _run_git(root, "rev-parse", "--show-toplevel")
    try:
        git_root = Path(reported_root.decode("utf-8").strip()).resolve()
    except UnicodeDecodeError as exc:
        raise ValueError("Git repository root is malformed") from exc
    if git_root != root:
        raise ValueError("eligibility repository root must equal Git top-level")
    try:
        head = _run_git(root, "rev-parse", "HEAD").decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise ValueError("Git HEAD is malformed") from exc
    status = _run_git(
        root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--ignored=matching",
    )
    return SourceSnapshotEligibility(
        repo_root=root.as_posix(),
        git_head=head,
        status_porcelain=status,
        status_sha256=_sha256(status),
        clean_input_eligible=not status,
    )
