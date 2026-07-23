from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

import datadiff_osc
import datadiff_osc._phase6_source_snapshot_eligibility as eligibility_module
from datadiff_osc._phase6_source_snapshot_eligibility import (
    SourceSnapshotEligibility,
    inspect_source_snapshot_eligibility,
)


def _git(repo: Path, *arguments: str) -> None:
    subprocess.run(
        ("git", "-C", str(repo), *arguments),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _repository(tmp_path: Path, *, name: str = "source-input") -> Path:
    repo = tmp_path / name
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Phase6 Test")
    _git(repo, "config", "user.email", "phase6@example.invalid")
    (repo / ".gitignore").write_text("*.ignored\n", encoding="utf-8")
    (repo / "tracked.txt").write_text("initial\n", encoding="utf-8")
    _git(repo, "add", ".gitignore", "tracked.txt")
    _git(repo, "commit", "-m", "initial")
    return repo


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_clean_temporary_repo_is_only_clean_input_eligible(tmp_path):
    record = inspect_source_snapshot_eligibility(_repository(tmp_path))

    assert record.clean_input_eligible is True
    assert record.status_porcelain == b""
    assert record.diagnostic_id
    assert record.authority_eligible is False
    assert record.gate_credit is False
    assert record.source_snapshot_created is False
    assert record.raw_artifact_admitted is False
    assert record.coverage_event_created is False
    assert record.candidate_confirmed is False
    assert record.bug_claimed is False
    assert "SourceSnapshotEligibility" not in datadiff_osc.__all__
    assert not hasattr(datadiff_osc, "SourceSnapshotEligibility")


def test_negative_review_marks_tracked_untracked_and_ignored_files_dirty(tmp_path):
    tracked_repo = _repository(tmp_path, name="tracked-input")
    (tracked_repo / "tracked.txt").write_text("changed\n", encoding="utf-8")
    tracked = inspect_source_snapshot_eligibility(tracked_repo)
    assert tracked.clean_input_eligible is False
    assert b" M tracked.txt\0" in tracked.status_porcelain

    untracked_repo = _repository(tmp_path, name="untracked-input")
    (untracked_repo / "untracked.txt").write_text("untracked\n", encoding="utf-8")
    untracked = inspect_source_snapshot_eligibility(untracked_repo)
    assert untracked.clean_input_eligible is False
    assert b"?? untracked.txt\0" in untracked.status_porcelain

    ignored_repo = _repository(tmp_path, name="ignored-input")
    (ignored_repo / "private.ignored").write_text("ignored\n", encoding="utf-8")
    ignored = inspect_source_snapshot_eligibility(ignored_repo)
    assert ignored.clean_input_eligible is False
    assert b"!! private.ignored\0" in ignored.status_porcelain


def test_independent_counterexamples_reject_root_and_status_substitution(tmp_path):
    repo = _repository(tmp_path)
    nested = repo / "nested"
    nested.mkdir()
    with pytest.raises(ValueError, match="Git top-level"):
        inspect_source_snapshot_eligibility(nested)
    with pytest.raises(ValueError, match="Git query failed"):
        inspect_source_snapshot_eligibility(tmp_path / "not-a-repository")

    clean = inspect_source_snapshot_eligibility(repo)
    with pytest.raises(ValueError, match="status SHA mismatch"):
        replace(clean, status_sha256="0" * 64)
    malformed = b"?? no-nul"
    with pytest.raises(ValueError, match="status is malformed"):
        SourceSnapshotEligibility(
            repo_root=clean.repo_root,
            git_head=clean.git_head,
            status_porcelain=malformed,
            status_sha256=_sha256(malformed),
            clean_input_eligible=False,
        )


def test_exceptional_git_failures_fail_closed(monkeypatch, tmp_path):
    repo = _repository(tmp_path)

    def unavailable(*_args, **_kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(eligibility_module.subprocess, "run", unavailable)
    with pytest.raises(ValueError, match="Git executable is unavailable"):
        inspect_source_snapshot_eligibility(repo)

    def malformed_result(*_args, **_kwargs):
        return SimpleNamespace(returncode=0, stdout=b"not-a-root\n", stderr=b"")

    monkeypatch.setattr(eligibility_module.subprocess, "run", malformed_result)
    with pytest.raises(ValueError, match="Git top-level"):
        inspect_source_snapshot_eligibility(repo)
