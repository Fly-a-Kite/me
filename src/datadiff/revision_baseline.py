from __future__ import annotations

import ast
import hashlib
import json
import platform
import subprocess
import sys
import tarfile
from collections import Counter, deque
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from datadiff.experiment_manifest import stable_digest


REVISION_BASELINE_SCHEMA_VERSION = "revision-baseline-v1"
EVIDENCE_INDEX_SCHEMA_VERSION = "p0-p7-evidence-index-v1"


def build_revision_baseline(repo_root: Path) -> dict[str, Any]:
    root = repo_root.resolve()
    worktree = classify_worktree(root)
    evidence = build_evidence_index(root)
    frozen_anchors = build_frozen_evidence_anchors(root)
    imports = build_active_import_graph(root)
    environment = runtime_environment()
    payload = {
        "schema_version": REVISION_BASELINE_SCHEMA_VERSION,
        "worktree": worktree,
        "evidence_index": evidence,
        "frozen_evidence_anchors": frozen_anchors,
        "active_import_graph": imports,
        "environment": environment,
        "freeze_ready": bool(
            not worktree["summary"]["entry_count"]
            and evidence["summary"]["checksum_failure_count"] == 0
            and frozen_anchors["verified"]
            and not imports["rlcmf_live_dependencies"]
        ),
    }
    payload["baseline_digest"] = stable_digest("revision-baseline", payload)
    return payload


def classify_worktree(repo_root: Path) -> dict[str, Any]:
    completed = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    entries: list[dict[str, str]] = []
    parts = completed.stdout.decode("utf-8", errors="surrogateescape").split("\0")
    index = 0
    while index < len(parts):
        raw = parts[index]
        index += 1
        if not raw:
            continue
        status = raw[:2]
        path = raw[3:]
        if status[0] in {"R", "C"} and index < len(parts):
            path = parts[index]
            index += 1
        entries.append(
            {
                "status": status,
                "path": path,
                "category": classify_repository_path(path),
            }
        )
    counts = Counter(entry["category"] for entry in entries)
    return {
        "summary": {
            "entry_count": len(entries),
            "tracked_change_count": sum(entry["status"] != "??" for entry in entries),
            "untracked_count": sum(entry["status"] == "??" for entry in entries),
            "category_counts": dict(sorted(counts.items())),
        },
        "entries": entries,
    }


def classify_repository_path(path: str) -> str:
    normalized = str(path).replace("\\", "/")
    name = Path(normalized).name.lower()
    lower = normalized.lower()
    if "__pycache__" in lower or name.endswith((".pyc", ".pyo")):
        return "cache_scratch"
    if lower.startswith((".pytest_cache/", ".mypy_cache/", ".ruff_cache/")):
        return "cache_scratch"
    if lower.startswith("tests/"):
        return "tests"
    if lower.startswith(("src/", "scripts/")) or name in {
        "pyproject.toml",
        "requirements-final.lock",
        "dockerfile.final",
    }:
        return "active_source"
    if lower.startswith("experiments/"):
        if any(token in lower for token in ("smoke", "diagnostic", "runtime", "worker")):
            return "smoke_diagnostic_evidence"
        if any(
            token in name
            for token in (
                "protocol",
                "preregistration",
                "manifest",
                "result",
                "analysis",
                "report",
                "seal",
                "checksum",
                "sha256",
            )
        ):
            return "formal_evidence"
        return "obsolete_generated_output"
    if lower.startswith(("logs/", "new_issue/generated/", "runs/", "corpus/")):
        return "obsolete_generated_output"
    if lower.startswith(("docs/", "reports/", "new_issue/")) or name.endswith(".md"):
        return "documentation_evidence"
    return "cache_scratch"


def build_evidence_index(repo_root: Path) -> dict[str, Any]:
    experiments = repo_root / "experiments"
    rows: list[dict[str, Any]] = []
    checksum_failures: list[dict[str, str]] = []
    checksum_manifests: list[dict[str, Any]] = []
    if experiments.exists():
        for path in sorted(experiments.rglob("*")):
            if not path.is_file() or not _is_evidence_file(path):
                continue
            relative = path.relative_to(repo_root).as_posix()
            rows.append(
                {
                    "path": relative,
                    "stage": _evidence_stage(relative),
                    "role": _evidence_role(path.name),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
        for checksum_file in _checksum_manifest_paths(experiments):
            verification = _verify_sha256s(checksum_file, repo_root)
            failures = verification["failures"]
            checksum_failures.extend(failures)
            checksum_manifests.append(
                {
                    "path": checksum_file.relative_to(repo_root).as_posix(),
                    "entry_count": verification["entry_count"],
                    "passed_count": verification["entry_count"] - len(failures),
                    "failure_count": len(failures),
                }
            )
    stage_counts = Counter(row["stage"] for row in rows)
    role_counts = Counter(row["role"] for row in rows)
    failure_counts = Counter(row["reason"] for row in checksum_failures)
    checksum_entry_count = sum(row["entry_count"] for row in checksum_manifests)
    payload = {
        "schema_version": EVIDENCE_INDEX_SCHEMA_VERSION,
        "summary": {
            "file_count": len(rows),
            "stage_counts": dict(sorted(stage_counts.items())),
            "role_counts": dict(sorted(role_counts.items())),
            "checksum_manifest_count": len(checksum_manifests),
            "checksum_entry_count": checksum_entry_count,
            "checksum_pass_count": checksum_entry_count - len(checksum_failures),
            "checksum_failure_count": len(checksum_failures),
            "checksum_missing_count": failure_counts["missing"],
            "checksum_mismatch_count": failure_counts["digest_mismatch"],
            "checksum_malformed_count": failure_counts["malformed_checksum_line"],
        },
        "checksum_manifests": checksum_manifests,
        "checksum_failures": checksum_failures,
        "files": rows,
    }
    payload["index_digest"] = stable_digest("evidence-index", payload)
    return payload


def build_active_import_graph(
    repo_root: Path,
    *,
    roots: Iterable[str] = ("datadiff.cli", "datadiff.runner", "datadiff.run_loaded"),
) -> dict[str, Any]:
    source_root = repo_root / "src"
    queue = deque(str(module) for module in roots)
    visited: set[str] = set()
    edges: list[dict[str, str]] = []
    source_loc = 0
    while queue:
        module = queue.popleft()
        if module in visited:
            continue
        visited.add(module)
        path = _module_path(source_root, module)
        if path is None:
            continue
        text = path.read_text(encoding="utf-8")
        source_loc += sum(bool(line.strip()) for line in text.splitlines())
        for dependency in _datadiff_imports(text):
            edges.append({"from": module, "to": dependency})
            if dependency not in visited:
                queue.append(dependency)
    rlcmf = sorted(
        module
        for module in visited
        if module.startswith("datadiff.rlcmf") or module.startswith("datadiff.p7_rlcmf")
    )
    payload = {
        "root_modules": list(roots),
        "reachable_module_count": len(visited),
        "active_nonblank_loc": source_loc,
        "rlcmf_live_dependencies": rlcmf,
        "modules": sorted(visited),
        "edges": sorted(edges, key=lambda row: (row["from"], row["to"])),
    }
    payload["graph_digest"] = stable_digest("active-import-graph", payload)
    return payload


def build_frozen_evidence_anchors(repo_root: Path) -> dict[str, Any]:
    p5_archive_manifest_path = (
        repo_root / "experiments/p5_formal_ablation_v1/frozen_source_manifest.json"
    )
    p5_adoption_path = (
        repo_root / "experiments/p5_formal_ablation_v1/adoption_analysis.json"
    )
    p5_final_manifest_path = (
        repo_root / "experiments/p6_unseen_generalization_v1/frozen_source_manifest.json"
    )
    p7_source_manifest_path = (
        repo_root
        / "experiments/p7_rlcmf_promotion_v1/frozen_source/archive_manifest.json"
    )
    p7_result_path = repo_root / "experiments/p7_rlcmf_promotion_v1/result.json"
    p7_analysis_path = repo_root / "experiments/p7_rlcmf_promotion_v1/analysis.json"

    p5_archive_manifest = _read_json(p5_archive_manifest_path)
    p5_adoption = _read_json(p5_adoption_path)
    p5_final_manifest = _read_json(p5_final_manifest_path)
    p5_archive_path = repo_root / p5_archive_manifest["archive"]
    p5_final_archive_path = repo_root / p5_final_manifest["archive"]
    method_arms_entry = next(
        row
        for row in p5_final_manifest["entries"]
        if row["path"] == "src/datadiff/method_arms.py"
    )
    p5_archive_verified = (
        _sha256(p5_archive_path) == p5_archive_manifest["archive_sha256"]
    )
    p5_final_archive_verified = (
        _sha256(p5_final_archive_path) == p5_final_manifest["archive_sha256"]
    )
    observed_method_arms_sha256 = _tar_member_sha256(
        p5_final_archive_path,
        method_arms_entry["path"],
    )
    p5_source_verified = observed_method_arms_sha256 == method_arms_entry["sha256"]
    adoption = p5_adoption["adoption_decision"]
    p5_decision_verified = bool(
        adoption["adopted_default_arm"] == "p5_promoted_method"
        and adoption["method_frozen_after_adoption"]
    )

    p7_source_manifest = _read_json(p7_source_manifest_path)
    p7_source_root = p7_source_manifest_path.parent
    p7_source_failures = [
        row["path"]
        for row in p7_source_manifest["files"]
        if not (p7_source_root / row["path"]).is_file()
        or _sha256(p7_source_root / row["path"]) != row["sha256"]
    ]
    p7_result = _read_json(p7_result_path)
    p7_analysis = _read_json(p7_analysis_path)
    result_promotion = p7_result["promotion"]
    analysis_promotion = p7_analysis["promotion"]
    p7_decision_verified = bool(
        result_promotion == analysis_promotion
        and result_promotion["decision"] == "remain_shadow"
        and result_promotion["eligible"] is False
    )

    p5_verified = bool(
        p5_archive_verified
        and p5_final_archive_verified
        and p5_source_verified
        and p5_decision_verified
    )
    p7_verified = not p7_source_failures and p7_decision_verified
    return {
        "verified": bool(p5_verified and p7_verified),
        "p5_promoted_method": {
            "verified": p5_verified,
            "adopted_arm_id": adoption["adopted_default_arm"],
            "formal_archive": p5_archive_path.relative_to(repo_root).as_posix(),
            "formal_archive_sha256": p5_archive_manifest["archive_sha256"],
            "formal_archive_verified": p5_archive_verified,
            "final_source_archive": p5_final_archive_path.relative_to(repo_root).as_posix(),
            "final_source_archive_sha256": p5_final_manifest["archive_sha256"],
            "final_source_archive_verified": p5_final_archive_verified,
            "method_arms_source_path": method_arms_entry["path"],
            "method_arms_source_sha256": method_arms_entry["sha256"],
            "method_arms_source_verified": p5_source_verified,
        },
        "p7_negative_result": {
            "verified": p7_verified,
            "decision": result_promotion["decision"],
            "eligible": result_promotion["eligible"],
            "source_manifest": p7_source_manifest_path.relative_to(repo_root).as_posix(),
            "source_sha256": p7_source_manifest["source_sha256"],
            "source_file_count": len(p7_source_manifest["files"]),
            "source_failure_count": len(p7_source_failures),
            "source_failures": p7_source_failures,
            "result": p7_result_path.relative_to(repo_root).as_posix(),
            "result_sha256": _sha256(p7_result_path),
            "analysis": p7_analysis_path.relative_to(repo_root).as_posix(),
            "analysis_sha256": _sha256(p7_analysis_path),
        },
    }


def runtime_environment() -> dict[str, Any]:
    payload = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "locale": _safe_locale(),
        "timezone": _safe_timezone(),
        "executable": sys.executable,
    }
    payload["environment_digest"] = stable_digest("environment", payload)
    return payload


def _is_evidence_file(path: Path) -> bool:
    name = path.name.lower()
    return any(
        token in name
        for token in (
            "protocol",
            "preregistration",
            "manifest",
            "result",
            "analysis",
            "report",
            "seal",
            "checksum",
            "sha256",
        )
    )


def _evidence_stage(path: str) -> str:
    lower = path.lower()
    for stage in range(8):
        if f"/p{stage}" in lower or f"p{stage}_" in lower or f"p{stage}-" in lower:
            return f"P{stage}"
    return "cross_stage"


def _evidence_role(name: str) -> str:
    lower = name.lower()
    if "checksum" in lower or "sha256" in lower:
        return "checksum"
    for role in (
        "preregistration",
        "protocol",
        "manifest",
        "result",
        "analysis",
        "report",
        "seal",
    ):
        if role in lower:
            return role
    return "other"


def _checksum_manifest_paths(experiments: Path) -> list[Path]:
    names = {"SHA256SUMS", "checksums.sha256"}
    return sorted(
        path
        for path in experiments.rglob("*")
        if path.is_file() and path.name in names
    )


def _verify_sha256s(checksum_file: Path, repo_root: Path) -> dict[str, Any]:
    failures: list[dict[str, str]] = []
    entry_count = 0
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        entry_count += 1
        try:
            expected, relative = stripped.split(maxsplit=1)
        except ValueError:
            failures.append(
                {
                    "checksum_file": checksum_file.relative_to(repo_root).as_posix(),
                    "path": "",
                    "reason": "malformed_checksum_line",
                }
            )
            continue
        if len(expected) != 64 or any(
            char not in "0123456789abcdefABCDEF" for char in expected
        ):
            failures.append(
                {
                    "checksum_file": checksum_file.relative_to(repo_root).as_posix(),
                    "path": relative.removeprefix("*"),
                    "reason": "malformed_checksum_line",
                }
            )
            continue
        relative_path = relative.removeprefix("*")
        repository_candidate = repo_root / relative_path
        local_candidate = checksum_file.parent / relative_path
        candidate = (
            repository_candidate
            if repository_candidate.is_file()
            else local_candidate
        )
        if not candidate.is_file():
            failures.append(
                {
                    "checksum_file": checksum_file.relative_to(repo_root).as_posix(),
                    "path": relative_path,
                    "reason": "missing",
                }
            )
        elif _sha256(candidate) != expected:
            failures.append(
                {
                    "checksum_file": checksum_file.relative_to(repo_root).as_posix(),
                    "path": candidate.relative_to(repo_root).as_posix(),
                    "reason": "digest_mismatch",
                }
            )
    return {"entry_count": entry_count, "failures": failures}


def _module_path(source_root: Path, module: str) -> Path | None:
    relative = Path(*module.split("."))
    module_file = source_root / relative.with_suffix(".py")
    if module_file.is_file():
        return module_file
    package_file = source_root / relative / "__init__.py"
    return package_file if package_file.is_file() else None


def _datadiff_imports(source: str) -> tuple[str, ...]:
    imports: set[str] = set()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(
                alias.name for alias in node.names if alias.name.startswith("datadiff")
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("datadiff"):
                imports.add(node.module)
    return tuple(sorted(imports))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tar_member_sha256(archive: Path, member_path: str) -> str:
    digest = hashlib.sha256()
    with tarfile.open(archive, "r:gz") as bundle:
        member = bundle.extractfile(member_path)
        if member is None:
            raise ValueError(f"missing frozen source member: {member_path}")
        for chunk in iter(lambda: member.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _safe_locale() -> str:
    try:
        import locale

        return str(locale.setlocale(locale.LC_ALL, None))
    except Exception:  # noqa: BLE001
        return "unknown"


def _safe_timezone() -> str:
    try:
        import time

        return str(time.tzname)
    except Exception:  # noqa: BLE001
        return "unknown"


def write_revision_baseline(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
