from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from datadiff.dsl import Case


CONFIRMED_LEDGER_STATUSES = frozenset({"upstream_labeled_bug", "fixed_upstream"})
DEFAULT_CANONICAL_BUG_CORPUS_MANIFEST = Path(
    "experiments/canonical_confirmed_bug_corpus/v2/manifest.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_canonical_bug_corpus(manifest_path: Path) -> dict[str, Any]:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def validate_canonical_bug_corpus(
    repo_root: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    manifest_path = manifest_path.resolve()
    payload = load_canonical_bug_corpus(manifest_path)
    errors: list[str] = []
    warnings: list[str] = []

    if payload.get("schema_version") != "canonical-confirmed-bug-corpus-v1":
        errors.append("unsupported or missing canonical corpus schema_version")

    native = payload.get("native_reproducer", {})
    native_path = repo_root / str(native.get("path", ""))
    _validate_file_checksum(
        path=native_path,
        expected=str(native.get("sha256", "")),
        label="native_reproducer",
        errors=errors,
    )

    confirmed = list(payload.get("confirmed_roots", []))
    pending = list(payload.get("pending_roots", []))
    all_roots = [*confirmed, *pending]
    root_ids = [str(root.get("root_id", "")) for root in all_roots]
    issue_urls = [str(root.get("issue_url", "")) for root in all_roots]
    _validate_unique_nonempty(root_ids, "root_id", errors)
    _validate_unique_nonempty(issue_urls, "issue_url", errors)

    target = payload.get("target", {})
    if int(target.get("confirmed_root_count", -1)) != len(confirmed):
        errors.append("target.confirmed_root_count does not match confirmed_roots")
    if int(target.get("pending_root_count", -1)) != len(pending):
        errors.append("target.pending_root_count does not match pending_roots")

    parsed_cases: list[str] = []
    for root in all_roots:
        root_id = str(root.get("root_id", ""))
        case_spec = root.get("dsl_case", {})
        case_path = repo_root / str(case_spec.get("path", ""))
        _validate_file_checksum(
            path=case_path,
            expected=str(case_spec.get("sha256", "")),
            label=f"{root_id}.dsl_case",
            errors=errors,
        )
        if case_path.is_file():
            try:
                case = Case.from_dict(json.loads(case_path.read_text(encoding="utf-8")))
            except Exception as exc:  # schema failure should identify the root
                errors.append(f"{root_id}.dsl_case cannot be parsed: {type(exc).__name__}: {exc}")
            else:
                parsed_cases.append(case.case_id)
                if case.metadata.get("canonical_root_id") != root_id:
                    errors.append(f"{root_id}.dsl_case canonical_root_id mismatch")

        observations = list(root.get("version_observations", []))
        if not observations:
            errors.append(f"{root_id} has no version_observations")
        for index, observation in enumerate(observations):
            versions = observation.get("versions", {})
            if not isinstance(versions, dict) or not versions:
                errors.append(f"{root_id}.version_observations[{index}] has no versions")
            if not isinstance(observation.get("expected_bug_present"), bool):
                errors.append(
                    f"{root_id}.version_observations[{index}] expected_bug_present is not boolean"
                )
            if observation.get("locally_validated") is not True:
                warnings.append(f"{root_id}.version_observations[{index}] is not locally validated")

        if root.get("affected_versions") and not any(
            observation.get("expected_bug_present") is True for observation in observations
        ):
            errors.append(f"{root_id} has affected_versions without an affected observation")
        if root.get("fixed_versions") and not any(
            observation.get("expected_bug_present") is False for observation in observations
        ):
            gap = str(root.get("fixed_validation_gap", ""))
            if gap:
                warnings.append(f"{root_id}: {gap}")
            else:
                errors.append(f"{root_id} has fixed_versions without a fixed observation or gap")

    ledger_path = repo_root / str(payload.get("source", {}).get("confirmation_ledger", ""))
    if not ledger_path.is_file():
        errors.append(f"confirmation ledger does not exist: {ledger_path}")
        ledger_confirmed: list[dict[str, Any]] = []
        ledger_pending: list[dict[str, Any]] = []
    else:
        ledger_payload = json.loads(ledger_path.read_text(encoding="utf-8"))
        ledger_rows = list(ledger_payload.get("confirmations", []))
        ledger_confirmed = [
            row for row in ledger_rows if row.get("upstream_status") in CONFIRMED_LEDGER_STATUSES
        ]
        ledger_pending = [
            row
            for row in ledger_rows
            if row.get("upstream_status") == "submitted_pending_independent_confirmation"
        ]
        expected_digest = str(
            payload.get("source", {}).get("confirmation_ledger_canonical_json_sha256", "")
        )
        observed_digest = canonical_json_sha256(ledger_path)
        if expected_digest != observed_digest:
            errors.append(
                "confirmation ledger canonical digest mismatch: "
                f"expected {expected_digest}, observed {observed_digest}"
            )

    ledger_confirmed_urls = {str(row.get("issue_url", "")) for row in ledger_confirmed}
    manifest_confirmed_urls = {str(root.get("issue_url", "")) for root in confirmed}
    if ledger_confirmed_urls != manifest_confirmed_urls:
        errors.append(
            "confirmed issue URL set differs between corpus and ledger: "
            f"corpus_only={sorted(manifest_confirmed_urls - ledger_confirmed_urls)}, "
            f"ledger_only={sorted(ledger_confirmed_urls - manifest_confirmed_urls)}"
        )

    ledger_pending_urls = {str(row.get("issue_url", "")) for row in ledger_pending}
    for root in pending:
        issue_url = str(root.get("issue_url", ""))
        if issue_url not in ledger_pending_urls:
            errors.append(f"pending root {root.get('root_id')} is missing from the pending ledger")
        if root.get("count_as_confirmed") is not False:
            errors.append(f"pending root {root.get('root_id')} must set count_as_confirmed=false")

    validation_summary = payload.get("validation_summary", {})
    fixed_gaps = sum(bool(root.get("fixed_validation_gap")) for root in confirmed)
    if int(validation_summary.get("fixed_source_validation_gaps", -1)) != fixed_gaps:
        errors.append("validation_summary.fixed_source_validation_gaps is inconsistent")
    if fixed_gaps and validation_summary.get("p3_3_exit_ready") is not False:
        errors.append("p3_3_exit_ready must be false while fixed-source validation gaps remain")

    return {
        "schema_version": "canonical-corpus-validation-v1",
        "manifest": str(manifest_path.relative_to(repo_root)),
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "confirmed_roots": len(confirmed),
            "pending_roots": len(pending),
            "parsed_dsl_cases": len(parsed_cases),
            "fixed_source_validation_gaps": fixed_gaps,
            "p3_3_exit_ready": bool(validation_summary.get("p3_3_exit_ready", False)),
        },
    }


def execute_canonical_bug_corpus(
    repo_root: Path,
    manifest_path: Path,
    *,
    python_executable: Path | None = None,
    include_pending: bool = False,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    payload = load_canonical_bug_corpus(manifest_path.resolve())
    roots = list(payload.get("confirmed_roots", []))
    if include_pending:
        roots.extend(payload.get("pending_roots", []))
    native_path = repo_root / payload["native_reproducer"]["path"]
    python_path = python_executable or Path(sys.executable)
    timeout = timeout_seconds or float(payload.get("config", {}).get("native_timeout_seconds", 60))
    results: list[dict[str, Any]] = []

    for root in roots:
        root_id = str(root["root_id"])
        completed = subprocess.run(
            [str(python_path), str(native_path), "--root-id", root_id],
            cwd=repo_root,
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout,
        )
        record: dict[str, Any] = {
            "root_id": root_id,
            "returncode": completed.returncode,
            "stderr": completed.stderr,
        }
        try:
            observation = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            record.update(
                {
                    "passed": False,
                    "classification": "invalid_native_output",
                    "error": str(exc),
                    "stdout": completed.stdout,
                }
            )
            results.append(record)
            continue

        versions = observation.get("versions", {})
        expected = _matching_version_observation(root, versions)
        if completed.returncode != 0 or observation.get("status") != "ok":
            passed = False
            classification = "native_execution_error"
        elif expected is None:
            passed = False
            classification = "unregistered_version_tuple"
        else:
            passed = observation.get("bug_present") is expected.get("expected_bug_present")
            classification = "matched_expected_state" if passed else "unexpected_bug_state"
        record.update(
            {
                "passed": passed,
                "classification": classification,
                "versions": versions,
                "bug_present": observation.get("bug_present"),
                "expected_bug_present": (
                    expected.get("expected_bug_present") if expected is not None else None
                ),
                "signature": observation.get("signature", ""),
                "observation": observation,
            }
        )
        results.append(record)

    passed_count = sum(bool(record.get("passed")) for record in results)
    return {
        "schema_version": "canonical-corpus-execution-v1",
        "manifest": str(manifest_path.resolve().relative_to(repo_root)),
        "python_executable": str(python_path),
        "include_pending": include_pending,
        "results": results,
        "summary": {
            "executed": len(results),
            "passed": passed_count,
            "failed": len(results) - passed_count,
            "all_passed": passed_count == len(results),
        },
    }


def _matching_version_observation(
    root: dict[str, Any],
    versions: dict[str, Any],
) -> dict[str, Any] | None:
    for observation in root.get("version_observations", []):
        expected_versions = observation.get("versions", {})
        if expected_versions == versions:
            return observation
    return None


def _validate_file_checksum(
    *,
    path: Path,
    expected: str,
    label: str,
    errors: list[str],
) -> None:
    if not path.is_file():
        errors.append(f"{label} does not exist: {path}")
        return
    observed = sha256_file(path)
    if not expected:
        errors.append(f"{label} has no frozen sha256")
    elif observed != expected:
        errors.append(f"{label} sha256 mismatch: expected {expected}, observed {observed}")


def _validate_unique_nonempty(values: list[str], label: str, errors: list[str]) -> None:
    if any(not value for value in values):
        errors.append(f"one or more {label} values are empty")
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        errors.append(f"duplicate {label} values: {duplicates}")
