#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from datadiff.config import DEFAULT_REPLAY_BUG_SOURCE_ISSUES  # noqa: E402
from datadiff.classification_oracle import documented_semantic_rule_records  # noqa: E402
from datadiff.classification_oracle import semantic_boundary_rule_records  # noqa: E402
from datadiff.dynamic_strategy import write_strategy_snapshot  # noqa: E402
from datadiff.experiment_catalog import (  # noqa: E402
    FINAL_ADAPTIVE_COMPONENT_ABLATION_MATRIX,
    FINAL_COMPARISON_MATRIX,
    FINAL_LIVE_DISCOVERY_MATRIX,
    FINAL_MODULE_ABLATION_MATRIX,
    FINAL_PROTOCOL_TRACKS,
    FINAL_SEEDED_SENSITIVITY_MATRIX,
    FINAL_VALIDATION_MATRIX,
    build_historical_experiment_meta,
)
from datadiff.historical import list_historical_bugs  # noqa: E402
from datadiff.run_journal import append_run_journal_entries, build_run_journal_entry  # noqa: E402
from datadiff.triage import standalone_reproducer_rule_records  # noqa: E402
from datadiff.util import REPORTS_DIR, load_json, read_jsonl, utc_now  # noqa: E402

DATADIFF = Path(sys.prefix) / "bin" / "datadiff"
if not DATADIFF.exists():
    DATADIFF = PROJECT_ROOT / ".venv" / "bin" / "datadiff"

FINAL_PLAN_TRACKS: tuple[str, ...] = (*FINAL_PROTOCOL_TRACKS, "postprocess")
FINAL_MATRIX_TRACK_ALIASES: tuple[str, ...] = (
    FINAL_VALIDATION_MATRIX.id,
    FINAL_LIVE_DISCOVERY_MATRIX.id,
    "historical_replay",
    FINAL_SEEDED_SENSITIVITY_MATRIX.id,
    FINAL_MODULE_ABLATION_MATRIX.id,
    FINAL_ADAPTIVE_COMPONENT_ABLATION_MATRIX.id,
    FINAL_COMPARISON_MATRIX.id,
    "version_ledger",
)
MANIFEST_INDEX_SCHEMA_VERSION = "final-experiment-manifest-index-v1"
DEFAULT_FINAL_STRATEGY_SNAPSHOT = REPORTS_DIR / "strategy-snapshots" / "final-frozen-strategy-snapshot.json"
FINAL_REQUIRED_MATRIX_IDS: tuple[str, ...] = (
    FINAL_VALIDATION_MATRIX.id,
    FINAL_LIVE_DISCOVERY_MATRIX.id,
    "historical_replay",
    FINAL_SEEDED_SENSITIVITY_MATRIX.id,
    FINAL_MODULE_ABLATION_MATRIX.id,
    FINAL_ADAPTIVE_COMPONENT_ABLATION_MATRIX.id,
    FINAL_COMPARISON_MATRIX.id,
)


@dataclass(frozen=True, slots=True)
class FinalCommand:
    track: str
    name: str
    command: list[str]
    purpose: str
    count_as_real_bugs: bool
    expected_output: str
    notes: str = ""
    replay_bug_policy: dict[str, object] = field(default_factory=dict)
    experiment_meta: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["shell"] = shell_join(self.command)
        return data


def main() -> int:
    return run_with_args(parse_args())


def run_with_args(args: argparse.Namespace) -> int:
    if args.execute and args.track == "all":
        print(
            "refusing --track all --execute because live/latest and historical/vulnerable "
            "runs require different Python environments; execute track-specific plans from "
            "the matching venv instead.",
            file=sys.stderr,
        )
        return 2
    index_path = Path(
        str(getattr(args, "manifest_index", "") or REPORTS_DIR / "final-experiment-manifest-index.json")
    )
    imported = _import_existing_evidence_if_requested(args, index_path=index_path)
    commands = build_plan(args)
    plan_path = write_plan(commands, args)
    print(f"final experiment plan: {plan_path}")
    for item in commands:
        print()
        print(f"[{item.track}] {item.name}")
        print(f"purpose: {item.purpose}")
        print(f"counts_as_real_bugs: {str(item.count_as_real_bugs).lower()}")
        if item.notes:
            print(f"notes: {item.notes}")
        print(shell_join(item.command))
    if args.execute:
        if args.track == "postprocess" and _requires_postprocess_evidence_preflight(commands):
            issues = _postprocess_evidence_preflight_issues(index_path, args=args)
            if issues:
                print(
                    "refusing postprocess final-readiness: manifest index is incomplete",
                    file=sys.stderr,
                )
                for issue in issues:
                    print(f"- {issue}", file=sys.stderr)
                return 2
        if args.track != "postprocess" and (
            bool(getattr(args, "reset_manifest_index", False)) or not index_path.exists()
        ):
            _write_initial_manifest_index(index_path, plan_path=plan_path, args=args, commands=commands)
            if imported:
                _append_manifest_index_import(
                    index_path,
                    manifest_files=imported["manifest_files"],
                    extra_manifest_files=imported["extra_manifest_files"],
                    paper_run_journal=imported["paper_run_journal"],
                    imported_run_files=imported["imported_run_files"],
                )
        for item in commands:
            print(f"\nexecuting [{item.track}] {item.name}", flush=True)
            result = _execute_command_with_manifest_capture(item)
            _record_evidence_validation(item, result)
            if args.track != "postprocess":
                _append_manifest_index_command(index_path, item, result)
            if result["returncode"] != 0:
                raise subprocess.CalledProcessError(int(result["returncode"]), item.command)
    return 0


def main_with_args_for_test(args: argparse.Namespace) -> int:
    # Compatibility alias for older tests and external wrappers.
    return run_with_args(args)


def _execute_command_with_manifest_capture(item: FinalCommand) -> dict[str, object]:
    observed: dict[str, object] = {
        "returncode": 1,
        "manifest_files": [],
        "extra_manifest_files": [],
        "final_readiness_files": [],
    }
    proc = subprocess.Popen(
        item.command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        _ingest_command_evidence_line(observed, line)
    observed["returncode"] = proc.wait()
    return observed


def _ingest_command_evidence_line(observed: dict[str, object], line: str) -> None:
    text = line.strip()
    if text.startswith("experiment manifest:"):
        _append_observed_path(observed, "manifest_files", text.split(":", 1)[1].strip())
    elif text.startswith("evidence_manifest="):
        _append_observed_path(observed, "extra_manifest_files", text.split("=", 1)[1].strip())
    elif text.startswith("final readiness json:"):
        _append_observed_path(observed, "final_readiness_files", text.split(":", 1)[1].strip())


def _record_evidence_validation(item: FinalCommand, observed: dict[str, object]) -> None:
    if int(observed.get("returncode", 1)) != 0:
        return
    issues = _missing_command_evidence(item, observed)
    if not issues:
        return
    observed["evidence_issues"] = issues
    observed["returncode"] = 2
    print(
        f"missing required evidence for [{item.track}] {item.name}: {', '.join(issues)}",
        file=sys.stderr,
        flush=True,
    )


def _missing_command_evidence(item: FinalCommand, observed: dict[str, object]) -> list[str]:
    subcommand = _datadiff_subcommand(item.command)
    issues: list[str] = []
    if subcommand in {"experiment", "replay-fixture"} and not _string_list(
        observed.get("manifest_files", [])
    ):
        issues.append("missing_experiment_manifest")
    if subcommand == "version-ledger" and "--evidence-manifest-output" in item.command and not _string_list(
        observed.get("extra_manifest_files", [])
    ):
        issues.append("missing_version_ledger_evidence_manifest")
    if subcommand == "final-readiness" and not _string_list(observed.get("final_readiness_files", [])):
        issues.append("missing_final_readiness_json")
    return issues


def _datadiff_subcommand(command: list[str]) -> str:
    known = {
        "experiment",
        "replay-fixture",
        "version-ledger",
        "final-readiness",
    }
    for part in command:
        if part in known:
            return part
    return ""


def _requires_postprocess_evidence_preflight(commands: list[FinalCommand]) -> bool:
    return any(
        _datadiff_subcommand(command.command) == "final-readiness"
        and "--manifest-index" in command.command
        for command in commands
    )


def _postprocess_evidence_preflight_issues(index_path: Path, *, args: argparse.Namespace) -> list[str]:
    payload, load_error = _load_json_object(index_path)
    if load_error:
        return [f"manifest_index_{load_error}:{index_path}"]
    assert payload is not None
    issues: list[str] = []
    if str(payload.get("schema_version", "") or "") != MANIFEST_INDEX_SCHEMA_VERSION:
        issues.append("manifest_index_schema_mismatch")
    failed_commands = _failed_manifest_index_commands(payload)
    if failed_commands:
        issues.append(f"failed_index_commands:{','.join(failed_commands[:10])}")
    manifest_files = _string_list(payload.get("manifest_files", []))
    if not manifest_files:
        issues.append("manifest_index_has_no_manifest_files")
    observed_matrix_ids, matrix_issues = _manifest_index_matrix_ids(manifest_files)
    issues.extend(matrix_issues)
    missing_matrix_ids = sorted(set(FINAL_REQUIRED_MATRIX_IDS) - observed_matrix_ids)
    if missing_matrix_ids:
        issues.append(f"missing_required_matrix_ids:{','.join(missing_matrix_ids)}")

    ledger_manifest = str(
        getattr(args, "ledger_evidence_manifest", "") or REPORTS_DIR / "experiment-final-version-ledger.json"
    ).strip()
    extra_manifest_files = _string_list(payload.get("extra_manifest_files", []))
    if not ledger_manifest:
        issues.append("missing_configured_version_ledger_evidence_manifest")
    elif not _contains_recorded_path(extra_manifest_files, ledger_manifest):
        issues.append(f"unrecorded_version_ledger_evidence_manifest:{ledger_manifest}")
    if ledger_manifest:
        issues.extend(_version_ledger_evidence_manifest_issues(ledger_manifest))
    return issues


def _failed_manifest_index_commands(payload: dict[str, object]) -> list[str]:
    failed: list[str] = []
    for row in payload.get("commands", []) or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("status", "") or "") == "completed":
            continue
        label = str(row.get("name", "") or row.get("track", "") or "unknown").strip()
        failed.append(label)
    return failed


def _manifest_index_matrix_ids(manifest_files: list[str]) -> tuple[set[str], list[str]]:
    matrix_ids: set[str] = set()
    issues: list[str] = []
    for path_text in manifest_files:
        path = _project_path(path_text)
        manifest, load_error = _load_json_object(path)
        if load_error:
            issues.append(f"manifest_{load_error}:{path_text}")
            continue
        assert manifest is not None
        matrix_ids.update(_manifest_matrix_ids(manifest))
    return matrix_ids, issues


def _manifest_matrix_ids(manifest: dict[str, object]) -> set[str]:
    matrix_ids: set[str] = set()

    def add(value: object) -> None:
        text = str(value or "").strip()
        if text:
            matrix_ids.add(text)

    add(manifest.get("matrix_id", ""))
    experiment_meta = manifest.get("experiment_meta", {})
    if isinstance(experiment_meta, dict):
        add(experiment_meta.get("matrix_id", ""))
    for run in manifest.get("runs", []) or []:
        if not isinstance(run, dict):
            continue
        add(run.get("matrix_id", ""))
        run_meta = run.get("experiment_meta", {})
        if isinstance(run_meta, dict):
            add(run_meta.get("matrix_id", ""))
    return matrix_ids


def _version_ledger_evidence_manifest_issues(path_text: str) -> list[str]:
    path = _project_path(path_text)
    payload, load_error = _load_json_object(path)
    if load_error:
        return [f"version_ledger_evidence_manifest_{load_error}:{path_text}"]
    assert payload is not None
    issues: list[str] = []
    if str(payload.get("schema_version", "") or "") != "version-ledger-evidence-manifest-v1":
        issues.append(f"version_ledger_evidence_manifest_schema_mismatch:{path_text}")
    ledger_file = str(payload.get("version_ledger_file", "") or "").strip()
    if not ledger_file:
        for run in payload.get("runs", []) or []:
            if isinstance(run, dict):
                ledger_file = str(run.get("version_ledger_file", "") or "").strip()
                if ledger_file:
                    break
    if not ledger_file:
        issues.append(f"version_ledger_evidence_manifest_missing_ledger_file:{path_text}")
        return issues
    ledger, ledger_load_error = _load_json_object(_project_path(ledger_file))
    if ledger_load_error:
        issues.append(f"version_ledger_{ledger_load_error}:{ledger_file}")
        return issues
    assert ledger is not None
    issues.extend(_version_ledger_payload_issues(ledger, ledger_file=ledger_file))
    return issues


def _version_ledger_payload_issues(ledger: dict[str, object], *, ledger_file: str) -> list[str]:
    issues: list[str] = []
    if str(ledger.get("schema_version", "") or "") != "version-ledger-v1":
        issues.append(f"version_ledger_schema_mismatch:{ledger_file}")
    summary = ledger.get("summary", {}) if isinstance(ledger.get("summary", {}), dict) else {}
    if int(summary.get("version_count", 0) or 0) < 2:
        issues.append(f"version_ledger_requires_two_versions:{ledger_file}")
    if int(summary.get("family_count", 0) or 0) < 1:
        issues.append(f"version_ledger_requires_candidate_families:{ledger_file}")
    health = ledger.get("health", {}) if isinstance(ledger.get("health", {}), dict) else {}
    if str(health.get("schema_version", "") or "") != "version-ledger-health-v1":
        issues.append(f"version_ledger_missing_health_feedback:{ledger_file}")
    report = (
        ledger.get("health_feedback_report", {})
        if isinstance(ledger.get("health_feedback_report", {}), dict)
        else {}
    )
    if str(report.get("schema_version", "") or "") != "version-ledger-health-feedback-report-v1":
        issues.append(f"version_ledger_missing_health_feedback_report:{ledger_file}")
    if int(report.get("health_observation_count", 0) or 0) <= 0:
        issues.append(f"version_ledger_requires_health_observations:{ledger_file}")
    return issues


def _load_json_object(path: Path) -> tuple[dict[str, object] | None, str]:
    if not path.is_file():
        return None, "missing"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None, "invalid_json"
    if not isinstance(data, dict):
        return None, "not_object"
    return data, ""


def _project_path(path_text: str | Path) -> Path:
    path = Path(str(path_text))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _contains_recorded_path(values: list[str], expected: str) -> bool:
    expected_text = str(expected).strip()
    if expected_text in values:
        return True
    expected_path = _project_path(expected_text).resolve()
    for value in values:
        try:
            if _project_path(value).resolve() == expected_path:
                return True
        except OSError:
            continue
    return False


def _append_observed_path(observed: dict[str, object], key: str, path: str) -> None:
    if not path:
        return
    values = observed.setdefault(key, [])
    if not isinstance(values, list):
        return
    if path not in values:
        values.append(path)


def _import_existing_evidence_if_requested(
    args: argparse.Namespace,
    *,
    index_path: Path,
) -> dict[str, list[str]] | None:
    manifest_files = _string_list(getattr(args, "import_manifest", []))
    extra_manifest_files = _string_list(getattr(args, "import_extra_manifest", []))
    if not manifest_files and not extra_manifest_files:
        return None
    if bool(getattr(args, "reset_manifest_index", False)) or not index_path.exists():
        _write_initial_manifest_index(index_path, plan_path=Path(""), args=args, commands=[])
    imported_run_files = _import_manifest_journal_entries(
        manifest_files=manifest_files,
        paper_run_journal=Path(
            str(getattr(args, "paper_run_journal", "") or REPORTS_DIR / "paper-run-journal.jsonl")
        ),
    )
    _append_manifest_index_import(
        index_path,
        manifest_files=manifest_files,
        extra_manifest_files=extra_manifest_files,
        paper_run_journal=str(
            getattr(args, "paper_run_journal", "") or REPORTS_DIR / "paper-run-journal.jsonl"
        ),
        imported_run_files=imported_run_files,
    )
    return {
        "manifest_files": manifest_files,
        "extra_manifest_files": extra_manifest_files,
        "paper_run_journal": [str(getattr(args, "paper_run_journal", "") or REPORTS_DIR / "paper-run-journal.jsonl")],
        "imported_run_files": imported_run_files,
    }


def _import_manifest_journal_entries(*, manifest_files: list[str], paper_run_journal: Path) -> list[str]:
    existing_run_files = (
        {
            str(row.get("run_file", "")).strip()
            for row in read_jsonl(paper_run_journal)
            if isinstance(row, dict)
        }
        if paper_run_journal.exists()
        else set()
    )
    entries = []
    imported_run_files: list[str] = []
    for manifest_text in manifest_files:
        manifest_path = _project_path(manifest_text)
        manifest = load_json(manifest_path)
        if not isinstance(manifest, dict):
            continue
        experiment_meta = manifest.get("experiment_meta", {}) if isinstance(manifest.get("experiment_meta", {}), dict) else {}
        paper_notes = str(manifest.get("paper_notes", "") or "")
        evidence_mode = str(manifest.get("evidence_mode", "") or "")
        known_bug_id = str(manifest.get("known_bug_id", "") or "")
        target_version = str(manifest.get("target_version", "") or "")
        for run in manifest.get("runs", []) or []:
            if not isinstance(run, dict):
                continue
            run_file_text = str(run.get("run_file", "") or "").strip()
            if not run_file_text or run_file_text in existing_run_files:
                continue
            run_file = Path(run_file_text)
            imported_run_files.append(run_file_text)
            existing_run_files.add(run_file_text)
            entries.append(
                build_run_journal_entry(
                    run_file,
                    {
                        "command": "experiment-import",
                        "theme": _manifest_run_theme(manifest, run),
                        "notes": paper_notes,
                        "evidence_mode": str(run.get("evidence_mode") or evidence_mode),
                        "known_bug_id": str(run.get("known_bug_id") or known_bug_id),
                        "target_version": str(run.get("target_version") or target_version),
                        "target_suite": str(run.get("target_suite", "") or ""),
                        "preset": str(run.get("preset", "") or ""),
                        "seed": run.get("seed", ""),
                        "backends": run.get("backends", []),
                        "manifest_file": str(manifest_path),
                        "experiment_meta": run.get("experiment_meta", {}) or experiment_meta,
                    },
                )
            )
    if entries:
        append_run_journal_entries(entries, paper_run_journal)
    return _unique_strings(imported_run_files)


def _manifest_run_theme(manifest: dict[str, object], run: dict[str, object]) -> str:
    base = str(manifest.get("run_theme", "") or "").strip()
    suffix = f"{run.get('target_suite', '')}:{run.get('preset', '')}:seed{run.get('seed', '')}"
    if base:
        return f"{base} | {suffix}"
    evidence_mode = str(run.get("evidence_mode") or manifest.get("evidence_mode", "live"))
    known_bug_id = str(run.get("known_bug_id") or manifest.get("known_bug_id", "") or "")
    if evidence_mode == "historical" and known_bug_id:
        return f"historical:{known_bug_id}:{suffix}"
    return f"{evidence_mode}:{suffix}"


def _append_manifest_index_import(
    index_path: Path,
    *,
    manifest_files: list[str],
    extra_manifest_files: list[str],
    paper_run_journal: list[str] | str,
    imported_run_files: list[str],
) -> None:
    if not index_path.exists():
        _write_initial_manifest_index(index_path, plan_path=Path(""), args=argparse.Namespace(), commands=[])
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    commands = payload.setdefault("commands", [])
    if not isinstance(commands, list):
        commands = []
        payload["commands"] = commands
    commands.append(
        {
            "track": "import",
            "name": "import_existing_evidence",
            "status": "completed",
            "returncode": 0,
            "manifest_files": manifest_files,
            "extra_manifest_files": extra_manifest_files,
            "paper_run_journal_files": _string_list(paper_run_journal),
            "imported_run_files": imported_run_files,
            "final_readiness_files": [],
            "evidence_issues": [],
            "shell": "import-existing-evidence",
        }
    )
    payload["manifest_files"] = _unique_strings(
        [*_string_list(payload.get("manifest_files", [])), *manifest_files]
    )
    payload["extra_manifest_files"] = _unique_strings(
        [*_string_list(payload.get("extra_manifest_files", [])), *extra_manifest_files]
    )
    payload["updated_at"] = utc_now()
    index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_initial_manifest_index(
    index_path: Path,
    *,
    plan_path: Path,
    args: argparse.Namespace,
    commands: list[FinalCommand],
) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": MANIFEST_INDEX_SCHEMA_VERSION,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "project_root": str(PROJECT_ROOT),
        "plan_file": str(plan_path),
        "args": vars(args),
        "planned_command_count": len(commands),
        "manifest_files": [],
        "extra_manifest_files": [],
        "commands": [],
    }
    index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_manifest_index_command(
    index_path: Path,
    item: FinalCommand,
    observed: dict[str, object],
) -> None:
    if not index_path.exists():
        _write_initial_manifest_index(index_path, plan_path=Path(""), args=argparse.Namespace(), commands=[])
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    commands = payload.setdefault("commands", [])
    if not isinstance(commands, list):
        commands = []
        payload["commands"] = commands
    manifest_files = _string_list(observed.get("manifest_files", []))
    extra_manifest_files = _string_list(observed.get("extra_manifest_files", []))
    returncode = int(observed.get("returncode", 1))
    commands.append(
        {
            "track": item.track,
            "name": item.name,
            "status": "completed" if returncode == 0 else "failed",
            "returncode": returncode,
            "manifest_files": manifest_files,
            "extra_manifest_files": extra_manifest_files,
            "final_readiness_files": _string_list(observed.get("final_readiness_files", [])),
            "evidence_issues": _string_list(observed.get("evidence_issues", [])),
            "shell": shell_join(item.command),
        }
    )
    payload["manifest_files"] = _unique_strings(
        [*_string_list(payload.get("manifest_files", [])), *manifest_files]
    )
    payload["extra_manifest_files"] = _unique_strings(
        [*_string_list(payload.get("extra_manifest_files", [])), *extra_manifest_files]
    )
    payload["updated_at"] = utc_now()
    index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _string_list(value: object) -> list[str]:
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, (list, tuple, set)):
        items = [str(item) for item in value]
    else:
        items = []
    return [item.strip() for item in items if item.strip()]


def _unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def jobs_arg(value: object) -> str:
    text = str(value).strip().lower()
    if text == "auto":
        return "auto"
    return str(max(1, int(text)))


def max_parallel_cost_args(value: object) -> list[str]:
    if value is None:
        return []
    return ["--max-parallel-cost", f"{max(0.0, float(value)):g}"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the frozen final experiment command set. By default this "
            "only writes/prints a plan; pass --execute to run it."
        ),
    )
    parser.add_argument(
        "--track",
        choices=["all", *FINAL_PLAN_TRACKS, *FINAL_MATRIX_TRACK_ALIASES],
        default="all",
        help="experiment track or final matrix id to plan",
    )
    parser.add_argument("--duration", default="24h", help="per-run wall-clock budget for live discovery")
    parser.add_argument(
        "--live-batch-duration",
        default="10m",
        help="adaptive live-discovery batch wall-clock budget; longer batches reduce scheduler/IO overhead",
    )
    parser.add_argument(
        "--adaptive-learning-weight",
        type=float,
        default=0.75,
        help="adaptive scheduler contextual-learning weight for final live and adaptive-ablation runs",
    )
    parser.add_argument(
        "--scheduler-annealing-temperature",
        type=float,
        default=0.35,
        help="initial annealing temperature for final adaptive scheduling; 0 disables annealed selection",
    )
    parser.add_argument(
        "--scheduler-annealing-decay",
        type=float,
        default=0.985,
        help="per-completed-batch decay for final adaptive scheduler annealing",
    )
    parser.add_argument(
        "--scheduler-annealing-min-temperature",
        type=float,
        default=0.02,
        help="minimum nonzero final adaptive scheduler annealing temperature",
    )
    parser.add_argument(
        "--continual-learning-ledgers",
        default="",
        help="comma-separated version ledgers used to cold-start final adaptive continual learning",
    )
    parser.add_argument("--validation-cases", type=int, default=200, help="cases per short validation run")
    parser.add_argument("--validation-seeds", default="1,101", help="short validation seeds")
    parser.add_argument("--live-seeds", default="1,1001,2001", help="comma-separated live discovery seeds")
    parser.add_argument(
        "--live-campaign",
        action="append",
        default=[],
        help=(
            "restrict --track live/live_discovery to one or more campaign names of the "
            "form target_suite:preset; useful for parallel 24h sessions with independent indexes"
        ),
    )
    parser.add_argument("--historical-seeds", default=None, help="override historical replay seeds")
    parser.add_argument("--seeded-cases", type=int, default=5000, help="cases per seeded sensitivity run")
    parser.add_argument("--seeded-seeds", default="1,1001,2001,3001,4001", help="seeded sensitivity seeds")
    parser.add_argument("--ablation-cases", type=int, default=2000, help="cases per module-ablation run")
    parser.add_argument("--ablation-seeds", default="1,1001,2001", help="module-ablation seeds")
    parser.add_argument("--comparison-cases", type=int, default=2000, help="cases per baseline/comparison run")
    parser.add_argument("--comparison-seeds", default="1,1001,2001", help="baseline/comparison seeds")
    parser.add_argument("--jobs", default="auto", help="parallel experiment jobs, or 'auto'")
    parser.add_argument(
        "--max-parallel-cost",
        type=float,
        default=None,
        help="override datadiff experiment cost limiter when planning parallel final runs",
    )
    parser.add_argument("--artifact-limit", type=int, default=50, help="bug artifacts per run")
    parser.add_argument(
        "--log-level",
        choices=["full", "compact", "minimal"],
        default="minimal",
        help="run JSONL detail level; minimal is the default for 24h runs to reduce per-case IO",
    )
    parser.add_argument("--include-pending-historical", action="store_true")
    parser.add_argument("--skip-run-reports", action="store_true", default=True)
    parser.add_argument(
        "--strategy-snapshot",
        default="",
        help=(
            "existing frozen strategy snapshot to reuse; default reuses "
            "reports/strategy-snapshots/final-frozen-strategy-snapshot.json"
        ),
    )
    parser.add_argument(
        "--reset-strategy-snapshot",
        action="store_true",
        help="regenerate the default final frozen strategy snapshot before planning this track",
    )
    parser.add_argument(
        "--ledger-run-files",
        default="",
        help="comma-separated cross-version run logs used to build the final regression ledger",
    )
    parser.add_argument(
        "--ledger-collection-cases",
        type=int,
        default=200,
        help="cases per C5 source/target collection run when --track version_ledger is executed",
    )
    parser.add_argument(
        "--ledger-collection-seeds",
        default="1,1001",
        help="seeds for C5 source/target collection runs",
    )
    parser.add_argument(
        "--ledger-source-version",
        default="c5-source",
        help="source dependency version/commit label for C5 champion-transfer collection",
    )
    parser.add_argument(
        "--ledger-target-version",
        default="c5-target",
        help="target dependency version/commit label for C5 champion-transfer collection",
    )
    parser.add_argument(
        "--ledger-target-suite",
        default="datafusion_cross",
        help="target suite for C5 source/target collection runs",
    )
    parser.add_argument(
        "--ledger-preset",
        default="live_deep_organic",
        help="preset for C5 source/target collection runs",
    )
    parser.add_argument(
        "--ledger-version-pair-learning-weight",
        type=float,
        default=0.75,
        help="version-pair learning weight used by the C5 target collection run",
    )
    parser.add_argument(
        "--ledger-versions",
        default="",
        help="comma-separated version ids matching --ledger-run-files",
    )
    parser.add_argument("--previous-ledger", default="", help="optional previous version ledger for regression detection")
    parser.add_argument(
        "--ledger-output",
        default=str(REPORTS_DIR / "final-version-ledger.json"),
        help="output path for the final cross-version ledger",
    )
    parser.add_argument(
        "--ledger-evidence-manifest",
        default=str(REPORTS_DIR / "experiment-final-version-ledger.json"),
        help="manifest path that exposes the version ledger to final readiness",
    )
    parser.add_argument(
        "--manifest-index",
        default=str(REPORTS_DIR / "final-experiment-manifest-index.json"),
        help=(
            "JSON index populated during --execute with final experiment manifest paths; "
            "postprocess audits this file instead of sweeping stale runs/experiment-*.json files"
        ),
    )
    parser.add_argument(
        "--import-manifest",
        action="append",
        default=[],
        help=(
            "existing experiment manifest to import into --manifest-index and paper-run-journal "
            "without rerunning the experiment; may be repeated"
        ),
    )
    parser.add_argument(
        "--import-extra-manifest",
        action="append",
        default=[],
        help=(
            "existing support evidence manifest to import into --manifest-index, such as "
            "reports/experiment-final-version-ledger.json; may be repeated"
        ),
    )
    parser.add_argument(
        "--paper-run-journal",
        default=str(REPORTS_DIR / "paper-run-journal.jsonl"),
        help="paper-run journal JSONL path used when importing existing experiment manifests",
    )
    parser.add_argument(
        "--reset-manifest-index",
        action="store_true",
        help="when executing a non-postprocess track, start a fresh manifest index before appending evidence",
    )
    parser.add_argument("--execute", action="store_true", help="execute commands instead of only printing them")
    return parser.parse_args()


def build_plan(args: argparse.Namespace) -> list[FinalCommand]:
    commands: list[FinalCommand] = []
    strategy_snapshot = _resolve_strategy_snapshot(args)
    selected = _selected_tracks(args.track)
    if _track_selected(selected, "validation", FINAL_VALIDATION_MATRIX.id):
        commands.append(short_validation_command(args, strategy_snapshot=strategy_snapshot))
    if _track_selected(selected, "live", FINAL_LIVE_DISCOVERY_MATRIX.id):
        commands.extend(live_discovery_commands(args, strategy_snapshot=strategy_snapshot))
    if _track_selected(selected, "historical", "historical_replay"):
        commands.extend(historical_replay_commands(args, strategy_snapshot=strategy_snapshot))
    if _track_selected(selected, "seeded", FINAL_SEEDED_SENSITIVITY_MATRIX.id):
        commands.append(seeded_sensitivity_command(args, strategy_snapshot=strategy_snapshot))
    if _track_selected(selected, "ablation", FINAL_MODULE_ABLATION_MATRIX.id):
        commands.append(module_ablation_command(args, strategy_snapshot=strategy_snapshot))
    if _track_selected(selected, "ablation", FINAL_ADAPTIVE_COMPONENT_ABLATION_MATRIX.id):
        commands.extend(adaptive_component_ablation_commands(args, strategy_snapshot=strategy_snapshot))
    if _track_selected(selected, "comparison", FINAL_COMPARISON_MATRIX.id):
        commands.append(method_comparison_command(args, strategy_snapshot=strategy_snapshot))
    if "version_ledger" in selected and not str(getattr(args, "ledger_run_files", "") or "").strip():
        commands.extend(version_ledger_collection_commands(args, strategy_snapshot=strategy_snapshot))
    if _track_selected(selected, "comparison", "version_ledger"):
        ledger_command = version_ledger_evidence_command(args)
        if ledger_command is not None:
            commands.append(ledger_command)
    if _track_selected(selected, "postprocess"):
        commands.append(final_readiness_audit_command(args))
    return commands


def _selected_tracks(track: str) -> set[str]:
    if track == "all":
        return set(FINAL_PLAN_TRACKS)
    return {track}


def _track_selected(selected: set[str], *aliases: str) -> bool:
    return any(alias in selected for alias in aliases)


def _append_experiment_meta(cmd: list[str], meta: dict[str, object]) -> None:
    cmd.extend(["--experiment-meta", json.dumps(meta, ensure_ascii=False, sort_keys=True)])


def _resolve_strategy_snapshot(args: argparse.Namespace) -> str:
    configured = str(getattr(args, "strategy_snapshot", "") or "").strip()
    if configured:
        return configured
    if DEFAULT_FINAL_STRATEGY_SNAPSHOT.is_file() and not bool(
        getattr(args, "reset_strategy_snapshot", False)
    ):
        return str(DEFAULT_FINAL_STRATEGY_SNAPSHOT)
    snapshot = write_strategy_snapshot(
        classification_documented_rules=list(documented_semantic_rule_records()),
        classification_boundary_rules=list(semantic_boundary_rule_records()),
        reproducer_rules=list(standalone_reproducer_rule_records()),
        metadata={
            "generated_by": "scripts/run_final_experiments.py",
            "track": str(args.track),
            "freeze_role": "final_experiment_strategy",
            "reuse_policy": "reuse_default_until_reset",
        },
        output_dir=DEFAULT_FINAL_STRATEGY_SNAPSHOT.parent,
        snapshot_id=DEFAULT_FINAL_STRATEGY_SNAPSHOT.stem,
    )
    return str(snapshot)


def _append_strategy_snapshot_args(cmd: list[str], *, strategy_snapshot: str) -> None:
    if strategy_snapshot:
        cmd.extend(["--strategy-snapshot", strategy_snapshot, "--freeze-strategy-snapshot"])


def short_validation_command(args: argparse.Namespace, *, strategy_snapshot: str) -> FinalCommand:
    cmd = [
        str(DATADIFF),
        "experiment",
        "--cases",
        str(max(1, int(args.validation_cases))),
        "--seeds",
        str(args.validation_seeds),
        "--presets",
        ",".join(variant.preset for variant in FINAL_VALIDATION_MATRIX.variants),
        "--target-suites",
        ",".join(FINAL_VALIDATION_MATRIX.target_suites),
        "--evidence-mode",
        "validation",
        "--run-theme",
        "final-validation-smoke",
        "--paper-notes",
        (
            "Short pre-freeze validation over live target families; inspect run-health, "
            "classify-run, experiment-summary, and methodology-report before starting 24h runs."
        ),
        "--replay-bug-source-issues",
        ",".join(replay_source_issues()),
        "--artifact-limit",
        str(max(0, int(args.artifact_limit))),
        "--log-level",
        "compact",
        "--jobs",
        jobs_arg(args.jobs),
        "--skip-run-reports",
    ]
    cmd.extend(max_parallel_cost_args(getattr(args, "max_parallel_cost", None)))
    _append_strategy_snapshot_args(cmd, strategy_snapshot=strategy_snapshot)
    _append_experiment_meta(
        cmd,
        FINAL_VALIDATION_MATRIX.command_experiment_meta(
            target_suites=FINAL_VALIDATION_MATRIX.target_suites,
        ),
    )
    return FinalCommand(
        track="validation",
        name="short_validation_smoke",
        command=cmd,
        purpose=(
            "Run a short validation-mode smoke matrix before the frozen 24h campaigns to catch "
            "adapter, oracle, preflight, classification, and evidence-pipeline noise."
        ),
        count_as_real_bugs=False,
        expected_output=(
            "short experiment manifest plus run-health/classify-run/experiment-summary/"
            "methodology-report checks; do not count candidates as final 24h bug evidence"
        ),
        notes=(
            "If this track exposes harness noise, fix it before freezing and regenerate all "
            "final plans. Validation keeps enable_replay_bug=false and only gates readiness."
        ),
        replay_bug_policy={
            "enable_replay_bug": False,
            "source_issues": replay_source_issues(),
        },
        experiment_meta=FINAL_VALIDATION_MATRIX.command_experiment_meta(
            target_suites=FINAL_VALIDATION_MATRIX.target_suites,
        ),
    )


def live_discovery_commands(args: argparse.Namespace, *, strategy_snapshot: str) -> list[FinalCommand]:
    commands = []
    replay_sources = replay_source_issues()
    selected_campaigns = _selected_live_campaigns(args)
    for campaign in FINAL_LIVE_DISCOVERY_MATRIX.campaigns:
        suite, preset, purpose = campaign.suite, campaign.preset, campaign.purpose
        if selected_campaigns and f"{suite}:{preset}" not in selected_campaigns:
            continue
        cmd = [
            str(DATADIFF),
            "experiment",
            "--duration",
            str(args.duration),
            "--seeds",
            str(args.live_seeds),
            "--presets",
            preset,
            "--target-suite",
            suite,
            "--evidence-mode",
            "live",
            "--schedule",
            "adaptive",
            "--batch-duration",
            str(getattr(args, "live_batch_duration", "10m") or "10m"),
            "--adaptive-learning-weight",
            str(_adaptive_learning_weight(args)),
            "--scheduler-annealing-temperature",
            str(_scheduler_annealing_temperature(args)),
            "--scheduler-annealing-decay",
            str(_scheduler_annealing_decay(args)),
            "--scheduler-annealing-min-temperature",
            str(_scheduler_annealing_min_temperature(args)),
            "--run-theme",
            f"final-live:{suite}:{preset}",
            "--paper-notes",
            purpose,
            "--replay-bug-source-issues",
            ",".join(replay_sources),
            "--artifact-limit",
            str(max(0, int(args.artifact_limit))),
            "--log-level",
            str(args.log_level),
            "--jobs",
            jobs_arg(args.jobs),
            "--persist-closed-loop-state",
        ]
        _append_continual_learning_args(cmd, args)
        if args.skip_run_reports:
            cmd.append("--skip-run-reports")
        cmd.extend(max_parallel_cost_args(getattr(args, "max_parallel_cost", None)))
        _append_strategy_snapshot_args(cmd, strategy_snapshot=strategy_snapshot)
        experiment_meta = FINAL_LIVE_DISCOVERY_MATRIX.command_experiment_meta_for_campaign(campaign)
        _append_experiment_meta(cmd, experiment_meta)
        commands.append(
            FinalCommand(
                track="live",
                name=f"{suite}:{preset}",
                command=cmd,
                purpose=purpose,
                count_as_real_bugs=True,
                expected_output=(
                    "runs/experiment-*.json plus experiment-summary/analyze-experiment/"
                    "methodology-report outputs"
                ),
                notes=(
                    "Fresh/latest mode keeps enable_replay_bug=false and filters known replay probes. "
                    "Run without changing generator/oracle code after inspecting findings. "
                    "Count unique candidate bug families, then separately mark maintainer-confirmed bugs."
                ),
                replay_bug_policy={
                    "enable_replay_bug": False,
                    "source_issues": replay_sources,
                },
                experiment_meta=experiment_meta,
            )
        )
    return commands


def _selected_live_campaigns(args: argparse.Namespace) -> set[str]:
    selected: set[str] = set()
    raw_values = getattr(args, "live_campaign", []) or []
    if isinstance(raw_values, str):
        raw_values = [raw_values]
    valid = {
        f"{campaign.suite}:{campaign.preset}"
        for campaign in FINAL_LIVE_DISCOVERY_MATRIX.campaigns
    }
    for raw_value in raw_values:
        for item in str(raw_value or "").split(","):
            name = item.strip()
            if not name:
                continue
            if name not in valid:
                raise SystemExit(
                    f"unknown live campaign {name!r}; expected one of: {', '.join(sorted(valid))}"
                )
            selected.add(name)
    return selected


def historical_replay_commands(args: argparse.Namespace, *, strategy_snapshot: str) -> list[FinalCommand]:
    specs = list_historical_bugs(include_pending=bool(args.include_pending_historical))
    commands: list[FinalCommand] = []
    for spec in specs:
        if spec.replay_kind == "fixture":
            commands.append(_historical_fixture_replay_command(spec, args, strategy_snapshot=strategy_snapshot))
            continue
        seeds = args.historical_seeds or ",".join(str(seed) for seed in spec.default_seeds)
        artifact_limit = (
            args.artifact_limit
            if getattr(spec, "default_artifact_limit", None) is None
            else int(getattr(spec, "default_artifact_limit"))
        )
        log_level = str(getattr(spec, "default_log_level", "") or args.log_level)
        replay_sources = replay_source_issues(spec.issue_url)
        cmd = [
            str(DATADIFF),
            "experiment",
            "--cases",
            str(spec.default_cases),
            "--seeds",
            seeds,
            "--presets",
            ",".join(spec.default_presets),
            "--target-suite",
            spec.target_suite,
            "--evidence-mode",
            "historical",
            "--known-bug-id",
            spec.bug_id,
            "--target-version",
            spec.target_version,
            "--enable-replay-bug",
            "--replay-bug-source-issues",
            ",".join(replay_sources),
            "--run-theme",
            f"final-historical:{spec.bug_id}",
            "--paper-notes",
            f"Historical replay for {spec.bug_id}; status={spec.status}.",
            "--artifact-limit",
            str(max(0, int(artifact_limit))),
            "--log-level",
            log_level,
            "--jobs",
            jobs_arg(args.jobs),
        ]
        if args.skip_run_reports:
            cmd.append("--skip-run-reports")
        cmd.extend(max_parallel_cost_args(getattr(args, "max_parallel_cost", None)))
        _append_strategy_snapshot_args(cmd, strategy_snapshot=strategy_snapshot)
        experiment_meta = build_historical_experiment_meta(spec)
        _append_experiment_meta(cmd, experiment_meta)
        commands.append(
            FinalCommand(
                track="historical",
                name=spec.bug_id,
                command=cmd,
                purpose=f"Replay previously reported {spec.project} bug on a vulnerable target version.",
                count_as_real_bugs=spec.status == "confirmed_fixed",
                expected_output="historical experiment manifest and expected-root detection metrics",
                notes=(
                    f"status={spec.status}; run inside an environment whose backend version is "
                    f"{spec.target_version}. Replay mode sets enable_replay_bug=true but uses the "
                    f"same generator/oracle/runner path as fresh mode. {spec.notes}"
                ).strip(),
                replay_bug_policy={
                    "enable_replay_bug": True,
                    "source_issues": replay_sources,
                },
                experiment_meta=experiment_meta,
            )
        )
    return commands


def _historical_fixture_replay_command(
    spec: object,
    args: argparse.Namespace,
    *,
    strategy_snapshot: str,
) -> FinalCommand:
    replay_sources = replay_source_issues(str(getattr(spec, "issue_url", "")))
    cmd = [
        str(DATADIFF),
        "replay-fixture",
        "--spec",
        str(getattr(spec, "fixture_spec")),
        "--fixture-env",
        str(getattr(spec, "fixture_env")),
        "--target-suite",
        str(getattr(spec, "target_suite")),
        "--evidence-mode",
        "historical",
        "--known-bug-id",
        str(getattr(spec, "bug_id")),
        "--target-version",
        str(getattr(spec, "target_version")),
        "--run-theme",
        f"final-historical:{getattr(spec, 'bug_id')}",
        "--paper-notes",
        f"Historical fixture replay for {getattr(spec, 'bug_id')}; status={getattr(spec, 'status')}.",
        "--artifact-limit",
        str(max(0, int(args.artifact_limit))),
        "--log-level",
        str(args.log_level),
    ]
    _append_strategy_snapshot_args(cmd, strategy_snapshot=strategy_snapshot)
    experiment_meta = build_historical_experiment_meta(spec)
    cmd.extend(["--experiment-meta", json.dumps(experiment_meta, ensure_ascii=False, sort_keys=True)])
    return FinalCommand(
        track="historical",
        name=str(getattr(spec, "bug_id")),
        command=cmd,
        purpose=f"Replay previously reported {getattr(spec, 'project')} bug using an upstream fixture.",
        count_as_real_bugs=getattr(spec, "status") == "confirmed_fixed",
        expected_output="runs/run-fixture-*.jsonl plus paper run journal entry",
        notes=(
            f"status={getattr(spec, 'status')}; set {getattr(spec, 'fixture_env')} to the "
            f"external fixture path before execution. {getattr(spec, 'notes')}"
        ).strip(),
        replay_bug_policy={
            "enable_replay_bug": True,
            "source_issues": replay_sources,
        },
        experiment_meta=experiment_meta,
    )


def seeded_sensitivity_command(args: argparse.Namespace, *, strategy_snapshot: str) -> FinalCommand:
    cmd = [
        str(DATADIFF),
        "experiment",
        "--cases",
        str(max(1, int(args.seeded_cases))),
        "--seeds",
        str(args.seeded_seeds),
        "--presets",
        ",".join(variant.preset for variant in FINAL_SEEDED_SENSITIVITY_MATRIX.variants),
        "--target-suites",
        ",".join(FINAL_SEEDED_SENSITIVITY_MATRIX.target_suites),
        "--evidence-mode",
        "seeded",
        "--run-theme",
        "final-seeded-sensitivity",
        "--paper-notes",
        "Controlled injected-fault sensitivity run; not counted as real backend bugs.",
        "--artifact-limit",
        "1",
        "--log-level",
        "minimal",
        "--jobs",
        jobs_arg(args.jobs),
        "--skip-run-reports",
    ]
    cmd.extend(max_parallel_cost_args(getattr(args, "max_parallel_cost", None)))
    _append_strategy_snapshot_args(cmd, strategy_snapshot=strategy_snapshot)
    experiment_meta = FINAL_SEEDED_SENSITIVITY_MATRIX.command_experiment_meta(
        target_suites=FINAL_SEEDED_SENSITIVITY_MATRIX.target_suites,
    )
    _append_experiment_meta(cmd, experiment_meta)
    return FinalCommand(
        track="seeded",
        name="seeded_sensitivity",
        command=cmd,
        purpose="Measure detection sensitivity and time-to-first on controlled injected faults.",
        count_as_real_bugs=False,
        expected_output="seeded-sensitivity report; do not include these in real bug counts",
        notes="Use this to support method validity, not as backend bug evidence.",
        replay_bug_policy={
            "enable_replay_bug": False,
            "source_issues": replay_source_issues(),
        },
        experiment_meta=experiment_meta,
    )


def module_ablation_command(args: argparse.Namespace, *, strategy_snapshot: str) -> FinalCommand:
    cmd = [
        str(DATADIFF),
        "experiment",
        "--cases",
        str(max(1, int(args.ablation_cases))),
        "--seeds",
        str(args.ablation_seeds),
        "--presets",
        ",".join(variant.preset for variant in FINAL_MODULE_ABLATION_MATRIX.variants),
        "--target-suites",
        ",".join(FINAL_MODULE_ABLATION_MATRIX.target_suites),
        "--evidence-mode",
        "ablation",
        "--run-theme",
        "final-ablation-modules",
        "--paper-notes",
        "Module ablation for generator typing, normalizer, feedback, reducer, and oracle composition.",
        "--replay-bug-source-issues",
        ",".join(replay_source_issues()),
        "--artifact-limit",
        str(max(0, int(args.artifact_limit))),
        "--log-level",
        str(args.log_level),
        "--jobs",
        jobs_arg(args.jobs),
        "--skip-run-reports",
    ]
    cmd.extend(max_parallel_cost_args(getattr(args, "max_parallel_cost", None)))
    _append_strategy_snapshot_args(cmd, strategy_snapshot=strategy_snapshot)
    experiment_meta = FINAL_MODULE_ABLATION_MATRIX.command_experiment_meta(
        target_suites=FINAL_MODULE_ABLATION_MATRIX.target_suites,
    )
    _append_experiment_meta(cmd, experiment_meta)
    return FinalCommand(
        track="ablation",
        name="module_ablation",
        command=cmd,
        purpose=(
            "Quantify sensitivity of type-aware generation, semantic normalization, feedback, "
            "reducer, and oracle composition across core target families."
        ),
        count_as_real_bugs=False,
        expected_output=(
            "experiment manifest plus experiment-summary/analyze-experiment/"
            "analyze-ablation-audit/methodology-report outputs"
        ),
        notes=(
            "Use for RQ ablation and baseline comparison tables. Candidate bugs from this track "
            "require the same live confirmation pipeline before they can be counted as real bugs."
        ),
        replay_bug_policy={
            "enable_replay_bug": False,
            "source_issues": replay_source_issues(),
        },
        experiment_meta=experiment_meta,
    )


def adaptive_component_ablation_commands(args: argparse.Namespace, *, strategy_snapshot: str) -> list[FinalCommand]:
    commands: list[FinalCommand] = []
    for variant in FINAL_ADAPTIVE_COMPONENT_ABLATION_MATRIX.variants:
        disabled_components = []
        if variant.id == "no_scheduler_learning":
            disabled_components.append("scheduler-learning")
        if variant.id == "no_scheduler_annealing":
            disabled_components.append("scheduler-annealing")
        if variant.id == "no_online_reward_model":
            disabled_components.append("online-reward-model")
        if variant.id == "no_continual_learning":
            disabled_components.append("continual-learning")
        if variant.id == "no_runtime_cost_learning":
            disabled_components.append("runtime-cost-learning")
        if variant.id == "no_quality_archive":
            disabled_components.append("quality-archive")
        if variant.id == "no_bd_axis_bandit":
            disabled_components.append("bd-axis-bandit")
        if variant.id == "no_bayesian_exploration":
            disabled_components.append("bayesian-exploration")
        if variant.id == "no_value_catalog":
            disabled_components.append("value-catalog")
        if variant.id == "no_hierarchical_archive":
            disabled_components.append("hierarchical-archive")
        if variant.id == "no_seed_quota":
            disabled_components.append("seed-quota")
        if variant.id == "no_seed_energy_batch":
            disabled_components.append("seed-energy-batch")
        if variant.id == "no_seed_energy_tier":
            disabled_components.append("seed-energy-tier")
        if variant.id == "no_per_operator_energy":
            disabled_components.append("per-operator-energy")
        if variant.id == "no_ir_rewrite_mutations":
            disabled_components.append("ir-rewrite-mutations")
        if variant.id == "no_operator_swarm":
            disabled_components.append("operator-swarm")
        if variant.id == "no_divergence_conditioned":
            disabled_components.append("divergence-conditioned")
        if variant.id == "no_shrink_mutations":
            disabled_components.append("shrink-mutations")
        if variant.id == "no_lineage_rarity":
            disabled_components.append("lineage-rarity")
        if variant.id == "no_minhash_dedup":
            disabled_components.append("minhash-dedup")
        if variant.id == "no_disagreement_bd_axis":
            disabled_components.append("disagreement-bd-axis")
        if variant.id == "no_lhs_seeding":
            disabled_components.append("lhs-seeding")
        if variant.id == "no_champion_corpus":
            disabled_components.append("champion-corpus")
        if variant.id == "no_champion_graft_donor":
            disabled_components.append("champion-graft-donor")
        if variant.id == "no_backend_pair_learning":
            disabled_components.append("backend-pair-learning")
        if variant.id == "no_cost_normalized_reward":
            disabled_components.append("cost-normalized-reward")
        if variant.id == "no_active_learning":
            disabled_components.append("active-learning")
        cmd = [
            str(DATADIFF),
            "experiment",
            "--cases",
            str(max(1, int(args.ablation_cases))),
            "--seeds",
            str(args.ablation_seeds),
            "--presets",
            variant.preset,
            "--target-suite",
            FINAL_ADAPTIVE_COMPONENT_ABLATION_MATRIX.target_suites[0],
            "--evidence-mode",
            "ablation",
            "--schedule",
            "adaptive",
            "--batch-cases",
            str(min(100, max(1, int(args.ablation_cases)))),
            "--adaptive-learning-weight",
            str(_adaptive_learning_weight(args)),
            "--scheduler-annealing-temperature",
            str(_scheduler_annealing_temperature(args)),
            "--scheduler-annealing-decay",
            str(_scheduler_annealing_decay(args)),
            "--scheduler-annealing-min-temperature",
            str(_scheduler_annealing_min_temperature(args)),
            "--run-theme",
            f"final-adaptive-ablation:{variant.id}",
            "--paper-notes",
            variant.notes
            or "Adaptive component ablation with fixed target, preset, seed budget, and oracle.",
            "--replay-bug-source-issues",
            ",".join(replay_source_issues()),
            "--artifact-limit",
            str(max(0, int(args.artifact_limit))),
            "--log-level",
            str(args.log_level),
            "--jobs",
            jobs_arg(args.jobs),
            "--skip-run-reports",
        ]
        _append_continual_learning_args(cmd, args)
        if disabled_components:
            cmd.extend(["--disable-adaptive-components", ",".join(disabled_components)])
        cmd.extend(max_parallel_cost_args(getattr(args, "max_parallel_cost", None)))
        _append_strategy_snapshot_args(cmd, strategy_snapshot=strategy_snapshot)
        experiment_meta = _adaptive_component_experiment_meta(variant)
        _append_experiment_meta(cmd, experiment_meta)
        commands.append(
            FinalCommand(
                track="ablation",
                name=f"adaptive_component_ablation:{variant.id}",
                command=cmd,
                purpose=FINAL_ADAPTIVE_COMPONENT_ABLATION_MATRIX.purpose,
                count_as_real_bugs=False,
                expected_output=(
                    "adaptive experiment manifest plus methodology-report adaptive component "
                    "rows comparing throughput, invalid rate, false positive rate, and candidate yield"
                ),
                notes=(
                    "Use together with module_ablation. This isolates closed-loop adaptive components "
                    "without changing generator preset, target suite, oracle mode, or seed budget."
                ),
                replay_bug_policy={
                    "enable_replay_bug": False,
                    "source_issues": replay_source_issues(),
                },
                experiment_meta=experiment_meta,
            )
        )
    return commands


def _append_continual_learning_args(cmd: list[str], args: argparse.Namespace) -> None:
    ledgers = str(getattr(args, "continual_learning_ledgers", "") or "").strip()
    if ledgers:
        cmd.extend(["--continual-learning-ledgers", ledgers])


def _adaptive_learning_weight(args: argparse.Namespace) -> float:
    return max(0.0, float(getattr(args, "adaptive_learning_weight", 0.75) or 0.0))


def _scheduler_annealing_temperature(args: argparse.Namespace) -> float:
    return max(0.0, float(getattr(args, "scheduler_annealing_temperature", 0.35) or 0.0))


def _scheduler_annealing_decay(args: argparse.Namespace) -> float:
    return min(1.0, max(0.0, float(getattr(args, "scheduler_annealing_decay", 0.985) or 0.0)))


def _scheduler_annealing_min_temperature(args: argparse.Namespace) -> float:
    return max(0.0, float(getattr(args, "scheduler_annealing_min_temperature", 0.02) or 0.0))


def _adaptive_component_experiment_meta(variant: object) -> dict[str, object]:
    meta = FINAL_ADAPTIVE_COMPONENT_ABLATION_MATRIX.to_experiment_meta(
        target_suites=FINAL_ADAPTIVE_COMPONENT_ABLATION_MATRIX.target_suites,
    )
    variant_meta = dict(variant.to_meta())
    variant_meta.pop("preset", None)
    meta["variant"] = variant_meta
    return meta


def method_comparison_command(args: argparse.Namespace, *, strategy_snapshot: str) -> FinalCommand:
    cmd = [
        str(DATADIFF),
        "experiment",
        "--cases",
        str(max(1, int(args.comparison_cases))),
        "--seeds",
        str(args.comparison_seeds),
        "--presets",
        ",".join(variant.preset for variant in FINAL_COMPARISON_MATRIX.variants),
        "--target-suites",
        ",".join(FINAL_COMPARISON_MATRIX.target_suites),
        "--evidence-mode",
        "comparison",
        "--run-theme",
        "final-baseline-and-scope-comparison",
        "--paper-notes",
        (
            "Baseline and related-scope comparison: SQL/DBMS-style target suites versus "
            "cross-ecosystem DataFrame/Arrow/SQL target suites under the same harness."
        ),
        "--replay-bug-source-issues",
        ",".join(replay_source_issues()),
        "--artifact-limit",
        str(max(0, int(args.artifact_limit))),
        "--log-level",
        str(args.log_level),
        "--jobs",
        jobs_arg(args.jobs),
        "--skip-run-reports",
    ]
    cmd.extend(max_parallel_cost_args(getattr(args, "max_parallel_cost", None)))
    _append_strategy_snapshot_args(cmd, strategy_snapshot=strategy_snapshot)
    experiment_meta = FINAL_COMPARISON_MATRIX.command_experiment_meta(
        target_suites=FINAL_COMPARISON_MATRIX.target_suites,
    )
    _append_experiment_meta(cmd, experiment_meta)
    return FinalCommand(
        track="comparison",
        name="baseline_and_related_scope",
        command=cmd,
        purpose=(
            "Compare random/guided/metamorphic/workflow presets and SQL/query-engine-only "
            "scope against the cross-ecosystem DataDiffFuzz scope."
        ),
        count_as_real_bugs=False,
        expected_output="experiment manifest plus baseline, methodology, and space/time efficiency analysis outputs",
        notes=(
            "This is not a reimplementation of SQLancer/SQUIRREL; it is a controlled scope "
            "baseline inside the same runner, used to isolate what DataFrame/Arrow/cross-family "
            "coverage adds beyond SQL/query-engine-oriented testing."
        ),
        replay_bug_policy={
            "enable_replay_bug": False,
            "source_issues": replay_source_issues(),
        },
        experiment_meta=experiment_meta,
    )


def version_ledger_collection_commands(args: argparse.Namespace, *, strategy_snapshot: str) -> list[FinalCommand]:
    source_version = str(getattr(args, "ledger_source_version", "") or "c5-source").strip()
    target_version = str(getattr(args, "ledger_target_version", "") or "c5-target").strip()
    cases = max(1, int(getattr(args, "ledger_collection_cases", 200) or 200))
    seeds = str(getattr(args, "ledger_collection_seeds", "") or "1,1001").strip()
    preset = str(getattr(args, "ledger_preset", "") or "live_deep_organic").strip()
    target_suite = str(getattr(args, "ledger_target_suite", "") or "datafusion_cross").strip()
    common = [
        str(DATADIFF),
        "experiment",
        "--cases",
        str(cases),
        "--seeds",
        seeds,
        "--presets",
        preset,
        "--target-suite",
        target_suite,
        "--evidence-mode",
        "comparison",
        "--schedule",
        "adaptive",
        "--batch-cases",
        str(min(100, cases)),
        "--adaptive-learning-weight",
        str(_adaptive_learning_weight(args)),
        "--scheduler-annealing-temperature",
        str(_scheduler_annealing_temperature(args)),
        "--scheduler-annealing-decay",
        str(_scheduler_annealing_decay(args)),
        "--scheduler-annealing-min-temperature",
        str(_scheduler_annealing_min_temperature(args)),
        "--replay-bug-source-issues",
        ",".join(replay_source_issues()),
        "--artifact-limit",
        str(max(0, int(args.artifact_limit))),
        "--log-level",
        str(args.log_level),
        "--jobs",
        jobs_arg(args.jobs),
        "--skip-run-reports",
        "--persist-closed-loop-state",
    ]
    _append_continual_learning_args(common, args)
    common.extend(max_parallel_cost_args(getattr(args, "max_parallel_cost", None)))
    _append_strategy_snapshot_args(common, strategy_snapshot=strategy_snapshot)
    source_meta = _version_transfer_experiment_meta(
        role="source",
        source_version=source_version,
        target_version=target_version,
        target_suite=target_suite,
        preset=preset,
    )
    target_meta = _version_transfer_experiment_meta(
        role="target",
        source_version=source_version,
        target_version=target_version,
        target_suite=target_suite,
        preset=preset,
    )
    source_cmd = [
        *common,
        "--target-version",
        source_version,
        "--run-theme",
        f"final-c5-transfer:source:{source_version}",
        "--paper-notes",
        (
            "C5 source-version collection run: promotes stable candidate families "
            "into the champion corpus for seed-level cross-version transfer."
        ),
    ]
    _append_experiment_meta(source_cmd, source_meta)
    target_cmd = [
        *common,
        "--target-version",
        target_version,
        "--fixed-version",
        source_version,
        "--version-pair-learning-weight",
        str(max(0.0, float(getattr(args, "ledger_version_pair_learning_weight", 0.75) or 0.0))),
        "--run-theme",
        f"final-c5-transfer:target:{source_version}->{target_version}",
        "--paper-notes",
        (
            "C5 target-version collection run: measures champion corpus injection, "
            "champion graft donor learning, and cold-start candidate yield."
        ),
    ]
    _append_experiment_meta(target_cmd, target_meta)
    return [
        FinalCommand(
            track="comparison",
            name="cross_version_transfer_source",
            command=source_cmd,
            purpose=(
                "Collect source-version candidate families and champion promotions for "
                "C5 cross-version seed transfer."
            ),
            count_as_real_bugs=False,
            expected_output="experiment manifest and run log tagged with the C5 source version",
            notes=(
                "Use exact dependency versions/commits via --ledger-source-version for paper evidence; "
                "default c5-source/c5-target labels are smoke labels only."
            ),
            replay_bug_policy={"enable_replay_bug": False, "source_issues": replay_source_issues()},
            experiment_meta=source_meta,
        ),
        FinalCommand(
            track="comparison",
            name="cross_version_transfer_target",
            command=target_cmd,
            purpose=(
                "Collect target-version champion injection/graft outcomes for C5 "
                "cross-version seed transfer."
            ),
            count_as_real_bugs=False,
            expected_output="experiment manifest and run log tagged with the C5 target version",
            notes=(
                "Run after the source collection in the matching target-version environment. "
                "The following version-ledger command converts both manifests into audited evidence."
            ),
            replay_bug_policy={"enable_replay_bug": False, "source_issues": replay_source_issues()},
            experiment_meta=target_meta,
        ),
    ]


def _version_transfer_experiment_meta(
    *,
    role: str,
    source_version: str,
    target_version: str,
    target_suite: str,
    preset: str,
) -> dict[str, object]:
    version = source_version if role == "source" else target_version
    return {
        "matrix_id": "baseline_scope_comparison",
        "comparison_group": "cross_version_continual_learning",
        "counts_as_real_bugs": False,
        "track": "comparison",
        "target_suites": [target_suite],
        "scope_kind": "cross_version",
        "analysis_tags": ["cross_version", "continual_learning", "champion_transfer", f"c5_{role}"],
        "variant": {
            "variant_id": f"c5_transfer_{role}",
            "comparison_role": "support",
            "component_focus": "cross_version_champion_transfer",
            "source_version": source_version,
            "target_version": target_version,
            "target_suite": target_suite,
            "preset": preset,
            "version": version,
        },
        "methodology_claim": (
            "C5 evidence is collected as a two-run source/target chain and reduced to a "
            "version ledger that reports seed-level champion transfer and cold-start yield."
        ),
    }


def version_ledger_evidence_command(args: argparse.Namespace) -> FinalCommand | None:
    run_files = str(getattr(args, "ledger_run_files", "") or "").strip()
    manifest_index = str(
        getattr(args, "manifest_index", "") or REPORTS_DIR / "final-experiment-manifest-index.json"
    ).strip()
    if not run_files and not manifest_index:
        return None
    ledger_output = str(getattr(args, "ledger_output", "") or REPORTS_DIR / "final-version-ledger.json")
    evidence_manifest = str(
        getattr(args, "ledger_evidence_manifest", "") or REPORTS_DIR / "experiment-final-version-ledger.json"
    )
    cmd = [
        str(DATADIFF),
        "version-ledger",
        "--output",
        ledger_output,
        "--evidence-manifest-output",
        evidence_manifest,
    ]
    if run_files:
        cmd.extend(["--run-files", run_files])
    else:
        cmd.extend(["--manifest-index", manifest_index])
    versions = str(getattr(args, "ledger_versions", "") or "").strip()
    if versions:
        cmd.extend(["--versions", versions])
    previous_ledger = str(getattr(args, "previous_ledger", "") or "").strip()
    if previous_ledger:
        cmd.extend(["--previous-ledger", previous_ledger])
    return FinalCommand(
        track="comparison",
        name="cross_version_regression_ledger",
        command=cmd,
        purpose=(
            "Build the final cross-version regression ledger used by readiness to verify "
            "continual-learning and multi-version stability claims."
        ),
        count_as_real_bugs=False,
        expected_output=(
            "version-ledger JSON plus experiment-final-version-ledger manifest referenced by final readiness"
        ),
        notes=(
            "Run after collecting comparable run logs from at least two target versions. "
            "This does not execute fuzzing; it converts existing cross-version outcomes into audited evidence."
        ),
        replay_bug_policy={"enable_replay_bug": False, "source_issues": replay_source_issues()},
        experiment_meta={
            "matrix_id": "baseline_scope_comparison",
            "comparison_group": "cross_version_continual_learning",
            "variant": {
                "variant_id": "version_ledger",
                "comparison_role": "support",
                "component_focus": "cross_version_continual_learning",
            },
            "analysis_tags": ["cross_version", "continual_learning", "regression_ledger"],
            "counts_as_real_bugs": False,
        },
    )


def final_readiness_audit_command(args: argparse.Namespace) -> FinalCommand:
    cmd = [
        str(DATADIFF),
        "final-readiness",
        "--manifest-index",
        str(getattr(args, "manifest_index", "") or REPORTS_DIR / "final-experiment-manifest-index.json"),
        "--full-run-log-scan",
        "--fail-on-missing",
    ]
    ledger_manifest = str(
        getattr(args, "ledger_evidence_manifest", "") or REPORTS_DIR / "experiment-final-version-ledger.json"
    ).strip()
    if ledger_manifest:
        cmd.extend(["--extra-manifest", ledger_manifest])
    return FinalCommand(
        track="postprocess",
        name="final_readiness_audit",
        command=cmd,
        purpose=(
            "Run the final ICSE evidence audit after all experiment tracks complete, including "
            "24h live depth, validation, seeded sensitivity, ablations, comparisons, runtime "
            "efficiency, discovery responsiveness, adaptive evidence, and cross-version ledgers."
        ),
        count_as_real_bugs=False,
        expected_output="reports/final-readiness-*.json plus reports/final-readiness-*.md with all gates passing",
        notes=(
            "This is the authoritative final-paper gate. Do not claim final readiness unless this "
            "command exits successfully after the frozen live and support tracks have completed."
        ),
        replay_bug_policy={"enable_replay_bug": False, "source_issues": replay_source_issues()},
        experiment_meta={
            "matrix_id": "final_readiness_audit",
            "comparison_group": "postprocess_readiness",
            "variant": {
                "variant_id": "final_readiness_audit",
                "comparison_role": "support",
                "component_focus": "final_readiness",
            },
            "analysis_tags": ["final_readiness", "postprocess", "icse_audit"],
            "counts_as_real_bugs": False,
        },
    )


def write_plan(commands: Iterable[FinalCommand], args: argparse.Namespace) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"final-experiment-plan-{utc_now().replace(':', '').replace('-', '').replace('Z', '')}-{time.time_ns()}.json"
    payload = {
        "created_at": utc_now(),
        "project_root": str(PROJECT_ROOT),
        "datadiff": str(DATADIFF),
        "args": vars(args),
        "commands": [command.to_dict() for command in commands],
        "policy": {
            "freeze_rule": "Do not change generator, oracle, normalizer, or triage code after starting final runs.",
            "live_counts": "Live latest-version runs count candidate/confirmed real backend bugs after family deduplication.",
            "historical_counts": "Historical runs count only confirmed_fixed specs; pending/candidate specs are case studies.",
            "seeded_counts": "Seeded runs measure sensitivity only and do not count as real bugs.",
            "validation_counts": "Short validation runs gate the harness before 24h runs and do not count as real bugs.",
            "ablation_counts": "Ablation/comparison runs support RQ tables and do not directly count as real bugs.",
            "paper_run_journal": (
                "Every final-plan command keeps paper-run-journal recording enabled so each counted or "
                "paper-facing support run is appended to reports/paper-run-journal.jsonl and .md."
            ),
            "manifest_index": (
                "When --execute is used, this script records each completed command's emitted experiment "
                "manifest in --manifest-index. The postprocess readiness audit consumes that index so stale "
                "pre-final runs do not contaminate the final ICSE evidence gate."
            ),
            "family_key": "root_cause + suspicious_backends",
            "replay_bug_gate": (
                "Final live commands explicitly keep enable_replay_bug=false; historical commands "
                "explicitly enable replay while using the same middle/bottom harness."
            ),
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def replay_source_issues(*extra_sources: str) -> list[str]:
    sources = [str(source).strip().rstrip("/") for source in DEFAULT_REPLAY_BUG_SOURCE_ISSUES]
    sources.extend(str(source).strip().rstrip("/") for source in extra_sources)
    return sorted({source for source in sources if source})


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


if __name__ == "__main__":
    raise SystemExit(main())
