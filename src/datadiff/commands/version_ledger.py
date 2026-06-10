from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from datadiff.guidance import parse_guidance_targets
from datadiff.manifest_index import manifest_index_files
from datadiff.reporter import latest_run_log_path
from datadiff.util import dump_json, load_json, utc_now
from datadiff.version_ledger import (
    build_version_ledger,
    observations_from_run_logs,
)


def cmd_version_ledger(args: argparse.Namespace) -> int:
    run_files = [Path(item) for item in parse_guidance_targets(getattr(args, "run_files", ""))]
    if not run_files and getattr(args, "run_file", None):
        run_files = [Path(getattr(args, "run_file"))]
    versions = parse_guidance_targets(getattr(args, "versions", ""))
    manifest_indexes = list(getattr(args, "manifest_index", []) or [])
    if not run_files:
        run_files, auto_versions = _version_ledger_runs_from_manifest_indexes(
            manifest_indexes
        )
        if auto_versions and not versions:
            versions = auto_versions
        if manifest_indexes and not run_files:
            print("no run logs found in --manifest-index for version-ledger", file=sys.stderr)
            return 2
    if not run_files:
        run_files = [latest_run_log_path()]
    observations = observations_from_run_logs(run_files, versions=versions)
    unique_versions = [
        observation.version_id
        for observation in observations
        if str(observation.version_id).strip()
    ]
    unique_versions = list(dict.fromkeys(unique_versions))
    if (manifest_indexes or evidence_manifest_requested(args)) and len(unique_versions) < 2:
        print(
            "version-ledger evidence requires run logs from at least two versions",
            file=sys.stderr,
        )
        return 2
    previous_ledger = {}
    previous_ledger_text = str(getattr(args, "previous_ledger", "") or "").strip()
    if previous_ledger_text:
        previous_ledger_path = Path(previous_ledger_text)
        if previous_ledger_path.is_file():
            loaded = load_json(previous_ledger_path)
            previous_ledger = loaded if isinstance(loaded, dict) else {}
    payload = build_version_ledger(
        observations,
        baseline_version=str(getattr(args, "baseline_version", "") or ""),
        previous_ledger=previous_ledger,
    )
    output = str(getattr(args, "output", "") or "").strip()
    if output:
        dump_json(payload, Path(output))
    evidence_manifest_output = str(getattr(args, "evidence_manifest_output", "") or "").strip()
    if evidence_manifest_output:
        if not output:
            raise SystemExit("--evidence-manifest-output requires --output")
        _write_version_ledger_evidence_manifest(
            Path(evidence_manifest_output),
            ledger_file=Path(output),
            run_files=run_files,
            versions=versions,
        )
    if getattr(args, "json", False):
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    summary = payload["summary"]
    print(f"schema={payload['schema_version']}")
    print(f"versions={summary['version_count']}")
    print(f"families={summary['family_count']}")
    print(f"new={summary['new_family_count']}")
    print(f"fixed={summary['fixed_family_count']}")
    print(f"regression={summary['regression_family_count']}")
    print(f"persistent={summary['persistent_family_count']}")
    print(f"health_observations={summary.get('health_observation_count', 0)}")
    print(f"invalid_cases={summary.get('invalid_case_count', 0)}")
    print(f"fallback_cases={summary.get('fallback_case_count', 0)}")
    print(f"false_positives={summary.get('false_positive_count', 0)}")
    print(f"min_throughput_cases_s={summary.get('min_throughput_cases_s', 0.0):.6g}")
    print(f"max_invalid_rate={summary.get('max_invalid_rate', 0.0):.6g}")
    print(f"max_false_positive_rate={summary.get('max_false_positive_rate', 0.0):.6g}")
    if output:
        print(f"ledger={output}")
    if evidence_manifest_output:
        print(f"evidence_manifest={evidence_manifest_output}")
    return 0


def evidence_manifest_requested(args: argparse.Namespace) -> bool:
    return bool(str(getattr(args, "evidence_manifest_output", "") or "").strip())


def _version_ledger_runs_from_manifest_indexes(index_files: list[str] | tuple[str, ...]) -> tuple[list[Path], list[str]]:
    manifest_files, _, _ = manifest_index_files(index_files)
    candidates: list[tuple[Path, str]] = []
    for manifest_file in manifest_files:
        if not manifest_file.is_file():
            continue
        manifest = load_json(manifest_file)
        if not isinstance(manifest, dict):
            continue
        manifest_target_version = str(manifest.get("target_version", "") or "").strip()
        for run in manifest.get("runs", []) or []:
            if not isinstance(run, dict):
                continue
            if str(run.get("evidence_kind", "") or manifest.get("evidence_kind", "") or "").strip():
                continue
            run_file_text = str(run.get("run_file", "") or "").strip()
            if not run_file_text:
                continue
            run_file = Path(run_file_text)
            version = (
                str(run.get("target_version", "") or "").strip()
                or manifest_target_version
                or _manifest_run_version_label(manifest, run)
            )
            if not version:
                continue
            candidates.append((run_file, version))
    seen: set[tuple[str, str]] = set()
    run_files: list[Path] = []
    versions: list[str] = []
    for run_file, version in candidates:
        key = (str(run_file), version)
        if key in seen:
            continue
        seen.add(key)
        run_files.append(run_file)
        versions.append(version)
    return run_files, versions


def _manifest_run_version_label(manifest: dict[str, Any], run: dict[str, Any]) -> str:
    for source in (run, manifest):
        experiment_meta = source.get("experiment_meta", {}) if isinstance(source, dict) else {}
        if not isinstance(experiment_meta, dict):
            continue
        historical = experiment_meta.get("historical", {})
        if isinstance(historical, dict):
            version = str(historical.get("target_version", "") or "").strip()
            if version:
                return version
        variant = experiment_meta.get("variant", {})
        if isinstance(variant, dict):
            version = str(variant.get("target_version", "") or "").strip()
            if version:
                return version
    return ""


def _write_version_ledger_evidence_manifest(
    manifest_path: Path,
    *,
    ledger_file: Path,
    run_files: list[Path],
    versions: list[str],
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_payload = _load_ledger_payload(ledger_file)
    champion_transfer = (
        ledger_payload.get("champion_transfer", {})
        if isinstance(ledger_payload.get("champion_transfer", {}), dict)
        else {}
    )
    run_payload = {
        "target_suite": "cross_version",
        "preset": "version_ledger",
        "seed": "",
        "run_file": str(run_files[-1]) if run_files else "",
        "backends": [],
        "report": "",
        "evidence_mode": "comparison",
        "evidence_kind": "postprocess_ledger",
        "version_ledger_file": str(ledger_file),
        "champion_transfer": champion_transfer,
    }
    dump_json(
        {
            "schema_version": "version-ledger-evidence-manifest-v1",
            "created_at": utc_now(),
            "evidence_mode": "comparison",
            "evidence_kind": "postprocess_ledger",
            "target_suite": "cross_version",
            "target_suites": ["cross_version"],
            "backends": [],
            "targets": [],
            "runs": [run_payload],
            "version_ledger_file": str(ledger_file),
            "champion_transfer": champion_transfer,
            "version_ledger_inputs": {
                "run_files": [str(path) for path in run_files],
                "versions": versions,
            },
            "experiment_meta": {
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
        },
        manifest_path,
    )


def _load_ledger_payload(ledger_file: Path) -> dict[str, Any]:
    try:
        loaded = load_json(ledger_file)
    except Exception:  # noqa: BLE001
        return {}
    return loaded if isinstance(loaded, dict) else {}
