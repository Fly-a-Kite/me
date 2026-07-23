from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from datadiff.classification_oracle import classify_finding
from datadiff.config import DEFAULT_KNOWN_SATURATED_BUG_FAMILIES
from datadiff.dsl import Case
from datadiff.normalizer import NormalizedResult, normalized_results_from_mapping
from datadiff.operation_semantics import operation_names
from datadiff.oracle import evaluate_case
from datadiff.finding_outcomes import backend_group_key, offline_finding_bucket
from datadiff.osc_diagnostic_facade import consume_opaque_diagnostic_ref_set
from datadiff.util import load_json, read_jsonl, read_jsonl_partial, run_meta_path


def _project_relative_summary_path(value: str | Path | None) -> str:
    if not value:
        return ""
    path = Path(value)
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _diagnostic_manifest_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    try:
        manifest = row.get("experiment_manifest")
        return dict(manifest) if isinstance(manifest, Mapping) else {}
    except Exception:
        return {}


def _diagnostic_backend_status_from_row(
    row: Mapping[str, Any],
) -> dict[str, str]:
    for key in ("normalized", "raw_results", "backend_status"):
        try:
            payload = row.get(key)
            items = tuple(payload.items()) if isinstance(payload, Mapping) else ()
        except Exception:
            continue
        status_by_backend: dict[str, str] = {}
        for backend, value in items:
            if not isinstance(backend, str) or not backend:
                continue
            try:
                status = value.get("status") if isinstance(value, Mapping) else value
            except Exception:
                status = None
            status_by_backend[backend] = (
                status if isinstance(status, str) and status else "unknown"
            )
        if status_by_backend:
            return status_by_backend
    return {}


def _diagnostic_backends_from_row(row: Mapping[str, Any]) -> list[str]:
    return list(_diagnostic_backend_status_from_row(row))


def _diagnostic_case_digest_from_row(row: Mapping[str, Any]) -> str:
    manifest = _diagnostic_manifest_from_row(row)
    case_digest = manifest.get("case_digest")
    return case_digest if isinstance(case_digest, str) else ""


def _opaque_diagnostic_refs_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return consume_opaque_diagnostic_ref_set(
        row.get("osc_diagnostic_refs"),
        expected_backends=_diagnostic_backends_from_row(row),
        case_digest=_diagnostic_case_digest_from_row(row),
    )


def _opaque_metamorphic_diagnostic_refs_from_row(
    row: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    source = row.get("osc_metamorphic_diagnostic_refs")
    if not isinstance(source, Mapping):
        source = row.get("metamorphic")
    if not isinstance(source, Mapping):
        return {}
    projections: dict[str, dict[str, Any]] = {}
    try:
        items = tuple(source.items())
    except Exception:
        return {}
    for name, variant in items:
        if not isinstance(name, str) or not name or not isinstance(variant, Mapping):
            continue
        projections[name] = {
            "experiment_manifest": _diagnostic_manifest_from_row(variant),
            "backend_status": _diagnostic_backend_status_from_row(variant),
            "osc_diagnostic_refs": _opaque_diagnostic_refs_from_row(variant),
        }
    return projections


def _summarize_run_health(run_file: Path, *, limit: int = 3) -> dict[str, Any]:
    rows, partial = read_jsonl_partial(run_file)
    statuses: Counter[str] = Counter()
    candidate_bug_families: Counter[str] = Counter()
    false_positive_reasons: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    known_saturated = set(DEFAULT_KNOWN_SATURATED_BUG_FAMILIES)
    candidate_origins: dict[str, Counter[str]] = {
        "organic": Counter(),
        "issue_inspired": Counter(),
    }

    for row in rows:
        statuses[row.get("status", "unknown")] += 1
        known_saturated.update(row.get("config", {}).get("known_saturated_bug_families", []) or [])
        candidate_findings: list[dict] = []
        for finding in row.get("findings", []) or []:
            if finding.get("false_positive_reason"):
                false_positive_reasons[finding["false_positive_reason"]] += 1
            if _is_candidate_issue_finding(finding):
                candidate_findings.append(finding)
                if len(examples) < limit:
                    examples.append(
                        {
                            "case_id": row.get("case", {}).get("case_id", ""),
                            "seed": row.get("case", {}).get("seed", ""),
                            "status": row.get("status", ""),
                            "kind": finding.get("kind", ""),
                            "root": finding.get("root_cause", "unknown"),
                            "suspicious": finding.get("suspicious_backends", []),
                        }
                    )
        candidate_keys = _candidate_issue_family_keys(candidate_findings)
        candidate_bug_families.update(candidate_keys)
        if candidate_keys:
            origin = _candidate_row_origin(row, candidate_findings)
            candidate_origins.setdefault(origin, Counter()).update(candidate_keys)

    candidate_items = dict(candidate_bug_families.most_common())
    known_items = {
        family: count
        for family, count in candidate_items.items()
        if _is_known_saturated_family_key(family, known_saturated)
    }
    unsaturated_items = {
        family: count
        for family, count in candidate_items.items()
        if not _is_known_saturated_family_key(family, known_saturated)
    }
    fresh_items = {
        family: count
        for family, count in candidate_origins["organic"].most_common()
        if family in unsaturated_items
    }
    runtime = _run_health_runtime_summary(run_file)
    return {
        "run_file": _project_relative_summary_path(run_file),
        "partial": partial,
        "bytes": run_file.stat().st_size if run_file.exists() else 0,
        "rows": len(rows),
        "statuses": dict(statuses.most_common()),
        "candidate_bug_families": candidate_items,
        "fresh_candidate_bug_families": fresh_items,
        "known_saturated_candidate_bug_families": known_items,
        "false_positive_reasons": dict(false_positive_reasons.most_common()),
        "examples": examples,
        "runtime": runtime,
    }


def _run_health_runtime_summary(run_file: Path) -> dict[str, Any]:
    meta_path = run_meta_path(run_file)
    meta = load_json(meta_path) if meta_path.exists() else {}
    checkpoint_text = str(meta.get("checkpoint_file", "") or "").strip() if isinstance(meta, dict) else ""
    checkpoint_path = Path(checkpoint_text).expanduser() if checkpoint_text else run_file.with_name(
        f"{run_file.stem.split('.jsonl', 1)[0]}.checkpoint.json"
    )
    checkpoint = load_json(checkpoint_path) if checkpoint_path is not None and checkpoint_path.exists() else {}
    snapshot = checkpoint if isinstance(checkpoint, dict) and checkpoint else meta if isinstance(meta, dict) else {}
    if not isinstance(snapshot, dict) or not snapshot:
        return {}
    elapsed_s = float(snapshot.get("elapsed_s", 0.0) or 0.0)
    executed = int(snapshot.get("executed_cases", 0) or 0)
    run_bytes = run_file.stat().st_size if run_file.exists() else 0
    throughput = float(snapshot.get("throughput_cases_s", 0.0) or 0.0)
    stage_profile = snapshot.get("stage_profile", {}) if isinstance(snapshot.get("stage_profile", {}), dict) else {}
    avg_ms = stage_profile.get("avg_ms_per_case", {}) if isinstance(stage_profile.get("avg_ms_per_case", {}), dict) else {}
    share = stage_profile.get("share_of_total", {}) if isinstance(stage_profile.get("share_of_total", {}), dict) else {}
    stage_profile_summary = {
        "avg_ms_per_case": {key: float(value or 0.0) for key, value in avg_ms.items()},
        "share_of_total": {key: float(value or 0.0) for key, value in share.items()},
        "case_count": int(stage_profile.get("case_count", executed) or executed),
    }
    closed_loop_state_file = str(snapshot.get("closed_loop_state_file", "") or "").strip()
    state_path = Path(closed_loop_state_file).expanduser() if closed_loop_state_file else None
    state_bytes = state_path.stat().st_size if state_path is not None and state_path.exists() else 0
    return {
        "status": str(snapshot.get("status", "")).strip(),
        "elapsed_s": elapsed_s,
        "executed_cases": executed,
        "throughput_cases_s": throughput,
        "next_seed": int(snapshot.get("next_seed", 0) or 0),
        "evidence_bytes_per_case": (run_bytes / executed) if executed else 0.0,
        "run_log_bytes": run_bytes,
        "checkpoint_file": (
            _project_relative_summary_path(checkpoint_path)
            if checkpoint_path is not None and checkpoint_path.exists()
            else ""
        ),
        "meta_file": _project_relative_summary_path(meta_path) if meta_path.exists() else "",
        "stage_profile": stage_profile_summary,
        "closed_loop_state_file": (
            _project_relative_summary_path(state_path) if state_path is not None and state_path.exists() else ""
        ),
        "closed_loop_state_bytes": state_bytes,
        "closed_loop_state_summary": snapshot.get("closed_loop_state_summary", {}),
    }


def _summarize_run_classification(
    run_file: Path,
    *,
    limit: int = 3,
    refresh: bool = False,
) -> dict[str, Any]:
    verdicts: Counter[str] = Counter()
    offline_buckets: Counter[str] = Counter()
    candidate_bug_families: Counter[str] = Counter()
    false_positive_reasons: Counter[str] = Counter()
    examples: dict[str, list[dict]] = {}
    cache: dict[tuple, dict] = {}
    known_saturated = set(DEFAULT_KNOWN_SATURATED_BUG_FAMILIES)
    candidate_origins: dict[str, Counter[str]] = {
        "organic": Counter(),
        "issue_inspired": Counter(),
    }

    rows = read_jsonl(run_file)
    for row in rows:
        known_saturated.update(row.get("config", {}).get("known_saturated_bug_families", []) or [])
    for row in rows:
        _classify_run_row(
            row,
            verdicts,
            offline_buckets,
            candidate_bug_families,
            false_positive_reasons,
            examples,
            limit,
            cache,
            refresh=refresh,
            candidate_origins=candidate_origins,
            known_saturated_bug_families=known_saturated,
        )
    candidate_items = dict(candidate_bug_families.most_common())
    known_items = {
        family: count
        for family, count in candidate_items.items()
        if _is_known_saturated_family_key(family, known_saturated)
    }
    unsaturated_items = {
        family: count
        for family, count in candidate_items.items()
        if not _is_known_saturated_family_key(family, known_saturated)
    }
    inspired_items = {
        family: count
        for family, count in candidate_origins["issue_inspired"].most_common()
        if family in unsaturated_items
    }
    organic_items = {
        family: count
        for family, count in candidate_origins["organic"].most_common()
        if family in unsaturated_items
    }
    return {
        "run_file": _project_relative_summary_path(run_file),
        "refresh": refresh,
        "offline_buckets": dict(offline_buckets.most_common()),
        "triage_verdicts": dict(verdicts.most_common()),
        "candidate_bug_families": candidate_items,
        "unsaturated_candidate_bug_families": unsaturated_items,
        "fresh_candidate_bug_families": organic_items,
        "issue_inspired_unsaturated_candidate_bug_families": inspired_items,
        "known_saturated_candidate_bug_families": known_items,
        "known_saturated_reference_count": len(known_saturated),
        "false_positive_reasons": dict(false_positive_reasons.most_common()),
        "examples": examples,
    }


def _classified_candidate_rows_for_run(
    run_file: Path,
    *,
    refresh: bool = False,
) -> list[dict[str, Any]]:
    rows = read_jsonl(run_file)
    known_saturated = set(DEFAULT_KNOWN_SATURATED_BUG_FAMILIES)
    for row in rows:
        known_saturated.update(row.get("config", {}).get("known_saturated_bug_families", []) or [])

    classified_rows: list[dict[str, Any]] = []
    cache: dict[tuple, dict] = {}
    for row in rows:
        _classify_run_row(
            row,
            Counter(),
            Counter(),
            Counter(),
            Counter(),
            {},
            0,
            cache,
            refresh=refresh,
            known_saturated_bug_families=known_saturated,
            classified_candidate_rows=classified_rows,
        )
    return classified_rows


def _is_known_saturated_family_key(family: str, known_saturated: set[str]) -> bool:
    if family in known_saturated:
        return True
    root, sep, backend_key = family.rpartition("@")
    if not sep:
        return False
    backends = [backend for backend in backend_key.split(",") if backend]
    return bool(backends) and all(f"{root}@{backend}" in known_saturated for backend in backends)


def _classify_run_row(
    row: dict,
    verdicts: Counter[str],
    offline_buckets: Counter[str],
    candidate_bug_families: Counter[str],
    false_positive_reasons: Counter[str],
    examples: dict[str, list[dict]],
    limit: int,
    cache: dict[tuple, dict],
    *,
    refresh: bool = False,
    candidate_origins: dict[str, Counter[str]] | None = None,
    known_saturated_bug_families: set[str] | None = None,
    classified_candidate_rows: list[dict[str, Any]] | None = None,
) -> None:
    if not row.get("findings"):
        return
    case = None
    normalized = None
    raw_results = row.get("raw_results", {})
    config = row.get("config", {})
    backends = list(row.get("normalized", {}))
    candidate_findings: list[dict] = []
    findings = row.get("findings", [])
    if refresh:
        refreshed = _refresh_differential_findings(row)
        if refreshed is not None:
            case, normalized, findings, config, backends = refreshed
    if refresh and normalized:
        refreshed = [finding.to_dict() for finding in evaluate_case(case, normalized)]
        if refreshed:
            findings = refreshed
    for finding in findings:
        if finding.get("triage_verdict") and finding.get("triage_verdict") != "unclassified":
            classification = {
                "verdict": finding.get("triage_verdict", "unclassified"),
                "false_positive_reason": finding.get("false_positive_reason", ""),
                "evidence": finding.get("triage_evidence", ""),
            }
        else:
            cache_key = _classification_cache_key(row, finding)
            classification = cache.get(cache_key)
            if classification is None:
                if case is None:
                    case = Case.from_dict(row["case"])
                if normalized is None:
                    normalized = _normalized_from_row(row)
                c = classify_finding(case, finding, normalized, raw_results, config, backends)
                classification = c.to_dict()
                cache[cache_key] = classification
        verdict = classification["verdict"]
        verdicts[verdict] += 1
        classified_finding = dict(finding)
        classified_finding["triage_verdict"] = verdict
        classified_finding["false_positive"] = bool(
            classification.get("false_positive", finding.get("false_positive", False))
        )
        classified_finding["false_positive_reason"] = classification.get(
            "false_positive_reason",
            finding.get("false_positive_reason", ""),
        )
        classified_finding["triage_evidence"] = classification.get(
            "evidence",
            finding.get("triage_evidence", ""),
        )
        if classification.get("implicated_backends"):
            classified_finding["suspicious_backends"] = list(classification["implicated_backends"])
        if not classified_finding.get("source_issue"):
            metadata = row.get("case", {}).get("metadata", {}) or {}
            classified_finding["source_issue"] = (
                metadata.get("source_issue")
                or metadata.get("source_issue_alt")
                or finding.get("source_issue", "")
            )
        if not classified_finding.get("discovery_origin"):
            metadata = row.get("case", {}).get("metadata", {}) or {}
            if metadata.get("source_issue") or metadata.get("source_issue_alt"):
                classified_finding["discovery_origin"] = "issue_inspired"
        offline_buckets[
            offline_finding_bucket(
                classified_finding,
                tuple(sorted(known_saturated_bug_families or set())),
            )
        ] += 1
        if verdict == "candidate_implementation_bug" and not classification.get("false_positive"):
            candidate_finding = dict(classified_finding)
            candidate_finding["false_positive"] = False
            candidate_findings.append(candidate_finding)
        if classification.get("false_positive_reason"):
            false_positive_reasons[classification["false_positive_reason"]] += 1
        if len(examples.setdefault(verdict, [])) < limit:
            examples[verdict].append(
                {
                    "case_id": row["case"]["case_id"],
                    "seed": row["case"]["seed"],
                    "kind": classified_finding.get("kind", ""),
                    "root": classified_finding.get("root_cause", "unknown"),
                    "suspicious": classified_finding.get("suspicious_backends", []),
                    "signature": classified_finding.get("signature", ""),
                    "evidence": classification.get("evidence", ""),
                }
            )
    candidate_keys = _candidate_issue_family_keys(candidate_findings)
    candidate_bug_families.update(candidate_keys)
    if candidate_origins is not None and candidate_keys:
        origin = _candidate_row_origin(row, candidate_findings)
        candidate_origins.setdefault(origin, Counter()).update(candidate_keys)
    if classified_candidate_rows is not None and candidate_findings:
        classified_candidate_rows.append(
            {
                "case": row.get("case", {}),
                "findings": candidate_findings,
                "normalized": row.get("normalized", {}),
                "raw_results": row.get("raw_results", {}),
                "backend_status": row.get("backend_status", {}),
                "config": row.get("config", {}),
                "candidate_recheck": row.get("candidate_recheck", {}),
                "bug_dir": row.get("bug_dir", ""),
                "status": row.get("status", ""),
                "case_index": row.get("case_index", ""),
                "elapsed_s": row.get("elapsed_s", ""),
                "candidate_bug_families": dict(candidate_keys),
                "experiment_manifest": row.get("experiment_manifest", {}),
                "osc_diagnostic_refs": _opaque_diagnostic_refs_from_row(row),
                "osc_metamorphic_diagnostic_refs": (
                    _opaque_metamorphic_diagnostic_refs_from_row(row)
                ),
            }
        )


def _candidate_row_origin(row: dict, candidate_findings: list[dict]) -> str:
    if any(
        finding.get("discovery_origin") == "issue_inspired" or finding.get("source_issue")
        for finding in candidate_findings
    ):
        return "issue_inspired"
    metadata = row.get("case", {}).get("metadata", {})
    if metadata.get("source_issue") or metadata.get("source_issue_alt"):
        return "issue_inspired"
    return "organic"


def _candidate_issue_family_key(finding: dict) -> str:
    return next(iter(_candidate_issue_family_keys([finding])), "")


def _candidate_issue_family_keys(findings: list[dict]) -> Counter:
    keys: Counter = Counter()
    root_by_backend_group: dict[str, str] = {}
    for finding in findings:
        if not _is_candidate_issue_finding(finding):
            continue
        root = str(finding.get("root_cause", "unknown"))
        if root.startswith("metamorphic_"):
            continue
        backend_group = backend_group_key(finding)
        root_by_backend_group.setdefault(backend_group, root)
    for finding in findings:
        if not _is_candidate_issue_finding(finding):
            continue
        root = str(finding.get("root_cause", "unknown"))
        backend_group = backend_group_key(finding)
        if root.startswith("metamorphic_") and backend_group in root_by_backend_group:
            root = root_by_backend_group[backend_group]
        keys[f"{root}@{backend_group}"] += 1
    return keys


def _is_candidate_issue_finding(finding: dict) -> bool:
    return finding.get("triage_verdict") == "candidate_implementation_bug" and not finding.get("false_positive")


_candidate_bug_family_key = _candidate_issue_family_key
_candidate_bug_family_keys = _candidate_issue_family_keys
_is_candidate_bug_finding = _is_candidate_issue_finding


def _classification_cache_key(row: dict, finding: dict) -> tuple:
    case = row.get("case", {})
    program = case.get("program", {})
    tables = case.get("tables", [])
    return (
        finding.get("signature", ""),
        finding.get("kind", ""),
        finding.get("root_cause", ""),
        finding.get("confidence", ""),
        tuple(finding.get("suspicious_backends", [])),
        row.get("config", {}).get("generator_profile", ""),
        tuple(operation_names(program.get("operations", []))),
        _case_has_special_float_data(tables),
        _case_has_null_data(tables),
        _case_has_non_ascii_data(tables),
    )


def _case_has_special_float_data(tables: list[dict]) -> bool:
    # JSONL stores NaN/Infinity as non-standard JSON tokens; Python json restores
    # them as floats, so this detects old and new run files.
    for table in tables:
        for row in table.get("rows", []):
            for value in row.values():
                if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                    return True
    return False


def _case_has_null_data(tables: list[dict]) -> bool:
    return any(value is None for table in tables for row in table.get("rows", []) for value in row.values())


def _case_has_non_ascii_data(tables: list[dict]) -> bool:
    return any(
        isinstance(value, str) and any(ord(ch) > 127 for ch in value)
        for table in tables
        for row in table.get("rows", [])
        for value in row.values()
    )


def _normalized_from_row(row: dict) -> dict[str, NormalizedResult]:
    return _normalized_from_mapping(row.get("normalized", {}))


def _refresh_differential_findings(
    row: dict,
) -> tuple[Case, dict[str, NormalizedResult], list[dict], dict, list[str]] | None:
    config = row.get("config", {})
    if row.get("normalized"):
        try:
            case = Case.from_dict(row["case"])
        except Exception:
            case = None
        if case is not None:
            normalized = _normalized_from_mapping(row.get("normalized", {}))
            return case, normalized, [finding.to_dict() for finding in evaluate_case(case, normalized)], config, list(normalized)

    bug_dir_text = row.get("bug_dir", "")
    if not bug_dir_text:
        return None
    bug_dir = Path(bug_dir_text)
    if not bug_dir.exists():
        bug_dir = Path.cwd() / bug_dir_text
    case_path = bug_dir / "case.json"
    normalized_path = bug_dir / "normalized.json"
    if not case_path.exists() or not normalized_path.exists():
        return None
    case = Case.from_dict(load_json(case_path))
    normalized = _normalized_from_mapping(load_json(normalized_path))
    config_path = bug_dir / "config.json"
    if config_path.exists():
        config = load_json(config_path)
    return case, normalized, [finding.to_dict() for finding in evaluate_case(case, normalized)], config, list(normalized)


def _normalized_from_mapping(mapping: dict) -> dict[str, NormalizedResult]:
    return normalized_results_from_mapping(mapping)
