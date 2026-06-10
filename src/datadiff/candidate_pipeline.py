from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from datadiff.artifact import save_issue_artifact
from datadiff.bug_discovery_system import (
    bug_discovery_system_descriptor,
    rank_candidate_pipeline_rows,
)
from datadiff.config import ExperimentConfig
from datadiff.dynamic_strategy import append_learning_event
from datadiff.dynamic_strategy import write_strategy_snapshot
from datadiff.dsl import Case
from datadiff.issue_readiness import build_issue_readiness
from datadiff.oracle import Finding
from datadiff.pathing import project_display_path as _project_display_path_impl
from datadiff.pathing import resolve_project_path as _resolve_project_path_impl
from datadiff.finding_outcomes import candidate_issue_family_keys
from datadiff.reducer import reduce_case
from datadiff.reproducer_scripts import write_reduced_reproducer
from datadiff.runner import run_loaded_case
from datadiff.semantic_contracts import finding_contract_axes, semantic_contract_lattice_payload
from datadiff.triage import (
    build_triage_report,
    standalone_reproducer_rule_records,
    supports_standalone_reproducer,
    write_standalone_reproducer,
    write_triage_artifact,
)
from datadiff.classification_oracle import documented_semantic_rule_records
from datadiff.classification_oracle import semantic_boundary_rule_records
from datadiff.mutator_ir import ir_rewrite_rule_metadata
from datadiff.util import PROJECT_ROOT, dump_json, load_json, slugify, utc_now

CANDIDATE_PIPELINE_SCHEMA_VERSION = "candidate-pipeline-v1"
DEFAULT_CANDIDATE_PIPELINE_DIR = PROJECT_ROOT / "new_issue" / "generated" / "candidate-pipelines"
save_bug_artifact = save_issue_artifact


def build_candidate_pipeline(
    *,
    evidence_files: list[Path] | None = None,
    manifest_file: Path | None = None,
    output_dir: Path | None = None,
    recheck_attempts: int = 2,
    reduce_artifacts: bool = True,
    standalone_reproducer: bool = True,
    latest_confirmation_files: list[Path] | None = None,
    new_issue_dir: Path | None = None,
    old_issue_dir: Path | None = None,
    generated_issue_dir: Path | None = None,
) -> dict[str, Any]:
    resolved_manifest = _resolve_project_path(manifest_file) if manifest_file is not None else None
    resolved_evidence_files = [
        _resolve_project_path(path)
        for path in (evidence_files or _candidate_pipeline_manifest_evidence_files(resolved_manifest))
    ]
    resolved_evidence_files = [path for path in resolved_evidence_files if path.is_file()]
    if not resolved_evidence_files:
        raise FileNotFoundError("no candidate evidence files were found for the candidate pipeline")

    resolved_new_issue_dir = _resolve_project_path(new_issue_dir or (PROJECT_ROOT / "new_issue"))
    resolved_old_issue_dir = _resolve_project_path(old_issue_dir or (PROJECT_ROOT / "old_issue"))
    resolved_generated_issue_dir = _resolve_project_path(
        generated_issue_dir or (resolved_new_issue_dir / "generated")
    )
    resolved_output_root = _resolve_project_path(output_dir or DEFAULT_CANDIDATE_PIPELINE_DIR)
    pipeline_dir = resolved_output_root / _pipeline_dir_name(resolved_manifest, resolved_evidence_files)
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    frozen_path = pipeline_dir / "frozen-candidates.json"
    strategy_snapshot_path = write_strategy_snapshot(
        classification_documented_rules=list(documented_semantic_rule_records()),
        classification_boundary_rules=list(semantic_boundary_rule_records()),
        reproducer_rules=list(standalone_reproducer_rule_records()),
        metadata={
            "generated_by": "datadiff candidate-pipeline",
            "pipeline_dir": _project_display_path(pipeline_dir),
            "source_manifest_file": _project_display_path(resolved_manifest) if resolved_manifest is not None else "",
        },
        output_dir=pipeline_dir,
        snapshot_id="strategy-snapshot",
    )
    issue_drafts_dir = pipeline_dir / "issue-drafts"
    issue_drafts_dir.mkdir(parents=True, exist_ok=True)

    evidence_payloads = [_load_evidence_payload(path) for path in resolved_evidence_files]
    frozen_rows: list[dict[str, Any]] = []
    existing_queue = build_issue_readiness(
        latest_confirmation_files=latest_confirmation_files,
        new_issue_dir=resolved_new_issue_dir,
        old_issue_dir=resolved_old_issue_dir,
        generated_issue_dir=resolved_generated_issue_dir,
        include_generated=True,
    )
    existing_by_family = _issues_by_family(existing_queue.get("issues", []))
    confirmed_latest_families = set(existing_queue.get("bug_status_summary", {}).get("confirmed_latest_families", []))

    candidates: list[dict[str, Any]] = []
    family_seen_in_pipeline: dict[str, str] = {}
    for evidence_index, evidence in enumerate(evidence_payloads):
        source_run_file = str(evidence.get("source_run_file", ""))
        for row_index, row in enumerate(evidence.get("candidate_rows", [])):
            if not isinstance(row, dict):
                continue
            row_config = row.get("config", {}) if isinstance(row.get("config", {}), dict) else {}
            known_families = tuple(row_config.get("known_saturated_bug_families", []) or ())
            row_families = sorted(
                candidate_issue_family_keys(
                    row.get("findings", []),
                    known_saturated_bug_families=known_families,
                ).keys()
            )
            if not row_families:
                continue
            primary_family = row_families[0]
            candidate_id = _candidate_id(primary_family, row, evidence_index, row_index)
            frozen_row = {
                "candidate_id": candidate_id,
                "source_evidence_file": _project_display_path(resolved_evidence_files[evidence_index]),
                "source_run_file": source_run_file,
                "case": row.get("case", {}),
                "findings": row.get("findings", []),
                "normalized": row.get("normalized", {}),
                "raw_results": row.get("raw_results", {}),
                "semantic_contract_evidence": _semantic_contract_evidence(row),
                "ir_rewrite_evidence": _ir_rewrite_evidence(row),
                "config": {
                    **row_config,
                    "strategy_snapshot_path": str(strategy_snapshot_path),
                    "freeze_strategy_snapshot": True,
                    "strategy_learning_path": str(pipeline_dir / "strategy-learning" / "candidate-pipeline-learning.json"),
                },
                "candidate_recheck": row.get("candidate_recheck", {}),
                "bug_dir": row.get("bug_dir", ""),
                "families": row_families,
            }
            frozen_rows.append(frozen_row)

    frozen_rows = rank_candidate_pipeline_rows(
        frozen_rows,
        confirmed_latest_families=confirmed_latest_families,
        existing_by_family=existing_by_family,
    )
    for frozen_row in frozen_rows:
        candidates.append(
            _process_candidate(
                candidate_id=str(frozen_row.get("candidate_id", "")),
                row=frozen_row,
                pipeline_dir=pipeline_dir,
                issue_drafts_dir=issue_drafts_dir,
                recheck_attempts=max(0, int(recheck_attempts)),
                reduce_artifacts=reduce_artifacts,
                standalone_reproducer=standalone_reproducer,
                confirmed_latest_families=confirmed_latest_families,
                existing_by_family=existing_by_family,
                family_seen_in_pipeline=family_seen_in_pipeline,
            )
        )

    frozen_manifest = {
        "schema_version": "candidate-freeze-v1",
        "generated_at": utc_now(),
        "generated_by": "datadiff candidate-pipeline",
        "manifest_file": _project_display_path(resolved_manifest) if resolved_manifest is not None else "",
        "evidence_files": [_project_display_path(path) for path in resolved_evidence_files],
        "strategy_snapshot_path": _project_display_path(strategy_snapshot_path),
        "candidate_count": len(frozen_rows),
        "bug_discovery_system": bug_discovery_system_descriptor(),
        "candidates": frozen_rows,
    }
    dump_json(frozen_manifest, frozen_path)

    queue = build_issue_readiness(
        latest_confirmation_files=latest_confirmation_files,
        new_issue_dir=issue_drafts_dir,
        old_issue_dir=resolved_old_issue_dir,
        generated_issue_dir=resolved_generated_issue_dir,
        include_generated=False,
    )
    readiness_by_path = {
        str(issue.get("path", "")): issue for issue in queue.get("issues", []) if str(issue.get("path", ""))
    }
    for candidate in candidates:
        draft_path = str(candidate.get("issue_draft", {}).get("path", ""))
        if draft_path and draft_path in readiness_by_path:
            candidate["issue_readiness"] = readiness_by_path[draft_path]

    manifest_path = pipeline_dir / "manifest.json"
    markdown_path = pipeline_dir / "manifest.md"
    manifest = {
        "schema_version": CANDIDATE_PIPELINE_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "generated_by": "datadiff candidate-pipeline",
        "manifest_path": _project_display_path(manifest_path),
        "markdown_path": _project_display_path(markdown_path),
        "inputs": {
            "manifest_file": _project_display_path(resolved_manifest) if resolved_manifest is not None else "",
            "evidence_files": [_project_display_path(path) for path in resolved_evidence_files],
            "output_dir": _project_display_path(pipeline_dir),
            "recheck_attempts": max(0, int(recheck_attempts)),
            "reduce_artifacts": bool(reduce_artifacts),
            "standalone_reproducer": bool(standalone_reproducer),
            "new_issue_dir": _project_display_path(resolved_new_issue_dir),
            "old_issue_dir": _project_display_path(resolved_old_issue_dir),
            "generated_issue_dir": _project_display_path(resolved_generated_issue_dir),
        },
        "frozen_candidates_path": _project_display_path(frozen_path),
        "strategy_snapshot_path": _project_display_path(strategy_snapshot_path),
        "bug_discovery_system": bug_discovery_system_descriptor(),
        "summary": _candidate_pipeline_summary(candidates, queue),
        "existing_issue_readiness_summary": existing_queue.get("summary", {}),
        "pipeline_issue_readiness_summary": queue.get("summary", {}),
        "submission_groups": queue.get("submission_groups", []),
        "candidates": candidates,
    }
    dump_json(manifest, manifest_path)
    markdown_path.write_text(render_candidate_pipeline_markdown(manifest), encoding="utf-8")
    return manifest


def render_candidate_pipeline_markdown(manifest: dict[str, Any]) -> str:
    summary = manifest.get("summary", {})
    lines = [
        "# DataDiffFuzz Candidate Pipeline",
        "",
        f"- Generated at: `{manifest.get('generated_at', '')}`",
        f"- Frozen candidates: `{summary.get('candidate_count', 0)}`",
        f"- Rechecked candidates: `{summary.get('rechecked_count', 0)}`",
        f"- Reproduced candidates: `{summary.get('reproduced_count', 0)}`",
        f"- Reduced artifacts: `{summary.get('reduced_count', 0)}`",
        f"- Candidate-bug triage verdicts: `{summary.get('candidate_bug_verdict_count', 0)}`",
        f"- Semantic-contract candidates: `{summary.get('semantic_contract_candidate_count', 0)}`; "
        f"boundary axes: `{', '.join(summary.get('semantic_contract_boundary_axes', [])) or 'none'}`",
        f"- IR rewrite candidates: `{summary.get('ir_rewrite_candidate_count', 0)}`; "
        f"rules: `{', '.join(summary.get('ir_rewrite_rules', [])) or 'none'}`",
        f"- Local duplicate families: `{summary.get('local_duplicate_candidate_count', 0)}`",
        f"- Issue drafts: `{summary.get('issue_draft_count', 0)}`",
        f"- Needs dedup check: `{summary.get('needs_dedup_check_count', 0)}`",
        "",
        "## Candidates",
        "",
        "| Candidate | Family | Score | True bug probability | Contract axes | IR rewrites | Reproduced | Triage | Dedup | Issue readiness |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for candidate in manifest.get("candidates", []):
        issue_readiness = candidate.get("issue_readiness", {})
        acquisition = candidate.get("candidate_acquisition", {}) if isinstance(candidate.get("candidate_acquisition"), dict) else {}
        contract_evidence = (
            candidate.get("semantic_contract_evidence", {})
            if isinstance(candidate.get("semantic_contract_evidence", {}), dict)
            else {}
        )
        ir_evidence = (
            candidate.get("ir_rewrite_evidence", {})
            if isinstance(candidate.get("ir_rewrite_evidence", {}), dict)
            else {}
        )
        lines.append(
            f"| `{candidate.get('candidate_id', '')}` | "
            f"`{candidate.get('primary_family', '')}` | "
            f"`{float(acquisition.get('acquisition_score', 0.0) or 0.0):.2f}` | "
            f"`{float(acquisition.get('true_bug_probability', 0.0) or 0.0):.2f}` | "
            f"`{', '.join(contract_evidence.get('matched_boundary_axes', []) or contract_evidence.get('finding_axes', [])) or 'none'}` | "
            f"`{', '.join(ir_evidence.get('operators', [])) or 'none'}` | "
            f"`{str(candidate.get('recheck', {}).get('reproduced', False)).lower()}` | "
            f"`{candidate.get('triage', {}).get('verdict', '')}` | "
            f"`{candidate.get('dedup', {}).get('status', '')}` | "
            f"`{issue_readiness.get('readiness_status', '')}` |"
        )
    if not manifest.get("candidates"):
        lines.append("| none |  | 0.00 | 0.00 | none | none | false |  |  |  |")
    lines.append("")
    return "\n".join(lines)


def _process_candidate(
    *,
    candidate_id: str,
    row: dict[str, Any],
    pipeline_dir: Path,
    issue_drafts_dir: Path,
    recheck_attempts: int,
    reduce_artifacts: bool,
    standalone_reproducer: bool,
    confirmed_latest_families: set[str],
    existing_by_family: dict[str, list[str]],
    family_seen_in_pipeline: dict[str, str],
) -> dict[str, Any]:
    families = list(row.get("families", []) or [])
    primary_family = str(families[0]) if families else "unknown"
    case = Case.from_dict(row.get("case", {}))
    backends = _candidate_backends(row)
    config_data = row.get("config", {}) if isinstance(row.get("config", {}), dict) else {}
    config = ExperimentConfig.from_payload(config_data)
    semantic_contract_evidence = (
        dict(row.get("semantic_contract_evidence", {}))
        if isinstance(row.get("semantic_contract_evidence", {}), dict)
        else {}
    )
    ir_rewrite_evidence = (
        dict(row.get("ir_rewrite_evidence", {}))
        if isinstance(row.get("ir_rewrite_evidence", {}), dict)
        else {}
    )
    artifact_dir, artifact_created = _ensure_candidate_artifact_dir(row)
    recheck = _recheck_candidate(case, backends, config, families, attempts=recheck_attempts)

    reduction: dict[str, Any] = {"requested": bool(reduce_artifacts), "performed": False}
    triage: dict[str, Any] = {}
    if artifact_dir is not None and recheck.get("reproduced"):
        original_case = (
            Case.from_dict(load_json(artifact_dir / "case.json"))
            if (artifact_dir / "case.json").is_file()
            else case
        )
        original_findings = _load_artifact_findings(artifact_dir, row)
        triage_case = original_case
        triage_backends = backends or _artifact_backends(artifact_dir)
        triage_config = _artifact_config_or_default(artifact_dir, config_data)
        if reduce_artifacts and triage_backends and original_findings:
            target_roots = [finding.get("root_cause", "unknown") for finding in original_findings]
            reduced = reduce_case(
                original_case,
                backends=triage_backends,
                config=triage_config,
                target_kinds=[finding.get("kind", "") for finding in original_findings],
                target_roots=target_roots,
                target_suspicious_backends=[
                    finding.get("suspicious_backends", []) for finding in original_findings
                ],
            )
            dump_json(reduced.to_dict(), artifact_dir / "reduced_case.json")
            write_reduced_reproducer(artifact_dir, triage_backends)
            triage_case = reduced
            reduction = {
                "requested": True,
                "performed": True,
                "original_rows": len(original_case.tables[0].rows) if original_case.tables else 0,
                "reduced_rows": len(reduced.tables[0].rows) if reduced.tables else 0,
                "original_operations": len(original_case.program.operations),
                "reduced_operations": len(reduced.program.operations),
                "reduced_case_path": _project_display_path(artifact_dir / "reduced_case.json"),
                "reduced_reproducer_path": _project_display_path(artifact_dir / "reproduce_reduced.py"),
            }
        reproduced = run_loaded_case(triage_case, backends=triage_backends, config=triage_config, save_artifact=False)
        report = build_triage_report(
            triage_case,
            original_findings=original_findings,
            reproduced_findings=reproduced.get("findings", []),
            config=triage_config.to_dict(),
            backends=triage_backends,
        )
        report["artifact"] = str(artifact_dir)
        report["reduced"] = bool(reduction.get("performed"))
        report["rows"] = len(triage_case.tables[0].rows) if triage_case.tables else 0
        report["operations"] = len(triage_case.program.operations)
        triage_json, triage_md = write_triage_artifact(artifact_dir, report)
        triage = {
            "verdict": report.get("verdict", ""),
            "paper_status": report.get("paper_status", ""),
            "triage_confidence": report.get("triage_confidence", ""),
            "triage_json": _project_display_path(triage_json),
            "triage_markdown": _project_display_path(triage_md),
            "suspicious_backends": list(report.get("suspicious_backends", []) or []),
            "reproduced_roots": list(report.get("reproduced_roots", []) or []),
        }
        if standalone_reproducer and supports_standalone_reproducer(report):
            standalone_path = write_standalone_reproducer(artifact_dir, report)
            triage["standalone_reproducer"] = _project_display_path(standalone_path)

    local_duplicate_paths = sorted(
        {
            path
            for family in families
            for path in existing_by_family.get(family, [])
        }
    )
    pipeline_duplicate_of = family_seen_in_pipeline.get(primary_family, "")
    if primary_family and primary_family not in family_seen_in_pipeline:
        family_seen_in_pipeline[primary_family] = candidate_id
    dedup = {
        "status": _dedup_status(
            families,
            local_duplicate_paths=local_duplicate_paths,
            pipeline_duplicate_of=pipeline_duplicate_of,
            confirmed_latest_families=confirmed_latest_families,
        ),
        "local_duplicate_paths": local_duplicate_paths,
        "pipeline_duplicate_of": pipeline_duplicate_of,
        "confirmed_latest_family": any(family in confirmed_latest_families for family in families),
    }

    issue_draft = {}
    if triage.get("verdict") == "candidate_implementation_bug" and artifact_dir is not None:
        issue_path = issue_drafts_dir / f"{slugify(candidate_id)}.md"
        issue_path.write_text(
            _issue_draft_markdown(
                candidate_id=candidate_id,
                primary_family=primary_family,
                families=families,
                row=row,
                artifact_dir=artifact_dir,
                pipeline_dir=pipeline_dir,
                triage=triage,
                dedup=dedup,
                semantic_contract_evidence=semantic_contract_evidence,
                ir_rewrite_evidence=ir_rewrite_evidence,
            ),
            encoding="utf-8",
        )
        issue_draft = {
            "path": _project_display_path(issue_path),
            "status_text": _issue_status_text(dedup),
        }
    learning_event = append_learning_event(
        {
            "candidate_id": candidate_id,
            "primary_family": primary_family,
            "families": families,
            "reproduced": bool(recheck.get("reproduced")),
            "reproduced_families": list(recheck.get("reproduced_families", []) or []),
            "triage_verdict": str(triage.get("verdict", "")),
            "suspicious_backends": list(triage.get("suspicious_backends", []) or []),
            "reproduced_roots": list(triage.get("reproduced_roots", []) or []),
            "dedup_status": str(dedup.get("status", "")),
            "source_run_file": str(row.get("source_run_file", "")),
            "source_evidence_file": str(row.get("source_evidence_file", "")),
            "semantic_contract_boundary_axes": list(semantic_contract_evidence.get("boundary_axes", []) or []),
            "semantic_contract_matched_boundary_axes": list(
                semantic_contract_evidence.get("matched_boundary_axes", []) or []
            ),
            "ir_rewrite_rules": list(ir_rewrite_evidence.get("rule_ids", []) or []),
        },
        output_dir=pipeline_dir / "strategy-learning",
        learning_id="candidate-pipeline-learning",
    )

    return {
        "candidate_id": candidate_id,
        "primary_family": primary_family,
        "families": families,
        "candidate_acquisition": dict(row.get("candidate_acquisition", {}) or {}),
        "source_evidence_file": str(row.get("source_evidence_file", "")),
        "source_run_file": str(row.get("source_run_file", "")),
        "case_id": row.get("case", {}).get("case_id", ""),
        "semantic_contract_evidence": semantic_contract_evidence,
        "ir_rewrite_evidence": ir_rewrite_evidence,
        "bug_dir": _project_display_path(artifact_dir) if artifact_dir is not None else "",
        "artifact_created": artifact_created,
        "initial_candidate_recheck": dict(row.get("candidate_recheck", {}) or {}),
        "recheck": recheck,
        "reduction": reduction,
        "triage": triage,
        "dedup": dedup,
        "issue_draft": issue_draft,
        "strategy_learning_path": _project_display_path(learning_event),
        "issue_readiness": {},
    }


def _load_evidence_payload(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"candidate evidence must be a JSON object: {path}")
    return payload


def _semantic_contract_evidence(row: dict[str, Any]) -> dict[str, Any]:
    lattice = _semantic_contract_lattice_from_row(row)
    if not lattice:
        return {}
    finding_axes = sorted(
        {
            axis
            for finding in row.get("findings", []) or []
            if isinstance(finding, dict)
            for axis in finding_contract_axes(finding)
        }
    )
    boundary_axes = _string_list(lattice.get("boundary_axes", []))
    strict_axes = _string_list(lattice.get("strict_axes", []))
    return {
        "schema_version": str(lattice.get("schema_version", "")),
        "case_id": str(lattice.get("case_id", "")),
        "boundary_axes": boundary_axes,
        "strict_axes": strict_axes,
        "contract_tags": _string_list(lattice.get("contract_tags", [])),
        "finding_axes": finding_axes,
        "matched_boundary_axes": sorted(set(boundary_axes) & set(finding_axes)),
        "operation_contract_count": len(lattice.get("operation_contracts", []) or []),
    }


def _semantic_contract_lattice_from_row(row: dict[str, Any]) -> dict[str, Any]:
    direct = row.get("semantic_contract_lattice", {})
    if isinstance(direct, dict) and direct:
        return direct
    case_payload = row.get("case", {}) if isinstance(row.get("case", {}), dict) else {}
    case_metadata = case_payload.get("metadata", {}) if isinstance(case_payload.get("metadata", {}), dict) else {}
    metadata_lattice = case_metadata.get("semantic_contract_lattice", {})
    if isinstance(metadata_lattice, dict) and metadata_lattice:
        return metadata_lattice
    try:
        return semantic_contract_lattice_payload(Case.from_dict(case_payload))
    except (KeyError, TypeError, ValueError):
        return {}


def _ir_rewrite_evidence(row: dict[str, Any]) -> dict[str, Any]:
    rules = _ir_rewrite_rules_from_row(row)
    if not rules:
        return {}
    return {
        "rule_count": len(rules),
        "rule_ids": sorted({str(rule.get("rule_id", "")) for rule in rules if rule.get("rule_id")}),
        "operators": sorted({str(rule.get("operator", "")) for rule in rules if rule.get("operator")}),
        "relations": sorted({str(rule.get("relation", "")) for rule in rules if rule.get("relation")}),
        "semantics_classes": sorted(
            {str(rule.get("semantics_class", "")) for rule in rules if rule.get("semantics_class")}
        ),
        "contract_axes": sorted(
            {
                axis
                for rule in rules
                for axis in _string_list(rule.get("contract_axes", []))
            }
        ),
        "rules": rules,
    }


def _ir_rewrite_rules_from_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    for metadata in _candidate_metadata_sources(row):
        mutation = metadata.get("mutation", {}) if isinstance(metadata.get("mutation", {}), dict) else {}
        existing_rules = mutation.get("ir_rewrite_rules", [])
        if isinstance(existing_rules, list):
            rules.extend(dict(rule) for rule in existing_rules if isinstance(rule, dict))
        existing_rule = mutation.get("ir_rewrite_rule", {})
        if isinstance(existing_rule, dict) and existing_rule:
            rules.append(dict(existing_rule))
        operator = str(mutation.get("operator", "") or "")
        if operator:
            inferred = ir_rewrite_rule_metadata(operator, detail=str(mutation.get("detail", "") or ""))
            if inferred:
                rules.append(inferred)
    return _dedupe_rule_payloads(rules)


def _candidate_metadata_sources(row: dict[str, Any]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    row_mutation = row.get("mutation", {})
    if isinstance(row_mutation, dict) and row_mutation:
        sources.append({"mutation": row_mutation})
    row_metadata = row.get("metadata", {})
    if isinstance(row_metadata, dict):
        sources.append(row_metadata)
    case_payload = row.get("case", {}) if isinstance(row.get("case", {}), dict) else {}
    case_metadata = case_payload.get("metadata", {}) if isinstance(case_payload.get("metadata", {}), dict) else {}
    if case_metadata:
        sources.append(case_metadata)
    return sources


def _dedupe_rule_payloads(rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for rule in rules:
        key = (
            str(rule.get("rule_id", "")),
            str(rule.get("operator", "")),
            str(rule.get("detail", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(rule)
    return deduped


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def _candidate_pipeline_manifest_evidence_files(manifest_file: Path | None) -> list[Path]:
    if manifest_file is None or not manifest_file.is_file():
        return []
    manifest = load_json(manifest_file)
    if not isinstance(manifest, dict):
        return []
    evidence_files: list[Path] = []
    fuzz_run = manifest.get("fuzz_run", {}) if isinstance(manifest.get("fuzz_run"), dict) else {}
    if fuzz_run.get("fresh_candidate_evidence"):
        evidence_files.append(_resolve_project_path(Path(str(fuzz_run.get("fresh_candidate_evidence")))))
    for run in manifest.get("runs", []) or []:
        if not isinstance(run, dict):
            continue
        if run.get("fresh_candidate_evidence"):
            evidence_files.append(_resolve_project_path(Path(str(run.get("fresh_candidate_evidence")))))
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in evidence_files:
        key = path.resolve() if path.exists() else path.absolute()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _issues_by_family(issues: list[dict[str, Any]]) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for issue in issues:
        family = str(issue.get("family_guess", "")).strip()
        path = str(issue.get("path", "")).strip()
        if not family or not path:
            continue
        mapping.setdefault(family, []).append(path)
    return {family: sorted(paths) for family, paths in mapping.items()}


def _candidate_id(primary_family: str, row: dict[str, Any], evidence_index: int, row_index: int) -> str:
    case_id = str(row.get("case", {}).get("case_id", "")).strip() or f"case-{evidence_index}-{row_index}"
    return slugify(f"{primary_family}-{case_id}", max_len=120)


def _candidate_backends(row: dict[str, Any]) -> list[str]:
    normalized = row.get("normalized", {})
    if isinstance(normalized, dict) and normalized:
        return list(normalized)
    raw_results = row.get("raw_results", {})
    if isinstance(raw_results, dict) and raw_results:
        return list(raw_results)
    backends = sorted(
        {
            backend
            for finding in row.get("findings", []) or []
            for backend in finding.get("suspicious_backends", []) or []
            if backend
        }
    )
    return backends


def _ensure_candidate_artifact_dir(row: dict[str, Any]) -> tuple[Path | None, bool]:
    artifact_dir_text = str(row.get("bug_dir", "") or "").strip()
    if artifact_dir_text:
        artifact_dir = _resolve_project_path(Path(artifact_dir_text))
        if artifact_dir.is_dir():
            return artifact_dir, False
    case = Case.from_dict(row.get("case", {}))
    findings = [_finding_from_dict(item) for item in row.get("findings", []) if isinstance(item, dict)]
    normalized = row.get("normalized", {}) if isinstance(row.get("normalized", {}), dict) else {}
    raw_results = row.get("raw_results", {}) if isinstance(row.get("raw_results", {}), dict) else {}
    if not findings or not normalized or not raw_results:
        return None, False
    artifact_dir = save_bug_artifact(
        case,
        raw_results=raw_results,
        normalized=normalized,
        findings=findings,
        config=row.get("config", {}),
    )
    return artifact_dir, True


def _finding_from_dict(data: dict[str, Any]) -> Finding:
    return Finding(
        finding_id=str(data.get("finding_id", data.get("signature", "candidate"))),
        kind=str(data.get("kind", "")),
        severity=str(data.get("severity", "medium")),
        suspicious_backends=list(data.get("suspicious_backends", []) or []),
        evidence=str(data.get("evidence", "")),
        signature=str(data.get("signature", data.get("finding_id", "candidate"))),
        root_cause=str(data.get("root_cause", "unknown")),
        oracle=str(data.get("oracle", "differential")),
        confidence=str(data.get("confidence", "medium")),
        triage_verdict=str(data.get("triage_verdict", "unclassified")),
        paper_status=str(data.get("paper_status", "unclassified")),
        triage_confidence=str(data.get("triage_confidence", "low")),
        false_positive=bool(data.get("false_positive", False)),
        false_positive_reason=str(data.get("false_positive_reason", "")),
        triage_evidence=str(data.get("triage_evidence", "")),
        recommendation=list(data.get("recommendation", []) or []),
        documentation_refs=list(data.get("documentation_refs", []) or []),
        mismatch_class=str(data.get("mismatch_class", "")),
        discovery_origin=str(data.get("discovery_origin", "organic")),
        source_issue=str(data.get("source_issue", "")),
    )


def _recheck_candidate(
    case: Case,
    backends: list[str],
    config: ExperimentConfig,
    families: list[str],
    *,
    attempts: int,
) -> dict[str, Any]:
    if attempts <= 0 or not backends or not families:
        return {"attempts": 0, "reproduced": False, "reproduced_families": [], "attempt_summaries": []}
    recheck_config_data = config.to_dict()
    recheck_config_data["candidate_recheck_count"] = 0
    recheck_config_data["enable_artifact"] = False
    recheck_config_data["enable_reducer"] = False
    recheck_config = ExperimentConfig.from_payload(recheck_config_data)
    original_families = set(families)
    reproduced_families: set[str] | None = None
    attempt_summaries: list[dict[str, Any]] = []
    for attempt in range(attempts):
        row = run_loaded_case(case, backends=backends, config=recheck_config, save_artifact=False)
        current_families = set(candidate_issue_family_keys(row.get("findings", [])).keys())
        matched = sorted(original_families & current_families)
        if reproduced_families is None:
            reproduced_families = set(matched)
        else:
            reproduced_families &= set(matched)
        attempt_summaries.append(
            {
                "attempt": attempt + 1,
                "status": row.get("status", ""),
                "finding_count": len(row.get("findings", []) or []),
                "candidate_families": sorted(current_families),
                "matched_families": matched,
            }
        )
    reproduced_families = reproduced_families or set()
    return {
        "attempts": attempts,
        "reproduced": bool(reproduced_families),
        "reproduced_families": sorted(reproduced_families),
        "attempt_summaries": attempt_summaries,
    }


def _load_artifact_findings(artifact_dir: Path, row: dict[str, Any]) -> list[dict[str, Any]]:
    findings_path = artifact_dir / "findings.json"
    if findings_path.is_file():
        payload = load_json(findings_path)
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
    return [item for item in row.get("findings", []) if isinstance(item, dict)]


def _artifact_config_or_default(artifact_dir: Path, fallback: dict[str, Any]) -> ExperimentConfig:
    config_path = artifact_dir / "config.json"
    if config_path.is_file():
        payload = load_json(config_path)
        if isinstance(payload, dict) and payload:
            return ExperimentConfig.from_payload(payload)
    return ExperimentConfig.from_payload(fallback)


def _artifact_backends(artifact_dir: Path) -> list[str]:
    results_path = artifact_dir / "results.json"
    if not results_path.is_file():
        return []
    payload = load_json(results_path)
    return list(payload) if isinstance(payload, dict) else []


def _dedup_status(
    families: list[str],
    *,
    local_duplicate_paths: list[str],
    pipeline_duplicate_of: str,
    confirmed_latest_families: set[str],
) -> str:
    if any(family in confirmed_latest_families for family in families):
        return "already_confirmed_latest"
    if pipeline_duplicate_of:
        return "duplicate_pipeline_family"
    if local_duplicate_paths:
        return "duplicate_local_family"
    return "needs_final_upstream_dedup"


def _issue_status_text(dedup: dict[str, Any]) -> str:
    status = str(dedup.get("status", ""))
    if status == "already_confirmed_latest":
        return "Already submitted or confirmed upstream"
    return "Needs final upstream dedup before submission"


def _issue_draft_markdown(
    *,
    candidate_id: str,
    primary_family: str,
    families: list[str],
    row: dict[str, Any],
    artifact_dir: Path,
    pipeline_dir: Path,
    triage: dict[str, Any],
    dedup: dict[str, Any],
    semantic_contract_evidence: dict[str, Any] | None = None,
    ir_rewrite_evidence: dict[str, Any] | None = None,
) -> str:
    reproducer_path = artifact_dir / "reproduce_reduced.py"
    if not reproducer_path.is_file():
        reproducer_path = artifact_dir / "reproduce.py"
    reproducer_source = reproducer_path.read_text(encoding="utf-8") if reproducer_path.is_file() else "print('missing')\n"
    local_duplicate_paths = list(dedup.get("local_duplicate_paths", []) or [])
    lines = [
        f"# Auto Issue Draft: {primary_family}",
        "",
        "## Discovery Record",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| Project family labels observed | `{', '.join(families)}` |",
        f"| First DataDiffFuzz signal | {utc_now()} |",
        f"| How found | Automated candidate pipeline from `{row.get('source_run_file', '')}` |",
        f"| Current status | {_issue_status_text(dedup)} |",
        "",
        "## Environment",
        "",
        f"- Target backend/version: see `{_project_display_path(artifact_dir / 'environment.json')}`",
        f"- Candidate artifact: `{_project_display_path(artifact_dir)}`",
        f"- Pipeline manifest directory: `{_project_display_path(pipeline_dir)}`",
        "",
        "## Reproducer",
        "",
        "```python",
        reproducer_source.rstrip(),
        "```",
        "",
        "## Expected Output",
        "",
        "```text",
        "Backends should agree on the final normalized result for this case.",
        "```",
        "",
        "## Actual Output",
        "",
        "```text",
        f"Triage verdict: {triage.get('verdict', '')}",
        f"Suspicious backends: {', '.join(triage.get('suspicious_backends', [])) or 'unknown'}",
        f"Reproduced roots: {', '.join(triage.get('reproduced_roots', [])) or 'unknown'}",
        f"Local duplicate drafts: {', '.join(local_duplicate_paths) or 'none'}",
        "```",
        "",
        "## DataDiffFuzz Evidence",
        "",
        f"- Frozen candidate evidence: `{_project_display_path(pipeline_dir / 'frozen-candidates.json')}`",
        f"- Source evidence file: `{row.get('source_evidence_file', '')}`",
        f"- Source run log: `{row.get('source_run_file', '')}`",
        f"- Candidate artifact: `{_project_display_path(artifact_dir)}`",
        f"- Triage JSON: `{triage.get('triage_json', '')}`",
        f"- Triage markdown: `{triage.get('triage_markdown', '')}`",
    ]
    contract_evidence = semantic_contract_evidence or {}
    if contract_evidence:
        lines.extend(
            [
                f"- Semantic contract boundary axes: `{', '.join(contract_evidence.get('boundary_axes', [])) or 'none'}`",
                f"- Finding contract axes: `{', '.join(contract_evidence.get('finding_axes', [])) or 'none'}`",
                f"- Matched contract boundary axes: `{', '.join(contract_evidence.get('matched_boundary_axes', [])) or 'none'}`",
            ]
        )
    rewrite_evidence = ir_rewrite_evidence or {}
    if rewrite_evidence:
        lines.extend(
            [
                f"- IR rewrite rules: `{', '.join(rewrite_evidence.get('rule_ids', [])) or 'none'}`",
                f"- IR rewrite semantic classes: `{', '.join(rewrite_evidence.get('semantics_classes', [])) or 'none'}`",
            ]
        )
    reduced_case_path = artifact_dir / "reduced_case.json"
    reduced_reproducer_path = artifact_dir / "reproduce_reduced.py"
    if reduced_case_path.is_file():
        lines.append(f"- Reduced case: `{_project_display_path(reduced_case_path)}`")
    if reduced_reproducer_path.is_file():
        lines.append(f"- Reduced reproducer: `{_project_display_path(reduced_reproducer_path)}`")
    lines.append("")
    return "\n".join(lines)


def _candidate_pipeline_summary(candidates: list[dict[str, Any]], queue: dict[str, Any]) -> dict[str, Any]:
    readiness_counts = Counter(
        candidate.get("issue_readiness", {}).get("readiness_status", "")
        for candidate in candidates
        if candidate.get("issue_readiness")
    )
    reproduced_count = sum(1 for candidate in candidates if candidate.get("recheck", {}).get("reproduced"))
    reduced_count = sum(1 for candidate in candidates if candidate.get("reduction", {}).get("performed"))
    local_duplicate_count = sum(
        1
        for candidate in candidates
        if candidate.get("dedup", {}).get("status") in {"duplicate_local_family", "duplicate_pipeline_family"}
    )
    contract_summary = _semantic_contract_candidate_summary(candidates)
    rewrite_summary = _ir_rewrite_candidate_summary(candidates)
    return {
        "candidate_count": len(candidates),
        "unique_family_count": len({candidate.get("primary_family", "") for candidate in candidates if candidate.get("primary_family")}),
        "rechecked_count": sum(1 for candidate in candidates if candidate.get("recheck", {}).get("attempts", 0) > 0),
        "reproduced_count": reproduced_count,
        "recheck_pass_rate": reproduced_count / len(candidates) if candidates else 0.0,
        "reduced_count": reduced_count,
        "artifact_created_count": sum(1 for candidate in candidates if candidate.get("artifact_created")),
        "candidate_bug_verdict_count": sum(
            1 for candidate in candidates if candidate.get("triage", {}).get("verdict") == "candidate_implementation_bug"
        ),
        "local_duplicate_candidate_count": local_duplicate_count,
        "issue_draft_count": sum(1 for candidate in candidates if candidate.get("issue_draft", {}).get("path")),
        "ready_to_submit_count": readiness_counts.get("ready_to_submit", 0),
        "needs_dedup_check_count": readiness_counts.get("needs_dedup_check", 0),
        "needs_reproducer_or_evidence_count": readiness_counts.get("needs_reproducer_or_evidence", 0),
        "already_submitted_or_confirmed_count": readiness_counts.get("already_submitted_or_confirmed", 0),
        "not_latest_reproducible_count": readiness_counts.get("not_latest_reproducible", 0),
        "submission_group_count": int(queue.get("summary", {}).get("submission_group_count", 0) or 0),
        "semantic_contract_candidate_count": contract_summary["candidate_count"],
        "semantic_contract_boundary_axes": contract_summary["boundary_axes"],
        "semantic_contract_matched_boundary_axes": contract_summary["matched_boundary_axes"],
        "semantic_contract_operation_contract_count": contract_summary["operation_contract_count"],
        "semantic_contract_evidence": contract_summary,
        "ir_rewrite_candidate_count": rewrite_summary["candidate_count"],
        "ir_rewrite_rule_count": rewrite_summary["rule_count"],
        "ir_rewrite_rules": rewrite_summary["rule_ids"],
        "ir_rewrite_operators": rewrite_summary["operators"],
        "ir_rewrite_semantics_classes": rewrite_summary["semantics_classes"],
        "ir_rewrite_evidence": rewrite_summary,
    }


def _semantic_contract_candidate_summary(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    evidence_rows = [
        candidate.get("semantic_contract_evidence", {})
        for candidate in candidates
        if isinstance(candidate.get("semantic_contract_evidence", {}), dict)
        and candidate.get("semantic_contract_evidence", {})
    ]
    return {
        "candidate_count": len(evidence_rows),
        "boundary_axes": sorted(
            {
                axis
                for evidence in evidence_rows
                for axis in _string_list(evidence.get("boundary_axes", []))
            }
        ),
        "matched_boundary_axes": sorted(
            {
                axis
                for evidence in evidence_rows
                for axis in _string_list(evidence.get("matched_boundary_axes", []))
            }
        ),
        "finding_axes": sorted(
            {
                axis
                for evidence in evidence_rows
                for axis in _string_list(evidence.get("finding_axes", []))
            }
        ),
        "contract_tags": sorted(
            {
                tag
                for evidence in evidence_rows
                for tag in _string_list(evidence.get("contract_tags", []))
            }
        ),
        "operation_contract_count": sum(
            int(evidence.get("operation_contract_count", 0) or 0) for evidence in evidence_rows
        ),
    }


def _ir_rewrite_candidate_summary(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    evidence_rows = [
        candidate.get("ir_rewrite_evidence", {})
        for candidate in candidates
        if isinstance(candidate.get("ir_rewrite_evidence", {}), dict)
        and candidate.get("ir_rewrite_evidence", {})
    ]
    return {
        "candidate_count": len(evidence_rows),
        "rule_count": sum(int(evidence.get("rule_count", 0) or 0) for evidence in evidence_rows),
        "rule_ids": sorted(
            {
                rule_id
                for evidence in evidence_rows
                for rule_id in _string_list(evidence.get("rule_ids", []))
            }
        ),
        "operators": sorted(
            {
                operator
                for evidence in evidence_rows
                for operator in _string_list(evidence.get("operators", []))
            }
        ),
        "semantics_classes": sorted(
            {
                semantics_class
                for evidence in evidence_rows
                for semantics_class in _string_list(evidence.get("semantics_classes", []))
            }
        ),
        "contract_axes": sorted(
            {
                axis
                for evidence in evidence_rows
                for axis in _string_list(evidence.get("contract_axes", []))
            }
        ),
    }


def _pipeline_dir_name(manifest_file: Path | None, evidence_files: list[Path]) -> str:
    stamp = utc_now().replace(":", "").replace("-", "").replace("Z", "")
    seed = manifest_file.stem if manifest_file is not None else evidence_files[0].stem
    return f"pipeline-{slugify(seed)}-{stamp}"


def _resolve_project_path(path: str | Path | None) -> Path:
    return _resolve_project_path_impl(path, project_root=PROJECT_ROOT)


def _project_display_path(path: str | Path | None) -> str:
    return _project_display_path_impl(path, project_root=PROJECT_ROOT)


_ensure_bug_dir = _ensure_candidate_artifact_dir
_load_candidate_findings = _load_artifact_findings
_bug_dir_backends = _artifact_backends
