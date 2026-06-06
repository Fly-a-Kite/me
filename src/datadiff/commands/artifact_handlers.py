from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from datadiff.finding_outcomes import row_has_rewardable_new_behavior


def _experiment_config_from_payload(
    experiment_config_factory: Callable[..., Any],
    payload: dict[str, Any] | None,
    **overrides: Any,
) -> Any:
    config_data = dict(payload or {})
    config_data.update(overrides)
    from_payload = getattr(experiment_config_factory, "from_payload", None)
    if callable(from_payload):
        return from_payload(config_data)
    return experiment_config_factory(**config_data) if config_data else experiment_config_factory()


def _apply_fixture_replay_new_behavior_flags(row: dict[str, Any], config: Any) -> tuple[bool, bool]:
    raw_new_behavior = bool(row.get("is_new_behavior", bool(row.get("findings"))))
    raw_signal_new_behavior = bool(row.get("signal_new_behavior", raw_new_behavior))
    row["is_new_behavior"] = raw_new_behavior
    row["signal_new_behavior"] = raw_signal_new_behavior
    signal_new_behavior = row_has_rewardable_new_behavior(
        row,
        known_saturated_bug_families=tuple(getattr(config, "known_saturated_bug_families", ()) or ()),
    )
    row["signal_new_behavior"] = signal_new_behavior
    return raw_new_behavior, signal_new_behavior


def cmd_reproduce_impl(
    args: argparse.Namespace,
    *,
    case_from_dict_func: Callable[[dict[str, Any]], Any],
    load_json_func: Callable[[Path], Any],
    experiment_config_factory: Callable[..., Any],
    parse_backends_func: Callable[[str], list[str]],
    run_loaded_case_func: Callable[..., dict[str, Any]],
) -> int:
    bug_dir = Path(args.bug)
    repro = bug_dir / "reproduce.py"
    if not repro.exists():
        raise FileNotFoundError(repro)
    if args.print_command:
        print(f"Run this command to reproduce:\npython {repro}")
        return 0
    case = case_from_dict_func(load_json_func(bug_dir / "case.json"))
    config_path = bug_dir / "config.json"
    config_data = load_json_func(config_path) if config_path.exists() else {}
    config = _experiment_config_from_payload(experiment_config_factory, config_data)
    backends = parse_backends_func(args.backends) if args.backends else list(load_json_func(bug_dir / "results.json"))
    result = run_loaded_case_func(case, backends=backends, config=config, save_artifact=False)
    print(f"status={result['status']}")
    for finding in result["findings"]:
        print(
            f"- {finding['kind']} root={finding.get('root_cause', 'unknown')} "
            f"oracle={finding.get('oracle', 'unknown')} signature={finding.get('signature', '')}"
        )
    return 0


def cmd_validate_artifact_impl(
    args: argparse.Namespace,
    *,
    case_from_dict_func: Callable[[dict[str, Any]], Any],
    load_json_func: Callable[[Path], Any],
    experiment_config_factory: Callable[..., Any],
    parse_backends_func: Callable[[str], list[str]],
    run_loaded_case_func: Callable[..., dict[str, Any]],
) -> int:
    bug_dir = Path(args.bug)
    case = case_from_dict_func(load_json_func(bug_dir / "case.json"))
    original_findings = load_json_func(bug_dir / "findings.json")
    config_path = bug_dir / "config.json"
    config_data = load_json_func(config_path) if config_path.exists() else {}
    config = _experiment_config_from_payload(experiment_config_factory, config_data)
    backends = parse_backends_func(args.backends) if args.backends else list(load_json_func(bug_dir / "results.json"))
    result = run_loaded_case_func(case, backends=backends, config=config, save_artifact=False)

    original_kinds = {f.get("kind", "") for f in original_findings}
    reproduced_kinds = {f.get("kind", "") for f in result.get("findings", [])}
    original_roots = {f.get("root_cause", "unknown") for f in original_findings}
    reproduced_roots = {f.get("root_cause", "unknown") for f in result.get("findings", [])}
    kind_ok = bool(original_kinds & reproduced_kinds)
    root_ok = bool(original_roots & reproduced_roots) if original_roots else True
    ok = result["status"] == "bug" and kind_ok
    status = "valid" if ok and root_ok else "valid-root-changed" if ok else "not-reproduced"
    print(f"artifact={bug_dir}")
    print(f"status={status}")
    print(f"original_kinds={sorted(original_kinds)}")
    print(f"reproduced_kinds={sorted(reproduced_kinds)}")
    print(f"original_roots={sorted(original_roots)}")
    print(f"reproduced_roots={sorted(reproduced_roots)}")
    return 0 if ok else 1


def cmd_triage_artifact_impl(
    args: argparse.Namespace,
    *,
    case_from_dict_func: Callable[[dict[str, Any]], Any],
    load_json_func: Callable[[Path], Any],
    dump_json_func: Callable[..., None],
    experiment_config_factory: Callable[..., Any],
    parse_backends_func: Callable[[str], list[str]],
    load_artifact_config_func: Callable[[Path], dict[str, Any]],
    reduce_case_func: Callable[..., Any],
    run_loaded_case_func: Callable[..., dict[str, Any]],
    build_triage_report_func: Callable[..., dict[str, Any]],
    write_triage_artifact_func: Callable[..., tuple[Path, Path]],
    supports_standalone_reproducer_func: Callable[[dict[str, Any]], bool],
    write_standalone_reproducer_func: Callable[[Path, dict[str, Any]], Path],
    write_reduced_reproducer_func: Callable[[Path, list[str]], None],
) -> int:
    bug_dir = Path(args.bug)
    case = case_from_dict_func(load_json_func(bug_dir / "case.json"))
    original_findings = load_json_func(bug_dir / "findings.json")
    config_data = load_artifact_config_func(bug_dir)
    config = _experiment_config_from_payload(experiment_config_factory, config_data)
    backends = parse_backends_func(args.backends) if args.backends else list(load_json_func(bug_dir / "results.json"))

    triage_case = case
    if args.reduce:
        target_roots = [] if args.reduce_ignore_roots else [
            finding.get("root_cause", "unknown") for finding in original_findings
        ]
        triage_case = reduce_case_func(
            case,
            backends=backends,
            config=config,
            target_kinds=[finding.get("kind", "") for finding in original_findings],
            target_roots=target_roots,
            target_suspicious_backends=[
                finding.get("suspicious_backends", []) for finding in original_findings
            ],
        )
        dump_json_func(triage_case.to_dict(), bug_dir / "reduced_case.json")
        write_reduced_reproducer_func(bug_dir, backends)

    result = run_loaded_case_func(triage_case, backends=backends, config=config, save_artifact=False)
    report = build_triage_report_func(
        triage_case,
        original_findings=original_findings,
        reproduced_findings=result.get("findings", []),
        config=config.to_dict(),
        backends=backends,
    )
    report["artifact"] = str(bug_dir)
    report["reduced"] = args.reduce
    report["rows"] = len(triage_case.tables[0].rows)
    report["operations"] = len(triage_case.program.operations)
    json_path, md_path = write_triage_artifact_func(bug_dir, report)
    print(f"verdict={report['verdict']}")
    print(f"paper_status={report['paper_status']}")
    print(f"triage_json={json_path}")
    print(f"triage_md={md_path}")
    if args.standalone_reproducer:
        if supports_standalone_reproducer_func(report):
            standalone_path = write_standalone_reproducer_func(bug_dir, report)
            print(f"standalone_reproducer={standalone_path}")
        else:
            print("standalone_reproducer=skipped (no standalone template for this root cause)")
    return 0


def cmd_reduce_impl(
    args: argparse.Namespace,
    *,
    case_from_dict_func: Callable[[dict[str, Any]], Any],
    load_json_func: Callable[[Path], Any],
    dump_json_func: Callable[..., None],
    experiment_config_factory: Callable[..., Any],
    parse_backends_func: Callable[[str], list[str]],
    load_artifact_config_func: Callable[[Path], dict[str, Any]],
    reduce_case_func: Callable[..., Any],
    run_loaded_case_func: Callable[..., dict[str, Any]],
    write_reduced_reproducer_func: Callable[[Path, list[str]], None],
) -> int:
    bug_dir = Path(args.bug)
    case = case_from_dict_func(load_json_func(bug_dir / "case.json"))
    backends = parse_backends_func(args.backends)
    original_findings = load_json_func(bug_dir / "findings.json")
    config_data = load_artifact_config_func(bug_dir)
    config = _experiment_config_from_payload(
        experiment_config_factory,
        config_data,
        enable_artifact=False,
    )
    config.enable_artifact = False
    config.enable_reducer = False
    target_roots = [] if args.ignore_roots else [
        finding.get("root_cause", "unknown") for finding in original_findings
    ]
    reduced = reduce_case_func(
        case,
        backends=backends,
        config=config,
        target_kinds=[finding.get("kind", "") for finding in original_findings],
        target_roots=target_roots,
        target_suspicious_backends=[
            finding.get("suspicious_backends", []) for finding in original_findings
        ],
    )
    result = run_loaded_case_func(reduced, backends=backends, config=config)
    dump_json_func(reduced.to_dict(), bug_dir / "reduced_case.json")
    write_reduced_reproducer_func(bug_dir, backends)
    print(f"original rows={len(case.tables[0].rows)} ops={len(case.program.operations)}")
    print(f"reduced rows={len(reduced.tables[0].rows)} ops={len(reduced.program.operations)}")
    print(f"status={result['status']} bug_dir={result.get('bug_dir', '')}")
    return 0


def cmd_replay_fixture_impl(
    args: argparse.Namespace,
    *,
    ensure_dirs_func: Callable[[], None],
    load_fixture_replay_spec_func: Callable[[str], dict[str, Any]],
    resolve_fixture_replay_path_func: Callable[[argparse.Namespace], Path],
    build_fixture_replay_case_func: Callable[[dict[str, Any], Path], Any],
    parse_backends_func: Callable[[str], list[str]],
    resolve_target_backends_func: Callable[..., list[str]],
    parse_experiment_meta_func: Callable[[Any], dict[str, Any]],
    normalize_experiment_meta_func: Callable[[dict[str, Any]], dict[str, Any]],
    replay_bug_enabled_by_default_func: Callable[[str], bool],
    registered_experiment_meta_defaults_func: Callable[..., dict[str, Any]],
    merge_experiment_meta_func: Callable[..., dict[str, Any]],
    experiment_config_factory: Callable[..., Any],
    parse_adaptive_components_func: Callable[[Any], set[str]],
    adaptive_component_config_func: Callable[[set[str]], dict[str, bool]],
    run_loaded_case_func: Callable[..., dict[str, Any]],
    fixture_artifact_budget_allows_func: Callable[[int | None], bool],
    describe_targets_func: Callable[[list[str]], list[dict[str, Any]]],
    fixture_replay_run_id_func: Callable[[str], str],
    runs_dir: Path,
    jsonl_writer_factory: Callable[..., Any],
    compact_log_row_func: Callable[[dict[str, Any], str], dict[str, Any]],
    target_context_func: Callable[[list[str]], Any],
    experiment_manifest_path_func: Callable[[], Path],
    resolved_run_semantics_func: Callable[..., dict[str, Any]],
    catalog_preset_metadata_func: Callable[..., dict[str, Any]],
    configured_guidance_targets_func: Callable[[Any], list[str]],
    describe_operation_combo_func: Callable[[list[dict[str, Any]]], Any],
    dump_json_func: Callable[..., None],
    run_meta_path_func: Callable[[Path], Path],
    record_run_journal_func: Callable[..., tuple[Path, Path]],
    reports_dir: Path,
    utc_now_func: Callable[[], str],
) -> int:
    ensure_dirs_func()
    spec = load_fixture_replay_spec_func(args.spec)
    fixture_path = resolve_fixture_replay_path_func(args)
    case = build_fixture_replay_case_func(spec, fixture_path)
    suite = str(args.target_suite or spec.get("target_suite") or "core")
    backends = parse_backends_func(args.backends) if args.backends else resolve_target_backends_func(None, suite)
    evidence_mode = str(args.evidence_mode)
    known_bug_id = str(args.known_bug_id or spec.get("known_bug_id") or "")
    target_version = str(args.target_version or spec.get("target_version") or "")
    run_theme = str(args.run_theme or f"{evidence_mode}-fixture:{known_bug_id or case.case_id}")
    paper_notes = str(args.paper_notes or spec.get("paper_notes") or "")
    explicit_experiment_meta = parse_experiment_meta_func(getattr(args, "experiment_meta", None))
    experiment_meta = normalize_experiment_meta_func(explicit_experiment_meta)
    if replay_bug_enabled_by_default_func(evidence_mode):
        experiment_defaults = registered_experiment_meta_defaults_func(
            evidence_mode=evidence_mode,
            known_bug_id=known_bug_id,
            target_suite=suite,
            target_version=target_version,
            include_pending_historical=True,
        )
        if experiment_defaults:
            experiment_meta = merge_experiment_meta_func(experiment_defaults, explicit_experiment_meta)
    config = experiment_config_factory(
        enable_replay_bug=replay_bug_enabled_by_default_func(evidence_mode),
        enable_artifact=not args.disable_artifact,
        compress_run_log=not args.no_compress_run_log,
        artifact_limit=args.artifact_limit,
        log_level=args.log_level,
    )
    disabled_components = parse_adaptive_components_func(getattr(args, "disable_adaptive_components", ""))
    adaptive_components = adaptive_component_config_func(disabled_components)
    started = time.perf_counter()
    row = run_loaded_case_func(
        case,
        backends=backends,
        config=config,
        save_artifact=not args.disable_artifact and fixture_artifact_budget_allows_func(args.artifact_limit),
        target_specs=describe_targets_func(backends),
    )
    elapsed_s = time.perf_counter() - started
    row["case_index"] = 0
    row["elapsed_s"] = round(elapsed_s, 6)
    row["candidate_source"] = "fixture"
    row["seed_lineage"] = {
        "root_seed": case.seed,
        "parent_seed": None,
        "parent_case_id": "",
        "mutation_seed": None,
        "depth": 0,
    }
    row["mutation"] = {"operator": "fixture", "detail": "fixture_replay", "changed": False}
    row["operation_combo"] = describe_operation_combo_func(case.program.operations)
    row["preflight"] = {
        "valid": True,
        "repaired": False,
        "fallback_used": False,
        "errors_before": [],
        "errors_after": [],
    }
    row["guidance"] = {
        "score": 0.0,
        "features": [],
        "matched_targets": [],
        "candidate_count": 1,
    }
    row["candidate_seed_start"] = case.seed
    row["candidate_pool_size"] = 1
    raw_new_behavior, signal_new_behavior = _apply_fixture_replay_new_behavior_flags(row, config)
    row["stored_in_feedback_corpus"] = False
    row["feedback_corpus_persisted"] = False
    row["source_reward"] = None
    row["source_scheduler"] = []

    run_id = fixture_replay_run_id_func(known_bug_id or case.case_id)
    suffix = ".jsonl.gz" if config.compress_run_log else ".jsonl"
    run_file = runs_dir / f"{run_id}{suffix}"
    with jsonl_writer_factory(run_file, compresslevel=1) as writer:
        writer.write(compact_log_row_func(row, config.log_level))

    backend_context = target_context_func(backends)
    manifest_path = experiment_manifest_path_func()
    run_identity = {
        "target_suite": suite,
        "backends": backends,
        "preset": "fixture_replay",
        "seed": case.seed,
        "evidence_mode": evidence_mode,
        "known_bug_id": known_bug_id,
        "target_version": target_version,
        "experiment_meta": experiment_meta,
    }
    run_semantics = resolved_run_semantics_func(run_identity, experiment_meta)
    preset_metadata = catalog_preset_metadata_func(
        "fixture_replay",
        base_preset=str(run_semantics.get("base_preset", "") or ""),
        overlays=list(run_semantics.get("overlays", []) or []),
    )
    configured_guidance_targets = list(config.guidance_targets)
    configured_effective_guidance_targets = configured_guidance_targets_func(config)
    run_payload = {
        "target_suite": suite,
        "backends": backends,
        "preset": "fixture_replay",
        "seed": case.seed,
        "evidence_mode": evidence_mode,
        "known_bug_id": known_bug_id,
        "target_version": target_version,
        "batch_index": None,
        "schedule_arm_id": "",
        "estimated_cost": "",
        "worker_thread_limit": "",
        "closed_loop_state_present": False,
        "adaptive_components": dict(adaptive_components),
        "disabled_adaptive_components": sorted(disabled_components),
        "experiment_meta": experiment_meta,
        "matrix_id": run_semantics["matrix_id"],
        "matrix_title": run_semantics["matrix_title"],
        "comparison_group": run_semantics["comparison_group"],
        "purpose": run_semantics["purpose"],
        "counts_as_real_bugs": run_semantics["counts_as_real_bugs"],
        "rq_tags": list(run_semantics["rq_tags"]),
        "analysis_tags": list(run_semantics["analysis_tags"]),
        "variant_id": run_semantics["variant_id"],
        "variant_title": run_semantics["variant_title"],
        "base_preset": run_semantics["base_preset"],
        "comparison_role": run_semantics["comparison_role"],
        "canonical_comparison_role": run_semantics["canonical_comparison_role"],
        "component_focus": run_semantics["component_focus"],
        "overlays": list(run_semantics["overlays"]),
        "semantic_focus_families": list(run_semantics["semantic_focus_families"]),
        "semantic_focus_signals": list(run_semantics["semantic_focus_signals"]),
        "factors": dict(run_semantics["factors"]),
        "oracle_profile": run_semantics["oracle_profile"],
        "scope_kind": run_semantics["scope_kind"],
        "preset_metadata": preset_metadata,
        "configured_guidance_targets": configured_guidance_targets,
        "configured_effective_guidance_targets": configured_effective_guidance_targets,
        "configured_semantic_focus_families": list(config.semantic_focus_families),
        "configured_semantic_focus_signals": list(config.semantic_focus_signals),
        "fixture_spec": str(args.spec),
        "fixture_path": str(fixture_path),
        "fixture_sha256": case.metadata.get("fixture_sha256", ""),
        "run_file": str(run_file),
        "report": "",
        "csv": "",
    }
    manifest = {
        "created_at": utc_now_func(),
        "presets": ["fixture_replay"],
        "seeds": [case.seed],
        "cases": 1,
        "duration_s": None,
        "evidence_mode": evidence_mode,
        "known_bug_id": known_bug_id,
        "target_version": target_version,
        "run_theme": run_theme,
        "paper_notes": paper_notes,
        "experiment_meta": experiment_meta,
        "backends": backends,
        "target_suite": suite,
        "target_suites": [suite],
        "backends_by_suite": {suite: backends},
        "targets": backend_context.target_dicts(),
        "common_capabilities": list(backend_context.common_capabilities),
        "target_context": backend_context.to_dict(),
        "log_level": config.log_level,
        "compress_run_log": config.compress_run_log,
        "metamorphic_variant_limit": config.metamorphic_variant_limit,
        "replay_bug_policy": {
            "enable_replay_bug": config.enable_replay_bug,
            "source_issues": list(config.replay_bug_source_issues),
        },
        "fixture_spec": str(args.spec),
        "fixture_path": str(fixture_path),
        "fixture_sha256": case.metadata.get("fixture_sha256", ""),
        "jobs": 1,
        "parallelism": {"requested_jobs": 1, "worker_count": 1},
        "schedule": "fixture_replay",
        "adaptive_methodology": {
            "components": dict(adaptive_components),
            "disabled_components": sorted(disabled_components),
        },
        "runs": [run_payload],
    }
    meta = {
        "run_id": run_id,
        "status": "completed",
        "run_file": str(run_file),
        "manifest_file": str(manifest_path),
        "case_log_file": "",
        "checkpoint_file": "",
        "requested_cases": 1,
        "executed_cases": 1,
        "duration_s": None,
        "elapsed_s": elapsed_s,
        "throughput_cases_s": 1 / elapsed_s if elapsed_s else 0.0,
        "findings": len(row.get("findings", [])),
        "new_behavior_cases": int(raw_new_behavior),
        "signal_new_behavior_cases": int(signal_new_behavior),
        "saved_artifacts": int(bool(row.get("bug_dir"))),
        "preflight": {"fixture_cases": 1},
        "quality_oracles": {},
        "seed": case.seed,
        "next_seed": case.seed + 1,
        "backends": backends,
        "targets": backend_context.target_dicts(),
        "common_capabilities": list(backend_context.common_capabilities),
        "target_context": backend_context.to_dict(),
        "config": config.to_dict(),
        "environment": row.get("environment", {}),
        "log_level": config.log_level,
        "target_suite": suite,
        "preset": "fixture_replay",
        "evidence_mode": evidence_mode,
        "known_bug_id": known_bug_id,
        "target_version": target_version,
        "run_theme": run_theme,
        "paper_notes": paper_notes,
        "experiment_meta": experiment_meta,
        "fixture_spec": str(args.spec),
        "fixture_path": str(fixture_path),
        "fixture_sha256": case.metadata.get("fixture_sha256", ""),
        "updated_at": utc_now_func(),
    }
    dump_json_func(meta, run_meta_path_func(run_file))
    dump_json_func(manifest, manifest_path)
    journal_path, journal_md = record_run_journal_func(
        run_file,
        context={
            "command": "replay-fixture",
            "theme": run_theme,
            "notes": paper_notes,
            "evidence_mode": evidence_mode,
            "known_bug_id": known_bug_id,
            "target_version": target_version,
            "target_suite": suite,
            "preset": "fixture_replay",
            "seed": case.seed,
            "backends": backends,
            "manifest_file": str(manifest_path),
            "experiment_meta": experiment_meta,
        },
        journal_file=reports_dir / "paper-run-journal.jsonl",
    )

    print(f"status={row['status']}")
    print(f"run_file={run_file}")
    print(f"meta_file={run_meta_path_func(run_file)}")
    print(f"experiment manifest: {manifest_path}")
    print(f"paper_run_journal={journal_path}")
    print(f"paper_run_journal_markdown={journal_md}")
    for finding in row.get("findings", []):
        print(
            f"- {finding.get('kind', '')} root={finding.get('root_cause', 'unknown')} "
            f"verdict={finding.get('triage_verdict', '')} "
            f"suspicious={','.join(finding.get('suspicious_backends', []) or [])}"
        )
    return 0


def cmd_historical_status_impl(
    args: argparse.Namespace,
    *,
    list_historical_bugs_func: Callable[..., list[Any]],
    historical_status_row_func: Callable[[object], dict[str, Any]],
) -> int:
    specs = list_historical_bugs_func(include_pending=bool(args.include_pending))
    rows = [historical_status_row_func(spec) for spec in specs]
    if args.json:
        print(json.dumps({"historical_bugs": rows}, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    print("bug_id\tstatus\tcounted\treplay_kind\ttarget_version\tfixture_env")
    for row in rows:
        print(
            "\t".join(
                [
                    row["bug_id"],
                    row["status"],
                    "yes" if row["counted"] else "no",
                    row["replay_kind"],
                    row["target_version"],
                    row["fixture_env_status"],
                ]
            )
        )
    return 0
