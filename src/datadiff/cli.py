from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
from collections import Counter
from pathlib import Path

from datadiff.ablation_audit import analyze_ablation_audit
from datadiff.classification_oracle import classify_finding
from datadiff.config import ExperimentConfig
from datadiff.dsl import Case
from datadiff.experiment_analysis import analyze_experiment
from datadiff.fixture_replay import build_fixture_replay_case, load_fixture_replay_spec
from datadiff.historical import list_historical_bugs
from datadiff.normalizer import NormalizedResult
from datadiff.guidance import parse_guidance_targets
from datadiff.operation_combo import classify_operation_combo
from datadiff.oracle import evaluate_case
from datadiff.pattern_analysis import analyze_pattern_variants
from datadiff.reporter import latest_run_file, write_experiment_summary, write_report
from datadiff.reducer import reduce_case
from datadiff.run_journal import (
    append_run_journal_entries,
    build_run_journal_entry,
    record_run_journal,
    write_run_journal_markdown,
)
from datadiff.runner import _compact_log_row, run_fuzz, run_loaded_case
from datadiff.scheduler import AdaptiveBudgetScheduler, AdaptiveScheduleConfig, summarize_batch_run
from datadiff.seeded_analysis import analyze_seeded_sensitivity
from datadiff.targets import (
    TARGETS,
    TARGET_SUITES,
    common_capabilities,
    describe_targets,
    list_target_suites,
    parse_backend_names,
    resolve_target_backends,
    target_capability_matrix,
)
from datadiff.triage import (
    build_triage_report,
    supports_standalone_reproducer,
    write_standalone_reproducer,
    write_triage_artifact,
)
from datadiff.util import (
    BUGS_DIR,
    CORPUS_DIR,
    JsonlWriter,
    REPORTS_DIR,
    RUNS_DIR,
    dump_json,
    ensure_dirs,
    load_json,
    parse_duration,
    read_jsonl,
    run_meta_path,
    slugify,
    utc_now,
)

LIVE_BUGHUNT_TARGETS = [
    "common_workflow",
    "operation_combo",
    "topk",
    "join",
    "join_null_truth_filter",
    "join_null_key_topk",
    "groupby",
    "mutate",
    "filter",
    "truth_filter",
    "nulls",
    "aggregation",
    "global_aggregation",
    "sort_limit",
    "topk_resort",
    "ordered_groupby_sort",
    "join_ordered_agg_topk",
    "global_null_aggregate",
    "string_count_groupby",
    "wide_offset_topk",
    "empty_filter_groupby",
    "expressions",
]

LIVE_ARROW_TARGETS = [
    "common_workflow",
    "operation_combo",
    "join",
    "join_null_truth_filter",
    "join_null_key_topk",
    "groupby",
    "mutate",
    "filter",
    "truth_filter",
    "strings",
    "casts",
    "nulls",
    "aggregation",
    "global_aggregation",
    "sort_limit",
    "topk",
    "topk_resort",
    "ordered_groupby_sort",
    "join_ordered_agg_topk",
    "global_null_aggregate",
    "string_count_groupby",
    "wide_offset_topk",
    "empty_filter_groupby",
    "expressions",
]

LIVE_POLARS_LAZY_TARGETS = [
    "common_workflow",
    "operation_combo",
    "join",
    "join_null_truth_filter",
    "join_null_key_topk",
    "filter",
    "truth_filter",
    "mutate",
    "groupby",
    "strings",
    "casts",
    "nulls",
    "global_aggregation",
    "sort_limit",
    "topk",
    "topk_resort",
    "ordered_groupby_sort",
    "join_ordered_agg_topk",
    "global_null_aggregate",
    "string_count_groupby",
    "empty_filter_groupby",
    "expressions",
]

LIVE_EMBEDDED_SQL_TARGETS = [
    "common_workflow",
    "operation_combo",
    "join",
    "join_null_truth_filter",
    "join_null_key_topk",
    "filter",
    "truth_filter",
    "mutate",
    "groupby",
    "nulls",
    "aggregation",
    "global_aggregation",
    "sort_limit",
    "topk",
    "topk_resort",
    "ordered_groupby_sort",
    "join_ordered_agg_topk",
    "global_null_aggregate",
    "string_count_groupby",
    "wide_offset_topk",
    "empty_filter_groupby",
    "casts",
    "expressions",
]

LIVE_CROSS_FAMILY_TARGETS = [
    "common_workflow",
    "operation_combo",
    "join",
    "join_null_truth_filter",
    "join_null_key_topk",
    "filter",
    "truth_filter",
    "mutate",
    "groupby",
    "strings",
    "casts",
    "nulls",
    "aggregation",
    "global_aggregation",
    "sort_limit",
    "topk",
    "topk_resort",
    "ordered_groupby_sort",
    "join_ordered_agg_topk",
    "global_null_aggregate",
    "string_count_groupby",
    "wide_offset_topk",
    "empty_filter_groupby",
    "expressions",
]


def _parse_backends(value: str) -> list[str]:
    return parse_backend_names(value)


def _parse_jobs(value: str) -> int | str:
    text = str(value).strip().lower()
    if text == "auto":
        return "auto"
    jobs = int(text)
    if jobs < 1:
        raise argparse.ArgumentTypeError("--jobs must be a positive integer or 'auto'")
    return jobs


def _resolve_run_backends(args: argparse.Namespace) -> list[str]:
    return resolve_target_backends(
        getattr(args, "backends", None),
        target_suite=getattr(args, "target_suite", "core"),
    )


def _config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        enable_type_aware_generation=not args.disable_type_aware_generation,
        enable_normalizer=not args.disable_normalizer,
        enable_differential_oracle=not args.disable_differential_oracle,
        enable_metamorphic_oracle=args.enable_metamorphic_oracle,
        enable_feedback=not args.disable_feedback,
        enable_reducer=args.enable_reducer,
        enable_artifact=not args.disable_artifact,
        enable_preflight_validation=not args.disable_preflight_validation,
        enable_preflight_repair=not args.disable_preflight_repair,
        persist_feedback_corpus=args.persist_feedback_corpus,
        feedback_persist_limit=max(0, int(getattr(args, "feedback_persist_limit", 4096))),
        enable_local_source_scheduler=bool(getattr(args, "enable_local_source_scheduler", False)),
        local_source_exploration_weight=max(
            0.0, float(getattr(args, "local_source_exploration_weight", 0.5))
        ),
        compress_run_log=not args.no_compress_run_log,
        artifact_limit=args.artifact_limit,
        oracle_mode="both" if args.enable_metamorphic_oracle else "differential",
        generator_profile=args.profile,
        guidance_strategy=getattr(args, "strategy", "random"),
        guidance_candidate_pool=max(1, int(getattr(args, "candidate_pool", 1))),
        guidance_targets=parse_guidance_targets(getattr(args, "targets", "")),
        metamorphic_variant_limit=max(0, int(getattr(args, "metamorphic_variant_limit", 4))),
        log_level=getattr(args, "log_level", "compact"),
    )


def add_ablation_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--disable-type-aware-generation", action="store_true")
    parser.add_argument("--disable-normalizer", action="store_true")
    parser.add_argument("--disable-differential-oracle", action="store_true")
    parser.add_argument("--enable-metamorphic-oracle", action="store_true")
    parser.add_argument("--disable-feedback", action="store_true")
    parser.add_argument("--enable-reducer", action="store_true")
    parser.add_argument("--disable-artifact", action="store_true")
    parser.add_argument("--disable-preflight-validation", action="store_true")
    parser.add_argument("--disable-preflight-repair", action="store_true")
    parser.add_argument("--persist-feedback-corpus", action="store_true")
    parser.add_argument(
        "--feedback-persist-limit",
        type=int,
        default=4096,
        help="maximum interesting feedback cases to write to corpus/interesting for this run",
    )
    parser.add_argument(
        "--enable-local-source-scheduler",
        action="store_true",
        help="adaptively choose between generated candidates and feedback mutations within a run",
    )
    parser.add_argument(
        "--local-source-exploration-weight",
        type=float,
        default=0.5,
        help="exploration weight for the within-run generated-vs-feedback source scheduler",
    )
    parser.add_argument("--no-compress-run-log", action="store_true")
    parser.add_argument(
        "--artifact-limit",
        type=int,
        default=None,
        help="maximum bug artifact directories to write for this run; 0 keeps only run-log finding summaries",
    )
    parser.add_argument(
        "--metamorphic-variant-limit",
        type=int,
        default=4,
        help="maximum metamorphic variants to execute per base case",
    )
    parser.add_argument(
        "--log-level",
        choices=["full", "compact", "minimal"],
        default="compact",
        help="run JSONL detail level; compact keeps full details only for finding rows",
    )


def add_guidance_flags(
    parser: argparse.ArgumentParser,
    *,
    default_strategy: str,
    default_candidate_pool: int,
) -> None:
    parser.add_argument("--strategy", choices=["random", "guided"], default=default_strategy)
    parser.add_argument(
        "--candidate-pool",
        type=int,
        default=default_candidate_pool,
        help="number of cheap generated candidates scored before executing one case",
    )
    parser.add_argument(
        "--targets",
        default="",
        help=(
            "comma-separated guided targets such as groupby,filter,mutate,sort_limit,"
            "nulls,strings,numeric,edge_float,aggregation,join,expressions,casts"
        ),
    )


def add_target_suite_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--target-suite",
        choices=sorted(TARGET_SUITES),
        default="core",
        help="backend target suite to execute when --backends is not provided",
    )
    parser.add_argument(
        "--backends",
        default=None,
        help="explicit comma-separated backend targets; overrides --target-suite",
    )


def add_paper_journal_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run-theme",
        default="",
        help="paper-facing run theme recorded in reports/paper-run-journal.*",
    )
    parser.add_argument(
        "--paper-notes",
        default="",
        help="short paper-facing notes recorded with the run journal entry",
    )
    parser.add_argument(
        "--skip-paper-journal",
        action="store_true",
        help="skip paper-run journal writes for IO-sensitive long runs; summarize later from the manifest/run log",
    )


def cmd_init(args: argparse.Namespace) -> int:
    ensure_dirs()
    print("Initialized DataDiffFuzz")
    print(f"runs:    {RUNS_DIR}")
    print(f"bugs:    {BUGS_DIR}")
    print(f"reports: {REPORTS_DIR}")
    return 0


def cmd_fuzz(args: argparse.Namespace) -> int:
    out = run_fuzz(
        cases=args.cases,
        seed=args.seed,
        backends=_resolve_run_backends(args),
        config=_config_from_args(args),
        duration_s=parse_duration(args.duration),
    )
    print(f"run log written: {out}")
    if not getattr(args, "skip_paper_journal", False):
        journal_path, journal_md = _record_cli_run_journal(out, args, command="fuzz")
        print(f"paper run journal: {journal_path}")
        if journal_md is not None:
            print(f"paper run journal markdown: {journal_md}")
    return 0


def _record_cli_run_journal(run_file: Path, args: argparse.Namespace, *, command: str) -> tuple[Path, Path | None]:
    context = {
        "command": command,
        "theme": _single_run_theme(args, command=command),
        "notes": str(getattr(args, "paper_notes", "") or ""),
        "evidence_mode": "live",
        "target_suite": str(getattr(args, "target_suite", "") or ""),
        "profile": str(getattr(args, "profile", "") or ""),
        "seed": getattr(args, "seed", ""),
        "backends": _resolve_run_backends(args),
    }
    return record_run_journal(
        run_file,
        context=context,
        journal_file=REPORTS_DIR / "paper-run-journal.jsonl",
    )


def _single_run_theme(args: argparse.Namespace, *, command: str) -> str:
    explicit = str(getattr(args, "run_theme", "") or "").strip()
    if explicit:
        return explicit
    target_suite = str(getattr(args, "target_suite", "") or "core")
    profile = str(getattr(args, "profile", "") or "common")
    seed = getattr(args, "seed", "")
    return f"{command}:{target_suite}:{profile}:seed{seed}"


def _print_longrun_progress(snapshot: dict) -> None:
    print(
        "progress "
        f"cases={snapshot['executed_cases']} "
        f"elapsed_s={snapshot['elapsed_s']:.1f} "
        f"cases_s={snapshot['throughput_cases_s']:.3f} "
        f"findings={snapshot['findings']} "
        f"next_seed={snapshot['next_seed']}",
        flush=True,
    )


def cmd_longrun(args: argparse.Namespace) -> int:
    case_log_file = Path(args.case_log) if args.case_log else None
    out = run_fuzz(
        cases=args.cases,
        seed=args.seed,
        backends=_resolve_run_backends(args),
        config=_config_from_args(args),
        duration_s=parse_duration(args.duration),
        save_cases=args.save_cases and not args.no_save_cases,
        case_log_file=case_log_file,
        checkpoint_interval_s=parse_duration(args.checkpoint_interval),
        progress_interval_s=parse_duration(args.progress_interval),
        progress_callback=_print_longrun_progress if not args.quiet else None,
    )
    print(f"run log written: {out}")
    if not getattr(args, "skip_paper_journal", False):
        journal_path, journal_md = _record_cli_run_journal(out, args, command="longrun")
        print(f"paper run journal: {journal_path}")
        if journal_md is not None:
            print(f"paper run journal markdown: {journal_md}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    run_file = Path(args.run_file) if args.run_file else latest_run_file()
    md_path, csv_path = write_report(run_file, csv_limit=args.csv_limit)
    print(f"markdown report: {md_path}")
    print(f"csv findings:    {csv_path}")
    return 0


def cmd_experiment_summary(args: argparse.Namespace) -> int:
    manifest_file = Path(args.manifest) if args.manifest else None
    md_path, csv_path = write_experiment_summary(manifest_file, refresh=bool(getattr(args, "refresh", False)))
    aggregate_csv_path = md_path.with_name(f"{md_path.stem}-aggregates.csv")
    print(f"markdown summary: {md_path}")
    print(f"csv summary:      {csv_path}")
    if aggregate_csv_path.exists():
        print(f"aggregate csv:    {aggregate_csv_path}")
    return 0


def cmd_analyze_experiment(args: argparse.Namespace) -> int:
    manifest_file = Path(args.manifest) if args.manifest else None
    compare_presets = _parse_presets(args.compare_presets) if args.compare_presets else None
    md_path, csv_path = analyze_experiment(
        manifest_file,
        baseline_preset=args.baseline_preset,
        compare_presets=compare_presets,
        refresh=bool(getattr(args, "refresh", False)),
    )
    print(f"analysis markdown: {md_path}")
    print(f"analysis csv:      {csv_path}")
    return 0


def cmd_analyze_seeded_sensitivity(args: argparse.Namespace) -> int:
    manifest_file = Path(args.manifest) if args.manifest else None
    md_path, csv_path = analyze_seeded_sensitivity(manifest_file)
    print(f"seeded sensitivity markdown: {md_path}")
    print(f"seeded sensitivity csv:      {csv_path}")
    return 0


def cmd_analyze_ablation_audit(args: argparse.Namespace) -> int:
    manifest_file = Path(args.manifest) if args.manifest else None
    trusted_presets = _parse_presets(args.trusted_presets) if args.trusted_presets else None
    ablation_presets = _parse_presets(args.ablation_presets) if args.ablation_presets else None
    md_path, csv_path = analyze_ablation_audit(
        manifest_file,
        trusted_presets=trusted_presets,
        ablation_presets=ablation_presets,
        refresh=bool(getattr(args, "refresh", False)),
    )
    print(f"ablation audit markdown: {md_path}")
    print(f"ablation audit csv:      {csv_path}")
    return 0


def cmd_analyze_pattern_variants(args: argparse.Namespace) -> int:
    manifest_file = Path(args.manifest) if args.manifest else None
    md_path, csv_path = analyze_pattern_variants(manifest_file, pattern=args.pattern)
    print(f"pattern variant markdown: {md_path}")
    print(f"pattern variant csv:      {csv_path}")
    return 0


def cmd_targets(args: argparse.Namespace) -> int:
    payload = {
        "suites": list_target_suites(),
        "targets": [target.to_dict() for target in sorted(TARGETS.values(), key=lambda item: item.name)],
        "capability_matrix": target_capability_matrix(),
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    print("target suites:")
    for suite in payload["suites"]:
        print(
            f"- {suite['suite']}: "
            f"backends={','.join(suite['backends'])} "
            f"families={','.join(suite['families'])} "
            f"common_capabilities={len(suite['common_capabilities'])}"
        )
    print("targets:")
    for target in sorted(TARGETS.values(), key=lambda item: item.name):
        print(
            f"- {target.name}: family={target.family} "
            f"layer={target.layer} status={target.status} "
            f"capabilities={len(target.capabilities)} adapter={target.adapter}"
        )
    return 0


def cmd_prune_corpus(args: argparse.Namespace) -> int:
    keep = max(0, int(args.keep))
    interesting_dir = CORPUS_DIR / "interesting"
    files = sorted(
        [path for path in interesting_dir.glob("*.json") if path.is_file()],
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )
    victims = files[keep:]
    bytes_to_free = sum(path.stat().st_size for path in victims)
    print(f"corpus_interesting={interesting_dir}")
    print(f"total_files={len(files)}")
    print(f"keep_latest={keep}")
    print(f"delete_candidates={len(victims)}")
    print(f"bytes_to_free={bytes_to_free}")
    if not args.yes:
        print("dry_run=true")
        print("rerun with --yes to delete candidates")
        return 0

    for path in victims:
        path.unlink()
    print("dry_run=false")
    print(f"deleted={len(victims)}")
    return 0


def cmd_show_bugs(args: argparse.Namespace) -> int:
    run_file = Path(args.run_file) if args.run_file else latest_run_file()
    rows = read_jsonl(run_file)
    shown = 0
    for row in rows:
        if not row.get("findings"):
            continue
        print("=" * 88)
        print(f"{row['case']['case_id']} seed={row['case']['seed']} status={row['status']}")
        print(f"ops={[op['op'] for op in row['case']['program']['operations']]}")
        print(f"bug_dir={row.get('bug_dir', '')}")
        for backend, norm in row.get("normalized", {}).items():
            print(
                f"  {backend}: {norm.get('status')} "
                f"rows={len(norm.get('rows', []))} cols={norm.get('columns', [])}"
            )
        for finding in row.get("findings", []):
            print(
                f"- [{finding['severity']}] {finding['kind']} "
                f"root={finding.get('root_cause', 'unknown')} "
                f"oracle={finding.get('oracle', 'unknown')} "
                f"triage={finding.get('triage_verdict', 'unclassified')} "
                f"suspicious={finding.get('suspicious_backends', [])}: {finding['evidence']}"
            )
            if finding.get("false_positive_reason"):
                print(f"  false_positive_reason={finding['false_positive_reason']}")
        shown += 1
        if shown >= args.limit:
            break
    if shown == 0:
        print("No bugs in selected run.")
    return 0


def cmd_classify_run(args: argparse.Namespace) -> int:
    run_file = Path(args.run_file) if args.run_file else latest_run_file()
    verdicts: Counter[str] = Counter()
    candidate_bug_families: Counter[str] = Counter()
    false_positive_reasons: Counter[str] = Counter()
    examples: dict[str, list[dict]] = {}
    cache: dict[tuple, dict] = {}

    for row in read_jsonl(run_file):
        _classify_run_row(
            row,
            verdicts,
            candidate_bug_families,
            false_positive_reasons,
            examples,
            args.limit,
            cache,
            refresh=bool(getattr(args, "refresh", False)),
        )

    print(f"run_file={run_file}")
    if getattr(args, "refresh", False):
        print("refresh=true")
    print("triage verdicts:")
    if not verdicts:
        print("- none")
    for verdict, count in verdicts.most_common():
        print(f"- {verdict}: {count}")
    print("candidate bug families:")
    if not candidate_bug_families:
        print("- none")
    for family, count in candidate_bug_families.most_common():
        print(f"- {family}: {count}")
    print("false positive reasons:")
    if not false_positive_reasons:
        print("- none")
    for reason, count in false_positive_reasons.most_common():
        print(f"- {reason}: {count}")
    for verdict, items in examples.items():
        print(f"examples[{verdict}]:")
        for item in items:
            print(
                f"- {item['case_id']} seed={item['seed']} kind={item['kind']} "
                f"root={item['root']} suspicious={item['suspicious']} signature={item['signature']} "
                f"evidence={item['evidence']}"
            )
    return 0


def _classify_run_row(
    row: dict,
    verdicts: Counter[str],
    candidate_bug_families: Counter[str],
    false_positive_reasons: Counter[str],
    examples: dict[str, list[dict]],
    limit: int,
    cache: dict[tuple, dict],
    *,
    refresh: bool = False,
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
        if verdict == "candidate_implementation_bug" and not classification.get("false_positive"):
            candidate_finding = dict(finding)
            candidate_finding["triage_verdict"] = verdict
            candidate_finding["false_positive"] = False
            candidate_findings.append(candidate_finding)
        if classification.get("false_positive_reason"):
            false_positive_reasons[classification["false_positive_reason"]] += 1
        if len(examples.setdefault(verdict, [])) < limit:
            examples[verdict].append(
                {
                    "case_id": row["case"]["case_id"],
                    "seed": row["case"]["seed"],
                    "kind": finding.get("kind", ""),
                    "root": finding.get("root_cause", "unknown"),
                    "suspicious": finding.get("suspicious_backends", []),
                    "signature": finding.get("signature", ""),
                    "evidence": classification.get("evidence", ""),
                }
            )
    candidate_bug_families.update(_candidate_bug_family_keys(candidate_findings))


def _candidate_bug_family_key(finding: dict) -> str:
    return next(iter(_candidate_bug_family_keys([finding])), "")


def _candidate_bug_family_keys(findings: list[dict]) -> Counter:
    keys: Counter = Counter()
    root_by_suspicious: dict[str, str] = {}
    for finding in findings:
        if not _is_candidate_bug_finding(finding):
            continue
        root = str(finding.get("root_cause", "unknown"))
        if root.startswith("metamorphic_"):
            continue
        suspicious = _suspicious_key(finding)
        root_by_suspicious.setdefault(suspicious, root)
    for finding in findings:
        if not _is_candidate_bug_finding(finding):
            continue
        root = str(finding.get("root_cause", "unknown"))
        suspicious = _suspicious_key(finding)
        if root.startswith("metamorphic_") and suspicious in root_by_suspicious:
            root = root_by_suspicious[suspicious]
        keys[f"{root}@{suspicious}"] += 1
    return keys


def _is_candidate_bug_finding(finding: dict) -> bool:
    return finding.get("triage_verdict") == "candidate_implementation_bug" and not finding.get("false_positive")


def _suspicious_key(finding: dict) -> str:
    return ",".join(sorted(finding.get("suspicious_backends", []) or [])) or "unknown"


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
        tuple(op.get("op", "") for op in program.get("operations", [])),
        _case_has_special_float_data(tables),
        _case_has_null_data(tables),
        _case_has_non_ascii_data(tables),
    )


def _case_has_special_float_data(tables: list[dict]) -> bool:
    # JSONL stores NaN/Infinity as non-standard JSON tokens; Python json restores
    # them as floats, so this detects old and new run files.
    import math

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
    out = {}
    for backend, data in mapping.items():
        out[backend] = NormalizedResult(
            backend=data.get("backend", backend),
            status=data.get("status", "unknown"),
            columns=data.get("columns", []),
            rows=data.get("rows", []),
            error_type=data.get("error_type", ""),
            error=data.get("error", ""),
        )
    return out


def cmd_reproduce(args: argparse.Namespace) -> int:
    bug_dir = Path(args.bug)
    repro = bug_dir / "reproduce.py"
    if not repro.exists():
        raise FileNotFoundError(repro)
    if args.print_command:
        print(f"Run this command to reproduce:\npython {repro}")
        return 0
    case = Case.from_dict(load_json(bug_dir / "case.json"))
    config_path = bug_dir / "config.json"
    config_data = load_json(config_path) if config_path.exists() else {}
    config = ExperimentConfig(**config_data) if config_data else ExperimentConfig()
    backends = _parse_backends(args.backends) if args.backends else list(load_json(bug_dir / "results.json"))
    result = run_loaded_case(case, backends=backends, config=config, save_artifact=False)
    print(f"status={result['status']}")
    for finding in result["findings"]:
        print(
            f"- {finding['kind']} root={finding.get('root_cause', 'unknown')} "
            f"oracle={finding.get('oracle', 'unknown')} signature={finding.get('signature', '')}"
        )
    return 0


def cmd_validate_artifact(args: argparse.Namespace) -> int:
    bug_dir = Path(args.bug)
    case = Case.from_dict(load_json(bug_dir / "case.json"))
    original_findings = load_json(bug_dir / "findings.json")
    config_path = bug_dir / "config.json"
    config_data = load_json(config_path) if config_path.exists() else {}
    config = ExperimentConfig(**config_data) if config_data else ExperimentConfig()
    backends = _parse_backends(args.backends) if args.backends else list(load_json(bug_dir / "results.json"))
    result = run_loaded_case(case, backends=backends, config=config, save_artifact=False)

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


def cmd_triage_artifact(args: argparse.Namespace) -> int:
    bug_dir = Path(args.bug)
    case = Case.from_dict(load_json(bug_dir / "case.json"))
    original_findings = load_json(bug_dir / "findings.json")
    config_data = _load_artifact_config(bug_dir)
    config = ExperimentConfig(**config_data) if config_data else ExperimentConfig()
    backends = _parse_backends(args.backends) if args.backends else list(load_json(bug_dir / "results.json"))

    triage_case = case
    if args.reduce:
        target_roots = [] if args.reduce_ignore_roots else [
            finding.get("root_cause", "unknown") for finding in original_findings
        ]
        triage_case = reduce_case(
            case,
            backends=backends,
            config=config,
            target_kinds=[finding.get("kind", "") for finding in original_findings],
            target_roots=target_roots,
        )
        dump_json(triage_case.to_dict(), bug_dir / "reduced_case.json")
        _write_reduced_reproducer(bug_dir, backends)

    result = run_loaded_case(triage_case, backends=backends, config=config, save_artifact=False)
    report = build_triage_report(
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
    json_path, md_path = write_triage_artifact(bug_dir, report)
    print(f"verdict={report['verdict']}")
    print(f"paper_status={report['paper_status']}")
    print(f"triage_json={json_path}")
    print(f"triage_md={md_path}")
    if args.standalone_reproducer:
        if supports_standalone_reproducer(report):
            standalone_path = write_standalone_reproducer(bug_dir, report)
            print(f"standalone_reproducer={standalone_path}")
        else:
            print("standalone_reproducer=skipped (no standalone template for this root cause)")
    return 0


def cmd_reduce(args: argparse.Namespace) -> int:
    bug_dir = Path(args.bug)
    case = Case.from_dict(load_json(bug_dir / "case.json"))
    backends = _parse_backends(args.backends)
    original_findings = load_json(bug_dir / "findings.json")
    config_data = _load_artifact_config(bug_dir)
    config = ExperimentConfig(**config_data) if config_data else ExperimentConfig(enable_artifact=False)
    config.enable_artifact = False
    config.enable_reducer = False
    target_roots = [] if args.ignore_roots else [
        finding.get("root_cause", "unknown") for finding in original_findings
    ]
    reduced = reduce_case(
        case,
        backends=backends,
        config=config,
        target_kinds=[finding.get("kind", "") for finding in original_findings],
        target_roots=target_roots,
    )
    result = run_loaded_case(reduced, backends=backends, config=config)
    dump_json(reduced.to_dict(), bug_dir / "reduced_case.json")
    _write_reduced_reproducer(bug_dir, backends)
    print(f"original rows={len(case.tables[0].rows)} ops={len(case.program.operations)}")
    print(f"reduced rows={len(reduced.tables[0].rows)} ops={len(reduced.program.operations)}")
    print(f"status={result['status']} bug_dir={result.get('bug_dir', '')}")
    return 0


def cmd_replay_fixture(args: argparse.Namespace) -> int:
    ensure_dirs()
    spec = load_fixture_replay_spec(args.spec)
    fixture_path = _resolve_fixture_replay_path(args)
    case = build_fixture_replay_case(spec, fixture_path)
    suite = str(args.target_suite or spec.get("target_suite") or "core")
    backends = _parse_backends(args.backends) if args.backends else resolve_target_backends(None, suite)
    evidence_mode = str(args.evidence_mode)
    known_bug_id = str(args.known_bug_id or spec.get("known_bug_id") or "")
    target_version = str(args.target_version or spec.get("target_version") or "")
    run_theme = str(args.run_theme or f"{evidence_mode}-fixture:{known_bug_id or case.case_id}")
    paper_notes = str(args.paper_notes or spec.get("paper_notes") or "")
    config = ExperimentConfig(
        enable_artifact=not args.disable_artifact,
        compress_run_log=not args.no_compress_run_log,
        artifact_limit=args.artifact_limit,
        log_level=args.log_level,
    )
    started = time.perf_counter()
    row = run_loaded_case(
        case,
        backends=backends,
        config=config,
        save_artifact=not args.disable_artifact and _fixture_artifact_budget_allows(args.artifact_limit),
        target_specs=describe_targets(backends),
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
    row["operation_combo"] = classify_operation_combo(case.program.operations)
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
    row["is_new_behavior"] = bool(row.get("findings"))
    row["stored_in_feedback_corpus"] = False
    row["feedback_corpus_persisted"] = False
    row["source_reward"] = None
    row["source_scheduler"] = []

    run_id = _fixture_replay_run_id(known_bug_id or case.case_id)
    suffix = ".jsonl.gz" if config.compress_run_log else ".jsonl"
    run_file = RUNS_DIR / f"{run_id}{suffix}"
    with JsonlWriter(run_file, compresslevel=1) as writer:
        writer.write(_compact_log_row(row, config.log_level))

    meta = {
        "run_id": run_id,
        "status": "completed",
        "run_file": str(run_file),
        "case_log_file": "",
        "checkpoint_file": "",
        "requested_cases": 1,
        "executed_cases": 1,
        "duration_s": None,
        "elapsed_s": elapsed_s,
        "throughput_cases_s": 1 / elapsed_s if elapsed_s else 0.0,
        "findings": len(row.get("findings", [])),
        "new_behavior_cases": int(bool(row.get("findings"))),
        "saved_artifacts": int(bool(row.get("bug_dir"))),
        "preflight": {"fixture_cases": 1},
        "quality_oracles": {},
        "seed": case.seed,
        "next_seed": case.seed + 1,
        "backends": backends,
        "targets": describe_targets(backends),
        "common_capabilities": common_capabilities(backends),
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
        "fixture_spec": str(args.spec),
        "fixture_path": str(fixture_path),
        "fixture_sha256": case.metadata.get("fixture_sha256", ""),
        "updated_at": utc_now(),
    }
    dump_json(meta, run_meta_path(run_file))
    journal_path, journal_md = record_run_journal(
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
        },
        journal_file=REPORTS_DIR / "paper-run-journal.jsonl",
    )

    print(f"status={row['status']}")
    print(f"run_file={run_file}")
    print(f"meta_file={run_meta_path(run_file)}")
    print(f"paper_run_journal={journal_path}")
    print(f"paper_run_journal_markdown={journal_md}")
    for finding in row.get("findings", []):
        print(
            f"- {finding.get('kind', '')} root={finding.get('root_cause', 'unknown')} "
            f"verdict={finding.get('triage_verdict', '')} "
            f"suspicious={','.join(finding.get('suspicious_backends', []) or [])}"
        )
    return 0


def cmd_historical_status(args: argparse.Namespace) -> int:
    specs = list_historical_bugs(include_pending=bool(args.include_pending))
    rows = [_historical_status_row(spec) for spec in specs]
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


def _historical_status_row(spec: object) -> dict:
    fixture_env = str(getattr(spec, "fixture_env", "") or "")
    fixture_env_value = os.environ.get(fixture_env) if fixture_env else ""
    return {
        "bug_id": str(getattr(spec, "bug_id")),
        "project": str(getattr(spec, "project")),
        "status": str(getattr(spec, "status")),
        "counted": getattr(spec, "status") == "confirmed_fixed",
        "replay_kind": str(getattr(spec, "replay_kind", "experiment")),
        "target_suite": str(getattr(spec, "target_suite")),
        "target_version": str(getattr(spec, "target_version")),
        "fixed_version": str(getattr(spec, "fixed_version", "")),
        "issue_url": str(getattr(spec, "issue_url", "")),
        "default_presets": list(getattr(spec, "default_presets", ())),
        "default_cases": int(getattr(spec, "default_cases", 0)),
        "default_seeds": list(getattr(spec, "default_seeds", ())),
        "expected_root_causes": list(getattr(spec, "expected_root_causes", ())),
        "expected_suspicious_backends": list(getattr(spec, "expected_suspicious_backends", ())),
        "fixture_spec": str(getattr(spec, "fixture_spec", "")),
        "fixture_env": fixture_env,
        "fixture_env_set": bool(fixture_env_value),
        "fixture_env_status": "set" if fixture_env_value else "unset" if fixture_env else "n/a",
        "notes": str(getattr(spec, "notes", "")),
    }


def _resolve_fixture_replay_path(args: argparse.Namespace) -> Path:
    if args.fixture and args.fixture_env:
        raise ValueError("use only one of --fixture or --fixture-env")
    if args.fixture:
        return Path(args.fixture)
    if args.fixture_env:
        value = os.environ.get(args.fixture_env)
        if not value:
            raise ValueError(f"environment variable {args.fixture_env} is not set")
        return Path(value)
    raise ValueError("one of --fixture or --fixture-env is required")


def _fixture_artifact_budget_allows(artifact_limit: int | None) -> bool:
    return artifact_limit is None or int(artifact_limit) > 0


def _fixture_replay_run_id(label: str) -> str:
    ts = utc_now().replace(":", "").replace("-", "").replace("Z", "")
    return f"run-fixture-{slugify(label)}-{ts}-{time.time_ns()}"


def _load_artifact_config(bug_dir: Path) -> dict:
    config_path = bug_dir / "config.json"
    return load_json(config_path) if config_path.exists() else {}


def _write_reduced_reproducer(bug_dir: Path, backends: list[str]) -> None:
    repro = f'''#!/usr/bin/env python3
from datadiff.config import ExperimentConfig
from datadiff.dsl import Case
from datadiff.runner import run_loaded_case
from datadiff.util import load_json

here = __import__("pathlib").Path(__file__).parent
case = Case.from_dict(load_json(here / "reduced_case.json"))
config_data = load_json(here / "config.json")
config = ExperimentConfig(**config_data) if config_data else ExperimentConfig()
result = run_loaded_case(case, backends={backends!r}, config=config, save_artifact=False)
print(result["status"])
for finding in result["findings"]:
    print(finding)
'''
    path = bug_dir / "reproduce_reduced.py"
    path.write_text(repro, encoding="utf-8")
    path.chmod(0o755)


def _parse_seeds(value: str) -> list[int]:
    seeds: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        seeds.append(int(part))
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def _parse_presets(value: str) -> list[str]:
    presets = []
    for part in value.split(","):
        preset = part.strip()
        if preset:
            presets.append(preset)
    if not presets:
        raise ValueError("at least one preset is required")
    return presets


def _parse_target_suites(value: str | None) -> list[str]:
    if not value:
        return []
    suites: list[str] = []
    for part in value.split(","):
        suite = part.strip()
        if not suite:
            continue
        if suite not in TARGET_SUITES:
            raise ValueError(f"unknown target suite: {suite}")
        if suite not in suites:
            suites.append(suite)
    return suites


def _experiment_target_runs(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    if getattr(args, "backends", None):
        return [(getattr(args, "target_suite", "custom"), _resolve_run_backends(args))]
    suites = _parse_target_suites(getattr(args, "target_suites", None)) or [args.target_suite]
    return [(suite, resolve_target_backends(target_suite=suite)) for suite in suites]


def _live_bughunt_config(
    *,
    guidance_targets: list[str],
    generator_profile: str = "bughunt",
    candidate_pool: int = 12,
    local_source_exploration_weight: float = 0.40,
    metamorphic: bool = False,
    metamorphic_variant_limit: int = 4,
) -> ExperimentConfig:
    return ExperimentConfig(
        generator_profile=generator_profile,
        enable_metamorphic_oracle=metamorphic,
        oracle_mode="both" if metamorphic else "differential",
        guidance_strategy="guided",
        guidance_candidate_pool=candidate_pool,
        guidance_targets=list(guidance_targets),
        enable_local_source_scheduler=True,
        local_source_exploration_weight=local_source_exploration_weight,
        metamorphic_variant_limit=metamorphic_variant_limit,
    )


def _preset_config(name: str) -> ExperimentConfig:
    if name == "baseline":
        return ExperimentConfig()
    if name == "no_type_aware":
        return ExperimentConfig(enable_type_aware_generation=False)
    if name == "no_normalizer":
        return ExperimentConfig(enable_normalizer=False)
    if name == "no_feedback":
        return ExperimentConfig(enable_feedback=False)
    if name == "metamorphic":
        return ExperimentConfig(enable_metamorphic_oracle=True, oracle_mode="both")
    if name == "reducer":
        return ExperimentConfig(enable_reducer=True)
    if name == "oracle_only_metamorphic":
        return ExperimentConfig(
            enable_differential_oracle=False,
            enable_metamorphic_oracle=True,
            oracle_mode="metamorphic",
        )
    if name == "edge_float":
        return ExperimentConfig(generator_profile="edge_float")
    if name == "edge_float_guided":
        return ExperimentConfig(
            generator_profile="edge_float",
            guidance_strategy="guided",
            guidance_candidate_pool=8,
            guidance_targets=["edge_float", "numeric", "expressions"],
        )
    if name == "edge_float_metamorphic":
        return ExperimentConfig(
            generator_profile="edge_float",
            enable_metamorphic_oracle=True,
            oracle_mode="both",
        )
    if name == "workflow":
        return ExperimentConfig(generator_profile="workflow")
    if name == "workflow_metamorphic":
        return ExperimentConfig(
            generator_profile="workflow",
            enable_metamorphic_oracle=True,
            oracle_mode="both",
        )
    if name == "bughunt":
        return ExperimentConfig(generator_profile="bughunt")
    if name == "bughunt_no_groupby":
        return ExperimentConfig(generator_profile="bughunt_no_groupby")
    if name == "bughunt_guided":
        return ExperimentConfig(
            generator_profile="bughunt",
            guidance_strategy="guided",
            guidance_candidate_pool=8,
            guidance_targets=["join", "groupby", "mutate", "filter", "expressions"],
        )
    if name == "bughunt_no_groupby_guided":
        return ExperimentConfig(
            generator_profile="bughunt_no_groupby",
            guidance_strategy="guided",
            guidance_candidate_pool=8,
            guidance_targets=["join", "mutate", "filter", "expressions", "sort_limit"],
        )
    if name == "bughunt_metamorphic":
        return ExperimentConfig(
            generator_profile="bughunt",
            enable_metamorphic_oracle=True,
            oracle_mode="both",
            metamorphic_variant_limit=8,
        )
    if name == "bughunt_no_groupby_metamorphic":
        return ExperimentConfig(
            generator_profile="bughunt_no_groupby",
            enable_metamorphic_oracle=True,
            oracle_mode="both",
            metamorphic_variant_limit=8,
        )
    if name == "bughunt_guided_metamorphic":
        return ExperimentConfig(
            generator_profile="bughunt",
            enable_metamorphic_oracle=True,
            oracle_mode="both",
            guidance_strategy="guided",
            guidance_candidate_pool=8,
            guidance_targets=["join", "groupby", "mutate", "filter", "expressions"],
            metamorphic_variant_limit=8,
        )
    if name == "bughunt_no_groupby_guided_metamorphic":
        return ExperimentConfig(
            generator_profile="bughunt_no_groupby",
            enable_metamorphic_oracle=True,
            oracle_mode="both",
            guidance_strategy="guided",
            guidance_candidate_pool=8,
            guidance_targets=["join", "mutate", "filter", "expressions", "sort_limit"],
            metamorphic_variant_limit=8,
        )
    if name == "null_groupby_topk":
        return ExperimentConfig(
            generator_profile="null_groupby_topk",
            guidance_strategy="guided",
            guidance_candidate_pool=4,
            guidance_targets=["null_groupby_topk", "groupby", "nulls", "sort_limit"],
            metamorphic_variant_limit=4,
        )
    if name == "null_agg_topk":
        return ExperimentConfig(
            generator_profile="null_agg_topk",
            guidance_strategy="guided",
            guidance_candidate_pool=4,
            guidance_targets=["null_agg_topk", "groupby", "nulls", "aggregation", "sort_limit"],
            metamorphic_variant_limit=4,
        )
    if name == "filter_null_agg_topk":
        return ExperimentConfig(
            generator_profile="filter_null_agg_topk",
            guidance_strategy="guided",
            guidance_candidate_pool=4,
            guidance_targets=[
                "filter_null_agg_topk",
                "filter",
                "groupby",
                "nulls",
                "aggregation",
                "sort_limit",
                "expressions",
            ],
            metamorphic_variant_limit=4,
        )
    if name == "join_null_agg_topk":
        return ExperimentConfig(
            generator_profile="join_null_agg_topk",
            guidance_strategy="guided",
            guidance_candidate_pool=4,
            guidance_targets=["join_null_agg_topk", "join", "nulls", "aggregation", "sort_limit", "expressions"],
            metamorphic_variant_limit=4,
        )
    if name == "join_null_key_topk":
        return ExperimentConfig(
            generator_profile="join_null_key_topk",
            guidance_strategy="guided",
            guidance_candidate_pool=4,
            guidance_targets=["join_null_key_topk", "join", "groupby", "nulls", "sort_limit", "topk"],
            metamorphic_variant_limit=4,
        )
    if name == "wide_offset_topk":
        return ExperimentConfig(
            generator_profile="wide_offset_topk",
            guidance_strategy="guided",
            guidance_candidate_pool=2,
            guidance_targets=["wide_offset_topk", "sort_limit", "sort_offset", "offset", "topk", "nulls"],
            metamorphic_variant_limit=2,
        )
    if name == "empty_filter_groupby":
        return ExperimentConfig(
            generator_profile="empty_filter_groupby",
            guidance_strategy="guided",
            guidance_candidate_pool=4,
            guidance_targets=["empty_filter_groupby", "filter", "groupby", "aggregation", "empty", "sort_limit"],
            metamorphic_variant_limit=4,
        )
    if name == "join_filter_groupby":
        return ExperimentConfig(
            generator_profile="join_filter_groupby",
            guidance_strategy="guided",
            guidance_candidate_pool=4,
            guidance_targets=["join_filter_groupby", "join", "filter", "groupby", "aggregation", "sort_limit", "expressions"],
            metamorphic_variant_limit=4,
        )
    if name == "join_null_truth_filter":
        return ExperimentConfig(
            generator_profile="join_null_truth_filter",
            guidance_strategy="guided",
            guidance_candidate_pool=4,
            guidance_targets=["join_null_truth_filter", "join", "filter", "truth_filter", "nulls"],
            metamorphic_variant_limit=4,
        )
    if name == "join_groupby_stress":
        return ExperimentConfig(
            generator_profile="join_groupby_stress",
            guidance_strategy="guided",
            guidance_candidate_pool=1,
            guidance_targets=["join", "groupby", "aggregation", "global_aggregation"],
            metamorphic_variant_limit=2,
        )
    if name == "storage_offset":
        return ExperimentConfig(
            generator_profile="storage_offset",
            guidance_strategy="guided",
            guidance_candidate_pool=1,
            guidance_targets=["sort_limit", "sort_offset", "offset"],
            metamorphic_variant_limit=0,
        )
    if name == "null_groupby_topk_metamorphic":
        return ExperimentConfig(
            generator_profile="null_groupby_topk",
            enable_metamorphic_oracle=True,
            oracle_mode="both",
            guidance_strategy="guided",
            guidance_candidate_pool=4,
            guidance_targets=["null_groupby_topk", "groupby", "nulls", "sort_limit"],
            metamorphic_variant_limit=8,
        )
    if name == "null_agg_topk_metamorphic":
        return ExperimentConfig(
            generator_profile="null_agg_topk",
            enable_metamorphic_oracle=True,
            oracle_mode="both",
            guidance_strategy="guided",
            guidance_candidate_pool=4,
            guidance_targets=["null_agg_topk", "groupby", "nulls", "aggregation", "sort_limit"],
            metamorphic_variant_limit=8,
        )
    if name == "filter_null_agg_topk_metamorphic":
        return ExperimentConfig(
            generator_profile="filter_null_agg_topk",
            enable_metamorphic_oracle=True,
            oracle_mode="both",
            guidance_strategy="guided",
            guidance_candidate_pool=4,
            guidance_targets=[
                "filter_null_agg_topk",
                "filter",
                "groupby",
                "nulls",
                "aggregation",
                "sort_limit",
                "expressions",
            ],
            metamorphic_variant_limit=8,
        )
    if name == "join_null_agg_topk_metamorphic":
        return ExperimentConfig(
            generator_profile="join_null_agg_topk",
            enable_metamorphic_oracle=True,
            oracle_mode="both",
            guidance_strategy="guided",
            guidance_candidate_pool=4,
            guidance_targets=["join_null_agg_topk", "join", "nulls", "aggregation", "sort_limit", "expressions"],
            metamorphic_variant_limit=8,
        )
    if name == "join_null_key_topk_metamorphic":
        return ExperimentConfig(
            generator_profile="join_null_key_topk",
            enable_metamorphic_oracle=True,
            oracle_mode="both",
            guidance_strategy="guided",
            guidance_candidate_pool=4,
            guidance_targets=["join_null_key_topk", "join", "groupby", "nulls", "sort_limit", "topk"],
            metamorphic_variant_limit=8,
        )
    if name == "wide_offset_topk_metamorphic":
        return ExperimentConfig(
            generator_profile="wide_offset_topk",
            enable_metamorphic_oracle=True,
            oracle_mode="both",
            guidance_strategy="guided",
            guidance_candidate_pool=2,
            guidance_targets=["wide_offset_topk", "sort_limit", "sort_offset", "offset", "topk", "nulls"],
            metamorphic_variant_limit=4,
        )
    if name == "empty_filter_groupby_metamorphic":
        return ExperimentConfig(
            generator_profile="empty_filter_groupby",
            enable_metamorphic_oracle=True,
            oracle_mode="both",
            guidance_strategy="guided",
            guidance_candidate_pool=4,
            guidance_targets=["empty_filter_groupby", "filter", "groupby", "aggregation", "empty", "sort_limit"],
            metamorphic_variant_limit=8,
        )
    if name == "join_filter_groupby_metamorphic":
        return ExperimentConfig(
            generator_profile="join_filter_groupby",
            enable_metamorphic_oracle=True,
            oracle_mode="both",
            guidance_strategy="guided",
            guidance_candidate_pool=4,
            guidance_targets=["join_filter_groupby", "join", "filter", "groupby", "aggregation", "sort_limit", "expressions"],
            metamorphic_variant_limit=8,
        )
    if name == "join_null_truth_filter_metamorphic":
        return ExperimentConfig(
            generator_profile="join_null_truth_filter",
            enable_metamorphic_oracle=True,
            oracle_mode="both",
            guidance_strategy="guided",
            guidance_candidate_pool=4,
            guidance_targets=["join_null_truth_filter", "join", "filter", "truth_filter", "nulls"],
            metamorphic_variant_limit=8,
        )
    if name == "join_groupby_stress_metamorphic":
        return ExperimentConfig(
            generator_profile="join_groupby_stress",
            enable_metamorphic_oracle=True,
            oracle_mode="both",
            guidance_strategy="guided",
            guidance_candidate_pool=1,
            guidance_targets=["join", "groupby", "aggregation", "global_aggregation"],
            metamorphic_variant_limit=4,
        )
    if name == "float_group_key":
        return ExperimentConfig(
            generator_profile="float_group_key",
            guidance_strategy="guided",
            guidance_candidate_pool=4,
            guidance_targets=["float_group_key", "join", "mutate", "groupby", "expressions"],
            metamorphic_variant_limit=4,
        )
    if name == "float_group_key_metamorphic":
        return ExperimentConfig(
            generator_profile="float_group_key",
            enable_metamorphic_oracle=True,
            oracle_mode="both",
            guidance_strategy="guided",
            guidance_candidate_pool=4,
            guidance_targets=["float_group_key", "join", "mutate", "groupby", "expressions"],
            metamorphic_variant_limit=8,
        )
    if name == "join_null_sort":
        return ExperimentConfig(
            generator_profile="join_null_sort",
            guidance_strategy="guided",
            guidance_candidate_pool=4,
            guidance_targets=["join_null_sort", "join", "nulls", "sort_limit", "expressions"],
            metamorphic_variant_limit=4,
        )
    if name == "join_null_sort_metamorphic":
        return ExperimentConfig(
            generator_profile="join_null_sort",
            enable_metamorphic_oracle=True,
            oracle_mode="both",
            guidance_strategy="guided",
            guidance_candidate_pool=4,
            guidance_targets=["join_null_sort", "join", "nulls", "sort_limit", "expressions"],
            metamorphic_variant_limit=8,
        )
    if name == "ordered_groupby_sort":
        return ExperimentConfig(
            generator_profile="ordered_groupby_sort",
            guidance_strategy="guided",
            guidance_candidate_pool=4,
            guidance_targets=["ordered_groupby_sort", "groupby", "aggregation", "sort_limit"],
            metamorphic_variant_limit=4,
        )
    if name == "ordered_groupby_sort_metamorphic":
        return ExperimentConfig(
            generator_profile="ordered_groupby_sort",
            enable_metamorphic_oracle=True,
            oracle_mode="both",
            guidance_strategy="guided",
            guidance_candidate_pool=4,
            guidance_targets=["ordered_groupby_sort", "groupby", "aggregation", "sort_limit"],
            metamorphic_variant_limit=8,
        )
    if name == "topk_resort":
        return ExperimentConfig(
            generator_profile="topk_resort",
            guidance_strategy="guided",
            guidance_candidate_pool=4,
            guidance_targets=["topk_resort", "sort_limit", "topk", "nulls"],
            metamorphic_variant_limit=4,
        )
    if name == "topk_resort_metamorphic":
        return ExperimentConfig(
            generator_profile="topk_resort",
            enable_metamorphic_oracle=True,
            oracle_mode="both",
            guidance_strategy="guided",
            guidance_candidate_pool=4,
            guidance_targets=["topk_resort", "sort_limit", "topk", "nulls"],
            metamorphic_variant_limit=8,
        )
    if name == "join_ordered_agg_topk":
        return ExperimentConfig(
            generator_profile="join_ordered_agg_topk",
            guidance_strategy="guided",
            guidance_candidate_pool=4,
            guidance_targets=["join_ordered_agg_topk", "join", "groupby", "aggregation", "sort_limit", "topk"],
            metamorphic_variant_limit=4,
        )
    if name == "join_ordered_agg_topk_metamorphic":
        return ExperimentConfig(
            generator_profile="join_ordered_agg_topk",
            enable_metamorphic_oracle=True,
            oracle_mode="both",
            guidance_strategy="guided",
            guidance_candidate_pool=4,
            guidance_targets=["join_ordered_agg_topk", "join", "groupby", "aggregation", "sort_limit", "topk"],
            metamorphic_variant_limit=8,
        )
    if name == "global_null_aggregate":
        return ExperimentConfig(
            generator_profile="global_null_aggregate",
            guidance_strategy="guided",
            guidance_candidate_pool=4,
            guidance_targets=["global_null_aggregate", "global_aggregation", "aggregation", "nulls", "sort_limit"],
            metamorphic_variant_limit=4,
        )
    if name == "global_null_aggregate_metamorphic":
        return ExperimentConfig(
            generator_profile="global_null_aggregate",
            enable_metamorphic_oracle=True,
            oracle_mode="both",
            guidance_strategy="guided",
            guidance_candidate_pool=4,
            guidance_targets=["global_null_aggregate", "global_aggregation", "aggregation", "nulls", "sort_limit"],
            metamorphic_variant_limit=8,
        )
    if name == "string_count_groupby":
        return ExperimentConfig(
            generator_profile="string_count_groupby",
            guidance_strategy="guided",
            guidance_candidate_pool=4,
            guidance_targets=["string_count_groupby", "groupby", "strings", "nulls", "aggregation", "sort_limit"],
            metamorphic_variant_limit=4,
        )
    if name == "string_count_groupby_metamorphic":
        return ExperimentConfig(
            generator_profile="string_count_groupby",
            enable_metamorphic_oracle=True,
            oracle_mode="both",
            guidance_strategy="guided",
            guidance_candidate_pool=4,
            guidance_targets=["string_count_groupby", "groupby", "strings", "nulls", "aggregation", "sort_limit"],
            metamorphic_variant_limit=8,
        )
    if name == "live_datafusion":
        return _live_bughunt_config(
            guidance_targets=list(LIVE_BUGHUNT_TARGETS),
            local_source_exploration_weight=0.35,
        )
    if name == "live_datafusion_metamorphic":
        return _live_bughunt_config(
            candidate_pool=8,
            guidance_targets=list(LIVE_BUGHUNT_TARGETS),
            local_source_exploration_weight=0.35,
            metamorphic=True,
            metamorphic_variant_limit=6,
        )
    if name == "live_arrow":
        return _live_bughunt_config(
            guidance_targets=list(LIVE_ARROW_TARGETS),
            local_source_exploration_weight=0.45,
        )
    if name == "live_arrow_metamorphic":
        return _live_bughunt_config(
            candidate_pool=8,
            guidance_targets=list(LIVE_ARROW_TARGETS),
            local_source_exploration_weight=0.45,
            metamorphic=True,
            metamorphic_variant_limit=6,
        )
    if name == "live_polars_lazy":
        return _live_bughunt_config(
            guidance_targets=list(LIVE_POLARS_LAZY_TARGETS),
            local_source_exploration_weight=0.45,
        )
    if name == "live_polars_lazy_metamorphic":
        return _live_bughunt_config(
            candidate_pool=8,
            guidance_targets=list(LIVE_POLARS_LAZY_TARGETS),
            local_source_exploration_weight=0.45,
            metamorphic=True,
            metamorphic_variant_limit=8,
        )
    if name == "live_embedded_sql":
        return _live_bughunt_config(
            guidance_targets=list(LIVE_EMBEDDED_SQL_TARGETS),
            local_source_exploration_weight=0.40,
        )
    if name == "live_embedded_sql_metamorphic":
        return _live_bughunt_config(
            candidate_pool=8,
            guidance_targets=list(LIVE_EMBEDDED_SQL_TARGETS),
            local_source_exploration_weight=0.40,
            metamorphic=True,
            metamorphic_variant_limit=8,
        )
    if name == "live_cross_family":
        return _live_bughunt_config(
            guidance_targets=list(LIVE_CROSS_FAMILY_TARGETS),
            local_source_exploration_weight=0.40,
        )
    if name == "live_cross_family_metamorphic":
        return _live_bughunt_config(
            candidate_pool=8,
            guidance_targets=list(LIVE_CROSS_FAMILY_TARGETS),
            local_source_exploration_weight=0.40,
            metamorphic=True,
            metamorphic_variant_limit=6,
        )
    if name == "guided":
        return ExperimentConfig(guidance_strategy="guided", guidance_candidate_pool=8)
    if name == "guided_filter":
        return ExperimentConfig(
            guidance_strategy="guided",
            guidance_candidate_pool=8,
            guidance_targets=["filter"],
        )
    if name == "guided_groupby":
        return ExperimentConfig(
            guidance_strategy="guided",
            guidance_candidate_pool=8,
            guidance_targets=["groupby", "aggregation"],
        )
    if name == "guided_join":
        return ExperimentConfig(
            generator_profile="bughunt_no_groupby",
            guidance_strategy="guided",
            guidance_candidate_pool=8,
            guidance_targets=["join", "sort_limit"],
        )
    if name == "guided_mutate":
        return ExperimentConfig(
            guidance_strategy="guided",
            guidance_candidate_pool=8,
            guidance_targets=["mutate", "expressions"],
        )
    raise ValueError(f"unknown experiment preset: {name}")


def cmd_experiment(args: argparse.Namespace) -> int:
    ensure_dirs()
    presets = _parse_presets(args.presets)
    seeds = _parse_seeds(args.seeds)
    target_runs = _experiment_target_runs(args)
    suite_names = [suite for suite, _ in target_runs]
    backend_union = sorted({backend for _, backends in target_runs for backend in backends})
    duration_s = parse_duration(args.duration)
    evidence_mode = _resolve_evidence_mode(getattr(args, "evidence_mode", "auto"), suite_names)
    planned_runs = [
        {
            "order": order,
            "arm_id": f"{target_suite}:{preset}:seed{seed}",
            "target_suite": target_suite,
            "backends": backends,
            "preset": preset,
            "seed": seed,
            "cases": args.cases,
            "duration_s": duration_s,
            "log_level": args.log_level,
            "compress_run_log": not args.no_compress_run_log,
            "artifact_limit": args.artifact_limit,
            "metamorphic_variant_limit": args.metamorphic_variant_limit,
            "evidence_mode": evidence_mode,
            "known_bug_id": str(getattr(args, "known_bug_id", "") or ""),
            "target_version": str(getattr(args, "target_version", "") or ""),
            "run_theme": str(getattr(args, "run_theme", "") or ""),
            "paper_notes": str(getattr(args, "paper_notes", "") or ""),
            "enable_local_source_scheduler": bool(getattr(args, "enable_local_source_scheduler", False)),
            "local_source_exploration_weight": max(
                0.0, float(getattr(args, "local_source_exploration_weight", 0.5))
            ),
            "skip_run_reports": args.skip_run_reports,
        }
        for order, (target_suite, backends, preset, seed) in enumerate(
            (target_suite, backends, preset, seed)
            for target_suite, backends in target_runs
            for preset in presets
            for seed in seeds
        )
    ]
    for job in planned_runs:
        job["estimated_cost"] = round(_experiment_job_weight(job), 4)
    parallelism = _resolve_experiment_parallelism(args, planned_runs)
    for job in planned_runs:
        job["worker_thread_limit"] = parallelism["worker_thread_limit"]
    jobs = int(parallelism["worker_count"])
    schedule = _resolve_experiment_schedule(args, jobs=jobs)
    manifest = {
        "created_at": utc_now(),
        "presets": presets,
        "seeds": seeds,
        "cases": args.cases,
        "duration_s": duration_s,
        "evidence_mode": evidence_mode,
        "known_bug_id": str(getattr(args, "known_bug_id", "") or ""),
        "target_version": str(getattr(args, "target_version", "") or ""),
        "run_theme": str(getattr(args, "run_theme", "") or ""),
        "paper_notes": str(getattr(args, "paper_notes", "") or ""),
        "backends": backend_union,
        "target_suite": suite_names[0] if len(suite_names) == 1 else ",".join(suite_names),
        "target_suites": suite_names,
        "backends_by_suite": {suite: backends for suite, backends in target_runs},
        "targets": describe_targets(backend_union),
        "common_capabilities": common_capabilities(backend_union),
        "log_level": args.log_level,
        "compress_run_log": not args.no_compress_run_log,
        "metamorphic_variant_limit": args.metamorphic_variant_limit,
        "jobs": jobs,
        "parallelism": parallelism,
        "schedule": schedule,
        "local_source_scheduler": {
            "enabled": bool(getattr(args, "enable_local_source_scheduler", False) or schedule == "adaptive"),
            "exploration_weight": max(0.0, float(getattr(args, "local_source_exploration_weight", 0.5))),
        },
        "runs": [],
    }
    if schedule == "adaptive":
        completed_runs = _run_experiment_adaptive(args, manifest, planned_runs, duration_s, jobs=jobs)
        for result in completed_runs:
            manifest["runs"].append(result["run"])
        manifest["adaptive_state"] = completed_runs[-1]["scheduler_state"] if completed_runs else []
        manifest_path = _experiment_manifest_path()
        dump_json(manifest, manifest_path)
        print(f"experiment manifest: {manifest_path}")
        if not getattr(args, "skip_paper_journal", False):
            journal_path, journal_md = _record_experiment_journal(manifest_path, manifest)
            print(f"paper run journal: {journal_path}")
            print(f"paper run journal markdown: {journal_md}")
        return 0
    completed_runs = []
    if jobs == 1:
        for job in planned_runs:
            result = _run_experiment_job(job)
            completed_runs.append(result)
            print(result["message"], flush=True)
    else:
        worker_count = min(jobs, len(planned_runs))
        try:
            completed_runs = _run_experiment_jobs_parallel(
                ProcessPoolExecutor,
                worker_count,
                planned_runs,
                max_parallel_cost=float(parallelism["max_parallel_cost"]),
            )
        except PermissionError:
            print("process parallelism unavailable; falling back to threaded workers", flush=True)
            completed_runs = _run_experiment_jobs_parallel(
                ThreadPoolExecutor,
                worker_count,
                planned_runs,
                max_parallel_cost=float(parallelism["max_parallel_cost"]),
            )
    for result in sorted(completed_runs, key=lambda item: item["order"]):
        manifest["runs"].append(result["run"])
    manifest_path = _experiment_manifest_path()
    dump_json(manifest, manifest_path)
    print(f"experiment manifest: {manifest_path}")
    if not getattr(args, "skip_paper_journal", False):
        journal_path, journal_md = _record_experiment_journal(manifest_path, manifest)
        print(f"paper run journal: {journal_path}")
        print(f"paper run journal markdown: {journal_md}")
    return 0


def _record_experiment_journal(manifest_path: Path, manifest: dict) -> tuple[Path, Path]:
    journal_file = REPORTS_DIR / "paper-run-journal.jsonl"
    entries = []
    for run in manifest.get("runs", []):
        run_file = Path(run.get("run_file", ""))
        context = {
            "command": "experiment",
            "theme": _experiment_run_theme(manifest, run),
            "notes": str(manifest.get("paper_notes", "") or ""),
            "evidence_mode": str(run.get("evidence_mode") or manifest.get("evidence_mode", "")),
            "known_bug_id": str(run.get("known_bug_id") or manifest.get("known_bug_id", "")),
            "target_version": str(run.get("target_version") or manifest.get("target_version", "")),
            "target_suite": str(run.get("target_suite", "")),
            "preset": str(run.get("preset", "")),
            "seed": run.get("seed", ""),
            "backends": run.get("backends", []),
            "manifest_file": str(manifest_path),
        }
        entries.append(build_run_journal_entry(run_file, context))
    append_run_journal_entries(entries, journal_file)
    md_path = write_run_journal_markdown(journal_file)
    return journal_file, md_path


def _experiment_run_theme(manifest: dict, run: dict) -> str:
    base = str(manifest.get("run_theme", "") or "").strip()
    suffix = f"{run.get('target_suite', '')}:{run.get('preset', '')}:seed{run.get('seed', '')}"
    if base:
        return f"{base} | {suffix}"
    evidence_mode = str(run.get("evidence_mode") or manifest.get("evidence_mode", "live"))
    known_bug_id = str(run.get("known_bug_id") or manifest.get("known_bug_id", "") or "")
    if evidence_mode == "historical" and known_bug_id:
        return f"historical:{known_bug_id}:{suffix}"
    return f"{evidence_mode}:{suffix}"


def _run_experiment_adaptive(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    planned_runs: list[dict[str, Any]],
    duration_s: float | None,
    *,
    jobs: int,
) -> list[dict[str, Any]]:
    default_batch_cases = min(100, max(1, int(args.cases or 100)))
    batch_cases = max(1, int(getattr(args, "batch_cases", 0) or default_batch_cases))
    batch_duration_s = parse_duration(getattr(args, "batch_duration", None))
    if batch_duration_s is None and duration_s is not None and args.cases is None:
        batch_duration_s = min(duration_s, 30.0)
    if args.cases is not None:
        total_cases_budget = len(planned_runs) * max(1, int(args.cases))
    elif duration_s is None:
        total_cases_budget = len(planned_runs) * 100
    else:
        total_cases_budget = None
    total_duration_budget_s = None if args.cases is not None or duration_s is None else len(planned_runs) * duration_s
    schedule_config = AdaptiveScheduleConfig(
        batch_cases=batch_cases,
        batch_duration_s=batch_duration_s,
        warmup_batches=max(1, int(getattr(args, "warmup_batches", 1))),
        exploration_weight=max(0.0, float(getattr(args, "exploration_weight", 0.75))),
    )
    scheduler = AdaptiveBudgetScheduler(
        planned_runs,
        total_cases_budget=total_cases_budget,
        total_duration_budget_s=total_duration_budget_s,
        config=schedule_config,
    )
    manifest["adaptive_config"] = {
        "total_cases_budget": total_cases_budget,
        "total_duration_budget_s": total_duration_budget_s,
        "batch_cases": batch_cases,
        "batch_duration_s": batch_duration_s,
        "warmup_batches": schedule_config.warmup_batches,
        "exploration_weight": schedule_config.exploration_weight,
        "fine_grained_local_source_scheduler": True,
        "local_source_exploration_weight": max(
            0.0, float(getattr(args, "local_source_exploration_weight", 0.5))
        ),
        "jobs": jobs,
        "parallelism": manifest.get("parallelism", {}),
    }
    completed_runs = []
    worker_count = min(max(1, jobs), len(planned_runs))
    if worker_count == 1:
        while scheduler.has_budget():
            batches = scheduler.next_round(1)
            if not batches:
                break
            completed_runs.extend(_complete_adaptive_round(scheduler, [_run_experiment_job(_adaptive_job(batches[0]))], batches))
            print(completed_runs[-1]["message"], flush=True)
        return completed_runs
    try:
        completed_runs.extend(_run_experiment_adaptive_parallel(ProcessPoolExecutor, worker_count, scheduler))
    except PermissionError:
        print("process parallelism unavailable; falling back to threaded workers", flush=True)
        completed_runs.extend(_run_experiment_adaptive_parallel(ThreadPoolExecutor, worker_count, scheduler))
    return completed_runs


def _resolve_experiment_parallelism(args: argparse.Namespace, planned_runs: list[dict[str, Any]]) -> dict[str, Any]:
    cpu_count = max(1, os.cpu_count() or 1)
    requested = getattr(args, "jobs", "auto")
    if requested == "auto":
        # Each matrix job runs several native engines that may use their own
        # thread pools. A conservative default gives better 24h throughput than
        # letting every target compete for every core.
        worker_count = max(1, min(len(planned_runs), max(1, cpu_count // 4), 6))
    else:
        worker_count = max(1, int(requested))
        worker_count = min(worker_count, max(1, len(planned_runs)))
    costs = [_experiment_job_weight(job) for job in planned_runs] or [1.0]
    max_job_cost = max(costs)
    requested_cost = getattr(args, "max_parallel_cost", None)
    if requested_cost is None:
        # Cost units are backend-weighted, not CPU cores. This budget allows
        # several light runs together but usually keeps latest_all_engines-style
        # jobs from stacking on top of each other.
        max_parallel_cost = max(max_job_cost, cpu_count * 0.75)
    else:
        max_parallel_cost = max(max_job_cost, float(requested_cost))
    worker_thread_limit = max(1, min(4, cpu_count // max(1, worker_count)))
    return {
        "requested_jobs": requested,
        "worker_count": worker_count,
        "cpu_count": cpu_count,
        "max_parallel_cost": round(max_parallel_cost, 4),
        "max_job_cost": round(max_job_cost, 4),
        "worker_thread_limit": worker_thread_limit,
        "bounded_submission": True,
        "cost_limited": True,
    }


def _run_experiment_adaptive_parallel(
    executor_cls: type,
    worker_count: int,
    scheduler: AdaptiveBudgetScheduler,
) -> list[dict[str, Any]]:
    completed_runs: list[dict[str, Any]] = []
    with executor_cls(max_workers=worker_count) as executor:
        while scheduler.has_budget():
            batches = scheduler.next_round(worker_count)
            if not batches:
                break
            futures = {
                executor.submit(_run_experiment_job, _adaptive_job(batch)): batch
                for batch in batches
            }
            round_results = [future.result() for future in as_completed(futures)]
            completed_runs.extend(_complete_adaptive_round(scheduler, round_results, batches))
            for item in completed_runs[-len(batches):]:
                print(item["message"], flush=True)
    return completed_runs


def _adaptive_job(batch: Any) -> dict[str, Any]:
    job = dict(batch.job)
    job["enable_local_source_scheduler"] = True
    return job


def _complete_adaptive_round(
    scheduler: AdaptiveBudgetScheduler,
    round_results: list[dict[str, Any]],
    batches: list[Any],
) -> list[dict[str, Any]]:
    results_by_batch = {int(item["run"]["batch_index"]): item for item in round_results}
    batch_by_index = {int(batch.batch_index): batch for batch in batches}
    completed: list[dict[str, Any]] = []
    for batch_index in sorted(batch_by_index):
        batch = batch_by_index[batch_index]
        result = results_by_batch[batch_index]
        run_file = Path(result["run"]["run_file"])
        observation = summarize_batch_run(run_file)
        meta_path = run_meta_path(run_file)
        meta = load_json(meta_path) if meta_path.exists() else {}
        reward = scheduler.record_result(
            batch,
            observation,
            next_seed=int(meta.get("next_seed", batch.seed + max(1, observation.cases))),
        )
        result["run"].update(
            {
                "schedule_arm_id": batch.arm_id,
                "batch_index": batch.batch_index,
                "scheduler_reward": reward,
                "scheduler_observation": {
                    "cases": observation.cases,
                    "elapsed_s": observation.elapsed_s,
                    "throughput_cases_s": observation.throughput_cases_s,
                    "findings": observation.findings,
                    "candidate_bug_cases": observation.candidate_bug_cases,
                    "candidate_bug_families": sorted(observation.candidate_bug_families),
                    "semantic_divergence_count": observation.semantic_divergence_count,
                    "false_positive_count": observation.false_positive_count,
                    "new_behavior_cases": observation.new_behavior_cases,
                    "first_candidate_bug_case_index": observation.first_candidate_bug_case_index,
                    "first_candidate_bug_elapsed_s": observation.first_candidate_bug_elapsed_s,
                    "candidate_bug_discovery_auc": observation.candidate_bug_discovery_auc,
                },
            }
        )
        completed.append(
            {
                **result,
                "scheduler_state": scheduler.snapshot(),
                "message": (
                    f"{result['message']} batch={batch.batch_index} "
                    f"reward={reward:.3f} remaining_cases={scheduler.remaining_cases_budget} "
                    f"remaining_duration_s={scheduler.remaining_duration_budget_s}"
                ),
            }
        )
    return completed


def _resolve_experiment_schedule(args: argparse.Namespace, *, jobs: int) -> str:
    schedule = getattr(args, "schedule", None)
    if schedule:
        return str(schedule)
    return "longest_first" if jobs > 1 else "matrix_order"


def _resolve_evidence_mode(value: str, suite_names: list[str]) -> str:
    if value != "auto":
        return value
    if suite_names and all(suite.startswith("seeded_") for suite in suite_names):
        return "seeded"
    return "live"


def _experiment_manifest_path() -> Path:
    ts = utc_now().replace(":", "").replace("-", "").replace("Z", "")
    return RUNS_DIR / f"experiment-{ts}-{time.time_ns()}.json"


def _run_experiment_jobs_parallel(
    executor_cls: type,
    worker_count: int,
    planned_runs: list[dict],
    *,
    max_parallel_cost: float,
) -> list[dict]:
    completed_runs = []
    scheduled_runs = sorted(planned_runs, key=_experiment_job_sort_key)
    queued_runs = list(scheduled_runs)
    running = {}
    running_cost = 0.0
    with executor_cls(max_workers=worker_count) as executor:
        while queued_runs or running:
            while len(running) < worker_count and queued_runs:
                available_cost = max_parallel_cost - running_cost
                index = _next_schedulable_job_index(queued_runs, available_cost)
                if index is None:
                    break
                job = queued_runs.pop(index)
                cost = _job_estimated_cost(job)
                future = executor.submit(_run_experiment_job, job)
                running[future] = cost
                running_cost += cost
            if not running and queued_runs:
                # A single run can exceed the current cost budget; allow it so
                # the queue cannot deadlock.
                job = queued_runs.pop(0)
                cost = _job_estimated_cost(job)
                future = executor.submit(_run_experiment_job, job)
                running[future] = cost
                running_cost += cost
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                running_cost -= running.pop(future)
                result = future.result()
                completed_runs.append(result)
                print(result["message"], flush=True)
    return completed_runs


def _next_schedulable_job_index(queued_runs: list[dict], available_cost: float) -> int | None:
    for index, job in enumerate(queued_runs):
        if _job_estimated_cost(job) <= max(0.0, available_cost):
            return index
    return None


def _job_estimated_cost(job: dict) -> float:
    return float(job.get("estimated_cost", _experiment_job_weight(job)))


def _experiment_job_sort_key(job: dict) -> tuple[float, int]:
    return (-_experiment_job_weight(job), int(job["order"]))


def _experiment_job_weight(job: dict) -> float:
    config = _preset_config(str(job["preset"]))
    backend_cost = sum(_backend_cost(str(backend)) for backend in job["backends"])
    metamorphic_multiplier = 1.0
    if config.enable_metamorphic_oracle:
        metamorphic_multiplier += max(1, int(config.metamorphic_variant_limit))
    profile_multiplier = {
        "float_group_key": 1.6,
        "join_null_sort": 1.5,
        "bughunt": 1.3,
        "bughunt_no_groupby": 1.2,
        "workflow": 1.2,
        "edge_float": 1.1,
        "null_groupby_topk": 1.0,
        "null_agg_topk": 1.0,
        "filter_null_agg_topk": 1.2,
        "join_null_agg_topk": 1.3,
        "join_null_key_topk": 1.3,
        "wide_offset_topk": 1.4,
        "empty_filter_groupby": 1.1,
        "join_filter_groupby": 1.4,
        "join_null_truth_filter": 1.3,
        "join_groupby_stress": 2.0,
        "storage_offset": 2.0,
        "ordered_groupby_sort": 1.2,
        "topk_resort": 1.1,
        "join_ordered_agg_topk": 1.5,
        "common": 1.0,
    }.get(config.generator_profile, 1.0)
    guidance_multiplier = 1.0 + 0.03 * max(0, int(config.guidance_candidate_pool) - 1)
    return backend_cost * metamorphic_multiplier * profile_multiplier * guidance_multiplier


def _backend_cost(backend: str) -> float:
    return {
        "datafusion": 1.5,
        "polars_lazy": 1.4,
        "duckdb": 1.2,
        "duckdb_persistent": 1.4,
        "sqlite": 1.0,
        "polars": 1.0,
        "pandas": 1.0,
        "pyarrow": 1.0,
    }.get(backend, 1.0)


def _run_experiment_job(job: dict) -> dict:
    _apply_native_thread_limits(int(job.get("worker_thread_limit", 1) or 1))
    preset_config = _preset_config(str(job["preset"]))
    preset_config.log_level = str(job["log_level"])
    preset_config.compress_run_log = bool(job["compress_run_log"])
    preset_config.artifact_limit = job["artifact_limit"]
    job_source_scheduler_enabled = bool(job.get("enable_local_source_scheduler", False))
    preset_config.enable_local_source_scheduler = preset_config.enable_local_source_scheduler or job_source_scheduler_enabled
    if job_source_scheduler_enabled:
        preset_config.local_source_exploration_weight = max(
            0.0,
            float(job.get("local_source_exploration_weight", preset_config.local_source_exploration_weight)),
        )
    if job["metamorphic_variant_limit"] is not None:
        preset_config.metamorphic_variant_limit = max(0, int(job["metamorphic_variant_limit"]))
    run_file = run_fuzz(
        cases=job["cases"],
        seed=int(job["seed"]),
        backends=list(job["backends"]),
        config=preset_config,
        duration_s=job["duration_s"],
    )
    md_path = csv_path = ""
    if not job["skip_run_reports"]:
        md_path, csv_path = write_report(run_file)
    run = {
        "target_suite": job["target_suite"],
        "backends": job["backends"],
        "preset": job["preset"],
        "seed": job["seed"],
        "evidence_mode": job.get("evidence_mode", ""),
        "known_bug_id": job.get("known_bug_id", ""),
        "target_version": job.get("target_version", ""),
        "batch_index": job.get("batch_index"),
        "schedule_arm_id": job.get("schedule_arm_id", ""),
        "estimated_cost": job.get("estimated_cost", ""),
        "worker_thread_limit": job.get("worker_thread_limit", ""),
        "run_file": str(run_file),
        "report": str(md_path),
        "csv": str(csv_path),
    }
    return {
        "order": int(job["order"]),
        "run": run,
        "message": f"{job['target_suite']} {job['preset']} seed={job['seed']} run={run_file}",
    }


def _apply_native_thread_limits(thread_limit: int) -> None:
    value = str(max(1, int(thread_limit)))
    for name in [
        "DATADIFF_DUCKDB_THREADS",
        "POLARS_MAX_THREADS",
        "RAYON_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "ARROW_NUM_THREADS",
    ]:
        os.environ[name] = value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="datadiff",
        description="Semantic differential fuzzing for DataFrame and embedded analytical engines.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="create project runtime directories")
    p_init.set_defaults(func=cmd_init)

    p_targets = sub.add_parser("targets", help="list supported backend targets and target suites")
    p_targets.add_argument("--json", action="store_true", help="emit target registry as JSON")
    p_targets.set_defaults(func=cmd_targets)

    p_prune = sub.add_parser("prune-corpus", help="dry-run prune of persisted feedback corpus cases")
    p_prune.add_argument(
        "--keep",
        type=int,
        default=4096,
        help="number of newest corpus/interesting JSON files to keep",
    )
    p_prune.add_argument("--yes", action="store_true", help="delete files beyond --keep")
    p_prune.set_defaults(func=cmd_prune_corpus)

    p_fuzz = sub.add_parser("fuzz", help="run differential fuzzing")
    p_fuzz.add_argument("--cases", type=int, default=None, help="maximum cases; defaults to 100 when --duration is absent")
    p_fuzz.add_argument("--duration", default=None, help="wall-clock budget such as 10s, 5m, 24h")
    p_fuzz.add_argument("--seed", type=int, default=1)
    add_target_suite_flags(p_fuzz)
    p_fuzz.add_argument("--profile", choices=["common", "edge_float", "workflow", "bughunt", "bughunt_no_groupby", "null_groupby_topk", "null_agg_topk", "filter_null_agg_topk", "join_null_agg_topk", "join_null_key_topk", "wide_offset_topk", "empty_filter_groupby", "join_filter_groupby", "join_null_truth_filter", "join_groupby_stress", "storage_offset", "float_group_key", "join_null_sort", "ordered_groupby_sort", "topk_resort", "join_ordered_agg_topk", "global_null_aggregate", "string_count_groupby"], default="common")
    add_guidance_flags(p_fuzz, default_strategy="random", default_candidate_pool=8)
    add_ablation_flags(p_fuzz)
    add_paper_journal_flags(p_fuzz)
    p_fuzz.set_defaults(func=cmd_fuzz)

    p_long = sub.add_parser("longrun", help="run long-duration fuzzing and persist generated test cases")
    p_long.add_argument("--cases", type=int, default=None, help="optional maximum cases; duration is the primary budget")
    p_long.add_argument("--duration", default="24h", help="wall-clock budget such as 10m, 24h, 2d")
    p_long.add_argument("--seed", type=int, default=1)
    add_target_suite_flags(p_long)
    p_long.add_argument("--profile", choices=["common", "edge_float", "workflow", "bughunt", "bughunt_no_groupby", "null_groupby_topk", "null_agg_topk", "filter_null_agg_topk", "join_null_agg_topk", "join_null_key_topk", "wide_offset_topk", "empty_filter_groupby", "join_filter_groupby", "join_null_truth_filter", "join_groupby_stress", "storage_offset", "float_group_key", "join_null_sort", "ordered_groupby_sort", "topk_resort", "join_ordered_agg_topk", "global_null_aggregate", "string_count_groupby"], default="common")
    add_guidance_flags(p_long, default_strategy="guided", default_candidate_pool=8)
    p_long.add_argument("--case-log", default=None, help="optional JSONL path for generated test cases")
    p_long.add_argument("--checkpoint-interval", default="60s", help="checkpoint write interval")
    p_long.add_argument("--progress-interval", default="60s", help="stdout progress interval")
    p_long.add_argument("--save-cases", action="store_true", help="persist every generated test case separately")
    p_long.add_argument("--no-save-cases", action="store_true", help="do not persist generated test cases separately")
    p_long.add_argument("--quiet", action="store_true", help="suppress periodic progress output")
    add_ablation_flags(p_long)
    add_paper_journal_flags(p_long)
    p_long.set_defaults(func=cmd_longrun)

    p_report = sub.add_parser("report", help="generate markdown/csv report")
    p_report.add_argument("--run-file", default=None)
    p_report.add_argument(
        "--csv-limit",
        type=int,
        default=None,
        help="maximum finding rows to export to CSV; useful for large longrun logs",
    )
    p_report.set_defaults(func=cmd_report)

    p_exp_summary = sub.add_parser("experiment-summary", help="summarize an experiment manifest")
    p_exp_summary.add_argument("--manifest", default=None)
    p_exp_summary.add_argument(
        "--refresh",
        action="store_true",
        help="recompute differential findings from stored normalized outputs or bug artifacts with the current oracle",
    )
    p_exp_summary.set_defaults(func=cmd_experiment_summary)

    p_exp_analysis = sub.add_parser(
        "analyze-experiment",
        help="compare experiment aggregate metrics against a baseline preset",
    )
    p_exp_analysis.add_argument("--manifest", default=None)
    p_exp_analysis.add_argument("--baseline-preset", default="baseline")
    p_exp_analysis.add_argument(
        "--compare-presets",
        default=None,
        help="optional comma-separated preset subset to compare against the baseline",
    )
    p_exp_analysis.add_argument(
        "--refresh",
        action="store_true",
        help="recompute experiment summary findings with the current oracle before analysis",
    )
    p_exp_analysis.set_defaults(func=cmd_analyze_experiment)

    p_seeded_analysis = sub.add_parser(
        "analyze-seeded-sensitivity",
        help="analyze expected-root detection for seeded fault experiments",
    )
    p_seeded_analysis.add_argument("--manifest", default=None)
    p_seeded_analysis.set_defaults(func=cmd_analyze_seeded_sensitivity)

    p_ablation_audit = sub.add_parser(
        "analyze-ablation-audit",
        help="audit candidate families and false positives introduced by ablation presets",
    )
    p_ablation_audit.add_argument("--manifest", default=None)
    p_ablation_audit.add_argument(
        "--trusted-presets",
        default=None,
        help="comma-separated presets considered part of the default soundness boundary",
    )
    p_ablation_audit.add_argument(
        "--ablation-presets",
        default=None,
        help="comma-separated weakened presets whose candidates should not be counted without triage",
    )
    p_ablation_audit.add_argument(
        "--refresh",
        action="store_true",
        help="recompute experiment summary findings with the current oracle before auditing",
    )
    p_ablation_audit.set_defaults(func=cmd_analyze_ablation_audit)

    p_pattern_variants = sub.add_parser(
        "analyze-pattern-variants",
        help="analyze generated pattern variants and candidate findings in an experiment",
    )
    p_pattern_variants.add_argument("--manifest", default=None)
    p_pattern_variants.add_argument(
        "--pattern",
        choices=["null_agg_topk"],
        default="null_agg_topk",
    )
    p_pattern_variants.set_defaults(func=cmd_analyze_pattern_variants)

    p_show = sub.add_parser("show-bugs", help="print bug findings")
    p_show.add_argument("--run-file", default=None)
    p_show.add_argument("--limit", type=int, default=10)
    p_show.set_defaults(func=cmd_show_bugs)

    p_classify = sub.add_parser("classify-run", help="classify findings as bugs, semantic divergences, or false positives")
    p_classify.add_argument("--run-file", default=None)
    p_classify.add_argument("--limit", type=int, default=3, help="examples per verdict")
    p_classify.add_argument(
        "--refresh",
        action="store_true",
        help="recompute differential findings from stored normalized outputs with the current oracle",
    )
    p_classify.set_defaults(func=cmd_classify_run)

    p_repro = sub.add_parser("reproduce", help="show reproduce command for a bug artifact")
    p_repro.add_argument("--bug", required=True)
    p_repro.add_argument("--backends", default=None)
    p_repro.add_argument("--print-command", action="store_true")
    p_repro.set_defaults(func=cmd_reproduce)

    p_validate = sub.add_parser("validate-artifact", help="rerun a bug artifact and check finding preservation")
    p_validate.add_argument("--bug", required=True)
    p_validate.add_argument("--backends", default=None)
    p_validate.set_defaults(func=cmd_validate_artifact)

    p_triage = sub.add_parser("triage-artifact", help="classify a reproduced artifact for paper use")
    p_triage.add_argument("--bug", required=True)
    p_triage.add_argument("--backends", default=None)
    p_triage.add_argument("--reduce", action="store_true")
    p_triage.add_argument(
        "--reduce-ignore-roots",
        action="store_true",
        help="when reducing, preserve finding kind only instead of the original root-cause label",
    )
    p_triage.add_argument(
        "--standalone-reproducer",
        action="store_true",
        help="also write an optional standalone diagnostic script when this root cause is supported",
    )
    p_triage.set_defaults(func=cmd_triage_artifact)

    p_reduce = sub.add_parser("reduce", help="minimize a bug artifact while preserving findings")
    p_reduce.add_argument("--bug", required=True)
    p_reduce.add_argument("--backends", default="pandas,polars,duckdb,sqlite")
    p_reduce.add_argument(
        "--ignore-roots",
        action="store_true",
        help="preserve finding kind only instead of the original root-cause label",
    )
    p_reduce.set_defaults(func=cmd_reduce)

    p_hist = sub.add_parser("historical-status", help="show historical replay registry admission status")
    p_hist.add_argument(
        "--include-pending",
        action="store_true",
        help="include candidate and pending historical case studies",
    )
    p_hist.add_argument("--json", action="store_true", help="emit historical registry status as JSON")
    p_hist.set_defaults(func=cmd_historical_status)

    p_fixture = sub.add_parser(
        "replay-fixture",
        help="run a declared fixture-backed case through normal differential oracle and journal recording",
    )
    p_fixture.add_argument("--spec", required=True, help="fixture replay spec JSON")
    p_fixture.add_argument("--fixture", default=None, help="path to the external fixture data file")
    p_fixture.add_argument(
        "--fixture-env",
        default=None,
        help="environment variable containing the external fixture data file path",
    )
    p_fixture.add_argument("--backends", default=None, help="explicit comma-separated backend targets")
    p_fixture.add_argument(
        "--target-suite",
        choices=sorted(TARGET_SUITES),
        default=None,
        help="backend target suite used when --backends is not provided; defaults to the spec target_suite",
    )
    p_fixture.add_argument(
        "--evidence-mode",
        choices=["live", "historical", "seeded"],
        default="historical",
        help="paper evidence layer for this replay",
    )
    p_fixture.add_argument("--known-bug-id", default="", help="historical bug id recorded in the run journal")
    p_fixture.add_argument("--target-version", default="", help="target dependency version or commit under replay")
    p_fixture.add_argument("--run-theme", default="", help="short paper-facing run theme")
    p_fixture.add_argument("--paper-notes", default="", help="brief paper-facing run notes")
    p_fixture.add_argument("--artifact-limit", type=int, default=None, help="0 disables artifact writes")
    p_fixture.add_argument("--log-level", choices=["full", "compact", "minimal"], default="compact")
    p_fixture.add_argument("--disable-artifact", action="store_true")
    p_fixture.add_argument("--no-compress-run-log", action="store_true")
    p_fixture.set_defaults(func=cmd_replay_fixture)

    p_exp = sub.add_parser("experiment", help="run repeatable ablation experiment matrix")
    p_exp.add_argument("--cases", type=int, default=None, help="maximum cases; defaults to 100 when --duration is absent")
    p_exp.add_argument("--duration", default=None, help="optional per-run wall-clock budget such as 10s, 5m, 24h")
    p_exp.add_argument("--seeds", default="1,1001,2001")
    p_exp.add_argument(
        "--evidence-mode",
        choices=["auto", "live", "historical", "seeded"],
        default="auto",
        help="experiment evidence layer: live latest-version finding, historical fixed-bug replay, or seeded fault ablation",
    )
    p_exp.add_argument(
        "--known-bug-id",
        default="",
        help="identifier for a replayed historical/upstream bug when --evidence-mode=historical",
    )
    p_exp.add_argument(
        "--target-version",
        default="",
        help="target dependency version or commit used by a historical replay run",
    )
    add_target_suite_flags(p_exp)
    p_exp.add_argument(
        "--target-suites",
        default=None,
        help="comma-separated backend target suites to run as an extra experiment dimension",
    )
    p_exp.add_argument(
        "--artifact-limit",
        type=int,
        default=None,
        help="maximum bug artifact directories to write for each preset run",
    )
    p_exp.add_argument("--log-level", choices=["full", "compact", "minimal"], default="compact")
    p_exp.add_argument(
        "--jobs",
        type=_parse_jobs,
        default="auto",
        help="number of experiment matrix runs to execute in parallel, or 'auto'",
    )
    p_exp.add_argument(
        "--max-parallel-cost",
        type=float,
        default=None,
        help="cost-token budget for concurrently running experiment jobs; defaults to a CPU-based budget",
    )
    p_exp.add_argument(
        "--schedule",
        choices=["matrix_order", "longest_first", "adaptive"],
        default=None,
        help="experiment scheduler; adaptive shares the matrix budget across runs",
    )
    p_exp.add_argument(
        "--batch-cases",
        type=int,
        default=None,
        help="adaptive-schedule batch size in cases; ignored by static schedules",
    )
    p_exp.add_argument(
        "--batch-duration",
        default=None,
        help="adaptive-schedule batch wall-clock budget such as 30s; ignored by static schedules",
    )
    p_exp.add_argument(
        "--warmup-batches",
        type=int,
        default=1,
        help="minimum adaptive batches to allocate to each arm before exploitation",
    )
    p_exp.add_argument(
        "--exploration-weight",
        type=float,
        default=0.75,
        help="adaptive scheduler exploration weight",
    )
    p_exp.add_argument(
        "--enable-local-source-scheduler",
        action="store_true",
        help="enable within-run generated-vs-feedback source scheduling for non-adaptive experiment jobs",
    )
    p_exp.add_argument(
        "--local-source-exploration-weight",
        type=float,
        default=0.5,
        help="exploration weight for the within-run generated-vs-feedback source scheduler",
    )
    p_exp.add_argument(
        "--metamorphic-variant-limit",
        type=int,
        default=None,
        help="maximum metamorphic variants to execute per base case",
    )
    p_exp.add_argument("--no-compress-run-log", action="store_true")
    p_exp.add_argument(
        "--skip-run-reports",
        action="store_true",
        help="do not write per-run markdown/csv reports during the matrix; use experiment-summary after completion",
    )
    p_exp.add_argument(
        "--presets",
        default="baseline,no_type_aware,no_normalizer,no_feedback,metamorphic,reducer",
        help="comma-separated presets",
    )
    add_paper_journal_flags(p_exp)
    p_exp.set_defaults(func=cmd_experiment)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
