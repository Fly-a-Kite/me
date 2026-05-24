from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

from datadiff.classification_oracle import annotate_findings
from datadiff.config import ExperimentConfig
from datadiff.dsl import Case
from datadiff.normalizer import NormalizedResult
from datadiff.oracle import evaluate_case
from datadiff.util import REPORTS_DIR, RUNS_DIR, ensure_dirs, jsonl_log_stem, load_json, read_jsonl, run_meta_path


def latest_run_file() -> Path:
    files = sorted([*RUNS_DIR.glob("run-*.jsonl"), *RUNS_DIR.glob("run-*.jsonl.gz")])
    if not files:
        raise FileNotFoundError(f"no run logs found in {RUNS_DIR}")
    return files[-1]


def write_report(run_file: Path | None = None, csv_limit: int | None = None) -> tuple[Path, Path]:
    ensure_dirs()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    run_file = run_file or latest_run_file()
    rows = read_jsonl(run_file)
    run_stem = jsonl_log_stem(run_file)
    md_path = REPORTS_DIR / f"report-{run_stem}.md"
    csv_path = REPORTS_DIR / f"findings-{run_stem}.csv"

    status_counts = Counter(r.get("status", "unknown") for r in rows)
    finding_kinds = Counter()
    root_causes = Counter()
    oracle_counts = Counter()
    confidence_counts = Counter()
    triage_verdicts = Counter()
    discovery_origins = Counter()
    candidate_bug_families = Counter()
    false_positive_reasons = Counter()
    backend_status = Counter()
    quality_oracle_verdicts = Counter()
    quality_oracle_pass_fail = Counter()
    signatures = Counter(r.get("behavior_signature", "") for r in rows)
    examples: dict[str, list[dict]] = defaultdict(list)
    meta_path = run_meta_path(run_file)
    meta = load_json(meta_path) if meta_path.exists() else {}
    target_specs = meta.get("targets") or (rows[0].get("targets", []) if rows else [])
    target_families = Counter(target.get("family", "unknown") for target in target_specs)
    target_layers = Counter(target.get("layer", "unknown") for target in target_specs)
    common_target_capabilities = meta.get("common_capabilities") or _common_capabilities_from_specs(target_specs)

    for row in rows:
        if row.get("normalized"):
            for backend, norm in row.get("normalized", {}).items():
                backend_status[f"{backend}:{norm.get('status')}"] += 1
        else:
            for backend, status in row.get("backend_status", {}).items():
                backend_status[f"{backend}:{status}"] += 1
        for oracle in row.get("quality_oracles", []):
            quality_oracle_verdicts[f"{oracle.get('name', 'unknown')}:{oracle.get('verdict', 'unknown')}"] += 1
            outcome = "passed" if oracle.get("passed") else "failed"
            quality_oracle_pass_fail[f"{oracle.get('name', 'unknown')}:{outcome}"] += 1
        candidate_bug_families.update(_candidate_bug_family_keys(row.get("findings", [])))
        for finding in row.get("findings", []):
            finding_kinds[finding["kind"]] += 1
            root_causes[finding.get("root_cause", "unknown")] += 1
            oracle_counts[finding.get("oracle", "unknown")] += 1
            confidence_counts[finding.get("confidence", "unknown")] += 1
            triage_verdicts[finding.get("triage_verdict", "unclassified")] += 1
            discovery_origins[finding.get("discovery_origin", "legacy") or "legacy"] += 1
            if finding.get("false_positive_reason"):
                false_positive_reasons[finding.get("false_positive_reason", "")] += 1
            if len(examples[finding["kind"]]) < 3:
                examples[finding["kind"]].append(
                    {
                        "case_id": row["case"]["case_id"],
                        "seed": row["case"]["seed"],
                        "evidence": finding["evidence"],
                        "root_cause": finding.get("root_cause", "unknown"),
                        "oracle": finding.get("oracle", "unknown"),
                        "confidence": finding.get("confidence", "unknown"),
                        "triage_verdict": finding.get("triage_verdict", "unclassified"),
                        "bug_dir": row.get("bug_dir", ""),
                    }
                )

    total = len(rows)
    bug_rows = [r for r in rows if r.get("findings")]
    lines = [
        "# DataDiffFuzz Report",
        "",
        f"- Run log: `{run_file}`",
        f"- Cases: {total}",
        f"- Bug-triggering cases: {len(bug_rows)}",
        f"- Bug rate: {len(bug_rows) / total:.1%}" if total else "- Bug rate: n/a",
        f"- Unique behavior signatures: {len(signatures)}",
        f"- New behavior cases: {sum(1 for r in rows if r.get('is_new_behavior'))}",
        f"- Elapsed seconds: {meta.get('elapsed_s', 'n/a')}",
        f"- Throughput cases/s: {meta.get('throughput_cases_s', 'n/a')}",
        f"- Backends: {', '.join(meta.get('backends', [])) if meta.get('backends') else 'n/a'}",
        f"- Target families: {_counter_summary(target_families)}",
        f"- Target layers: {_counter_summary(target_layers)}",
        f"- Common target capabilities: {len(common_target_capabilities)}",
        f"- Findings CSV limit: {'none' if csv_limit is None else max(0, csv_limit)}",
        "",
        "## Status",
    ]
    for key, count in status_counts.most_common():
        lines.append(f"- {key}: {count}")
    lines.append("")
    lines.append("## Backend Status")
    for key, count in backend_status.most_common():
        lines.append(f"- {key}: {count}")
    lines.append("")
    lines.append("## Finding Kinds")
    if finding_kinds:
        for key, count in finding_kinds.most_common():
            lines.append(f"- {key}: {count}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Root Causes")
    if root_causes:
        for key, count in root_causes.most_common():
            lines.append(f"- {key}: {count}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Oracle Sources")
    if oracle_counts:
        for key, count in oracle_counts.most_common():
            lines.append(f"- {key}: {count}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Confidence")
    if confidence_counts:
        for key, count in confidence_counts.most_common():
            lines.append(f"- {key}: {count}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Triage Verdicts")
    if triage_verdicts:
        for key, count in triage_verdicts.most_common():
            lines.append(f"- {key}: {count}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Discovery Origins")
    if discovery_origins:
        for key, count in discovery_origins.most_common():
            lines.append(f"- {key}: {count}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Candidate Bug Families")
    if candidate_bug_families:
        for key, count in candidate_bug_families.most_common():
            lines.append(f"- {key}: {count}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## False Positive Reasons")
    if false_positive_reasons:
        for key, count in false_positive_reasons.most_common():
            lines.append(f"- {key}: {count}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Target Capability Intersection")
    if common_target_capabilities:
        for capability in common_target_capabilities:
            lines.append(f"- {capability}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Quality Oracles")
    if quality_oracle_verdicts:
        lines.append("Verdicts:")
        for key, count in quality_oracle_verdicts.most_common():
            lines.append(f"- {key}: {count}")
        lines.append("")
        lines.append("Pass/fail:")
        for key, count in quality_oracle_pass_fail.most_common():
            lines.append(f"- {key}: {count}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Representative Evidence")
    if not examples:
        lines.append("- No findings.")
    for kind, items in examples.items():
        lines.append(f"### {kind}")
        for item in items:
            lines.append(
                f"- `{item['case_id']}` seed={item['seed']} "
                f"root={item['root_cause']} oracle={item['oracle']} confidence={item['confidence']} "
                f"triage={item['triage_verdict']}: "
                f"{item['evidence']} "
                f"artifact=`{item['bug_dir']}`"
            )
        lines.append("")
    if not examples:
        lines.append("")
    lines.append("## Ablation-Oriented Interpretation")
    lines.append("")
    lines.append(
        "Each row records the active modules in `config`: type-aware generation, "
        "semantic normalization, differential oracle, feedback, reducer, and artifact generation. "
        "Run the same seed range with modules disabled to quantify their contribution."
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "seed",
                "kind",
                "severity",
                "root_cause",
                "oracle",
                "confidence",
                "triage_verdict",
                "paper_status",
                "triage_confidence",
                "false_positive",
                "false_positive_reason",
                "suspicious_backends",
                "signature",
                "evidence",
                "triage_evidence",
                "bug_dir",
            ],
        )
        writer.writeheader()
        written = 0
        max_rows = None if csv_limit is None else max(0, int(csv_limit))
        for row in rows:
            for finding in row.get("findings", []):
                if max_rows is not None and written >= max_rows:
                    break
                writer.writerow(
                    {
                        "case_id": row["case"]["case_id"],
                        "seed": row["case"]["seed"],
                        "kind": finding["kind"],
                        "severity": finding["severity"],
                        "root_cause": finding.get("root_cause", "unknown"),
                        "oracle": finding.get("oracle", "unknown"),
                        "confidence": finding.get("confidence", "unknown"),
                        "triage_verdict": finding.get("triage_verdict", "unclassified"),
                        "paper_status": finding.get("paper_status", "unclassified"),
                        "triage_confidence": finding.get("triage_confidence", "low"),
                        "false_positive": finding.get("false_positive", False),
                        "false_positive_reason": finding.get("false_positive_reason", ""),
                        "suspicious_backends": ",".join(finding.get("suspicious_backends", [])),
                        "signature": finding["signature"],
                        "evidence": finding["evidence"],
                        "triage_evidence": finding.get("triage_evidence", ""),
                        "bug_dir": row.get("bug_dir", ""),
                    }
                )
                written += 1
            if max_rows is not None and written >= max_rows:
                break
    return md_path, csv_path


def latest_experiment_manifest() -> Path:
    files = sorted(RUNS_DIR.glob("experiment-*.json"))
    if not files:
        raise FileNotFoundError(f"no experiment manifests found in {RUNS_DIR}")
    return files[-1]


def write_experiment_summary(manifest_file: Path | None = None, *, refresh: bool = False) -> tuple[Path, Path]:
    ensure_dirs()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_file = manifest_file or latest_experiment_manifest()
    manifest = load_json(manifest_file)
    md_path = REPORTS_DIR / f"experiment-summary-{manifest_file.stem}.md"
    csv_path = REPORTS_DIR / f"experiment-summary-{manifest_file.stem}.csv"
    aggregate_csv_path = REPORTS_DIR / f"experiment-summary-{manifest_file.stem}-aggregates.csv"

    rows = []
    for run in manifest.get("runs", []):
        run_file = Path(run["run_file"])
        run_rows = read_jsonl(run_file)
        if refresh:
            run_rows = [_refresh_summary_row_findings(row) for row in run_rows]
        meta_path = run_meta_path(run_file)
        meta = load_json(meta_path) if meta_path.exists() else {}
        preflight = meta.get("preflight", {}) if isinstance(meta.get("preflight", {}), dict) else {}
        findings = [finding for row in run_rows for finding in row.get("findings", [])]
        unique_finding_signatures = {f.get("signature", "") for f in findings}
        root_causes = Counter(f.get("root_cause", "unknown") for f in findings)
        finding_kinds = Counter(f.get("kind", "unknown") for f in findings)
        triage_verdicts = Counter(f.get("triage_verdict", "unclassified") for f in findings)
        discovery_origins = Counter(f.get("discovery_origin", "legacy") or "legacy" for f in findings)
        issue_replay_candidate_count = sum(
            1 for finding in findings if _is_candidate_bug_finding(finding) and _is_issue_replay_finding(finding)
        )
        rewardable_candidate_count = sum(1 for finding in findings if _is_rewardable_candidate_bug_finding(finding))
        candidate_bug_families = Counter()
        for row in run_rows:
            candidate_bug_families.update(_candidate_bug_family_keys(row.get("findings", [])))
        target_families = Counter(target.get("family", "unknown") for target in meta.get("targets", []))
        total = len(run_rows)
        bug_cases = sum(1 for row in run_rows if row.get("findings"))
        candidate_bug_cases = sum(
            1
            for row in run_rows
            if any(
                _is_candidate_bug_finding(finding)
                for finding in row.get("findings", [])
            )
        )
        first_finding_case_index = _first_case_index(run_rows, lambda finding: True)
        first_candidate_bug_case_index = _first_case_index(
            run_rows,
            _is_candidate_bug_finding,
        )
        first_candidate_bug_elapsed_s = _first_case_elapsed_s(run_rows, _is_candidate_bug_finding)
        candidate_bug_discovery_auc = _candidate_bug_discovery_auc(run_rows)
        guidance_metrics = _guidance_metrics(run_rows)
        new_behavior_cases = sum(1 for row in run_rows if row.get("is_new_behavior"))
        rows.append(
            {
                "target_suite": run.get("target_suite", manifest.get("target_suite", "")),
                "preset": run.get("preset", ""),
                "seed": run.get("seed", ""),
                "evidence_mode": run.get("evidence_mode", manifest.get("evidence_mode", "")),
                "known_bug_id": run.get("known_bug_id", manifest.get("known_bug_id", "")),
                "target_version": run.get("target_version", manifest.get("target_version", "")),
                "batch_index": run.get("batch_index", ""),
                "schedule_arm_id": run.get("schedule_arm_id", ""),
                "scheduler_reward": run.get("scheduler_reward", ""),
                "cases": total,
                "bug_cases": bug_cases,
                "bug_rate": bug_cases / total if total else 0.0,
                "candidate_bug_cases": candidate_bug_cases,
                "candidate_bug_case_rate": candidate_bug_cases / total if total else 0.0,
                "first_finding_case_index": first_finding_case_index,
                "first_candidate_bug_case_index": first_candidate_bug_case_index,
                "first_candidate_bug_elapsed_s": first_candidate_bug_elapsed_s,
                "candidate_bug_discovery_auc": candidate_bug_discovery_auc,
                "findings": len(findings),
                "unique_findings": len(unique_finding_signatures),
                "new_behavior_cases": new_behavior_cases,
                "new_behavior_rate": new_behavior_cases / total if total else 0.0,
                "preflight_repaired_cases": int(preflight.get("repaired_cases", 0) or 0),
                "preflight_fallback_cases": int(preflight.get("fallback_cases", 0) or 0),
                "preflight_invalid_cases": int(preflight.get("invalid_cases", 0) or 0),
                "preflight_repaired_rate": (int(preflight.get("repaired_cases", 0) or 0) / total if total else 0.0),
                "preflight_fallback_rate": (int(preflight.get("fallback_cases", 0) or 0) / total if total else 0.0),
                "preflight_invalid_rate": (int(preflight.get("invalid_cases", 0) or 0) / total if total else 0.0),
                "elapsed_s": meta.get("elapsed_s", ""),
                "throughput_cases_s": meta.get("throughput_cases_s", ""),
                **guidance_metrics,
                "top_root_causes": _counter_summary(root_causes),
                "top_finding_kinds": _counter_summary(finding_kinds),
                "top_triage_verdicts": _counter_summary(triage_verdicts),
                "top_discovery_origins": _counter_summary(discovery_origins),
                "candidate_bug_families": len(candidate_bug_families),
                "top_candidate_bug_families": _counter_summary(candidate_bug_families),
                "candidate_implementation_bug_count": triage_verdicts["candidate_implementation_bug"],
                "rewardable_candidate_implementation_bug_count": rewardable_candidate_count,
                "issue_replay_candidate_bug_count": issue_replay_candidate_count,
                "documented_semantic_divergence_count": triage_verdicts["documented_semantic_divergence"],
                "expected_semantic_divergence_count": triage_verdicts["expected_semantic_divergence"],
                "semantic_divergence_needs_confirmation_count": triage_verdicts[
                    "semantic_divergence_needs_confirmation"
                ],
                "generator_false_positive_count": triage_verdicts["generator_false_positive"],
                "normalizer_false_positive_count": triage_verdicts["normalizer_false_positive"],
                "needs_manual_confirmation_count": triage_verdicts["needs_manual_confirmation"],
                "target_families": _counter_summary(target_families),
                "backends": ",".join(run.get("backends", manifest.get("backends", []))),
                "run_file": str(run_file),
                "report": run.get("report", ""),
            }
        )

    lines = [
        "# DataDiffFuzz Experiment Summary",
        "",
        f"- Manifest: `{manifest_file}`",
        f"- Runs: {len(rows)}",
        f"- Presets: {', '.join(manifest.get('presets', []))}",
        f"- Seeds: {', '.join(str(s) for s in manifest.get('seeds', []))}",
        f"- Backends: {', '.join(manifest.get('backends', []))}",
        f"- Target suite: {manifest.get('target_suite', 'n/a')}",
        f"- Target suites: {', '.join(manifest.get('target_suites', [])) or manifest.get('target_suite', 'n/a')}",
        f"- Target families: {_counter_summary(Counter(target.get('family', 'unknown') for target in manifest.get('targets', [])))}",
        f"- Common target capabilities: {len(manifest.get('common_capabilities', []))}",
        f"- Schedule: {manifest.get('schedule', 'matrix_order')}",
        f"- Evidence mode: {manifest.get('evidence_mode', 'live')}",
        f"- Known bug id: {manifest.get('known_bug_id', '') or 'n/a'}",
        f"- Target version: {manifest.get('target_version', '') or 'n/a'}",
        f"- Local source scheduler: {manifest.get('local_source_scheduler', {'enabled': False, 'exploration_weight': ''})}",
        f"- Aggregate CSV: `{aggregate_csv_path}`",
        "- Triage columns distinguish candidate implementation bugs from documented/expected semantic divergences and oracle false positives.",
        f"- Refreshed with current oracle: {'yes' if refresh else 'no'}",
        "",
        "## Runs",
        "",
        "| target suite | preset | seed | batch | arm | reward | cases | findings | candidate bugs | rewardable candidates | issue replays | candidate case % | first candidate | first candidate s | discovery AUC | semantic divs | false positives | new behavior % | cases/s | data sensitivity | path proxy | frontier | contribution | pruned % | roots | triage | origins |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        semantic_divergence_count = (
            row["documented_semantic_divergence_count"]
            + row["expected_semantic_divergence_count"]
            + row["semantic_divergence_needs_confirmation_count"]
        )
        false_positive_count = row["generator_false_positive_count"] + row["normalizer_false_positive_count"]
        lines.append(
            "| {target_suite} | {preset} | {seed} | {batch_index} | {schedule_arm_id} | {scheduler_reward} | {cases} | {findings} | "
            "{candidate_implementation_bug_count} | {rewardable_candidate_implementation_bug_count} | {issue_replay_candidate_bug_count} | {candidate_bug_case_rate} | "
            "{first_candidate_bug_case_index} | {first_candidate_bug_elapsed_s} | {candidate_bug_discovery_auc} | {semantic_divergence_count} | "
            "{false_positive_count} | {new_behavior_rate} | {throughput_cases_s} | {avg_data_sensitivity} | "
            "{avg_path_coverage_proxy} | {avg_frontier_conformance} | "
            "{avg_contribution_potential} | {pruned_candidate_rate} | {top_root_causes} | {top_triage_verdicts} | {top_discovery_origins} |".format(
                **{
                    **row,
                    "semantic_divergence_count": semantic_divergence_count,
                    "false_positive_count": false_positive_count,
                    "batch_index": _fmt_optional_int(row["batch_index"]),
                    "scheduler_reward": _fmt_optional_float(row["scheduler_reward"]),
                    "candidate_bug_case_rate": _fmt_percent(row["candidate_bug_case_rate"]),
                    "first_candidate_bug_case_index": _fmt_optional_int(row["first_candidate_bug_case_index"]),
                    "first_candidate_bug_elapsed_s": _fmt_optional_float(row["first_candidate_bug_elapsed_s"]),
                    "candidate_bug_discovery_auc": _fmt_float(row["candidate_bug_discovery_auc"]),
                    "new_behavior_rate": _fmt_percent(row["new_behavior_rate"]),
                    "throughput_cases_s": _fmt_float(row["throughput_cases_s"]),
                    "avg_data_sensitivity": _fmt_float(row["avg_data_sensitivity"]),
                    "avg_path_coverage_proxy": _fmt_float(row["avg_path_coverage_proxy"]),
                    "avg_frontier_conformance": _fmt_float(row["avg_frontier_conformance"]),
                    "avg_contribution_potential": _fmt_float(row["avg_contribution_potential"]),
                    "pruned_candidate_rate": _fmt_percent(row["pruned_candidate_rate"]),
                }
            )
        )
    aggregate_rows = _aggregate_experiment_rows(rows)
    lines.extend(
        [
            "",
            "## Aggregates",
            "",
            "| target suite | preset | runs | cases | findings | candidate bugs | rewardable candidates | issue replays | candidate case % | candidate cases/s | median first candidate | median first s | avg discovery AUC | avg reward | semantic divs | false positives | avg new behavior % | avg cases/s | avg data sensitivity | avg path proxy |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in aggregate_rows:
        lines.append(
            "| {target_suite} | {preset} | {runs} | {cases} | {findings} | "
            "{candidate_implementation_bug_count} | {rewardable_candidate_implementation_bug_count} | {issue_replay_candidate_bug_count} | {candidate_bug_case_rate} | "
            "{candidate_bug_cases_per_s} | {median_first_candidate_bug_case_index} | "
            "{median_first_candidate_bug_elapsed_s} | {avg_candidate_bug_discovery_auc} | {avg_scheduler_reward} | "
            "{semantic_divergence_count} | {false_positive_count} | {avg_new_behavior_rate} | "
            "{avg_throughput_cases_s} | {avg_data_sensitivity} | {avg_path_coverage_proxy} |".format(
                **{
                    **row,
                    "candidate_bug_case_rate": _fmt_percent(row["candidate_bug_case_rate"]),
                    "candidate_bug_cases_per_s": _fmt_float(row["candidate_bug_cases_per_s"]),
                    "median_first_candidate_bug_case_index": _fmt_optional_number(
                        row["median_first_candidate_bug_case_index"]
                    ),
                    "median_first_candidate_bug_elapsed_s": _fmt_optional_number(
                        row["median_first_candidate_bug_elapsed_s"]
                    ),
                    "avg_candidate_bug_discovery_auc": _fmt_float(row["avg_candidate_bug_discovery_auc"]),
                    "avg_scheduler_reward": _fmt_optional_float(row["avg_scheduler_reward"]),
                    "avg_new_behavior_rate": _fmt_percent(row["avg_new_behavior_rate"]),
                    "avg_throughput_cases_s": _fmt_float(row["avg_throughput_cases_s"]),
                    "avg_data_sensitivity": _fmt_float(row["avg_data_sensitivity"]),
                    "avg_path_coverage_proxy": _fmt_float(row["avg_path_coverage_proxy"]),
                }
            )
        )
    lines.extend(
        [
            "",
            "## Preflight Integrity",
            "",
            "| target suite | preset | runs | cases | invalid % | repaired % | fallback % |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in aggregate_rows:
        lines.append(
            "| {target_suite} | {preset} | {runs} | {cases} | {preflight_invalid_rate} | {preflight_repaired_rate} | {preflight_fallback_rate} |".format(
                **{
                    **row,
                    "preflight_invalid_rate": _fmt_percent(row["preflight_invalid_rate"]),
                    "preflight_repaired_rate": _fmt_percent(row["preflight_repaired_rate"]),
                    "preflight_fallback_rate": _fmt_percent(row["preflight_fallback_rate"]),
                }
            )
        )
    lines.extend(
        [
            "",
            "## Candidate Bug-Family Deduplication",
            "",
            "Families are conservatively grouped by `root_cause + suspicious_backends`; metamorphic relation failures are mapped to a co-occurring differential root for the same suspicious backend when one exists. Use this table for paper-level bug-family counts; use finding and case counts above for sensitivity and time-to-first measurements.",
            "",
            "| target suite | preset | candidate bug families | top families |",
            "|---|---|---:|---|",
        ]
    )
    for row in aggregate_rows:
        lines.append(
            "| {target_suite} | {preset} | {candidate_bug_families} | {top_candidate_bug_families} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- Compare `baseline` with `no_type_aware` to measure generator validity and useful behavior discovery.",
            "- Compare `baseline` with `no_normalizer` to estimate false-positive pressure from representation differences.",
            "- Compare `baseline` with `no_feedback` to measure whether corpus feedback improves new behavior discovery.",
            "- Compare `baseline` with `metamorphic` to separate cross-engine differential bugs from single-engine relation violations.",
            "- Use `edge_float` separately from `common`; it studies boundary semantics rather than the default common subset.",
        ]
    )
    if manifest.get("schedule") == "adaptive":
        adaptive_summary = _adaptive_schedule_summary(rows)
        lines.extend(
            [
                "",
                "## Adaptive Schedule",
                "",
                f"- Config: `{manifest.get('adaptive_config', {})}`",
                f"- Final arm state entries: {len(manifest.get('adaptive_state', []))}",
                f"- Global first candidate case: {_fmt_optional_int(adaptive_summary['global_first_candidate_case'])}",
                f"- Batches by suite: {adaptive_summary['batches_by_suite']}",
                f"- Cases by suite: {adaptive_summary['cases_by_suite']}",
                f"- Candidate cases by suite: {adaptive_summary['candidate_cases_by_suite']}",
            ]
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "target_suite",
                "preset",
                "seed",
                "evidence_mode",
                "known_bug_id",
                "target_version",
                "batch_index",
                "schedule_arm_id",
                "scheduler_reward",
                "cases",
                "bug_cases",
                "bug_rate",
                "candidate_bug_cases",
                "candidate_bug_case_rate",
                "first_finding_case_index",
                "first_candidate_bug_case_index",
                "first_candidate_bug_elapsed_s",
                "candidate_bug_discovery_auc",
                "findings",
                "unique_findings",
                "new_behavior_cases",
                "new_behavior_rate",
                "preflight_repaired_cases",
                "preflight_fallback_cases",
                "preflight_invalid_cases",
                "preflight_repaired_rate",
                "preflight_fallback_rate",
                "preflight_invalid_rate",
                "elapsed_s",
                "throughput_cases_s",
                "avg_guidance_score",
                "avg_data_sensitivity",
                "avg_path_coverage_proxy",
                "avg_frontier_conformance",
                "avg_contribution_potential",
                "avg_candidate_count",
                "avg_contributing_candidate_count",
                "avg_pruned_candidate_count",
                "pruned_candidate_rate",
                "avg_feature_count",
                "avg_frontier_bucket_count",
                "top_root_causes",
                "top_finding_kinds",
                "top_triage_verdicts",
                "top_discovery_origins",
                "candidate_bug_families",
                "top_candidate_bug_families",
                "candidate_implementation_bug_count",
                "rewardable_candidate_implementation_bug_count",
                "issue_replay_candidate_bug_count",
                "documented_semantic_divergence_count",
                "expected_semantic_divergence_count",
                "semantic_divergence_needs_confirmation_count",
                "generator_false_positive_count",
                "normalizer_false_positive_count",
                "needs_manual_confirmation_count",
                "target_families",
                "backends",
                "run_file",
                "report",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with aggregate_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "target_suite",
                "preset",
                "runs",
                "cases",
                "findings",
                "candidate_implementation_bug_count",
                "rewardable_candidate_implementation_bug_count",
                "issue_replay_candidate_bug_count",
                "candidate_bug_cases",
                "candidate_bug_case_rate",
                "candidate_bug_cases_per_s",
                "median_first_candidate_bug_case_index",
                "median_first_candidate_bug_elapsed_s",
                "avg_candidate_bug_discovery_auc",
                "avg_scheduler_reward",
                "preflight_repaired_cases",
                "preflight_fallback_cases",
                "preflight_invalid_cases",
                "preflight_repaired_rate",
                "preflight_fallback_rate",
                "preflight_invalid_rate",
                "semantic_divergence_count",
                "false_positive_count",
                "avg_new_behavior_rate",
                "avg_throughput_cases_s",
                "avg_data_sensitivity",
                "avg_path_coverage_proxy",
                "candidate_bug_families",
                "top_candidate_bug_families",
            ],
        )
        writer.writeheader()
        writer.writerows(aggregate_rows)

    return md_path, csv_path


def _counter_summary(counter: Counter) -> str:
    return "; ".join(f"{key}:{count}" for key, count in counter.most_common(5)) or "none"


def _parse_counter_summary(value: str) -> Counter:
    out: Counter = Counter()
    if not value or value == "none":
        return out
    for item in value.split("; "):
        if ":" not in item:
            continue
        key, count = item.rsplit(":", 1)
        try:
            out[key] += int(count)
        except ValueError:
            continue
    return out


def _refresh_summary_row_findings(row: dict) -> dict:
    refreshed = _recompute_differential_findings(row)
    if refreshed is None:
        return row
    case, normalized, findings, config, backends = refreshed
    annotate_findings(
        case,
        findings,
        normalized,
        row.get("raw_results", {}),
        config,
        backends,
    )
    out = dict(row)
    out["findings"] = [finding.to_dict() for finding in findings]
    return out


def _recompute_differential_findings(
    row: dict,
) -> tuple[Case, dict[str, NormalizedResult], list, dict, list[str]] | None:
    if row.get("normalized"):
        try:
            case = Case.from_dict(row["case"])
        except Exception:
            return None
        normalized = _normalized_from_mapping(row.get("normalized", {}))
        config = row.get("config", {})
        return case, normalized, evaluate_case(case, normalized), config, list(normalized)

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
    config = load_json(config_path) if config_path.exists() else ExperimentConfig().to_dict()
    return case, normalized, evaluate_case(case, normalized), config, list(normalized)


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


def _candidate_bug_family_keys(findings: list[dict]) -> Counter:
    keys: Counter = Counter()
    root_by_suspicious: dict[str, str] = {}
    for finding in findings:
        if not _is_rewardable_candidate_bug_finding(finding):
            continue
        root = str(finding.get("root_cause", "unknown"))
        if root.startswith("metamorphic_"):
            continue
        suspicious = _suspicious_key(finding)
        root_by_suspicious.setdefault(suspicious, root)
    for finding in findings:
        if not _is_rewardable_candidate_bug_finding(finding):
            continue
        root = str(finding.get("root_cause", "unknown"))
        suspicious = _suspicious_key(finding)
        if root.startswith("metamorphic_") and suspicious in root_by_suspicious:
            root = root_by_suspicious[suspicious]
        keys[f"{root}@{suspicious}"] += 1
    return keys


def _is_candidate_bug_finding(finding: dict) -> bool:
    if finding.get("triage_verdict") != "candidate_implementation_bug":
        return False
    if finding.get("false_positive"):
        return False
    return True


def _is_issue_replay_finding(finding: dict) -> bool:
    return str(finding.get("discovery_origin", "")).strip() == "issue_replay"


def _is_rewardable_candidate_bug_finding(finding: dict) -> bool:
    return _is_candidate_bug_finding(finding) and not _is_issue_replay_finding(finding)


def _candidate_bug_family_key(finding: dict) -> str:
    return next(iter(_candidate_bug_family_keys([finding])), "")


def _suspicious_key(finding: dict) -> str:
    return ",".join(sorted(finding.get("suspicious_backends", []) or [])) or "unknown"


def _fmt_float(value) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.2f}"
    return str(value)


def _fmt_percent(value) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.1%}"
    return str(value)


def _fmt_optional_int(value) -> str:
    return "" if value is None else str(value)


def _fmt_optional_float(value) -> str:
    if value is None or value == "":
        return ""
    return _fmt_float(value)


def _fmt_optional_number(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def _first_case_index(rows: list[dict], predicate) -> int | None:
    for idx, row in enumerate(rows):
        for finding in row.get("findings", []):
            if predicate(finding):
                return int(row.get("case_index", idx))
    return None


def _first_case_elapsed_s(rows: list[dict], predicate) -> float | None:
    for row in rows:
        for finding in row.get("findings", []):
            if predicate(finding):
                value = row.get("elapsed_s")
                return float(value) if value not in (None, "") else None
    return None


def _candidate_bug_discovery_auc(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    hits = [
        int(any(_is_candidate_bug_finding(finding) for finding in row.get("findings", [])))
        for row in rows
    ]
    total = sum(hits)
    if total == 0:
        return 0.0
    cumulative = 0
    area = 0
    for hit in hits:
        cumulative += hit
        area += cumulative
    return area / (len(rows) * total)


def _aggregate_experiment_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["target_suite"]), str(row["preset"]))].append(row)

    out = []
    for (target_suite, preset), items in sorted(grouped.items()):
        cases = sum(int(row["cases"]) for row in items)
        findings = sum(int(row["findings"]) for row in items)
        candidate_count = sum(int(row["candidate_implementation_bug_count"]) for row in items)
        rewardable_candidate_count = sum(
            int(row.get("rewardable_candidate_implementation_bug_count", 0) or 0) for row in items
        )
        issue_replay_candidate_count = sum(
            int(row.get("issue_replay_candidate_bug_count", 0) or 0) for row in items
        )
        family_counter = Counter()
        for row in items:
            family_counter.update(_parse_counter_summary(str(row.get("top_candidate_bug_families", ""))))
        candidate_bug_cases = sum(int(row["candidate_bug_cases"]) for row in items)
        semantic_divergence_count = sum(
            int(row["documented_semantic_divergence_count"])
            + int(row["expected_semantic_divergence_count"])
            + int(row["semantic_divergence_needs_confirmation_count"])
            for row in items
        )
        false_positive_count = sum(
            int(row["generator_false_positive_count"]) + int(row["normalizer_false_positive_count"])
            for row in items
        )
        preflight_repaired_cases = sum(int(row.get("preflight_repaired_cases", 0) or 0) for row in items)
        preflight_fallback_cases = sum(int(row.get("preflight_fallback_cases", 0) or 0) for row in items)
        preflight_invalid_cases = sum(int(row.get("preflight_invalid_cases", 0) or 0) for row in items)
        avg_throughput = _avg_value(items, "throughput_cases_s")
        candidate_bug_case_rate = candidate_bug_cases / cases if cases else 0.0
        out.append(
            {
                "target_suite": target_suite,
                "preset": preset,
                "runs": len(items),
                "cases": cases,
                "findings": findings,
                "candidate_implementation_bug_count": candidate_count,
                "rewardable_candidate_implementation_bug_count": rewardable_candidate_count,
                "issue_replay_candidate_bug_count": issue_replay_candidate_count,
                "candidate_bug_families": len(family_counter),
                "top_candidate_bug_families": _counter_summary(family_counter),
                "candidate_bug_cases": candidate_bug_cases,
                "candidate_bug_case_rate": candidate_bug_case_rate,
                "candidate_bug_cases_per_s": candidate_bug_case_rate * avg_throughput,
                "median_first_candidate_bug_case_index": _median_optional_int(
                    row["first_candidate_bug_case_index"] for row in items
                ),
                "median_first_candidate_bug_elapsed_s": _median_optional_float(
                    row.get("first_candidate_bug_elapsed_s") for row in items
                ),
                "avg_candidate_bug_discovery_auc": _avg_value(items, "candidate_bug_discovery_auc"),
                "avg_scheduler_reward": _avg_value(items, "scheduler_reward"),
                "preflight_repaired_cases": preflight_repaired_cases,
                "preflight_fallback_cases": preflight_fallback_cases,
                "preflight_invalid_cases": preflight_invalid_cases,
                "preflight_repaired_rate": preflight_repaired_cases / cases if cases else 0.0,
                "preflight_fallback_rate": preflight_fallback_cases / cases if cases else 0.0,
                "preflight_invalid_rate": preflight_invalid_cases / cases if cases else 0.0,
                "semantic_divergence_count": semantic_divergence_count,
                "false_positive_count": false_positive_count,
                "avg_new_behavior_rate": _avg_value(items, "new_behavior_rate"),
                "avg_throughput_cases_s": avg_throughput,
                "avg_data_sensitivity": _avg_value(items, "avg_data_sensitivity"),
                "avg_path_coverage_proxy": _avg_value(items, "avg_path_coverage_proxy"),
            }
        )
    return out


def _adaptive_schedule_summary(rows: list[dict]) -> dict[str, object]:
    batches_by_suite = Counter()
    cases_by_suite = Counter()
    candidate_cases_by_suite = Counter()
    global_first_candidate_case: int | None = None
    offset = 0
    for row in rows:
        suite = str(row.get("target_suite", "unknown"))
        batches_by_suite[suite] += 1
        cases = int(row.get("cases", 0) or 0)
        cases_by_suite[suite] += cases
        candidate_cases = int(row.get("candidate_bug_cases", 0) or 0)
        candidate_cases_by_suite[suite] += candidate_cases
        first_candidate = row.get("first_candidate_bug_case_index")
        if global_first_candidate_case is None and first_candidate not in (None, ""):
            global_first_candidate_case = offset + int(first_candidate) + 1
        offset += cases
    return {
        "global_first_candidate_case": global_first_candidate_case,
        "batches_by_suite": _counter_summary(batches_by_suite),
        "cases_by_suite": _counter_summary(cases_by_suite),
        "candidate_cases_by_suite": _counter_summary(candidate_cases_by_suite),
    }


def _avg_value(rows: list[dict], key: str) -> float:
    values = [_float_or_none(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else 0.0


def _median_optional_int(values) -> float | None:
    ints = sorted(int(value) for value in values if value is not None and value != "")
    if not ints:
        return None
    middle = len(ints) // 2
    if len(ints) % 2 == 1:
        return float(ints[middle])
    return (ints[middle - 1] + ints[middle]) / 2.0


def _median_optional_float(values) -> float | None:
    floats = sorted(float(value) for value in values if value is not None and value != "")
    if not floats:
        return None
    middle = len(floats) // 2
    if len(floats) % 2 == 1:
        return floats[middle]
    return (floats[middle - 1] + floats[middle]) / 2.0


def _float_or_none(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _guidance_metrics(rows: list[dict]) -> dict[str, float]:
    guidance_rows = [row.get("guidance", {}) for row in rows if row.get("guidance")]
    return {
        "avg_guidance_score": _avg_metric(guidance_rows, "score"),
        "avg_data_sensitivity": _avg_metric(guidance_rows, "data_sensitivity"),
        "avg_path_coverage_proxy": _avg_metric(guidance_rows, "path_coverage_proxy"),
        "avg_frontier_conformance": _avg_metric(guidance_rows, "frontier_conformance"),
        "avg_contribution_potential": _avg_metric(guidance_rows, "contribution_potential"),
        "avg_candidate_count": _avg_metric(guidance_rows, "candidate_count"),
        "avg_contributing_candidate_count": _avg_metric(guidance_rows, "contributing_candidate_count"),
        "avg_pruned_candidate_count": _avg_metric(guidance_rows, "pruned_candidate_count"),
        "pruned_candidate_rate": _candidate_rate(guidance_rows, "pruned_candidate_count"),
        "avg_feature_count": _avg_metric(guidance_rows, "feature_count"),
        "avg_frontier_bucket_count": _avg_metric(guidance_rows, "frontier_bucket_count"),
    }


def _avg_metric(rows: list[dict], key: str) -> float:
    values = [float(row.get(key, 0.0) or 0.0) for row in rows]
    return sum(values) / len(values) if values else 0.0


def _candidate_rate(rows: list[dict], numerator_key: str) -> float:
    numerator = sum(float(row.get(numerator_key, 0.0) or 0.0) for row in rows)
    denominator = sum(float(row.get("candidate_count", 0.0) or 0.0) for row in rows)
    return numerator / denominator if denominator else 0.0


def _common_capabilities_from_specs(target_specs: list[dict]) -> list[str]:
    if not target_specs:
        return []
    capabilities = [set(target.get("capabilities", [])) for target in target_specs]
    common = capabilities[0]
    for item in capabilities[1:]:
        common &= item
    return sorted(common)
