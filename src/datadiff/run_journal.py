from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from datadiff.experiment_catalog import counting_policy_for_evidence_mode
from datadiff.experiment_metadata import (
    experiment_row_group_id,
    experiment_row_variant_label,
    manifest_experiment_meta,
    resolved_run_semantics,
)
from datadiff.reward import (
    candidate_issue_family_keys,
    is_rewardable_candidate_issue_finding,
)
from datadiff.util import (
    JsonlWriter,
    REPORTS_DIR,
    jsonl_log_stem,
    load_json,
    read_jsonl,
    run_meta_path,
    utc_now,
)

JOURNAL_SCHEMA_VERSION = 1
DEFAULT_JOURNAL_NAME = "paper-run-journal.jsonl"


def default_journal_path() -> Path:
    return REPORTS_DIR / DEFAULT_JOURNAL_NAME


def build_run_journal_entry(run_file: Path, context: dict[str, Any] | None = None) -> dict[str, Any]:
    context = dict(context or {})
    rows = read_jsonl(run_file) if run_file.exists() else []
    meta_path = run_meta_path(run_file)
    meta = load_json(meta_path) if meta_path.exists() else {}
    meta_with_context = dict(meta)
    if isinstance(context.get("experiment_meta"), dict) and context.get("experiment_meta"):
        meta_with_context["experiment_meta"] = context["experiment_meta"]
    experiment_meta = manifest_experiment_meta(meta_with_context)
    run_semantics = resolved_run_semantics(context, experiment_meta)
    config = meta.get("config", {}) if isinstance(meta.get("config", {}), dict) else {}
    known_bug_families = list(config.get("known_saturated_bug_families", []) or [])
    findings = [finding for row in rows for finding in row.get("findings", [])]
    candidate_families = _candidate_bug_family_keys(findings, known_bug_families)
    semantic_divergences = sum(_is_semantic_divergence(finding) for finding in findings)
    false_positives = sum(_is_false_positive(finding) for finding in findings)
    target_specs = meta.get("targets", [])
    target_families = Counter(target.get("family", "unknown") for target in target_specs)
    target_layers = Counter(target.get("layer", "unknown") for target in target_specs)
    replay_filter = (
        meta.get("replay_bug_filter", {}) if isinstance(meta.get("replay_bug_filter", {}), dict) else {}
    )
    evidence_mode = str(context.get("evidence_mode") or meta.get("evidence_mode") or "live")
    theme = str(context.get("theme") or meta.get("run_theme") or _default_theme(context, meta, run_file))
    executed_cases = int(meta.get("executed_cases", len(rows)))
    raw_new_behavior_cases = int(meta.get("new_behavior_cases", sum(1 for row in rows if row.get("is_new_behavior"))))
    signal_new_behavior_cases = int(
        meta.get(
            "signal_new_behavior_cases",
            sum(1 for row in rows if row.get("signal_new_behavior", row.get("is_new_behavior"))),
        )
    )

    return {
        "schema_version": JOURNAL_SCHEMA_VERSION,
        "recorded_at": utc_now(),
        "theme": theme,
        "notes": str(context.get("notes") or ""),
        "command": str(context.get("command") or ""),
        "evidence_mode": evidence_mode,
        "counting_policy": counting_policy_for_evidence_mode(evidence_mode),
        "known_bug_id": str(context.get("known_bug_id") or meta.get("known_bug_id") or ""),
        "target_version": str(context.get("target_version") or meta.get("target_version") or ""),
        "target_suite": str(context.get("target_suite") or meta.get("target_suite") or ""),
        "preset": str(context.get("preset") or meta.get("preset") or ""),
        "matrix_id": run_semantics["matrix_id"],
        "matrix_title": run_semantics["matrix_title"],
        "comparison_group": run_semantics["comparison_group"],
        "variant_id": run_semantics["variant_id"],
        "variant_label": experiment_row_variant_label(run_semantics),
        "variant_group_id": "|".join(experiment_row_group_id(run_semantics)),
        "base_preset": run_semantics["base_preset"],
        "comparison_role": run_semantics["comparison_role"],
        "canonical_comparison_role": run_semantics["canonical_comparison_role"],
        "component_focus": run_semantics["component_focus"],
        "semantic_focus_families": list(run_semantics["semantic_focus_families"]),
        "semantic_focus_signals": list(run_semantics["semantic_focus_signals"]),
        "scope_kind": run_semantics["scope_kind"],
        "oracle_profile": run_semantics["oracle_profile"],
        "rq_tags": list(run_semantics["rq_tags"]),
        "analysis_tags": list(run_semantics["analysis_tags"]),
        "counts_as_real_bugs": run_semantics["counts_as_real_bugs"],
        "seed": context.get("seed", meta.get("seed", "")),
        "run_file": str(run_file),
        "meta_file": str(meta_path) if meta_path.exists() else "",
        "manifest_file": str(context.get("manifest_file") or ""),
        "case_log_file": str(meta.get("case_log_file") or ""),
        "checkpoint_file": str(meta.get("checkpoint_file") or ""),
        "requested_cases": meta.get("requested_cases"),
        "executed_cases": executed_cases,
        "duration_s": meta.get("duration_s"),
        "elapsed_s": meta.get("elapsed_s", _last_elapsed_s(rows)),
        "throughput_cases_s": meta.get("throughput_cases_s", 0.0),
        "backends": list(meta.get("backends", context.get("backends", [])) or []),
        "target_families": dict(sorted(target_families.items())),
        "target_layers": dict(sorted(target_layers.items())),
        "common_capabilities_count": len(meta.get("common_capabilities", [])),
        "config_summary": {
            "generator_profile": config.get("generator_profile", context.get("profile", "")),
            "oracle_mode": config.get("oracle_mode", ""),
            "normalizer": config.get("enable_normalizer"),
            "type_aware_generation": config.get("enable_type_aware_generation"),
            "differential_oracle": config.get("enable_differential_oracle"),
            "metamorphic_oracle": config.get("enable_metamorphic_oracle"),
            "metamorphic_variant_limit": config.get("metamorphic_variant_limit"),
            "feedback": config.get("enable_feedback"),
            "enable_replay_bug": config.get("enable_replay_bug"),
            "reducer": config.get("enable_reducer"),
            "artifact_limit": config.get("artifact_limit"),
            "guidance_strategy": config.get("guidance_strategy"),
            "guidance_candidate_pool": config.get("guidance_candidate_pool"),
            "guidance_targets": config.get("guidance_targets", []),
            "effective_guidance_targets": config.get("effective_guidance_targets", []),
            "log_level": config.get("log_level", meta.get("log_level", "")),
        },
        "replay_bug_policy": {
            "enable_replay_bug": bool(config.get("enable_replay_bug", False)),
            "source_issue_count": len(config.get("replay_bug_source_issues", []) or []),
            "filter_enabled": replay_filter.get("enabled", ""),
            "filtered_candidates": int(replay_filter.get("filtered_candidates", 0) or 0),
            "fallback_candidates": int(replay_filter.get("fallback_candidates", 0) or 0),
        },
        "result_summary": {
            "raw_findings": len(findings),
            "bug_triggering_cases": sum(1 for row in rows if row.get("findings")),
            "candidate_bug_cases": sum(
                1
                for row in rows
                if any(
                    is_rewardable_candidate_issue_finding(finding, known_bug_families)
                    for finding in row.get("findings", [])
                )
            ),
            "candidate_bug_families": dict(candidate_families),
            "candidate_bug_family_count": len(candidate_families),
            "semantic_divergence_findings": semantic_divergences,
            "false_positive_findings": false_positives,
            "new_behavior_cases": raw_new_behavior_cases,
            "new_behavior_rate": raw_new_behavior_cases / executed_cases if executed_cases else 0.0,
            "signal_new_behavior_cases": signal_new_behavior_cases,
            "signal_new_behavior_rate": signal_new_behavior_cases / executed_cases if executed_cases else 0.0,
            "saved_artifacts": int(meta.get("saved_artifacts", sum(1 for row in rows if row.get("bug_dir")))),
            "first_finding_case_index": _first_case_index(rows, lambda finding: True),
            "first_candidate_bug_case_index": _first_case_index(
                rows,
                lambda finding: is_rewardable_candidate_issue_finding(finding, known_bug_families),
            ),
            "first_candidate_bug_elapsed_s": _first_case_elapsed_s(
                rows,
                lambda finding: is_rewardable_candidate_issue_finding(finding, known_bug_families),
            ),
            "candidate_family_first_seen": _candidate_family_first_seen(rows, known_bug_families),
            "preflight": meta.get("preflight", {}),
            "quality_oracles": meta.get("quality_oracles", {}),
        },
        "environment_summary": _environment_summary(meta.get("environment", {})),
        "run_provenance": _run_provenance_summary(meta.get("run_provenance", {})),
    }


def record_run_journal(
    run_file: Path,
    *,
    context: dict[str, Any] | None = None,
    journal_file: Path | None = None,
    render_markdown: bool = True,
) -> tuple[Path, Path | None]:
    journal_file = journal_file or default_journal_path()
    append_run_journal_entries([build_run_journal_entry(run_file, context)], journal_file)
    md_path = write_run_journal_markdown(journal_file) if render_markdown else None
    return journal_file, md_path


def append_run_journal_entries(entries: list[dict[str, Any]], journal_file: Path | None = None) -> Path:
    journal_file = journal_file or default_journal_path()
    with JsonlWriter(journal_file) as writer:
        for entry in entries:
            writer.write(entry)
    return journal_file


def write_run_journal_markdown(journal_file: Path | None = None) -> Path:
    journal_file = journal_file or default_journal_path()
    entries = read_jsonl(journal_file) if journal_file.exists() else []
    md_path = journal_file.with_suffix(".md")
    lines = [
        "# DataDiffFuzz Paper Run Journal",
        "",
        "This is a paper-facing index of completed runs. It records the run theme, methodology track, target setup, budget, seed, and headline results without changing the experiment semantics.",
        "",
        "## Run Index",
        "",
        "| Recorded | Theme | Evidence | Target | Variant/Preset | Seed | Cases | Raw Findings | Raw novelty % | Signal novelty % | Candidate Families | First Candidate s |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for entry in entries:
        summary = entry.get("result_summary", {})
        config = entry.get("config_summary", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    _cell(entry.get("recorded_at", "")),
                    _cell(entry.get("theme", "")),
                    _cell(entry.get("evidence_mode", "")),
                    _cell(entry.get("target_suite", "")),
                    _cell(entry.get("variant_label") or entry.get("preset") or config.get("generator_profile", "")),
                    _cell(entry.get("seed", "")),
                    str(entry.get("executed_cases", "")),
                    str(summary.get("raw_findings", "")),
                    _cell(_fmt_percent(summary.get("new_behavior_rate", 0.0))),
                    _cell(_fmt_percent(summary.get("signal_new_behavior_rate", 0.0))),
                    str(summary.get("candidate_bug_family_count", "")),
                    _cell(summary.get("first_candidate_bug_elapsed_s", "")),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Details", ""])
    for idx, entry in enumerate(entries, start=1):
        summary = entry.get("result_summary", {})
        config = entry.get("config_summary", {})
        replay_policy = entry.get("replay_bug_policy", {})
        families = summary.get("candidate_bug_families", {})
        family_text = ", ".join(f"{name}:{count}" for name, count in sorted(families.items())) or "none"
        lines.extend(
            [
                f"### {idx}. {entry.get('theme', 'untitled run')}",
                "",
                f"- Recorded: `{entry.get('recorded_at', '')}`",
                f"- Evidence mode: `{entry.get('evidence_mode', '')}`; policy: {entry.get('counting_policy', '')}",
                f"- Target suite: `{entry.get('target_suite', '')}`; backends: `{', '.join(entry.get('backends', []))}`",
                f"- Variant/preset: `{entry.get('variant_label') or entry.get('preset') or config.get('generator_profile', '')}`",
                (
                    f"- Experiment identity: matrix=`{entry.get('matrix_id', '')}`; "
                    f"group=`{entry.get('comparison_group', '')}`; "
                    f"variant=`{entry.get('variant_id', '')}`; variant_label=`{entry.get('variant_label', '')}`; "
                    f"variant_group=`{entry.get('variant_group_id', '')}`; role=`{entry.get('comparison_role', '')}`; "
                    f"canonical_role=`{entry.get('canonical_comparison_role', '')}`; "
                    f"component_focus=`{entry.get('component_focus', '')}`; "
                    f"semantic_focus_families=`{', '.join(entry.get('semantic_focus_families', []))}`; "
                    f"semantic_focus_signals=`{', '.join(entry.get('semantic_focus_signals', []))}`"
                ),
                (
                    f"- Replay policy: enable_replay_bug=`{replay_policy.get('enable_replay_bug', False)}`; "
                    f"filter_enabled=`{replay_policy.get('filter_enabled', '')}`; "
                    f"filtered_candidates=`{replay_policy.get('filtered_candidates', 0)}`; "
                    f"fallback_candidates=`{replay_policy.get('fallback_candidates', 0)}`"
                ),
                f"- Seed: `{entry.get('seed', '')}`; requested cases: `{entry.get('requested_cases', '')}`; duration_s: `{entry.get('duration_s', '')}`",
                f"- Executed cases: `{entry.get('executed_cases', 0)}`; elapsed_s: `{entry.get('elapsed_s', '')}`; throughput_cases_s: `{entry.get('throughput_cases_s', '')}`",
                (
                    f"- Raw/signal new behavior cases: `{summary.get('new_behavior_cases', 0)}` / "
                    f"`{summary.get('signal_new_behavior_cases', 0)}`; rates: "
                    f"`{_fmt_percent(summary.get('new_behavior_rate', 0.0))}` / "
                    f"`{_fmt_percent(summary.get('signal_new_behavior_rate', 0.0))}`"
                ),
                f"- Raw findings: `{summary.get('raw_findings', 0)}`; candidate bug cases: `{summary.get('candidate_bug_cases', 0)}`; candidate families: {family_text}",
                f"- First candidate case/time: `{summary.get('first_candidate_bug_case_index', None)}` / `{summary.get('first_candidate_bug_elapsed_s', None)}`",
                f"- Run log: `{entry.get('run_file', '')}`",
            ]
        )
        if entry.get("known_bug_id"):
            lines.append(f"- Known bug id: `{entry.get('known_bug_id')}`; target version: `{entry.get('target_version', '')}`")
        if entry.get("manifest_file"):
            lines.append(f"- Manifest: `{entry.get('manifest_file')}`")
        if entry.get("notes"):
            lines.append(f"- Notes: {entry.get('notes')}")
        lines.append("")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def _default_theme(context: dict[str, Any], meta: dict[str, Any], run_file: Path) -> str:
    parts = [
        str(context.get("command") or "run"),
        str(context.get("target_suite") or meta.get("target_suite") or ""),
        str(context.get("preset") or meta.get("preset") or meta.get("config", {}).get("generator_profile", "")),
    ]
    text = ":".join(part for part in parts if part)
    return text or jsonl_log_stem(run_file)


def _candidate_issue_family_keys(
    findings: list[dict[str, Any]],
    known_saturated_bug_families: list[str] | tuple[str, ...] | None = None,
) -> Counter[str]:
    return candidate_issue_family_keys(
        findings,
        known_saturated_bug_families=known_saturated_bug_families,
    )


def _candidate_family_first_seen(
    rows: list[dict[str, Any]],
    known_saturated_bug_families: list[str] | tuple[str, ...] | None = None,
) -> dict[str, dict[str, Any]]:
    first_seen: dict[str, dict[str, Any]] = {}
    for fallback_idx, row in enumerate(rows):
        case = row.get("case", {})
        for family in _candidate_issue_family_keys(row.get("findings", []), known_saturated_bug_families):
            first_seen.setdefault(
                family,
                {
                    "case_index": row.get("case_index", fallback_idx),
                    "elapsed_s": row.get("elapsed_s"),
                    "case_id": case.get("case_id", ""),
                    "seed": case.get("seed", ""),
                },
            )
    return first_seen


_candidate_bug_family_keys = _candidate_issue_family_keys


def _first_case_index(rows: list[dict[str, Any]], predicate: Any) -> int | None:
    for fallback_idx, row in enumerate(rows):
        if any(predicate(finding) for finding in row.get("findings", [])):
            return int(row.get("case_index", fallback_idx))
    return None


def _first_case_elapsed_s(rows: list[dict[str, Any]], predicate: Any) -> float | None:
    for row in rows:
        if any(predicate(finding) for finding in row.get("findings", [])):
            elapsed = row.get("elapsed_s")
            return float(elapsed) if elapsed is not None else None
    return None


def _last_elapsed_s(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    elapsed = rows[-1].get("elapsed_s", 0.0)
    return float(elapsed) if elapsed is not None else 0.0


def _is_false_positive(finding: dict[str, Any]) -> bool:
    if finding.get("false_positive"):
        return True
    return finding.get("triage_verdict") in {"generator_false_positive", "normalizer_false_positive"}


def _is_semantic_divergence(finding: dict[str, Any]) -> bool:
    return finding.get("triage_verdict") in {
        "documented_semantic_divergence",
        "expected_semantic_divergence",
    }


def _environment_summary(environment: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(environment, dict):
        return {}
    keys = [
        "python",
        "platform",
        "system",
        "machine",
        "processor",
        "pandas",
        "polars",
        "duckdb",
        "pyarrow",
        "datafusion",
    ]
    return {key: environment[key] for key in keys if key in environment}


def _run_provenance_summary(provenance: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(provenance, dict):
        return {}
    vcs = provenance.get("vcs", {}) if isinstance(provenance.get("vcs", {}), dict) else {}
    launch = provenance.get("launch", {}) if isinstance(provenance.get("launch", {}), dict) else {}
    harness = provenance.get("harness", {}) if isinstance(provenance.get("harness", {}), dict) else {}
    summary = {
        "git_commit": str(vcs.get("git_commit", "") or ""),
        "git_branch": str(vcs.get("git_branch", "") or ""),
        "workspace_dirty": vcs.get("workspace_dirty"),
        "authority": bool(harness.get("authority", False)),
        "freeze_intent": bool(harness.get("freeze_intent", False)),
        "latest_code_claim": bool(harness.get("latest_code_claim", False)),
        "evidence_role": str(harness.get("evidence_role", "") or ""),
        "launch_source": str(launch.get("source", "") or ""),
        "launch_duration": str(launch.get("duration", "") or ""),
    }
    freeze_manifest = str((provenance.get("freeze_artifacts", {}) or {}).get("manifest", "") or "")
    if freeze_manifest:
        summary["freeze_manifest"] = freeze_manifest
    return summary


def _cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _fmt_percent(value: Any) -> str:
    try:
        return f"{float(value or 0.0):.1%}"
    except (TypeError, ValueError):
        return "0.0%"
