from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from datadiff.bug_status import build_issue_status
from datadiff.config import DEFAULT_KNOWN_SATURATED_BUG_FAMILIES, DEFAULT_REPLAY_BUG_SOURCE_ISSUES
from datadiff.experiment_catalog import FINAL_PROTOCOL_TRACKS
from datadiff.final_readiness import DEFAULT_A_LEVEL_READINESS_POLICY
from datadiff.issue_readiness import build_issue_readiness
from datadiff.targets import TARGETS, TARGET_SUITES, target_context
from datadiff.util import PROJECT_ROOT, REPORTS_DIR, dump_json, utc_now

REVIEW_READINESS_SCHEMA_VERSION = "review-readiness-v1"


@dataclass(frozen=True, slots=True)
class ReviewThresholds:
    target_confirmed_bug_families: int = 20
    min_audit_candidate_families: int = 1
    min_discovery_workflow_manifests: int = 1
    min_generated_issue_drafts: int = 1
    min_issue_bundle_families: int = 1
    min_pending_issue_drafts: int = 1
    min_old_known_upstream_issues: int = 1


REQUIRED_LIVE_TARGET_SUITES = DEFAULT_A_LEVEL_READINESS_POLICY.required_live_suites
REQUIRED_LIVE_TARGET_FAMILIES = DEFAULT_A_LEVEL_READINESS_POLICY.required_live_families
REQUIRED_DOC_PATHS = (
    "README.md",
    "docs/project_architecture.md",
    "docs/automated_bug_detection.md",
)
REQUIRED_PIPELINE_PATHS = (
    "src/datadiff/targets.py",
    "src/datadiff/datagen.py",
    "src/datadiff/runner.py",
    "src/datadiff/oracle.py",
    "src/datadiff/metamorphic.py",
    "src/datadiff/triage.py",
    "src/datadiff/bug_audit.py",
    "src/datadiff/bug_status.py",
    "src/datadiff/issue_bundle.py",
    "src/datadiff/issue_readiness.py",
    "src/datadiff/methodology_report.py",
    "src/datadiff/run_journal.py",
)
REQUIRED_REPRO_COMMAND_MARKERS = (
    "datadiff targets",
    "datadiff bug-audit",
    "datadiff discovery-run",
    "datadiff discovery-campaign",
    "datadiff classify-run",
    "datadiff bug-status",
    "datadiff issue-bundle",
    "datadiff issue-readiness",
    "datadiff methodology-report",
    "datadiff final-readiness",
    "datadiff review-readiness",
    "scripts/run_final_experiments.py --track validation",
)
REQUIRED_METHOD_TEST_PATHS = (
    "tests/test_methodology_contracts.py",
    "tests/test_targets.py",
    "tests/test_bug_audit.py",
    "tests/test_bug_status.py",
    "tests/test_issue_bundle.py",
    "tests/test_issue_readiness.py",
    "tests/test_final_readiness.py",
    "tests/test_methodology_report.py",
    "tests/test_run_journal.py",
    "tests/test_reward.py",
)


def build_review_readiness(
    *,
    project_root: Path | None = None,
    latest_confirmation_files: list[Path] | None = None,
    thresholds: ReviewThresholds | None = None,
) -> dict[str, Any]:
    root = (project_root or PROJECT_ROOT).resolve()
    thresholds = thresholds or ReviewThresholds()
    latest_confirmation_files = latest_confirmation_files or [root / "experiments" / "latest_confirmations.json"]
    status = build_issue_status(
        latest_confirmation_files=latest_confirmation_files,
        new_issue_dir=root / "new_issue",
        old_issue_dir=root / "old_issue",
        generated_issue_dir=root / "new_issue" / "generated",
    )
    issue_queue = build_issue_readiness(
        latest_confirmation_files=latest_confirmation_files,
        new_issue_dir=root / "new_issue",
        old_issue_dir=root / "old_issue",
        generated_issue_dir=root / "new_issue" / "generated",
    )
    criteria = [
        _criterion_documentation(root),
        _criterion_layered_pipeline_modules(root),
        _criterion_target_breadth(),
        _criterion_final_experiment_protocol(root),
        _criterion_automated_bug_pipeline(status, thresholds),
        _criterion_fresh_replay_separation(status),
        _criterion_reproducibility_commands(root),
        _criterion_artifact_hygiene(root),
        _criterion_methodology_tests(root),
        _criterion_confirmed_bug_target(status, thresholds),
        _criterion_reportable_issue_queue(status, thresholds),
    ]
    required = [item for item in criteria if item["required"]]
    summary = {
        "required_passed": sum(1 for item in required if item["status"] == "pass"),
        "required_total": len(required),
        "warnings": sum(1 for item in criteria if item["status"] == "warn"),
        "failed_required": [item["id"] for item in required if item["status"] != "pass"],
        "confirmed_latest_count": status["summary"]["confirmed_latest_count"],
        "target_confirmed_bug_families": thresholds.target_confirmed_bug_families,
        "current_fresh_candidate_families": status["summary"]["fresh_candidate_families"],
        "pending_issue_drafts": status["summary"]["pending_manual_issue_drafts"],
        "issue_submission_group_count": issue_queue["summary"].get("submission_group_count", 0),
        "ready_to_submit_families": issue_queue["summary"].get("ready_to_submit_families", []),
        "needs_dedup_check_families": issue_queue["summary"].get("needs_dedup_check_families", []),
        "needs_reproducer_or_evidence": issue_queue["summary"].get("needs_reproducer_or_evidence", []),
        "duplicate_family_draft_count": issue_queue["summary"].get("duplicate_family_draft_count", 0),
    }
    return {
        "schema_version": REVIEW_READINESS_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "generated_by": "datadiff review-readiness",
        "project_root": str(root),
        "ready": not summary["failed_required"],
        "thresholds": asdict(thresholds),
        "summary": summary,
        "criteria": criteria,
        "recommendations": _recommendations(summary, status, issue_queue),
        "bug_status_summary": status["summary"],
        "issue_readiness_summary": issue_queue["summary"],
        "issue_submission_groups": issue_queue.get("submission_groups", []),
    }


def write_review_readiness_outputs(audit: dict[str, Any], *, output_dir: Path | None = None) -> tuple[Path, Path]:
    output_dir = output_dir or REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = str(audit.get("generated_at") or utc_now()).replace(":", "").replace("-", "").replace("Z", "")
    json_path = output_dir / f"review-readiness-{stamp}.json"
    md_path = output_dir / f"review-readiness-{stamp}.md"
    dump_json(audit, json_path)
    md_path.write_text(render_review_readiness_markdown(audit), encoding="utf-8")
    return json_path, md_path


def render_review_readiness_markdown(audit: dict[str, Any]) -> str:
    summary = audit.get("summary", {})
    lines = [
        "# DataDiffFuzz Review Readiness",
        "",
        f"- Generated at: `{audit.get('generated_at', '')}`",
        f"- Ready: `{audit.get('ready', False)}`",
        f"- Required gates: `{summary.get('required_passed', 0)}/{summary.get('required_total', 0)}`",
        f"- Confirmed latest bug families: `{summary.get('confirmed_latest_count', 0)}` / `{summary.get('target_confirmed_bug_families', 0)}`",
        f"- Ready issue families: `{len(summary.get('ready_to_submit_families', []))}`",
        f"- Dedup-check issue families: `{len(summary.get('needs_dedup_check_families', []))}`",
        f"- Needs evidence issue drafts: `{len(summary.get('needs_reproducer_or_evidence', []))}`",
        "",
        "## Gates",
        "",
        "| Gate | Status | Evidence |",
        "| --- | --- | --- |",
    ]
    for item in audit.get("criteria", []):
        required = "required" if item.get("required") else "supporting"
        evidence = str(item.get("evidence", "")).replace("\n", " ")
        lines.append(f"| `{item.get('id', '')}` ({required}) | `{item.get('status', '')}` | {evidence} |")
    lines.extend(["", "## Recommendations", ""])
    recommendations = audit.get("recommendations", [])
    if recommendations:
        lines.extend(f"- {item}" for item in recommendations)
    else:
        lines.append("- No blocking recommendations.")
    lines.append("")
    groups = audit.get("issue_submission_groups", [])
    lines.extend(["## Issue Submission Groups", ""])
    if groups:
        lines.extend(["| Family | Status | Primary Issue | Documents |", "| --- | --- | --- | ---: |"])
        for group in groups:
            lines.append(
                f"| `{group.get('family', '')}` | `{group.get('submission_status', '')}` | "
                f"`{group.get('primary_issue_path', '')}` | {group.get('issue_count', 0)} |"
            )
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def _criterion(
    criterion_id: str,
    *,
    status: str,
    evidence: str,
    required: bool = True,
    summary: str = "",
) -> dict[str, Any]:
    return {
        "id": criterion_id,
        "status": status,
        "required": required,
        "summary": summary or evidence,
        "evidence": evidence,
    }


def _criterion_documentation(root: Path) -> dict[str, Any]:
    missing = [path for path in REQUIRED_DOC_PATHS if not (root / path).is_file()]
    return _criterion(
        "documented_methodology",
        status="pass" if not missing else "fail",
        evidence="all required docs present" if not missing else f"missing docs: {', '.join(missing)}",
    )


def _criterion_layered_pipeline_modules(root: Path) -> dict[str, Any]:
    missing = [path for path in REQUIRED_PIPELINE_PATHS if not (root / path).is_file()]
    return _criterion(
        "layered_pipeline_implementation",
        status="pass" if not missing else "fail",
        evidence="target/generator/runner/oracle/triage/evidence modules present"
        if not missing
        else f"missing modules: {', '.join(missing)}",
    )


def _criterion_target_breadth() -> dict[str, Any]:
    missing_suites = [suite for suite in REQUIRED_LIVE_TARGET_SUITES if suite not in TARGET_SUITES]
    families = set(target_context(sorted(TARGETS)).families)
    missing_families = [family for family in REQUIRED_LIVE_TARGET_FAMILIES if family not in families]
    missing = missing_suites + missing_families
    return _criterion(
        "target_registry_breadth",
        status="pass" if not missing else "fail",
        evidence=(
            f"suites={len(TARGET_SUITES)}, targets={len(TARGETS)}, families={','.join(sorted(families))}"
            if not missing
            else f"missing target registry coverage: {', '.join(missing)}"
        ),
    )


def _criterion_final_experiment_protocol(root: Path) -> dict[str, Any]:
    required_files = (
        "scripts/run_final_experiments.py",
        "experiments/final_protocol.md",
        "src/datadiff/final_readiness.py",
        "tests/test_final_experiment_plan.py",
    )
    missing_files = [path for path in required_files if not (root / path).is_file()]
    if missing_files:
        return _criterion(
            "final_experiment_protocol",
            status="fail",
            evidence=f"missing files: {', '.join(missing_files)}",
        )

    script_text = (root / "scripts/run_final_experiments.py").read_text(encoding="utf-8")
    protocol_text = (root / "experiments/final_protocol.md").read_text(encoding="utf-8")
    readiness_text = (root / "src/datadiff/final_readiness.py").read_text(encoding="utf-8")
    test_text = (root / "tests/test_final_experiment_plan.py").read_text(encoding="utf-8")

    failures: list[str] = []
    missing_script_tracks = [
        track
        for track in FINAL_PROTOCOL_TRACKS
        if f'"{track}"' not in script_text and f"'{track}'" not in script_text
    ]
    if missing_script_tracks:
        failures.append(f"plan script missing tracks: {', '.join(missing_script_tracks)}")
    missing_protocol_tracks = [track for track in FINAL_PROTOCOL_TRACKS if f"--track {track}" not in protocol_text]
    if missing_protocol_tracks:
        failures.append(f"protocol doc missing track commands: {', '.join(missing_protocol_tracks)}")

    required_script_markers = (
        "short_validation_command",
        "module_ablation_command",
        "method_comparison_command",
        '"validation"',
        '"--evidence-mode"',
        "paper_run_journal",
    )
    missing_script_markers = [marker for marker in required_script_markers if marker not in script_text]
    if missing_script_markers:
        failures.append(f"plan script missing markers: {', '.join(missing_script_markers)}")

    required_readiness_markers = (
        "paper_run_journal",
        "short_validation",
        "validation_runs",
        "require_validation",
        "ablation_runs",
        "comparison_runs",
        "require_ablation",
        "require_comparison",
    )
    missing_readiness_markers = [marker for marker in required_readiness_markers if marker not in readiness_text]
    if missing_readiness_markers:
        failures.append(f"readiness audit missing validation markers: {', '.join(missing_readiness_markers)}")

    required_test_markers = (
        "test_validation_command_gates_short_before_long_runs",
        "test_ablation_and_comparison_commands_cover_method_rqs",
        "test_final_plan_keeps_paper_run_journal_enabled",
    )
    missing_test_markers = [marker for marker in required_test_markers if marker not in test_text]
    if missing_test_markers:
        failures.append(f"plan tests missing markers: {', '.join(missing_test_markers)}")

    return _criterion(
            "final_experiment_protocol",
            status="pass" if not failures else "fail",
            evidence=(
            f"tracks={','.join(FINAL_PROTOCOL_TRACKS)}; validation/support readiness gates and plan tests present"
            if not failures
            else "; ".join(failures)
        ),
    )


def _criterion_automated_bug_pipeline(status: dict[str, Any], thresholds: ReviewThresholds) -> dict[str, Any]:
    summary = status["summary"]
    failures = []
    if summary["audit_candidate_family_count"] < thresholds.min_audit_candidate_families:
        failures.append("deterministic audit candidate evidence")
    if summary["discovery_workflow_manifest_count"] < thresholds.min_discovery_workflow_manifests:
        failures.append("discovery-run/discovery-campaign workflow manifests")
    if summary["generated_issue_draft_count"] < thresholds.min_generated_issue_drafts:
        failures.append("generated issue drafts")
    if summary.get("issue_bundle_family_count", 0) < thresholds.min_issue_bundle_families:
        failures.append("issue bundle reproducers")
    if summary.get("issue_bundle_missing_reproducer_count", 0):
        failures.append("issue bundle issues without reproducers")
    if summary.get("issue_bundle_compile_failure_count", 0):
        failures.append("issue bundle compile failures")
    if summary.get("issue_bundle_flaky_reproducer_count", 0):
        failures.append("issue bundle flaky reproducers")
    if summary.get("issue_bundle_nonzero_exit_count", 0):
        failures.append("issue bundle reproducers with nonzero exits")
    if summary.get("issue_bundle_timeout_count", 0):
        failures.append("issue bundle reproducer timeouts")
    return _criterion(
        "automated_bug_detection_pipeline",
        status="pass" if not failures else "fail",
        evidence=(
            "audit_candidates={audit_candidate_family_count}, workflows={discovery_workflow_manifest_count}, "
            "generated_issue_drafts={generated_issue_draft_count}, "
            "issue_bundle_families={issue_bundle_family_count}, "
            "issue_bundle_reproducers={issue_bundle_reproducer_count}, "
            "issue_bundle_expected_failures={issue_bundle_expected_failure_reproducer_count}, "
            "issue_bundle_fixed_upstream_not_reproduced={issue_bundle_fixed_upstream_not_reproduced_count}, "
            "issue_bundle_missing={issue_bundle_missing_reproducer_count}, "
            "issue_bundle_compile_failures={issue_bundle_compile_failure_count}, "
            "issue_bundle_flaky={issue_bundle_flaky_reproducer_count}, "
            "issue_bundle_nonzero={issue_bundle_nonzero_exit_count}, "
            "issue_bundle_timeouts={issue_bundle_timeout_count}"
        ).format(**summary)
        if not failures
        else f"missing: {', '.join(failures)}",
    )


def _criterion_fresh_replay_separation(status: dict[str, Any]) -> dict[str, Any]:
    summary = status["summary"]
    ok = (
        summary["known_saturated_family_count"] == len(DEFAULT_KNOWN_SATURATED_BUG_FAMILIES)
        and len(DEFAULT_REPLAY_BUG_SOURCE_ISSUES) > 0
        and summary["old_known_upstream_issue_count"] > 0
    )
    return _criterion(
        "fresh_replay_known_separation",
        status="pass" if ok else "fail",
        evidence=(
            f"known_saturated={summary['known_saturated_family_count']}, "
            f"replay_source_issues={len(DEFAULT_REPLAY_BUG_SOURCE_ISSUES)}, "
            f"old_known_upstream_issues={summary['old_known_upstream_issue_count']}"
        ),
    )


def _criterion_reproducibility_commands(root: Path) -> dict[str, Any]:
    readme = root / "README.md"
    text = readme.read_text(encoding="utf-8") if readme.is_file() else ""
    missing = [marker for marker in REQUIRED_REPRO_COMMAND_MARKERS if marker not in text]
    return _criterion(
        "reproducibility_commands_documented",
        status="pass" if not missing else "fail",
        evidence=(
            "README contains target/audit/discovery/classify/status/issue-bundle/"
            "methodology/final-plan/readiness commands"
        )
        if not missing
        else f"missing command markers: {', '.join(missing)}",
    )


def _criterion_artifact_hygiene(root: Path) -> dict[str, Any]:
    gitignore = root / ".gitignore"
    text = gitignore.read_text(encoding="utf-8") if gitignore.is_file() else ""
    required_ignored = ("runs/", "bugs/", "reports/", "corpus/", "study/")
    missing = [item for item in required_ignored if item not in text]
    study_present = (root / "study").exists()
    status = "pass" if not missing and not study_present else "fail"
    evidence = "runtime outputs ignored and study/ absent"
    if missing or study_present:
        parts = []
        if missing:
            parts.append(f"missing gitignore entries: {', '.join(missing)}")
        if study_present:
            parts.append("study/ exists")
        evidence = "; ".join(parts)
    return _criterion("artifact_hygiene", status=status, evidence=evidence)


def _criterion_methodology_tests(root: Path) -> dict[str, Any]:
    missing = [path for path in REQUIRED_METHOD_TEST_PATHS if not (root / path).is_file()]
    return _criterion(
        "methodology_contract_tests_present",
        status="pass" if not missing else "fail",
        evidence="methodology, target, audit, status, reward, run-journal, issue-bundle, methodology-report, and readiness tests present"
        if not missing
        else f"missing tests: {', '.join(missing)}",
    )


def _criterion_confirmed_bug_target(status: dict[str, Any], thresholds: ReviewThresholds) -> dict[str, Any]:
    count = int(status["summary"]["confirmed_latest_count"])
    target = int(thresholds.target_confirmed_bug_families)
    return _criterion(
        "confirmed_bug_family_target",
        status="pass" if count >= target else "fail",
        evidence=f"confirmed_latest={count}, target={target}",
    )


def _criterion_reportable_issue_queue(status: dict[str, Any], thresholds: ReviewThresholds) -> dict[str, Any]:
    pending = int(status["summary"]["pending_manual_issue_draft_count"])
    required = pending >= thresholds.min_pending_issue_drafts
    return _criterion(
        "reportable_issue_queue",
        status="pass" if required else "warn",
        required=False,
        evidence=f"pending_issue_drafts={pending}",
    )


def _recommendations(
    summary: dict[str, Any],
    status: dict[str, Any],
    issue_queue: dict[str, Any],
) -> list[str]:
    out: list[str] = []
    confirmed = int(summary.get("confirmed_latest_count", 0))
    target = int(summary.get("target_confirmed_bug_families", 0))
    if confirmed < target:
        out.append(
            f"Need {target - confirmed} more upstream-confirmed latest bug families to reach the paper target."
        )
    if not status["summary"].get("fresh_candidate_families"):
        out.append("Current status has no unsaturated fresh candidate family; run longer or add more focused discovery lanes.")
    issue_summary = issue_queue.get("summary", {})
    ready_families = issue_summary.get("ready_to_submit_families", []) or []
    if ready_families:
        out.append(f"Submit ready unique issue families: {', '.join(ready_families[:5])}.")
    dedup_families = issue_summary.get("needs_dedup_check_families", []) or []
    if dedup_families:
        out.append(f"Run final upstream duplicate checks for issue families: {', '.join(dedup_families[:5])}.")
    needs_evidence = issue_summary.get("needs_reproducer_or_evidence", []) or []
    if needs_evidence:
        out.append(f"Stabilize or complete reproducer/evidence for drafts: {', '.join(needs_evidence[:5])}.")
    duplicate_count = int(issue_summary.get("duplicate_family_draft_count", 0) or 0)
    if duplicate_count:
        out.append(
            f"Keep {duplicate_count} duplicate family draft(s) grouped so issue documents are not counted as distinct bug families."
        )
    if summary.get("failed_required"):
        out.append(f"Fix required review gates: {', '.join(summary['failed_required'])}.")
    return out
