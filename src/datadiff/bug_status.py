from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

from datadiff.config import DEFAULT_KNOWN_SATURATED_BUG_FAMILIES
from datadiff.classification_oracle import classify_finding
from datadiff.dsl import Case
from datadiff.final_readiness import CONFIRMED_LATEST_UPSTREAM_STATUSES, DEFAULT_LATEST_CONFIRMATIONS_FILE
from datadiff.normalizer import NormalizedResult
from datadiff.oracle import classify_root_cause
from datadiff.util import PROJECT_ROOT, REPORTS_DIR, dump_json, load_json, utc_now

BUG_STATUS_SCHEMA_VERSION = "bug-status-v1"
DEFAULT_NEW_ISSUE_DIR = PROJECT_ROOT / "new_issue"
DEFAULT_OLD_ISSUE_DIR = PROJECT_ROOT / "old_issue"
DEFAULT_GENERATED_ISSUE_DIR = DEFAULT_NEW_ISSUE_DIR / "generated"
CANONICAL_FRESH_FAMILY_ALIASES = {
    "metamorphic_limit_idempotence@datafusion": "datafusion_limit_idempotence@datafusion",
}


def build_issue_status(
    *,
    latest_confirmation_files: list[Path] | None = None,
    new_issue_dir: Path | None = None,
    old_issue_dir: Path | None = None,
    generated_issue_dir: Path | None = None,
) -> dict[str, Any]:
    latest_confirmation_files = latest_confirmation_files or _default_latest_confirmation_files()
    new_issue_dir = _resolve_project_path(new_issue_dir or DEFAULT_NEW_ISSUE_DIR)
    old_issue_dir = _resolve_project_path(old_issue_dir or DEFAULT_OLD_ISSUE_DIR)
    generated_issue_dir = _resolve_project_path(generated_issue_dir or DEFAULT_GENERATED_ISSUE_DIR)

    confirmations = _load_latest_confirmations(latest_confirmation_files)
    confirmed_latest = [_confirmation_summary(item) for item in confirmations if _confirmation_family(item)]
    confirmed_latest_families = sorted({item["family"] for item in confirmed_latest})
    manual_issue_drafts = _collect_markdown_issue_drafts(new_issue_dir)
    generated_issue_drafts = _collect_markdown_issue_drafts(generated_issue_dir)
    pending_issue_drafts = [item for item in manual_issue_drafts if _is_pending_issue_status(item.get("status", ""))]
    audit_manifest = _load_audit_manifest(generated_issue_dir)
    fresh_evidence = _collect_fresh_candidate_evidence(generated_issue_dir)
    discovery_run_manifests = _collect_discovery_run_manifests(generated_issue_dir)
    discovery_campaign_manifests = _collect_discovery_campaign_manifests(generated_issue_dir)
    issue_bundle_manifest = _load_issue_bundle_manifest(generated_issue_dir)
    old_known = _collect_old_known_issues(old_issue_dir)

    audit_families = list(audit_manifest.get("candidate_bug_families", []))
    fresh_counter = Counter()
    for item in fresh_evidence:
        fresh_counter.update(item.get("fresh_candidate_bug_families", {}))
    known_saturated = set(DEFAULT_KNOWN_SATURATED_BUG_FAMILIES)
    current_fresh_counter = Counter(
        {family: count for family, count in fresh_counter.items() if family not in known_saturated}
    )

    return {
        "schema_version": BUG_STATUS_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "generated_by": "datadiff bug-status",
        "inputs": {
            "latest_confirmation_files": [_project_display_path(path) for path in latest_confirmation_files],
            "new_issue_dir": _project_display_path(new_issue_dir),
            "old_issue_dir": _project_display_path(old_issue_dir),
            "generated_issue_dir": _project_display_path(generated_issue_dir),
        },
        "summary": {
            "confirmed_latest_count": len(confirmed_latest_families),
            "confirmed_latest_families": confirmed_latest_families,
            "manual_new_issue_draft_count": len(manual_issue_drafts),
            "pending_manual_issue_draft_count": len(pending_issue_drafts),
            "pending_manual_issue_drafts": [item["path"] for item in pending_issue_drafts],
            "generated_issue_draft_count": len(generated_issue_drafts),
            "audit_candidate_family_count": len(audit_families),
            "audit_candidate_families": audit_families,
            "fresh_candidate_family_count": len(current_fresh_counter),
            "fresh_candidate_families": dict(sorted(current_fresh_counter.items())),
            "recorded_fresh_candidate_family_count": len(fresh_counter),
            "recorded_fresh_candidate_families": dict(sorted(fresh_counter.items())),
            "known_saturated_family_count": len(known_saturated),
            "discovery_run_manifest_count": len(discovery_run_manifests),
            "discovery_campaign_manifest_count": len(discovery_campaign_manifests),
            "discovery_workflow_manifest_count": len(discovery_run_manifests) + len(discovery_campaign_manifests),
            "issue_bundle_present": bool(issue_bundle_manifest.get("present")),
            "issue_bundle_family_count": int(issue_bundle_manifest.get("family_count", 0) or 0),
            "issue_bundle_reproducer_count": int(issue_bundle_manifest.get("extracted_reproducer_count", 0) or 0),
            "issue_bundle_missing_reproducer_count": int(
                issue_bundle_manifest.get("missing_reproducer_count", 0) or 0
            ),
            "issue_bundle_compile_failure_count": int(issue_bundle_manifest.get("compile_failure_count", 0) or 0),
            "issue_bundle_executed_reproducer_count": int(
                issue_bundle_manifest.get("executed_reproducer_count", 0) or 0
            ),
            "issue_bundle_executed_reproducer_attempt_count": int(
                issue_bundle_manifest.get("executed_reproducer_attempt_count", 0) or 0
            ),
            "issue_bundle_flaky_reproducer_count": int(
                issue_bundle_manifest.get("flaky_reproducer_count", 0) or 0
            ),
            "issue_bundle_nonzero_exit_count": int(issue_bundle_manifest.get("nonzero_exit_count", 0) or 0),
            "issue_bundle_nonzero_exit_attempt_count": int(
                issue_bundle_manifest.get("nonzero_exit_attempt_count", 0) or 0
            ),
            "issue_bundle_timeout_count": int(issue_bundle_manifest.get("timeout_count", 0) or 0),
            "issue_bundle_timeout_attempt_count": int(issue_bundle_manifest.get("timeout_attempt_count", 0) or 0),
            "old_known_document_count": len(old_known["documents"]),
            "old_known_upstream_issue_count": len(old_known["upstream_issue_urls"]),
        },
        "latest_confirmations": confirmed_latest,
        "manual_issue_drafts": manual_issue_drafts,
        "generated_issue_drafts": generated_issue_drafts,
        "audit_manifest": audit_manifest,
        "fresh_candidate_evidence": fresh_evidence,
        "discovery_run_manifests": discovery_run_manifests,
        "discovery_campaign_manifests": discovery_campaign_manifests,
        "issue_bundle_manifest": issue_bundle_manifest,
        "old_known": old_known,
    }


def write_issue_status_outputs(status: dict[str, Any], *, output_dir: Path | None = None) -> tuple[Path, Path]:
    output_dir = output_dir or REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = str(status.get("generated_at") or utc_now()).replace(":", "").replace("-", "").replace("Z", "")
    json_path = output_dir / f"bug-status-{stamp}.json"
    md_path = output_dir / f"bug-status-{stamp}.md"
    dump_json(status, json_path)
    md_path.write_text(render_issue_status_markdown(status), encoding="utf-8")
    return json_path, md_path


def render_issue_status_markdown(status: dict[str, Any]) -> str:
    summary = status.get("summary", {})
    lines = [
        "# DataDiffFuzz Bug Status",
        "",
        f"- Generated at: `{status.get('generated_at', '')}`",
        f"- Confirmed latest families: `{summary.get('confirmed_latest_count', 0)}`",
        f"- Audit candidate families: `{summary.get('audit_candidate_family_count', 0)}`",
        f"- Currently unsaturated fresh fuzz candidate families: `{summary.get('fresh_candidate_family_count', 0)}`",
        f"- Recorded fresh fuzz candidate families: `{summary.get('recorded_fresh_candidate_family_count', 0)}`",
        f"- Discovery workflow manifests: `{summary.get('discovery_workflow_manifest_count', 0)}`",
        f"- Issue bundle families: `{summary.get('issue_bundle_family_count', 0)}`",
        f"- Issue bundle reproducers: `{summary.get('issue_bundle_reproducer_count', 0)}`",
        f"- Issue bundle flaky reproducers: `{summary.get('issue_bundle_flaky_reproducer_count', 0)}`",
        f"- Issue bundle nonzero exits: `{summary.get('issue_bundle_nonzero_exit_count', 0)}`",
        f"- Issue bundle timeouts: `{summary.get('issue_bundle_timeout_count', 0)}`",
        f"- Pending manual issue drafts: `{summary.get('pending_manual_issue_draft_count', 0)}`",
        f"- Old known upstream issues: `{summary.get('old_known_upstream_issue_count', 0)}`",
        "",
        "## Confirmed Latest Families",
        "",
    ]
    confirmed = summary.get("confirmed_latest_families", [])
    if confirmed:
        lines.extend(f"- `{family}`" for family in confirmed)
    else:
        lines.append("- none")
    lines.extend(["", "## Audit Candidates", ""])
    audit = summary.get("audit_candidate_families", [])
    if audit:
        lines.extend(f"- `{family}`" for family in audit)
    else:
        lines.append("- none")
    lines.extend(["", "## Currently Unsaturated Fresh Fuzz Candidates", ""])
    fresh = summary.get("fresh_candidate_families", {})
    if fresh:
        lines.extend(f"- `{family}`: {count}" for family, count in sorted(fresh.items()))
    else:
        lines.append("- none")
    lines.extend(["", "## Recorded Fresh Fuzz Candidates", ""])
    recorded = summary.get("recorded_fresh_candidate_families", {})
    if recorded:
        lines.extend(f"- `{family}`: {count}" for family, count in sorted(recorded.items()))
    else:
        lines.append("- none")
    lines.extend(["", "## Pending Manual Issue Drafts", ""])
    pending = summary.get("pending_manual_issue_drafts", [])
    if pending:
        lines.extend(f"- `{path}`" for path in pending)
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def _default_latest_confirmation_files() -> list[Path]:
    return [DEFAULT_LATEST_CONFIRMATIONS_FILE] if DEFAULT_LATEST_CONFIRMATIONS_FILE.is_file() else []


def _resolve_project_path(path: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_latest_confirmations(paths: list[Path]) -> list[dict[str, Any]]:
    confirmations: list[dict[str, Any]] = []
    for path in paths:
        resolved = _resolve_project_path(path)
        data = _load_json_if_exists(resolved)
        raw_items = data.get("confirmations", data) if isinstance(data, dict) else data
        if not isinstance(raw_items, list):
            continue
        for raw_item in raw_items:
            if not isinstance(raw_item, dict):
                continue
            item = dict(raw_item)
            item["_confirmation_file"] = _project_display_path(resolved)
            confirmations.append(item)
    return confirmations


def _confirmation_summary(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "family": _confirmation_family(item),
        "issue_url": str(item.get("issue_url", "")),
        "issue_title": str(item.get("issue_title", "")),
        "upstream_status": str(item.get("upstream_status", "")),
        "state": str(item.get("state", "")),
        "labels": list(item.get("labels", []) or []),
        "assignee": str(item.get("assignee", "")),
        "checked_at": str(item.get("checked_at", "")),
        "source_file": str(item.get("_confirmation_file", "")),
    }


def _confirmation_family(item: dict[str, Any]) -> str:
    if str(item.get("upstream_status", "")).strip() not in CONFIRMED_LATEST_UPSTREAM_STATUSES:
        return ""
    if not str(item.get("issue_url", "")).strip():
        return ""
    family = str(item.get("family", "")).strip()
    if family:
        return family
    root = str(item.get("root_cause", "")).strip()
    suspicious = item.get("suspicious_backends", [])
    if not root or not isinstance(suspicious, list):
        return ""
    backends = ",".join(sorted(str(backend).strip() for backend in suspicious if str(backend).strip()))
    return f"{root}@{backends}" if backends else ""


def _collect_markdown_issue_drafts(directory: Path) -> list[dict[str, str]]:
    if not directory.is_dir():
        return []
    drafts = []
    for path in sorted(directory.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        text = path.read_text(encoding="utf-8")
        drafts.append(
            {
                "path": _project_display_path(path),
                "title": _extract_markdown_title(text) or path.stem,
                "status": _extract_issue_status(text),
            }
        )
    return drafts


def _load_audit_manifest(generated_issue_dir: Path) -> dict[str, Any]:
    path = generated_issue_dir / "manifest.json"
    data = _load_json_if_exists(path)
    if not isinstance(data, dict):
        return {"path": _project_display_path(path), "present": False, "candidate_bug_families": []}
    return {
        "path": _project_display_path(path),
        "present": True,
        "schema_version": str(data.get("schema_version", "")),
        "generated_at": str(data.get("generated_at", "")),
        "generated_by": str(data.get("generated_by", "")),
        "candidate_bug_families": list(data.get("candidate_bug_families", []) or []),
        "issue_files": list(data.get("issue_files", []) or []),
        "output_json": str(data.get("output_json", "")),
        "output_markdown": str(data.get("output_markdown", "")),
        "reproduction_command": str(data.get("reproduction_command", "")),
    }


def _collect_fresh_candidate_evidence(generated_issue_dir: Path) -> list[dict[str, Any]]:
    if not generated_issue_dir.is_dir():
        return []
    evidence = []
    for path in sorted(generated_issue_dir.glob("*-fresh-candidates.json")):
        data = _load_json_if_exists(path)
        if not isinstance(data, dict):
            continue
        raw_families = dict(data.get("fresh_candidate_bug_families", {}) or {})
        canonical_families = _canonical_fresh_candidate_families(data)
        if canonical_families is None:
            canonical_families = raw_families
        evidence.append(
            {
                "path": _project_display_path(path),
                "generated_at": str(data.get("generated_at", "")),
                "source_run_file": str(data.get("source_run_file", "")),
                "fresh_candidate_bug_families": canonical_families,
                "raw_fresh_candidate_bug_families": raw_families,
                "candidate_row_count": int(data.get("candidate_row_count", 0) or 0),
            }
        )
    return evidence


def _canonical_fresh_candidate_families(data: dict[str, Any]) -> dict[str, int] | None:
    counter = Counter()
    if "candidate_rows" not in data:
        return None
    rows = data.get("candidate_rows", [])
    if not isinstance(rows, list):
        return None
    saw_reclassifiable_finding = False
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            case = Case.from_dict(row.get("case", {}))
            normalized = {
                str(backend): _normalized_from_dict(str(backend), payload)
                for backend, payload in dict(row.get("normalized", {}) or {}).items()
                if isinstance(payload, dict)
            }
        except Exception:  # noqa: BLE001
            continue
        raw_results = row.get("raw_results", {})
        if not isinstance(raw_results, dict):
            raw_results = {}
        config = row.get("config", {})
        if not isinstance(config, dict):
            config = {}
        backends = sorted(normalized)
        for finding in row.get("findings", []) or []:
            if not isinstance(finding, dict):
                continue
            saw_reclassifiable_finding = True
            classification = classify_finding(case, finding, normalized, raw_results, config, backends)
            if classification.false_positive or classification.verdict in {
                "documented_semantic_divergence",
                "expected_semantic_divergence",
                "generator_false_positive",
                "normalizer_false_positive",
            }:
                continue
            was_recorded_candidate = (
                finding.get("triage_verdict") == "candidate_implementation_bug"
                and not bool(finding.get("false_positive"))
            )
            if classification.verdict != "candidate_implementation_bug" and not was_recorded_candidate:
                continue
            suspicious = classification.implicated_backends or [
                str(backend)
                for backend in finding.get("suspicious_backends", []) or []
                if str(backend).strip()
            ]
            if not suspicious:
                continue
            root = str(finding.get("root_cause", "unknown"))
            if str(finding.get("oracle", "")) == "metamorphic" or root.startswith("metamorphic_"):
                family = _canonical_fresh_family(f"{root}@{','.join(sorted(suspicious))}")
            else:
                root = classify_root_cause(case, normalized, str(finding.get("kind", "")))
                family = _canonical_fresh_family(f"{root}@{','.join(sorted(suspicious))}")
            counter[family] += 1
    if not saw_reclassifiable_finding and rows:
        return None
    return dict(sorted(counter.items()))


def _canonical_fresh_family(family: str) -> str:
    return CANONICAL_FRESH_FAMILY_ALIASES.get(family, family)


def _normalized_from_dict(backend: str, payload: dict[str, Any]) -> NormalizedResult:
    return NormalizedResult.from_dict(payload, backend=backend)


def _collect_discovery_run_manifests(generated_issue_dir: Path) -> list[dict[str, Any]]:
    if not generated_issue_dir.is_dir():
        return []
    manifests = []
    paths = [*generated_issue_dir.glob("discovery-run*-manifest.json")]
    for path in sorted(dict.fromkeys(paths)):
        data = _load_json_if_exists(path)
        if not isinstance(data, dict):
            continue
        classification = data.get("classification", {}) if isinstance(data.get("classification"), dict) else {}
        manifests.append(
            {
                "path": _project_display_path(path),
                "generated_at": str(data.get("generated_at", "")),
                "target_suite": str(data.get("target_suite", "")),
                "preset": str(data.get("preset", "")),
                "cases": data.get("cases", ""),
                "duration": data.get("duration", ""),
                "seed": data.get("seed", ""),
                "fresh_candidate_bug_families": dict(classification.get("fresh_candidate_bug_families", {}) or {}),
                "known_saturated_candidate_bug_families": dict(
                    classification.get("known_saturated_candidate_bug_families", {}) or {}
                ),
            }
        )
    return manifests


def _collect_discovery_campaign_manifests(generated_issue_dir: Path) -> list[dict[str, Any]]:
    if not generated_issue_dir.is_dir():
        return []
    manifests = []
    paths = [*generated_issue_dir.glob("discovery-campaign*-manifest.json")]
    for path in sorted(dict.fromkeys(paths)):
        data = _load_json_if_exists(path)
        if not isinstance(data, dict):
            continue
        summary = data.get("summary", {}) if isinstance(data.get("summary"), dict) else {}
        manifests.append(
            {
                "path": _project_display_path(path),
                "generated_at": str(data.get("generated_at", "")),
                "lane_ids": list(data.get("lane_ids", []) or []),
                "seeds": list(data.get("seeds", []) or []),
                "cases_per_lane_seed": data.get("cases_per_lane_seed", ""),
                "duration": data.get("duration", ""),
                "fresh_candidate_bug_families": dict(summary.get("fresh_candidate_bug_families", {}) or {}),
                "known_saturated_candidate_bug_families": dict(
                    summary.get("known_saturated_candidate_bug_families", {}) or {}
                ),
                "run_count": int(summary.get("run_count", 0) or 0),
            }
        )
    return manifests


def _load_issue_bundle_manifest(generated_issue_dir: Path) -> dict[str, Any]:
    path = generated_issue_dir / "issue-bundles" / "manifest.json"
    data = _load_json_if_exists(path)
    if not isinstance(data, dict):
        return {
            "path": _project_display_path(path),
            "present": False,
            "family_count": 0,
            "families": [],
            "extracted_reproducer_count": 0,
            "missing_reproducer_count": 0,
            "compile_failure_count": 0,
            "executed_reproducer_count": 0,
            "executed_reproducer_attempt_count": 0,
            "flaky_reproducer_count": 0,
            "nonzero_exit_count": 0,
            "nonzero_exit_attempt_count": 0,
            "timeout_count": 0,
            "timeout_attempt_count": 0,
        }
    summary = data.get("summary", {}) if isinstance(data.get("summary"), dict) else {}
    executed_reproducer_count = int(summary.get("executed_reproducer_count", 0) or 0)
    return {
        "path": _project_display_path(path),
        "present": True,
        "schema_version": str(data.get("schema_version", "")),
        "generated_at": str(data.get("generated_at", "")),
        "generated_by": str(data.get("generated_by", "")),
        "family_count": int(summary.get("family_count", 0) or 0),
        "families": list(summary.get("families", []) or []),
        "extracted_reproducer_count": int(summary.get("extracted_reproducer_count", 0) or 0),
        "missing_reproducer_count": int(summary.get("missing_reproducer_count", 0) or 0),
        "compile_failure_count": int(summary.get("compile_failure_count", 0) or 0),
        "executed_reproducer_count": executed_reproducer_count,
        "executed_reproducer_attempt_count": int(
            summary.get("executed_reproducer_attempt_count", executed_reproducer_count) or 0
        ),
        "flaky_reproducer_count": int(summary.get("flaky_reproducer_count", 0) or 0),
        "nonzero_exit_count": int(summary.get("nonzero_exit_count", 0) or 0),
        "nonzero_exit_attempt_count": int(summary.get("nonzero_exit_attempt_count", 0) or 0),
        "timeout_count": int(summary.get("timeout_count", 0) or 0),
        "timeout_attempt_count": int(summary.get("timeout_attempt_count", 0) or 0),
        "manifest_path": str(data.get("manifest_path", "")),
        "markdown_path": str(data.get("markdown_path", "")),
    }


def _collect_old_known_issues(old_issue_dir: Path) -> dict[str, Any]:
    documents = _collect_markdown_issue_drafts(old_issue_dir)
    urls: set[str] = set()
    if old_issue_dir.is_dir():
        for path in sorted(old_issue_dir.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            urls.update(_extract_upstream_issue_urls(text))
    return {
        "documents": documents,
        "upstream_issue_urls": sorted(urls),
    }


def _load_json_if_exists(path: Path) -> Any:
    if not path.is_file():
        return None
    return load_json(path)


def _extract_markdown_title(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def _extract_issue_status(text: str) -> str:
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        key = cells[0].strip("` ").lower()
        if key in {"status", "current status", "当前状态"}:
            return cells[1]
    return ""


def _is_pending_issue_status(status: str) -> bool:
    text = status.lower()
    if not text:
        return False
    pending_markers = (
        "not yet submitted",
        "new candidate",
        "待提交",
        "未提交",
        "needs final",
        "needs stable reproducer",
        "needs stable reproduction",
        "flaky reproducer",
        "需",
    )
    submitted_markers = ("已提交", "closed", "fixed", "accepted")
    return any(marker in text for marker in pending_markers) and not any(
        marker in text for marker in submitted_markers
    )


def _extract_upstream_issue_urls(text: str) -> list[str]:
    return re.findall(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/issues/\d+", text)


def _project_display_path(path: str | Path) -> str:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    try:
        return str(resolved.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


build_bug_status = build_issue_status
write_bug_status_outputs = write_issue_status_outputs
render_bug_status_markdown = render_issue_status_markdown
