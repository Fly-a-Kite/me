from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from datadiff.bug_status import build_issue_status
from datadiff.final_readiness import DEFAULT_LATEST_CONFIRMATIONS_FILE
from datadiff.pathing import project_display_path as _project_display_path_impl
from datadiff.pathing import resolve_project_path as _resolve_project_path_impl
from datadiff.util import PROJECT_ROOT, REPORTS_DIR, dump_json, utc_now

ISSUE_READINESS_SCHEMA_VERSION = "issue-readiness-v1"

READY_TO_SUBMIT = "ready_to_submit"
NEEDS_DEDUP_CHECK = "needs_dedup_check"
NEEDS_REPRODUCER_OR_EVIDENCE = "needs_reproducer_or_evidence"
ALREADY_SUBMITTED_OR_CONFIRMED = "already_submitted_or_confirmed"
NOT_LATEST_REPRODUCIBLE = "not_latest_reproducible"

_STATUS_ORDER = (
    READY_TO_SUBMIT,
    NEEDS_DEDUP_CHECK,
    NEEDS_REPRODUCER_OR_EVIDENCE,
    ALREADY_SUBMITTED_OR_CONFIRMED,
    NOT_LATEST_REPRODUCIBLE,
)
_PATH_REF_RE = re.compile(
    r"(?<![\w/-])"
    r"((?:bugs|runs|reports|new_issue|old_issue|experiments|src|tests|study)/"
    r"[A-Za-z0-9_.:/-]+(?:\.jsonl\.gz|\.jsonl|\.json|\.md|\.py|\.csv|\.txt)?)"
)
_UPSTREAM_ISSUE_RE = re.compile(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/issues/\d+")
_FAMILY_RE = re.compile(r"`?([A-Za-z0-9][A-Za-z0-9_.-]*@[A-Za-z0-9_,.-]+)`?")


def build_issue_readiness(
    *,
    latest_confirmation_files: list[Path] | None = None,
    new_issue_dir: Path | None = None,
    old_issue_dir: Path | None = None,
    generated_issue_dir: Path | None = None,
    include_generated: bool = False,
) -> dict[str, Any]:
    latest_confirmation_files = latest_confirmation_files or _default_latest_confirmation_files()
    new_issue_dir = _resolve_project_path(new_issue_dir or PROJECT_ROOT / "new_issue")
    old_issue_dir = _resolve_project_path(old_issue_dir or PROJECT_ROOT / "old_issue")
    generated_issue_dir = _resolve_project_path(generated_issue_dir or new_issue_dir / "generated")

    status = build_issue_status(
        latest_confirmation_files=latest_confirmation_files,
        new_issue_dir=new_issue_dir,
        old_issue_dir=old_issue_dir,
        generated_issue_dir=generated_issue_dir,
    )
    confirmed_by_url = {
        str(item.get("issue_url", "")).strip(): item for item in status.get("latest_confirmations", [])
    }
    confirmed_by_family = {
        str(item.get("family", "")).strip(): item for item in status.get("latest_confirmations", [])
    }
    old_known_urls = set(status.get("old_known", {}).get("upstream_issue_urls", []) or [])
    audit_families = list(status.get("summary", {}).get("audit_candidate_families", []) or [])

    documents = _collect_issue_documents(new_issue_dir, source="manual")
    if include_generated:
        documents.extend(_collect_issue_documents(generated_issue_dir, source="generated"))

    issues = [
        _assess_issue_document(
            document,
            confirmed_by_url=confirmed_by_url,
            confirmed_by_family=confirmed_by_family,
            old_known_urls=old_known_urls,
            audit_families=audit_families,
        )
        for document in documents
    ]
    submission_groups = _submission_groups(issues)
    summary = _summarize_issue_readiness(issues, submission_groups)
    return {
        "schema_version": ISSUE_READINESS_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "generated_by": "datadiff issue-readiness",
        "inputs": {
            "latest_confirmation_files": [_project_display_path(path) for path in latest_confirmation_files],
            "new_issue_dir": _project_display_path(new_issue_dir),
            "old_issue_dir": _project_display_path(old_issue_dir),
            "generated_issue_dir": _project_display_path(generated_issue_dir),
            "include_generated": include_generated,
        },
        "summary": summary,
        "submission_groups": submission_groups,
        "issues": issues,
        "bug_status_summary": status.get("summary", {}),
    }


def write_issue_readiness_outputs(audit: dict[str, Any], *, output_dir: Path | None = None) -> tuple[Path, Path]:
    output_dir = output_dir or REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = str(audit.get("generated_at") or utc_now()).replace(":", "").replace("-", "").replace("Z", "")
    json_path = output_dir / f"issue-readiness-{stamp}.json"
    md_path = output_dir / f"issue-readiness-{stamp}.md"
    dump_json(audit, json_path)
    md_path.write_text(render_issue_readiness_markdown(audit), encoding="utf-8")
    return json_path, md_path


def render_issue_readiness_markdown(audit: dict[str, Any]) -> str:
    summary = audit.get("summary", {})
    lines = [
        "# DataDiffFuzz Issue Readiness",
        "",
        f"- Generated at: `{audit.get('generated_at', '')}`",
        f"- Ready to submit: `{summary.get('ready_to_submit_count', 0)}` documents / "
        f"`{summary.get('ready_to_submit_family_count', 0)}` families",
        f"- Needs dedup check: `{summary.get('needs_dedup_check_count', 0)}` documents / "
        f"`{summary.get('needs_dedup_check_family_count', 0)}` families",
        f"- Needs evidence/reproducer: `{summary.get('needs_reproducer_or_evidence_count', 0)}`",
        f"- Already submitted or confirmed: `{summary.get('already_submitted_or_confirmed_count', 0)}`",
        f"- Not latest reproducible: `{summary.get('not_latest_reproducible_count', 0)}`",
        f"- Submission groups: `{summary.get('submission_group_count', 0)}` unique families",
        f"- Duplicate family drafts: `{summary.get('duplicate_family_draft_count', 0)}`",
        "",
        "## Submission Groups",
        "",
        "| Family | Primary Issue | Documents | Statuses |",
        "| --- | --- | ---: | --- |",
    ]
    for group in audit.get("submission_groups", []):
        statuses = ", ".join(
            f"{status}:{count}" for status, count in sorted(group.get("status_counts", {}).items())
        )
        lines.append(
            f"| `{group.get('family', '')}` | `{group.get('primary_issue_path', '')}` | "
            f"{group.get('issue_count', 0)} | {statuses or '-'} |"
        )
    if not audit.get("submission_groups", []):
        lines.append("| none |  | 0 | - |")
    lines.extend(
        [
        "",
        "## Queue",
        "",
        "| Issue | Status | Priority | Blocking Reasons |",
        "| --- | --- | --- | --- |",
        ]
    )
    for issue in audit.get("issues", []):
        reasons = "; ".join(issue.get("blocking_reasons", []) or []) or "-"
        lines.append(
            f"| `{issue.get('path', '')}` | `{issue.get('readiness_status', '')}` | "
            f"`{issue.get('priority', '')}` | {reasons} |"
        )
    lines.append("")
    return "\n".join(lines)


def _collect_issue_documents(directory: Path, *, source: str) -> list[dict[str, Any]]:
    if not directory.is_dir():
        return []
    documents = []
    for path in sorted(directory.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        text = path.read_text(encoding="utf-8")
        documents.append(
            {
                "path": _project_display_path(path),
                "absolute_path": path,
                "title": _extract_markdown_title(text) or path.stem,
                "text": text,
                "source": source,
            }
        )
    return documents


def _assess_issue_document(
    document: dict[str, Any],
    *,
    confirmed_by_url: dict[str, dict[str, Any]],
    confirmed_by_family: dict[str, dict[str, Any]],
    old_known_urls: set[str],
    audit_families: list[str],
) -> dict[str, Any]:
    text = str(document["text"])
    status_text = _extract_table_value(text, ("status", "current status", "当前状态"))
    upstream_urls = sorted(set(_UPSTREAM_ISSUE_RE.findall(text)))
    family_guess = _guess_family(text, str(document["absolute_path"]), audit_families)
    evidence = _evidence_flags(text)
    missing_evidence = [name for name, present in evidence.items() if not present]
    missing_refs = _missing_referenced_paths(text)
    old_mentions = sorted(url for url in upstream_urls if url in old_known_urls)
    confirmed_match = _confirmed_match(upstream_urls, family_guess, confirmed_by_url, confirmed_by_family)

    status_lower = status_text.lower()
    text_lower = text.lower()
    blocking_reasons: list[str] = []
    warnings: list[str] = []

    if missing_refs:
        warnings.append("referenced local evidence paths are missing or ignored: " + ", ".join(missing_refs[:8]))

    if confirmed_match:
        readiness_status = ALREADY_SUBMITTED_OR_CONFIRMED
        blocking_reasons.append("already registered in latest confirmation evidence")
    elif _looks_submitted_or_confirmed(status_text, upstream_urls):
        readiness_status = ALREADY_SUBMITTED_OR_CONFIRMED
        blocking_reasons.append("document status says it has already been submitted or confirmed upstream")
    elif _looks_not_latest_reproducible(status_text, text_lower):
        readiness_status = NOT_LATEST_REPRODUCIBLE
        blocking_reasons.append("document says the issue is not reproducible on the current latest version")
    elif document["source"] == "generated":
        readiness_status = NEEDS_REPRODUCER_OR_EVIDENCE
        blocking_reasons.append("raw generated draft must be reviewed in new_issue/ before upstream submission")
    elif _looks_needs_reproducer_stabilization(status_lower, text_lower):
        readiness_status = NEEDS_REPRODUCER_OR_EVIDENCE
        blocking_reasons.append("document says the reproducer is flaky or still needs stable evidence")
    elif missing_evidence:
        readiness_status = NEEDS_REPRODUCER_OR_EVIDENCE
        blocking_reasons.append("missing required issue evidence sections: " + ", ".join(missing_evidence))
    elif _needs_dedup_check(status_lower, text_lower, old_mentions):
        readiness_status = NEEDS_DEDUP_CHECK
        blocking_reasons.append("needs final upstream duplicate check before submission")
        if old_mentions:
            blocking_reasons.append("mentions old-known upstream issue URL(s): " + ", ".join(old_mentions))
    else:
        readiness_status = READY_TO_SUBMIT

    priority_score, priority = _priority(readiness_status, evidence, warnings)
    return {
        "path": document["path"],
        "source": document["source"],
        "title": document["title"],
        "family_guess": family_guess,
        "status_text": status_text,
        "readiness_status": readiness_status,
        "priority": priority,
        "priority_score": priority_score,
        "blocking_reasons": blocking_reasons,
        "warnings": warnings,
        "evidence_found": evidence,
        "missing_evidence": missing_evidence,
        "missing_referenced_paths": missing_refs,
        "upstream_urls": upstream_urls,
        "old_known_urls_mentioned": old_mentions,
        "confirmed_latest_match": confirmed_match,
        "discovery_time": _extract_discovery_time(text),
        "discovery_basis": _extract_discovery_basis(text),
        "counts_as_confirmed": readiness_status == ALREADY_SUBMITTED_OR_CONFIRMED and bool(confirmed_match),
        "safe_for_paper_confirmed_count": readiness_status == ALREADY_SUBMITTED_OR_CONFIRMED and bool(confirmed_match),
    }


def _summarize_issue_readiness(
    issues: list[dict[str, Any]],
    submission_groups: list[dict[str, Any]],
) -> dict[str, Any]:
    counts = {status: 0 for status in _STATUS_ORDER}
    for issue in issues:
        counts[str(issue.get("readiness_status", ""))] = counts.get(str(issue.get("readiness_status", "")), 0) + 1
    queue = sorted(
        issues,
        key=lambda item: (
            -int(item.get("priority_score", 0)),
            _STATUS_ORDER.index(str(item.get("readiness_status", "")))
            if str(item.get("readiness_status", "")) in _STATUS_ORDER
            else len(_STATUS_ORDER),
            str(item.get("path", "")),
        ),
    )
    families_by_status = {
        status: sorted(
            {
                str(issue.get("family_guess", "")).strip()
                for issue in issues
                if issue.get("readiness_status") == status and str(issue.get("family_guess", "")).strip()
            }
        )
        for status in _STATUS_ORDER
    }
    confirmed_families = sorted(
        {
            str(issue.get("family_guess", "")).strip()
            for issue in issues
            if issue.get("safe_for_paper_confirmed_count") and str(issue.get("family_guess", "")).strip()
        }
    )
    duplicate_groups = [group for group in submission_groups if int(group.get("issue_count", 0)) > 1]
    return {
        "issue_document_count": len(issues),
        "ready_to_submit_count": counts.get(READY_TO_SUBMIT, 0),
        "ready_to_submit_family_count": len(families_by_status[READY_TO_SUBMIT]),
        "ready_to_submit_families": families_by_status[READY_TO_SUBMIT],
        "needs_dedup_check_count": counts.get(NEEDS_DEDUP_CHECK, 0),
        "needs_dedup_check_family_count": len(families_by_status[NEEDS_DEDUP_CHECK]),
        "needs_dedup_check_families": families_by_status[NEEDS_DEDUP_CHECK],
        "needs_reproducer_or_evidence_count": counts.get(NEEDS_REPRODUCER_OR_EVIDENCE, 0),
        "needs_reproducer_or_evidence_family_count": len(families_by_status[NEEDS_REPRODUCER_OR_EVIDENCE]),
        "already_submitted_or_confirmed_count": counts.get(ALREADY_SUBMITTED_OR_CONFIRMED, 0),
        "already_submitted_or_confirmed_family_count": len(families_by_status[ALREADY_SUBMITTED_OR_CONFIRMED]),
        "not_latest_reproducible_count": counts.get(NOT_LATEST_REPRODUCIBLE, 0),
        "not_latest_reproducible_family_count": len(families_by_status[NOT_LATEST_REPRODUCIBLE]),
        "confirmed_count_eligible_count": sum(1 for issue in issues if issue.get("safe_for_paper_confirmed_count")),
        "confirmed_count_eligible_family_count": len(confirmed_families),
        "confirmed_count_eligible_families": confirmed_families,
        "ready_to_submit": [issue["path"] for issue in issues if issue["readiness_status"] == READY_TO_SUBMIT],
        "needs_dedup_check": [issue["path"] for issue in issues if issue["readiness_status"] == NEEDS_DEDUP_CHECK],
        "needs_reproducer_or_evidence": [
            issue["path"] for issue in issues if issue["readiness_status"] == NEEDS_REPRODUCER_OR_EVIDENCE
        ],
        "submission_group_count": len(submission_groups),
        "submission_group_families": [group["family"] for group in submission_groups],
        "submission_groups_ready_to_submit_count": sum(
            1 for group in submission_groups if group.get("status_counts", {}).get(READY_TO_SUBMIT, 0)
        ),
        "submission_groups_needs_dedup_count": sum(
            1 for group in submission_groups if group.get("needs_dedup_check")
        ),
        "duplicate_family_draft_count": sum(int(group.get("issue_count", 0)) - 1 for group in duplicate_groups),
        "duplicate_family_draft_groups": {
            group["family"]: group.get("issue_paths", []) for group in duplicate_groups
        },
        "top_priority": [issue["path"] for issue in queue[:5]],
    }


def _submission_groups(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for issue in issues:
        if issue.get("readiness_status") not in {READY_TO_SUBMIT, NEEDS_DEDUP_CHECK}:
            continue
        family = str(issue.get("family_guess", "")).strip() or str(issue.get("path", "")).strip()
        grouped.setdefault(family, []).append(issue)

    out: list[dict[str, Any]] = []
    for family, family_issues in grouped.items():
        ordered = sorted(
            family_issues,
            key=lambda item: (
                -int(item.get("priority_score", 0)),
                _STATUS_ORDER.index(str(item.get("readiness_status", "")))
                if str(item.get("readiness_status", "")) in _STATUS_ORDER
                else len(_STATUS_ORDER),
                str(item.get("path", "")),
            ),
        )
        primary = ordered[0]
        status_counts: dict[str, int] = {}
        for issue in ordered:
            status = str(issue.get("readiness_status", ""))
            status_counts[status] = status_counts.get(status, 0) + 1
        submission_status = READY_TO_SUBMIT if status_counts.get(READY_TO_SUBMIT, 0) else NEEDS_DEDUP_CHECK
        out.append(
            {
                "family": family,
                "submission_status": submission_status,
                "primary_issue_path": str(primary.get("path", "")),
                "primary_title": str(primary.get("title", "")),
                "priority": str(primary.get("priority", "")),
                "priority_score": int(primary.get("priority_score", 0) or 0),
                "issue_count": len(ordered),
                "issue_paths": [str(issue.get("path", "")) for issue in ordered],
                "supporting_issue_paths": [str(issue.get("path", "")) for issue in ordered[1:]],
                "status_counts": dict(sorted(status_counts.items())),
                "needs_dedup_check": bool(status_counts.get(NEEDS_DEDUP_CHECK, 0)),
            }
        )
    return sorted(
        out,
        key=lambda group: (
            -int(group.get("priority_score", 0)),
            str(group.get("family", "")),
        ),
    )


def _default_latest_confirmation_files() -> list[Path]:
    return [DEFAULT_LATEST_CONFIRMATIONS_FILE] if DEFAULT_LATEST_CONFIRMATIONS_FILE.is_file() else []


def _resolve_project_path(path: str | Path | None) -> Path:
    return _resolve_project_path_impl(path, project_root=PROJECT_ROOT)


def _project_display_path(path: str | Path | None) -> str:
    return _project_display_path_impl(path, project_root=PROJECT_ROOT)


def _extract_markdown_title(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def _extract_table_value(text: str, keys: tuple[str, ...]) -> str:
    normalized_keys = {_normalize_table_key(key) for key in keys}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or "---" in stripped:
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) < 2:
            continue
        if _normalize_table_key(cells[0]) in normalized_keys:
            return cells[1]
    return ""


def _normalize_table_key(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip("` :：").lower())


def _guess_family(text: str, path: str, audit_families: list[str]) -> str:
    aligned_family = _audit_family_for_path(path, audit_families)
    if aligned_family:
        return aligned_family
    preferred = _extract_table_value(
        text,
        (
            "candidate family",
            "project family label observed",
            "project family labels observed",
            "本地 family",
            "原始本地标签",
            "根因 family",
        ),
    )
    family_matches = _FAMILY_RE.findall(preferred) or _FAMILY_RE.findall(text)
    if family_matches:
        return family_matches[0].strip("`")
    return Path(path).stem


def _audit_family_for_path(path: str, audit_families: list[str]) -> str:
    stem_tokens = _family_tokens(Path(path).stem)
    best_family = ""
    best_score = 0
    for family in audit_families:
        root = family.split("@", 1)[0]
        root_tokens = _family_tokens(root)
        overlap = len(stem_tokens & root_tokens)
        score = overlap
        stem = Path(path).stem.lower()
        root_lower = root.lower()
        if root_lower in stem or stem in root_lower:
            score += 4
        if score > best_score:
            best_family = family
            best_score = score
    return best_family if best_score >= 3 else ""


def _family_tokens(value: str) -> set[str]:
    return {token for token in re.split(r"[^A-Za-z0-9]+", value.lower()) if len(token) > 1}


def _evidence_flags(text: str) -> dict[str, bool]:
    lower = text.lower()
    return {
        "discovery_record": "## discovery record" in lower or "## 发现记录" in text,
        "environment": "## environment" in lower or "target backend/version" in lower or "环境" in text,
        "reproducer": "## reproducer" in lower or "## reproduction" in lower or "复现" in text,
        "expected": "## expected" in lower or "## expected output" in lower or "## 预期" in text,
        "observed_or_actual": "## actual" in lower or "## observed" in lower or "observed normalized outputs" in lower,
        "project_evidence": "datadifffuzz evidence" in lower or "## evidence" in lower or "## 证据" in text,
    }


def _missing_referenced_paths(text: str) -> list[str]:
    missing = []
    for match in sorted(set(_PATH_REF_RE.findall(text))):
        cleaned = match.rstrip(".,);:")
        if not cleaned:
            continue
        path = PROJECT_ROOT / cleaned
        if not path.exists():
            missing.append(cleaned)
    return missing


def _confirmed_match(
    upstream_urls: list[str],
    family_guess: str,
    confirmed_by_url: dict[str, dict[str, Any]],
    confirmed_by_family: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    for url in upstream_urls:
        if url in confirmed_by_url:
            return {
                "matched_by": "issue_url",
                "issue_url": url,
                "family": str(confirmed_by_url[url].get("family", "")),
                "upstream_status": str(confirmed_by_url[url].get("upstream_status", "")),
            }
    if family_guess in confirmed_by_family:
        item = confirmed_by_family[family_guess]
        return {
            "matched_by": "family",
            "issue_url": str(item.get("issue_url", "")),
            "family": family_guess,
            "upstream_status": str(item.get("upstream_status", "")),
        }
    return {}


def _looks_submitted_or_confirmed(status_text: str, upstream_urls: list[str]) -> bool:
    lower = status_text.lower()
    submitted = ("已提交", "上游 issue", "upstream issue", "submitted", "open", "closed", "accepted")
    confirmed = ("标签含 `bug`", "labeled bug", "bug", "assigned", "已分配")
    pending = ("not yet submitted", "待提交", "未提交", "needs final", "需先", "需判断")
    return bool(upstream_urls) and any(marker in lower for marker in submitted + confirmed) and not any(
        marker in lower for marker in pending
    )


def _looks_not_latest_reproducible(status_text: str, text_lower: str) -> bool:
    status_lower = status_text.lower()
    markers = (
        "未复现",
        "not reproduced",
        "not reproducible",
        "暂不按最新版",
        "暂不计当前",
        "not latest",
    )
    return any(marker in status_lower or marker in text_lower for marker in markers)


def _needs_dedup_check(status_lower: str, text_lower: str, old_mentions: list[str]) -> bool:
    explicit_status_markers = (
        "needs final",
        "需先判断",
        "需判断是否",
    )
    if any(marker in status_lower for marker in explicit_status_markers):
        return True
    markers = (
        "needs final upstream dedup",
        "needs final upstream search",
        "final upstream search",
        "final upstream dedup",
        "需先判断",
        "需判断是否",
        "重复",
        "dedup check",
    )
    if old_mentions:
        return True
    if any(marker in status_lower or marker in text_lower for marker in markers):
        no_duplicate_markers = (
            "did not locate an obvious existing issue",
            "did not identify an existing",
            "未 locate",
            "未找到",
        )
        return not any(marker in text_lower for marker in no_duplicate_markers)
    return False


def _looks_needs_reproducer_stabilization(status_lower: str, text_lower: str) -> bool:
    markers = (
        "needs stable reproducer",
        "needs stable reproduction",
        "flaky reproducer",
        "flaky reproduction",
        "unstable reproducer",
        "reproducer is flaky",
        "复现不稳定",
        "需要稳定复现",
    )
    return any(marker in status_lower or marker in text_lower for marker in markers)


def _priority(readiness_status: str, evidence: dict[str, bool], warnings: list[str]) -> tuple[int, str]:
    evidence_score = sum(1 for present in evidence.values() if present)
    if readiness_status == READY_TO_SUBMIT:
        score = 90 + evidence_score
    elif readiness_status == NEEDS_DEDUP_CHECK:
        score = 75 + evidence_score
    elif readiness_status == NEEDS_REPRODUCER_OR_EVIDENCE:
        score = 40 + evidence_score
    elif readiness_status == ALREADY_SUBMITTED_OR_CONFIRMED:
        score = 20 + evidence_score
    else:
        score = 10 + evidence_score
    if warnings:
        score -= 3
    if score >= 90:
        label = "high"
    elif score >= 70:
        label = "medium"
    else:
        label = "low"
    return score, label


def _extract_discovery_time(text: str) -> str:
    return _extract_table_value(
        text,
        (
            "discovery time",
            "first automated signal",
            "first datadifffuzz signal",
            "first recorded signal",
            "first recorded live signal",
            "首个可核验草稿时间",
            "本地草稿最早可核验时间",
            "提交时间",
            "发现/首个可核验记录时间（北京时间）",
        ),
    )


def _extract_discovery_basis(text: str) -> str:
    return _extract_table_value(text, ("how found", "如何发现", "发现依据"))
