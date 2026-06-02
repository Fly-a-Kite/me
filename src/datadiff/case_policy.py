from __future__ import annotations

from typing import Any, Literal

from datadiff.case_features import PROBE_ROOTS
from datadiff.dsl import Case
from datadiff.operation_semantics import operation_names

CaseOrigin = Literal["organic", "issue_inspired", "issue_replay"]

ISSUE_REPLAY_OPS = frozenset(PROBE_ROOTS) | {"running_sum", "tuple_absence_filter", "row_number_filter"}


def canonical_source_issue_key(value: Any) -> str:
    return str(value or "").strip().rstrip("/")


def case_source_issues(case: Case) -> list[str]:
    metadata = case.metadata if isinstance(case.metadata, dict) else {}
    sources = []
    for field_name in ("source_issue", "source_issue_alt"):
        source_value = canonical_source_issue_key(metadata.get(field_name))
        if source_value:
            sources.append(source_value)
    return sources


def primary_source_issue(case: Case) -> str:
    return next(iter(case_source_issues(case)), "")


def case_operation_names(case: Case) -> list[str]:
    return operation_names(case.program.operations)


def uses_replay_probe_operations(operation_names: list[str]) -> bool:
    return any(operation_name in ISSUE_REPLAY_OPS for operation_name in operation_names)


def case_discovery_origin(case: Case) -> CaseOrigin:
    if not primary_source_issue(case):
        return "organic"
    if uses_replay_probe_operations(case_operation_names(case)):
        return "issue_replay"
    return "issue_inspired"


def known_replay_source_filter_reason(
    case: Case,
    *,
    enable_replay_bug: bool,
    replay_bug_source_issues: list[str] | tuple[str, ...] = (),
) -> str:
    if enable_replay_bug:
        return ""
    if case_discovery_origin(case) == "issue_replay":
        return "issue_replay_probe"
    replay_sources = {canonical_source_issue_key(source) for source in replay_bug_source_issues}
    if set(case_source_issues(case)) & replay_sources:
        return "known_replay_source_issue"
    return ""


issue_source_key = canonical_source_issue_key
uses_issue_replay_ops = uses_replay_probe_operations
replay_bug_filter_reason = known_replay_source_filter_reason
