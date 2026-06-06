from __future__ import annotations

from collections.abc import Iterable

from datadiff.config import ExperimentConfig
from datadiff.dsl import Case
from datadiff.operation_semantics import op_kind, op_table
from datadiff.runner import run_loaded_case


def reduce_case(
    case: Case,
    backends: list[str],
    config: ExperimentConfig | None = None,
    target_kinds: Iterable[str] | None = None,
    target_roots: Iterable[str] | None = None,
    target_suspicious_backends: Iterable[Iterable[str]] | None = None,
) -> Case:
    """Small deterministic reducer for the MVP.

    It currently minimizes rows and trailing operations. The reducer keeps a
    candidate only if it still triggers at least one finding. This module exists
    so ablation experiments can compare artifact quality with and without
    reduction; later versions should add column/value/expression reducers.
    """

    config = config or ExperimentConfig(enable_artifact=False)
    target = set(target_kinds or [])
    target_root_set = set(target_roots or [])
    target_suspicious_set = {
        _suspicious_key(backends)
        for backends in (target_suspicious_backends or [])
    }
    best = case

    changed = True
    while changed:
        changed = False

        for table_idx, table in enumerate(best.tables):
            for row_idx in range(len(table.rows)):
                if len(table.rows) <= 1:
                    break
                candidate_rows = table.rows[:row_idx] + table.rows[row_idx + 1 :]
                candidate_tables = list(best.tables)
                candidate_tables[table_idx] = type(table)(table.name, table.columns, candidate_rows)
                candidate = Case(best.case_id, best.seed, candidate_tables, best.program, best.metadata)
                if _preserves_target(candidate, backends, config, target, target_root_set, target_suspicious_set):
                    best = candidate
                    changed = True
                    break
            if changed:
                break

        if changed:
            continue

        ops = best.program.operations
        for idx in range(len(ops) - 1, -1, -1):
            if len(ops) <= 1:
                break
            if not _can_remove_operation(ops, idx):
                continue
            candidate_program = type(best.program)(best.program.program_id, best.program.seed, ops[:idx] + ops[idx + 1 :])
            candidate = Case(best.case_id, best.seed, best.tables, candidate_program, best.metadata)
            if _preserves_target(candidate, backends, config, target, target_root_set, target_suspicious_set):
                best = candidate
                changed = True
                break

        if changed:
            continue

        for idx in range(1, len(best.tables)):
            table_name = best.tables[idx].name
            if _program_references_table(best.program.operations, table_name):
                continue
            candidate = Case(best.case_id, best.seed, best.tables[:idx] + best.tables[idx + 1 :], best.program, best.metadata)
            if _preserves_target(candidate, backends, config, target, target_root_set, target_suspicious_set):
                best = candidate
                changed = True
                break

        if changed:
            continue

        for table_idx, table in enumerate(best.tables):
            for col_idx, _column in enumerate(table.columns):
                if len(table.columns) <= 1:
                    break
                candidate_columns = table.columns[:col_idx] + table.columns[col_idx + 1 :]
                candidate_rows = [
                    {col.name: row.get(col.name) for col in candidate_columns}
                    for row in table.rows
                ]
                candidate_tables = list(best.tables)
                candidate_tables[table_idx] = type(table)(table.name, candidate_columns, candidate_rows)
                candidate = Case(best.case_id, best.seed, candidate_tables, best.program, best.metadata)
                if _preserves_target(candidate, backends, config, target, target_root_set, target_suspicious_set):
                    best = candidate
                    changed = True
                    break
            if changed:
                break

    return best


def _can_remove_operation(ops: list[dict], idx: int) -> bool:
    op = ops[idx]
    if op_kind(op) != "sort":
        return True
    # Dropping a sort while keeping a later limit/offset turns deterministic top-k
    # semantics into an arbitrary prefix. That can preserve a finding for the
    # wrong reason and produce a misleading reduced artifact. A later sort
    # supersedes the current one before any limit observes it.
    for later in ops[idx + 1 :]:
        if op_kind(later) in {"limit", "offset"}:
            return False
        if op_kind(later) == "sort":
            return True
    return True


def _program_references_table(ops: list[dict], table_name: str) -> bool:
    return any(op_kind(op) in {"join", "tuple_absence_filter"} and op_table(op) == table_name for op in ops)


def _preserves_target(
    candidate: Case,
    backends: list[str],
    config: ExperimentConfig,
    target_kinds: set[str],
    target_roots: set[str],
    target_suspicious_backends: set[tuple[str, ...]],
) -> bool:
    findings = run_loaded_case(candidate, backends, config=config, save_artifact=False)["findings"]
    findings = [finding for finding in findings if not _is_false_positive_reduction(finding)]
    if not findings:
        return False
    if not target_kinds and not target_roots and not target_suspicious_backends:
        return True
    return any(_matches_target(finding, target_kinds, target_roots, target_suspicious_backends) for finding in findings)


def _matches_target(
    finding: dict,
    target_kinds: set[str],
    target_roots: set[str],
    target_suspicious_backends: set[tuple[str, ...]],
) -> bool:
    if target_kinds and finding.get("kind") not in target_kinds:
        return False
    if target_roots and finding.get("root_cause", "unknown") not in target_roots:
        return False
    if target_suspicious_backends and _suspicious_key(finding.get("suspicious_backends", [])) not in target_suspicious_backends:
        return False
    return True


def _suspicious_key(backends: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({str(backend) for backend in backends if str(backend)}))


def _is_false_positive_reduction(finding: dict) -> bool:
    if finding.get("false_positive"):
        return True
    return finding.get("triage_verdict") in {"generator_false_positive", "normalizer_false_positive"}
