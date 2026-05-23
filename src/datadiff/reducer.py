from __future__ import annotations

from collections.abc import Iterable

from datadiff.config import ExperimentConfig
from datadiff.dsl import Case
from datadiff.runner import run_loaded_case


def reduce_case(
    case: Case,
    backends: list[str],
    config: ExperimentConfig | None = None,
    target_kinds: Iterable[str] | None = None,
    target_roots: Iterable[str] | None = None,
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
                if _preserves_target(candidate, backends, config, target, target_root_set):
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
            if _preserves_target(candidate, backends, config, target, target_root_set):
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
            if _preserves_target(candidate, backends, config, target, target_root_set):
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
                if _preserves_target(candidate, backends, config, target, target_root_set):
                    best = candidate
                    changed = True
                    break
            if changed:
                break

    return best


def _can_remove_operation(ops: list[dict], idx: int) -> bool:
    op = ops[idx]
    if op.get("op") != "sort":
        return True
    # Dropping a sort while keeping a later limit/offset turns deterministic top-k
    # semantics into an arbitrary prefix. That can preserve a finding for the
    # wrong reason and produce a misleading reduced artifact. A later sort
    # supersedes the current one before any limit observes it.
    for later in ops[idx + 1 :]:
        if later.get("op") in {"limit", "offset"}:
            return False
        if later.get("op") == "sort":
            return True
    return True


def _program_references_table(ops: list[dict], table_name: str) -> bool:
    return any(op.get("op") in {"join", "tuple_absence_filter"} and op.get("table") == table_name for op in ops)


def _preserves_target(
    candidate: Case,
    backends: list[str],
    config: ExperimentConfig,
    target_kinds: set[str],
    target_roots: set[str],
) -> bool:
    findings = run_loaded_case(candidate, backends, config=config, save_artifact=False)["findings"]
    findings = [finding for finding in findings if not _is_false_positive_reduction(finding)]
    if not findings:
        return False
    if not target_kinds:
        return True
    if not target_roots:
        return bool(target_kinds & {finding["kind"] for finding in findings})
    return any(
        finding["kind"] in target_kinds and finding.get("root_cause", "unknown") in target_roots
        for finding in findings
    )


def _is_false_positive_reduction(finding: dict) -> bool:
    if finding.get("false_positive"):
        return True
    return finding.get("triage_verdict") in {"generator_false_positive", "normalizer_false_positive"}
