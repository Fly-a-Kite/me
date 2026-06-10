from __future__ import annotations

from typing import Any

from datadiff.canonicalization import compare_row_set_batch
from datadiff.dsl import Case, Program, normalize_sort_keys
from datadiff.filtering import evaluate_filter_predicate
from datadiff.join_keys import join_key_pairs
from datadiff.normalizer import NormalizedResult, _norm_value
from datadiff.ordering_semantics import (
    rows_have_observable_duplicate_sort_key,
    sort_row_mappings,
    sort_window_boundary_splits_observable_tie,
)
from datadiff.operation_semantics import (
    aggregate_functions,
    condition_cmp,
    condition_column,
    condition_value,
    op_column,
    op_kind,
    op_n,
    op_table,
)
from datadiff.program_analysis import case_uses_precision_sensitive_float_arithmetic
from datadiff.reference_semantics import reference_result


ROW_ORDER_PRESERVING_AFTER_ORDER = frozenset(
    {
        "filter",
        "tuple_absence_filter",
        "drop_nulls",
        "semi_join",
        "anti_join",
        "fill_null",
        "coalesce",
        "case_when",
        "select",
        "mutate",
        "limit",
        "offset",
    }
)


def normalizer_errors(normalized: dict[str, NormalizedResult | dict[str, Any]]) -> list[str]:
    out = []
    for backend, result in normalized.items():
        if _result_get(result, "status") == "normalization_error":
            out.append(f"{backend}:{_result_get(result, 'error_type')}")
    return out


def is_pyarrow_empty_global_bool_aggregate_adapter_error(
    case: Case,
    finding: Any,
    normalized: dict[str, NormalizedResult | dict[str, Any]],
    raw_results: dict[str, dict[str, Any]],
) -> bool:
    suspicious = {str(backend) for backend in (_get(finding, "suspicious_backends", []) or [])}
    if "pyarrow" not in suspicious:
        return False
    pyarrow_raw = raw_results.get("pyarrow", {})
    pyarrow_normalized = normalized.get("pyarrow")
    status = str(pyarrow_raw.get("status") or _result_get(pyarrow_normalized, "status", ""))
    if status != "error":
        return False
    error_text = " ".join(
        str(part)
        for part in [
            pyarrow_raw.get("error_type", ""),
            pyarrow_raw.get("error", ""),
            _result_get(pyarrow_normalized, "error_type", ""),
            _result_get(pyarrow_normalized, "error", ""),
        ]
    ).lower()
    if "arrowinvalid" not in error_text or "invalid null value" not in error_text:
        return False
    return _has_empty_input_global_bool_aggregate(case)


def all_backends_rejected_due_to_generated_invalidity(raw_results: dict[str, dict[str, Any]]) -> bool:
    if not raw_results:
        return False
    if any(result.get("status") == "ok" for result in raw_results.values()):
        return False
    text = " ".join(
        f"{result.get('error_type', '')} {result.get('error', '')}".lower()
        for result in raw_results.values()
    )
    invalid_markers = [
        "column",
        "not found",
        "no such",
        "binder",
        "schema",
        "keyerror",
        "invalid operation",
        "cannot resolve",
    ]
    return any(marker in text for marker in invalid_markers)


def is_float_precision_boundary_mismatch(
    case: Case,
    normalized: dict[str, NormalizedResult | dict[str, Any]],
) -> bool:
    if not case_uses_precision_sensitive_float_arithmetic(case):
        return False
    ok_results = [result for result in normalized.values() if _result_get(result, "status") == "ok"]
    if len(ok_results) < 2:
        return False
    column_sets = [_result_get(result, "columns", []) for result in ok_results]
    first_columns = column_sets[0]
    if any(columns != first_columns for columns in column_sets):
        return False
    comparison = compare_row_set_batch(
        [_result_get(result, "rows", []) for result in ok_results],
        column_sets=column_sets,
    )
    if not comparison.has_mismatch:
        return False
    relaxed_rows = [
        [_relaxed_float_precision_row(row) for row in _result_get(result, "rows", [])]
        for result in ok_results
    ]
    relaxed_comparison = compare_row_set_batch(
        relaxed_rows,
        column_sets=column_sets,
    )
    if relaxed_comparison.is_exact_match():
        return True
    return relaxed_comparison.is_order_only_mismatch()


def has_clear_minority_backend(finding: Any, backends: list[str]) -> bool:
    suspicious = list(_get(finding, "suspicious_backends", []) or [])
    confidence = _get(finding, "confidence", "")
    return 0 < len(suspicious) < max(1, len(backends)) and confidence == "high"


def is_order_only_mismatch(normalized: dict[str, NormalizedResult | dict[str, Any]]) -> bool:
    ok_results = [result for result in normalized.values() if _result_get(result, "status") == "ok"]
    if len(ok_results) < 2:
        return False
    comparison = compare_row_set_batch(
        [_result_get(result, "rows", []) for result in ok_results],
        column_sets=[_result_get(result, "columns", []) for result in ok_results],
    )
    return comparison.is_order_only_mismatch()


def is_sort_tie_order_only_mismatch(case: Case) -> bool:
    op_index = _last_order_defining_operation_index(case)
    if op_index is None:
        return False
    op = case.program.operations[op_index]
    if op_kind(op) != "sort":
        return False
    try:
        sort_keys = normalize_sort_keys(op)
    except ValueError:
        return False
    rows = _reference_rows_before_operation(case, op_index)
    if rows is None:
        return False
    return rows_have_observable_duplicate_sort_key(rows, sort_keys)


def is_sort_topk_tie_cutoff_mismatch(case: Case) -> bool:
    op_index = _last_order_defining_operation_index(case)
    if op_index is None:
        return False
    op = case.program.operations[op_index]
    if op_kind(op) != "sort":
        return False
    try:
        sort_keys = normalize_sort_keys(op)
    except ValueError:
        return False
    rows = _reference_rows_before_operation(case, op_index)
    if rows is None or len(rows) < 2:
        return False
    sorted_rows = list(sort_row_mappings(rows, sort_keys))
    start = 0
    end = len(sorted_rows)
    saw_topk = False
    sort_columns = {key.column for key in sort_keys}
    for later in case.program.operations[op_index + 1 :]:
        kind = op_kind(later)
        if kind == "select":
            continue
        if kind in {"fill_null", "coalesce", "case_when"}:
            continue
        if kind == "mutate":
            if op_column(later) in sort_columns:
                return False
            continue
        if kind == "limit":
            try:
                limit = max(0, op_n(later))
            except (TypeError, ValueError):
                return False
            end = min(end, start + limit)
            saw_topk = True
            if sort_window_boundary_splits_observable_tie(sorted_rows, sort_keys, start, end):
                return True
            continue
        if kind == "offset":
            try:
                offset = max(0, op_n(later))
            except (TypeError, ValueError):
                return False
            start = min(end, start + offset)
            saw_topk = True
            if sort_window_boundary_splits_observable_tie(sorted_rows, sort_keys, start, end):
                return True
            continue
        if saw_topk:
            continue
        return False
    return False


def has_limit_or_offset_without_defined_order(case: Case) -> bool:
    order_defined = False
    for idx, op in enumerate(case.program.operations):
        kind = op_kind(op)
        if kind in {"sort", "running_sum", "row_number_filter"}:
            order_defined = True
            continue
        if kind == "limit":
            try:
                limit = op_n(op)
            except (TypeError, ValueError):
                limit = 1
            if not order_defined and _unordered_limit_changes_result(case, idx, limit):
                return True
            continue
        if kind == "offset":
            try:
                offset = op_n(op)
            except (TypeError, ValueError):
                offset = 1
            if not order_defined and _unordered_offset_changes_result(case, idx, offset):
                return True
            continue
        if kind in ROW_ORDER_PRESERVING_AFTER_ORDER:
            continue
        if kind == "sortedness_check":
            order_defined = True
            continue
        order_defined = False
    return False


def _unordered_limit_changes_result(case: Case, op_index: int, limit: int) -> bool:
    if limit <= 0:
        return False
    rows = _reference_rows_before_operation(case, op_index)
    if rows is None:
        return True
    return int(limit) < len(rows)


def _unordered_offset_changes_result(case: Case, op_index: int, offset: int) -> bool:
    if offset <= 0:
        return False
    rows = _reference_rows_before_operation(case, op_index)
    if rows is None:
        return True
    return int(offset) < len(rows)


def join_keys_contain_null(case: Case) -> bool:
    tables = {table.name: table for table in case.tables}
    for op in case.program.operations:
        if op_kind(op) != "join":
            continue
        left_keys, right_keys = join_key_pairs(op)
        right = tables.get(op_table(op))
        left_has_null = any(any(row.get(left_key) is None for left_key in left_keys) for row in case.tables[0].rows)
        right_has_null = bool(
            right and any(any(row.get(right_key) is None for right_key in right_keys) for row in right.rows)
        )
        if left_has_null or right_has_null:
            return True
    return False


def _has_empty_input_global_bool_aggregate(case: Case) -> bool:
    rows = [dict(row) for row in case.tables[0].rows] if case.tables else []
    for op in case.program.operations:
        kind = op_kind(op)
        if kind == "filter":
            try:
                rows = [
                    row
                    for row in rows
                    if evaluate_filter_predicate(row.get(condition_column(op)), condition_cmp(op), condition_value(op))
                ]
            except Exception:  # noqa: BLE001
                return False
            continue
        if kind != "aggregate":
            return False
        return not rows and bool(set(aggregate_functions(op)) & {"any", "all"})
    return False


def _relaxed_float_precision_row(row: list[Any]) -> list[Any]:
    return [_norm_value(value, preserve_float_precision=False) for value in row]


def _last_order_defining_operation_index(case: Case) -> int | None:
    for idx in range(len(case.program.operations) - 1, -1, -1):
        kind = op_kind(case.program.operations[idx])
        if kind in {"sort", "running_sum", "row_number_filter", "sortedness_check"}:
            return idx
        if kind in ROW_ORDER_PRESERVING_AFTER_ORDER:
            continue
        return None
    return None


def _reference_rows_before_operation(case: Case, op_index: int) -> list[dict[str, Any]] | None:
    prefix = Case(
        case_id=f"{case.case_id}-prefix-{op_index}",
        seed=case.seed,
        tables=case.tables,
        program=Program(
            program_id=f"{case.program.program_id}-prefix-{op_index}",
            seed=case.program.seed,
            operations=list(case.program.operations[:op_index]),
        ),
        metadata=case.metadata,
    )
    reference = reference_result(prefix)
    if reference is None or reference.status != "ok":
        return None
    return [
        {column: row[idx] for idx, column in enumerate(reference.columns)}
        for row in reference.rows
    ]


def _get(finding: Any, key: str, default: Any = None) -> Any:
    if isinstance(finding, dict):
        return finding.get(key, default)
    return getattr(finding, key, default)


def _result_get(result: NormalizedResult | dict[str, Any] | None, key: str, default: Any = "") -> Any:
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)
