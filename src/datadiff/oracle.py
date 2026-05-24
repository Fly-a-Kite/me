from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from datadiff.dsl import Case, normalize_sort_keys
from datadiff.filtering import evaluate_filter_predicate, parse_filter_comparator
from datadiff.normalizer import NormalizedResult


@dataclass(slots=True)
class Finding:
    finding_id: str
    kind: str
    severity: str
    suspicious_backends: list[str]
    evidence: str
    signature: str
    root_cause: str = "unknown"
    oracle: str = "differential"
    confidence: str = "medium"
    triage_verdict: str = "unclassified"
    paper_status: str = "unclassified"
    triage_confidence: str = "low"
    false_positive: bool = False
    false_positive_reason: str = ""
    triage_evidence: str = ""
    recommendation: list[str] = field(default_factory=list)
    documentation_refs: list[dict[str, str]] = field(default_factory=list)
    mismatch_class: str = ""
    discovery_origin: str = "organic"
    source_issue: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _payload(norm: NormalizedResult) -> dict[str, Any]:
    return {
        "status": norm.status,
        "columns": norm.columns,
        "rows": norm.rows,
        "error_type": norm.error_type,
    }


def _signature(case: Case, normalized: dict[str, NormalizedResult], kind: str) -> str:
    payload = {
        "kind": kind,
        "ops": case.program.op_sequence(),
        "results": {k: _payload(v) for k, v in sorted(normalized.items())},
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def classify_root_cause(case: Case, normalized: dict[str, NormalizedResult], kind: str) -> str:
    ops = case.program.op_sequence()
    if kind == "exception_mismatch":
        return "exception_taxonomy"
    probe_root = _last_probe_root(case)
    if probe_root is not None:
        return probe_root
    if _case_has_path_projection_keyed_pick(case):
        return "path_projection_keyed_pick"
    if _case_has_running_sum(case):
        return "running_sum_precision"
    if _case_contains_special_float(case):
        return "nan_inf_semantics"
    if _case_uses_modulo(case):
        return "arithmetic_expression"
    if _case_has_reverse_division_columns(case):
        return "reverse_division_operand_order"
    if _case_has_grouped_topk_null_sort_key(case):
        return "grouped_topk_null_sort_key"
    if _case_has_float_group_key_instability(case, normalized):
        return "float_group_key_instability"
    if _case_has_negative_zero_comparison(case):
        return "negative_zero_comparison"
    if _case_has_tuple_absence_filter(case):
        return "tuple_absence_null_filter"
    if _case_has_outer_join_truth_filter(case):
        return "outer_join_truth_filter"
    if _case_has_post_topk_filter(case):
        return "topk_filter_pushdown"
    if _case_has_joined_order_offset_projection(case):
        return "joined_order_offset_projection"
    if _case_has_ordered_topk_projection(case):
        return "ordered_topk_projection"
    if any(op in {"groupby", "aggregate"} for op in ops):
        return "groupby_aggregation"
    if any(op == "join" for op in ops):
        return "join_semantics"
    if any(op == "filter" for op in ops):
        return "filter_predicate"
    if any(op == "mutate" for op in ops):
        kinds = {
            operation.get("expr", {}).get("kind", "")
            for operation in case.program.operations
            if operation.get("op") == "mutate"
        }
        if kinds & {"string_length", "string_lower"}:
            return "string_expression"
        if "cast" in kinds:
            return "type_cast"
        return "arithmetic_expression"
    if any(op in {"sort", "limit", "offset"} for op in ops):
        return "ordering_or_limit"
    ok_results = [r for r in normalized.values() if r.status == "ok"]
    if ok_results and len({tuple(r.columns) for r in ok_results}) > 1:
        return "schema_projection"
    if _case_contains_null(case):
        return "null_semantics"
    return "unknown"


def _case_contains_null(case: Case) -> bool:
    for table in case.tables:
        for row in table.rows:
            if any(v is None for v in row.values()):
                return True
    return False


PROBE_ROOTS = {
    "sortedness_check": "sortedness_null_placement",
    "random_case_probe": "simple_case_random_subject",
    "group_quantile_probe": "group_quantile_key_expression",
    "scalar_subquery_probe": "scalar_subquery_double_parentheses",
    "window_avg_probe": "window_avg_rows_frame",
    "struct_distinct_probe": "struct_distinct_unnest",
    "bit_compare_probe": "bit_compare_unequal_length",
    "round_even_probe": "round_even_float_scale",
    "float_literal_precision_probe": "duckdb_float_literal_precision",
    "timestamp_precision_filter_probe": "polars_timestamp_precision_filter",
    "series_rtruediv_probe": "series_rtruediv_operand_order",
    "uint64_isin_probe": "pandas_uint64_isin_precision",
    "tuple_anti_null_probe": "duckdb_tuple_anti_null_semantics",
    "setop_all_duplicate_probe": "datafusion_setop_all_duplicate_count",
    "json_predicate_order_probe": "duckdb_json_predicate_order_semantics",
    "sparse_mask_probe": "pandas_sparse_array_mask_semantics",
    "float_wrap_probe": "polars_float_wrap_numerical_semantics",
    "index_bool_probe": "pandas_index_bool_result_type",
    "empty_literal_groupby_probe": "polars_empty_literal_groupby_semantics",
    "arrow_string_eq_sum_probe": "pandas_arrow_string_eq_sum_semantics",
    "arrow_timestamp_loc_slice_probe": "pandas_arrow_timestamp_loc_slice_semantics",
    "arrow_timestamp_index_attr_probe": "pandas_arrow_timestamp_index_attr_semantics",
    "eval_inplace_alias_probe": "pandas_eval_inplace_aliasing_semantics",
    "bool_reduction_skipna_probe": "pandas_bool_reduction_skipna_semantics",
    "dataset_isin_all_match_probe": "pyarrow_dataset_isin_all_match_semantics",
    "run_end_null_compute_probe": "pyarrow_run_end_null_compute_semantics",
    "large_string_partition_probe": "pyarrow_large_string_partition_schema_semantics",
    "hash_pivot_wider_probe": "pyarrow_hash_pivot_wider_order_semantics",
    "rolling_mean_by_null_count_probe": "polars_rolling_mean_by_null_count_semantics",
    "csv_long_numeric_roundtrip_probe": "csv_long_numeric_roundtrip",
}


def _last_probe_root(case: Case) -> str | None:
    for op in reversed(case.program.operations):
        root = PROBE_ROOTS.get(str(op.get("op", "")))
        if root is not None:
            return root
    return None


def _case_contains_special_float(case: Case) -> bool:
    import math

    for table in case.tables:
        for row in table.rows:
            for value in row.values():
                if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                    return True
    return False


def _case_has_running_sum(case: Case) -> bool:
    return any(op.get("op") == "running_sum" for op in case.program.operations)


def _case_has_path_projection_keyed_pick(case: Case) -> bool:
    has_basename = any(
        op.get("op") == "mutate" and op.get("expr", {}).get("kind") == "string_basename"
        for op in case.program.operations
    )
    has_keyed_pick = any(op.get("op") == "row_number_filter" for op in case.program.operations)
    return has_basename and has_keyed_pick


def _case_has_sortedness_check(case: Case) -> bool:
    return any(op.get("op") == "sortedness_check" for op in case.program.operations)


def _case_has_random_case_probe(case: Case) -> bool:
    return any(op.get("op") == "random_case_probe" for op in case.program.operations)


def _case_has_group_quantile_probe(case: Case) -> bool:
    return any(op.get("op") == "group_quantile_probe" for op in case.program.operations)


def _case_has_scalar_subquery_probe(case: Case) -> bool:
    return any(op.get("op") == "scalar_subquery_probe" for op in case.program.operations)


def _case_has_window_avg_probe(case: Case) -> bool:
    return any(op.get("op") == "window_avg_probe" for op in case.program.operations)


def _case_has_struct_distinct_probe(case: Case) -> bool:
    return any(op.get("op") == "struct_distinct_probe" for op in case.program.operations)


def _case_has_bit_compare_probe(case: Case) -> bool:
    return any(op.get("op") == "bit_compare_probe" for op in case.program.operations)


def _case_has_round_even_probe(case: Case) -> bool:
    return any(op.get("op") == "round_even_probe" for op in case.program.operations)


def _case_has_series_rtruediv_probe(case: Case) -> bool:
    return any(op.get("op") == "series_rtruediv_probe" for op in case.program.operations)


def _case_has_uint64_isin_probe(case: Case) -> bool:
    return any(op.get("op") == "uint64_isin_probe" for op in case.program.operations)


def _case_has_tuple_anti_null_probe(case: Case) -> bool:
    return any(op.get("op") == "tuple_anti_null_probe" for op in case.program.operations)


def _case_has_json_predicate_order_probe(case: Case) -> bool:
    return any(op.get("op") == "json_predicate_order_probe" for op in case.program.operations)


def _case_has_sparse_mask_probe(case: Case) -> bool:
    return any(op.get("op") == "sparse_mask_probe" for op in case.program.operations)


def _case_has_float_wrap_probe(case: Case) -> bool:
    return any(op.get("op") == "float_wrap_probe" for op in case.program.operations)


def _case_has_index_bool_probe(case: Case) -> bool:
    return any(op.get("op") == "index_bool_probe" for op in case.program.operations)


def _case_has_empty_literal_groupby_probe(case: Case) -> bool:
    return any(op.get("op") == "empty_literal_groupby_probe" for op in case.program.operations)


def _case_has_arrow_string_eq_sum_probe(case: Case) -> bool:
    return any(op.get("op") == "arrow_string_eq_sum_probe" for op in case.program.operations)


def _case_has_arrow_timestamp_loc_slice_probe(case: Case) -> bool:
    return any(op.get("op") == "arrow_timestamp_loc_slice_probe" for op in case.program.operations)


def _case_has_arrow_timestamp_index_attr_probe(case: Case) -> bool:
    return any(op.get("op") == "arrow_timestamp_index_attr_probe" for op in case.program.operations)


def _case_has_eval_inplace_alias_probe(case: Case) -> bool:
    return any(op.get("op") == "eval_inplace_alias_probe" for op in case.program.operations)


def _case_has_dataset_isin_all_match_probe(case: Case) -> bool:
    return any(op.get("op") == "dataset_isin_all_match_probe" for op in case.program.operations)


def _case_has_large_string_partition_probe(case: Case) -> bool:
    return any(op.get("op") == "large_string_partition_probe" for op in case.program.operations)


def _case_has_hash_pivot_wider_probe(case: Case) -> bool:
    return any(op.get("op") == "hash_pivot_wider_probe" for op in case.program.operations)


def _case_has_rolling_mean_by_null_count_probe(case: Case) -> bool:
    return any(op.get("op") == "rolling_mean_by_null_count_probe" for op in case.program.operations)


def _case_uses_modulo(case: Case) -> bool:
    return any(
        op.get("op") == "mutate"
        and op.get("expr", {}).get("kind") == "arith_const"
        and op.get("expr", {}).get("op") == "mod"
        for op in case.program.operations
    )


def _case_has_reverse_division_columns(case: Case) -> bool:
    return any(
        op.get("op") == "mutate" and op.get("expr", {}).get("kind") == "reverse_division_columns"
        for op in case.program.operations
    )


def _case_has_outer_join_truth_filter(case: Case) -> bool:
    after_left_join = False
    for op in case.program.operations:
        if op.get("op") == "join":
            after_left_join = op.get("how") == "left"
        elif op.get("op") == "filter" and after_left_join:
            parsed = parse_filter_comparator(op.get("cmp", ""))
            if parsed is not None and parsed.truth_test is not None:
                return True
        elif op.get("op") in {"groupby", "aggregate"}:
            after_left_join = False
    return False


def _case_has_post_topk_filter(case: Case) -> bool:
    ops = case.program.operations
    for sort_idx, op in enumerate(ops):
        if op.get("op") != "sort":
            continue
        for topk_idx in range(sort_idx + 1, len(ops)):
            if ops[topk_idx].get("op") not in {"limit", "offset"}:
                continue
            if any(later.get("op") == "filter" for later in ops[topk_idx + 1 :]):
                return True
    return False


def _case_has_joined_order_offset_projection(case: Case) -> bool:
    saw_join = False
    saw_order_after_join = False
    for op in case.program.operations:
        kind = op.get("op")
        if kind == "join":
            saw_join = True
        elif kind == "sort" and saw_join:
            saw_order_after_join = True
        elif kind == "offset" and saw_order_after_join:
            return True
    return False


def _case_has_ordered_topk_projection(case: Case) -> bool:
    saw_order = False
    saw_topk_after_order = False
    for op in case.program.operations:
        kind = op.get("op")
        if kind == "sort":
            if saw_topk_after_order:
                return True
            saw_order = True
        elif kind in {"limit", "offset"} and saw_order:
            saw_topk_after_order = True
        elif kind == "select" and saw_topk_after_order:
            return True
    return False


def _case_has_negative_zero_comparison(case: Case) -> bool:
    zero_source_columns = _columns_with_numeric_zero(case)
    negative_zero_columns = set()
    for op in case.program.operations:
        kind = op.get("op")
        if kind == "mutate":
            expr = op.get("expr", {})
            if (
                expr.get("kind") == "arith_const"
                and expr.get("op") == "mul"
                and _is_numeric_value(expr.get("value"), -1.0)
                and expr.get("source") in zero_source_columns
            ):
                negative_zero_columns.add(str(op.get("column")))
        elif kind == "filter" and op.get("column") in negative_zero_columns:
            parsed = parse_filter_comparator(op.get("cmp"))
            if parsed is not None and _filter_comparator_touches_zero(parsed.base, op.get("value")):
                return True
    return False


def _filter_comparator_touches_zero(base: str, value: Any) -> bool:
    if base in {">", ">=", "<", "<=", "==", "!="}:
        return _is_numeric_value(value, 0.0)
    if base == "range_closed" and isinstance(value, (list, tuple)) and len(value) == 2:
        lower, upper = value
        if _is_numeric_value(lower, 0.0) or _is_numeric_value(upper, 0.0):
            return True
        if _is_orderable_number(lower) and _is_orderable_number(upper):
            return float(lower) < 0.0 < float(upper)
    return False


def _columns_with_numeric_zero(case: Case) -> set[str]:
    columns = set()
    for table in case.tables:
        for row in table.rows:
            for column, value in row.items():
                if _is_numeric_value(value, 0.0):
                    columns.add(column)
    return columns


def _is_numeric_value(value: Any, target: float) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return float(value) == target
    except Exception:
        return False


def _is_orderable_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        number = float(value)
    except Exception:
        return False
    return not math.isnan(number)


def _case_has_tuple_absence_filter(case: Case) -> bool:
    return any(op.get("op") == "tuple_absence_filter" for op in case.program.operations)


def _case_has_grouped_topk_null_sort_key(case: Case) -> bool:
    if not case.tables:
        return False
    table_samples = {
        table.name: {
            column.name: [row.get(column.name) for row in table.rows]
            for column in table.columns
        }
        for table in case.tables
    }
    samples = dict(table_samples.get(case.tables[0].name, {}))
    grouped = False
    ops = case.program.operations
    for idx, op in enumerate(ops):
        kind = op.get("op")
        if kind == "join":
            samples = _join_samples(samples, table_samples.get(str(op.get("table", "")), {}), op)
        elif kind == "filter":
            samples = _filter_samples(samples, op)
        elif kind == "select":
            selected = [str(column) for column in op.get("columns", []) if str(column) in samples]
            samples = {column: samples[column] for column in selected}
        elif kind == "mutate":
            column = str(op.get("column", ""))
            values = _eval_expr_samples(samples, op.get("expr", {}))
            if column and values is not None:
                samples[column] = values
        elif kind == "groupby":
            samples = _groupby_output_samples(samples, op)
            grouped = True
        elif kind == "sort" and grouped:
            try:
                sort_columns = [key.column for key in normalize_sort_keys(op) if key.column in samples]
            except ValueError:
                sort_columns = []
            if any(any(value is None for value in samples[column]) for column in sort_columns):
                if any(later.get("op") in {"limit", "offset"} for later in ops[idx + 1 :]):
                    return True
    return False


def _join_samples(
    left_samples: dict[str, list[Any]],
    right_samples: dict[str, list[Any]],
    op: dict[str, Any],
) -> dict[str, list[Any]]:
    left_on = str(op.get("left_on", ""))
    right_on = str(op.get("right_on", ""))
    if not right_samples or left_on not in left_samples or right_on not in right_samples:
        return left_samples

    right_extra_columns = [
        column
        for column in right_samples
        if column != right_on and column not in left_samples
    ]
    output_columns = list(left_samples) + right_extra_columns
    right_index: dict[Any, list[dict[str, Any]]] = {}
    for row in _rows_from_samples(right_samples):
        key = row.get(right_on)
        if key is not None:
            right_index.setdefault(key, []).append(row)

    joined_rows: list[dict[str, Any]] = []
    how = str(op.get("how", "inner"))
    for left in _rows_from_samples(left_samples):
        key = left.get(left_on)
        matches = [] if key is None else right_index.get(key, [])
        if matches:
            for right in matches:
                row = dict(left)
                for column in right_extra_columns:
                    row[column] = right.get(column)
                joined_rows.append(row)
        elif how == "left":
            row = dict(left)
            for column in right_extra_columns:
                row[column] = None
            joined_rows.append(row)
    return _samples_from_rows(joined_rows, output_columns)


def _rows_from_samples(samples: dict[str, list[Any]]) -> list[dict[str, Any]]:
    row_count = min((len(values) for values in samples.values()), default=0)
    return [
        {column: values[idx] for column, values in samples.items()}
        for idx in range(row_count)
    ]


def _samples_from_rows(rows: list[dict[str, Any]], columns: list[str]) -> dict[str, list[Any]]:
    return {
        column: [row.get(column) for row in rows]
        for column in columns
    }


def _case_has_float_group_key_instability(
    case: Case,
    normalized: dict[str, NormalizedResult],
) -> bool:
    groupby_keys = _final_groupby_keys(case)
    if not groupby_keys:
        return False
    column_types = _column_types_before_groupby(case)
    if not any(column_types.get(key) == "float" for key in groupby_keys):
        return False
    if not _case_uses_arithmetic_float_lineage(case, groupby_keys):
        return False

    ok_results = [result for result in normalized.values() if result.status == "ok"]
    if len(ok_results) < 2:
        return False
    row_counts = {len(result.rows) for result in ok_results}
    if len(row_counts) < 2:
        return False
    return any(_has_duplicate_normalized_rows(result) for result in ok_results)


def _final_groupby_keys(case: Case) -> list[str]:
    for op in reversed(case.program.operations):
        if op.get("op") == "groupby":
            return [str(key) for key in op.get("keys", [])]
    return []


def _column_types_before_groupby(case: Case) -> dict[str, str]:
    if not case.tables:
        return {}
    table_by_name = {table.name: table for table in case.tables}
    col_types = {column.name: column.type for column in case.tables[0].columns}
    for op in case.program.operations:
        kind = op.get("op")
        if kind == "join":
            right = table_by_name.get(str(op.get("table", "")))
            if right is None:
                continue
            right_on = str(op.get("right_on", ""))
            for column in right.columns:
                if column.name == right_on or column.name in col_types:
                    continue
                col_types[column.name] = column.type
        elif kind == "select":
            selected = {str(column) for column in op.get("columns", [])}
            col_types = {name: typ for name, typ in col_types.items() if name in selected}
        elif kind == "mutate":
            out_type = _expr_output_type(op.get("expr", {}), col_types)
            if out_type is not None:
                col_types[str(op.get("column", ""))] = out_type
        elif kind == "groupby":
            return col_types
    return col_types


def _case_uses_arithmetic_float_lineage(case: Case, groupby_keys: list[str]) -> bool:
    lineage: dict[str, set[str]] = {}
    if case.tables:
        for column in case.tables[0].columns:
            lineage[column.name] = {column.name}
        for table in case.tables[1:]:
            for column in table.columns:
                lineage.setdefault(column.name, {column.name})
    arithmetic_float_columns: set[str] = set()
    for op in case.program.operations:
        kind = op.get("op")
        if kind == "select":
            selected = {str(column) for column in op.get("columns", [])}
            lineage = {name: deps for name, deps in lineage.items() if name in selected}
        elif kind == "mutate":
            column = str(op.get("column", ""))
            expr = op.get("expr", {})
            source = str(expr.get("source", ""))
            deps = set(lineage.get(source, {source}))
            if column:
                lineage[column] = deps | {column}
            if (
                expr.get("kind") == "arith_const"
                and expr.get("op") == "div"
                and column
            ) or (expr.get("kind") == "cast" and expr.get("to") == "float" and column):
                arithmetic_float_columns.add(column)
        elif kind == "groupby":
            break
    for key in groupby_keys:
        deps = lineage.get(key, {key})
        if deps & arithmetic_float_columns:
            return True
        if key in arithmetic_float_columns:
            return True
    return False


def _expr_output_type(expr: dict[str, Any], col_types: dict[str, str]) -> str | None:
    source = str(expr.get("source", ""))
    source_type = col_types.get(source)
    if source_type is None:
        return None
    kind = expr.get("kind")
    if kind == "add_const":
        return source_type if source_type in {"int", "float"} else None
    if kind == "arith_const":
        op = expr.get("op")
        if source_type not in {"int", "float"} or op not in {"sub", "mul", "div", "mod"}:
            return None
        return "float" if op == "div" or source_type == "float" else source_type
    if kind == "cast" and expr.get("to") == "float":
        return "float" if source_type in {"int", "float"} else None
    if kind == "string_length":
        return "int" if source_type == "str" else None
    if kind == "string_lower":
        return "str" if source_type == "str" else None
    if kind == "string_basename":
        return "str" if source_type == "str" else None
    return None


def _has_duplicate_normalized_rows(result: NormalizedResult) -> bool:
    seen: set[str] = set()
    for row in result.rows:
        key = json.dumps(row, ensure_ascii=False, sort_keys=True)
        if key in seen:
            return True
        seen.add(key)
    return False


def _filter_samples(samples: dict[str, list[Any]], op: dict[str, Any]) -> dict[str, list[Any]]:
    column = str(op.get("column", ""))
    values = samples.get(column)
    if values is None:
        return samples
    mask = [_compare_value(value, str(op.get("cmp", "")), op.get("value")) for value in values]
    return {
        name: [value for value, keep in zip(column_values, mask) if keep]
        for name, column_values in samples.items()
    }


def _compare_value(left: Any, comparator: str, right: Any) -> bool:
    try:
        return evaluate_filter_predicate(left, comparator, right)
    except Exception:
        return False


def _eval_expr_samples(samples: dict[str, list[Any]], expr: dict[str, Any]) -> list[Any] | None:
    source = str(expr.get("source", ""))
    values = samples.get(source)
    if values is None:
        return None
    out = []
    for value in values:
        if value is None:
            out.append(None)
            continue
        try:
            kind = expr.get("kind")
            if kind == "add_const":
                out.append(value + expr.get("value", 0))
            elif kind == "arith_const":
                op = expr.get("op")
                operand = expr.get("value", 0)
                if op == "sub":
                    out.append(value - operand)
                elif op == "mul":
                    out.append(value * operand)
                elif op == "div":
                    out.append(value / operand)
                elif op == "mod":
                    out.append(value % operand)
                else:
                    out.append(None)
            elif kind == "cast" and expr.get("to") == "float":
                out.append(float(value))
            elif kind == "string_length":
                out.append(len(value) if isinstance(value, str) else None)
            elif kind == "string_lower":
                out.append(value.lower() if isinstance(value, str) else None)
            elif kind == "string_basename":
                text = str(value)
                slash = max(text.rfind("/"), text.rfind("\\"))
                out.append(text[slash + 1 :] if slash >= 0 else text)
            else:
                out.append(None)
        except Exception:
            out.append(None)
    return out


def _groupby_output_samples(samples: dict[str, list[Any]], op: dict[str, Any]) -> dict[str, list[Any]]:
    keys = [str(key) for key in op.get("keys", []) if str(key) in samples]
    row_count = min((len(samples[key]) for key in keys), default=0)
    groups: dict[tuple[Any, ...], list[int]] = {}
    for idx in range(row_count):
        key_tuple = tuple(samples[key][idx] for key in keys)
        groups.setdefault(key_tuple, []).append(idx)
    out: dict[str, list[Any]] = {key: [] for key in keys}
    for key_tuple in groups:
        for idx, key in enumerate(keys):
            out[key].append(key_tuple[idx])
    for agg in op.get("aggs", []):
        alias = str(agg.get("as", ""))
        source_values = samples.get(str(agg.get("column", "")), [])
        if not alias:
            continue
        values = []
        for indices in groups.values():
            group_values = [source_values[idx] for idx in indices if idx < len(source_values)]
            non_null = [value for value in group_values if value is not None]
            func = agg.get("func")
            if func == "count":
                values.append(len(non_null))
            elif func == "nunique":
                values.append(len(set(non_null)))
            elif not non_null:
                values.append(None)
            elif func == "any":
                values.append(any(bool(value) for value in non_null))
            elif func == "all":
                values.append(all(bool(value) for value in non_null))
            elif func == "sum":
                values.append(sum(non_null))
            elif func == "min":
                values.append(min(non_null))
            elif func == "max":
                values.append(max(non_null))
            else:
                values.append(None)
        out[alias] = values
    return out


def evaluate_case(case: Case, normalized: dict[str, NormalizedResult]) -> list[Finding]:
    findings: list[Finding] = []
    statuses = {b: r.status for b, r in normalized.items()}
    ok = {b: r for b, r in normalized.items() if r.status == "ok"}
    non_ok = {b: r for b, r in normalized.items() if r.status != "ok"}

    if ok and non_ok:
        sig = _signature(case, normalized, "accept_reject_mismatch")
        kind = "accept_reject_mismatch"
        findings.append(Finding(
            finding_id=f"finding-{sig}",
            kind=kind,
            severity="high",
            suspicious_backends=list(non_ok),
            evidence=f"Some backends accepted while others rejected: {statuses}",
            signature=sig,
            root_cause=classify_root_cause(case, normalized, kind),
            oracle="differential",
            confidence="high",
            mismatch_class="accept_reject",
        ))
        return findings

    if len(non_ok) == len(normalized) and len(set((r.status, r.error_type) for r in non_ok.values())) > 1:
        sig = _signature(case, normalized, "exception_mismatch")
        kind = "exception_mismatch"
        findings.append(Finding(
            finding_id=f"finding-{sig}",
            kind=kind,
            severity="medium",
            suspicious_backends=list(non_ok),
            evidence=f"All backends rejected but with different errors: { {b: r.error_type for b, r in non_ok.items()} }",
            signature=sig,
            root_cause=classify_root_cause(case, normalized, kind),
            oracle="differential",
            confidence="medium",
            mismatch_class="exception_taxonomy",
        ))
        return findings

    if len(ok) >= 2:
        payloads = {b: _payload(r) for b, r in ok.items()}
        unique = {}
        for b, p in payloads.items():
            key = json.dumps(p, ensure_ascii=False, sort_keys=True)
            unique.setdefault(key, []).append(b)
        if len(unique) > 1:
            group_sizes = [len(group) for group in unique.values()]
            max_size = max(group_sizes)
            if group_sizes.count(max_size) > 1:
                suspicious = sorted(ok)
            else:
                largest_group = next(group for group in unique.values() if len(group) == max_size)
                suspicious = sorted(b for group in unique.values() if group is not largest_group for b in group)
            sig = _signature(case, normalized, "semantic_output_mismatch")
            kind = "semantic_output_mismatch"
            shapes = {b: (len(r.rows), len(r.columns)) for b, r in ok.items()}
            mismatch_class = _semantic_mismatch_class(ok)
            findings.append(Finding(
                finding_id=f"finding-{sig}",
                kind=kind,
                severity="critical",
                suspicious_backends=suspicious,
                evidence=f"Backends returned different canonical tables; mismatch_class={mismatch_class}; shapes={shapes}",
                signature=sig,
                root_cause=classify_root_cause(case, normalized, kind),
                oracle="differential",
                confidence="high" if len(suspicious) < len(ok) else "medium",
                mismatch_class=mismatch_class,
            ))
    return findings


def _semantic_mismatch_class(ok_results: dict[str, NormalizedResult]) -> str:
    columns = [tuple(result.columns) for result in ok_results.values()]
    if len(set(columns)) > 1:
        return "schema"
    row_counts = [len(result.rows) for result in ok_results.values()]
    if len(set(row_counts)) > 1:
        return "row_count"
    ordered = [_stable_rows(result.rows) for result in ok_results.values()]
    if len({tuple(rows) for rows in ordered}) <= 1:
        return "none"
    unordered = [tuple(sorted(rows)) for rows in ordered]
    if len(set(unordered)) == 1:
        return "row_order"
    return "value"


def _stable_rows(rows: list[list[Any]]) -> list[str]:
    return [json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows]
