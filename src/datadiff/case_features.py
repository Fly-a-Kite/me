from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from datadiff.dsl import Case
from datadiff.filtering import parse_filter_comparator
from datadiff.operation_semantics import (
    condition_cmp,
    condition_value,
    expr_kind,
    expr_operator,
    has_any_expr_kind,
    has_post_topk_filter as operations_have_post_topk_filter,
    has_operation,
    is_left_join,
    op_kind,
)

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
    "series_reflected_arithmetic_probe": (
        "polars_reflected_arithmetic_operand_order"
    ),
    "datafusion_grouped_null_topk_probe": "grouped_topk_null_sort_key",
    "uint64_isin_probe": "pandas_uint64_isin_precision",
    "tuple_anti_null_probe": "duckdb_tuple_anti_null_semantics",
    "setop_all_duplicate_probe": "datafusion_setop_all_duplicate_count",
    "json_predicate_order_probe": "duckdb_json_predicate_order_semantics",
    "sparse_mask_probe": "pandas_sparse_array_mask_semantics",
    "float_wrap_probe": "polars_float_wrap_numerical_semantics",
    "index_bool_probe": "pandas_index_bool_result_type",
    "empty_literal_groupby_probe": "polars_empty_literal_groupby_semantics",
    "arrow_string_eq_sum_probe": "pandas_arrow_string_eq_sum_semantics",
    "arrow_string_contains_na_probe": "pandas_arrow_string_contains_na_semantics",
    "arrow_timestamp_loc_slice_probe": "pandas_arrow_timestamp_loc_slice_semantics",
    "arrow_timestamp_index_attr_probe": "pandas_arrow_timestamp_index_attr_semantics",
    "eval_inplace_alias_probe": "pandas_eval_inplace_aliasing_semantics",
    "bool_reduction_skipna_probe": "pandas_bool_reduction_skipna_semantics",
    "arrow_bool_groupby_reduction_probe": "pandas_arrow_bool_groupby_reduction_semantics",
    "polars_timezone_filter_probe": "polars_timezone_filter_semantics",
    "dataset_isin_all_match_probe": "pyarrow_dataset_isin_all_match_semantics",
    "run_end_null_compute_probe": "pyarrow_run_end_null_compute_semantics",
    "large_string_partition_probe": "pyarrow_large_string_partition_schema_semantics",
    "hash_pivot_wider_probe": "pyarrow_hash_pivot_wider_order_semantics",
    "list_flatten_parent_indices_probe": "pyarrow_list_flatten_parent_indices_semantics",
    "rolling_mean_by_null_count_probe": "polars_rolling_mean_by_null_count_semantics",
    "csv_long_numeric_roundtrip_probe": "csv_long_numeric_roundtrip",
}


def case_contains_null(case: Case) -> bool:
    return any(value is None for table in case.tables for row in table.rows for value in row.values())


def case_contains_nan(case: Case) -> bool:
    return any(
        isinstance(value, float) and math.isnan(value)
        for table in case.tables
        for row in table.rows
        for value in row.values()
    )


def case_contains_inf(case: Case) -> bool:
    return any(
        isinstance(value, float) and math.isinf(value)
        for table in case.tables
        for row in table.rows
        for value in row.values()
    )


def case_contains_special_float(case: Case) -> bool:
    return case_contains_nan(case) or case_contains_inf(case)


def case_contains_non_ascii_string(case: Case) -> bool:
    return any(
        isinstance(value, str) and any(ord(ch) > 127 for ch in value)
        for table in case.tables
        for row in table.rows
        for value in row.values()
    )


def case_uses_unicode_case_mapping(case: Case) -> bool:
    return has_any_expr_kind(case.program.operations, {"string_lower", "string_upper"})


def case_uses_modulo(case: Case) -> bool:
    return any(
        op_kind(op) == "mutate" and expr_kind(op) == "arith_const" and expr_operator(op) == "mod"
        for op in case.program.operations
    )


def case_has_running_sum(case: Case) -> bool:
    return has_operation(case.program.operations, "running_sum")


def case_has_path_projection_keyed_pick(case: Case) -> bool:
    return has_any_expr_kind(case.program.operations, {"string_basename"}) and has_operation(
        case.program.operations, "row_number_filter"
    )


def last_probe_root(case: Case) -> str | None:
    return last_probe_root_for_operations(case.program.operations)


def last_probe_root_for_operations(operations: Iterable[dict[str, Any]]) -> str | None:
    for op in reversed(list(operations)):
        kind = op_kind(op)
        if kind == "confirmed_root_witness_probe":
            root = str(op.get("root_cause", "") or "")
            if root:
                return root
        root = PROBE_ROOTS.get(kind)
        if root is not None:
            return root
    return None


def case_has_null_filter_literal(case: Case) -> bool:
    for op in case.program.operations:
        if op_kind(op) != "filter" or condition_value(op) is not None:
            continue
        parsed = parse_filter_comparator(condition_cmp(op))
        base = parsed.base if parsed is not None else ""
        if base not in {"is_null", "is_not_null", "bool_predicate"}:
            return True
    return False


def case_has_special_float_filter_literal(case: Case) -> bool:
    """True when a filter compares against NaN or Infinity."""
    for op in case.program.operations:
        if op_kind(op) != "filter":
            continue
        value = condition_value(op)
        if isinstance(value, float) and not math.isfinite(value):
            return True
    return False


def case_has_outer_join_truth_filter(case: Case) -> bool:
    after_left_join = False
    for op in case.program.operations:
        kind = op_kind(op)
        if kind == "join":
            after_left_join = is_left_join(op)
            continue
        if kind == "filter" and after_left_join:
            parsed = parse_filter_comparator(condition_cmp(op))
            if parsed is not None and parsed.truth_test is not None:
                return True
            continue
        if kind in {"groupby", "aggregate"}:
            after_left_join = False
    return False


def case_has_post_topk_filter(case: Case) -> bool:
    return operations_have_post_topk_filter(case.program.operations)


def has_post_topk_filter(operations: Iterable[dict[str, Any]]) -> bool:
    return operations_have_post_topk_filter(operations)
