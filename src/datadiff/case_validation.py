from __future__ import annotations

from typing import Any

from datadiff.dsl import Case
from datadiff.identifiers import is_reserved_output_name
from datadiff.join_keys import join_key_pairs
from datadiff.operation_semantics import (
    op_branches,
    op_kind,
    op_output_alias,
    op_quantiles,
    op_rows,
    op_table,
    op_values,
    op_literal,
)
from datadiff.program_state import ProgramState, apply_operation_state
from datadiff.program_validation import (
    ValidationContext,
    valid_digit_string_values,
    valid_float_literal_text,
    valid_group_quantile_values,
    validate_core_operation,
)


BOOL_ALIAS_PROBES = frozenset(
    {
        "arrow_string_eq_sum_probe",
        "arrow_string_contains_na_probe",
        "arrow_timestamp_index_attr_probe",
        "arrow_timestamp_loc_slice_probe",
        "arrow_bool_groupby_reduction_probe",
        "bit_compare_probe",
        "bool_reduction_skipna_probe",
        "dataset_isin_all_match_probe",
        "empty_literal_groupby_probe",
        "eval_inplace_alias_probe",
        "float_wrap_probe",
        "hash_pivot_wider_probe",
        "index_bool_probe",
        "json_predicate_order_probe",
        "large_string_partition_probe",
        "list_flatten_parent_indices_probe",
        "rolling_mean_by_null_count_probe",
        "round_even_probe",
        "run_end_null_compute_probe",
        "series_rtruediv_probe",
        "series_reflected_arithmetic_probe",
        "datafusion_grouped_null_topk_probe",
        "confirmed_root_witness_probe",
        "setop_all_duplicate_probe",
        "sparse_mask_probe",
        "struct_distinct_probe",
        "timestamp_precision_filter_probe",
        "polars_timezone_filter_probe",
        "tuple_anti_null_probe",
        "uint64_isin_probe",
        "window_avg_probe",
    }
)


CORE_OPERATIONS = frozenset(
    {
        "aggregate",
        "anti_join",
        "case_when",
        "coalesce",
        "distinct",
        "drop_nulls",
        "fill_null",
        "filter",
        "groupby",
        "join",
        "limit",
        "mutate",
        "offset",
        "row_number_filter",
        "running_sum",
        "select",
        "semi_join",
        "sort",
        "sortedness_check",
        "tuple_absence_filter",
        "union_all",
    }
)


def validate_case_program(case: Case) -> list[str]:
    if not case.tables:
        return ["case has no tables"]
    errors: list[str] = []
    table_names = [table.name for table in case.tables]
    duplicate_table_names = _duplicates(table_names)
    for name in duplicate_table_names:
        errors.append(f"case has duplicate table name {name!r}")
    for table in case.tables:
        for column in _duplicates([column.name for column in table.columns]):
            errors.append(f"table {table.name!r} contains duplicate column {column!r}")
    if errors:
        return errors
    tables = {table.name: table for table in case.tables}
    state = ProgramState.from_table(case.tables[0])

    for idx, op in enumerate(case.program.operations):
        kind = op_kind(op)
        ctx = ValidationContext(tables=tables, state=state, errors=errors)
        if kind in CORE_OPERATIONS:
            if validate_core_operation(ctx, op, idx):
                apply_operation_state(state, op)
                if kind == "join":
                    _extend_join_state(state, tables, op)
            continue
        if kind == "random_case_probe":
            _validate_random_case_probe(state, errors, op, idx)
        elif kind == "group_quantile_probe":
            _validate_group_quantile_probe(state, errors, op, idx)
        elif kind in {"scalar_subquery_probe", "window_avg_probe"}:
            _validate_bool_alias_probe(state, errors, op, idx, kind)
        elif kind == "float_literal_precision_probe":
            _validate_float_literal_precision_probe(state, errors, op, idx)
        elif kind in BOOL_ALIAS_PROBES:
            _validate_bool_alias_probe(state, errors, op, idx, kind)
        elif kind == "csv_long_numeric_roundtrip_probe":
            _validate_csv_long_numeric_roundtrip_probe(state, errors, op, idx)
        else:
            errors.append(f"op {idx}: unknown operation {kind!r}")
    return errors


def _duplicates(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


def _extend_join_state(state: ProgramState, tables: dict[str, Any], op: Any) -> None:
    right = tables.get(op_table(op))
    if right is None:
        return
    _, right_keys = join_key_pairs(op)
    right_key_set = set(right_keys)
    for column in right.columns:
        if column.name in right_key_set or column.name in state.available:
            continue
        state.columns.append(column.name)
        state.column_types[column.name] = column.type


def _validate_alias(errors: list[str], op: Any, idx: int, operation_name: str) -> str:
    alias = op_output_alias(op)
    if not alias:
        errors.append(f"op {idx}: {operation_name} output alias is empty")
    elif is_reserved_output_name(alias):
        errors.append(f"op {idx}: {operation_name} output alias {alias!r} is reserved")
    return alias


def _validate_random_case_probe(state: ProgramState, errors: list[str], op: Any, idx: int) -> None:
    alias = _validate_alias(errors, op, idx, "random_case_probe")
    try:
        if op_rows(op) <= 0:
            errors.append(f"op {idx}: random_case_probe rows must be positive")
    except (TypeError, ValueError):
        errors.append(f"op {idx}: random_case_probe rows must be an integer")
    try:
        branches = op_branches(op)
        if branches <= 0 or branches > 16:
            errors.append(f"op {idx}: random_case_probe branches must be between 1 and 16")
    except (TypeError, ValueError):
        errors.append(f"op {idx}: random_case_probe branches must be an integer")
    if alias:
        state.collapse_to_alias(alias, "bool")


def _validate_group_quantile_probe(state: ProgramState, errors: list[str], op: Any, idx: int) -> None:
    alias = _validate_alias(errors, op, idx, "group_quantile_probe")
    values = op_values(op)
    quantiles = op_quantiles(op)
    if not valid_group_quantile_values(values, quantiles):
        errors.append(f"op {idx}: group_quantile_probe requires numeric values and quantiles in [0, 1]")
    if alias:
        state.collapse_to_alias(alias, "bool")


def _validate_bool_alias_probe(
    state: ProgramState,
    errors: list[str],
    op: Any,
    idx: int,
    operation_name: str,
) -> None:
    alias = _validate_alias(errors, op, idx, operation_name)
    if alias:
        state.collapse_to_alias(alias, "bool")


def _validate_float_literal_precision_probe(state: ProgramState, errors: list[str], op: Any, idx: int) -> None:
    alias = _validate_alias(errors, op, idx, "float_literal_precision_probe")
    if not valid_float_literal_text(op_literal(op)):
        errors.append(f"op {idx}: float_literal_precision_probe literal is invalid")
    if alias:
        state.collapse_to_alias(alias, "bool")


def _validate_csv_long_numeric_roundtrip_probe(state: ProgramState, errors: list[str], op: Any, idx: int) -> None:
    alias = _validate_alias(errors, op, idx, "csv_long_numeric_roundtrip_probe")
    if not valid_digit_string_values(op_values(op)):
        errors.append(f"op {idx}: csv_long_numeric_roundtrip_probe requires non-empty digit-string values")
    if alias:
        state.collapse_to_alias(alias, "bool")
