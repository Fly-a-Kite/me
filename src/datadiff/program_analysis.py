from __future__ import annotations

from typing import Any

from datadiff.dsl import Case, normalize_sort_keys
from datadiff.join_keys import join_key_pairs
from datadiff.operation_semantics import (
    aggregate_alias,
    aggregate_column,
    aggregate_func,
    aggregate_specs,
    expr_kind,
    expr_operator,
    expr_source,
    expr_target_type,
    groupby_keys,
    op_column,
    op_columns,
    op_kind,
    op_table,
)
from datadiff.operation_type_semantics import aggregate_result_type
from datadiff.program_state import state_before_operation


def numeric_cast_string_columns(case: Case) -> set[str]:
    columns: set[str] = set()
    for index, operation in enumerate(case.program.operations):
        if op_kind(operation) != "mutate":
            continue
        column = op_column(operation)
        columns.discard(column)
        if expr_kind(operation) != "cast" or expr_target_type(operation) != "str":
            continue
        state = state_before_operation(case, index)
        if state.column_types.get(expr_source(operation)) in {"int", "float"}:
            columns.add(column)
    return columns


def case_uses_precision_sensitive_float_arithmetic(case: Case) -> bool:
    if not case.tables:
        return False
    table_by_name = {table.name: table for table in case.tables}
    column_types = {column.name: column.type for column in case.tables[0].columns}
    float_lineage = {column.name for column in case.tables[0].columns if column.type == "float"}
    precision_columns: set[str] = set(float_lineage)
    saw_precision_arithmetic = False
    for operation in case.program.operations:
        kind = op_kind(operation)
        if kind == "join":
            right_table = table_by_name.get(op_table(operation))
            if right_table is None:
                continue
            _, right_keys = _join_keys(operation)
            right_key_set = set(right_keys[:1])
            for column in right_table.columns:
                if column.name in right_key_set or column.name in column_types:
                    continue
                column_types[column.name] = column.type
                if column.type == "float":
                    float_lineage.add(column.name)
                    precision_columns.add(column.name)
        elif kind == "select":
            selected_columns = set(op_columns(operation))
            column_types = {name: value_type for name, value_type in column_types.items() if name in selected_columns}
            float_lineage &= selected_columns
            precision_columns &= selected_columns
        elif kind == "mutate":
            output_column = op_column(operation)
            result_type = _mutate_float_precision_result_type(operation.get("expr", {}), column_types)
            if not output_column or result_type is None:
                continue
            source_name = expr_source(operation)
            column_types[output_column] = result_type
            if result_type != "float":
                continue
            source_is_float = source_name in float_lineage or column_types.get(source_name) == "float"
            precision_sensitive = (
                expr_kind(operation) == "cast"
                or source_is_float
                or expr_operator(operation) in {"div", "mul"}
            )
            if precision_sensitive:
                saw_precision_arithmetic = True
                float_lineage.add(output_column)
                precision_columns.add(output_column)
        elif kind in {"groupby", "aggregate"}:
            input_column_types = dict(column_types)
            aggregate_outputs = _float_precision_aggregate_outputs(operation, input_column_types, precision_columns)
            if aggregate_outputs:
                saw_precision_arithmetic = True
            if kind == "groupby":
                keys = groupby_keys(operation)
                column_types = {key: input_column_types.get(key, "derived") for key in keys}
                precision_columns = {key for key in keys if key in precision_columns}
                float_lineage = {key for key in keys if key in float_lineage}
            else:
                column_types = {}
                precision_columns = set()
                float_lineage = set()
            for aggregate in aggregate_specs(operation):
                output_column = aggregate_alias(aggregate)
                if not output_column:
                    continue
                source_type = input_column_types.get(aggregate_column(aggregate), "float")
                result_type = aggregate_result_type(source_type, aggregate_func(aggregate))
                column_types[output_column] = result_type
                if output_column in aggregate_outputs:
                    precision_columns.add(output_column)
                    float_lineage.add(output_column)
        elif kind == "sort":
            try:
                sort_columns = {sort_key.column for sort_key in normalize_sort_keys(operation)}
            except ValueError:
                sort_columns = set()
            if saw_precision_arithmetic and sort_columns & precision_columns:
                return True
    return saw_precision_arithmetic


def case_uses_arithmetic_float_lineage(case: Case, groupby_output_keys: list[str]) -> bool:
    lineage: dict[str, set[str]] = {}
    if case.tables:
        for column in case.tables[0].columns:
            lineage[column.name] = {column.name}
        for table in case.tables[1:]:
            for column in table.columns:
                lineage.setdefault(column.name, {column.name})
    arithmetic_float_columns: set[str] = set()
    for operation in case.program.operations:
        kind = op_kind(operation)
        if kind == "select":
            selected = set(op_columns(operation))
            lineage = {name: deps for name, deps in lineage.items() if name in selected}
        elif kind == "mutate":
            column = op_column(operation)
            source = expr_source(operation)
            deps = set(lineage.get(source, {source}))
            if column:
                lineage[column] = deps | {column}
            if (
                expr_kind(operation) == "arith_const"
                and expr_operator(operation) == "div"
                and column
            ) or (expr_kind(operation) == "cast" and expr_target_type(operation) == "float" and column):
                arithmetic_float_columns.add(column)
        elif kind == "groupby":
            break
    for key in groupby_output_keys:
        deps = lineage.get(key, {key})
        if deps & arithmetic_float_columns:
            return True
        if key in arithmetic_float_columns:
            return True
    return False


def _mutate_float_precision_result_type(
    expression: dict[str, Any],
    column_types: dict[str, str],
) -> str | None:
    source_name = str(expression.get("source", ""))
    source_type = column_types.get(source_name)
    if source_type is None:
        return None
    expression_kind = str(expression.get("kind", ""))
    if expression_kind == "add_const":
        return source_type if source_type in {"int", "float"} else None
    if expression_kind == "arith_const":
        expression_operator = str(expression.get("op", ""))
        if source_type not in {"int", "float"} or expression_operator not in {"sub", "mul", "div", "mod"}:
            return None
        return "float" if expression_operator == "div" or source_type == "float" else source_type
    if expression_kind == "cast" and str(expression.get("to", "")) == "float":
        return "float" if source_type in {"int", "float"} else None
    return None


def _float_precision_aggregate_outputs(
    operation: Any,
    column_types: dict[str, str],
    precision_columns: set[str],
) -> set[str]:
    outputs: set[str] = set()
    for aggregate in aggregate_specs(operation):
        source_name = aggregate_column(aggregate)
        aggregate_function = aggregate_func(aggregate)
        output_column = aggregate_alias(aggregate)
        if not output_column:
            continue
        source_type = column_types.get(source_name)
        precision_sensitive_sum = aggregate_function == "sum" and (
            source_name in precision_columns or source_type == "float"
        )
        precision_sensitive_mean = (
            aggregate_function == "mean"
            and source_type in {"int", "float"}
        )
        if precision_sensitive_sum or precision_sensitive_mean:
            outputs.add(output_column)
    return outputs


def _join_keys(operation: Any) -> tuple[list[str], list[str]]:
    return join_key_pairs(operation)
