from __future__ import annotations

from typing import Any

from datadiff.canonicalization import canonicalize_rows, dedupe_by_canonical_key
from datadiff.dsl import Case, normalize_sort_keys
from datadiff.expression_semantics import eval_expr_on_row
from datadiff.filtering import evaluate_filter_predicate
from datadiff.join_keys import join_key_pairs, join_key_value
from datadiff.normalizer import NormalizedResult, _norm_value
from datadiff.ordering_semantics import sort_row_mappings
from datadiff.operation_semantics import (
    aggregate_alias,
    aggregate_column,
    aggregate_func,
    aggregate_specs,
    case_else_value,
    case_then_value,
    coalesce_fallback,
    coalesce_has_fallback,
    coalesce_sources,
    condition_cmp,
    condition_column,
    condition_value,
    expr_payload,
    expr_kind,
    expr_other,
    groupby_keys,
    join_how,
    op_ascending,
    op_comparator,
    op_column,
    op_columns,
    op_n,
    op_nulls,
    op_output_alias,
    op_right_columns,
    op_source,
    op_table,
    op_value,
    op_kind,
)
from datadiff.running import (
    running_sum_partition_columns,
    running_sum_sort_keys,
    sort_rows_for_running,
    stable_running_sum_values,
)
from datadiff.sortedness import is_sorted_values
from datadiff.tuple_logic import evaluate_tuple_absence
from datadiff.windowing import row_number_filter_rows


def reference_result(case: Case, *, backend_name: str = "dsl_reference") -> NormalizedResult | None:
    try:
        if not case.tables:
            return None
        columns = [column.name for column in case.tables[0].columns]
        rows = [{column: row.get(column) for column in columns} for row in case.tables[0].rows]
        tables = {table.name: table for table in case.tables}
        for op in case.program.operations:
            kind = op_kind(op)
            if kind == "join":
                rows, columns = _reference_join(rows, columns, tables, op)
            elif kind == "union_all":
                right = tables[op_table(op)]
                rows = [*rows, *[{column: right_row.get(column) for column in columns} for right_row in right.rows]]
            elif kind in {"semi_join", "anti_join"}:
                rows = _reference_semi_anti_join(rows, tables, op, kind)
            elif kind == "drop_nulls":
                kept_columns = op_columns(op)
                rows = [
                    row
                    for row in rows
                    if all(evaluate_filter_predicate(row.get(column), "is_not_null", None) for column in kept_columns)
                ]
            elif kind == "filter":
                rows = [
                    row
                    for row in rows
                    if evaluate_filter_predicate(
                        row.get(op_column(op)),
                        condition_cmp(op) or op_comparator(op),
                        condition_value(op) if "condition" in op else op_value(op),
                    )
                ]
            elif kind == "tuple_absence_filter":
                right = tables[op_table(op)]
                left_columns = op_columns(op)
                right_columns = op_right_columns(op)
                rows = [row for row in rows if evaluate_tuple_absence(row, left_columns, right.rows, right_columns)]
            elif kind == "row_number_filter":
                rows = row_number_filter_rows(rows, op)
            elif kind == "running_sum":
                out_column = op_column(op)
                rows = sort_rows_for_running(rows, running_sum_sort_keys(op))
                values = stable_running_sum_values(rows, op_source(op), running_sum_partition_columns(op))
                rows = [{**row, out_column: value} for row, value in zip(rows, values)]
                columns = [name for name in columns if name != out_column] + [out_column]
            elif kind == "sortedness_check":
                alias = op_output_alias(op)
                ok = is_sorted_values(
                    [row.get(op_column(op)) for row in rows],
                    ascending=op_ascending(op),
                    nulls=op_nulls(op),
                )
                columns = [alias]
                rows = [{alias: ok}]
            elif kind.endswith("_probe"):
                alias = op_output_alias(op)
                columns = [alias]
                rows = [{alias: False}]
            elif kind == "select":
                columns = op_columns(op)
                rows = [{column: row.get(column) for column in columns} for row in rows]
            elif kind == "distinct":
                columns = op_columns(op)
                projected_rows = [{column: row.get(column) for column in columns} for row in rows]
                rows = dedupe_by_canonical_key(projected_rows)
            elif kind == "sort":
                sort_keys = normalize_sort_keys(op)
                rows = list(sort_row_mappings(rows, sort_keys))
            elif kind == "limit":
                rows = rows[: op_n(op)]
            elif kind == "offset":
                rows = rows[op_n(op) :]
            elif kind == "mutate":
                out_column = op_column(op)
                rows = [{**row, out_column: eval_expr_on_row(expr_payload(op), row)} for row in rows]
                if out_column not in columns:
                    columns.append(out_column)
            elif kind == "fill_null":
                column = op_column(op)
                value = op_value(op)
                rows = [{**row, column: value if row.get(column) is None else row.get(column)} for row in rows]
            elif kind == "coalesce":
                alias = op_output_alias(op)
                source_columns = coalesce_sources(op)
                has_fallback = coalesce_has_fallback(op)
                fallback = coalesce_fallback(op)
                out_rows = []
                for row in rows:
                    value = None
                    for column in source_columns:
                        candidate = row.get(column)
                        if candidate is not None:
                            value = candidate
                            break
                    if value is None and has_fallback:
                        value = fallback
                    out_rows.append({**row, alias: value})
                rows = out_rows
                columns = [column for column in columns if column != alias] + [alias]
            elif kind == "case_when":
                alias = op_output_alias(op)
                column = condition_column(op)
                comparator = condition_cmp(op)
                value = condition_value(op)
                then_value = case_then_value(op)
                else_value = case_else_value(op)
                rows = [
                    {
                        **row,
                        alias: then_value if evaluate_filter_predicate(row.get(column), comparator, value) else else_value,
                    }
                    for row in rows
                ]
                if alias not in columns:
                    columns.append(alias)
            elif kind == "groupby":
                rows, columns = _reference_groupby(rows, op)
            elif kind == "aggregate":
                rows, columns = _reference_aggregate(rows, op)
            else:
                return None
        return normalize_reference_rows(
            columns,
            rows,
            preserve_order=case.program.output_order_sensitive,
            preserve_float_precision=case.program.order_sensitive,
            backend_name=backend_name,
        )
    except Exception:
        return None


def normalize_reference_rows(
    columns: list[str],
    rows: list[dict[str, Any]],
    *,
    preserve_order: bool = False,
    preserve_float_precision: bool | None = None,
    backend_name: str = "dsl_reference",
) -> NormalizedResult:
    if preserve_float_precision is None:
        preserve_float_precision = preserve_order
    column_positions = sorted(enumerate(columns), key=lambda item: (item[1], item[0]))
    out_columns = [name for _, name in column_positions]
    out_rows = [
        [
            _norm_value(
                row.get(columns[idx]),
                preserve_float_precision=preserve_float_precision,
            )
            for idx, _ in column_positions
        ]
        for row in rows
    ]
    normalized = NormalizedResult(backend_name, "ok", out_columns, out_rows)
    if not preserve_order:
        normalized.adopt_canonicalized_rows(canonicalize_rows(out_rows))
    return normalized


def reference_aggregate_value(rows: list[dict[str, Any]], column: str, func: str) -> Any:
    values = [row.get(column) for row in rows if row.get(column) is not None]
    if func == "count":
        return len(values)
    if func == "nunique":
        return len(set(values))
    if not values:
        return None
    if func == "any":
        return any(bool(value) for value in values)
    if func == "all":
        return all(bool(value) for value in values)
    if func == "sum":
        return sum(values)
    if func == "mean":
        return sum(values) / len(values)
    if func == "min":
        return min(values)
    if func == "max":
        return max(values)
    raise ValueError(func)


def _reference_join(
    rows: list[dict[str, Any]],
    columns: list[str],
    tables: dict[str, Any],
    op: Any,
) -> tuple[list[dict[str, Any]], list[str]]:
    right = tables[op_table(op)]
    left_keys, right_keys = join_key_pairs(op)
    right_key_set = set(right_keys)
    right_columns = [column.name for column in right.columns if column.name not in right_key_set and column.name not in columns]
    index: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for right_row in right.rows:
        key = join_key_value(right_row, right_keys)
        if key is None:
            continue
        index.setdefault(key, []).append(right_row)
    joined: list[dict[str, Any]] = []
    for left_row in rows:
        key = join_key_value(left_row, left_keys)
        matches = [] if key is None else index.get(key, [])
        if matches:
            for right_row in matches:
                out = dict(left_row)
                for column in right_columns:
                    out[column] = right_row.get(column)
                joined.append(out)
        elif join_how(op) == "left":
            out = dict(left_row)
            for column in right_columns:
                out[column] = None
            joined.append(out)
    return joined, [*columns, *right_columns]


def _reference_semi_anti_join(rows: list[dict[str, Any]], tables: dict[str, Any], op: Any, kind: str) -> list[dict[str, Any]]:
    right = tables[op_table(op)]
    left_keys, right_keys = join_key_pairs(op)
    right_key_values = {
        join_key_value(right_row, right_keys)
        for right_row in right.rows
        if join_key_value(right_row, right_keys) is not None
    }
    filtered = []
    for row in rows:
        value = join_key_value(row, left_keys)
        matched = value is not None and value in right_key_values
        if (kind == "semi_join" and matched) or (kind == "anti_join" and not matched):
            filtered.append(row)
    return filtered
def _reference_groupby(rows: list[dict[str, Any]], op: Any) -> tuple[list[dict[str, Any]], list[str]]:
    keys = groupby_keys(op)
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    order: list[tuple[Any, ...]] = []
    for row in rows:
        key = tuple(row.get(column) for column in keys)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(row)
    out_rows = []
    aliases = [aggregate_alias(agg) for agg in aggregate_specs(op)]
    for key in order:
        group_rows = grouped[key]
        out = {column: value for column, value in zip(keys, key)}
        for agg in aggregate_specs(op):
            out[aggregate_alias(agg)] = reference_aggregate_value(group_rows, aggregate_column(agg), aggregate_func(agg))
        out_rows.append(out)
    return out_rows, keys + aliases


def _reference_aggregate(rows: list[dict[str, Any]], op: Any) -> tuple[list[dict[str, Any]], list[str]]:
    out = {}
    aliases = [aggregate_alias(agg) for agg in aggregate_specs(op)]
    for agg in aggregate_specs(op):
        out[aggregate_alias(agg)] = reference_aggregate_value(rows, aggregate_column(agg), aggregate_func(agg))
    return [out], aliases
