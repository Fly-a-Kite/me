from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from datadiff.expression_semantics import eval_expr_on_row
from datadiff.filtering import evaluate_filter_predicate
from datadiff.join_keys import join_key_pairs, join_key_value
from datadiff.operation_semantics import (
    aggregate_alias,
    aggregate_column,
    aggregate_func,
    aggregate_specs,
    condition_cmp,
    expr_kind,
    expr_numerator,
    expr_other,
    expr_source,
    groupby_keys,
    join_how,
    op_columns,
    op_value,
)


def rows_from_samples(samples: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    row_count = min((len(values) for values in samples.values()), default=0)
    return [
        {column: values[idx] for column, values in samples.items()}
        for idx in range(row_count)
    ]


def samples_from_rows(rows: list[dict[str, Any]], columns: list[str]) -> dict[str, list[Any]]:
    return {
        column: [row.get(column) for row in rows]
        for column in columns
    }


def distinct_output_samples(samples: dict[str, list[Any]], op: Any) -> dict[str, list[Any]]:
    columns = [column for column in op_columns(op) if column in samples]
    if not columns:
        return samples
    seen: set[tuple[Any, ...]] = set()
    rows: list[dict[str, Any]] = []
    for row in rows_from_samples(samples):
        key = tuple(row.get(column) for column in columns)
        if key in seen:
            continue
        seen.add(key)
        rows.append({column: row.get(column) for column in columns})
    return samples_from_rows(rows, columns)


def join_samples(
    left_samples: dict[str, list[Any]],
    right_samples: dict[str, list[Any]],
    op: Any,
) -> dict[str, list[Any]]:
    left_keys, right_keys = join_key_pairs(op)
    if (
        not right_samples
        or not left_keys
        or len(left_keys) != len(right_keys)
        or any(left_key not in left_samples for left_key in left_keys)
        or any(right_key not in right_samples for right_key in right_keys)
    ):
        return left_samples

    right_key_set = set(right_keys)
    right_extra_columns = [
        column
        for column in right_samples
        if column not in right_key_set and column not in left_samples
    ]
    output_columns = list(left_samples) + right_extra_columns
    right_index: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows_from_samples(right_samples):
        key = join_key_value(row, right_keys)
        if key is not None:
            right_index.setdefault(key, []).append(row)

    joined_rows: list[dict[str, Any]] = []
    how = join_how(op, "inner")
    for left in rows_from_samples(left_samples):
        key = join_key_value(left, left_keys)
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
    return samples_from_rows(joined_rows, output_columns)


def filter_samples(samples: dict[str, list[Any]], op: Any, *, comparator: str | None = None) -> dict[str, list[Any]]:
    from datadiff.operation_semantics import op_column

    column = op_column(op)
    values = samples.get(column)
    if values is None:
        return samples
    cmp = comparator if comparator is not None else condition_cmp(op)
    mask = [_compare_value(value, cmp, op_value(op)) for value in values]
    return {
        name: [value for value, keep in zip(column_values, mask) if keep]
        for name, column_values in samples.items()
    }


def eval_expr_samples(samples: dict[str, list[Any]], expr: Mapping[str, Any]) -> list[Any] | None:
    source = expr_source({"expr": expr})
    values = samples.get(source)
    if values is None:
        return None
    dependent_columns = {source}
    if expr_kind({"expr": expr}) == "string_concat":
        dependent_columns.add(expr_other({"expr": expr}))
    if expr_kind({"expr": expr}) == "reverse_division_columns":
        dependent_columns.add(expr_numerator({"expr": expr}))
    if any(column not in samples for column in dependent_columns):
        return None
    out: list[Any] = []
    row_count = len(values)
    for idx in range(row_count):
        row = {
            column: samples[column][idx] if idx < len(samples[column]) else None
            for column in dependent_columns
        }
        try:
            out.append(eval_expr_on_row(expr, row))
        except Exception:
            out.append(None)
    return out


def groupby_output_samples(samples: dict[str, list[Any]], op: Any) -> dict[str, list[Any]]:
    keys = [key for key in groupby_keys(op) if key in samples]
    row_count = min((len(samples[key]) for key in keys), default=0)
    groups: dict[tuple[Any, ...], list[int]] = {}
    for idx in range(row_count):
        key_tuple = tuple(samples[key][idx] for key in keys)
        groups.setdefault(key_tuple, []).append(idx)
    out: dict[str, list[Any]] = {key: [] for key in keys}
    for key_tuple in groups:
        for idx, key in enumerate(keys):
            out[key].append(key_tuple[idx])
    for agg in aggregate_specs(op):
        alias = aggregate_alias(agg)
        source_values = samples.get(aggregate_column(agg), [])
        if not alias:
            continue
        values = []
        for indices in groups.values():
            group_values = [source_values[idx] for idx in indices if idx < len(source_values)]
            non_null = [value for value in group_values if value is not None]
            func = aggregate_func(agg)
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


def _compare_value(left: Any, comparator: str, right: Any) -> bool:
    try:
        return evaluate_filter_predicate(left, comparator, right)
    except Exception:
        return False
