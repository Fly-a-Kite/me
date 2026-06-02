from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
    expr_index,
    expr_kind,
    expr_length,
    expr_lower,
    expr_needle,
    expr_new,
    expr_numerator,
    expr_old,
    expr_operator,
    expr_other,
    expr_part,
    expr_separator,
    expr_source,
    expr_start,
    expr_target_type,
    expr_upper,
    expr_value,
    groupby_keys,
    op_comparator,
    op_column,
    op_columns,
    op_input_dtype,
    op_output_alias,
    op_partition_columns,
    op_right_columns,
    op_source,
    op_table,
    op_value,
)


def aggregate_triplets(op: Any) -> list[tuple[str, str, str]]:
    return [
        (aggregate_column(agg), aggregate_func(agg), aggregate_alias(agg))
        for agg in aggregate_specs(op)
    ]


def coalesce_plan(op: Any) -> tuple[str, list[str], bool, Any]:
    return op_output_alias(op), coalesce_sources(op), coalesce_has_fallback(op), coalesce_fallback(op)


def case_when_plan(op: Any) -> tuple[str, str, str, Any, Any, Any]:
    return (
        op_output_alias(op),
        condition_column(op),
        condition_cmp(op),
        condition_value(op),
        case_then_value(op),
        case_else_value(op),
    )


def filter_plan(op: Any) -> tuple[str, str, Any]:
    return op_column(op), op_comparator(op), op_value(op)


def tuple_absence_plan(op: Any) -> tuple[str, list[str], list[str]]:
    return op_table(op), op_columns(op), op_right_columns(op)


@dataclass(frozen=True, slots=True)
class MutatePlan:
    column: str
    kind: str
    source: str
    operator: str
    value: Any
    numerator: str
    lower: Any
    upper: Any
    target_type: str
    old: Any
    new: Any
    start: Any
    length: Any
    separator: str
    index: Any
    other: str
    needle: Any
    part: str


@dataclass(frozen=True, slots=True)
class RunningSumPlan:
    column: str
    source: str
    partition_columns: list[str]
    input_dtype: str


def mutate_plan(op: Any) -> MutatePlan:
    return MutatePlan(
        column=op_column(op),
        kind=expr_kind(op),
        source=expr_source(op),
        operator=expr_operator(op),
        value=expr_value(op),
        numerator=expr_numerator(op),
        lower=expr_lower(op),
        upper=expr_upper(op),
        target_type=expr_target_type(op),
        old=expr_old(op),
        new=expr_new(op),
        start=expr_start(op),
        length=expr_length(op),
        separator=expr_separator(op),
        index=expr_index(op),
        other=expr_other(op),
        needle=expr_needle(op),
        part=expr_part(op),
    )


def running_sum_plan(op: Any) -> RunningSumPlan:
    return RunningSumPlan(
        column=op_column(op),
        source=op_source(op),
        partition_columns=op_partition_columns(op),
        input_dtype=op_input_dtype(op),
    )


def select_columns(op: Any) -> list[str]:
    return op_columns(op)


def distinct_columns(op: Any) -> list[str]:
    return op_columns(op)


def groupby_plan(op: Any) -> tuple[list[str], list[tuple[str, str, str]]]:
    return groupby_keys(op), aggregate_triplets(op)
