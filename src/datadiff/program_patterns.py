from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from datadiff.case_features import has_post_topk_filter
from datadiff.filtering import parse_filter_comparator
from datadiff.operation_semantics import (
    OperationLike,
    condition_cmp,
    condition_value,
    expr_kind,
    expr_numerator,
    expr_operator,
    groupby_agg_aliases,
    groupby_keys,
    has_fractional_float_literal,
    is_left_join,
    join_how,
    op_column,
    op_input_dtype,
    op_nulls,
    op_kind,
    operation_names,
    sort_has_nulls_first,
    sort_key_columns,
    sort_null_placement_mismatch,
)


def program_pattern_features(
    operations: Sequence[OperationLike],
    frontier_buckets: Iterable[str],
    *,
    op_names: Sequence[str] | None = None,
    has_range_filter: bool = False,
    has_running_sum_precision: bool = False,
    has_sortedness_check: bool = False,
) -> set[str]:
    names = list(op_names) if op_names is not None else operation_names(operations)
    patterns: set[str] = set()
    if has_null_groupby_topk_pattern(names, frontier_buckets):
        patterns.add("pattern:null_groupby_topk")
    if has_null_agg_topk_pattern(operations, frontier_buckets):
        patterns.add("pattern:null_agg_topk")
    if has_filter_null_agg_topk_pattern(operations, frontier_buckets):
        patterns.add("pattern:filter_null_agg_topk")
    if has_join_null_agg_topk_pattern(operations, frontier_buckets):
        patterns.add("pattern:join_null_agg_topk")
    if has_join_null_key_topk_pattern(operations, frontier_buckets):
        patterns.add("pattern:join_null_key_topk")
    if has_distinct_null_topk_pattern(operations, frontier_buckets):
        patterns.add("pattern:distinct_null_topk")
    if has_wide_offset_topk_pattern(operations):
        patterns.add("pattern:wide_offset_topk")
    if has_empty_filter_groupby_pattern(operations, frontier_buckets):
        patterns.add("pattern:empty_filter_groupby")
    if has_empty_filter_aggregate_pattern(operations, frontier_buckets):
        patterns.add("pattern:empty_filter_aggregate")
    if has_join_filter_groupby_pattern(operations):
        patterns.add("pattern:join_filter_groupby")
    if has_join_null_truth_filter_pattern(operations):
        patterns.add("pattern:join_null_truth_filter")
    if has_float_group_key_pattern(operations):
        patterns.add("pattern:float_group_key")
    if has_join_null_sort_pattern(operations, frontier_buckets):
        patterns.add("pattern:join_null_sort")
    if has_ordered_groupby_sort_pattern(operations):
        patterns.add("pattern:ordered_groupby_sort")
    if has_topk_resort_pattern(operations):
        patterns.add("pattern:topk_resort")
    if has_join_ordered_agg_topk_pattern(operations):
        patterns.add("pattern:join_ordered_agg_topk")
    if has_global_null_aggregate_pattern(operations, frontier_buckets):
        patterns.add("pattern:global_null_aggregate")
    if has_groupby_filter_cast_membership_pattern(operations):
        patterns.add("pattern:pyarrow_groupby_filter_cast_membership")
    if has_range_filter and has_post_topk_filter(operations):
        patterns.add("pattern:post_topk_range_filter")
    if has_running_sum_precision or has_running_sum_precision_pattern(operations):
        patterns.add("pattern:running_sum_precision")
    if has_sortedness_check and has_sortedness_null_placement_pattern(operations):
        patterns.add("pattern:sortedness_null_placement")
    if has_reverse_division_columns_pattern(operations):
        patterns.add("pattern:polars_reverse_division_columns")
    return patterns


def has_null_groupby_topk_pattern(op_names: Sequence[str], frontier_buckets: Iterable[str]) -> bool:
    frontier = set(frontier_buckets)
    if "groupby:null-key" not in frontier or "sort:null-order" not in frontier:
        return False
    try:
        groupby_idx = list(op_names).index("groupby")
        sort_idx = next(idx for idx, name in enumerate(op_names[groupby_idx + 1 :], start=groupby_idx + 1) if name == "sort")
        next(idx for idx, name in enumerate(op_names[sort_idx + 1 :], start=sort_idx + 1) if name == "limit")
    except (StopIteration, ValueError):
        return False
    return True


def has_null_agg_topk_pattern(operations: Sequence[OperationLike], frontier_buckets: Iterable[str]) -> bool:
    frontier = set(frontier_buckets)
    if "groupby:null-agg-output" not in frontier or "sort:null-order" not in frontier:
        return False
    for groupby_idx, op in enumerate(operations):
        if op_kind(op) != "groupby":
            continue
        agg_aliases = groupby_agg_aliases(op)
        if not agg_aliases:
            continue
        for sort_idx in range(groupby_idx + 1, len(operations)):
            sort_op = operations[sort_idx]
            if op_kind(sort_op) != "sort":
                continue
            if not (agg_aliases & sort_key_columns(sort_op)):
                continue
            if any(op_kind(later) == "limit" for later in operations[sort_idx + 1 :]):
                return True
    return False


def has_distinct_null_topk_pattern(operations: Sequence[OperationLike], frontier_buckets: Iterable[str]) -> bool:
    if "sort:null-order" not in set(frontier_buckets):
        return False
    distinct_idx = next((idx for idx, op in enumerate(operations) if op_kind(op) == "distinct"), -1)
    if distinct_idx < 0:
        return False
    for sort_idx in range(distinct_idx + 1, len(operations)):
        sort_op = operations[sort_idx]
        if op_kind(sort_op) != "sort":
            continue
        if sort_has_nulls_first(sort_op) and any(
            op_kind(later) in {"limit", "offset"} for later in operations[sort_idx + 1 :]
        ):
            return True
    return False


def has_filter_null_agg_topk_pattern(operations: Sequence[OperationLike], frontier_buckets: Iterable[str]) -> bool:
    frontier = set(frontier_buckets)
    if "groupby:null-agg-output" not in frontier or "sort:null-order" not in frontier:
        return False
    try:
        filter_idx = next(idx for idx, op in enumerate(operations) if op_kind(op) == "filter")
        mutate_idx = next(
            idx for idx, op in enumerate(operations[filter_idx + 1 :], start=filter_idx + 1) if op_kind(op) == "mutate"
        )
        select_idx = next(
            idx for idx, op in enumerate(operations[mutate_idx + 1 :], start=mutate_idx + 1) if op_kind(op) == "select"
        )
        groupby_idx = next(
            idx for idx, op in enumerate(operations[select_idx + 1 :], start=select_idx + 1) if op_kind(op) == "groupby"
        )
        post_group_select_idx = next(
            idx for idx, op in enumerate(operations[groupby_idx + 1 :], start=groupby_idx + 1) if op_kind(op) == "select"
        )
        sort_idx = next(
            idx
            for idx, op in enumerate(operations[post_group_select_idx + 1 :], start=post_group_select_idx + 1)
            if op_kind(op) == "sort"
        )
        next(idx for idx, op in enumerate(operations[sort_idx + 1 :], start=sort_idx + 1) if op_kind(op) == "limit")
    except StopIteration:
        return False
    return True


def has_join_null_agg_topk_pattern(operations: Sequence[OperationLike], frontier_buckets: Iterable[str]) -> bool:
    frontier = set(frontier_buckets)
    if "groupby:null-agg-output" not in frontier or "sort:null-order" not in frontier:
        return False
    join_idx = next((idx for idx, op in enumerate(operations) if is_left_join(op)), None)
    if join_idx is None:
        return False
    try:
        groupby_idx = next(
            idx for idx, op in enumerate(operations[join_idx + 1 :], start=join_idx + 1) if op_kind(op) == "groupby"
        )
        sort_idx = next(
            idx for idx, op in enumerate(operations[groupby_idx + 1 :], start=groupby_idx + 1) if op_kind(op) == "sort"
        )
        next(idx for idx, op in enumerate(operations[sort_idx + 1 :], start=sort_idx + 1) if op_kind(op) == "limit")
    except StopIteration:
        return False
    return True


def has_join_null_key_topk_pattern(operations: Sequence[OperationLike], frontier_buckets: Iterable[str]) -> bool:
    frontier = set(frontier_buckets)
    if "groupby:null-key" not in frontier or "sort:null-order" not in frontier:
        return False
    join_idx = next((idx for idx, op in enumerate(operations) if is_left_join(op)), None)
    if join_idx is None:
        return False
    try:
        groupby_idx = next(
            idx for idx, op in enumerate(operations[join_idx + 1 :], start=join_idx + 1) if op_kind(op) == "groupby"
        )
    except StopIteration:
        return False
    group_keys = set(groupby_keys(operations[groupby_idx]))
    if not group_keys:
        return False
    for sort_idx, sort_op in enumerate(operations[groupby_idx + 1 :], start=groupby_idx + 1):
        if op_kind(sort_op) != "sort":
            continue
        if not (sort_key_columns(sort_op) & group_keys):
            continue
        if any(op_kind(later) == "limit" for later in operations[sort_idx + 1 :]):
            return True
    return False


def has_wide_offset_topk_pattern(operations: Sequence[OperationLike]) -> bool:
    try:
        sort_idx = next(idx for idx, op in enumerate(operations) if op_kind(op) == "sort")
        offset_idx = next(
            idx for idx, op in enumerate(operations[sort_idx + 1 :], start=sort_idx + 1) if op_kind(op) == "offset"
        )
        next(idx for idx, op in enumerate(operations[offset_idx + 1 :], start=offset_idx + 1) if op_kind(op) == "limit")
    except StopIteration:
        return False
    return True


def has_empty_filter_groupby_pattern(operations: Sequence[OperationLike], frontier_buckets: Iterable[str]) -> bool:
    if "filter:empty-output" not in set(frontier_buckets):
        return False
    try:
        filter_idx = next(idx for idx, op in enumerate(operations) if op_kind(op) == "filter")
        groupby_idx = next(
            idx for idx, op in enumerate(operations[filter_idx + 1 :], start=filter_idx + 1) if op_kind(op) == "groupby"
        )
    except StopIteration:
        return False
    return any(op_kind(op) in {"select", "sort", "limit"} for op in operations[groupby_idx + 1 :])


def has_empty_filter_aggregate_pattern(operations: Sequence[OperationLike], frontier_buckets: Iterable[str]) -> bool:
    if "filter:empty-output" not in set(frontier_buckets):
        return False
    try:
        filter_idx = next(idx for idx, op in enumerate(operations) if op_kind(op) == "filter")
        next(
            idx for idx, op in enumerate(operations[filter_idx + 1 :], start=filter_idx + 1) if op_kind(op) == "aggregate"
        )
    except StopIteration:
        return False
    return True


def has_join_filter_groupby_pattern(operations: Sequence[OperationLike]) -> bool:
    try:
        join_idx = next(
            idx
            for idx, op in enumerate(operations)
            if op_kind(op) == "join" and join_how(op) == "inner"
        )
        filter_idx = next(
            idx for idx, op in enumerate(operations[join_idx + 1 :], start=join_idx + 1) if op_kind(op) == "filter"
        )
        next(
            idx for idx, op in enumerate(operations[filter_idx + 1 :], start=filter_idx + 1) if op_kind(op) == "groupby"
        )
        sort_idx = next(
            idx for idx, op in enumerate(operations[filter_idx + 1 :], start=filter_idx + 1) if op_kind(op) == "sort"
        )
        next(idx for idx, op in enumerate(operations[sort_idx + 1 :], start=sort_idx + 1) if op_kind(op) == "limit")
    except StopIteration:
        return False
    return True


def has_join_null_truth_filter_pattern(operations: Sequence[OperationLike]) -> bool:
    try:
        join_idx = next(idx for idx, op in enumerate(operations) if is_left_join(op))
        filter_op = next(op for op in operations[join_idx + 1 :] if op_kind(op) == "filter")
    except StopIteration:
        return False
    parsed = parse_filter_comparator(condition_cmp(filter_op))
    return parsed is not None and parsed.truth_test in {"is_not_true", "is_not_false", "is_unknown"}


def has_groupby_filter_cast_membership_pattern(operations: Sequence[OperationLike]) -> bool:
    agg_aliases: set[str] = set()
    for op in operations:
        kind = op_kind(op)
        if kind == "groupby":
            agg_aliases = groupby_agg_aliases(op)
            continue
        if kind != "filter" or not agg_aliases:
            continue
        parsed = parse_filter_comparator(condition_cmp(op))
        if parsed is None or parsed.base not in {"in_set", "not_in_set"}:
            continue
        if op_column(op) in agg_aliases and has_fractional_float_literal(condition_value(op)):
            return True
    return False


def has_reverse_division_columns_pattern(operations: Sequence[OperationLike]) -> bool:
    return any(
        op_kind(op) == "mutate"
        and expr_kind(op) == "reverse_division_columns"
        and bool(expr_numerator(op))
        for op in operations
    )


def has_float_group_key_pattern(operations: Sequence[OperationLike]) -> bool:
    div_columns: set[str] = set()
    for op in operations:
        kind = op_kind(op)
        if kind == "mutate":
            if expr_kind(op) == "arith_const" and expr_operator(op) == "div":
                div_columns.add(op_column(op))
        elif kind == "groupby":
            keys = set(groupby_keys(op))
            return bool(keys & div_columns)
    return False


def has_ordered_groupby_sort_pattern(operations: Sequence[OperationLike]) -> bool:
    try:
        first_sort_idx = next(idx for idx, op in enumerate(operations) if op_kind(op) == "sort")
        groupby_idx = next(
            idx for idx, op in enumerate(operations[first_sort_idx + 1 :], start=first_sort_idx + 1) if op_kind(op) == "groupby"
        )
        agg_aliases = groupby_agg_aliases(operations[groupby_idx])
        second_sort = next(
            op
            for op in operations[groupby_idx + 1 :]
            if op_kind(op) == "sort" and bool(agg_aliases & sort_key_columns(op))
        )
    except StopIteration:
        return False
    return bool(second_sort)


def has_topk_resort_pattern(operations: Sequence[OperationLike]) -> bool:
    try:
        first_sort_idx = next(idx for idx, op in enumerate(operations) if op_kind(op) == "sort")
        limit_idx = next(
            idx for idx, op in enumerate(operations[first_sort_idx + 1 :], start=first_sort_idx + 1) if op_kind(op) == "limit"
        )
        next_op = operations[limit_idx + 1]
    except (IndexError, StopIteration):
        return False
    return op_kind(next_op) in {"sort", "offset"}


def has_running_sum_precision_pattern(operations: Sequence[OperationLike]) -> bool:
    return any(op_kind(op) == "running_sum" and op_input_dtype(op) == "float32" for op in operations)


def has_sortedness_null_placement_pattern(operations: Sequence[OperationLike]) -> bool:
    for idx, op in enumerate(operations):
        if op_kind(op) != "sortedness_check":
            continue
        column = op_column(op)
        check_nulls = op_nulls(op)
        for previous in reversed(operations[:idx]):
            if op_kind(previous) != "sort":
                continue
            if sort_null_placement_mismatch(previous, column, check_nulls):
                return True
    return False


def has_join_ordered_agg_topk_pattern(operations: Sequence[OperationLike]) -> bool:
    try:
        join_idx = next(idx for idx, op in enumerate(operations) if op_kind(op) == "join")
        first_sort_idx = next(
            idx for idx, op in enumerate(operations[join_idx + 1 :], start=join_idx + 1) if op_kind(op) == "sort"
        )
        groupby_idx = next(
            idx for idx, op in enumerate(operations[first_sort_idx + 1 :], start=first_sort_idx + 1) if op_kind(op) == "groupby"
        )
        agg_aliases = groupby_agg_aliases(operations[groupby_idx])
        sort_idx = next(
            idx
            for idx, op in enumerate(operations[groupby_idx + 1 :], start=groupby_idx + 1)
            if op_kind(op) == "sort" and bool(agg_aliases & sort_key_columns(op))
        )
        next(idx for idx, op in enumerate(operations[sort_idx + 1 :], start=sort_idx + 1) if op_kind(op) == "limit")
    except StopIteration:
        return False
    return True


def has_global_null_aggregate_pattern(operations: Sequence[OperationLike], frontier_buckets: Iterable[str]) -> bool:
    frontier = set(frontier_buckets)
    if "aggregate:global" not in frontier:
        return False
    if not {"aggregate:null-output", "aggregate:empty-input"} & frontier:
        return False
    return any(op_kind(op) == "aggregate" for op in operations)


def has_join_null_sort_pattern(operations: Sequence[OperationLike], frontier_buckets: Iterable[str]) -> bool:
    if "sort:null-order" not in set(frontier_buckets):
        return False
    join_idx = next((idx for idx, op in enumerate(operations) if is_left_join(op)), None)
    if join_idx is None:
        return False
    try:
        sort_idx = next(
            idx for idx, op in enumerate(operations[join_idx + 1 :], start=join_idx + 1) if op_kind(op) == "sort"
        )
        next(idx for idx, op in enumerate(operations[sort_idx + 1 :], start=sort_idx + 1) if op_kind(op) == "limit")
    except StopIteration:
        return False
    return True
