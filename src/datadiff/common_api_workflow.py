from __future__ import annotations

import random
from typing import Any

from .dsl import Case, ColumnSpec, Program, TableData


def _string_pattern_literal_for_type(rnd: random.Random, typ: str) -> str:
    if typ != "str":
        raise ValueError(typ)
    return rnd.choice(["a", "A", "beta", "中文", "space", "value", "missing"])


def _string_pattern_literal_for_column(rnd: random.Random, table: TableData, col: str, comparator: str) -> str:
    values = [row.get(col) for row in table.rows if isinstance(row.get(col), str) and row.get(col) != ""]
    if values and rnd.random() < 0.75:
        value = str(rnd.choice(values))
        if comparator == "str_starts_with":
            return value[: rnd.randint(1, min(len(value), 4))]
        if comparator == "str_ends_with":
            width = rnd.randint(1, min(len(value), 4))
            return value[-width:]
        if " " in value and rnd.random() < 0.5:
            return rnd.choice([" ", value.split(" ", 1)[0]])
        if len(value) == 1:
            return value
        start = rnd.randint(0, len(value) - 1)
        end = rnd.randint(start + 1, min(len(value), start + 4))
        return value[start:end]
    return _string_pattern_literal_for_type(rnd, "str")


def _clip_bounds_for_type(rnd: random.Random, typ: str) -> tuple[Any, Any]:
    if typ == "int":
        lower, upper = sorted(rnd.sample([-10, -5, -2, 0, 2, 5, 10], 2))
        return lower, upper
    if typ == "float":
        lower, upper = sorted(rnd.sample([-10.0, -2.5, -1.0, 0.0, 1.0, 2.5, 10.0], 2))
        return lower, upper
    raise ValueError(typ)


COMMON_API_WORKFLOW_TEMPLATES = (
    "filter_mutate_project_topk",
    "membership_groupby_aggregate",
    "input_partition_union_groupby",
    "filter_input_materialization_groupby",
    "negative_membership_topk",
    "negative_membership_groupby",
    "string_derive_groupby",
    "string_upper_groupby",
    "string_upper_topk",
    "string_contains_groupby",
    "string_contains_topk",
    "string_strip_groupby",
    "string_strip_topk",
    "string_replace_groupby",
    "string_replace_topk",
    "string_slice_groupby",
    "string_slice_topk",
    "string_concat_groupby",
    "string_concat_topk",
    "string_contains_flag_groupby",
    "string_contains_flag_topk",
    "string_starts_with_flag_groupby",
    "string_starts_with_flag_topk",
    "string_ends_with_flag_groupby",
    "string_ends_with_flag_topk",
    "bool_not_groupby",
    "bool_not_topk",
    "join_filter_groupby",
    "nullable_bool_groupby",
    "bool_reduction_groupby_topk",
    "bool_reduction_global_summary",
    "ordered_slice_projection",
    "range_cast_groupby",
    "type_cast_boundary_groupby",
    "type_cast_boundary_topk",
    "abs_groupby",
    "abs_topk",
    "clip_groupby",
    "clip_topk",
    "double_filter_topk",
    "distinct_sort_topk",
    "distinct_null_topk",
    "filter_distinct_groupby",
    "fill_null_groupby",
    "fill_null_distinct_topk",
    "fill_null_filter_groupby_topk",
    "coalesce_fill_null_groupby_topk",
    "coalesce_groupby",
    "coalesce_topk",
    "case_when_classify_topk",
    "case_when_groupby",
    "union_all_filter_groupby",
    "union_all_case_when_topk",
    "drop_nulls_groupby",
    "union_all_drop_nulls_topk",
    "semi_join_filter_topk",
    "anti_join_groupby",
    "top_n_per_group",
    "dedup_latest_per_id",
    "running_total_by_group",
    "left_join_fill_groupby",
    "clean_key_join_groupby",
    "filtered_global_aggregate",
    "groupby_having_topk",
    "pagination_filter_select",
    "nunique_groupby",
    "join_nunique_groupby",
    "global_nunique_summary",
    "multi_key_groupby_summary",
    "multi_key_groupby_topk",
    "empty_filter_groupby_common",
    "empty_filter_global_aggregate",
    "multi_key_join_groupby",
    "multi_key_join_topk",
    "date_part_groupby",
    "date_part_topk",
    "string_split_part_groupby",
    "string_split_part_topk",
    "string_null_if_empty_groupby",
    "string_null_if_empty_topk",
    "string_length_groupby",
    "string_length_topk",
    "string_lower_groupby",
    "string_lower_topk",
    "string_basename_groupby",
    "string_basename_topk",
    "normalized_string_join_groupby",
    "normalized_string_join_topk",
    "normalized_string_semi_join_topk",
    "normalized_string_anti_join_groupby",
    "normalized_string_case_when_groupby",
    "normalized_string_case_when_topk",
    "string_pattern_case_when_groupby",
    "string_pattern_case_when_topk",
    "sql_distinct_null_coalesce_topk",
    "sql_left_join_coalesce_membership",
    "sql_case_membership_distinct_topk",
    "sql_union_coalesce_distinct_topk",
    "sql_left_join_case_membership_groupby",
    "sql_left_join_null_predicate_aggregate",
    "sql_coalesce_case_distinct_groupby",
    "sql_numeric_text_cast_membership_groupby",
    "sql_bool_membership_case_aggregate",
    "sql_left_join_bool_case_groupby",
    "sql_left_join_bool_coalesce_case_groupby",
    "sql_bool_antijoin_case_aggregate",
    "sql_left_join_bool_coalesce_filter_groupby",
    "sql_numeric_text_cast_bool_antijoin_groupby",
    "sql_multi_key_semijoin_case_groupby",
    "sql_multi_key_antijoin_case_groupby",
)


def generate_common_api_workflow_case(seed: int) -> Case:
    """Generate short, everyday DataFrame/SQL workflows from organic data.

    The templates intentionally stay close to operations users write in notebooks
    and ETL scripts: filter, select, mutate, fill-null, coalesce, case-when,
    positive/negative set membership filters, string pattern filters, string length/upper/strip/null-if-empty/replace/slice/split/basename/concat,
    normalized string-key joins and semi/anti membership joins,
    DuckDB/SQLite-sensitive DISTINCT NULL top-k, union/materialization, NULL predicates,
    text casts, and join/coalesce/case membership rewrites,
    int/float/string boundary casts,
    ISO date-string part extraction, nullable boolean negation/reductions, physical input partition/union boundaries,
    filter input-materialization boundaries, union-all, semi/anti join,
    fill-null/coalesce interactions with groupby and top-k,
    distinct with nullable top-k, count-distinct/nunique, single/multi-key groupby,
    empty-filter aggregation, single/multi-key join, window-style top-N per group, running totals, sort, limit, offset,
    null predicates, boolean predicates, and set membership.
    """

    rnd = random.Random(seed * 2147483647 + 101)
    template = COMMON_API_WORKFLOW_TEMPLATES[seed % len(COMMON_API_WORKFLOW_TEMPLATES)]
    table = _common_api_base_table(rnd)
    if template in {"date_part_groupby", "date_part_topk"}:
        table = _common_api_with_date_column(table, rnd)
    if template in {
        "type_cast_boundary_groupby",
        "type_cast_boundary_topk",
        "sql_numeric_text_cast_membership_groupby",
        "sql_numeric_text_cast_bool_antijoin_groupby",
    }:
        table = _common_api_with_numeric_string_column(table, rnd)
    if template in {"string_basename_groupby", "string_basename_topk"}:
        table = _common_api_with_path_column(table, rnd)
    tables = [table]
    operations: list[dict[str, Any]]

    if template == "filter_mutate_project_topk":
        operations = [
            {"op": "filter", "column": "x", "cmp": ">=", "value": rnd.choice([-2, 0, 1])},
            {"op": "mutate", "column": "x_plus_one", "expr": {"kind": "add_const", "source": "x", "value": 1}},
            {
                "op": "sort",
                "keys": [
                    {"column": "g", "ascending": True, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                    {"column": "row_nr", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "select", "columns": ["id", "g", "x_plus_one"]},
            {"op": "limit", "n": rnd.randint(2, 6)},
        ]
    elif template == "membership_groupby_aggregate":
        operations = [
            {"op": "filter", "column": "g", "cmp": "in_set", "value": ["a", "b", "space value"]},
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [
                    {"column": "x", "func": "sum", "as": "sum_x"},
                    {"column": "id", "func": "count", "as": "count_id"},
                    {"column": "flag", "func": "any", "as": "any_flag"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "g", "ascending": True, "nulls": "last"},
                    {"column": "count_id", "ascending": False, "nulls": "last"},
                ],
            },
        ]
    elif template == "input_partition_union_groupby":
        operations = [
            {"op": "mutate", "column": "s_norm", "expr": {"kind": "string_null_if_empty", "source": "s"}},
            {"op": "fill_null", "column": "s_norm", "value": "missing"},
            {
                "op": "groupby",
                "keys": ["g", "s_norm"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "count_id"},
                    {"column": "x", "func": "sum", "as": "sum_x"},
                    {"column": "flag", "func": "any", "as": "any_flag"},
                ],
            },
        ]
    elif template == "filter_input_materialization_groupby":
        operations = [
            {"op": "filter", "column": "s", "cmp": "in_set", "value": ["Alpha", "Beta"]},
            {"op": "fill_null", "column": "g", "value": "missing"},
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "count_id"},
                    {"column": "x", "func": "sum", "as": "sum_x"},
                    {"column": "flag", "func": "any", "as": "any_flag"},
                ],
            },
        ]
    elif template == "negative_membership_topk":
        operations = [
            {"op": "filter", "column": "g", "cmp": "not_in_set", "value": ["a", "space value"]},
            {
                "op": "sort",
                "keys": [
                    {"column": "g", "ascending": True, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                    {"column": "row_nr", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "select", "columns": ["id", "g", "s"]},
            {"op": "limit", "n": rnd.randint(2, 6)},
        ]
    elif template == "negative_membership_groupby":
        operations = [
            {"op": "filter", "column": "s", "cmp": "not_in_set", "value": ["Alpha", "Beta"]},
            {"op": "fill_null", "column": "g", "value": "missing"},
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "count_id"},
                    {"column": "x", "func": "sum", "as": "sum_x"},
                    {"column": "s", "func": "nunique", "as": "unique_s"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "g", "ascending": True, "nulls": "last"},
                    {"column": "count_id", "ascending": False, "nulls": "last"},
                ],
            },
        ]
    elif template == "string_derive_groupby":
        operations = [
            {"op": "mutate", "column": "s_lower", "expr": {"kind": "string_lower", "source": "s"}},
            {"op": "mutate", "column": "s_len", "expr": {"kind": "string_length", "source": "s_lower"}},
            {"op": "filter", "column": "s_len", "cmp": ">=", "value": 0},
            {
                "op": "groupby",
                "keys": ["s_lower"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "count_id"},
                    {"column": "s_len", "func": "min", "as": "min_s_len"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "s_lower", "ascending": True, "nulls": "last"},
                    {"column": "count_id", "ascending": False, "nulls": "last"},
                ],
            },
        ]
    elif template == "string_upper_groupby":
        operations = [
            {"op": "mutate", "column": "g_upper", "expr": {"kind": "string_upper", "source": "g"}},
            {"op": "filter", "column": "g_upper", "cmp": "is_not_null", "value": None},
            {
                "op": "groupby",
                "keys": ["g_upper"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "count_id"},
                    {"column": "x", "func": "sum", "as": "sum_x"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "g_upper", "ascending": True, "nulls": "last"},
                    {"column": "count_id", "ascending": False, "nulls": "last"},
                ],
            },
        ]
    elif template == "string_upper_topk":
        operations = [
            {"op": "mutate", "column": "s_upper", "expr": {"kind": "string_upper", "source": "s"}},
            {
                "op": "sort",
                "keys": [
                    {"column": "s_upper", "ascending": True, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                    {"column": "row_nr", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "select", "columns": ["id", "g", "s_upper"]},
            {"op": "limit", "n": rnd.randint(2, 6)},
        ]
    elif template == "string_contains_groupby":
        comparator = rnd.choice(["str_contains", "str_starts_with", "str_ends_with"])
        operations = [
            {"op": "filter", "column": "s", "cmp": comparator, "value": _string_pattern_literal_for_column(rnd, table, "s", comparator)},
            {"op": "fill_null", "column": "g", "value": "missing"},
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "count_id"},
                    {"column": "x", "func": "sum", "as": "sum_x"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "g", "ascending": True, "nulls": "last"},
                    {"column": "count_id", "ascending": False, "nulls": "last"},
                ],
            },
        ]
    elif template == "string_contains_topk":
        comparator = rnd.choice(["str_contains", "str_starts_with", "str_ends_with"])
        operations = [
            {"op": "filter", "column": "g", "cmp": comparator, "value": _string_pattern_literal_for_column(rnd, table, "g", comparator)},
            {
                "op": "sort",
                "keys": [
                    {"column": "g", "ascending": True, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                    {"column": "row_nr", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "select", "columns": ["id", "g", "s"]},
            {"op": "limit", "n": rnd.randint(2, 6)},
        ]
    elif template == "string_strip_groupby":
        operations = [
            {"op": "mutate", "column": "s_clean", "expr": {"kind": "string_strip", "source": "s"}},
            {"op": "filter", "column": "s_clean", "cmp": "is_not_null", "value": None},
            {
                "op": "groupby",
                "keys": ["s_clean"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "count_id"},
                    {"column": "x", "func": "sum", "as": "sum_x"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "s_clean", "ascending": True, "nulls": "last"},
                    {"column": "count_id", "ascending": False, "nulls": "last"},
                ],
            },
        ]
    elif template == "string_strip_topk":
        operations = [
            {"op": "mutate", "column": "g_clean", "expr": {"kind": "string_strip", "source": "g"}},
            {
                "op": "sort",
                "keys": [
                    {"column": "g_clean", "ascending": True, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                    {"column": "row_nr", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "select", "columns": ["id", "g_clean", "s"]},
            {"op": "limit", "n": rnd.randint(2, 6)},
        ]
    elif template == "string_replace_groupby":
        operations = [
            {"op": "mutate", "column": "s_token", "expr": {"kind": "string_replace", "source": "s", "old": " ", "new": "_"}},
            {"op": "filter", "column": "s_token", "cmp": "is_not_null", "value": None},
            {
                "op": "groupby",
                "keys": ["s_token"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "count_id"},
                    {"column": "x", "func": "sum", "as": "sum_x"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "s_token", "ascending": True, "nulls": "last"},
                    {"column": "count_id", "ascending": False, "nulls": "last"},
                ],
            },
        ]
    elif template == "string_replace_topk":
        operations = [
            {"op": "mutate", "column": "g_token", "expr": {"kind": "string_replace", "source": "g", "old": " ", "new": "_"}},
            {
                "op": "sort",
                "keys": [
                    {"column": "g_token", "ascending": True, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                    {"column": "row_nr", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "select", "columns": ["id", "g_token", "s"]},
            {"op": "limit", "n": rnd.randint(2, 6)},
        ]
    elif template == "string_slice_groupby":
        operations = [
            {"op": "mutate", "column": "s_prefix", "expr": {"kind": "string_slice", "source": "s", "start": 0, "length": 3}},
            {"op": "filter", "column": "s_prefix", "cmp": "is_not_null", "value": None},
            {
                "op": "groupby",
                "keys": ["s_prefix"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "count_id"},
                    {"column": "x", "func": "sum", "as": "sum_x"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "s_prefix", "ascending": True, "nulls": "last"},
                    {"column": "count_id", "ascending": False, "nulls": "last"},
                ],
            },
        ]
    elif template == "string_slice_topk":
        operations = [
            {"op": "mutate", "column": "g_prefix", "expr": {"kind": "string_slice", "source": "g", "start": 0, "length": 3}},
            {
                "op": "sort",
                "keys": [
                    {"column": "g_prefix", "ascending": True, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                    {"column": "row_nr", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "select", "columns": ["id", "g_prefix", "s"]},
            {"op": "limit", "n": rnd.randint(2, 6)},
        ]
    elif template == "string_concat_groupby":
        operations = [
            {"op": "mutate", "column": "g_s_key", "expr": {"kind": "string_concat", "source": "g", "other": "s", "sep": "-"}},
            {"op": "filter", "column": "g_s_key", "cmp": "is_not_null", "value": None},
            {
                "op": "groupby",
                "keys": ["g_s_key"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "count_id"},
                    {"column": "x", "func": "sum", "as": "sum_x"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "g_s_key", "ascending": True, "nulls": "last"},
                    {"column": "count_id", "ascending": False, "nulls": "last"},
                ],
            },
        ]
    elif template == "string_concat_topk":
        operations = [
            {"op": "mutate", "column": "g_s_key", "expr": {"kind": "string_concat", "source": "g", "other": "s", "sep": "-"}},
            {
                "op": "sort",
                "keys": [
                    {"column": "g_s_key", "ascending": True, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                    {"column": "row_nr", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "select", "columns": ["id", "g_s_key"]},
            {"op": "limit", "n": rnd.randint(2, 6)},
        ]
    elif template == "string_contains_flag_groupby":
        operations = [
            {"op": "mutate", "column": "has_space", "expr": {"kind": "string_contains", "source": "s", "needle": "space"}},
            {
                "op": "groupby",
                "keys": ["has_space"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "count_id"},
                    {"column": "x", "func": "sum", "as": "sum_x"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "has_space", "ascending": True, "nulls": "last"},
                    {"column": "count_id", "ascending": False, "nulls": "last"},
                ],
            },
        ]
    elif template == "string_contains_flag_topk":
        operations = [
            {"op": "mutate", "column": "has_pad", "expr": {"kind": "string_contains", "source": "g", "needle": "pad"}},
            {
                "op": "sort",
                "keys": [
                    {"column": "has_pad", "ascending": False, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                    {"column": "row_nr", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "select", "columns": ["id", "has_pad", "g"]},
            {"op": "limit", "n": rnd.randint(2, 6)},
        ]
    elif template == "string_starts_with_flag_groupby":
        operations = [
            {
                "op": "mutate",
                "column": "starts_space",
                "expr": {"kind": "string_starts_with", "source": "s", "needle": "space"},
            },
            {
                "op": "groupby",
                "keys": ["starts_space"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "count_id"},
                    {"column": "x", "func": "sum", "as": "sum_x"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "starts_space", "ascending": True, "nulls": "last"},
                    {"column": "count_id", "ascending": False, "nulls": "last"},
                ],
            },
        ]
    elif template == "string_starts_with_flag_topk":
        operations = [
            {
                "op": "mutate",
                "column": "starts_pad",
                "expr": {"kind": "string_starts_with", "source": "g", "needle": "pad"},
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "starts_pad", "ascending": False, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                    {"column": "row_nr", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "select", "columns": ["id", "starts_pad", "g"]},
            {"op": "limit", "n": rnd.randint(2, 6)},
        ]
    elif template == "string_ends_with_flag_groupby":
        operations = [
            {
                "op": "mutate",
                "column": "ends_a",
                "expr": {"kind": "string_ends_with", "source": "s", "needle": "a"},
            },
            {
                "op": "groupby",
                "keys": ["ends_a"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "count_id"},
                    {"column": "x", "func": "sum", "as": "sum_x"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "ends_a", "ascending": True, "nulls": "last"},
                    {"column": "count_id", "ascending": False, "nulls": "last"},
                ],
            },
        ]
    elif template == "string_ends_with_flag_topk":
        operations = [
            {
                "op": "mutate",
                "column": "ends_d",
                "expr": {"kind": "string_ends_with", "source": "g", "needle": "d"},
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "ends_d", "ascending": False, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                    {"column": "row_nr", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "select", "columns": ["id", "ends_d", "g"]},
            {"op": "limit", "n": rnd.randint(2, 6)},
        ]
    elif template == "bool_not_groupby":
        operations = [
            {"op": "mutate", "column": "not_flag", "expr": {"kind": "bool_not", "source": "flag"}},
            {"op": "filter", "column": "not_flag", "cmp": "bool_is_not_unknown", "value": None},
            {
                "op": "groupby",
                "keys": ["not_flag"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "count_id"},
                    {"column": "x", "func": "sum", "as": "sum_x"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "not_flag", "ascending": True, "nulls": "last"},
                    {"column": "count_id", "ascending": False, "nulls": "last"},
                ],
            },
        ]
    elif template == "bool_not_topk":
        operations = [
            {"op": "mutate", "column": "not_flag", "expr": {"kind": "bool_not", "source": "flag"}},
            {
                "op": "sort",
                "keys": [
                    {"column": "not_flag", "ascending": True, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                    {"column": "row_nr", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "select", "columns": ["id", "not_flag", "g"]},
            {"op": "limit", "n": rnd.randint(2, 6)},
        ]
    elif template == "join_filter_groupby":
        tables.append(_common_api_dimension_table(rnd))
        operations = [
            {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": rnd.choice(["inner", "left"])},
            {"op": "filter", "column": "tag", "cmp": "is_not_null", "value": None},
            {"op": "mutate", "column": "j_float", "expr": {"kind": "cast", "source": "j", "to": "float"}},
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [
                    {"column": "j_float", "func": "mean", "as": "mean_j"},
                    {"column": "id", "func": "count", "as": "count_id"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "g", "ascending": True, "nulls": "last"},
                    {"column": "count_id", "ascending": False, "nulls": "last"},
                ],
            },
            {"op": "limit", "n": 5},
        ]
    elif template == "nullable_bool_groupby":
        operations = [
            {"op": "filter", "column": "flag", "cmp": "bool_is_not_false", "value": None},
            {
                "op": "groupby",
                "keys": ["flag"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "count_id"},
                    {"column": "x", "func": "min", "as": "min_x"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "flag", "ascending": False, "nulls": "last"},
                    {"column": "count_id", "ascending": False, "nulls": "last"},
                ],
            },
        ]
    elif template == "bool_reduction_groupby_topk":
        operations = [
            {"op": "fill_null", "column": "g", "value": "missing"},
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [
                    {"column": "flag", "func": "any", "as": "any_flag"},
                    {"column": "flag", "func": "all", "as": "all_flag"},
                    {"column": "flag", "func": "count", "as": "count_flag"},
                    {"column": "x", "func": "sum", "as": "sum_x"},
                ],
            },
            {"op": "filter", "column": "any_flag", "cmp": "bool_is_not_false", "value": None},
            {
                "op": "sort",
                "keys": [
                    {"column": "all_flag", "ascending": False, "nulls": "last"},
                    {"column": "count_flag", "ascending": False, "nulls": "last"},
                    {"column": "g", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "select", "columns": ["g", "any_flag", "all_flag", "count_flag", "sum_x"]},
            {"op": "limit", "n": rnd.randint(2, 5)},
        ]
    elif template == "bool_reduction_global_summary":
        operations = [
            {"op": "filter", "column": "g", "cmp": "is_not_null", "value": None},
            {
                "op": "aggregate",
                "aggs": [
                    {"column": "flag", "func": "any", "as": "any_flag"},
                    {"column": "flag", "func": "all", "as": "all_flag"},
                    {"column": "flag", "func": "count", "as": "count_flag"},
                    {"column": "id", "func": "count", "as": "count_id"},
                ],
            },
            {"op": "filter", "column": "any_flag", "cmp": "bool_is_not_unknown", "value": None},
            {
                "op": "sort",
                "keys": [
                    {"column": "all_flag", "ascending": False, "nulls": "last"},
                    {"column": "count_flag", "ascending": False, "nulls": "last"},
                ],
            },
            {"op": "select", "columns": ["any_flag", "all_flag", "count_flag", "count_id"]},
            {"op": "limit", "n": 1},
        ]
    elif template == "ordered_slice_projection":
        operations = [
            {
                "op": "sort",
                "keys": [
                    {"column": "x", "ascending": rnd.choice([True, False]), "nulls": rnd.choice(["first", "last"])},
                    {"column": "id", "ascending": True, "nulls": "last"},
                    {"column": "row_nr", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "offset", "n": rnd.randint(0, 2)},
            {"op": "limit", "n": rnd.randint(2, 5)},
            {"op": "select", "columns": ["id", "g", "x"]},
        ]
    elif template == "range_cast_groupby":
        operations = [
            {"op": "filter", "column": "x", "cmp": "range_closed", "value": [-10, 10]},
            {"op": "mutate", "column": "x_float", "expr": {"kind": "cast", "source": "x", "to": "float"}},
            {"op": "mutate", "column": "x_scaled", "expr": {"kind": "arith_const", "source": "x_float", "op": "div", "value": 2}},
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [
                    {"column": "x_scaled", "func": "mean", "as": "mean_x_scaled"},
                    {"column": "x_float", "func": "max", "as": "max_x_float"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "g", "ascending": True, "nulls": "last"},
                    {"column": "mean_x_scaled", "ascending": False, "nulls": "last"},
                ],
            },
        ]
    elif template == "type_cast_boundary_groupby":
        operations = [
            {
                "op": "mutate",
                "column": "num_i",
                "expr": {"kind": "cast", "source": "num_s", "to": "int", "input_domain": "integer_string"},
            },
            {"op": "mutate", "column": "num_f", "expr": {"kind": "cast", "source": "num_i", "to": "float"}},
            {"op": "mutate", "column": "num_label", "expr": {"kind": "cast", "source": "num_i", "to": "str"}},
            {"op": "filter", "column": "num_f", "cmp": "range_closed", "value": [-10.0, 10.0]},
            {
                "op": "groupby",
                "keys": ["num_label"],
                "aggs": [
                    {"column": "num_i", "func": "sum", "as": "sum_num_i"},
                    {"column": "num_f", "func": "mean", "as": "mean_num_f"},
                    {"column": "id", "func": "count", "as": "count_id"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "num_label", "ascending": True, "nulls": "last"},
                    {"column": "sum_num_i", "ascending": False, "nulls": "last"},
                    {"column": "count_id", "ascending": False, "nulls": "last"},
                ],
            },
        ]
    elif template == "type_cast_boundary_topk":
        operations = [
            {
                "op": "mutate",
                "column": "num_i",
                "expr": {"kind": "cast", "source": "num_s", "to": "int", "input_domain": "integer_string"},
            },
            {"op": "mutate", "column": "x_float", "expr": {"kind": "cast", "source": "x", "to": "float"}},
            {"op": "mutate", "column": "id_label", "expr": {"kind": "cast", "source": "id", "to": "str"}},
            {"op": "filter", "column": "num_i", "cmp": "!=", "value": 0},
            {
                "op": "sort",
                "keys": [
                    {"column": "num_i", "ascending": False, "nulls": "last"},
                    {"column": "x_float", "ascending": True, "nulls": "last"},
                    {"column": "id_label", "ascending": True, "nulls": "last"},
                    {"column": "row_nr", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "select", "columns": ["id_label", "num_s", "num_i", "x_float"]},
            {"op": "limit", "n": rnd.randint(2, 6)},
        ]
    elif template == "abs_groupby":
        operations = [
            {"op": "mutate", "column": "x_abs", "expr": {"kind": "abs", "source": "x"}},
            {"op": "filter", "column": "x_abs", "cmp": "is_not_null", "value": None},
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [
                    {"column": "x_abs", "func": "sum", "as": "sum_x_abs"},
                    {"column": "id", "func": "count", "as": "count_id"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "g", "ascending": True, "nulls": "last"},
                    {"column": "sum_x_abs", "ascending": False, "nulls": "last"},
                ],
            },
        ]
    elif template == "abs_topk":
        operations = [
            {"op": "mutate", "column": "y_abs", "expr": {"kind": "abs", "source": "y"}},
            {
                "op": "sort",
                "keys": [
                    {"column": "y_abs", "ascending": False, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                    {"column": "row_nr", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "select", "columns": ["id", "g", "y_abs"]},
            {"op": "limit", "n": rnd.randint(2, 6)},
        ]
    elif template == "clip_groupby":
        lower, upper = _clip_bounds_for_type(rnd, "int")
        operations = [
            {"op": "mutate", "column": "x_clip", "expr": {"kind": "clip", "source": "x", "lower": lower, "upper": upper}},
            {"op": "filter", "column": "x_clip", "cmp": "is_not_null", "value": None},
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [
                    {"column": "x_clip", "func": "sum", "as": "sum_x_clip"},
                    {"column": "id", "func": "count", "as": "count_id"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "g", "ascending": True, "nulls": "last"},
                    {"column": "sum_x_clip", "ascending": False, "nulls": "last"},
                ],
            },
        ]
    elif template == "clip_topk":
        lower, upper = _clip_bounds_for_type(rnd, "float")
        operations = [
            {"op": "mutate", "column": "y_clip", "expr": {"kind": "clip", "source": "y", "lower": lower, "upper": upper}},
            {
                "op": "sort",
                "keys": [
                    {"column": "y_clip", "ascending": False, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                    {"column": "row_nr", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "select", "columns": ["id", "g", "y_clip"]},
            {"op": "limit", "n": rnd.randint(2, 6)},
        ]
    elif template == "double_filter_topk":
        operations = [
            {"op": "filter", "column": "g", "cmp": "is_not_null", "value": None},
            {"op": "filter", "column": "x", "cmp": "!=", "value": 0},
            {
                "op": "sort",
                "keys": [
                    {"column": "g", "ascending": True, "nulls": "last"},
                    {"column": "x", "ascending": False, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                    {"column": "row_nr", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "select", "columns": ["id", "g", "x", "flag"]},
            {"op": "limit", "n": rnd.randint(1, 6)},
        ]
    elif template == "distinct_sort_topk":
        operations = [
            {"op": "filter", "column": "g", "cmp": "is_not_null", "value": None},
            {"op": "distinct", "columns": ["g", "x", "flag"]},
            {
                "op": "sort",
                "keys": [
                    {"column": "g", "ascending": True, "nulls": "last"},
                    {"column": "x", "ascending": True, "nulls": "last"},
                    {"column": "flag", "ascending": False, "nulls": "last"},
                ],
            },
            {"op": "limit", "n": rnd.randint(2, 6)},
        ]
    elif template == "distinct_null_topk":
        table.rows[0]["s"] = None
        table.rows[1]["s"] = ""
        operations = [
            {"op": "distinct", "columns": ["s"]},
            {"op": "sort", "keys": [{"column": "s", "ascending": True, "nulls": "first"}]},
            {"op": "limit", "n": 1},
        ]
    elif template == "filter_distinct_groupby":
        operations = [
            {"op": "filter", "column": "x", "cmp": "range_closed", "value": [-10, 10]},
            {"op": "distinct", "columns": ["g", "id", "x"]},
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "count_id"},
                    {"column": "x", "func": "max", "as": "max_x"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "g", "ascending": True, "nulls": "last"},
                    {"column": "count_id", "ascending": False, "nulls": "last"},
                ],
            },
        ]
    elif template == "fill_null_groupby":
        operations = [
            {"op": "fill_null", "column": "g", "value": "missing"},
            {"op": "fill_null", "column": "x", "value": 0},
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [
                    {"column": "x", "func": "sum", "as": "sum_x"},
                    {"column": "id", "func": "count", "as": "count_id"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "g", "ascending": True, "nulls": "last"},
                    {"column": "sum_x", "ascending": False, "nulls": "last"},
                ],
            },
        ]
    elif template == "fill_null_distinct_topk":
        operations = [
            {"op": "fill_null", "column": "flag", "value": False},
            {"op": "filter", "column": "flag", "cmp": "bool_is_not_false", "value": None},
            {"op": "fill_null", "column": "s", "value": ""},
            {"op": "distinct", "columns": ["s", "g", "flag"]},
            {
                "op": "sort",
                "keys": [
                    {"column": "s", "ascending": True, "nulls": "last"},
                    {"column": "g", "ascending": True, "nulls": "last"},
                    {"column": "flag", "ascending": False, "nulls": "last"},
                ],
            },
            {"op": "limit", "n": rnd.randint(2, 6)},
        ]
    elif template == "fill_null_filter_groupby_topk":
        operations = [
            {"op": "fill_null", "column": "g", "value": "missing"},
            {"op": "fill_null", "column": "x", "value": 0},
            {"op": "filter", "column": "g", "cmp": "!=", "value": ""},
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [
                    {"column": "x", "func": "sum", "as": "sum_x"},
                    {"column": "id", "func": "count", "as": "count_id"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "sum_x", "ascending": False, "nulls": "last"},
                    {"column": "g", "ascending": True, "nulls": "last"},
                    {"column": "count_id", "ascending": False, "nulls": "last"},
                ],
            },
            {"op": "limit", "n": rnd.randint(2, 5)},
        ]
    elif template == "coalesce_fill_null_groupby_topk":
        operations = [
            {"op": "coalesce", "columns": ["g", "s"], "as": "label", "fallback": "missing"},
            {"op": "fill_null", "column": "x", "value": 0},
            {
                "op": "groupby",
                "keys": ["label"],
                "aggs": [
                    {"column": "x", "func": "sum", "as": "sum_x"},
                    {"column": "flag", "func": "any", "as": "any_flag"},
                    {"column": "id", "func": "count", "as": "count_id"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "sum_x", "ascending": False, "nulls": "last"},
                    {"column": "label", "ascending": True, "nulls": "last"},
                    {"column": "count_id", "ascending": False, "nulls": "last"},
                ],
            },
            {"op": "select", "columns": ["label", "sum_x", "any_flag", "count_id"]},
            {"op": "limit", "n": rnd.randint(2, 5)},
        ]
    elif template == "coalesce_groupby":
        operations = [
            {"op": "coalesce", "columns": ["g", "s"], "as": "g_or_s", "fallback": "missing"},
            {
                "op": "groupby",
                "keys": ["g_or_s"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "count_id"},
                    {"column": "x", "func": "sum", "as": "sum_x"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "g_or_s", "ascending": True, "nulls": "last"},
                    {"column": "count_id", "ascending": False, "nulls": "last"},
                ],
            },
        ]
    elif template == "coalesce_topk":
        operations = [
            {"op": "coalesce", "columns": ["x", "id"], "as": "x_or_id", "fallback": 0},
            {"op": "filter", "column": "x_or_id", "cmp": ">=", "value": 0},
            {
                "op": "sort",
                "keys": [
                    {"column": "x_or_id", "ascending": False, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                    {"column": "row_nr", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "select", "columns": ["id", "g", "x_or_id"]},
            {"op": "limit", "n": rnd.randint(2, 6)},
        ]
    elif template == "case_when_classify_topk":
        operations = [
            {
                "op": "case_when",
                "as": "x_bucket",
                "condition": {"column": "x", "cmp": ">=", "value": 0},
                "then": "nonnegative",
                "else": "negative_or_null",
            },
            {"op": "filter", "column": "x_bucket", "cmp": "!=", "value": "negative_or_null"},
            {
                "op": "sort",
                "keys": [
                    {"column": "x_bucket", "ascending": True, "nulls": "last"},
                    {"column": "x", "ascending": False, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                    {"column": "row_nr", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "select", "columns": ["id", "x", "x_bucket"]},
            {"op": "limit", "n": rnd.randint(2, 6)},
        ]
    elif template == "case_when_groupby":
        operations = [
            {
                "op": "case_when",
                "as": "has_flag",
                "condition": {"column": "flag", "cmp": "bool_is_true", "value": None},
                "then": "true_flag",
                "else": "not_true",
            },
            {
                "op": "groupby",
                "keys": ["has_flag"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "count_id"},
                    {"column": "x", "func": "sum", "as": "sum_x"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "has_flag", "ascending": True, "nulls": "last"},
                    {"column": "count_id", "ascending": False, "nulls": "last"},
                ],
            },
        ]
    elif template == "union_all_filter_groupby":
        tables.append(_common_api_append_table(rnd, table))
        operations = [
            {"op": "union_all", "table": "t_append"},
            {"op": "filter", "column": "g", "cmp": "is_not_null", "value": None},
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "count_id"},
                    {"column": "x", "func": "sum", "as": "sum_x"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "g", "ascending": True, "nulls": "last"},
                    {"column": "count_id", "ascending": False, "nulls": "last"},
                ],
            },
        ]
    elif template == "union_all_case_when_topk":
        tables.append(_common_api_append_table(rnd, table))
        operations = [
            {"op": "union_all", "table": "t_append"},
            {
                "op": "case_when",
                "as": "flag_label",
                "condition": {"column": "flag", "cmp": "bool_is_true", "value": None},
                "then": "true_flag",
                "else": "not_true",
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "flag_label", "ascending": True, "nulls": "last"},
                    {"column": "x", "ascending": False, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                    {"column": "row_nr", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "select", "columns": ["id", "x", "flag_label"]},
            {"op": "limit", "n": rnd.randint(2, 6)},
        ]
    elif template == "drop_nulls_groupby":
        operations = [
            {"op": "drop_nulls", "columns": ["g", "x"]},
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "count_id"},
                    {"column": "x", "func": "sum", "as": "sum_x"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "g", "ascending": True, "nulls": "last"},
                    {"column": "sum_x", "ascending": False, "nulls": "last"},
                ],
            },
        ]
    elif template == "union_all_drop_nulls_topk":
        tables.append(_common_api_append_table(rnd, table))
        operations = [
            {"op": "union_all", "table": "t_append"},
            {"op": "drop_nulls", "columns": ["flag", "x"]},
            {
                "op": "case_when",
                "as": "flag_label",
                "condition": {"column": "flag", "cmp": "bool_is_true", "value": None},
                "then": "true_flag",
                "else": "not_true",
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "flag_label", "ascending": True, "nulls": "last"},
                    {"column": "x", "ascending": False, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                    {"column": "row_nr", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "select", "columns": ["id", "x", "flag_label"]},
            {"op": "limit", "n": rnd.randint(2, 6)},
        ]
    elif template == "semi_join_filter_topk":
        tables.append(_common_api_lookup_table(rnd))
        operations = [
            {"op": "semi_join", "table": "t_lookup", "left_on": "id", "right_on": "id"},
            {"op": "filter", "column": "x", "cmp": "is_not_null", "value": None},
            {
                "op": "sort",
                "keys": [
                    {"column": "g", "ascending": True, "nulls": "last"},
                    {"column": "x", "ascending": False, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                    {"column": "row_nr", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "select", "columns": ["id", "g", "x"]},
            {"op": "limit", "n": rnd.randint(2, 6)},
        ]
    elif template == "anti_join_groupby":
        tables.append(_common_api_lookup_table(rnd))
        operations = [
            {"op": "anti_join", "table": "t_lookup", "left_on": "id", "right_on": "id"},
            {"op": "fill_null", "column": "g", "value": "missing"},
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "count_id"},
                    {"column": "x", "func": "sum", "as": "sum_x"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "g", "ascending": True, "nulls": "last"},
                    {"column": "count_id", "ascending": False, "nulls": "last"},
                ],
            },
        ]
    elif template == "top_n_per_group":
        operations = [
            {
                "op": "row_number_filter",
                "partition_by": ["g"],
                "order_by": [
                    {"column": "x", "ascending": False, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                    {"column": "row_nr", "ascending": True, "nulls": "last"},
                ],
                "cmp": "<=",
                "value": rnd.choice([1, 2]),
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "g", "ascending": True, "nulls": "last"},
                    {"column": "x", "ascending": False, "nulls": "last"},
                    {"column": "row_nr", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "select", "columns": ["id", "g", "x", "s"]},
        ]
    elif template == "dedup_latest_per_id":
        operations = [
            {
                "op": "row_number_filter",
                "partition_by": ["id"],
                "order_by": [
                    {"column": "y", "ascending": False, "nulls": "last"},
                    {"column": "row_nr", "ascending": True, "nulls": "last"},
                ],
                "cmp": "==",
                "value": 1,
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "id", "ascending": True, "nulls": "last"},
                    {"column": "row_nr", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "select", "columns": ["id", "g", "y", "s"]},
        ]
    elif template == "running_total_by_group":
        operations = [
            {
                "op": "running_sum",
                "source": "x",
                "column": "run_x",
                "partition_by": ["g"],
                "order_by": [
                    {"column": "id", "ascending": True, "nulls": "last"},
                    {"column": "row_nr", "ascending": True, "nulls": "last"},
                ],
                "input_dtype": "int64",
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "g", "ascending": True, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                    {"column": "row_nr", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "select", "columns": ["id", "g", "x", "run_x"]},
        ]
    elif template == "left_join_fill_groupby":
        tables.append(_common_api_dimension_table(rnd))
        operations = [
            {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"},
            {"op": "fill_null", "column": "tag", "value": "missing"},
            {"op": "fill_null", "column": "j", "value": 0},
            {
                "op": "groupby",
                "keys": ["tag"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "count_id"},
                    {"column": "j", "func": "sum", "as": "sum_j"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "tag", "ascending": True, "nulls": "last"},
                    {"column": "count_id", "ascending": False, "nulls": "last"},
                ],
            },
        ]
    elif template == "clean_key_join_groupby":
        tables.append(_common_api_group_lookup_table())
        operations = [
            {"op": "mutate", "column": "g_clean", "expr": {"kind": "string_strip", "source": "g"}},
            {"op": "join", "table": "t_group_lookup", "left_on": "g_clean", "right_on": "g_clean", "how": "left"},
            {"op": "fill_null", "column": "segment", "value": "unknown"},
            {
                "op": "groupby",
                "keys": ["segment"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "count_id"},
                    {"column": "x", "func": "sum", "as": "sum_x"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "segment", "ascending": True, "nulls": "last"},
                    {"column": "count_id", "ascending": False, "nulls": "last"},
                ],
            },
        ]
    elif template == "filtered_global_aggregate":
        operations = [
            {"op": "filter", "column": "g", "cmp": "is_not_null", "value": None},
            {"op": "filter", "column": "x", "cmp": "range_closed", "value": [-10, 10]},
            {
                "op": "aggregate",
                "aggs": [
                    {"column": "id", "func": "count", "as": "row_count"},
                    {"column": "x", "func": "sum", "as": "sum_x"},
                    {"column": "flag", "func": "any", "as": "any_flag"},
                    {"column": "flag", "func": "all", "as": "all_flag"},
                ],
            },
        ]
    elif template == "groupby_having_topk":
        operations = [
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "count_id"},
                    {"column": "x", "func": "sum", "as": "sum_x"},
                ],
            },
            {"op": "filter", "column": "count_id", "cmp": ">=", "value": 2},
            {
                "op": "sort",
                "keys": [
                    {"column": "sum_x", "ascending": False, "nulls": "last"},
                    {"column": "g", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "limit", "n": rnd.randint(1, 4)},
        ]
    elif template == "pagination_filter_select":
        operations = [
            {"op": "filter", "column": "g", "cmp": "is_not_null", "value": None},
            {
                "op": "sort",
                "keys": [
                    {"column": "g", "ascending": True, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                    {"column": "row_nr", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "offset", "n": rnd.randint(0, 3)},
            {"op": "limit", "n": rnd.randint(2, 6)},
            {"op": "select", "columns": ["id", "g", "s"]},
        ]
    elif template == "nunique_groupby":
        operations = [
            {"op": "filter", "column": "g", "cmp": "is_not_null", "value": None},
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [
                    {"column": "s", "func": "nunique", "as": "unique_s"},
                    {"column": "x", "func": "nunique", "as": "unique_x"},
                    {"column": "id", "func": "count", "as": "count_id"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "unique_s", "ascending": False, "nulls": "last"},
                    {"column": "g", "ascending": True, "nulls": "last"},
                ],
            },
        ]
    elif template == "join_nunique_groupby":
        tables.append(_common_api_dimension_table(rnd))
        operations = [
            {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": rnd.choice(["inner", "left"])},
            {"op": "fill_null", "column": "tag", "value": "missing"},
            {
                "op": "groupby",
                "keys": ["tag"],
                "aggs": [
                    {"column": "g", "func": "nunique", "as": "unique_g"},
                    {"column": "j", "func": "nunique", "as": "unique_j"},
                    {"column": "id", "func": "count", "as": "count_id"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "unique_g", "ascending": False, "nulls": "last"},
                    {"column": "tag", "ascending": True, "nulls": "last"},
                ],
            },
        ]
    elif template == "global_nunique_summary":
        operations = [
            {"op": "filter", "column": "x", "cmp": "range_closed", "value": [-10, 10]},
            {
                "op": "aggregate",
                "aggs": [
                    {"column": "g", "func": "nunique", "as": "unique_g"},
                    {"column": "s", "func": "nunique", "as": "unique_s"},
                    {"column": "flag", "func": "nunique", "as": "unique_flag"},
                    {"column": "id", "func": "count", "as": "row_count"},
                ],
            },
        ]
    elif template == "multi_key_groupby_summary":
        operations = [
            {"op": "mutate", "column": "g_clean", "expr": {"kind": "string_strip", "source": "g"}},
            {"op": "fill_null", "column": "g_clean", "value": "missing"},
            {
                "op": "groupby",
                "keys": ["g_clean", "flag"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "row_count"},
                    {"column": "x", "func": "sum", "as": "sum_x"},
                    {"column": "s", "func": "nunique", "as": "unique_s"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "g_clean", "ascending": True, "nulls": "last"},
                    {"column": "flag", "ascending": False, "nulls": "last"},
                    {"column": "row_count", "ascending": False, "nulls": "last"},
                ],
            },
        ]
    elif template == "multi_key_groupby_topk":
        operations = [
            {
                "op": "groupby",
                "keys": ["g", "flag"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "row_count"},
                    {"column": "y", "func": "max", "as": "max_y"},
                    {"column": "x", "func": "mean", "as": "mean_x"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "max_y", "ascending": False, "nulls": "last"},
                    {"column": "g", "ascending": True, "nulls": "last"},
                    {"column": "flag", "ascending": False, "nulls": "last"},
                ],
            },
            {"op": "limit", "n": rnd.randint(2, 6)},
        ]
    elif template == "empty_filter_groupby_common":
        operations = [
            {"op": "filter", "column": "x", "cmp": ">", "value": 999},
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "row_count"},
                    {"column": "x", "func": "sum", "as": "sum_x"},
                    {"column": "flag", "func": "any", "as": "any_flag"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "g", "ascending": True, "nulls": "last"},
                    {"column": "row_count", "ascending": False, "nulls": "last"},
                ],
            },
        ]
    elif template == "empty_filter_global_aggregate":
        operations = [
            {"op": "filter", "column": "id", "cmp": "<", "value": 0},
            {
                "op": "aggregate",
                "aggs": [
                    {"column": "id", "func": "count", "as": "row_count"},
                    {"column": "x", "func": "sum", "as": "sum_x"},
                    {"column": "y", "func": "mean", "as": "mean_y"},
                    {"column": "flag", "func": "any", "as": "any_flag"},
                    {"column": "flag", "func": "all", "as": "all_flag"},
                ],
            },
        ]
    elif template == "multi_key_join_groupby":
        tables.append(_common_api_composite_dimension_table(rnd))
        operations = [
            {"op": "mutate", "column": "g_clean", "expr": {"kind": "string_strip", "source": "g"}},
            {"op": "filter", "column": "g_clean", "cmp": "is_not_null", "value": None},
            {
                "op": "join",
                "table": "t_composite_dim",
                "left_on": ["id", "g_clean"],
                "right_on": ["id", "g_clean"],
                "how": rnd.choice(["inner", "left"]),
            },
            {"op": "fill_null", "column": "tag2", "value": "missing"},
            {"op": "fill_null", "column": "j2", "value": 0},
            {
                "op": "groupby",
                "keys": ["tag2"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "count_id"},
                    {"column": "j2", "func": "sum", "as": "sum_j2"},
                    {"column": "s", "func": "nunique", "as": "unique_s"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "tag2", "ascending": True, "nulls": "last"},
                    {"column": "count_id", "ascending": False, "nulls": "last"},
                ],
            },
        ]
    elif template == "multi_key_join_topk":
        tables.append(_common_api_composite_dimension_table(rnd))
        operations = [
            {"op": "mutate", "column": "g_clean", "expr": {"kind": "string_strip", "source": "g"}},
            {"op": "filter", "column": "g_clean", "cmp": "is_not_null", "value": None},
            {
                "op": "join",
                "table": "t_composite_dim",
                "left_on": ["id", "g_clean"],
                "right_on": ["id", "g_clean"],
                "how": "left",
            },
            {"op": "fill_null", "column": "tag2", "value": "missing"},
            {
                "op": "sort",
                "keys": [
                    {"column": "tag2", "ascending": True, "nulls": "last"},
                    {"column": "x", "ascending": False, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                    {"column": "row_nr", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "select", "columns": ["id", "g_clean", "tag2", "x"]},
            {"op": "limit", "n": rnd.randint(2, 6)},
        ]
    elif template == "date_part_groupby":
        operations = [
            {"op": "mutate", "column": "dt_year", "expr": {"kind": "date_part", "source": "dt", "part": "year"}},
            {"op": "mutate", "column": "dt_month", "expr": {"kind": "date_part", "source": "dt", "part": "month"}},
            {"op": "fill_null", "column": "x", "value": 0},
            {
                "op": "groupby",
                "keys": ["dt_year", "dt_month"],
                "aggs": [
                    {"column": "x", "func": "sum", "as": "sum_x"},
                    {"column": "id", "func": "count", "as": "count_id"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "dt_year", "ascending": True, "nulls": "last"},
                    {"column": "dt_month", "ascending": True, "nulls": "last"},
                    {"column": "sum_x", "ascending": False, "nulls": "last"},
                ],
            },
            {"op": "select", "columns": ["dt_year", "dt_month", "sum_x", "count_id"]},
            {"op": "limit", "n": rnd.randint(2, 6)},
        ]
    elif template == "date_part_topk":
        operations = [
            {"op": "mutate", "column": "dt_year", "expr": {"kind": "date_part", "source": "dt", "part": "year"}},
            {"op": "mutate", "column": "dt_month", "expr": {"kind": "date_part", "source": "dt", "part": "month"}},
            {"op": "mutate", "column": "dt_day", "expr": {"kind": "date_part", "source": "dt", "part": "day"}},
            {"op": "filter", "column": "dt_year", "cmp": ">=", "value": 2024},
            {
                "op": "sort",
                "keys": [
                    {"column": "dt_year", "ascending": False, "nulls": "last"},
                    {"column": "dt_month", "ascending": True, "nulls": "last"},
                    {"column": "dt_day", "ascending": True, "nulls": "last"},
                    {"column": "row_nr", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "select", "columns": ["id", "dt_year", "dt_month", "dt_day", "x"]},
            {"op": "limit", "n": rnd.randint(2, 6)},
        ]
    elif template == "string_split_part_groupby":
        operations = [
            {"op": "mutate", "column": "s_token", "expr": {"kind": "string_split_part", "source": "s", "sep": " ", "index": 0}},
            {"op": "filter", "column": "s_token", "cmp": "is_not_null", "value": None},
            {
                "op": "groupby",
                "keys": ["s_token"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "count_id"},
                    {"column": "x", "func": "sum", "as": "sum_x"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "s_token", "ascending": True, "nulls": "last"},
                    {"column": "count_id", "ascending": False, "nulls": "last"},
                ],
            },
        ]
    elif template == "string_split_part_topk":
        operations = [
            {"op": "mutate", "column": "g_token", "expr": {"kind": "string_split_part", "source": "g", "sep": " ", "index": 0}},
            {
                "op": "sort",
                "keys": [
                    {"column": "g_token", "ascending": True, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                    {"column": "row_nr", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "select", "columns": ["id", "g_token", "s"]},
            {"op": "limit", "n": rnd.randint(2, 6)},
        ]
    elif template == "string_null_if_empty_groupby":
        operations = [
            {"op": "mutate", "column": "s_norm", "expr": {"kind": "string_null_if_empty", "source": "s"}},
            {"op": "fill_null", "column": "s_norm", "value": "missing"},
            {
                "op": "groupby",
                "keys": ["s_norm"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "count_id"},
                    {"column": "x", "func": "sum", "as": "sum_x"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "s_norm", "ascending": True, "nulls": "last"},
                    {"column": "count_id", "ascending": False, "nulls": "last"},
                ],
            },
        ]
    elif template == "string_null_if_empty_topk":
        operations = [
            {"op": "mutate", "column": "g_norm", "expr": {"kind": "string_null_if_empty", "source": "g"}},
            {
                "op": "sort",
                "keys": [
                    {"column": "g_norm", "ascending": True, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                    {"column": "row_nr", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "select", "columns": ["id", "g_norm", "s"]},
            {"op": "limit", "n": rnd.randint(2, 6)},
        ]
    elif template == "string_length_groupby":
        operations = [
            {"op": "mutate", "column": "s_len", "expr": {"kind": "string_length", "source": "s"}},
            {"op": "fill_null", "column": "s_len", "value": -1},
            {
                "op": "groupby",
                "keys": ["s_len"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "count_id"},
                    {"column": "x", "func": "sum", "as": "sum_x"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "s_len", "ascending": True, "nulls": "last"},
                    {"column": "count_id", "ascending": False, "nulls": "last"},
                ],
            },
        ]
    elif template == "string_length_topk":
        operations = [
            {"op": "mutate", "column": "g_len", "expr": {"kind": "string_length", "source": "g"}},
            {
                "op": "sort",
                "keys": [
                    {"column": "g_len", "ascending": True, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                    {"column": "row_nr", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "select", "columns": ["id", "g", "g_len", "s"]},
            {"op": "limit", "n": rnd.randint(2, 6)},
        ]
    elif template == "string_lower_groupby":
        operations = [
            {"op": "mutate", "column": "s_lower", "expr": {"kind": "string_lower", "source": "s"}},
            {"op": "fill_null", "column": "s_lower", "value": "missing"},
            {
                "op": "groupby",
                "keys": ["s_lower"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "count_id"},
                    {"column": "x", "func": "sum", "as": "sum_x"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "s_lower", "ascending": True, "nulls": "last"},
                    {"column": "count_id", "ascending": False, "nulls": "last"},
                ],
            },
        ]
    elif template == "string_lower_topk":
        operations = [
            {"op": "mutate", "column": "g_lower", "expr": {"kind": "string_lower", "source": "g"}},
            {
                "op": "sort",
                "keys": [
                    {"column": "g_lower", "ascending": True, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                    {"column": "row_nr", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "select", "columns": ["id", "g_lower", "s"]},
            {"op": "limit", "n": rnd.randint(2, 6)},
        ]
    elif template == "string_basename_groupby":
        operations = [
            {"op": "mutate", "column": "path_base", "expr": {"kind": "string_basename", "source": "path_value"}},
            {"op": "fill_null", "column": "path_base", "value": "missing"},
            {
                "op": "groupby",
                "keys": ["path_base"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "count_id"},
                    {"column": "x", "func": "sum", "as": "sum_x"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "path_base", "ascending": True, "nulls": "last"},
                    {"column": "count_id", "ascending": False, "nulls": "last"},
                ],
            },
        ]
    elif template == "string_basename_topk":
        operations = [
            {"op": "mutate", "column": "path_base", "expr": {"kind": "string_basename", "source": "path_value"}},
            {
                "op": "sort",
                "keys": [
                    {"column": "path_base", "ascending": True, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                    {"column": "row_nr", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "select", "columns": ["id", "path_base", "path_value"]},
            {"op": "limit", "n": rnd.randint(2, 6)},
        ]
    elif template == "normalized_string_join_groupby":
        tables.append(_common_api_string_key_lookup_table())
        operations = [
            {"op": "mutate", "column": "s_clean", "expr": {"kind": "string_strip", "source": "s"}},
            {"op": "mutate", "column": "s_key", "expr": {"kind": "string_lower", "source": "s_clean"}},
            {"op": "join", "table": "t_string_lookup", "left_on": "s_key", "right_on": "s_key", "how": "left"},
            {"op": "fill_null", "column": "label", "value": "unmatched"},
            {
                "op": "groupby",
                "keys": ["label"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "count_id"},
                    {"column": "x", "func": "sum", "as": "sum_x"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "label", "ascending": True, "nulls": "last"},
                    {"column": "count_id", "ascending": False, "nulls": "last"},
                ],
            },
        ]
    elif template == "normalized_string_join_topk":
        tables.append(_common_api_string_key_lookup_table())
        operations = [
            {"op": "mutate", "column": "s_clean", "expr": {"kind": "string_strip", "source": "s"}},
            {"op": "mutate", "column": "s_key", "expr": {"kind": "string_lower", "source": "s_clean"}},
            {"op": "join", "table": "t_string_lookup", "left_on": "s_key", "right_on": "s_key", "how": "left"},
            {"op": "fill_null", "column": "label", "value": "unmatched"},
            {
                "op": "sort",
                "keys": [
                    {"column": "label", "ascending": True, "nulls": "last"},
                    {"column": "x", "ascending": False, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                    {"column": "row_nr", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "select", "columns": ["id", "s_key", "label", "s"]},
            {"op": "limit", "n": rnd.randint(2, 6)},
        ]
    elif template == "normalized_string_semi_join_topk":
        tables.append(_common_api_partial_string_key_lookup_table())
        operations = [
            {"op": "mutate", "column": "s_clean", "expr": {"kind": "string_strip", "source": "s"}},
            {"op": "mutate", "column": "s_key", "expr": {"kind": "string_lower", "source": "s_clean"}},
            {"op": "filter", "column": "s_key", "cmp": "is_not_null", "value": None},
            {"op": "semi_join", "table": "t_string_membership", "left_on": "s_key", "right_on": "s_key"},
            {
                "op": "sort",
                "keys": [
                    {"column": "s_key", "ascending": True, "nulls": "last"},
                    {"column": "x", "ascending": False, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                    {"column": "row_nr", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "select", "columns": ["id", "s_key", "s", "x"]},
            {"op": "limit", "n": rnd.randint(2, 6)},
        ]
    elif template == "normalized_string_anti_join_groupby":
        tables.append(_common_api_partial_string_key_lookup_table())
        operations = [
            {"op": "mutate", "column": "s_clean", "expr": {"kind": "string_strip", "source": "s"}},
            {"op": "mutate", "column": "s_key", "expr": {"kind": "string_lower", "source": "s_clean"}},
            {"op": "filter", "column": "s_key", "cmp": "is_not_null", "value": None},
            {"op": "anti_join", "table": "t_string_membership", "left_on": "s_key", "right_on": "s_key"},
            {
                "op": "groupby",
                "keys": ["s_key"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "count_id"},
                    {"column": "x", "func": "sum", "as": "sum_x"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "s_key", "ascending": True, "nulls": "last"},
                    {"column": "count_id", "ascending": False, "nulls": "last"},
                ],
            },
        ]
    elif template == "normalized_string_case_when_groupby":
        operations = [
            {"op": "mutate", "column": "s_clean", "expr": {"kind": "string_strip", "source": "s"}},
            {"op": "mutate", "column": "s_key", "expr": {"kind": "string_lower", "source": "s_clean"}},
            {
                "op": "case_when",
                "as": "string_bucket",
                "condition": {"column": "s_key", "cmp": "in_set", "value": ["alpha", "space value", "padded", ""]},
                "then": "interesting",
                "else": "other_or_null",
            },
            {
                "op": "groupby",
                "keys": ["string_bucket"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "count_id"},
                    {"column": "s_key", "func": "nunique", "as": "unique_key"},
                    {"column": "x", "func": "sum", "as": "sum_x"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "string_bucket", "ascending": True, "nulls": "last"},
                    {"column": "count_id", "ascending": False, "nulls": "last"},
                    {"column": "unique_key", "ascending": False, "nulls": "last"},
                ],
            },
        ]
    elif template == "normalized_string_case_when_topk":
        operations = [
            {"op": "mutate", "column": "s_clean", "expr": {"kind": "string_strip", "source": "s"}},
            {"op": "mutate", "column": "s_key", "expr": {"kind": "string_lower", "source": "s_clean"}},
            {
                "op": "case_when",
                "as": "string_bucket",
                "condition": {"column": "s_key", "cmp": "in_set", "value": ["alpha", "space value", "padded", ""]},
                "then": "interesting",
                "else": "other_or_null",
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "string_bucket", "ascending": True, "nulls": "last"},
                    {"column": "s_key", "ascending": True, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                    {"column": "row_nr", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "select", "columns": ["id", "string_bucket", "s_key", "s"]},
            {"op": "limit", "n": rnd.randint(2, 6)},
        ]
    elif template == "string_pattern_case_when_groupby":
        operations = [
            {
                "op": "mutate",
                "column": "starts_space",
                "expr": {"kind": "string_starts_with", "source": "s", "needle": "space"},
            },
            {
                "op": "case_when",
                "as": "pattern_bucket",
                "condition": {"column": "starts_space", "cmp": "bool_is_true", "value": None},
                "then": "starts-space",
                "else": "other-or-null",
            },
            {
                "op": "groupby",
                "keys": ["pattern_bucket"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "count_id"},
                    {"column": "flag", "func": "any", "as": "any_flag"},
                    {"column": "x", "func": "sum", "as": "sum_x"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "pattern_bucket", "ascending": True, "nulls": "last"},
                    {"column": "count_id", "ascending": False, "nulls": "last"},
                ],
            },
        ]
    elif template == "string_pattern_case_when_topk":
        operations = [
            {
                "op": "mutate",
                "column": "ends_d",
                "expr": {"kind": "string_ends_with", "source": "g", "needle": "d"},
            },
            {
                "op": "case_when",
                "as": "pattern_bucket",
                "condition": {"column": "ends_d", "cmp": "bool_is_true", "value": None},
                "then": "ends-d",
                "else": "other-or-null",
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "pattern_bucket", "ascending": True, "nulls": "last"},
                    {"column": "ends_d", "ascending": False, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                    {"column": "row_nr", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "select", "columns": ["id", "pattern_bucket", "ends_d", "g"]},
            {"op": "limit", "n": rnd.randint(2, 6)},
        ]
    elif template == "sql_distinct_null_coalesce_topk":
        table.rows[0]["s"] = None
        table.rows[1]["s"] = ""
        table.rows[2]["g"] = None
        operations = [
            {"op": "mutate", "column": "s_nonempty", "expr": {"kind": "string_null_if_empty", "source": "s"}},
            {"op": "coalesce", "columns": ["s_nonempty", "g"], "as": "label", "fallback": "missing"},
            {"op": "distinct", "columns": ["label", "flag", "x"]},
            {
                "op": "sort",
                "keys": [
                    {"column": "flag", "ascending": False, "nulls": "first"},
                    {"column": "label", "ascending": True, "nulls": "first"},
                    {"column": "x", "ascending": False, "nulls": "last"},
                ],
            },
            {"op": "offset", "n": rnd.randint(0, 1)},
            {"op": "limit", "n": rnd.randint(2, 5)},
        ]
    elif template == "sql_left_join_coalesce_membership":
        tables.append(_common_api_dimension_table(rnd))
        tables.append(_common_api_segment_membership_table())
        join_kind = rnd.choice(["semi_join", "anti_join"])
        operations = [
            {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"},
            {"op": "coalesce", "columns": ["tag", "g"], "as": "segment_key", "fallback": "missing"},
            {"op": "fill_null", "column": "j", "value": 0},
            {"op": join_kind, "table": "t_segment_membership", "left_on": "segment_key", "right_on": "segment_key"},
        ]
        if join_kind == "semi_join":
            operations.extend(
                [
                    {
                        "op": "groupby",
                        "keys": ["segment_key"],
                        "aggs": [
                            {"column": "id", "func": "count", "as": "count_id"},
                            {"column": "j", "func": "sum", "as": "sum_j"},
                            {"column": "flag", "func": "any", "as": "any_flag"},
                        ],
                    },
                    {
                        "op": "sort",
                        "keys": [
                            {"column": "segment_key", "ascending": True, "nulls": "last"},
                            {"column": "count_id", "ascending": False, "nulls": "last"},
                        ],
                    },
                ]
            )
        else:
            operations.extend(
                [
                    {
                        "op": "sort",
                        "keys": [
                            {"column": "segment_key", "ascending": True, "nulls": "first"},
                            {"column": "j", "ascending": False, "nulls": "last"},
                            {"column": "id", "ascending": True, "nulls": "last"},
                            {"column": "row_nr", "ascending": True, "nulls": "last"},
                        ],
                    },
                    {"op": "select", "columns": ["id", "segment_key", "j", "g"]},
                    {"op": "limit", "n": rnd.randint(2, 6)},
                ]
            )
    elif template == "sql_case_membership_distinct_topk":
        operations = [
            {"op": "mutate", "column": "g_clean", "expr": {"kind": "string_strip", "source": "g"}},
            {
                "op": "case_when",
                "as": "membership_bucket",
                "condition": {"column": "g_clean", "cmp": "in_set", "value": ["a", "space value", "padded"]},
                "then": "member",
                "else": "other_or_null",
            },
            {"op": "distinct", "columns": ["membership_bucket", "flag", "g_clean"]},
            {
                "op": "sort",
                "keys": [
                    {"column": "membership_bucket", "ascending": True, "nulls": "last"},
                    {"column": "flag", "ascending": False, "nulls": "first"},
                    {"column": "g_clean", "ascending": True, "nulls": "first"},
                ],
            },
            {"op": "limit", "n": rnd.randint(2, 6)},
        ]
    elif template == "sql_union_coalesce_distinct_topk":
        tables.append(_common_api_append_table(rnd, table))
        operations = [
            {"op": "union_all", "table": "t_append"},
            {"op": "mutate", "column": "s_nonempty", "expr": {"kind": "string_null_if_empty", "source": "s"}},
            {"op": "coalesce", "columns": ["s_nonempty", "g"], "as": "label", "fallback": "missing"},
            {"op": "distinct", "columns": ["label", "flag", "x"]},
            {
                "op": "sort",
                "keys": [
                    {"column": "label", "ascending": True, "nulls": "first"},
                    {"column": "flag", "ascending": False, "nulls": "first"},
                    {"column": "x", "ascending": False, "nulls": "last"},
                ],
            },
            {"op": "offset", "n": rnd.randint(0, 2)},
            {"op": "limit", "n": rnd.randint(2, 6)},
        ]
    elif template == "sql_left_join_case_membership_groupby":
        tables.append(_common_api_dimension_table(rnd))
        operations = [
            {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"},
            {
                "op": "case_when",
                "as": "tag_bucket",
                "condition": {"column": "tag", "cmp": "in_set", "value": ["dim-a", "space value"]},
                "then": "interesting",
                "else": "other_or_null",
            },
            {"op": "fill_null", "column": "j", "value": 0},
            {
                "op": "groupby",
                "keys": ["tag_bucket"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "count_id"},
                    {"column": "j", "func": "sum", "as": "sum_j"},
                    {"column": "flag", "func": "any", "as": "any_flag"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "tag_bucket", "ascending": True, "nulls": "last"},
                    {"column": "count_id", "ascending": False, "nulls": "last"},
                ],
            },
        ]
    elif template == "sql_left_join_null_predicate_aggregate":
        tables.append(_common_api_dimension_table(rnd))
        operations = [
            {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"},
            {"op": "filter", "column": "tag", "cmp": "is_null", "value": None},
            {
                "op": "aggregate",
                "aggs": [
                    {"column": "id", "func": "count", "as": "count_id"},
                    {"column": "x", "func": "sum", "as": "sum_x"},
                    {"column": "y", "func": "mean", "as": "mean_y"},
                    {"column": "flag", "func": "any", "as": "any_flag"},
                    {"column": "flag", "func": "all", "as": "all_flag"},
                ],
            },
        ]
    elif template == "sql_coalesce_case_distinct_groupby":
        operations = [
            {"op": "mutate", "column": "s_norm", "expr": {"kind": "string_null_if_empty", "source": "s"}},
            {"op": "coalesce", "columns": ["s_norm", "g"], "as": "label", "fallback": "missing"},
            {
                "op": "case_when",
                "as": "label_bucket",
                "condition": {"column": "label", "cmp": "in_set", "value": ["a", "alpha", "space value", "padded"]},
                "then": "interesting",
                "else": "other_or_null",
            },
            {"op": "distinct", "columns": ["label_bucket", "label", "flag"]},
            {
                "op": "groupby",
                "keys": ["label_bucket"],
                "aggs": [
                    {"column": "label", "func": "count", "as": "count_label"},
                    {"column": "label", "func": "nunique", "as": "unique_label"},
                    {"column": "flag", "func": "any", "as": "any_flag"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "label_bucket", "ascending": True, "nulls": "last"},
                    {"column": "count_label", "ascending": False, "nulls": "last"},
                ],
            },
        ]
    elif template == "sql_numeric_text_cast_membership_groupby":
        tables.append(_common_api_numeric_membership_table())
        operations = [
            {
                "op": "mutate",
                "column": "num_value",
                "expr": {"kind": "cast", "source": "num_s", "to": "int", "input_domain": "integer_string"},
            },
            {"op": "filter", "column": "num_value", "cmp": "is_not_null", "value": None},
            {"op": "semi_join", "table": "t_numeric_membership", "left_on": "num_value", "right_on": "num_value"},
            {
                "op": "groupby",
                "keys": ["num_value"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "count_id"},
                    {"column": "x", "func": "sum", "as": "sum_x"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "num_value", "ascending": True, "nulls": "last"},
                    {"column": "count_id", "ascending": False, "nulls": "last"},
                ],
            },
        ]
    elif template == "sql_bool_membership_case_aggregate":
        tables.append(_common_api_boolean_membership_table())
        operations = [
            {"op": "filter", "column": "flag", "cmp": "bool_is_not_unknown", "value": None},
            {"op": "semi_join", "table": "t_bool_membership", "left_on": "flag", "right_on": "flag_key"},
            {
                "op": "case_when",
                "as": "flag_bucket",
                "condition": {"column": "flag", "cmp": "bool_is_true", "value": None},
                "then": "true_member",
                "else": "false_member",
            },
            {
                "op": "groupby",
                "keys": ["flag_bucket"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "count_id"},
                    {"column": "x", "func": "sum", "as": "sum_x"},
                    {"column": "flag", "func": "any", "as": "any_flag"},
                    {"column": "flag", "func": "all", "as": "all_flag"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "flag_bucket", "ascending": True, "nulls": "last"},
                    {"column": "count_id", "ascending": False, "nulls": "last"},
                ],
            },
        ]
    elif template == "sql_left_join_bool_case_groupby":
        tables.append(_common_api_dimension_table(rnd))
        operations = [
            {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"},
            {
                "op": "case_when",
                "as": "dim_missing",
                "condition": {"column": "tag", "cmp": "is_null", "value": None},
                "then": True,
                "else": False,
            },
            {
                "op": "groupby",
                "keys": ["dim_missing"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "count_id"},
                    {"column": "x", "func": "sum", "as": "sum_x"},
                    {"column": "flag", "func": "any", "as": "any_flag"},
                    {"column": "flag", "func": "all", "as": "all_flag"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "dim_missing", "ascending": True, "nulls": "last"},
                    {"column": "count_id", "ascending": False, "nulls": "last"},
                ],
            },
        ]
    elif template == "sql_left_join_bool_coalesce_case_groupby":
        tables.append(_common_api_boolean_dimension_table(rnd))
        operations = [
            {"op": "join", "table": "t_bool_dim", "left_on": "id", "right_on": "id", "how": "left"},
            {"op": "coalesce", "columns": ["dim_flag", "flag"], "as": "flag_effective", "fallback": False},
            {
                "op": "case_when",
                "as": "flag_bucket",
                "condition": {"column": "flag_effective", "cmp": "bool_is_true", "value": None},
                "then": "effective_true",
                "else": "effective_false_or_missing",
            },
            {
                "op": "groupby",
                "keys": ["flag_bucket"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "count_id"},
                    {"column": "x", "func": "sum", "as": "sum_x"},
                    {"column": "flag_effective", "func": "any", "as": "any_flag_effective"},
                    {"column": "flag_effective", "func": "all", "as": "all_flag_effective"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "flag_bucket", "ascending": True, "nulls": "last"},
                    {"column": "count_id", "ascending": False, "nulls": "last"},
                ],
            },
        ]
    elif template == "sql_bool_antijoin_case_aggregate":
        tables.append(_common_api_boolean_membership_table())
        operations = [
            {"op": "filter", "column": "flag", "cmp": "bool_is_not_unknown", "value": None},
            {"op": "anti_join", "table": "t_bool_membership", "left_on": "flag", "right_on": "flag_key"},
            {
                "op": "case_when",
                "as": "flag_bucket",
                "condition": {"column": "flag", "cmp": "bool_is_false", "value": None},
                "then": "false_non_member",
                "else": "other_non_member",
            },
            {
                "op": "groupby",
                "keys": ["flag_bucket"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "count_id"},
                    {"column": "x", "func": "sum", "as": "sum_x"},
                    {"column": "flag", "func": "any", "as": "any_flag"},
                    {"column": "flag", "func": "all", "as": "all_flag"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "flag_bucket", "ascending": True, "nulls": "last"},
                    {"column": "count_id", "ascending": False, "nulls": "last"},
                ],
            },
        ]
    elif template == "sql_left_join_bool_coalesce_filter_groupby":
        tables.append(_common_api_boolean_dimension_table(rnd))
        operations = [
            {"op": "join", "table": "t_bool_dim", "left_on": "id", "right_on": "id", "how": "left"},
            {"op": "coalesce", "columns": ["dim_flag", "flag"], "as": "flag_effective", "fallback": False},
            {"op": "filter", "column": "flag_effective", "cmp": "==", "value": False},
            {
                "op": "case_when",
                "as": "flag_bucket",
                "condition": {"column": "flag_effective", "cmp": "bool_is_false", "value": None},
                "then": "effective_false",
                "else": "effective_true_or_missing",
            },
            {
                "op": "groupby",
                "keys": ["flag_bucket"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "count_id"},
                    {"column": "x", "func": "sum", "as": "sum_x"},
                    {"column": "flag_effective", "func": "any", "as": "any_flag_effective"},
                    {"column": "flag_effective", "func": "all", "as": "all_flag_effective"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "flag_bucket", "ascending": True, "nulls": "last"},
                    {"column": "count_id", "ascending": False, "nulls": "last"},
                ],
            },
        ]
    elif template == "sql_numeric_text_cast_bool_antijoin_groupby":
        tables.append(_common_api_numeric_membership_table())
        operations = [
            {
                "op": "mutate",
                "column": "num_value",
                "expr": {"kind": "cast", "source": "num_s", "to": "int", "input_domain": "integer_string"},
            },
            {"op": "filter", "column": "num_value", "cmp": "is_not_null", "value": None},
            {"op": "filter", "column": "flag", "cmp": "bool_is_not_false", "value": None},
            {"op": "anti_join", "table": "t_numeric_membership", "left_on": "num_value", "right_on": "num_value"},
            {
                "op": "case_when",
                "as": "flag_bucket",
                "condition": {"column": "flag", "cmp": "bool_is_true", "value": None},
                "then": "true_unmatched_number",
                "else": "null_or_false_unmatched_number",
            },
            {
                "op": "groupby",
                "keys": ["flag_bucket"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "count_id"},
                    {"column": "num_value", "func": "sum", "as": "sum_num_value"},
                    {"column": "flag", "func": "any", "as": "any_flag"},
                    {"column": "flag", "func": "all", "as": "all_flag"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "flag_bucket", "ascending": True, "nulls": "last"},
                    {"column": "count_id", "ascending": False, "nulls": "last"},
                ],
            },
        ]
    elif template == "sql_multi_key_semijoin_case_groupby":
        tables.append(_common_api_composite_dimension_table(rnd))
        operations = [
            {"op": "mutate", "column": "g_clean", "expr": {"kind": "string_strip", "source": "g"}},
            {"op": "filter", "column": "g_clean", "cmp": "is_not_null", "value": None},
            {
                "op": "semi_join",
                "table": "t_composite_dim",
                "left_on": ["id", "g_clean"],
                "right_on": ["id", "g_clean"],
            },
            {
                "op": "case_when",
                "as": "flag_bucket",
                "condition": {"column": "flag", "cmp": "bool_is_true", "value": None},
                "then": "matched_true",
                "else": "matched_false_or_null",
            },
            {
                "op": "groupby",
                "keys": ["flag_bucket"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "count_id"},
                    {"column": "x", "func": "sum", "as": "sum_x"},
                    {"column": "flag", "func": "any", "as": "any_flag"},
                    {"column": "flag", "func": "all", "as": "all_flag"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "flag_bucket", "ascending": True, "nulls": "last"},
                    {"column": "count_id", "ascending": False, "nulls": "last"},
                ],
            },
        ]
    elif template == "sql_multi_key_antijoin_case_groupby":
        tables.append(_common_api_composite_dimension_table(rnd))
        operations = [
            {"op": "mutate", "column": "g_clean", "expr": {"kind": "string_strip", "source": "g"}},
            {"op": "filter", "column": "g_clean", "cmp": "is_not_null", "value": None},
            {
                "op": "anti_join",
                "table": "t_composite_dim",
                "left_on": ["id", "g_clean"],
                "right_on": ["id", "g_clean"],
            },
            {
                "op": "case_when",
                "as": "flag_bucket",
                "condition": {"column": "flag", "cmp": "bool_is_true", "value": None},
                "then": "unmatched_true",
                "else": "unmatched_false_or_null",
            },
            {
                "op": "groupby",
                "keys": ["flag_bucket"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "count_id"},
                    {"column": "x", "func": "sum", "as": "sum_x"},
                    {"column": "flag", "func": "any", "as": "any_flag"},
                    {"column": "flag", "func": "all", "as": "all_flag"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "flag_bucket", "ascending": True, "nulls": "last"},
                    {"column": "count_id", "ascending": False, "nulls": "last"},
                ],
            },
        ]
    else:
        raise ValueError(template)

    program = Program(f"prog-{seed:08d}-common-api-workflow", seed, operations)
    return Case(
        case_id=f"case-{seed:08d}-common-api-workflow",
        seed=seed,
        tables=tables,
        program=program,
        metadata={
            "generator_profile": "common_api_workflow",
            "workflow_template": template,
            "discovery_origin": "organic",
        },
    )


def _common_api_base_table(rnd: random.Random) -> TableData:
    columns = [
        ColumnSpec("row_nr", "int", nullable=False),
        ColumnSpec("id", "int", nullable=False),
        ColumnSpec("g", "str", nullable=True),
        ColumnSpec("x", "int", nullable=True),
        ColumnSpec("y", "float", nullable=True),
        ColumnSpec("flag", "bool", nullable=True),
        ColumnSpec("s", "str", nullable=True),
    ]
    group_values = ["a", "b", "c", "space value", " padded ", "", None]
    string_values = ["Alpha", " alpha ", "Beta", "", "space value", "  padded  ", None]
    rows: list[dict[str, Any]] = []
    row_count = rnd.randint(8, 16)
    for idx in range(row_count):
        rows.append(
            {
                "row_nr": idx,
                "id": idx % rnd.choice([5, 7, 11]),
                "g": group_values[(idx + rnd.randint(0, 3)) % len(group_values)],
                "x": None if idx % 7 == 0 else rnd.choice([-10, -2, -1, 0, 1, 2, 5, 10]),
                "y": None if idx % 6 == 0 else rnd.choice([-1.5, -0.5, 0.0, 0.5, 1.5, 10.0]),
                "flag": None if idx % 5 == 0 else rnd.choice([True, False]),
                "s": string_values[(idx + rnd.randint(0, 2)) % len(string_values)],
            }
        )
    return TableData("t0", columns, rows)


def _common_api_with_date_column(table: TableData, rnd: random.Random) -> TableData:
    date_values = [
        "2024-01-03",
        "2024-02-14",
        "2025-02-14T08:30:00",
        "2025-12-31",
        "2026-01-01T00:00:00",
        None,
    ]
    columns = [*table.columns, ColumnSpec("dt", "str", nullable=True)]
    rows = []
    for idx, row in enumerate(table.rows):
        copied = dict(row)
        copied["dt"] = date_values[(idx + rnd.randint(0, 2)) % len(date_values)]
        rows.append(copied)
    return TableData(table.name, columns, rows)


def _common_api_with_numeric_string_column(table: TableData, rnd: random.Random) -> TableData:
    numeric_text_values = ["-10", "-2", "-1", "0", "1", "2", "5", "10", None]
    columns = [*table.columns, ColumnSpec("num_s", "str", nullable=True)]
    rows = []
    for idx, row in enumerate(table.rows):
        copied = dict(row)
        copied["num_s"] = numeric_text_values[(idx + rnd.randint(0, 3)) % len(numeric_text_values)]
        rows.append(copied)
    return TableData(table.name, columns, rows)


def _common_api_with_path_column(table: TableData, rnd: random.Random) -> TableData:
    path_values = [
        "/tmp/alpha.csv",
        "relative/beta.parquet",
        "C:/warehouse/gamma.json",
        "D:\\archive\\delta.log",
        "plain_name.txt",
        "",
        None,
    ]
    columns = [*table.columns, ColumnSpec("path_value", "str", nullable=True)]
    rows = []
    for idx, row in enumerate(table.rows):
        copied = dict(row)
        copied["path_value"] = path_values[(idx + rnd.randint(0, 3)) % len(path_values)]
        rows.append(copied)
    return TableData(table.name, columns, rows)


def _common_api_dimension_table(rnd: random.Random) -> TableData:
    columns = [
        ColumnSpec("id", "int", nullable=False),
        ColumnSpec("j", "int", nullable=True),
        ColumnSpec("tag", "str", nullable=True),
    ]
    rows: list[dict[str, Any]] = []
    for idx in range(12):
        rows.append(
            {
                "id": idx % 6,
                "j": None if idx % 5 == 0 else rnd.choice([-3, 0, 1, 2, 7, 10]),
                "tag": None if idx % 4 == 0 else rnd.choice(["dim-a", "dim-b", "space value"]),
            }
        )
    return TableData("t1", columns, rows)


def _common_api_append_table(rnd: random.Random, base: TableData) -> TableData:
    group_values = ["a", "b", "append", "space value", " padded ", None]
    string_values = ["append", " Alpha ", "", "space value", "  padded  ", None]
    rows: list[dict[str, Any]] = []
    row_count = rnd.randint(3, 8)
    for idx in range(row_count):
        rows.append(
            {
                "row_nr": 1000 + idx,
                "id": 100 + idx % rnd.choice([3, 5, 7]),
                "g": group_values[(idx + rnd.randint(0, 2)) % len(group_values)],
                "x": None if idx % 4 == 0 else rnd.choice([-5, -1, 0, 1, 5, 20]),
                "y": None if idx % 3 == 0 else rnd.choice([-2.0, 0.0, 0.25, 2.5]),
                "flag": None if idx % 4 == 1 else rnd.choice([True, False]),
                "s": string_values[(idx + rnd.randint(0, 2)) % len(string_values)],
            }
        )
    return TableData("t_append", list(base.columns), rows)


def _common_api_lookup_table(rnd: random.Random) -> TableData:
    columns = [
        ColumnSpec("id", "int", nullable=True),
        ColumnSpec("bucket", "str", nullable=True),
    ]
    rows = [
        {"id": 0, "bucket": "keep"},
        {"id": 1, "bucket": "keep"},
        {"id": 1, "bucket": "duplicate"},
        {"id": 3, "bucket": "keep"},
        {"id": None, "bucket": "null-key"},
        {"id": rnd.choice([4, 5, 6]), "bucket": "sampled"},
    ]
    return TableData("t_lookup", columns, rows)


def _common_api_group_lookup_table() -> TableData:
    return TableData(
        "t_group_lookup",
        [
            ColumnSpec("g_clean", "str", nullable=True),
            ColumnSpec("segment", "str", nullable=True),
        ],
        [
            {"g_clean": "a", "segment": "alpha"},
            {"g_clean": "b", "segment": "beta"},
            {"g_clean": "c", "segment": "gamma"},
            {"g_clean": "space value", "segment": "space"},
            {"g_clean": "padded", "segment": "trimmed"},
            {"g_clean": "", "segment": "empty"},
        ],
    )


def _common_api_string_key_lookup_table() -> TableData:
    return TableData(
        "t_string_lookup",
        [
            ColumnSpec("s_key", "str", nullable=False),
            ColumnSpec("label", "str", nullable=False),
        ],
        [
            {"s_key": "alpha", "label": "letter-a"},
            {"s_key": "beta", "label": "letter-b"},
            {"s_key": "gamma", "label": "letter-g"},
            {"s_key": "delta", "label": "letter-d"},
            {"s_key": "space value", "label": "space"},
            {"s_key": "padded", "label": "trimmed"},
            {"s_key": "", "label": "empty"},
        ],
    )


def _common_api_partial_string_key_lookup_table() -> TableData:
    return TableData(
        "t_string_membership",
        [
            ColumnSpec("s_key", "str", nullable=False),
        ],
        [
            {"s_key": "alpha"},
            {"s_key": "space value"},
            {"s_key": "padded"},
        ],
    )


def _common_api_segment_membership_table() -> TableData:
    return TableData(
        "t_segment_membership",
        [
            ColumnSpec("segment_key", "str", nullable=False),
        ],
        [
            {"segment_key": "dim-a"},
            {"segment_key": "dim-b"},
            {"segment_key": "space value"},
            {"segment_key": "missing"},
        ],
    )


def _common_api_numeric_membership_table() -> TableData:
    return TableData(
        "t_numeric_membership",
        [ColumnSpec("num_value", "int", nullable=False)],
        [
            {"num_value": -10},
            {"num_value": -1},
            {"num_value": 0},
            {"num_value": 2},
            {"num_value": 10},
        ],
    )


def _common_api_boolean_membership_table() -> TableData:
    return TableData(
        "t_bool_membership",
        [ColumnSpec("flag_key", "bool", nullable=False)],
        [
            {"flag_key": True},
        ],
    )


def _common_api_boolean_dimension_table(rnd: random.Random) -> TableData:
    rows = []
    for idx in range(12):
        rows.append(
            {
                "id": idx % 7,
                "dim_flag": None if idx % 4 == 0 else rnd.choice([True, False]),
            }
        )
    return TableData(
        "t_bool_dim",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("dim_flag", "bool", nullable=True),
        ],
        rows,
    )


def _common_api_composite_dimension_table(rnd: random.Random) -> TableData:
    rows = []
    groups = ["a", "b", "c", "space value", "padded", ""]
    for idx in range(18):
        group = groups[(idx + rnd.randint(0, 2)) % len(groups)]
        rows.append(
            {
                "id": idx % 7,
                "g_clean": group,
                "j2": None if idx % 6 == 0 else rnd.choice([-2, 0, 1, 3, 8]),
                "tag2": rnd.choice(["mk-a", "mk-b", "mk-c"]),
            }
        )
    return TableData(
        "t_composite_dim",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g_clean", "str", nullable=False),
            ColumnSpec("j2", "int", nullable=True),
            ColumnSpec("tag2", "str", nullable=False),
        ],
        rows,
    )
