from __future__ import annotations

import math
import random
import string
from typing import Any, Callable, Literal

from .csv_roundtrip import DEFAULT_LONG_NUMERIC_CSV_VALUES, csv_long_numeric_values
from .dsl import Case, ColumnSpec, Program, SortKey, TableData, coerce_expression, normalize_sort_keys
from .operation_semantics import (
    aggregate_alias,
    aggregate_column,
    aggregate_func,
    aggregate_specs,
    case_else_value,
    case_then_value,
    coalesce_fallback,
    coalesce_sources,
    condition_cmp,
    condition_column,
    condition_value,
    expr_kind,
    groupby_keys,
    join_how,
    op_ascending,
    op_branches,
    op_column,
    op_columns,
    op_comparator,
    op_input_dtype,
    op_kind,
    op_literal,
    op_output_alias,
    op_partition_columns,
    op_quantiles,
    op_right_columns,
    op_rows,
    op_table,
    op_value,
    op_values,
    operation_names,
    operation_count,
    op_n,
    expr_payload,
)
from .operation_type_semantics import aggregate_accepts_type, case_when_output_type
from .program_state import ProgramState, state_after_operations
from .program_validation import (
    ValidationContext,
    normalized_row_number_filter_op,
    normalized_running_sum_op,
    normalized_sort_op,
    validate_core_operation,
)
from .expression_semantics import aggregate_output_type, cast_output_type, expr_output_type, literal_output_type
from .filtering import filter_comparator_supports_type, parse_filter_comparator
from .identifiers import is_reserved_output_name, make_safe_output_name
from .join_keys import join_key_pairs
from .util import unique_preserve_order

GeneratorProfile = Literal[
    "common",
    "edge_float",
    "workflow",
    "discovery",
    "discovery_fresh",
    "discovery_no_groupby",
    "common_api_workflow",
    "issue_focus",
    "deep_probe_rotation",
    "null_groupby_topk",
    "null_agg_topk",
    "filter_null_agg_topk",
    "join_null_agg_topk",
    "join_null_key_topk",
    "wide_offset_topk",
    "empty_filter_groupby",
    "join_filter_groupby",
    "join_null_truth_filter",
    "join_groupby_stress",
    "storage_offset",
    "float_group_key",
    "join_null_sort",
    "ordered_groupby_sort",
    "topk_resort",
    "join_ordered_agg_topk",
    "global_null_aggregate",
    "string_count_groupby",
    "unique_count_groupby",
    "bool_null_groupby_agg",
    "large_int_filter_groupby",
    "set_membership_filter",
    "pyarrow_groupby_filter_cast_membership",
    "null_predicate_filter",
    "boolean_predicate_filter",
    "post_topk_range_filter",
    "tuple_absence_filter",
    "row_value_absence_filter",
    "running_sum_precision",
    "partitioned_running_sum",
    "path_basename_keyed_pick",
    "sortedness_null_placement",
    "simple_case_random_subject",
    "group_quantile_key_probe",
    "scalar_subquery_double_parentheses",
    "window_avg_rows_frame",
    "struct_distinct_unnest",
    "bit_compare_unequal_length",
    "round_even_float_scale",
    "duckdb_float_literal_precision",
    "polars_timestamp_precision_filter",
    "series_rtruediv_operand_order",
    "polars_reverse_division_columns",
    "pandas_uint64_isin_precision",
    "duckdb_tuple_anti_null_semantics",
    "datafusion_setop_all_duplicate_count",
    "duckdb_json_predicate_order_semantics",
    "pandas_sparse_array_mask_semantics",
    "polars_float_wrap_numerical_semantics",
    "pandas_index_bool_result_type",
    "polars_empty_literal_groupby_semantics",
    "pandas_arrow_string_eq_sum_semantics",
    "pandas_arrow_timestamp_loc_slice_semantics",
    "pandas_arrow_timestamp_index_attr_semantics",
    "pandas_eval_inplace_aliasing_semantics",
    "pandas_bool_reduction_skipna_semantics",
    "pyarrow_dataset_isin_all_match_semantics",
    "pyarrow_run_end_null_compute_semantics",
    "pyarrow_large_string_partition_schema_semantics",
    "pyarrow_hash_pivot_wider_order_semantics",
    "pyarrow_list_flatten_parent_indices_semantics",
    "polars_rolling_mean_by_null_count_semantics",
    "csv_long_numeric_roundtrip",
]


def _is_discovery_profile(profile: GeneratorProfile) -> bool:
    return profile in {"discovery", "discovery_fresh", "discovery_no_groupby", "issue_focus", "deep_probe_rotation"}


def _discovery_profile_allows_groupby(profile: GeneratorProfile) -> bool:
    return profile in {"discovery", "discovery_fresh", "issue_focus", "deep_probe_rotation"}


def _aggregate_accepts_type(source_type: str, func: str, is_numeric_column: bool) -> bool:
    return aggregate_accepts_type(source_type, func, is_numeric_column)


def _aggregate_output_type(source_type: str, func: str) -> str:
    return aggregate_output_type(source_type, func)


def _rand_str(rnd: random.Random) -> str | None:
    choices = ["", "alpha", "beta", "gamma", "δelta", "中文", "space value", "A", "a"]
    if rnd.random() < 0.7:
        return rnd.choice(choices)
    return "".join(rnd.choice(string.ascii_letters) for _ in range(rnd.randint(1, 8)))


def _value_for_type(
    rnd: random.Random,
    typ: str,
    nullable: bool = True,
    profile: GeneratorProfile = "common",
) -> Any:
    if nullable and rnd.random() < 0.18:
        return None
    if typ == "int":
        return rnd.choice([0, 1, -1, 2, -2, 10, -10, rnd.randint(-100, 100)])
    if typ == "float":
        if profile == "edge_float":
            special = rnd.random()
            if special < 0.05:
                return float("nan")
            if special < 0.08:
                return float("inf")
            if special < 0.11:
                return float("-inf")
        # The common profile targets a stable subset. NaN/Infinity are valuable,
        # but they belong in edge_float experiments because engines intentionally
        # disagree on their comparison semantics.
        return rnd.choice([0.0, 1.0, -1.0, 0.5, -0.5, round(rnd.uniform(-50, 50), 3)])
    if typ == "bool":
        return rnd.choice([True, False])
    if typ == "str":
        return _rand_str(rnd)
    raise ValueError(typ)


def generate_table(
    seed: int,
    name: str = "t0",
    min_rows: int = 0,
    max_rows: int = 20,
    profile: GeneratorProfile = "common",
) -> TableData:
    rnd = random.Random(seed)
    base_cols = [
        ColumnSpec("id", "int", nullable=False),
        ColumnSpec("g", "str", nullable=True),
        ColumnSpec("x", "int", nullable=True),
        ColumnSpec("y", "float", nullable=True),
        ColumnSpec("flag", "bool", nullable=True),
        ColumnSpec("s", "str", nullable=True),
    ]
    ncols = len(base_cols) if _is_discovery_profile(profile) else rnd.randint(3, len(base_cols))
    columns = base_cols[:ncols]
    nrows = rnd.randint(min_rows, max_rows)
    rows: list[dict[str, Any]] = []
    for i in range(nrows):
        row: dict[str, Any] = {}
        for col in columns:
            if col.name == "id":
                # Repeated IDs are useful for joins and group-like behavior.
                choices = [i, i % 5, 0, 1]
                if _is_discovery_profile(profile):
                    choices.extend([i % 3, i % 2])
                row[col.name] = rnd.choice(choices)
            else:
                row[col.name] = _value_for_type(rnd, col.type, col.nullable, profile=profile)
        rows.append(row)
    return TableData(name=name, columns=columns, rows=rows)


def generate_join_table(seed: int, profile: GeneratorProfile = "common") -> TableData:
    rnd = random.Random(seed * 3571 + 29)
    columns = [
        ColumnSpec("id", "int", nullable=False),
        ColumnSpec("j", "int", nullable=True),
        ColumnSpec("z", "float", nullable=True),
        ColumnSpec("tag", "str", nullable=True),
    ]
    nrows = rnd.randint(8, 18) if _is_discovery_profile(profile) else rnd.randint(1, 12)
    rows: list[dict[str, Any]] = []
    for i in range(nrows):
        rows.append(
            {
                "id": (
                    rnd.choice([i, i % 5, i % 3, 0, 1, 2])
                    if _is_discovery_profile(profile)
                    else rnd.choice([i, i % 5, 0, 1, 2])
                ),
                "j": _value_for_type(rnd, "int", True, profile=profile),
                "z": _value_for_type(rnd, "float", True, profile=profile),
                "tag": _value_for_type(rnd, "str", True, profile=profile),
            }
        )
    return TableData(name="t1", columns=columns, rows=rows)


def _literal_for_column(rnd: random.Random, table: TableData, col: str) -> Any:
    typ = table.column_type(col)
    values = [r.get(col) for r in table.rows if r.get(col) is not None]
    if values and rnd.random() < 0.65:
        v = rnd.choice(values)
        # Avoid NaN as a comparison literal because NaN equality is intentionally special.
        if isinstance(v, float) and math.isnan(v):
            return 0.0
        return v
    if typ == "int":
        return rnd.choice([-10, -1, 0, 1, 2, 10])
    if typ == "float":
        return rnd.choice([-1.0, 0.0, 0.5, 1.0, 10.0])
    if typ == "bool":
        return rnd.choice([True, False])
    return rnd.choice(["", "alpha", "beta", "中文", "missing"])


def _literal_list_for_column(rnd: random.Random, table: TableData, col: str) -> list[Any]:
    typ = table.column_type(col)
    values = unique_preserve_order([row.get(col) for row in table.rows if row.get(col) is not None])
    values = [value for value in values if not (isinstance(value, float) and math.isnan(value))]
    rnd.shuffle(values)
    selected = values[: rnd.randint(1, max(1, min(3, len(values))))]
    fallback = _literal_list_for_type(rnd, typ)
    return unique_preserve_order([*selected, *fallback])[:3]


def _literal_range_for_column(rnd: random.Random, table: TableData, col: str) -> list[Any]:
    typ = table.column_type(col)
    values = [
        row.get(col)
        for row in table.rows
        if isinstance(row.get(col), (int, float))
        and not isinstance(row.get(col), bool)
        and not (isinstance(row.get(col), float) and math.isnan(row.get(col)))
    ]
    fallback = _literal_list_for_type(rnd, typ)
    pool = unique_preserve_order([*values, *fallback])
    if len(pool) < 2:
        pool = [*_literal_list_for_type(rnd, typ), *_literal_list_for_type(rnd, typ)]
    lower, upper = sorted(rnd.sample(pool, 2))
    return [lower, upper]


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


def _literal_for_type(rnd: random.Random, typ: str) -> Any:
    if typ == "int":
        return rnd.choice([-10, -1, 0, 1, 2, 10])
    if typ == "float":
        return rnd.choice([-1.0, 0.0, 0.5, 1.0, 10.0])
    if typ == "bool":
        return rnd.choice([True, False])
    return rnd.choice(["", "alpha", "beta", "中文", "missing"])


def _string_pattern_literal_for_type(rnd: random.Random, typ: str) -> str:
    if typ != "str":
        raise ValueError(typ)
    return rnd.choice(["a", "A", "beta", "中文", "space", "value", "missing"])


def _date_part_span(part: str) -> tuple[int, int] | None:
    if part == "year":
        return 0, 4
    if part == "month":
        return 5, 2
    if part == "day":
        return 8, 2
    return None


def _cast_output_type(source_type: str, expr: dict[str, Any]) -> str | None:
    return cast_output_type(source_type, expr)


def _literal_list_for_type(rnd: random.Random, typ: str) -> list[Any]:
    if typ == "int":
        return rnd.sample([-10, -1, 0, 1, 2, 10], k=3)
    if typ == "float":
        return rnd.sample([-1.0, 0.0, 0.5, 1.0, 10.0], k=3)
    if typ == "bool":
        return rnd.sample([True, False], k=rnd.randint(1, 2))
    return rnd.sample(["", "alpha", "beta", "中文", "missing"], k=3)


def _case_when_literals(rnd: random.Random, typ: str) -> tuple[Any, Any]:
    if typ == "int":
        return rnd.choice([1, 10]), rnd.choice([0, -1])
    if typ == "float":
        return rnd.choice([1.0, 0.5]), rnd.choice([0.0, -1.0])
    if typ == "bool":
        return True, False
    return rnd.choice(["matched", "yes", "high"]), rnd.choice(["other", "no", "low"])


def _clip_bounds_for_type(rnd: random.Random, typ: str) -> tuple[Any, Any]:
    if typ == "int":
        lower, upper = sorted(rnd.sample([-10, -5, -2, 0, 2, 5, 10], 2))
        return lower, upper
    if typ == "float":
        lower, upper = sorted(rnd.sample([-10.0, -2.5, -1.0, 0.0, 1.0, 2.5, 10.0], 2))
        return lower, upper
    raise ValueError(typ)


def _table_has_compatible_columns(table: TableData, columns: list[str], col_types: dict[str, str]) -> bool:
    table_types = {column.name: column.type for column in table.columns}
    return all(table_types.get(column) == col_types.get(column) for column in columns)


def _compatible_membership_key_pairs(
    extra_tables: list[TableData],
    available_cols: list[str],
    col_types: dict[str, str],
) -> list[tuple[TableData, str, str]]:
    pairs: list[tuple[TableData, str, str]] = []
    for extra in extra_tables:
        right_types = {column.name: column.type for column in extra.columns}
        for left_on in available_cols:
            right_type = right_types.get(left_on)
            if right_type is not None and right_type == col_types.get(left_on):
                pairs.append((extra, left_on, left_on))
    return pairs


def _coalesce_column_groups(available_cols: list[str], col_types: dict[str, str]) -> list[tuple[str, list[str]]]:
    groups: dict[str, list[str]] = {}
    for column in available_cols:
        typ = col_types.get(column)
        if typ is not None:
            groups.setdefault(typ, []).append(column)
    return [(typ, columns) for typ, columns in groups.items() if len(columns) >= 2]


def generate_program(
    seed: int,
    table: TableData,
    max_ops: int = 6,
    type_aware: bool = True,
    extra_tables: list[TableData] | None = None,
    profile: GeneratorProfile = "common",
) -> Program:
    rnd = random.Random(seed * 7919 + 17)
    extra_tables = extra_tables or []
    ops: list[dict[str, Any]] = []
    available_cols = [c.name for c in table.columns]
    col_types = {c.name: c.type for c in table.columns}
    numeric_cols = table.numeric_columns()
    string_cols = [c.name for c in table.columns if c.type == "str"]
    bool_cols = [c.name for c in table.columns if c.type == "bool"]
    comparable_cols = table.comparable_columns()

    op_pool = [
        "filter",
        "drop_nulls",
        "select",
        "distinct",
        "fill_null",
        "coalesce",
        "case_when",
        "sort",
        "limit",
        "offset",
        "mutate",
        "groupby",
    ]
    compatible_union_tables = [
        extra
        for extra in extra_tables
        if _table_has_compatible_columns(extra, available_cols, col_types)
    ]
    compatible_membership_pairs = _compatible_membership_key_pairs(extra_tables, available_cols, col_types)
    if compatible_union_tables:
        op_pool.append("union_all")
    if compatible_membership_pairs:
        op_pool.extend(["semi_join", "anti_join"])
    discovery_profile = _is_discovery_profile(profile)
    if discovery_profile:
        op_pool = [
            "filter",
            "filter",
            "drop_nulls",
            "mutate",
            "mutate",
            "sort",
            "limit",
            "offset",
            "select",
            "distinct",
            "fill_null",
            "coalesce",
            "case_when",
        ]
        if compatible_union_tables:
            op_pool.append("union_all")
        if compatible_membership_pairs:
            op_pool.extend(["semi_join", "anti_join"])
        if _discovery_profile_allows_groupby(profile):
            op_pool.extend(["groupby", "groupby"])
    if extra_tables:
        op_pool.extend(["join", "join"] if discovery_profile else ["join"])
    if discovery_profile:
        nops = rnd.randint(4, max(4, max_ops + 2))
    else:
        nops = rnd.randint(1, max_ops)
    grouped = False
    joined_tables: set[str] = set()
    emitted_ops: set[str] = set()

    for index in range(nops):
        if not type_aware:
            ops.append(
                _generate_type_oblivious_operation(
                    rnd,
                    table,
                    available_cols,
                    allow_groupby=profile != "discovery_no_groupby",
                )
            )
            continue

        before_len = len(ops)
        possible = list(op_pool)
        if grouped:
            possible = ["sort", "limit", "offset", "select"]
        if joined_tables:
            possible = [p for p in possible if p != "join"]
        if "union_all" in possible:
            compatible_union_tables = [
                extra
                for extra in extra_tables
                if extra.name not in joined_tables and _table_has_compatible_columns(extra, available_cols, col_types)
            ]
            if not compatible_union_tables:
                possible = [p for p in possible if p != "union_all"]
        if "semi_join" in possible or "anti_join" in possible:
            compatible_membership_pairs = _compatible_membership_key_pairs(extra_tables, available_cols, col_types)
            if not compatible_membership_pairs:
                possible = [p for p in possible if p not in {"semi_join", "anti_join"}]
        remaining = nops - index
        if discovery_profile and not grouped:
            if extra_tables and not joined_tables and "id" in available_cols and (not ops or rnd.random() < 0.8):
                op = "join"
            elif "mutate" not in emitted_ops and (numeric_cols or string_cols) and len(ops) >= int(bool(extra_tables)):
                op = "mutate"
            elif "filter" not in emitted_ops and comparable_cols and len(ops) >= 2 and remaining > 2:
                op = "filter"
            elif (
                _discovery_profile_allows_groupby(profile)
                and
                "groupby" not in emitted_ops
                and (numeric_cols or bool_cols)
                and len(ops) >= 3
                and (remaining <= 3 or rnd.random() < 0.75)
            ):
                op = "groupby"
            else:
                op = rnd.choice(possible)
        else:
            op = rnd.choice(possible)

        if op == "join" and extra_tables and not grouped and "id" in available_cols:
            right = rnd.choice([t for t in extra_tables if t.name not in joined_tables] or extra_tables)
            if any(c.name == "id" for c in right.columns):
                ops.append(
                    {
                        "op": "join",
                        "table": right.name,
                        "left_on": "id",
                        "right_on": "id",
                        "how": rnd.choice(["inner", "left"]),
                    }
                )
                joined_tables.add(right.name)
                for col in right.columns:
                    if col.name == "id" or col.name in available_cols:
                        continue
                    available_cols.append(col.name)
                    col_types[col.name] = col.type
                    comparable_cols.append(col.name)
                    if col.type in {"int", "float"}:
                        numeric_cols.append(col.name)
                    if col.type == "str":
                        string_cols.append(col.name)
                    if col.type == "bool":
                        bool_cols.append(col.name)

        elif op == "union_all" and compatible_union_tables and not grouped:
            right = rnd.choice(compatible_union_tables)
            ops.append({"op": "union_all", "table": right.name})

        elif op in {"semi_join", "anti_join"} and compatible_membership_pairs and not grouped:
            right, left_on, right_on = rnd.choice(compatible_membership_pairs)
            ops.append({"op": op, "table": right.name, "left_on": left_on, "right_on": right_on})

        elif op == "drop_nulls" and available_cols and not grouped:
            width = rnd.randint(1, min(3, len(available_cols)))
            ops.append({"op": "drop_nulls", "columns": rnd.sample(available_cols, k=width)})

        elif op == "filter" and comparable_cols and not grouped:
            col = rnd.choice(comparable_cols)
            typ = col_types[col]
            cmp_ops = ["==", "!="] if typ in {"str", "bool"} else [">", ">=", "<", "<=", "==", "!="]
            if discovery_profile and typ in {"int", "float"} and rnd.random() < 0.35:
                cmp_ops = [
                    "gt_is_not_true",
                    "ge_is_not_true",
                    "lt_is_not_false",
                    "le_is_not_false",
                    *cmp_ops,
                ]
            if discovery_profile and rnd.random() < 0.20:
                cmp_ops = [rnd.choice(["in_set", "not_in_set"]), *cmp_ops]
            if discovery_profile and rnd.random() < 0.15:
                cmp_ops = [rnd.choice(["is_null", "is_not_null"]), *cmp_ops]
            if discovery_profile and typ == "bool" and rnd.random() < 0.35:
                cmp_ops = [
                    rnd.choice(["bool_is_true", "bool_is_not_true", "bool_is_false", "bool_is_not_false"]),
                    *cmp_ops,
                ]
            if discovery_profile and typ in {"int", "float"} and rnd.random() < 0.20:
                cmp_ops = ["range_closed", *cmp_ops]
            base_cols = {c.name for c in table.columns}
            cmp = rnd.choice(cmp_ops)
            if cmp in {"in_set", "not_in_set"}:
                value = _literal_list_for_column(rnd, table, col) if col in base_cols else _literal_list_for_type(rnd, typ)
            elif cmp == "range_closed":
                value = _literal_range_for_column(rnd, table, col) if col in base_cols else sorted(rnd.sample(_literal_list_for_type(rnd, typ), 2))
            elif cmp in {"is_null", "is_not_null", "bool_is_true", "bool_is_not_true", "bool_is_false", "bool_is_not_false"}:
                value = None
            else:
                value = _literal_for_column(rnd, table, col) if col in base_cols else _literal_for_type(rnd, typ)
            ops.append({"op": "filter", "column": col, "cmp": cmp, "value": value})

        elif op == "select" and available_cols:
            k = rnd.randint(1, len(available_cols))
            cols = sorted(rnd.sample(available_cols, k))
            ops.append({"op": "select", "columns": cols})
            available_cols = cols
            numeric_cols = [c for c in numeric_cols if c in available_cols]
            string_cols = [c for c in string_cols if c in available_cols]
            bool_cols = [c for c in bool_cols if c in available_cols]
            comparable_cols = [c for c in comparable_cols if c in available_cols]

        elif op == "distinct" and available_cols:
            k = rnd.randint(1, min(3, len(available_cols)))
            cols = sorted(rnd.sample(available_cols, k))
            ops.append({"op": "distinct", "columns": cols})
            available_cols = cols
            numeric_cols = [c for c in numeric_cols if c in available_cols]
            string_cols = [c for c in string_cols if c in available_cols]
            bool_cols = [c for c in bool_cols if c in available_cols]
            comparable_cols = [c for c in comparable_cols if c in available_cols]

        elif op == "fill_null" and available_cols and not grouped:
            candidates = [col for col in available_cols if col in col_types]
            if not candidates:
                continue
            col = rnd.choice(candidates)
            ops.append({"op": "fill_null", "column": col, "value": _literal_for_type(rnd, col_types[col])})

        elif op == "coalesce" and not grouped:
            groups = _coalesce_column_groups(available_cols, col_types)
            if not groups:
                continue
            output_type, candidates = rnd.choice(groups)
            width = rnd.randint(2, min(3, len(candidates)))
            columns = rnd.sample(candidates, width)
            alias = make_safe_output_name(
                f"co_{columns[0]}",
                used=set(available_cols) | {op_output_alias(op) for op in ops if op_output_alias(op)},
            )
            coalesce_op: dict[str, Any] = {"op": "coalesce", "columns": columns, "as": alias}
            if rnd.random() < 0.75:
                coalesce_op["fallback"] = _literal_for_type(rnd, output_type)
            ops.append(coalesce_op)
            available_cols.append(alias)
            col_types[alias] = output_type
            comparable_cols.append(alias)
            if output_type in {"int", "float"}:
                numeric_cols.append(alias)
            if output_type == "str":
                string_cols.append(alias)
            if output_type == "bool":
                bool_cols.append(alias)

        elif op == "case_when" and comparable_cols and not grouped:
            predicate_col = rnd.choice(comparable_cols)
            predicate_type = col_types[predicate_col]
            cmp_ops = ["==", "!="] if predicate_type in {"str", "bool"} else [">", ">=", "<", "<=", "==", "!="]
            if discovery_profile and predicate_type in {"int", "float"} and rnd.random() < 0.25:
                cmp_ops = [
                    rnd.choice(["gt_is_not_true", "ge_is_not_true", "lt_is_not_false", "le_is_not_false"]),
                    *cmp_ops,
                ]
            if discovery_profile and predicate_type == "bool" and rnd.random() < 0.35:
                cmp_ops = [
                    rnd.choice(["bool_is_true", "bool_is_not_true", "bool_is_false", "bool_is_not_false"]),
                    *cmp_ops,
                ]
            cmp = rnd.choice(cmp_ops)
            base_cols = {c.name for c in table.columns}
            if cmp in {"is_null", "is_not_null", "bool_is_true", "bool_is_not_true", "bool_is_false", "bool_is_not_false"}:
                predicate_value = None
            else:
                predicate_value = (
                    _literal_for_column(rnd, table, predicate_col)
                    if predicate_col in base_cols
                    else _literal_for_type(rnd, predicate_type)
                )
            output_type = rnd.choice(["str", "int", "bool"])
            then_value, else_value = _case_when_literals(rnd, output_type)
            alias = make_safe_output_name(f"cw_{operation_count(ops, 'case_when')}", used=available_cols)
            ops.append(
                {
                    "op": "case_when",
                    "as": alias,
                    "condition": {"column": predicate_col, "cmp": cmp, "value": predicate_value},
                    "then": then_value,
                    "else": else_value,
                }
            )
            available_cols.append(alias)
            col_types[alias] = output_type
            comparable_cols.append(alias)
            if output_type in {"int", "float"}:
                numeric_cols.append(alias)
            if output_type == "str":
                string_cols.append(alias)
            if output_type == "bool":
                bool_cols.append(alias)

        elif op == "sort" and available_cols:
            first = rnd.choice(available_cols)
            cols = [first] + sorted(c for c in available_cols if c != first)
            ops.append(_random_sort_op(rnd, cols, allow_mixed=_is_discovery_profile(profile)))

        elif op == "limit":
            ops.append({"op": "limit", "n": rnd.randint(0, max(1, len(table.rows) + 2))})

        elif op == "offset":
            ops.append({"op": "offset", "n": rnd.randint(0, max(1, len(table.rows) + 2))})

        elif op == "mutate" and (numeric_cols or string_cols) and not grouped:
            new_col = f"m_{operation_count(ops, 'mutate')}"
            expr, out_type = _random_mutate_expr(rnd, numeric_cols, string_cols, col_types, profile=profile)
            ops.append({"op": "mutate", "column": new_col, "expr": expr})
            available_cols.append(new_col)
            col_types[new_col] = out_type
            comparable_cols.append(new_col)
            if out_type in {"int", "float"}:
                numeric_cols.append(new_col)
            if out_type == "str":
                string_cols.append(new_col)

        elif op == "groupby" and (numeric_cols or bool_cols) and available_cols and not grouped:
            keys = [rnd.choice(available_cols)]
            # Avoid grouping by float columns for common-subset stability.
            key_candidates = [c for c in available_cols if col_types.get(c) in {"int", "str", "bool"}]
            if key_candidates:
                key_count = 1 if len(key_candidates) == 1 or rnd.random() < 0.8 else 2
                keys = sorted(rnd.sample(key_candidates, key_count))
            agg_candidates = list(numeric_cols)
            if discovery_profile:
                agg_candidates.extend(c for c in bool_cols if c not in agg_candidates)
            if not agg_candidates:
                continue
            agg_count = rnd.randint(1, min(3, len(agg_candidates)))
            aggs = []
            used_aliases = set()
            for val in rnd.sample(agg_candidates, agg_count):
                if col_types.get(val) == "bool":
                    func = rnd.choice(["any", "all", "min", "max", "count", "nunique"])
                else:
                    func = rnd.choice(["sum", "mean", "min", "max", "count", "nunique"])
                alias = make_safe_output_name(f"{func}_{val}", used=used_aliases | set(keys))
                used_aliases.add(alias)
                aggs.append({"column": val, "func": func, "as": alias})
            ops.append({"op": "groupby", "keys": keys, "aggs": aggs})
            available_cols = keys + [aggregate_alias(agg) for agg in aggs]
            numeric_cols = []
            bool_cols = []
            for agg in aggs:
                source_column = aggregate_column(agg)
                alias = aggregate_alias(agg)
                output_type = _aggregate_output_type(col_types.get(source_column, "float"), aggregate_func(agg))
                col_types[alias] = output_type
                if output_type in {"int", "float"}:
                    numeric_cols.append(alias)
                if output_type == "bool":
                    bool_cols.append(alias)
            string_cols = [c for c in keys if col_types.get(c) == "str"]
            comparable_cols = available_cols
            grouped = True
        if len(ops) > before_len:
            emitted_ops.add(op_kind(ops[-1]))

    if type_aware:
        ops = repair_operations(table, ops, extra_tables=extra_tables)
        if discovery_profile:
            ops = _add_discovery_order_projection_probe(ops, table, extra_tables, rnd)
    if not ops:
        ops.append({"op": "limit", "n": len(table.rows)})
    return Program(program_id=f"prog-{seed:08d}", seed=seed, operations=ops)


def _add_discovery_order_projection_probe(
    ops: list[dict[str, Any]],
    table: TableData,
    extra_tables: list[TableData],
    rnd: random.Random,
) -> list[dict[str, Any]]:
    if rnd.random() >= 0.35:
        return ops
    available = _available_columns_after_operations(table, ops, extra_tables=extra_tables)
    if len(available) < 2:
        return ops

    primary = rnd.choice(available)
    selected_candidates = [column for column in available if column != primary]
    selected_count = rnd.randint(1, min(3, len(selected_candidates)))
    selected = sorted(rnd.sample(selected_candidates, selected_count))
    sort_columns = [primary] + sorted(column for column in available if column != primary)
    out = list(ops)
    out.append(_random_sort_op(rnd, sort_columns, allow_mixed=True))
    out.append({"op": "select", "columns": selected})
    if rnd.random() < 0.75:
        if rnd.random() < 0.70:
            out.append({"op": "limit", "n": rnd.randint(1, max(1, min(len(table.rows) + 2, 8)))})
        else:
            out.append({"op": "offset", "n": rnd.randint(0, 2)})
    return out


def _available_columns_after_operations(
    table: TableData,
    ops: list[dict[str, Any]],
    extra_tables: list[TableData] | None = None,
) -> list[str]:
    state = state_after_operations(table, ops, extra_tables=extra_tables or [])
    return unique_preserve_order(state.columns)


def _random_mutate_expr(
    rnd: random.Random,
    numeric_cols: list[str],
    string_cols: list[str],
    col_types: dict[str, str],
    profile: GeneratorProfile = "common",
) -> tuple[dict[str, Any], str]:
    choices: list[str] = []
    if numeric_cols:
        choices.extend(["add_const", "arith_const", "cast_float", "cast_string"])
    if string_cols:
        choices.extend([
            "string_length",
            "string_lower",
            "string_upper",
            "string_null_if_empty",
            "string_split_part",
            "string_basename",
            "string_contains",
            "string_starts_with",
            "string_ends_with",
        ])
    kind = rnd.choice(choices)
    if kind == "add_const":
        src = rnd.choice(numeric_cols)
        return {"kind": "add_const", "source": src, "value": rnd.choice([-2, -1, 0, 1, 2, 10])}, col_types[src]
    if kind == "arith_const":
        src = rnd.choice(numeric_cols)
        op_choices = ["sub", "mul", "div"]
        if profile == "edge_float":
            op_choices.append("mod")
        op = rnd.choice(op_choices)
        value = rnd.choice([2, 3, 5, 10]) if op in {"div", "mod"} else rnd.choice([-2, -1, 1, 2, 10])
        out_type = "float" if op == "div" or col_types[src] == "float" else col_types[src]
        return {"kind": "arith_const", "source": src, "op": op, "value": value}, out_type
    if kind == "cast_float":
        src = rnd.choice(numeric_cols)
        return {"kind": "cast", "source": src, "to": "float"}, "float"
    if kind == "cast_string":
        int_cols = [col for col in numeric_cols if col_types.get(col) == "int"]
        src = rnd.choice(int_cols or numeric_cols)
        return {"kind": "cast", "source": src, "to": "str"}, "str"
    if kind == "string_length":
        return {"kind": "string_length", "source": rnd.choice(string_cols)}, "int"
    if kind == "string_lower":
        return {"kind": "string_lower", "source": rnd.choice(string_cols)}, "str"
    if kind == "string_upper":
        return {"kind": "string_upper", "source": rnd.choice(string_cols)}, "str"
    if kind == "string_null_if_empty":
        return {"kind": "string_null_if_empty", "source": rnd.choice(string_cols)}, "str"
    if kind == "string_split_part":
        return {
            "kind": "string_split_part",
            "source": rnd.choice(string_cols),
            "sep": rnd.choice([" ", "-", "_"]),
            "index": 0,
        }, "str"
    if kind == "string_basename":
        return {"kind": "string_basename", "source": rnd.choice(string_cols)}, "str"
    if kind == "string_contains":
        return {
            "kind": "string_contains",
            "source": rnd.choice(string_cols),
            "needle": rnd.choice(["a", "A", "space", "pad"]),
        }, "bool"
    if kind == "string_starts_with":
        return {
            "kind": "string_starts_with",
            "source": rnd.choice(string_cols),
            "needle": rnd.choice(["a", "A", "space", "pad"]),
        }, "bool"
    if kind == "string_ends_with":
        return {
            "kind": "string_ends_with",
            "source": rnd.choice(string_cols),
            "needle": rnd.choice(["a", "A", "e", "d"]),
        }, "bool"
    raise ValueError(kind)


def _generate_type_oblivious_operation(
    rnd: random.Random,
    table: TableData,
    available_cols: list[str],
    *,
    allow_groupby: bool = True,
) -> dict[str, Any]:
    col = rnd.choice(available_cols)
    kinds = ["filter", "select", "sort", "limit", "offset", "mutate"]
    if allow_groupby:
        kinds.append("groupby")
    kind = rnd.choice(kinds)
    if kind == "filter":
        return {
            "op": "filter",
            "column": col,
            "cmp": rnd.choice([">", ">=", "<", "<=", "==", "!="]),
            "value": rnd.choice([None, -1, 0, 1, 0.5, True, False, "alpha", "missing"]),
        }
    if kind == "select":
        return {"op": "select", "columns": sorted(rnd.sample(available_cols, rnd.randint(1, len(available_cols))))}
    if kind == "sort":
        cols = [col] + sorted(c for c in available_cols if c != col)
        return {"op": "sort", "columns": cols, "ascending": rnd.choice([True, False])}
    if kind == "limit":
        return {"op": "limit", "n": rnd.randint(0, max(1, len(table.rows) + 2))}
    if kind == "offset":
        return {"op": "offset", "n": rnd.randint(0, max(1, len(table.rows) + 2))}
    if kind == "mutate":
        return {
            "op": "mutate",
            "column": f"u_{rnd.randint(0, 9)}",
            "expr": {"kind": "add_const", "source": col, "value": rnd.choice([-1, 0, 1])},
        }
    numeric_cols = table.numeric_columns()
    agg_col = rnd.choice(numeric_cols or available_cols)
    func = rnd.choice(
        ["sum", "mean", "min", "max", "count", "nunique"] if agg_col in numeric_cols else ["count", "nunique"]
    )
    alias = make_safe_output_name(f"{func}_{agg_col}", used={col})
    return {"op": "groupby", "keys": [col], "aggs": [{"column": agg_col, "func": func, "as": alias}]}


def repair_operations(
    table: TableData,
    ops: list[dict[str, Any]],
    extra_tables: list[TableData] | None = None,
) -> list[dict[str, Any]]:
    """Keep generated programs inside the common semantic subset.

    This is a guardrail for paper-quality experiments: invalid generated
    programs mostly measure generator bugs and backend error-message variance.
    """

    repaired: list[dict[str, Any]] = []
    extra_tables = extra_tables or []
    table_by_name = {t.name: t for t in [table] + extra_tables}
    available = {c.name for c in table.columns}
    col_types = {c.name: c.type for c in table.columns}
    numeric = {c.name for c in table.columns if c.type in {"int", "float"}}
    strings = {c.name for c in table.columns if c.type == "str"}

    def current_state() -> ProgramState:
        ordered = [column.name for column in table.columns if column.name in available]
        extras = sorted(column for column in available if column not in ordered)
        nullable = {
            column.name
            for column in table.columns
            if column.nullable or any(row.get(column.name) is None for row in table.rows if column.name in row)
        }
        nullable |= {column for column in available if column not in col_types}
        return ProgramState(ordered + extras, dict(col_types), nullable)

    order_pending = False
    pending_order_columns: set[str] = set()
    for op in ops:
        kind = op_kind(op)
        if kind == "join":
            right = table_by_name.get(op_table(op))
            left_keys, right_keys = join_key_pairs(op)
            right_types = {c.name: c.type for c in right.columns} if right is not None else {}
            if (
                right is None
                or not left_keys
                or len(left_keys) != len(right_keys)
                or any(left_key not in available for left_key in left_keys)
                or any(right_key not in right_types for right_key in right_keys)
                or any(col_types.get(left_key) != right_types.get(right_key) for left_key, right_key in zip(left_keys, right_keys))
                or join_how(op) not in {"inner", "left"}
            ):
                continue
            repaired.append(op)
            right_key_set = set(right_keys)
            for col in right.columns:
                if col.name in right_key_set or col.name in available:
                    continue
                available.add(col.name)
                col_types[col.name] = col.type
                if col.type in {"int", "float"}:
                    numeric.add(col.name)
                if col.type == "str":
                    strings.add(col.name)
            order_pending = False
            pending_order_columns = set()
        elif kind == "union_all":
            right = table_by_name.get(op_table(op))
            if right is None:
                continue
            right_types = {column.name: column.type for column in right.columns}
            if any(right_types.get(column) != col_types.get(column) for column in available):
                continue
            repaired.append({"op": "union_all", "table": right.name})
            order_pending = False
            pending_order_columns = set()
        elif kind in {"semi_join", "anti_join"}:
            right = table_by_name.get(op_table(op))
            left_keys, right_keys = join_key_pairs(op)
            if right is None or not left_keys or len(left_keys) != len(right_keys):
                continue
            right_types = {column.name: column.type for column in right.columns}
            if any(left_key not in available for left_key in left_keys):
                continue
            if any(right_key not in right_types for right_key in right_keys):
                continue
            if any(col_types.get(left_key) != right_types.get(right_key) for left_key, right_key in zip(left_keys, right_keys)):
                continue
            left_on: str | list[str] = left_keys[0] if len(left_keys) == 1 else left_keys
            right_on: str | list[str] = right_keys[0] if len(right_keys) == 1 else right_keys
            repaired.append({"op": kind, "table": right.name, "left_on": left_on, "right_on": right_on})
        elif kind == "drop_nulls":
            cols = unique_preserve_order([column for column in op_columns(op) if column in available])
            if not cols:
                continue
            repaired.append({"op": "drop_nulls", "columns": cols})
        elif kind == "filter":
            if op_column(op) not in available:
                continue
            column_type = col_types.get(op_column(op), "")
            if not _filter_literal_is_valid(column_type, op_comparator(op), op_value(op)):
                continue
            repaired.append(op)
        elif kind == "tuple_absence_filter":
            right = table_by_name.get(op_table(op))
            columns = unique_preserve_order(op_columns(op))
            right_columns = op_right_columns(op)
            if right is None or not columns or len(columns) != len(right_columns):
                continue
            right_types = {column.name: column.type for column in right.columns}
            if any(column not in available for column in columns):
                continue
            if any(column not in right_types for column in right_columns):
                continue
            if any(col_types.get(left) != right_types.get(right_col) for left, right_col in zip(columns, right_columns)):
                continue
            repaired.append(
                {
                    "op": "tuple_absence_filter",
                    "columns": columns,
                    "table": right.name,
                    "right_columns": right_columns,
                }
            )
        elif kind == "running_sum":
            repaired_op = normalized_running_sum_op(op, available)
            if repaired_op is None:
                continue
            ctx = ValidationContext(tables=table_by_name, state=current_state(), errors=[])
            if not validate_core_operation(ctx, repaired_op, len(repaired)):
                continue
            repaired.append(repaired_op)
            column = op_column(repaired_op)
            available.add(column)
            col_types[column] = "float"
            numeric.add(column)
            strings.discard(column)
            order_pending = True
            pending_order_columns = {key.column for key in normalize_sort_keys(repaired_op)}
        elif kind == "row_number_filter":
            repaired_op = normalized_row_number_filter_op(op, available)
            if repaired_op is None:
                continue
            ctx = ValidationContext(tables=table_by_name, state=current_state(), errors=[])
            if not validate_core_operation(ctx, repaired_op, len(repaired)):
                continue
            repaired.append(repaired_op)
            order_pending = True
            pending_order_columns = {key.column for key in normalize_sort_keys({"keys": repaired_op["order_by"]})} | set(
                op_partition_columns(repaired_op)
            )
        elif kind == "sortedness_check":
            column = op_column(op)
            alias = op_output_alias(op)
            ascending = op_ascending(op)
            nulls = op_nulls(op)
            if column not in available:
                continue
            if not alias or is_reserved_output_name(alias):
                continue
            if nulls not in {"first", "last"}:
                continue
            repaired.append(
                {
                    "op": "sortedness_check",
                    "column": column,
                    "as": alias,
                    "ascending": ascending,
                    "nulls": nulls,
                }
            )
            available = {alias}
            col_types = {alias: "bool"}
            numeric = set()
            strings = set()
            order_pending = False
            pending_order_columns = set()
        elif kind == "random_case_probe":
            alias = op_output_alias(op)
            if not alias or is_reserved_output_name(alias):
                continue
            try:
                row_count = op_rows(op, 100_000)
                branch_count = op_branches(op, 3)
            except (TypeError, ValueError):
                continue
            if row_count <= 0 or branch_count <= 0 or branch_count > 16:
                continue
            repaired.append(
                {
                    "op": "random_case_probe",
                    "as": alias,
                    "rows": row_count,
                    "branches": branch_count,
                }
            )
            available = {alias}
            col_types = {alias: "bool"}
            numeric = set()
            strings = set()
            order_pending = False
            pending_order_columns = set()
        elif kind == "group_quantile_probe":
            alias = op_output_alias(op)
            if not alias or is_reserved_output_name(alias):
                continue
            values = op_values(op) or [1, 2, 3]
            quantiles = op_quantiles(op) or [0.0, 0.5, 1.0]
            if not _valid_quantile_probe_values(values, quantiles):
                continue
            repaired.append(
                {
                    "op": "group_quantile_probe",
                    "as": alias,
                    "values": list(values),
                    "quantiles": list(quantiles),
                }
            )
            available = {alias}
            col_types = {alias: "bool"}
            numeric = set()
            strings = set()
            order_pending = False
            pending_order_columns = set()
        elif kind == "scalar_subquery_probe":
            alias = op_output_alias(op)
            if not alias or is_reserved_output_name(alias):
                continue
            repaired.append({"op": "scalar_subquery_probe", "as": alias})
            available = {alias}
            col_types = {alias: "bool"}
            numeric = set()
            strings = set()
            order_pending = False
            pending_order_columns = set()
        elif kind == "window_avg_probe":
            alias = op_output_alias(op)
            if not alias or is_reserved_output_name(alias):
                continue
            repaired.append({"op": "window_avg_probe", "as": alias})
            available = {alias}
            col_types = {alias: "bool"}
            numeric = set()
            strings = set()
            order_pending = False
            pending_order_columns = set()
        elif kind == "struct_distinct_probe":
            alias = op_output_alias(op)
            if not alias or is_reserved_output_name(alias):
                continue
            repaired.append({"op": "struct_distinct_probe", "as": alias})
            available = {alias}
            col_types = {alias: "bool"}
            numeric = set()
            strings = set()
            order_pending = False
            pending_order_columns = set()
        elif kind == "bit_compare_probe":
            alias = op_output_alias(op)
            if not alias or is_reserved_output_name(alias):
                continue
            repaired.append({"op": "bit_compare_probe", "as": alias})
            available = {alias}
            col_types = {alias: "bool"}
            numeric = set()
            strings = set()
            order_pending = False
            pending_order_columns = set()
        elif kind == "round_even_probe":
            alias = op_output_alias(op)
            if not alias or is_reserved_output_name(alias):
                continue
            repaired.append({"op": "round_even_probe", "as": alias})
            available = {alias}
            col_types = {alias: "bool"}
            numeric = set()
            strings = set()
            order_pending = False
            pending_order_columns = set()
        elif kind == "float_literal_precision_probe":
            alias = op_output_alias(op)
            literal = op_literal(op)
            if not alias or is_reserved_output_name(alias) or not _valid_float_literal_text(literal):
                continue
            repaired.append({"op": "float_literal_precision_probe", "as": alias, "literal": literal})
            available = {alias}
            col_types = {alias: "bool"}
            numeric = set()
            strings = set()
            order_pending = False
            pending_order_columns = set()
        elif kind == "timestamp_precision_filter_probe":
            alias = op_output_alias(op)
            if not alias or is_reserved_output_name(alias):
                continue
            repaired.append({"op": "timestamp_precision_filter_probe", "as": alias})
            available = {alias}
            col_types = {alias: "bool"}
            numeric = set()
            strings = set()
            order_pending = False
            pending_order_columns = set()
        elif kind == "series_rtruediv_probe":
            alias = op_output_alias(op)
            if not alias or is_reserved_output_name(alias):
                continue
            repaired.append({"op": "series_rtruediv_probe", "as": alias})
            available = {alias}
            col_types = {alias: "bool"}
            numeric = set()
            strings = set()
            order_pending = False
            pending_order_columns = set()
        elif kind == "uint64_isin_probe":
            alias = op_output_alias(op)
            if not alias or is_reserved_output_name(alias):
                continue
            repaired.append({"op": "uint64_isin_probe", "as": alias})
            available = {alias}
            col_types = {alias: "bool"}
            numeric = set()
            strings = set()
            order_pending = False
            pending_order_columns = set()
        elif kind == "tuple_anti_null_probe":
            alias = op_output_alias(op)
            if not alias or is_reserved_output_name(alias):
                continue
            repaired.append({"op": "tuple_anti_null_probe", "as": alias})
            available = {alias}
            col_types = {alias: "bool"}
            numeric = set()
            strings = set()
            order_pending = False
            pending_order_columns = set()
        elif kind == "setop_all_duplicate_probe":
            alias = op_output_alias(op)
            if not alias or is_reserved_output_name(alias):
                continue
            repaired.append({"op": "setop_all_duplicate_probe", "as": alias})
            available = {alias}
            col_types = {alias: "bool"}
            numeric = set()
            strings = set()
            order_pending = False
            pending_order_columns = set()
        elif kind == "json_predicate_order_probe":
            alias = op_output_alias(op)
            if not alias or is_reserved_output_name(alias):
                continue
            repaired.append({"op": "json_predicate_order_probe", "as": alias})
            available = {alias}
            col_types = {alias: "bool"}
            numeric = set()
            strings = set()
            order_pending = False
            pending_order_columns = set()
        elif kind == "sparse_mask_probe":
            alias = op_output_alias(op)
            if not alias or is_reserved_output_name(alias):
                continue
            repaired.append({"op": "sparse_mask_probe", "as": alias})
            available = {alias}
            col_types = {alias: "bool"}
            numeric = set()
            strings = set()
            order_pending = False
            pending_order_columns = set()
        elif kind == "float_wrap_probe":
            alias = op_output_alias(op)
            if not alias or is_reserved_output_name(alias):
                continue
            repaired.append({"op": "float_wrap_probe", "as": alias})
            available = {alias}
            col_types = {alias: "bool"}
            numeric = set()
            strings = set()
            order_pending = False
            pending_order_columns = set()
        elif kind == "index_bool_probe":
            alias = op_output_alias(op)
            if not alias or is_reserved_output_name(alias):
                continue
            repaired.append({"op": "index_bool_probe", "as": alias})
            available = {alias}
            col_types = {alias: "bool"}
            numeric = set()
            strings = set()
            order_pending = False
            pending_order_columns = set()
        elif kind == "empty_literal_groupby_probe":
            alias = op_output_alias(op)
            if not alias or is_reserved_output_name(alias):
                continue
            repaired.append({"op": "empty_literal_groupby_probe", "as": alias})
            available = {alias}
            col_types = {alias: "bool"}
            numeric = set()
            strings = set()
            order_pending = False
            pending_order_columns = set()
        elif kind == "arrow_string_eq_sum_probe":
            alias = op_output_alias(op)
            if not alias or is_reserved_output_name(alias):
                continue
            repaired.append({"op": "arrow_string_eq_sum_probe", "as": alias})
            available = {alias}
            col_types = {alias: "bool"}
            numeric = set()
            strings = set()
            order_pending = False
            pending_order_columns = set()
        elif kind == "arrow_timestamp_loc_slice_probe":
            alias = op_output_alias(op)
            if not alias or is_reserved_output_name(alias):
                continue
            repaired.append({"op": "arrow_timestamp_loc_slice_probe", "as": alias})
            available = {alias}
            col_types = {alias: "bool"}
            numeric = set()
            strings = set()
            order_pending = False
            pending_order_columns = set()
        elif kind == "arrow_timestamp_index_attr_probe":
            alias = op_output_alias(op)
            if not alias or is_reserved_output_name(alias):
                continue
            repaired.append({"op": "arrow_timestamp_index_attr_probe", "as": alias})
            available = {alias}
            col_types = {alias: "bool"}
            numeric = set()
            strings = set()
            order_pending = False
            pending_order_columns = set()
        elif kind == "eval_inplace_alias_probe":
            alias = op_output_alias(op)
            if not alias or is_reserved_output_name(alias):
                continue
            repaired.append({"op": "eval_inplace_alias_probe", "as": alias})
            available = {alias}
            col_types = {alias: "bool"}
            numeric = set()
            strings = set()
            order_pending = False
            pending_order_columns = set()
        elif kind == "bool_reduction_skipna_probe":
            alias = op_output_alias(op)
            if not alias or is_reserved_output_name(alias):
                continue
            repaired.append({"op": "bool_reduction_skipna_probe", "as": alias})
            available = {alias}
            col_types = {alias: "bool"}
            numeric = set()
            strings = set()
            order_pending = False
            pending_order_columns = set()
        elif kind == "dataset_isin_all_match_probe":
            alias = op_output_alias(op)
            if not alias or is_reserved_output_name(alias):
                continue
            repaired.append({"op": "dataset_isin_all_match_probe", "as": alias})
            available = {alias}
            col_types = {alias: "bool"}
            numeric = set()
            strings = set()
            order_pending = False
            pending_order_columns = set()
        elif kind == "run_end_null_compute_probe":
            alias = op_output_alias(op)
            if not alias or is_reserved_output_name(alias):
                continue
            repaired.append({"op": "run_end_null_compute_probe", "as": alias})
            available = {alias}
            col_types = {alias: "bool"}
            numeric = set()
            strings = set()
            order_pending = False
            pending_order_columns = set()
        elif kind == "large_string_partition_probe":
            alias = op_output_alias(op)
            if not alias or is_reserved_output_name(alias):
                continue
            repaired.append({"op": "large_string_partition_probe", "as": alias})
            available = {alias}
            col_types = {alias: "bool"}
            numeric = set()
            strings = set()
            order_pending = False
            pending_order_columns = set()
        elif kind == "hash_pivot_wider_probe":
            alias = op_output_alias(op)
            if not alias or is_reserved_output_name(alias):
                continue
            repaired.append({"op": "hash_pivot_wider_probe", "as": alias})
            available = {alias}
            col_types = {alias: "bool"}
            numeric = set()
            strings = set()
            order_pending = False
            pending_order_columns = set()
        elif kind == "list_flatten_parent_indices_probe":
            alias = op_output_alias(op)
            if not alias or is_reserved_output_name(alias):
                continue
            repaired.append({"op": "list_flatten_parent_indices_probe", "as": alias})
            available = {alias}
            col_types = {alias: "bool"}
            numeric = set()
            strings = set()
            order_pending = False
            pending_order_columns = set()
        elif kind == "rolling_mean_by_null_count_probe":
            alias = op_output_alias(op)
            if not alias or is_reserved_output_name(alias):
                continue
            repaired.append({"op": "rolling_mean_by_null_count_probe", "as": alias})
            available = {alias}
            col_types = {alias: "bool"}
            numeric = set()
            strings = set()
            order_pending = False
            pending_order_columns = set()
        elif kind == "csv_long_numeric_roundtrip_probe":
            alias = op_output_alias(op)
            if not alias or is_reserved_output_name(alias):
                continue
            values = [str(value).strip() for value in op_values(op)]
            values = [value for value in values if value and value.isdigit()]
            repaired_op = {"op": "csv_long_numeric_roundtrip_probe", "as": alias}
            if values:
                repaired_op["values"] = values
            repaired.append(repaired_op)
            available = {alias}
            col_types = {alias: "bool"}
            numeric = set()
            strings = set()
            order_pending = False
            pending_order_columns = set()
        elif kind == "select":
            cols = unique_preserve_order([c for c in op_columns(op) if c in available])
            if not cols:
                continue
            repaired.append({"op": "select", "columns": cols})
            available = set(cols)
            numeric &= available
            strings &= available
        elif kind == "distinct":
            cols = unique_preserve_order([c for c in op_columns(op) if c in available])
            if not cols:
                continue
            repaired.append({"op": "distinct", "columns": cols})
            available = set(cols)
            col_types = {column: col_types[column] for column in cols}
            numeric &= available
            strings &= available
            order_pending = False
            pending_order_columns = set()
        elif kind == "fill_null":
            column = op_column(op)
            if column not in available or column not in col_types:
                continue
            repaired.append({"op": "fill_null", "column": column, "value": op_value(op)})
            if column in pending_order_columns:
                order_pending = False
                pending_order_columns = set()
        elif kind == "coalesce":
            columns = unique_preserve_order([column for column in op_columns(op) if column in available])
            if len(columns) < 2:
                continue
            output_type = col_types.get(columns[0])
            if output_type is None or any(col_types.get(column) != output_type for column in columns):
                continue
            fallback = coalesce_fallback(op)
            if "fallback" in op and fallback is not None and not _filter_literal_is_valid(output_type, "==", fallback):
                continue
            alias = op_output_alias(op)
            if not alias or is_reserved_output_name(alias):
                continue
            repaired_op: dict[str, Any] = {"op": "coalesce", "columns": columns, "as": alias}
            if "fallback" in op:
                repaired_op["fallback"] = fallback
            repaired.append(repaired_op)
            available.add(alias)
            col_types[alias] = output_type
            if output_type in {"int", "float"}:
                numeric.add(alias)
            else:
                numeric.discard(alias)
            if output_type == "str":
                strings.add(alias)
            else:
                strings.discard(alias)
            if alias in pending_order_columns:
                order_pending = False
                pending_order_columns = set()
        elif kind == "case_when":
            predicate_column = condition_column(op)
            if predicate_column not in available:
                continue
            column_type = col_types.get(predicate_column, "")
            if not _filter_literal_is_valid(column_type, condition_cmp(op), condition_value(op)):
                continue
            output_type = _case_when_output_type(case_then_value(op), case_else_value(op))
            if output_type is None:
                continue
            alias = op_output_alias(op)
            if not alias or is_reserved_output_name(alias):
                continue
            repaired.append(
                {
                    "op": "case_when",
                    "as": alias,
                    "condition": {
                        "column": predicate_column,
                        "cmp": condition_cmp(op),
                        "value": condition_value(op),
                    },
                    "then": case_then_value(op),
                    "else": case_else_value(op),
                }
            )
            available.add(alias)
            col_types[alias] = output_type
            if output_type in {"int", "float"}:
                numeric.add(alias)
            else:
                numeric.discard(alias)
            if output_type == "str":
                strings.add(alias)
            else:
                strings.discard(alias)
            if alias in pending_order_columns:
                order_pending = False
                pending_order_columns = set()
        elif kind == "sort":
            repaired_op = normalized_sort_op(op, available)
            if repaired_op is None:
                continue
            ctx = ValidationContext(tables=table_by_name, state=current_state(), errors=[])
            if not validate_core_operation(ctx, repaired_op, len(repaired)):
                continue
            repaired.append(repaired_op)
            try:
                pending_order_columns = {key.column for key in normalize_sort_keys(repaired_op)}
            except ValueError:
                pending_order_columns = set()
            order_pending = bool(pending_order_columns)
        elif kind == "limit":
            if order_pending:
                repaired.append(op)
            break
        elif kind == "offset":
            if order_pending:
                repaired.append({"op": "offset", "n": max(0, op_n(op))})
        elif kind == "mutate":
            expr = expr_payload(op)
            out_type = _mutate_output_type(expr, available, numeric, strings, col_types)
            if out_type is None:
                continue
            column = op_column(op)
            repaired.append(op)
            available.add(column)
            col_types[column] = out_type
            if out_type in {"int", "float"}:
                numeric.add(column)
            if out_type == "str":
                strings.add(column)
            if column in pending_order_columns:
                order_pending = False
                pending_order_columns = set()
        elif kind == "groupby":
            keys = unique_preserve_order([k for k in groupby_keys(op) if k in available])
            aggs = [
                a
                for a in aggregate_specs(op)
                if aggregate_column(a) in available
                and _aggregate_accepts_type(
                    col_types.get(aggregate_column(a), "derived"),
                    aggregate_func(a),
                    aggregate_column(a) in numeric,
                )
            ]
            unique_aggs: list[Any] = []
            seen_aliases: set[str] = set()
            for agg in aggs:
                alias = aggregate_alias(agg)
                if not alias or alias in seen_aliases or alias in keys or is_reserved_output_name(alias):
                    continue
                seen_aliases.add(alias)
                unique_aggs.append(agg)
            aggs = unique_aggs
            if not keys or not aggs:
                continue
            repaired_op = dict(op.to_dict()) if hasattr(op, "to_dict") else dict(op)
            repaired_op["keys"] = keys
            repaired_op["aggs"] = [dict(agg.to_dict()) if hasattr(agg, "to_dict") else dict(agg) for agg in aggs]
            repaired.append(repaired_op)
            available = set(keys) | {aggregate_alias(agg) for agg in aggs}
            numeric = {k for k in keys if col_types.get(k) in {"int", "float"}}
            strings = {k for k in keys if col_types.get(k) == "str"}
            for agg in aggs:
                output_type = _aggregate_output_type(
                    col_types.get(aggregate_column(agg), "float"),
                    aggregate_func(agg),
                )
                alias = aggregate_alias(agg)
                col_types[alias] = output_type
                if output_type in {"int", "float"}:
                    numeric.add(alias)
            order_pending = False
            pending_order_columns = set()
        elif kind == "aggregate":
            aggs = [
                a
                for a in aggregate_specs(op)
                if aggregate_column(a) in available
                and _aggregate_accepts_type(
                    col_types.get(aggregate_column(a), "derived"),
                    aggregate_func(a),
                    aggregate_column(a) in numeric,
                )
            ]
            unique_aggs = []
            seen_aliases: set[str] = set()
            for agg in aggs:
                alias = aggregate_alias(agg)
                if not alias or alias in seen_aliases or is_reserved_output_name(alias):
                    continue
                seen_aliases.add(alias)
                unique_aggs.append(agg)
            if not unique_aggs:
                continue
            repaired_op = dict(op.to_dict()) if hasattr(op, "to_dict") else dict(op)
            repaired_op["aggs"] = [
                dict(agg.to_dict()) if hasattr(agg, "to_dict") else dict(agg) for agg in unique_aggs
            ]
            repaired.append(repaired_op)
            available = {aggregate_alias(agg) for agg in unique_aggs}
            numeric = set()
            strings = set()
            for agg in unique_aggs:
                output_type = _aggregate_output_type(
                    col_types.get(aggregate_column(agg), "float"),
                    aggregate_func(agg),
                )
                alias = aggregate_alias(agg)
                col_types[alias] = output_type
                if output_type in {"int", "float"}:
                    numeric.add(alias)
            order_pending = False
            pending_order_columns = set()
    return repaired


def _random_sort_op(rnd: random.Random, columns: list[str], *, allow_mixed: bool = False) -> dict[str, Any]:
    if allow_mixed and len(columns) > 1 and rnd.random() < 0.35:
        return {
            "op": "sort",
            "keys": [
                {
                    "column": column,
                    "ascending": rnd.choice([True, False]),
                    "nulls": rnd.choice(["first", "last"]),
                }
                for column in columns
            ],
        }
    return {"op": "sort", "columns": columns, "ascending": rnd.choice([True, False])}


def _dedupe_sort_keys(keys: list[SortKey]) -> list[SortKey]:
    seen = set()
    out = []
    for key in keys:
        if key.column in seen:
            continue
        seen.add(key.column)
        out.append(key)
    return out


def _mutate_output_type(
    expr: dict[str, Any],
    available: set[str],
    numeric: set[str],
    strings: set[str],
    col_types: dict[str, str],
) -> str | None:
    src = coerce_expression(expr).source
    if src not in available:
        return None
    return expr_output_type(expr, col_types)


def _case_when_output_type(then_value: Any, else_value: Any) -> str | None:
    return case_when_output_type(then_value, else_value)


def _literal_output_type(value: Any) -> str | None:
    return literal_output_type(value)


def _filter_literal_is_valid(column_type: str, comparator: Any, value: Any) -> bool:
    if not filter_comparator_supports_type(column_type, comparator):
        return False
    parsed = parse_filter_comparator(comparator)
    if parsed is not None and parsed.base in {"in_set", "not_in_set"}:
        if not isinstance(value, list) or not value or any(item is None for item in value):
            return False
        return all(_filter_literal_is_valid(column_type, "==", item) for item in value)
    if parsed is not None and parsed.base == "range_closed":
        if not isinstance(value, list) or len(value) != 2 or any(item is None for item in value):
            return False
        if not all(_filter_literal_is_valid(column_type, "==", item) for item in value):
            return False
        return value[0] <= value[1]
    if parsed is not None and parsed.base in {"str_contains", "str_starts_with", "str_ends_with"}:
        return column_type == "str" and isinstance(value, str) and value != ""
    if comparator in {"is_null", "is_not_null"}:
        return value is None
    if parsed is not None and parsed.base == "bool_predicate":
        return value is None
    if value is None:
        return True
    if column_type == "str":
        return isinstance(value, str)
    if column_type == "bool":
        return isinstance(value, bool)
    if column_type in {"int", "float"}:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return False


def _valid_quantile_probe_values(values: Any, quantiles: Any) -> bool:
    if not isinstance(values, list) or not isinstance(quantiles, list):
        return False
    if len(values) < 2 or len(quantiles) < 2:
        return False
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in values):
        return False
    return all(
        isinstance(quantile, (int, float))
        and not isinstance(quantile, bool)
        and 0.0 <= float(quantile) <= 1.0
        for quantile in quantiles
    )


def _valid_float_literal_text(value: Any) -> bool:
    text = str(value or "").strip()
    if not text or not any(ch.isdigit() for ch in text):
        return False
    allowed = set("0123456789+-.eE")
    if any(ch not in allowed for ch in text):
        return False
    try:
        float(text)
    except ValueError:
        return False
    return True


_TYPE_AWARE_PROFILE_GENERATORS: dict[str, Callable[[int], Case]] | None = None
_GENERIC_CASE_SUFFIXES = {
    "discovery": "-discovery",
    "discovery_fresh": "-discovery-fresh",
    "discovery_no_groupby": "-discovery-no-groupby",
    "common_api_workflow": "-common-api-workflow",
    "issue_focus": "-issue-focus",
    "deep_probe_rotation": "-deep-probe-rotation",
}


def _type_aware_profile_generators() -> dict[str, Callable[[int], Case]]:
    global _TYPE_AWARE_PROFILE_GENERATORS
    if _TYPE_AWARE_PROFILE_GENERATORS is None:
        _TYPE_AWARE_PROFILE_GENERATORS = {
            "null_groupby_topk": generate_null_groupby_topk_case,
            "null_agg_topk": generate_null_agg_topk_case,
            "filter_null_agg_topk": generate_filter_null_agg_topk_case,
            "join_null_agg_topk": generate_join_null_agg_topk_case,
            "join_null_key_topk": generate_join_null_key_topk_case,
            "wide_offset_topk": generate_wide_offset_topk_case,
            "empty_filter_groupby": generate_empty_filter_groupby_case,
            "join_filter_groupby": generate_join_filter_groupby_case,
            "join_null_truth_filter": generate_join_null_truth_filter_case,
            "join_groupby_stress": generate_join_groupby_stress_case,
            "storage_offset": generate_storage_offset_case,
            "float_group_key": generate_float_group_key_case,
            "join_null_sort": generate_join_null_sort_case,
            "ordered_groupby_sort": generate_ordered_groupby_sort_case,
            "topk_resort": generate_topk_resort_case,
            "join_ordered_agg_topk": generate_join_ordered_agg_topk_case,
            "global_null_aggregate": generate_global_null_aggregate_case,
            "string_count_groupby": generate_string_count_groupby_case,
            "unique_count_groupby": generate_unique_count_groupby_case,
            "bool_null_groupby_agg": generate_bool_null_groupby_agg_case,
            "large_int_filter_groupby": generate_large_int_filter_groupby_case,
            "set_membership_filter": generate_set_membership_filter_case,
            "pyarrow_groupby_filter_cast_membership": generate_pyarrow_groupby_filter_cast_membership_case,
            "null_predicate_filter": generate_null_predicate_filter_case,
            "boolean_predicate_filter": generate_boolean_predicate_filter_case,
            "post_topk_range_filter": generate_post_topk_range_filter_case,
            "tuple_absence_filter": generate_tuple_absence_filter_case,
            "row_value_absence_filter": generate_row_value_absence_filter_case,
            "running_sum_precision": generate_running_sum_precision_case,
            "partitioned_running_sum": generate_partitioned_running_sum_case,
            "path_basename_keyed_pick": generate_path_basename_keyed_pick_case,
            "sortedness_null_placement": generate_sortedness_null_placement_case,
            "simple_case_random_subject": generate_simple_case_random_subject_case,
            "group_quantile_key_probe": generate_group_quantile_key_probe_case,
            "scalar_subquery_double_parentheses": generate_scalar_subquery_double_parentheses_case,
            "window_avg_rows_frame": generate_window_avg_rows_frame_case,
            "struct_distinct_unnest": generate_struct_distinct_unnest_case,
            "bit_compare_unequal_length": generate_bit_compare_unequal_length_case,
            "round_even_float_scale": generate_round_even_float_scale_case,
            "duckdb_float_literal_precision": generate_duckdb_float_literal_precision_case,
            "polars_timestamp_precision_filter": generate_polars_timestamp_precision_filter_case,
            "series_rtruediv_operand_order": generate_series_rtruediv_operand_order_case,
            "polars_reverse_division_columns": generate_polars_reverse_division_columns_case,
            "pandas_uint64_isin_precision": generate_pandas_uint64_isin_precision_case,
            "duckdb_tuple_anti_null_semantics": generate_duckdb_tuple_anti_null_semantics_case,
            "datafusion_setop_all_duplicate_count": generate_datafusion_setop_all_duplicate_count_case,
            "duckdb_json_predicate_order_semantics": generate_duckdb_json_predicate_order_semantics_case,
            "pandas_sparse_array_mask_semantics": generate_pandas_sparse_array_mask_semantics_case,
            "polars_float_wrap_numerical_semantics": generate_polars_float_wrap_numerical_semantics_case,
            "pandas_index_bool_result_type": generate_pandas_index_bool_result_type_case,
            "polars_empty_literal_groupby_semantics": generate_polars_empty_literal_groupby_semantics_case,
            "pandas_arrow_string_eq_sum_semantics": generate_pandas_arrow_string_eq_sum_semantics_case,
            "pandas_arrow_timestamp_loc_slice_semantics": generate_pandas_arrow_timestamp_loc_slice_semantics_case,
            "pandas_arrow_timestamp_index_attr_semantics": generate_pandas_arrow_timestamp_index_attr_semantics_case,
            "pandas_eval_inplace_aliasing_semantics": generate_pandas_eval_inplace_aliasing_semantics_case,
            "pandas_bool_reduction_skipna_semantics": generate_pandas_bool_reduction_skipna_semantics_case,
            "pyarrow_dataset_isin_all_match_semantics": generate_pyarrow_dataset_isin_all_match_semantics_case,
            "pyarrow_run_end_null_compute_semantics": generate_pyarrow_run_end_null_compute_semantics_case,
            "pyarrow_large_string_partition_schema_semantics": generate_pyarrow_large_string_partition_schema_semantics_case,
            "pyarrow_hash_pivot_wider_order_semantics": generate_pyarrow_hash_pivot_wider_order_semantics_case,
            "pyarrow_list_flatten_parent_indices_semantics": generate_pyarrow_list_flatten_parent_indices_semantics_case,
            "polars_rolling_mean_by_null_count_semantics": generate_polars_rolling_mean_by_null_count_semantics_case,
            "csv_long_numeric_roundtrip": generate_csv_long_numeric_roundtrip_case,
            "workflow": generate_workflow_case,
            "common_api_workflow": generate_common_api_workflow_case,
            "deep_probe_rotation": _deep_probe_rotation_case,
            "issue_focus": _issue_focus_case,
        }
    return _TYPE_AWARE_PROFILE_GENERATORS


def _profile_dispatch_case(seed: int, profile: str, *, type_aware: bool) -> Case | None:
    if not type_aware:
        return None
    if profile == "discovery":
        return _discovery_issue_inspired_case(seed)
    if profile == "discovery_no_groupby":
        mixed = _discovery_no_groupby_issue_inspired_case(seed)
        if mixed is not None:
            return mixed
    generator = _type_aware_profile_generators().get(profile)
    return generator(seed) if generator is not None else None


def _generic_case_suffix(profile: str) -> str:
    return _GENERIC_CASE_SUFFIXES.get(profile, "")


def generate_case(seed: int, type_aware: bool = True, profile: GeneratorProfile = "common") -> Case:
    dispatched = _profile_dispatch_case(seed, profile, type_aware=type_aware)
    if dispatched is not None:
        return dispatched
    discovery_profile = _is_discovery_profile(profile)
    table = generate_table(
        seed,
        name="t0",
        min_rows=8 if discovery_profile else 0,
        max_rows=30 if discovery_profile else 20,
        profile=profile,
    )
    rnd = random.Random(seed * 15485863 + 11)
    join_probability = 0.85 if discovery_profile else 0.4
    extra_tables = [generate_join_table(seed, profile=profile)] if type_aware and rnd.random() < join_probability else []
    if discovery_profile and extra_tables:
        extra_tables = [_cover_join_table_keys(table, extra_tables[0])]
    program = generate_program(
        seed,
        table,
        max_ops=8 if discovery_profile else 6,
        type_aware=type_aware,
        extra_tables=extra_tables,
        profile=profile,
    )
    suffix = _generic_case_suffix(profile)
    return Case(case_id=f"case-{seed:08d}{suffix}", seed=seed, tables=[table] + extra_tables, program=program)


DEEP_PROBE_ROTATION_PROFILES = (
    "simple_case_random_subject",
    "group_quantile_key_probe",
    "scalar_subquery_double_parentheses",
    "window_avg_rows_frame",
    "struct_distinct_unnest",
    "bit_compare_unequal_length",
    "round_even_float_scale",
    "series_rtruediv_operand_order",
    "pandas_uint64_isin_precision",
    "duckdb_tuple_anti_null_semantics",
    "datafusion_setop_all_duplicate_count",
    "duckdb_json_predicate_order_semantics",
    "pandas_sparse_array_mask_semantics",
    "polars_float_wrap_numerical_semantics",
    "pandas_index_bool_result_type",
    "polars_empty_literal_groupby_semantics",
    "pandas_arrow_string_eq_sum_semantics",
    "pandas_arrow_timestamp_loc_slice_semantics",
    "pandas_arrow_timestamp_index_attr_semantics",
    "pandas_eval_inplace_aliasing_semantics",
    "pandas_bool_reduction_skipna_semantics",
    "pyarrow_dataset_isin_all_match_semantics",
    "pyarrow_run_end_null_compute_semantics",
    "pyarrow_large_string_partition_schema_semantics",
    "pyarrow_hash_pivot_wider_order_semantics",
    "pyarrow_list_flatten_parent_indices_semantics",
    "polars_rolling_mean_by_null_count_semantics",
    "pyarrow_groupby_filter_cast_membership",
    "partitioned_running_sum",
    "path_basename_keyed_pick",
    "csv_long_numeric_roundtrip",
)


def _deep_probe_rotation_case(seed: int) -> Case:
    profile = DEEP_PROBE_ROTATION_PROFILES[seed % len(DEEP_PROBE_ROTATION_PROFILES)]
    case = generate_case(seed, profile=profile)  # type: ignore[arg-type]
    return _as_discovery_mixed_case(
        case,
        seed,
        profile,
        generator_profile="deep_probe_rotation",
    )


ISSUE_FOCUS_MIXED_PROFILES = (
    "null_groupby_topk",
    "null_agg_topk",
    "filter_null_agg_topk",
    "join_null_agg_topk",
    "empty_filter_groupby",
    "join_filter_groupby",
    "join_null_sort",
    "ordered_groupby_sort",
    "topk_resort",
    "join_ordered_agg_topk",
    "global_null_aggregate",
    "string_count_groupby",
    "unique_count_groupby",
    "bool_null_groupby_agg",
    "large_int_filter_groupby",
    "set_membership_filter",
    "pyarrow_groupby_filter_cast_membership",
    "null_predicate_filter",
    "row_value_absence_filter",
    "path_basename_keyed_pick",
    "polars_reverse_division_columns",
    "pandas_bool_reduction_skipna_semantics",
    "csv_long_numeric_roundtrip",
)


def _issue_focus_case(seed: int) -> Case:
    issue_case = _discovery_issue_inspired_case(seed)
    if issue_case is not None:
        mixed_profile = str(issue_case.metadata.get("mixed_generator_profile", "issue_inspired"))
        return _as_discovery_mixed_case(
            issue_case,
            seed,
            mixed_profile,
            generator_profile="issue_focus",
        )
    start_index = seed % len(ISSUE_FOCUS_MIXED_PROFILES)
    for offset in range(len(ISSUE_FOCUS_MIXED_PROFILES)):
        mixed_profile = ISSUE_FOCUS_MIXED_PROFILES[(start_index + offset) % len(ISSUE_FOCUS_MIXED_PROFILES)]
        case = generate_case(seed + offset, profile=mixed_profile)  # type: ignore[arg-type]
        return _as_discovery_mixed_case(
            case,
            seed,
            mixed_profile,
            generator_profile="issue_focus",
        )
    fallback = generate_case(seed, profile="discovery_fresh")
    return _as_discovery_mixed_case(
        fallback,
        seed,
        "discovery_fresh",
        generator_profile="issue_focus",
    )


def _discovery_issue_inspired_case(seed: int) -> Case | None:
    if seed % 211 == 111:
        return _as_discovery_mixed_case(
            generate_duckdb_float_literal_precision_case(seed),
            seed,
            "duckdb_float_literal_precision",
        )
    if seed % 227 == 114:
        return _as_discovery_mixed_case(
            generate_polars_timestamp_precision_filter_case(seed),
            seed,
            "polars_timestamp_precision_filter",
        )
    selector = seed % 60
    if selector == 2:
        return _as_discovery_mixed_case(generate_join_null_truth_filter_case(seed), seed, "join_null_truth_filter")
    if selector == 5:
        return _as_discovery_mixed_case(generate_global_null_aggregate_case(seed), seed, "global_null_aggregate")
    if selector == 11:
        return _as_discovery_mixed_case(generate_empty_filter_groupby_case(seed), seed, "empty_filter_groupby")
    if selector == 20:
        return _as_discovery_mixed_case(generate_wide_offset_topk_case(seed), seed, "wide_offset_topk")
    if selector == 31:
        return _as_discovery_mixed_case(generate_ordered_groupby_sort_case(seed), seed, "ordered_groupby_sort")
    if selector == 40:
        return _as_discovery_mixed_case(generate_join_null_key_topk_case(seed), seed, "join_null_key_topk")
    if selector == 47:
        return _as_discovery_mixed_case(generate_string_count_groupby_case(seed), seed, "string_count_groupby")
    if selector == 50:
        return _as_discovery_mixed_case(generate_set_membership_filter_case(seed), seed, "set_membership_filter")
    if selector == 51:
        return _as_discovery_mixed_case(generate_null_predicate_filter_case(seed), seed, "null_predicate_filter")
    if selector == 52:
        return _as_discovery_mixed_case(
            generate_polars_reverse_division_columns_case(seed),
            seed,
            "polars_reverse_division_columns",
        )
    if selector == 54:
        return _as_discovery_mixed_case(generate_boolean_predicate_filter_case(seed), seed, "boolean_predicate_filter")
    if selector == 55:
        return _as_discovery_mixed_case(generate_post_topk_range_filter_case(seed), seed, "post_topk_range_filter")
    if selector == 56:
        return _as_discovery_mixed_case(generate_unique_count_groupby_case(seed), seed, "unique_count_groupby")
    if selector == 57:
        return _as_discovery_mixed_case(generate_tuple_absence_filter_case(seed), seed, "tuple_absence_filter")
    if selector == 53:
        return _as_discovery_mixed_case(generate_topk_resort_case(seed), seed, "topk_resort")
    if selector == 58:
        return _as_discovery_mixed_case(generate_join_ordered_agg_topk_case(seed), seed, "join_ordered_agg_topk")
    if selector == 59:
        return _as_discovery_mixed_case(generate_running_sum_precision_case(seed), seed, "running_sum_precision")
    if seed % 71 == 60:
        return _as_discovery_mixed_case(generate_sortedness_null_placement_case(seed), seed, "sortedness_null_placement")
    if seed % 73 == 61:
        return _as_discovery_mixed_case(generate_simple_case_random_subject_case(seed), seed, "simple_case_random_subject")
    if seed % 79 == 63:
        return _as_discovery_mixed_case(generate_group_quantile_key_probe_case(seed), seed, "group_quantile_key_probe")
    if seed % 83 == 64:
        return _as_discovery_mixed_case(
            generate_scalar_subquery_double_parentheses_case(seed),
            seed,
            "scalar_subquery_double_parentheses",
        )
    if seed % 89 == 66:
        return _as_discovery_mixed_case(generate_window_avg_rows_frame_case(seed), seed, "window_avg_rows_frame")
    if seed % 97 == 67:
        return _as_discovery_mixed_case(generate_struct_distinct_unnest_case(seed), seed, "struct_distinct_unnest")
    if seed % 101 == 68:
        return _as_discovery_mixed_case(
            generate_bit_compare_unequal_length_case(seed),
            seed,
            "bit_compare_unequal_length",
        )
    if seed % 103 == 69:
        return _as_discovery_mixed_case(generate_round_even_float_scale_case(seed), seed, "round_even_float_scale")
    if seed % 107 == 70:
        return _as_discovery_mixed_case(
            generate_series_rtruediv_operand_order_case(seed),
            seed,
            "series_rtruediv_operand_order",
        )
    if seed % 109 == 72:
        return _as_discovery_mixed_case(
            generate_pandas_uint64_isin_precision_case(seed),
            seed,
            "pandas_uint64_isin_precision",
        )
    if seed % 113 == 73:
        return _as_discovery_mixed_case(
            generate_duckdb_tuple_anti_null_semantics_case(seed),
            seed,
            "duckdb_tuple_anti_null_semantics",
        )
    if seed % 199 == 102:
        return _as_discovery_mixed_case(
            generate_datafusion_setop_all_duplicate_count_case(seed),
            seed,
            "datafusion_setop_all_duplicate_count",
        )
    if seed % 181 == 105:
        return _as_discovery_mixed_case(
            generate_duckdb_json_predicate_order_semantics_case(seed),
            seed,
            "duckdb_json_predicate_order_semantics",
        )
    if seed % 181 == 101:
        return _as_discovery_mixed_case(
            generate_row_value_absence_filter_case(seed),
            seed,
            "row_value_absence_filter",
        )
    if seed % 127 == 74:
        return _as_discovery_mixed_case(
            generate_pandas_sparse_array_mask_semantics_case(seed),
            seed,
            "pandas_sparse_array_mask_semantics",
        )
    if seed % 131 == 75:
        return _as_discovery_mixed_case(
            generate_polars_float_wrap_numerical_semantics_case(seed),
            seed,
            "polars_float_wrap_numerical_semantics",
        )
    if seed % 137 == 76:
        return _as_discovery_mixed_case(
            generate_pandas_index_bool_result_type_case(seed),
            seed,
            "pandas_index_bool_result_type",
        )
    if seed % 139 == 77:
        return _as_discovery_mixed_case(
            generate_polars_empty_literal_groupby_semantics_case(seed),
            seed,
            "polars_empty_literal_groupby_semantics",
        )
    if seed % 149 == 78:
        return _as_discovery_mixed_case(
            generate_pandas_arrow_string_eq_sum_semantics_case(seed),
            seed,
            "pandas_arrow_string_eq_sum_semantics",
        )
    if seed % 151 == 79:
        return _as_discovery_mixed_case(
            generate_pandas_arrow_timestamp_loc_slice_semantics_case(seed),
            seed,
            "pandas_arrow_timestamp_loc_slice_semantics",
        )
    if seed % 157 == 81:
        return _as_discovery_mixed_case(
            generate_pandas_arrow_timestamp_index_attr_semantics_case(seed),
            seed,
            "pandas_arrow_timestamp_index_attr_semantics",
        )
    if seed % 181 == 108:
        return _as_discovery_mixed_case(
            generate_pandas_eval_inplace_aliasing_semantics_case(seed),
            seed,
            "pandas_eval_inplace_aliasing_semantics",
        )
    if seed % 193 == 110:
        return _as_discovery_mixed_case(
            generate_pandas_bool_reduction_skipna_semantics_case(seed),
            seed,
            "pandas_bool_reduction_skipna_semantics",
        )
    if seed % 163 == 82:
        return _as_discovery_mixed_case(
            generate_pyarrow_dataset_isin_all_match_semantics_case(seed),
            seed,
            "pyarrow_dataset_isin_all_match_semantics",
        )
    if seed % 197 == 112:
        return _as_discovery_mixed_case(
            generate_pyarrow_run_end_null_compute_semantics_case(seed),
            seed,
            "pyarrow_run_end_null_compute_semantics",
        )
    if seed % 173 == 104:
        return _as_discovery_mixed_case(
            generate_pyarrow_large_string_partition_schema_semantics_case(seed),
            seed,
            "pyarrow_large_string_partition_schema_semantics",
        )
    if seed % 191 == 106:
        return _as_discovery_mixed_case(
            generate_pyarrow_hash_pivot_wider_order_semantics_case(seed),
            seed,
            "pyarrow_hash_pivot_wider_order_semantics",
        )
    if seed % 233 == 8:
        return _as_discovery_mixed_case(
            generate_pyarrow_list_flatten_parent_indices_semantics_case(seed),
            seed,
            "pyarrow_list_flatten_parent_indices_semantics",
        )
    if seed % 167 == 83:
        return _as_discovery_mixed_case(
            generate_polars_rolling_mean_by_null_count_semantics_case(seed),
            seed,
            "polars_rolling_mean_by_null_count_semantics",
        )
    if seed % 197 == 109:
        return _as_discovery_mixed_case(
            generate_pyarrow_groupby_filter_cast_membership_case(seed),
            seed,
            "pyarrow_groupby_filter_cast_membership",
        )
    if seed % 223 == 103:
        return _as_discovery_mixed_case(generate_partitioned_running_sum_case(seed), seed, "partitioned_running_sum")
    if seed % 229 == 107:
        return _as_discovery_mixed_case(
            generate_path_basename_keyed_pick_case(seed),
            seed,
            "path_basename_keyed_pick",
        )
    if seed % 251 == 48:
        return _as_discovery_mixed_case(
            generate_csv_long_numeric_roundtrip_case(seed),
            seed,
            "csv_long_numeric_roundtrip",
        )
    return None


def _discovery_no_groupby_issue_inspired_case(seed: int) -> Case | None:
    if seed % 53 == 18:
        return _as_discovery_mixed_case(
            generate_datafusion_setop_all_duplicate_count_case(seed),
            seed,
            "datafusion_setop_all_duplicate_count",
            generator_profile="discovery_no_groupby",
        )
    return None


def _as_discovery_mixed_case(
    case: Case,
    seed: int,
    mixed_profile: str,
    *,
    generator_profile: str = "discovery",
) -> Case:
    metadata = dict(case.metadata)
    metadata["generator_profile"] = generator_profile
    metadata["mixed_generator_profile"] = mixed_profile
    profile_slug = generator_profile.replace("_", "-")
    return Case(
        case_id=f"case-{seed:08d}-{profile_slug}-{mixed_profile.replace('_', '-')}",
        seed=seed,
        tables=case.tables,
        program=case.program,
        metadata=metadata,
    )


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


def generate_null_groupby_topk_case(seed: int) -> Case:
    rnd = random.Random(seed * 32452843 + 19)
    string_values = [None, "", "a", "alpha", "space value", "delta"]
    row_count = rnd.randint(1, 8)
    rows = []
    for idx in range(row_count):
        value = string_values[idx % len(string_values)]
        if idx == 0:
            value = None
        rows.append(
            {
                "x": rnd.choice([-10, -1, 0, 1, 2, 10, None]),
                "s": value,
            }
        )
    table = TableData(
        "t0",
        [
            ColumnSpec("x", "int", nullable=True),
            ColumnSpec("s", "str", nullable=True),
        ],
        rows,
    )
    ascending = rnd.choice([True, False])
    limit = rnd.randint(1, max(1, row_count + 2))
    program = Program(
        f"prog-{seed:08d}-null-groupby-topk",
        seed,
        [
            {"op": "mutate", "column": "m_0", "expr": {"kind": "string_length", "source": "s"}},
            {
                "op": "groupby",
                "keys": ["m_0"],
                "aggs": [{"column": "x", "func": "count", "as": "count_x"}],
            },
            {"op": "select", "columns": ["m_0"]},
            {"op": "sort", "columns": ["m_0"], "ascending": ascending},
            {"op": "limit", "n": limit},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-null-groupby-topk",
        seed=seed,
        tables=[table],
        program=program,
    )


def generate_storage_offset_case(seed: int) -> Case:
    row_count = 300_000
    offset = 0 if seed % 2 == 0 else 200_000
    table = TableData(
        "t0",
        [ColumnSpec("id", "int", nullable=False)],
        [{"id": idx} for idx in range(row_count)],
    )
    program = Program(
        f"prog-{seed:08d}-storage-offset",
        seed,
        [
            {"op": "sort", "columns": ["id"], "ascending": True},
            {"op": "offset", "n": offset},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-storage-offset",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "storage_offset",
            "source_issue": "https://github.com/duckdb/duckdb/issues/22656",
            "row_count": row_count,
            "offset": offset,
        },
    )


def generate_null_agg_topk_case(seed: int) -> Case:
    rnd = random.Random(seed * 67867967 + 23)
    row_count = rnd.randint(1, 8)
    rows = []
    group_values = ["b", "c", "d", "e"]
    for idx in range(row_count):
        if idx == 0:
            rows.append({"g": "a", "x": None})
            continue
        group = group_values[idx % len(group_values)]
        value = rnd.choice([None, -10, -1, 0, 1, 2, 5, 10])
        rows.append({"g": group, "x": value})
    table = TableData(
        "t0",
        [
            ColumnSpec("g", "str", nullable=False),
            ColumnSpec("x", "int", nullable=True),
        ],
        rows,
    )
    agg_func = rnd.choice(["min", "max"])
    agg_alias = f"{agg_func}_x"
    ascending = agg_func == "min"
    program = Program(
        f"prog-{seed:08d}-null-agg-topk",
        seed,
        [
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [{"column": "x", "func": agg_func, "as": agg_alias}],
            },
            {"op": "select", "columns": [agg_alias]},
            {"op": "sort", "columns": [agg_alias], "ascending": ascending},
            {"op": "limit", "n": row_count + 2},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-null-agg-topk",
        seed=seed,
        tables=[table],
        program=program,
    )


def generate_filter_null_agg_topk_case(seed: int) -> Case:
    rnd = random.Random(seed * 701408733 + 29)
    rows = [
        {"id": 0, "g": "a", "x": None, "lane": "keep"},
        {"id": 1, "g": "a", "x": None, "lane": "keep"},
        {"id": 2, "g": "b", "x": rnd.choice([1, 2, 5]), "lane": "keep"},
        {"id": 3, "g": "b", "x": rnd.choice([None, 0, 3]), "lane": "keep"},
        {"id": 4, "g": "c", "x": rnd.choice([-1, 0, 7]), "lane": "keep"},
        {"id": 5, "g": "drop", "x": rnd.choice([None, 9]), "lane": "skip"},
    ]
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str", nullable=False),
            ColumnSpec("x", "int", nullable=True),
            ColumnSpec("lane", "str", nullable=False),
        ],
        rows,
    )
    agg_func = rnd.choice(["min", "max"])
    agg_alias = f"{agg_func}_x"
    ascending = agg_func == "min"
    program = Program(
        f"prog-{seed:08d}-filter-null-agg-topk",
        seed,
        [
            {"op": "filter", "column": "lane", "cmp": "!=", "value": "skip"},
            {"op": "mutate", "column": "m_0", "expr": {"kind": "add_const", "source": "x", "value": 0}},
            {"op": "select", "columns": ["g", "m_0"]},
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [{"column": "m_0", "func": agg_func, "as": agg_alias}],
            },
            {"op": "select", "columns": [agg_alias]},
            {"op": "sort", "columns": [agg_alias], "ascending": ascending},
            {"op": "limit", "n": 6},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-filter-null-agg-topk",
        seed=seed,
        tables=[table],
        program=program,
    )


def generate_join_null_agg_topk_case(seed: int) -> Case:
    rnd = random.Random(seed * 74649677 + 27)
    left = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str", nullable=False),
            ColumnSpec("x", "int", nullable=True),
        ],
        [
            {"id": 0, "g": "a", "x": 1},
            {"id": 1, "g": "a", "x": None},
            {"id": 2, "g": "b", "x": 2},
            {"id": 3, "g": "c", "x": -1},
            {"id": 4, "g": "d", "x": 0},
        ],
    )
    right = TableData(
        "t1",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("j", "int", nullable=True),
            ColumnSpec("tag", "str", nullable=True),
        ],
        [
            {"id": 0, "j": 10, "tag": "alpha"},
            {"id": 0, "j": None, "tag": "beta"},
            {"id": 2, "j": 5, "tag": "gamma"},
            {"id": 2, "j": 7, "tag": "delta"},
            {"id": 5, "j": 9, "tag": "orphan"},
        ],
    )
    agg_func = rnd.choice(["min", "max"])
    agg_alias = f"{agg_func}_j"
    ascending = agg_func == "min"
    program = Program(
        f"prog-{seed:08d}-join-null-agg-topk",
        seed,
        [
            {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"},
            {"op": "mutate", "column": "m_0", "expr": {"kind": "cast", "source": "j", "to": "float"}},
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [{"column": "m_0", "func": agg_func, "as": agg_alias}],
            },
            {"op": "select", "columns": [agg_alias]},
            {"op": "sort", "columns": [agg_alias], "ascending": ascending},
            {"op": "limit", "n": 8},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-join-null-agg-topk",
        seed=seed,
        tables=[left, right],
        program=program,
    )


def generate_join_null_key_topk_case(seed: int) -> Case:
    rnd = random.Random(seed * 82_589_933 + 43)
    left = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str", nullable=True),
            ColumnSpec("x", "int", nullable=True),
        ],
        [
            {"id": 1, "g": "a", "x": -1},
            {"id": 2, "g": "b", "x": 2},
            {"id": 3, "g": "c", "x": rnd.choice([0, 3, 5])},
        ],
    )
    right = TableData(
        "t1",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("j", "int", nullable=True),
            ColumnSpec("tag", "str", nullable=True),
        ],
        [
            {"id": 2, "j": rnd.choice([5, 7, 9]), "tag": "match"},
            {"id": 9, "j": rnd.choice([11, None]), "tag": "orphan"},
        ],
    )
    program = Program(
        f"prog-{seed:08d}-join-null-key-topk",
        seed,
        [
            {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"},
            {
                "op": "groupby",
                "keys": ["j"],
                "aggs": [{"column": "x", "func": "count", "as": "count_x"}],
            },
            {"op": "select", "columns": ["j"]},
            {
                "op": "sort",
                "keys": [
                    {
                        "column": "j",
                        "ascending": rnd.choice([True, False]),
                        "nulls": rnd.choice(["first", "last"]),
                    }
                ],
            },
            {"op": "limit", "n": 5},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-join-null-key-topk",
        seed=seed,
        tables=[left, right],
        program=program,
        metadata={
            "generator_profile": "join_null_key_topk",
            "source_issue": "https://github.com/apache/datafusion/issues/22190",
        },
    )


def generate_wide_offset_topk_case(seed: int) -> Case:
    rnd = random.Random(seed * 99_991 + 47)
    payload_columns = [
        ColumnSpec(f"p_{idx}", "float" if idx % 3 == 0 else "int", nullable=True)
        for idx in range(18)
    ]
    columns = [
        ColumnSpec("id", "int", nullable=False),
        ColumnSpec("sort_key", "int", nullable=True),
        *payload_columns,
    ]
    row_count = 384
    rows: list[dict[str, Any]] = []
    for idx in range(row_count):
        row: dict[str, Any] = {
            "id": idx,
            "sort_key": None if idx % 97 == 0 else (row_count - idx + (idx % 7)),
        }
        for column in payload_columns:
            payload_idx = int(column.name.split("_", 1)[1])
            if idx % (payload_idx + 11) == 0:
                row[column.name] = None
            elif column.type == "float":
                row[column.name] = round((idx * (payload_idx + 1)) / 13.0, 6)
            else:
                row[column.name] = idx * (payload_idx + 1)
        rows.append(row)
    offset_n = rnd.randint(240, 340)
    limit_n = rnd.randint(1, 4)
    table = TableData("t0", columns, rows)
    program = Program(
        f"prog-{seed:08d}-wide-offset-topk",
        seed,
        [
            {
                "op": "sort",
                "keys": [
                    {"column": "sort_key", "ascending": True, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "offset", "n": offset_n},
            {"op": "limit", "n": limit_n},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-wide-offset-topk",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "wide_offset_topk",
            "source_issue": "https://github.com/duckdb/duckdb/issues/11261",
            "row_count": row_count,
            "offset": offset_n,
            "limit": limit_n,
        },
    )


def generate_empty_filter_groupby_case(seed: int) -> Case:
    rnd = random.Random(seed * 1_299_709 + 53)
    rows = [
        {"id": 0, "g": "a", "x": 1, "lane": "keep"},
        {"id": 1, "g": "a", "x": None, "lane": "keep"},
        {"id": 2, "g": "b", "x": 2, "lane": "skip"},
        {"id": 3, "g": None, "x": rnd.choice([3, None]), "lane": "skip"},
    ]
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str", nullable=True),
            ColumnSpec("x", "int", nullable=True),
            ColumnSpec("lane", "str", nullable=False),
        ],
        rows,
    )
    program = Program(
        f"prog-{seed:08d}-empty-filter-groupby",
        seed,
        [
            {"op": "filter", "column": "lane", "cmp": "==", "value": "missing"},
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [{"column": "x", "func": "count", "as": "count_x"}],
            },
            {"op": "select", "columns": ["g", "count_x"]},
            {
                "op": "sort",
                "keys": [
                    {"column": "g", "ascending": True, "nulls": "last"},
                    {"column": "count_x", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "limit", "n": 4},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-empty-filter-groupby",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "empty_filter_groupby",
            "source_issue": "https://github.com/pandas-dev/pandas/issues/43767",
        },
    )


def generate_join_filter_groupby_case(seed: int) -> Case:
    rnd = random.Random(seed * 91815541 + 33)
    left = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str", nullable=False),
            ColumnSpec("x", "int", nullable=True),
            ColumnSpec("y", "float", nullable=True),
        ],
        [
            {"id": 0, "g": "a", "x": 1, "y": 0.5},
            {"id": 1, "g": "a", "x": 0, "y": -0.5},
            {"id": 1, "g": "b", "x": 2, "y": 1.0},
            {"id": 2, "g": "b", "x": -1, "y": 0.0},
            {"id": 3, "g": "c", "x": 3, "y": None},
        ],
    )
    right = TableData(
        "t1",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("j", "int", nullable=True),
            ColumnSpec("z", "float", nullable=True),
            ColumnSpec("tag", "str", nullable=True),
        ],
        [
            {"id": 0, "j": 5, "z": 0.5, "tag": "alpha"},
            {"id": 1, "j": 7, "z": 1.0, "tag": "beta"},
            {"id": 1, "j": -2, "z": -0.5, "tag": "gamma"},
            {"id": 3, "j": 1, "z": 2.0, "tag": "delta"},
        ],
    )
    ascending = rnd.choice([True, False])
    program = Program(
        f"prog-{seed:08d}-join-filter-groupby",
        seed,
        [
            {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "inner"},
            {"op": "filter", "column": "j", "cmp": ">=", "value": 0},
            {"op": "mutate", "column": "m_0", "expr": {"kind": "add_const", "source": "x", "value": 1}},
            {"op": "mutate", "column": "m_1", "expr": {"kind": "arith_const", "source": "m_0", "op": "mul", "value": 2}},
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [
                    {"column": "m_1", "func": "sum", "as": "sum_m_1"},
                    {"column": "j", "func": "count", "as": "count_j"},
                    {"column": "z", "func": "max", "as": "max_z"},
                ],
            },
            {"op": "select", "columns": ["g", "sum_m_1", "count_j", "max_z"]},
            {"op": "sort", "columns": ["sum_m_1", "count_j", "g"], "ascending": ascending},
            {"op": "limit", "n": 6},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-join-filter-groupby",
        seed=seed,
        tables=[left, right],
        program=program,
    )


def generate_join_null_truth_filter_case(seed: int) -> Case:
    rnd = random.Random(seed * 97_409 + 37)
    left = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str", nullable=False),
            ColumnSpec("x", "int", nullable=True),
        ],
        [
            {"id": 1, "g": "a", "x": 10},
            {"id": 2, "g": "b", "x": 20},
            {"id": 3, "g": "c", "x": 30},
            {"id": 4, "g": "d", "x": 40},
        ],
    )
    right = TableData(
        "t1",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("j", "int", nullable=True),
            ColumnSpec("z", "float", nullable=True),
            ColumnSpec("tag", "str", nullable=True),
        ],
        [
            {"id": 1, "j": 100, "z": 1.0, "tag": "p"},
            {"id": 2, "j": 200, "z": 2.0, "tag": "q"},
            {"id": 5, "j": 300, "z": 3.0, "tag": "r"},
        ],
    )
    threshold = rnd.choice([125, 150, 175])
    program = Program(
        f"prog-{seed:08d}-join-null-truth-filter",
        seed,
        [
            {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"},
            {"op": "filter", "column": "j", "cmp": "gt_is_not_true", "value": threshold},
            {"op": "select", "columns": ["id", "g", "j", "tag"]},
            {
                "op": "sort",
                "keys": [
                    {"column": "g", "ascending": True, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                ],
            },
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-join-null-truth-filter",
        seed=seed,
        tables=[left, right],
        program=program,
        metadata={
            "generator_profile": "join_null_truth_filter",
            "source_issue": "https://github.com/apache/datafusion/issues/22441",
        },
    )


def generate_join_groupby_stress_case(seed: int) -> Case:
    row_count = 100_000
    edges = _stable_graph_edges(row_count)
    left_rows = [{"fromnode": source, "tonode": target} for source, target in edges]
    path_rows = [{"p6_fromnode": source, "p6_tonode": target} for source, target in edges]
    probe_rows = [
        {
            "p4_fromnode": source,
            "p4_tonode_join": target,
            "p4_tonode": target,
        }
        for source, target in edges
    ]
    left = TableData(
        "t0",
        [
            ColumnSpec("fromnode", "int", nullable=False),
            ColumnSpec("tonode", "int", nullable=False),
        ],
        left_rows,
    )
    path = TableData(
        "t1",
        [
            ColumnSpec("p6_fromnode", "int", nullable=False),
            ColumnSpec("p6_tonode", "int", nullable=False),
        ],
        path_rows,
    )
    probe = TableData(
        "t2",
        [
            ColumnSpec("p4_fromnode", "int", nullable=False),
            ColumnSpec("p4_tonode_join", "int", nullable=False),
            ColumnSpec("p4_tonode", "int", nullable=False),
        ],
        probe_rows,
    )
    program = Program(
        f"prog-{seed:08d}-join-groupby-stress",
        seed,
        [
            {
                "op": "join",
                "table": "t1",
                "left_on": "tonode",
                "right_on": "p6_fromnode",
                "how": "inner",
            },
            {
                "op": "groupby",
                "keys": ["fromnode", "tonode"],
                "aggs": [{"column": "p6_tonode", "func": "count", "as": "path2_count"}],
            },
            {
                "op": "join",
                "table": "t2",
                "left_on": "fromnode",
                "right_on": "p4_tonode_join",
                "how": "inner",
            },
            {
                "op": "groupby",
                "keys": ["p4_fromnode", "p4_tonode"],
                "aggs": [{"column": "path2_count", "func": "sum", "as": "path3_count"}],
            },
            {
                "op": "aggregate",
                "aggs": [
                    {"column": "path3_count", "func": "sum", "as": "total_path3_count"},
                    {"column": "path3_count", "func": "count", "as": "path3_group_count"},
                ],
            },
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-join-groupby-stress",
        seed=seed,
        tables=[left, path, probe],
        program=program,
        metadata={"generator_profile": "join_groupby_stress", "row_count": row_count},
    )


def _stable_graph_edges(row_count: int) -> list[tuple[int, int]]:
    left_span = row_count // 2 + row_count // 4
    right_span = row_count // 2 + row_count
    edges: list[tuple[int, int]] = []
    for idx in range(row_count):
        left_seed = idx * 1_000_003
        right_seed = idx * 9_176
        if _stable_hash64(left_seed) == _stable_hash64(right_seed):
            continue
        source = min(_stable_hash64(left_seed) % left_span, _stable_hash64(left_seed + 7) % left_span)
        target = min(_stable_hash64(right_seed) % right_span, _stable_hash64(right_seed + 11) % right_span)
        edges.append((int(source), int(target)))
    return edges


def _stable_hash64(value: int) -> int:
    mask = (1 << 64) - 1
    value &= mask
    value ^= value >> 32
    value = (value * 0xD6E8FEB86659FD93) & mask
    value ^= value >> 32
    value = (value * 0xD6E8FEB86659FD93) & mask
    value ^= value >> 32
    return value & mask


def generate_float_group_key_case(seed: int) -> Case:
    rnd = random.Random(seed * 86028121 + 31)
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str", nullable=True),
            ColumnSpec("x", "int", nullable=True),
            ColumnSpec("y", "float", nullable=True),
            ColumnSpec("flag", "bool", nullable=True),
            ColumnSpec("s", "str", nullable=True),
        ],
        [
            {"id": 0, "g": None, "x": None, "y": 0.5, "flag": True, "s": "scnvvnMt"},
            {"id": 0, "g": "", "x": 0, "y": -0.5, "flag": None, "s": None},
            {"id": 0, "g": "cn", "x": 0, "y": 0.0, "flag": None, "s": "a"},
            {"id": 0, "g": "A", "x": -18, "y": 0.0, "flag": True, "s": "space value"},
        ],
    )
    join_table = TableData(
        "t1",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("j", "int", nullable=True),
            ColumnSpec("z", "float", nullable=True),
            ColumnSpec("tag", "str", nullable=True),
        ],
        [
            {"id": 0, "j": -10, "z": -0.5, "tag": "space value"},
            {"id": 1, "j": None, "z": 0.5, "tag": "A"},
            {"id": 0, "j": 0, "z": 1.0, "tag": "beta"},
            {"id": 0, "j": 2, "z": -0.5, "tag": "beta"},
        ],
    )
    program = Program(
        f"prog-{seed:08d}-float-group-key",
        seed,
        [
            {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "inner"},
            {"op": "mutate", "column": "m_0", "expr": {"kind": "add_const", "source": "x", "value": -1}},
            {"op": "filter", "column": "m_0", "cmp": "==", "value": -1},
            {
                "op": "sort",
                "columns": ["s", "flag", "g", "id", "j", "m_0", "tag", "x", "y", "z"],
                "ascending": rnd.choice([True, False]),
            },
            {"op": "mutate", "column": "m_1", "expr": {"kind": "arith_const", "source": "m_0", "op": "mul", "value": 10}},
            {"op": "mutate", "column": "m_2", "expr": {"kind": "cast", "source": "m_0", "to": "float"}},
            {"op": "mutate", "column": "m_3", "expr": {"kind": "arith_const", "source": "m_1", "op": "div", "value": 3}},
            {
                "op": "groupby",
                "keys": ["m_3"],
                "aggs": [{"column": "m_2", "func": "min", "as": "min_m_2"}],
            },
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-float-group-key",
        seed=seed,
        tables=[table, join_table],
        program=program,
    )


def generate_join_null_sort_case(seed: int) -> Case:
    rnd = random.Random(seed * 9999991 + 41)
    left = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("x", "int", nullable=True),
            ColumnSpec("y", "float", nullable=True),
            ColumnSpec("s", "str", nullable=True),
        ],
        [
            {"id": 0, "x": 2, "y": 0.5, "s": "alpha"},
            {"id": 1, "x": -1, "y": None, "s": "Beta"},
            {"id": 1, "x": 0, "y": -0.5, "s": "space value"},
            {"id": 2, "x": 3, "y": 1.0, "s": "gamma"},
            {"id": 4, "x": None, "y": 0.0, "s": ""},
        ],
    )
    right = TableData(
        "t1",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("j", "int", nullable=True),
            ColumnSpec("z", "float", nullable=True),
            ColumnSpec("tag", "str", nullable=True),
        ],
        [
            {"id": 1, "j": None, "z": 1.0, "tag": "Alpha"},
            {"id": 1, "j": 2, "z": None, "tag": "beta"},
            {"id": 2, "j": -2, "z": -0.5, "tag": "MIXED"},
            {"id": 3, "j": 7, "z": 0.5, "tag": "orphan"},
        ],
    )
    ascending = rnd.choice([True, False])
    program = Program(
        f"prog-{seed:08d}-join-null-sort",
        seed,
        [
            {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"},
            {"op": "mutate", "column": "m_0", "expr": {"kind": "cast", "source": "j", "to": "float"}},
            {"op": "mutate", "column": "m_1", "expr": {"kind": "string_length", "source": "tag"}},
            {"op": "select", "columns": ["id", "x", "y", "s", "tag", "m_0", "m_1"]},
            {"op": "sort", "columns": ["m_0", "m_1", "id", "s"], "ascending": ascending},
            {"op": "limit", "n": 6},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-join-null-sort",
        seed=seed,
        tables=[left, right],
        program=program,
    )


def generate_ordered_groupby_sort_case(seed: int) -> Case:
    rnd = random.Random(seed * 1_000_003 + 59)
    groups = ["a", "b", "c", "d"]
    rows: list[dict[str, Any]] = []
    for idx in range(8):
        group = groups[idx % len(groups)]
        if idx == 0:
            group = "a"
        value_choices = [None, -2, -1, 0, 1, 2, 5, 10]
        rows.append(
            {
                "id": idx,
                "g": group,
                "x": rnd.choice(value_choices),
                "z": rnd.choice([None, -1, 0, 1, 2, 4, 8]),
            }
        )
    # Ensure two non-null aggregate outputs whose input-order and value-order
    # disagree. This exercises stale sortedness/ordering metadata generally,
    # without hard-coding a backend-specific oracle.
    rows[0] = {"id": 0, "g": "a", "x": 1, "z": 1}
    rows[1] = {"id": 1, "g": "b", "x": 2, "z": 2}
    rows[2] = {"id": 2, "g": "b", "x": 0, "z": 0}
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str", nullable=False),
            ColumnSpec("x", "int", nullable=True),
            ColumnSpec("z", "int", nullable=True),
        ],
        rows,
    )
    value_col = rnd.choice(["x", "z"])
    agg_func = rnd.choice(["max", "min"])
    alias = f"{agg_func}_{value_col}"
    pre_sort_ascending = rnd.choice([True, False])
    post_sort_ascending = rnd.choice([True, False])
    program = Program(
        f"prog-{seed:08d}-ordered-groupby-sort",
        seed,
        [
            {
                "op": "sort",
                "keys": [
                    {"column": value_col, "ascending": pre_sort_ascending, "nulls": "last"},
                    {"column": "g", "ascending": True, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                ],
            },
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [{"column": value_col, "func": agg_func, "as": alias}],
            },
            {"op": "select", "columns": ["g", alias]},
            {
                "op": "sort",
                "keys": [
                    {"column": alias, "ascending": post_sort_ascending, "nulls": "last"},
                    {"column": "g", "ascending": True, "nulls": "last"},
                ],
            },
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-ordered-groupby-sort",
        seed=seed,
        tables=[table],
        program=program,
        metadata={"generator_profile": "ordered_groupby_sort"},
    )


def generate_topk_resort_case(seed: int) -> Case:
    rnd = random.Random(seed * 1_618_033 + 61)
    rows = []
    for idx in range(14):
        rows.append(
            {
                "id": idx,
                "g": rnd.choice(["a", "b", "c", None]),
                "x": rnd.choice([None, -10, -2, -1, 0, 1, 2, 10]),
                "z": rnd.choice([None, -3, 0, 1, 3, 8]),
                "s": rnd.choice(["", "A", "a", "space value", None]),
            }
        )
    rows[0] = {"id": 0, "g": "a", "x": None, "z": 8, "s": "A"}
    rows[1] = {"id": 1, "g": "b", "x": 10, "z": None, "s": "space value"}
    rows[2] = {"id": 2, "g": "c", "x": -10, "z": -3, "s": ""}
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str", nullable=True),
            ColumnSpec("x", "int", nullable=True),
            ColumnSpec("z", "int", nullable=True),
            ColumnSpec("s", "str", nullable=True),
        ],
        rows,
    )
    first_col = rnd.choice(["x", "z", "g", "s"])
    second_col = rnd.choice([col for col in ["x", "z", "g", "s"] if col != first_col])
    limit_n = rnd.randint(1, 6)
    offset_n = rnd.randint(0, 2)
    program = Program(
        f"prog-{seed:08d}-topk-resort",
        seed,
        [
            {
                "op": "sort",
                "keys": [
                    {"column": first_col, "ascending": rnd.choice([True, False]), "nulls": rnd.choice(["first", "last"])},
                    {"column": "id", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "limit", "n": limit_n},
            {
                "op": "sort",
                "keys": [
                    {"column": second_col, "ascending": rnd.choice([True, False]), "nulls": rnd.choice(["first", "last"])},
                    {"column": "id", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "offset", "n": offset_n},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-topk-resort",
        seed=seed,
        tables=[table],
        program=program,
        metadata={"generator_profile": "topk_resort", "limit": limit_n, "offset": offset_n},
    )


def generate_join_ordered_agg_topk_case(seed: int) -> Case:
    rnd = random.Random(seed * 2_147_483 + 67)
    left = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str", nullable=True),
            ColumnSpec("x", "int", nullable=True),
        ],
        [
            {"id": 0, "g": "a", "x": 1},
            {"id": 1, "g": "b", "x": 2},
            {"id": 1, "g": "b", "x": 0},
            {"id": 2, "g": "c", "x": None},
            {"id": 3, "g": "d", "x": -1},
            {"id": 4, "g": None, "x": 5},
        ],
    )
    right = TableData(
        "t1",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("j", "int", nullable=True),
            ColumnSpec("z", "int", nullable=True),
            ColumnSpec("tag", "str", nullable=True),
        ],
        [
            {"id": 0, "j": 10, "z": 1, "tag": "alpha"},
            {"id": 1, "j": 2, "z": None, "tag": "beta"},
            {"id": 1, "j": 7, "z": 3, "tag": "Beta"},
            {"id": 2, "j": None, "z": 8, "tag": "space value"},
            {"id": 5, "j": 9, "z": -1, "tag": "orphan"},
        ],
    )
    agg_col = rnd.choice(["j", "z", "x"])
    agg_func = rnd.choice(["min", "max", "sum", "count"])
    alias = f"{agg_func}_{agg_col}"
    sort_ascending = rnd.choice([True, False])
    program = Program(
        f"prog-{seed:08d}-join-ordered-agg-topk",
        seed,
        [
            {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"},
            {
                "op": "sort",
                "keys": [
                    {"column": "j", "ascending": rnd.choice([True, False]), "nulls": rnd.choice(["first", "last"])},
                    {"column": "g", "ascending": True, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                ],
            },
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [{"column": agg_col, "func": agg_func, "as": alias}],
            },
            {"op": "select", "columns": ["g", alias]},
            {
                "op": "sort",
                "keys": [
                    {"column": alias, "ascending": sort_ascending, "nulls": "last"},
                    {"column": "g", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "limit", "n": rnd.randint(1, 6)},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-join-ordered-agg-topk",
        seed=seed,
        tables=[left, right],
        program=program,
        metadata={"generator_profile": "join_ordered_agg_topk"},
    )


def generate_global_null_aggregate_case(seed: int) -> Case:
    empty_input = seed % 2 == 0
    rows = [] if empty_input else [
        {"id": 0, "g": "a", "x": None, "y": None},
        {"id": 1, "g": "b", "x": None, "y": None},
        {"id": 2, "g": None, "x": None, "y": 5},
        {"id": 3, "g": "space value", "x": None, "y": -1},
    ]
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str", nullable=True),
            ColumnSpec("x", "int", nullable=True),
            ColumnSpec("y", "int", nullable=True),
        ],
        rows,
    )
    program = Program(
        f"prog-{seed:08d}-global-null-aggregate",
        seed,
        [
            {
                "op": "aggregate",
                "aggs": [
                    {"column": "x", "func": "sum", "as": "sum_x"},
                    {"column": "x", "func": "min", "as": "min_x"},
                    {"column": "x", "func": "max", "as": "max_x"},
                    {"column": "x", "func": "count", "as": "count_x"},
                    {"column": "y", "func": "sum", "as": "sum_y"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "sum_x", "ascending": True, "nulls": "first"},
                    {"column": "count_x", "ascending": False, "nulls": "last"},
                ],
            },
            {"op": "limit", "n": 1},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-global-null-aggregate",
        seed=seed,
        tables=[table],
        program=program,
        metadata={"generator_profile": "global_null_aggregate", "empty_input": empty_input},
    )


def generate_string_count_groupby_case(seed: int) -> Case:
    rows = [
        {"id": 0, "g": "alpha", "s": "A", "x": 1},
        {"id": 1, "g": "alpha", "s": None, "x": 2},
        {"id": 2, "g": "alpha", "s": "", "x": None},
        {"id": 3, "g": "beta", "s": "space value", "x": -1},
        {"id": 4, "g": "beta", "s": "中文", "x": 0},
        {"id": 5, "g": None, "s": None, "x": 5},
        {"id": 6, "g": None, "s": "delta", "x": None},
        {"id": 7, "g": "gamma", "s": None, "x": 3},
    ]
    if seed % 2:
        rows.append({"id": 8, "g": "gamma", "s": "", "x": -3})
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str", nullable=True),
            ColumnSpec("s", "str", nullable=True),
            ColumnSpec("x", "int", nullable=True),
        ],
        rows,
    )
    program = Program(
        f"prog-{seed:08d}-string-count-groupby",
        seed,
        [
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [
                    {"column": "s", "func": "count", "as": "count_s"},
                    {"column": "x", "func": "sum", "as": "sum_x"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "count_s", "ascending": False, "nulls": "last"},
                    {"column": "g", "ascending": True, "nulls": "first"},
                ],
            },
            {"op": "limit", "n": 4},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-string-count-groupby",
        seed=seed,
        tables=[table],
        program=program,
        metadata={"generator_profile": "string_count_groupby"},
    )


def generate_unique_count_groupby_case(seed: int) -> Case:
    rows = [
        {"id": 0, "g": "alpha", "s": "red", "x": 1},
        {"id": 1, "g": "alpha", "s": "red", "x": 1},
        {"id": 2, "g": "alpha", "s": None, "x": None},
        {"id": 3, "g": "beta", "s": "", "x": 2},
        {"id": 4, "g": "beta", "s": "space value", "x": 3},
        {"id": 5, "g": "beta", "s": "", "x": 2},
        {"id": 6, "g": None, "s": None, "x": None},
        {"id": 7, "g": None, "s": "中文", "x": 4},
        {"id": 8, "g": "gamma", "s": None, "x": None},
    ]
    if seed % 2:
        rows.append({"id": 9, "g": "gamma", "s": "red", "x": 4})
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str", nullable=True),
            ColumnSpec("s", "str", nullable=True),
            ColumnSpec("x", "int", nullable=True),
        ],
        rows,
    )
    program = Program(
        f"prog-{seed:08d}-unique-count-groupby",
        seed,
        [
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [
                    {"column": "s", "func": "nunique", "as": "uniq_s_count"},
                    {"column": "x", "func": "nunique", "as": "uniq_x_count"},
                    {"column": "s", "func": "count", "as": "count_s"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "uniq_s_count", "ascending": False, "nulls": "last"},
                    {"column": "uniq_x_count", "ascending": False, "nulls": "last"},
                    {"column": "g", "ascending": True, "nulls": "first"},
                ],
            },
            {"op": "limit", "n": 5},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-unique-count-groupby",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "unique_count_groupby",
            "source_issue": "https://github.com/apache/arrow/issues/36149",
        },
    )


def generate_bool_null_groupby_agg_case(seed: int) -> Case:
    rows = [
        {"id": 0, "g": "alpha", "flag": True, "x": 1},
        {"id": 1, "g": "alpha", "flag": None, "x": 2},
        {"id": 2, "g": "alpha", "flag": False, "x": None},
        {"id": 3, "g": "beta", "flag": None, "x": -1},
        {"id": 4, "g": "beta", "flag": None, "x": 0},
        {"id": 5, "g": None, "flag": True, "x": 5},
        {"id": 6, "g": None, "flag": False, "x": None},
        {"id": 7, "g": "gamma", "flag": None, "x": 3},
    ]
    if seed % 2:
        rows.append({"id": 8, "g": "gamma", "flag": True, "x": -3})
    if seed % 3 == 0:
        rows.append({"id": 9, "g": "delta", "flag": False, "x": 4})
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str", nullable=True),
            ColumnSpec("flag", "bool", nullable=True),
            ColumnSpec("x", "int", nullable=True),
        ],
        rows,
    )
    program = Program(
        f"prog-{seed:08d}-bool-null-groupby-agg",
        seed,
        [
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [
                    {"column": "flag", "func": "any", "as": "flag_any_value"},
                    {"column": "flag", "func": "all", "as": "flag_all_value"},
                    {"column": "flag", "func": "min", "as": "flag_min_value"},
                    {"column": "flag", "func": "max", "as": "flag_max_value"},
                    {"column": "flag", "func": "count", "as": "flag_seen_count"},
                    {"column": "flag", "func": "nunique", "as": "flag_distinct_count"},
                    {"column": "x", "func": "sum", "as": "x_sum_value"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "flag_seen_count", "ascending": False, "nulls": "last"},
                    {"column": "flag_any_value", "ascending": False, "nulls": "last"},
                    {"column": "flag_all_value", "ascending": True, "nulls": "last"},
                    {"column": "flag_min_value", "ascending": True, "nulls": "last"},
                    {"column": "g", "ascending": True, "nulls": "first"},
                ],
            },
            {"op": "limit", "n": 5},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-bool-null-groupby-agg",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "bool_null_groupby_agg",
            "source_issue": "https://github.com/pola-rs/polars/issues/26671",
        },
    )


def generate_large_int_filter_groupby_case(seed: int) -> Case:
    high = 9_007_199_254_740_992
    rows = [
        {"id": 0, "g": "alpha", "x": high - 1, "flag": True},
        {"id": 1, "g": "alpha", "x": high, "flag": False},
        {"id": 2, "g": "alpha", "x": high + 1, "flag": None},
        {"id": 3, "g": "beta", "x": -high + 1, "flag": True},
        {"id": 4, "g": "beta", "x": -high, "flag": False},
        {"id": 5, "g": None, "x": 0, "flag": None},
        {"id": 6, "g": None, "x": 42, "flag": True},
        {"id": 7, "g": "gamma", "x": None, "flag": False},
    ]
    if seed % 2:
        rows.append({"id": 8, "g": "gamma", "x": high + 3, "flag": True})
    if seed % 3 == 0:
        rows.append({"id": 9, "g": "delta", "x": -high - 3, "flag": None})
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str", nullable=True),
            ColumnSpec("x", "int", nullable=True),
            ColumnSpec("flag", "bool", nullable=True),
        ],
        rows,
    )
    filter_mode = seed % 3
    if filter_mode == 0:
        filter_op = {"op": "filter", "column": "x", "cmp": "in_set", "value": [high - 1, high + 1, -high, 0]}
    elif filter_mode == 1:
        filter_op = {"op": "filter", "column": "x", "cmp": "range_closed", "value": [-high, high]}
    else:
        filter_op = {"op": "filter", "column": "x", "cmp": "!=", "value": 42}
    program = Program(
        f"prog-{seed:08d}-large-int-filter-groupby",
        seed,
        [
            filter_op,
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [
                    {"column": "x", "func": "count", "as": "x_seen_count"},
                    {"column": "x", "func": "min", "as": "x_min_value"},
                    {"column": "x", "func": "max", "as": "x_max_value"},
                    {"column": "flag", "func": "count", "as": "flag_seen_count"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "x_max_value", "ascending": False, "nulls": "last"},
                    {"column": "g", "ascending": True, "nulls": "first"},
                ],
            },
            {"op": "limit", "n": 5},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-large-int-filter-groupby",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "large_int_filter_groupby",
            "source_issue": "https://github.com/pola-rs/polars/issues/27726",
            "source_issue_alt": "https://github.com/duckdb/duckdb/issues/22676",
        },
    )


def generate_set_membership_filter_case(seed: int) -> Case:
    rows = [
        {"id": 0, "g": "alpha", "s": "red", "x": 1, "flag": True},
        {"id": 1, "g": "alpha", "s": "", "x": 2, "flag": False},
        {"id": 2, "g": "alpha", "s": None, "x": None, "flag": None},
        {"id": 3, "g": "beta", "s": "blue", "x": 2, "flag": True},
        {"id": 4, "g": "beta", "s": "中文", "x": 3, "flag": False},
        {"id": 5, "g": "beta", "s": "space value", "x": None, "flag": True},
        {"id": 6, "g": None, "s": "", "x": 4, "flag": None},
        {"id": 7, "g": None, "s": "red", "x": 4, "flag": False},
        {"id": 8, "g": "gamma", "s": None, "x": 5, "flag": True},
    ]
    if seed % 2:
        rows.append({"id": 9, "g": "gamma", "s": "中文", "x": 5, "flag": False})
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str", nullable=True),
            ColumnSpec("s", "str", nullable=True),
            ColumnSpec("x", "int", nullable=True),
            ColumnSpec("flag", "bool", nullable=True),
        ],
        rows,
    )
    program = Program(
        f"prog-{seed:08d}-set-membership-filter",
        seed,
        [
            {"op": "filter", "column": "s", "cmp": "in_set", "value": ["red", "", "中文"]},
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [
                    {"column": "s", "func": "count", "as": "count_s"},
                    {"column": "x", "func": "nunique", "as": "uniq_x_count"},
                ],
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "count_s", "ascending": False, "nulls": "last"},
                    {"column": "g", "ascending": True, "nulls": "first"},
                ],
            },
            {"op": "limit", "n": 5},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-set-membership-filter",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "set_membership_filter",
            "source_issue": "https://github.com/pola-rs/polars/issues/22149",
        },
    )


def generate_pyarrow_groupby_filter_cast_membership_case(seed: int) -> Case:
    bucket_code = seed % 3
    other_bucket = (bucket_code + 1) % 3
    last_bucket = (bucket_code + 2) % 3
    membership_values = [float(bucket_code) + 0.5, float(-(bucket_code + 1)), 10.0]
    rows = [
        {"bucket_code": bucket_code, "amount_code": 10, "metric_value": -0.5},
        {"bucket_code": bucket_code, "amount_code": 2, "metric_value": -1.0},
        {"bucket_code": other_bucket, "amount_code": 10, "metric_value": 0.5},
        {"bucket_code": last_bucket, "amount_code": None, "metric_value": 1.0},
    ]
    if seed % 2:
        rows.append({"bucket_code": other_bucket, "amount_code": 2, "metric_value": None})
    table = TableData(
        "t0",
        [
            ColumnSpec("bucket_code", "int", nullable=False),
            ColumnSpec("amount_code", "int", nullable=True),
            ColumnSpec("metric_value", "float", nullable=True),
        ],
        rows,
    )
    program = Program(
        f"prog-{seed:08d}-pyarrow-groupby-filter-cast-membership",
        seed,
        [
            {
                "op": "groupby",
                "keys": ["bucket_code", "amount_code"],
                "aggs": [
                    {"column": "bucket_code", "func": "min", "as": "agg_min_bucket"},
                    {"column": "amount_code", "func": "max", "as": "agg_max_amount"},
                ],
            },
            {"op": "filter", "column": "agg_min_bucket", "cmp": "in_set", "value": membership_values},
            {
                "op": "sort",
                "keys": [
                    {"column": "agg_min_bucket", "ascending": True, "nulls": "last"},
                    {"column": "amount_code", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "select", "columns": ["bucket_code", "amount_code", "agg_min_bucket", "agg_max_amount"]},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-pyarrow-groupby-filter-cast-membership",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "pyarrow_groupby_filter_cast_membership",
            "membership_values": membership_values,
        },
    )


def generate_null_predicate_filter_case(seed: int) -> Case:
    use_is_null = seed % 2 == 0
    predicate = "is_null" if use_is_null else "is_not_null"
    rows = [
        {"id": 0, "g": "alpha", "s": "red", "x": 1, "flag": True},
        {"id": 1, "g": "alpha", "s": None, "x": 2, "flag": False},
        {"id": 2, "g": "beta", "s": "", "x": None, "flag": None},
        {"id": 3, "g": "beta", "s": "blue", "x": 2, "flag": True},
        {"id": 4, "g": None, "s": None, "x": 3, "flag": False},
        {"id": 5, "g": None, "s": "中文", "x": None, "flag": None},
        {"id": 6, "g": "gamma", "s": "space value", "x": 4, "flag": True},
        {"id": 7, "g": "gamma", "s": None, "x": 4, "flag": False},
    ]
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str", nullable=True),
            ColumnSpec("s", "str", nullable=True),
            ColumnSpec("x", "int", nullable=True),
            ColumnSpec("flag", "bool", nullable=True),
        ],
        rows,
    )
    program = Program(
        f"prog-{seed:08d}-null-predicate-filter",
        seed,
        [
            {"op": "filter", "column": "s", "cmp": predicate, "value": None},
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
                    {"column": "count_id", "ascending": False, "nulls": "last"},
                    {"column": "g", "ascending": True, "nulls": "first"},
                ],
            },
            {"op": "limit", "n": 5},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-null-predicate-filter",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "null_predicate_filter",
            "predicate": predicate,
            "source_issue": "https://github.com/duckdb/duckdb/issues/4978",
        },
    )


def generate_boolean_predicate_filter_case(seed: int) -> Case:
    predicates = ["bool_is_true", "bool_is_not_true", "bool_is_false", "bool_is_not_false"]
    predicate = predicates[seed % len(predicates)]
    rows = [
        {"id": 0, "g": "alpha", "flag": True, "x": 1, "s": "red"},
        {"id": 1, "g": "alpha", "flag": False, "x": 2, "s": ""},
        {"id": 2, "g": "alpha", "flag": None, "x": None, "s": None},
        {"id": 3, "g": "beta", "flag": True, "x": 2, "s": "blue"},
        {"id": 4, "g": "beta", "flag": False, "x": 3, "s": "中文"},
        {"id": 5, "g": "beta", "flag": None, "x": None, "s": "space value"},
        {"id": 6, "g": None, "flag": True, "x": 4, "s": "red"},
        {"id": 7, "g": None, "flag": False, "x": 4, "s": None},
        {"id": 8, "g": "gamma", "flag": None, "x": 5, "s": ""},
    ]
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str", nullable=True),
            ColumnSpec("flag", "bool", nullable=True),
            ColumnSpec("x", "int", nullable=True),
            ColumnSpec("s", "str", nullable=True),
        ],
        rows,
    )
    program = Program(
        f"prog-{seed:08d}-boolean-predicate-filter",
        seed,
        [
            {"op": "filter", "column": "flag", "cmp": predicate, "value": None},
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
                    {"column": "count_id", "ascending": False, "nulls": "last"},
                    {"column": "g", "ascending": True, "nulls": "first"},
                ],
            },
            {"op": "limit", "n": 5},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-boolean-predicate-filter",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "boolean_predicate_filter",
            "predicate": predicate,
            "source_issue": "https://github.com/pola-rs/polars/issues/8516",
            "source_issue_alt": "https://github.com/apache/datafusion/issues/22441",
        },
    )


def generate_post_topk_range_filter_case(seed: int) -> Case:
    limit_n = 6 if seed % 2 == 0 else 7
    rows = [
        {"id": 0, "g": "top", "score": 100, "x": -5, "flag": True},
        {"id": 1, "g": "top", "score": 95, "x": 7, "flag": False},
        {"id": 2, "g": "keep", "score": 90, "x": 1, "flag": None},
        {"id": 3, "g": "top", "score": 85, "x": None, "flag": True},
        {"id": 4, "g": "keep", "score": 80, "x": 2, "flag": False},
        {"id": 5, "g": "top", "score": 75, "x": -3, "flag": None},
        {"id": 6, "g": "late", "score": 70, "x": 0, "flag": True},
        {"id": 7, "g": "late", "score": 65, "x": 2, "flag": False},
        {"id": 8, "g": "late", "score": 60, "x": 1, "flag": None},
        {"id": 9, "g": None, "score": None, "x": 1, "flag": True},
    ]
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str", nullable=True),
            ColumnSpec("score", "int", nullable=True),
            ColumnSpec("x", "int", nullable=True),
            ColumnSpec("flag", "bool", nullable=True),
        ],
        rows,
    )
    program = Program(
        f"prog-{seed:08d}-post-topk-range-filter",
        seed,
        [
            {
                "op": "sort",
                "keys": [
                    {"column": "score", "ascending": False, "nulls": "last"},
                    {"column": "id", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "limit", "n": limit_n},
            {"op": "filter", "column": "x", "cmp": "range_closed", "value": [0, 2]},
            {"op": "select", "columns": ["id", "g", "score", "x"]},
            {
                "op": "sort",
                "keys": [
                    {"column": "id", "ascending": True, "nulls": "last"},
                ],
            },
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-post-topk-range-filter",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "post_topk_range_filter",
            "source_issue": "https://github.com/pola-rs/polars/issues/26803",
            "source_issue_alt": "https://github.com/duckdb/duckdb/issues/22075",
            "limit": limit_n,
        },
    )


def generate_tuple_absence_filter_case(seed: int) -> Case:
    left_rows = [
        {"row_id": 0, "a": 1, "b": 1, "payload": "matched"},
        {"row_id": 1, "a": 2, "b": 2, "payload": "survivor"},
        {"row_id": 2, "a": 3, "b": None, "payload": "unknown-left"},
        {"row_id": 3, "a": None, "b": 4, "payload": "unknown-both"},
    ]
    if seed % 2:
        left_rows.append({"row_id": 4, "a": 5, "b": 4, "payload": "definite-false"})
    left = TableData(
        "t0",
        [
            ColumnSpec("row_id", "int", nullable=False),
            ColumnSpec("a", "int", nullable=True),
            ColumnSpec("b", "int", nullable=True),
            ColumnSpec("payload", "str", nullable=True),
        ],
        left_rows,
    )
    right = TableData(
        "t1",
        [
            ColumnSpec("a", "int", nullable=True),
            ColumnSpec("b", "int", nullable=True),
        ],
        [
            {"a": 1, "b": 1},
            {"a": None, "b": 4},
        ],
    )
    program = Program(
        f"prog-{seed:08d}-tuple-absence-filter",
        seed,
        [
            {"op": "tuple_absence_filter", "columns": ["a", "b"], "table": "t1", "right_columns": ["a", "b"]},
            {"op": "select", "columns": ["row_id", "a", "b", "payload"]},
            {"op": "sort", "keys": [{"column": "row_id", "ascending": True, "nulls": "last"}]},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-tuple-absence-filter",
        seed=seed,
        tables=[left, right],
        program=program,
        metadata={
            "generator_profile": "tuple_absence_filter",
            "source_issue": "https://github.com/duckdb/duckdb/issues/22418",
        },
    )


def generate_row_value_absence_filter_case(seed: int) -> Case:
    left_rows = [
        {"sample_id": 0, "left_a": 1, "left_b": 1, "label_value": "matched"},
        {"sample_id": 1, "left_a": 2, "left_b": 2, "label_value": "survivor"},
        {"sample_id": 2, "left_a": 3, "left_b": None, "label_value": "null-left"},
        {"sample_id": 3, "left_a": None, "left_b": 4, "label_value": "null-pair"},
    ]
    if seed % 2:
        left_rows.append({"sample_id": 4, "left_a": 5, "left_b": 4, "label_value": "unknown-tail"})
    left = TableData(
        "t0",
        [
            ColumnSpec("sample_id", "int", nullable=False),
            ColumnSpec("left_a", "int", nullable=True),
            ColumnSpec("left_b", "int", nullable=True),
            ColumnSpec("label_value", "str", nullable=True),
        ],
        left_rows,
    )
    right = TableData(
        "t1",
        [
            ColumnSpec("right_a", "int", nullable=True),
            ColumnSpec("right_b", "int", nullable=True),
        ],
        [
            {"right_a": 1, "right_b": 1},
            {"right_a": None, "right_b": 4},
        ],
    )
    program = Program(
        f"prog-{seed:08d}-row-value-absence-filter",
        seed,
        [
            {
                "op": "tuple_absence_filter",
                "columns": ["left_a", "left_b"],
                "table": "t1",
                "right_columns": ["right_a", "right_b"],
            },
            {"op": "select", "columns": ["sample_id", "left_a", "left_b", "label_value"]},
            {"op": "sort", "keys": [{"column": "sample_id", "ascending": True, "nulls": "last"}]},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-row-value-absence-filter",
        seed=seed,
        tables=[left, right],
        program=program,
        metadata={"generator_profile": "row_value_absence_filter"},
    )


def generate_running_sum_precision_case(seed: int) -> Case:
    row_count = 20_000
    increment = 0.0005
    rows = [{"row_id": idx, "x": increment} for idx in range(row_count)]
    table = TableData(
        "t0",
        [
            ColumnSpec("row_id", "int", nullable=False),
            ColumnSpec("x", "float", nullable=False),
        ],
        rows,
    )
    program = Program(
        f"prog-{seed:08d}-running-sum-precision",
        seed,
        [
            {
                "op": "running_sum",
                "source": "x",
                "column": "run_x",
                "order_by": [{"column": "row_id", "ascending": True, "nulls": "last"}],
                "input_dtype": "float32",
            },
            {
                "op": "sort",
                "keys": [{"column": "row_id", "ascending": False, "nulls": "last"}],
            },
            {"op": "limit", "n": 1},
            {"op": "select", "columns": ["row_id", "run_x"]},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-running-sum-precision",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "running_sum_precision",
            "source_issue": "https://github.com/pola-rs/polars/issues/27662",
            "source_issue_alt": "https://github.com/pola-rs/polars/issues/26800",
            "row_count": row_count,
            "increment": increment,
            "expected_total": row_count * increment,
        },
    )


def generate_partitioned_running_sum_case(seed: int) -> Case:
    rnd = random.Random(seed * 4001 + 17)
    rows = []
    row_id = 0
    for group in ["A", "B", None]:
        for seq, value in enumerate([1.0, None, 2.5, -0.5], start=1):
            adjusted = value if value is None else value + (0.25 if group == "B" else 0.0)
            rows.append({"row_id": row_id, "grp": group, "seq": seq, "x": adjusted})
            row_id += 1
    rnd.shuffle(rows)
    table = TableData(
        "t0",
        [
            ColumnSpec("row_id", "int", nullable=False),
            ColumnSpec("grp", "str", nullable=True),
            ColumnSpec("seq", "int", nullable=False),
            ColumnSpec("x", "float", nullable=True),
        ],
        rows,
    )
    program = Program(
        f"prog-{seed:08d}-partitioned-running-sum",
        seed,
        [
            {
                "op": "running_sum",
                "source": "x",
                "column": "run_x",
                "partition_by": ["grp"],
                "order_by": [
                    {"column": "seq", "ascending": True, "nulls": "last"},
                    {"column": "row_id", "ascending": True, "nulls": "last"},
                ],
                "input_dtype": "float64",
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "grp", "ascending": True, "nulls": "last"},
                    {"column": "seq", "ascending": True, "nulls": "last"},
                    {"column": "row_id", "ascending": True, "nulls": "last"},
                ],
            },
            {"op": "select", "columns": ["grp", "seq", "run_x"]},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-partitioned-running-sum",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "partitioned_running_sum",
            "issue_inspiration": "partitioned ROWS-frame cumulative aggregates",
        },
    )


def generate_path_basename_keyed_pick_case(seed: int) -> Case:
    rnd = random.Random(seed * 5021 + 31)
    path_values = [
        "C:/warehouse/january/report.csv",
        "/tmp/releases/build.tar",
        "relative/name.parquet",
        "plain_name.txt",
        r"D:\archive\delta.log",
        None,
    ]
    rows = []
    row_id = 0
    for group in ["A", "B", None]:
        for offset, path_value in enumerate(path_values):
            rows.append(
                {
                    "row_id": row_id,
                    "grp": group,
                    "pick_key": (offset * 3 + row_id) % 11,
                    "path_value": path_value,
                    "payload": row_id % 5,
                }
            )
            row_id += 1
    rnd.shuffle(rows)
    table = TableData(
        "t0",
        [
            ColumnSpec("row_id", "int", nullable=False),
            ColumnSpec("grp", "str", nullable=True),
            ColumnSpec("pick_key", "int", nullable=False),
            ColumnSpec("path_value", "str", nullable=True),
            ColumnSpec("payload", "int", nullable=False),
        ],
        rows,
    )
    basename_column = make_safe_output_name(
        "path_base_value",
        used={column.name for column in table.columns},
    )
    row_limit_cmp = "<=" if seed % 3 == 0 else "=="
    row_limit_value = 2 if row_limit_cmp == "<=" else 1
    pick_keys = [{"column": "pick_key", "ascending": True, "nulls": "last"}]
    output_sort_keys = [
        {"column": "pick_key", "ascending": True, "nulls": "last"},
        {"column": "row_id", "ascending": True, "nulls": "last"},
    ]
    program = Program(
        f"prog-{seed:08d}-path-basename-keyed-pick",
        seed,
        [
            {
                "op": "mutate",
                "column": basename_column,
                "expr": {"kind": "string_basename", "source": "path_value"},
            },
            {
                "op": "row_number_filter",
                "partition_by": ["grp"],
                "order_by": pick_keys,
                "cmp": row_limit_cmp,
                "value": row_limit_value,
            },
            {
                "op": "sort",
                "keys": [
                    {"column": "grp", "ascending": True, "nulls": "last"},
                    *output_sort_keys,
                ],
            },
            {"op": "select", "columns": ["grp", "pick_key", basename_column]},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-path-basename-keyed-pick",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "path_basename_keyed_pick",
            "source_issue": "https://github.com/duckdb/duckdb/issues/22849",
            "issue_inspiration": "path-string projection composed with keyed row selection",
        },
    )


def generate_sortedness_null_placement_case(seed: int) -> Case:
    rows = [
        {"row_id": 0, "x": 3},
        {"row_id": 1, "x": 1},
        {"row_id": 2, "x": 2},
        {"row_id": 3, "x": None},
    ]
    table = TableData(
        "t0",
        [
            ColumnSpec("row_id", "int", nullable=False),
            ColumnSpec("x", "int", nullable=True),
        ],
        rows,
    )
    alias = make_safe_output_name("sorted_ok_x", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-sortedness-null-placement",
        seed,
        [
            {
                "op": "sort",
                "keys": [{"column": "x", "ascending": True, "nulls": "last"}],
            },
            {
                "op": "sortedness_check",
                "column": "x",
                "as": alias,
                "ascending": True,
                "nulls": "first",
            },
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-sortedness-null-placement",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "sortedness_null_placement",
            "source_issue": "https://github.com/pola-rs/polars/issues/26993",
            "expected_sortedness": False,
        },
    )


def generate_simple_case_random_subject_case(seed: int) -> Case:
    row_count = 100_000
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("unexpected_else_seen", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-simple-case-random-subject",
        seed,
        [
            {
                "op": "random_case_probe",
                "as": alias,
                "rows": row_count,
                "branches": 3,
            }
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-simple-case-random-subject",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "simple_case_random_subject",
            "source_issue": "https://github.com/duckdb/duckdb/issues/22576",
            "row_count": row_count,
            "expected_unexpected_else_seen": False,
        },
    )


def generate_group_quantile_key_probe_case(seed: int) -> Case:
    values = [1, 2, 3]
    quantiles = [0.0, 0.5, 1.0]
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("quantile_key_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-group-quantile-key-probe",
        seed,
        [
            {
                "op": "group_quantile_probe",
                "as": alias,
                "values": values,
                "quantiles": quantiles,
            }
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-group-quantile-key-probe",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "group_quantile_key_probe",
            "source_issue": "https://github.com/pola-rs/polars/issues/25888",
            "expected_quantiles": [1.0, 2.0, 3.0],
            "expected_quantile_key_mismatch": False,
        },
    )


def generate_scalar_subquery_double_parentheses_case(seed: int) -> Case:
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("scalar_subquery_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-scalar-subquery-double-parentheses",
        seed,
        [{"op": "scalar_subquery_probe", "as": alias}],
    )
    return Case(
        case_id=f"case-{seed:08d}-scalar-subquery-double-parentheses",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "scalar_subquery_double_parentheses",
            "source_issue": "https://github.com/duckdb/duckdb/issues/19851",
            "expected_scalar_subquery_mismatch": False,
            "expected_rows": [[2]],
        },
    )


def generate_window_avg_rows_frame_case(seed: int) -> Case:
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("window_avg_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-window-avg-rows-frame",
        seed,
        [{"op": "window_avg_probe", "as": alias}],
    )
    return Case(
        case_id=f"case-{seed:08d}-window-avg-rows-frame",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "window_avg_rows_frame",
            "source_issue": "https://github.com/pola-rs/polars/issues/26065",
            "expected_window_avg_mismatch": False,
            "expected_window_avg_values": [10.0, 15.0, 20.0, 25.0, 30.0],
        },
    )


def generate_struct_distinct_unnest_case(seed: int) -> Case:
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("struct_distinct_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-struct-distinct-unnest",
        seed,
        [{"op": "struct_distinct_probe", "as": alias}],
    )
    return Case(
        case_id=f"case-{seed:08d}-struct-distinct-unnest",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "struct_distinct_unnest",
            "source_issue": "https://github.com/duckdb/duckdb/issues/17278",
            "expected_struct_distinct_mismatch": False,
            "expected_rows": [["0", "0"], ["0", "1"]],
        },
    )


def generate_bit_compare_unequal_length_case(seed: int) -> Case:
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("bit_compare_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-bit-compare-unequal-length",
        seed,
        [{"op": "bit_compare_probe", "as": alias}],
    )
    return Case(
        case_id=f"case-{seed:08d}-bit-compare-unequal-length",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "bit_compare_unequal_length",
            "source_issue": "https://github.com/duckdb/duckdb/issues/22527",
            "expected_bit_compare_mismatch": False,
            "expected_bit_less": True,
        },
    )


def generate_round_even_float_scale_case(seed: int) -> Case:
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("round_even_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-round-even-float-scale",
        seed,
        [{"op": "round_even_probe", "as": alias}],
    )
    return Case(
        case_id=f"case-{seed:08d}-round-even-float-scale",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "round_even_float_scale",
            "source_issue": "https://github.com/duckdb/duckdb/issues/19491",
            "expected_round_even_mismatch": False,
            "expected_round_even_value": 2.67,
        },
    )


def generate_duckdb_float_literal_precision_case(seed: int) -> Case:
    literals = [
        "0.41000000000000003",
        "0.9999999999999999",
        "0.1000000000000000055511151231257827",
    ]
    literal = literals[seed % len(literals)]
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("float_literal_precision_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-duckdb-float-literal-precision",
        seed,
        [{"op": "float_literal_precision_probe", "as": alias, "literal": literal}],
    )
    return Case(
        case_id=f"case-{seed:08d}-duckdb-float-literal-precision",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "duckdb_float_literal_precision",
            "source_issue": "https://github.com/duckdb/duckdb/issues/22837",
            "issue_inspiration": "decimal float literal and string cast should round identically",
            "literal": literal,
            "expected_float_literal_precision_mismatch": False,
        },
    )


def generate_polars_timestamp_precision_filter_case(seed: int) -> Case:
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("timestamp_precision_filter_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-polars-timestamp-precision-filter",
        seed,
        [{"op": "timestamp_precision_filter_probe", "as": alias}],
    )
    return Case(
        case_id=f"case-{seed:08d}-polars-timestamp-precision-filter",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "polars_timestamp_precision_filter",
            "source_issue": "https://github.com/pola-rs/polars/issues/27726",
            "issue_inspiration": "timestamp filter should compare us columns and ns literals without lossy downcast",
            "expected_timestamp_precision_filter_rows": 1,
        },
    )


def generate_series_rtruediv_operand_order_case(seed: int) -> Case:
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("series_rtruediv_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-series-rtruediv-operand-order",
        seed,
        [{"op": "series_rtruediv_probe", "as": alias}],
    )
    return Case(
        case_id=f"case-{seed:08d}-series-rtruediv-operand-order",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "series_rtruediv_operand_order",
            "source_issue": "https://github.com/pola-rs/polars/issues/17760",
            "expected_series_rtruediv_mismatch": False,
            "expected_series_rtruediv_values": [2.0, 1.5, 4.0 / 3.0],
        },
    )


def generate_polars_reverse_division_columns_case(seed: int) -> Case:
    offset = seed % 3
    rows = [
        {"divisor_value": 1 + offset, "numerator_value": 2 + offset, "group_code": 0},
        {"divisor_value": 2 + offset, "numerator_value": 3 + offset, "group_code": 1},
        {"divisor_value": 3 + offset, "numerator_value": 4 + offset, "group_code": 1},
    ]
    table = TableData(
        "t0",
        [
            ColumnSpec("divisor_value", "int", nullable=False),
            ColumnSpec("numerator_value", "int", nullable=False),
            ColumnSpec("group_code", "int", nullable=False),
        ],
        rows,
    )
    program = Program(
        f"prog-{seed:08d}-polars-reverse-division-columns",
        seed,
        [
            {
                "op": "mutate",
                "column": "ratio_value",
                "expr": {
                    "kind": "reverse_division_columns",
                    "source": "divisor_value",
                    "numerator": "numerator_value",
                },
            },
            {"op": "select", "columns": ["group_code", "ratio_value"]},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}-polars-reverse-division-columns",
        seed=seed,
        tables=[table],
        program=program,
        metadata={"generator_profile": "polars_reverse_division_columns"},
    )


def generate_pandas_uint64_isin_precision_case(seed: int) -> Case:
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("uint64_isin_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-pandas-uint64-isin-precision",
        seed,
        [{"op": "uint64_isin_probe", "as": alias}],
    )
    return Case(
        case_id=f"case-{seed:08d}-pandas-uint64-isin-precision",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "pandas_uint64_isin_precision",
            "source_issue": "https://github.com/pandas-dev/pandas/issues/59609",
            "expected_uint64_isin_mismatch": False,
            "expected_uint64_isin_value": False,
        },
    )


def generate_duckdb_tuple_anti_null_semantics_case(seed: int) -> Case:
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("tuple_anti_null_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-duckdb-tuple-anti-null-semantics",
        seed,
        [{"op": "tuple_anti_null_probe", "as": alias}],
    )
    return Case(
        case_id=f"case-{seed:08d}-duckdb-tuple-anti-null-semantics",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "duckdb_tuple_anti_null_semantics",
            "source_issue": "https://github.com/duckdb/duckdb/issues/22418",
            "expected_tuple_anti_null_mismatch": False,
            "expected_tuple_anti_null_counts": [1, 1, 2],
        },
    )


def generate_datafusion_setop_all_duplicate_count_case(seed: int) -> Case:
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("setop_all_duplicate_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-datafusion-setop-all-duplicate-count",
        seed,
        [{"op": "setop_all_duplicate_probe", "as": alias}],
    )
    return Case(
        case_id=f"case-{seed:08d}-datafusion-setop-all-duplicate-count",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "datafusion_setop_all_duplicate_count",
            "source_issue": "https://github.com/apache/datafusion/issues/12956",
            "source_issue_alt": "https://github.com/apache/datafusion/issues/12955",
            "expected_setop_all_duplicate_mismatch": False,
            "expected_except_all_counts": {"a": 1, "b": 1, "c": 2},
            "expected_intersect_all_counts": {"b": 2, "c": 2},
        },
    )


def generate_duckdb_json_predicate_order_semantics_case(seed: int) -> Case:
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("json_predicate_order_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-duckdb-json-predicate-order-semantics",
        seed,
        [{"op": "json_predicate_order_probe", "as": alias}],
    )
    return Case(
        case_id=f"case-{seed:08d}-duckdb-json-predicate-order-semantics",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "duckdb_json_predicate_order_semantics",
            "source_issue": "https://github.com/duckdb/duckdb/issues/20366",
            "expected_json_predicate_order_mismatch": False,
            "expected_json_predicate_order_rows": 1,
        },
    )


def generate_pandas_sparse_array_mask_semantics_case(seed: int) -> Case:
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("sparse_mask_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-pandas-sparse-array-mask-semantics",
        seed,
        [{"op": "sparse_mask_probe", "as": alias}],
    )
    return Case(
        case_id=f"case-{seed:08d}-pandas-sparse-array-mask-semantics",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "pandas_sparse_array_mask_semantics",
            "source_issue": "https://github.com/pandas-dev/pandas/issues/45284",
            "expected_sparse_mask_mismatch": False,
            "expected_sparse_mask_values": [4.0],
        },
    )


def generate_polars_float_wrap_numerical_semantics_case(seed: int) -> Case:
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("float_wrap_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-polars-float-wrap-numerical-semantics",
        seed,
        [{"op": "float_wrap_probe", "as": alias}],
    )
    return Case(
        case_id=f"case-{seed:08d}-polars-float-wrap-numerical-semantics",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "polars_float_wrap_numerical_semantics",
            "source_issue": "https://github.com/pola-rs/polars/issues/18546",
            "expected_float_wrap_mismatch": False,
            "expected_wrapped_uint8": [100, 44],
        },
    )


def generate_pandas_index_bool_result_type_case(seed: int) -> Case:
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("index_bool_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-pandas-index-bool-result-type",
        seed,
        [{"op": "index_bool_probe", "as": alias}],
    )
    return Case(
        case_id=f"case-{seed:08d}-pandas-index-bool-result-type",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "pandas_index_bool_result_type",
            "source_issue": "https://github.com/pandas-dev/pandas/issues/62766",
            "expected_index_bool_mismatch": False,
            "expected_index_bool_type": "Index",
        },
    )


def generate_polars_empty_literal_groupby_semantics_case(seed: int) -> Case:
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("empty_literal_groupby_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-polars-empty-literal-groupby-semantics",
        seed,
        [{"op": "empty_literal_groupby_probe", "as": alias}],
    )
    return Case(
        case_id=f"case-{seed:08d}-polars-empty-literal-groupby-semantics",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "polars_empty_literal_groupby_semantics",
            "source_issue": "https://github.com/pola-rs/polars/issues/23870",
            "expected_empty_literal_groupby_mismatch": False,
            "expected_empty_literal_groupby_rows": 0,
        },
    )


def generate_pandas_arrow_string_eq_sum_semantics_case(seed: int) -> Case:
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("arrow_string_eq_sum_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-pandas-arrow-string-eq-sum-semantics",
        seed,
        [{"op": "arrow_string_eq_sum_probe", "as": alias}],
    )
    return Case(
        case_id=f"case-{seed:08d}-pandas-arrow-string-eq-sum-semantics",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "pandas_arrow_string_eq_sum_semantics",
            "source_issue": "https://github.com/pandas-dev/pandas/issues/63458",
            "expected_arrow_string_eq_sum_mismatch": False,
            "expected_arrow_string_eq_sum_value": 1,
        },
    )


def generate_pandas_arrow_timestamp_loc_slice_semantics_case(seed: int) -> Case:
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("arrow_timestamp_loc_slice_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-pandas-arrow-timestamp-loc-slice-semantics",
        seed,
        [{"op": "arrow_timestamp_loc_slice_probe", "as": alias}],
    )
    return Case(
        case_id=f"case-{seed:08d}-pandas-arrow-timestamp-loc-slice-semantics",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "pandas_arrow_timestamp_loc_slice_semantics",
            "source_issue": "https://github.com/pandas-dev/pandas/issues/63526",
            "expected_arrow_timestamp_loc_slice_mismatch": False,
            "expected_arrow_timestamp_loc_slice_rows": 1,
        },
    )


def generate_pandas_arrow_timestamp_index_attr_semantics_case(seed: int) -> Case:
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("arrow_timestamp_index_attr_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-pandas-arrow-timestamp-index-attr-semantics",
        seed,
        [{"op": "arrow_timestamp_index_attr_probe", "as": alias}],
    )
    return Case(
        case_id=f"case-{seed:08d}-pandas-arrow-timestamp-index-attr-semantics",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "pandas_arrow_timestamp_index_attr_semantics",
            "source_issue": "https://github.com/pandas-dev/pandas/issues/63527",
            "expected_arrow_timestamp_index_attr_mismatch": False,
            "expected_arrow_timestamp_index_months": [5, 6],
        },
    )


def generate_pandas_eval_inplace_aliasing_semantics_case(seed: int) -> Case:
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("eval_inplace_alias_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-pandas-eval-inplace-aliasing-semantics",
        seed,
        [{"op": "eval_inplace_alias_probe", "as": alias}],
    )
    return Case(
        case_id=f"case-{seed:08d}-pandas-eval-inplace-aliasing-semantics",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "pandas_eval_inplace_aliasing_semantics",
            "source_issue": "https://github.com/pandas-dev/pandas/issues/65664",
            "expected_eval_inplace_alias_mismatch": False,
            "expected_eval_inplace_alias_rows": 3,
        },
    )


def generate_pandas_bool_reduction_skipna_semantics_case(seed: int) -> Case:
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("bool_reduction_skipna_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-pandas-bool-reduction-skipna-semantics",
        seed,
        [{"op": "bool_reduction_skipna_probe", "as": alias}],
    )
    return Case(
        case_id=f"case-{seed:08d}-pandas-bool-reduction-skipna-semantics",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "pandas_bool_reduction_skipna_semantics",
            "source_issue": "https://github.com/pandas-dev/pandas/issues/65710",
            "expected_bool_reduction_skipna_mismatch": False,
        },
    )


def generate_pyarrow_dataset_isin_all_match_semantics_case(seed: int) -> Case:
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("dataset_isin_all_match_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-pyarrow-dataset-isin-all-match-semantics",
        seed,
        [{"op": "dataset_isin_all_match_probe", "as": alias}],
    )
    return Case(
        case_id=f"case-{seed:08d}-pyarrow-dataset-isin-all-match-semantics",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "pyarrow_dataset_isin_all_match_semantics",
            "source_issue": "https://github.com/apache/arrow/issues/46183",
            "expected_dataset_isin_all_match_mismatch": False,
            "expected_dataset_isin_all_match_rows": 1,
        },
    )


def generate_pyarrow_run_end_null_compute_semantics_case(seed: int) -> Case:
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("run_end_null_compute_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-pyarrow-run-end-null-compute-semantics",
        seed,
        [{"op": "run_end_null_compute_probe", "as": alias}],
    )
    return Case(
        case_id=f"case-{seed:08d}-pyarrow-run-end-null-compute-semantics",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "pyarrow_run_end_null_compute_semantics",
            "source_issue": "https://github.com/apache/arrow/issues/49889",
            "expected_run_end_null_compute_mismatch": False,
            "expected_run_end_null_compute_values": [True, None, True, True, True],
        },
    )


def generate_pyarrow_large_string_partition_schema_semantics_case(seed: int) -> Case:
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("large_string_partition_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-pyarrow-large-string-partition-schema-semantics",
        seed,
        [{"op": "large_string_partition_probe", "as": alias}],
    )
    return Case(
        case_id=f"case-{seed:08d}-pyarrow-large-string-partition-schema-semantics",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "pyarrow_large_string_partition_schema_semantics",
            "source_issue": "https://github.com/apache/arrow/issues/47177",
            "expected_large_string_partition_mismatch": False,
            "expected_large_string_partition_rows": 4,
        },
    )


def generate_pyarrow_hash_pivot_wider_order_semantics_case(seed: int) -> Case:
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("hash_pivot_wider_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-pyarrow-hash-pivot-wider-order-semantics",
        seed,
        [{"op": "hash_pivot_wider_probe", "as": alias}],
    )
    return Case(
        case_id=f"case-{seed:08d}-pyarrow-hash-pivot-wider-order-semantics",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "pyarrow_hash_pivot_wider_order_semantics",
            "source_issue": "https://github.com/apache/arrow/issues/48679",
            "expected_hash_pivot_wider_mismatch": False,
            "expected_hash_pivot_wider_rows": 3,
        },
    )


def generate_pyarrow_list_flatten_parent_indices_semantics_case(seed: int) -> Case:
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("list_flatten_parent_indices_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-pyarrow-list-flatten-parent-indices-semantics",
        seed,
        [{"op": "list_flatten_parent_indices_probe", "as": alias}],
    )
    return Case(
        case_id=f"case-{seed:08d}-pyarrow-list-flatten-parent-indices-semantics",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "pyarrow_list_flatten_parent_indices_semantics",
            "expected_list_flatten_parent_indices_mismatch": False,
            "expected_list_flatten_values": [1, 2, None, 3],
            "expected_list_parent_indices": [0, 0, 3, 3],
        },
    )


def generate_polars_rolling_mean_by_null_count_semantics_case(seed: int) -> Case:
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("rolling_mean_by_null_count_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-polars-rolling-mean-by-null-count-semantics",
        seed,
        [{"op": "rolling_mean_by_null_count_probe", "as": alias}],
    )
    return Case(
        case_id=f"case-{seed:08d}-polars-rolling-mean-by-null-count-semantics",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "polars_rolling_mean_by_null_count_semantics",
            "source_issue": "https://github.com/pola-rs/polars/issues/27661",
            "expected_rolling_mean_by_null_count_mismatch": False,
            "expected_rolling_mean_by_null_count_values": [None, 400.5, None],
        },
    )


def generate_csv_long_numeric_roundtrip_case(seed: int) -> Case:
    base_values = list(DEFAULT_LONG_NUMERIC_CSV_VALUES)
    rotation = seed % len(base_values)
    values = base_values[rotation:] + base_values[:rotation]
    if seed % 2:
        values.append(str(10**20 + (seed % 997)))
    values = csv_long_numeric_values({"values": values})
    table = TableData(
        "t0",
        [ColumnSpec("probe_id", "int", nullable=False)],
        [{"probe_id": 0}],
    )
    alias = make_safe_output_name("csv_long_numeric_roundtrip_mismatch", used={column.name for column in table.columns})
    program = Program(
        f"prog-{seed:08d}-csv-long-numeric-roundtrip",
        seed,
        [{"op": "csv_long_numeric_roundtrip_probe", "as": alias, "values": values}],
    )
    return Case(
        case_id=f"case-{seed:08d}-csv-long-numeric-roundtrip",
        seed=seed,
        tables=[table],
        program=program,
        metadata={
            "generator_profile": "csv_long_numeric_roundtrip",
            "issue_inspiration": "CSV long numeric identifiers should round-trip without lossy inference",
            "expected_csv_long_numeric_roundtrip_mismatch": False,
            "expected_csv_values": values,
        },
    )


def _cover_join_table_keys(primary: TableData, right: TableData) -> TableData:
    existing = {row.get("id") for row in right.rows}
    rows = list(right.rows)
    left_ids = sorted(
        {
            row.get("id")
            for row in primary.rows
            if isinstance(row.get("id"), int) and not isinstance(row.get("id"), bool)
        }
    )
    for value in left_ids:
        if value in existing:
            continue
        row: dict[str, Any] = {}
        for column in right.columns:
            if column.name == "id":
                row[column.name] = value
            elif column.type == "int":
                row[column.name] = value
            elif column.type == "float":
                row[column.name] = float(value)
            elif column.type == "bool":
                row[column.name] = bool(value % 2)
            else:
                row[column.name] = f"tag_{value}"
        rows.append(row)
        existing.add(value)
    return TableData(right.name, right.columns, rows)


def generate_workflow_case(seed: int) -> Case:
    workflows = [
        _etl_cleanup_workflow,
        _log_aggregation_workflow,
        _feature_engineering_workflow,
        _join_enrichment_workflow,
        _null_heavy_workflow,
    ]
    builder = workflows[seed % len(workflows)]
    return builder(seed)


def _etl_cleanup_workflow(seed: int) -> Case:
    rnd = random.Random(seed * 8191 + 101)
    rows = [
        {
            "id": idx,
            "g": rnd.choice(["retail", "enterprise", "trial", ""]),
            "x": rnd.choice([0, 1, 2, 10, None]),
            "y": rnd.choice([0.0, 1.0, 3.5, 10.0, None]),
            "flag": rnd.choice([True, False, None]),
        }
        for idx in range(12)
    ]
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str", nullable=True),
            ColumnSpec("x", "int", nullable=True),
            ColumnSpec("y", "float", nullable=True),
            ColumnSpec("flag", "bool", nullable=True),
        ],
        rows,
    )
    program = Program(
        f"prog-{seed:08d}-workflow-etl",
        seed,
        [
            {"op": "filter", "column": "flag", "cmp": "==", "value": True},
            {"op": "mutate", "column": "y_float", "expr": {"kind": "cast", "source": "y", "to": "float"}},
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [
                    {"column": "y_float", "func": "sum", "as": "sum_y"},
                    {"column": "x", "func": "count", "as": "count_x"},
                ],
            },
        ],
    )
    return Case(f"case-{seed:08d}-workflow-etl", seed, [table], program)


def _log_aggregation_workflow(seed: int) -> Case:
    rnd = random.Random(seed * 8191 + 211)
    services = ["api", "worker", "frontend", "scheduler", None]
    rows = [
        {
            "id": idx,
            "g": rnd.choice(services),
            "x": rnd.choice([0, 1, 2, 5, 10, -1]),
            "s": rnd.choice(["INFO", "WARN", "ERROR", "error", ""]),
        }
        for idx in range(18)
    ]
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str", nullable=True),
            ColumnSpec("x", "int", nullable=True),
            ColumnSpec("s", "str", nullable=True),
        ],
        rows,
    )
    program = Program(
        f"prog-{seed:08d}-workflow-log",
        seed,
        [
            {"op": "filter", "column": "x", "cmp": ">=", "value": 0},
            {"op": "mutate", "column": "level", "expr": {"kind": "string_lower", "source": "s"}},
            {
                "op": "groupby",
                "keys": ["g", "level"],
                "aggs": [
                    {"column": "id", "func": "count", "as": "events"},
                    {"column": "x", "func": "sum", "as": "total_x"},
                ],
            },
        ],
    )
    return Case(f"case-{seed:08d}-workflow-log", seed, [table], program)


def _feature_engineering_workflow(seed: int) -> Case:
    rnd = random.Random(seed * 8191 + 307)
    rows = [
        {
            "id": idx,
            "g": rnd.choice(["A", "B", "C", None]),
            "x": rnd.choice([-2, -1, 0, 1, 2, 10, None]),
            "y": rnd.choice([0.0, 0.5, 1.0, 2.5, None]),
            "s": rnd.choice(["alpha", "beta", "space value", "", None]),
        }
        for idx in range(14)
    ]
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str", nullable=True),
            ColumnSpec("x", "int", nullable=True),
            ColumnSpec("y", "float", nullable=True),
            ColumnSpec("s", "str", nullable=True),
        ],
        rows,
    )
    program = Program(
        f"prog-{seed:08d}-workflow-feature",
        seed,
        [
            {"op": "mutate", "column": "x_shift", "expr": {"kind": "add_const", "source": "x", "value": 2}},
            {"op": "mutate", "column": "s_len", "expr": {"kind": "string_length", "source": "s"}},
            {"op": "filter", "column": "x_shift", "cmp": ">=", "value": 0},
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [
                    {"column": "x_shift", "func": "max", "as": "max_x_shift"},
                    {"column": "s_len", "func": "sum", "as": "sum_s_len"},
                ],
            },
        ],
    )
    return Case(f"case-{seed:08d}-workflow-feature", seed, [table], program)


def _join_enrichment_workflow(seed: int) -> Case:
    rnd = random.Random(seed * 8191 + 401)
    fact_rows = [
        {
            "id": rnd.choice([idx, idx % 4, 0, 1]),
            "g": rnd.choice(["north", "south", "west", None]),
            "x": rnd.choice([0, 1, 2, 10, None]),
            "y": rnd.choice([0.0, 1.0, 4.5, None]),
        }
        for idx in range(16)
    ]
    dim_rows = [
        {
            "id": idx,
            "j": rnd.choice([0, 1, 5, 10, None]),
            "z": rnd.choice([0.0, 1.0, 2.0, 10.0, None]),
            "tag": rnd.choice(["gold", "silver", "bronze", None]),
        }
        for idx in range(6)
    ]
    fact = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str", nullable=True),
            ColumnSpec("x", "int", nullable=True),
            ColumnSpec("y", "float", nullable=True),
        ],
        fact_rows,
    )
    dim = TableData(
        "t1",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("j", "int", nullable=True),
            ColumnSpec("z", "float", nullable=True),
            ColumnSpec("tag", "str", nullable=True),
        ],
        dim_rows,
    )
    program = Program(
        f"prog-{seed:08d}-workflow-join",
        seed,
        [
            {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"},
            {"op": "filter", "column": "j", "cmp": ">=", "value": 0},
            {"op": "mutate", "column": "tag_norm", "expr": {"kind": "string_lower", "source": "tag"}},
            {
                "op": "groupby",
                "keys": ["tag_norm"],
                "aggs": [
                    {"column": "x", "func": "sum", "as": "sum_x"},
                    {"column": "z", "func": "max", "as": "max_z"},
                ],
            },
        ],
    )
    return Case(f"case-{seed:08d}-workflow-join", seed, [fact, dim], program)


def _null_heavy_workflow(seed: int) -> Case:
    rnd = random.Random(seed * 8191 + 503)
    rows = [
        {
            "id": idx,
            "g": rnd.choice(["known", "unknown", None]),
            "x": rnd.choice([None, None, 0, 1, 2, 10]),
            "y": rnd.choice([None, None, 0.0, 1.0, -1.0]),
            "flag": rnd.choice([True, False, None]),
        }
        for idx in range(20)
    ]
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str", nullable=True),
            ColumnSpec("x", "int", nullable=True),
            ColumnSpec("y", "float", nullable=True),
            ColumnSpec("flag", "bool", nullable=True),
        ],
        rows,
    )
    program = Program(
        f"prog-{seed:08d}-workflow-null",
        seed,
        [
            {"op": "filter", "column": "g", "cmp": "!=", "value": "unknown"},
            {"op": "mutate", "column": "x_plus", "expr": {"kind": "add_const", "source": "x", "value": 1}},
            {
                "op": "groupby",
                "keys": ["flag"],
                "aggs": [
                    {"column": "x_plus", "func": "sum", "as": "sum_x_plus"},
                    {"column": "y", "func": "count", "as": "count_y"},
                ],
            },
        ],
    )
    return Case(f"case-{seed:08d}-workflow-null", seed, [table], program)
