from __future__ import annotations

import math
import random
import string
from typing import Any, Literal

from .csv_roundtrip import DEFAULT_LONG_NUMERIC_CSV_VALUES, csv_long_numeric_values
from .dsl import Case, ColumnSpec, Program, SortKey, TableData, normalize_sort_keys
from .filtering import filter_comparator_supports_type, parse_filter_comparator
from .identifiers import is_reserved_output_name, make_safe_output_name
from .util import unique_preserve_order

GeneratorProfile = Literal[
    "common",
    "edge_float",
    "workflow",
    "bughunt",
    "bughunt_fresh",
    "bughunt_no_groupby",
    "issue_focus",
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
    "polars_rolling_mean_by_null_count_semantics",
    "csv_long_numeric_roundtrip",
]


def _is_bughunt_profile(profile: GeneratorProfile) -> bool:
    return profile in {"bughunt", "bughunt_fresh", "bughunt_no_groupby", "issue_focus"}


def _bughunt_allows_groupby(profile: GeneratorProfile) -> bool:
    return profile in {"bughunt", "bughunt_fresh", "issue_focus"}


def _aggregate_accepts_type(source_type: str, func: str, is_numeric_column: bool) -> bool:
    if func in {"count", "nunique"}:
        return True
    if func in {"any", "all"}:
        return source_type == "bool"
    if func in {"min", "max"} and source_type == "bool":
        return True
    return is_numeric_column


def _aggregate_output_type(source_type: str, func: str) -> str:
    if func in {"count", "nunique"}:
        return "int"
    if func in {"any", "all"}:
        return "bool"
    if func == "mean":
        return "float"
    return source_type


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
    ncols = len(base_cols) if _is_bughunt_profile(profile) else rnd.randint(3, len(base_cols))
    columns = base_cols[:ncols]
    nrows = rnd.randint(min_rows, max_rows)
    rows: list[dict[str, Any]] = []
    for i in range(nrows):
        row: dict[str, Any] = {}
        for col in columns:
            if col.name == "id":
                # Repeated IDs are useful for joins and group-like behavior.
                choices = [i, i % 5, 0, 1]
                if _is_bughunt_profile(profile):
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
    nrows = rnd.randint(8, 18) if _is_bughunt_profile(profile) else rnd.randint(1, 12)
    rows: list[dict[str, Any]] = []
    for i in range(nrows):
        rows.append(
            {
                "id": rnd.choice([i, i % 5, i % 3, 0, 1, 2]) if _is_bughunt_profile(profile) else rnd.choice([i, i % 5, 0, 1, 2]),
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


def _literal_for_type(rnd: random.Random, typ: str) -> Any:
    if typ == "int":
        return rnd.choice([-10, -1, 0, 1, 2, 10])
    if typ == "float":
        return rnd.choice([-1.0, 0.0, 0.5, 1.0, 10.0])
    if typ == "bool":
        return rnd.choice([True, False])
    return rnd.choice(["", "alpha", "beta", "中文", "missing"])


def _literal_list_for_type(rnd: random.Random, typ: str) -> list[Any]:
    if typ == "int":
        return rnd.sample([-10, -1, 0, 1, 2, 10], k=3)
    if typ == "float":
        return rnd.sample([-1.0, 0.0, 0.5, 1.0, 10.0], k=3)
    if typ == "bool":
        return rnd.sample([True, False], k=rnd.randint(1, 2))
    return rnd.sample(["", "alpha", "beta", "中文", "missing"], k=3)


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

    op_pool = ["filter", "select", "sort", "limit", "offset", "mutate", "groupby"]
    bughunt_profile = _is_bughunt_profile(profile)
    if bughunt_profile:
        op_pool = [
            "filter",
            "filter",
            "mutate",
            "mutate",
            "sort",
            "limit",
            "offset",
            "select",
        ]
        if _bughunt_allows_groupby(profile):
            op_pool.extend(["groupby", "groupby"])
    if extra_tables:
        op_pool.extend(["join", "join"] if bughunt_profile else ["join"])
    if bughunt_profile:
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
                    allow_groupby=profile != "bughunt_no_groupby",
                )
            )
            continue

        before_len = len(ops)
        possible = list(op_pool)
        if grouped:
            possible = ["sort", "limit", "offset", "select"]
        if joined_tables:
            possible = [p for p in possible if p != "join"]
        remaining = nops - index
        if bughunt_profile and not grouped:
            if extra_tables and not joined_tables and "id" in available_cols and (not ops or rnd.random() < 0.8):
                op = "join"
            elif "mutate" not in emitted_ops and (numeric_cols or string_cols) and len(ops) >= int(bool(extra_tables)):
                op = "mutate"
            elif "filter" not in emitted_ops and comparable_cols and len(ops) >= 2 and remaining > 2:
                op = "filter"
            elif (
                _bughunt_allows_groupby(profile)
                and
                "groupby" not in emitted_ops
                and (numeric_cols or bool_cols)
                and len(ops) >= 3
                and (remaining <= 3 or rnd.random() < 0.65)
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

        elif op == "filter" and comparable_cols and not grouped:
            col = rnd.choice(comparable_cols)
            typ = col_types[col]
            cmp_ops = ["==", "!="] if typ in {"str", "bool"} else [">", ">=", "<", "<=", "==", "!="]
            if bughunt_profile and typ in {"int", "float"} and rnd.random() < 0.35:
                cmp_ops = [
                    "gt_is_not_true",
                    "ge_is_not_true",
                    "lt_is_not_false",
                    "le_is_not_false",
                    *cmp_ops,
                ]
            if bughunt_profile and rnd.random() < 0.20:
                cmp_ops = ["in_set", *cmp_ops]
            if bughunt_profile and rnd.random() < 0.15:
                cmp_ops = [rnd.choice(["is_null", "is_not_null"]), *cmp_ops]
            if bughunt_profile and typ == "bool" and rnd.random() < 0.35:
                cmp_ops = [
                    rnd.choice(["bool_is_true", "bool_is_not_true", "bool_is_false", "bool_is_not_false"]),
                    *cmp_ops,
                ]
            if bughunt_profile and typ in {"int", "float"} and rnd.random() < 0.20:
                cmp_ops = ["range_closed", *cmp_ops]
            base_cols = {c.name for c in table.columns}
            cmp = rnd.choice(cmp_ops)
            if cmp == "in_set":
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

        elif op == "sort" and available_cols:
            first = rnd.choice(available_cols)
            cols = [first] + sorted(c for c in available_cols if c != first)
            ops.append(_random_sort_op(rnd, cols, allow_mixed=_is_bughunt_profile(profile)))

        elif op == "limit":
            ops.append({"op": "limit", "n": rnd.randint(0, max(1, len(table.rows) + 2))})

        elif op == "offset":
            ops.append({"op": "offset", "n": rnd.randint(0, max(1, len(table.rows) + 2))})

        elif op == "mutate" and (numeric_cols or string_cols) and not grouped:
            new_col = f"m_{len([o for o in ops if o.get('op') == 'mutate'])}"
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
            if bughunt_profile:
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
            available_cols = keys + [a["as"] for a in aggs]
            numeric_cols = []
            bool_cols = []
            for agg in aggs:
                output_type = _aggregate_output_type(col_types.get(agg["column"], "float"), agg["func"])
                col_types[agg["as"]] = output_type
                if output_type in {"int", "float"}:
                    numeric_cols.append(agg["as"])
                if output_type == "bool":
                    bool_cols.append(agg["as"])
            string_cols = [c for c in keys if col_types.get(c) == "str"]
            comparable_cols = available_cols
            grouped = True
        if len(ops) > before_len:
            emitted_ops.add(str(ops[-1].get("op", "")))

    if type_aware:
        ops = repair_operations(table, ops, extra_tables=extra_tables)
        if bughunt_profile:
            ops = _add_bughunt_order_projection_probe(ops, table, extra_tables, rnd)
    if not ops:
        ops.append({"op": "limit", "n": len(table.rows)})
    return Program(program_id=f"prog-{seed:08d}", seed=seed, operations=ops)


def _add_bughunt_order_projection_probe(
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
    extra_tables = extra_tables or []
    table_by_name = {t.name: t for t in [table] + extra_tables}
    available = [column.name for column in table.columns]
    for op in ops:
        kind = op.get("op")
        if kind == "join":
            right = table_by_name.get(str(op.get("table", "")))
            if right is not None:
                available.extend(
                    column.name
                    for column in right.columns
                    if column.name != op.get("right_on") and column.name not in available
                )
        elif kind == "select":
            selected = [str(column) for column in op.get("columns", [])]
            available = [column for column in selected if column in available]
        elif kind == "mutate":
            column = str(op.get("column", ""))
            if column:
                available = [existing for existing in available if existing != column] + [column]
        elif kind == "running_sum":
            column = str(op.get("column", ""))
            if column:
                available = [existing for existing in available if existing != column] + [column]
        elif kind == "sortedness_check":
            alias = str(op.get("as", ""))
            available = [alias] if alias else []
        elif kind == "random_case_probe":
            alias = str(op.get("as", ""))
            available = [alias] if alias else []
        elif kind == "group_quantile_probe":
            alias = str(op.get("as", ""))
            available = [alias] if alias else []
        elif kind == "scalar_subquery_probe":
            alias = str(op.get("as", ""))
            available = [alias] if alias else []
        elif kind == "window_avg_probe":
            alias = str(op.get("as", ""))
            available = [alias] if alias else []
        elif kind == "struct_distinct_probe":
            alias = str(op.get("as", ""))
            available = [alias] if alias else []
        elif kind == "bit_compare_probe":
            alias = str(op.get("as", ""))
            available = [alias] if alias else []
        elif kind == "round_even_probe":
            alias = str(op.get("as", ""))
            available = [alias] if alias else []
        elif kind == "float_literal_precision_probe":
            alias = str(op.get("as", ""))
            available = [alias] if alias else []
        elif kind == "timestamp_precision_filter_probe":
            alias = str(op.get("as", ""))
            available = [alias] if alias else []
        elif kind == "series_rtruediv_probe":
            alias = str(op.get("as", ""))
            available = [alias] if alias else []
        elif kind == "uint64_isin_probe":
            alias = str(op.get("as", ""))
            available = [alias] if alias else []
        elif kind == "tuple_anti_null_probe":
            alias = str(op.get("as", ""))
            available = [alias] if alias else []
        elif kind == "setop_all_duplicate_probe":
            alias = str(op.get("as", ""))
            available = [alias] if alias else []
        elif kind == "json_predicate_order_probe":
            alias = str(op.get("as", ""))
            available = [alias] if alias else []
        elif kind == "sparse_mask_probe":
            alias = str(op.get("as", ""))
            available = [alias] if alias else []
        elif kind == "float_wrap_probe":
            alias = str(op.get("as", ""))
            available = [alias] if alias else []
        elif kind == "index_bool_probe":
            alias = str(op.get("as", ""))
            available = [alias] if alias else []
        elif kind == "empty_literal_groupby_probe":
            alias = str(op.get("as", ""))
            available = [alias] if alias else []
        elif kind == "arrow_string_eq_sum_probe":
            alias = str(op.get("as", ""))
            available = [alias] if alias else []
        elif kind == "arrow_timestamp_loc_slice_probe":
            alias = str(op.get("as", ""))
            available = [alias] if alias else []
        elif kind == "arrow_timestamp_index_attr_probe":
            alias = str(op.get("as", ""))
            available = [alias] if alias else []
        elif kind == "eval_inplace_alias_probe":
            alias = str(op.get("as", ""))
            available = [alias] if alias else []
        elif kind == "dataset_isin_all_match_probe":
            alias = str(op.get("as", ""))
            available = [alias] if alias else []
        elif kind == "run_end_null_compute_probe":
            alias = str(op.get("as", ""))
            available = [alias] if alias else []
        elif kind == "large_string_partition_probe":
            alias = str(op.get("as", ""))
            available = [alias] if alias else []
        elif kind == "hash_pivot_wider_probe":
            alias = str(op.get("as", ""))
            available = [alias] if alias else []
        elif kind == "rolling_mean_by_null_count_probe":
            alias = str(op.get("as", ""))
            available = [alias] if alias else []
        elif kind == "csv_long_numeric_roundtrip_probe":
            alias = str(op.get("as", ""))
            available = [alias] if alias else []
        elif kind == "groupby":
            available = unique_preserve_order(
                [str(key) for key in op.get("keys", [])]
                + [str(agg.get("as", "")) for agg in op.get("aggs", []) if agg.get("as")]
            )
        elif kind == "aggregate":
            available = unique_preserve_order(
                [str(agg.get("as", "")) for agg in op.get("aggs", []) if agg.get("as")]
            )
    return unique_preserve_order(available)


def _random_mutate_expr(
    rnd: random.Random,
    numeric_cols: list[str],
    string_cols: list[str],
    col_types: dict[str, str],
    profile: GeneratorProfile = "common",
) -> tuple[dict[str, Any], str]:
    choices: list[str] = []
    if numeric_cols:
        choices.extend(["add_const", "arith_const", "cast_float"])
    if string_cols:
        choices.extend(["string_length", "string_lower"])
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
    if kind == "string_length":
        return {"kind": "string_length", "source": rnd.choice(string_cols)}, "int"
    if kind == "string_lower":
        return {"kind": "string_lower", "source": rnd.choice(string_cols)}, "str"
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
    order_pending = False
    pending_order_columns: set[str] = set()
    for op in ops:
        kind = op["op"]
        if kind == "join":
            right = table_by_name.get(op.get("table", ""))
            left_on = op.get("left_on")
            right_on = op.get("right_on")
            if (
                right is None
                or left_on not in available
                or right_on not in {c.name for c in right.columns}
                or op.get("how") not in {"inner", "left"}
            ):
                continue
            repaired.append(op)
            for col in right.columns:
                if col.name == right_on or col.name in available:
                    continue
                available.add(col.name)
                col_types[col.name] = col.type
                if col.type in {"int", "float"}:
                    numeric.add(col.name)
                if col.type == "str":
                    strings.add(col.name)
            order_pending = False
            pending_order_columns = set()
        elif kind == "filter":
            if op["column"] not in available:
                continue
            column_type = col_types.get(op["column"], "")
            if not _filter_literal_is_valid(column_type, op.get("cmp"), op.get("value")):
                continue
            repaired.append(op)
        elif kind == "tuple_absence_filter":
            right = table_by_name.get(str(op.get("table", "")))
            columns = unique_preserve_order([str(column) for column in op.get("columns", [])])
            right_columns = [str(column) for column in op.get("right_columns", [])]
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
            source = str(op.get("source", ""))
            column = str(op.get("column", ""))
            if source not in available or source not in numeric:
                continue
            if not column or is_reserved_output_name(column):
                continue
            partition_by = unique_preserve_order(
                [str(partition_column) for partition_column in op.get("partition_by", []) or []]
            )
            partition_by = [partition_column for partition_column in partition_by if partition_column in available]
            try:
                order_keys = normalize_sort_keys({"keys": op.get("order_by", [])})
            except ValueError:
                continue
            order_keys = _dedupe_sort_keys([key for key in order_keys if key.column in available])
            if not order_keys:
                continue
            repaired_op = {
                "op": "running_sum",
                "source": source,
                "column": column,
                "order_by": [key.to_dict() for key in order_keys],
                "input_dtype": op.get("input_dtype", "float64"),
            }
            if partition_by:
                repaired_op["partition_by"] = partition_by
            repaired.append(repaired_op)
            available.add(column)
            col_types[column] = "float"
            numeric.add(column)
            strings.discard(column)
            order_pending = True
            pending_order_columns = {key.column for key in order_keys}
        elif kind == "row_number_filter":
            partition_by = unique_preserve_order(
                [str(partition_column) for partition_column in op.get("partition_by", []) or []]
            )
            partition_by = [partition_column for partition_column in partition_by if partition_column in available]
            try:
                order_keys = normalize_sort_keys({"keys": op.get("order_by", [])})
            except ValueError:
                continue
            order_keys = _dedupe_sort_keys([key for key in order_keys if key.column in available])
            if not order_keys:
                continue
            comparator = str(op.get("cmp", "=="))
            if comparator not in {"==", "<", "<="}:
                continue
            try:
                value = int(op.get("value", 1))
            except (TypeError, ValueError):
                continue
            if value <= 0:
                continue
            repaired_op = {
                "op": "row_number_filter",
                "partition_by": partition_by,
                "order_by": [key.to_dict() for key in order_keys],
                "cmp": comparator,
                "value": value,
            }
            repaired.append(repaired_op)
            order_pending = True
            pending_order_columns = {key.column for key in order_keys} | set(partition_by)
        elif kind == "sortedness_check":
            column = str(op.get("column", ""))
            alias = str(op.get("as", ""))
            ascending = op.get("ascending", True)
            nulls = str(op.get("nulls", "last"))
            if column not in available:
                continue
            if not alias or is_reserved_output_name(alias):
                continue
            if not isinstance(ascending, bool) or nulls not in {"first", "last"}:
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
            alias = str(op.get("as", ""))
            if not alias or is_reserved_output_name(alias):
                continue
            try:
                row_count = int(op.get("rows", 100_000))
                branch_count = int(op.get("branches", 3))
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
            alias = str(op.get("as", ""))
            if not alias or is_reserved_output_name(alias):
                continue
            values = op.get("values", [1, 2, 3])
            quantiles = op.get("quantiles", [0.0, 0.5, 1.0])
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
            alias = str(op.get("as", ""))
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
            alias = str(op.get("as", ""))
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
            alias = str(op.get("as", ""))
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
            alias = str(op.get("as", ""))
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
            alias = str(op.get("as", ""))
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
            alias = str(op.get("as", ""))
            literal = str(op.get("literal", ""))
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
            alias = str(op.get("as", ""))
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
            alias = str(op.get("as", ""))
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
            alias = str(op.get("as", ""))
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
            alias = str(op.get("as", ""))
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
            alias = str(op.get("as", ""))
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
            alias = str(op.get("as", ""))
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
            alias = str(op.get("as", ""))
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
            alias = str(op.get("as", ""))
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
            alias = str(op.get("as", ""))
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
            alias = str(op.get("as", ""))
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
            alias = str(op.get("as", ""))
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
            alias = str(op.get("as", ""))
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
            alias = str(op.get("as", ""))
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
            alias = str(op.get("as", ""))
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
            alias = str(op.get("as", ""))
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
            alias = str(op.get("as", ""))
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
            alias = str(op.get("as", ""))
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
            alias = str(op.get("as", ""))
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
            alias = str(op.get("as", ""))
            if not alias or is_reserved_output_name(alias):
                continue
            repaired.append({"op": "hash_pivot_wider_probe", "as": alias})
            available = {alias}
            col_types = {alias: "bool"}
            numeric = set()
            strings = set()
            order_pending = False
            pending_order_columns = set()
        elif kind == "rolling_mean_by_null_count_probe":
            alias = str(op.get("as", ""))
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
            alias = str(op.get("as", ""))
            if not alias or is_reserved_output_name(alias):
                continue
            values = [str(value).strip() for value in op.get("values", []) or []]
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
            cols = unique_preserve_order([c for c in op["columns"] if c in available])
            if not cols:
                continue
            repaired.append({"op": "select", "columns": cols})
            available = set(cols)
            numeric &= available
            strings &= available
        elif kind == "sort":
            try:
                keys = normalize_sort_keys(op)
            except ValueError:
                continue
            keys = _dedupe_sort_keys([key for key in keys if key.column in available])
            if keys:
                existing = {key.column for key in keys}
                tail = [SortKey(column=c) for c in sorted(c for c in available if c not in existing)]
                full_keys = keys + tail
                if "keys" in op:
                    repaired.append({"op": "sort", "keys": [key.to_dict() for key in full_keys]})
                else:
                    repaired.append({**op, "columns": [key.column for key in full_keys]})
                order_pending = True
                pending_order_columns = {key.column for key in full_keys}
        elif kind == "limit":
            if order_pending:
                repaired.append(op)
            break
        elif kind == "offset":
            if order_pending:
                repaired.append({"op": "offset", "n": max(0, int(op.get("n", 0)))})
        elif kind == "mutate":
            expr = op["expr"]
            out_type = _mutate_output_type(expr, available, numeric, strings, col_types)
            if out_type is None:
                continue
            column = str(op["column"])
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
            keys = unique_preserve_order([k for k in op["keys"] if k in available])
            aggs = [
                a
                for a in op["aggs"]
                if a["column"] in available
                and _aggregate_accepts_type(
                    col_types.get(str(a["column"]), "derived"),
                    str(a["func"]),
                    a["column"] in numeric,
                )
            ]
            unique_aggs: list[dict[str, Any]] = []
            seen_aliases: set[str] = set()
            for agg in aggs:
                alias = str(agg.get("as", ""))
                if not alias or alias in seen_aliases or alias in keys or is_reserved_output_name(alias):
                    continue
                seen_aliases.add(alias)
                unique_aggs.append(agg)
            aggs = unique_aggs
            if not keys or not aggs:
                continue
            repaired.append({**op, "keys": keys, "aggs": aggs})
            available = set(keys) | {a["as"] for a in aggs}
            numeric = {k for k in keys if col_types.get(k) in {"int", "float"}}
            strings = {k for k in keys if col_types.get(k) == "str"}
            for agg in aggs:
                output_type = _aggregate_output_type(
                    col_types.get(str(agg["column"]), "float"),
                    str(agg["func"]),
                )
                col_types[agg["as"]] = output_type
                if output_type in {"int", "float"}:
                    numeric.add(agg["as"])
            order_pending = False
            pending_order_columns = set()
        elif kind == "aggregate":
            aggs = [
                a
                for a in op["aggs"]
                if a["column"] in available
                and _aggregate_accepts_type(
                    col_types.get(str(a["column"]), "derived"),
                    str(a["func"]),
                    a["column"] in numeric,
                )
            ]
            unique_aggs = []
            seen_aliases: set[str] = set()
            for agg in aggs:
                alias = str(agg.get("as", ""))
                if not alias or alias in seen_aliases or is_reserved_output_name(alias):
                    continue
                seen_aliases.add(alias)
                unique_aggs.append(agg)
            if not unique_aggs:
                continue
            repaired.append({**op, "aggs": unique_aggs})
            available = {a["as"] for a in unique_aggs}
            numeric = set()
            strings = set()
            for agg in unique_aggs:
                output_type = _aggregate_output_type(
                    col_types.get(str(agg["column"]), "float"),
                    str(agg["func"]),
                )
                col_types[agg["as"]] = output_type
                if output_type in {"int", "float"}:
                    numeric.add(agg["as"])
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
    kind = expr.get("kind")
    src = expr.get("source")
    if src not in available:
        return None
    if kind == "add_const":
        return col_types[src] if src in numeric else None
    if kind == "arith_const":
        if src not in numeric or expr.get("op") not in {"sub", "mul", "div", "mod"}:
            return None
        if expr.get("op") in {"div", "mod"} and expr.get("value") == 0:
            return None
        return "float" if expr.get("op") == "div" or col_types[src] == "float" else col_types[src]
    if kind == "reverse_division_columns":
        numerator = expr.get("numerator")
        if src not in numeric or numerator not in numeric:
            return None
        return "float"
    if kind == "cast":
        return "float" if src in numeric and expr.get("to") == "float" else None
    if kind == "string_length":
        return "int" if src in strings else None
    if kind == "string_lower":
        return "str" if src in strings else None
    if kind == "string_basename":
        return "str" if src in strings else None
    return None


def _filter_literal_is_valid(column_type: str, comparator: Any, value: Any) -> bool:
    if not filter_comparator_supports_type(column_type, comparator):
        return False
    parsed = parse_filter_comparator(comparator)
    if comparator == "in_set":
        if not isinstance(value, list) or not value or any(item is None for item in value):
            return False
        return all(_filter_literal_is_valid(column_type, "==", item) for item in value)
    if parsed is not None and parsed.base == "range_closed":
        if not isinstance(value, list) or len(value) != 2 or any(item is None for item in value):
            return False
        if not all(_filter_literal_is_valid(column_type, "==", item) for item in value):
            return False
        return value[0] <= value[1]
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


def generate_case(seed: int, type_aware: bool = True, profile: GeneratorProfile = "common") -> Case:
    if profile == "bughunt" and type_aware:
        mixed = _bughunt_issue_inspired_case(seed)
        if mixed is not None:
            return mixed
    if profile == "null_groupby_topk" and type_aware:
        return generate_null_groupby_topk_case(seed)
    if profile == "null_agg_topk" and type_aware:
        return generate_null_agg_topk_case(seed)
    if profile == "filter_null_agg_topk" and type_aware:
        return generate_filter_null_agg_topk_case(seed)
    if profile == "join_null_agg_topk" and type_aware:
        return generate_join_null_agg_topk_case(seed)
    if profile == "join_null_key_topk" and type_aware:
        return generate_join_null_key_topk_case(seed)
    if profile == "wide_offset_topk" and type_aware:
        return generate_wide_offset_topk_case(seed)
    if profile == "empty_filter_groupby" and type_aware:
        return generate_empty_filter_groupby_case(seed)
    if profile == "join_filter_groupby" and type_aware:
        return generate_join_filter_groupby_case(seed)
    if profile == "join_null_truth_filter" and type_aware:
        return generate_join_null_truth_filter_case(seed)
    if profile == "join_groupby_stress" and type_aware:
        return generate_join_groupby_stress_case(seed)
    if profile == "storage_offset" and type_aware:
        return generate_storage_offset_case(seed)
    if profile == "float_group_key" and type_aware:
        return generate_float_group_key_case(seed)
    if profile == "join_null_sort" and type_aware:
        return generate_join_null_sort_case(seed)
    if profile == "ordered_groupby_sort" and type_aware:
        return generate_ordered_groupby_sort_case(seed)
    if profile == "topk_resort" and type_aware:
        return generate_topk_resort_case(seed)
    if profile == "join_ordered_agg_topk" and type_aware:
        return generate_join_ordered_agg_topk_case(seed)
    if profile == "global_null_aggregate" and type_aware:
        return generate_global_null_aggregate_case(seed)
    if profile == "string_count_groupby" and type_aware:
        return generate_string_count_groupby_case(seed)
    if profile == "unique_count_groupby" and type_aware:
        return generate_unique_count_groupby_case(seed)
    if profile == "bool_null_groupby_agg" and type_aware:
        return generate_bool_null_groupby_agg_case(seed)
    if profile == "large_int_filter_groupby" and type_aware:
        return generate_large_int_filter_groupby_case(seed)
    if profile == "set_membership_filter" and type_aware:
        return generate_set_membership_filter_case(seed)
    if profile == "pyarrow_groupby_filter_cast_membership" and type_aware:
        return generate_pyarrow_groupby_filter_cast_membership_case(seed)
    if profile == "null_predicate_filter" and type_aware:
        return generate_null_predicate_filter_case(seed)
    if profile == "boolean_predicate_filter" and type_aware:
        return generate_boolean_predicate_filter_case(seed)
    if profile == "post_topk_range_filter" and type_aware:
        return generate_post_topk_range_filter_case(seed)
    if profile == "tuple_absence_filter" and type_aware:
        return generate_tuple_absence_filter_case(seed)
    if profile == "row_value_absence_filter" and type_aware:
        return generate_row_value_absence_filter_case(seed)
    if profile == "running_sum_precision" and type_aware:
        return generate_running_sum_precision_case(seed)
    if profile == "partitioned_running_sum" and type_aware:
        return generate_partitioned_running_sum_case(seed)
    if profile == "path_basename_keyed_pick" and type_aware:
        return generate_path_basename_keyed_pick_case(seed)
    if profile == "sortedness_null_placement" and type_aware:
        return generate_sortedness_null_placement_case(seed)
    if profile == "simple_case_random_subject" and type_aware:
        return generate_simple_case_random_subject_case(seed)
    if profile == "group_quantile_key_probe" and type_aware:
        return generate_group_quantile_key_probe_case(seed)
    if profile == "scalar_subquery_double_parentheses" and type_aware:
        return generate_scalar_subquery_double_parentheses_case(seed)
    if profile == "window_avg_rows_frame" and type_aware:
        return generate_window_avg_rows_frame_case(seed)
    if profile == "struct_distinct_unnest" and type_aware:
        return generate_struct_distinct_unnest_case(seed)
    if profile == "bit_compare_unequal_length" and type_aware:
        return generate_bit_compare_unequal_length_case(seed)
    if profile == "round_even_float_scale" and type_aware:
        return generate_round_even_float_scale_case(seed)
    if profile == "duckdb_float_literal_precision" and type_aware:
        return generate_duckdb_float_literal_precision_case(seed)
    if profile == "polars_timestamp_precision_filter" and type_aware:
        return generate_polars_timestamp_precision_filter_case(seed)
    if profile == "series_rtruediv_operand_order" and type_aware:
        return generate_series_rtruediv_operand_order_case(seed)
    if profile == "polars_reverse_division_columns" and type_aware:
        return generate_polars_reverse_division_columns_case(seed)
    if profile == "pandas_uint64_isin_precision" and type_aware:
        return generate_pandas_uint64_isin_precision_case(seed)
    if profile == "duckdb_tuple_anti_null_semantics" and type_aware:
        return generate_duckdb_tuple_anti_null_semantics_case(seed)
    if profile == "datafusion_setop_all_duplicate_count" and type_aware:
        return generate_datafusion_setop_all_duplicate_count_case(seed)
    if profile == "duckdb_json_predicate_order_semantics" and type_aware:
        return generate_duckdb_json_predicate_order_semantics_case(seed)
    if profile == "pandas_sparse_array_mask_semantics" and type_aware:
        return generate_pandas_sparse_array_mask_semantics_case(seed)
    if profile == "polars_float_wrap_numerical_semantics" and type_aware:
        return generate_polars_float_wrap_numerical_semantics_case(seed)
    if profile == "pandas_index_bool_result_type" and type_aware:
        return generate_pandas_index_bool_result_type_case(seed)
    if profile == "polars_empty_literal_groupby_semantics" and type_aware:
        return generate_polars_empty_literal_groupby_semantics_case(seed)
    if profile == "pandas_arrow_string_eq_sum_semantics" and type_aware:
        return generate_pandas_arrow_string_eq_sum_semantics_case(seed)
    if profile == "pandas_arrow_timestamp_loc_slice_semantics" and type_aware:
        return generate_pandas_arrow_timestamp_loc_slice_semantics_case(seed)
    if profile == "pandas_arrow_timestamp_index_attr_semantics" and type_aware:
        return generate_pandas_arrow_timestamp_index_attr_semantics_case(seed)
    if profile == "pandas_eval_inplace_aliasing_semantics" and type_aware:
        return generate_pandas_eval_inplace_aliasing_semantics_case(seed)
    if profile == "pandas_bool_reduction_skipna_semantics" and type_aware:
        return generate_pandas_bool_reduction_skipna_semantics_case(seed)
    if profile == "pyarrow_dataset_isin_all_match_semantics" and type_aware:
        return generate_pyarrow_dataset_isin_all_match_semantics_case(seed)
    if profile == "pyarrow_run_end_null_compute_semantics" and type_aware:
        return generate_pyarrow_run_end_null_compute_semantics_case(seed)
    if profile == "pyarrow_large_string_partition_schema_semantics" and type_aware:
        return generate_pyarrow_large_string_partition_schema_semantics_case(seed)
    if profile == "pyarrow_hash_pivot_wider_order_semantics" and type_aware:
        return generate_pyarrow_hash_pivot_wider_order_semantics_case(seed)
    if profile == "polars_rolling_mean_by_null_count_semantics" and type_aware:
        return generate_polars_rolling_mean_by_null_count_semantics_case(seed)
    if profile == "csv_long_numeric_roundtrip" and type_aware:
        return generate_csv_long_numeric_roundtrip_case(seed)
    if profile == "workflow" and type_aware:
        return generate_workflow_case(seed)
    if profile == "bughunt_no_groupby" and type_aware:
        mixed = _bughunt_no_groupby_issue_inspired_case(seed)
        if mixed is not None:
            return mixed
    if profile == "issue_focus" and type_aware:
        return _issue_focus_case(seed)
    bughunt_profile = _is_bughunt_profile(profile)
    table = generate_table(
        seed,
        name="t0",
        min_rows=8 if bughunt_profile else 0,
        max_rows=30 if bughunt_profile else 20,
        profile=profile,
    )
    rnd = random.Random(seed * 15485863 + 11)
    join_probability = 0.85 if bughunt_profile else 0.4
    extra_tables = [generate_join_table(seed, profile=profile)] if type_aware and rnd.random() < join_probability else []
    if bughunt_profile and extra_tables:
        extra_tables = [_cover_join_table_keys(table, extra_tables[0])]
    program = generate_program(
        seed,
        table,
        max_ops=8 if bughunt_profile else 6,
        type_aware=type_aware,
        extra_tables=extra_tables,
        profile=profile,
    )
    suffix = (
        "-bughunt"
        if profile == "bughunt"
        else "-bughunt-fresh"
        if profile == "bughunt_fresh"
        else "-bughunt-no-groupby"
        if profile == "bughunt_no_groupby"
        else "-issue-focus"
        if profile == "issue_focus"
        else ""
    )
    return Case(case_id=f"case-{seed:08d}{suffix}", seed=seed, tables=[table] + extra_tables, program=program)


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
    issue_case = _bughunt_issue_inspired_case(seed)
    if issue_case is not None:
        mixed_profile = str(issue_case.metadata.get("mixed_generator_profile", "issue_inspired"))
        return _as_bughunt_mixed_case(
            issue_case,
            seed,
            mixed_profile,
            generator_profile="issue_focus",
        )
    start_index = seed % len(ISSUE_FOCUS_MIXED_PROFILES)
    for offset in range(len(ISSUE_FOCUS_MIXED_PROFILES)):
        mixed_profile = ISSUE_FOCUS_MIXED_PROFILES[(start_index + offset) % len(ISSUE_FOCUS_MIXED_PROFILES)]
        case = generate_case(seed + offset, profile=mixed_profile)  # type: ignore[arg-type]
        return _as_bughunt_mixed_case(
            case,
            seed,
            mixed_profile,
            generator_profile="issue_focus",
        )
    fallback = generate_case(seed, profile="bughunt_fresh")
    return _as_bughunt_mixed_case(
        fallback,
        seed,
        "bughunt_fresh",
        generator_profile="issue_focus",
    )


def _bughunt_issue_inspired_case(seed: int) -> Case | None:
    if seed % 211 == 111:
        return _as_bughunt_mixed_case(
            generate_duckdb_float_literal_precision_case(seed),
            seed,
            "duckdb_float_literal_precision",
        )
    if seed % 227 == 114:
        return _as_bughunt_mixed_case(
            generate_polars_timestamp_precision_filter_case(seed),
            seed,
            "polars_timestamp_precision_filter",
        )
    selector = seed % 60
    if selector == 2:
        return _as_bughunt_mixed_case(generate_join_null_truth_filter_case(seed), seed, "join_null_truth_filter")
    if selector == 5:
        return _as_bughunt_mixed_case(generate_global_null_aggregate_case(seed), seed, "global_null_aggregate")
    if selector == 11:
        return _as_bughunt_mixed_case(generate_empty_filter_groupby_case(seed), seed, "empty_filter_groupby")
    if selector == 20:
        return _as_bughunt_mixed_case(generate_wide_offset_topk_case(seed), seed, "wide_offset_topk")
    if selector == 31:
        return _as_bughunt_mixed_case(generate_ordered_groupby_sort_case(seed), seed, "ordered_groupby_sort")
    if selector == 40:
        return _as_bughunt_mixed_case(generate_join_null_key_topk_case(seed), seed, "join_null_key_topk")
    if selector == 47:
        return _as_bughunt_mixed_case(generate_string_count_groupby_case(seed), seed, "string_count_groupby")
    if selector == 50:
        return _as_bughunt_mixed_case(generate_set_membership_filter_case(seed), seed, "set_membership_filter")
    if selector == 51:
        return _as_bughunt_mixed_case(generate_null_predicate_filter_case(seed), seed, "null_predicate_filter")
    if selector == 52:
        return _as_bughunt_mixed_case(
            generate_polars_reverse_division_columns_case(seed),
            seed,
            "polars_reverse_division_columns",
        )
    if selector == 54:
        return _as_bughunt_mixed_case(generate_boolean_predicate_filter_case(seed), seed, "boolean_predicate_filter")
    if selector == 55:
        return _as_bughunt_mixed_case(generate_post_topk_range_filter_case(seed), seed, "post_topk_range_filter")
    if selector == 56:
        return _as_bughunt_mixed_case(generate_unique_count_groupby_case(seed), seed, "unique_count_groupby")
    if selector == 57:
        return _as_bughunt_mixed_case(generate_tuple_absence_filter_case(seed), seed, "tuple_absence_filter")
    if selector == 53:
        return _as_bughunt_mixed_case(generate_topk_resort_case(seed), seed, "topk_resort")
    if selector == 58:
        return _as_bughunt_mixed_case(generate_join_ordered_agg_topk_case(seed), seed, "join_ordered_agg_topk")
    if selector == 59:
        return _as_bughunt_mixed_case(generate_running_sum_precision_case(seed), seed, "running_sum_precision")
    if seed % 71 == 60:
        return _as_bughunt_mixed_case(generate_sortedness_null_placement_case(seed), seed, "sortedness_null_placement")
    if seed % 73 == 61:
        return _as_bughunt_mixed_case(generate_simple_case_random_subject_case(seed), seed, "simple_case_random_subject")
    if seed % 79 == 63:
        return _as_bughunt_mixed_case(generate_group_quantile_key_probe_case(seed), seed, "group_quantile_key_probe")
    if seed % 83 == 64:
        return _as_bughunt_mixed_case(
            generate_scalar_subquery_double_parentheses_case(seed),
            seed,
            "scalar_subquery_double_parentheses",
        )
    if seed % 89 == 66:
        return _as_bughunt_mixed_case(generate_window_avg_rows_frame_case(seed), seed, "window_avg_rows_frame")
    if seed % 97 == 67:
        return _as_bughunt_mixed_case(generate_struct_distinct_unnest_case(seed), seed, "struct_distinct_unnest")
    if seed % 101 == 68:
        return _as_bughunt_mixed_case(
            generate_bit_compare_unequal_length_case(seed),
            seed,
            "bit_compare_unequal_length",
        )
    if seed % 103 == 69:
        return _as_bughunt_mixed_case(generate_round_even_float_scale_case(seed), seed, "round_even_float_scale")
    if seed % 107 == 70:
        return _as_bughunt_mixed_case(
            generate_series_rtruediv_operand_order_case(seed),
            seed,
            "series_rtruediv_operand_order",
        )
    if seed % 109 == 72:
        return _as_bughunt_mixed_case(
            generate_pandas_uint64_isin_precision_case(seed),
            seed,
            "pandas_uint64_isin_precision",
        )
    if seed % 113 == 73:
        return _as_bughunt_mixed_case(
            generate_duckdb_tuple_anti_null_semantics_case(seed),
            seed,
            "duckdb_tuple_anti_null_semantics",
        )
    if seed % 199 == 102:
        return _as_bughunt_mixed_case(
            generate_datafusion_setop_all_duplicate_count_case(seed),
            seed,
            "datafusion_setop_all_duplicate_count",
        )
    if seed % 181 == 105:
        return _as_bughunt_mixed_case(
            generate_duckdb_json_predicate_order_semantics_case(seed),
            seed,
            "duckdb_json_predicate_order_semantics",
        )
    if seed % 181 == 101:
        return _as_bughunt_mixed_case(
            generate_row_value_absence_filter_case(seed),
            seed,
            "row_value_absence_filter",
        )
    if seed % 127 == 74:
        return _as_bughunt_mixed_case(
            generate_pandas_sparse_array_mask_semantics_case(seed),
            seed,
            "pandas_sparse_array_mask_semantics",
        )
    if seed % 131 == 75:
        return _as_bughunt_mixed_case(
            generate_polars_float_wrap_numerical_semantics_case(seed),
            seed,
            "polars_float_wrap_numerical_semantics",
        )
    if seed % 137 == 76:
        return _as_bughunt_mixed_case(
            generate_pandas_index_bool_result_type_case(seed),
            seed,
            "pandas_index_bool_result_type",
        )
    if seed % 139 == 77:
        return _as_bughunt_mixed_case(
            generate_polars_empty_literal_groupby_semantics_case(seed),
            seed,
            "polars_empty_literal_groupby_semantics",
        )
    if seed % 149 == 78:
        return _as_bughunt_mixed_case(
            generate_pandas_arrow_string_eq_sum_semantics_case(seed),
            seed,
            "pandas_arrow_string_eq_sum_semantics",
        )
    if seed % 151 == 79:
        return _as_bughunt_mixed_case(
            generate_pandas_arrow_timestamp_loc_slice_semantics_case(seed),
            seed,
            "pandas_arrow_timestamp_loc_slice_semantics",
        )
    if seed % 157 == 81:
        return _as_bughunt_mixed_case(
            generate_pandas_arrow_timestamp_index_attr_semantics_case(seed),
            seed,
            "pandas_arrow_timestamp_index_attr_semantics",
        )
    if seed % 181 == 108:
        return _as_bughunt_mixed_case(
            generate_pandas_eval_inplace_aliasing_semantics_case(seed),
            seed,
            "pandas_eval_inplace_aliasing_semantics",
        )
    if seed % 193 == 110:
        return _as_bughunt_mixed_case(
            generate_pandas_bool_reduction_skipna_semantics_case(seed),
            seed,
            "pandas_bool_reduction_skipna_semantics",
        )
    if seed % 163 == 82:
        return _as_bughunt_mixed_case(
            generate_pyarrow_dataset_isin_all_match_semantics_case(seed),
            seed,
            "pyarrow_dataset_isin_all_match_semantics",
        )
    if seed % 197 == 112:
        return _as_bughunt_mixed_case(
            generate_pyarrow_run_end_null_compute_semantics_case(seed),
            seed,
            "pyarrow_run_end_null_compute_semantics",
        )
    if seed % 173 == 104:
        return _as_bughunt_mixed_case(
            generate_pyarrow_large_string_partition_schema_semantics_case(seed),
            seed,
            "pyarrow_large_string_partition_schema_semantics",
        )
    if seed % 191 == 106:
        return _as_bughunt_mixed_case(
            generate_pyarrow_hash_pivot_wider_order_semantics_case(seed),
            seed,
            "pyarrow_hash_pivot_wider_order_semantics",
        )
    if seed % 167 == 83:
        return _as_bughunt_mixed_case(
            generate_polars_rolling_mean_by_null_count_semantics_case(seed),
            seed,
            "polars_rolling_mean_by_null_count_semantics",
        )
    if seed % 197 == 109:
        return _as_bughunt_mixed_case(
            generate_pyarrow_groupby_filter_cast_membership_case(seed),
            seed,
            "pyarrow_groupby_filter_cast_membership",
        )
    if seed % 223 == 103:
        return _as_bughunt_mixed_case(generate_partitioned_running_sum_case(seed), seed, "partitioned_running_sum")
    if seed % 229 == 107:
        return _as_bughunt_mixed_case(
            generate_path_basename_keyed_pick_case(seed),
            seed,
            "path_basename_keyed_pick",
        )
    if seed % 251 == 48:
        return _as_bughunt_mixed_case(
            generate_csv_long_numeric_roundtrip_case(seed),
            seed,
            "csv_long_numeric_roundtrip",
        )
    return None


def _bughunt_no_groupby_issue_inspired_case(seed: int) -> Case | None:
    if seed % 53 == 18:
        return _as_bughunt_mixed_case(
            generate_datafusion_setop_all_duplicate_count_case(seed),
            seed,
            "datafusion_setop_all_duplicate_count",
            generator_profile="bughunt_no_groupby",
        )
    return None


def _as_bughunt_mixed_case(
    case: Case,
    seed: int,
    mixed_profile: str,
    *,
    generator_profile: str = "bughunt",
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
