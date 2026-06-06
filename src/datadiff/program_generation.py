from __future__ import annotations

from itertools import combinations
import math
import random
from typing import Any

from .dsl import Program, SortKey, TableData, coerce_expression, normalize_sort_keys
from .expression_semantics import aggregate_output_type, cast_output_type, expr_output_type, literal_output_type
from .filtering import filter_comparator_supports_type, parse_filter_comparator
from .identifiers import is_reserved_output_name, make_safe_output_name
from .join_keys import join_key_arg, join_key_pairs
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
    op_nulls,
    op_output_alias,
    op_partition_columns,
    op_quantiles,
    op_right_columns,
    op_rows,
    op_table,
    op_value,
    op_values,
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
from .string_literals import (
    string_pattern_literal_for_column,
    string_pattern_literal_for_type,
)
from .util import unique_preserve_order

GeneratorProfile = str


def _is_discovery_profile(profile: str) -> bool:
    return profile in {"discovery", "discovery_fresh", "discovery_no_groupby", "issue_focus", "deep_probe_rotation"}


def _discovery_profile_allows_groupby(profile: str) -> bool:
    return profile in {"discovery", "discovery_fresh", "issue_focus", "deep_probe_rotation"}


def _aggregate_accepts_type(source_type: str, func: str, is_numeric_column: bool) -> bool:
    return aggregate_accepts_type(source_type, func, is_numeric_column)


def _aggregate_output_type(source_type: str, func: str) -> str:
    return aggregate_output_type(source_type, func)


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
    return string_pattern_literal_for_column(rnd, table, col, comparator)


def _literal_for_type(rnd: random.Random, typ: str) -> Any:
    if typ == "int":
        return rnd.choice([-10, -1, 0, 1, 2, 10])
    if typ == "float":
        return rnd.choice([-1.0, 0.0, 0.5, 1.0, 10.0])
    if typ == "bool":
        return rnd.choice([True, False])
    return rnd.choice(["", "alpha", "beta", "中文", "missing"])


def _string_pattern_literal_for_type(rnd: random.Random, typ: str) -> str:
    return string_pattern_literal_for_type(rnd, typ)


def _looks_like_date_column(column: str) -> bool:
    lowered = column.lower()
    return lowered in {"dt", "date", "event_date", "timestamp"} or lowered.endswith(("_dt", "_date"))


def _looks_like_numeric_string_column(column: str) -> bool:
    lowered = column.lower()
    return lowered in {"num_s", "number_text", "numeric_text"} or lowered.endswith(
        ("_num_s", "_number_text", "_numeric_text")
    )


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
) -> list[tuple[TableData, list[str], list[str]]]:
    pairs: list[tuple[TableData, list[str], list[str]]] = []
    seen: set[tuple[str, tuple[str, ...], tuple[str, ...]]] = set()
    for extra in extra_tables:
        right_types = {column.name: column.type for column in extra.columns}
        matching = [
            column
            for column in available_cols
            if right_types.get(column) is not None and right_types.get(column) == col_types.get(column)
        ]
        for width in range(1, min(2, len(matching)) + 1):
            for key_columns in combinations(matching, width):
                left_columns = list(key_columns)
                right_columns = list(key_columns)
                key = (extra.name, tuple(left_columns), tuple(right_columns))
                if key in seen:
                    continue
                seen.add(key)
                pairs.append((extra, left_columns, right_columns))
    return pairs


def _choose_compatible_key_pair(
    pairs: list[tuple[TableData, list[str], list[str]]],
    rnd: random.Random,
    *,
    prefer_multi_key: bool,
) -> tuple[TableData, list[str], list[str]]:
    if prefer_multi_key:
        multi_key_pairs = [pair for pair in pairs if len(pair[1]) > 1]
        if multi_key_pairs and rnd.random() < 0.65:
            return rnd.choice(multi_key_pairs)
    return rnd.choice(pairs)


def _coalesce_column_groups(available_cols: list[str], col_types: dict[str, str]) -> list[tuple[str, list[str]]]:
    groups: dict[str, list[str]] = {}
    for column in available_cols:
        typ = col_types.get(column)
        if typ is not None:
            groups.setdefault(typ, []).append(column)
    return [(typ, columns) for typ, columns in groups.items() if len(columns) >= 2]


def _preferred_comparable_column(
    rnd: random.Random,
    comparable_cols: list[str],
    *,
    derived_cols: set[str],
    nullable_cols: set[str],
    prefer_derived: bool,
) -> str:
    derived_candidates = [column for column in comparable_cols if column in derived_cols]
    if prefer_derived and derived_candidates and rnd.random() < 0.70:
        return rnd.choice(derived_candidates)
    nullable_candidates = [column for column in comparable_cols if column in nullable_cols]
    if nullable_candidates and rnd.random() < 0.35:
        return rnd.choice(nullable_candidates)
    return rnd.choice(comparable_cols)


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
        op_pool.extend(["semi_join", "anti_join", "tuple_absence_filter"])
    discovery_profile = _is_discovery_profile(profile)
    if discovery_profile:
        op_pool = [
            "filter",
            "filter",
            "drop_nulls",
            "mutate",
            "mutate",
            "row_number_filter",
            "running_sum",
            "sort",
            "limit",
            "offset",
            "select",
            "distinct",
            "fill_null",
            "coalesce",
            "case_when",
            "tuple_absence_filter",
        ]
        if compatible_union_tables:
            op_pool.append("union_all")
        if compatible_membership_pairs:
            op_pool.extend(["semi_join", "semi_join", "anti_join", "tuple_absence_filter"])
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
    derived_cols: set[str] = set()
    base_columns = {column.name for column in table.columns}
    nullable_cols = {
        column.name
        for column in table.columns
        if column.nullable or any(row.get(column.name) is None for row in table.rows if column.name in row)
    }

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
        if {"semi_join", "anti_join", "tuple_absence_filter"} & set(possible):
            compatible_membership_pairs = _compatible_membership_key_pairs(extra_tables, available_cols, col_types)
            if not compatible_membership_pairs:
                possible = [p for p in possible if p not in {"semi_join", "anti_join", "tuple_absence_filter"}]
        remaining = nops - index
        if discovery_profile and not grouped:
            derived_numeric = [column for column in numeric_cols if column in derived_cols]
            derived_nullable = [column for column in comparable_cols if column in derived_cols and column in nullable_cols]
            if extra_tables and not joined_tables and compatible_membership_pairs and (not ops or rnd.random() < 0.8):
                op = "join"
            elif "mutate" not in emitted_ops and (numeric_cols or bool_cols or string_cols) and len(ops) >= int(bool(extra_tables)):
                op = "mutate"
            elif (
                derived_numeric
                and "running_sum" in possible
                and "running_sum" not in emitted_ops
                and remaining > 1
            ):
                op = "running_sum"
            elif (
                derived_nullable
                and "case_when" in possible
                and "case_when" not in emitted_ops
                and remaining > 1
            ):
                op = "case_when"
            elif (
                derived_cols
                and "filter" in possible
                and any(column in comparable_cols for column in derived_cols)
                and remaining > 1
                and rnd.random() < 0.45
            ):
                op = "filter"
            elif "filter" not in emitted_ops and comparable_cols and len(ops) >= 2 and remaining > 2:
                op = "filter"
            elif (
                "tuple_absence_filter" not in emitted_ops
                and compatible_membership_pairs
                and len(ops) >= int(bool(extra_tables))
                and remaining > 2
                and rnd.random() < 0.28
            ):
                op = "tuple_absence_filter"
            elif "row_number_filter" not in emitted_ops and comparable_cols and len(ops) >= 2 and remaining > 1 and rnd.random() < 0.35:
                op = "row_number_filter"
            elif "running_sum" not in emitted_ops and numeric_cols and len(ops) >= 2 and remaining > 1 and rnd.random() < 0.30:
                op = "running_sum"
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
        elif profile == "edge_float" and not grouped:
            if "mutate" not in emitted_ops and numeric_cols and rnd.random() < 0.75:
                op = "mutate"
            elif (
                derived_cols
                and "filter" in possible
                and any(column in comparable_cols for column in derived_cols)
                and rnd.random() < 0.25
            ):
                op = "filter"
            else:
                op = rnd.choice(possible)
        else:
            op = rnd.choice(possible)

        if op == "join" and extra_tables and not grouped:
            join_candidates = [
                pair
                for pair in compatible_membership_pairs
                if pair[0].name not in joined_tables
            ]
            if join_candidates:
                right, left_on, right_on = _choose_compatible_key_pair(
                    join_candidates,
                    rnd,
                    prefer_multi_key=discovery_profile,
                )
                ops.append(
                    {
                        "op": "join",
                        "table": right.name,
                        "left_on": join_key_arg(left_on),
                        "right_on": join_key_arg(right_on),
                        "how": rnd.choice(["inner", "left"]),
                    }
                )
                joined_tables.add(right.name)
                right_key_set = set(right_on)
                for col in right.columns:
                    if col.name in right_key_set or col.name in available_cols:
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
            right, left_on, right_on = _choose_compatible_key_pair(
                compatible_membership_pairs,
                rnd,
                prefer_multi_key=discovery_profile,
            )
            ops.append(
                {
                    "op": op,
                    "table": right.name,
                    "left_on": join_key_arg(left_on),
                    "right_on": join_key_arg(right_on),
                }
            )

        elif op == "tuple_absence_filter" and compatible_membership_pairs and not grouped:
            right, left_columns, right_columns = _choose_compatible_key_pair(
                compatible_membership_pairs,
                rnd,
                prefer_multi_key=True,
            )
            ops.append(
                {
                    "op": "tuple_absence_filter",
                    "columns": left_columns,
                    "table": right.name,
                    "right_columns": right_columns,
                }
            )

        elif op == "drop_nulls" and available_cols and not grouped:
            width = rnd.randint(1, min(3, len(available_cols)))
            ops.append({"op": "drop_nulls", "columns": rnd.sample(available_cols, k=width)})

        elif op == "filter" and comparable_cols and not grouped:
            col = _preferred_comparable_column(
                rnd,
                comparable_cols,
                derived_cols=derived_cols,
                nullable_cols=nullable_cols,
                prefer_derived=(discovery_profile or profile == "edge_float"),
            )
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
            if (discovery_profile or profile == "edge_float") and col in nullable_cols and rnd.random() < 0.35:
                cmp_ops = [rnd.choice(["is_null", "is_not_null"]), *cmp_ops]
            if discovery_profile and typ == "bool" and rnd.random() < 0.35:
                cmp_ops = [
                    rnd.choice(["bool_is_true", "bool_is_not_true", "bool_is_false", "bool_is_not_false"]),
                    *cmp_ops,
                ]
            if (discovery_profile or profile == "edge_float") and typ in {"int", "float"} and rnd.random() < 0.20:
                cmp_ops = ["range_closed", *cmp_ops]
            cmp = rnd.choice(cmp_ops)
            if cmp in {"in_set", "not_in_set"}:
                value = _literal_list_for_column(rnd, table, col) if col in base_columns else _literal_list_for_type(rnd, typ)
            elif cmp == "range_closed":
                value = (
                    _literal_range_for_column(rnd, table, col)
                    if col in base_columns
                    else sorted(rnd.sample(_literal_list_for_type(rnd, typ), 2))
                )
            elif cmp in {"is_null", "is_not_null", "bool_is_true", "bool_is_not_true", "bool_is_false", "bool_is_not_false"}:
                value = None
            else:
                value = _literal_for_column(rnd, table, col) if col in base_columns else _literal_for_type(rnd, typ)
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
            derived_cols.add(alias)
            nullable_cols.add(alias)

        elif op == "case_when" and comparable_cols and not grouped:
            derived_nullable = [column for column in comparable_cols if column in derived_cols and column in nullable_cols]
            if discovery_profile and derived_nullable and rnd.random() < 0.75:
                predicate_col = rnd.choice(derived_nullable)
            else:
                predicate_col = _preferred_comparable_column(
                    rnd,
                    comparable_cols,
                    derived_cols=derived_cols,
                    nullable_cols=nullable_cols,
                    prefer_derived=discovery_profile,
                )
            predicate_type = col_types[predicate_col]
            cmp_ops = ["==", "!="] if predicate_type in {"str", "bool"} else [">", ">=", "<", "<=", "==", "!="]
            if discovery_profile and predicate_type in {"int", "float"} and rnd.random() < 0.25:
                cmp_ops = [
                    rnd.choice(["gt_is_not_true", "ge_is_not_true", "lt_is_not_false", "le_is_not_false"]),
                    *cmp_ops,
                ]
            if discovery_profile and predicate_col in derived_nullable:
                cmp_ops = ["is_null", "is_not_null", *cmp_ops]
            elif discovery_profile and predicate_col in nullable_cols and rnd.random() < 0.70:
                cmp_ops = ["is_null", "is_not_null", *cmp_ops]
            if discovery_profile and predicate_type == "bool" and rnd.random() < 0.35:
                cmp_ops = [
                    rnd.choice(["bool_is_true", "bool_is_not_true", "bool_is_false", "bool_is_not_false"]),
                    *cmp_ops,
                ]
            cmp = rnd.choice(cmp_ops)
            if cmp in {"is_null", "is_not_null", "bool_is_true", "bool_is_not_true", "bool_is_false", "bool_is_not_false"}:
                predicate_value = None
            else:
                predicate_value = (
                    _literal_for_column(rnd, table, predicate_col)
                    if predicate_col in base_columns
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
            derived_cols.add(alias)
            nullable_cols.add(alias)

        elif op == "row_number_filter" and comparable_cols and not grouped:
            order_candidates = [column for column in available_cols if col_types.get(column) in {"int", "float", "str", "bool"}]
            if not order_candidates:
                continue
            order_width = 1 if len(order_candidates) == 1 else rnd.randint(2, min(3, len(order_candidates)))
            derived_order_candidates = [column for column in order_candidates if column in derived_cols]
            if discovery_profile and derived_order_candidates and rnd.random() < 0.55:
                first_order = rnd.choice(derived_order_candidates)
                remaining_order = [column for column in order_candidates if column != first_order]
                order_by_columns = [first_order, *rnd.sample(remaining_order, k=max(0, order_width - 1))]
            else:
                order_by_columns = rnd.sample(order_candidates, k=order_width)
            partition_candidates = [column for column in available_cols if col_types.get(column) in {"int", "str", "bool"}]
            partition_by: list[str] = []
            if partition_candidates and rnd.random() < 0.80:
                partition_width = rnd.randint(1, min(2, len(partition_candidates)))
                partition_by = unique_preserve_order(rnd.sample(partition_candidates, k=partition_width))
            comparator = "==" if rnd.random() < 0.45 else rnd.choice(["<", "<="])
            value = 1 if comparator == "==" else rnd.choice([1, 2, 3])
            ops.append(
                {
                    "op": "row_number_filter",
                    "partition_by": partition_by,
                    "order_by": [
                        {
                            "column": column,
                            "ascending": rnd.choice([True, False]),
                            "nulls": (
                                "first"
                                if column in nullable_cols and discovery_profile and rnd.random() < 0.60
                                else rnd.choice(["first", "last"])
                            ),
                        }
                        for column in order_by_columns
                    ],
                    "cmp": comparator,
                    "value": value,
                }
            )

        elif op == "running_sum" and numeric_cols and not grouped:
            derived_numeric = [column for column in numeric_cols if column in derived_cols]
            if discovery_profile and derived_numeric and rnd.random() < 0.95:
                source = rnd.choice(derived_numeric)
            else:
                source = rnd.choice(numeric_cols)
            order_candidates = [column for column in available_cols if col_types.get(column) in {"int", "float", "str", "bool"}]
            if not order_candidates:
                continue
            order_width = 1 if len(order_candidates) == 1 else rnd.randint(2, min(3, len(order_candidates)))
            derived_order_candidates = [column for column in order_candidates if column in derived_cols]
            if discovery_profile and derived_order_candidates and rnd.random() < 0.55:
                first_order = rnd.choice(derived_order_candidates)
                remaining_order = [column for column in order_candidates if column != first_order]
                order_by_columns = [first_order, *rnd.sample(remaining_order, k=max(0, order_width - 1))]
            else:
                order_by_columns = rnd.sample(order_candidates, k=order_width)
            partition_candidates = [
                column
                for column in available_cols
                if column != source and col_types.get(column) in {"int", "str", "bool"}
            ]
            partition_by: list[str] = []
            if partition_candidates and rnd.random() < 0.75:
                partition_width = rnd.randint(1, min(2, len(partition_candidates)))
                partition_by = unique_preserve_order(rnd.sample(partition_candidates, k=partition_width))
            alias = make_safe_output_name(
                f"run_{source}",
                used=set(available_cols) | {op_output_alias(existing) for existing in ops if op_output_alias(existing)},
            )
            running_sum_op: dict[str, Any] = {
                "op": "running_sum",
                "source": source,
                "column": alias,
                "order_by": [
                    {
                        "column": column,
                        "ascending": rnd.choice([True, False]),
                        "nulls": (
                            "first"
                            if column in nullable_cols and discovery_profile and rnd.random() < 0.60
                            else rnd.choice(["first", "last"])
                        ),
                    }
                    for column in order_by_columns
                ],
                "input_dtype": "float32" if col_types.get(source) == "float" or rnd.random() < 0.5 else "float64",
            }
            if partition_by:
                running_sum_op["partition_by"] = partition_by
            ops.append(running_sum_op)
            available_cols.append(alias)
            col_types[alias] = "float"
            comparable_cols.append(alias)
            numeric_cols.append(alias)
            derived_cols.add(alias)
            nullable_cols.add(alias)

        elif op == "sort" and available_cols:
            first = rnd.choice(available_cols)
            cols = [first] + sorted(c for c in available_cols if c != first)
            ops.append(_random_sort_op(rnd, cols, allow_mixed=_is_discovery_profile(profile)))

        elif op == "limit":
            ops.append({"op": "limit", "n": rnd.randint(0, max(1, len(table.rows) + 2))})

        elif op == "offset":
            ops.append({"op": "offset", "n": rnd.randint(0, max(1, len(table.rows) + 2))})

        elif op == "mutate" and (numeric_cols or bool_cols or string_cols) and not grouped:
            new_col = f"m_{operation_count(ops, 'mutate')}"
            expr, out_type = _random_mutate_expr(rnd, numeric_cols, bool_cols, string_cols, col_types, profile=profile)
            if (
                discovery_profile
                and numeric_cols
                and not any(col_types.get(column) in {"int", "float"} for column in derived_cols)
                and out_type not in {"int", "float"}
                and rnd.random() < 0.65
            ):
                expr, out_type = _random_mutate_expr(rnd, numeric_cols, [], [], col_types, profile=profile)
            ops.append({"op": "mutate", "column": new_col, "expr": expr})
            available_cols.append(new_col)
            col_types[new_col] = out_type
            comparable_cols.append(new_col)
            if out_type in {"int", "float"}:
                numeric_cols.append(new_col)
            if out_type == "str":
                string_cols.append(new_col)
            if out_type == "bool":
                bool_cols.append(new_col)
            derived_cols.add(new_col)
            nullable_cols.add(new_col)

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
            derived_cols &= set(available_cols)
            nullable_cols &= set(available_cols)
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
    bool_cols: list[str],
    string_cols: list[str],
    col_types: dict[str, str],
    profile: GeneratorProfile = "common",
) -> tuple[dict[str, Any], str]:
    date_strings = [column for column in string_cols if _looks_like_date_column(column)]
    numeric_strings = [column for column in string_cols if _looks_like_numeric_string_column(column)]
    choices: list[str] = []
    if numeric_cols:
        choices.extend(["add_const", "arith_const", "abs", "clip", "cast_float", "cast_string"])
    if bool_cols:
        choices.append("bool_not")
    if string_cols:
        choices.extend([
            "string_length",
            "string_lower",
            "string_upper",
            "string_replace",
            "string_slice",
            "string_null_if_empty",
            "string_split_part",
            "string_basename",
            "string_contains",
            "string_starts_with",
            "string_ends_with",
        ])
        if len(string_cols) > 1:
            choices.append("string_concat")
    if date_strings:
        choices.append("date_part")
    if numeric_strings:
        choices.extend(["cast_int_from_numeric_string", "cast_float_from_numeric_string"])
    if profile == "edge_float" and numeric_cols and rnd.random() < 0.65:
        kind = "arith_const"
    else:
        kind = rnd.choice(choices)
    if kind == "add_const":
        src = rnd.choice(numeric_cols)
        return {"kind": "add_const", "source": src, "value": rnd.choice([-2, -1, 0, 1, 2, 10])}, col_types[src]
    if kind == "arith_const":
        src = rnd.choice(numeric_cols)
        op_choices = ["sub", "mul", "div"]
        if profile == "edge_float":
            op_choices.extend(["mod", "mod"])
        op = "mod" if profile == "edge_float" and rnd.random() < 0.45 else rnd.choice(op_choices)
        value = rnd.choice([2, 3, 5, 10]) if op in {"div", "mod"} else rnd.choice([-2, -1, 1, 2, 10])
        out_type = "float" if op == "div" or col_types[src] == "float" else col_types[src]
        return {"kind": "arith_const", "source": src, "op": op, "value": value}, out_type
    if kind == "abs":
        src = rnd.choice(numeric_cols)
        return {"kind": "abs", "source": src}, col_types[src]
    if kind == "clip":
        src = rnd.choice(numeric_cols)
        lower, upper = _clip_bounds_for_type(rnd, col_types[src])
        return {"kind": "clip", "source": src, "lower": lower, "upper": upper}, col_types[src]
    if kind == "cast_float":
        src = rnd.choice(numeric_cols)
        return {"kind": "cast", "source": src, "to": "float"}, "float"
    if kind == "cast_string":
        int_cols = [col for col in numeric_cols if col_types.get(col) == "int"]
        src = rnd.choice(int_cols or numeric_cols)
        return {"kind": "cast", "source": src, "to": "str"}, "str"
    if kind == "bool_not":
        return {"kind": "bool_not", "source": rnd.choice(bool_cols)}, "bool"
    if kind == "cast_int_from_numeric_string":
        src = rnd.choice(numeric_strings)
        return {"kind": "cast", "source": src, "to": "int", "input_domain": "integer_string"}, "int"
    if kind == "cast_float_from_numeric_string":
        src = rnd.choice(numeric_strings)
        return {"kind": "cast", "source": src, "to": "float", "input_domain": "integer_string"}, "float"
    if kind == "string_length":
        return {"kind": "string_length", "source": rnd.choice(string_cols)}, "int"
    if kind == "string_lower":
        return {"kind": "string_lower", "source": rnd.choice(string_cols)}, "str"
    if kind == "string_upper":
        return {"kind": "string_upper", "source": rnd.choice(string_cols)}, "str"
    if kind == "string_replace":
        src = rnd.choice(string_cols)
        return {
            "kind": "string_replace",
            "source": src,
            "old": rnd.choice([" ", "a", "A", "_"]),
            "new": rnd.choice(["", "-", "_", "X"]),
        }, "str"
    if kind == "string_slice":
        return {
            "kind": "string_slice",
            "source": rnd.choice(string_cols),
            "start": 0,
            "length": rnd.randint(1, 4),
        }, "str"
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
    if kind == "string_concat":
        src = rnd.choice(string_cols)
        other = rnd.choice([column for column in string_cols if column != src])
        return {
            "kind": "string_concat",
            "source": src,
            "other": other,
            "sep": rnd.choice(["", "-", "_", "/"]),
        }, "str"
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
    if kind == "date_part":
        return {
            "kind": "date_part",
            "source": rnd.choice(date_strings),
            "part": rnd.choice(["year", "month", "day"]),
        }, "int"
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
