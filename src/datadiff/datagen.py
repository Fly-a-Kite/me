from __future__ import annotations

import math
import random
import string
from typing import Any, Literal

from .dsl import Case, ColumnSpec, Program, SortKey, TableData, normalize_sort_keys
from .util import unique_preserve_order

GeneratorProfile = Literal[
    "common",
    "edge_float",
    "workflow",
    "bughunt",
    "bughunt_no_groupby",
    "null_groupby_topk",
    "null_agg_topk",
    "filter_null_agg_topk",
    "join_null_agg_topk",
    "join_filter_groupby",
    "join_groupby_stress",
    "storage_offset",
    "float_group_key",
    "join_null_sort",
    "ordered_groupby_sort",
    "topk_resort",
    "join_ordered_agg_topk",
]


def _is_bughunt_profile(profile: GeneratorProfile) -> bool:
    return profile in {"bughunt", "bughunt_no_groupby"}


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


def _literal_for_type(rnd: random.Random, typ: str) -> Any:
    if typ == "int":
        return rnd.choice([-10, -1, 0, 1, 2, 10])
    if typ == "float":
        return rnd.choice([-1.0, 0.0, 0.5, 1.0, 10.0])
    if typ == "bool":
        return rnd.choice([True, False])
    return rnd.choice(["", "alpha", "beta", "中文", "missing"])


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
        if profile == "bughunt":
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
                profile == "bughunt"
                and
                "groupby" not in emitted_ops
                and numeric_cols
                and len(ops) >= 3
                and (remaining <= 3 or rnd.random() < 0.5)
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

        elif op == "filter" and comparable_cols and not grouped:
            col = rnd.choice(comparable_cols)
            typ = col_types[col]
            cmp_ops = ["==", "!="] if typ in {"str", "bool"} else [">", ">=", "<", "<=", "==", "!="]
            base_cols = {c.name for c in table.columns}
            value = _literal_for_column(rnd, table, col) if col in base_cols else _literal_for_type(rnd, typ)
            ops.append({"op": "filter", "column": col, "cmp": rnd.choice(cmp_ops), "value": value})

        elif op == "select" and available_cols:
            k = rnd.randint(1, len(available_cols))
            cols = sorted(rnd.sample(available_cols, k))
            ops.append({"op": "select", "columns": cols})
            available_cols = cols
            numeric_cols = [c for c in numeric_cols if c in available_cols]
            string_cols = [c for c in string_cols if c in available_cols]
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

        elif op == "groupby" and numeric_cols and available_cols and not grouped:
            keys = [rnd.choice(available_cols)]
            # Avoid grouping by float columns for common-subset stability.
            key_candidates = [c for c in available_cols if col_types.get(c) in {"int", "str", "bool"}]
            if key_candidates:
                key_count = 1 if len(key_candidates) == 1 or rnd.random() < 0.8 else 2
                keys = sorted(rnd.sample(key_candidates, key_count))
            agg_count = rnd.randint(1, min(3, len(numeric_cols)))
            aggs = []
            used_aliases = set()
            for val in rnd.sample(numeric_cols, agg_count):
                func = rnd.choice(["sum", "min", "max", "count"])
                alias = f"{func}_{val}"
                if alias in used_aliases:
                    alias = f"{alias}_{len(used_aliases)}"
                used_aliases.add(alias)
                aggs.append({"column": val, "func": func, "as": alias})
            ops.append({"op": "groupby", "keys": keys, "aggs": aggs})
            available_cols = keys + [a["as"] for a in aggs]
            numeric_cols = [a["as"] for a in aggs]
            string_cols = [c for c in keys if col_types.get(c) == "str"]
            comparable_cols = available_cols
            grouped = True
        if len(ops) > before_len:
            emitted_ops.add(str(ops[-1].get("op", "")))

    if type_aware:
        ops = repair_operations(table, ops, extra_tables=extra_tables)
    if not ops:
        ops.append({"op": "limit", "n": len(table.rows)})
    return Program(program_id=f"prog-{seed:08d}", seed=seed, operations=ops)


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
    func = rnd.choice(["sum", "min", "max", "count"])
    return {"op": "groupby", "keys": [col], "aggs": [{"column": agg_col, "func": func, "as": f"{func}_{agg_col}"}]}


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
        elif kind == "filter":
            if op["column"] not in available:
                continue
            column_type = col_types.get(op["column"], "")
            if not _filter_literal_is_valid(column_type, op.get("cmp"), op.get("value")):
                continue
            repaired.append(op)
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
        elif kind == "limit":
            if repaired and repaired[-1].get("op") == "sort":
                repaired.append(op)
            break
        elif kind == "offset":
            if repaired and repaired[-1].get("op") == "sort":
                repaired.append({"op": "offset", "n": max(0, int(op.get("n", 0)))})
        elif kind == "mutate":
            expr = op["expr"]
            out_type = _mutate_output_type(expr, available, numeric, strings, col_types)
            if out_type is None:
                continue
            repaired.append(op)
            available.add(op["column"])
            col_types[op["column"]] = out_type
            if out_type in {"int", "float"}:
                numeric.add(op["column"])
            if out_type == "str":
                strings.add(op["column"])
        elif kind == "groupby":
            keys = unique_preserve_order([k for k in op["keys"] if k in available])
            aggs = [a for a in op["aggs"] if a["column"] in available and a["column"] in numeric]
            unique_aggs: list[dict[str, Any]] = []
            seen_aliases: set[str] = set()
            for agg in aggs:
                alias = str(agg.get("as", ""))
                if not alias or alias in seen_aliases:
                    continue
                seen_aliases.add(alias)
                unique_aggs.append(agg)
            aggs = unique_aggs
            if not keys or not aggs:
                continue
            repaired.append({**op, "keys": keys, "aggs": aggs})
            available = set(keys) | {a["as"] for a in aggs}
            numeric = {k for k in keys if col_types.get(k) in {"int", "float"}}
            numeric |= {a["as"] for a in aggs}
            strings = {k for k in keys if col_types.get(k) == "str"}
            for agg in aggs:
                col_types[agg["as"]] = "int" if agg["func"] == "count" else col_types.get(agg["column"], "float")
        elif kind == "aggregate":
            aggs = [a for a in op["aggs"] if a["column"] in available and a["column"] in numeric]
            unique_aggs = []
            seen_aliases: set[str] = set()
            for agg in aggs:
                alias = str(agg.get("as", ""))
                if not alias or alias in seen_aliases:
                    continue
                seen_aliases.add(alias)
                unique_aggs.append(agg)
            if not unique_aggs:
                continue
            repaired.append({**op, "aggs": unique_aggs})
            available = {a["as"] for a in unique_aggs}
            numeric = set(available)
            strings = set()
            for agg in unique_aggs:
                col_types[agg["as"]] = "int" if agg["func"] == "count" else col_types.get(agg["column"], "float")
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
    if kind == "cast":
        return "float" if src in numeric and expr.get("to") == "float" else None
    if kind == "string_length":
        return "int" if src in strings else None
    if kind == "string_lower":
        return "str" if src in strings else None
    return None


def _filter_literal_is_valid(column_type: str, comparator: Any, value: Any) -> bool:
    if column_type in {"str", "bool"} and comparator not in {"==", "!="}:
        return False
    if value is None:
        return True
    if column_type == "str":
        return isinstance(value, str)
    if column_type == "bool":
        return isinstance(value, bool)
    if column_type in {"int", "float"}:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return False


def generate_case(seed: int, type_aware: bool = True, profile: GeneratorProfile = "common") -> Case:
    if profile == "null_groupby_topk" and type_aware:
        return generate_null_groupby_topk_case(seed)
    if profile == "null_agg_topk" and type_aware:
        return generate_null_agg_topk_case(seed)
    if profile == "filter_null_agg_topk" and type_aware:
        return generate_filter_null_agg_topk_case(seed)
    if profile == "join_null_agg_topk" and type_aware:
        return generate_join_null_agg_topk_case(seed)
    if profile == "join_filter_groupby" and type_aware:
        return generate_join_filter_groupby_case(seed)
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
    if profile == "workflow" and type_aware:
        return generate_workflow_case(seed)
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
    suffix = "-bughunt" if profile == "bughunt" else "-bughunt-no-groupby" if profile == "bughunt_no_groupby" else ""
    return Case(case_id=f"case-{seed:08d}{suffix}", seed=seed, tables=[table] + extra_tables, program=program)


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
        metadata={"generator_profile": "storage_offset", "row_count": row_count, "offset": offset},
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
    second_col = rnd.choice([col for col in ["x", "z", "g", "s", "id"] if col != first_col])
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
