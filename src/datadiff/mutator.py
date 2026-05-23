from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import Any, Callable

from datadiff.datagen import repair_operations
from datadiff.dsl import Case, Program, TableData, normalize_sort_keys
from datadiff.identifiers import make_safe_output_name
from datadiff.util import unique_preserve_order


@dataclass(slots=True)
class MutationResult:
    case: Case
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class MutationOperator:
    name: str
    apply: Callable[[list[TableData], list[dict[str, Any]], random.Random], str]


def mutate_case(case: Case, seed: int) -> Case:
    return mutate_case_with_metadata(case, seed).case


def mutate_case_with_metadata(case: Case, seed: int) -> MutationResult:
    rnd = random.Random(seed * 104729 + case.seed)
    tables = copy.deepcopy(case.tables)
    table = tables[0]
    operations = copy.deepcopy(case.program.operations)
    original_rows = copy.deepcopy(table.rows)
    original_ops = copy.deepcopy(operations)
    operator = rnd.choice(MUTATION_OPERATORS)
    choice = operator.name
    detail = operator.apply(tables, operations, rnd)

    operations = repair_operations(table, operations, extra_tables=tables[1:])
    if not operations:
        operations = [{"op": "limit", "n": len(table.rows)}]
    changed = table.rows != original_rows or operations != original_ops
    parent_lineage = case.metadata.get("seed_lineage", {}) if isinstance(case.metadata, dict) else {}
    root_seed = parent_lineage.get("root_seed", case.seed)
    depth = int(parent_lineage.get("depth", 0) or 0) + 1
    metadata = {
        "seed_lineage": {
            "root_seed": root_seed,
            "parent_seed": case.seed,
            "parent_case_id": case.case_id,
            "mutation_seed": seed,
            "depth": depth,
        },
        "mutation": {
            "operator": choice,
            "detail": detail,
            "changed": changed,
        },
    }
    program = Program(
        program_id=f"{case.program.program_id}-mut-{seed}",
        seed=seed,
        operations=operations,
    )
    mutated = Case(
        f"{case.case_id}-mut-{seed}",
        seed,
        tables,
        program,
        metadata=metadata,
    )
    return MutationResult(mutated, metadata)


def _mutate_scalar_value(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    del operations
    if not tables:
        return "value:none"
    return _mutate_value(tables[0], rnd)


def _mutate_value(table: TableData, rnd: random.Random) -> str:
    if not table.rows:
        return "value:none"
    col = rnd.choice(table.columns)
    row = rnd.choice(table.rows)
    current = row.get(col.name)
    if current is None:
        row[col.name] = _literal_for_type(col.type, rnd)
        return f"value:{col.type}:fill-null"
    if col.type == "int":
        row[col.name] = int(current) + rnd.choice([-10, -1, 0, 1, 10])
    elif col.type == "float":
        if isinstance(current, float) and (math.isnan(current) or math.isinf(current)):
            row[col.name] = 0.0
        else:
            row[col.name] = float(current) + rnd.choice([-1.0, -0.5, 0.5, 1.0])
    elif col.type == "bool":
        row[col.name] = not bool(current)
    elif col.type == "str":
        row[col.name] = rnd.choice(["", "alpha", "ALPHA", "中文", str(current) + "_x"])
    return f"value:{col.type}:{col.name}"


def _nullify_value(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    del operations
    if not tables or not tables[0].rows:
        return "nullify:none"
    table = tables[0]
    nullable = [col for col in table.columns if col.nullable]
    if not nullable:
        return "nullify:no-nullable-column"
    candidates = [
        (row, col)
        for row in table.rows
        for col in nullable
        if row.get(col.name) is not None
    ]
    if not candidates:
        candidates = [(row, col) for row in table.rows for col in nullable]
    row, col = rnd.choice(candidates)
    row[col.name] = None
    return f"nullify:{col.type}:{col.name}"


def _duplicate_row(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    del operations
    if not tables or not tables[0].rows:
        return "duplicate_row:none"
    tables[0].rows.append(copy.deepcopy(rnd.choice(tables[0].rows)))
    return f"duplicate_row:{tables[0].name}"


def _drop_row(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    del operations
    if not tables or not tables[0].rows:
        return "drop_row:none"
    removed_index = rnd.randrange(len(tables[0].rows))
    tables[0].rows.pop(removed_index)
    return f"drop_row:{tables[0].name}:{removed_index}"


def _shuffle_rows(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    del operations
    if not tables or len(tables[0].rows) < 2:
        return "shuffle_rows:none"
    before = copy.deepcopy(tables[0].rows)
    rnd.shuffle(tables[0].rows)
    if tables[0].rows == before:
        tables[0].rows.reverse()
    return f"shuffle_rows:{tables[0].name}"


def _append_operation(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    op = _random_operation(tables, operations, rnd)
    if op is not None:
        operations.append(op)
        return f"append:{op.get('op', 'unknown')}"
    return "append:none"


def _drop_operation(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    del tables
    if len(operations) <= 1:
        return "drop:none"
    removed = operations.pop(rnd.randrange(len(operations)))
    return f"drop:{removed.get('op', 'unknown')}"


def _tweak_random_operation(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    if not operations:
        return "tweak:none"
    op = operations[rnd.randrange(len(operations))]
    return _tweak_operation(tables, op, rnd)


def _append_order_projection_probe(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    if not tables:
        return "append_order_projection:none"
    available = _available_columns(tables, operations)
    if len(available) < 2:
        return "append_order_projection:too-few-columns"
    primary = rnd.choice(available)
    selected_candidates = [column for column in available if column != primary]
    selected_count = rnd.randint(1, min(3, len(selected_candidates)))
    selected = sorted(rnd.sample(selected_candidates, selected_count))
    sort_columns = [primary] + sorted(column for column in available if column != primary)
    operations.append(
        {
            "op": "sort",
            "keys": [
                {
                    "column": column,
                    "ascending": rnd.choice([True, False]),
                    "nulls": rnd.choice(["first", "last"]),
                }
                for column in sort_columns
            ],
        }
    )
    operations.append({"op": "select", "columns": selected})
    if rnd.random() < 0.75:
        if rnd.random() < 0.70:
            operations.append({"op": "limit", "n": rnd.randint(1, max(1, min(len(tables[0].rows) + 2, 8)))})
            tail = "limit"
        else:
            operations.append({"op": "offset", "n": rnd.randint(0, 2)})
            tail = "offset"
    else:
        tail = "none"
    return f"append_order_projection:{primary}:tail={tail}"


def _append_truth_filter_probe(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    if not tables:
        return "append_truth_filter:none"
    available = _available_columns(tables, operations)
    numeric = [column for column in available if _column_type(tables, column) in {"int", "float"}]
    if not numeric:
        return "append_truth_filter:no-numeric-column"
    column = rnd.choice(numeric)
    comparator = rnd.choice(["gt_is_not_true", "ge_is_not_true", "lt_is_not_false", "le_is_not_false"])
    operations.append(
        {
            "op": "filter",
            "column": column,
            "cmp": comparator,
            "value": _literal_for_type(_column_type(tables, column), rnd),
        }
    )
    return f"append_truth_filter:{column}:{comparator}"


def _append_boolean_predicate_filter_probe(
    tables: list[TableData],
    operations: list[dict[str, Any]],
    rnd: random.Random,
) -> str:
    if not tables:
        return "append_boolean_predicate_filter:none"
    available = _available_columns(tables, operations)
    boolean_columns = [column for column in available if _column_type(tables, column) == "bool"]
    if not boolean_columns:
        return "append_boolean_predicate_filter:no-bool-column"
    column = rnd.choice(boolean_columns)
    comparator = rnd.choice(["bool_is_true", "bool_is_not_true", "bool_is_false", "bool_is_not_false"])
    operations.append({"op": "filter", "column": column, "cmp": comparator, "value": None})
    return f"append_boolean_predicate_filter:{column}:{comparator}"


def _append_range_filter_probe(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    if not tables:
        return "append_range_filter:none"
    available = _available_columns(tables, operations)
    numeric_columns = [column for column in available if _column_type(tables, column) in {"int", "float"}]
    if not numeric_columns:
        return "append_range_filter:no-numeric-column"
    column = rnd.choice(numeric_columns)
    values = sorted(rnd.sample(_literal_list_for_type(_column_type(tables, column), rnd), 2))
    operations.append({"op": "filter", "column": column, "cmp": "range_closed", "value": values})
    return f"append_range_filter:{column}:{values[0]}:{values[1]}"


def _append_grouped_topk_probe(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> str:
    if not tables:
        return "append_grouped_topk:none"
    available = _available_columns(tables, operations)
    if not available:
        return "append_grouped_topk:no-columns"
    numeric = [
        column
        for column in available
        if _column_type(tables, column) in {"int", "float"}
        or column.startswith(("m_", "sum_", "min_", "max_", "count_", "nunique_", "uniq_"))
    ]
    if not numeric:
        return "append_grouped_topk:no-numeric-column"
    key = rnd.choice(numeric)
    value = rnd.choice([column for column in numeric if column != key] or numeric)
    alias = make_safe_output_name(f"count_{value}", used={key})
    operations.extend(
        [
            {
                "op": "groupby",
                "keys": [key],
                "aggs": [{"column": value, "func": "count", "as": alias}],
            },
            {"op": "select", "columns": [key]},
            {
                "op": "sort",
                "keys": [
                    {
                        "column": key,
                        "ascending": rnd.choice([True, False]),
                        "nulls": rnd.choice(["first", "last"]),
                    }
                ],
            },
            {"op": "limit", "n": rnd.randint(1, max(1, min(len(tables[0].rows) + 2, 8)))},
        ]
    )
    return f"append_grouped_topk:{key}:{value}"


def _random_operation(tables: list[TableData], operations: list[dict[str, Any]], rnd: random.Random) -> dict[str, Any] | None:
    table = tables[0]
    available = _available_columns(tables, operations)
    if not available:
        return None
    numeric = [
        c
        for c in available
        if _column_type(tables, c) in {"int", "float"}
        or c.startswith(("m_", "sum_", "min_", "max_", "count_", "nunique_", "uniq_"))
    ]
    strings = [c for c in available if _column_type(tables, c) == "str"]
    choices = ["filter", "select", "sort", "limit"]
    if numeric or strings:
        choices.append("mutate")
    if numeric:
        choices.append("groupby")
        choices.append("aggregate")
    kind = rnd.choice(choices)
    if kind == "filter":
        col = rnd.choice(available)
        typ = _column_type(tables, col)
        cmp = rnd.choice(["==", "!="] if typ in {"str", "bool"} else [">", ">=", "<", "<=", "==", "!="])
        if typ in {"int", "float"} and rnd.random() < 0.20:
            cmp = rnd.choice(["gt_is_not_true", "ge_is_not_true", "lt_is_not_false", "le_is_not_false"])
        if rnd.random() < 0.15:
            cmp = "in_set"
        if rnd.random() < 0.12:
            cmp = rnd.choice(["is_null", "is_not_null"])
        if typ == "bool" and rnd.random() < 0.25:
            cmp = rnd.choice(["bool_is_true", "bool_is_not_true", "bool_is_false", "bool_is_not_false"])
        if typ in {"int", "float"} and rnd.random() < 0.18:
            cmp = "range_closed"
        if cmp == "in_set":
            value = _literal_list_for_type(typ, rnd)
        elif cmp == "range_closed":
            value = sorted(rnd.sample(_literal_list_for_type(typ, rnd), 2))
        elif cmp in {"is_null", "is_not_null", "bool_is_true", "bool_is_not_true", "bool_is_false", "bool_is_not_false"}:
            value = None
        else:
            value = _literal_for_type(typ, rnd)
        return {"op": "filter", "column": col, "cmp": cmp, "value": value}
    if kind == "select":
        count = rnd.randint(1, len(available))
        return {"op": "select", "columns": sorted(rnd.sample(available, count))}
    if kind == "sort":
        first = rnd.choice(available)
        cols = [first] + sorted(c for c in available if c != first)
        if len(cols) > 1 and rnd.random() < 0.35:
            return {
                "op": "sort",
                "keys": [
                    {
                        "column": column,
                        "ascending": rnd.choice([True, False]),
                        "nulls": rnd.choice(["first", "last"]),
                    }
                    for column in cols
                ],
            }
        return {"op": "sort", "columns": cols, "ascending": rnd.choice([True, False])}
    if kind == "limit":
        return {"op": "limit", "n": rnd.randint(0, max(1, len(table.rows) + 3))}
    if kind == "mutate" and (numeric or strings):
        if strings and (not numeric or rnd.random() < 0.3):
            src = rnd.choice(strings)
            expr = rnd.choice([
                {"kind": "string_length", "source": src},
                {"kind": "string_lower", "source": src},
            ])
        else:
            src = rnd.choice(numeric)
            expr = {"kind": "add_const", "source": src, "value": rnd.choice([-10, -1, 0, 1, 10])}
        return {
            "op": "mutate",
            "column": f"m_{len([o for o in operations if o.get('op') == 'mutate'])}",
            "expr": expr,
        }
    if kind == "groupby" and numeric:
        keys = [rnd.choice(available)]
        val = rnd.choice(numeric)
        func = rnd.choice(["sum", "min", "max", "count", "nunique"])
        alias = make_safe_output_name(f"{func}_{val}", used=set(keys))
        return {"op": "groupby", "keys": keys, "aggs": [{"column": val, "func": func, "as": alias}]}
    if kind == "aggregate" and numeric:
        val = rnd.choice(numeric)
        func = rnd.choice(["sum", "min", "max", "count", "nunique"])
        alias = make_safe_output_name(f"{func}_{val}_all")
        return {"op": "aggregate", "aggs": [{"column": val, "func": func, "as": alias}]}
    return None


def _tweak_operation(tables: list[TableData], op: dict[str, Any], rnd: random.Random) -> str:
    kind = op.get("op")
    if kind == "filter":
        op["cmp"] = rnd.choice([">", ">=", "<", "<=", "==", "!="])
        op["value"] = _literal_for_type(_column_type(tables, op["column"]), rnd)
        return "tweak:filter"
    elif kind == "sort":
        if "keys" in op:
            keys = [key.to_dict() for key in normalize_sort_keys(op)]
            if not keys:
                return "tweak:sort:no-keys"
            key = keys[rnd.randrange(len(keys))]
            if rnd.random() < 0.5:
                key["ascending"] = not bool(key.get("ascending", True))
            else:
                key["nulls"] = "first" if key.get("nulls", "last") == "last" else "last"
            op["keys"] = keys
        else:
            op["ascending"] = not bool(op.get("ascending", True))
        return "tweak:sort"
    elif kind == "limit":
        op["n"] = max(0, int(op.get("n", 0)) + rnd.choice([-2, -1, 1, 2]))
        return "tweak:limit"
    elif kind == "mutate":
        if "value" in op["expr"]:
            op["expr"]["value"] = op["expr"].get("value", 0) + rnd.choice([-2, -1, 1, 2])
            return "tweak:mutate"
    return f"tweak:{kind or 'unknown'}:noop"


def _available_columns(tables: list[TableData], operations: list[dict[str, Any]]) -> list[str]:
    available = [c.name for c in tables[0].columns]
    table_by_name = {table.name: table for table in tables}
    for op in operations:
        if op.get("op") == "join":
            right = table_by_name.get(op.get("table", ""))
            if right is not None:
                available.extend(c.name for c in right.columns if c.name != op.get("right_on"))
        elif op.get("op") == "select":
            available = [c for c in unique_preserve_order(op.get("columns", [])) if c in available]
        elif op.get("op") == "mutate":
            available.append(op["column"])
        elif op.get("op") == "groupby":
            available = unique_preserve_order(list(op.get("keys", [])) + [agg["as"] for agg in op.get("aggs", [])])
        elif op.get("op") == "aggregate":
            available = unique_preserve_order([agg["as"] for agg in op.get("aggs", [])])
    return unique_preserve_order(available)


def _column_type(tables: list[TableData], name: str) -> str:
    for table in tables:
        for col in table.columns:
            if col.name == name:
                return col.type
    return "float" if name.startswith(("m_", "sum_", "min_", "max_")) else "int"


def _literal_for_type(typ: str, rnd: random.Random) -> Any:
    if typ == "int":
        return rnd.choice([-10, -1, 0, 1, 2, 10])
    if typ == "float":
        return rnd.choice([-1.0, 0.0, 0.5, 1.0, 10.0])
    if typ == "bool":
        return rnd.choice([True, False])
    return rnd.choice(["", "alpha", "beta", "中文", "missing"])


def _literal_list_for_type(typ: str, rnd: random.Random) -> list[Any]:
    if typ == "int":
        return rnd.sample([-10, -1, 0, 1, 2, 10], k=3)
    if typ == "float":
        return rnd.sample([-1.0, 0.0, 0.5, 1.0, 10.0], k=3)
    if typ == "bool":
        return rnd.sample([True, False], k=rnd.randint(1, 2))
    return rnd.sample(["", "alpha", "beta", "中文", "missing"], k=3)


MUTATION_OPERATORS: tuple[MutationOperator, ...] = (
    MutationOperator("value", _mutate_scalar_value),
    MutationOperator("nullify_value", _nullify_value),
    MutationOperator("duplicate_row", _duplicate_row),
    MutationOperator("drop_row", _drop_row),
    MutationOperator("shuffle_rows", _shuffle_rows),
    MutationOperator("append_op", _append_operation),
    MutationOperator("append_order_projection", _append_order_projection_probe),
    MutationOperator("append_truth_filter", _append_truth_filter_probe),
    MutationOperator("append_boolean_predicate_filter", _append_boolean_predicate_filter_probe),
    MutationOperator("append_range_filter", _append_range_filter_probe),
    MutationOperator("append_grouped_topk", _append_grouped_topk_probe),
    MutationOperator("drop_op", _drop_operation),
    MutationOperator("tweak_op", _tweak_random_operation),
)
MUTATION_OPERATOR_NAMES = tuple(operator.name for operator in MUTATION_OPERATORS)
