from __future__ import annotations

import time
from typing import Any

from datadiff.backends.base import Backend, BackendResult
from datadiff.dsl import Program, SortKey, TableData, normalize_sort_keys
from datadiff.sqlite_runtime import sqlite3


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _lit(value: Any) -> str:
    import math

    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "NULL"
        if math.isinf(value):
            return "1e999" if value > 0 else "-1e999"
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


def _sql_type(kind: str) -> str:
    if kind == "int":
        return "INTEGER"
    if kind == "float":
        return "REAL"
    if kind == "bool":
        return "INTEGER"
    return "TEXT"


def _replace_projection(cols: list[str], column: str, expr_sql: str) -> tuple[str, list[str]]:
    kept_cols = [col for col in cols if col != column]
    select_parts = [f"q.{_quote(col)}" for col in kept_cols]
    select_parts.append(f"{expr_sql} AS {_quote(column)}")
    return ", ".join(select_parts), kept_cols + [column]


def _order_clause(sort_keys: list[SortKey]) -> str:
    # Use an explicit discriminator so SQLite follows the common DSL null
    # placement independently of its default ORDER BY behavior.
    order_parts = []
    for key in sort_keys:
        q = _quote(key.column)
        null_direction = "DESC" if key.nulls == "first" else "ASC"
        order_parts.append(f"({q} IS NULL) {null_direction}")
        order_parts.append(f"{q} {'ASC' if key.ascending else 'DESC'}")
    return ", ".join(order_parts)


class SQLiteBackend(Backend):
    name = "sqlite"

    def run(self, tables: list[TableData], program: Program, timeout_s: float = 5.0) -> BackendResult:
        start = time.perf_counter()
        try:
            import pandas as pd

            table_by_name = {table.name: table for table in tables}
            column_types = {c.name: c.type for table in tables for c in table.columns}
            current_cols = [c.name for c in tables[0].columns]
            con = sqlite3.connect(":memory:")
            for table in tables:
                col_defs = ", ".join(f"{_quote(c.name)} {_sql_type(c.type)}" for c in table.columns)
                con.execute(f"CREATE TABLE {_quote(table.name)} ({col_defs})")
                if table.rows:
                    cols = [c.name for c in table.columns]
                    placeholders = ", ".join("?" for _ in cols)
                    con.executemany(
                        f"INSERT INTO {_quote(table.name)} ({', '.join(_quote(c) for c in cols)}) VALUES ({placeholders})",
                        [[row.get(c) for c in cols] for row in table.rows],
                    )
            query = "SELECT * FROM t0"
            pending_order: list[SortKey] | None = None
            for op in program.operations:
                kind = op["op"]
                if kind == "join":
                    right = table_by_name[op["table"]]
                    right_cols = [
                        f"r.{_quote(c.name)} AS {_quote(c.name)}"
                        for c in right.columns
                        if c.name != op["right_on"]
                    ]
                    select_right = ", " + ", ".join(right_cols) if right_cols else ""
                    join_kind = "LEFT JOIN" if op["how"] == "left" else "INNER JOIN"
                    query = (
                        f"SELECT q.*{select_right} FROM ({query}) q {join_kind} {_quote(right.name)} r "
                        f"ON q.{_quote(op['left_on'])} = r.{_quote(op['right_on'])}"
                    )
                    current_cols.extend(
                        c.name
                        for c in right.columns
                        if c.name != op["right_on"] and c.name not in current_cols
                    )
                    pending_order = None
                elif kind == "filter":
                    query = (
                        f"SELECT * FROM ({query}) q "
                        f"WHERE {_quote(op['column'])} {op['cmp']} {_lit(op['value'])}"
                    )
                elif kind == "select":
                    cols = ", ".join(_quote(c) for c in op["columns"])
                    query = f"SELECT {cols} FROM ({query}) q"
                    current_cols = list(op["columns"])
                    if pending_order is not None:
                        pending_order = pending_order if {key.column for key in pending_order}.issubset(current_cols) else None
                elif kind == "sort":
                    pending_order = normalize_sort_keys(op)
                elif kind == "limit":
                    if pending_order is not None:
                        query = (
                            f"SELECT * FROM ({query}) q "
                            f"ORDER BY {_order_clause(pending_order)} LIMIT {int(op['n'])}"
                        )
                    else:
                        query = f"SELECT * FROM ({query}) q LIMIT {int(op['n'])}"
                    pending_order = None
                elif kind == "offset":
                    if pending_order is not None:
                        query = (
                            f"SELECT * FROM ({query}) q "
                            f"ORDER BY {_order_clause(pending_order)} LIMIT -1 OFFSET {int(op['n'])}"
                        )
                    else:
                        query = f"SELECT * FROM ({query}) q LIMIT -1 OFFSET {int(op['n'])}"
                    pending_order = None
                elif kind == "mutate":
                    expr = op["expr"]
                    if expr["kind"] == "add_const":
                        expr_sql = f"q.{_quote(expr['source'])} + {_lit(expr['value'])}"
                    elif expr["kind"] == "arith_const":
                        if expr["op"] == "div":
                            expr_sql = f"1.0 * q.{_quote(expr['source'])} / {_lit(expr['value'])}"
                        else:
                            op_sql = {"sub": "-", "mul": "*", "mod": "%"}[expr["op"]]
                            expr_sql = f"q.{_quote(expr['source'])} {op_sql} {_lit(expr['value'])}"
                    elif expr["kind"] == "cast" and expr["to"] == "float":
                        expr_sql = f"CAST(q.{_quote(expr['source'])} AS REAL)"
                    elif expr["kind"] == "string_length":
                        expr_sql = f"LENGTH(q.{_quote(expr['source'])})"
                    elif expr["kind"] == "string_lower":
                        expr_sql = f"LOWER(q.{_quote(expr['source'])})"
                    else:
                        raise ValueError(expr["kind"])
                    projection, current_cols = _replace_projection(current_cols, op["column"], expr_sql)
                    query = f"SELECT {projection} FROM ({query}) q"
                    if pending_order is not None and op["column"] in {key.column for key in pending_order}:
                        pending_order = None
                elif kind == "groupby":
                    keys = list(op["keys"])
                    key_sql = ", ".join(_quote(k) for k in keys)
                    agg_sql = []
                    for agg in op["aggs"]:
                        func = "COUNT" if agg["func"] == "count" else agg["func"].upper()
                        agg_sql.append(f"{func}({_quote(agg['column'])}) AS {_quote(agg['as'])}")
                    query = (
                        f"SELECT {key_sql}, {', '.join(agg_sql)} "
                        f"FROM ({query}) q GROUP BY {key_sql}"
                    )
                    current_cols = keys + [agg["as"] for agg in op["aggs"]]
                    pending_order = None
                elif kind == "aggregate":
                    agg_sql = []
                    for agg in op["aggs"]:
                        func = "COUNT" if agg["func"] == "count" else agg["func"].upper()
                        agg_sql.append(f"{func}({_quote(agg['column'])}) AS {_quote(agg['as'])}")
                    query = f"SELECT {', '.join(agg_sql)} FROM ({query}) q"
                    current_cols = [agg["as"] for agg in op["aggs"]]
                    pending_order = None
                else:
                    raise ValueError(kind)
            if pending_order is not None:
                query = f"SELECT * FROM ({query}) q ORDER BY {_order_clause(pending_order)}"
            cur = con.execute(query)
            columns = [desc[0] for desc in cur.description]
            out = pd.DataFrame(cur.fetchall(), columns=columns)
            for col in out.columns:
                if column_types.get(str(col)) == "bool":
                    out[col] = out[col].map(lambda v: None if pd.isna(v) else bool(v))
            con.close()
            return BackendResult(self.name, "ok", data=out, duration_ms=(time.perf_counter() - start) * 1000)
        except Exception as exc:  # noqa: BLE001
            return BackendResult(
                self.name,
                "error",
                error_type=type(exc).__name__,
                error=str(exc),
                duration_ms=(time.perf_counter() - start) * 1000,
            )
