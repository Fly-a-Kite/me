from __future__ import annotations

import tempfile
import time
import os

from datadiff.backends.base import Backend, BackendResult
from datadiff.dsl import Program, SortKey, TableData, normalize_sort_keys


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _lit(value):
    import math
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "'NaN'::DOUBLE"
        if math.isinf(value):
            return "'Infinity'::DOUBLE" if value > 0 else "'-Infinity'::DOUBLE"
        return repr(value)
    s = str(value).replace("'", "''")
    return f"'{s}'"


def _sql_type(kind: str) -> str:
    if kind == "int":
        return "INTEGER"
    if kind == "float":
        return "DOUBLE"
    if kind == "bool":
        return "BOOLEAN"
    return "VARCHAR"


def _replace_projection(cols: list[str], column: str, expr_sql: str) -> tuple[str, list[str]]:
    kept_cols = [col for col in cols if col != column]
    select_parts = [f"q.{_quote(col)}" for col in kept_cols]
    select_parts.append(f"{expr_sql} AS {_quote(column)}")
    return ", ".join(select_parts), kept_cols + [column]


def _order_clause(sort_keys: list[SortKey]) -> str:
    return ", ".join(
        f"{_quote(key.column)} {'ASC' if key.ascending else 'DESC'} NULLS {key.nulls.upper()}"
        for key in sort_keys
    )


class DuckDBBackend(Backend):
    name = "duckdb"
    persistent_storage = False

    def run(self, tables: list[TableData], program: Program, timeout_s: float = 5.0) -> BackendResult:
        start = time.perf_counter()
        tempdir = tempfile.TemporaryDirectory() if self.persistent_storage else None
        con = None
        try:
            import duckdb
            import pandas as pd

            database = f"{tempdir.name}/datadiff.duckdb" if tempdir is not None else ":memory:"
            con = duckdb.connect(database=database)
            thread_limit = _duckdb_threads()
            if thread_limit is not None:
                con.execute(f"PRAGMA threads={thread_limit}")
            table_by_name = {table.name: table for table in tables}
            current_cols = [c.name for c in tables[0].columns]
            for table in tables:
                df = pd.DataFrame(table.rows, columns=[c.name for c in table.columns])
                registered_name = f"__datadiff_input_{table.name}"
                con.register(registered_name, df)
                col_defs = ", ".join(f"{_quote(c.name)} {_sql_type(c.type)}" for c in table.columns)
                scope = "" if self.persistent_storage else "TEMP "
                con.execute(f"CREATE {scope}TABLE {_quote(table.name)} ({col_defs})")
                if table.columns:
                    cols = ", ".join(_quote(c.name) for c in table.columns)
                    cast_cols = ", ".join(
                        f"CAST({_quote(c.name)} AS {_sql_type(c.type)}) AS {_quote(c.name)}"
                        for c in table.columns
                    )
                    con.execute(
                        f"INSERT INTO {_quote(table.name)} ({cols}) "
                        f"SELECT {cast_cols} FROM {_quote(registered_name)}"
                    )
                con.unregister(registered_name)
            if self.persistent_storage:
                con.execute("CHECKPOINT")
            ctes: list[tuple[str, str]] = []
            relation = _quote(tables[0].name)
            pending_order: list[SortKey] | None = None

            def add_step(sql: str) -> str:
                name = f"step_{len(ctes)}"
                ctes.append((name, sql))
                return _quote(name)

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
                    relation = add_step(
                        f"SELECT q.*{select_right} FROM {relation} q {join_kind} {_quote(right.name)} r "
                        f"ON q.{_quote(op['left_on'])} = r.{_quote(op['right_on'])}"
                    )
                    current_cols.extend(
                        c.name
                        for c in right.columns
                        if c.name != op["right_on"] and c.name not in current_cols
                    )
                    pending_order = None
                elif kind == "filter":
                    cmp = op["cmp"]
                    col = _quote(op["column"])
                    relation = add_step(f"SELECT * FROM {relation} q WHERE q.{col} {cmp} {_lit(op['value'])}")
                elif kind == "select":
                    cols = ", ".join(_quote(c) for c in op["columns"])
                    relation = add_step(f"SELECT {cols} FROM {relation} q")
                    current_cols = list(op["columns"])
                    if pending_order is not None:
                        pending_order = pending_order if {key.column for key in pending_order}.issubset(current_cols) else None
                elif kind == "sort":
                    pending_order = normalize_sort_keys(op)
                elif kind == "limit":
                    if pending_order is not None:
                        relation = add_step(
                            f"SELECT * FROM {relation} q "
                            f"ORDER BY {_order_clause(pending_order)} LIMIT {int(op['n'])}"
                        )
                    else:
                        relation = add_step(f"SELECT * FROM {relation} q LIMIT {int(op['n'])}")
                    pending_order = None
                elif kind == "offset":
                    if pending_order is not None:
                        relation = add_step(
                            f"SELECT * FROM {relation} q "
                            f"ORDER BY {_order_clause(pending_order)} OFFSET {int(op['n'])}"
                        )
                    else:
                        relation = add_step(f"SELECT * FROM {relation} q OFFSET {int(op['n'])}")
                    pending_order = None
                elif kind == "mutate":
                    expr = op["expr"]
                    if expr["kind"] == "add_const":
                        expr_sql = f"q.{_quote(expr['source'])} + {_lit(expr['value'])}"
                    elif expr["kind"] == "arith_const":
                        op_sql = {"sub": "-", "mul": "*", "div": "/", "mod": "%"}[expr["op"]]
                        source_sql = f"CAST(q.{_quote(expr['source'])} AS DOUBLE)" if expr["op"] == "div" else f"q.{_quote(expr['source'])}"
                        expr_sql = f"{source_sql} {op_sql} {_lit(expr['value'])}"
                    elif expr["kind"] == "cast" and expr["to"] == "float":
                        expr_sql = f"CAST(q.{_quote(expr['source'])} AS DOUBLE)"
                    elif expr["kind"] == "string_length":
                        expr_sql = f"LENGTH(q.{_quote(expr['source'])})"
                    elif expr["kind"] == "string_lower":
                        expr_sql = f"LOWER(q.{_quote(expr['source'])})"
                    else:
                        raise ValueError(expr["kind"])
                    projection, current_cols = _replace_projection(current_cols, op["column"], expr_sql)
                    relation = add_step(f"SELECT {projection} FROM {relation} q")
                    if pending_order is not None and op["column"] in {key.column for key in pending_order}:
                        pending_order = None
                elif kind == "groupby":
                    keys = list(op["keys"])
                    key_sql = ", ".join(_quote(k) for k in keys)
                    agg_sql = []
                    for agg in op["aggs"]:
                        func = "COUNT" if agg["func"] == "count" else agg["func"].upper()
                        agg_sql.append(f"{func}({_quote(agg['column'])}) AS {_quote(agg['as'])}")
                    relation = add_step(
                        f"SELECT {key_sql}, {', '.join(agg_sql)} FROM {relation} q "
                        f"GROUP BY {key_sql}"
                    )
                    current_cols = keys + [agg["as"] for agg in op["aggs"]]
                    pending_order = None
                elif kind == "aggregate":
                    agg_sql = []
                    for agg in op["aggs"]:
                        func = "COUNT" if agg["func"] == "count" else agg["func"].upper()
                        agg_sql.append(f"{func}({_quote(agg['column'])}) AS {_quote(agg['as'])}")
                    relation = add_step(f"SELECT {', '.join(agg_sql)} FROM {relation} q")
                    current_cols = [agg["as"] for agg in op["aggs"]]
                    pending_order = None
                else:
                    raise ValueError(kind)
            if pending_order is not None:
                relation = add_step(f"SELECT * FROM {relation} q ORDER BY {_order_clause(pending_order)}")
            if ctes:
                cte_sql = ", ".join(f"{_quote(name)} AS ({sql})" for name, sql in ctes)
                query = f"WITH {cte_sql} SELECT * FROM {relation}"
            else:
                query = f"SELECT * FROM {relation}"
            out = con.execute(query).df()
            con.close()
            if tempdir is not None:
                tempdir.cleanup()
            return BackendResult(self.name, "ok", data=out, duration_ms=(time.perf_counter()-start)*1000)
        except Exception as exc:  # noqa: BLE001
            if con is not None:
                try:
                    con.close()
                except Exception:
                    pass
            if tempdir is not None:
                tempdir.cleanup()
            return BackendResult(self.name, "error", error_type=type(exc).__name__, error=str(exc), duration_ms=(time.perf_counter()-start)*1000)


class DuckDBPersistentBackend(DuckDBBackend):
    name = "duckdb_persistent"
    persistent_storage = True


def _duckdb_threads() -> int | None:
    raw = os.environ.get("DATADIFF_DUCKDB_THREADS")
    if raw is None:
        return None
    try:
        return max(1, int(raw))
    except ValueError:
        return 1
