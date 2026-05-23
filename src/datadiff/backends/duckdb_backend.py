from __future__ import annotations

import tempfile
import time
import os

from datadiff.backends.base import Backend, BackendResult
from datadiff.dsl import Program, SortKey, TableData, normalize_sort_keys
from datadiff.filtering import sql_filter_condition
from datadiff.sortedness import is_sorted_values


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _lit(value):
    import math
    if value is None:
        return "NULL"
    if isinstance(value, (list, tuple)):
        return "(" + ", ".join(_lit(item) for item in value) + ")"
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


def _agg_expr(column: str, func: str) -> str:
    if func == "nunique":
        return f"COUNT(DISTINCT {_quote(column)})"
    sql_func = "COUNT" if func == "count" else func.upper()
    return f"{sql_func}({_quote(column)})"


def _tuple_absence_native_condition(op: dict) -> str:
    left_sql = ", ".join(f"q.{_quote(column)}" for column in op["columns"])
    right_sql = ", ".join(_quote(column) for column in op["right_columns"])
    return f"({left_sql}) NOT IN (SELECT {right_sql} FROM {_quote(op['table'])})"


def _running_sum_projection(cols: list[str], op: dict) -> tuple[str, list[str]]:
    kept_cols = [col for col in cols if col != op["column"]]
    select_parts = [f"q.{_quote(col)}" for col in kept_cols]
    order_sql = _order_clause(normalize_sort_keys({"keys": op["order_by"]}))
    expr_sql = (
        f"SUM(CAST(q.{_quote(op['source'])} AS DOUBLE)) OVER ("
        f"ORDER BY {order_sql} ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
        f") AS {_quote(op['column'])}"
    )
    select_parts.append(expr_sql)
    return ", ".join(select_parts), kept_cols + [op["column"]]


def _random_case_probe_sql(op: dict) -> str:
    rows = int(op.get("rows", 100_000))
    branches = int(op.get("branches", 3))
    when_sql = " ".join(f"WHEN {idx} THEN 'branch_{idx}'" for idx in range(branches))
    return (
        f"SELECT (COUNT(*) > 0) AS {_quote(op['as'])} FROM ("
        f"SELECT CASE CAST(FLOOR(random() * {branches}) AS INTEGER) "
        f"{when_sql} ELSE 'unexpected_else' END AS result "
        f"FROM generate_series(1, {rows})"
        f") q WHERE result = 'unexpected_else'"
    )


def _scalar_subquery_probe_sql(op: dict) -> str:
    return (
        "WITH tenk1(unique1, unique2, two, four, ten, twenty, hundred, thousand) AS ("
        "VALUES (1,1,1,1,1,1,1,1), (2,2,2,2,2,2,2,2)"
        "), got AS ("
        "SELECT (SELECT max((SELECT i.unique2 FROM tenk1 i WHERE i.unique1 = o.unique1))) AS probe_value "
        "FROM tenk1 o"
        ") "
        f"SELECT NOT (COUNT(*) = 1 AND MIN(probe_value) = 2 AND MAX(probe_value) = 2) AS {_quote(op['as'])} "
        "FROM got"
    )


def _struct_distinct_probe_sql(op: dict) -> str:
    return (
        "WITH got AS ("
        "SELECT DISTINCT unnest(s) "
        "FROM (SELECT {'a':'0','b':'0'} AS s UNION ALL SELECT {'a':'0','b':'1'})"
        ") "
        "SELECT NOT ("
        "COUNT(*) = 2 "
        "AND SUM(CASE WHEN a = '0' AND b = '0' THEN 1 ELSE 0 END) = 1 "
        "AND SUM(CASE WHEN a = '0' AND b = '1' THEN 1 ELSE 0 END) = 1"
        f") AS {_quote(op['as'])} "
        "FROM got"
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
            visible_cols = list(current_cols)
            hidden_order_cols: list[str] = []

            def add_step(sql: str) -> str:
                name = f"step_{len(ctes)}"
                ctes.append((name, sql))
                return _quote(name)

            def visible_projection() -> str:
                return ", ".join(f"q.{_quote(col)}" for col in visible_cols)

            def drop_hidden_order_cols() -> None:
                nonlocal relation, current_cols, hidden_order_cols
                if not hidden_order_cols:
                    return
                relation = add_step(f"SELECT {visible_projection()} FROM {relation} q")
                current_cols = list(visible_cols)
                hidden_order_cols = []

            def materialize_visible_relation():
                if pending_order is not None:
                    body = (
                        f"SELECT {visible_projection()} FROM {relation} q "
                        f"ORDER BY {_order_clause(pending_order)}"
                    )
                else:
                    drop_hidden_order_cols()
                    body = f"SELECT {visible_projection()} FROM {relation} q"
                if ctes:
                    cte_sql = ", ".join(f"{_quote(name)} AS ({sql})" for name, sql in ctes)
                    body = f"WITH {cte_sql} {body}"
                return con.execute(body).df()

            def select_with_pending_order(cols: list[str]) -> str:
                nonlocal pending_order, hidden_order_cols
                if pending_order is None:
                    return ", ".join(f"q.{_quote(c)}" for c in cols)
                projection = [f"q.{_quote(c)}" for c in cols]
                selected = set(cols)
                updated_order: list[SortKey] = []
                for idx, key in enumerate(pending_order):
                    if key.column in selected:
                        updated_order.append(key)
                        continue
                    hidden = (
                        key.column
                        if key.column in hidden_order_cols
                        else f"__datadiff_order_{len(hidden_order_cols)}_{idx}"
                    )
                    if hidden not in hidden_order_cols:
                        projection.append(f"q.{_quote(key.column)} AS {_quote(hidden)}")
                        hidden_order_cols.append(hidden)
                    else:
                        projection.append(f"q.{_quote(hidden)}")
                    updated_order.append(SortKey(hidden, key.ascending, key.nulls))
                pending_order = updated_order
                return ", ".join(projection)

            for op in program.operations:
                kind = op["op"]
                if kind == "join":
                    drop_hidden_order_cols()
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
                    visible_cols = list(current_cols)
                    pending_order = None
                elif kind == "filter":
                    condition = sql_filter_condition(f"q.{_quote(op['column'])}", _lit(op["value"]), op["cmp"])
                    relation = add_step(f"SELECT * FROM {relation} q WHERE {condition}")
                elif kind == "tuple_absence_filter":
                    relation = add_step(f"SELECT * FROM {relation} q WHERE {_tuple_absence_native_condition(op)}")
                elif kind == "running_sum":
                    drop_hidden_order_cols()
                    order_keys = normalize_sort_keys({"keys": op["order_by"]})
                    projection, current_cols = _running_sum_projection(current_cols, op)
                    relation = add_step(f"SELECT {projection} FROM {relation} q")
                    visible_cols = [col for col in visible_cols if col != op["column"]] + [op["column"]]
                    pending_order = order_keys
                elif kind == "sortedness_check":
                    materialized = materialize_visible_relation()
                    ok = is_sorted_values(
                        materialized[op["column"]].tolist(),
                        ascending=bool(op.get("ascending", True)),
                        nulls=str(op.get("nulls", "last")),
                    )
                    materialized_name = f"__datadiff_sortedness_{len(ctes)}"
                    con.register(materialized_name, pd.DataFrame({op["as"]: [ok]}))
                    ctes = []
                    relation = _quote(materialized_name)
                    current_cols = [op["as"]]
                    visible_cols = [op["as"]]
                    hidden_order_cols = []
                    pending_order = None
                elif kind == "random_case_probe":
                    ctes = []
                    relation = add_step(_random_case_probe_sql(op))
                    current_cols = [op["as"]]
                    visible_cols = [op["as"]]
                    hidden_order_cols = []
                    pending_order = None
                elif kind == "group_quantile_probe":
                    ctes = []
                    relation = add_step(f"SELECT FALSE AS {_quote(op['as'])}")
                    current_cols = [op["as"]]
                    visible_cols = [op["as"]]
                    hidden_order_cols = []
                    pending_order = None
                elif kind == "scalar_subquery_probe":
                    ctes = []
                    relation = add_step(_scalar_subquery_probe_sql(op))
                    current_cols = [op["as"]]
                    visible_cols = [op["as"]]
                    hidden_order_cols = []
                    pending_order = None
                elif kind == "window_avg_probe":
                    ctes = []
                    relation = add_step(f"SELECT FALSE AS {_quote(op['as'])}")
                    current_cols = [op["as"]]
                    visible_cols = [op["as"]]
                    hidden_order_cols = []
                    pending_order = None
                elif kind == "struct_distinct_probe":
                    ctes = []
                    relation = add_step(_struct_distinct_probe_sql(op))
                    current_cols = [op["as"]]
                    visible_cols = [op["as"]]
                    hidden_order_cols = []
                    pending_order = None
                elif kind == "select":
                    cols = list(op["columns"])
                    projection = select_with_pending_order(cols)
                    relation = add_step(f"SELECT {projection} FROM {relation} q")
                    visible_cols = cols
                    current_cols = cols + [col for col in hidden_order_cols if col not in cols]
                elif kind == "sort":
                    drop_hidden_order_cols()
                    pending_order = normalize_sort_keys(op)
                elif kind == "limit":
                    if pending_order is not None:
                        relation = add_step(
                            f"SELECT * FROM {relation} q "
                            f"ORDER BY {_order_clause(pending_order)} LIMIT {int(op['n'])}"
                        )
                    else:
                        relation = add_step(f"SELECT * FROM {relation} q LIMIT {int(op['n'])}")
                elif kind == "offset":
                    if pending_order is not None:
                        relation = add_step(
                            f"SELECT * FROM {relation} q "
                            f"ORDER BY {_order_clause(pending_order)} OFFSET {int(op['n'])}"
                        )
                    else:
                        relation = add_step(f"SELECT * FROM {relation} q OFFSET {int(op['n'])}")
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
                    visible_cols = [col for col in visible_cols if col != op["column"]] + [op["column"]]
                    if pending_order is not None and op["column"] in {key.column for key in pending_order}:
                        pending_order = None
                        drop_hidden_order_cols()
                elif kind == "groupby":
                    drop_hidden_order_cols()
                    keys = list(op["keys"])
                    key_sql = ", ".join(_quote(k) for k in keys)
                    agg_sql = []
                    for agg in op["aggs"]:
                        agg_sql.append(f"{_agg_expr(agg['column'], agg['func'])} AS {_quote(agg['as'])}")
                    relation = add_step(
                        f"SELECT {key_sql}, {', '.join(agg_sql)} FROM {relation} q "
                        f"GROUP BY {key_sql}"
                    )
                    current_cols = keys + [agg["as"] for agg in op["aggs"]]
                    visible_cols = list(current_cols)
                    pending_order = None
                elif kind == "aggregate":
                    drop_hidden_order_cols()
                    agg_sql = []
                    for agg in op["aggs"]:
                        agg_sql.append(f"{_agg_expr(agg['column'], agg['func'])} AS {_quote(agg['as'])}")
                    relation = add_step(f"SELECT {', '.join(agg_sql)} FROM {relation} q")
                    current_cols = [agg["as"] for agg in op["aggs"]]
                    visible_cols = list(current_cols)
                    pending_order = None
                else:
                    raise ValueError(kind)
            if pending_order is not None:
                relation = add_step(f"SELECT {visible_projection()} FROM {relation} q ORDER BY {_order_clause(pending_order)}")
            else:
                drop_hidden_order_cols()
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
