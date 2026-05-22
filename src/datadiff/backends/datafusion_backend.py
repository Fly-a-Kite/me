from __future__ import annotations

import math
import time
from typing import Any

from datadiff.backends.base import Backend, BackendResult
from datadiff.dsl import Program, SortKey, TableData, normalize_sort_keys


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _lit(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "CAST('NaN' AS DOUBLE)"
        if math.isinf(value):
            sign = "" if value > 0 else "-"
            return f"CAST('{sign}Infinity' AS DOUBLE)"
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


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


class DataFusionBackend(Backend):
    name = "datafusion"

    def _to_record_batch(self, table: TableData, pa):
        arrays = []
        fields = []
        for column in table.columns:
            typ = _arrow_type(pa, column.type)
            arrays.append(pa.array([row.get(column.name) for row in table.rows], type=typ))
            fields.append(pa.field(column.name, typ, nullable=column.nullable))
        return pa.RecordBatch.from_arrays(arrays, schema=pa.schema(fields))

    def run(self, tables: list[TableData], program: Program, timeout_s: float = 5.0) -> BackendResult:
        start = time.perf_counter()
        try:
            from datafusion import SessionContext
            import pyarrow as pa

            ctx = SessionContext()
            table_by_name = {table.name: table for table in tables}
            current_cols = [c.name for c in tables[0].columns]
            for table in tables:
                ctx.register_record_batches(table.name, [[self._to_record_batch(table, pa)]])

            query = f"SELECT * FROM {_quote(tables[0].name)}"
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
                        f"WHERE q.{_quote(op['column'])} {op['cmp']} {_lit(op['value'])}"
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
                            f"ORDER BY {_order_clause(pending_order)} OFFSET {int(op['n'])}"
                        )
                    else:
                        query = f"SELECT * FROM ({query}) q OFFSET {int(op['n'])}"
                    pending_order = None
                elif kind == "mutate":
                    expr = op["expr"]
                    if expr["kind"] == "add_const":
                        expr_sql = f"q.{_quote(expr['source'])} + {_lit(expr['value'])}"
                    elif expr["kind"] == "arith_const":
                        op_sql = {"sub": "-", "mul": "*", "div": "/", "mod": "%"}[expr["op"]]
                        source_sql = (
                            f"CAST(q.{_quote(expr['source'])} AS DOUBLE)"
                            if expr["op"] == "div"
                            else f"q.{_quote(expr['source'])}"
                        )
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
            out = ctx.sql(query).to_pandas()
            return BackendResult(self.name, "ok", data=out, duration_ms=(time.perf_counter() - start) * 1000)
        except Exception as exc:  # noqa: BLE001
            return BackendResult(
                self.name,
                "error",
                error_type=type(exc).__name__,
                error=str(exc),
                duration_ms=(time.perf_counter() - start) * 1000,
            )


def _arrow_type(pa, kind: str):
    if kind == "int":
        return pa.int64()
    if kind == "float":
        return pa.float64()
    if kind == "bool":
        return pa.bool_()
    return pa.string()
