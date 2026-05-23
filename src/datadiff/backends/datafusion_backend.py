from __future__ import annotations

import math
import time
from typing import Any

from datadiff.backends.base import Backend, BackendResult
from datadiff.dsl import Program, SortKey, TableData, normalize_sort_keys
from datadiff.filtering import sql_filter_condition
from datadiff.sortedness import is_sorted_values


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _lit(value: Any) -> str:
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


def _agg_expr(column: str, func: str) -> str:
    if func == "nunique":
        return f"COUNT(DISTINCT {_quote(column)})"
    sql_func = "COUNT" if func == "count" else func.upper()
    return f"{sql_func}({_quote(column)})"


def _tuple_absence_safe_condition(op: dict[str, Any]) -> str:
    definite_inequalities = [
        (
            f"q.{_quote(left)} IS NOT NULL AND r.{_quote(right)} IS NOT NULL "
            f"AND q.{_quote(left)} <> r.{_quote(right)}"
        )
        for left, right in zip(op["columns"], op["right_columns"])
    ]
    row_equality_is_not_false = "NOT (" + " OR ".join(definite_inequalities) + ")"
    return (
        "NOT EXISTS ("
        f"SELECT 1 FROM {_quote(op['table'])} r "
        f"WHERE {row_equality_is_not_false}"
        ")"
    )


def _running_sum_projection(cols: list[str], op: dict[str, Any]) -> tuple[str, list[str]]:
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
            visible_cols = list(current_cols)
            hidden_order_cols: list[str] = []

            def visible_projection() -> str:
                return ", ".join(f"q.{_quote(col)}" for col in visible_cols)

            def drop_hidden_order_cols() -> None:
                nonlocal query, current_cols, hidden_order_cols
                if not hidden_order_cols:
                    return
                query = f"SELECT {visible_projection()} FROM ({query}) q"
                current_cols = list(visible_cols)
                hidden_order_cols = []

            def materialize_visible_query():
                if pending_order is not None:
                    body = (
                        f"SELECT {visible_projection()} FROM ({query}) q "
                        f"ORDER BY {_order_clause(pending_order)}"
                    )
                else:
                    drop_hidden_order_cols()
                    body = f"SELECT {visible_projection()} FROM ({query}) q"
                return ctx.sql(body).to_pandas()

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
                    query = (
                        f"SELECT q.*{select_right} FROM ({query}) q {join_kind} {_quote(right.name)} r "
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
                    query = (
                        f"SELECT * FROM ({query}) q "
                        f"WHERE {condition}"
                    )
                elif kind == "tuple_absence_filter":
                    query = (
                        f"SELECT * FROM ({query}) q "
                        f"WHERE {_tuple_absence_safe_condition(op)}"
                    )
                elif kind == "running_sum":
                    drop_hidden_order_cols()
                    order_keys = normalize_sort_keys({"keys": op["order_by"]})
                    projection, current_cols = _running_sum_projection(current_cols, op)
                    query = f"SELECT {projection} FROM ({query}) q"
                    visible_cols = [col for col in visible_cols if col != op["column"]] + [op["column"]]
                    pending_order = order_keys
                elif kind == "sortedness_check":
                    materialized = materialize_visible_query()
                    ok = is_sorted_values(
                        materialized[op["column"]].tolist(),
                        ascending=bool(op.get("ascending", True)),
                        nulls=str(op.get("nulls", "last")),
                    )
                    materialized_name = f"__datadiff_sortedness_{len(current_cols)}"
                    arrow_table = pa.Table.from_pydict({op["as"]: [ok]})
                    ctx.register_record_batches(materialized_name, [arrow_table.to_batches()])
                    query = f"SELECT * FROM {_quote(materialized_name)}"
                    current_cols = [op["as"]]
                    visible_cols = [op["as"]]
                    hidden_order_cols = []
                    pending_order = None
                elif kind == "random_case_probe":
                    query = f"SELECT false AS {_quote(op['as'])}"
                    current_cols = [op["as"]]
                    visible_cols = [op["as"]]
                    hidden_order_cols = []
                    pending_order = None
                elif kind == "group_quantile_probe":
                    query = f"SELECT false AS {_quote(op['as'])}"
                    current_cols = [op["as"]]
                    visible_cols = [op["as"]]
                    hidden_order_cols = []
                    pending_order = None
                elif kind == "scalar_subquery_probe":
                    query = f"SELECT false AS {_quote(op['as'])}"
                    current_cols = [op["as"]]
                    visible_cols = [op["as"]]
                    hidden_order_cols = []
                    pending_order = None
                elif kind == "window_avg_probe":
                    query = f"SELECT false AS {_quote(op['as'])}"
                    current_cols = [op["as"]]
                    visible_cols = [op["as"]]
                    hidden_order_cols = []
                    pending_order = None
                elif kind == "struct_distinct_probe":
                    query = f"SELECT false AS {_quote(op['as'])}"
                    current_cols = [op["as"]]
                    visible_cols = [op["as"]]
                    hidden_order_cols = []
                    pending_order = None
                elif kind == "bit_compare_probe":
                    query = f"SELECT false AS {_quote(op['as'])}"
                    current_cols = [op["as"]]
                    visible_cols = [op["as"]]
                    hidden_order_cols = []
                    pending_order = None
                elif kind == "round_even_probe":
                    query = f"SELECT false AS {_quote(op['as'])}"
                    current_cols = [op["as"]]
                    visible_cols = [op["as"]]
                    hidden_order_cols = []
                    pending_order = None
                elif kind == "series_rtruediv_probe":
                    query = f"SELECT false AS {_quote(op['as'])}"
                    current_cols = [op["as"]]
                    visible_cols = [op["as"]]
                    hidden_order_cols = []
                    pending_order = None
                elif kind == "uint64_isin_probe":
                    query = f"SELECT false AS {_quote(op['as'])}"
                    current_cols = [op["as"]]
                    visible_cols = [op["as"]]
                    hidden_order_cols = []
                    pending_order = None
                elif kind == "tuple_anti_null_probe":
                    query = f"SELECT false AS {_quote(op['as'])}"
                    current_cols = [op["as"]]
                    visible_cols = [op["as"]]
                    hidden_order_cols = []
                    pending_order = None
                elif kind == "sparse_mask_probe":
                    query = f"SELECT false AS {_quote(op['as'])}"
                    current_cols = [op["as"]]
                    visible_cols = [op["as"]]
                    hidden_order_cols = []
                    pending_order = None
                elif kind == "float_wrap_probe":
                    query = f"SELECT false AS {_quote(op['as'])}"
                    current_cols = [op["as"]]
                    visible_cols = [op["as"]]
                    hidden_order_cols = []
                    pending_order = None
                elif kind == "index_bool_probe":
                    query = f"SELECT false AS {_quote(op['as'])}"
                    current_cols = [op["as"]]
                    visible_cols = [op["as"]]
                    hidden_order_cols = []
                    pending_order = None
                elif kind == "empty_literal_groupby_probe":
                    query = f"SELECT false AS {_quote(op['as'])}"
                    current_cols = [op["as"]]
                    visible_cols = [op["as"]]
                    hidden_order_cols = []
                    pending_order = None
                elif kind == "arrow_string_eq_sum_probe":
                    query = f"SELECT false AS {_quote(op['as'])}"
                    current_cols = [op["as"]]
                    visible_cols = [op["as"]]
                    hidden_order_cols = []
                    pending_order = None
                elif kind == "arrow_timestamp_loc_slice_probe":
                    query = f"SELECT false AS {_quote(op['as'])}"
                    current_cols = [op["as"]]
                    visible_cols = [op["as"]]
                    hidden_order_cols = []
                    pending_order = None
                elif kind == "arrow_timestamp_index_attr_probe":
                    query = f"SELECT false AS {_quote(op['as'])}"
                    current_cols = [op["as"]]
                    visible_cols = [op["as"]]
                    hidden_order_cols = []
                    pending_order = None
                elif kind == "dataset_isin_all_match_probe":
                    query = f"SELECT false AS {_quote(op['as'])}"
                    current_cols = [op["as"]]
                    visible_cols = [op["as"]]
                    hidden_order_cols = []
                    pending_order = None
                elif kind == "select":
                    cols = list(op["columns"])
                    projection = select_with_pending_order(cols)
                    query = f"SELECT {projection} FROM ({query}) q"
                    visible_cols = cols
                    current_cols = cols + [col for col in hidden_order_cols if col not in cols]
                elif kind == "sort":
                    drop_hidden_order_cols()
                    pending_order = normalize_sort_keys(op)
                elif kind == "limit":
                    if pending_order is not None:
                        query = (
                            f"SELECT * FROM ({query}) q "
                            f"ORDER BY {_order_clause(pending_order)} LIMIT {int(op['n'])}"
                        )
                    else:
                        query = f"SELECT * FROM ({query}) q LIMIT {int(op['n'])}"
                elif kind == "offset":
                    if pending_order is not None:
                        query = (
                            f"SELECT * FROM ({query}) q "
                            f"ORDER BY {_order_clause(pending_order)} OFFSET {int(op['n'])}"
                        )
                    else:
                        query = f"SELECT * FROM ({query}) q OFFSET {int(op['n'])}"
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
                    query = (
                        f"SELECT {key_sql}, {', '.join(agg_sql)} "
                        f"FROM ({query}) q GROUP BY {key_sql}"
                    )
                    current_cols = keys + [agg["as"] for agg in op["aggs"]]
                    visible_cols = list(current_cols)
                    pending_order = None
                elif kind == "aggregate":
                    drop_hidden_order_cols()
                    agg_sql = []
                    for agg in op["aggs"]:
                        agg_sql.append(f"{_agg_expr(agg['column'], agg['func'])} AS {_quote(agg['as'])}")
                    query = f"SELECT {', '.join(agg_sql)} FROM ({query}) q"
                    current_cols = [agg["as"] for agg in op["aggs"]]
                    visible_cols = list(current_cols)
                    pending_order = None
                else:
                    raise ValueError(kind)
            if pending_order is not None:
                query = f"SELECT {visible_projection()} FROM ({query}) q ORDER BY {_order_clause(pending_order)}"
            else:
                drop_hidden_order_cols()
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
