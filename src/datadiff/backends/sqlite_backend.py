from __future__ import annotations

import time
from typing import Any

from datadiff.backends.base import Backend, BackendResult, PreparedTable, prepare_table
from datadiff.backends.dataframe_semantics import running_sum_plan, tuple_absence_plan
from datadiff.backends.probe_semantics import EXTENDED_FALSE_PROBE_KINDS
from datadiff.backends.sql_lowering import (
    SqlDialect,
    cast_logical_expression,
    render_aggregate_sql,
    render_case_when_expr,
    render_coalesce_expr,
    render_filter_condition,
    render_fill_null_expr,
    render_groupby_sql,
    render_join_condition,
    render_join_right_projection,
    render_mutate_expr,
    render_semi_anti_join_condition,
)
from datadiff.backends.sql_runtime import build_subquery_runtime
from datadiff.dsl import Program, SortKey, TableData, normalize_sort_keys
from datadiff.join_keys import join_key_pairs
from datadiff.operation_semantics import (
    groupby_keys,
    is_default_false_probe_kind,
    op_ascending,
    op_column,
    op_columns,
    op_comparator,
    op_kind,
    op_n,
    op_nulls,
    op_output_alias,
    op_table,
    op_value,
)
from datadiff.pathing import path_basename
from datadiff.program_state import ProgramState, state_after_operation
from datadiff.running import running_sum_partition_columns, running_sum_sort_keys
from datadiff.sortedness import is_sorted_values
from datadiff.sqlite_runtime import sqlite3
from datadiff.windowing import row_number_order_keys, row_number_partition_columns


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _lit(value: Any) -> str:
    import math

    if value is None:
        return "NULL"
    if isinstance(value, (list, tuple)):
        return "(" + ", ".join(_lit(item) for item in value) + ")"
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
    return SQLITE_DIALECT.cast_type(kind)


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


def _agg_expr(column: str, func: str) -> str:
    quoted = _quote(column)
    if func == "nunique":
        return f"COUNT(DISTINCT {quoted})"
    if func == "any":
        return (
            f"CASE WHEN COUNT({quoted}) = 0 THEN NULL "
            f"ELSE MAX(CASE WHEN {quoted} IS NULL THEN NULL WHEN {quoted} THEN 1 ELSE 0 END) END"
        )
    if func == "all":
        return (
            f"CASE WHEN COUNT({quoted}) = 0 THEN NULL "
            f"ELSE MIN(CASE WHEN {quoted} IS NULL THEN NULL WHEN {quoted} THEN 1 ELSE 0 END) END"
        )
    if func == "mean":
        return f"AVG({quoted})"
    sql_func = "COUNT" if func == "count" else func.upper()
    return f"{sql_func}({quoted})"


SQLITE_DIALECT = SqlDialect(
    logical_type_sql={
        "bool": "INTEGER",
        "float": "REAL",
        "int": "INTEGER",
        "str": "TEXT",
    },
    string_slice_fn="SUBSTR",
    string_length_sql=lambda source: f"LENGTH({source})",
    string_contains_sql=lambda source, needle: (
        f"CASE WHEN {source} IS NULL THEN NULL ELSE INSTR({source}, {needle}) > 0 END"
    ),
    string_startswith_fn=lambda source, needle: (
        f"CASE WHEN {source} IS NULL THEN NULL ELSE SUBSTR({source}, 1, LENGTH({needle})) = {needle} END"
    ),
    string_endswith_fn=lambda source, needle: (
        f"CASE WHEN {source} IS NULL THEN NULL ELSE SUBSTR({source}, LENGTH({source}) - LENGTH({needle}) + 1, LENGTH({needle})) = {needle} END"
    ),
    date_part_spans={"year": (1, 4), "month": (6, 2), "day": (9, 2)},
    basename_sql=lambda source: f"__datadiff_basename({source})",
    split_part_sql=lambda source, sep, _index: (
        f"CASE WHEN {source} IS NULL THEN NULL "
        f"WHEN INSTR({source}, {sep}) > 0 THEN SUBSTR({source}, 1, INSTR({source}, {sep}) - 1) "
        f"ELSE {source} END"
    ),
    division_sql=lambda source, value: f"1.0 * {source} / {value}",
    reverse_division_sql=lambda numerator, source: f"1.0 * {numerator} / {source}",
)


def _dataframe_preserving_sqlite_scalars(pd: Any, rows: list[tuple[Any, ...]], columns: list[str]):
    # A nullable INTEGER result must not be inferred as float64 before normalization:
    # values above 2**53 would then be irreversibly rounded.
    return pd.DataFrame(rows, columns=columns, dtype=object)


def _tuple_absence_native_condition(op: dict[str, Any]) -> str:
    right_table, left_columns, right_columns = tuple_absence_plan(op)
    left_sql = ", ".join(f"q.{_quote(column)}" for column in left_columns)
    right_sql = ", ".join(_quote(column) for column in right_columns)
    return f"({left_sql}) NOT IN (SELECT {right_sql} FROM {_quote(right_table)})"


def _running_sum_projection(cols: list[str], op: dict[str, Any]) -> tuple[str, list[str]]:
    plan = running_sum_plan(op)
    kept_cols = [col for col in cols if col != plan.column]
    select_parts = [f"q.{_quote(col)}" for col in kept_cols]
    order_sql = _order_clause(running_sum_sort_keys(op))
    partition_columns = running_sum_partition_columns(op)
    partition_sql = ""
    if partition_columns:
        partition_sql = "PARTITION BY " + ", ".join(f"q.{_quote(column)}" for column in partition_columns) + " "
    expr_sql = (
        f"SUM(CAST(q.{_quote(plan.source)} AS REAL)) OVER ("
        f"{partition_sql}ORDER BY {order_sql} ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
        f") AS {_quote(plan.column)}"
    )
    select_parts.append(expr_sql)
    return ", ".join(select_parts), kept_cols + [plan.column]


def _row_number_window_sql(op: dict[str, Any]) -> str:
    partition_columns = row_number_partition_columns(op)
    partition_sql = ""
    if partition_columns:
        partition_sql = "PARTITION BY " + ", ".join(f"q.{_quote(column)}" for column in partition_columns) + " "
    return f"{partition_sql}ORDER BY {_order_clause(row_number_order_keys(op))}"


def _row_number_filter_condition(op: dict[str, Any], column: str) -> str:
    comparator = {"==": "=", "<": "<", "<=": "<="}[op_comparator(op, "==")]
    return f"{_quote(column)} {comparator} {int(op_value(op) or 1)}"


def _scalar_subquery_probe_sql(op: dict[str, Any]) -> str:
    alias = op_output_alias(op)
    return (
        "WITH tenk1(unique1, unique2, two, four, ten, twenty, hundred, thousand) AS ("
        "VALUES (1,1,1,1,1,1,1,1), (2,2,2,2,2,2,2,2)"
        "), got AS ("
        "SELECT (SELECT max((SELECT i.unique2 FROM tenk1 i WHERE i.unique1 = o.unique1))) AS probe_value "
        "FROM tenk1 o"
        ") "
        f"SELECT NOT (COUNT(*) = 1 AND MIN(probe_value) = 2 AND MAX(probe_value) = 2) AS {_quote(alias)} "
        "FROM got"
    )


class SQLiteBackend(Backend):
    name = "sqlite"

    def execute_lowered(
        self,
        tables: list[TableData | PreparedTable],
        program: Program,
        timeout_s: float = 5.0,
    ) -> BackendResult:
        start = time.perf_counter()
        try:
            import pandas as pd

            prepared_tables = [prepare_table(table) for table in tables]
            table_by_name = {table.name: table for table in prepared_tables}
            semantic_state = ProgramState.from_table(prepared_tables[0])
            current_cols = [c.name for c in prepared_tables[0].columns]
            con = sqlite3.connect(":memory:")
            con.create_function("__datadiff_basename", 1, path_basename)
            for table in prepared_tables:
                col_defs = ", ".join(f"{_quote(c.name)} {_sql_type(c.type)}" for c in table.columns)
                con.execute(f"CREATE TABLE {_quote(table.name)} ({col_defs})")
                if table.row_tuples:
                    cols = list(table.column_names)
                    placeholders = ", ".join("?" for _ in cols)
                    con.executemany(
                        f"INSERT INTO {_quote(table.name)} ({', '.join(_quote(c) for c in cols)}) VALUES ({placeholders})",
                        table.row_tuples,
                    )
            query = "SELECT * FROM t0"
            runtime = build_subquery_runtime(
                query,
                current_cols,
                quote=_quote,
                order_clause=_order_clause,
            )

            def _reset_probe_query(alias: str, query_sql: str) -> None:
                nonlocal query
                query = runtime.assign_source(query_sql)
                runtime.reset_source(query_sql, alias)

            def visible_projection() -> str:
                return runtime.visible_projection()

            def drop_hidden_order_cols() -> None:
                nonlocal query
                if runtime.drop_hidden_order_cols():
                    query = runtime.source

            def materialize_visible_query():
                body = runtime.materialize_sql()
                cur = con.execute(body)
                columns = [desc[0] for desc in cur.description]
                return _dataframe_preserving_sqlite_scalars(pd, cur.fetchall(), columns)

            def select_with_pending_order(cols: list[str]) -> str:
                return runtime.select_with_pending_order(cols)

            def freeze_pending_order() -> None:
                nonlocal query
                if runtime.freeze_pending_order():
                    query = runtime.source

            for op in program.operations:
                kind = op_kind(op)
                next_semantic_state = state_after_operation(
                    semantic_state,
                    op,
                    tables=table_by_name,
                )
                if kind == "join":
                    drop_hidden_order_cols()
                    right = table_by_name[op_table(op)]
                    _, right_keys = join_key_pairs(op)
                    right_types = {column.name: column.type for column in right.columns}
                    select_right, projected_right = render_join_right_projection(
                        right.column_names,
                        right_keys,
                        semantic_state.columns,
                        _quote,
                    )
                    join_kind = "LEFT JOIN" if op["how"] == "left" else "INNER JOIN"
                    query = runtime.assign_source(
                        f"SELECT q.*{select_right} FROM ({query}) q {join_kind} {_quote(right.name)} r "
                        f"ON {render_join_condition(op, SQLITE_DIALECT, _quote, semantic_state.column_types, right_types)}"
                    )
                    runtime.state.current_cols.extend(projected_right)
                    runtime.state.visible_cols = list(runtime.state.current_cols)
                    runtime.state.pending_order = None
                elif kind == "union_all":
                    drop_hidden_order_cols()
                    right_projection = ", ".join(_quote(col) for col in runtime.state.visible_cols)
                    query = runtime.assign_source(
                        f"SELECT {visible_projection()} FROM ({query}) q "
                        f"UNION ALL SELECT {right_projection} FROM {_quote(op_table(op))}"
                    )
                    runtime.state.current_cols = list(runtime.state.visible_cols)
                    runtime.state.pending_order = None
                elif kind in {"semi_join", "anti_join"}:
                    right = table_by_name[op_table(op)]
                    right_types = {column.name: column.type for column in right.columns}
                    condition = render_semi_anti_join_condition(
                        op,
                        kind,
                        SQLITE_DIALECT,
                        _quote,
                        semantic_state.column_types,
                        right_types,
                    )
                    query = runtime.assign_source(f"SELECT * FROM ({query}) q WHERE {condition}")
                elif kind == "drop_nulls":
                    condition = " AND ".join(f"q.{_quote(column)} IS NOT NULL" for column in op["columns"])
                    query = runtime.assign_source(f"SELECT * FROM ({query}) q WHERE {condition}")
                elif kind == "filter":
                    condition = render_filter_condition(
                        f"q.{_quote(op_column(op))}",
                        _lit(op_value(op)),
                        op_comparator(op),
                        SQLITE_DIALECT,
                    )
                    query = runtime.assign_source(
                        f"SELECT * FROM ({query}) q "
                        f"WHERE {condition}"
                    )
                elif kind == "tuple_absence_filter":
                    query = runtime.assign_source(
                        f"SELECT * FROM ({query}) q "
                        f"WHERE {_tuple_absence_native_condition(op)}"
                    )
                elif kind == "row_number_filter":
                    drop_hidden_order_cols()
                    ordinal_col = "__datadiff_row_number"
                    query = runtime.assign_source(
                        f"SELECT {visible_projection()} FROM ("
                        f"SELECT q.*, ROW_NUMBER() OVER ({_row_number_window_sql(op)}) AS {_quote(ordinal_col)} "
                        f"FROM ({query}) q"
                        f") q WHERE {_row_number_filter_condition(op, ordinal_col)}"
                    )
                    runtime.state.current_cols = list(runtime.state.visible_cols)
                    runtime.state.pending_order = [
                        *(SortKey(column, True, "last") for column in row_number_partition_columns(op)),
                        *row_number_order_keys(op),
                    ]
                elif kind == "running_sum":
                    drop_hidden_order_cols()
                    order_keys = running_sum_sort_keys(op)
                    projection, runtime.state.current_cols = _running_sum_projection(runtime.state.current_cols, op)
                    query = runtime.assign_source(f"SELECT {projection} FROM ({query}) q")
                    out_column = op_column(op)
                    runtime.state.visible_cols = [col for col in runtime.state.visible_cols if col != out_column] + [out_column]
                    runtime.state.pending_order = order_keys
                elif kind == "sortedness_check":
                    materialized = materialize_visible_query()
                    ok = is_sorted_values(
                        materialized[op_column(op)].tolist(),
                        ascending=op_ascending(op),
                        nulls=op_nulls(op),
                    )
                    alias = op_output_alias(op)
                    _reset_probe_query(alias, f"SELECT {1 if ok else 0} AS {_quote(alias)}")
                elif kind == "scalar_subquery_probe":
                    alias = op_output_alias(op)
                    _reset_probe_query(alias, _scalar_subquery_probe_sql(op))
                elif kind in EXTENDED_FALSE_PROBE_KINDS or is_default_false_probe_kind(kind):
                    alias = op_output_alias(op)
                    _reset_probe_query(alias, f"SELECT 0 AS {_quote(alias)}")
                elif kind == "select":
                    cols = op_columns(op)
                    projection = select_with_pending_order(cols)
                    query = runtime.assign_source(f"SELECT {projection} FROM ({query}) q")
                elif kind == "distinct":
                    drop_hidden_order_cols()
                    cols = op_columns(op)
                    projection = ", ".join(f"q.{_quote(c)}" for c in cols)
                    query = runtime.assign_source(f"SELECT DISTINCT {projection} FROM ({query}) q")
                    runtime.state.reset_projection(cols)
                elif kind == "fill_null":
                    column, expr_sql = render_fill_null_expr(op, _quote, _lit)
                    output_type = next_semantic_state.column_types.get(column)
                    if output_type is not None:
                        expr_sql = cast_logical_expression(expr_sql, output_type, SQLITE_DIALECT)
                    if runtime.state.pending_order_mentions(column):
                        freeze_pending_order()
                    projection = runtime.state.replace_projection_expr(column, expr_sql, _quote)
                    query = runtime.assign_source(f"SELECT {projection} FROM ({query}) q")
                    runtime.state.replace_visible_column(column)
                    if runtime.state.pending_order_mentions(column):
                        runtime.state.clear_pending_order()
                        drop_hidden_order_cols()
                elif kind == "coalesce":
                    alias, expr_sql = render_coalesce_expr(op, _quote, _lit)
                    output_type = next_semantic_state.column_types.get(alias)
                    if output_type is not None:
                        expr_sql = cast_logical_expression(expr_sql, output_type, SQLITE_DIALECT)
                    if runtime.state.pending_order_mentions(alias):
                        freeze_pending_order()
                    projection = runtime.state.replace_projection_expr(alias, expr_sql, _quote)
                    query = runtime.assign_source(f"SELECT {projection} FROM ({query}) q")
                    runtime.state.replace_visible_column(alias)
                    if runtime.state.pending_order_mentions(alias):
                        runtime.state.clear_pending_order()
                        drop_hidden_order_cols()
                elif kind == "case_when":
                    alias, expr_sql = render_case_when_expr(op, SQLITE_DIALECT, _quote, _lit)
                    output_type = next_semantic_state.column_types.get(alias)
                    if output_type is not None:
                        expr_sql = cast_logical_expression(expr_sql, output_type, SQLITE_DIALECT)
                    if runtime.state.pending_order_mentions(alias):
                        freeze_pending_order()
                    projection = runtime.state.replace_projection_expr(alias, expr_sql, _quote)
                    query = runtime.assign_source(f"SELECT {projection} FROM ({query}) q")
                    runtime.state.replace_visible_column(alias)
                    if runtime.state.pending_order_mentions(alias):
                        runtime.state.clear_pending_order()
                        drop_hidden_order_cols()
                elif kind == "sort":
                    drop_hidden_order_cols()
                    runtime.state.pending_order = normalize_sort_keys(op)
                elif kind == "limit":
                    if runtime.state.pending_order is not None:
                        query = runtime.assign_source(
                            f"SELECT * FROM ({query}) q "
                            f"ORDER BY {_order_clause(runtime.state.pending_order)} LIMIT {op_n(op)}"
                        )
                    else:
                        query = runtime.assign_source(f"SELECT * FROM ({query}) q LIMIT {op_n(op)}")
                elif kind == "offset":
                    if runtime.state.pending_order is not None:
                        query = runtime.assign_source(
                            f"SELECT * FROM ({query}) q "
                            f"ORDER BY {_order_clause(runtime.state.pending_order)} LIMIT -1 OFFSET {op_n(op)}"
                        )
                    else:
                        query = runtime.assign_source(f"SELECT * FROM ({query}) q LIMIT -1 OFFSET {op_n(op)}")
                elif kind == "mutate":
                    out_column, expr_sql = render_mutate_expr(
                        op,
                        SQLITE_DIALECT,
                        _quote,
                        _lit,
                        semantic_state.column_types,
                    )
                    output_type = next_semantic_state.column_types.get(out_column)
                    if output_type is not None:
                        expr_sql = cast_logical_expression(expr_sql, output_type, SQLITE_DIALECT)
                    if runtime.state.pending_order_mentions(out_column):
                        freeze_pending_order()
                    projection = runtime.state.replace_projection_expr(out_column, expr_sql, _quote)
                    query = runtime.assign_source(f"SELECT {projection} FROM ({query}) q")
                    runtime.state.replace_visible_column(out_column)
                    if runtime.state.pending_order_mentions(out_column):
                        runtime.state.clear_pending_order()
                        drop_hidden_order_cols()
                elif kind == "groupby":
                    drop_hidden_order_cols()
                    keys = groupby_keys(op)
                    select_sql, aliases = render_groupby_sql(op, _agg_expr, _quote, keys)
                    query = runtime.assign_source(
                        f"SELECT {select_sql} "
                        f"FROM ({query}) q GROUP BY {', '.join(_quote(key) for key in keys)}"
                    )
                    runtime.state.reset_projection(keys + aliases)
                elif kind == "aggregate":
                    drop_hidden_order_cols()
                    agg_sql, aliases = render_aggregate_sql(op, _agg_expr, _quote)
                    query = runtime.assign_source(f"SELECT {', '.join(agg_sql)} FROM ({query}) q")
                    runtime.state.reset_projection(aliases)
                else:
                    raise ValueError(kind)
                semantic_state = next_semantic_state
            query = runtime.assign_source(runtime.finalize_source())
            cur = con.execute(query)
            columns = [desc[0] for desc in cur.description]
            out = _dataframe_preserving_sqlite_scalars(pd, cur.fetchall(), columns)
            for col in out.columns:
                if semantic_state.column_types.get(str(col)) == "bool":
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
