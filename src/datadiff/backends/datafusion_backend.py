from __future__ import annotations

import math
import time
from typing import Any

from datadiff.backends.base import Backend, BackendResult, PreparedTable, prepare_table
from datadiff.backends.dataframe_semantics import running_sum_plan, tuple_absence_plan
from datadiff.backends.probe_semantics import EXTENDED_FALSE_PROBE_KINDS
from datadiff.backends.sql_lowering import (
    SqlDialect,
    render_aggregate_sql,
    render_case_when_expr,
    render_coalesce_expr,
    render_fill_null_expr,
    render_groupby_sql,
    render_mutate_expr,
)
from datadiff.backends.sql_runtime import build_subquery_runtime
from datadiff.dsl import Program, SortKey, TableData, normalize_sort_keys
from datadiff.filtering import sql_filter_condition
from datadiff.join_keys import join_key_pairs
from datadiff.operation_semantics import groupby_keys, is_default_false_probe_kind, join_how, op_ascending, op_column, op_columns, op_comparator, op_kind, op_n, op_nulls, op_output_alias, op_table, op_value
from datadiff.running import running_sum_partition_columns, running_sum_sort_keys
from datadiff.sortedness import is_sorted_values
from datadiff.windowing import row_number_order_keys, row_number_partition_columns


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


def _order_clause(sort_keys: list[SortKey]) -> str:
    return ", ".join(
        f"{_quote(key.column)} {'ASC' if key.ascending else 'DESC'} NULLS {key.nulls.upper()}"
        for key in sort_keys
    )


def _agg_expr(column: str, func: str) -> str:
    if func == "nunique":
        return f"COUNT(DISTINCT {_quote(column)})"
    if func == "any":
        return f"BOOL_OR({_quote(column)})"
    if func == "all":
        return f"BOOL_AND({_quote(column)})"
    if func == "mean":
        return f"AVG({_quote(column)})"
    sql_func = "COUNT" if func == "count" else func.upper()
    return f"{sql_func}({_quote(column)})"


DATAFUSION_DIALECT = SqlDialect(
    float_cast_type="DOUBLE",
    int_cast_type="BIGINT",
    str_cast_type="VARCHAR",
    string_slice_fn="SUBSTRING",
    string_startswith_fn=lambda source, needle: (
        f"CASE WHEN {source} IS NULL THEN NULL ELSE SUBSTRING({source}, 1, LENGTH({needle})) = {needle} END"
    ),
    string_endswith_fn=lambda source, needle: (
        f"CASE WHEN {source} IS NULL THEN NULL ELSE SUBSTRING({source}, LENGTH({source}) - LENGTH({needle}) + 1, LENGTH({needle})) = {needle} END"
    ),
    date_part_spans={"year": (1, 4), "month": (6, 2), "day": (9, 2)},
    basename_sql=lambda source: f"regexp_replace({source}, '^.*[\\\\/]', '')",
    split_part_sql=lambda source, sep, index: f"SPLIT_PART({source}, {sep}, {index + 1})",
    division_sql=lambda source, value: f"CAST({source} AS DOUBLE) / {value}",
    reverse_division_sql=lambda numerator, source: f"CAST({numerator} AS DOUBLE) / {source}",
)


def _join_condition(op: dict[str, Any]) -> str:
    left_keys, right_keys = join_key_pairs(op)
    return " AND ".join(f"q.{_quote(left)} = r.{_quote(right)}" for left, right in zip(left_keys, right_keys))


def _semi_anti_join_condition(op: dict[str, Any], kind: str) -> str:
    left_keys, right_keys = join_key_pairs(op)
    predicates = [
        *(f"r.{_quote(right)} IS NOT NULL" for right in right_keys),
        *(f"q.{_quote(left)} = r.{_quote(right)}" for left, right in zip(left_keys, right_keys)),
    ]
    exists_sql = f"EXISTS (SELECT 1 FROM {_quote(op_table(op))} r WHERE {' AND '.join(predicates)})"
    return exists_sql if kind == "semi_join" else f"NOT {exists_sql}"


def _tuple_absence_safe_condition(op: dict[str, Any]) -> str:
    right_table, left_columns, right_columns = tuple_absence_plan(op)
    definite_inequalities = [
        (
            f"q.{_quote(left)} IS NOT NULL AND r.{_quote(right)} IS NOT NULL "
            f"AND q.{_quote(left)} <> r.{_quote(right)}"
        )
        for left, right in zip(left_columns, right_columns)
    ]
    row_equality_is_not_false = "NOT (" + " OR ".join(definite_inequalities) + ")"
    return (
        "NOT EXISTS ("
        f"SELECT 1 FROM {_quote(right_table)} r "
        f"WHERE {row_equality_is_not_false}"
        ")"
    )


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
        f"SUM(CAST(q.{_quote(plan.source)} AS DOUBLE)) OVER ("
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
    return f"q.{_quote(column)} {comparator} {int(op_value(op) or 1)}"


def _setop_all_duplicate_probe_sql(op: dict[str, Any]) -> str:
    alias = op_output_alias(op)
    return (
        "WITH except_got AS ("
        "SELECT * FROM (VALUES ('a'), ('b'), ('b'), ('c'), ('c'), ('c')) AS lhs(v) "
        "EXCEPT ALL "
        "SELECT * FROM (VALUES ('b'), ('c')) AS rhs(v)"
        "), intersect_got AS ("
        "SELECT * FROM (VALUES ('a'), ('b'), ('b'), ('c'), ('c'), ('c')) AS lhs(v) "
        "INTERSECT ALL "
        "SELECT * FROM (VALUES ('b'), ('b'), ('b'), ('c'), ('c')) AS rhs(v)"
        "), observed(bucket, row_count) AS ("
        "SELECT 'except_a', COUNT(*) FROM except_got WHERE v = 'a' "
        "UNION ALL SELECT 'except_b', COUNT(*) FROM except_got WHERE v = 'b' "
        "UNION ALL SELECT 'except_c', COUNT(*) FROM except_got WHERE v = 'c' "
        "UNION ALL SELECT 'intersect_b', COUNT(*) FROM intersect_got WHERE v = 'b' "
        "UNION ALL SELECT 'intersect_c', COUNT(*) FROM intersect_got WHERE v = 'c'"
        ") "
        "SELECT NOT ("
        "SUM(CASE WHEN bucket = 'except_a' THEN row_count ELSE 0 END) = 1 "
        "AND SUM(CASE WHEN bucket = 'except_b' THEN row_count ELSE 0 END) = 1 "
        "AND SUM(CASE WHEN bucket = 'except_c' THEN row_count ELSE 0 END) = 2 "
        "AND SUM(CASE WHEN bucket = 'intersect_b' THEN row_count ELSE 0 END) = 2 "
        "AND SUM(CASE WHEN bucket = 'intersect_c' THEN row_count ELSE 0 END) = 2"
        f") AS {_quote(alias)} "
        "FROM observed"
    )


class DataFusionBackend(Backend):
    name = "datafusion"

    def _to_record_batch(self, table: TableData | PreparedTable, pa):
        prepared = prepare_table(table)
        arrays = []
        fields = []
        for column in prepared.columns:
            typ = _arrow_type(pa, column.type)
            arrays.append(pa.array(prepared.columns_data[column.name], type=typ))
            fields.append(pa.field(column.name, typ, nullable=column.nullable))
        return pa.RecordBatch.from_arrays(arrays, schema=pa.schema(fields))

    def run(
        self,
        tables: list[TableData | PreparedTable],
        program: Program,
        timeout_s: float = 5.0,
    ) -> BackendResult:
        start = time.perf_counter()
        try:
            from datafusion import SessionContext
            import pyarrow as pa

            ctx = SessionContext()
            table_by_name = {table.name: table for table in tables}
            for table in tables:
                ctx.register_record_batches(table.name, [[self._to_record_batch(table, pa)]])

            query = f"SELECT * FROM {_quote(tables[0].name)}"
            runtime = build_subquery_runtime(
                query,
                [c.name for c in tables[0].columns],
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
                return ctx.sql(body).to_pandas()

            def select_with_pending_order(cols: list[str]) -> str:
                return runtime.select_with_pending_order(cols)

            def freeze_pending_order() -> None:
                nonlocal query
                if runtime.freeze_pending_order():
                    query = runtime.source

            for op in program.operations:
                kind = op_kind(op)
                if kind == "join":
                    drop_hidden_order_cols()
                    right = table_by_name[op_table(op)]
                    _, right_keys = join_key_pairs(op)
                    right_key_set = set(right_keys)
                    right_cols = [
                        f"r.{_quote(c.name)} AS {_quote(c.name)}"
                        for c in right.columns
                        if c.name not in right_key_set
                    ]
                    select_right = ", " + ", ".join(right_cols) if right_cols else ""
                    join_kind = "LEFT JOIN" if join_how(op) == "left" else "INNER JOIN"
                    query = runtime.assign_source(
                        f"SELECT q.*{select_right} FROM ({query}) q {join_kind} {_quote(right.name)} r "
                        f"ON {_join_condition(op)}"
                    )
                    runtime.state.current_cols.extend(
                        c.name
                        for c in right.columns
                        if c.name not in right_key_set and c.name not in runtime.state.current_cols
                    )
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
                    condition = _semi_anti_join_condition(op, kind)
                    query = runtime.assign_source(f"SELECT * FROM ({query}) q WHERE {condition}")
                elif kind == "drop_nulls":
                    condition = " AND ".join(f"q.{_quote(column)} IS NOT NULL" for column in op["columns"])
                    query = runtime.assign_source(f"SELECT * FROM ({query}) q WHERE {condition}")
                elif kind == "filter":
                    condition = sql_filter_condition(f"q.{_quote(op_column(op))}", _lit(op_value(op)), op_comparator(op))
                    query = runtime.assign_source(
                        f"SELECT * FROM ({query}) q "
                        f"WHERE {condition}"
                    )
                elif kind == "tuple_absence_filter":
                    query = runtime.assign_source(
                        f"SELECT * FROM ({query}) q "
                        f"WHERE {_tuple_absence_safe_condition(op)}"
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
                    materialized_name = f"__datadiff_sortedness_{len(runtime.state.current_cols)}"
                    arrow_table = pa.Table.from_pydict({alias: [ok]})
                    ctx.register_record_batches(materialized_name, [arrow_table.to_batches()])
                    _reset_probe_query(alias, f"SELECT * FROM {_quote(materialized_name)}")
                elif kind in EXTENDED_FALSE_PROBE_KINDS or is_default_false_probe_kind(kind):
                    alias = op_output_alias(op)
                    _reset_probe_query(alias, f"SELECT false AS {_quote(alias)}")
                elif kind == "setop_all_duplicate_probe":
                    alias = op_output_alias(op)
                    _reset_probe_query(alias, _setop_all_duplicate_probe_sql(op))
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
                    if runtime.state.pending_order_mentions(alias):
                        freeze_pending_order()
                    projection = runtime.state.replace_projection_expr(alias, expr_sql, _quote)
                    query = runtime.assign_source(f"SELECT {projection} FROM ({query}) q")
                    runtime.state.replace_visible_column(alias)
                    if runtime.state.pending_order_mentions(alias):
                        runtime.state.clear_pending_order()
                        drop_hidden_order_cols()
                elif kind == "case_when":
                    alias, expr_sql = render_case_when_expr(op, _quote, _lit)
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
                            f"ORDER BY {_order_clause(runtime.state.pending_order)} OFFSET {op_n(op)}"
                        )
                    else:
                        query = runtime.assign_source(f"SELECT * FROM ({query}) q OFFSET {op_n(op)}")
                elif kind == "mutate":
                    out_column, expr_sql = render_mutate_expr(op, DATAFUSION_DIALECT, _quote, _lit)
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
            query = runtime.assign_source(runtime.finalize_source())
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
