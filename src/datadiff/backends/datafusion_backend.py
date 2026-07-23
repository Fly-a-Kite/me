from __future__ import annotations

import math
import os
import time
from typing import Any

from datadiff.backends.base import Backend, BackendResult, NativeRows, PreparedTable, prepare_table
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
from datadiff.operation_semantics import groupby_keys, is_default_false_probe_kind, join_how, op_ascending, op_column, op_columns, op_comparator, op_kind, op_n, op_nulls, op_output_alias, op_table, op_value
from datadiff.physical_plan import collect_physical_plan_bundle
from datadiff.program_state import ProgramState, state_after_operation
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
    logical_type_sql={
        "bool": "BOOLEAN",
        "float": "DOUBLE",
        "int": "BIGINT",
        "str": "VARCHAR",
    },
    string_slice_fn="SUBSTRING",
    string_length_sql=lambda source: f"LENGTH({source})",
    string_contains_sql=lambda source, needle: (
        f"CASE WHEN {source} IS NULL THEN NULL ELSE INSTR({source}, {needle}) > 0 END"
    ),
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


def _datafusion_grouped_null_topk_mismatch(
    ctx: Any,
    source_sql: str,
    op: Any,
) -> bool:
    aggregate = str(op.get("aggregate", "") or "").lower()
    if aggregate not in {"min", "max"}:
        raise ValueError(f"unsupported grouped-null TopK aggregate: {aggregate}")
    direction = str(op.get("direction", "") or "").lower()
    if direction not in {"asc", "desc"}:
        raise ValueError(f"unsupported grouped-null TopK direction: {direction}")
    nulls = str(op.get("nulls", "") or "").lower()
    if nulls not in {"first", "last"}:
        raise ValueError(f"unsupported grouped-null TopK null placement: {nulls}")
    limit = max(1, int(op.get("limit", 1) or 1))
    group_column = str(op.get("group_column", "g") or "g")
    value_column = str(op.get("value_column", "x") or "x")
    aggregate_alias = "__datadiff_grouped_null_aggregate"
    grouped_sql = (
        f"SELECT {_quote(group_column)}, "
        f"{aggregate.upper()}({_quote(value_column)}) AS {_quote(aggregate_alias)} "
        f"FROM ({source_sql}) q GROUP BY {_quote(group_column)}"
    )
    observed_table = ctx.sql(
        f"SELECT {_quote(aggregate_alias)} FROM ({grouped_sql}) q "
        f"ORDER BY {_quote(aggregate_alias)} {direction.upper()} "
        f"NULLS {nulls.upper()} LIMIT {limit}"
    ).to_arrow_table()
    observed_values = [
        row.get(aggregate_alias) for row in observed_table.to_pylist()
    ]
    expected_values = list(op.get("expected_values", []) or [])
    return observed_values != expected_values


def _datafusion_confirmed_root_mismatch(
    ctx: Any,
    source_sql: str,
    op: Any,
) -> bool:
    if str(op.get("target_backend", "") or "") != "datafusion":
        return False
    root_id = str(op.get("root_id", "") or "")
    params = op.get("native_parameters", {})
    if not isinstance(params, dict):
        params = dict(params) if params else {}

    def rows(sql: str) -> list[dict[str, Any]]:
        return ctx.sql(sql).to_arrow_table().to_pylist()

    if root_id == "datafusion-limit-offset-pushdown-001":
        limit_n = max(1, int(params.get("inner_limit", 8) or 8))
        offset_n = max(1, int(params.get("outer_offset", 1) or 1))
        base = (
            "SELECT l.id, COUNT(l.id) AS count_id, "
            "COUNT(DISTINCT r.j) AS nunique_j "
            f"FROM ({source_sql}) l LEFT JOIN {_quote('t1')} r ON l.id = r.id "
            "GROUP BY l.id"
        )
        outer_order = (
            "ORDER BY id DESC NULLS LAST, count_id DESC NULLS LAST, "
            "nunique_j ASC NULLS LAST"
        )
        inner_order = (
            "ORDER BY id DESC NULLS LAST, count_id DESC NULLS LAST, "
            "nunique_j DESC NULLS LAST"
        )
        expected = rows(
            f"SELECT * FROM ({base}) q {outer_order} OFFSET {offset_n}"
        )
        observed = rows(
            f"SELECT * FROM (SELECT * FROM ({base}) q {inner_order} "
            f"LIMIT {limit_n}) q2 {outer_order} OFFSET {offset_n}"
        )
        return observed != expected

    if root_id == "datafusion-negative-zero-comparison-001":
        expression_mode = str(params.get("expression_mode", "") or "")
        expression = "y * -1" if expression_mode == "multiply_neg_one" else "-y"
        comparator = str(params.get("comparator", "") or "")
        operator = ">=" if comparator == "ge_zero" else "="
        observed = rows(
            f"SELECT id, ({expression}) AS m, "
            f"(({expression}) {operator} 0.0) AS cmp "
            f"FROM ({source_sql}) q WHERE y = 0.0 ORDER BY id"
        )
        return not observed or any(row.get("cmp") is not True for row in observed)

    if root_id == "datafusion-distinct-null-topk-001":
        projection_mode = str(params.get("projection_mode", "") or "")
        if projection_mode == "aliased_subquery":
            distinct_source = (
                f"SELECT probe_v AS v FROM (SELECT v AS probe_v FROM ({source_sql}) q) p"
            )
        else:
            distinct_source = f"SELECT v FROM ({source_sql}) q"
        order_mode = str(params.get("order_mode", "") or "")
        direction = "DESC" if order_mode == "desc_nulls_first" else "ASC"
        base = f"SELECT DISTINCT v FROM ({distinct_source}) d"
        full = rows(f"{base} ORDER BY v {direction} NULLS FIRST")
        top = rows(f"{base} ORDER BY v {direction} NULLS FIRST LIMIT 1")
        return top != full[:1]

    if root_id == "datafusion-ordered-limit-idempotence-001":
        limit_n = max(1, int(params.get("limit_n", 5) or 5))
        offset_n = max(1, int(params.get("offset_n", 2) or 2))
        inner_order = "ORDER BY s ASC NULLS FIRST, id ASC NULLS LAST"
        outer_order = "ORDER BY x DESC NULLS FIRST, id ASC NULLS LAST"
        limited = f"SELECT * FROM ({source_sql}) q {inner_order} LIMIT {limit_n}"
        expected_sql = (
            f"SELECT * FROM (SELECT * FROM ({limited}) q {outer_order} "
            f"OFFSET {offset_n}) q {outer_order}"
        )
        duplicate_sql = (
            "SELECT * FROM (SELECT * FROM ("
            f"SELECT * FROM ({limited}) q {inner_order} LIMIT {limit_n}"
            f") q {outer_order} OFFSET {offset_n}) q {outer_order}"
        )
        return rows(duplicate_sql) != rows(expected_sql)

    raise ValueError(f"unsupported DataFusion confirmed root probe: {root_id}")


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
    session_reuse_policy = "reset"
    session_reset_managed_by_backend = True
    plan_collection_support = "logical+physical_inline"

    def __init__(self) -> None:
        self._context = None
        self._registered_tables: set[str] = set()
        self._target_partitions = _datafusion_target_partitions()

    def reset_for_case(self) -> None:
        if self._context is None:
            self._registered_tables.clear()
            return
        for table_name in tuple(self._registered_tables):
            try:
                self._context.deregister_table(table_name)
            except Exception:  # noqa: BLE001
                pass
        self._registered_tables.clear()

    def close(self) -> None:
        self.reset_for_case()
        self._context = None

    def _context_for_case(self, SessionContext, SessionConfig):
        if self._context is None:
            self._context = SessionContext(
                SessionConfig().with_target_partitions(self._target_partitions)
            )
        else:
            self.reset_for_case()
        return self._context

    def _register_record_batches(self, ctx, table_name: str, batches) -> None:
        ctx.register_record_batches(table_name, batches)
        self._registered_tables.add(str(table_name))

    def _to_record_batch(self, table: TableData | PreparedTable, pa):
        prepared = prepare_table(table)
        arrays = []
        fields = []
        for column in prepared.columns:
            typ = _arrow_type(pa, column.type)
            arrays.append(pa.array(prepared.columns_data[column.name], type=typ))
            fields.append(pa.field(column.name, typ, nullable=column.nullable))
        return pa.RecordBatch.from_arrays(arrays, schema=pa.schema(fields))

    def execute_lowered(
        self,
        tables: list[TableData | PreparedTable],
        program: Program,
        timeout_s: float = 5.0,
    ) -> BackendResult:
        start = time.perf_counter()
        physical_plan = None
        try:
            import datafusion
            from datafusion import SessionConfig, SessionContext
            import pyarrow as pa

            ctx = self._context_for_case(SessionContext, SessionConfig)
            table_by_name = {table.name: table for table in tables}
            semantic_state = ProgramState.from_table(tables[0])
            for table in tables:
                self._register_record_batches(
                    ctx,
                    table.name,
                    [[self._to_record_batch(table, pa)]],
                )

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
                    join_kind = "LEFT JOIN" if join_how(op) == "left" else "INNER JOIN"
                    query = runtime.assign_source(
                        f"SELECT q.*{select_right} FROM ({query}) q {join_kind} {_quote(right.name)} r "
                        f"ON {render_join_condition(op, DATAFUSION_DIALECT, _quote, semantic_state.column_types, right_types)}"
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
                        DATAFUSION_DIALECT,
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
                        DATAFUSION_DIALECT,
                    )
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
                    self._register_record_batches(
                        ctx,
                        materialized_name,
                        [arrow_table.to_batches()],
                    )
                    _reset_probe_query(alias, f"SELECT * FROM {_quote(materialized_name)}")
                elif kind == "datafusion_grouped_null_topk_probe":
                    alias = op_output_alias(op)
                    mismatch = _datafusion_grouped_null_topk_mismatch(
                        ctx,
                        query,
                        op,
                    )
                    _reset_probe_query(
                        alias,
                        f"SELECT {'true' if mismatch else 'false'} AS {_quote(alias)}",
                    )
                elif kind == "confirmed_root_witness_probe":
                    alias = op_output_alias(op)
                    mismatch = _datafusion_confirmed_root_mismatch(ctx, query, op)
                    _reset_probe_query(
                        alias,
                        f"SELECT {'true' if mismatch else 'false'} AS {_quote(alias)}",
                    )
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
                    output_type = next_semantic_state.column_types.get(column)
                    if output_type is not None:
                        expr_sql = cast_logical_expression(expr_sql, output_type, DATAFUSION_DIALECT)
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
                        expr_sql = cast_logical_expression(expr_sql, output_type, DATAFUSION_DIALECT)
                    if runtime.state.pending_order_mentions(alias):
                        freeze_pending_order()
                    projection = runtime.state.replace_projection_expr(alias, expr_sql, _quote)
                    query = runtime.assign_source(f"SELECT {projection} FROM ({query}) q")
                    runtime.state.replace_visible_column(alias)
                    if runtime.state.pending_order_mentions(alias):
                        runtime.state.clear_pending_order()
                        drop_hidden_order_cols()
                elif kind == "case_when":
                    alias, expr_sql = render_case_when_expr(op, DATAFUSION_DIALECT, _quote, _lit)
                    output_type = next_semantic_state.column_types.get(alias)
                    if output_type is not None:
                        expr_sql = cast_logical_expression(expr_sql, output_type, DATAFUSION_DIALECT)
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
                    out_column, expr_sql = render_mutate_expr(
                        op,
                        DATAFUSION_DIALECT,
                        _quote,
                        _lit,
                        semantic_state.column_types,
                    )
                    output_type = next_semantic_state.column_types.get(out_column)
                    if output_type is not None:
                        expr_sql = cast_logical_expression(expr_sql, output_type, DATAFUSION_DIALECT)
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
            dataframe = ctx.sql(query)
            if self.physical_plan_collection_enabled:
                physical_plan = collect_physical_plan_bundle(
                    backend=self.name,
                    backend_version=str(getattr(datafusion, "__version__", "")),
                    sources={
                        "logical": lambda: str(dataframe.logical_plan()),
                        "optimized_logical": lambda: str(dataframe.optimized_logical_plan()),
                        "physical": lambda: str(dataframe.execution_plan()),
                    },
                    detail=self.physical_plan_collection_mode,
                )
            out = _arrow_native_rows(dataframe.to_arrow_table())
            return BackendResult(
                self.name,
                "ok",
                data=out,
                duration_ms=(time.perf_counter() - start) * 1000,
                physical_plan=physical_plan,
            )
        except Exception as exc:  # noqa: BLE001
            return BackendResult(
                self.name,
                "error",
                error_type=type(exc).__name__,
                error=str(exc),
                duration_ms=(time.perf_counter() - start) * 1000,
                physical_plan=physical_plan,
            )


def _arrow_type(pa, kind: str):
    if kind == "int":
        return pa.int64()
    if kind == "float":
        return pa.float64()
    if kind == "bool":
        return pa.bool_()
    return pa.string()


def _datafusion_target_partitions() -> int:
    """Use a deterministic single-partition default for bounded fuzz cases."""

    raw = os.environ.get("DATADIFF_DATAFUSION_TARGET_PARTITIONS")
    if raw is None:
        return 1
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


def _arrow_native_rows(table: Any) -> NativeRows:
    columns = [str(column) for column in table.column_names]
    return NativeRows(
        columns=columns,
        row_values=[[row.get(column) for column in columns] for row in table.to_pylist()],
        column_types=[str(field.type) for field in table.schema],
    )
