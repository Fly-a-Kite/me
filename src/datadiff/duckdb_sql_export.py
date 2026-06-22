from __future__ import annotations

from datadiff.backends.base import PreparedTable, prepare_table
from datadiff.backends.duckdb_backend import (
    DUCKDB_DIALECT,
    _agg_expr,
    _join_condition,
    _lit,
    _order_clause,
    _quote,
    _row_number_condition_sql,
    _running_sum_projection,
    _semi_anti_join_condition,
    _sql_type,
    _tuple_absence_native_condition,
)
from datadiff.backends.sql_lowering import (
    render_aggregate_sql,
    render_case_when_expr,
    render_coalesce_expr,
    render_fill_null_expr,
    render_groupby_sql,
    render_mutate_expr,
)
from datadiff.backends.sql_runtime import build_relation_step_runtime
from datadiff.dsl import Case, Program, TableData, normalize_sort_keys
from datadiff.filtering import sql_filter_condition
from datadiff.join_keys import join_key_pairs
from datadiff.operation_semantics import (
    groupby_keys,
    join_how,
    op_column,
    op_columns,
    op_comparator,
    op_kind,
    op_n,
    op_table,
    op_value,
)
from datadiff.running import running_sum_sort_keys
from datadiff.windowing import row_number_order_keys, row_number_partition_columns


class UnsupportedDuckDBSqlExport(ValueError):
    pass


SQL_EXPORT_SUPPORTED_OPS = {
    "aggregate",
    "anti_join",
    "case_when",
    "coalesce",
    "distinct",
    "drop_nulls",
    "fill_null",
    "filter",
    "groupby",
    "join",
    "limit",
    "mutate",
    "offset",
    "row_number_filter",
    "running_sum",
    "select",
    "semi_join",
    "sort",
    "tuple_absence_filter",
    "union_all",
}


def render_duckdb_case_sql(case: Case, *, header_lines: list[str] | None = None) -> str:
    lines: list[str] = [
        "-- DataDiffFuzz DuckDB native SQL reproducer",
        f"-- case_id: {case.case_id}",
        f"-- seed: {case.seed}",
    ]
    for line in header_lines or []:
        lines.append(f"-- {line}")
    lines.extend(["", "PRAGMA threads=1;", ""])
    for table in case.tables:
        lines.extend(render_duckdb_table_sql(table))
        lines.append("")
    lines.extend(["-- Final query", render_duckdb_query_sql(case.tables, case.program) + ";", ""])
    return "\n".join(lines)


def render_duckdb_table_sql(table: TableData | PreparedTable) -> list[str]:
    prepared = prepare_table(table)
    column_defs = ", ".join(f"{_quote(column.name)} {_sql_type(column.type)}" for column in prepared.columns)
    lines = [
        f"DROP TABLE IF EXISTS {_quote(prepared.name)};",
        f"CREATE TABLE {_quote(prepared.name)} ({column_defs});",
    ]
    if not prepared.columns or not prepared.rows:
        return lines
    columns = ", ".join(_quote(column.name) for column in prepared.columns)
    values = []
    for row in prepared.rows:
        values.append("(" + ", ".join(_lit(row.get(column.name)) for column in prepared.columns) + ")")
    lines.append(f"INSERT INTO {_quote(prepared.name)} ({columns}) VALUES")
    lines.extend(f"  {value}{',' if index < len(values) - 1 else ';'}" for index, value in enumerate(values))
    return lines


def render_duckdb_query_sql(tables: list[TableData | PreparedTable], program: Program) -> str:
    if not tables:
        raise UnsupportedDuckDBSqlExport("DuckDB SQL export requires at least one table")
    prepared_tables = [prepare_table(table) for table in tables]
    table_by_name = {table.name: table for table in prepared_tables}
    ctes: list[tuple[str, str]] = []
    relation = _quote(prepared_tables[0].name)

    def add_step(sql: str) -> str:
        name = f"step_{len(ctes)}"
        ctes.append((name, sql))
        return _quote(name)

    runtime = build_relation_step_runtime(
        relation,
        [column.name for column in prepared_tables[0].columns],
        quote=_quote,
        order_clause=_order_clause,
        add_step=add_step,
    )

    def visible_projection() -> str:
        return runtime.visible_projection()

    def drop_hidden_order_cols() -> None:
        nonlocal relation
        if runtime.drop_hidden_order_cols():
            relation = runtime.source

    def select_with_pending_order(cols: list[str]) -> str:
        return runtime.select_with_pending_order(cols)

    def freeze_pending_order() -> None:
        nonlocal relation
        if runtime.freeze_pending_order():
            relation = runtime.source

    for op in program.operations:
        kind = op_kind(op)
        if kind not in SQL_EXPORT_SUPPORTED_OPS:
            raise UnsupportedDuckDBSqlExport(f"unsupported DuckDB SQL export operation: {kind}")
        if kind == "join":
            drop_hidden_order_cols()
            right = table_by_name[op_table(op)]
            _, right_keys = join_key_pairs(op)
            right_key_set = set(right_keys)
            right_cols = [
                f"r.{_quote(column.name)} AS {_quote(column.name)}"
                for column in right.columns
                if column.name not in right_key_set
            ]
            select_right = ", " + ", ".join(right_cols) if right_cols else ""
            join_kind = "LEFT JOIN" if join_how(op) == "left" else "INNER JOIN"
            relation = runtime.assign_source(
                add_step(
                    f"SELECT q.*{select_right} FROM {relation} q {join_kind} {_quote(right.name)} r "
                    f"ON {_join_condition(op)}"
                )
            )
            runtime.state.current_cols.extend(
                column.name
                for column in right.columns
                if column.name not in right_key_set and column.name not in runtime.state.current_cols
            )
            runtime.state.visible_cols = list(runtime.state.current_cols)
            runtime.state.pending_order = None
        elif kind == "union_all":
            drop_hidden_order_cols()
            right_projection = ", ".join(_quote(column) for column in runtime.state.visible_cols)
            relation = runtime.assign_source(
                add_step(
                    f"SELECT {visible_projection()} FROM {relation} q "
                    f"UNION ALL SELECT {right_projection} FROM {_quote(op_table(op))}"
                )
            )
            runtime.state.current_cols = list(runtime.state.visible_cols)
            runtime.state.pending_order = None
        elif kind in {"semi_join", "anti_join"}:
            condition = _semi_anti_join_condition(op, kind)
            relation = runtime.assign_source(add_step(f"SELECT * FROM {relation} q WHERE {condition}"))
        elif kind == "drop_nulls":
            condition = " AND ".join(f"q.{_quote(column)} IS NOT NULL" for column in op["columns"])
            relation = runtime.assign_source(add_step(f"SELECT * FROM {relation} q WHERE {condition}"))
        elif kind == "filter":
            condition = sql_filter_condition(f"q.{_quote(op_column(op))}", _lit(op_value(op)), op_comparator(op))
            relation = runtime.assign_source(add_step(f"SELECT * FROM {relation} q WHERE {condition}"))
        elif kind == "tuple_absence_filter":
            relation = runtime.assign_source(
                add_step(f"SELECT * FROM {relation} q WHERE {_tuple_absence_native_condition(op)}")
            )
        elif kind == "row_number_filter":
            relation = runtime.assign_source(add_step(f"SELECT * FROM {relation} q QUALIFY {_row_number_condition_sql(op)}"))
            runtime.state.pending_order = [
                *(_partition_sort_key(column) for column in row_number_partition_columns(op)),
                *row_number_order_keys(op),
            ]
        elif kind == "running_sum":
            drop_hidden_order_cols()
            order_keys = running_sum_sort_keys(op)
            projection, runtime.state.current_cols = _running_sum_projection(runtime.state.current_cols, op)
            relation = runtime.assign_source(add_step(f"SELECT {projection} FROM {relation} q"))
            runtime.state.visible_cols = [
                column for column in runtime.state.visible_cols if column != op["column"]
            ] + [op["column"]]
            runtime.state.pending_order = order_keys
        elif kind == "select":
            cols = op_columns(op)
            projection = select_with_pending_order(cols)
            relation = runtime.assign_source(add_step(f"SELECT {projection} FROM {relation} q"))
        elif kind == "distinct":
            drop_hidden_order_cols()
            cols = op_columns(op)
            projection = ", ".join(f"q.{_quote(column)}" for column in cols)
            relation = runtime.assign_source(add_step(f"SELECT DISTINCT {projection} FROM {relation} q"))
            runtime.state.reset_projection(cols)
        elif kind == "fill_null":
            column, expr_sql = render_fill_null_expr(op, _quote, _lit)
            if runtime.state.pending_order_mentions(column):
                freeze_pending_order()
            projection = runtime.state.replace_projection_expr(column, expr_sql, _quote)
            relation = runtime.assign_source(add_step(f"SELECT {projection} FROM {relation} q"))
            runtime.state.replace_visible_column(column)
            if runtime.state.pending_order_mentions(column):
                runtime.state.clear_pending_order()
                drop_hidden_order_cols()
        elif kind == "coalesce":
            alias, expr_sql = render_coalesce_expr(op, _quote, _lit)
            if runtime.state.pending_order_mentions(alias):
                freeze_pending_order()
            projection = runtime.state.replace_projection_expr(alias, expr_sql, _quote)
            relation = runtime.assign_source(add_step(f"SELECT {projection} FROM {relation} q"))
            runtime.state.replace_visible_column(alias)
            if runtime.state.pending_order_mentions(alias):
                runtime.state.clear_pending_order()
                drop_hidden_order_cols()
        elif kind == "case_when":
            alias, expr_sql = render_case_when_expr(op, _quote, _lit)
            if runtime.state.pending_order_mentions(alias):
                freeze_pending_order()
            projection = runtime.state.replace_projection_expr(alias, expr_sql, _quote)
            relation = runtime.assign_source(add_step(f"SELECT {projection} FROM {relation} q"))
            runtime.state.replace_visible_column(alias)
            if runtime.state.pending_order_mentions(alias):
                runtime.state.clear_pending_order()
                drop_hidden_order_cols()
        elif kind == "sort":
            drop_hidden_order_cols()
            runtime.state.pending_order = normalize_sort_keys(op)
        elif kind == "limit":
            if runtime.state.pending_order is not None:
                relation = runtime.assign_source(
                    add_step(
                        f"SELECT * FROM {relation} q "
                        f"ORDER BY {_order_clause(runtime.state.pending_order)} LIMIT {op_n(op)}"
                    )
                )
            else:
                relation = runtime.assign_source(add_step(f"SELECT * FROM {relation} q LIMIT {op_n(op)}"))
        elif kind == "offset":
            if runtime.state.pending_order is not None:
                relation = runtime.assign_source(
                    add_step(
                        f"SELECT * FROM {relation} q "
                        f"ORDER BY {_order_clause(runtime.state.pending_order)} OFFSET {op_n(op)}"
                    )
                )
            else:
                relation = runtime.assign_source(add_step(f"SELECT * FROM {relation} q OFFSET {op_n(op)}"))
        elif kind == "mutate":
            out_column, expr_sql = render_mutate_expr(op, DUCKDB_DIALECT, _quote, _lit)
            if runtime.state.pending_order_mentions(out_column):
                freeze_pending_order()
            projection = runtime.state.replace_projection_expr(out_column, expr_sql, _quote)
            relation = runtime.assign_source(add_step(f"SELECT {projection} FROM {relation} q"))
            runtime.state.replace_visible_column(out_column)
            if runtime.state.pending_order_mentions(out_column):
                runtime.state.clear_pending_order()
                drop_hidden_order_cols()
        elif kind == "groupby":
            drop_hidden_order_cols()
            keys = groupby_keys(op)
            select_sql, aliases = render_groupby_sql(op, _agg_expr, _quote, keys)
            relation = runtime.assign_source(
                add_step(
                    f"SELECT {select_sql} FROM {relation} q "
                    f"GROUP BY {', '.join(_quote(key) for key in keys)}"
                )
            )
            runtime.state.reset_projection(keys + aliases)
        elif kind == "aggregate":
            drop_hidden_order_cols()
            agg_sql, aliases = render_aggregate_sql(op, _agg_expr, _quote)
            relation = runtime.assign_source(add_step(f"SELECT {', '.join(agg_sql)} FROM {relation} q"))
            runtime.state.reset_projection(aliases)
    relation = runtime.assign_source(runtime.finalize_source())
    if ctes:
        cte_sql = ",\n     ".join(f"{_quote(name)} AS ({sql})" for name, sql in ctes)
        return f"WITH {cte_sql}\nSELECT * FROM {relation}"
    return f"SELECT * FROM {relation}"


def _partition_sort_key(column: str):
    from datadiff.dsl import SortKey

    return SortKey(column, True, "last")
