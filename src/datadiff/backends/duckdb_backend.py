from __future__ import annotations

import os
import tempfile
import time

from datadiff.backends.base import Backend, BackendResult
from datadiff.backends.sql_lowering import (
    SqlDialect,
    render_aggregate_sql,
    render_case_when_expr,
    render_coalesce_expr,
    render_fill_null_expr,
    render_groupby_sql,
    render_mutate_expr,
)
from datadiff.backends.sql_runtime import build_relation_step_runtime
from datadiff.csv_roundtrip import (
    csv_long_numeric_roundtrip_mismatch,
    csv_long_numeric_values,
    long_numeric_csv_path,
)
from datadiff.dsl import Program, SortKey, TableData, normalize_sort_keys
from datadiff.filtering import sql_filter_condition
from datadiff.join_keys import join_key_pairs
from datadiff.operation_semantics import (
    aggregate_alias,
    aggregate_specs,
    groupby_keys,
    op_ascending,
    op_branches,
    op_column,
    op_columns,
    op_comparator,
    op_kind,
    op_literal,
    op_n,
    op_nulls,
    op_output_alias,
    op_table,
    op_rows,
    op_value,
    join_how,
)
from datadiff.running import running_sum_partition_columns, running_sum_sort_keys
from datadiff.sortedness import is_sorted_values
from datadiff.windowing import row_number_order_keys, row_number_partition_columns


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
        return "BIGINT"
    if kind == "float":
        return "DOUBLE"
    if kind == "bool":
        return "BOOLEAN"
    return "VARCHAR"


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


DUCKDB_DIALECT = SqlDialect(
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
    basename_sql=lambda source: f"parse_filename({source})",
    split_part_sql=lambda source, sep, index: f"SPLIT_PART({source}, {sep}, {index + 1})",
    division_sql=lambda source, value: f"CAST({source} AS DOUBLE) / {value}",
    reverse_division_sql=lambda numerator, source: f"CAST({numerator} AS DOUBLE) / {source}",
)


def _join_condition(op: dict) -> str:
    left_keys, right_keys = join_key_pairs(op)
    return " AND ".join(f"q.{_quote(left)} = r.{_quote(right)}" for left, right in zip(left_keys, right_keys))


def _semi_anti_join_condition(op: dict, kind: str) -> str:
    left_keys, right_keys = join_key_pairs(op)
    predicates = [
        *(f"r.{_quote(right)} IS NOT NULL" for right in right_keys),
        *(f"q.{_quote(left)} = r.{_quote(right)}" for left, right in zip(left_keys, right_keys)),
    ]
    exists_sql = f"EXISTS (SELECT 1 FROM {_quote(op['table'])} r WHERE {' AND '.join(predicates)})"
    return exists_sql if kind == "semi_join" else f"NOT {exists_sql}"


def _tuple_absence_native_condition(op: dict) -> str:
    left_sql = ", ".join(f"q.{_quote(column)}" for column in op["columns"])
    right_sql = ", ".join(_quote(column) for column in op["right_columns"])
    return f"({left_sql}) NOT IN (SELECT {right_sql} FROM {_quote(op['table'])})"


def _running_sum_projection(cols: list[str], op: dict) -> tuple[str, list[str]]:
    kept_cols = [col for col in cols if col != op["column"]]
    select_parts = [f"q.{_quote(col)}" for col in kept_cols]
    order_sql = _order_clause(normalize_sort_keys({"keys": op["order_by"]}))
    partition_columns = running_sum_partition_columns(op)
    partition_sql = ""
    if partition_columns:
        partition_sql = "PARTITION BY " + ", ".join(f"q.{_quote(column)}" for column in partition_columns) + " "
    expr_sql = (
        f"SUM(CAST(q.{_quote(op['source'])} AS DOUBLE)) OVER ("
        f"{partition_sql}ORDER BY {order_sql} ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
        f") AS {_quote(op['column'])}"
    )
    select_parts.append(expr_sql)
    return ", ".join(select_parts), kept_cols + [op["column"]]


def _row_number_window_sql(op: dict) -> str:
    partition_columns = row_number_partition_columns(op)
    partition_sql = ""
    if partition_columns:
        partition_sql = "PARTITION BY " + ", ".join(f"q.{_quote(column)}" for column in partition_columns) + " "
    return f"{partition_sql}ORDER BY {_order_clause(row_number_order_keys(op))}"


def _row_number_condition_sql(op: dict) -> str:
    comparator = {"==": "=", "<": "<", "<=": "<="}[op_comparator(op, "==")]
    return f"ROW_NUMBER() OVER ({_row_number_window_sql(op)}) {comparator} {int(op_value(op) or 1)}"


def _random_case_probe_sql(op: dict) -> str:
    rows = op_rows(op, 100_000)
    branches = op_branches(op, 3)
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


def _bit_compare_probe_sql(op: dict) -> str:
    return f"SELECT NOT (('0'::bit < '10101010'::bit) IS TRUE) AS {_quote(op['as'])}"


def _round_even_probe_sql(op: dict) -> str:
    return f"SELECT NOT (round_even(2.675::DOUBLE, 2) = 2.67::DOUBLE) AS {_quote(op['as'])}"


def _float_literal_text(op: dict) -> str:
    literal = op_literal(op).strip()
    allowed = set("0123456789+-.eE")
    if not literal or not any(ch.isdigit() for ch in literal) or any(ch not in allowed for ch in literal):
        raise ValueError(f"invalid float literal probe value: {literal!r}")
    float(literal)
    return literal


def _float_literal_precision_probe_sql(op: dict) -> str:
    literal = _float_literal_text(op)
    quoted_literal = literal.replace("'", "''")
    return (
        "SELECT ("
        f"printf('%.17g', {literal}) <> "
        f"printf('%.17g', CAST('{quoted_literal}' AS DOUBLE))"
        f") AS {_quote(op['as'])}"
    )


def _tuple_anti_null_probe_sql(op: dict) -> str:
    return (
        "WITH t_a(i, j) AS (VALUES (1, 1), (2, 2), (3, NULL), (NULL, 4)), "
        "t_b(i, j) AS (VALUES (1, 1), (NULL, 4)), "
        "got(bucket, row_count) AS ("
        "SELECT 'truthy', COUNT(*) FROM t_a WHERE ((i, j) NOT IN (SELECT i, j FROM t_b)) "
        "UNION ALL "
        "SELECT 'falsy', COUNT(*) FROM t_a WHERE NOT (((i, j) NOT IN (SELECT i, j FROM t_b))) "
        "UNION ALL "
        "SELECT 'nullish', COUNT(*) FROM t_a WHERE (((i, j) NOT IN (SELECT i, j FROM t_b))) IS NULL"
        ") "
        "SELECT NOT ("
        "SUM(CASE WHEN bucket = 'truthy' THEN row_count ELSE 0 END) = 1 "
        "AND SUM(CASE WHEN bucket = 'falsy' THEN row_count ELSE 0 END) = 1 "
        "AND SUM(CASE WHEN bucket = 'nullish' THEN row_count ELSE 0 END) = 2"
        f") AS {_quote(op['as'])} "
        "FROM got"
    )


def _setop_all_duplicate_probe_sql(op: dict) -> str:
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
        f") AS {_quote(op['as'])} "
        "FROM observed"
    )


def _duckdb_json_predicate_order_mismatch(con) -> bool:
    con.execute("DROP TABLE IF EXISTS __datadiff_json_predicate_data")
    con.execute(
        """
        CREATE TEMP TABLE __datadiff_json_predicate_data (
            data_id INTEGER NOT NULL,
            data_json JSON
        )
        """
    )
    con.execute(
        """
        INSERT INTO __datadiff_json_predicate_data VALUES
            (1, '{"data": [{"id":0,"type":"type1","val":"305.123"},{"id":1,"type":"type2","val":"39.35"}]}'),
            (2, '{"data": [{"id":0,"type":"type1","val":"223.752"},{"id":1,"type":"type2","val":"160.875"}]}')
        """
    )
    select_sql = (
        "SELECT je.value->>'type' AS ct, CAST(je.value->>'val' AS DOUBLE) AS duration, d.data_id "
        "FROM __datadiff_json_predicate_data d, json_each(d.data_json, 'data') je "
    )
    safe_query = (
        select_sql
        + "WHERE je.value->>'type' = 'type1' "
        + "AND CAST(je.value->>'val' AS DOUBLE) > 300 "
        + "ORDER BY d.data_id"
    )
    reordered_query = (
        select_sql
        + "WHERE CAST(je.value->>'val' AS DOUBLE) > 300 "
        + "AND je.value->>'type' = 'type1' "
        + "ORDER BY d.data_id"
    )
    expected = [("type1", 305.123, 1)]
    try:
        safe_result = con.execute(safe_query).fetchall()
        reordered_result = con.execute(reordered_query).fetchall()
    except Exception:
        return True
    return safe_result != expected or reordered_result != expected


def _duckdb_csv_long_numeric_roundtrip_mismatch(con, op: dict) -> bool:
    expected_values = csv_long_numeric_values(op)
    with long_numeric_csv_path(expected_values) as csv_path:
        path_sql = str(csv_path).replace("'", "''")
        observed_values = [
            row[0]
            for row in con.execute(
                f"SELECT CAST(value AS VARCHAR) FROM read_csv_auto('{path_sql}')"
            ).fetchall()
        ]
    return csv_long_numeric_roundtrip_mismatch(observed_values, expected_values)


def _table_to_dataframe(pd, table: TableData):
    data = {}
    for column in table.columns:
        values = [row.get(column.name) for row in table.rows]
        if column.type == "int":
            data[column.name] = pd.array(values, dtype="Int64")
        elif column.type == "bool":
            data[column.name] = pd.array(values, dtype="boolean")
        elif column.type == "str":
            data[column.name] = pd.array(values, dtype="string")
        else:
            data[column.name] = values
    return pd.DataFrame(data, columns=[c.name for c in table.columns])


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
                df = _table_to_dataframe(pd, table)
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

            def add_step(sql: str) -> str:
                name = f"step_{len(ctes)}"
                ctes.append((name, sql))
                return _quote(name)

            runtime = build_relation_step_runtime(
                relation,
                current_cols,
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

            def materialize_visible_relation():
                body = runtime.materialize_sql()
                if ctes:
                    cte_sql = ", ".join(f"{_quote(name)} AS ({sql})" for name, sql in ctes)
                    body = f"WITH {cte_sql} {body}"
                return con.execute(body).df()

            def select_with_pending_order(cols: list[str]) -> str:
                return runtime.select_with_pending_order(cols)

            def freeze_pending_order() -> None:
                nonlocal relation
                if runtime.freeze_pending_order():
                    relation = runtime.source

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
                    relation = runtime.assign_source(add_step(
                        f"SELECT q.*{select_right} FROM {relation} q {join_kind} {_quote(right.name)} r "
                        f"ON {_join_condition(op)}"
                    ))
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
                    relation = runtime.assign_source(add_step(
                        f"SELECT {visible_projection()} FROM {relation} q "
                        f"UNION ALL SELECT {right_projection} FROM {_quote(op_table(op))}"
                    ))
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
                    relation = runtime.assign_source(add_step(f"SELECT * FROM {relation} q WHERE {_tuple_absence_native_condition(op)}"))
                elif kind == "row_number_filter":
                    relation = runtime.assign_source(add_step(f"SELECT * FROM {relation} q QUALIFY {_row_number_condition_sql(op)}"))
                    runtime.state.pending_order = [
                        *(SortKey(column, True, "last") for column in row_number_partition_columns(op)),
                        *row_number_order_keys(op),
                    ]
                elif kind == "running_sum":
                    drop_hidden_order_cols()
                    order_keys = running_sum_sort_keys(op)
                    projection, runtime.state.current_cols = _running_sum_projection(runtime.state.current_cols, op)
                    relation = runtime.assign_source(add_step(f"SELECT {projection} FROM {relation} q"))
                    runtime.state.visible_cols = [col for col in runtime.state.visible_cols if col != op["column"]] + [op["column"]]
                    runtime.state.pending_order = order_keys
                elif kind == "sortedness_check":
                    materialized = materialize_visible_relation()
                    ok = is_sorted_values(
                        materialized[op["column"]].tolist(),
                        ascending=op_ascending(op),
                        nulls=op_nulls(op),
                    )
                    materialized_name = f"__datadiff_sortedness_{len(ctes)}"
                    con.register(materialized_name, pd.DataFrame({op["as"]: [ok]}))
                    ctes = []
                    relation = runtime.assign_source(_quote(materialized_name))
                    runtime.reset_source(relation, op["as"])
                elif kind == "random_case_probe":
                    ctes = []
                    relation = runtime.assign_source(add_step(_random_case_probe_sql(op)))
                    runtime.reset_source(relation, op["as"])
                elif kind == "group_quantile_probe":
                    ctes = []
                    relation = runtime.assign_source(add_step(f"SELECT FALSE AS {_quote(op['as'])}"))
                    runtime.reset_source(relation, op["as"])
                elif kind == "scalar_subquery_probe":
                    ctes = []
                    relation = runtime.assign_source(add_step(_scalar_subquery_probe_sql(op)))
                    runtime.reset_source(relation, op["as"])
                elif kind == "window_avg_probe":
                    ctes = []
                    relation = runtime.assign_source(add_step(f"SELECT FALSE AS {_quote(op['as'])}"))
                    runtime.reset_source(relation, op["as"])
                elif kind == "struct_distinct_probe":
                    ctes = []
                    relation = runtime.assign_source(add_step(_struct_distinct_probe_sql(op)))
                    runtime.reset_source(relation, op["as"])
                elif kind == "bit_compare_probe":
                    ctes = []
                    relation = runtime.assign_source(add_step(_bit_compare_probe_sql(op)))
                    runtime.reset_source(relation, op["as"])
                elif kind == "round_even_probe":
                    ctes = []
                    relation = runtime.assign_source(add_step(_round_even_probe_sql(op)))
                    runtime.reset_source(relation, op["as"])
                elif kind == "float_literal_precision_probe":
                    ctes = []
                    relation = runtime.assign_source(add_step(_float_literal_precision_probe_sql(op)))
                    runtime.reset_source(relation, op["as"])
                elif kind == "timestamp_precision_filter_probe":
                    ctes = []
                    relation = runtime.assign_source(add_step(f"SELECT FALSE AS {_quote(op['as'])}"))
                    runtime.reset_source(relation, op["as"])
                elif kind == "series_rtruediv_probe":
                    ctes = []
                    relation = runtime.assign_source(add_step(f"SELECT FALSE AS {_quote(op['as'])}"))
                    runtime.reset_source(relation, op["as"])
                elif kind == "uint64_isin_probe":
                    ctes = []
                    relation = runtime.assign_source(add_step(f"SELECT FALSE AS {_quote(op['as'])}"))
                    runtime.reset_source(relation, op["as"])
                elif kind == "tuple_anti_null_probe":
                    ctes = []
                    relation = runtime.assign_source(add_step(_tuple_anti_null_probe_sql(op)))
                    runtime.reset_source(relation, op["as"])
                elif kind == "setop_all_duplicate_probe":
                    ctes = []
                    relation = runtime.assign_source(add_step(_setop_all_duplicate_probe_sql(op)))
                    runtime.reset_source(relation, op["as"])
                elif kind == "json_predicate_order_probe":
                    ctes = []
                    mismatch = _duckdb_json_predicate_order_mismatch(con)
                    materialized_name = f"__datadiff_json_predicate_order_{len(ctes)}"
                    con.register(materialized_name, pd.DataFrame({op["as"]: [mismatch]}))
                    relation = runtime.assign_source(_quote(materialized_name))
                    runtime.reset_source(relation, op["as"])
                elif kind == "sparse_mask_probe":
                    ctes = []
                    relation = runtime.assign_source(add_step(f"SELECT FALSE AS {_quote(op['as'])}"))
                    runtime.reset_source(relation, op["as"])
                elif kind == "float_wrap_probe":
                    ctes = []
                    relation = runtime.assign_source(add_step(f"SELECT FALSE AS {_quote(op['as'])}"))
                    runtime.reset_source(relation, op["as"])
                elif kind == "index_bool_probe":
                    ctes = []
                    relation = runtime.assign_source(add_step(f"SELECT FALSE AS {_quote(op['as'])}"))
                    runtime.reset_source(relation, op["as"])
                elif kind == "empty_literal_groupby_probe":
                    ctes = []
                    relation = runtime.assign_source(add_step(f"SELECT FALSE AS {_quote(op['as'])}"))
                    runtime.reset_source(relation, op["as"])
                elif kind == "arrow_string_eq_sum_probe":
                    ctes = []
                    relation = runtime.assign_source(add_step(f"SELECT FALSE AS {_quote(op['as'])}"))
                    runtime.reset_source(relation, op["as"])
                elif kind == "arrow_timestamp_loc_slice_probe":
                    ctes = []
                    relation = runtime.assign_source(add_step(f"SELECT FALSE AS {_quote(op['as'])}"))
                    runtime.reset_source(relation, op["as"])
                elif kind == "arrow_timestamp_index_attr_probe":
                    ctes = []
                    relation = runtime.assign_source(add_step(f"SELECT FALSE AS {_quote(op['as'])}"))
                    runtime.reset_source(relation, op["as"])
                elif kind == "eval_inplace_alias_probe":
                    ctes = []
                    relation = runtime.assign_source(add_step(f"SELECT FALSE AS {_quote(op['as'])}"))
                    runtime.reset_source(relation, op["as"])
                elif kind == "bool_reduction_skipna_probe":
                    ctes = []
                    relation = runtime.assign_source(add_step(f"SELECT FALSE AS {_quote(op['as'])}"))
                    runtime.reset_source(relation, op["as"])
                elif kind == "dataset_isin_all_match_probe":
                    ctes = []
                    relation = runtime.assign_source(add_step(f"SELECT FALSE AS {_quote(op['as'])}"))
                    runtime.reset_source(relation, op["as"])
                elif kind == "run_end_null_compute_probe":
                    ctes = []
                    relation = runtime.assign_source(add_step(f"SELECT FALSE AS {_quote(op['as'])}"))
                    runtime.reset_source(relation, op["as"])
                elif kind == "large_string_partition_probe":
                    ctes = []
                    relation = runtime.assign_source(add_step(f"SELECT FALSE AS {_quote(op['as'])}"))
                    runtime.reset_source(relation, op["as"])
                elif kind == "hash_pivot_wider_probe":
                    ctes = []
                    relation = runtime.assign_source(add_step(f"SELECT FALSE AS {_quote(op['as'])}"))
                    runtime.reset_source(relation, op["as"])
                elif kind == "list_flatten_parent_indices_probe":
                    ctes = []
                    relation = runtime.assign_source(add_step(f"SELECT FALSE AS {_quote(op['as'])}"))
                    runtime.reset_source(relation, op["as"])
                elif kind == "rolling_mean_by_null_count_probe":
                    ctes = []
                    relation = runtime.assign_source(add_step(f"SELECT FALSE AS {_quote(op['as'])}"))
                    runtime.reset_source(relation, op["as"])
                elif kind == "csv_long_numeric_roundtrip_probe":
                    ctes = []
                    mismatch = _duckdb_csv_long_numeric_roundtrip_mismatch(con, op)
                    materialized_name = f"__datadiff_csv_long_numeric_{len(ctes)}"
                    con.register(materialized_name, pd.DataFrame({op["as"]: [mismatch]}))
                    relation = runtime.assign_source(_quote(materialized_name))
                    runtime.reset_source(relation, op["as"])
                elif kind == "select":
                    cols = op_columns(op)
                    projection = select_with_pending_order(cols)
                    relation = runtime.assign_source(add_step(f"SELECT {projection} FROM {relation} q"))
                elif kind == "distinct":
                    drop_hidden_order_cols()
                    cols = op_columns(op)
                    projection = ", ".join(f"q.{_quote(c)}" for c in cols)
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
                        relation = runtime.assign_source(add_step(
                            f"SELECT * FROM {relation} q "
                            f"ORDER BY {_order_clause(runtime.state.pending_order)} LIMIT {op_n(op)}"
                        ))
                    else:
                        relation = runtime.assign_source(add_step(f"SELECT * FROM {relation} q LIMIT {op_n(op)}"))
                elif kind == "offset":
                    if runtime.state.pending_order is not None:
                        relation = runtime.assign_source(add_step(
                            f"SELECT * FROM {relation} q "
                            f"ORDER BY {_order_clause(runtime.state.pending_order)} OFFSET {op_n(op)}"
                        ))
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
                    relation = runtime.assign_source(add_step(
                        f"SELECT {select_sql} FROM {relation} q "
                        f"GROUP BY {', '.join(_quote(key) for key in keys)}"
                    ))
                    runtime.state.reset_projection(keys + aliases)
                elif kind == "aggregate":
                    drop_hidden_order_cols()
                    agg_sql, aliases = render_aggregate_sql(op, _agg_expr, _quote)
                    relation = runtime.assign_source(add_step(f"SELECT {', '.join(agg_sql)} FROM {relation} q"))
                    runtime.state.reset_projection(aliases)
                else:
                    raise ValueError(kind)
            relation = runtime.assign_source(runtime.finalize_source())
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
