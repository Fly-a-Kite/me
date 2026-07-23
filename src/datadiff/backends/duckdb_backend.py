from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
from typing import Any

from datadiff.backends.base import Backend, BackendResult, PreparedTable, prepare_table
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
from datadiff.backends.sql_runtime import build_relation_step_runtime
from datadiff.csv_roundtrip import (
    csv_long_numeric_roundtrip_mismatch,
    csv_long_numeric_values,
    long_numeric_csv_path,
)
from datadiff.dsl import Program, SortKey, TableData, normalize_sort_keys
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
from datadiff.physical_plan import collect_physical_plan_bundle
from datadiff.program_state import ProgramState, state_after_operation
from datadiff.running import running_sum_partition_columns, running_sum_sort_keys
from datadiff.sortedness import is_sorted_values
from datadiff.windowing import row_number_order_keys, row_number_partition_columns


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


@dataclass(slots=True)
class _DuckDBNativeRows:
    columns: list[str]
    row_values: list[list[Any]]
    column_types: list[str]

    def rows(self, named: bool = False):
        if named:
            return [dict(zip(self.columns, row)) for row in self.row_values]
        return self.row_values


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
    s = str(value)
    if "\x00" in s:
        parts = []
        for index, part in enumerate(s.split("\x00")):
            if part:
                escaped_part = part.replace("'", "''")
                parts.append(f"'{escaped_part}'")
            if index < s.count("\x00"):
                parts.append("CHR(0)")
        return "(" + " || ".join(parts or ["''"]) + ")"
    s = s.replace("'", "''")
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
    basename_sql=lambda source: f"parse_filename({source})",
    split_part_sql=lambda source, sep, index: f"SPLIT_PART({source}, {sep}, {index + 1})",
    division_sql=lambda source, value: f"CAST({source} AS DOUBLE) / {value}",
    reverse_division_sql=lambda numerator, source: f"CAST({numerator} AS DOUBLE) / {source}",
)


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


_DUCKDB_JOIN_FILTER_NEGATIVE_RECHECKS = 16


def _duckdb_confirmed_root_mismatch(con, op: dict[str, Any]) -> bool:
    if str(op.get("target_backend", "") or "") != "duckdb":
        return False
    root_id = str(op.get("root_id", "") or "")
    if root_id != "duckdb-join-filter-pushdown-limit-001":
        raise ValueError(f"unsupported DuckDB confirmed root probe: {root_id}")
    params = op.get("native_parameters", {})
    if not isinstance(params, dict):
        params = dict(params) if params else {}
    cut_mode = str(params.get("cut_mode", "") or "")
    membership_mode = str(params.get("membership_mode", "") or "")
    if cut_mode == "offset1_desc":
        order_sql = (
            "ord DESC NULLS LAST, flag DESC NULLS LAST, "
            "min_flag DESC NULLS LAST"
        )
        offset_n = 1
    else:
        order_sql = (
            "ord ASC NULLS LAST, flag ASC NULLS LAST, "
            "min_flag ASC NULLS LAST"
        )
        offset_n = 2
    membership_sql = "SEMI JOIN" if membership_mode == "semi_join" else "INNER JOIN"
    cte = f"""
WITH step_0 AS (
  SELECT q.id, q.flag, r.z, r.tag
  FROM {_quote('t0')} q INNER JOIN {_quote('t1')} r ON q.id = r.id
), step_1 AS (
  SELECT q.id, q.flag, q.z, q.tag, q.z * 2 AS m_0 FROM step_0 q
), step_2 AS (
  SELECT * FROM step_1 q WHERE q.tag NOT IN ('beta', '中文', 'alpha')
), step_3 AS (
  SELECT flag, MIN(flag) AS min_flag, MAX(z) AS max_z
  FROM step_2 q GROUP BY flag
), step_4 AS (
  SELECT q.flag, q.min_flag, q.max_z AS ord FROM step_3 q
), step_5 AS (
  SELECT * FROM step_4 q ORDER BY {order_sql} OFFSET {offset_n}
), step_6 AS (
  SELECT q.flag, q.min_flag FROM step_5 q
)
"""
    treatment_sql = (
        cte
        + f"SELECT q.* FROM step_6 q {membership_sql} {_quote('keys')} r "
        "ON q.flag = r.flag_key"
    )
    treatment = con.execute(treatment_sql).fetchall()
    materialized_name = "__datadiff_duckdb_join_filter_materialized"
    con.execute(f"DROP TABLE IF EXISTS {_quote(materialized_name)}")
    con.execute(
        f"CREATE TEMP TABLE {_quote(materialized_name)} AS "
        + cte
        + "SELECT * FROM step_6"
    )
    expected = con.execute(
        f"SELECT q.* FROM {_quote(materialized_name)} q "
        f"{membership_sql} {_quote('keys')} r ON q.flag = r.flag_key"
    ).fetchall()
    if treatment != expected:
        return True
    # DuckDB 1.5.4's affected runtime-filter plan is normally deterministic
    # with one thread, but a heavily loaded full-suite process can rarely
    # produce the correct row for several consecutive executions. Retry only
    # that negative observation; active cells keep the one-query fast path.
    return any(
        con.execute(treatment_sql).fetchall() != expected
        for _ in range(_DUCKDB_JOIN_FILTER_NEGATIVE_RECHECKS)
    )


def _table_to_dataframe(pd, table: TableData | PreparedTable):
    prepared = prepare_table(table)
    data = {}
    for column in prepared.columns:
        values = prepared.columns_data[column.name]
        if column.type == "int":
            data[column.name] = pd.array(values, dtype="Int64")
        elif column.type == "bool":
            data[column.name] = pd.array(values, dtype="boolean")
        elif column.type == "str":
            data[column.name] = pd.array(values, dtype="string")
        else:
            data[column.name] = values
    return pd.DataFrame(data, columns=[c.name for c in prepared.columns])


class DuckDBBackend(Backend):
    name = "duckdb"
    session_reuse_policy = "reset"
    session_reset_managed_by_backend = True
    persistent_storage = False
    plan_collection_support = "logical+physical_inline"

    def __init__(self) -> None:
        self._connection = None
        self._registered_relations: set[str] = set()

    def _configure_connection(self, con) -> None:
        thread_limit = _duckdb_threads()
        if thread_limit is not None:
            con.execute(f"PRAGMA threads={thread_limit}")

    def _connection_for_case(self, duckdb):
        if self._connection is None:
            self._connection = duckdb.connect(database=":memory:")
            self._configure_connection(self._connection)
        else:
            self.reset_for_case()
        return self._connection

    def reset_for_case(self) -> None:
        con = self._connection
        if con is None:
            self._registered_relations.clear()
            return
        for name in tuple(self._registered_relations):
            try:
                con.unregister(name)
            except Exception:  # noqa: BLE001
                pass
        self._registered_relations.clear()
        try:
            table_names = [
                str(row[0])
                for row in con.execute(
                    "SELECT table_name FROM duckdb_tables()"
                ).fetchall()
            ]
        except Exception:  # noqa: BLE001
            table_names = []
        for table_name in table_names:
            try:
                con.execute(f"DROP TABLE IF EXISTS {_quote(table_name)}")
            except Exception:  # noqa: BLE001
                pass

    def close(self) -> None:
        con = self._connection
        self._connection = None
        self._registered_relations.clear()
        if con is not None:
            con.close()

    def execute_lowered(
        self,
        tables: list[TableData | PreparedTable],
        program: Program,
        timeout_s: float = 5.0,
    ) -> BackendResult:
        start = time.perf_counter()
        tempdir = tempfile.TemporaryDirectory() if self.persistent_storage else None
        con = None
        physical_plan = None
        try:
            import duckdb
            import pandas as pd

            if tempdir is not None:
                database = f"{tempdir.name}/datadiff.duckdb"
                con = duckdb.connect(database=database)
                self._configure_connection(con)
            else:
                con = self._connection_for_case(duckdb)
            table_by_name = {table.name: table for table in tables}
            semantic_state = ProgramState.from_table(tables[0])
            current_cols = [c.name for c in tables[0].columns]
            for table in tables:
                df = _table_to_dataframe(pd, table)
                registered_name = f"__datadiff_input_{table.name}"
                con.register(registered_name, df)
                self._registered_relations.add(registered_name)
                try:
                    col_defs = ", ".join(
                        f"{_quote(c.name)} {_sql_type(c.type)}"
                        for c in table.columns
                    )
                    scope = "" if self.persistent_storage else "TEMP "
                    con.execute(
                        f"CREATE {scope}TABLE {_quote(table.name)} ({col_defs})"
                    )
                    if table.columns:
                        cols = ", ".join(_quote(c.name) for c in table.columns)
                        cast_cols = ", ".join(
                            f"CAST({_quote(c.name)} AS {_sql_type(c.type)}) "
                            f"AS {_quote(c.name)}"
                            for c in table.columns
                        )
                        con.execute(
                            f"INSERT INTO {_quote(table.name)} ({cols}) "
                            f"SELECT {cast_cols} FROM {_quote(registered_name)}"
                        )
                finally:
                    try:
                        con.unregister(registered_name)
                    finally:
                        self._registered_relations.discard(registered_name)
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
                    relation = runtime.assign_source(add_step(
                        f"SELECT q.*{select_right} FROM {relation} q {join_kind} {_quote(right.name)} r "
                        f"ON {render_join_condition(op, DUCKDB_DIALECT, _quote, semantic_state.column_types, right_types)}"
                    ))
                    runtime.state.current_cols.extend(projected_right)
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
                    right = table_by_name[op_table(op)]
                    right_types = {column.name: column.type for column in right.columns}
                    condition = render_semi_anti_join_condition(
                        op,
                        kind,
                        DUCKDB_DIALECT,
                        _quote,
                        semantic_state.column_types,
                        right_types,
                    )
                    relation = runtime.assign_source(add_step(f"SELECT * FROM {relation} q WHERE {condition}"))
                elif kind == "drop_nulls":
                    condition = " AND ".join(f"q.{_quote(column)} IS NOT NULL" for column in op["columns"])
                    relation = runtime.assign_source(add_step(f"SELECT * FROM {relation} q WHERE {condition}"))
                elif kind == "filter":
                    condition = render_filter_condition(
                        f"q.{_quote(op_column(op))}",
                        _lit(op_value(op)),
                        op_comparator(op),
                        DUCKDB_DIALECT,
                    )
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
                    self._registered_relations.add(materialized_name)
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
                elif kind == "series_reflected_arithmetic_probe":
                    ctes = []
                    relation = runtime.assign_source(add_step(f"SELECT FALSE AS {_quote(op['as'])}"))
                    runtime.reset_source(relation, op["as"])
                elif kind == "datafusion_grouped_null_topk_probe":
                    ctes = []
                    relation = runtime.assign_source(add_step(f"SELECT FALSE AS {_quote(op['as'])}"))
                    runtime.reset_source(relation, op["as"])
                elif kind == "confirmed_root_witness_probe":
                    ctes = []
                    mismatch = _duckdb_confirmed_root_mismatch(con, op)
                    relation = runtime.assign_source(
                        add_step(
                            f"SELECT {'TRUE' if mismatch else 'FALSE'} "
                            f"AS {_quote(op['as'])}"
                        )
                    )
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
                    self._registered_relations.add(materialized_name)
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
                elif kind == "arrow_bool_groupby_reduction_probe":
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
                    self._registered_relations.add(materialized_name)
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
                    output_type = next_semantic_state.column_types.get(column)
                    if output_type is not None:
                        expr_sql = cast_logical_expression(expr_sql, output_type, DUCKDB_DIALECT)
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
                    output_type = next_semantic_state.column_types.get(alias)
                    if output_type is not None:
                        expr_sql = cast_logical_expression(expr_sql, output_type, DUCKDB_DIALECT)
                    if runtime.state.pending_order_mentions(alias):
                        freeze_pending_order()
                    projection = runtime.state.replace_projection_expr(alias, expr_sql, _quote)
                    relation = runtime.assign_source(add_step(f"SELECT {projection} FROM {relation} q"))
                    runtime.state.replace_visible_column(alias)
                    if runtime.state.pending_order_mentions(alias):
                        runtime.state.clear_pending_order()
                        drop_hidden_order_cols()
                elif kind == "case_when":
                    alias, expr_sql = render_case_when_expr(op, DUCKDB_DIALECT, _quote, _lit)
                    output_type = next_semantic_state.column_types.get(alias)
                    if output_type is not None:
                        expr_sql = cast_logical_expression(expr_sql, output_type, DUCKDB_DIALECT)
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
                    out_column, expr_sql = render_mutate_expr(
                        op,
                        DUCKDB_DIALECT,
                        _quote,
                        _lit,
                        semantic_state.column_types,
                    )
                    output_type = next_semantic_state.column_types.get(out_column)
                    if output_type is not None:
                        expr_sql = cast_logical_expression(expr_sql, output_type, DUCKDB_DIALECT)
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
                semantic_state = next_semantic_state
            relation = runtime.assign_source(runtime.finalize_source())
            if ctes:
                cte_sql = ", ".join(f"{_quote(name)} AS ({sql})" for name, sql in ctes)
                query = f"WITH {cte_sql} SELECT * FROM {relation}"
            else:
                query = f"SELECT * FROM {relation}"
            if self.physical_plan_collection_enabled:
                def duckdb_physical_plan() -> str:
                    rows = con.execute(f"EXPLAIN {query}").fetchall()
                    for plan_kind, plan_text in rows:
                        if str(plan_kind) == "physical_plan":
                            return str(plan_text)
                    return "\n".join(
                        f"{plan_kind}\n{plan_text}"
                        for plan_kind, plan_text in rows
                    )

                physical_plan = collect_physical_plan_bundle(
                    backend=self.name,
                    backend_version=str(getattr(duckdb, "__version__", "")),
                    sources={"physical": duckdb_physical_plan},
                    unsupported={
                        "logical": "DuckDB Python EXPLAIN does not expose a stable logical plan"
                    },
                    detail=self.physical_plan_collection_mode,
                )
            cursor = con.execute(query)
            columns = [str(description[0]) for description in cursor.description]
            column_types = [str(description[1]) for description in cursor.description]
            out = _DuckDBNativeRows(
                columns=columns,
                row_values=[list(row) for row in cursor.fetchall()],
                column_types=column_types,
            )
            if tempdir is not None:
                con.close()
            if tempdir is not None:
                tempdir.cleanup()
            return BackendResult(
                self.name,
                "ok",
                data=out,
                duration_ms=(time.perf_counter()-start)*1000,
                physical_plan=physical_plan,
            )
        except Exception as exc:  # noqa: BLE001
            if con is not None and tempdir is not None:
                try:
                    con.close()
                except Exception:
                    pass
            if tempdir is not None:
                tempdir.cleanup()
            return BackendResult(
                self.name,
                "error",
                error_type=type(exc).__name__,
                error=str(exc),
                duration_ms=(time.perf_counter()-start)*1000,
                physical_plan=physical_plan,
            )


class DuckDBPersistentBackend(DuckDBBackend):
    name = "duckdb_persistent"
    session_reuse_policy = "fresh_only"
    persistent_storage = True


def _duckdb_threads() -> int | None:
    raw = os.environ.get("DATADIFF_DUCKDB_THREADS")
    if raw is None:
        return 1
    try:
        return max(1, int(raw))
    except ValueError:
        return 1
