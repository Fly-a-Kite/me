from __future__ import annotations

import json
import struct
import time
from typing import Any

from datadiff.backends.base import Backend, BackendResult, PreparedTable, prepare_table
from datadiff.backends.probe_semantics import EXTENDED_FALSE_PROBE_KINDS
from datadiff.backends.sql_lowering import (
    SqlDialect,
    cast_logical_expression,
    render_aggregate_sql,
    render_case_when_expr,
    render_coalesce_expr,
    render_filter_condition,
    render_fill_null_expr,
    render_join_condition,
    render_join_right_projection,
    render_mutate_expr,
    render_semi_anti_join_condition,
    typed_column_sql,
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
    op_kind,
    op_n,
    op_nulls,
    op_output_alias,
    op_source,
    op_table,
)
from datadiff.program_state import ProgramState, state_after_operation
from datadiff.running import running_sum_partition_columns, running_sum_sort_keys


def _quote(name: str) -> str:
    # ClickHouse identifier quoting uses backticks; double quotes also work
    # if `enable_quoted_identifiers` is on (default true). Use backticks for
    # safety across CH versions.
    return "`" + name.replace("`", "``") + "`"


def _lit(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, (list, tuple)):
        return "(" + ", ".join(_lit(item) for item in value) + ")"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        raw_hex = struct.pack("<d", value).hex()
        return f"reinterpretAsFloat64(unhex('{raw_hex}'))"
    return "'" + str(value).replace("\\", "\\\\").replace("'", "\\'") + "'"


def _ch_type(kind: str) -> str:
    return CHDB_DIALECT.cast_type(kind)


def _order_clause(
    sort_keys: list[SortKey],
    column_types: dict[str, str] | None = None,
) -> str:
    # ClickHouse supports NULLS FIRST/LAST natively since 18.x
    return ", ".join(
        f"{_typed_column_sql('', key.column, (column_types or {}).get(key.column))} "
        f"{'ASC' if key.ascending else 'DESC'} NULLS {key.nulls.upper()}"
        for key in sort_keys
    )


def _typed_column_sql(alias: str, column: str, logical_type: str | None) -> str:
    prefix = f"{alias}." if alias else ""
    source = f"{prefix}{_quote(column)}"
    return (
        typed_column_sql(alias, column, logical_type, CHDB_DIALECT, _quote)
        if logical_type
        else source
    )


def _agg_expr(column: str, func: str) -> str:
    quoted = _quote(column)
    if func == "nunique":
        return f"uniqExact({quoted})"
    if func == "any":
        # OR semantics over a nullable Bool column; NULL-only group → NULL
        return (
            f"CASE WHEN count({quoted}) = 0 THEN NULL "
            f"ELSE max(toUInt8({quoted})) > 0 END"
        )
    if func == "all":
        return (
            f"CASE WHEN count({quoted}) = 0 THEN NULL "
            f"ELSE min(toUInt8({quoted})) > 0 END"
        )
    if func == "mean":
        return f"avg({quoted})"
    if func == "count":
        return f"count({quoted})"
    return f"{func}({quoted})"


def _running_sum_projection(
    columns: list[str],
    op: dict[str, Any],
    column_types: dict[str, str],
) -> tuple[str, list[str]]:
    output_column = op_column(op)
    kept_columns = [column for column in columns if column != output_column]
    select_parts = [f"q.{_quote(column)}" for column in kept_columns]
    partition_columns = running_sum_partition_columns(op)
    partition_sql = ""
    if partition_columns:
        partition_sql = "PARTITION BY " + ", ".join(
            _typed_column_sql("q", column, column_types.get(column))
            for column in partition_columns
        ) + " "
    source_sql = cast_logical_expression(
        f"q.{_quote(op_source(op))}",
        "float",
        CHDB_DIALECT,
    )
    order_sql = _order_clause(running_sum_sort_keys(op), column_types)
    select_parts.append(
        f"sum({source_sql}) OVER ("
        f"{partition_sql}ORDER BY {order_sql} "
        "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
        f") AS {_quote(output_column)}"
    )
    return ", ".join(select_parts), [*kept_columns, output_column]


def _semantic_cast_sql(
    source_sql: str,
    source_type: str | None,
    target_type: str,
) -> str:
    if source_type == "float" and target_type == "str":
        float_sql = cast_logical_expression(source_sql, "float", CHDB_DIALECT)
        text_sql = f"toString({float_sql})"
        python_style_text = (
            f"CASE WHEN {source_sql} IS NULL THEN NULL "
            f"WHEN isFinite({float_sql}) "
            f"AND position({text_sql}, '.') = 0 "
            f"AND position({text_sql}, 'e') = 0 "
            f"AND position({text_sql}, 'E') = 0 "
            f"THEN concat({text_sql}, '.0') ELSE {text_sql} END"
        )
        return cast_logical_expression(python_style_text, "str", CHDB_DIALECT)
    return cast_logical_expression(source_sql, target_type, CHDB_DIALECT)


CHDB_DIALECT = SqlDialect(
    logical_type_sql={
        "bool": "Nullable(Bool)",
        "float": "Nullable(Float64)",
        "int": "Nullable(Int64)",
        "str": "Nullable(String)",
    },
    string_slice_fn="substring",
    string_length_sql=lambda source: f"lengthUTF8({source})",
    string_contains_sql=lambda source, needle: (
        f"CASE WHEN {source} IS NULL THEN NULL ELSE position({source}, {needle}) > 0 END"
    ),
    string_startswith_fn=lambda source, needle: (
        f"CASE WHEN {source} IS NULL THEN NULL ELSE startsWith({source}, {needle}) END"
    ),
    string_endswith_fn=lambda source, needle: (
        f"CASE WHEN {source} IS NULL THEN NULL ELSE endsWith({source}, {needle}) END"
    ),
    date_part_spans={"year": (1, 4), "month": (6, 2), "day": (9, 2)},
    # ClickHouse: basename via `path()` only takes URLs; we use a generic regex
    basename_sql=lambda source: (
        f"replaceRegexpAll({source}, '^.*/', '')"
    ),
    split_part_sql=lambda source, sep, index: (
        f"arrayElement(splitByString({sep}, {source}), {index + 1})"
    ),
    division_sql=lambda source, value: f"toFloat64({source}) / ({value})",
    reverse_division_sql=lambda numerator, source: f"toFloat64({numerator}) / ({source})",
    semantic_cast_sql=_semantic_cast_sql,
)


def _open_session() -> Any:
    try:
        import chdb.session as chdb_session  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "chdb backend requested but the `chdb` package is not installed; "
            "install with `pip install chdb` to enable it"
        ) from exc
    return chdb_session.Session()


def _execute_query_json(session: Any, sql: str) -> tuple[list[str], list[list[Any]]]:
    """Run a SELECT and return (column_names, rows)."""
    result = session.query(sql, "JSONCompactColumns")
    payload = str(result)
    if not payload.strip():
        return [], []
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"chdb returned non-JSON payload: {payload[:200]}") from exc
    # JSONCompactColumns is list[list[Any]] in column-major order; metadata via separate format
    if not isinstance(parsed, list):
        raise RuntimeError(f"unexpected chdb JSONCompactColumns shape: {type(parsed).__name__}")
    if not parsed:
        return [], []
    # We separately query column names via DESCRIBE
    return [], parsed  # column names filled by caller via DESCRIBE


def _describe_columns(session: Any, query_sql: str) -> list[str]:
    """Return ordered column names for a SELECT query via DESCRIBE."""
    described = session.query(f"DESCRIBE ({query_sql})", "JSONCompact")
    text = str(described)
    if not text.strip():
        return []
    parsed = json.loads(text)
    rows = parsed.get("data", []) if isinstance(parsed, dict) else parsed
    return [str(row[0]) for row in rows]


def _materialize_dataframe(pd: Any, session: Any, query_sql: str, column_types: dict[str, str]):
    """Run query, return a pandas DataFrame with dtype=object to preserve int64 ranges."""
    columns = _describe_columns(session, query_sql)
    # Use JSONCompactColumns for column-major fetch (more compact than row-major)
    result = session.query(
        f"{query_sql} SETTINGS output_format_json_quote_64bit_floats = 1",
        "JSONCompactColumns",
    )
    text = str(result)
    if not text.strip():
        return pd.DataFrame({col: [] for col in columns}, dtype=object)
    parsed = json.loads(text)
    if not isinstance(parsed, list) or (parsed and not isinstance(parsed[0], list)):
        raise RuntimeError(f"unexpected JSONCompactColumns shape: {type(parsed).__name__}")
    column_data: dict[str, list[Any]] = {}
    for idx, col in enumerate(columns):
        values = parsed[idx] if idx < len(parsed) else []
        kind = column_types.get(col, "str")
        column_data[col] = [_coerce_value(value, kind) for value in values]
    return pd.DataFrame(column_data, dtype=object)


def _coerce_value(value: Any, kind: str) -> Any:
    if value is None:
        return None
    if kind == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1"}:
                return True
            if lowered in {"false", "0"}:
                return False
            return None
        return None
    if kind == "int":
        try:
            return int(value)
        except (TypeError, ValueError):
            return value
    if kind == "float":
        try:
            return float(value)
        except (TypeError, ValueError):
            return value
    return value


class ChDBBackend(Backend):
    """ClickHouse-local (chDB) backend adapter.

    Uses chdb's stateful Session API. ClickHouse SQL dialect is broadly
    compatible with DuckDB/SQLite for the operators emitted by our IR; for
    probe operators that map cleanly into ANSI SQL we render them, otherwise
    we emit a deterministic stub (matching SQLite's strategy) so the case
    completes rather than failing the entire batch.
    """

    name = "chdb"

    def execute_lowered(
        self,
        tables: list[TableData | PreparedTable],
        program: Program,
        timeout_s: float = 5.0,
    ) -> BackendResult:
        start = time.perf_counter()
        session: Any = None
        try:
            import pandas as pd

            prepared_tables = [prepare_table(table) for table in tables]
            table_by_name = {table.name: table for table in prepared_tables}
            semantic_state = ProgramState.from_table(prepared_tables[0])
            current_cols = [c.name for c in prepared_tables[0].columns]
            session = _open_session()

            for table in prepared_tables:
                col_defs = ", ".join(
                    f"{_quote(c.name)} {_ch_type(c.type)}" for c in table.columns
                )
                # ENGINE = Memory: in-process, no disk IO
                session.query(
                    f"CREATE OR REPLACE TABLE {_quote(table.name)} ({col_defs}) ENGINE = Memory"
                )
                if table.row_tuples:
                    values_sql = ", ".join(
                        "(" + ", ".join(_lit(v) for v in row) + ")"
                        for row in table.row_tuples
                    )
                    cols_sql = ", ".join(_quote(c) for c in table.column_names)
                    session.query(
                        f"INSERT INTO {_quote(table.name)} ({cols_sql}) VALUES {values_sql}"
                    )

            query = f"SELECT * FROM {_quote(prepared_tables[0].name)}"
            runtime = build_subquery_runtime(
                query,
                current_cols,
                quote=_quote,
                order_clause=lambda keys: _order_clause(keys, semantic_state.column_types),
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
                        f"SELECT q.*{select_right} FROM ({query}) q {join_kind} "
                        f"{_quote(right.name)} r ON "
                        f"{render_join_condition(op, CHDB_DIALECT, _quote, semantic_state.column_types, right_types)}"
                    )
                    runtime.state.current_cols.extend(projected_right)
                    runtime.state.visible_cols = list(runtime.state.current_cols)
                    runtime.state.pending_order = None
                elif kind == "union_all":
                    drop_hidden_order_cols()
                    right_projection = ", ".join(
                        _quote(col) for col in runtime.state.visible_cols
                    )
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
                        CHDB_DIALECT,
                        _quote,
                        semantic_state.column_types,
                        right_types,
                    )
                    query = runtime.assign_source(
                        f"SELECT * FROM ({query}) q WHERE {condition}"
                    )
                elif kind == "drop_nulls":
                    condition = " AND ".join(
                        f"q.{_quote(column)} IS NOT NULL" for column in op["columns"]
                    )
                    query = runtime.assign_source(
                        f"SELECT * FROM ({query}) q WHERE {condition}"
                    )
                elif kind == "filter":
                    condition = render_filter_condition(
                        f"q.{_quote(op_column(op))}",
                        _lit(op.get("value")),
                        op.get("cmp"),
                        CHDB_DIALECT,
                    )
                    query = runtime.assign_source(
                        f"SELECT * FROM ({query}) q WHERE {condition}"
                    )
                elif kind == "running_sum":
                    drop_hidden_order_cols()
                    order_keys = running_sum_sort_keys(op)
                    projection, runtime.state.current_cols = _running_sum_projection(
                        runtime.state.current_cols,
                        op,
                        semantic_state.column_types,
                    )
                    query = runtime.assign_source(
                        f"SELECT {projection} FROM ({query}) q"
                    )
                    output_column = op_column(op)
                    runtime.state.visible_cols = [
                        column
                        for column in runtime.state.visible_cols
                        if column != output_column
                    ] + [output_column]
                    runtime.state.pending_order = order_keys
                elif kind == "select":
                    cols = op_columns(op)
                    projection = select_with_pending_order(cols)
                    query = runtime.assign_source(f"SELECT {projection} FROM ({query}) q")
                elif kind == "distinct":
                    drop_hidden_order_cols()
                    cols = op_columns(op)
                    projection = ", ".join(f"q.{_quote(c)}" for c in cols)
                    query = runtime.assign_source(
                        f"SELECT DISTINCT {projection} FROM ({query}) q"
                    )
                    runtime.state.reset_projection(cols)
                elif kind == "fill_null":
                    column, expr_sql = render_fill_null_expr(op, _quote, _lit)
                    output_type = next_semantic_state.column_types.get(column)
                    if output_type is not None:
                        expr_sql = cast_logical_expression(expr_sql, output_type, CHDB_DIALECT)
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
                        expr_sql = cast_logical_expression(expr_sql, output_type, CHDB_DIALECT)
                    if runtime.state.pending_order_mentions(alias):
                        freeze_pending_order()
                    projection = runtime.state.replace_projection_expr(alias, expr_sql, _quote)
                    query = runtime.assign_source(f"SELECT {projection} FROM ({query}) q")
                    runtime.state.replace_visible_column(alias)
                    if runtime.state.pending_order_mentions(alias):
                        runtime.state.clear_pending_order()
                        drop_hidden_order_cols()
                elif kind == "case_when":
                    alias, expr_sql = render_case_when_expr(op, CHDB_DIALECT, _quote, _lit)
                    output_type = next_semantic_state.column_types.get(alias)
                    if output_type is not None:
                        expr_sql = cast_logical_expression(expr_sql, output_type, CHDB_DIALECT)
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
                            f"ORDER BY {_order_clause(runtime.state.pending_order, semantic_state.column_types)} LIMIT {op_n(op)}"
                        )
                    else:
                        query = runtime.assign_source(
                            f"SELECT * FROM ({query}) q LIMIT {op_n(op)}"
                        )
                elif kind == "offset":
                    if runtime.state.pending_order is not None:
                        query = runtime.assign_source(
                            f"SELECT * FROM ({query}) q "
                            f"ORDER BY {_order_clause(runtime.state.pending_order, semantic_state.column_types)} "
                            f"LIMIT 18446744073709551615 OFFSET {op_n(op)}"
                        )
                    else:
                        query = runtime.assign_source(
                            f"SELECT * FROM ({query}) q "
                            f"LIMIT 18446744073709551615 OFFSET {op_n(op)}"
                        )
                elif kind == "mutate":
                    out_column, expr_sql = render_mutate_expr(
                        op,
                        CHDB_DIALECT,
                        _quote,
                        _lit,
                        semantic_state.column_types,
                    )
                    output_type = next_semantic_state.column_types.get(out_column)
                    if output_type is not None:
                        expr_sql = cast_logical_expression(expr_sql, output_type, CHDB_DIALECT)
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
                    agg_sql, aliases = render_aggregate_sql(op, _agg_expr, _quote)
                    key_sql = [
                        f"{_typed_column_sql('q', key, semantic_state.column_types.get(key))} "
                        f"AS {_quote(key)}"
                        for key in keys
                    ]
                    select_sql = ", ".join([*key_sql, *agg_sql])
                    query = runtime.assign_source(
                        f"SELECT {select_sql} FROM ({query}) q "
                        f"GROUP BY {', '.join(_quote(key) for key in keys)}"
                    )
                    runtime.state.reset_projection(keys + aliases)
                elif kind == "aggregate":
                    drop_hidden_order_cols()
                    agg_sql, aliases = render_aggregate_sql(op, _agg_expr, _quote)
                    query = runtime.assign_source(f"SELECT {', '.join(agg_sql)} FROM ({query}) q")
                    runtime.state.reset_projection(aliases)
                elif (
                    kind in EXTENDED_FALSE_PROBE_KINDS
                    or is_default_false_probe_kind(kind)
                ):
                    # Same stubbing strategy as SQLite: probes we don't natively
                    # implement deterministically return 0 so the case completes.
                    alias = op_output_alias(op)
                    _reset_probe_query(alias, f"SELECT 0 AS {_quote(alias)}")
                else:
                    return BackendResult(
                        self.name,
                        "missing",
                        error_type="UnsupportedOperation",
                        error=f"chdb backend does not yet support op kind={kind}",
                        duration_ms=(time.perf_counter() - start) * 1000,
                    )
                semantic_state = next_semantic_state

            query = runtime.assign_source(runtime.finalize_source())
            out = _materialize_dataframe(pd, session, query, semantic_state.column_types)
            return BackendResult(
                self.name,
                "ok",
                data=out,
                duration_ms=(time.perf_counter() - start) * 1000,
            )
        except Exception as exc:  # noqa: BLE001
            return BackendResult(
                self.name,
                "error",
                error_type=type(exc).__name__,
                error=str(exc),
                duration_ms=(time.perf_counter() - start) * 1000,
            )
        finally:
            if session is not None:
                try:
                    session.close()
                except Exception:  # noqa: BLE001
                    pass
