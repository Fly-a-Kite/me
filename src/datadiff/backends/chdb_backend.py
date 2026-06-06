from __future__ import annotations

import json
import time
from typing import Any

from datadiff.backends.base import Backend, BackendResult, PreparedTable, prepare_table
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
from datadiff.operation_semantics import (
    aggregate_column,
    aggregate_alias,
    aggregate_func,
    aggregate_specs,
    case_else_value,
    case_then_value,
    expr_kind,
    expr_target_type,
    groupby_keys,
    is_default_false_probe_kind,
    op_ascending,
    op_column,
    op_columns,
    op_kind,
    op_n,
    op_nulls,
    op_output_alias,
    op_table,
)


def _quote(name: str) -> str:
    # ClickHouse identifier quoting uses backticks; double quotes also work
    # if `enable_quoted_identifiers` is on (default true). Use backticks for
    # safety across CH versions.
    return "`" + name.replace("`", "``") + "`"


def _lit(value: Any) -> str:
    import math
    if value is None:
        return "NULL"
    if isinstance(value, (list, tuple)):
        return "(" + ", ".join(_lit(item) for item in value) + ")"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return repr(value)
    return "'" + str(value).replace("\\", "\\\\").replace("'", "\\'") + "'"


def _ch_type(kind: str) -> str:
    # All columns wrapped in Nullable since our DSL allows NULLs everywhere
    if kind == "int":
        return "Nullable(Int64)"
    if kind == "float":
        return "Nullable(Float64)"
    if kind == "bool":
        return "Nullable(Bool)"
    return "Nullable(String)"


def _order_clause(sort_keys: list[SortKey]) -> str:
    # ClickHouse supports NULLS FIRST/LAST natively since 18.x
    return ", ".join(
        f"{_quote(key.column)} {'ASC' if key.ascending else 'DESC'} NULLS {key.nulls.upper()}"
        for key in sort_keys
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


CHDB_DIALECT = SqlDialect(
    float_cast_type="Float64",
    int_cast_type="Int64",
    str_cast_type="String",
    string_slice_fn="substring",
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
)


def _join_condition(op: dict[str, Any]) -> str:
    left_keys, right_keys = join_key_pairs(op)
    return " AND ".join(
        f"q.{_quote(left)} = r.{_quote(right)}" for left, right in zip(left_keys, right_keys)
    )


def _semi_anti_join_condition(op: dict[str, Any], kind: str) -> str:
    left_keys, right_keys = join_key_pairs(op)
    predicates = [
        *(f"r.{_quote(right)} IS NOT NULL" for right in right_keys),
        *(
            f"q.{_quote(left)} = r.{_quote(right)}"
            for left, right in zip(left_keys, right_keys)
        ),
    ]
    exists_sql = (
        f"EXISTS (SELECT 1 FROM {_quote(op_table(op))} r WHERE {' AND '.join(predicates)})"
    )
    return exists_sql if kind == "semi_join" else f"NOT {exists_sql}"


def _agg_result_type(column_types: dict[str, str], agg: Any) -> str:
    func = aggregate_func(agg)
    if func in {"count", "nunique"}:
        return "int"
    if func in {"any", "all"}:
        return "bool"
    if func == "mean":
        return "float"
    return column_types.get(aggregate_column(agg), "float")


def _case_when_chdb_type(then_value: Any, else_value: Any) -> str:
    values = [then_value, else_value]
    if all(isinstance(value, bool) for value in values):
        return "bool"
    if all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        return "int"
    if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
        return "float"
    return "str"


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
    result = session.query(query_sql, "JSONCompactColumns")
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

    def run(
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
            column_types = {c.name: c.type for table in prepared_tables for c in table.columns}
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
                    join_kind = "LEFT JOIN" if op["how"] == "left" else "INNER JOIN"
                    query = runtime.assign_source(
                        f"SELECT q.*{select_right} FROM ({query}) q {join_kind} "
                        f"{_quote(right.name)} r ON {_join_condition(op)}"
                    )
                    runtime.state.current_cols.extend(
                        c.name
                        for c in right.columns
                        if c.name not in right_key_set
                        and c.name not in runtime.state.current_cols
                    )
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
                    condition = _semi_anti_join_condition(op, kind)
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
                    condition = sql_filter_condition(
                        _quote(op["column"]), _lit(op["value"]), op["cmp"]
                    )
                    query = runtime.assign_source(
                        f"SELECT * FROM ({query}) q WHERE {condition}"
                    )
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
                    column_types = {
                        column: column_types[column]
                        for column in cols
                        if column in column_types
                    }
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
                    sources = op_columns(op)
                    column_types[alias] = (
                        column_types.get(str(sources[0]), "str") if sources else "str"
                    )
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
                    column_types[alias] = _case_when_chdb_type(
                        case_then_value(op), case_else_value(op)
                    )
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
                        query = runtime.assign_source(
                            f"SELECT * FROM ({query}) q LIMIT {op_n(op)}"
                        )
                elif kind == "offset":
                    if runtime.state.pending_order is not None:
                        query = runtime.assign_source(
                            f"SELECT * FROM ({query}) q "
                            f"ORDER BY {_order_clause(runtime.state.pending_order)} "
                            f"LIMIT 18446744073709551615 OFFSET {op_n(op)}"
                        )
                    else:
                        query = runtime.assign_source(
                            f"SELECT * FROM ({query}) q "
                            f"LIMIT 18446744073709551615 OFFSET {op_n(op)}"
                        )
                elif kind == "mutate":
                    out_column, expr_sql = render_mutate_expr(op, CHDB_DIALECT, _quote, _lit)
                    if runtime.state.pending_order_mentions(out_column):
                        freeze_pending_order()
                    projection = runtime.state.replace_projection_expr(out_column, expr_sql, _quote)
                    query = runtime.assign_source(f"SELECT {projection} FROM ({query}) q")
                    runtime.state.replace_visible_column(out_column)
                    if expr_kind(op) in {
                        "bool_not",
                        "string_contains",
                        "string_starts_with",
                        "string_ends_with",
                    }:
                        column_types[out_column] = "bool"
                    elif expr_kind(op) == "cast":
                        column_types[out_column] = expr_target_type(op) or column_types.get(
                            out_column, "str"
                        )
                    elif expr_kind(op) == "date_part":
                        column_types[out_column] = "int"
                    if runtime.state.pending_order_mentions(out_column):
                        runtime.state.clear_pending_order()
                        drop_hidden_order_cols()
                elif kind == "groupby":
                    drop_hidden_order_cols()
                    keys = groupby_keys(op)
                    select_sql, aliases = render_groupby_sql(op, _agg_expr, _quote, keys)
                    query = runtime.assign_source(
                        f"SELECT {select_sql} FROM ({query}) q "
                        f"GROUP BY {', '.join(_quote(key) for key in keys)}"
                    )
                    for agg in aggregate_specs(op):
                        column_types[aggregate_alias(agg)] = _agg_result_type(column_types, agg)
                    runtime.state.reset_projection(keys + aliases)
                elif kind == "aggregate":
                    drop_hidden_order_cols()
                    agg_sql, aliases = render_aggregate_sql(op, _agg_expr, _quote)
                    query = runtime.assign_source(f"SELECT {', '.join(agg_sql)} FROM ({query}) q")
                    for agg in aggregate_specs(op):
                        column_types[aggregate_alias(agg)] = _agg_result_type(column_types, agg)
                    runtime.state.reset_projection(aliases)
                elif (
                    kind in EXTENDED_FALSE_PROBE_KINDS
                    or is_default_false_probe_kind(kind)
                ):
                    # Same stubbing strategy as SQLite: probes we don't natively
                    # implement deterministically return 0 so the case completes.
                    alias = op_output_alias(op)
                    _reset_probe_query(alias, f"SELECT 0 AS {_quote(alias)}")
                    column_types[alias] = "bool"
                else:
                    return BackendResult(
                        self.name,
                        "missing",
                        error_type="UnsupportedOperation",
                        error=f"chdb backend does not yet support op kind={kind}",
                        duration_ms=(time.perf_counter() - start) * 1000,
                    )

            query = runtime.assign_source(runtime.finalize_source())
            out = _materialize_dataframe(pd, session, query, column_types)
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
