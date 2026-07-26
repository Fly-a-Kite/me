#!/usr/bin/env python3
"""Version-aware native reproducers for the P3 canonical bug corpus.

This module intentionally imports no DataDiffFuzz code.  Each probe constructs
the smallest backend-native program needed to observe one canonical root and
emits a uniform JSON record whose ``bug_present`` field has the same meaning
for every backend and version.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from typing import Any


Probe = Callable[[], dict[str, Any]]


def _datafusion_rows(frame: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for batch in frame.collect():
        rows.extend(batch.to_pylist())
    return rows


def _register_arrow_rows(context: Any, name: str, rows: list[dict[str, Any]], schema: Any) -> None:
    import pyarrow as pa

    context.register_record_batches(
        name,
        [[pa.RecordBatch.from_pylist(rows, schema=schema)]],
    )


def datafusion_grouped_null_topk() -> dict[str, Any]:
    import datafusion
    import pyarrow as pa
    from datafusion import SessionContext

    context = SessionContext()
    _register_arrow_rows(
        context,
        "t0",
        [{"g": "a", "x": None}],
        pa.schema(
            [
                pa.field("g", pa.string(), nullable=True),
                pa.field("x", pa.int64(), nullable=True),
            ]
        ),
    )
    base = "SELECT g, MIN(x) AS min_x FROM t0 GROUP BY g"
    control = _datafusion_rows(context.sql(f"SELECT min_x FROM ({base}) q LIMIT 20"))
    observed = _datafusion_rows(
        context.sql(
            f"SELECT min_x FROM ({base}) q "
            "ORDER BY min_x ASC NULLS LAST LIMIT 20"
        )
    )
    return {
        "backend": "datafusion",
        "versions": {"datafusion": datafusion.__version__, "pyarrow": pa.__version__},
        "bug_present": observed != control,
        "expected": control,
        "observed": observed,
        "signature": "grouped MIN(NULL) row must survive ORDER BY aggregate + LIMIT",
    }


def datafusion_groupby_limit_offset() -> dict[str, Any]:
    import datafusion
    import pyarrow as pa
    from datafusion import SessionContext

    context = SessionContext()
    _register_arrow_rows(
        context,
        "t0",
        [{"id": 0}, {"id": 1}],
        pa.schema([pa.field("id", pa.int64(), nullable=False)]),
    )
    _register_arrow_rows(
        context,
        "t1",
        [{"id": 1, "j": 1}],
        pa.schema(
            [
                pa.field("id", pa.int64(), nullable=False),
                pa.field("j", pa.int64(), nullable=True),
            ]
        ),
    )
    base = (
        "SELECT t0.id, COUNT(t0.id) AS count_id, COUNT(DISTINCT j) AS nunique_j "
        "FROM t0 LEFT JOIN t1 ON t0.id = t1.id GROUP BY t0.id"
    )
    control_sql = (
        f"SELECT * FROM ({base}) q "
        "ORDER BY id DESC NULLS LAST, count_id DESC NULLS LAST, "
        "nunique_j ASC NULLS LAST OFFSET 1"
    )
    observed_sql = (
        f"SELECT * FROM (SELECT * FROM ({base}) q "
        "ORDER BY id DESC NULLS LAST, count_id DESC NULLS LAST, "
        "nunique_j DESC NULLS LAST LIMIT 8) q2 "
        "ORDER BY id DESC NULLS LAST, count_id DESC NULLS LAST, "
        "nunique_j ASC NULLS LAST OFFSET 1"
    )
    expected = _datafusion_rows(context.sql(control_sql))
    observed = _datafusion_rows(context.sql(observed_sql))
    return {
        "backend": "datafusion",
        "versions": {"datafusion": datafusion.__version__, "pyarrow": pa.__version__},
        "bug_present": observed != expected,
        "expected": expected,
        "observed": observed,
        "signature": "inner ordered LIMIT must not inherit a handled outer OFFSET",
    }


def datafusion_negative_zero_comparison() -> dict[str, Any]:
    import datafusion
    import pyarrow as pa
    from datafusion import SessionContext

    context = SessionContext()
    _register_arrow_rows(
        context,
        "t0",
        [{"id": 1, "y": 0.0}],
        pa.schema(
            [
                pa.field("id", pa.int64(), nullable=False),
                pa.field("y", pa.float64(), nullable=True),
            ]
        ),
    )
    diagnostic = _datafusion_rows(
        context.sql("SELECT id, y * -1 AS m_0, (y * -1) >= 0.0 AS cmp FROM t0")
    )
    filtered = _datafusion_rows(
        context.sql(
            "SELECT id, y * -1 AS m_0 FROM t0 "
            "WHERE NOT (((y * -1) >= 0.0) IS TRUE)"
        )
    )
    comparison = bool(diagnostic[0]["cmp"])
    bug_present = comparison is not True or bool(filtered)
    return {
        "backend": "datafusion",
        "versions": {"datafusion": datafusion.__version__, "pyarrow": pa.__version__},
        "bug_present": bug_present,
        "expected": {"comparison": True, "filtered_rows": []},
        "observed": {"comparison": comparison, "filtered_rows": filtered},
        "signature": "-0.0 >= 0.0 must be true under SQL numeric comparison semantics",
    }


def datafusion_distinct_null_topk() -> dict[str, Any]:
    import datafusion
    import pyarrow as pa
    from datafusion import SessionContext

    context = SessionContext()
    _register_arrow_rows(
        context,
        "t0",
        [{"v": None}, {"v": ""}, {"v": "a"}],
        pa.schema([pa.field("v", pa.string(), nullable=True)]),
    )
    full_rows = _datafusion_rows(
        context.sql("SELECT DISTINCT v FROM t0 ORDER BY v ASC NULLS FIRST")
    )
    top_rows = _datafusion_rows(
        context.sql(
            "SELECT DISTINCT v FROM t0 ORDER BY v ASC NULLS FIRST LIMIT 1"
        )
    )
    expected = full_rows[:1]
    return {
        "backend": "datafusion",
        "versions": {"datafusion": datafusion.__version__, "pyarrow": pa.__version__},
        "bug_present": top_rows != expected,
        "expected": expected,
        "observed": top_rows,
        "signature": "ordered DISTINCT top-1 must equal the first full ordered row",
    }


def datafusion_limit_idempotence() -> dict[str, Any]:
    import datafusion
    import pyarrow as pa
    from datafusion import SessionContext

    rows = [
        {"id": 0, "g": "a", "x": None, "z": 8, "s": "A"},
        {"id": 1, "g": "b", "x": 10, "z": None, "s": "space value"},
        {"id": 2, "g": "c", "x": -10, "z": -3, "s": ""},
        {"id": 3, "g": "b", "x": -2, "z": 0, "s": "space value"},
        {"id": 4, "g": "b", "x": 10, "z": None, "s": "a"},
        {"id": 5, "g": "a", "x": None, "z": 0, "s": ""},
        {"id": 6, "g": None, "x": None, "z": 0, "s": "space value"},
        {"id": 7, "g": "a", "x": 0, "z": 1, "s": "a"},
        {"id": 8, "g": "c", "x": 0, "z": 0, "s": "A"},
        {"id": 9, "g": "c", "x": -10, "z": 8, "s": "A"},
        {"id": 10, "g": "c", "x": -10, "z": -3, "s": "space value"},
        {"id": 11, "g": "b", "x": -10, "z": 8, "s": "A"},
        {"id": 12, "g": "a", "x": -2, "z": -3, "s": ""},
        {"id": 13, "g": "a", "x": 2, "z": 1, "s": "space value"},
    ]
    context = SessionContext()
    _register_arrow_rows(
        context,
        "t0",
        rows,
        pa.schema(
            [
                pa.field("id", pa.int64(), nullable=False),
                pa.field("g", pa.string(), nullable=True),
                pa.field("x", pa.int64(), nullable=True),
                pa.field("z", pa.int64(), nullable=True),
                pa.field("s", pa.string(), nullable=True),
            ]
        ),
    )
    base_sql = """
SELECT * FROM (
  SELECT * FROM (
    SELECT * FROM t0 ORDER BY s ASC NULLS FIRST, id ASC NULLS LAST LIMIT 5
  ) q ORDER BY x DESC NULLS FIRST, id ASC NULLS LAST OFFSET 2
) q ORDER BY x DESC NULLS FIRST, id ASC NULLS LAST
"""
    duplicate_sql = """
SELECT * FROM (
  SELECT * FROM (
    SELECT * FROM (
      SELECT * FROM t0 ORDER BY s ASC NULLS FIRST, id ASC NULLS LAST LIMIT 5
    ) q ORDER BY s ASC NULLS FIRST, id ASC NULLS LAST LIMIT 5
  ) q ORDER BY x DESC NULLS FIRST, id ASC NULLS LAST OFFSET 2
) q ORDER BY x DESC NULLS FIRST, id ASC NULLS LAST
"""
    expected = _datafusion_rows(context.sql(base_sql))
    observed = _datafusion_rows(context.sql(duplicate_sql))
    return {
        "backend": "datafusion",
        "versions": {"datafusion": datafusion.__version__, "pyarrow": pa.__version__},
        "bug_present": observed != expected,
        "expected": expected,
        "observed": observed,
        "signature": "repeating an identical ordered LIMIT must be idempotent",
    }


def polars_grouped_max_sort_metadata() -> dict[str, Any]:
    import polars as pl

    frame = pl.DataFrame(
        {"s": ["a", "b", "b", "b"], "z": [1, 2, 0, None]},
        schema={"s": pl.Utf8, "z": pl.Int64},
    )
    grouped = (
        frame.sort("z", nulls_last=True)
        .group_by("s", maintain_order=True)
        .agg(pl.col("z").max().alias("max_z"))
    )
    observed = grouped.sort("max_z", descending=True, nulls_last=True).to_dicts()
    expected = (
        pl.DataFrame(grouped.to_dicts())
        .sort("max_z", descending=True, nulls_last=True)
        .to_dicts()
    )
    return {
        "backend": "polars",
        "versions": {"polars": pl.__version__},
        "bug_present": observed != expected,
        "expected": expected,
        "observed": observed,
        "details": {"max_z_flags": grouped["max_z"].flags},
        "signature": "grouped max output must not carry stale ascending metadata",
    }


def polars_reflected_arithmetic_operand_order() -> dict[str, Any]:
    import polars as pl

    lhs = pl.Series("lhs", [2, 3, 4])
    rhs = pl.Series("rhs", [5, 7, 9])
    checks: dict[str, dict[str, Any]] = {}
    operations = {
        "rsub": (lambda: lhs.__rsub__(rhs).to_list(), lambda: (rhs - lhs).to_list()),
        "rtruediv": (
            lambda: lhs.__rtruediv__(rhs).to_list(),
            lambda: (rhs / lhs).to_list(),
        ),
        "rfloordiv": (
            lambda: lhs.__rfloordiv__(rhs).to_list(),
            lambda: (rhs // lhs).to_list(),
        ),
        "rmod": (lambda: lhs.__rmod__(rhs).to_list(), lambda: (rhs % lhs).to_list()),
        "rpow": (lambda: lhs.__rpow__(rhs).to_list(), lambda: (rhs**lhs).to_list()),
    }
    bug_present = False
    for name, (reflected, direct) in operations.items():
        expected = direct()
        try:
            observed = reflected()
            error = ""
        except Exception as exc:  # backend exception is part of the observation
            observed = None
            error = f"{type(exc).__name__}: {exc}"
        mismatch = observed != expected
        bug_present = bug_present or mismatch
        checks[name] = {
            "expected": expected,
            "observed": observed,
            "error": error,
            "mismatch": mismatch,
        }
    return {
        "backend": "polars",
        "versions": {"polars": pl.__version__},
        "bug_present": bug_present,
        "expected": "Series.__r*__(other) must equal other <op> Series",
        "observed": checks,
        "signature": "reflected non-commutative Series operators must preserve operand order",
    }


def pyarrow_sliced_bool_groupby_any_all() -> dict[str, Any]:
    import pyarrow as pa

    base = pa.table({"g": [99, 10, 10], "flag": [True, False, None]})
    sliced = base.slice(1)
    rebuilt = pa.table(sliced.to_pydict())

    def aggregate(table: Any, name: str) -> list[dict[str, Any]]:
        return table.group_by("g", use_threads=False).aggregate([("flag", name)]).to_pylist()

    expected = {name: aggregate(rebuilt, name) for name in ("any", "all")}
    observed = {name: aggregate(sliced, name) for name in ("any", "all")}
    return {
        "backend": "pyarrow",
        "versions": {"pyarrow": pa.__version__},
        "bug_present": observed != expected,
        "expected": expected,
        "observed": observed,
        "details": {"flag_offset": sliced["flag"].chunk(0).offset},
        "signature": "sliced Boolean hash_any/hash_all must honor the values bitmap offset",
    }


def duckdb_join_filter_pushdown_limit() -> dict[str, Any]:
    import duckdb

    connection = duckdb.connect(database=":memory:")
    # The affected runtime-filter path is schedule-sensitive with DuckDB's
    # default worker count.  A single execution thread makes the affected
    # 1.5.4 plan deterministic while preserving the same optimizer fault.
    connection.execute("SET threads = 1")
    connection.execute("CREATE TABLE t0(id BIGINT, flag BOOLEAN)")
    connection.execute(
        """
INSERT INTO t0 VALUES
    (1, FALSE), (1, TRUE), (1, FALSE), (0, TRUE), (1, TRUE),
    (0, NULL), (0, TRUE), (2, FALSE), (1, FALSE), (0, TRUE),
    (1, TRUE), (24, FALSE), (1, TRUE), (2, FALSE)
"""
    )
    connection.execute("CREATE TABLE t1(id BIGINT, z DOUBLE, tag VARCHAR)")
    connection.execute(
        "INSERT INTO t1 VALUES (0, 1.0, 'BkcDeJw'), (24, 24.0, 'tag_24')"
    )
    connection.execute("CREATE TABLE keys(flag_key BOOLEAN)")
    connection.execute("INSERT INTO keys VALUES (FALSE)")
    cte = """
WITH step_0 AS (
  SELECT q.id, q.flag, r.z, r.tag FROM t0 q INNER JOIN t1 r ON q.id = r.id
), step_1 AS (
  SELECT q.id, q.flag, q.z, q.tag, q.z * 2 AS m_0 FROM step_0 q
), step_2 AS (
  SELECT * FROM step_1 q WHERE q.tag NOT IN ('beta', '中文', 'alpha')
), step_3 AS (
  SELECT flag, MIN(flag) AS min_flag, MAX(z) AS max_z FROM step_2 q GROUP BY flag
), step_4 AS (
  SELECT q.flag, q.min_flag, q.max_z AS ord FROM step_3 q
), step_5 AS (
  SELECT * FROM step_4 q
  ORDER BY ord ASC NULLS LAST, flag ASC NULLS LAST, min_flag ASC NULLS LAST
  OFFSET 2
), step_6 AS (
  SELECT q.flag, q.min_flag FROM step_5 q
)
"""
    step_rows = connection.execute(cte + "SELECT * FROM step_6").fetchall()
    exists_rows = connection.execute(
        cte
        + """
SELECT q.* FROM step_6 q
WHERE EXISTS (
  SELECT 1 FROM keys r WHERE r.flag_key IS NOT NULL AND q.flag = r.flag_key
)
"""
    ).fetchall()
    join_rows = connection.execute(
        cte + "SELECT q.* FROM step_6 q INNER JOIN keys r ON q.flag = r.flag_key"
    ).fetchall()
    connection.execute("CREATE TEMP TABLE materialized_step_6 AS " + cte + " SELECT * FROM step_6")
    materialized_rows = connection.execute(
        "SELECT q.* FROM materialized_step_6 q INNER JOIN keys r ON q.flag = r.flag_key"
    ).fetchall()
    expected = [list(row) for row in materialized_rows]
    observed = [list(row) for row in join_rows]
    return {
        "backend": "duckdb",
        "versions": {"duckdb": duckdb.__version__},
        "bug_present": observed != expected,
        "expected": expected,
        "observed": observed,
        "details": {
            "threads": 1,
            "step_6": [list(row) for row in step_rows],
            "exists": [list(row) for row in exists_rows],
        },
        "signature": "join filter pushdown must not cross ORDER BY/OFFSET or LIMIT",
    }


def datafusion_order_by_offset_subquery_groupby() -> dict[str, Any]:
    import datafusion
    import pyarrow as pa
    from datafusion import SessionContext

    context = SessionContext()
    _register_arrow_rows(
        context,
        "t0",
        [
            {"grp": "physical_second", "sort_key": 2},
            {"grp": "sorted_first", "sort_key": 1},
        ],
        pa.schema(
            [
                pa.field("grp", pa.string(), nullable=False),
                pa.field("sort_key", pa.int64(), nullable=False),
            ]
        ),
    )
    inner_sql = "SELECT grp, sort_key FROM t0 ORDER BY sort_key ASC NULLS LAST OFFSET 1"
    grouped_sql = (
        "SELECT grp, COUNT(*) AS n FROM ("
        + inner_sql
        + ") q GROUP BY grp ORDER BY grp"
    )
    inner_rows = _datafusion_rows(context.sql(inner_sql))
    observed = _datafusion_rows(context.sql(grouped_sql))
    expected = [{"grp": "physical_second", "n": 1}]
    return {
        "backend": "datafusion",
        "versions": {"datafusion": datafusion.__version__, "pyarrow": pa.__version__},
        "bug_present": observed != expected,
        "expected": expected,
        "observed": observed,
        "details": {"inner_rows": inner_rows},
        "signature": "Sort below an observable OFFSET must survive outer GROUP BY planning",
    }


PROBES: dict[str, Probe] = {
    "datafusion-grouped-null-topk-001": datafusion_grouped_null_topk,
    "datafusion-limit-offset-pushdown-001": datafusion_groupby_limit_offset,
    "datafusion-negative-zero-comparison-001": datafusion_negative_zero_comparison,
    "datafusion-distinct-null-topk-001": datafusion_distinct_null_topk,
    "datafusion-ordered-limit-idempotence-001": datafusion_limit_idempotence,
    "polars-grouped-max-sort-metadata-001": polars_grouped_max_sort_metadata,
    "polars-reflected-arithmetic-operand-order-001": polars_reflected_arithmetic_operand_order,
    "pyarrow-sliced-bool-hash-aggregate-001": pyarrow_sliced_bool_groupby_any_all,
    "duckdb-join-filter-pushdown-limit-001": duckdb_join_filter_pushdown_limit,
    "datafusion-order-by-offset-subquery-groupby-001": datafusion_order_by_offset_subquery_groupby,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-id", choices=sorted(PROBES))
    parser.add_argument("--list", action="store_true", help="list supported canonical root IDs")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list:
        print(json.dumps(sorted(PROBES), indent=2))
        return 0
    if not args.root_id:
        raise SystemExit("--root-id is required unless --list is used")
    try:
        payload = PROBES[args.root_id]()
    except Exception as exc:
        print(
            json.dumps(
                {
                    "schema_version": "canonical-native-reproducer-v1",
                    "root_id": args.root_id,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    payload = {
        "schema_version": "canonical-native-reproducer-v1",
        "root_id": args.root_id,
        "status": "ok",
        **payload,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
