#!/usr/bin/env python3
"""Standalone reproduction for DuckDB row-value NOT IN with NULL.

This script does not import DataDiffFuzz. It compares a minimal row-value
NOT IN query against SQLite and DuckDB's direct row equality result.
"""

from __future__ import annotations

import sqlite3

import duckdb


DUCKDB_QUERY = """
WITH t0(left_a, left_b) AS (VALUES (2, 2)),
     t1(right_a, right_b) AS (VALUES (NULL, 4))
SELECT left_a, left_b
FROM t0
WHERE (left_a, left_b) NOT IN (SELECT right_a, right_b FROM t1)
"""

SQLITE_QUERY = """
WITH t0(left_a, left_b) AS (VALUES (2, 2)),
     t1(right_a, right_b) AS (VALUES (NULL, 4))
SELECT left_a, left_b
FROM t0
WHERE (left_a, left_b) NOT IN (SELECT right_a, right_b FROM t1)
"""


def main() -> None:
    duck = duckdb.connect(database=":memory:")
    sqlite = sqlite3.connect(":memory:")

    duck_rows = duck.execute(DUCKDB_QUERY).fetchall()
    sqlite_rows = sqlite.execute(SQLITE_QUERY).fetchall()
    duck_row_equal = duck.execute("SELECT (2, 2) = (NULL, 4)").fetchone()[0]
    duck_row_not_in = duck.execute(
        "SELECT (2, 2) NOT IN (SELECT * FROM (VALUES (NULL, 4)) AS t(a, b))"
    ).fetchone()[0]

    print(f"duckdb={duckdb.__version__}")
    print(f"sqlite={sqlite3.sqlite_version}")
    print(f"duckdb filtered rows: {duck_rows}")
    print(f"sqlite filtered rows: {sqlite_rows}")
    print(f"duckdb row equality (2,2) = (NULL,4): {duck_row_equal!r}")
    print(f"duckdb row NOT IN result: {duck_row_not_in!r}")

    assert duck_rows == sqlite_rows == [(2, 2)], (
        "DuckDB filtered out the row even though the unequal second tuple "
        "component makes the row comparison false."
    )


if __name__ == "__main__":
    main()
