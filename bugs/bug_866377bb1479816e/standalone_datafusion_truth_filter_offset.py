#!/usr/bin/env python3
"""Standalone reproduction for DataFusion truth filter plus OFFSET.

This script does not import DataDiffFuzz. It uses the SQL shape emitted by the
DataFusion adapter for a left join, arithmetic mutation, truth-tested filter,
ordered OFFSET, and final sort. DataFusion 53.0.0 returns a different row than
the one selected by the filtered ordered input.
"""

from __future__ import annotations

import datafusion
import pandas as pd
import pyarrow as pa
from datafusion import SessionContext


def _register(ctx: SessionContext, name: str, rows: list[dict], schema: pa.Schema) -> None:
    batch = pa.RecordBatch.from_pylist(rows, schema=schema)
    ctx.register_record_batches(name, [[batch]])


def main() -> None:
    ctx = SessionContext()
    _register(
        ctx,
        "t0",
        [
            {"id": 2, "g": "alpha", "x": 0, "y": 0.5, "flag": None, "s": "zh"},
            {"id": 0, "g": "M", "x": 2, "y": -1.0, "flag": False, "s": "zh"},
        ],
        pa.schema(
            [
                pa.field("id", pa.int64(), nullable=False),
                pa.field("g", pa.string()),
                pa.field("x", pa.int64()),
                pa.field("y", pa.float64()),
                pa.field("flag", pa.bool_()),
                pa.field("s", pa.string()),
            ]
        ),
    )
    _register(
        ctx,
        "t1",
        [
            {"id": 2, "j": -10, "z": 0.0, "tag": None},
            {"id": 2, "j": -2, "z": -1.0, "tag": "beta"},
        ],
        pa.schema(
            [
                pa.field("id", pa.int64(), nullable=False),
                pa.field("j", pa.int64()),
                pa.field("z", pa.float64()),
                pa.field("tag", pa.string()),
            ]
        ),
    )

    query = """
    SELECT
      q."id", q."g", q."x", q."y", q."flag",
      q."s", q."j", q."z", q."tag", q."m_0"
    FROM (
      SELECT * FROM (
        SELECT * FROM (
          SELECT
            q."id", q."g", q."x", q."y", q."flag", q."s",
            q."j", q."z", q."tag", q."z" * 10 AS "m_0"
          FROM (
            SELECT q.*, r."j" AS "j", r."z" AS "z", r."tag" AS "tag"
            FROM (SELECT * FROM "t0") q
            LEFT JOIN "t1" r ON q."id" = r."id"
          ) q
        ) q
        WHERE NOT ((q."m_0" <= 0.5) IS FALSE)
      ) q
      ORDER BY
        "tag" ASC NULLS LAST, "flag" ASC NULLS LAST, "g" ASC NULLS LAST,
        "id" ASC NULLS LAST, "j" ASC NULLS LAST, "m_0" ASC NULLS LAST,
        "s" ASC NULLS LAST, "x" ASC NULLS LAST, "y" ASC NULLS LAST,
        "z" ASC NULLS LAST
      OFFSET 2
    ) q
    ORDER BY "z" ASC NULLS LAST
    """

    print(f"datafusion={getattr(datafusion, '__version__', 'unknown')}")
    print(f"pyarrow={pa.__version__}")
    result = ctx.sql(query).to_pandas()
    print(result)

    assert len(result) == 1
    row = result.iloc[0]
    assert int(row["id"]) == 2
    assert pd.isna(row["flag"])
    assert int(row["j"]) == -10
    assert float(row["z"]) == 0.0
    assert float(row["m_0"]) == 0.0


if __name__ == "__main__":
    main()
