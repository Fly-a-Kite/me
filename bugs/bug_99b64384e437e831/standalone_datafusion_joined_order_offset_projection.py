#!/usr/bin/env python3
"""Standalone reproduction for DataFusion joined ORDER BY/OFFSET projection.

This script does not import DataDiffFuzz. A joined input is ordered, OFFSET
keeps the second row, then a later projection carries hidden order columns for
an outer ORDER BY. DataFusion 53.0.0 returns the skipped row from the nested
ORDER BY/OFFSET subquery.
"""

from __future__ import annotations

import datafusion
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
            {"id": 0, "g": "filfM", "x": None, "y": -0.5, "flag": True, "s": "Iqm"},
            {"id": 0, "g": "a", "x": -1, "y": 1.0, "flag": True, "s": "alpha"},
        ],
        pa.schema(
            [
                pa.field("id", pa.int64(), nullable=False),
                pa.field("g", pa.string()),
                pa.field("x", pa.int64(), nullable=True),
                pa.field("y", pa.float64()),
                pa.field("flag", pa.bool_()),
                pa.field("s", pa.string()),
            ]
        ),
    )
    _register(
        ctx,
        "t1",
        [{"id": 0, "j": 0, "z": -39.802, "tag": "A"}],
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
    SELECT q."g" FROM (
      SELECT
        q."g",
        q."s" AS "__datadiff_order_0_0",
        q."j" AS "__datadiff_order_1_2"
      FROM (
        SELECT * FROM (
          SELECT
            q."id", q."g", q."x", q."y", q."flag", q."s",
            q."j", q."z", q."tag", q."id" + -2 AS "m_0"
          FROM (
            SELECT q.*, r."j" AS "j", r."z" AS "z", r."tag" AS "tag"
            FROM (SELECT * FROM "t0") q
            INNER JOIN "t1" r ON q."id" = r."id"
          ) q
        ) q
        ORDER BY
          "id" ASC NULLS FIRST, "flag" ASC NULLS LAST, "g" ASC NULLS FIRST,
          "j" DESC NULLS FIRST, "m_0" ASC NULLS LAST, "s" ASC NULLS FIRST,
          "tag" ASC NULLS LAST, "x" DESC NULLS FIRST, "y" DESC NULLS FIRST,
          "z" ASC NULLS LAST
        OFFSET 1
      ) q
    ) q
    ORDER BY
      "__datadiff_order_0_0" DESC NULLS FIRST,
      "g" DESC NULLS FIRST,
      "__datadiff_order_1_2" ASC NULLS LAST
    """

    print(f"datafusion={getattr(datafusion, '__version__', 'unknown')}")
    print(f"pyarrow={pa.__version__}")
    result = ctx.sql(query).to_pydict()["g"]
    print(f"observed={result!r}")

    assert result == ["filfM"], "DataFusion returned the row that OFFSET should skip"


if __name__ == "__main__":
    main()
