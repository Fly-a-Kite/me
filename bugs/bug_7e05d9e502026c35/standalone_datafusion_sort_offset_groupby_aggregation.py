#!/usr/bin/env python3
"""Standalone reproduction for DataFusion sort/OFFSET before GROUP BY aggregation.

This script does not import DataDiffFuzz. It orders a left-join result, keeps
only the final row with OFFSET, then groups that single-row input. DataFusion
53.0.0 groups and aggregates a different row than the ordered/OFFSET input
selects.
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
            {"flag": True, "g": None, "id": 1, "s": "sszMgRQI", "x": -85, "y": -0.5},
            {"flag": True, "g": "Vk", "id": 0, "s": "A", "x": None, "y": -1.0},
            {"flag": False, "g": "A", "id": 0, "s": "sutAXn", "x": -2, "y": -0.5},
        ],
        pa.schema(
            [
                pa.field("flag", pa.bool_()),
                pa.field("g", pa.string()),
                pa.field("id", pa.int64()),
                pa.field("s", pa.string()),
                pa.field("x", pa.int64(), nullable=True),
                pa.field("y", pa.float64()),
            ]
        ),
    )
    _register(
        ctx,
        "t1",
        [
            {"id": 0, "j": -2, "tag": "NANtwaaW", "z": None},
            {"id": 0, "j": -1, "tag": None, "z": 0.5},
            {"id": 0, "j": 10, "tag": "zh", "z": 0.5},
            {"id": 1, "j": -10, "tag": None, "z": -0.5},
            {"id": 0, "j": 10, "tag": "VbSl", "z": -1.0},
            {"id": 1, "j": -10, "tag": "GBkrEvBt", "z": 1.0},
            {"id": 0, "j": -1, "tag": "", "z": -0.5},
        ],
        pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field("j", pa.int64()),
                pa.field("tag", pa.string()),
                pa.field("z", pa.float64(), nullable=True),
            ]
        ),
    )

    query = """
    WITH joined AS (
      SELECT
        t0.flag, t0.g, t0.id, t0.s, t0.x, t0.y,
        t1.j, t1.tag, t1.z, t1.z + 2 AS m_0
      FROM t0 LEFT JOIN t1 ON t0.id = t1.id
    ),
    offset_rows AS (
      SELECT * FROM joined
      ORDER BY
        z ASC NULLS LAST, flag ASC NULLS LAST, g ASC NULLS LAST,
        id ASC NULLS LAST, j ASC NULLS LAST, m_0 ASC NULLS LAST,
        s ASC NULLS LAST, tag ASC NULLS LAST, x ASC NULLS LAST,
        y ASC NULLS LAST
      OFFSET 11
    )
    SELECT flag, MIN(m_0) AS min_m_0, SUM(x) AS sum_x, SUM(y) AS sum_y
    FROM offset_rows GROUP BY flag
    """

    print(f"datafusion={getattr(datafusion, '__version__', 'unknown')}")
    print(f"pyarrow={pa.__version__}")
    result = ctx.sql(query).to_pandas()
    print(result)

    assert len(result) == 1
    row = result.iloc[0]
    assert bool(row["flag"]) is True
    assert pd.isna(row["min_m_0"])
    assert pd.isna(row["sum_x"])
    assert float(row["sum_y"]) == -1.0


if __name__ == "__main__":
    main()
