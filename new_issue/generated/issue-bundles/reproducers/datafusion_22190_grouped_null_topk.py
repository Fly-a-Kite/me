#!/usr/bin/env python3
"""Standalone reproduction for DataFusion grouped top-k NULL sort-key loss.

This script does not import DataDiffFuzz. It constructs one grouped aggregate
whose sort key is NULL, then compares the control query with the top-k query.
"""

from __future__ import annotations

import datafusion
import pyarrow as pa
from datafusion import SessionContext


def _register(rows: list[tuple[str, int | None]]) -> SessionContext:
    ctx = SessionContext()
    batch = pa.RecordBatch.from_arrays(
        [
            pa.array([row[0] for row in rows], type=pa.string()),
            pa.array([row[1] for row in rows], type=pa.int64()),
        ],
        schema=pa.schema(
            [
                pa.field("g", pa.string(), nullable=True),
                pa.field("x", pa.int64(), nullable=True),
            ]
        ),
    )
    ctx.register_record_batches("t0", [[batch]])
    return ctx


def _row_count(ctx: SessionContext, sql: str) -> int:
    return sum(batch.num_rows for batch in ctx.sql(sql).collect())


def main() -> None:
    base = "SELECT g, MIN(x) AS min_x FROM t0 GROUP BY g"
    control_query = f"SELECT min_x FROM ({base}) q LIMIT 20"
    failing_query = f"SELECT min_x FROM ({base}) q ORDER BY min_x ASC NULLS LAST LIMIT 20"

    print(f"datafusion={getattr(datafusion, '__version__', 'unknown')}")
    print(f"pyarrow={pa.__version__}")

    ctx = _register([("a", None)])
    control = ctx.sql(control_query).to_pandas()
    print("control:")
    print(control)

    ctx = _register([("a", None)])
    failing = ctx.sql(failing_query).to_pandas()
    print("top-k:")
    print(failing)

    ctx = _register([("a", None)])
    failing_rows = _row_count(ctx, failing_query)
    print(f"top-k record-batch rows={failing_rows}")

    assert len(control) == 1, "control query should return the grouped NULL aggregate"
    assert failing_rows == 1, "DataFusion dropped the group whose aggregate sort key is NULL"


if __name__ == "__main__":
    main()
