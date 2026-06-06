#!/usr/bin/env python3
"""Standalone reproduction for DataFusion groupby ORDER/LIMIT/OFFSET row loss.

This script does not import DataDiffFuzz. It builds a two-row grouped result,
applies an inner ordered LIMIT followed by an outer ORDER BY/OFFSET, and
compares the expected second row with DataFusion's observed empty result.
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
    _register(ctx, "t0", [{"id": 0}, {"id": 1}], pa.schema([pa.field("id", pa.int64())]))
    _register(
        ctx,
        "t1",
        [{"id": 1, "j": 1}],
        pa.schema([pa.field("id", pa.int64()), pa.field("j", pa.int64(), nullable=True)]),
    )

    base = (
        "SELECT t0.id, COUNT(t0.id) AS count_id, COUNT(DISTINCT j) AS nunique_j "
        "FROM t0 LEFT JOIN t1 ON t0.id = t1.id GROUP BY t0.id"
    )
    control_query = (
        f"SELECT * FROM ({base}) q "
        "ORDER BY id DESC NULLS LAST, count_id DESC NULLS LAST, nunique_j ASC NULLS LAST OFFSET 1"
    )
    failing_query = (
        f"SELECT * FROM (SELECT * FROM ({base}) q "
        "ORDER BY id DESC NULLS LAST, count_id DESC NULLS LAST, nunique_j DESC NULLS LAST LIMIT 8) q2 "
        "ORDER BY id DESC NULLS LAST, count_id DESC NULLS LAST, nunique_j ASC NULLS LAST OFFSET 1"
    )

    print(f"datafusion={getattr(datafusion, '__version__', 'unknown')}")
    print(f"pyarrow={pa.__version__}")
    control = ctx.sql(control_query).to_pandas()
    failing = ctx.sql(failing_query).to_pandas()
    print("control:")
    print(control)
    print("with inner limit:")
    print(failing)

    assert len(control) == 1 and control.iloc[0]["id"] == 0
    assert len(failing) == 1 and failing.iloc[0]["id"] == 0, (
        "DataFusion dropped the second grouped row after inner ORDER BY/LIMIT "
        "and outer ORDER BY/OFFSET."
    )


if __name__ == "__main__":
    main()
