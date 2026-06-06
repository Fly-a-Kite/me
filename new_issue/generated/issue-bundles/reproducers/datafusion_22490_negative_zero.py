#!/usr/bin/env python3
"""Standalone reproduction for DataFusion negative-zero truth filtering.

This script does not import DataDiffFuzz. It shows that DataFusion evaluates
(-0.0 >= 0.0) as false, so an IS NOT TRUE filter keeps a row that should be
removed under the common numeric semantics used by Python and DuckDB.
"""

from __future__ import annotations

import datafusion
import duckdb
import pyarrow as pa
from datafusion import SessionContext


def main() -> None:
    expected_cmp = (-0.0 >= 0.0)
    duck_cmp = duckdb.connect(database=":memory:").execute("SELECT (-0.0 >= 0.0)").fetchone()[0]

    ctx = SessionContext()
    batch = pa.RecordBatch.from_pylist(
        [{"id": 1, "y": 0.0}],
        schema=pa.schema([pa.field("id", pa.int64()), pa.field("y", pa.float64(), nullable=True)]),
    )
    ctx.register_record_batches("t0", [[batch]])

    diagnostic = ctx.sql("SELECT id, y * -1 AS m_0, (y * -1) >= 0.0 AS cmp FROM t0").to_pandas()
    filtered = ctx.sql(
        "SELECT id, y * -1 AS m_0 FROM t0 "
        "WHERE NOT (((y * -1) >= 0.0) IS TRUE)"
    ).to_pandas()

    print(f"datafusion={getattr(datafusion, '__version__', 'unknown')}")
    print(f"pyarrow={pa.__version__}")
    print(f"duckdb={duckdb.__version__}")
    print(f"python comparison: {expected_cmp!r}")
    print(f"duckdb comparison: {duck_cmp!r}")
    print("datafusion diagnostic:")
    print(diagnostic)
    print("datafusion filtered rows:")
    print(filtered)

    assert expected_cmp is True and duck_cmp is True
    assert bool(diagnostic.iloc[0]["cmp"]) is True, "DataFusion evaluated -0.0 >= 0.0 as false"
    assert len(filtered) == 0, "DataFusion kept a row that should not pass IS NOT TRUE"


if __name__ == "__main__":
    main()
