#!/usr/bin/env python3
"""Standalone reproduction for Polars Series.__rtruediv__ operand order.

This script does not import DataDiffFuzz. It compares an eager Series
right-division expression with the equivalent lazy column expression.
"""

from __future__ import annotations

import polars as pl


def main() -> None:
    df = pl.DataFrame({"divisor_value": [4], "numerator_value": [5]})

    eager = df.with_columns(
        df["divisor_value"].__rtruediv__(df["numerator_value"]).alias("ratio_value")
    )
    lazy = df.lazy().with_columns(
        (pl.col("numerator_value") / pl.col("divisor_value")).alias("ratio_value")
    ).collect()

    eager_ratio = eager.get_column("ratio_value").item()
    lazy_ratio = lazy.get_column("ratio_value").item()

    print(f"polars={pl.__version__}")
    print("eager Series.__rtruediv__ result:")
    print(eager)
    print("lazy numerator / divisor result:")
    print(lazy)

    assert eager_ratio == lazy_ratio, (
        "Series.__rtruediv__ returned a different operand order: "
        f"eager={eager_ratio!r}, expected={lazy_ratio!r}"
    )


if __name__ == "__main__":
    main()
