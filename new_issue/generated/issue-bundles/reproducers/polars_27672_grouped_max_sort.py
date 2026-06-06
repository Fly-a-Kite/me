#!/usr/bin/env python3
from __future__ import annotations

import polars as pl


def main() -> int:
    df = pl.DataFrame(
        {
            "s": ["a", "b", "b", "b"],
            "z": [1, 2, 0, None],
        },
        schema={"s": pl.Utf8, "z": pl.Int64},
    )

    grouped = (
        df.sort("z", nulls_last=True)
        .group_by("s", maintain_order=True)
        .agg(pl.col("z").max().alias("max_z"))
    )
    observed = grouped.sort("max_z", descending=True, nulls_last=True)
    expected = pl.DataFrame(grouped.to_dicts()).sort("max_z", descending=True, nulls_last=True)

    lazy = (
        df.lazy()
        .sort("z", nulls_last=True)
        .group_by("s")
        .agg(pl.col("z").max().alias("max_z"))
        .sort("max_z", descending=True, nulls_last=True)
        .collect()
    )

    print(f"polars={pl.__version__}")
    print("\ninput")
    print(df)
    print("\ngrouped")
    print(grouped)
    print(f"grouped['max_z'].flags={grouped['max_z'].flags}")
    print(f"grouped['max_z'].arg_sort()={grouped['max_z'].arg_sort().to_list()}")
    print(f"grouped['max_z'].arg_sort(descending=True)={grouped['max_z'].arg_sort(descending=True).to_list()}")
    print("\nobserved eager sort descending")
    print(observed)
    print("\nexpected after materializing rows")
    print(expected)
    print("\nlazy reference")
    print(lazy)

    if observed.to_dicts() == expected.to_dicts():
        print("\nNo mismatch reproduced.")
        return 1

    print("\nMismatch reproduced: eager descending sort returns the smaller max_z first.")
    print(f"observed_top={observed.head(1).to_dicts()}")
    print(f"expected_top={expected.head(1).to_dicts()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
