# Issue Draft: Polars Vectorized Division Returns Next-Up Float For `7 / 5`

## Discovery Record

| Item | Value |
| --- | --- |
| Project family label observed | `grouped_topk_null_sort_key@polars,polars_lazy` |
| First recorded live signal | 2026-05-25 22:46:16 CST (`latest_no_datafusion`, case index `120427`) |
| Manual isolation date | 2026-05-26 CST |
| How found | 24h fresh live differential test: pandas, DuckDB, SQLite, and PyArrow agreed on `1.4`, while Polars eager/lazy returned `1.4000000000000001` after `mutate -> groupby(max) -> sort -> limit`; prefix execution showed the first visible divergence after order-sensitive sorting preserved full float precision |
| Status | Submitted upstream as [pola-rs/polars#27753](https://github.com/pola-rs/polars/issues/27753); upstream issue was closed as `invalid`, so this does not count as a confirmed latest-version bug. |

The live root label mentions grouped top-k/null-sort because that was the
workflow where the difference became observable. The reduced native trigger is
more specific: Polars vectorized expression division can round `7 / 5` to the
next representable double.

## Title

Vectorized expression division returns next-up float for `7 / 5`, while scalar/single-row paths return nearest double

## Environment

```text
polars: 1.41.0
Python: 3.12.3
Platform: Linux
```

`1.41.0` was the latest PyPI release when reproduced on 2026-05-26.

## Reproducer

```python
import struct
import polars as pl


def bits(value: float) -> str:
    return hex(struct.unpack(">Q", struct.pack(">d", float(value)))[0])


single = (
    pl.DataFrame({"x": [7]}, schema={"x": pl.Int64})
    .with_columns((pl.col("x") / 5).alias("y"))
    .get_column("y")
    .to_list()
)

vector = (
    pl.DataFrame({"x": [7, 7]}, schema={"x": pl.Int64})
    .with_columns((pl.col("x") / 5).alias("y"))
    .get_column("y")
    .to_list()
)

print("polars:", pl.__version__)
print("single:", [repr(v) for v in single], [bits(v) for v in single])
print("vector:", [repr(v) for v in vector], [bits(v) for v in vector])
```

## Actual Output

```text
polars: 1.41.0
single: ['1.4'] ['0x3ff6666666666666']
vector: ['1.4000000000000001', '1.4000000000000001'] ['0x3ff6666666666667', '0x3ff6666666666667']
```

The same next-up value is also observed for a `Float64` input column:

```python
pl.DataFrame({"x": [7.0, 7.0]}, schema={"x": pl.Float64}).with_columns(
    (pl.col("x") / 5).alias("y")
)
```

## Expected Output

Both paths should produce the same rounded double for the same arithmetic
operation. Python, NumPy, and pandas produce `0x3ff6666666666666` for `7 / 5`:

```text
python 7 / 5 -> 1.4, bits 0x3ff6666666666666
numpy  7 / 5 -> 1.4, bits 0x3ff6666666666666
pandas 7 / 5 -> 1.4, bits 0x3ff6666666666666
```

The next-up value `0x3ff6666666666667` is farther from the exact rational
`7/5` than `0x3ff6666666666666`.

## DataDiffFuzz Evidence

- Live run: `runs/run-20260525T072821-1779694101242817526.jsonl.gz`
- Case index: `120427`
- Case id: `case-01909290-bughunt`
- Original operation sequence:
  `join -> mutate -> mutate -> groupby -> sort -> select -> limit`
- Differential result: pandas, DuckDB, SQLite, and PyArrow returned `1.4`;
  Polars eager and Polars lazy returned `1.4000000000000001`.

## Duplicate Search

On 2026-05-26, targeted searches in `pola-rs/polars` for
`1.4000000000000001`, nullable/vectorized integer division rounding, and
`7 / 5` with null/division terms did not locate an obvious existing issue for
this value mismatch.
