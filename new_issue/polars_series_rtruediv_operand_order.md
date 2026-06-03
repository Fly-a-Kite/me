# Issue Draft: Polars `Series.__rtruediv__(Series)` Uses The Wrong Operand Order

## Discovery Record

| Item | Value |
| --- | --- |
| Project family labels observed | `reverse_division_operand_order@polars`, `series_rtruediv_operand_order@polars` |
| First recorded signal | 2026-05-24 02:49:55 CST (`run-20260523T184954-1779562194466079119`, case index `9`) |
| Manual reconfirmation date | 2026-05-26 CST |
| How found | DataDiffFuzz generated a reverse-division expression and compared Polars eager with Polars lazy and other backends. Lazy expression API, pandas, DuckDB, SQLite, PyArrow, and DataFusion agree on `numerator / divisor`; Polars eager `Series.__rtruediv__(Series)` returns `divisor / numerator`. |
| Status | Covered by submitted upstream issue [pola-rs/polars#27752](https://github.com/pola-rs/polars/issues/27752); do not submit separately. |

This is separate from `polars_vector_division_rounding.md`. That issue is
about one-ulp rounding in vectorized division. This one is an operand-order
error: the division direction is reversed for a Series right-division dunder
call with another Series.

## Title

`Series.__rtruediv__(Series)` computes `self / other` instead of `other / self`

## Environment

```text
polars: 1.41.0
Python: 3.12.3
Platform: Linux
```

`pip index versions polars` reported `1.41.0` as both installed and latest on
2026-05-26.

## Reproducer

```python
import polars as pl

divisor = pl.Series("divisor", [2, 3, 4])
numerator = pl.Series("numerator", [3, 4, 5])

print("numerator / divisor:", (numerator / divisor).to_list())
print("divisor.__rtruediv__(numerator):", divisor.__rtruediv__(numerator).to_list())
print("divisor.__rtruediv__(1):", divisor.__rtruediv__(1).to_list())
```

## Actual Output

```text
numerator / divisor: [1.5, 1.3333333333333333, 1.25]
divisor.__rtruediv__(numerator): [0.6666666666666666, 0.75, 0.8]
divisor.__rtruediv__(1): [0.5, 0.3333333333333333, 0.25]
```

For scalar right division, `divisor.__rtruediv__(1)` behaves as expected:
`1 / divisor`. For Series right division, however, the result matches
`divisor / numerator`.

## Expected Output

```text
divisor.__rtruediv__(numerator): [1.5, 1.3333333333333333, 1.25]
```

Python's right-division protocol means `self.__rtruediv__(other)` should
represent `other / self` when the reflected operation is used. That is also
how Polars behaves for scalar right division and how the lazy expression
`pl.col("numerator") / pl.col("divisor")` behaves.

## DataDiffFuzz Evidence

- First live run: `runs/run-20260523T184954-1779562194466079119.jsonl.gz`
- First recorded case index: `9`
- First recorded case id: `case-00009412-discovery-polars-reverse-division-columns`
- Later artifact: `bugs/bug_2cc0dbdcd7bedbc7`
- Primary executable reproducer for the merged reflected-arithmetic family:
  `new_issue/generated/issue-bundles/reproducers/polars_series_reflected_arithmetic_operand_order.py`
- Bundle manifest:
  `new_issue/generated/issue-bundles/manifest.json`

Original DataDiffFuzz table:

```text
divisor_value:   [2, 3, 4]
numerator_value: [3, 4, 5]
```

Observed normalized outputs:

| Backend | Result |
| --- | --- |
| Polars eager | `[0.6666666667, 0.75, 0.8]` |
| Polars lazy | `[1.5, 1.25, 1.3333333333]` |
| pandas / DuckDB / SQLite / PyArrow / DataFusion | agree with `numerator / divisor` |

## Duplicate Search

On 2026-05-26, targeted searches in `pola-rs/polars` for
`__rtruediv__`, `rtruediv`, `Series`, and operand-order wording did not locate
an obvious existing issue for this exact Series-vs-Series right-division
behavior. This is a duplicate-search result, not upstream confirmation.
