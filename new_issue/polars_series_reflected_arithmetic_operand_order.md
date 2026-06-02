# Issue Draft: Polars `Series` Reflected Arithmetic Uses Wrong Operand Order

## Discovery Record

| Item | Value |
| --- | --- |
| Project family labels observed | `reverse_division_operand_order@polars`, `series_rtruediv_operand_order@polars` |
| First DataDiffFuzz signal | 2026-05-24 02:49:55 CST (`run-20260523T184954-1779562194466079119`, case index `9`) |
| Expanded API audit | 2026-05-26 20:38:32 CST |
| How found | DataDiffFuzz first isolated `Series.__rtruediv__(Series)`. A follow-up API audit checked the same reflected-operator pattern for other non-commutative arithmetic methods and found the same operand-order issue in `__rsub__`, `__rfloordiv__`, and `__rmod__`; `__rpow__(Series)` raises `ColumnNotFoundError`. |
| Status | Submitted upstream as [pola-rs/polars#27752](https://github.com/pola-rs/polars/issues/27752); upstream issue is open and labeled `bug`, `python`, and `P-medium`. |

This draft generalizes `polars_series_rtruediv_operand_order.md`. For paper
counting, these should be treated as one Polars reflected-arithmetic bug
family unless maintainers split them into separate root causes.

## Title

`Series.__rsub__`, `__rtruediv__`, `__rfloordiv__`, and `__rmod__` use `self op other` instead of `other op self`; `__rpow__(Series)` errors

## Environment

```text
polars: 1.41.0
Python: 3.12.3
Platform: Linux
```

`pip index versions polars` previously reported `1.41.0` as both installed and
latest on 2026-05-26.

## Reproducer

```python
import polars as pl

lhs = pl.Series("lhs", [2, 3, 4])
rhs = pl.Series("rhs", [5, 7, 9])

print("lhs.__rsub__(rhs):", lhs.__rsub__(rhs).to_list())
print("rhs - lhs:", (rhs - lhs).to_list())

print("lhs.__rtruediv__(rhs):", lhs.__rtruediv__(rhs).to_list())
print("rhs / lhs:", (rhs / lhs).to_list())

print("lhs.__rfloordiv__(rhs):", lhs.__rfloordiv__(rhs).to_list())
print("rhs // lhs:", (rhs // lhs).to_list())

print("lhs.__rmod__(rhs):", lhs.__rmod__(rhs).to_list())
print("rhs % lhs:", (rhs % lhs).to_list())

try:
    print("lhs.__rpow__(rhs):", lhs.__rpow__(rhs).to_list())
except Exception as exc:
    print("lhs.__rpow__(rhs):")
    print(f"{type(exc).__name__}: {exc}")
print("rhs ** lhs:", (rhs ** lhs).to_list())
```

## Actual Output

```text
lhs.__rsub__(rhs): [-3, -4, -5]
rhs - lhs: [3, 4, 5]

lhs.__rtruediv__(rhs): [0.4, 0.42857142857142855, 0.4444444444444444]
rhs / lhs: [2.5, 2.3333333333333335, 2.25]

lhs.__rfloordiv__(rhs): [0, 0, 0]
rhs // lhs: [2, 2, 2]

lhs.__rmod__(rhs): [2, 3, 4]
rhs % lhs: [1, 1, 1]

lhs.__rpow__(rhs):
ColumnNotFoundError: unable to find column "lhs"; valid columns: ["rhs"]
rhs ** lhs: [25, 343, 6561]
```

The observed values for `__rsub__`, `__rtruediv__`, `__rfloordiv__`, and
`__rmod__` match `lhs op rhs`, not the reflected operation `rhs op lhs`.
Scalar reflected operations behave correctly, so the issue appears specific
to `Series`-vs-`Series` reflected arithmetic.

## Expected Output

```text
lhs.__rsub__(rhs): [3, 4, 5]
lhs.__rtruediv__(rhs): [2.5, 2.3333333333333335, 2.25]
lhs.__rfloordiv__(rhs): [2, 2, 2]
lhs.__rmod__(rhs): [1, 1, 1]
lhs.__rpow__(rhs): [25, 343, 6561]
```

Python's reflected operator protocol means `lhs.__rsub__(rhs)` represents
`rhs - lhs`, and the same applies to other reflected non-commutative
operators. Polars already follows this behavior when the right operand is a
scalar.

## DataDiffFuzz Evidence

- First live signal: `runs/run-20260523T184954-1779562194466079119.jsonl.gz`
- First recorded case index: `9`
- First recorded case id: `case-00009412-bughunt-polars-reverse-division-columns`
- Later artifact: `bugs/bug_2cc0dbdcd7bedbc7`
- Extracted executable reproducer:
  `new_issue/generated/issue-bundles/reproducers/polars_series_reflected_arithmetic_operand_order.py`
- Bundle manifest:
  `new_issue/generated/issue-bundles/manifest.json`

## Duplicate Search

On 2026-05-26, targeted GitHub searches in `pola-rs/polars` for
`__rsub__`, `__rtruediv__`, `__rfloordiv__`, `__rmod__`, `__rpow__`,
`ColumnNotFoundError`, and reflected arithmetic wording did not locate an
obvious existing issue for this exact `Series`-vs-`Series` reflected
arithmetic behavior. Related broad operator-completeness issues exist, but
they do not appear to report this operand-order bug.
