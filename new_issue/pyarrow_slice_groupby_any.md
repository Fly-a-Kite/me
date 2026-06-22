# Issue Draft: `Table.group_by(...).aggregate([("flag", "any")])` Returns Wrong Result For Sliced Boolean Array

## Discovery Record

| Item | Value |
| --- | --- |
| Project family labels observed | `grouped_topk_null_sort_key@pyarrow`, `topk_filter_pushdown@pyarrow` |
| First recorded live signal | 2026-05-25 15:41:41 CST (`arrow_cross`, case index `8816`) |
| Second confirming live shape | 2026-05-25 15:51:34 CST (`arrow_cross`, case index `14067`) |
| Manual isolation date | 2026-05-26 CST |
| How found | 24h fresh live differential test: pandas/DuckDB agreed while PyArrow returned a different Boolean aggregate; prefix execution located the first divergence at `groupby`, followed by an adapter-free native reproducer |
| Status | Submitted upstream as [apache/arrow#50043](https://github.com/apache/arrow/issues/50043); upstream issue is closed/completed and labeled `Type: bug`, `Component: C++`, and `Component: Python`. |

The two live family labels above reduce to the same likely root cause and must
not be counted as two independent bugs.

## Title

`Table.group_by(...).aggregate([("flag", "any")])` returns incorrect `True` on a sliced Boolean array with nulls

## Environment

```text
pyarrow: 24.0.0
Python: 3.12.3
Platform: Linux
```

`24.0.0` was the latest PyPI release when reproduced on 2026-05-26.

## Reproducer

```python
import pyarrow as pa
import pyarrow.compute as pc

base = pa.table({"g": [99, 10, 10], "flag": [True, False, None]})
sliced = base.slice(1)
rebuilt = pa.table(sliced.to_pydict())

def grouped_any(table):
    return (
        table.group_by("g", use_threads=False)
        .aggregate([("flag", "any")])
        .column("flag_any")
        .to_pylist()
    )

print("offset:", sliced["flag"].chunk(0).offset)
print("scalar any:", pc.any(pa.array([False, None])).as_py())
print("sliced:", grouped_any(sliced))
print("rebuilt:", grouped_any(rebuilt))
```

## Actual Output

```text
offset: 1
scalar any: False
sliced: [True]
rebuilt: [False]
```

## Expected Output

```text
sliced: [False]
rebuilt: [False]
```

After `slice(1)`, the only rows in group `g=10` have `flag` values
`[False, None]`. The grouped `any` result should therefore agree with scalar
`compute.any` and the zero-offset rebuilt table.

## Evidence

- Extracted executable reproducer:
  `new_issue/generated/issue-bundles/reproducers/pyarrow_slice_groupby_any.py`
- Bundle manifest:
  `new_issue/generated/issue-bundles/manifest.json`
- Reduced DSL artifacts: `bugs/bug_425750e242985a47/reduced_case.json` and
  `bugs/bug_7b3fcc033c85a371/reduced_case.json`
- Duplicate search on 2026-05-26 did not identify an existing Arrow issue for
  the sliced Boolean `group_by(... any ...)` value mismatch.
