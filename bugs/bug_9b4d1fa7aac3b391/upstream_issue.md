# DataFusion drops grouped MIN/MAX rows with NULL sort keys under ORDER BY + LIMIT

Submitted upstream as https://github.com/apache/datafusion/issues/22190 on 2026-05-15.

Post-submission status: DataFusion member `xiedeyantu` replied `take` on 2026-05-17 UTC and is assigned to the issue. This is an upstream triage signal, not yet a confirmation of the bug.

## Version

- Python: `3.12.3`
- Python package: `datafusion==53.0.0`
- Arrow input via `pyarrow==24.0.0`
- Platform used for reproduction: Ubuntu 24.04 x86_64, Linux `6.17.0-23-generic`

## Minimal Reproducer

```python
import pyarrow as pa
from datafusion import SessionContext

ctx = SessionContext()
batch = pa.RecordBatch.from_arrays(
    [
        pa.array(["a"], type=pa.string()),
        pa.array([None], type=pa.int64()),
    ],
    schema=pa.schema(
        [
            pa.field("g", pa.string(), nullable=True),
            pa.field("x", pa.int64(), nullable=True),
        ]
    ),
)
ctx.register_record_batches("t0", [[batch]])

base = "SELECT g, MIN(x) AS min_x FROM t0 GROUP BY g"
print(ctx.sql(f"SELECT min_x FROM ({base}) q LIMIT 20").to_pandas())
print(ctx.sql(f"SELECT min_x FROM ({base}) q ORDER BY min_x ASC NULLS LAST LIMIT 20").to_pandas())
```

## Actual Result

Standalone reproducer output:

```text
datafusion=53.0.0
pyarrow=24.0.0
control:
   min_x
0    NaN
top-k:
Empty DataFrame
Columns: [min_x]
Index: []
top-k record-batch rows=0
AssertionError: DataFusion dropped the group whose aggregate sort key is NULL
```

The control query returns the grouped row with a NULL aggregate result:

```text
   min_x
0    NaN
```

The query with `ORDER BY min_x ASC NULLS LAST LIMIT 20` returns no rows:

```text
Empty DataFrame
Columns: [min_x]
Index: []
```

With two groups, one having `MIN(x) = NULL` and one having `MIN(x) = 5`, the `ORDER BY ... LIMIT 20` query returns only the non-NULL sort-key group. The limit is larger than the number of groups, so no grouped row should be removed.

## Expected Result

Both single-group queries should return one row. `GROUP BY g` forms a group for `g = 'a'`; `MIN(x)` is NULL because all values in that group are NULL. Ordering with `NULLS LAST` and a limit larger than the result cardinality should preserve the row.

## Fuzzer Evidence

DataDiffFuzz found this via a cross-backend differential case. Pandas, Polars eager, Polars lazy, DuckDB, SQLite, and the independent DSL reference return the grouped NULL aggregate row; DataFusion alone returns an empty result.

Artifact: `bugs/bug_9b4d1fa7aac3b391`
Standalone script: `bugs/bug_9b4d1fa7aac3b391/standalone_datafusion_groupby_null_sortkey_limit.py`
Preflight boundary script: `scripts/datafusion_topk_null_sortkey_preflight.py`
Captured preflight output: `reports/datafusion-topk-null-sortkey-preflight.txt`

Additional preflight result on `datafusion==53.0.0`:

```text
ok raw_null_sort_topk_control: expected_rows=1 actual_rows=1 rows=[(None,)]
ok null_group_key_nonnull_min_control: expected_rows=1 actual_rows=1 rows=[(5,)]
ok single_null_control: expected_rows=1 actual_rows=1 rows=[(None,)]
ok single_null_order_only: expected_rows=1 actual_rows=1 rows=[(None,)]
FAIL single_null_topk_asc_last: expected_rows=1 actual_rows=0 rows=[]
FAIL single_null_topk_asc_default: expected_rows=1 actual_rows=0 rows=[]
FAIL single_null_topk_asc_last_limit_one: expected_rows=1 actual_rows=0 rows=[]
FAIL single_null_topk_asc_first: expected_rows=1 actual_rows=0 rows=[]
ok single_null_topk_desc_last: expected_rows=1 actual_rows=1 rows=[(None,)]
ok single_null_topk_desc_first: expected_rows=1 actual_rows=1 rows=[(None,)]
FAIL mixed_null_and_value_topk: expected_rows=2 actual_rows=1 rows=[(5,)]
FAIL single_null_max_desc_topk: expected_rows=1 actual_rows=0 rows=[]
FAIL mixed_null_and_value_max_desc_topk: expected_rows=2 actual_rows=1 rows=[(5,)]
ok single_null_sum_asc_topk_control: expected_rows=1 actual_rows=1 rows=[(None,)]
ok single_null_avg_asc_topk_control: expected_rows=1 actual_rows=1 rows=[(None,)]
ok mixed_limit_one_expected_value_only: expected_rows=1 actual_rows=1 rows=[(5,)]
```

This narrows the trigger to grouped top-k over NULL `MIN`/`MAX` aggregate sort keys. Plain NULL column top-k is OK, a NULL group key with a non-NULL aggregate is OK, and grouped `SUM`/`AVG` NULL aggregate top-k is OK in this preflight. The failing combinations observed here are `MIN(x)` ordered ascending and `MAX(x)` ordered descending when the aggregate result is NULL.

Ubuntu campaign evidence:

- Main manifest: `runs/experiment-20260514T175130.json`
- Main summary: `reports/experiment-summary-experiment-20260514T175130.md`
- Confirmation manifest: `runs/experiment-20260514T180106.json`
- Confirmation summary: `reports/experiment-summary-experiment-20260514T180106.md`
- Evidence snapshot: `reports/ubuntu-datafusion-confirmation.md`
- Independent-seed domain manifest: `runs/experiment-20260514T192105.json`
- Independent-seed domain summary: `reports/experiment-summary-experiment-20260514T192105.md`
- Pre-issue sanity manifest: `runs/experiment-20260514T194522.json`
- Pre-issue sanity summary: `reports/experiment-summary-experiment-20260514T194522.md`
- Post-enhancement null-aggregate manifest: `runs/experiment-20260514T195546.json`
- Pattern variant analysis: `reports/pattern-variants-null_agg_topk-experiment-20260514T195546.md`
- `core_datafusion/null_groupby_topk`: 3,797 candidate findings across 5,000 cases.
- `core_datafusion/null_agg_topk`: 2,934 candidate findings across 5,000 cases.
- `datafusion_cross/null_groupby_topk`: 3,798 candidate findings across 5,000 cases.
- `datafusion_cross/null_agg_topk`: 2,970 candidate findings across 5,000 cases.
- Confirmation run with new seeds executed 10,000 cases and found 6,744 candidate findings, all deduplicating to `grouped_topk_null_sort_key@datafusion`, with 0 oracle false positives.
- Independent-seed domain run executed 60,000 cases with seeds `5001,6001,7001,8001,9001` and found 13,601 candidate findings, all deduplicating to `grouped_topk_null_sort_key@datafusion`, with 0 oracle false positives and 0 semantic divergences.
- Pre-issue sanity run executed 3,600 cases with seeds `12001,13001,14001` and found 2,336 candidate findings, all deduplicating to `grouped_topk_null_sort_key@datafusion`, with 0 oracle false positives.
- Post-enhancement null-aggregate run executed 3,000 cases with seeds `15001,16001,17001` and found 2,562 candidate findings, all deduplicating to `grouped_topk_null_sort_key@datafusion`, with 0 oracle false positives.
- Variant analysis of that run shows the boundary:
  - `MIN ASC`: 1,298 generated cases, 1,252 candidate cases.
  - `MAX DESC`: 1,412 generated cases, 1,310 candidate cases.
  - `MIN DESC`: 89 generated cases, 0 candidate cases.
  - `MAX ASC`: 143 generated cases, 0 candidate cases.
- Current triage root: `grouped_topk_null_sort_key`.
- Current artifact validation status: finding still reproduces; root classification changed from the older broad `groupby_aggregation` label to the narrower `grouped_topk_null_sort_key` label.

This appears related to the previously minimized NULL group-key/top-k case, but this reproducer uses a non-NULL group key and a NULL `MIN(x)` aggregate sort key. The boundary script suggests the issue is specifically tied to grouped top-k over NULL `MIN`/`MAX` aggregate sort keys, not to all NULL sort keys.

Related variant found by the same fuzzer:

```sql
SELECT m_0
FROM (
  SELECT m_0, COUNT(*) AS c
  FROM t0
  GROUP BY m_0
) q
ORDER BY m_0 DESC NULLS LAST
LIMIT 5
```

With a single input row where `m_0 = NULL`, the control query without `ORDER BY` returns the NULL group, while the ordered/limited query returns zero rows.
