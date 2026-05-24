# DataFusion evaluates -0.0 >= 0.0 as false

## Summary

DataFusion `53.0.0` evaluates `-0.0 >= 0.0` as `false` in a SQL expression.
Python and DuckDB evaluate the same comparison as `true`. This also changes
filter results for predicates using `IS TRUE` / `IS NOT TRUE`.

## Environment

- `datafusion==53.0.0`
- `pyarrow==24.0.0`
- `duckdb==1.5.3`
- Python `3.12.3`

## Reproduction

```bash
python standalone_datafusion_negative_zero_truth_filter.py
```

The standalone script is included in this artifact:

```text
bugs/bug_72710816fcc5d0d4/standalone_datafusion_negative_zero_truth_filter.py
```

Minimal DataFusion query:

```sql
SELECT id, y * -1 AS m_0, (y * -1) >= 0.0 AS cmp
FROM t0;
```

with:

```text
t0(id, y): (1, 0.0)
```

## Expected

`y * -1` is `-0.0`, and `-0.0 >= 0.0` should be `true`.

The corresponding filter should remove the row:

```sql
SELECT id, y * -1 AS m_0
FROM t0
WHERE NOT (((y * -1) >= 0.0) IS TRUE);
```

Python and DuckDB agree with this expectation.

## Actual

DataFusion reports the comparison as `false` and keeps the row under the
`IS NOT TRUE` filter.
