# DataFusion drops grouped row after inner ORDER BY/LIMIT and outer ORDER BY/OFFSET

## Summary

DataFusion `53.0.0` returns an empty result for a two-row grouped aggregate
when an inner ordered `LIMIT` is followed by an outer `ORDER BY ... OFFSET 1`.
The same query without the inner `LIMIT` returns the expected second row.

## Environment

- `datafusion==53.0.0`
- `pyarrow==24.0.0`
- Python `3.12.3`

## Reproduction

```bash
python standalone_datafusion_groupby_limit_offset.py
```

The standalone script is included in this artifact:

```text
bugs/bug_615197514c6c857f/standalone_datafusion_groupby_limit_offset.py
```

Core query:

```sql
SELECT *
FROM (
  SELECT *
  FROM (
    SELECT
      t0.id,
      COUNT(t0.id) AS count_id,
      COUNT(DISTINCT j) AS nunique_j
    FROM t0
    LEFT JOIN t1 ON t0.id = t1.id
    GROUP BY t0.id
  ) q
  ORDER BY id DESC NULLS LAST, count_id DESC NULLS LAST, nunique_j DESC NULLS LAST
  LIMIT 8
) q2
ORDER BY id DESC NULLS LAST, count_id DESC NULLS LAST, nunique_j ASC NULLS LAST
OFFSET 1;
```

Input tables:

```text
t0(id): (0), (1)
t1(id, j): (1, 1)
```

## Expected

The grouped result contains two rows. Applying `OFFSET 1` after the final order
should return the second row:

```text
id=0, count_id=1, nunique_j=0
```

DuckDB and pandas-backed reference execution return that row.

## Actual

DataFusion returns an empty result.

The control query without the inner ordered `LIMIT` returns the expected row,
so the issue appears related to the interaction between grouped aggregate,
inner `ORDER BY/LIMIT`, and outer `ORDER BY/OFFSET`.
