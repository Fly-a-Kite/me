# Issue Draft: DuckDB CTE Boolean INNER JOIN Returns Empty Result While EXISTS and Materialized Join Match

## Discovery Record

| Item | Value |
| --- | --- |
| Project family labels observed | `metamorphic_semi_anti_join_rewrite@duckdb` |
| First automated signal | 2026-05-28 CST, `discovery-campaign-operator-cooldown-s46010` |
| Artifact | `bugs/bug_1d25bb8cd42bf069` |
| How found | DataDiffFuzz generated a boolean membership workflow. The metamorphic oracle rewrote a `semi_join` as an equivalent distinct-key `INNER JOIN`; pandas and SQLite matched, while DuckDB returned an empty result for the rewritten query. |
| Current status | Submitted upstream as [duckdb/duckdb#22924](https://github.com/duckdb/duckdb/issues/22924); upstream issue is open with `needs triage`. The local repeated reproducer was previously flaky on DuckDB 1.5.3, so keep it out of confirmed paper counts until upstream triage or stable local replay confirms it. |

## Title

CTE boolean `INNER JOIN` returns no rows, but equivalent `EXISTS` and materialized join return the row

## Environment

```text
duckdb: 1.5.3
Python: 3.12.3
Platform: Linux
```

## Reproducer

```python
import duckdb

print("duckdb", duckdb.__version__)
con = duckdb.connect()
con.execute("CREATE TABLE t0(id BIGINT, flag BOOLEAN)")
con.execute("""
INSERT INTO t0 VALUES
    (1, FALSE), (1, TRUE), (1, FALSE), (0, TRUE), (1, TRUE),
    (0, NULL), (0, TRUE), (2, FALSE), (1, FALSE), (0, TRUE),
    (1, TRUE), (24, FALSE), (1, TRUE), (2, FALSE)
""")
con.execute("CREATE TABLE t1(id BIGINT, z DOUBLE, tag VARCHAR)")
con.execute("""
INSERT INTO t1 VALUES
    (0, 1.0, 'BkcDeJw'), (24, 24.0, 'tag_24')
""")
con.execute("CREATE TABLE keys(flag_key BOOLEAN)")
con.execute("INSERT INTO keys VALUES (FALSE)")

cte = """
WITH step_0 AS (
  SELECT q.id, q.flag, r.z, r.tag FROM t0 q INNER JOIN t1 r ON q.id = r.id
), step_1 AS (
  SELECT q.id, q.flag, q.z, q.tag, q.z * 2 AS m_0 FROM step_0 q
), step_2 AS (
  SELECT * FROM step_1 q WHERE q.tag NOT IN ('beta', '中文', 'alpha')
), step_3 AS (
  SELECT flag, MIN(flag) AS min_flag, MAX(z) AS max_z FROM step_2 q GROUP BY flag
), step_4 AS (
  SELECT q.flag, q.min_flag, q.max_z AS ord FROM step_3 q
), step_5 AS (
  SELECT * FROM step_4 q
  ORDER BY ord ASC NULLS LAST, flag ASC NULLS LAST, min_flag ASC NULLS LAST
  OFFSET 2
), step_6 AS (
  SELECT q.flag, q.min_flag FROM step_5 q
)
"""

step_rows = con.execute(cte + "SELECT * FROM step_6").fetchall()
exists_rows = con.execute(cte + """
SELECT q.* FROM step_6 q
WHERE EXISTS (
  SELECT 1 FROM keys r WHERE r.flag_key IS NOT NULL AND q.flag = r.flag_key
)
""").fetchall()
join_rows = con.execute(
    cte + "SELECT q.* FROM step_6 q INNER JOIN keys r ON q.flag = r.flag_key"
).fetchall()
con.execute("CREATE TEMP TABLE materialized_step_6 AS " + cte + " SELECT * FROM step_6")
materialized_join_rows = con.execute(
    "SELECT q.* FROM materialized_step_6 q INNER JOIN keys r ON q.flag = r.flag_key"
).fetchall()

print("step_6", step_rows)
print("exists", exists_rows)
print("inner_join", join_rows)
print("materialized_inner_join", materialized_join_rows)
```

## Actual Output

```text
duckdb 1.5.3
step_6 [(False, False)]
exists [(False, False)]
inner_join []
materialized_inner_join [(False, False)]
```

## Expected Output

`inner_join` should return `[(False, False)]`, matching both the equivalent
`EXISTS` query and the same join after materializing `step_6`.

## DataDiffFuzz Evidence

- Validated artifact: `bugs/bug_1d25bb8cd42bf069`
- Artifact validation command:

```bash
.venv/bin/datadiff validate-artifact --bug bugs/bug_1d25bb8cd42bf069
```

- Validation result:

```text
status=valid
original_roots=['metamorphic_semi_anti_join_rewrite']
reproduced_roots=['metamorphic_semi_anti_join_rewrite']
```

- Triage command:

```bash
.venv/bin/datadiff triage-artifact --bug bugs/bug_1d25bb8cd42bf069
```

- Triage result: `candidate_implementation_bug`

## Dedup Notes

Initial search did not find an exact DuckDB issue for this CTE boolean
`INNER JOIN` / `EXISTS` / materialization discrepancy. Do one final upstream
search before submission and link any exact duplicate if found.

## Reproducibility Note

On 2026-05-28, `datadiff issue-bundle --run-reproducers --repeat 3` observed
both `inner_join []` and `inner_join [(False, False)]` for this same standalone
script under DuckDB 1.5.3. Keep the draft as local evidence, but do not submit
or count it until a stable latest-version reproducer is isolated.
