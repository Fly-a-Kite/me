# Issue Draft: DataFusion Duplicate Ordered LIMIT Changes Query Result

## Discovery Record

| Item | Value |
| --- | --- |
| Project family labels observed | `metamorphic_limit_idempotence@datafusion`, `datafusion_limit_idempotence@datafusion` |
| First automated signal | 2026-05-26 22:45 CST, `discovery-run-deep-existing-manifest.json` |
| Deterministic audit record | 2026-05-26 22:53:51 CST, `datadiff bug-audit --probes datafusion_limit_idempotence` |
| How found | DataDiffFuzz generated a top-k/sort/offset workflow and the metamorphic oracle duplicated an already applied ordered `LIMIT`. The base and transformed programs should be equivalent, but DataFusion returned fewer rows after the duplicate `LIMIT`. |
| Current status | Submitted upstream as [apache/datafusion#22541](https://github.com/apache/datafusion/issues/22541); upstream issue is open, labeled `bug`, and assigned. |

## Title

Repeating the same ordered `LIMIT` over a subquery changes the result

## Environment

```text
datafusion: 53.0.0
pyarrow: 24.0.0
Python: 3.12.3
Platform: Linux
```

## Why This Should Be Equivalent

The inner query first computes:

```sql
SELECT * FROM t0 ORDER BY s ASC NULLS FIRST, id ASC NULLS LAST LIMIT 5
```

Applying the same `ORDER BY ... LIMIT 5` again to that already limited ordered
result should not remove additional rows. The outer `ORDER BY x ... OFFSET 2`
is the same in both queries, so the final result should match.

## Reproducer

```python
import pyarrow as pa
from datafusion import SessionContext

rows = [
    {"id": 0, "g": "a", "x": None, "z": 8, "s": "A"},
    {"id": 1, "g": "b", "x": 10, "z": None, "s": "space value"},
    {"id": 2, "g": "c", "x": -10, "z": -3, "s": ""},
    {"id": 3, "g": "b", "x": -2, "z": 0, "s": "space value"},
    {"id": 4, "g": "b", "x": 10, "z": None, "s": "a"},
    {"id": 5, "g": "a", "x": None, "z": 0, "s": ""},
    {"id": 6, "g": None, "x": None, "z": 0, "s": "space value"},
    {"id": 7, "g": "a", "x": 0, "z": 1, "s": "a"},
    {"id": 8, "g": "c", "x": 0, "z": 0, "s": "A"},
    {"id": 9, "g": "c", "x": -10, "z": 8, "s": "A"},
    {"id": 10, "g": "c", "x": -10, "z": -3, "s": "space value"},
    {"id": 11, "g": "b", "x": -10, "z": 8, "s": "A"},
    {"id": 12, "g": "a", "x": -2, "z": -3, "s": ""},
    {"id": 13, "g": "a", "x": 2, "z": 1, "s": "space value"},
]

ctx = SessionContext()
table = pa.table(
    {key: [row[key] for row in rows] for key in ["id", "g", "x", "z", "s"]},
    schema=pa.schema(
        [
            pa.field("id", pa.int64(), nullable=False),
            pa.field("g", pa.string(), nullable=True),
            pa.field("x", pa.int64(), nullable=True),
            pa.field("z", pa.int64(), nullable=True),
            pa.field("s", pa.string(), nullable=True),
        ]
    ),
)
ctx.register_record_batches("t0", [table.to_batches()])

base = """
SELECT * FROM (
  SELECT * FROM (
    SELECT * FROM t0 ORDER BY s ASC NULLS FIRST, id ASC NULLS LAST LIMIT 5
  ) q ORDER BY x DESC NULLS FIRST, id ASC NULLS LAST OFFSET 2
) q ORDER BY x DESC NULLS FIRST, id ASC NULLS LAST
"""

duplicate_limit = """
SELECT * FROM (
  SELECT * FROM (
    SELECT * FROM (
      SELECT * FROM t0 ORDER BY s ASC NULLS FIRST, id ASC NULLS LAST LIMIT 5
    ) q ORDER BY s ASC NULLS FIRST, id ASC NULLS LAST LIMIT 5
  ) q ORDER BY x DESC NULLS FIRST, id ASC NULLS LAST OFFSET 2
) q ORDER BY x DESC NULLS FIRST, id ASC NULLS LAST
"""

print("base")
print(ctx.sql(base).to_pandas().to_string(index=False))
print("duplicate_limit")
print(ctx.sql(duplicate_limit).to_pandas().to_string(index=False))
```

## Actual Output

```text
base
 id g   x  z s
  8 c   0  0 A
 12 a  -2 -3
  2 c -10 -3

duplicate_limit
 id g  x  z s
 12 a -2 -3
```

## Expected Output

Both queries should return the same three rows as the base query.

## DataDiffFuzz Evidence

- Integrated manifest: `new_issue/generated/discovery-run-deep-existing-manifest.json`
- Fresh candidate evidence: `new_issue/generated/discovery-run-deep-existing-manifest-fresh-candidates.json`
- First example artifact: `bugs/bug_1bf9d3c0ae839abf`
- Generated audit draft: `new_issue/generated/datafusion_limit_idempotence.md`
- Deterministic probe: `src/datadiff/bug_audit.py`, probe id `datafusion_limit_idempotence`
- Saturated family keys after audit: `datafusion_limit_idempotence@datafusion`
  and the original fuzz root `metamorphic_limit_idempotence@datafusion`

Project reproduction command:

```bash
.venv/bin/datadiff bug-audit --probes datafusion_limit_idempotence
```

## Upstream Status

Submitted by Fly-a-Kite as
[apache/datafusion#22541](https://github.com/apache/datafusion/issues/22541).
As of 2026-05-28 20:10:59 +08:00, GitHub shows the issue open, labeled `bug`,
and assigned to `raman20`. This is recorded in
`experiments/latest_confirmations.json` as `upstream_labeled_bug`.

Initial GitHub issue search found related DataFusion sorting/limit topics such
as `apache/datafusion#9488`, `apache/datafusion#20048`, and
`apache/datafusion#21973`; #22541 is the project-submitted issue for this exact
duplicate ordered `LIMIT` metamorphic case.
