# DataFusion DISTINCT ORDER BY NULLS FIRST LIMIT returns the wrong top row

## Discovery Record

| Item | Value |
| --- | --- |
| Candidate family | `distinct_null_topk@datafusion` |
| First automated signal | 2026-05-27 07:07:33 CST |
| First deterministic audit | 2026-05-27 07:21:17 CST |
| How found | Integrated DataDiffFuzz bug-sprint found `join_semantics@datafusion`; minimization showed join was unnecessary and the real trigger was `DISTINCT + ORDER BY NULLS FIRST + LIMIT`. The new `datadiff bug-audit` probe now reproduces it automatically. |
| Evidence artifact | `bugs/bug_e9e3a5c96a9390d1` |
| Fresh candidate manifest | `new_issue/generated/bug-sprint-bool-not-organic-s8902-manifest-datafusion_optimizer-seed8902-fresh-candidates.json` |
| Integrated focused sprint | `new_issue/generated/bug-sprint-datafusion-common-api-distinct-null-topk-focused-s25-manifest.json` |
| Deterministic audit manifest | `new_issue/generated/manifest.json` |
| Deterministic report | `reports/bug-audit-20260526T232117.md` |
| Target backend/version | DataFusion `53.0.0` |
| Status | Submitted upstream as [apache/datafusion#22554](https://github.com/apache/datafusion/issues/22554); upstream labeled `bug`. The submitted issue body currently has the Output and Expected snippets reversed, so a follow-up correction comment is needed. |

## Summary

`SELECT DISTINCT v FROM t0 ORDER BY v ASC NULLS FIRST LIMIT 1` should return
the first row of the same ordered `DISTINCT` query without `LIMIT`. On
DataFusion `53.0.0`, the full ordered query puts `NULL` first, but adding
`LIMIT 1` returns the first non-null value for string, integer, and float
columns.

This is a low-complexity, common SQL pattern: deduplicate a nullable column,
sort it with explicit null placement, and take the top row.

## Reproducer

```python
import pyarrow as pa
from datafusion import SessionContext

ctx = SessionContext()
batch = pa.RecordBatch.from_pylist(
    [{"v": None}, {"v": ""}, {"v": "a"}],
    schema=pa.schema([pa.field("v", pa.string(), nullable=True)]),
)
ctx.register_record_batches("t0", [[batch]])

full_sql = "SELECT DISTINCT v FROM t0 ORDER BY v ASC NULLS FIRST"
top1_sql = full_sql + " LIMIT 1"

full = ctx.sql(full_sql).collect()[0].to_pydict()["v"]
top1 = ctx.sql(top1_sql).collect()[0].to_pydict()["v"]

print("full:", full)
print("top1:", top1)
assert top1 == full[:1]
```

## Expected

```text
full: [None, '', 'a']
top1: [None]
```

## Actual

```text
full: [None, '', 'a']
top1: ['']
```

## Evidence

DataDiffFuzz now has an automated deterministic probe:

```bash
datadiff bug-audit --probes datafusion_distinct_null_topk
```

The latest full audit report recorded:

```text
Candidate family: distinct_null_topk@datafusion
DataFusion version: 53.0.0
Checked cases: 16
Mismatches: 6
First mismatch: utf8 ASC NULLS FIRST LIMIT 1 returned '' instead of NULL
```

The same probe also found the issue for `int64` and `float64` values with
`ASC NULLS FIRST` and `DESC NULLS FIRST`. `NULLS LAST` cases matched the
full ordered query.

The focused integrated sprint also reproduces the same family through the
normal runner/oracle/classification/evidence path:

```bash
datadiff bug-sprint --cases 20 --seeds 25 --lanes datafusion_common_api --skip-bug-audit --output-manifest new_issue/generated/bug-sprint-datafusion-common-api-distinct-null-topk-focused-s25-manifest.json --classify-limit 10
```

That run produced `distinct_null_topk@datafusion: 4` fresh candidates. The
first candidate used exactly:

```json
[
  {"op": "distinct", "columns": ["s"]},
  {"op": "sort", "keys": [{"column": "s", "ascending": true, "nulls": "first"}]},
  {"op": "limit", "n": 1}
]
```

Normalized outputs were pandas/DuckDB `[[null]]` versus DataFusion `[[""]]`.

## Environment

Reproduced with the Python DataFusion package.

- OS: Ubuntu 24.04.1, Linux kernel `6.17.0-29-generic`
- Architecture: `x86_64`
- Python: `3.12.3`
- datafusion: `53.0.0`
- pyarrow: `24.0.0`

Additional packages used for differential checking:

- pandas: `3.0.3`
- duckdb: `1.5.3`
- polars: `1.41.0`
- sqlite3 module SQLite version: `3.45.1`

Install command:

```bash
python -m pip install datafusion==53.0.0 pyarrow==24.0.0
```

## Duplicate Search Notes

Initial GitHub searches checked Apache DataFusion issues and PRs for:

- `DISTINCT "NULLS FIRST" LIMIT`
- `"ORDER BY" "NULLS FIRST" DISTINCT LIMIT`
- `"NULLS FIRST" "LIMIT 1"`
- `"nulls_first" "distinct"`

The visible related items were performance or different TopK/sort issues,
for example `#16620`, `#16837`, `#12423`, `#18202`, `#21580`, and `#21621`.
They did not describe this minimal `DISTINCT + NULLS FIRST + LIMIT` wrong-row
case. A final upstream search should still be done immediately before filing.

Final duplicate search on 2026-05-27 found
[apache/datafusion#22554](https://github.com/apache/datafusion/issues/22554),
opened by `Fly-a-Kite` with the same title and labeled `bug`. No older exact
duplicate was found in the required exact/key-phrase searches:

- `DISTINCT ORDER BY NULLS FIRST LIMIT`
- `DISTINCT NULLS FIRST LIMIT wrong row`
- `TopK NULLS FIRST DISTINCT`
- `LIMIT returns non-null before null`

Follow-up comment/edit needed for #22554: the local reproducer prints the
actual DataFusion output `top1: ['']` and expects `top1: [None]`, but the
submitted issue body currently has those two snippets reversed. A ready-to-post
correction draft is stored in `new_issue/datafusion_22554_followup_comment.md`.

## Project Integration

The issue is no longer only an AI/manual observation. It is encoded in project
code as:

- `src/datadiff/bug_audit.py`: `datafusion_distinct_null_topk` deterministic probe.
- `src/datadiff/oracle.py`: `distinct_null_topk` root-cause classifier.
- `src/datadiff/guidance.py`: `pattern:distinct_null_topk` target feature.
- `src/datadiff/datagen.py`: common API workflow template `distinct_null_topk`.
- `src/datadiff/cli.py`: `datafusion_common_api` bug-sprint lane for low-complexity DataFusion workflows.
- `tests/test_bug_audit.py`, `tests/test_oracle.py`, `tests/test_guidance.py`,
  `tests/test_operation_combo.py`, and `tests/test_datagen.py`: regression tests.
