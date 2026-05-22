# Historical Bug Candidate Intake

Read `FINAL_GOAL.md` first. This document is the intake list for historical bugs that may become replay evidence after isolated validation. It is not the counted historical registry.

## Status Meanings

- `candidate`: plausible wrong-result bug, but not yet locally replayed with the final DataDiffFuzz code.
- `pending_validation`: upstream minimal reproducer works, and the next step is to express or search for it through the DataDiffFuzz DSL.
- `pending_dsl_support`: upstream reproducer is locally validated on vulnerable/fixed versions, but the current final DSL cannot yet express the required semantics without a general extension.
- `promoted`: locally replayed with final DataDiffFuzz code on a vulnerable version and verified absent on a fixed version; the counted registry entry lives in `src/datadiff/historical.py`.
- `validation_failed`: attempted locally, but the stated vulnerable package/version did not reproduce the upstream symptom in the isolated environment.
- `out_of_scope`: real upstream bug, but it requires operators or data sources outside the current DSL/backends.
- `rejected`: not suitable for historical replay evidence.

A bug can move into `src/datadiff/historical.py` only after the vulnerable and fixed versions are known and a local isolated replay succeeds.

## Shortlist

| Candidate | Status | Upstream evidence | Vulnerable target | Fixed target | Current fit | Next validation step |
| --- | --- | --- | --- | --- | --- | --- |
| `duckdb-3015` | `candidate` | https://github.com/duckdb/duckdb/issues/3015, fixed by https://github.com/duckdb/duckdb/pull/3040 | DuckDB 0.3.1 Python client | DuckDB 0.3.2 locally verified fixed | The final code now has general per-column sort/null ordering, fixture import, and a `replay-fixture` entry point. The official fixture spec is `experiments/historical_replays/duckdb-3015.fixture.json`. A fixed-version Python 3.10 replay with DuckDB 0.3.2 produced no findings. | Do not count yet. A Python 3.10 isolated install of `duckdb==0.3.1` had no compatible wheel and source build failed, so the vulnerable replay is still diagnostic-only unless a Python>=3.10-compatible vulnerable build is produced or a general compatibility-runner protocol is approved. Evidence summary: `reports/duckdb-3015-datadiff-fixture-replay.md`. |
| `duckdb-22075` | `promoted` | https://github.com/duckdb/duckdb/issues/22075, fixed by merged PR https://github.com/duckdb/duckdb/pull/22218 (`5909259229afa28a33cee6075dc37fa602477325`) | DuckDB 1.5.0-1.5.2 | DuckDB 1.5.3.dev26 was locally verified fixed; merge commit `5909259229afa28a33cee6075dc37fa602477325` | The general DSL now supports post-groupby joins and `aggregate` for global aggregation. The replay uses the generic `join_groupby_stress` profile over `cross_family` backends. | Promoted to `src/datadiff/historical.py` on 2026-05-20. A 20-case local replay with DuckDB 1.5.2 produced exactly one candidate family, `groupby_aggregation@duckdb`; the same 20-case replay on DuckDB 1.5.3.dev26 produced no findings. Evidence summary: `reports/duckdb-22075-historical-replay.md`. |
| `duckdb-22656` | `promoted` | https://github.com/duckdb/duckdb/issues/22656, fixed by merged PR https://github.com/duckdb/duckdb/pull/22744 (`67af30b260d7f0d8ad33092fc8e45a2ee87946c0`) | DuckDB 1.5.2 | DuckDB 1.5.3 locally verified fixed | The general DSL now supports `offset`, and the generic `duckdb_persistent` target exercises persisted storage row-group pruning paths without bug-specific backend code. The replay uses the `storage_offset` profile over `duckdb_storage_cross`. | Promoted to `src/datadiff/historical.py` on 2026-05-21 after isolated vulnerable/fixed replay. Evidence summary: `reports/duckdb-22656-historical-replay.md`. |
| `datafusion-20244` | `candidate` | https://github.com/apache/datafusion/issues/20244, fixed by https://github.com/apache/datafusion/pull/20245 and https://github.com/apache/datafusion/pull/20247 | DataFusion 52.1.0 CLI according to issue | DataFusion 52.2.0 release track according to issue timeline | Groupby + sort + expression mutation is partly in scope. The upstream bug depends on ordered Parquet metadata and a `WITH ORDER` external table, which the current in-memory Python adapter does not generate. | Check whether the Python `datafusion` package has a vulnerable version that exposes the same optimizer path. If not, this needs a source-built CLI replay outside the default registry. |
| `datafusion-16638` | `out_of_scope` | https://github.com/apache/datafusion/issues/16638, fixed by https://github.com/apache/datafusion/pull/16641 | DataFusion version before 2025-07-03 fix | post-PR #16641 | Sort + limit + join is in scope, but the reproducer uses `LEFT ANTI JOIN`; the current DSL only emits `inner` and `left`. | Only revisit after adding general semi/anti join support across reference and target backends. Do not add a one-off anti-join reproducer to the final registry. |
| `datafusion-14335` | `out_of_scope` | https://github.com/apache/datafusion/issues/14335, fixed by https://github.com/apache/datafusion/pull/14338 | DataFusion before the 2025-01-28 fix; linked to release 45.0.0 timeline | post-PR #14338 | Limit + join is in scope, but the reproducer requires `FULL OUTER JOIN`; the current DSL only emits `inner` and `left`. | Only revisit after adding general full outer join support across reference and target backends. |
| `polars-26803` | `validation_failed` | https://github.com/pola-rs/polars/issues/26803, fixed by https://github.com/pola-rs/polars/pull/26804 | Issue reports Polars 1.38.1 | post-PR #26804 | Sort + head/limit + filter over eager vs lazy is directly in scope. | On 2026-05-20, an isolated venv with PyPI `polars==1.38.1` produced the correct upstream and DataDiffFuzz outputs. Keep out of the counted registry until a truly vulnerable wheel or commit is identified. |
| `polars-21439` | `out_of_scope` | https://github.com/pola-rs/polars/issues/21439 | Polars 1.23.0 according to issue | closed 2025-10-28; fixed version must be verified | It is a lazy optimizer error involving datetime extraction in `group_by` and a later filter. The current DSL lacks datetime columns and groupby expression keys. | Revisit only if datetime/groupby-expression support is added as a general DSL extension. |

## Rejected Or Deferred Search Hits

| Issue | Decision | Reason |
| --- | --- | --- |
| https://github.com/pola-rs/polars/issues/27171 | `rejected` | Requires grouped `arg_min`/`arg_max`, binary values, and `max_by`; not represented by current aggregations. |
| https://github.com/pola-rs/polars/issues/27610 | `rejected` | Requires list aggregation with `unique().sort()`; not represented by current aggregations. |
| https://github.com/pola-rs/polars/issues/26680 | `rejected` | Requires `sort_by(nulls_last=True)` inside `group_by().agg()` with `map_batches`; not represented by current DSL. |
| https://github.com/pola-rs/polars/issues/22926 | `rejected` | Polars SQL `HAVING` issue; current Polars adapters use expression APIs, not SQL. |
| https://github.com/duckdb/duckdb/issues/21719 | `rejected` | Requires `SELECT DISTINCT`, CTEs, and intermittent behavior over UUID data; not represented by current DSL. |
| https://github.com/duckdb/duckdb/issues/19313 | `rejected` | SQL `HAVING` without group by; not represented by current DSL. |
| https://github.com/duckdb/duckdb/issues/20486 | `rejected` | Natural join semantics; current DSL uses explicit equi-joins. |
| https://github.com/apache/datafusion/issues/15032 | `rejected` | Correlated subquery aggregate bug; current DSL has no subqueries. |
| https://github.com/apache/datafusion/issues/12570 | `rejected` | Grouping sets/ROLLUP/CUBE surface; current DSL has plain groupby only. |
| https://github.com/apache/datafusion/issues/9586 | `rejected` | Multiple `count(distinct)` aggregates; current DSL has no distinct aggregates. |

## Current Priority

1. Use `duckdb-22075` as the first confirmed historical replay, while continuing to add other confirmed fixed historical bugs from the shortlist.
2. Keep `duckdb-3015` as a case-study candidate until the vulnerable DuckDB 0.3.1 target can be replayed by the final code path without a compatibility shim.
3. Revisit `datafusion-14335` and `datafusion-16638` only after adding full/anti join support as a general DSL extension.
4. Do not count `polars-26803` unless a vulnerable wheel or commit is identified; the PyPI 1.38.1 wheel tested locally did not reproduce.
