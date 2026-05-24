# Live Exploration Reproducers - 2026-05-24

This bundle summarizes the post-fix large live exploration run. The run used
the current project code after fixing PyArrow `in_set` literal-cast noise.

## Large Run Summary

| Suite | Preset | Seed | Cases | Findings | Candidate bugs | Rewardable candidates | Issue replays | False positives |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `datafusion_cross` | `live_datafusion` | 9701 | 15000 | 304 | 234 | 177 | 57 | 0 |
| `arrow_cross` | `live_arrow` | 9702 | 15000 | 82 | 73 | 29 | 44 | 0 |
| `dataframe_lazy` | `live_polars_lazy` | 9703 | 15000 | 4115 | 4113 | 3138 | 975 | 0 |
| `embedded_sql` | `live_embedded_sql` | 9704 | 15000 | 191 | 187 | 159 | 28 | 0 |

Manifests:

- `runs/experiment-20260524T021412-1779588852257941078.json`
- `runs/experiment-20260524T021233-1779588753680175152.json`
- `runs/experiment-20260524T020329-1779588209001613129.json`
- `runs/experiment-20260524T021025-1779588625461679944.json`

## New Independent Candidates

| Label | Artifact | Source row | Suspicious backend | Reduced shape | Reduced output |
| --- | --- | ---: | --- | --- | --- |
| DataFusion groupby inner-limit outer-offset row loss | `bugs/bug_615197514c6c857f` | 1044 | `datafusion` | rows `[2,1]`, cols `[1,2]`, ops `6` | pandas/duckdb return `[[1,0,0]]`; DataFusion returns `[]` |
| DataFusion negative-zero truth filter | `bugs/bug_72710816fcc5d0d4` | 10958 | `datafusion` | rows `[1,1]`, cols `[2,1]`, ops `3` | pandas/duckdb return `[]`; DataFusion returns `[[1,0,0]]` |

Standalone reproducers:

```bash
./.venv/bin/python bugs/bug_615197514c6c857f/standalone_datafusion_groupby_limit_offset.py
./.venv/bin/python bugs/bug_72710816fcc5d0d4/standalone_datafusion_negative_zero_truth_filter.py
```

Both standalone scripts fail an assertion on DataFusion `53.0.0`, which is the
expected signal for these candidate implementation bugs.

## Secondary Candidates Not Counted As New Families

- Polars `filter_predicate@polars` and `groupby_aggregation@polars` reduce to
  the same `reverse_division_columns` value error already captured in
  `bugs/final_repros_20260524/bug_f82da91dcce39d31`.
- DuckDB `outer_join_truth_filter@duckdb` reduces to the same row-value
  `NOT IN` with `NULL` behavior already captured in
  `bugs/final_repros_20260524/bug_c57a8db95c205092`.
- Arrow no longer produced the old PyArrow fractional membership cast false
  positive as a rewardable candidate.

## Focused Verification

```bash
./.venv/bin/python -m pytest tests/test_triage.py \
  tests/test_regression_findings.py::test_datafusion_groupby_limit_offset_is_candidate_bug \
  tests/test_regression_findings.py::test_datafusion_negative_zero_truth_filter_is_candidate_bug
```
