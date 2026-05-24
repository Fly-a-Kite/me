# Final Reduced Reproducers - 2026-05-24

This bundle keeps only candidates reproduced by the current project code after
removing adapter noise. All three counted candidates are organic, reduced, and
triaged as `candidate_implementation_bug` with
`candidate_bug_needs_external_confirmation`.

## Valid candidates

| Label | Artifact | Suspicious backend | Reduced shape | Observed difference |
| --- | --- | --- | --- | --- |
| DataFusion grouped top-k NULL sort key | `bug_443e55c7231a3902` | `datafusion` | rows `[1,2]`, cols `[2,3]`, ops `6` | pandas/duckdb return `[1, null, "zDo"]`; DataFusion returns `[]` |
| Polars reverse division operand order | `bug_f82da91dcce39d31` | `polars` | rows `[1]`, cols `[2]`, ops `1` | eager Polars gives `ratio_value=0.8`; lazy/reference gives `1.25` |
| DuckDB row-value absence NULL filter | `bug_c57a8db95c205092` | `duckdb` | rows `[1,1]`, cols `[2,2]`, ops `1` | DuckDB returns `[]`; SQLite/reference returns `[(2, 2)]` |

## Standalone reproducers

These scripts do not import DataDiffFuzz. On the affected local versions they
print the minimal difference and fail an assertion:

```bash
./.venv/bin/python bugs/final_repros_20260524/bug_443e55c7231a3902/standalone_datafusion_groupby_null_sortkey_limit.py
./.venv/bin/python bugs/final_repros_20260524/bug_f82da91dcce39d31/standalone_polars_reverse_division_columns.py
./.venv/bin/python bugs/final_repros_20260524/bug_c57a8db95c205092/standalone_duckdb_tuple_absence_null_filter.py
```

Observed local versions during standalone runs:

| Backend | Version |
| --- | --- |
| DataFusion | `53.0.0` |
| PyArrow | `24.0.0` |
| Polars | `1.40.1` |
| DuckDB | `1.5.3` |
| SQLite | `3.45.1` |

## Validation commands

These commands passed and preserved the original root cause:

```bash
./.venv/bin/python -m datadiff.cli validate-artifact --bug bugs/final_repros_20260524/bug_443e55c7231a3902 --backends pandas,duckdb,datafusion
./.venv/bin/python -m datadiff.cli validate-artifact --bug bugs/final_repros_20260524/bug_f82da91dcce39d31 --backends polars,polars_lazy
./.venv/bin/python -m datadiff.cli validate-artifact --bug bugs/final_repros_20260524/bug_c57a8db95c205092 --backends duckdb,sqlite
```

## Rejected noise

`bug_fafd8cd305b9c253` is no longer counted. The previous PyArrow difference
came from the project adapter casting `in_set` literals to the column type,
turning `0.5` into `0`. The current code compares membership literals without
that unsafe cast, and the artifact now validates as `not-reproduced`.

## Source canaries

| Suite/preset/seed | Manifest | Run | Counted status |
| --- | --- | --- | --- |
| `datafusion_cross/live_datafusion/9501` | `runs/experiment-20260523T191615-1779563775028932760.json` | `runs/run-20260523T191528-1779563728215530436.jsonl.gz` | counted |
| `dataframe_lazy/live_polars_lazy/9503` | `runs/experiment-20260523T191547-1779563747820855431.json` | `runs/run-20260523T191528-1779563728284618301.jsonl.gz` | counted |
| `embedded_sql/live_embedded_sql/9504` | `runs/experiment-20260523T191607-1779563767052582721.json` | `runs/run-20260523T191528-1779563728306017720.jsonl.gz` | counted |
| `arrow_cross/live_arrow/9502` | `runs/experiment-20260523T191607-1779563767632035394.json` | `runs/run-20260523T191528-1779563728240668270.jsonl.gz` | rejected adapter noise |

## Focused tests run

```bash
./.venv/bin/python -m pytest tests/test_triage.py tests/test_regression_findings.py tests/test_runner.py::test_pyarrow_backend_matches_common_join_groupby_case
```
