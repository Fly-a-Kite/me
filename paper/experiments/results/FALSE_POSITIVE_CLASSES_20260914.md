# False-positive classes found by the 24 h run (2026-09-14)

The longhaul produced 3 fresh candidate families. Manual adjudication shows **none of them are
implementation bugs yet**; two are clean false-positive classes. This is direct RQ2
(noise-control) material.

## Raw candidate yield (our run, `20260914T140327Z`)

| Family | Candidates | Localisation |
| --- | ---: | --- |
| `conditional_expression@pyarrow` | 4 | special-float filter literal (NaN-as-value vs NaN-as-null) |
| `conditional_expression@sqlite` | 6 | SQLite adapter cannot pass NUL-containing strings to SQL |
| `running_sum_precision@sqlite` | 2 | same NUL-adapter cause (`accept_reject`, sqlite error) |

## Class FP-1: special-float filter literal — **fixed**

- Cause: the reference treats NaN as nullish, PyArrow follows IEEE; `sum_x <= NaN` with
  `le_is_not_false` therefore keeps vs drops the row. All capability models already declare
  `nan_distinct_from_null`.
- Fix: new pre-reference boundary rule `boundary:special_float_filter_literal`
  (`case_has_special_float_filter_literal`); re-triage of the candidate now returns
  `expected_semantic_divergence`.
- Files: `case_features.py`, `classification_oracle.py`, `semantic_boundaries.py`;
  test `tests/test_special_float_filter_literal_boundary.py` (4 passed); 158 related tests pass.

## Class FP-2: NUL-character string literal on SQLite — **fixed**

- Cause: `value_catalog.py:310` contains `"a\x00z"`. The DuckDB adapter escapes NUL via
  `CHR(0)` (`duckdb_backend.py:91-99`); the SQLite adapter did not, so `sqlite3` raised
  `ValueError: the query contains a null character`. Result: `accept_reject` mismatch with
  sqlite as the "suspicious minority", even though the other five backends simply have an
  adapter that can carry NUL.
- This was a **framework/adapter limitation**, not a SQLite engine bug.
- Fix applied: `sqlite_backend._lit` now raises the existing typed
  `HarnessLoweringError("sqlite cannot represent NUL characters in string literals")`. The
  pipeline's existing `harness_lowering_errors` path then excludes the case, so no new
  classification code was needed.
- Verified with a fresh re-execution: `classify_finding` now returns
  `verdict=harness_lowering_error`, `paper_status=exclude_harness_lowering_failure`,
  `false_positive=True` (`shared_sql_lowering_failure`).
- Tests: `tests/test_sqlite_nul_literal.py` (2) plus 152 classification/triage tests pass;
  broader sqlite-related run is 397 passed / 6 known W1 interpreter-drift failures.
- Files: `sqlite_backend.py`.

## Implication

The pipeline's **detection** works (it found real divergences), but its **adjudication** needs
the two boundary classes above before any of these can be called bugs. Both are already
described in the capability models, which is exactly the "declared boundary vs bug" discipline
the paper claims.

## Scale and effect of FP-2 (measured)

- Our run produced **516 candidate artifact directories** and the aggregate reported
  **17 distinct "fresh families"** (all `@sqlite`).
- A random sample of **60/60** candidates shares the *same* sqlite error
  (`the query contains a null character`). The 17 labels are the same adapter defect surfacing
  under different finding root causes — a **1-defect / 17-labels / 516-artifacts** inflation.
- After the `HarnessLoweringError` fix, batches **0704–0710 report zero fresh families**, while
  the pre-fix batches 0702–0703 still do.

This is the paper's cleanest noise-control anecdote: raw candidate counts are dominated by a
harness defect until the boundary is declared, after which the same workload reports nothing.
