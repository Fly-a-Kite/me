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

## Class FP-2: NUL-character string literal on SQLite — **open**

- Cause: `value_catalog.py:310` contains `"a\x00z"`. The DuckDB adapter escapes NUL via
  `CHR(0)` (`duckdb_backend.py:91-99`); the SQLite adapter does not, so `sqlite3` raises
  `ValueError: the query contains a null character`. Result: `accept_reject` mismatch with
  sqlite as the "suspicious minority", even though the other five backends simply have an
  adapter that can carry NUL.
- This is a **framework/adapter limitation**, not a SQLite engine bug.
- Recommended fix (pick one):
  1. Capability-gate NUL strings: declare the token unsupported for the SQLite target and skip
     such cases (cleanest; SQLite TEXT cannot represent NUL at all).
  2. Mirror the DuckDB `CHR(0)` handling in `sqlite_backend._lit`; note SQLite's `char(0)`
     yields an empty string, so this converts the error into a boundary value mismatch rather
     than removing the noise.
  3. Classify a suspicious backend whose status is `error` with an adapter/lowering
     `ValueError` as `adapter_error`, not a candidate bug.
- Deferred: choosing between (1) and (3) is a design decision; no risky catalog edit was made.

## Implication

The pipeline's **detection** works (it found real divergences), but its **adjudication** needs
the two boundary classes above before any of these can be called bugs. Both are already
described in the capability models, which is exactly the "declared boundary vs bug" discipline
the paper claims.
