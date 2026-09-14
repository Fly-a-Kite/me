# New candidate: `conditional_expression@pyarrow`

Found by the 24 h engineering longhaul (batch0167, lane `common_api_workflow`, seed 31233502).
First fresh candidate family produced by this sprint.

## Triage summary (from the pipeline, not yet externally confirmed)

| Field | Value |
| --- | --- |
| Backend | **pyarrow** (sole outlier) |
| Root cause | `conditional_expression` |
| Mismatch class | `row_count` |
| Shapes | pandas/polars/polars_lazy/duckdb/sqlite = `(1, 3)`; **pyarrow = `(0, 3)`** |
| Oracle | differential, `reference_majority_support` |
| Evidence | "Independent DSL reference agrees with 5 backends and disagrees with pyarrow." |
| Verdict | `candidate_implementation_bug`, confidence `high`, severity `critical` |
| Needs | external confirmation, recheck, minimization |

Artifact: `bugs/bug_9610b2b66a27d6ac/` (case.json, normalized.json, findings.json,
reproduce.py). Reproducer prints `bug` and the finding dict.

## Program (minimal shape)

`filter k1 IS NOT NULL` → `filter k2 IS NOT NULL` → `semi_join(t_lookup ON k1=k2)`
→ `case_when(flag IS TRUE → "true_key" ELSE "not_true_key") AS survivor_bucket`
→ `groupby(survivor_bucket){count(row_id), sum(x)}` → `sort` →
`filter sum_x <= NaN` with comparator **`le_is_not_false`**.

Five backends plus the DSL reference return `[[6, 0, "true_key"]]`; pyarrow returns zero rows.

### Localisation (bisection, 2026-09-14)

| Program prefix | pyarrow | pandas |
| --- | --- | --- |
| first 6 ops (… `sort`) | `[[6, 0, "true_key"]]` | `[[6, 0, "true_key"]]` |
| all 7 ops (+ final `filter sum_x <= NaN`, `le_is_not_false`) | **`[]`** | `[[6, 0, "true_key"]]` |

The fault is isolated to the final NaN boundary filter: `le_is_not_false` on
`sum_x <= NaN`. PyArrow evaluates the comparison as `False` (IEEE NaN ordering) and drops the
row, while the other five backends and the DSL reference treat it as **not-False**
(unknown/NULL) and keep it.

## Why this matters

- It is on **PyArrow**, one of the execution models no competitor targets (RQ7).
- It is exactly the class of null/NaN boundary semantics this project claims to test, and it is
  discovered **organically**, not from an issue-inspired lane.

## Next steps

1. Recheck + minimize the case; save a standalone reproducer.
2. Determine the exact PyArrow kernel/semantics at fault (`le_is_not_false` with NaN, or
   `case_when` + null).
3. Search upstream for duplicates (Apache Arrow issues).
4. File upstream; track confirmation.

Status: **candidate only** — `candidate_confirmed=false`, `bug_claimed=false`.
