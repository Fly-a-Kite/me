# Confirmed Bug Quality And 20+ Plan

Last updated: 2026-06-14 CST.

## Current Confirmed Count

`experiments/latest_confirmations.json` currently records 9 confirmed
latest-version bug families.

| Family | Backend | Evidence status | Quality |
| --- | --- | --- | --- |
| `grouped_topk_null_sort_key@datafusion` | DataFusion | `apache/datafusion#22190`, open, labeled `bug`, assigned | Strong confirmed, still open. |
| `groupby_aggregation@datafusion` | DataFusion | `apache/datafusion#22489`, closed/completed, labeled `bug,regression` | Strong fixed/completed. |
| `negative_zero_comparison@datafusion` | DataFusion | `apache/datafusion#22490`, closed/completed, labeled `bug` | Strong fixed/completed. |
| `distinct_null_topk@datafusion` | DataFusion | `apache/datafusion#22554`, closed/completed, labeled `bug,regression` | Strong fixed/completed. |
| `datafusion_limit_idempotence@datafusion` | DataFusion | `apache/datafusion#22541`, closed/completed, labeled `bug` | Strong fixed/completed. |
| `metamorphic_semi_anti_join_rewrite@duckdb` | DuckDB | `duckdb/duckdb#22924`, closed/completed; fixed by merged PR `duckdb/duckdb#22963` | Strong fixed/completed. |
| `groupby_aggregation@polars,polars_lazy` | Polars | `pola-rs/polars#27672`, closed/completed, labeled `bug,accepted,P-high` | Strong fixed/completed. |
| `polars_reflected_arithmetic_operand_order@polars` | Polars | `pola-rs/polars#27752`, open, labeled `bug,P-medium` | Strong confirmed, still open. |
| `pyarrow_sliced_bool_groupby_any_all@pyarrow` | PyArrow | `apache/arrow#50043`, closed/completed, labeled `Type: bug` | Strong fixed/completed. |

Quality summary:

- 7 / 9 are closed/completed or fixed upstream.
- 2 / 9 are open but labeled as bugs by upstream.
- The confirmed set spans 4 backend families: DataFusion, DuckDB, Polars, and
  PyArrow.
- This is above the minimum submission bar, but below the 20+ SOTA-like target.

Do not count:

- `polars_vector_division_rounding@polars` / `pola-rs/polars#27753`: closed as
  duplicate/invalid, so it is not a confirmed latest-version bug.

## Path From 9 To 20+

The target is at least 20 confirmed or fixed latest-version bug families. That
means 11 more confirmations are needed.

The latest machine-ranked triage queue is:

- `reports/bug-20plus-triage-plan-20260614.json`;
- `reports/bug-20plus-triage-plan-20260614.md`.

The first DuckDB P0 reproducer queue is:

- `reports/candidate-family-evidence-p0-duckdb-live-20260615.md`;
- `reports/candidate-family-evidence-p0-duckdb-live-20260615-rows/manifest.json`;
- `reports/reproducer-queue-p0-duckdb-live-20260615.md`.
- minimized audit:
  `reports/duckdb-sql-reproducers/p0-duckdb-live-20260615-minimized/manifest.json`.

It contains 4 DuckDB families, 8 prioritized queue rows, and all 8 rows have
immediate recheck reproduction evidence. The regenerated queue also has 8 / 8
available SQLancer/PQS-style witness plans and the batch validator reports 8 /
8 witness revalidations. Use this queue as the entry point for DuckDB
minimization before upstream deduplication and issue filing.

The current top P0 candidates are intentionally biased toward DuckDB, PyArrow,
Pandas, and Polars because DataFusion already has 5 confirmed families. The
first four families converted into minimized reproducer audits are:

1. `union_all_row_append@duckdb`
2. `groupby_aggregation@duckdb`
3. `grouped_topk_null_sort_key@duckdb`
4. `ordering_or_limit@duckdb`

The minimized audit currently has 4 / 4 DataDiff-level target reproductions and
2 / 4 strict native SQL matches. It now re-derives reduced witness plans from
the reduced rerun outputs and preserves the original queue plan as
`source_witness_plan`. Submit or deduplicate the strict native SQL matches
first:

- `grouped_topk_null_sort_key@duckdb`
- `ordering_or_limit@duckdb`

Issue-ready bundles for these two strict native SQL matches have been generated
at:

- `reports/duckdb-issue-ready-bundles/p0-duckdb-live-20260615/manifest.json`;
- `reports/duckdb-issue-ready-bundles/p0-duckdb-live-20260615.tar.gz`;
- checksum:
  `reports/duckdb-issue-ready-bundles/p0-duckdb-live-20260615.tar.gz.sha256`.

The bundle intentionally skips the two non-strict-native-match families below.
It is candidate submission evidence only and does not change the confirmed
family count until DuckDB confirms, labels, or fixes a report.

Treat these as DataDiff/DuckDB evidence requiring an ingestion-path or
precision-path reproducer before SQL-only upstream filing:

- `groupby_aggregation@duckdb`
- `union_all_row_append@duckdb`

For those two mismatch families, use the SQLancer/PQS-inspired witness plan as
a triage aid, not as a new counted discovery source:

- design note: `docs/sqlancer_pqs_transfer_plan.md`;
- start from the already generated reduced witness facts in
  `reports/duckdb-sql-reproducers/p0-duckdb-live-20260615-minimized/manifest.json`;
- determine whether the mismatch is in DuckDB SQL execution, Pandas/Arrow
  ingestion into DuckDB, or numeric/precision representation;
- file upstream only with the narrow proven backend path.

Witness-assisted reproducer work must not change the 9 confirmed-family count
until upstream labels, acknowledges, fixes, or otherwise confirms the issue.

Submit no more than one or two DuckDB reports at a time with the minimized
reproducer, the strict native-match status, and DataDiffFuzz artifact links.

### Highest-Yield Triage Buckets

Start from the 277 rewardable live candidate families in
`reports/final-readiness-20260614T103547.json`.

Prioritize candidates using this order:

1. Backend diversity: DuckDB, Pandas, Arrow/PyArrow, and Polars first, because
   DataFusion already has 5 confirmed families.
2. Reproducibility: prefer families already present in `new_issue/` or
   `new_issue/generated/issue-bundles/manifest.json` with executable
   reproducers.
3. Novelty: exclude `old_issue/` duplicates and any issue closed as invalid or
   duplicate unless maintainers identify a separate accepted root cause.
4. Root-cause clarity: prefer candidates that reduce to one backend and one
   semantic rule rather than broad cross-backend numeric tolerances.
5. Upstream responsiveness: submit polished minimal reproducers to projects
   that recently labeled/fixed existing reports.

### Candidate Families To Inspect First

High-count families from the final 24h report are not automatically bugs, but
they are useful triage queues:

- `drop_nulls_null_filter@duckdb`
- `groupby_aggregation@pandas`
- `semi_join_membership@polars_lazy`
- `semi_join_membership@duckdb`
- `outer_join_truth_filter@pandas`
- `union_all_row_append@duckdb`
- `coalesce_null_semantics@pyarrow`
- `drop_nulls_null_filter@pandas`
- `nan_inf_semantics@duckdb`
- `path_projection_keyed_pick@pandas`
- `union_all_row_append@pyarrow`
- `grouped_topk_null_sort_key@pyarrow`
- `tuple_absence_null_filter@polars_lazy`
- `tuple_absence_null_filter@datafusion`
- `distinct_null_topk@duckdb`
- `groupby_aggregation@duckdb`
- `filter_predicate@pyarrow`
- `groupby_aggregation@pyarrow`
- `anti_join_exclusion@polars_lazy`
- `outer_join_truth_filter@pyarrow`

Be careful with these noisy buckets:

- `running_sum_precision`
- `conditional_expression`
- `nan_inf_semantics`

These can produce real issues, but they often require stronger spec arguments
or tolerance analysis before upstream will accept them.

## Immediate Work Plan

1. Update/read the confirmation registry before every readiness claim:
   `experiments/latest_confirmations.json`.
2. Build a top-30 triage queue from rewardable live families, excluding known
   saturated and old-issue duplicates.
3. For each queue item, produce one native minimal reproducer and one DataDiff
   artifact link.
4. Submit no more than 3-5 polished issues per upstream project at a time.
5. After upstream labels/fixes an issue, add it to
   `experiments/latest_confirmations.json` and rerun final readiness.

## Reporting Guidance

Paper language can now say:

> We currently have 9 confirmed latest-version bug families across DataFusion,
> DuckDB, Polars, and PyArrow, including multiple upstream-completed fixes.

Do not yet claim:

> We found 20+ confirmed bugs.

That claim needs at least 11 more upstream-labeled, maintainer-acknowledged, or
fixed latest-version families.
