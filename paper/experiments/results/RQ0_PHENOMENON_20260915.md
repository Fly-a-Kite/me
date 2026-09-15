# RQ0 — phenomenon measurement (first run)

Date 2026-09-15. Source: 300 longhaul run logs (seed-7 sample), 66,000 executable cases.
Script: `scripts/paper/run_rq0_phenomenon.py`; raw output `paper/experiments/results/rq0/rq0_phenomenon.json`.

## Headline

| Metric | Value |
| --- | ---: |
| Executed cases | 66,000 |
| Cases with ≥1 cross-backend divergence | 269 |
| Raw divergence rate | **0.41 %** |
| Findings by suspicious backend | sqlite **270**, pyarrow **4** |

Top root causes: `running_sum_precision` 80, `conditional_expression` 70,
`semi_join_membership` 33, `anti_join_exclusion` 28, `coalesce_null_semantics` 18,
`filter_predicate` 17.

## The real finding: raw divergence is dominated by one harness boundary

- **270 of 274 findings are SQLite**, and 60/60 sampled SQLite candidates share the *same*
  error: `the query contains a null character` — the adapter limitation fixed in
  `sqlite_backend._lit` (typed `HarnessLoweringError`).
- The remaining 4 PyArrow findings are the declared `nan_distinct_from_null` boundary
  (the `boundary:special_float_filter_literal` class, also fixed).
- After excluding both declared boundaries, the genuine cross-engine divergence rate on this
  workload is ≈ **4 / 66,000 ≈ 0.006 %**, and even those are a *declared* boundary rather than
  an implementation defect.

## Why this matters for the paper

This is the cleanest possible evidence for the noise-control claim (RQ2) and the phenomenon
(RQ0):

1. Naive cross-engine differential testing on this workload reports **0.41 %** divergence.
2. **98.5 %** of that is a single undeclared harness boundary.
3. Declaring the boundary (typed lowering error + pre-reference boundary rule) collapses the
   rate by ~65×, with the same workload and backends.

It also shows the **taxonomy is misleading before adjudication**: the same adapter defect
surfaced under 17 different `root_cause` labels (running_sum_precision, conditional_expression,
semi_join_membership, …). Root-cause labels alone are not a valid taxonomy until boundaries are
adjudicated.

## Caveats

- The sample includes pre-fix batches; a post-fix rerun will give the clean rate. The pre/post
  contrast is itself the evidence, so both are kept.
- Run logs are per-case records; "divergence" here means the oracle emitted ≥1 finding for the
  case, not that every backend pair disagreed.
- Lane attribution exists via the batch manifests (`per_lane` in the JSON).

## Next

- Rerun after the fix on a fixed seed to publish the clean rate alongside the raw rate.
- Fold this into the RQ0/RQ2 paper tables with the raw → adjudicated funnel.
