# Ubuntu DataFusion Candidate Bug Confirmation

## Environment

- Python: 3.12.3
- DataFusion: 53.0.0
- PyArrow: 24.0.0
- Polars: 1.40.1
- Environment evidence:
  - `reports/pip-freeze-ubuntu.txt`
  - `reports/uname-ubuntu.txt`
  - `reports/lscpu-ubuntu.txt`
  - `reports/target-registry-ubuntu.json`

## Sanity Checks

- `pytest -q`: 210 passed after adding regression, CLI triage, manifest, bundle, seeded-sensitivity, ablation-audit, top-k boundary, pattern-variant, and upstream-issue bundle coverage.
- `python -m compileall -q src tests scripts`: passed.
- `git diff --check`: passed.

## Standalone Reproduction

- DataFusion reproducer: `bugs/bug_9b4d1fa7aac3b391/standalone_datafusion_groupby_null_sortkey_limit.py`
- Result: reproduces on Ubuntu. The query drops the grouped row when a `MIN(x)` aggregate sort key is NULL and an ascending top-k limit is applied.
- Polars lazy float group-key reproducer: `bugs/bug_0f37902bbb377813/standalone_polars_lazy_float_groupby_instability.py`
- Result: not reproduced with Polars 1.40.1.

## Upstream Submission

- Issue: https://github.com/apache/datafusion/issues/22190
- Submitted: 2026-05-15
- Project member response: `xiedeyantu` replied `take` on 2026-05-17 UTC and is assigned to the issue.
- Current paper-facing status: `submitted_upstream_needs_external_confirmation`
- Interpretation: this is now under upstream member triage. Keep counting it as one high-confidence submitted candidate bug family until upstream confirms, rejects, or documents the behavior.

## Main Matrix

- Manifest: `runs/experiment-20260514T175130.json`
- Summary: `reports/experiment-summary-experiment-20260514T175130.md`
- Analysis: `reports/experiment-analysis-experiment-20260514T175130.md`
- Cases: 60,000
- Findings: 13,503
- Candidate implementation bugs: 13,503
- Oracle false positives: 0
- Deduplicated candidate bug family: `grouped_topk_null_sort_key@datafusion`

## Confirmation Matrix

- Manifest: `runs/experiment-20260514T180106.json`
- Summary: `reports/experiment-summary-experiment-20260514T180106.md`
- Analysis: `reports/experiment-analysis-experiment-20260514T180106.md`
- Cases: 10,000
- Findings: 6,744
- Candidate implementation bugs: 6,744
- Oracle false positives: 0
- Deduplicated candidate bug family: `grouped_topk_null_sort_key@datafusion`

## Independent-Seed Domain Matrix

- Manifest: `runs/experiment-20260514T192105.json`
- Summary: `reports/experiment-summary-experiment-20260514T192105.md`
- Analysis: `reports/experiment-analysis-experiment-20260514T192105.md`
- Aggregate CSV: `reports/experiment-summary-experiment-20260514T192105-aggregates.csv`
- Seeds: `5001`, `6001`, `7001`, `8001`, `9001`
- Cases: 60,000
- Findings: 13,601
- Candidate implementation bugs: 13,601
- Oracle false positives: 0
- Semantic divergences: 0
- Deduplicated candidate bug family: `grouped_topk_null_sort_key@datafusion`
- `core_lazy` produced zero candidates across the same domain presets and seeds.

## Pre-Issue Sanity Matrix

- Manifest: `runs/experiment-20260514T194522.json`
- Summary: `reports/experiment-summary-experiment-20260514T194522.md`
- Analysis: `reports/experiment-analysis-experiment-20260514T194522.md`
- Aggregate CSV: `reports/experiment-summary-experiment-20260514T194522-aggregates.csv`
- Seeds: `12001`, `13001`, `14001`
- Cases: 3,600
- Findings: 2,336
- Candidate implementation bugs: 2,336
- Oracle false positives: 0
- Deduplicated candidate bug family: `grouped_topk_null_sort_key@datafusion`
- Purpose: quick fresh-seed validation before preparing an upstream issue.

## Post-Enhancement Null-Aggregate Matrix

- Manifest: `runs/experiment-20260514T195546.json`
- Summary: `reports/experiment-summary-experiment-20260514T195546.md`
- Analysis: `reports/experiment-analysis-experiment-20260514T195546.md`
- Aggregate CSV: `reports/experiment-summary-experiment-20260514T195546-aggregates.csv`
- Seeds: `15001`, `16001`, `17001`
- Cases: 3,000
- Findings: 2,562
- Candidate implementation bugs: 2,562
- Oracle false positives: 0
- Deduplicated candidate bug family: `grouped_topk_null_sort_key@datafusion`
- Purpose: sanity check after extending `null_agg_topk` generation to cover both NULL `MIN(x)` ascending top-k and NULL `MAX(x)` descending top-k.
- Variant analysis: `reports/pattern-variants-null_agg_topk-experiment-20260514T195546.md`
- Variant result: `MIN ASC` and `MAX DESC` account for all candidate cases; `MIN DESC` and `MAX ASC` produce zero candidate cases in the same run.

## Top-K Boundary Preflight

- Script: `scripts/datafusion_topk_null_sortkey_preflight.py`
- Result: plain NULL column top-k, NULL group-key with non-NULL aggregate, plain grouped query, and grouped `ORDER BY` without `LIMIT` all preserve the row.
- Failing variants include NULL `MIN(x)` ordered ascending with `ASC NULLS LAST`, `ASC` default, `ASC NULLS LAST LIMIT 1`, `ASC NULLS FIRST`, and a two-group query whose `LIMIT 20` should preserve both groups.
- Failing variants also include NULL `MAX(x)` ordered descending with `DESC NULLS LAST`.
- Passing variants include NULL `MIN(x)` ordered descending, NULL `MAX(x)` ordered ascending, grouped `SUM(x)`/`AVG(x)` NULL aggregate top-k, and a two-group `LIMIT 1` query where the non-NULL row is expected to be the only returned row.

## Post-Triage Health Check

- Manifest: `runs/experiment-20260514T181521.json`
- Summary: `reports/experiment-summary-experiment-20260514T181521.md`
- Analysis: `reports/experiment-analysis-experiment-20260514T181521.md`
- Cases: 100
- Findings: 75
- Candidate implementation bugs: 75
- Oracle false positives: 0
- Deduplicated candidate bug family: `grouped_topk_null_sort_key@datafusion`
- Purpose: quick regression check after adding automated triage and standalone-reproducer support.

## Generalization and Ablation Matrix

- Manifest: `runs/experiment-20260514T190631.json`
- Summary: `reports/experiment-summary-experiment-20260514T190631.md`
- Analysis: `reports/experiment-analysis-experiment-20260514T190631.md`
- Aggregate CSV: `reports/experiment-summary-experiment-20260514T190631-aggregates.csv`
- Suites: `core`, `core_lazy`, `core_datafusion`, `core_arrow`, `datafusion_cross`, `arrow_cross`
- Presets: `baseline`, `guided`, `no_feedback`, `no_type_aware`, `no_normalizer`, `metamorphic`
- Seeds: `1`, `1001`, `2001`, `3001`, `4001`
- Cases: 90,000
- Findings: 4,322
- Candidate implementation-bug cases: 61
- Oracle false positives: 4,128, all from `no_normalizer`
- Default-style presets `baseline`, `guided`, `no_feedback`, and `metamorphic` produced zero oracle false positives; their only candidate cases were the known DataFusion NULL aggregate top-k family in DataFusion suites.
- Ablation audit: `reports/ablation-audit-experiment-20260514T190631.md`
- Audit verdict: all new candidate families introduced only by `no_type_aware` are excluded from bug counts until separately triaged; only `grouped_topk_null_sort_key@datafusion` is detected by both trusted and ablation presets.

## Non-DataFusion Negative Control

- Manifest: `runs/experiment-20260514T193946.json`
- Summary: `reports/experiment-summary-experiment-20260514T193946.md`
- Analysis: `reports/experiment-analysis-experiment-20260514T193946.md`
- Aggregate CSV: `reports/experiment-summary-experiment-20260514T193946-aggregates.csv`
- Suites: `core`, `core_lazy`, `core_arrow`, `arrow_cross`
- Presets: `bughunt_no_groupby`, `bughunt_no_groupby_guided`, `bughunt_no_groupby_metamorphic`
- Cases: 36,000
- Findings: 0
- Candidate implementation bugs: 0
- Oracle false positives: 0
- Purpose: negative control showing that trusted non-DataFusion, no-groupby bug-hunting presets do not manufacture spurious findings.

## Regression Evidence Test

- Test: `tests/test_regression_findings.py::test_datafusion_grouped_topk_null_sort_key_is_candidate_bug`
- Minimal case: one group with `MIN(x) = NULL`, followed by `ORDER BY min_x ASC NULLS LAST LIMIT 20`.
- Expected comparison engines: pandas and DuckDB return the grouped row.
- Observed suspicious backend: DataFusion returns an empty result.
- Triage verdict: `candidate_implementation_bug`.
- Paper status: `submitted_upstream_needs_external_confirmation`; upstream member `xiedeyantu` has taken the issue.
- Additional test: `tests/test_regression_findings.py::test_datafusion_grouped_topk_null_max_sort_key_is_candidate_bug`
- Additional minimal case: one group with `MAX(x) = NULL`, followed by `ORDER BY max_x DESC LIMIT 20`.

## Automated Triage Artifact

- Command: `datadiff triage-artifact --bug bugs/bug_9b4d1fa7aac3b391 --backends pandas,duckdb,datafusion --standalone-reproducer`
- Triage JSON: `bugs/bug_9b4d1fa7aac3b391/triage.json`
- Triage Markdown: `bugs/bug_9b4d1fa7aac3b391/triage.md`
- Reduced case: `bugs/bug_9b4d1fa7aac3b391/reduced_case.json`
- Reduced reproducer: `bugs/bug_9b4d1fa7aac3b391/reproduce_reduced.py`
- Generated standalone reproducer: `bugs/bug_9b4d1fa7aac3b391/standalone_datafusion_groupby_null_sortkey_limit.py`
- Generated reproducer result: prints the control row, prints an empty top-k result, then raises `AssertionError` because DataFusion dropped the NULL aggregate sort-key group.
- Upstream issue draft: `bugs/bug_9b4d1fa7aac3b391/upstream_issue.md`
- Artifact manifest: `reports/ubuntu-datafusion-artifact-manifest.json`
- Artifact bundle: `reports/ubuntu-datafusion-artifact-bundle.tar.gz`
- Artifact bundle SHA256: `reports/ubuntu-datafusion-artifact-bundle.tar.gz.sha256`

## Implementation Note

The DataFusion backend now imports PyArrow once per backend run instead of once per table conversion. This keeps behavior unchanged while removing repeated import work during multi-table cases.
The triage tool now also recognizes `grouped_topk_null_sort_key` as a root cause with a standalone DataFusion reproducer template.
