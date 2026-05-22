# DataFusion Upstream Issue Checklist

## Candidate

- Family: `grouped_topk_null_sort_key@datafusion`
- Status: `submitted_upstream_needs_external_confirmation`
- Upstream issue: https://github.com/apache/datafusion/issues/22190
- Submitted: 2026-05-15
- Upstream workflow: open bug; DataFusion member `xiedeyantu` replied `take` on 2026-05-17 UTC and is assigned.
- Primary issue draft: `bugs/bug_9b4d1fa7aac3b391/upstream_issue.md`
- Minimal standalone reproducer: `bugs/bug_9b4d1fa7aac3b391/standalone_datafusion_groupby_null_sortkey_limit.py`
- Boundary preflight script: `scripts/datafusion_topk_null_sortkey_preflight.py`
- Captured boundary output: `reports/datafusion-topk-null-sortkey-preflight.txt`
- Minimal attachment bundle: `reports/datafusion-upstream-issue-bundle.tar.gz`
- Minimal attachment bundle checksum: `reports/datafusion-upstream-issue-bundle.tar.gz.sha256`

## Current Environment

- DataFusion: `53.0.0`
- PyArrow: `24.0.0`
- Python: `3.12.3`
- Platform evidence: `reports/uname-ubuntu.txt`, `reports/lscpu-ubuntu.txt`
- Full dependency capture: `reports/pip-freeze-ubuntu.txt`

## Fresh Validation Results

- Standalone reproducer: reproduces; exits with `AssertionError` because the top-k query returns zero rows.
- Reduced reproducer: reproduces `candidate_implementation_bug`.
- `datadiff triage-artifact`: `candidate_implementation_bug`, paper status `candidate_bug_needs_external_confirmation`.
- `datadiff validate-artifact`: finding still reproduces; root label refined from old `groupby_aggregation` to `grouped_topk_null_sort_key`.
- Pre-issue sanity matrix `runs/experiment-20260514T194522.json`: 3,600 cases, 2,336 candidate findings, 0 false positives, all deduplicated to `grouped_topk_null_sort_key@datafusion`.
- Post-enhancement null-aggregate matrix `runs/experiment-20260514T195546.json`: 3,000 cases, 2,562 candidate findings, 0 false positives, all deduplicated to `grouped_topk_null_sort_key@datafusion`.
- Variant analysis `reports/pattern-variants-null_agg_topk-experiment-20260514T195546.md`: `MIN ASC` and `MAX DESC` produce candidate cases; `MIN DESC` and `MAX ASC` produce zero candidate cases in the same run.
- Local regression suite: `210 passed`.

## Boundary Summary

- Plain NULL column top-k preserves the row.
- NULL group key with non-NULL aggregate preserves the row.
- Grouped `ORDER BY` without `LIMIT` preserves the row.
- NULL `MIN(x)` with ascending `ORDER BY ... LIMIT` drops the row.
- NULL `MAX(x)` with descending `ORDER BY ... LIMIT` drops the row.
- NULL `SUM(x)` and `AVG(x)` top-k controls preserve the row in the current preflight.

## Submission Note

Submitted upstream as https://github.com/apache/datafusion/issues/22190. A DataFusion project member has taken the issue, but has not yet confirmed, rejected, or explained the behavior. Keep the paper-facing status as `submitted_upstream_needs_external_confirmation` until that happens.

The issue draft should avoid claiming that all NULL aggregate sort keys are affected. The current evidence supports a narrower claim: grouped top-k can drop groups whose `MIN`/`MAX` aggregate sort key is NULL, specifically in the order direction that corresponds to the aggregate extremum (`MIN ASC`, `MAX DESC`).
