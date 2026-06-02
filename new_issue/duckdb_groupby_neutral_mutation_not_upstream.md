# Reviewed Candidate: `metamorphic_groupby_neutral_mutation@duckdb`

## Status

Do not submit upstream.

## Conclusion

This candidate was caused by a DataDiffFuzz bug in metamorphic variant
generation/type propagation, not by a DuckDB engine bug.

## What Happened

The reduced case in `bugs/bug_6106a2d4ad837c90/reduced_case.json` produces a
boolean aggregate column `any_cw_0` and then a later `groupby`.

DataDiffFuzz generated a `groupby_neutral_mutation` variant that inserted:

```text
mr_any_cw_0_plus0 = any_cw_0 + 0
```

DuckDB rejected that variant with a binder error for `+(BOOLEAN,
INTEGER_LITERAL)`, while pandas and SQLite accepted the rewritten expression in
their own execution paths. The original reduced case itself runs successfully on
all three backends.

## Why This Is Not An Upstream DuckDB Bug

`BOOLEAN + 0` is not a semantics-preserving neutral rewrite for DuckDB SQL.
The generated metamorphic relation was invalid for a boolean aggregate source,
so the resulting DuckDB binder error is expected behavior rather than an engine
bug.

The project bug was:

1. `case_when` output type was not propagated in
   `src/datadiff/metamorphic.py`.
2. aggregate result typing in the same file fell back to `"float"` for unknown
   sources.
3. that caused a boolean `any(...)` result to be treated as numeric and fed
   into the `+ 0` neutral-mutation rule.

## Fix

The local fix now:

1. propagates `case_when` output types in
   `src/datadiff/metamorphic.py`
2. uses aggregate-aware output typing for `count`, `nunique`, `any`, `all`,
   and `mean`
3. adds a regression test that prevents `groupby_neutral_mutation` from being
   built over boolean aggregate outputs

## Validation

- `rtk .venv/bin/python -m pytest tests/test_metamorphic.py -q`
- `rtk .venv/bin/python -m pytest tests/test_cli.py tests/test_candidate_pipeline.py tests/test_methodology_report.py tests/test_review_readiness.py -q`
- `rtk .venv/bin/python -m compileall -q src tests`
- replayed `bugs/bug_6106a2d4ad837c90/reduced_case.json`

After the fix, replaying the reduced artifact returns:

```text
status = ok
findings = []
contains_groupby_neutral_mutation = False
```

## Evidence

- Original reduced artifact: `bugs/bug_6106a2d4ad837c90`
- Auto draft that should not be submitted:
  `new_issue/generated/candidate-pipelines/pipeline-bug-sprint-system-fresh-20260531T0010-manifest-20260531T061726/issue-drafts/metamorphic_groupby_neutral_mutation-duckdb-case-00054113-bughunt-fresh-mut-6155.md`
