# DataDiffFuzz Project Idea And Progress

Last updated: 2026-05-27 15:38:57 CST (+0800)

This file is a handoff note for the next Codex session. At the start of a new
session, read this file together with `FINAL_GOAL.md` and
`experiments/final_protocol.md`, then continue from the "Next Actions" section.

## 2026-05-27 Continuation Update

Use `NEXT_SESSION_TODO.md` as the most current operational handoff. The older
PID section below is historical: the six `final-live-depth-required` PIDs listed
there are no longer running.

2026-05-28 update: `datafusion_limit_idempotence@datafusion` is now recorded as
upstream-confirmed latest evidence via
[apache/datafusion#22541](https://github.com/apache/datafusion/issues/22541),
which is open, labeled `bug`, and assigned upstream. The latest confirmation
registry should report six confirmed latest families, not five.

`distinct_null_topk@datafusion` has been submitted upstream as
[apache/datafusion#22554](https://github.com/apache/datafusion/issues/22554)
and is labeled `bug`, so it is now recorded as latest-version confirmation
evidence and added to known/saturated filtering. The submitted issue body
currently has Output and Expected snippets reversed; local evidence remains
correct:

```text
actual DataFusion output: top1: ['']
expected output:          top1: [None]
```

Nullable boolean string pattern expressions were completed as a general
framework extension: `string_starts_with` and `string_ends_with` now mirror
`string_contains` through backend adapters, type validation, generation,
mutation, guidance features, operation-combo risks, oracle/classification,
target capabilities, runner tests, and a short `discovery-campaign` manifest. The sprint
manifest is:

```text
new_issue/generated/discovery-campaign-string-pattern-expr-s10701-manifest.json
```

It reported no fresh, issue-inspired unsaturated, or known saturated candidate
families.

Fill-null/coalesce grouped top-k interactions were also integrated as general
common-API workflow templates:

```text
fill_null_filter_groupby_topk
coalesce_fill_null_groupby_topk
```

They are covered by datagen, operation-combo risk classification, runner
execution tests, and a short discovery-campaign manifest:

```text
new_issue/generated/discovery-campaign-fill-coalesce-interactions-s10839-manifest.json
```

The sprint reported no fresh, issue-inspired unsaturated, or known saturated
candidate families. Because the guided sprint did not deterministically hit the
two new templates, a direct `datadiff fuzz` smoke was run with seeds 39 and 40.
Both templates executed successfully across pandas, PyArrow, Polars eager,
Polars lazy, DuckDB, and SQLite with valid unrepaired preflight and zero
findings:

```text
runs/run-20260527T051022-1779858622531292565.jsonl
```

ISO date/datetime string part extraction was integrated as a conservative
general expression rather than a full DSL type expansion. The new `date_part`
mutate expression extracts `year`, `month`, or `day` from generated ISO date or
datetime strings and returns integer columns. It is wired through backend
adapters, common API templates, mutator, guidance targets/features,
operation-combo risks, oracle/classification type handling, target
capabilities, and tests.

Common API workflow templates:

```text
date_part_groupby
date_part_topk
```

Sprint manifest:

```text
new_issue/generated/discovery-campaign-date-part-expr-s10901-manifest.json
```

The sprint reported no fresh, issue-inspired unsaturated, or known saturated
candidate families. Deterministic full-pipeline smoke runs covered both date
templates across pandas, PyArrow, Polars eager, Polars lazy, DuckDB, SQLite, and
DataFusion with valid unrepaired preflight and zero findings:

```text
runs/run-20260527T052641-1779859601233384287.jsonl
runs/run-20260527T052723-1779859643348834620.jsonl
```

Nullable boolean reductions were broadened as organic common-API workflows,
using the existing `agg:any` and `agg:all` semantics rather than a special
replay probe. The new templates are:

```text
bool_reduction_groupby_topk
bool_reduction_global_summary
```

They exercise nullable boolean any/all with groupby/global aggregation,
post-aggregation boolean filters, sort, projection, and limit. The integration
is wired through common API generation, guidance targets/features,
operation-combo risks, runner tests, and sprint evidence.

Sprint manifest:

```text
new_issue/generated/discovery-campaign-nullable-bool-reduction-s11001-manifest.json
```

The sprint reported no fresh, issue-inspired unsaturated, or known saturated
candidate families. Deterministic full-pipeline smoke covered both templates
across pandas, PyArrow, Polars eager, Polars lazy, DuckDB, SQLite, and
DataFusion with valid unrepaired preflight and zero findings:

```text
runs/run-20260527T054145-1779860505516750583.jsonl
```

Conservative int/float/string cast-boundary support was integrated as a
general common-API workflow extension. Existing numeric-to-float casts were
broadened with safe numeric-string-to-int/float semantics and int-to-string
normalization. Arbitrary string casts are still avoided; string-to-number casts
carry an `input_domain` such as `integer_string` so the generator does not turn
free-form labels into backend-specific parse behavior.

Common API workflow templates:

```text
type_cast_boundary_groupby
type_cast_boundary_topk
```

The integration is wired through backend adapters for pandas, PyArrow, Polars
eager/lazy, DuckDB, SQLite, and DataFusion; common API generation; mutation;
guidance features; operation-combo risks; oracle/classification/metamorphic
type handling; runner tests; and sprint evidence.

Sprint manifest:

```text
new_issue/generated/discovery-campaign-type-cast-boundary-s11101-manifest.json
```

The sprint reported no fresh, issue-inspired unsaturated, or known saturated
candidate families. It executed 180 cases with 180 ok statuses and zero
findings:

```text
runs/run-20260527T061404-1779862444336611298.jsonl.gz
```

Deterministic full-pipeline smoke covered both cast-boundary templates across
pandas, PyArrow, Polars eager, Polars lazy, DuckDB, SQLite, and DataFusion with
valid unrepaired preflight and zero findings:

```text
runs/run-20260527T061128-1779862288896487663.jsonl
```

Latest targeted validation after this extension:

```text
556 passed in 14.64s
compileall passed
git diff --check passed
```

Arrow list/array-like layout coverage was added as a PyArrow-focused probe
rather than a full DSL list scalar type. The new profile/op are:

```text
pyarrow_list_flatten_parent_indices_semantics
list_flatten_parent_indices_probe
```

The probe checks `pyarrow.compute.list_flatten` and
`pyarrow.compute.list_parent_indices` on a nullable list array containing a
null parent, an empty list, and a null child:

```text
input:                 [[1, 2], None, [], [None, 3]]
expected flattened:    [1, 2, None, 3]
expected parent index: [0, 0, 3, 3]
```

Only the PyArrow backend executes the real compute check; pandas, Polars
eager/lazy, DuckDB, SQLite, and DataFusion return the control value `False`.
This keeps the list layout coverage low-noise without expanding the shared DSL
type system to list scalars.

Direct deterministic smoke:

```text
runs/run-20260527T065100-1779864660610553829.jsonl
```

Verification:

```text
case-00000150-pyarrow-list-flatten-parent-indices-semantics:
  ops: list_flatten_parent_indices_probe
  pandas/pyarrow/duckdb: ok
  preflight: valid, unrepaired
  findings: 0
```

Sprint manifest:

```text
new_issue/generated/discovery-campaign-arrow-list-layout-s11201-manifest.json
```

The sprint reported no fresh, issue-inspired unsaturated, or known saturated
candidate families. It executed 180 cases with 178 ok statuses and 2 bug
statuses, both classified as expected semantic divergences for
`grouped_topk_null_sort_key@pandas`; there were zero candidate implementation
bugs. The sprint did not deterministically hit the new probe, so the direct
profile smoke above is the authoritative execution evidence for the probe.

Latest targeted validation after this extension:

```text
574 passed in 14.78s
compileall passed
git diff --check passed
```

Constrained first-token string splitting was integrated as a low-noise common
API expression rather than a broad string-token DSL. The new `string_split_part`
mutate expression is intentionally limited to a non-empty literal separator and
`index=0`; null input propagates null and strings without the separator return
the whole string. This keeps semantics aligned across pandas, PyArrow, Polars
eager/lazy, DuckDB, SQLite, and DataFusion while still exercising common ETL
tokenization, nullable strings, groupby keys, and top-k sort keys.

Common API workflow templates:

```text
string_split_part_groupby
string_split_part_topk
```

The integration is wired through backend adapters for pandas, PyArrow, Polars
eager/lazy, DuckDB, SQLite, and DataFusion; common API generation; mutation;
guidance targets/features; operation-combo risks; oracle/classification type
handling; metamorphic split-first-token idempotence; target capabilities; CLI
target lists; runner tests; and sprint evidence.

Direct deterministic smoke with feedback disabled:

```text
runs/run-20260527T071043-1779865843802063128.jsonl
runs/run-20260527T071441-1779866081677942275.jsonl
```

Verification:

```text
case-00000074-common-api-workflow: string_split_part_groupby
  ops: mutate, filter, groupby, sort
  pandas/pyarrow/polars/polars_lazy/duckdb/sqlite/datafusion: ok
  preflight: valid, unrepaired
  findings: 0

case-00000075-common-api-workflow: string_split_part_topk
  ops: mutate, sort, select, limit
  pandas/pyarrow/polars/polars_lazy/duckdb/sqlite/datafusion: ok
  preflight: valid, unrepaired
  findings: 0

case-00000074-common-api-workflow with metamorphic oracle:
  relation: string_split_part_idempotence:repeat-0
  pandas/pyarrow/polars/polars_lazy/duckdb/sqlite/datafusion: ok
  findings: 0
```

Sprint manifest:

```text
new_issue/generated/discovery-campaign-string-split-part-s11301-manifest.json
```

The sprint reported no fresh, issue-inspired unsaturated, or known saturated
candidate families. It executed 180 common API cases with 180 ok statuses and
zero findings. The sprint did not randomly hit the split templates, so the
direct smoke above is the authoritative template execution evidence.

Latest targeted validation after this extension:

```text
571 passed in 16.11s
9 passed in 0.08s
compileall passed
git diff --check passed
```

Empty-string normalization was integrated as another conservative common API
ETL expression. The new `string_null_if_empty` mutate expression maps the empty
string to null, preserves existing nulls, and leaves all other strings
unchanged. This exercises a common cleaning boundary between empty strings and
nullable strings without broadening into backend-specific string parsing or
collation behavior.

Common API workflow templates:

```text
string_null_if_empty_groupby
string_null_if_empty_topk
```

The integration is wired through backend adapters for pandas, PyArrow, Polars
eager/lazy, DuckDB, SQLite, and DataFusion; common API generation; mutation;
guidance targets/features; operation-combo risks; oracle/classification type
handling; metamorphic null-if-empty idempotence; target capabilities; CLI target
lists; runner tests; and sprint evidence.

Direct deterministic smoke with feedback disabled:

```text
runs/run-20260527T072704-1779866824929397380.jsonl
```

Verification:

```text
case-00000076-common-api-workflow: string_null_if_empty_groupby
  ops: mutate, fill_null, groupby, sort
  pandas/pyarrow/polars/polars_lazy/duckdb/sqlite/datafusion: ok
  preflight: valid, unrepaired
  findings: 0

case-00000077-common-api-workflow: string_null_if_empty_topk
  ops: mutate, sort, select, limit
  pandas/pyarrow/polars/polars_lazy/duckdb/sqlite/datafusion: ok
  preflight: valid, unrepaired
  findings: 0

case-00000076-common-api-workflow with metamorphic oracle:
  relation: string_null_if_empty_idempotence:repeat-0
  pandas/pyarrow/polars/polars_lazy/duckdb/sqlite/datafusion: ok
  findings: 0
```

Metamorphic smoke:

```text
runs/run-20260527T072715-1779866835345417930.jsonl
```

Sprint manifest:

```text
new_issue/generated/discovery-campaign-string-null-if-empty-s11401-manifest.json
```

The sprint reported no fresh, issue-inspired unsaturated, or known saturated
candidate families. It executed 180 common API cases with 180 ok statuses and
zero findings. The sprint did not randomly hit the null-if-empty templates, so
the direct smoke above is the authoritative template execution evidence.

Latest targeted validation after this extension:

```text
577 passed in 16.13s
compileall passed
git diff --check passed
```

Standalone string-length derivation was promoted from incidental mutate
coverage to explicit common API workflow coverage. The `string_length`
expression was already implemented across backends; this continuation added
dedicated templates and scheduling/classification signals for length-based
grouping, null-boundary fill, and top-k ordering.

Common API workflow templates:

```text
string_length_groupby
string_length_topk
```

The integration is wired through common API generation, guidance target/features,
operation-combo risks, CLI target lists, runner tests, and sprint evidence. The
backend semantics continue to use the existing shared `string_length` adapters.

Direct deterministic smoke with feedback disabled:

```text
runs/run-20260527T073711-1779867431235173223.jsonl
```

Verification:

```text
case-00000078-common-api-workflow: string_length_groupby
  ops: mutate, fill_null, groupby, sort
  pandas/pyarrow/polars/polars_lazy/duckdb/sqlite/datafusion: ok
  preflight: valid, unrepaired
  findings: 0

case-00000079-common-api-workflow: string_length_topk
  ops: mutate, sort, select, limit
  pandas/pyarrow/polars/polars_lazy/duckdb/sqlite/datafusion: ok
  preflight: valid, unrepaired
  findings: 0
```

Sprint manifest:

```text
new_issue/generated/discovery-campaign-string-length-s11501-manifest.json
```

The sprint reported no fresh, issue-inspired unsaturated, or known saturated
candidate families. It executed 180 common API cases with 180 ok statuses and
zero findings. The sprint did not randomly hit the string-length templates, so
the direct smoke above is the authoritative template execution evidence.

Latest targeted validation after this extension:

```text
580 passed in 14.94s
compileall passed
git diff --check passed
```

Latest `datadiff bug-status` after this extension:

```text
confirmed latest families: 5
currently unsaturated fresh candidate families: 0
bug workflow manifests: 80
```

## Abstract

DataDiffFuzz is a semantic differential fuzzing framework for DataFrame,
Arrow, embedded SQL, and analytical query engines. The core idea is to generate
one abstract data-processing program, translate it to multiple backends, execute
the translated programs deterministically, normalize the outputs with a
type-aware semantic normalizer, and use differential plus metamorphic oracles to
find reproducible semantic divergences.

The project is not meant to be a collection of one-off bug reproducers. It is a
layered experiment harness:

- Bottom layer: shared DSL, table/program generators, backend adapters,
  execution, semantic normalizer, differential oracle, metamorphic oracle,
  reducer, and artifact writing.
- Middle layer: shared case policy, replay policy, feature extraction,
  feedback guidance, scheduling, reward accounting, candidate classification,
  readiness auditing, and reporting.
- Top layer: target suites, target-specific translation rules, presets,
  issue-inspired generator profiles, historical replay manifests, and final
  experiment plans.

Fresh latest-version discovery and replay/historical reproduction must share
the same bottom and middle layers. The policy split is only:

- `enable_replay_bug=false`: fresh/latest-version mode. Known submitted,
  saturated, or replay-only bug cases are treated as already fixed or already
  known and must not inflate new-bug evidence.
- `enable_replay_bug=true`: replay/historical mode. Known/submitted/historical
  cases are allowed through the same runner, normalizer, oracle, reducer, and
  reporting pipeline for reproducibility evidence.

This structure is intended to support a paper claim that the method is general:
the same middle and bottom layers can test DataFusion, Polars, DuckDB, PyArrow,
and future targets; only the top-layer target adapters and rules change.

## Motivation

Modern data and AI pipelines often move the same logical table operation across
DataFrame APIs, Arrow tables, SQL engines, CSV/Parquet/JSON interchange, and
lazy execution engines. Small semantic differences in null handling, NaN,
negative zero, sorting, top-k, groupby, joins, casts, timestamp precision, CSV
parsing, decimal/large integer handling, or optimizer rewrites can silently
change analytical results.

Existing fuzzing work is strong for compilers, DBMSs, deep learning frameworks,
and language runtimes, but modern DataFrame/Arrow/embedded analytical ecosystems
have a large cross-engine semantic surface that is not covered well by a single
traditional DBMS fuzzer or by API-specific unit tests.

DataDiffFuzz targets that gap by treating data-processing semantics as the unit
under test. It generates constrained programs that can be mapped to several
engines, then distinguishes likely implementation bugs from expected semantic
differences using normalization, classification, replay policy, and
confirmation evidence.

## Experiment Method

The intended final experiment has three evidence tracks.

### 1. Latest-Version Live Discovery

Goal: find new real bugs in the latest available versions of supported
backends.

Protocol:

- Freeze generator, normalizer, oracle, scheduler, reducer, and triage code
  before final 24h live runs.
- Run required target suites for 24h wall-clock depth.
- Keep `enable_replay_bug=false`.
- Exclude known saturated/submitted/replay-only families from rewardable fresh
  evidence.
- Report executed cases, raw findings, unique candidate families, time to first
  candidate, expected semantic divergences, non-reproducible candidates, and
  confirmed/fixed upstream bugs.
- Count bug families as `root_cause + suspicious_backends`; never count raw
  finding rows as bug count.

Required core live suites currently being run:

- `datafusion_cross` with `live_datafusion_fresh`
- `dataframe_lazy` with `live_polars_lazy`
- `arrow_cross` with `live_arrow`
- `embedded_sql` with `live_embedded_sql`
- `latest_all_engines` with `live_cross_family`
- `latest_no_datafusion` with `live_cross_family`

Additional issue-focus suites exist in the final protocol and may be used after
the current required-suite depth run is analyzed.

### 2. Historical / Replay Evidence

Goal: prove the same final code can rediscover known bugs in vulnerable backend
versions.

Protocol:

- Run in isolated vulnerable-version environments.
- Use `enable_replay_bug=true` through replay presets, evidence mode, or
  fixture replay.
- Keep the same bottom and middle layers as fresh live discovery.
- Report separately from fresh latest-version bugs.
- Count only confirmed fixed or maintainer-confirmed historical bugs as
  historical replay evidence.

### 3. Seeded Sensitivity

Goal: validate method sensitivity with controlled injected faults.

Protocol:

- Inject known faults into controlled reference backends or adapters.
- Measure detection rate, cases to first detection, and unique candidate
  families.
- Do not count seeded faults as real backend bugs.

## Experiment Goals

The paper-grade target is not just "find a bug". The project needs evidence for:

- Breadth: multiple targets and target families, including DataFusion, Polars,
  DuckDB, PyArrow/Arrow, and cross-family suites.
- Depth: 24h latest-version live discovery depth for required suites.
- Correctness: bottom-layer oracle/normalizer must avoid known noise such as
  order-sensitive float precision loss, expected null ordering differences, and
  semantic equivalences that should not be called bugs.
- Freshness: fresh mode must not count already submitted or replay-only bugs.
- Reproducibility: candidate artifacts must be replayable, reducible where
  possible, and reportable without referring to local machine paths.
- Confirmation: latest-version bugs become confirmed evidence only after
  maintainer acknowledgement, bug labels, fixes, or other independent upstream
  confirmation.
- Transferability: target-specific code should remain top-layer only; shared
  generation, mutation, scheduling, oracle, reward, and reporting should be
  reusable across backends.

## Current Repository State

As of this update:

- Working directory: `/root/datadiff_fuzz_lab`
- Branch: `main`
- Current commit: `725ab8d Classify negative zero set membership correctly`
- Last pushed commit: `725ab8d`
- `git status --short`: only untracked `rrr` was present before this file was
  added. Treat `rrr` as unrelated user/generated state unless the user says
  otherwise.
- Current untracked files: `PROJECT_IDEA_AND_PROGRESS.md`, `report.md`, and
  `rrr`. Treat `rrr` as unrelated user/generated state unless the user says
  otherwise.

Recent validation before the current live run:

- `pytest tests/test_normalizer.py tests/test_oracle.py tests/test_classification_oracle.py -q`
  passed with `150 passed`.
- `pytest tests/test_regression_findings.py::test_datafusion_negative_zero_truth_filter_is_candidate_bug -q`
  passed.
- `pytest tests/test_runner.py::test_run_loaded_case_flags_duckdb_float_literal_precision_candidate tests/test_runner.py::test_run_loaded_case_flags_polars_timestamp_precision_filter_candidate -q`
  passed.
- `python -m compileall -q src tests` passed.
- `git diff --check` passed.

Do not run the full test suite unless a later code change makes it necessary.
The user explicitly asked not to run full tests during this phase.

## Completed Code Work

Bottom-layer oracle and normalizer noise fixes were completed and committed:

- `src/datadiff/normalizer.py`
  - Added float precision preservation for order-sensitive programs.
  - Avoids collapsing meaningful ordered float output differences into false
    equality.
- `src/datadiff/classification_oracle.py`
  - Reference normalization now preserves float precision for order-sensitive
    programs.
  - Relaxed-equal float arithmetic/order differences are classified as
    `expected_semantic_divergence` instead of fresh candidate bugs.
- `src/datadiff/oracle.py`
  - Added missing `math` import.
  - Negative-zero classification now detects float zero multiplied by any
    negative numeric constant, not only `-1`.
- Tests were added/updated in:
  - `tests/test_normalizer.py`
  - `tests/test_oracle.py`
  - `tests/test_classification_oracle.py`

Artifact rechecks after the fix:

- `bugs/bug_220f412afb63cad0` now classifies as
  `negative_zero_comparison`, still a candidate DataFusion behavior.
- `bugs/bug_8f79b6fad996bd36/reduced_case.json` now triages as
  `expected_semantic_divergence` due float precision/order, so it should not be
  claimed as a DuckDB bug without deeper evidence.

Fresh reward policy fix completed and committed:

- Commit: `5bcd36d Exclude known CSV roundtrip duplicates from fresh reward`
- `src/datadiff/config.py`
  - Added `csv_long_numeric_roundtrip@duckdb` and
    `csv_long_numeric_roundtrip@pyarrow` to
    `DEFAULT_KNOWN_SATURATED_BUG_FAMILIES`.
  - Added `https://github.com/duckdb/duckdb/issues/22750` and
    `https://github.com/apache/arrow/issues/32171` to
    `DEFAULT_REPLAY_BUG_SOURCE_ISSUES`.
- Tests added/updated:
  - `tests/test_reward.py`
  - `tests/test_cli.py`
- Validation after this fix:
  - `pytest tests/test_reward.py tests/test_cli.py::test_cli_parses_live_datafusion_presets tests/test_cli.py::test_cli_parses_non_datafusion_live_presets -q`
    passed with `6 passed`.
  - `python -m compileall -q src tests` passed.
  - `git diff --check` passed.

Negative-zero root-cause classification fix completed and committed:

- Commit: `725ab8d Classify negative zero set membership correctly`
- Reason: reducing `bugs/bug_eb5796eeba5e25ec` showed the apparent
  `join_semantics@datafusion` finding minimized to a known negative-zero
  predicate issue:
  `z = 0.0`, `m_0 = z * -2`, then `m_0 IN (1.0, 0.0, 10.0)`.
  DataFusion returned zero rows while pandas and DuckDB returned one row.
- `src/datadiff/oracle.py`
  - `_filter_comparator_touches_zero` now treats `in_set` literals containing
    numeric zero as a negative-zero-sensitive predicate.
- `tests/test_oracle.py`
  - Added `test_oracle_classifies_negative_zero_set_membership_after_join`.
- Validation after this fix:
  - Negative-zero focused oracle tests passed: `4 passed`.
  - `tests/test_regression_findings.py::test_datafusion_negative_zero_truth_filter_is_candidate_bug`
    and reward known-family focused test passed: `2 passed`.
  - Re-running `bugs/bug_eb5796eeba5e25ec/reproduce_reduced.py` now reports
    `negative_zero_comparison@datafusion`, `known=True`, `rewardable=False`.
  - `python -m compileall -q src tests` passed.
  - `git diff --check` passed.

## Upstream / External Evidence

- User submitted DataFusion issue:
  `https://github.com/apache/datafusion/issues/22190`
- `experiments/latest_confirmations.json` records DataFusion `#22190` as
  externally confirmed/latest evidence.
- DuckDB CSV long numeric roundtrip is not fresh: GitHub search found
  `duckdb/duckdb#22750`, open with label `reproduced`.
- Arrow CSV long numeric / unsafe CSV conversion is not fresh:
  `apache/arrow#32171` is open with labels `Type: bug` and `Component: Python`.
- Local DuckDB issue draft exists at
  `/root/duckdb_csv_long_numeric_roundtrip_issue.md`, but it should not be
  submitted as a new fresh issue unless upstream status changes.

## Current Live Run Status

The first post-normalizer 24h live run from commit `804a4aa` was stopped after
about `10:46:12` because it exposed a reward-policy problem: the reproducible
DuckDB/PyArrow `csv_long_numeric_roundtrip` family was already known upstream
(`duckdb/duckdb#22750` and `apache/arrow#32171`) but was still counted as
fresh rewardable evidence. That run is useful diagnostic evidence, not final
fresh evidence.

After commit `5bcd36d`, the 6 required-suite live runs were restarted, but were
stopped after a few minutes because reducing the old DataFusion
`join_semantics` artifact exposed another classification-noise issue:
negative-zero set-membership after a join was being labeled as `join_semantics`
instead of the known/saturated `negative_zero_comparison`.

After commit `725ab8d`, the 6 required-suite 24h live runs were restarted again.
These are the current candidates for 24h depth evidence.

PID manifest:

```text
runs/final-live-depth-required.latest
```

The latest pointer currently resolves to:

```text
/root/datadiff_fuzz_lab/runs/final-live-depth-required-20260524T202025Z.pids
```

At `2026-05-25 15:20:21 CST`, the latest pointer resolves to:

```text
/root/datadiff_fuzz_lab/runs/final-live-depth-required-20260525T072821Z.pids
```

All 6 current processes were running, with elapsed time about `00:30`.

Current live processes:

```text
datafusion_cross      live_datafusion_fresh  PID 3283425
dataframe_lazy        live_polars_lazy       PID 3283426
arrow_cross           live_arrow             PID 3283427
embedded_sql          live_embedded_sql      PID 3283428
latest_all_engines    live_cross_family      PID 3283429
latest_no_datafusion  live_cross_family      PID 3283430
```

Run JSONL gzip logs found through open file descriptors:

```text
Current run JSONL gzip logs use the prefix:

```text
runs/run-20260525T072821-*.jsonl.gz
```

These gzip files are still being written. Parsing to EOF usually raises
`EOFError`; treat that as expected and use completed records only.

## Latest Parsed Live Summary

Parsed from the stopped/discarded `804a4aa` run. This section is diagnostic
only. It must not be used as final fresh 24h evidence because the run was
stopped before 24h and its reward policy still counted known CSV roundtrip
duplicates as fresh reward.

### `datafusion_cross` / `live_datafusion_fresh`

- Rows parsed: `98110`
- Statuses: `ok=96980`, `bug=1130`
- Rewardable families seen:
  - `join_semantics@datafusion`: `46`
  - `filter_predicate@datafusion`: `31`
- Known/saturated families seen:
  - `negative_zero_comparison@datafusion`: `6`
  - `tuple_absence_null_filter@duckdb`: `248`
  - `joined_order_offset_projection@datafusion`: `89`
  - `ordered_topk_projection@datafusion`: `46`
  - `outer_join_truth_filter@datafusion`: `21`
  - `tuple_absence_null_filter@datafusion,duckdb`: `17`
  - `topk_filter_pushdown@datafusion`: `17`
- Verdicts:
  - `candidate_implementation_bug=521`
  - `expected_semantic_divergence=609`
- Example:
  - `join_semantics@datafusion`: `/root/datadiff_fuzz_lab/bugs/bug_eb5796eeba5e25ec`
  - `filter_predicate@datafusion`: no artifact directory recorded in this
    partial parse.

### `dataframe_lazy` / `live_polars_lazy`

- Rows parsed: `225830`
- Statuses: `ok=223918`, `bug=1912`
- Rewardable families seen:
  - `grouped_topk_null_sort_key@polars_lazy`: `3`
- Verdicts:
  - `expected_semantic_divergence=1909`
  - `candidate_implementation_bug=3`
- Example:
  - `grouped_topk_null_sort_key@polars_lazy`: no artifact directory recorded in
    this partial parse.

### `arrow_cross` / `live_arrow`

- Rows parsed: `174917`
- Statuses: `ok=169028`, `bug=5889`
- Rewardable families seen:
  - `csv_long_numeric_roundtrip@duckdb,pyarrow`: `4`
  - `grouped_topk_null_sort_key@pyarrow`: `343`
  - `grouped_topk_null_sort_key@duckdb,pyarrow`: `2`
  - `grouped_topk_null_sort_key@duckdb`: `20`
  - `joined_order_offset_projection@pyarrow`: `12`
  - `topk_filter_pushdown@pyarrow`: `28`
  - `groupby_aggregation@pyarrow`: `4`
- Known/saturated families seen:
  - `tuple_absence_null_filter@duckdb,pyarrow`: `33`
  - `tuple_absence_null_filter@duckdb`: `93`
- Verdicts:
  - `candidate_implementation_bug=539`
  - `expected_semantic_divergence=5350`
- Example:
  - `csv_long_numeric_roundtrip@duckdb,pyarrow`:
    `/root/datadiff_fuzz_lab/bugs/bug_f813d694c48c0b6b`
  - Other rewardable families in this partial parse often have no bug directory
    yet and need later artifact/recheck triage.

### `embedded_sql` / `live_embedded_sql`

- Rows parsed: `161701`
- Statuses: `ok=155358`, `bug=6343`
- Rewardable families seen:
  - `csv_long_numeric_roundtrip@duckdb`: `4`
  - `grouped_topk_null_sort_key@duckdb`: `268`
  - `reverse_division_operand_order@duckdb`: `4`
- Verdicts:
  - `candidate_implementation_bug=276`
  - `expected_semantic_divergence=6067`
- Example:
  - `csv_long_numeric_roundtrip@duckdb`:
    `/root/datadiff_fuzz_lab/bugs/bug_af8798034add718a`

### `latest_all_engines` / `live_cross_family`

- Rows parsed: `185217`
- Statuses: `ok=177141`, `bug=8076`
- Rewardable families seen:
  - `csv_long_numeric_roundtrip@duckdb,pyarrow`: `4`
  - `grouped_topk_null_sort_key@duckdb`: `1`
  - `grouped_topk_null_sort_key@pyarrow`: `1`
  - `grouped_topk_null_sort_key@duckdb,polars,polars_lazy`: `2`
- Known/saturated families include DataFusion grouped-topk, joined-order-offset,
  outer-join truth filter, groupby aggregation, ordered top-k projection,
  negative zero, and tuple absence null filters.
- Verdicts:
  - `candidate_implementation_bug=1030`
  - `expected_semantic_divergence=7046`
  - `non_reproducible_candidate=58`
- Example:
  - `csv_long_numeric_roundtrip@duckdb,pyarrow`:
    `/root/datadiff_fuzz_lab/bugs/bug_1d1ead3177b08fd2`

### `latest_no_datafusion` / `live_cross_family`

- Rows parsed: `154028`
- Statuses: `ok=145948`, `bug=8080`
- Rewardable families seen:
  - `csv_long_numeric_roundtrip@duckdb,pyarrow`: `4`
  - `grouped_topk_null_sort_key@duckdb,pyarrow`: `65`
  - `grouped_topk_null_sort_key@duckdb`: `2`
  - `joined_order_offset_projection@pyarrow`: `2`
  - `joined_order_offset_projection@duckdb,polars,polars_lazy,pyarrow`: `2`
  - `topk_filter_pushdown@pyarrow`: `4`
  - `grouped_topk_null_sort_key@pyarrow`: `3`
  - `grouped_topk_null_sort_key@polars,polars_lazy`: `6`
  - `ordered_topk_projection@pyarrow`: `3`
- Verdicts:
  - `expected_semantic_divergence=7989`
  - `candidate_implementation_bug=91`
- Example:
  - `csv_long_numeric_roundtrip@duckdb,pyarrow`:
    `/root/datadiff_fuzz_lab/bugs/bug_7f276715138bc583`

## Current Interpretation

The project is past the "fix obvious oracle noise" phase and is currently in a
fresh 24h required-suite live discovery run. The live run has not reached the
24h depth target yet, so it is too early to claim final A-conference readiness.

Early signals:

- DataFusion still shows rewardable `join_semantics` and `filter_predicate`
  families after the noise fix. These need reduction and upstream-duplication
  triage after enough artifacts are available.
- Polars lazy has a small `grouped_topk_null_sort_key@polars_lazy` signal.
- DuckDB/PyArrow CSV long numeric roundtrip is reproducible, but not fresh:
  it overlaps `duckdb/duckdb#22750` and `apache/arrow#32171`, and is now
  excluded from rewardable fresh evidence.
- After excluding CSV roundtrip duplicates, the remaining interesting signals
  should be treated carefully. The old DataFusion `join_semantics` artifact
  `bugs/bug_eb5796eeba5e25ec` was reduced and reclassified as known
  `negative_zero_comparison@datafusion`, not fresh `join_semantics`.
  Remaining future signals of interest are Polars lazy grouped top-k/null-sort,
  DuckDB/PyArrow grouped top-k/null-sort, PyArrow order/filter/groupby, and any
  DataFusion join/filter family that still survives the `725ab8d` classifier.
- A large number of findings are now classified as
  `expected_semantic_divergence`, which suggests the recent normalizer/oracle
  fixes are filtering noise instead of blindly producing candidate bugs.

Do not modify generator/oracle/scheduler code while the current 24h evidence
run is intended to count as a frozen final run. If a new code change is truly
needed, stop the 6 live processes first, implement and test the change, commit
it, then restart the required 24h run from the new commit.
