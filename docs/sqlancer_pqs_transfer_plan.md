# SQLancer/PQS Transfer Plan

Last updated: 2026-06-15 CST.

## Purpose

This document records what DataDiffFuzz can learn from the SQLancer/PQS OSDI
2020 paper and how those ideas are integrated without weakening the current
ICSE evidence boundary.

Source artifact:

```text
papers/related_work/sqlancer_pqs_osdi2020.pdf
```

Paper count policy:

- SQLancer/PQS reported 121 issues.
- The paper classifies 96 as previously unknown true bugs:
  64 SQLite, 24 MySQL, and 8 PostgreSQL.
- True-bug kinds: 61 logic bugs, 32 internal-error bugs, and 3 crash/segfault
  bugs.
- Outcomes: 78 code fixes, 8 documentation fixes, and 10 developer
  confirmations.
- The remaining 25 reports were intended behavior or duplicates.

These are historical SOTA calibration numbers. They are not local SQLancer
DuckDB baseline results.

## Main Transferable Idea

SQLancer/PQS is effective because it uses a cheap partial oracle instead of
trying to compute a full expected query result.

The key abstraction is:

```text
pick pivot row -> generate predicates -> evaluate predicates locally ->
rectify predicates to true for that row -> run query -> require row containment
```

The AST interpreter is deliberately narrow. It interprets generated expressions
on one pivot row, not the full DBMS execution model. This keeps the oracle exact
for the local claim while avoiding the need to implement query planning,
storage, joins, concurrency, or persistence.

## DataDiffFuzz Analogue

The closest DataDiffFuzz analogue is a witness-level oracle:

```text
pick witness row/group -> generate workflow predicates/transforms ->
evaluate a narrow DSL subset locally -> require the witness fact after backend execution
```

A witness fact can be one of:

- row containment: a row that must survive a filter, join, semi-join, or
  projection;
- group containment: a group key that must exist after groupby/aggregation;
- local aggregate fact: a key whose count/min/max/sum can be computed from a
  small selected subset;
- top-k/order prefix fact: a row that must be before or after another row under
  a deterministic tie-broken ordering;
- layout-preservation fact: an Arrow slice/chunk/null-bitmap witness that must
  remain semantically equivalent after conversion or lowering.

This should complement the existing differential and metamorphic oracles. It
should not replace whole-table comparison, because many current DataDiffFuzz
findings depend on cross-backend output differences that are not expressible as
a single witness fact.

## Freeze Boundary

The final 24h authority evidence already imported into the current readiness
set was produced under a frozen generator/oracle/normalizer/guidance/reducer
protocol. Therefore:

- do not add witness-row oracle output to the existing 24h authority bug counts;
- do not rerun selected current findings with a new witness oracle and claim
  they came from the frozen final matrix;
- do not change the current confirmed-family count unless upstream
  confirmation/fix evidence exists under the existing confirmation policy.

Safe uses now:

- use witness facts as an issue-triage helper for current candidate families;
- use witness facts to make minimized reproducers easier to explain;
- add a future `with_witness_oracle` experiment track and compare it against
  the frozen baseline in a new, separately labeled run.

## Proposed Implementation Track

### 1. Artifact Schema

Case artifacts can carry an optional witness contract in `case.metadata`:

```json
{
  "witness_contract": {
    "schema_version": 1,
    "kind": "row_containment",
    "source_table": "t0",
    "row": {"id": 7, "key": "a", "value": 3},
    "required_after_operation": 4,
    "normalization": "datadiff-canonical-row-v1",
    "reason": "filter predicate rectified to true for this row"
  }
}
```

Current artifacts without this field remain valid.

### 2. Narrow Local Evaluator

The default-off implementation lives in:

```text
src/datadiff/witness_oracle.py
```

It currently supports these contract kinds:

- `row_containment`;
- `row_absence`;
- `group_containment`;
- `aggregate_value`.

The module also includes predicate helpers for the same narrow comparator
semantics already used by DataDiffFuzz filters. This is the implemented subset:

- comparisons: `==`, `!=`, `<`, `<=`, `>`, `>=`;
- boolean logic with SQL-style null handling where the contract explicitly
  chooses SQL semantics;
- null, boolean, membership, range, and string filter comparators supported by
  `datadiff.filtering.evaluate_filter_predicate`;
- membership predicates;

Do not start with full workflow execution. The goal is local proof of a witness
fact, not a second DataFrame engine.

### 3. Predicate Rectification

After generating a predicate, evaluate it on the selected witness row. If it is
false or null, rectify it with a small set of transformations. The implemented
first step is intentionally conservative:

- keep predicates that already evaluate to true for the witness row;
- otherwise replace the predicate with equality on the same witness column.

Future extensions can add `NOT` wrapping and disjunction rectification after
null-behavior tests are added for each target semantic model.

Every rectified predicate should carry a reason string in the artifact.

### 4. Backend-Specific Pressure

Use witness contracts to target feature paths that matter for DataDiffFuzz:

- DuckDB: SQL lowering, optimizer rewrites, Arrow/Pandas ingestion, CSV numeric
  inference, order/limit boundaries;
- DataFusion: logical/physical plan rewrites, limit/sort/top-k, null ordering,
  negative zero and numeric edge cases;
- Polars: eager/lazy divergence, streaming boundaries, nullable dtype coercion;
- PyArrow: sliced arrays, chunked arrays, null bitmaps, boolean aggregation;
- Pandas: nullable booleans, object/string coercion, groupby edge cases.

### 5. Reporting And Ablation

Runtime integration:

- config field: `enable_witness_oracle`, default `False`;
- CLI flag: `--enable-witness-oracle`;
- preset overlay: `enable_witness_oracle`;
- run-log field: `witness_oracle`;
- when disabled, witness contracts are evaluated and recorded for triage but
  do not create findings or change row status;
- when enabled, failed witness contracts produce `oracle="witness"` findings
  under the same candidate/confirmation policy as other candidate bugs.
- when enabled and no explicit `case.metadata.witness_contract` exists,
  DataDiffFuzz can infer a conservative `row_containment` witness for
  single-table row-preserving `filter`/`select`/`sort` pipelines only. It skips
  joins, groupby, mutate, limit/offset, and other operations until stronger
  local proof rules are added.

Artifact/reproducer integration:

- `scripts/plan_reproducer_queue.py` now derives a
  `datadiff-reference-witness-plan-v1` witness plan from existing normalized
  queue outputs when reference backends agree and the suspicious backend
  violates the same local fact.
- The generated evidence-queue batch reproducer attaches that witness contract
  to the rerun case and validates both the original finding key and the witness
  outcome.
- DuckDB SQL export and minimized SQL export carry the witness plan into
  manifests, README evidence sections, and SQL header comments.
- Minimized export re-derives the witness plan from the reduced rerun outputs
  and stores the original queue witness as `source_witness_plan`, so reduced
  artifacts do not accidentally cite stale pre-reduction rows or groups.

Current P0 DuckDB artifact status:

- `reports/reproducer-queue-p0-duckdb-live-20260615.json` has 8 / 8 prioritized
  rows with available witness plans.
- `reports/reproducer-queue-p0-duckdb-live-20260615-validation.json` validates
  8 / 8 finding keys and 8 / 8 witness plans.
- `reports/duckdb-sql-reproducers/p0-duckdb-live-20260615-minimized/manifest.json`
  re-derives reduced witness plans for all 4 P0 DuckDB families while preserving
  the existing strict native-SQL result: 4 / 4 DataDiff reproductions, 4 / 4
  local SQL execution, and 2 / 4 strict native SQL matches.

Report witness-oracle evidence as a separate experiment line:

- `with_witness_oracle` vs `without_witness_oracle`;
- time to first candidate family;
- candidate families/hour;
- false-positive or expected-divergence rate;
- reducer success rate;
- issue-ready reproducer success rate;
- overlap with differential/metamorphic findings.

The paper claim should be:

> Inspired by SQLancer/PQS's pivot-row containment oracle, DataDiffFuzz can
> optionally attach witness-level contracts to cross-ecosystem workflows. These
> contracts provide cheap local proof obligations for selected rows, groups, or
> layout facts, while whole-output differential and metamorphic oracles remain
> responsible for the broader cross-backend comparison.

Do not claim this as part of the already completed final 24h authority matrix
unless a new authority run is executed with the feature enabled and clearly
labeled.

## Expected Benefits

- Stronger false-positive control for candidates where one local fact is enough
  to prove the suspicious backend is wrong.
- Smaller issue reproducers because the report can focus on one witness row or
  group.
- Better paper connection to SQLancer/PQS while preserving DataDiffFuzz's
  cross-ecosystem novelty.
- A clean ablation target for future ICSE-strengthening experiments.

## Risks

- Overextending the evaluator can turn into a fragile partial reimplementation
  of DataFrame/SQL semantics.
- SQL, Pandas, Arrow, and Polars null semantics differ; witness contracts must
  state which semantic model is being asserted.
- Witness containment is weaker than whole-output comparison and can miss bugs
  that only appear in aggregate distribution, ordering, or multi-row
  interactions.
- Adding the oracle before a new authority freeze would invalidate comparisons
  against the current frozen 24h evidence unless reported as a separate track.

## Immediate Next Step

Do not modify the frozen final oracle yet. The immediate practical use is to
prototype witness contracts on the two P0 DuckDB native-SQL mismatch families:

- `groupby_aggregation@duckdb`;
- `union_all_row_append@duckdb`.

For these, build ingestion-path or precision-path reproducers that identify the
specific witness row/group whose DataDiffFuzz backend result and native SQL
result diverge. If the witness fact proves the issue is in data loading,
conversion, or numeric representation rather than SQL execution, file the
upstream report with that narrower claim.
