# ICSE Submission Gap And Reinforcement Plan

Last updated: 2026-06-15 CST.

Current machine-readable consolidated audit:

- `reports/icse-sota-gap-audit-20260615.json`;
- `reports/icse-sota-gap-audit-20260615.md`;
- generator: `scripts/build_icse_sota_gap_audit.py`.

This audit is the current single-entry summary for metrics gaps, project gaps,
code/pipeline defects, real-evidence gaps, claim adjustments, and the next
reinforcement plan.

## Current State

Final-readiness compliance is green for the current evidence set:

- latest full readiness source: `reports/final-readiness-20260614T103547.json`;
- readiness: `ready: true`;
- live authority evidence: 1580 live runs, 184,459 live cases, 950,434.48 live elapsed seconds;
- run traceability: 2099 required paper-run journal entries covered, 2099 run logs scanned, 0 scan skips;
- latest-version live candidate breadth: 277 rewardable candidate families;
- latest-version confirmations: 9 total in `experiments/latest_confirmations.json`.

The project is not yet in a high-probability ICSE state. The current ICSE quality
summary is grade B, score 85.197, `ready_for_icse_claim=false`. The weakest
dimension is throughput, with average 1.299 cases/s against the aspirational
5 cases/s target. The paper-level evidence gaps are real confirmed bug count,
strict external SOTA comparison, and issue-ready minimized reproducers.

## Defects Fixed In This Reinforcement Pass

This pass fixed five credibility-impacting pipeline issues:

1. Evidence-queue reproducer gap:
   - Before: `scripts/plan_reproducer_queue.py` produced JSON/Markdown only.
   - After: it also writes an executable batch rerunner next to the queue JSON.
   - New artifact: `reports/reproducer-queue-p0-duckdb-live-20260615_reproduce.py`.
   - The queue JSON now carries `case`, `config`, `verification_backends`, and
     `expected_finding_keys`, so reruns validate the selected row artifacts
     directly instead of requiring manual reconstruction.

2. External-baseline evidence strictness:
   - SQLancer summaries now preserve DuckDB target-version alignment from each
     SQLancer manifest.
   - The summary reports strict version-match run counts, version-mismatch run
     counts, and SQLancer reported-failure signals separately.
   - This prevents DuckDB JDBC 1.3.0 support pilots from being accidentally
     interpreted as strict DuckDB 1.5.3 head-to-head evidence.

3. Report output path ambiguity:
   - Several report scripts treated `--output-base reports/name` as relative to
     the default reports directory, producing `reports/reports/name`.
   - The fixed rule is: absolute paths are used as-is; relative paths with a
     directory component are resolved from the project root; bare filenames go
     under the script default output directory.

4. Native SQL actionability gap:
   - Added a DuckDB SQL exporter for relation-only DataDiffFuzz cases.
   - Added `scripts/export_duckdb_queue_sql_reproducers.py` to export validated
     queue rows into native DuckDB SQL scripts.
   - Added `scripts/minimize_duckdb_queue_sql_reproducers.py` to reduce
     validated queue rows, rerun the reduced DataDiff case, execute the
     resulting DuckDB SQL, and record whether the native SQL result strictly
     matches the reduced DuckDB backend result.
   - Current P0 export manifest:
     `reports/duckdb-sql-reproducers/p0-duckdb-live-20260615/manifest.json`.
   - The export manifest records local SQL execution status plus normalized
     DuckDB/reference rows from the evidence queue.
   - Current minimized audit manifest:
     `reports/duckdb-sql-reproducers/p0-duckdb-live-20260615-minimized/manifest.json`.
   - The minimized audit found 4/4 DataDiff-level target reproductions, 4/4
     locally executable SQL scripts, and 2/4 strict native SQL matches to the
     reduced DuckDB backend rerun. The mismatch cases are explicitly marked
     and must not be claimed as SQL-only upstream reproducers.

5. External-baseline long-run observability:
   - Before: `scripts/run_sqlancer_baseline.py` buffered SQLancer stdout in
     memory and wrote the per-run log only after the Java child exited, which
     made long 2h/24h baseline runs hard to monitor and less robust.
   - After: the runner streams stdout to the per-run log file while SQLancer is
     executing, parses summary/failure signals from disk, and terminates the
     Java process group on timeout.
   - Tests cover streamed-log parsing and timeout handling in
     `tests/test_sqlancer_baseline_runner.py`.

## Current SQLancer Baseline Status

SQLancer is the first practical published external SOTA baseline for the
DuckDB/SQL-overlap scope.

Version-aligned strict setup exists:

- root: `experiments/external_tools/sqlancer_duckdb153`;
- SQLancer commit: `30248509ba72`;
- DuckDB JDBC dependency: `1.5.3.0`;
- jar SHA256:
  `58fd71e69d6635eedc707ee0e76fb1ebb10bedeac3d79ee629cdadb6d3af78fe`;
- smoke manifest: `reports/external-baselines/sqlancer-duckdb153-smoke-20260615.json`;
- smoke result: 2/2 successful, strict target version match true.

The version-aligned 2h x 3-seed run is active in tmux session
`sqlancer_fair_duckdb153_2h_20260615`. As of this update it is still executing
the first SQLancer suite, `duckdb-query-partitioning`, and has progressed to
seed `1001`. Do not draw head-to-head conclusions until both SQLancer suites
and all matching DataDiffFuzz seed runs complete and are summarized.

Runtime interpretation: the plan is sequential and uses 7200s per SQLancer
seed, not 7200s for the entire SQLancer suite. With 3 seeds per suite, one
suite can take roughly 6h and the two SQLancer suites can take roughly 12h
before DataDiffFuzz comparison commands begin. On 2026-06-15 17:46 CST the
active process was still on `duckdb-query-partitioning` seed `1001`; this was
consistent with the configured budget.

Runner hardening completed after the active first-suite process launched:
`scripts/run_sqlancer_baseline.py` now streams SQLancer stdout to per-run log
files, parses summary/failure signals from disk, and terminates the Java
process group on timeout. This fix applies to future runner invocations; the
already-running first-suite Python process keeps its launch-time behavior until
it exits.

Use only these metrics for the SQLancer comparison:

- primary: independently confirmed unique latest-version bug families in the
  DuckDB SQL-overlap scope;
- supporting: SQLancer queries/databases/throughput/reported failures and
  DataDiffFuzz cases/candidate cases/rewardable families/case throughput;
- evidence quality: native/minimized reproducer success, immediate recheck
  stability, executable issue bundle, and family-level deduplication.

Do not use raw SQLancer queries versus raw DataDiffFuzz cases as a paper
conclusion.

## Remaining Evidence Gaps

1. Confirmed latest-version bug count:
   - Current: 9 confirmed families.
   - Strong ICSE bar: 12-20 confirmed or maintainer-acknowledged families.
   - Stretch SOTA-like target: 20+.
   - Need at least 3 more for the strong lower bar and 11 more for 20+.

2. P0 DuckDB reproducer conversion:
   - Current queue: `reports/reproducer-queue-p0-duckdb-live-20260615.json`.
   - Queue validation:
     `reports/reproducer-queue-p0-duckdb-live-20260615-validation.json`.
   - Current result: 8 / 8 prioritized rows passed batch rerun validation; all
     8 also have SQLancer/PQS-style witness plans that revalidated on rerun.
   - Native SQL export:
     `reports/duckdb-sql-reproducers/p0-duckdb-live-20260615/manifest.json`.
   - Minimized audit:
     `reports/duckdb-sql-reproducers/p0-duckdb-live-20260615-minimized/manifest.json`.
   - Current minimized result: 4 / 4 families still reproduce the target
     DataDiff finding after reduction; rows reduced 94 -> 11 and operations
     reduced 30 -> 17; all 4 SQL scripts execute locally.
   - Strict native SQL result: 2 / 4 minimized scripts match the reduced DuckDB
     backend output exactly (`grouped_topk_null_sort_key@duckdb` and
     `ordering_or_limit@duckdb`).
   - Issue-ready bundle for those 2 strict native SQL matches:
     `reports/duckdb-issue-ready-bundles/p0-duckdb-live-20260615/manifest.json`;
     tarball:
     `reports/duckdb-issue-ready-bundles/p0-duckdb-live-20260615.tar.gz`.
     The bundle contains upstream issue drafts, reduced SQL reproducers,
     reduced case JSON, evidence JSON, and submission checklists.
   - The other 2 families (`groupby_aggregation@duckdb` and
     `union_all_row_append@duckdb`) are currently DataDiff/DuckDB evidence
     involving the DataFrame/ingestion path and precision-sensitive values;
     they need an ingestion-path reproducer or a reframed upstream report before
     they can be counted as native SQL issue-ready evidence.
   - The minimized audit now re-derives witness plans from reduced rerun output
     and stores pre-reduction plans separately as `source_witness_plan`, so
     issue text can cite the reduced witness row/group without mixing evidence
     scopes.

3. External SOTA comparison:
   - Current: strict SQLancer smoke passed; strict 2h x 3-seed run active.
   - Required before paper claims: completed strict run summary, candidate/failure
     triage under the same confirmation protocol, then scale to 24h x 5-10
     seeds if resource-stable.

4. Throughput:
   - Current average: 1.299 cases/s.
   - Aspirational target: 5 cases/s.
   - Claim impact: report the final throughput honestly; do not claim SOTA-level
     execution speed unless a targeted optimization pass improves it.

5. Adaptive-component claim scope:
   - `champion_graft_donor` and `continual_learning` remain optional/unproven in
     live adaptive evidence.
   - Paper claims should focus on components with live evidence unless these are
     separately proven.

## SQLancer/PQS Method Transfer

SQLancer/PQS's most useful transferable idea is not just SQL AST generation. It
is the pivot-row containment oracle: prove a small witness fact locally, then
check that the target system preserves it. DataDiffFuzz now has a default-off
experimental witness oracle for selected DataFrame/Arrow/SQL workflow facts.

Design note:

```text
docs/sqlancer_pqs_transfer_plan.md
```

Freeze boundary:

- do not add this oracle to the already imported 24h authority evidence;
- do not count witness-assisted reruns as part of the frozen final matrix;
- safe current use is issue triage and reproducer explanation, especially for
  `groupby_aggregation@duckdb` and `union_all_row_append@duckdb`, where native
  SQL execution succeeds but strict native SQL output does not yet match the
  reduced DuckDB backend rerun.

Current integration points:

- `src/datadiff/witness_oracle.py`;
- `scripts/plan_reproducer_queue.py` writes reference-consensus
  `witness_plan` entries;
- generated queue reproducer scripts validate `witness_plan` outcomes during
  batch reruns;
- DuckDB SQL export/minimized export carry witness plans into manifests,
  README evidence, and SQL headers;
- config field `enable_witness_oracle`, default `False`;
- CLI flag `--enable-witness-oracle`;
- preset overlay `enable_witness_oracle`;
- run-log field `witness_oracle`;
- tests:
  `tests/test_witness_oracle.py`,
  `tests/test_witness_oracle_config.py`,
  `tests/test_run_loaded.py`.

Future experiment line:

- `with_witness_oracle` vs `without_witness_oracle`;
- report candidate families/hour, false-positive rate, reducer success,
  issue-ready reproducer success, and overlap with differential/metamorphic
  findings.

## Claim Adjustment

Safe claim:

> DataDiffFuzz provides a reproducible cross-ecosystem semantic differential
> fuzzing harness with paper-grade final-readiness evidence, 24h latest-version
> authority runs, confirmed bugs across DataFusion/DuckDB/Polars/PyArrow, and a
> shared evidence pipeline for reproducer, triage, and submission artifacts.

Unsafe until more evidence exists:

- global outperformance over SQLancer-style DBMS fuzzers;
- SOTA-level throughput;
- 20+ confirmed latest-version bug families;
- live-proven benefit from optional adaptive components without live evidence;
- counting comparison/ablation/SQLancer-support candidates as real live bugs.

## Next Execution Plan

1. Let `sqlancer_fair_duckdb153_2h_20260615` finish without interruption.
2. Summarize the strict SQLancer/DataDiffFuzz 2h x 3-seed comparison with
   `scripts/summarize_sqlancer_fair_comparison.py`.
3. Deduplicate the 2 strict native SQL matches against upstream DuckDB issues,
   then file at most one or two polished reports with minimized SQL and
   DataDiffFuzz evidence links.
4. For the 2 native-SQL mismatch families, build a DuckDB ingestion/DataFrame
   load reproducer or downscope them to DataDiff evidence until the backend path
   is proven independently.
5. Update `experiments/latest_confirmations.json` only after maintainer label,
   acknowledgement, merged fix, or equivalent confirmation.
6. Refresh ICSE quality/readiness reports and update this document with the new
   confirmation count and strict SQLancer comparison result.
