# ICSE SOTA Success Criteria

Last updated: 2026-06-15 CST.

This document records the paper-grade bar for pushing DataDiffFuzz from
`final-readiness` compliance toward a high-probability ICSE submission. It is a
handoff document: future sessions should update it whenever readiness, bug
triage, or SOTA baseline experiments change.

Current consolidated status artifact:

- `reports/icse-sota-gap-audit-20260615.json`;
- `reports/icse-sota-gap-audit-20260615.md`;
- generator: `scripts/build_icse_sota_gap_audit.py`.

Use that artifact as the current machine-readable gap snapshot before updating
paper claims or declaring high-probability submission readiness.

## SOTA Papers Used As Calibration

The local related-work PDFs under `papers/related_work/` were converted to text
and checked for evaluation scale, bug evidence, and comparison design.

| Work | Venue | Relevant standard to match or explain |
| --- | --- | --- |
| SQLancer/PQS | OSDI 2020 | Reported 121 issues and classified 96 as previously unknown true bugs: 64 SQLite, 24 MySQL, 8 PostgreSQL; 61 logic, 32 internal-error, and 3 crash bugs; 78 code fixes, 8 documentation fixes, and 10 developer confirmations. |
| SQLancer/NoREC | FSE 2020 | Reported 168 issues; 159 were previously unknown true bugs; 141 addressed by code changes; 14 verified but not yet addressed; 9 false bugs. |
| SQLancer/TLP | OOPSLA 2020 | Evaluation on 6 DBMSs; detected 77 logic bugs, at least 17 outside existing techniques. |
| SQUIRREL | CCS 2020 | 63 DBMS bugs across SQLite/MySQL/MariaDB; 52 fixed at paper time; compares with five SOTA fuzzers over 24h repeated runs and reports unique crashes, unique bugs, new edges, syntax correctness, and semantic correctness. |
| SQLancer+QPG | ICSE 2023 | 53 unique previously unknown bugs, all confirmed, 35 fixed; compares against SQLancer and SQLRight with 24h x 10-run experiments; reports bug-finding speed, unique plans, coverage, and ablations. |
| SQLRight | USENIX Security 2022 | Useful SOTA baseline for coverage-guided SQL logic testing; relevant mainly for SQL-only overlap, not for DataFrame/Arrow claims. |
| SQLBull | NDSS 2026 | 63 zero-day bugs across MySQL/MariaDB/CockroachDB/DuckDB/PostgreSQL; 24h per tool, 10 repetitions, latest versions at evaluation time, multiple SOTA baselines. |
| SparkFuzz | DBTest 2020 | Over a year, found >10 analysis errors, >10 runtime crashes, and >20 wrong results; demonstrates correctness testing with 500 generated SQL queries on Spark/PostgreSQL. |
| FuzzyData | DBTest 2022 | DataFrame workflow generation/replay baseline; found one Modin correctness bug and performance issues. Strong related work but not a direct bug-finding SOTA baseline. |

SQLancer/PQS count source and policy:

- Local paper artifact: `papers/related_work/sqlancer_pqs_osdi2020.pdf`.
- Paper sections used: Section 4.2, Table 2, and Table 3.
- Use the 96 true bugs as the historical SOTA calibration number.
- Do not treat the 121 reported issues as confirmed bugs; 25 reports were
  later classified as intended behavior or duplicates.
- Do not treat SQLancer/PQS historical bugs as local DuckDB baseline results.
  Local SQLancer counts in this repository come only from the executed
  SQLancer manifests and the shared confirmation protocol.

## High-Success Submission Bar

DataDiffFuzz should not try to claim it simply outperforms SQLancer-style tools
globally. The defensible high-success target is:

1. Final evidence audit is green.
   - `datadiff final-readiness --full-run-log-scan --fail-on-missing` must
     produce `ready: true`.
   - The manifest index, paper run journal, target-version audit, frozen
     provenance, and full run-log scan must be reproducible from local files.

2. Latest-version real-bug evidence is strong enough.
   - Minimum paper-submission bar: at least 8 confirmed latest-version bug
     families spanning at least 3 backend families.
   - Strong ICSE bar: 12-20 confirmed or maintainer-acknowledged latest-version
     bug families, with at least 2-4 polished case studies.
   - Stretch SOTA-like bar: 20+ confirmed or fixed latest-version bug families.
   - Candidate families are not enough. Each counted family needs a minimized
     reproducer, false-positive exclusion, suspicious backend, root-cause label,
     and upstream issue/PR or maintainer confirmation.

3. Historical and seeded evidence supports method validity.
   - Historical replay should keep at least 2 confirmed fixed historical bugs,
     and ideally expand to 4-6 across at least two backend families.
   - Seeded sensitivity should stay clearly separated from real bugs and report
     detection rate, false positives, and time/cases to detection.

4. Comparison experiments should be broad and controlled.
   - Keep internal module ablation and scope comparison as the main evaluation:
     these are fair because they share the same harness, targets, oracle, and
     artifact pipeline.
   - Required comparison tables:
     - module ablation;
     - adaptive component ablation;
     - SQL/query-engine-oriented scope vs cross-ecosystem scope;
     - oracle complementarity;
     - actionability/evidence pipeline.
   - For external SOTA, run only scope-limited overlap experiments:
     SQLancer on DuckDB/DataFusion/SQLite-like SQL targets is fair; SQLancer is
     not a fair baseline for Pandas/Polars/PyArrow API and Arrow layout claims.

5. External SOTA baseline evidence should be included but carefully scoped.
   - Preferred first baseline: SQLancer mainline, because it is public, mature,
     and supports several relevant SQL oracles/providers.
   - First executable target: DuckDB or SQLite smoke run, then DataFusion if the
     provider/build path is practical.
   - Report commands, version, run time, seeds, target DBMS, oracle, raw failures,
     unique failures, and whether failures overlap with DataDiffFuzz findings.
   - Do not import external baseline candidates into latest-version real bug
     counts unless they pass the same triage/confirmation protocol.

6. Performance and workflow metrics should match SOTA evaluation style.
   - Use 24h authority runs for the final live matrix.
   - Use repeated runs for comparison/baseline claims where feasible. SOTA DBMS
     fuzzing papers often use 5-10 repeats; if resources are limited, report the
     resource gap explicitly and use confidence language.
   - Include cases/sec, time-to-first-family, families/hour, valid-case rate,
     false-positive rate, run-log/artifact bytes per case, reduction success,
     and issue-readiness status.

## Current DataDiffFuzz Status

Source report: `reports/final-readiness-20260614T103547.json`.

Current confirmation registry:
`experiments/latest_confirmations.json`.

Confirmed bug quality and 20+ expansion plan:
`docs/confirmed_bug_quality_and_20plus_plan.md`.

Machine-ranked 20+ triage queue:
`reports/bug-20plus-triage-plan-20260614.json`.

### Achieved

- 24h latest-version live authority matrix completed and imported.
- Live coverage:
  - 1580 live runs;
  - 8 live suites: `arrow_cross`, `dataframe_lazy`, `datafusion_cross`,
    `embedded_sql`, `embedded_sql_cross`, `latest_all_engines`,
    `latest_no_datafusion`, `polars_cross`;
  - 4 target families: `arrow`, `dataframe`, `embedded_sql`, `query_engine`;
  - 184,459 live cases;
  - 950,434.48 live elapsed seconds.
- Candidate-family breadth:
  - 277 rewardable latest-version live candidate families.
  - 9 confirmed latest-version bug families:
    - `datafusion_limit_idempotence@datafusion`
    - `distinct_null_topk@datafusion`
    - `groupby_aggregation@datafusion`
    - `groupby_aggregation@polars,polars_lazy`
    - `grouped_topk_null_sort_key@datafusion`
    - `metamorphic_semi_anti_join_rewrite@duckdb`
    - `negative_zero_comparison@datafusion`
    - `polars_reflected_arithmetic_operand_order@polars`
    - `pyarrow_sliced_bool_groupby_any_all@pyarrow`
- Traceability:
  - 2099 required paper-run journal entries covered;
  - 2099 run logs scanned;
  - 0 run-log scan skips.
- Comparison/ablation support:
  - baseline/scope comparison: 116 runs, 4 complete groups, 12 reference runs,
    96 contrast runs;
  - module ablation: 105 runs, 5 complete groups, 15 reference runs, 90 contrast
    runs;
  - adaptive component ablation: 54 runs, 27 disabled components.

### Current Blockers

Final readiness is now `ready: true` with 0 failed gates. The previous
`runtime_efficiency`, `adaptive_live_component_evidence`, and
`baseline_comparison` blockers were fixed and verified by a full run-log scan.

Remaining ICSE high-success blockers are paper-level evidence gaps:

1. Confirmed latest-version bug count is 9; the target is 20+.
2. External SOTA baseline strict run is active but not yet complete or
   summarized.
3. Throughput is adequate for readiness but still below the aspirational ICSE
   quality target: average 1.299 cases/s vs target 5 cases/s.
4. `champion_graft_donor` and `continual_learning` remain optional/unproven in
   live adaptive evidence and should not be claimed as live-proven components.

## Gap Against SOTA-Level Expectations

| Dimension | Current | High-success target | Gap |
| --- | --- | --- | --- |
| Final readiness | `ready: true`, 0 failed gates | `ready: true` | Done for the current evidence set. |
| Confirmed latest-version families | 9 | 12-20 strong, 20+ stretch | Need at least 11 more confirmed or fixed families. |
| Backend-family spread | DataFusion/DuckDB/Polars/PyArrow confirmed; many candidates elsewhere | Confirmed families across at least 3-4 backend families | Already spans 4 backend families; now improve balance and add Pandas/Arrow/DuckDB diversity. |
| External SOTA baseline | SQLancer build/smoke executed; strict DuckDB 1.5.3 SQLancer run active | 2h pilot, then 24h x 5-10 seeds on DuckDB SQL overlap | Need strict run completion, final summary, and final triage/counting table. |
| Repeated comparison runs | Internal support exists | 5-10 repeats for claims where resources permit | Current final comparison has seeds/groups, but external baseline repeats not yet done. |
| Issue-ready evidence | Candidate volume high; P0 DuckDB live row evidence extracted; queue validation has 8/8 finding and witness revalidations; minimized DuckDB audit has 4/4 DataDiff reproductions and 2/4 strict native SQL matches; issue-ready bundle generated for the 2 strict matches | 2-4 polished case studies plus minimized reproducers | Need upstream dedup/submission for strict native matches and ingestion-path reproducers for the precision/load-path cases. |
| Paper claims | Broad method scope is promising | Claims must match actual evidence | Downscope `champion_graft_donor`/`continual_learning` unless live evidence is added. |

Historical SOTA calibration gap:

- SQLancer/PQS reached 96 true bugs across SQLite, MySQL, and PostgreSQL in its
  OSDI 2020 evaluation and reporting cycle.
- DataDiffFuzz currently has 9 confirmed latest-version families across
  DataFusion, DuckDB, Polars, and PyArrow, plus 277 rewardable live candidate
  families that still require reduction, deduplication, and upstream
  confirmation before they count.
- The fair paper claim is therefore not global bug-count outperformance over
  SQLancer/PQS. The stronger defensible claim is cross-ecosystem semantic
  coverage with paper-grade evidence, plus a scope-limited SQLancer DuckDB
  baseline for the SQL-overlap slice.

## Next Work Items

1. Triage confirmed-family expansion toward 20+:
   - rank the 277 rewardable candidate families by backend diversity, count,
     reproducibility, and novelty;
   - select 10-15 for reduction/upstream confirmation;
   - produce 2-4 polished case studies.
   - current first P0 DuckDB evidence report:
     `reports/candidate-family-evidence-p0-duckdb-live-20260615.md`.
   - current minimized DuckDB reproducer audit:
     `reports/duckdb-sql-reproducers/p0-duckdb-live-20260615-minimized/manifest.json`.
2. Build an external SOTA baseline support track:
   - SQLancer source is under `experiments/external_tools/sqlancer_duckdb153`;
   - Java 21, Maven, and the SQLancer shaded jar are available;
   - SQLancer jar:
     `experiments/external_tools/sqlancer_duckdb153/target/sqlancer-2.0.0.jar`;
   - SQLancer git commit: `30248509ba72`;
   - first verified target is DuckDB with `QUERY_PARTITIONING` and `NOREC`;
   - results must import as `comparison_role=support`, not live bug evidence.
3. Update paper tables and claims after the new readiness report and external
   baseline smoke are available.

## SQLancer External Baseline Status

SQLancer has been built and executed using the official CLI layout:

```bash
java -jar target/sqlancer-2.0.0.jar \
  --num-threads 1 \
  --timeout-seconds <seconds> \
  --random-seed <seed> \
  duckdb \
  --oracle QUERY_PARTITIONING
```

General SQLancer options are placed before the DBMS name and DBMS-specific
oracle options are placed after the DBMS name, matching SQLancer's README.

Current executable support:

- runner: `scripts/run_sqlancer_baseline.py`;
- fair-comparison planner: `scripts/plan_sqlancer_fair_comparison.py`;
- fair-comparison summarizer:
  `scripts/summarize_sqlancer_fair_comparison.py`;
- tests: `tests/test_sqlancer_baseline_runner.py`,
  `tests/test_sqlancer_fair_comparison_plan.py`, and
  `tests/test_sqlancer_fair_comparison_summary.py`;
- smoke manifest:
  `reports/external-baselines/sqlancer-baseline-smoke-20260614T1930CST.json`;
- 3-seed DuckDB pilot manifest:
  `reports/external-baselines/sqlancer-duckdb-pilot-3seed-20s-20260614T1945CST.json`.

The 20s pilot executed 6/6 SQLancer runs successfully:

- suites: `duckdb-query-partitioning`, `duckdb-norec`;
- seeds: `1`, `1001`, `2001`;
- budget: single thread, 20s per run;
- total SQLancer summary queries: `33,000`;
- total generated databases: `6`.

The fair-comparison planner was also smoke-tested end to end:

- plan:
  `reports/external-baselines/plans/sqlancer-fair-smoke-20260614b.json`;
- script:
  `reports/external-baselines/plans/sqlancer-fair-smoke-20260614b.sh`;
- SQLancer manifests:
  `reports/external-baselines/sqlancer-fair-smoke-20260614b-duckdb-query-partitioning-20s.json`
  and
  `reports/external-baselines/sqlancer-fair-smoke-20260614b-duckdb-norec-20s.json`;
- matching DataDiffFuzz manifest:
  `runs/experiment-20260614T142234-1781446954375063436.json`.
- smoke summary:
  `reports/external-baselines/sqlancer-fair-smoke-20260614b-summary.json`.

The final SQLancer comparison metric policy is:

- primary metric: independently confirmed unique latest-version bug families in
  the DuckDB SQL-overlap scope;
- supporting efficiency metrics: SQLancer generated queries/databases/query
  throughput and DataDiffFuzz executed cases/candidate cases/rewardable
  candidate families/case throughput;
- evidence-quality metrics: minimized/native reproducer success, immediate
  recheck stability, executable issue bundle, and family-level deduplication;
- scope metrics: SQL-overlap results separated from DataFrame/Arrow/Polars/
  PyArrow/DataFusion cross-ecosystem value.

Do not compare SQLancer raw query count against DataDiffFuzz raw case count as
a paper conclusion. They are different execution units.

Final strict DuckDB comparison must use the same DuckDB engine version on both
sides. The version baseline follows DataDiffFuzz's current latest-version target
rather than SQLancer's default dependency. On 2026-06-15 CST, DataDiffFuzz uses
DuckDB Python `1.5.3` / engine `v1.5.3` (`14eca11bd9`), while the currently
built SQLancer jar was compiled with `duckdb_jdbc 1.3.0.0`. Therefore the
current 2h run is a pipeline/support pilot, not the final strict
version-aligned SQLancer comparison. Before claiming the four SQLancer
comparison metrics, rebuild SQLancer with a matching DuckDB JDBC dependency if
available, or explicitly report the version mismatch as a threat/limitation.

`duckdb_jdbc 1.5.3.0` is available and a version-aligned SQLancer build has now
been prepared:

- root: `experiments/external_tools/sqlancer_duckdb153`;
- SQLancer commit: `30248509ba72`;
- DuckDB JDBC dependency: `1.5.3.0`;
- jar: `experiments/external_tools/sqlancer_duckdb153/target/sqlancer-2.0.0.jar`;
- jar SHA256:
  `58fd71e69d6635eedc707ee0e76fb1ebb10bedeac3d79ee629cdadb6d3af78fe`;
- smoke manifest:
  `reports/external-baselines/sqlancer-duckdb153-smoke-20260615.json`;
- smoke result: 2/2 successful runs, strict target version match true.

The strict version-aligned 2h x 3-seed plan is active:

- plan:
  `reports/external-baselines/plans/sqlancer-fair-duckdb153-2h-3seed-20260615.json`;
- script:
  `reports/external-baselines/plans/sqlancer-fair-duckdb153-2h-3seed-20260615.sh`.

It is running in tmux session `sqlancer_fair_duckdb153_2h_20260615`. Wait for
all SQLancer/DataDiffFuzz seed runs and the final summary before using SQLancer
comparison numbers as final paper evidence.

Current local SQLancer bug count policy: completed smoke/support pilots have
0 SQLancer-reported failure signals and 0 independently confirmed SQLancer bug
families in this repository. The strict DuckDB 1.5.3 run is still pending, so
the final SQLancer bug count is not available yet.

Important SQLancer runtime constraint: current mainline cannot safely run with
`--log-each-select false`. SQLancer's `MainOptions.logExecutionTime()` asserts
when `log-each-select` is false, so the local runner rejects
`--no-log-each-select`. Fair baseline runs should keep statement logging
enabled and use `--no-log-execution-time` only. Monitor
`experiments/external_tools/sqlancer_duckdb153/target/logs` during 2h/24h runs.

A 2h x 3-seed fair comparison was launched in tmux session
`sqlancer_fair_2h_20260614` on 2026-06-14 CST using:

- plan: `reports/external-baselines/plans/sqlancer-fair-2h-3seed-20260614.json`;
- script: `reports/external-baselines/plans/sqlancer-fair-2h-3seed-20260614.sh`;
- log: `logs/external-baselines/sqlancer-fair-2h-3seed-20260614.log`.

The matching DataDiffFuzz pilot was run as three separate 20s executions because
`datadiff experiment --duration` is a total matrix budget under adaptive
scheduling, not a per-seed budget. The executed manifests were:

- seed `1`: `runs/experiment-20260614T135154-1781445114853515800.json`;
- seed `1001`: `runs/experiment-20260614T135334-1781445214617838291.json`;
- seed `2001`: `runs/experiment-20260614T135339-1781445219441234279.json`.

Important fairness note: the current final-live DataDiffFuzz campaign
`embedded_sql_cross:live_duckdb_issue_focus` runs `pandas,duckdb,sqlite`, so it
is SQL-oriented but not identical to SQLancer's single-DBMS DuckDB scope. For a
strict SQL overlap table, prefer a dedicated DataDiffFuzz command using
`--target-suite embedded_sql` or explicit `--backends duckdb,sqlite`, and report
that DataDiffFuzz uses cross-backend differential evidence while SQLancer uses
single-DBMS metamorphic oracles.

## Claim Discipline

Safe high-confidence claim shape:

> DataDiffFuzz complements SQL-only DBMS fuzzers by using a unified typed
> workflow representation, semantic normalization, and a shared evidence
> pipeline across DataFrame, Arrow, embedded SQL, and query-engine backends.
> We compare against weaker configurations and SQL-oriented scopes to isolate
> the value of cross-ecosystem testing, and we use a scope-limited SQLancer
> baseline where the target/input overlap is fair.

Avoid:

> DataDiffFuzz outperforms SQLancer/SQUIRREL/SQLRight overall.

That claim is not currently supported and is not a fair comparison for the
DataFrame/Arrow parts of the system.
