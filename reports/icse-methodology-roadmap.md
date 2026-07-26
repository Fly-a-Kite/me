# ICSE-Oriented Methodology Roadmap

## Thesis

DataDiffFuzz is a reusable semantic differential fuzzing methodology for dataframe, embedded SQL, Arrow, and query-engine frameworks. The core claim should not be "we found one DataFusion bug"; it should be:

> A common typed DSL, semantic normalization layer, differential/metamorphic oracles, feedback-guided generation, and artifact triage pipeline can expose cross-framework semantic implementation bugs while separating real bugs from documented boundary divergences and generator/normalizer false positives.

## Method Components To Evaluate

1. Common DSL and target adapters
   - Keep the same generated programs across pandas, Polars eager/lazy, DuckDB, SQLite, DataFusion, and PyArrow.
   - Report target families, layers, and common capability intersection from `datadiff targets --json`.

2. Type-aware and profile-aware generation
   - Compare `baseline` vs `no_type_aware`.
   - Compare generic profiles against domain profiles such as `null_groupby_topk`, `null_agg_topk`, `float_group_key`, `workflow`, and `edge_float`.

3. Guidance and feedback
   - Compare `baseline`, `guided`, and `no_feedback`.
   - Report first-candidate index, candidate cases/s, new behavior rate, path proxy, frontier score, and contribution score.

4. Oracle stack
   - Differential oracle for cross-backend mismatches.
   - Metamorphic oracle for single-backend relation violations.
   - Classification oracle to separate candidate implementation bugs from documented/expected semantic divergences and false positives.

5. Triage and reproducibility
   - Automatically produce triage JSON/Markdown, reduced cases, standalone reproducers for supported root causes, upstream issue drafts, and artifact bundles.

## Required Research Questions

- RQ1: How many unique candidate bug families does the method find across framework families?
- RQ2: How much do guidance and feedback improve time-to-first bug and candidate yield?
- RQ3: How much do type-aware generation, preflight repair, and normalization reduce invalid cases and false positives?
- RQ4: How well do metamorphic relations complement cross-framework differential testing?
- RQ5: How portable is the method across dataframe, embedded SQL, Arrow, and query-engine targets?
- RQ6: How reproducible are the findings under new seeds, Ubuntu x86_64, latest dependency versions, and standalone scripts?

## Current Evidence

- DataFusion `grouped_topk_null_sort_key@datafusion` is stable across standalone reproduction, 60k-case main matrix, 10k-case confirmation matrix, 100-case post-triage health check, regression tests, and generated artifact bundle. It was submitted upstream as https://github.com/apache/datafusion/issues/22190 on 2026-05-15; DataFusion member `xiedeyantu` replied `take` on 2026-05-17 UTC and is assigned. It should remain `submitted_upstream_needs_external_confirmation` until maintainers confirm, reject, or document the behavior.
- An independent-seed 60k-case domain matrix `runs/experiment-20260514T192105.json` used seeds `5001,6001,7001,8001,9001` and again produced only `grouped_topk_null_sort_key@datafusion`: 13,601 candidate implementation-bug cases, 0 false positives, and 0 semantic divergences. The same domain presets produced zero candidates in `core_lazy`.
- A pre-issue sanity matrix `runs/experiment-20260514T194522.json` used fresh seeds `12001,13001,14001` for 3,600 targeted DataFusion cases and again produced only `grouped_topk_null_sort_key@datafusion`: 2,336 candidate implementation-bug cases and 0 false positives.
- Boundary preflight script `scripts/datafusion_topk_null_sortkey_preflight.py` narrows the issue: grouped `ORDER BY` without `LIMIT`, plain NULL column top-k, and NULL group-key/non-NULL aggregate cases preserve rows, while grouped top-k over NULL `MIN(x)` ordered ascending and NULL `MAX(x)` ordered descending drops rows.
- After updating the `null_agg_topk` generator to cover both `MIN ASC` and `MAX DESC`, post-enhancement matrix `runs/experiment-20260514T195546.json` executed 3,000 fresh targeted cases with 2,562 candidate implementation-bug cases, 0 false positives, and the same deduplicated DataFusion family.
- Pattern variant analysis `reports/pattern-variants-null_agg_topk-experiment-20260514T195546.md` confirms the generator-level boundary: `MIN ASC` and `MAX DESC` variants account for all candidates, while `MIN DESC` and `MAX ASC` variants are negative controls in the same run.
- Multi-op second-layer profile `filter_null_agg_topk` extends the same semantic trigger with `filter -> mutate -> select -> groupby -> select -> sort -> limit` rather than a bare aggregate-top-k shape. Matrix `runs/experiment-20260519T164232-1779208952365217503.json` executed 320 cases across `core_datafusion`, `datafusion_cross`, `core_lazy`, and `arrow_cross`; it preserved the same `grouped_topk_null_sort_key@datafusion` family with 73 candidate cases out of 80 in both DataFusion suites (91.2%), while `core_lazy` and `arrow_cross` remained clean and all suites again had 0 false positives and 0 semantic divergences. This is the current best evidence that the second-layer target testing is not limited to join-heavy or one-operator toy programs.
- Polars lazy float group-key candidate did not reproduce with Polars 1.40.1 on Ubuntu and should not be counted as a confirmed candidate bug.
- Methodology contract tests now enforce multi-framework target coverage, common DSL capability coverage, ablation axes, targeted bug-hunting profiles, and seeded-fault sensitivity suites.
- Seeded sensitivity matrix `runs/experiment-20260514T184753.json` separates expected injected-fault roots from all candidate findings. Targeted presets improve expected-root detection over baseline:
  - `seeded_filter/guided_filter`: 17.8% to 48.5% expected fault cases, 2.73x.
  - `seeded_groupby/guided_groupby`: 31.3% to 66.9%, 2.14x.
  - `seeded_join/guided_join`: 7.2% to 33.4%, 4.63x after switching `guided_join` to a join-heavy no-groupby profile.
  - `seeded_mutate/guided_mutate`: 10.9% to 24.6%, 2.26x.
  This supports the sensitivity-evaluation story, while the mutate time-to-first regression should be discussed as a guidance tradeoff.
- Generalization/ablation matrix `runs/experiment-20260514T190631.json` covers six suites (`core`, `core_lazy`, `core_datafusion`, `core_arrow`, `datafusion_cross`, `arrow_cross`), six presets (`baseline`, `guided`, `no_feedback`, `no_type_aware`, `no_normalizer`, `metamorphic`), and five fixed seeds for 90,000 cases. It produced 4,322 findings, 61 candidate implementation-bug cases, 133 expected semantic divergences, and 4,128 oracle false positives. All false positives came from `no_normalizer`; `baseline`, `guided`, `no_feedback`, and `metamorphic` produced zero false positives, and their only candidate cases were the confirmed DataFusion NULL aggregate top-k family in DataFusion suites. This is the current strongest ablation evidence that semantic normalization is necessary, and that the default oracle stack avoids the obvious reviewer critique that cross-framework differences are mostly formatting or ordering artifacts.
- Ablation audit `reports/ablation-audit-experiment-20260514T190631.md` turns the 90k matrix into a paper-facing soundness boundary. Trusted presets (`baseline`, `guided`, `no_feedback`, `metamorphic`) executed 60,000 cases with 0 oracle false positives. Ablation presets (`no_type_aware`, `no_normalizer`) executed 30,000 cases with 4,128 oracle false positives. Every new family introduced by `no_type_aware` is marked `ablation_only_do_not_count_without_triage`; only `grouped_topk_null_sort_key@datafusion` appears in both trusted and ablation settings.
- Non-DataFusion no-groupby exploration `runs/experiment-20260514T193946.json` executed 36,000 trusted-configuration cases across `core`, `core_lazy`, `core_arrow`, and `arrow_cross` with `bughunt_no_groupby`, guided, and metamorphic presets. It produced 0 findings and 0 false positives. This is a useful negative control: the method is not generating arbitrary candidate bugs outside the known DataFusion risk area.
- Boundary profile expansion on 2026-07-06 added six latest-version bug-hunting profiles for normalized join-key membership, coalesce/union/distinct/groupby, multi-key semi/anti joins with null guards, empty-input union/groupby, nullable boolean coalesce membership, and numeric-text cast membership aggregation. See `reports/real-review-20260706/boundary-profile-expansion.md`. Initial structure/guidance tests and seven-backend smoke checks passed. The profile-cycle tmux run completed 86,180 latest-target case-log rows across `latest_all_engines` and `latest_no_datafusion`; all statuses were `ok`, with 0 fresh candidates and 0 false-positive reasons. This is a clean negative-control result for adding the profiles to longer discovery campaigns.
- Boundary profile expansion V2 on 2026-07-06 added five more profiles for string tokenization before join/distinct, date-part extraction feeding partitioned row-number filters, post-groupby join plus global aggregate, distinct-to-membership top-k, and coalesced row-number top-k. See `reports/real-review-20260706/boundary-profile-expansion-v2.md`. Structure/guidance tests and seven-backend smoke checks passed. A high-volume `date_part_row_number_union` raw family from a pre-parser-fix process was downgraded to a local `condition_cmp` false positive; clean post-window-cmp-fix date-part reruns are active under `tmp/boundary-combo-v2-datepart-rerun-20260706-post-window-cmp-fix/` and started with 0 findings.
- Boundary profile expansion V3 on 2026-07-06 added three additional latest-version profiles for union/distinct materialization before anti-join and running windows, nullable coalesce/case/semi-join/groupby, and date/string/cast transformations before row-number top-k. See `reports/real-review-20260706/boundary-profile-expansion-v3.md`. The same change fixed a local `condition_cmp` default-order bug and added dynamic window tie-breaker repair for `running_sum` and `row_number_filter`; targeted tests and seven-backend smoke checks now pass with 0 surviving smoke findings. New tmux runs are active under `tmp/boundary-combo-v3-longrun-20260706-post-window-cmp-fix/`.
- Common API boundary supplement on 2026-07-06 started non-invasive latest-version tmux discovery lanes using the existing `common_api_workflow` generator and differential plus metamorphic checking over strings, fill-null, union/distinct/offset/top-k, membership joins, coalesce, case_when, row-number, date parts, casts, joins, and aggregation. See `reports/real-review-20260706/common-api-boundary-supplement.md`. Early raw candidates exposed local adapter/parser/schema noise and are not counted. Post-window-cmp-fix feedback-off replacement runs are active under `tmp/common-api-boundary-longrun-20260706-post-window-cmp-fix/` and started with 0 findings.
- Latest-version audit on 2026-07-06 passed for all registered target packages: `pandas 3.0.3`, `polars 1.42.1`, `duckdb 1.5.4`, `pyarrow 24.0.0`, `datafusion 54.0.0`, `chdb 4.2.0`, and `pysqlite3-binary 0.5.4.post2` all matched PyPI latest versions. See `reports/real-review-20260706/target-version-audit-20260706-post-window.json` and `reports/real-review-20260706/latest-version-discovery-status-post-window-20260706.md`.

## Next Experiments

1. Full generalization matrix depth run
   - Suites: `core`, `core_lazy`, `core_datafusion`, `core_arrow`, `datafusion_cross`, `arrow_cross`.
   - Presets: `baseline,guided,no_feedback,no_type_aware,no_normalizer,metamorphic`.
   - Increase cases per seed beyond the current 500-case, five-seed matrix if time permits.

2. Seeded sensitivity matrix scale-up
   - Suites: `seeded_filter,seeded_groupby,seeded_join,seeded_mutate`.
   - Presets: `baseline,guided_filter,guided_groupby,guided_join,guided_mutate`.
   - Metrics: injected fault detection rate, first detection index, detection throughput, and guidance regressions.

3. Domain bug-hunting matrix
   - Suites: `core_datafusion,core_lazy,datafusion_cross,arrow_cross`.
   - Presets: `null_groupby_topk,null_agg_topk,float_group_key,float_group_key_metamorphic`.
   - Metrics: deduplicated bug families, candidate case rate, and false-positive count.

4. Non-DataFusion depth run
   - Extend the current 36k-case no-groupby negative control with more seeds or duration-based long runs.
   - Keep it as a precision/soundness experiment unless it finds a stable new candidate family.

5. Reproducibility package
   - Keep minimal logs, summaries, aggregate CSVs, target registry JSON, environment captures, standalone scripts, triage artifacts, upstream issue drafts, and bundle checksums.

## Paper Positioning

Frame the paper as an engineering method and evaluation pipeline, not as a single-tool bug report. The contribution list should be:

- A typed dataframe/query DSL shared across heterogeneous analytical frameworks.
- A normalization and oracle stack for distinguishing common-subset bugs from semantic boundary divergences.
- A feedback-guided, target-profiled generation strategy for semantic risk areas.
- A triage pipeline that turns findings into reduced, reproducible, externally reportable artifacts.
- An empirical evaluation across multiple target families, ablations, seeded faults, and real candidate bugs.
