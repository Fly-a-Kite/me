# Engineering run results — 2026-09-14

Honest record of the first execution runs. No new bugs yet; this establishes the mechanism and
a throughput baseline.

## 1. Campaign pipeline pilot

- 2 lanes (`common_api_workflow`, `datafusion_optimizer`) × 20 cases, seed 30700001.
- Exit 0, manifest + run logs + report written. Pipeline validated end to end.
- Fresh candidate families: 0.

## 2. 30-minute engineering longhaul (tmux `wtpaper-eng`)

| Metric | Value |
| --- | --- |
| Manifests | 95 |
| Runs completed / planned | 259 / 260 |
| Executed cases | 5,180 / 5,200 |
| Wall time | 1,788 s (~30 min) |
| Throughput | 6.60 cases/s |
| Lanes / presets | 11 / 10 |
| Seeds | from 31000401 (reserved blocks avoided) |
| **Fresh candidate bug families** | **0** |
| Triage verdicts | none |

Code under test: `scripts/paper/wt_python.sh` (worktree source, not the base editable install).
Provenance freeze manifest, pip freeze, git status/diff, and target audit were produced.

## 3. Cross-version scans (`cross-version-scan`)

| Target | Cases | Findings |
| --- | --- | --- |
| DataFusion 53.0.0 vs 54.0.0, canonical corpus | 10 | **1** — `datafusion_distinct_null_topk` (df53 `[""]` vs df54 `[null]`) |
| DataFusion 53 vs 54, fresh `datafusion_targeted_rotation` | 40 | 0 |
| DataFusion 53 vs 54, fresh `null_agg_topk` | 30 | 0 |
| Polars 1.40.1 vs 1.42.1, canonical corpus | 10 | **1** — `polars_grouped_max_sort_metadata` (1.40.1 `[[1,"a"]]` vs 1.42.1 `[[2,"b"]]`) |

## 4. Interpretation

- The cross-version mechanism is correct: it re-detects both known version-sensitive corpus
  roots, and reports no false positives on the stable ones.
- Fresh random/targeted cases do **not** yet hit version differences. Version-differential yield
  needs either the exact version-sensitive family shapes (the corpus cases) or much larger
  budgets; the generic profiles are not sufficient.
- The campaign pipeline scales cleanly (5,180 cases in 30 min) with a documented throughput,
  which is directly usable as an RQ4 efficiency baseline.

## 5. Next levers

1. Full 24 h × 2 longhaul with the frozen code (the planned W2/W3 run).
2. Cross-version scanning seeded with the **known-root family shapes** (mutate corpus cases)
   rather than generic profiles.
3. Targeted family profiles matched to the version-sensitive roots
   (`null_agg_topk`, `ordered_groupby_sort`, `nested_topk_offset_aggregate`).
