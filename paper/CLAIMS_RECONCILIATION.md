# Claims Reconciliation — canonical numbers

Purpose: single source of truth for every numeric claim in the paper. Any number not listed
here must not appear in the submission. Derived from the frozen source at `a87d380`
(`experiments/latest_confirmations.json`, `bugs/`) on 2026-09-14.

## 1. Bug-family accounting

`experiments/latest_confirmations.json` contains **11 confirmation records**, all with
`discovery_credit = datadiff_submitted`. The honest partition is:

| Tier | Definition | Count |
| --- | --- | --- |
| **Strict confirmed** | upstream labeled the issue `bug` **or** the fix landed (`upstream_labeled_bug` + `fixed_upstream`) | **9** |
| Pending | submitted, awaiting independent confirmation | 1 |
| Non-original | matches a pre-existing upstream issue (`similar_existing_upstream_issue`) | 1 |
| **Total records** | | **11** |

Strict confirmed by backend:

| Backend family | Strict confirmed | Which |
| --- | ---: | --- |
| DataFusion (query engine) | 5 | grouped_topk_null_sort_key; groupby_aggregation; negative_zero_comparison; distinct_null_topk; datafusion_limit_idempotence |
| Polars (DataFrame) | 2 | polars_reflected_arithmetic_operand_order (labeled); groupby_aggregation@polars (fixed) |
| PyArrow (Arrow) | 1 | pyarrow_sliced_bool_groupby_any_all |
| DuckDB (embedded SQL) | 1 | metamorphic_semi_anti_join_rewrite (fixed) |

Excluded from the strict count: `polars_vector_division_rounding` (similar existing issue),
`path_projection_keyed_pick` (pending confirmation).

**Rule:** the paper reports `9 strict confirmed families` (+ 1 pending, 1 non-original shown
separately). It must **not** report 5, 8, 11, or "20+" as the confirmed count.

## 2. Fixes to doc inconsistencies

| Doc | Old claim | Corrected |
| --- | --- | --- |
| `PAPER.md` abstract / §6.1 | 8 bugs | 9 strict confirmed families |
| `PROJECT_IDEA_AND_PROGRESS.md` | 5 | 9 strict |
| `docs/icse_sota_success_criteria.md` | 9 + `ready:true` (report missing) | 9 strict; **readiness claim retracted** until W1/W3 regenerate the report |
| `experiments/latest_confirmations.json` | 11 records | keep 11 records, but label tiers as above |
| `PAPER.md` §3.2 vs §6.1 | 9/42 vs 12/45 probes/relations | **recount from code before writing** (pending W3) |

## 3. Probe / metamorphic-relation counts

`PAPER.md` gives 9 vs 12 probes and 42 vs 45 relations. Do **not** publish either until
recounted from the live registry. Task: `paper/experiments/rq3_oracle_complementarity.md`
must emit the counts programmatically.

## 4. Innovation count

Docs say "16 innovations"; the M/S/Q lists total 18. The paper will **not** use an innovation
count. It uses contributions C1/C2/C6 (+C7 as oracle core) from `PLAN.md` §3.

## 5. Reproducibility status of result tables

At `a87d380`, `runs/` contains 0 `experiment-*.json` manifests and the reports cited by
`PAPER.md` §6.2–§6.6 are absent. Therefore **every quantitative table in §6 is currently
unverified** and must be regenerated in W3 or removed. Surviving evidence: 1,327
`run-20260726-*` run artifacts, 9 strict confirmation records, 6 `bug_*` bundles.

## 6. 24-hour authorization

Status: **not authorized**. `PAPER.md` §6.6 ("12 h run"), the readiness doc ("matrix
complete"), and `src/datadiff_osc/_phase6_gate_authority.py` ("can never authorize a 24-hour
campaign") conflict. The paper will not claim a completed 24 h campaign unless a signed
launch authorization exists.
