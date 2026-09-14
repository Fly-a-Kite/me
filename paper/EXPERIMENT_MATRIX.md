# Experiment Matrix — DataDiffFuzz ICSE/FSE evaluation

Every cell names a treatment, a control, a frozen seed policy, a metric, and the artifact that
proves it. Nothing here is a result yet; results go to `paper/experiments/results/`.

## 0. Common protocol

Environment (frozen target):

```bash
V=/data1/lbw/xjx/datadiff_fuzz_lab/.venv/bin/python
export PYTHONPATH="$PWD/src:$PWD"
```

- **Source**: dev worktree `wt-paper` at a tagged commit; record `git rev-parse HEAD` per run.
- **Backends**: pandas 3.0.3, polars 1.42.1, duckdb 1.5.4, pyarrow 25.0.0, datafusion 54.0.0,
  chDB 4.2.1 (SQLite via `pysqlite3` if exercised).
- **Fresh seed policy**: excluded blocks `2026071800–2026071805`, `30000001–30000113`,
  `30500001–30500102`, `30600001–30600102`; longhaul reserved from `31000001`.
  Paper discovery uses fresh pairs starting at **30700001 / 30700102** (P6-02c-3 recommendation).
- **Live default method**: `p8_candidate_v1`; coverage/regression arm
  `p8_semantic_witness_global_v4`.
- **Control arms** (explicit `--research-control-arm`): `legacy_cartesian`,
  `contract_cartesian`, `contract_lattice_static|plan|shared_cost|full`,
  `p4_lattice_plan_unguided_cache_off`, `p4_lattice_plan_cache_off`, `p5_physical_plan`,
  `p5_goal_first`, `p5_priority_v2`, `p5_quality_diversity`, `p5_full_method`.
- **Counting contract**: finding → recheck survivor → native candidate → issue-ready root →
  strict confirmed family (`paper/CLAIMS_RECONCILIATION.md`).
- **Statistics**: paired seed blocks; report per-block values, median, bootstrap 95% CI
  (10,000 resamples), and paired effect size. Zero-yield and negative runs are kept.
- **Separation**: fresh discovery, regression/known-root, saturated, issue-inspired, and
  diagnostic evidence live in physically separate output roots.

Standard bounded command shape:

```bash
$V -m datadiff.cli discovery-campaign \
  --lanes <lane-spec> --cases 100 --seeds 30700001,30700102 \
  --output-manifest paper/experiments/results/<rq>/manifest-<arm>-<seedpair>.json \
  --candidate-recheck-count 3
$V -m datadiff.cli candidate-pipeline --manifest <...>
$V -m datadiff.cli discovery-campaign-aggregate --manifests '<...>' --output <...>
$V -m datadiff.cli run-health --manifest <...>
```

## RQ1 — Effectiveness (fresh latest-version discovery)

| | |
| --- | --- |
| Hypothesis | The full system finds reproducible, upstream-confirmable latest-version bug families across ≥3 backend families. |
| Treatment | `p8_candidate_v1`, 11 formal lanes, fresh seeds |
| Control | none (descriptive) + historical replay arm for sanity |
| Budget | 3 fresh seed pairs, 100 cases × 11 lanes each (6,600 cases) bounded; optional 24 h run for RQ1-long |
| Primary metrics | strict survivors / 10k executed cases; unique candidate families; time-to-first-survivor; strict confirmed families (9 baseline + new) |
| Secondary | per-backend family counts; preflight failure rate; false-positive rate |
| Artifacts | per-run journals, campaign manifests, candidate-pipeline bundles, `bug-status --json`, confirmation table |
| Status | needs re-run (manifests lost) |

## RQ2 — Noise control (normalizer + classification)

| | |
| --- | --- |
| Hypothesis | Semantic normalization + classification materially reduce false positives and expected-divergence noise without hiding real families. |
| Treatment | `contract_cartesian` (lossless contract comparison + lossless observation) |
| Control A | `legacy_cartesian` (legacy comparison/observation) |
| Control B | `contract_ccs_cartesian`, `contract_lattice_static` (partial normalization) |
| Budget | paired seeds, 3 blocks |
| Primary metrics | raw findings vs post-classification survivors; FP rate; expected-semantic-divergence rate; normalization-before/after candidate counts |
| Artifacts | `analyze-experiment`, `classify-run`, `analyze-ablation-audit` outputs |
| Status | code exists; tables must be regenerated |

## RQ3 — Oracle complementarity

| | |
| --- | --- |
| Hypothesis | Differential, metamorphic, and deterministic probes cover different error classes and are complementary. |
| Treatment | all three enabled |
| Controls | differential-only (`--no-...` metamorphic; probes disabled), metamorphic-only (`--enable-metamorphic-oracle`, differential off), probe-only (`--probes ...`, fuzz off) |
| Budget | paired seeds, 3 blocks; plus a deterministic probe-only sweep on the 9 strict roots |
| Primary metrics | families unique to each oracle; pairwise Jaccard; union vs intersection; per-oracle family distribution |
| Extra | recount probes and metamorphic relations from the live registry (resolves 9-vs-12 / 42-vs-45) |
| Status | needs instrumentation + re-run |

## RQ4 — Generation and scheduling efficiency

| | |
| --- | --- |
| Hypothesis | Typed generation + guidance + adaptive/coverage-debt scheduling beat random/unguided/no-feedback controls. |
| Treatment | `p8_candidate_v1` (adaptive) |
| Controls | `p4_lattice_plan_unguided_cache_off` (unguided); `--schedule matrix_order`; `--adaptive-learning-weight 0`; random generator profile |
| Budget | paired seeds, ≥12 h/cell if long mode; bounded 3-block pilot first |
| Primary metrics | valid-program ratio; executed cases/s; candidate families/hour; time-to-first family; novelty-weighted yield |
| Secondary | CPU hours, wall time, artifact bytes/case |
| Artifacts | `experiment` manifests, `adaptive-benchmark`, `experiment-summary` |
| Status | code exists; tables lost |

## RQ5 — Evidence-pipeline actionability

| | |
| --- | --- |
| Hypothesis | Triage → recheck → reduce → dedup → bundle turns fuzz findings into reproducible, submission-ready evidence. |
| Treatment | full `candidate-pipeline` |
| Control | audit-only (no recheck/reduction/bundling) |
| Budget | all RQ1 candidates |
| Primary metrics | reduction ratio; reproducer success rate; issue-bundle completeness; readiness status; flaky/timeout rate |
| Artifacts | `issue-bundle`, `issue-readiness`, `reduce`, `reproduce`, `validate-artifact` outputs; 9 strict root bundles |
| Status | 9 roots available; aggregate report must be regenerated |

## RQ6 — Transferability / reuse across backend families

| | |
| --- | --- |
| Hypothesis | One typed IR + normalizer + oracle complex is reusable across DataFrame / Arrow / embedded SQL / query-engine families without per-backend ad-hoc scripts. |
| Treatment | static accounting + runtime reuse counters |
| Controls | n/a |
| Budget | static + one bounded cross-family campaign |
| Primary metrics | workflow reuse rate across backends; capability coverage per family; adapter LOC + implementation time per new target; per-family strict families; shared oracle/relation coverage |
| Artifacts | `targets`, `semantic-registry`, adapter LOC table, run journals |
| Status | new measurement; no prior evidence |

## RQ-C7 — Semantic HyperContract oracle gates (phase 6)

| | |
| --- | --- |
| Hypothesis | The compositional HyperContract oracle satisfies correctness and efficiency gates at scale. |
| Gate 1 (contract) | exact-vs-staged verdict discrepancy = 0 over ≥100,000 result groups; 9/9 root recall; 100% adjudicated-FP precision fixes; ≥1 FPF/mutant; p95 compile/match ≤2 ms or ≤5% case wall; paired throughput regression ≤10%. |
| Gate 2 (granularity/reachability) | 232/232 cells constructed+activated; 384/384 edges observed; 16/16 families, each ≥2 seed blocks; scheduled activation ≥95% overall / ≥90% per family; mutation ≥90% / ≥80%; 11 lanes; 21/21 ops, 8/8 aggregates, 7/7 risk classes, 18/18 pipelines. |
| Gate 3 (runtime-correctness) | full repo suite 0 failures; v3 11 lanes × 2 seeds × 100 cases = 2,200/2,200 executed, 0 failures/non-OK; parallel worker invariance; exact-vs-staged escalation completeness. |
| Dependency | requires W2 tasks T1–T12 (`paper/research/CODE_GAP_REPORT.md`); no gate green in isolation |
| Budget | compute ≈8–20 h once built; engineering ≈4–6 weeks |
| Status | unrun; this is the phase6 software-engineering contribution |

## RQ7 — Competitor reachability of our confirmed bugs

| | |
| --- | --- |
| Hypothesis | Most of our strict confirmed families are unreachable by the strongest competitors' published target sets and oracle families. |
| Method | Per-bug scope+oracle coverage judgment from the competitors' own papers; then an empirical `tdiff_style` / `sqlancer_common_scope` arm. |
| Primary metric | number of strict confirmed families inside each competitor's (target systems × oracle family) box |
| Artifacts | `experiments/rq7_competitor_coverage.md` (argument), plus arm manifests |
| Status | scope argument complete (7/9 outside TDiFf scope; 9/9 outside SQL-only oracle scope); empirical arms pending |

## Baselines

| Baseline | Source | Fairness rule |
| --- | --- | --- |
| SQLancer (NoREC / TLP / PQS) | `src/datadiff/sqlancer_baseline_runner.py`, `docs/sqlancer_pqs_transfer_plan.md` | same machine, same wall/CPU budget, same strict counting; incompatibilities reported separately, never counted as a win |
| Random / unguided generator | `--research-control-arm p4_lattice_plan_unguided_cache_off`, random profile | identical case budget and seeds |
| Legacy comparison | `--research-control-arm legacy_cartesian` | isolates normalization contribution |
| No-feedback / static schedule | `--adaptive-learning-weight 0`, `--schedule matrix_order` | isolates adaptive scheduling |

## Execution order

1. **Pilot** (bounded, ~1 h): smoke each RQ cell; freeze commands and seed pairs.
2. **RQ2/RQ3/RQ5** regenerate from pilot + 3-block runs.
3. **RQ1-long / RQ4-long**: background multi-hour to 24 h paired runs.
4. **RQ6** static accounting.
5. **RQ-C7** after W2 code completion.
6. Freeze artifacts, rerun the reproduction script from scratch, then write results.

## Threats to validity (to be maintained)

- Cross-backend divergence is not automatically a bug; classification + native reproduction
  and upstream confirmation gate the strict count.
- Backend versions drift; freeze the venv and record package hashes per run.
- Seed-sensitive discovery; use paired blocks and report CIs, not point estimates.
- Engine coverage differs (DataFusion has more lanes), so per-family rates are reported
  alongside totals.
- Historical manifests are lost; any regenerated number must be labelled with its run date.
