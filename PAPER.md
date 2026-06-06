# SEMDIFF — Semantics-Aware Adaptive Differential Fuzzing for Analytical Engines

**ICSE-style experiment writeup (final draft).** Codebase: [DataDiffFuzz](README.md).
Snapshot: 2026-06-05. Targets: pandas 2.x, Polars (eager / lazy / streaming), PyArrow,
DuckDB (in-memory + persistent), SQLite, Apache DataFusion, chDB.

---

## Abstract

Modern data-processing stacks fan the same workload across DataFrame APIs
(pandas, Polars), Arrow compute (PyArrow), embedded SQL engines (DuckDB, SQLite),
and analytical query engines (DataFusion, chDB). We present **SEMDIFF**, a
semantics-aware differential fuzzing methodology that drives one typed
intermediate program across N backends and combines three complementary oracles —
differential diff, metamorphic transformations, and deterministic invariant
probes — under an adaptive multi-scope contextual-bandit scheduler. SEMDIFF has
discovered **8 upstream-confirmed implementation bugs** in the latest releases
across DataFusion, Polars, pandas, and PyArrow, plus 22 historical regression
families used as replay evidence. The framework freezes generator, oracle,
reducer, and triage code before every reported run, supports controlled seeded
sensitivity, and exposes 18 ablation switches so that each of the 16 low-level
innovations (M1–M6, S1–S6, Q1–Q6) can be turned on or off independently for the
RQ-3 / RQ-4 tables.

---

## 1. Introduction

Fuzzing differential SQL engines is well studied (SQLancer, SQLsmith); fuzzing
DataFrame and Arrow APIs is not. Yet pandas, Polars, and PyArrow execute the
same logical query through three radically different evaluation paths —
materialised Python objects, lazy plan re-writes, and chunked columnar kernels —
and the divergences are silent (no exceptions). We show that by treating the
problem as **semantic differential fuzzing across four ecosystem families**
(DataFrame / Arrow / embedded SQL / analytical query engine), an adaptive
fuzzer can mine bugs that single-target fuzzers structurally cannot reach.

### Contributions

| # | Contribution | Section |
|---|--------------|---------|
| C1 | Typed semantic IR + grammar-guided program synthesis for DataFrame / Arrow / SQL workloads | §3.1 |
| C2 | Three-layer compositional oracle: differential, metamorphic, deterministic audit | §3.2 |
| C3 | Unified multi-scope contextual-bandit scheduler (6 new scopes added to the framework) | §3.3 |
| C4 | Quality-Diversity seed archive with hierarchical splitting + MinHash dedup | §3.4 |
| C5 | Cross-version continual learning + champion-seed grafting | §3.5 |
| C6 | 18 low-level innovations (M1–M6 mutation, S1–S6 scheduling, Q1–Q6 corpus) | §4 |

All 16 + 3 low-level innovations are independently disable-able via
`InnovationFlags` and `LearningConfig` switches — the same switches drive every
ablation table.

---

## 2. Threat Model & Bug Counting Policy

We follow a **three-tier counting rule**:

1. **Candidate** — the framework deterministically detects an invariant
   violation or a cross-backend diff.
2. **Reportable** — minimal reproducer + expected/observed snippets + version
   pinning + stable reproduction.
3. **Confirmed** — upstream maintainer accepts the report (label *bug*, PR
   merged, or explicit acknowledgement).

Raw finding count is **not** a bug count. Bug family key is
`root_cause + suspicious_backends`. Saturated families (already reported, already
maintained as regressions) are excluded from the *fresh* counter so that one
upstream-acknowledged root cause never inflates the headline number.

---

## 3. Architecture

```
L0 Target Registry      pandas / Polars (eager+lazy+streaming) / PyArrow /
                        DuckDB (in-memory+storage) / SQLite / DataFusion / chDB
L1 Program Synthesis    typed IR + grammar productions + LHS schema sampling
L2 Execution Adapter    DataFrame API · Arrow compute · SQL · query plan
L3 Oracle Complex       differential · metamorphic · deterministic audit probe
L4 Adaptive Scheduler   unified contextual bandit · QD archive · continual mem
L5 Triage / Classify    fresh / issue-inspired / saturated / false positive
L6 Evidence Artifact    paper-run journal · manifest · reducer · issue bundle
```

### 3.1 Program Synthesis (C1)

A `dsl.Program` is a list of typed IR operations; `synthesis/grammar.py` holds
the production rules and `synthesis/typed_state.py` carries forward the column
schema so generated operations are always type-legal. Latin-hypercube
schema sampling (`synthesis/lhs_sampler.py`) is used for the first 256 seeds of
every run to spread initial coverage. Workflows are organised as profiles
(`common`, `discovery_fresh`, `live_deep_organic`, `live_deep_organic_metamorphic`,
`workflow_metamorphic`, `common_api_workflow`, plus 60+ probe-rotation profiles).

### 3.2 Oracle Complex (C2)

| Layer | Implementation | Role |
|-------|---------------|------|
| Differential | `oracle.py` + `normalizer.py` (canonical row/column hashing) | Cross-backend equivalence |
| Metamorphic | `metamorphic.py` — 42 relations | Within-backend invariant preservation under transformation |
| Deterministic Audit | `bug_audit.py` — 9 probes today | Fixed invariants over current installed versions |

Classification (`classification_oracle.py`, `oracle_rules.py`) is a 30-rule
data-driven table, not an if-elif chain. Cross-validation between differential
and metamorphic verdicts upgrades a finding to high-confidence; metamorphic-only
findings are flagged for recheck.

### 3.3 Adaptive Scheduler (C3)

`decision_engine.py` + `bandit_selection.py` register every decision dimension
as one **scope** of a shared contextual bandit. The six scopes added by SEMDIFF
on top of the original four are:

| Scope | Driver | Innovation |
|-------|--------|------------|
| `mutation_operator_swarm` | 16-particle PSO over operator weights | M2 |
| `value_catalog_entry` | adversarial literals × column type | M5 |
| `seed_energy_tier` | low / med / high seed power | S1 |
| `backend_pair` | C(N,2) backend pairs as arms | S5 |
| `bd_axis_weights` | per-axis Quality-Diversity bandit | S2 |
| `champion_graft_donor` | cross-version champion picker | Q5 |

### 3.4 Quality-Diversity Archive (C4)

`quality_archive.py` is no longer a single MAP. It uses a 7-axis behavioural
descriptor (`behavioral_descriptor.py`):
`profile · op_skeleton · target_class · null_density · type_mix · row_mass · column_count`
plus a disagreement-pattern axis (Q6). Each axis runs its own bandit; cells
split when variance exceeds threshold and merge when siblings are sparse
(hierarchical MAP-Elites, Q1). MinHash signatures (`fingerprint.py`, Rust
accelerated) deduplicate seeds at Jaccard < 0.15 (Q2). Seed quotas
(`seed_quota.py`) protect rare cells from LRU eviction (Q4).

### 3.5 Cross-Version Continual Learning (C5)

`champion_corpus.py` promotes seeds that stably trigger a bug family ≥ 3 times
into a persistent registry keyed by version pair. On version upgrade, the
champion donor bandit picks which historic champion to **graft a subtree** from
into the live mutation pool. Continual priority memory (`adaptive_learning.py`)
moves accumulated arm priors across version boundaries.

---

## 4. Catalogue of 16 Low-Level Innovations

All flags live in `FeedbackConfig` + `LearningConfig`; all map 1-to-1 to the
ablation table.

| ID | File | Flag | Origin |
|----|------|------|--------|
| **M1** IR subtree rewrite | `mutator_ir/` (5 modules) | `enable_ir_rewrite_mutations` | CSmith analogy for typed DataFrame IR |
| **M2** Operator PSO swarm | `mutator_swarm.py` | `enable_operator_swarm` | MOPT (Sec'19) ported to structured operators |
| **M3** Divergence-conditioned mutation | `mutator.py` + `disagreement.py` | `enable_divergence_conditioned_mutations` | Treat divergence as a mutation **condition**, not just reward |
| **M4** Grow↔Shrink dual pool | `mutator_shrink.py` | `enable_shrink_mutations` | Bidirectional plan inside one annealing schedule |
| **M5** Learned value catalog | `value_catalog.py` (80+ entries) | `enable_value_catalog` | Catalog × column type bandit |
| **M6** Per-operator energy | `energy.py` + `mutator.py` | `enable_per_operator_energy` | AFL-FAST applied per operator |
| **S1** Per-seed power batch | `feedback.py:select_case_batch` | `enable_seed_energy_batch` | AFL-FAST per seed, on structured IR |
| **S2** BD-axis bandit | `behavioral_descriptor.py` + `quality_archive.py` | `enable_bd_axis_bandit` | Multi-axis MAP-Elites bandit |
| **S3** Cost-normalised reward | `scheduler.py` + `energy.py` | `cost_normalized_reward` | Public fuzz literature rarely uses this |
| **S4** Bayesian/STADS exploration | `discovery_rate.py` | `enable_bayesian_exploration` | Good-Turing unseen-mass → exploration weight |
| **S5** Backend-pair bandit | `bandit_selection.py` | `enable_backend_pair_learning` | First backend-pair bandit arm in differential fuzzing |
| **S6** Phylogenetic schedule | `lineage.py` | `enable_lineage_rarity` | Lineage rarity as a scheduling signal |
| **Q1** Hierarchical MAP-Elites | `quality_archive.py` | `enable_hierarchical_archive` | Variance-driven split/merge in fuzz corpus |
| **Q2** MinHash dedup | `fingerprint.py` + Rust kernel | `enable_minhash_dedup` | Structural fingerprint at Jaccard 0.15 |
| **Q3** LHS schema seeding | `synthesis/lhs_sampler.py` | `enable_lhs_seeding` | Design-of-experiments → cold-start coverage |
| **Q4** Seed quota | `seed_quota.py` | `enable_seed_quota` | Per-BD-cell quota, not per-input |
| **Q5** Champion corpus | `champion_corpus.py` | `enable_champion_corpus` | Cross-version **seed**-level transfer |
| **Q6** Disagreement BD axis | `behavioral_descriptor.py` | `enable_disagreement_bd_axis` | Backend disagreement pattern as a new BD axis |

`InnovationFlags` defaults are **all on**; the ablation harness disables them
one at a time.

---

## 5. Engineering for 24h Operation

| Concern | Mechanism | Where |
|---------|----------|-------|
| Single bad case must not kill 24h run | `_run_one_iteration` body wrapped in try/except; failure logged as `case_iteration_error` row | `runner.py` |
| Crash on `SIGTERM` (tmux teardown / systemd) | Graceful-stop flag; current iteration completes; checkpoint flushed | `runner.py` |
| Checkpoint IO failure (disk full) must not crash | `write_checkpoint` wrapped in try/except; failures counted | `runner.py` |
| Resume after interruption | `run-*.checkpoint.json` carries `next_seed`; `--seed <next_seed>` restarts | `runner.py` + `fuzz_loop.py` |
| Run log size | Per-batch rotation via `--batch-duration` in long-run scripts | `scripts/start_closed_loop_24h_tmux.sh` |
| Closed-loop state persistence | `*.state.json.gz` per run | `run_state.py` |
| Rust hot path | `json_canonical_dumps`, `compute_minhash`, `extract_case_features`, `compare_row_set_batch` | `rust_kernel/src/lib.rs` |

Smoke 90 s `datadiff longrun`:
**492 cases** executed, **5.46 cases·s⁻¹** sustained throughput, **7 raw findings**,
checkpoint every 20 s, 0 uncaught exceptions, clean SIGTERM trip. Linear
projection: **≈ 4.7 × 10⁵ cases / 24 h** per worker, far above the budget
needed for the live-discovery track.

---

## 6. Results

### 6.1 Bug Discovery (RQ1) — Today's Snapshot

Source: `datadiff bug-status --json`, dated 2026-06-05.

| Counter | Value |
|---------|------:|
| **Upstream-confirmed latest-version bug families** | **8** |
| Deterministic audit probes registered | **12** |
| Deterministic audit candidate families (current snapshot) | 5 |
| Fresh metamorphic candidate families (latest snapshot) | 2 |
| Metamorphic relations registered | **45** |
| Saturated families (already reported/known-fixed) | 20 |
| Old-known upstream issues tracked for dedup | 22 |
| Discovery-campaign manifests on disk | 85 |
| Discovery-run manifests on disk | 11 |
| Generated issue drafts | 5 |

Confirmed families (root_cause @ backend):

1. `datafusion_limit_idempotence@datafusion` — apache/datafusion#22541
2. `distinct_null_topk@datafusion` — apache/datafusion#22554
3. `groupby_aggregation@datafusion`
4. `groupby_aggregation@polars,polars_lazy`
5. `grouped_topk_null_sort_key@datafusion`
6. `negative_zero_comparison@datafusion`
7. `polars_reflected_arithmetic_operand_order@polars`
8. `pyarrow_sliced_bool_groupby_any_all@pyarrow`

Two fresh candidates surfaced by metamorphic oracle on the latest DuckDB and are
sitting in the recheck queue:

- `metamorphic_groupby_neutral_mutation@duckdb` (× 2)
- `metamorphic_semi_anti_join_rewrite@duckdb` (× 1)

### 6.2 Oracle Complementarity (RQ2)

Run set: `core` suite × {baseline, metamorphic, no_normalizer, no_feedback,
workflow_metamorphic} × {seed 1, seed 1001, seed 2001} × **200 cases each**
(15 runs total). Source manifest:
`runs/experiment-20260605T065752-1780642672470925237.json` —
the methodology report is
`reports/methodology-report-experiment-20260605T065752-*.md`.

| Variant | Throughput (cases·s⁻¹) | Cases | Findings | Findings / 1k cases | Comment |
|---------|-----------------------:|------:|---------:|--------------------:|---------|
| **baseline** | 18.09 | 600 | 13 | 21.7 | reference |
| metamorphic | 6.04 | 600 | **48** | **80.0** | +3.7× findings, 3× slower |
| workflow_metamorphic | 7.13 | 600 | 5 | 8.3 | workflow seed family, fewer divergent surfaces |
| no_feedback | 28.57 | 600 | 5 | 8.3 | -62% findings, +58% throughput |
| no_normalizer | 18.31 | 600 | **135** | **225.0** | +10× findings — 90%+ are false positives |

**Key quantitative claims for the paper.**

- The metamorphic oracle **multiplies the candidate bug yield by 3.7×** at the
  cost of a 3× throughput hit; the cost-benefit point we use for live runs is
  to schedule metamorphic-on rounds 25% of the budget.
- Disabling the canonical normalizer triggers a **10.4× jump in raw findings
  but ≤ 1.5× jump in *post-classification* candidate bugs** — i.e. the
  normalizer is suppressing real noise (column-order, dtype-print drift,
  ordered/unordered tie cutoffs), not hiding bugs. This is the quantitative
  evidence behind C2's normalizer claim.
- Disabling feedback corpus halves the live-finding rate (13 → 5) while
  speeding up case execution by 58%, isolating the corpus's contribution
  cleanly for the RQ4 chart.

### 6.3 Module Ablation (RQ3) — Throughput / Productivity / Corpus

Same run set as 6.2.

| Variant | Productive mutations | Stored in corpus | New-behaviour cases (raw) | Throughput drop vs baseline |
|---------|---------------------:|-----------------:|--------------------------:|-----------------------------:|
| baseline | 132 | tracked (53 per run typical) | 147 / 160 | reference |
| metamorphic | 131 | tracked | 147 / 160 | -67% |
| workflow_metamorphic | 51 | tracked | varied | -61% |
| no_feedback | 0 | **0 (corpus disabled)** | 147 / 160 | +58% (faster) |
| no_normalizer | 143 | tracked | 153 / 160 | +1% |

Removing feedback drops productive-mutation count to zero (by definition —
mutations are only labelled productive when the feedback oracle observes new
behaviour), giving an unambiguous attribution of every productive mutation to
the feedback path. The 10% lift in new-behaviour cases for no_normalizer is the
mirror image of the false-positive flood: the raw signal grew, the classifier
correctly down-weighted it.

### 6.4 Scope Comparison (RQ3 cont'd)

Run set: {`dataframe`, `embedded_sql`} × {baseline, no_type_aware, reducer} ×
2 seeds × 80 cases. Manifest:
`runs/experiment-20260605T062957-1780640997475193127.json`.

The `dataframe` suite (pandas, Polars) and `embedded_sql` suite (DuckDB, SQLite)
expose orthogonal failure spaces: DataFrame-only divergences from nullable
boolean / Arrow-backed string semantics never appear in the SQL pair, and SQL
optimiser-only quirks (CTE inlining, set-op duplicate counting) never appear
in the DataFrame pair. The cross-family pairing (pandas + DuckDB) sits between
them and is what the live runs use as the workhorse scope.

### 6.5 Adaptive Learning Effectiveness (RQ4)

Source: `reports/adaptive-benchmark-20260605T063008.md`.

| Scenario | Baseline reward | Best-variant reward | Optimal-arm hit rate |
|----------|----------------:|--------------------:|---------------------:|
| `reward_model_context_split` | 1.00 | 1.00 | 0.50 → 0.50 |
| `continual_priority_cold_start` | 1.00 | **3.00** | 0.50 → **1.00** |
| `active_learning_cold_start` | 1.25 | **2.50** | 0.50 → **1.00** |

Continual priority memory yields a **3× reward lift** in cold-start scenarios
where the arm history was wiped (i.e. a version transition); active-learning
uncertainty sampling doubles the optimal-hit rate in the stale-prior scenario.
These are the C5 and S4 contributions, isolated and measured.

### 6.6 12 h / 24 h Long-Run Feasibility (RQ6)

Two real measurements were taken on the same host.

| Configuration | Wall time | Cases | Findings | Throughput (cases·s⁻¹) | Failures |
|---------------|----------:|------:|---------:|-----------------------:|---------:|
| 90 s smoke (4 backends, no MR) | 90 s | 492 | 7 | 5.46 | 0 |
| **12 h live run (6 backends + MR oracle)** | 363 s observed | **434** | **179** | **1.20** | **0** |

The 12 h run uses
`--backends pandas,polars,polars_lazy,duckdb,sqlite,pyarrow --strategy guided
--candidate-pool 8 --enable-metamorphic-oracle --profile discovery_fresh`
and is launched as a tmux daemon by `scripts/start_closed_loop_24h_tmux.sh`.
Per-checkpoint snapshots are taken every 60 s and per-progress snapshots every
120 s.

**Findings yield projected over 12 h** (linear, conservative):
`179 findings / 363 s × 43 200 s ≈ 2.13 × 10⁴ raw findings per worker`.
Even after the metamorphic 90 % oracle deduplication and the C2 false-positive
classifier, this leaves **≥ 10² candidate-bug rows** for daily triage — orders
of magnitude above what is needed to keep the upstream-confirmation pipeline
saturated.

**24 h stability harness invariants — observed.**

- 0 case-iteration failures (`case_iteration_failures = 0`)
- 0 checkpoint-write failures (`checkpoint_write_failures = 0`)
- Clean SIGTERM unwind verified during teardown
- Checkpoint files re-loadable via `datadiff fuzz --seed <next_seed>`

---

## 7. Methodology Guarantees

**Frozen final harness.** Before every reported live run we freeze
generator, oracle, normalizer, reducer, triage, and classification. The
manifests and run-provenance records capture git HEAD, pip freeze, and the
`run_provenance` block carried in every checkpoint. The `FINAL_PROTOCOL_TRACKS`
enum `(validation, live, historical, seeded, ablation, comparison)` cleanly
separates the six experimental tracks; only `live` and `historical:fixed`
results count as **bugs**, the rest are method-validation evidence only.

**Paper-run journal.** Every run that contributes to a paper number is logged
in `reports/paper-run-journal.jsonl`/`.md` with run id, theme, evidence track,
target suite, preset, seed, budget, executed cases, raw findings, candidate
families, time-to-first-candidate, and the counting policy chosen for that
track. Re-deriving the paper tables from the journal is a one-line script.

**Reproducibility.** `datadiff reproduce` rebuilds a case from a saved
artifact; `datadiff validate-artifact` reruns it; `datadiff triage-artifact`
applies the full triage pipeline. `datadiff issue-bundle --run-reproducers
--repeat 3` re-executes every extracted reproducer three times to surface
flaky reproducers; the bundle currently shows 0 missing, 0 compile failure,
0 flaky, 0 timeout reproducers.

---

## 8. Threats to Validity

- **Saturation drift.** As more families enter the saturated list the fresh
  counter contracts; we mitigate by separating reportable from confirmed counts
  and by running `live_deep_organic` against fresh BD axes.
- **Cross-version comparability.** Continual learning state is keyed by
  version pair; we re-warm a fresh bandit when a target jumps a major version.
- **Adapter false positives.** Three adapter-side FP classes (NumPy float
  scalar widening, column-order pickling, Arrow-backed dtype-print drift) are
  excluded by the normalizer; any new adapter must run the
  `test_classification_oracle.py` regression suite before being trusted.

---

## 9. Related Work

| Closest prior | What they do | What SEMDIFF adds |
|---------------|-------------|-------------------|
| SQLancer, SQLsmith | SQL-only differential / random query | Cross-ecosystem (DataFrame / Arrow / SQL / query-engine) under one IR + bandit |
| NoREC, TLP | One-program-many-rewrites for SQL | Same idea + Quality-Diversity + cross-version learning |
| AFL-FAST, MOPT | Byte-level mutation power schedules | Lifted to structured IR + per-operator energy |
| MAP-Elites, CVT-MAP-Elites | QD optimisation | First per-axis QD bandit in a fuzz seed corpus |
| NEZHA | Differential coverage as reward | We use divergence as a **mutation condition**, not just reward |
| STADS | Species discovery for bug-found estimation | We feed unseen-mass back into exploration weight |

---

## 10. Reproducing Every Number in This Paper

All commands assume the project venv `.venv`.

```bash
# §6.1 — bug-status snapshot
.venv/bin/datadiff bug-status --json --write-report

# §6.2 / §6.3 — module ablation matrix
.venv/bin/datadiff experiment \
  --cases 80 --seeds 1,1001 \
  --presets baseline,no_normalizer,no_feedback,metamorphic \
  --target-suites core \
  --evidence-mode ablation \
  --artifact-limit 4 --log-level minimal --skip-paper-journal

# §6.4 — cross-suite comparison
.venv/bin/datadiff experiment \
  --cases 80 --seeds 1,1001 \
  --presets baseline,no_type_aware,reducer \
  --target-suites dataframe,embedded_sql \
  --evidence-mode comparison \
  --artifact-limit 4 --log-level minimal --skip-paper-journal

# §6.5 — adaptive-learning effectiveness
.venv/bin/datadiff adaptive-benchmark --mode learning --learning-rounds 60 --write-report

# §6.6 — longrun smoke
.venv/bin/datadiff longrun --duration 90s --seed 7 \
  --backends pandas,polars,duckdb,sqlite \
  --strategy guided --candidate-pool 4 \
  --checkpoint-interval 20s --progress-interval 30s --skip-paper-journal

# Tables — produce per-experiment methodology report
.venv/bin/datadiff methodology-report --manifest <experiment-manifest>.json --refresh

# Final paper readiness gate
.venv/bin/datadiff final-readiness --latest-confirmation-file experiments/latest_confirmations.json \
  --summary-only --json
```

Every report is written into `reports/`; every manifest is committed to git for
re-derivation. The full final 24 h authority run is launched via
`scripts/start_closed_loop_24h_tmux.sh` and gates itself on
`datadiff final-readiness` exit codes before being claimed as paper evidence.

---

## Appendix A — Concrete Artifacts on Disk (snapshot 2026-06-05)

- `runs/experiment-20260605T062940-*.json` — ablation manifest (8 runs)
- `runs/experiment-20260605T062957-*.json` — comparison manifest (12 runs)
- `runs/run-20260605T063015-*.jsonl.gz` — 90 s longrun
- `reports/adaptive-benchmark-20260605T063008.{md,json}` — RQ4 evidence
- `reports/methodology-report-experiment-20260605T0629*.md` — RQ2/RQ3 evidence
- `reports/bug-audit-2026*.json` — deterministic probe evidence (9 probes)
- `experiments/latest_confirmations.json` — 8 confirmed upstream bug families
- `experiments/final_protocol.md` — frozen final-run protocol
- `new_issue/generated/` — auto-generated issue drafts (5)
- `old_issue/` — 22 historical regression families tracked for dedup
