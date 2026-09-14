# DataDiffFuzz / SEMDIFF — Paper & Code Completion Plan

Status date: 2026-09-14 (Asia/Shanghai)
Role of this document: planning and traceability only. It is **not** an execution authority,
not a test receipt, and not a gate result.

## 0. Decisions taken (user, 2026-09-14)

| Decision | Choice |
| --- | --- |
| Paper framing | **A+B fusion**: unified cross-ecosystem semantic differential fuzzing framework as the spine, compositional semantic HyperContract as the oracle core. |
| Priority | **Parallel**: paper skeleton + experiment matrix first, while closing the critical code gaps. |
| Compute | **Long runs allowed**: background multi-hour to 24 h campaigns and multi-backend executions are permitted on this machine. |
| Venue | **ICSE/FSE reviewing standards**, no fixed deadline. English submission. |

## 1. Authoritative baselines

| Role | Location | State |
| --- | --- | --- |
| Authority archive | `/data1/lbw/xjx/phase6_execution_authority_artifacts_20260726` | 6.5 GB, immutable mirror |
| Frozen source (canonical) | `.../phase6_formal_target_version_producer_r1.YxlpWU/source` | git HEAD `a87d380`, 1924 files, clean |
| Latest forward commit | `edf69f6` (v3 run receipt builder + provenance bridge) | on branch `p6-v3-receipt-core` |
| **Dev worktree (this plan)** | `.../worktrees/wt-paper` (branch `p6-paper-dev`) | off `edf69f6`; paper lives in `paper/` |
| Frozen backend env | `/data1/lbw/xjx/datadiff_fuzz_lab/.venv` | pandas 3.0.3, polars 1.42.1, duckdb 1.5.4, pyarrow 25.0.0, datafusion 54.0.0, chDB 4.2.1 |
| Confirmed roots | `.../source/bugs/`, `experiments/latest_confirmations.json` | 9 strict dirs; 11 confirmation records |
| Protected user workspace | `/data1/lbw/xjx/datadiff_fuzz_lab` | bulk evidence dirs emptied by user; **not** authority source |

Run tests without installing:

```bash
cd .../worktrees/wt-paper
PYTHONPATH="$PWD/src:$PWD" /data1/lbw/xjx/datadiff_fuzz_lab/.venv/bin/python -m pytest -q
```

Historical `/tmp/phase6_*` paths are satisfied by symlinks into the archive (see
`paper/experiments/ENV_REBIND.md`, created in W1).

## 2. Two known blockers (must be fixed, not worked around)

1. **Interpreter drift.** Authorities bind `interpreter_sha256 = 1643dacd…`; the host
   `/usr/bin/python3.12` is now `e50d468e…`. P6-01/P6-05 authority-bound runs fail closed.
   Fix = W1 rebind round producing a **new** authority with the current interpreter hash,
   keeping the old documents as immutable history. Never edit a frozen SHA.
2. **Lost experiment manifests.** The snapshot contains 0 `runs/experiment-*.json` and lacks
   the reports cited by `PAPER.md` §6. Tables are therefore not reproducible. Fix = W3
   re-runs the experiment commands under frozen seeds, or W1 reconstructs manifests from
   surviving `run-20260726-*` artifacts where legitimate.

## 3. Reconciled contributions

`docs/novelty_claims.md` forbids over-broad "first" claims. The defensible set is:

| ID | Contribution | Strength | Evidence status |
| --- | --- | --- | --- |
| **C1** | Unified cross-ecosystem testing surface: one typed workflow IR + capability-aware lowering over DataFrame APIs, Arrow compute, embedded SQL and query engines. | **Primary** | Code complete; needs RQ6 reuse/capability table |
| **C2** | Semantic normalizer + compositional multi-oracle: differential + metamorphic + deterministic probes fused per case with verdict confidence and root-cause classification. | **Primary** | Code complete; quantitative tables must be regenerated |
| **C6** | End-to-end latest-version bug-evidence pipeline: candidate → recheck → native candidate → issue-ready root → confirmed root, with fresh/replay physical separation. | **Primary** | 9 strict roots survive; readiness reports lost, must be regenerated |
| **C7** | Compositional semantic HyperContract: forward-property + backward-observability compilation, derivation/applicability/observation certificates, finite multi-endpoint contracts, target lattice + coverage-debt scheduling. | Oracle core (phase6) | Pre-implementation review only; gates unrun |
| C3 | Multi-scope contextual bandit scheduler | Supporting | Ablation tables lost |
| C4 | Quality-diversity seed archive (7-axis BD, hierarchical MAP-Elites, MinHash dedup) | Supporting | Design + unit tests only |
| C5 | DataFrame-IR mutation operators (subtree rewrite, divergence-conditioned scoring, value catalog) | Supporting | No isolated ablation |

**Crosswalk policy.** The 18 M/S/Q flags in `SEMDIFF_METHODOLOGY.md` are *mechanisms*, not
contributions; they map into C1–C7. No paper claim may assert "first" for fuzzing,
differential testing, metamorphic testing, or DataFrame testing.

## 4. Research questions and experiment matrix

| RQ | Question | Treatment vs control | Primary metric | Budget | Status |
| --- | --- | --- | --- | --- | --- |
| **RQ1** | Can the framework find reproducible, upstream-confirmable latest-version bug families? | full system, fresh seeds | confirmed families; strict survivors / 10k cases; time-to-first | ≥3 fresh seed blocks | needs re-run + confirmation audit |
| **RQ2** | Does semantic normalization + classification control false positives? | `full` vs `no_normalizer` vs `no_classification` | FP rate; expected-divergence rate; post-classification survivor count | paired seeds, ≥3 blocks | code exists; manifests lost |
| **RQ3** | Do differential / metamorphic / deterministic probes complement each other? | oracle-source ablations, union/intersection | families unique to each oracle; Jaccard | paired seeds | code exists; needs instrumentation |
| **RQ4** | Do typed generation + guidance + coverage-debt scheduling beat weaker baselines? | `p8_candidate_v1` vs random/unguided vs no-feedback | candidate families/hour; valid-case ratio; time-to-first | paired seeds, ≥12 h/cell | code exists; manifests lost |
| **RQ5** | Does the evidence pipeline turn findings into submission-ready evidence? | pipeline on vs audit-only | reproducer success; reduction ratio; bundle completeness; readiness | per-candidate | 9 roots; regenerate report |
| **RQ6** | Is the same surface reusable across backend families without per-backend ad-hoc scripts? | reuse/capability accounting | workflow reuse rate; capability coverage; adapter LOC/time | static + runtime | needs new measurement |
| **RQs (C7)** | Does the HyperContract oracle satisfy correctness/efficiency gates at scale? | OSC gate suite | verdict discrepancy=0; mutant kill=100%; p95 ≤2 ms or ≤5% wall | P6-03/04/05 | unrun; depends on W2 |

Baselines: SQLancer-family comparison (`src/datadiff/sqlancer_*`, `docs/sqlancer_pqs_transfer_plan.md`)
plus random/unguided and no-feedback arms. Statistics: paired seed blocks, bootstrap 95% CI,
effect size; report negative and zero-yield outcomes.

## 5. Workstreams

### W1 — Reproducibility rebind
- New canonical environment snapshot binding current interpreter + exact package versions.
- `paper/experiments/ENV_REBIND.md` documenting `/tmp` symlinks, hashes, and the drift.
- Restore/regenerate missing `experiment-*.json` manifests; full suite green.

### W2 — Code completion (dependency order)

Structural fact (verified): the dynamic gate plan requires all 18 sections
(`_phase6_gate_authority.py:190,1452`) and the receipt index requires exactly the 10 producer
kinds (`:500-513,3631,4474`); partial evaluation is never authority (`gates.py:177,532`).
**No gate can go green in isolation** — the minimal path to *any* green gate is a complete
10-kind index. Cheapest first gate is **runtime-correctness** (2/10 kinds already have real
retained evidence). Full detail: `paper/research/CODE_GAP_REPORT.md`.

| ID | Task | Files | Verify | Effort |
| --- | --- | --- | --- | --- |
| T1 | E1: replace unconditional fail-close + context binding module | `semantic_replay.py:80-93,1299-1300` | negatives still fail closed; real `FormalLanePlan` passes | 5–8 d |
| T2 | C3: register context router + `ContextReplayResult`→tuple adapter | `_phase6_gate_authority.py:689-707,1990-2022`; `context_replay.py` | 25 envelope pairs stay single-owner | 1–2 d |
| T3 | C1: coverage-event producer (`LedgerEvent` + `CoverageLedger.admit`) | new `runtime/_phase6_coverage_event_producer.py` | replay + credits == universe | 2–3 d |
| T4 | C2: generalize reachability binding + driver | `_phase6_reachability_execution_binding.py:310,319` | 232/384 constructed+activated | 3–5 d |
| T5 | C4: raw→typed rebuilders | `_producer_provenance.py:1010-1034`; new provenance modules | rebuild == typed | 2–3 d |
| T6 | v3 merge: cherry-pick `edf69f63` + v3 rebuilder key + exemptions | `_phase6_gate_authority.py`, `_producer_provenance.py` | v3 admitted | 1 d |
| T7 | v3 hot-path typed capture | campaign runner + capture module | 2,200 cases, 0 pipeline errors | 3–5 d |
| T9 | E2: frozen family→operator mutation policy (9 capturable operators) | `context_receipts.py:2018-2073`, `mutator.py` | ≥90% overall, ≥80%/family | 3–5 d |
| T10 | E5: typed unsupported-obligation producer | `context_receipts.py`, `semantic_replay.py:1180-1190` | unsupported admits | 2–3 d |
| T11 | E4: atom-extraction instrumentation + frozen 5%-of-wall decision | gate/contract modules | decision recorded | 1–2 d |
| T12 | Contract corpus (≥100k decisions, ≥1k samples) + exact-replay bridge | new corpus producer | real admission | 5–8 d |

Order: `T1 ∥ T3 → T2 → T4/T5 → T6 → T7 → T9/T10/T11 → T12`. Each item is a contract-scoped
change with its own tests; authority documents are regenerated, never edited.

### W3 — Experiments
Regenerate RQ1–RQ6 tables; run C7 gate suites; keep fresh/replay physically separated;
persist raw transcripts and receipts.

### W4 — Paper
Reconcile all numeric claims; write English submission; artifact package + reproduction script.

## 6. Paper structure (target: ICSE/FSE research track)

1. Introduction — problem, gap, thesis, contributions C1/C2/C6 (+C7 as oracle core).
2. Background & Related Work — SQLancer family, FuzzyData, EET/CODDTest/QTRAN, NOETHER/SemConT.
3. Approach — typed IR & capability-aware lowering; semantic normalizer; compositional oracle
   (differential/metamorphic/probes + HyperContract); search & scheduling; evidence pipeline.
4. Implementation — target registry, adapters, determinism, counting contract.
5. Evaluation — RQ1–RQ6, baselines, ablations, threats to validity.
6. Discussion / Lessons / Limitations.
7. Related work (full), Conclusion, Artifact appendix.

## 7. Non-negotiable evidence rules

1. No frozen SHA, comparator, tolerance, denominator, seed rule or gate is relaxed to make a
   result pass.
2. Fresh discovery, regression, known-root, saturated and issue-inspired evidence stay
   physically and logically separated.
3. `candidate_confirmed` and `bug_claimed` stay `false` until native reproduction + independent
   confirmation complete.
4. Every authoritative run uses frozen source, the exact target environment, fresh allowed
   seeds and durable raw evidence.
5. Failed or zero-yield runs are preserved and reported, never silently rerun.

## 8. Risks

| Risk | Response |
| --- | --- |
| Environment drift (already realized) | W1 rebind; pin interpreter copy inside the archive |
| Lost manifests make paper tables unverifiable | W3 regenerate under frozen seeds; mark any unverifiable claim |
| Engineering is the schedule (est. 4–6 weeks for C7 gates) | W2 dependency order; ship C1/C2/C6 paper even if C7 gates lag |
| Bug-count weakness | Lean on RQ2/RQ5/RQ6 methodology narrative per `novelty_claims.md` |
| Overclaim vs reviewers | Enforce the crosswalk policy and "no first" rule |
