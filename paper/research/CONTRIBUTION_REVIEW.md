# Contribution & Positioning Review (frozen HEAD a87d380)

Source: independent read-only research pass, 2026-09-14. No files modified.

## 1. Core problem and thesis

Modern tabular stacks (pandas, Polars, PyArrow, DuckDB, SQLite, DataFusion, chDB) are used
as interchangeable executors of the same table semantics, but silently diverge on
null/NaN/negative-zero, sorting/top-k, groupby/join, casts, Arrow layout and optimizer
rewrites. Thesis: one typed workflow IR, capability-aware lowering, a semantic normalizer,
and a compositional oracle complex (differential + metamorphic + deterministic probes) under
an adaptive multi-scope bandit and a QD seed archive can find upstream-confirmable
latest-version silent wrong-result bugs across four backend families, with an evidence
pipeline separating candidates from confirmed families.

## 2. Contributions, mechanisms, evidence

| ID | Contribution | Key code | Evidence | Missing |
| --- | --- | --- | --- | --- |
| C1 | Unified cross-ecosystem semantic testing surface | `ccs_ir.py`, `dsl.py`, `synthesis/*`, `capability_matrix.py`, `targets.py`, `adapters/`, `backends/` | design in `PAPER.md`; `reports/target-registry-ubuntu.json`; smoke runs | RQ6 reuse-rate, per-family capability coverage, adapter LOC/time |
| C2 | Semantic normalizer + multi-oracle complex | `normalizer.py`, `canonicalization.py`, `oracle.py`, `oracle_complex.py`, `oracle_rules.py`, `metamorphic.py`, `adjudication.py`, `classification_oracle.py` | `PAPER.md` §6.2 claims metamorphic ×3.7 yield; `no_normalizer` ×10.4 raw but ≤1.5× post-classification | cited manifest and report absent; relation count 42 vs 45 |
| C6 | Cross-version evidence pipeline + counting contract | `champion_corpus.py`, `adaptive_learning.py`, `case_policy.py`, `triage.py`, `issue_bundle.py`, `family_novelty.py`, `reducer.py`, `final_readiness.py`, `target_version_audit.py` | `PAPER.md` §2/§7; readiness doc claims 2,099 journal entries, 184,459 cases, 277 families, 9 confirmed | readiness report + all `experiment-*.json` absent; `latest_confirmations.json` = 11 |
| C7 | Compositional semantic HyperContract (phase6) | `datadiff_osc/contract_engine/*`, `contract_universe.py`, `semantic_contracts.py`, `obligation_coverage.py`, `coverage_debt.py` | pre-implementation review only; verdict `passed_for_low_complexity_core_implementation`, `24h_authorized=false` | E0–E2 unrun; 100k zero-discrepancy, 100% mutant kill, 3-certificate gates unchecked |
| C3 | Multi-scope contextual bandit scheduler | `decision_engine.py`, `bandit_selection.py`, `adaptive_learning.py`, `feedback.py`, `scheduler.py`, `energy.py` | `PAPER.md` §6.5 claims 3× cold-start reward | reports absent; graft/continual learning unproven |
| C4 | Quality-diversity seed archive | `quality_archive.py`, `behavioral_descriptor.py`, `fingerprint.py`, `disagreement.py`, `seed_quota.py`, `rust_kernel/src/lib.rs` | design + unit tests only | no isolated RQ4 table |
| C5 | DataFrame-IR mutation operators | `mutator_ir/*`, `mutator_shrink.py`, `value_catalog.py`, `energy.py`, `mutator.py` | blueprint metric definitions only | no ablation isolates M1–M6 |

## 3. Strongest defensible contributions (ranked)

1. **C1 unified cross-ecosystem surface.** No 2023–2026 paper covers Python API + Arrow
   compute + embedded SQL + query engine under one IR; verifiable from `targets.py`/`adapters/`.
   Needs only a reuse/adapter-cost table.
2. **C6 evidence pipeline + counting contract.** Methodology that DBMS-testing papers
   underdo; defensible even with a small bug count.
3. **C2 normalizer + multi-oracle.** Best quantitative story; must be re-run to regenerate
   absent manifests.

Weakest as independent contributions: C3, C4, C5, C7 — each ports a known mechanism
(MOPT, MAP-Elites, AFL-FAST, PQS/NOETHER) and lacks an isolating experiment.

## 4. Internal inconsistencies / overclaims

- Bug count diverges: `PAPER.md` abstract/§6.1 = 8; `icse_sota_success_criteria.md` = 9;
  `latest_confirmations.json` = 11; `PROJECT_IDEA_AND_PROGRESS.md` = 5; protocol TODO = 9.
- `PAPER.md` self-contradiction: §3.2 "9 probes"/"42 relations" vs §6.1 "12 probes"/"45".
- Innovation count: docs say 16, list totals 18.
- Readiness overclaim: `icse_sota_success_criteria.md` asserts `ready: true`, citing
  `reports/final-readiness-20260614T103547.json` and `reports/icse-sota-gap-audit-20260615.*`
  — neither exists in this snapshot.
- Unreproducible tables: `PAPER.md` §6.2–§6.6 cite `runs/experiment-*.json` and
  `reports/adaptive-benchmark-*.md`; `runs/` has 0 `experiment-*` files.
- 24h authorization conflict: readiness doc says matrix complete; `PAPER.md` §6.6 says 12 h;
  TODO launch boxes unchecked; OSC review sets `24h_authorized=false`; gate authority states
  it "can never authorize a 24-hour campaign".
- Novelty discipline violated: `novelty_claims.md` forbids "first" claims, yet
  `SEMDIFF_INNOVATION_BLUEPRINT.md`/`SEMDIFF_METHODOLOGY.md` Appendices assert 首次/完全原创.
- Unreconciled framing: 18 M/S/Q flags vs the 4 OSC contributions; never crosswalked.
