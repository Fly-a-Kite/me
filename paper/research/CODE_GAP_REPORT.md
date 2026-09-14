# Phase 6 Gate Verification & Code Work-Breakdown

Source: independent read-only research pass, 2026-09-14, frozen HEAD `a87d380`
(dev worktree `wt-paper` = `edf69f6`). No files were modified by that pass.

## 0. Frame correction

`phase_6_contract` / `phase_6_granularity_reachability` / `phase_6_runtime_correctness`
are **coordination-ledger status labels, not code symbols** (0 matches in `src/`).
They are `pending` at `coordination_ledger.yaml:2863-2865`, defined at `:3110`, `:3275`,
`:3295`. The real artefacts are:

- `verify_dynamic_gate_plan` — `_phase6_gate_authority.py:1403-1663`
- `verify_artifact_receipt_index` — `:2855-3666`
- `build_phase6_gate_authority_bundle` — `:4497-4828`
- gate calculator — `runtime/gates.py:209-254`

**Structural fact:** the dynamic plan requires all 18 sections (`_DYNAMIC_RECORD_FIELDS`
`:190`, loop `:1452`); the index requires exactly the 10 producer kinds
(`_REQUIRED_PRODUCER_KINDS` `:500-513`; missing check `:3631`; publishable requires
`set(kinds) == _REQUIRED_PRODUCER_KINDS` `:4474`). `evaluate_gate_specs` is never
authority (`gates.py:177,532`). **No gate can go green in isolation.**

## 1. Gate status and blocking gaps

### (a) contract — pending
Sections: `contract_comparison_decisions` (≥100,000, `:1468`), `false_positive_fixes`,
`contract_performance_samples` (≥1,000). Kinds: `contract_exact_replay`,
`performance_paired_replay`.
- No real corpus executor (`scripts/osc/audit_staged_parity.py` self-declares ineligible).
- `contract_exact_replay` has **no execution-provenance bridge**; the pending-provenance
  branch (`_phase6_gate_authority.py:2029`, error id `:2073`) exempts only 4 candidates
  (`:2036-2070`), so every clean `ContractComparisonReceipt` fails
  `typed_admission_root_provenance_pending_phase6`.
- Thresholds frozen at `gates.py:213-218` (9/9 recall, ≥1 FPF/mutant, 100k, 2 ms/5%, ≤10%).

### (b) granularity/reachability — pending
Kinds: `coverage_admission_replay`, `reachability_admission_replay`, `focus_admission_replay`.
1. **E1 (real, 5–8 d).** `replay_search_admission` unconditionally fail-closes 10 subjects:
   `semantic_replay.py:80-93`, branch `:1299-1300`. These are the denominators at
   `gates.py:225,226,229,230,231,232,233` plus `scheduled_target_attempts`/`mutation_attempts`/
   `formal_lanes`/`focus_signals` (`gates.py:234,235`). Escape hatch
   `_ROOT_CONTEXT_PENDING_SUBJECTS` lists only `staged_parity_groups`
   (`_phase6_gate_authority.py:709-714`). CR-OSC-6-003A named at `semantic_replay.py:135`.
2. **C3 (real, 1–2 d).** Router registers only contract/search/runtime
   (`_phase6_gate_authority.py:689-701`, owner guard `:706-707`).
   `replay_context_admission` (`search/context_replay.py:966`) is imported only by tests —
   dead production code. Return type `ContextReplayResult` (`context_replay.py:82`) is not
   the `tuple[str,...]` the dispatcher expects (`_phase6_gate_authority.py:2000-2022`),
   so wiring needs an adapter.
3. Four subjects remain context-required after C3: `context_receipts.py:3054-3071`
   (`scheduled_target_attempts`, `mutation_attempts`, `observation_contexts`, `focus_hits`;
   guard `context_replay.py:1012-1019`).
4. **C1 (real, 2–3 d).** No production code constructs `LedgerEvent` (`schemas.py:451-491`)
   or calls `CoverageLedger.admit` (`search/ledger.py:309-358`); tests only.
5. **B2/E3 (real, 2–3 d).** `_phase6_reachability_execution_binding.py:310` requires exactly
   one unparameterized cell; `:319` "cannot bind contrast edges" → cannot scale to 384 edges.
6. **E5/C6 (real, 2–3 d).** `ExecutionStatus.UNSUPPORTED` +
   `unsupported_evidence_digest` exist (`schemas.py:73,247,266`; replayed
   `runtime/_semantic_replay.py:212,226-228`) but gate numerator
   `backend_pair_obligations_observed_or_unsupported` (`gates.py:224`) has no typed producer.
7. **B3/E2 (real, 3–5 d).** Mutation snapshot recomputes the full callable dependency graph
   (`context_receipts.py:2018-2073` → `_code_dependency_specs` `:984-1020`); `getattr` banned
   (`:824-834`, raise `:1003-1006`). ~74 operator profiles (`mutator.py:4811`).
8. **E4 (1–2 d).** Budget defined (`gates.py:217`; `contract_engine/gates.py:250-251`;
   TODO.md:57) but no instrumentation/decision.

### (c) runtime-correctness — pending
Kinds: `v3_typed_run_replay`, `repository_test_replay`, `target_version_replay`,
`candidate_classification_replay`, `parallel_authority_replay`, `performance_paired_replay`.
**2 of 6 already have real retained evidence** (repository R4/R5, target R1/R2 — ledger
`:3780-3790`, `:3871-3890`).
- No `build_v3_run_receipt` at HEAD — builder + provenance live only on **unmerged commit
  `edf69f63`** (`p6-v3-receipt-core`, 1,650 insertions).
- Hot-path typed capture absent.
- `_RAW_TYPED_REBUILDERS` has one entry (`_producer_provenance.py:1010-1034`, lookup `:1047`).
- Exemptions missing for `v3_typed_run_replay`/`candidate_classification_replay`
  (`_phase6_gate_authority.py:2036-2070`).
- Parallel producer `_phase6_parallel_authority_producer.py` untracked in `wt-parallel-authority`.

## 2. Dependency-ordered tasks (each needs a task contract first)

| ID | Task | Files | Verify | Effort |
| --- | --- | --- | --- | --- |
| T1 | **E1** replace unconditional fail-close + context binding module | `semantic_replay.py:80-93,1299-1300` | existing negatives still fail closed; real `FormalLanePlan` passes; full `tests/osc` | 5–8 d |
| T2 | **C3** register context router + `ContextReplayResult`→tuple adapter | `_phase6_gate_authority.py:689-707,1990-2022`; `context_replay.py` | 25 envelope pairs stay single-owner | 1–2 d |
| T3 | **C1** coverage-event producer; construct `LedgerEvent`, call `admit` | new `runtime/_phase6_coverage_event_producer.py` | replay + credits == universe | 2–3 d |
| T4 | **C2** generalize reachability binding + driver | `_phase6_reachability_execution_binding.py:310,319` | 232/384 constructed+activated, edges observed | 3–5 d |
| T5 | **C4** raw→typed rebuilders | `_producer_provenance.py:1010-1034`; new `_phase6_{reachability,coverage}_provenance.py` | rebuild == typed | 2–3 d |
| T6 | **v3 merge** cherry-pick `edf69f63` + v3 exemptions + rebuilder key | `_phase6_gate_authority.py`, `_producer_provenance.py` | v3 admitted | 1 d |
| T7 | **v3 hot-path capture** | campaign runner + new capture module (canonical Case bytes, `SeedLineage`, `TaskSpec`, `ResultGroup`, `EvidenceEnvelope` + 4 cert digests) | 2,200 cases, 0 pipeline errors | 3–5 d |
| T9 | **E2** frozen family→operator policy over 9 capturable operators | `context_receipts.py:2018-2073`, `mutator.py` | ≥90% overall, ≥80%/family | 3–5 d |
| T10 | **E5** typed unsupported obligation | `context_receipts.py`, `semantic_replay.py:1180-1190` | unsupported admits | 2–3 d |
| T11 | **E4** atom-extraction instrumentation + frozen 5%-of-wall decision | gate/contract modules | budget decision recorded | 1–2 d |
| T12 | **contract corpus** | new corpus producer; ≥100k decisions, ≥1k samples, `contract_exact_replay` bridge | real admission | 5–8 d |

Order: `T1 ∥ T3 → T2 → T4/T5 → T6 → T7 → T9/T10/T11 → T12`.

## 3. Genuine code vs bookkeeping

**Genuine code:** E1, C3 router+adapter, C1, C2, C4, v3 hot-path capture, mutation policy,
unsupported schema, budget instrumentation, contract corpus, relaxing
`_phase6_reachability_execution_binding.py:310,319`.
**Written but unmerged:** v3 builder/provenance `edf69f63`; parallel producer (untracked).
**Bookkeeping/authority only:** authoring the plan+index (P6-02d) once producers exist
(verifiers frozen); seed-exclusion list is declarative except runner encoding;
`latest_target_baseline` decision; `performance_paired_replay` provisional exemption policy.

## 4. Minimal path to one green gate

The index needs exactly all 10 producer kinds (`:3631`), the plan all 18 sections (`:1452`),
and partial evaluation is never authority (`gates.py:177,532`). Therefore **there is no
gate-specific partial path**: the minimal path to *any* green gate is a complete 10-kind index.
Cheapest first gate is **runtime-correctness** (2/10 kinds already real; 2 more written on
unmerged branches). Engineering ≈ 4–6 weeks (T1 longest pole); compute ≈ 8–20 h once built.
Nothing is green today.
