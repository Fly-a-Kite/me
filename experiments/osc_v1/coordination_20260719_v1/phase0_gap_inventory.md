# Phase 0 gap inventory

Run: `osc-20260719T144951+0800`

This is a pre-audit inventory, not an implementation verdict. Phase 1 must replace each uncertain item with file/symbol/test evidence.

## Present

- `src/datadiff_osc/contract_engine/` contains model, finite domains, forward and backward analysis, compiler, certificates/evidence, relations, planner, monitor, compatibility, mutation and hardening modules.
- `src/datadiff_osc/semantic_targets/` contains only `model.py`, `taxonomy.py`, `declarations.py`, and `compiler.py`.
- Shared package files `src/datadiff_osc/__init__.py` and `_canonical.py` exist.
- Four OSC contract test modules and `tests/osc/conftest.py` exist; the current repository-venv baseline is `66 passed`.
- The preimplementation credibility review and reproduction note already exist under `experiments/osc_v1/`.

## Structurally absent from the OSC package

- `src/datadiff_osc/generation/**`
- `src/datadiff_osc/search/**`
- `src/datadiff_osc/scheduler/**`
- `src/datadiff_osc/comparison/**`
- `src/datadiff_osc/parallel/**`
- `src/datadiff_osc/runtime/**`
- `scripts/osc/**`
- OSC target/generation/search/scheduler/comparison/parallel/runtime scoped tests

Legacy `src/datadiff/**` may contain reusable implementations, but it remains read-only to child agents and is not OSC authority without a root-owned facade and integration decision.

## Unverified hard requirements

- HyperContract: exact two-pass semantics, four-valued preservation, three certificate validity, capability/permission separation, relation entailment, exact escalation, v1 facade boundary, all mutant/fault gates, 9/9 root recall, false-positive precision, 100,000-group exact differential, and p95/throughput gates.
- Target/search: declaration-derived fresh 16/232 and regression 9/144 physical partition, 384 edges, 502 obligations, generic atom extraction/matching, bitmap ledger, keyed substreams, no-replacement epochs, typed backward construction, target-preserving mutation, and constrained scheduler.
- Runtime: contract-driven component DAG, O(B) clustering, complete cache key, deterministic task DAG/resource tokens/epoch barrier, worker failure taxonomy, worker-count invariance, gate report generation, 2,200-case v3 tooling, and 2h paired-pilot tooling.
- Integration: unique frozen public schemas for semantic atoms, target/contrast objects, endpoint/observation/contracts/certificates, seed lineage/tasks/results/evidence, ledger events, and staged comparison requests/results.
- All Phase 6 evidence remains absent or unverified; `twenty_four_hour_run_authorized` is therefore `false`.

## Dirty-worktree risk

The OSC source and test trees are currently untracked user work. No existing file may be treated as disposable scaffolding. Phase 1 is read-only. Before Phase 3, each writer must identify exact existing and new files, and root must compare them against the frozen Phase 0 snapshot.
