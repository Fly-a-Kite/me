# OSC-1-C Runtime audit

Status: complete, read-only. No files changed. No pilot or long run started.

Validation: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest -q -p no:cacheprovider tests/osc` -> `66 passed in 0.22s`.

## Evidence matrix

| Requirement | Classification | Evidence |
| --- | --- | --- |
| staged comparison | partial | local prototype in `contract_engine/planner.py`; not connected to execution/runner and lacks parity evidence |
| component DAG | partial | `PlanNode`/`ComparisonPlan` are descriptive; execution does not execute the DAG |
| O(B) clustering | contradictory | dictionary insertion is O(B), but returned clusters are sorted and no pairwise/scaling equivalence test exists |
| cache identity/isolation | partial | legacy key omits explicit HyperContract/canonical versions; fresh/native cache denial is not integrated |
| deterministic task DAG | missing | no immutable task IDs, dependencies, keyed lineage or frozen assignment |
| resource tokens | missing | legacy scalar cost/RSS recycling is not CPU/RSS/I/O/internal-thread/exclusive-state accounting |
| epoch barrier | missing | no frozen epoch decision or deterministic merge |
| worker failure/retry | partial | reusable process/retry patterns exist, but results preserve completion order and no retry invariance proof exists |
| failure taxonomy | contradictory | OSC can encode statuses, but monitor/certificate and legacy execution collapse distinctions |
| workers 1 versus N | unverified | no OSC assignment/case/coverage/verdict/evidence equality audit |
| machine-readable gates | missing | legacy readiness lacks OSC exactness, 232/384/502, determinism and authority checks |
| 100k/scaling | missing | no OSC benchmark/artifact |
| post-refactor 2200 gate | missing | old 22/22, 2200/2200 artifact is diagnostic only and activation was 533/1929 |
| 2h pilot tooling | missing | no preregistered OSC paired pilot implementation |

Legacy components are implementation inputs only. No production module outside OSC tests imports `datadiff_osc`.

Critical/high risks: legacy 24h wrapper defaults missing readiness evidence to non-fatal; timeout/crash/domain/unsupported taxonomy is lost; legacy exceptions map to `missing/error`; runner advances seed after failures; no resource isolation; hash-equal screening lacks mandatory promotion evidence.

Change requests: root-owned runtime/task/evidence/cache schemas; root-owned BackendResult-to-Observation facade; fail-closed gate manifest and launch preflight.

Completion report invariants: `allowed_write_paths=[]`, `changed_files=[]`, `forbidden_paths_modified=[]`, `deviations=[]`, `candidate_status=not_assessed`, `twenty_four_hour_run_authorized=false`.
