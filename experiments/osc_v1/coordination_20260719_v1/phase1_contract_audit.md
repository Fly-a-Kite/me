# OSC-1-A Contract audit

Status: complete, read-only. No files changed. No pilot or long run started.

Validation: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest -q -p no:cacheprovider tests/osc/test_contract_*.py` -> `66 passed in 0.23s`.

## Evidence matrix

| Requirement | Classification | Evidence |
| --- | --- | --- |
| two-pass compiler | partial | `forward.py:forward_analyze`, `demand.py:backward_analyze`, `compiler.py:compile_hypercontract`; backward rules do not consume node forward state and candidate obligations do not drive relation synthesis |
| four-valued verdict / failures | partial | `model.py:VerdictKind`, `monitor.py:monitor_exact`; applicability certificate binding and structured denominators are not validated |
| three certificates | contradictory | classes exist in `derivation.py`, `applicability.py`, `evidence.py`, but empty/unbound certificates can report valid and are not connected to an observed ledger |
| capability / permission | partial | separate types exist in `capability.py` and `overlays.py`; semantic permission is not consumed by compiler/monitor/planner |
| entailment | contradictory | `hypergraph.py:EntailmentGraph` accepts `a -> b -> a`; planner does not consume entailment; `numeric_tolerant` is incorrectly marked transitive |
| staged planner | contradictory | `planner.py` defines S0-S4 and clustering, but malformed partial-order parameters produced staged `SATISFIED` versus exact `INCONCLUSIVE` |
| v1 facade | contradictory | `compatibility.py` is narrow, but legacy v1 remains production authority and no production module imports OSC |
| fault mutants | partial | palettes/helpers exist in `mutations.py`; tests do not execute or kill the authority-path mutants |
| 100k differential | missing | no runner/artifact; a small staged/exact counterexample already exists |
| p95 / throughput | missing | no OSC benchmark or paired evidence |

## Reproduced authority counterexamples

```text
numeric_exact_distinct_large_ints=SATISFIED
partial_order_empty_params_staged=SATISFIED
partial_order_empty_params_exact=INCONCLUSIVE
cyclic_entailment_graph_accepted=(('a','b'),('b','a'))
empty_contract_exact=SATISFIED
restricted_contract_with_old_cert=SATISFIED
empty_derivation_valid=True
unbound_observation_certificate_valid=True
bytes_observation_digest=TypeError
direct_observation_nested_mutation_changes_digest=True
```

Blockers: typed exact comparison, certificate/contract binding, relation parameter validation, acyclic entailment, staged/exact soundness, legacy authority migration, and every mandatory corpus/performance gate.

Change requests: root-owned canonical tagged scalar and deep immutability; root public API/evidence tier/failure schemas; shared atom/ledger/compiler input; root-owned legacy v1 facade integration.

Completion report invariants: `allowed_write_paths=[]`, `changed_files=[]`, `forbidden_paths_modified=[]`, `deviations=[]`, `twenty_four_hour_run_authorized=false`.
