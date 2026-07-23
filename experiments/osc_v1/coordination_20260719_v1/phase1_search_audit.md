# OSC-1-B Search audit

Status: complete, read-only. No files changed. No candidate or bug discovery was run.

Validation: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest -q -p no:cacheprovider tests/osc` -> `66 passed in 0.22s`.

## Independently recomputed denominators

| Partition | Families | Cells | Baseline-star edges | Backend-pair obligations |
| --- | ---: | ---: | ---: | ---: |
| fresh | 16 | 232 | 384 | 502 |
| regression | 9 | 144 | 264 (not a frozen gate) | 228 |

Compiler output has `232` fresh cells, `144` regression cells, `384` fresh edges and `202` shadow tiles. Its single `CompiledTargetUniverse.backend_pair_obligations` collection contains `730 = 502 + 228`; this contradicts the unambiguous fresh 502 denominator.

## Evidence matrix

| Requirement | Classification | Evidence |
| --- | --- | --- |
| declarations/provenance | partial | `declarations.py:legacy_v4_target_templates` and `partition_templates`; partitions exist in memory, but builders remain legacy adapters and contamination checking is substring-based |
| atom taxonomy | partial | `taxonomy.py:AtomTaxonomy`; deterministic namespaces exist but known-root probe operations are not excluded |
| atom extraction | missing | no OSC extractor or provenance-bearing semantic atom model |
| compiler | contradictory | deterministic cells/edges are correct; unknown pipelines silently lower to no atoms and obligations mix fresh/regression |
| matcher/certificates | missing | only an activation certificate data class exists; no generic matcher or edge/pair observation matching |
| four-level ledger | missing | no constructed/activated/executed/observed bitmap, event, merge or replay implementation |
| keyed substreams | partial | legacy case-level seed derivation exists; OSC lineage is an unconstrained string and stages are not isolated |
| no-replacement epochs | missing | no keyed permutation or invariance tests |
| typed backward synthesis | partial | typed IR/state primitives exist; generation remains forward/family-specific with no bounded infeasibility result |
| target-preserving mutation | partial | legacy affinity/repair exists but no re-extract/match proof for selected cells |
| max-min/Pareto scheduler | contradictory | legacy schedulers use scalar utility/UCB; no worst-group floor or bounded Pareto selection |
| reachability/performance gates | unverified | no OSC evidence or benchmark implementation exists |

Reusable only as primitives: CCS IR, program state/operation semantics, dense bitmap storage, case-level seed primitive, typed capability/source concepts and deterministic batch patterns. Legacy family evaluators, goal repairs, scalar schedulers and metadata activation remain shadow/parity only.

Critical risks: circular activation proof from assignment metadata; 730/502 denominator drift; fresh provenance pollution through legacy builders/probes; stage/worker seed drift; worst-family starvation.

Change request: root must freeze provenance-bearing atoms, typed seed/task identity, four-level coverage event, contract-bound observation/unsupported evidence, and separately typed fresh/regression obligation collections.

Completion report invariants: `allowed_write_paths=[]`, `changed_files=[]`, `forbidden_paths_modified=[]`, `deviations=[]`, `candidate_status=none`, `twenty_four_hour_run_authorized=false`.
