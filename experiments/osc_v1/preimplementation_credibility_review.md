# OSC Pre-implementation Credibility Review

Date: 2026-07-19 CST  
Schema: `osc-preimplementation-credibility-review-v1`  
Machine-readable authority: `preimplementation_credibility_review.json`

## Decision

The review authorizes implementation of the low-complexity OSC core. It does
not authorize a 24-hour run, primary interaction-tile scheduling, an e-graph,
general SMT synthesis, a generic MAP-Elites archive, or a native Rust kernel.

The central design survives adversarial review only under these constraints:

- four-valued verdicts remain distinct;
- unknown rules fail closed;
- capability and semantic permission remain separate;
- no family/root/candidate string has oracle authority;
- fresh and regression registries are physically separate;
- full exact comparison remains the authority path;
- complex mechanisms remain removable switches until paired evidence promotes
  them.

The formal 24-hour launch flag remains `false`.

## Audited facts and denominators

The current registry was independently evaluated through its public frozen
registrations, not copied from the architecture prose:

| Partition | Families | Cells |
| --- | ---: | ---: |
| Fresh coverage expansion | 16 | 232 |
| Confirmed-root regression | 9 | 144 |
| Total | 25 | 376 |

Target-to-control expansion over fresh cells gives exactly 502 backend-pair
obligations. For each family and categorical axis `a`, baseline-star edges were
recomputed as `(n_a - 1) * product(n_other)`. The per-family total is 384.

The same registry induces 202 raw baseline-centered 2×2 tiles when every
admissible pair of axes is enumerated. This is at most 808 endpoint instances
and 2,224 backend executions without reuse. Consequently, tiles are a shadow
denominator only. The compiler must generate and freeze them; no handwritten
tile count is authoritative.

The legacy post-triage gate remains an execution-health result, not a launch
authority: 22/22 runs, 2,200/2,200 cases, 6,296/6,296 OK backend results, zero
iteration failures and zero candidate families, but only 533/1,929 scheduled
semantic activations (27.63%).

The four `running_sum_precision@duckdb,pyarrow` rows remain suppressed by a
local, component-precise fix: `running_sum` consumes evaluation/partition order,
but a later join kills final presentation order. Their exact values and bag
multiplicities remain strict. The strict confirmed-root count remains nine.

## Forward semantics and backward observability

Eighteen representative workflows were manually derived. Full step traces are
in the JSON artifact. The decisive results are:

| ID | Workflow | Forward property | Backward demand / final relation |
| --- | --- | --- | --- |
| W01 | total sort → limit | total evaluation and presentation order | exact sequence, schema and cardinality |
| W02 | tied sort → limit | non-total key; cut crosses tie | legal top-k partial-order set, exact `n`, strict non-tie rows |
| W03 | limit without order | membership is not uniquely determined | cardinality only; values are inconclusive absent scoped input order |
| W04 | running sum → join | internal ordered evaluation; join kills presentation order | exact-value bag and join multiplicity; no final sequence demand |
| W05 | running sum → join → total sort | final sort re-establishes presentation order | sequence plus independent running-sum evaluation-order precondition |
| W06 | groupby NULL aggregates | unordered groups; per-aggregate NULL/dtype rules | exact group bag/cardinality and component-specific aggregate rules |
| W07 | distinct | input multiplicity killed; output must be unique | set membership plus uniqueness and exact cardinality |
| W08 | inner join | product multiplicity; order killed | exact bag and explicit NULL-key equality |
| W09 | semi join | right multiplicity ignored, left multiplicity retained | existential membership and right-duplicate invariance |
| W10 | anti join | complement membership; not automatically SQL `NOT IN` | explicit NULL policy and right-duplicate invariance |
| W11 | valid cast | exact defined conversion and dtype | tagged value plus logical dtype/nullability |
| W12 | invalid cast | scoped domain error; unsupported/timeout/crash distinct | accept/reject plus semantic error class |
| W13 | signed zero | sign bit retained; roles separated | exact or explicitly scoped role equivalence |
| W14 | NaN/Inf | predicate/order/group/aggregate/conversion are separate | only demanded roles can select a relation |
| W15 | Arrow slice | offset/layout changes, logical result preserved | activated slice plus exact logical output |
| W16 | Arrow chunks | chunk boundaries change representation only | activated multi-chunk layout plus exact logical output |
| W17 | Arrow dictionary | dictionary indices/order are representation | exact decoded values/schema; physical indices unobserved |
| W18 | eager/lazy/streaming | mode changes planning/materialization metadata | mode equivalence only for supported, version-scoped operations |

Two important corrections follow from these traces:

1. `DISTINCT` cannot be compared by a plain set alone. That would miss a backend
   that incorrectly emits duplicates. The output needs set membership,
   uniqueness, and exact cardinality.
2. Tied top-k cannot be implemented as either exact sequence or unrestricted
   bag tolerance. Only rows within the boundary tie are free; all other rows and
   the cardinality remain strict.

For sequence, bag, set-with-uniqueness, partial order, numeric, error, layout,
differential isolation, monotonic boundary, and interaction-tile relations, the
review records one satisfied, violated, inapplicable, and inconclusive example.

## v1 policy audit

All sources of `boundary`, `tolerated`, and `probe` authority in
`semantic_contracts.py` and `contract_comparison.py` were classified. The 28
rule groups are fully listed in the JSON crosswalk. Their dispositions are:

- **Exact component relations:** limit/order, tie handling, running/window
  order, joins, NULL predicates and rewrites, aggregate NULL behavior,
  successful casts, scoped domain errors, numeric special values, and
  structured mismatch components.
- **Versioned overlays or exact relations:** backend-specific aggregate/numeric
  dtype, the narrowly scoped pandas nullable-scalar NaN materialization, and
  documented error categories. An overlay must identify backend version,
  adapter revision, exact preconditions, affected component, evidence and
  expiry.
- **Explicit unsupported/inconclusive:** string API error equivalence without a
  domain rule, decimal-rounding `numeric_tolerant` without explicit
  abs/rel/ULP/forward-error parameters, and unknown probes/rules.
- **Deleted:** generic join layout tolerance, case-level special-float boundary,
  root/mismatch substring permissions, the wildcard `*_probe` permission, and
  probe-based boundary bypass.

The 35 known non-regression probe kinds become explicit lossless witness
predicate plugins. `confirmed_root_witness_probe` is regression-only and must be
decomposed into nine semantic predicates before v1 authority is removed.
Unknown probes fail closed. Probe identity never suppresses or creates a
violation.

## Alternatives and complexity decisions

Each layer was compared against both a simple baseline and the strongest
feasible alternative. The core promotion choices are:

| Layer | Simple baseline | OSC core | Strong alternative | Decision |
| --- | --- | --- | --- | --- |
| Contract | explicit relation registry | finite two-pass HyperContract | full executable cross-ecosystem semantics | OSC core, with explicit-registry fallback |
| Target space | cells only | cells + 384 edges | constrained 2×2 tiles | edges primary, tiles shadow |
| Generation | typed random/goal-first | backward constructor + repair | constructive + evolutionary + SMT | core only; fallbacks need evidence |
| Rewrites | sequential list | bounded proof-gated DAG | Rust `egg` | bounded DAG only |
| Archive | epoch queue | bounded certificate Pareto front | generic MAP-Elites | epoch authority; archive switch/shadow |
| Scheduler | round robin | floor + max-min debt | broad online control | deterministic core; adaptation bounded |
| Comparison | full exact pairs | fingerprints + exact escalation | native Rust/Arrow kernel | Python staged planner; exact authority |
| Parallel | static shards | keyed task DAG + epoch barrier | async work stealing/learning | deterministic DAG |

Named technology decisions are frozen as follows:

- e-graph: rejected as a pre-24h dependency; profile-triggered future shadow.
- SMT: shadow-only bounded fallback after constructive and repair failure.
- MAP-Elites: generic adoption rejected; bounded certificate archive is a
  paired-evidence candidate.
- CEGAR: adopted only as offline, versioned contract hardening.
- covering arrays: adopted only to compile/select the shadow tile denominator.
- submodular selection: shadow, with max-min/round-robin fallback.
- native Rust kernel: rejected as a pre-24h dependency.

These mechanisms are established prior work and are not novelty claims.
Primary-source checks include SemConT, NOETHER, Semantic Mutation Score,
Metamorphic Coverage, FANDANGO, NIST covering arrays and AFLRUN; URLs and the
claim boundary for each are frozen in the JSON artifact.

## Offline planner evidence

Nine frozen real-result shards from
`next_architecture_capability_audit_v6` contain 156 result groups and 312 backend
results. A conservative structural simulation escalated every stored mismatch
group (33/156):

| Metric | Value |
| --- | ---: |
| Full normalized payload | 492,081 bytes |
| Exact-escalated payload | 62,499 bytes |
| Full materialization avoided | 429,582 bytes (87.30%) |
| Conservative compact fingerprint transport | 107,664 bytes |
| Total staged transport estimate | 170,163 bytes |
| Estimated total transport reduction | 65.42% |

This is enough to justify implementing a Python component planner. It is not a
throughput benchmark and does not authorize a native kernel. The required
100,000-group zero-discrepancy benchmark and paired throughput gate remain open.

## Provenance findings

Endpoint count is not implementation independence:

- Polars eager/lazy/streaming share a lineage.
- DataFusion and PyArrow share Arrow representation/kernel risk.
- DuckDB, SQLite, DataFusion and chDB use the shared DataDiff SQL lowering layer
  even though their engines differ.
- several endpoints use pandas or Arrow for transport/materialization.

Lineage may affect endpoint cover and evidence strength only. It cannot change
the semantic relation or create a majority-vote verdict.

## Remaining blockers

Before a 24-hour launch, the implementation still must pass every mandatory
gate: compatibility and approved precision fixes, nine-root recall, complete
false-positive suppression, 100% high-risk mutant kills, 232/384/502
reachability, scheduled activation and mutation preservation, operation/risk/
pipeline coverage, 100,000 exact-comparator parity groups, performance,
worker-count replay, complete repository pytest, paired 2-hour pilots, the
post-pilot credibility review, and a clean frozen preregistration/manifest.

## Reproduction commands

Run from the repository root:

```bash
jq empty experiments/osc_v1/preimplementation_credibility_review.json

PYTHONPATH=src .venv/bin/python - <<'PY'
from datadiff.family_witness_registry import (
    GLOBAL_FAMILY_WITNESS_V4_GENERATION_MODE,
    global_family_witness_registrations,
)
from datadiff.semantic_family_universe_v3 import semantic_family_v3_definitions

source = {item.family_id: item.source for item in semantic_family_v3_definitions()}
regs = global_family_witness_registrations(GLOBAL_FAMILY_WITNESS_V4_GENERATION_MODE)
fresh = [item for item in regs if source[item.family_id] == "coverage_expansion"]
regression = [item for item in regs if source[item.family_id] == "confirmed_root"]
edges = 0
for item in fresh:
    dims = [len(values) for _, values in item.axes]
    for index, size in enumerate(dims):
        count = size - 1
        for other, other_size in enumerate(dims):
            if other != index:
                count *= other_size
        edges += count
print(len(fresh), sum(item.cell_count for item in fresh), edges)
print(len(regression), sum(item.cell_count for item in regression))
PY
```

Expected output:

```text
16 232 384
9 144
```

The exact offline-planner reproduction is frozen separately in
`preimplementation_reproduce.md` to keep shell quoting and calculation details
out of the authority JSON.
