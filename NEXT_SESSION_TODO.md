# Next Session TODO

Last updated: 2026-06-01 CST (+0800)

This file is the operational handoff for the next Codex session.

## What Was Completed In This Session

1. `ExperimentConfig` now normalizes and rehydrates `discovery_biases` on construction.
2. `DiscoveryBias` now has:
   - input normalization
   - stable dedup key
   - noop filtering
3. Added shared helpers in [src/datadiff/config.py](/root/datadiff_fuzz_lab/src/datadiff/config.py):
   - `coerce_discovery_bias`
   - `merge_discovery_biases`
4. Bug-sprint lane discovery bias metadata was promoted into a reusable source via
   [src/datadiff/strategy_registry.py](/root/datadiff_fuzz_lab/src/datadiff/strategy_registry.py):
   - `discovery_biases_for_lanes(...)`
5. Live presets now reuse lane-level discovery bias instead of leaving that logic only inside bug-sprint:
   - `live_common_api_workflow(_metamorphic)`
   - `live_datafusion_common_api_metamorphic`
   - `live_datafusion_fresh(_metamorphic)`
   - `live_deep_organic(_metamorphic)`
   - `live_cross_family_metamorphic`
   - `live_arrow_deep_organic_metamorphic`
   - `live_polars_deep_organic_metamorphic`
   - `live_polars_streaming_deep_organic_metamorphic`
   - `live_datafusion_deep_organic_metamorphic`
   - `live_embedded_sql_deep_organic_metamorphic`
6. Bug-sprint lane bias merge now deduplicates against preset defaults instead of double-applying identical bias.
7. Tests added/updated for:
   - preset default discovery bias presence
   - config round-trip rehydration
   - candidate pipeline artifact config rehydration
   - bug-sprint bias dedup behavior

## Verification Completed

Focused:

```bash
rtk proxy .venv/bin/python -m pytest -q tests/test_cli.py tests/test_candidate_pipeline.py tests/test_guidance.py tests/test_runner.py -k 'discovery_bias or live_common_api_workflow or live_datafusion_deep_organic or candidate_pipeline or recheck or bug_sprint'
```

Result:

```text
15 passed, 334 deselected
```

Broader touched-area regression:

```bash
rtk proxy .venv/bin/python -m pytest -q tests/test_cli.py tests/test_candidate_pipeline.py tests/test_guidance.py tests/test_runner.py tests/test_targets.py tests/test_reporter.py tests/test_run_journal.py
```

Result:

```text
384 passed in 21.19s
```

## Current State / Why This Matters

The project now has a better structural path for increasing fresh-bug yield:

1. discovery bias is no longer a bug-sprint-only add-on
2. saved config JSON can round-trip back into typed bias objects
3. live presets can push the search toward high-risk semantic boundaries without hardcoding one-off bug scripts

This is still not the final architecture. It is only the first systematic layer for improving bug-finding probability.

## Highest Priority Next Work

### 1. Extract preset registry out of `_preset_config`

Why:
- `src/datadiff/cli.py::_preset_config()` is still a massive if-chain
- discovery bias, target suites, methodology categories, and run intent should live in data, not in branch logic

Target outcome:
- add a typed preset registry module
- keep `_preset_config()` as a thin compatibility wrapper
- each preset entry should carry:
  - preset id
  - generator profile
  - metamorphic mode
  - guidance targets
  - default discovery bias
  - target suite affinity
  - reporting/methodology label

### 2. Push “new bug probability” deeper than preset weighting

Current limitation:
- discovery bias currently acts at scoring time
- it does not yet reshape generation/mutation aggressively enough

Next step:
- connect high-risk semantic families into seed selection and mutation choice
- do this generically, not per-bug

Concrete work:
- add a reusable “risk family” layer derived from features/patterns
- use it in:
  - seed scheduling
  - mutation operator selection
  - candidate-pool composition
- avoid using raw program complexity as a primary scheduler priority

### 3. Add stage-level performance profiling before any native rewrite

Do this before more Rust/C++ scope expansion.

Measure per case:
- generate/mutate
- backend execution
- normalize
- oracle/classification
- scheduler/feedback
- logging/artifact

Why:
- right now native descent decisions are still too assumption-driven
- the most likely high-value native target remains:
  - canonicalization
  - row/signature hashing
  - multiset diff
  - batch comparison kernel

### 4. Continue shrinking dict-style access at module boundaries

Typed IR migration is not fully complete in semantic layers.

Next convergence areas:
- `datagen`
- `metamorphic`
- `oracle`
- backend lowering/runtime seams

Goal:
- fewer ad hoc dict field reads
- more stable typed semantic accessors
- easier future Rust lowering and safer mutation logic

### 5. Continue moving result comparison kernel toward native conclusion output

User’s requested direction is still valid:
- not only “canonical artifacts”
- native side should increasingly produce comparison conclusions directly

Next narrow target:
- move from
  - normalize -> hash -> diff -> python oracle decision
- toward
  - native batch compare API returning structured semantic comparison result

But only after stage profiling.

### 6. Strengthen system-level “find new bug” methodology

Add more of these, as reusable architecture rather than test-only patches:

1. cross-layer boundary stress
2. same-semantics multi-representation execution
3. metamorphic relation families
4. native/fallback consistency loops
5. whole-system smoke around runner/candidate pipeline/artifacts
6. high-risk low-complexity seed templates

Important:
- do not add explicit names like `test_methodology_contracts`
- integrate the methodology into architecture, reporting, and scheduler semantics

## Strong Recommendation For Next Session

Do not start a 12-hour run first.

First do:

1. preset registry extraction
2. profiling hooks
3. risk-family-aware scheduling/mutation hooks
4. focused regression

Then run:

1. short smoke bug-sprint
2. short live preset run
3. only then a long concurrent run

## Suggested Execution Order Next Session

1. Read:
   - `FINAL_GOAL.md`
   - `PROJECT_IDEA_AND_PROGRESS.md`
   - `experiments/final_protocol.md`
   - this file
2. Extract preset registry
3. Add stage-level profiling
4. Thread risk-family metadata into scheduler + mutator
5. Add focused tests
6. Run medium regression
7. Start short integrated live runs
8. Decide whether native kernel descent should target canonicalization/diff next

## Files Touched This Session

- [src/datadiff/config.py](/root/datadiff_fuzz_lab/src/datadiff/config.py)
- [src/datadiff/strategy_registry.py](/root/datadiff_fuzz_lab/src/datadiff/strategy_registry.py)
- [src/datadiff/cli.py](/root/datadiff_fuzz_lab/src/datadiff/cli.py)
- [tests/test_cli.py](/root/datadiff_fuzz_lab/tests/test_cli.py)
- [tests/test_candidate_pipeline.py](/root/datadiff_fuzz_lab/tests/test_candidate_pipeline.py)

