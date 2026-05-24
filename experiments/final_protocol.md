# Final Experiment Protocol

Read `FINAL_GOAL.md` first. This file expands the protocol details for that goal.

This protocol is the paper-facing plan. It separates evidence for new latest-version bugs from historical replay and seeded sensitivity. Do not tune generator, oracle, normalizer, guidance, or triage code after starting the final runs.

## Layered Method

The experiment has one shared middle and bottom layer. Target-specific choices are kept in the top layer.

- Top layer: target suites, presets, guidance targets, saturated families, and replay-source issue lists. This is where DataFusion, Polars, DuckDB, Arrow, and cross-family campaigns differ.
- Middle layer: case origin and replay policy, feature extraction, guidance, feedback scheduling, quality oracles, reward accounting, and classification. The same `case_policy` decides whether a case is organic, issue-inspired, or issue-replay for every target family.
- Bottom layer: DSL case representation, generators, preflight validation/repair, backend adapters, execution, normalization, differential oracle, metamorphic oracle, reducer, artifact writing, and run logging.

Fresh latest-version discovery and replay therefore do not use different harnesses. They run through the same generator, runner, normalizer, oracle, classification, and reporting code. The only policy difference is `enable_replay_bug`:

- `enable_replay_bug=false`: fresh/latest-version mode. Known submitted or replay-only issue cases are filtered before execution and replaced by fresh generated candidates.
- `enable_replay_bug=true`: replay mode. The same cases are allowed through the same execution and oracle path, so historical/submitted bugs can be reproduced and measured separately.

Presets with the `_replay` suffix only flip this policy switch on the same preset body. For example, `live_datafusion` and `live_datafusion_replay` share generator profile, guidance targets, scheduler, oracle, and backend execution; the replay preset allows known replay cases that the fresh preset filters. The same convention applies to project-specific replay probes such as DataFusion set operations, DuckDB tuple anti-null semantics, and Polars rolling-window semantics.

## Tracks

### 1. Latest-Version Live Discovery

Goal: discover new real backend bugs in the latest released versions.

Run one fixed 24h discovery campaign per target suite and seed:

- `datafusion_cross` with `live_datafusion`
- `datafusion_cross` with `live_datafusion_fresh`
- `dataframe_lazy` with `live_polars_lazy`
- `arrow_cross` with `live_arrow`
- `embedded_sql` with `live_embedded_sql`
- `latest_all_engines` with `live_cross_family`
- `latest_no_datafusion` with `live_cross_family`
- `polars_cross` with `live_polars_issue_focus`
- `embedded_sql_cross` with `live_duckdb_issue_focus`
- `arrow_cross` with `live_arrow_issue_focus`
- `latest_no_datafusion` with `live_issue_focus`

Report:

- executed cases
- raw findings
- candidate bug cases
- rewardable candidate families after excluding known saturated/submitted families
- unique candidate bug families
- maintainer-confirmed/fixed bug families
- time to first candidate bug family
- time to each newly discovered family
- false positives and expected semantic divergences

Family key: `root_cause + suspicious_backends`.

Fresh live runs must keep `enable_replay_bug=false`. A candidate whose source issue or replay probe matches the known replay policy can still influence the written method description, but it is not executed or counted in fresh latest-version discovery. This prevents already submitted issues from inflating new-bug evidence.

Issue-focus live runs use the bottom-layer `issue_focus` generator profile. It rotates semantic sketches derived from issue classes and records source metadata, but it does not decide freshness. Exact known/submitted replay probes are excluded by the same middle-layer replay filter unless the replay switch is explicitly enabled.

Live presets also enable middle-layer candidate recheck. A finding must reproduce across the configured immediate recheck attempts to remain a rewardable candidate; otherwise it is kept as diagnostic noise with `non_reproducible_candidate` status and excluded from bug evidence. Single-case fresh issue-focus intentionally excludes sequence-state sketches such as `float_group_key` and broad NaN/Inf boundary profiles such as `edge_float`; those belong in separate sequence or boundary-semantics experiments before they can count.

### 2. Historical Replay

Goal: show that the same final code can rediscover bugs that existed in previous vulnerable versions.

Historical replay must be run in an environment where the vulnerable backend version is installed. Each replay manifest must record:

- known bug id
- vulnerable target version
- fixed version or upstream issue
- expected root cause
- expected suspicious backend
- seeds and budget

Historical replay runs must use `enable_replay_bug=true`, either through `--enable-replay-bug`, a `_replay` preset, `--evidence-mode historical`, or fixture replay. The replay result is reported separately from latest-version discovery even when the same target suite and generator profile are used.

Only `confirmed_fixed` historical specs count as historical bug evidence. Pending or candidate specs are case studies until upstream confirmation/fix.

Environment setup and admission criteria live in `experiments/historical_envs.md`.
Historical issue intake lives in `experiments/historical_candidates.md`; entries there are not counted until they are promoted to the confirmed registry.
Fixture-backed historical bugs use `datadiff replay-fixture` with a declared spec under `experiments/historical_replays/`; they still must satisfy the same vulnerable/fixed isolated replay criteria before counting.
When generating or executing historical plans from an isolated replay venv, run
`scripts/run_final_experiments.py` with that venv's Python interpreter. The generated
commands use the `datadiff` entry point from the same venv, so the backend version under
test is not accidentally taken from the project development environment.

### 3. Seeded Sensitivity

Goal: validate detection sensitivity on controlled faults.

Seeded suites inject known faults into pandas-compatible reference backends. These runs do not count as real backend bugs. Report detection rate, candidate bug cases/s, and first candidate case/time for each seeded suite.

## Command Generation

Generate the frozen command plan:

```bash
.venv/bin/python scripts/run_final_experiments.py --track all --duration 24h --jobs 1
```

Execute the plan only after confirming the code is frozen:

```bash
<latest-live-venv>/bin/python scripts/run_final_experiments.py --track live --duration 24h --jobs 1 --execute
<latest-live-venv>/bin/python scripts/run_final_experiments.py --track seeded --duration 24h --jobs 1 --execute
<historical-vulnerable-venv>/bin/python scripts/run_final_experiments.py --track historical --duration 24h --jobs 1 --execute
```

The mixed `--track all` command is for plan review only. Direct `--track all --execute`
is rejected because the latest-version and historical tracks must run from different
isolated Python environments.

For historical replay including pending case studies:

```bash
.venv/bin/python scripts/run_final_experiments.py --track historical --include-pending-historical
```

## Analysis

Each `datadiff fuzz`, `datadiff longrun`, and `datadiff experiment` invocation records a paper-facing run journal entry:

```text
reports/paper-run-journal.jsonl
reports/paper-run-journal.md
```

Use `--run-theme` for the short paper theme and `--paper-notes` for brief context. The final plan generator adds these automatically. The journal is an index over run logs and metadata; it is not an oracle and must not be used to tune code after final runs start.

After each experiment manifest:

```bash
.venv/bin/datadiff experiment-summary --manifest runs/experiment-*.json --refresh
.venv/bin/datadiff analyze-experiment --manifest runs/experiment-*.json --refresh
```

For seeded runs:

```bash
.venv/bin/datadiff analyze-seeded-sensitivity --manifest runs/experiment-*.json
```

Before claiming that the final experiment satisfies the paper target, run the readiness audit over
the live, historical, and seeded manifests:

```bash
.venv/bin/datadiff final-readiness \
  --manifest runs/experiment-live-datafusion.json \
  --manifest runs/experiment-live-polars.json \
  --manifest runs/experiment-live-arrow.json \
  --manifest runs/experiment-live-embedded-sql.json \
  --manifest runs/experiment-live-cross-family.json \
  --manifest runs/experiment-historical.json \
  --manifest runs/experiment-seeded.json
```

The audit is intentionally strict by default: it checks all five live suites, 24h
depth per live suite, fresh replay-policy isolation, confirmed latest-version bug
evidence, confirmed historical replay evidence, and seeded sensitivity evidence.
Those requirements are top-layer policy inputs to the audit (`--required-live-suites`,
`--required-live-families`, and threshold flags). The audit engine itself only reads
manifests/run logs and evaluates the supplied policy; it does not change generation,
execution, normalization, or oracle behavior.

## Paper Wording

Use "candidate bug family" for unconfirmed latest-version findings. Use "confirmed bug" only after maintainer acknowledgement, fix, or clear spec violation. Use seeded results only for sensitivity and ablation claims.

Do not report raw finding count as bug count; raw findings are often many executions of the same root cause.
