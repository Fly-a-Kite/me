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
- run-log bytes, artifact bytes, and evidence bytes per case
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

### 4. Module Ablation And Related-Scope Comparison

Goal: quantify which framework modules and target scope choices contribute to
bug discovery quality, time efficiency, space efficiency, and false-positive
control.

The `ablation` track runs the same harness across core target families with
module variants:

- `baseline`
- `no_type_aware`
- `no_normalizer`
- `no_feedback`
- `metamorphic`
- `oracle_only_metamorphic`
- `reducer`

The `comparison` track compares random/guided/metamorphic/workflow presets over
SQL/query-engine-oriented suites (`embedded_sql`, `datafusion_cross`) and the
cross-ecosystem suites (`latest_all_engines`, `latest_no_datafusion`). This is a
controlled scope baseline inside DataDiffFuzz, not a reimplementation of
SQLancer or SQUIRREL. Use it to measure what the DataFrame/Arrow/cross-family
target registry adds beyond SQL-oriented testing under the same runner and
oracle.

These tracks support RQ tables only. They do not directly count as real backend
bug evidence unless a candidate is independently promoted through the latest
live confirmation workflow.

## Command Generation

Generate the frozen command plan:

```bash
.venv/bin/python scripts/run_final_experiments.py --track all --duration 24h --jobs 1
```

Execute the plan only after confirming the code is frozen:

```bash
<latest-live-venv>/bin/python scripts/run_final_experiments.py --track validation --validation-cases 200 --jobs 1 --execute
<latest-live-venv>/bin/python scripts/run_final_experiments.py --track live --duration 24h --jobs 1 --execute
<latest-live-venv>/bin/python scripts/run_final_experiments.py --track seeded --duration 24h --jobs 1 --execute
<latest-live-venv>/bin/python scripts/run_final_experiments.py --track ablation --ablation-cases 2000 --jobs 1 --execute
<latest-live-venv>/bin/python scripts/run_final_experiments.py --track comparison --comparison-cases 2000 --jobs 1 --execute
<historical-vulnerable-venv>/bin/python scripts/run_final_experiments.py --track historical --duration 24h --jobs 1 --execute
```

The mixed `--track all` command is for plan review only. Direct `--track all --execute`
is rejected because the latest-version and historical tracks must run from different
isolated Python environments.

Run `--track validation` first in the latest-live environment. It is a short
validation evidence-mode smoke matrix over live target families for
adapter/oracle/preflight/classification/evidence noise; candidates from this
track are not final 24h bug evidence. If validation finds harness noise, fix it
before freezing and regenerate the final plan.

The `ablation` and `comparison` tracks use their own evidence modes, not
`live`. They support module/RQ and related-scope tables and are audited by
`final-readiness`, but their candidates must be promoted through a separate
latest-version live confirmation workflow before they can affect real bug
counts.

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
.venv/bin/datadiff classify-run --run-file runs/run-*.jsonl.gz --json
.venv/bin/datadiff methodology-report --manifest runs/experiment-*.json --refresh
```

`experiment-summary` records both time efficiency (`throughput_cases_s`, time to
first candidate) and space efficiency (`run_log_bytes`, `artifact_bytes`,
`evidence_bytes_per_case`). `analyze-experiment` compares those storage metrics
against the selected baseline preset, so a faster preset that produces much
larger evidence artifacts is visible in the paper tables instead of being hidden
behind candidate yield alone.
`classify-run` is the offline oracle-facing triage summary: its
`offline_buckets` field separates `new_bug`, `known_bug`, `false_positive`,
`semantic_divergence`, and residual `needs_triage`/`unclassified` findings from
the stored run evidence without rerunning the generator.
`methodology-report` ties those run-level artifacts back to the experiment
manifest, summary Markdown, run CSV, aggregate CSV, run logs, and baseline
comparisons, producing the paper-facing evidence-chain JSON/Markdown. It also
aggregates the same offline oracle buckets across the manifest's run logs, so
paper tables can distinguish new bugs, known/replay bugs, false positives, and
semantic divergences from the final evidence chain. Its reproducibility block
counts referenced bug artifact directories,
`reproduce.py`, reduced reproducers, standalone reproducers, and triage reports
so reproduction readiness is visible as a metric instead of an anecdote.
The lighter pre-submission `issue-bundle --run-reproducers` manifest is also
audited. Use `--repeat N` before submission when a candidate may depend on
optimizer state or execution order. Use `--primary-per-family` when multiple
drafts describe the same submission family, so only the primary draft's
reproducer is executed while supporting draft paths remain in the manifest.
Missing reproducer blocks, compile
failures, flaky repeated outputs, nonzero exits, and timeouts are exposed
through `bug-status` and fail `review-readiness` until the bundle evidence
executes cleanly. `methodology-report` records the same issue-bundle manifest,
reproducer paths, execution coverage, and failure counts in the paper-facing
reproducibility/evidence-chain artifact.
The bug-discovery block records both the global first candidate and each
rewardable candidate family's first-seen case/time/run context, preserving the
time-to-each-new-family metric required by the final protocol.
The report streams each run log once for run-log-derived evidence metrics, so
offline buckets, artifact reproducibility, and first-seen family timing stay
practical on 24h compressed logs. For quick checks over already generated
experiment-summary CSVs, `methodology-report --summary-only` skips those
run-log-derived sections and records `run_logs_scan_skipped=true`; this mode is
for iteration and is not a substitute for the full final evidence report.
The same report records which planned ablation modules were actually covered
(`type_aware_generation`, `semantic_normalizer`, `feedback_corpus`,
`metamorphic_oracle`, `differential_oracle`, and `reducer`) so incomplete
module-ablation evidence is visible before paper tables are written.

For seeded runs:

```bash
.venv/bin/datadiff analyze-seeded-sensitivity --manifest runs/experiment-*.json
```

Before claiming that the final experiment satisfies the paper target, run the readiness audit over
the validation, live, historical, seeded, ablation, and comparison manifests:

```bash
.venv/bin/datadiff final-readiness \
  --manifest runs/experiment-validation.json \
  --manifest runs/experiment-live-datafusion.json \
  --manifest runs/experiment-live-polars.json \
  --manifest runs/experiment-live-arrow.json \
  --manifest runs/experiment-live-embedded-sql.json \
  --manifest runs/experiment-live-cross-family.json \
  --manifest runs/experiment-historical.json \
  --manifest runs/experiment-seeded.json \
  --manifest runs/experiment-ablation.json \
  --manifest runs/experiment-comparison.json
```

The audit is intentionally strict by default: it checks the short validation
smoke, all six live suites, 24h depth per live suite, fresh replay-policy
isolation, confirmed latest-version bug evidence, confirmed historical replay
evidence, seeded sensitivity evidence, module-ablation evidence, and
baseline/related-scope comparison evidence. When no `--manifest` is supplied,
the CLI audits only the latest bounded set of experiment manifests so old
scratch runs do not make the command unusable. Without explicit manifests this
is a metadata-only status check and the `run_log_scan` gate will remain false;
use explicit `--manifest` values for the paper claim, or add
`--full-run-log-scan`/`--all-manifests` only when you intentionally want to scan
large accumulated logs.
Those requirements are top-layer policy inputs to the audit (`--required-live-suites`,
`--required-live-families`, confirmation files, and threshold flags). Confirmed
latest-version evidence may come from run-log findings whose `paper_status` is
maintainer-confirmed, or from `experiments/latest_confirmations.json` when an
upstream issue/PR has independent confirmation such as a maintainer `bug` label,
assignment, acknowledgement, or fix. This confirmation evidence is separate from
fresh discovery reward: known/saturated or already submitted families still do not
inflate rewardable latest candidate counts. The audit engine itself only reads
manifests/run logs, confirmation evidence, and supplied policy; it does not change
generation, execution, normalization, or oracle behavior.

## Paper Wording

Use "candidate bug family" for unconfirmed latest-version findings. Use "confirmed bug" only after maintainer acknowledgement, fix, or clear spec violation. Use seeded results only for sensitivity and ablation claims.

Do not report raw finding count as bug count; raw findings are often many executions of the same root cause.
