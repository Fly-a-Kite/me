# Final Goal

This project is now oriented toward a reproducible paper-grade experiment, not ad-hoc bug hunting.

## Primary Objective

Build and freeze a final DataDiffFuzz experiment harness that can:

1. Discover new bugs in the latest released versions of supported dataframe, Arrow, SQL, and query-engine backends.
2. Rediscover bugs that existed in previous vulnerable backend versions using the same final code and fixed replay protocol.
3. Validate method sensitivity with controlled seeded faults, without counting seeded faults as real backend bugs.

All future work should preserve this objective. Do not add one-off logic for a specific newly found bug unless it is part of the general experiment harness, reporting pipeline, or reproducibility protocol.

## Evidence Tracks

### Paper Run Journal

Every CLI run that can contribute to the paper must append a paper-facing entry to:

```text
reports/paper-run-journal.jsonl
reports/paper-run-journal.md
```

Each entry records the run theme, evidence track, target suite/backend set, preset/profile, seed, budget, executed cases, raw findings, candidate bug families, first candidate time, and the counting policy for that track. These journal entries are for paper traceability only; they must not change generator, oracle, reducer, or triage behavior.

### Latest-Version Live Discovery

Purpose: find new real backend bugs in current releases.

Required reporting:

- 24h wall-clock budget per target suite and seed.
- Executed cases.
- Raw findings.
- Candidate bug cases.
- Unique candidate bug families.
- Maintainer-confirmed or fixed bug families.
- Time to first candidate bug.
- Time to each new bug family.
- False positives and expected semantic divergences.

Bug family key:

```text
root_cause + suspicious_backends
```

Raw finding count is not a bug count.

### Historical Replay

Purpose: show that the same final code can rediscover previously known bugs in vulnerable backend versions.

Each historical replay must record:

- bug id
- upstream issue or PR
- vulnerable target version
- fixed version if available
- expected root cause
- expected suspicious backend
- seeds and budget

Only confirmed fixed or maintainer-confirmed historical bugs count as historical replay evidence. Pending issues are case studies until confirmed.

### Seeded Sensitivity

Purpose: measure detection sensitivity on controlled injected faults.

Seeded faults are method-validation evidence only. They must not be counted as real backend bugs.

The short final validation track uses `--evidence-mode validation`. It is a
pre-freeze harness gate and must not be counted as live latest-version bug
evidence.

### Ablation And Comparison Support

Purpose: quantify which framework modules and target-scope choices contribute
to bug discovery, false-positive control, throughput, storage cost, and
reproducibility.

The final module-ablation and related-scope/baseline tracks use
`--evidence-mode ablation` and `--evidence-mode comparison`. They support RQ
tables only. Candidates from those tracks must be rerun or promoted through the
latest-version live confirmation workflow before they can affect real bug
counts.

## Freeze Rule

Before starting final 24h runs:

- freeze generator, oracle, normalizer, guidance, reducer, and triage code;
- generate the final command plan;
- record dependency versions and platform metadata;
- do not tune code in response to intermediate findings.

After final runs start, allowed work is limited to:

- monitoring jobs;
- collecting outputs;
- validating and reducing artifacts;
- classifying findings;
- preparing upstream reports;
- writing analysis and paper tables.

## Final Experiment Entry Point

Use this command to generate the frozen plan:

```bash
.venv/bin/python scripts/run_final_experiments.py --track all --duration 24h --jobs 1
```

Do not execute the mixed `--track all` plan directly because latest-version live
runs and historical vulnerable-version replays use different isolated Python
environments. Execute frozen plans track-by-track from the matching venv:

```bash
<latest-live-venv>/bin/python scripts/run_final_experiments.py --track validation --validation-cases 200 --jobs 1 --execute
<latest-live-venv>/bin/python scripts/run_final_experiments.py --track live --duration 24h --jobs 1 --execute
<latest-live-venv>/bin/python scripts/run_final_experiments.py --track seeded --duration 24h --jobs 1 --execute
<latest-live-venv>/bin/python scripts/run_final_experiments.py --track ablation --ablation-cases 2000 --jobs 1 --execute
<latest-live-venv>/bin/python scripts/run_final_experiments.py --track comparison --comparison-cases 2000 --jobs 1 --execute
<historical-vulnerable-venv>/bin/python scripts/run_final_experiments.py --track historical --duration 24h --jobs 1 --execute
```

Protocol details live in:

```text
experiments/final_protocol.md
```

## Working Rule For Future Sessions

At the start of each new continuation on this project, read this file first,
then read `NEXT_SESSION_TODO.md` and align the next action with the final
experiment goal.

Default priority order:

1. Preserve reproducibility and paper-valid methodology.
2. Improve the general experiment harness or analysis pipeline.
3. Run or prepare final live, historical, or seeded experiments.
4. Triage findings only through the fixed family-dedup and confirmation protocol.
5. Avoid ad-hoc post-hoc tuning for individual bugs.
