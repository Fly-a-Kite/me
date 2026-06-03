# Reproduction Guide

This file is the working checklist for rerunning the current DataDiffFuzz evidence on an Ubuntu machine. Local macOS runs are useful for development and smoke tests; final paper tables should use Ubuntu x86_64.

## 1. Environment

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e '.[all]'
.venv/bin/python -m pip freeze > reports/pip-freeze-ubuntu.txt
uname -a > reports/uname-ubuntu.txt
lscpu > reports/lscpu-ubuntu.txt
```

Run sanity checks:

```bash
.venv/bin/pytest -q
.venv/bin/python -m compileall -q src
git diff --check
```

## 2. Confirm Candidate Bugs

DataFusion grouped top-k NULL sort-key bug:

```bash
.venv/bin/python bugs/bug_9b4d1fa7aac3b391/standalone_datafusion_groupby_null_sortkey_limit.py
```

Upstream issue: https://github.com/apache/datafusion/issues/22190. DataFusion member
`xiedeyantu` replied `take` on 2026-05-17 UTC and is assigned to the issue. Treat the
paper-facing status as `submitted_upstream_needs_external_confirmation` until the DataFusion
maintainers confirm, reject, or document the behavior.

Polars lazy float group-key instability candidate:

```bash
.venv/bin/python bugs/bug_0f37902bbb377813/standalone_polars_lazy_float_groupby_instability.py
```

These standalone scripts assert the expected correct behavior; an `AssertionError`
after the printed divergent output means the local dependency version still
reproduces the candidate bug.

Expected paper accounting:

- Count DataFusion as one high-confidence submitted candidate bug family if Ubuntu reproduces it.
- Count Polars only after Ubuntu x86_64 and latest-version confirmation.
- Do not count seeded faults as real bugs.

## 3. Medium Main Matrix

```bash
.venv/bin/datadiff experiment \
  --cases 1000 \
  --seeds 1,1001,2001,3001,4001 \
  --presets baseline,guided,no_feedback,no_type_aware,no_normalizer,metamorphic \
  --target-suite core \
  --log-level minimal \
  --artifact-limit 2 \
  --skip-run-reports
```

After completion:

```bash
.venv/bin/datadiff experiment-summary --manifest runs/<manifest>.json --refresh
.venv/bin/datadiff analyze-experiment --manifest runs/<manifest>.json --refresh
```

Optional follow-up for scheduler experiments outside the paper tables:

```bash
.venv/bin/datadiff experiment \
  --cases 1000 \
  --seeds 1,1001,2001 \
  --presets baseline \
  --target-suites core,datafusion_cross \
  --schedule adaptive \
  --batch-cases 100 \
  --local-source-exploration-weight 0.25 \
  --skip-run-reports
```

## 4. Bug-Hunting Matrix

```bash
.venv/bin/datadiff experiment \
  --cases 1000 \
  --seeds 1,1001,2001,3001,4001 \
  --presets discovery_no_groupby_guided_metamorphic,null_groupby_topk,null_agg_topk,float_group_key,float_group_key_metamorphic \
  --target-suites core_lazy,core_datafusion,datafusion_cross \
  --log-level minimal \
  --artifact-limit 2 \
  --skip-run-reports
```

Use current oracle labels on compact/full logs, or artifact-backed minimal logs:

```bash
.venv/bin/datadiff classify-run --run-file runs/<run>.jsonl.gz --refresh --limit 5
```

`--refresh` recomputes differential findings with the current oracle when the run row stores normalized outputs, or when the row points to a bug artifact containing `case.json` and `normalized.json`.

## 5. Seeded Sensitivity Matrix

```bash
.venv/bin/datadiff experiment \
  --cases 1000 \
  --seeds 1,1001,2001,3001,4001 \
  --presets baseline,guided_filter,guided_groupby,guided_join,guided_mutate \
  --target-suites seeded_filter,seeded_groupby,seeded_join,seeded_mutate \
  --log-level minimal \
  --artifact-limit 1 \
  --skip-run-reports
```

## 6. Keep For Paper Artifact

- `reports/experiment-summary-*.md`
- `reports/*-aggregates.csv`
- `reports/experiment-analysis-*.md`
- `bugs/bug_9b4d1fa7aac3b391`
- `bugs/bug_0f37902bbb377813`
- `runs/experiment-*.json`
- `reports/pip-freeze-ubuntu.txt`
- `reports/uname-ubuntu.txt`
- `reports/lscpu-ubuntu.txt`

Avoid publishing all `runs/run-*.jsonl.gz` files unless needed; minimal logs plus bug artifacts are usually enough for storage-conscious reproduction.
