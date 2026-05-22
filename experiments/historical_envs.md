# Historical Replay Environments

Historical replay is a separate evidence track from latest-version discovery. It must run the final DataDiffFuzz code against an explicitly vulnerable dependency version.

## Admission Criteria

A historical bug can enter the default final experiment plan only when all of the following are known:

- upstream issue or PR URL;
- vulnerable backend version or commit;
- fixed backend version or commit;
- reproduction surface is covered by the DataDiffFuzz DSL/backends;
- expected `root_cause`;
- expected `suspicious_backends`;
- successful local replay in an isolated environment.

If any field is missing, keep the bug as `candidate` or `pending_merge`; do not count it as historical replay evidence.

Candidate intake and rejected search hits are tracked in:

```text
experiments/historical_candidates.md
```

## Environment Layout

Use one isolated environment per historical backend/version:

```text
historical-envs/
  datafusion-<bug-id>-vulnerable/
  polars-<bug-id>-vulnerable/
  duckdb-<bug-id>-vulnerable/
  pyarrow-<bug-id>-vulnerable/
```

These environments are intentionally outside the repository and should not be committed.

## Python Package Replay

For Python-package backends such as Polars, DuckDB, PyArrow, pandas, and DataFusion Python bindings:

```bash
python3 -m venv /tmp/datadiff-historical-<bug-id>
/tmp/datadiff-historical-<bug-id>/bin/python -m pip install --upgrade pip
/tmp/datadiff-historical-<bug-id>/bin/python -m pip install pandas duckdb polars
/tmp/datadiff-historical-<bug-id>/bin/python -m pip install -e . --no-deps
/tmp/datadiff-historical-<bug-id>/bin/python -m pip install --force-reinstall '<backend>==<vulnerable-version>'
/tmp/datadiff-historical-<bug-id>/bin/datadiff experiment \
  --cases <cases> \
  --seeds <seeds> \
  --presets <presets> \
  --target-suite <suite> \
  --evidence-mode historical \
  --known-bug-id <bug-id> \
  --target-version <vulnerable-version> \
  --artifact-limit 20 \
  --log-level compact \
  --skip-run-reports
```

Record:

```bash
/tmp/datadiff-historical-<bug-id>/bin/python -m pip freeze > reports/pip-freeze-<bug-id>.txt
uname -a > reports/uname-<bug-id>.txt
```

Use `--no-deps` when installing this project into historical environments so the current dependency lower bounds do not silently upgrade the backend under test. After installing the vulnerable backend, verify the version inside the same interpreter before running a replay.
Generate or execute historical final plans with the isolated environment's Python, for example
`/tmp/datadiff-historical-<bug-id>/bin/python scripts/run_final_experiments.py --track historical`.
The plan generator resolves `datadiff` from that same venv.

Before adding a candidate to `src/datadiff/historical.py`, run the upstream minimal reproducer in the isolated vulnerable environment and in a fixed environment. A candidate is not admitted if the stated vulnerable package no longer reproduces locally.

## Fixture Replay

Some historical bugs require an upstream data fixture rather than generated in-memory rows. These may use the generic fixture replay path only when the data source and operations are declared in a JSON spec:

```bash
DATADIFF_<BUG_ID>_FIXTURE=/tmp/path/to/upstream-fixture \
  /tmp/datadiff-historical-<bug-id>/bin/python -m datadiff.cli replay-fixture \
  --spec experiments/historical_replays/<bug-id>.fixture.json \
  --fixture-env DATADIFF_<BUG_ID>_FIXTURE \
  --target-suite <suite> \
  --evidence-mode historical \
  --known-bug-id <bug-id> \
  --target-version <vulnerable-version> \
  --artifact-limit 20 \
  --log-level compact
```

The fixture spec must record the selected columns, operation sequence, upstream source, and expected SHA-256 when available. This path is still DataDiffFuzz evidence because it uses the same backend adapters, normalizer, oracle, triage, run log, and paper journal; it is not counted if it requires bug-specific backend code or a local compatibility shim.

## Source-Built Backend Replay

If the vulnerable version is only available as a source commit, build that backend outside this repository and point the corresponding Python adapter at the built package/library. Record the exact commit hash and build command in the historical bug notes.

## Registry Statuses

- `confirmed_fixed`: can be included in default historical final runs and counted as historical replay evidence.
- `pending_merge`: upstream report exists, but fix/version evidence is not complete.
- `candidate`: plausible historical issue found, but local isolated replay is not complete.

## Current Registry State

As of this protocol:

- `duckdb-22075` is tracked as `confirmed_fixed` and is included in default historical replay plans.
- `duckdb-22656` is tracked as `confirmed_fixed` and is included in default historical replay plans.
- `duckdb-3015` is tracked as a `candidate` fixture replay case study. The generic final code path is implemented, but the vulnerable `duckdb==0.3.1` Python package did not install under isolated Python 3.10, so it is not counted yet.
- `datafusion-22190` is tracked as `pending_merge`. It can be used as a case study only until a vulnerable version and fixed version are recorded.

Generate pending historical case-study commands with:

```bash
.venv/bin/python scripts/run_final_experiments.py --track historical --include-pending-historical
```

Generate confirmed historical replay commands with:

```bash
.venv/bin/python scripts/run_final_experiments.py --track historical
```

The second command includes only `confirmed_fixed` replay entries.

Check counted versus case-study registry status with:

```bash
.venv/bin/datadiff historical-status --include-pending
```
