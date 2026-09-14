# Environment Rebind — working notes

Recorded 2026-09-14. This documents how the frozen phase-6 environment is reconstructed in the
dev worktree, and the one place where the environment has **irreversibly drifted**.

## 1. Historical `/tmp` paths

Only one test file references historical `/tmp/phase6_*` paths:
`tests/osc/test_runtime_phase6_formal_target_version_producer.py`. Three symlinks restore the
materials from the immutable archive:

| `/tmp` path | Archive target |
| --- | --- |
| `/tmp/phase6_formal_execution_authority_r1.Smql7A` | `phase6_formal_execution_authority_r1.Smql7A/` |
| `/tmp/phase6_formal_latest_target_remediation_r1.20260724` | `tmp_evidence_mirror_20260726/phase6_formal_latest_target_remediation_r1.20260724/` |
| `/tmp/phase6_formal_target_version_execution_capability_source_snapshot_r1b.gfqAvj` | `tmp_evidence_mirror_20260726/phase6_formal_target_version_execution_capability_source_snapshot_r1b.gfqAvj/` |

With these in place the file collects cleanly (was: `FileNotFoundError` at import). These are
local test-environment links only; they do **not** modify any frozen artifact.

## 2. Interpreter drift (real blocker) — **two layers**

### Layer 1 — SHA
| Binding | Value |
| --- | --- |
| Frozen authority `interpreter_sha256` | `1643dacd9feaedc58f3cc581e4d22577dfe25c09b10282936186ccf0f2e61118` |
| Current `/usr/bin/python3.12` | `e50d468e8b0adfb05733f5b87b3cff34829c4a8c1aea50c865aa8bdfe4bb150f` |
| Resolved path (frozen) | `/usr/bin/python3.12` |

### Layer 2 — build/version (discovered 2026-09-14)
The environment audit records
`interpreter.version = 3.12.3 (main, Jun 19 2026, 12:46:00) [GCC 13.3.0]`, but the current
interpreter reports `3.12.3 (main, Aug 31 2026, 10:18:26) [GCC 13.3.0]`. The metadata probe
emits `sys.executable` + `version`, and
`_phase6_formal_target_version_producer.py:2222-2226` compares both against the audit. So even
after rebinding the SHA, 4 tests still fail with
`target-version metadata probe interpreter binding mismatch`.

**Cause:** the host Python was rebuilt/patched between the 2026-07-24 run and now; the
historical binary is unrecoverable.

### What was done this round (safe, partial)
- Replaced the mirror venv's `bin/python3.12` **symlink** with a **frozen real copy** of the
  current interpreter (sha `e50d468e…`), so future host patches cannot drift the bytes again.
  Verified the venv still imports pandas/polars/duckdb/pyarrow/datafusion/chDB.
- This alone does **not** fix the tests and is not committed as a test change.

### What the full rebind still requires
Regenerate a consistent chain under a **new** root (never edit the historical audit in place):
1. new `venv/` with the frozen interpreter + `pip-install-report.json`;
2. new `environment-audit.json` recording the current `interpreter.executable` and
   `interpreter.version`, plus correct `source_snapshot`/`pip_install_report` paths;
3. new `target-version-revalidation.json` whose `environment_audit.path` points at (2);
4. a new authority document binding the new audit/revalidation hashes;
5. update the test fixture (`_TARGET_*` constants) to the new root/hashes.

Failing tests before rebind: 6; after SHA-only rebind: 4 (all `metadata probe interpreter
binding mismatch`). Full suite baseline is otherwise clean (3044 passed / 1 skipped).

## 3. Python environments present

| Venv | Interpreter | Packages | Project import | Use |
| --- | --- | --- | --- | --- |
| `datadiff_fuzz_lab/.venv` | 3.12.3 | pandas 3.0.3, polars 1.42.1, duckdb 1.5.4, pyarrow 25.0.0, datafusion 54.0.0, chDB 4.2.1 | editable install pointing at the wiped lab path | **preferred**: exact frozen target versions |
| `tmp_evidence_mirror_20260726/.../20260724/venv` | 3.12.3 | pandas 3.0.5, polars 1.43.0, duckdb 1.5.5, pyarrow 25.0.0, datafusion 54.0.0, chDB 4.2.1 | not installed | the interpreter path the authorities bind |

Run tests against the dev source by shadowing the editable install:

```bash
cd worktrees/wt-paper
PYTHONPATH="$PWD/src:$PWD" /data1/lbw/xjx/datadiff_fuzz_lab/.venv/bin/python -m pytest -q
```

Verified: `import datadiff` / `import datadiff_osc` resolve to `worktrees/wt-paper/src/...`.

## 4. Open W1 actions

1. Freeze a durable environment descriptor (package versions + wheel hashes + interpreter hash)
   for the current host. *(partially done: `ENVIRONMENT.md`; interpreter already frozen)*
2. Build the new rebind root and regenerate `environment-audit.json` +
   `target-version-revalidation.json` (see §2) so the 4 remaining tests pass honestly.
3. Bind the new authority document to the new hashes and update the test fixture constants.
4. Locate or regenerate the missing `runs/experiment-*.json` manifests (W3).
