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

## 2. Interpreter drift (real blocker)

| Binding | Value |
| --- | --- |
| Frozen authority `interpreter_sha256` | `1643dacd9feaedc58f3cc581e4d22577dfe25c09b10282936186ccf0f2e61118` |
| Current `/usr/bin/python3.12` | `e50d468e8b0adfb05733f5b87b3cff34829c4a8c1aea50c865aa8bdfe4bb150f` |
| Resolved path (frozen) | `/usr/bin/python3.12` |

The host Python was patched after the phase-6 runs, so authority-bound execution fails closed
with `target-version target interpreter SHA-256 mismatch`
(`_phase6_formal_target_version_producer.py:2063`). This affects the six
`test_runtime_phase6_formal_target_version_producer.py` executor/capability tests.

**This must not be "fixed" by editing the frozen SHA.** The correct fix is a W1 rebind round
that publishes a new authority document binding the current interpreter hash and keeps the old
documents as immutable history.

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
   for the current host.
2. Decide whether to pin a copy of the interpreter binary inside the archive or to rebind to
   the host hash with an explicit drift note.
3. Regenerate the six failing executor/capability tests' expected bindings in the new round.
4. Locate or regenerate the missing `runs/experiment-*.json` manifests.
