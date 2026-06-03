# Next Session TODO

Last updated: 2026-06-03 14:42 CST (+0800)

Current session id: `icse-final-evidence-20260603T064235Z`

This file is the operational handoff for the next Codex session.

## Priority Handoff For `icse-final-evidence-20260603T064235Z`

The adaptive method/code layer is substantially implemented, but the ICSE-final
claim is not complete until the final experiment evidence passes readiness.
Current best estimate:

- method/code capability: about 75-85%
- automation/readiness gates: about 65-75%
- final ICSE experiment evidence: about 35-45%
- real 24h authority-run evidence: about 0-10%
- overall paper-ready completion: about 50-60%

Do not mark the active goal complete until the final readiness audit passes with
the manifest index and cross-version ledger evidence from real runs.

## Immediate TODO To Reach 100%

1. Prepare authority run prerequisites.
   - Make or stage a clean authority worktree state; the 24h launcher requires a clean tree by default.
   - Preserve provenance: commit hash, strategy snapshot, learning state, run journal, and launcher config.
   - Use the current launcher name only: `scripts/start_closed_loop_24h_tmux.sh`.

2. Execute the missing final matrix tracks with manifest-index capture.
   - `module_ablation`
   - `adaptive_component_ablation`
   - `baseline_scope_comparison`
   - Keep using `reports/final-experiment-manifest-index.json`.
   - Do not rely on stale `runs/experiment-*.json` sweeping.

3. Execute or import real 24h live discovery evidence.
   - The final report must show `live_depth` passing for the required suites.
   - The run must have clean frozen provenance and closed-loop state persistence.
   - The final readiness gates currently fail on live depth/provenance/responsiveness.

4. Build the final cross-version continual-learning ledger.
   - Use final comparable run logs from at least two target versions.
   - Output:
     - `reports/final-version-ledger.json`
     - `reports/experiment-final-version-ledger.json`
   - Required ledger schemas:
     - `version-ledger-v1`
     - `version-ledger-health-v1`
     - `version-ledger-health-feedback-report-v1`
     - `version-ledger-evidence-manifest-v1`
   - The ledger must include at least two versions, at least one candidate family,
     and health observations from real run logs.

5. Run final postprocess readiness only after evidence is complete.
   - Command shape:
     ```bash
     rtk .venv/bin/python scripts/run_final_experiments.py \
       --track postprocess \
       --execute \
       --manifest-index reports/final-experiment-manifest-index.json \
       --ledger-evidence-manifest reports/experiment-final-version-ledger.json
     ```
   - The script now refuses postprocess execution when the manifest index is
     missing required matrices or the final version-ledger evidence manifest.

6. Final completion condition.
   - `datadiff final-readiness` must exit successfully.
   - The JSON report must have `ready: true`.
   - No final gate may be failed.
   - Only then can the active goal be marked complete.

## Current Known Blockers

- Latest readiness reports are still `ready: false`.
- Missing matrix IDs in latest substantive evidence:
  - `module_ablation`
  - `adaptive_component_ablation`
  - `baseline_scope_comparison`
- Latest readiness summary has zero valid cross-version ledgers.
- Real clean 24h authority-run evidence has not been produced in the current
  worktree state.

## Verification From This Session

Code and gate hardening completed:

- Added postprocess preflight in `scripts/run_final_experiments.py`.
- Postprocess now refuses incomplete manifest-index evidence before running the
  final readiness audit.
- Cross-version continual ledger health/report path is implemented and tested.
- Old public discovery naming scan had no legacy matches.

Verification:

```text
tests/test_final_experiment_plan.py: 23 passed
final_readiness/CLI targeted tests: 50 passed
full pytest: 1408 passed, 2 warnings
git diff --check: clean
```
