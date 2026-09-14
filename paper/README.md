# Paper workspace

Planning and drafting area for the ICSE/FSE submission on \sys{} (DataDiffFuzz), built on the
phase-6 authority archive.

| Path | Role |
| --- | --- |
| `PLAN.md` | Master roadmap: decisions, baselines, contributions, RQ matrix, workstreams W1–W4, risks |
| `CLAIMS_RECONCILIATION.md` | Single source of truth for every numeric claim |
| `EXPERIMENT_MATRIX.md` | Per-RQ protocols, arms, seeds, metrics, artifacts, execution order |
| `research/CONTRIBUTION_REVIEW.md` | Independent review of the claims and their evidence |
| `research/CODE_GAP_REPORT.md` | Verified phase-6 gate code gaps and task list T1–T12 |
| `RELATED_WORK_DOSSIER.md` | Per-paper detailed analysis of all competing published work (verified abstracts) |
| `COMPETITIVE_ANALYSIS_DEEP.md` | Synthesis: overlap verdict, moat, adjustment directions D1–D9 |
| `COMPETITIVE_POSITIONING.md` | Positioning, threat tiers, rebuttal table |
| `INNOVATION_ANALYSIS.md` | First-principles significance and the five innovations (C-A…C-E) |
| `BUG_AUDIT_AND_VALIDITY.md` | Audit of the 11 confirmation records: are they really bugs, and the four weaknesses |
| `BUG_GENERATION_STRATEGY.md` | Breadth/depth/efficiency design to raise confirmed bugs (with code map) |
| `COMPETITIVE_TARGETS.md` | Multi-axis target metrics and go/no-go thresholds for head-to-head competition |
| `experiments/ENV_REBIND.md` | Frozen-environment reconstruction and interpreter drift |
| `experiments/results/` | Regenerated results and manifests (W3) |
| `main.tex`, `sections/`, `refs.bib` | English paper skeleton (acmart, ICSE/FSE format) |
| `writing/` | Per-section writing specifications |

## Status

- Paper skeleton: **drafted** (structure + C1/C2/C6 + oracle core; results placeholders).
- Canonical claims: **reconciled** (9 strict confirmed families; no "first" claims).
- Experiments: **not yet run**; historical manifests are missing and must be regenerated.
- Code completion: **not started**; see `research/CODE_GAP_REPORT.md` (no gate green in
  isolation; ~4–6 weeks engineering).

## Build the paper

No LaTeX toolchain is installed locally. Use Overleaf or TeX Live with the `acmart` class:

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Before submission: verify every `refs.bib` entry (entries marked `VERIFY` are provisional) and
replace every `\todo{}`.
