# Cross-version corpus-variant scan — result

Run 2026-09-14. `datafusion-53.0.0--pyarrow-24.0.0` vs `frozen-target` (54.0.0), backend
`datafusion`, all 10 canonical corpus cases × (1 + 10 row-resampled variants) = 110 cases.

| Metric | Value |
| --- | --- |
| Cases | 110 |
| Findings | **9** |
| Distinct roots among findings | **1** — `distinct_null_topk` (the known df53→df54 fix) |
| Other 9 corpus families | 0 findings (version-stable or fixed in both) |

Variant hit rate for the version-sensitive shape: 9/11 (original + 10 variants).

## Interpretation (honest)

- Row-resampling a **version-sensitive program shape** reliably re-triggers the cross-version
  divergence: 9 instances vs 1 from the original case.
- These are **instances of one known root**, not new roots. Variant seeding amplifies
  reproduction/detection, it does not by itself create new root causes.
- The 9 version-stable corpus families produce no false positives — the comparator is clean.

## Implication for the plan

- Variant seeding is useful for **confirmation and stability evidence** (how reliably a
  version difference reproduces), not for raising the root count.
- Raising the root count still needs either the 24 h campaign (running) or genuinely new
  program shapes (P0 operator/generator breadth).
