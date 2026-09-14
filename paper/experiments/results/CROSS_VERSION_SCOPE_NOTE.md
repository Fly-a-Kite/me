# Cross-version scope note — what it does and does not yield

Run 2026-09-14.

| Env pair | Backend(s) | Cases | Findings |
| --- | --- | ---: | ---: |
| df 53.0.0 → 54.0.0 | datafusion | 10 (corpus) | 1 known root |
| df 53.0.0 → 54.0.0 | datafusion | 110 (corpus + 10 variants each) | 9, all one known root |
| df 53.0.0 → 54.0.0 | datafusion | 40 + 30 (fresh targeted) | 0 |
| polars 1.40.1 → 1.42.1 | polars | 10 (corpus) | 1 known root |
| mirror → frozen (pandas 3.0.5/3.0.3, polars 1.43.0/1.42.1, duckdb 1.5.5/1.5.4) | pandas, polars, duckdb | 180 (corpus + 5 variants each) | 0 |

## Conclusion

- Cross-version differential is a **confirmation / stability** tool: it reliably re-detects the
  known version-sensitive roots and reports no false positives on stable ones.
- It is **not** by itself a new-root generator at the version gaps currently installed. Small
  patch/minor gaps (pandas 3.0.3↔3.0.5, polars 1.42.1↔1.43.0, duckdb 1.5.4↔1.5.5) are stable.
- To produce new roots it would need many more historical versions spanning known fix
  boundaries, or genuinely new program shapes/targets.

## Where new roots must come from

1. **New target systems / interfaces** (new bug surface) — highest historical yield.
2. **New operators / program shapes** (window, asof, set ops) — 7-backend cost.
3. **Longer campaigns** (24 h running) — historical rate ≈ 1 root per ~20k executed cases.
4. **Deeper oracle families** (phase-6 OSC contract/granularity) — planned but deprioritised.

Implication for the S2 target: +11–20 new roots in ~8 weeks is aggressive given the observed
fresh yield; the S2/S3 boundary remains the realistic landing zone.
