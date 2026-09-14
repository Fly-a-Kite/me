# Skeleton content requirements for the introduction.

Write in English, ~1.5 pages. Structure:

1. **Setting (1 paragraph).** Data pipelines mix pandas/Polars, PyArrow, DuckDB/SQLite, and
   DataFusion/chDB. Practitioners assume semantic equivalence; the paper's premise is that this
   assumption fails at edge cases and fails *silently*.

2. **Problem (1 paragraph).** Why this is hard: (i) four execution models with different type
   systems, null/NaN conventions, ordering guarantees, and Arrow layouts; (ii) direct output
   comparison produces false positives, so a normalizer and classification are required;
   (iii) single-model techniques (SQL DBMS logic-bug testing) do not transfer directly;
   (iv) findings are only useful if they survive reproduction, reduction, dedup, and upstream
   confirmation.

3. **Gap (1 paragraph).** Prior work covers either DataFrame workload generation or SQL/query
   engine oracles, not one shared surface across all four families. Cite the related-work
   matrix. Respect `docs/novelty_claims.md`: do **not** claim first fuzzing/differential/
   metamorphic/DataFrame testing.

4. **Thesis (1--2 sentences).** One typed workflow IR plus a semantic normalizer plus a
   compositional multi-endpoint oracle complex plus an evidence pipeline can find and
   substantiate latest-version silent wrong-result bugs across the whole tabular ecosystem.

5. **Contributions (bulleted C1/C2/C6, C7 as oracle core).** Copy from `paper/PLAN.md` §3,
   with each bullet pointing forward to the section that proves it. Do not use an innovation
   count; do not repeat "first" claims.

6. **Results preview (1 short paragraph).** Nine strict confirmed families
   (`paper/CLAIMS_RECONCILIATION.md`), plus the RQ2/RQ5/RQ6 methodology findings once
   regenerated.

7. **Availability.** Artifact package layout (to be finalized in W4).

Constraints: every number must trace to `paper/CLAIMS_RECONCILIATION.md`; mark unverified
numbers with `\todo{}`.
