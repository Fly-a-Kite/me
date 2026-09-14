# RQ7 — competitor-scope arm results

- confirmed roots: 9
- reachable (buggy backend in scope): {'tdiff_style': 2, 'sqlancer_common_scope': 1, 'ours_full': 9}
- detected (plain differential flagged it): {'tdiff_style': 1, 'sqlancer_common_scope': 0, 'ours_full': 3}
- competitor union reachable: 3/9 → 6/9 out of reach for all competitor scopes

| root | buggy backend | tdiff reach | tdiff detect | sqlancer reach | sqlancer detect | ours reach | ours detect |
| --- | --- | --- | --- | --- | --- | --- | --- |
| datafusion-grouped-null-topk-001 | datafusion | False | False | False | False | True | True |
| datafusion-limit-offset-pushdown-001 | datafusion | False | False | False | False | True | False |
| datafusion-negative-zero-comparison-001 | datafusion | False | False | False | False | True | True |
| datafusion-distinct-null-topk-001 | datafusion | False | False | False | False | True | False |
| datafusion-ordered-limit-idempotence-001 | datafusion | False | False | False | False | True | False |
| polars-grouped-max-sort-metadata-001 | polars | True | False | False | False | True | False |
| polars-reflected-arithmetic-operand-order-001 | polars | True | True | False | False | True | True |
| pyarrow-sliced-bool-hash-aggregate-001 | pyarrow | False | False | False | False | True | False |
| duckdb-join-filter-pushdown-limit-001 | duckdb | False | False | True | False | True | False |
