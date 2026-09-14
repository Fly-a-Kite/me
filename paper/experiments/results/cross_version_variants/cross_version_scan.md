# Cross-version scan

- env pairs: [['datafusion-53.0.0--pyarrow-24.0.0', 'frozen-target']]
- backends: ['datafusion']
- cases: 110
- results: 110
- findings: 9

| case | backend | left | right | match | reason |
| --- | --- | --- | --- | --- | --- |
| datafusion_distinct_null_topk.json | datafusion | ok | ok | False | result |
| canonical-datafusion-distinct-null-topk-001-v0 | datafusion | ok | ok | False | result |
| canonical-datafusion-distinct-null-topk-001-v1 | datafusion | ok | ok | True | equal |
| canonical-datafusion-distinct-null-topk-001-v2 | datafusion | ok | ok | True | equal |
| canonical-datafusion-distinct-null-topk-001-v3 | datafusion | ok | ok | False | result |
| canonical-datafusion-distinct-null-topk-001-v4 | datafusion | ok | ok | False | result |
| canonical-datafusion-distinct-null-topk-001-v5 | datafusion | ok | ok | False | result |
| canonical-datafusion-distinct-null-topk-001-v6 | datafusion | ok | ok | False | result |
| canonical-datafusion-distinct-null-topk-001-v7 | datafusion | ok | ok | False | result |
| canonical-datafusion-distinct-null-topk-001-v8 | datafusion | ok | ok | False | result |
| canonical-datafusion-distinct-null-topk-001-v9 | datafusion | ok | ok | False | result |
| datafusion_groupby_limit_offset.json | datafusion | ok | ok | True | equal |
| canonical-datafusion-groupby-limit-offset-001-v0 | datafusion | ok | ok | True | equal |
| canonical-datafusion-groupby-limit-offset-001-v1 | datafusion | ok | ok | True | equal |
| canonical-datafusion-groupby-limit-offset-001-v2 | datafusion | ok | ok | True | equal |
| canonical-datafusion-groupby-limit-offset-001-v3 | datafusion | ok | ok | True | equal |
| canonical-datafusion-groupby-limit-offset-001-v4 | datafusion | ok | ok | True | equal |
| canonical-datafusion-groupby-limit-offset-001-v5 | datafusion | ok | ok | True | equal |
| canonical-datafusion-groupby-limit-offset-001-v6 | datafusion | ok | ok | True | equal |
| canonical-datafusion-groupby-limit-offset-001-v7 | datafusion | ok | ok | True | equal |
| canonical-datafusion-groupby-limit-offset-001-v8 | datafusion | ok | ok | True | equal |
| canonical-datafusion-groupby-limit-offset-001-v9 | datafusion | ok | ok | True | equal |
| datafusion_grouped_null_topk.json | datafusion | ok | ok | True | equal |
| canonical-datafusion-grouped-null-topk-001-v0 | datafusion | ok | ok | True | equal |
| canonical-datafusion-grouped-null-topk-001-v1 | datafusion | ok | ok | True | equal |
| canonical-datafusion-grouped-null-topk-001-v2 | datafusion | ok | ok | True | equal |
| canonical-datafusion-grouped-null-topk-001-v3 | datafusion | ok | ok | True | equal |
| canonical-datafusion-grouped-null-topk-001-v4 | datafusion | ok | ok | True | equal |
| canonical-datafusion-grouped-null-topk-001-v5 | datafusion | ok | ok | True | equal |
| canonical-datafusion-grouped-null-topk-001-v6 | datafusion | ok | ok | True | equal |
| canonical-datafusion-grouped-null-topk-001-v7 | datafusion | ok | ok | True | equal |
| canonical-datafusion-grouped-null-topk-001-v8 | datafusion | ok | ok | True | equal |
| canonical-datafusion-grouped-null-topk-001-v9 | datafusion | ok | ok | True | equal |
| datafusion_limit_idempotence.json | datafusion | ok | ok | True | equal |
| canonical-datafusion-limit-idempotence-001-v0 | datafusion | ok | ok | True | equal |
| canonical-datafusion-limit-idempotence-001-v1 | datafusion | ok | ok | True | equal |
| canonical-datafusion-limit-idempotence-001-v2 | datafusion | ok | ok | True | equal |
| canonical-datafusion-limit-idempotence-001-v3 | datafusion | ok | ok | True | equal |
| canonical-datafusion-limit-idempotence-001-v4 | datafusion | ok | ok | True | equal |
| canonical-datafusion-limit-idempotence-001-v5 | datafusion | ok | ok | True | equal |
| canonical-datafusion-limit-idempotence-001-v6 | datafusion | ok | ok | True | equal |
| canonical-datafusion-limit-idempotence-001-v7 | datafusion | ok | ok | True | equal |
| canonical-datafusion-limit-idempotence-001-v8 | datafusion | ok | ok | True | equal |
| canonical-datafusion-limit-idempotence-001-v9 | datafusion | ok | ok | True | equal |
| datafusion_negative_zero_comparison.json | datafusion | ok | ok | True | equal |
| canonical-datafusion-negative-zero-comparison-001-v0 | datafusion | ok | ok | True | equal |
| canonical-datafusion-negative-zero-comparison-001-v1 | datafusion | ok | ok | True | equal |
| canonical-datafusion-negative-zero-comparison-001-v2 | datafusion | ok | ok | True | equal |
| canonical-datafusion-negative-zero-comparison-001-v3 | datafusion | ok | ok | True | equal |
| canonical-datafusion-negative-zero-comparison-001-v4 | datafusion | ok | ok | True | equal |
| canonical-datafusion-negative-zero-comparison-001-v5 | datafusion | ok | ok | True | equal |
| canonical-datafusion-negative-zero-comparison-001-v6 | datafusion | ok | ok | True | equal |
| canonical-datafusion-negative-zero-comparison-001-v7 | datafusion | ok | ok | True | equal |
| canonical-datafusion-negative-zero-comparison-001-v8 | datafusion | ok | ok | True | equal |
| canonical-datafusion-negative-zero-comparison-001-v9 | datafusion | ok | ok | True | equal |
| datafusion_order_by_offset_subquery_groupby.json | datafusion | ok | ok | True | equal |
| canonical-datafusion-order-by-offset-subquery-groupby-001-v0 | datafusion | ok | ok | True | equal |
| canonical-datafusion-order-by-offset-subquery-groupby-001-v1 | datafusion | ok | ok | True | equal |
| canonical-datafusion-order-by-offset-subquery-groupby-001-v2 | datafusion | ok | ok | True | equal |
| canonical-datafusion-order-by-offset-subquery-groupby-001-v3 | datafusion | ok | ok | True | equal |
| canonical-datafusion-order-by-offset-subquery-groupby-001-v4 | datafusion | ok | ok | True | equal |
| canonical-datafusion-order-by-offset-subquery-groupby-001-v5 | datafusion | ok | ok | True | equal |
| canonical-datafusion-order-by-offset-subquery-groupby-001-v6 | datafusion | ok | ok | True | equal |
| canonical-datafusion-order-by-offset-subquery-groupby-001-v7 | datafusion | ok | ok | True | equal |
| canonical-datafusion-order-by-offset-subquery-groupby-001-v8 | datafusion | ok | ok | True | equal |
| canonical-datafusion-order-by-offset-subquery-groupby-001-v9 | datafusion | ok | ok | True | equal |
| duckdb_join_filter_pushdown_limit.json | datafusion | ok | ok | True | equal |
| canonical-duckdb-join-filter-pushdown-limit-001-v0 | datafusion | ok | ok | True | equal |
| canonical-duckdb-join-filter-pushdown-limit-001-v1 | datafusion | ok | ok | True | equal |
| canonical-duckdb-join-filter-pushdown-limit-001-v2 | datafusion | ok | ok | True | equal |
| canonical-duckdb-join-filter-pushdown-limit-001-v3 | datafusion | ok | ok | True | equal |
| canonical-duckdb-join-filter-pushdown-limit-001-v4 | datafusion | ok | ok | True | equal |
| canonical-duckdb-join-filter-pushdown-limit-001-v5 | datafusion | ok | ok | True | equal |
| canonical-duckdb-join-filter-pushdown-limit-001-v6 | datafusion | ok | ok | True | equal |
| canonical-duckdb-join-filter-pushdown-limit-001-v7 | datafusion | ok | ok | True | equal |
| canonical-duckdb-join-filter-pushdown-limit-001-v8 | datafusion | ok | ok | True | equal |
| canonical-duckdb-join-filter-pushdown-limit-001-v9 | datafusion | ok | ok | True | equal |
| polars_grouped_max_sort_metadata.json | datafusion | ok | ok | True | equal |
| canonical-polars-grouped-max-sort-metadata-001-v0 | datafusion | ok | ok | True | equal |
| canonical-polars-grouped-max-sort-metadata-001-v1 | datafusion | ok | ok | True | equal |
| canonical-polars-grouped-max-sort-metadata-001-v2 | datafusion | ok | ok | True | equal |
| canonical-polars-grouped-max-sort-metadata-001-v3 | datafusion | ok | ok | True | equal |
| canonical-polars-grouped-max-sort-metadata-001-v4 | datafusion | ok | ok | True | equal |
| canonical-polars-grouped-max-sort-metadata-001-v5 | datafusion | ok | ok | True | equal |
| canonical-polars-grouped-max-sort-metadata-001-v6 | datafusion | ok | ok | True | equal |
| canonical-polars-grouped-max-sort-metadata-001-v7 | datafusion | ok | ok | True | equal |
| canonical-polars-grouped-max-sort-metadata-001-v8 | datafusion | ok | ok | True | equal |
| canonical-polars-grouped-max-sort-metadata-001-v9 | datafusion | ok | ok | True | equal |
| polars_reflected_arithmetic_operand_order.json | datafusion | ok | ok | True | equal |
| canonical-polars-reflected-arithmetic-operand-order-001-v0 | datafusion | ok | ok | True | equal |
| canonical-polars-reflected-arithmetic-operand-order-001-v1 | datafusion | ok | ok | True | equal |
| canonical-polars-reflected-arithmetic-operand-order-001-v2 | datafusion | ok | ok | True | equal |
| canonical-polars-reflected-arithmetic-operand-order-001-v3 | datafusion | ok | ok | True | equal |
| canonical-polars-reflected-arithmetic-operand-order-001-v4 | datafusion | ok | ok | True | equal |
| canonical-polars-reflected-arithmetic-operand-order-001-v5 | datafusion | ok | ok | True | equal |
| canonical-polars-reflected-arithmetic-operand-order-001-v6 | datafusion | ok | ok | True | equal |
| canonical-polars-reflected-arithmetic-operand-order-001-v7 | datafusion | ok | ok | True | equal |
| canonical-polars-reflected-arithmetic-operand-order-001-v8 | datafusion | ok | ok | True | equal |
| canonical-polars-reflected-arithmetic-operand-order-001-v9 | datafusion | ok | ok | True | equal |
| pyarrow_sliced_bool_groupby.json | datafusion | ok | ok | True | equal |
| canonical-pyarrow-sliced-bool-groupby-001-v0 | datafusion | ok | ok | True | equal |
| canonical-pyarrow-sliced-bool-groupby-001-v1 | datafusion | ok | ok | True | equal |
| canonical-pyarrow-sliced-bool-groupby-001-v2 | datafusion | ok | ok | True | equal |
| canonical-pyarrow-sliced-bool-groupby-001-v3 | datafusion | ok | ok | True | equal |
| canonical-pyarrow-sliced-bool-groupby-001-v4 | datafusion | ok | ok | True | equal |
| canonical-pyarrow-sliced-bool-groupby-001-v5 | datafusion | ok | ok | True | equal |
| canonical-pyarrow-sliced-bool-groupby-001-v6 | datafusion | ok | ok | True | equal |
| canonical-pyarrow-sliced-bool-groupby-001-v7 | datafusion | ok | ok | True | equal |
| canonical-pyarrow-sliced-bool-groupby-001-v8 | datafusion | ok | ok | True | equal |
| canonical-pyarrow-sliced-bool-groupby-001-v9 | datafusion | ok | ok | True | equal |

## Finding: datafusion_distinct_null_topk.json / datafusion

- left (datafusion-53.0.0--pyarrow-24.0.0): `[[""]]`
- right (frozen-target): `[[null]]`

## Finding: canonical-datafusion-distinct-null-topk-001-v0 / datafusion

- left (datafusion-53.0.0--pyarrow-24.0.0): `[[""]]`
- right (frozen-target): `[[null]]`

## Finding: canonical-datafusion-distinct-null-topk-001-v3 / datafusion

- left (datafusion-53.0.0--pyarrow-24.0.0): `[[""]]`
- right (frozen-target): `[[null]]`

## Finding: canonical-datafusion-distinct-null-topk-001-v4 / datafusion

- left (datafusion-53.0.0--pyarrow-24.0.0): `[["a"]]`
- right (frozen-target): `[[null]]`

## Finding: canonical-datafusion-distinct-null-topk-001-v5 / datafusion

- left (datafusion-53.0.0--pyarrow-24.0.0): `[[""]]`
- right (frozen-target): `[[null]]`

## Finding: canonical-datafusion-distinct-null-topk-001-v6 / datafusion

- left (datafusion-53.0.0--pyarrow-24.0.0): `[[""]]`
- right (frozen-target): `[[null]]`

## Finding: canonical-datafusion-distinct-null-topk-001-v7 / datafusion

- left (datafusion-53.0.0--pyarrow-24.0.0): `[[""]]`
- right (frozen-target): `[[null]]`

## Finding: canonical-datafusion-distinct-null-topk-001-v8 / datafusion

- left (datafusion-53.0.0--pyarrow-24.0.0): `[["a"]]`
- right (frozen-target): `[[null]]`

## Finding: canonical-datafusion-distinct-null-topk-001-v9 / datafusion

- left (datafusion-53.0.0--pyarrow-24.0.0): `[[""]]`
- right (frozen-target): `[[null]]`
