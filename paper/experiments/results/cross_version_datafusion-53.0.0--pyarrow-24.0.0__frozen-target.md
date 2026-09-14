# Cross-version smoke: `datafusion-53.0.0--pyarrow-24.0.0` vs `frozen-target`

- left packages: {'chdb': '', 'datafusion': '53.0.0', 'duckdb': '1.5.4', 'pandas': '3.0.0', 'polars': '', 'pyarrow': '24.0.0'}
- right packages: {'chdb': '4.2.1', 'datafusion': '54.0.0', 'duckdb': '1.5.4', 'pandas': '3.0.3', 'polars': '1.42.1', 'pyarrow': '25.0.0'}
- cases: 10; mismatches: 1

| case | backend | left | right | match | reason |
| --- | --- | --- | --- | --- | --- |
| datafusion_distinct_null_topk.json | datafusion | ok | ok | False | result |
| datafusion_groupby_limit_offset.json | datafusion | ok | ok | True | equal |
| datafusion_grouped_null_topk.json | datafusion | ok | ok | True | equal |
| datafusion_limit_idempotence.json | datafusion | ok | ok | True | equal |
| datafusion_negative_zero_comparison.json | datafusion | ok | ok | True | equal |
| datafusion_order_by_offset_subquery_groupby.json | datafusion | ok | ok | True | equal |
| duckdb_join_filter_pushdown_limit.json | datafusion | ok | ok | True | equal |
| polars_grouped_max_sort_metadata.json | datafusion | ok | ok | True | equal |
| polars_reflected_arithmetic_operand_order.json | datafusion | ok | ok | True | equal |
| pyarrow_sliced_bool_groupby.json | datafusion | ok | ok | True | equal |

## Mismatch: datafusion_distinct_null_topk.json

- left rows: `[[""]]`
- right rows: `[[null]]`
