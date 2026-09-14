# Cross-version smoke: `polars-1.40.1` vs `frozen-target`

- left packages: {'chdb': '', 'datafusion': '', 'duckdb': '', 'pandas': '3.0.3', 'polars': '1.40.1', 'pyarrow': '25.0.0'}
- right packages: {'chdb': '4.2.1', 'datafusion': '54.0.0', 'duckdb': '1.5.4', 'pandas': '3.0.3', 'polars': '1.42.1', 'pyarrow': '25.0.0'}
- cases: 10; mismatches: 1

| case | backend | left | right | match | reason |
| --- | --- | --- | --- | --- | --- |
| datafusion_distinct_null_topk.json | polars | ok | ok | True | equal |
| datafusion_groupby_limit_offset.json | polars | ok | ok | True | equal |
| datafusion_grouped_null_topk.json | polars | ok | ok | True | equal |
| datafusion_limit_idempotence.json | polars | ok | ok | True | equal |
| datafusion_negative_zero_comparison.json | polars | ok | ok | True | equal |
| datafusion_order_by_offset_subquery_groupby.json | polars | ok | ok | True | equal |
| duckdb_join_filter_pushdown_limit.json | polars | ok | ok | True | equal |
| polars_grouped_max_sort_metadata.json | polars | ok | ok | False | result |
| polars_reflected_arithmetic_operand_order.json | polars | ok | ok | True | equal |
| pyarrow_sliced_bool_groupby.json | polars | ok | ok | True | equal |

## Mismatch: polars_grouped_max_sort_metadata.json

- left rows: `[[1, "a"]]`
- right rows: `[[2, "b"]]`
