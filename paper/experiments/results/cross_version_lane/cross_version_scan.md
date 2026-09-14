# Cross-version scan

- env pairs: [['datafusion-53.0.0--pyarrow-24.0.0', 'frozen-target']]
- backends: ['datafusion']
- cases: 10
- results: 10
- findings: 1

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

## Finding: datafusion_distinct_null_topk.json / datafusion

- left (datafusion-53.0.0--pyarrow-24.0.0): `[[""]]`
- right (frozen-target): `[[null]]`
