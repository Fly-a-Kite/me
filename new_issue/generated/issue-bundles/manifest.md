# DataDiffFuzz Issue Bundle

- Generated at: `2026-06-06T18:44:09Z`
- Primary per family: `true`
- Eligible issue documents: `11`
- Bundled issue documents: `10`
- Bundled families: `10`
- Duplicate family drafts: `0`
- Supporting duplicate drafts skipped: `1`
- Extracted reproducers: `10`
- Compile failures: `0`
- Executed reproducers: `10`
- Executed attempts: `10`
- Expected assertion-failure reproducers: `4`
- Fixed-upstream no-longer-reproduced scripts: `1`
- Flaky reproducers: `0`
- Non-zero reproducer exits: `0`

## Family Groups

| Family | Documents | Issues |
| --- | ---: | --- |
| `datafusion_limit_idempotence@datafusion` | 1 | `new_issue/datafusion_limit_idempotence.md` |
| `distinct_null_topk@datafusion` | 1 | `new_issue/datafusion_distinct_null_topk.md` |
| `groupby_aggregation@datafusion` | 1 | `new_issue/datafusion_22489_grouped_limit_offset.md` |
| `groupby_aggregation@polars,polars_lazy` | 1 | `new_issue/polars_27672_grouped_max_sort.md` |
| `grouped_topk_null_sort_key@datafusion` | 1 | `new_issue/datafusion_22190_grouped_null_topk.md` |
| `metamorphic_semi_anti_join_rewrite@duckdb` | 1 | `new_issue/duckdb_cte_boolean_inner_join_empty_result.md` |
| `negative_zero_comparison@datafusion` | 1 | `new_issue/datafusion_22490_negative_zero.md` |
| `polars_reflected_arithmetic_operand_order@polars` | 1 | `new_issue/polars_series_reflected_arithmetic_operand_order.md` |
| `polars_vector_division_rounding@polars` | 1 | `new_issue/polars_vector_division_rounding.md` |
| `pyarrow_sliced_bool_groupby_any_all@pyarrow` | 1 | `new_issue/pyarrow_slice_groupby_any.md` |

## Bundle Selection

| Family | Selected Primary | Supporting Drafts | Skipped Supporting Drafts |
| --- | --- | --- | --- |
| `grouped_topk_null_sort_key@datafusion` | `new_issue/datafusion_22190_grouped_null_topk.md` | - | - |
| `groupby_aggregation@datafusion` | `new_issue/datafusion_22489_grouped_limit_offset.md` | - | - |
| `negative_zero_comparison@datafusion` | `new_issue/datafusion_22490_negative_zero.md` | - | - |
| `distinct_null_topk@datafusion` | `new_issue/datafusion_distinct_null_topk.md` | - | - |
| `datafusion_limit_idempotence@datafusion` | `new_issue/datafusion_limit_idempotence.md` | - | - |
| `metamorphic_semi_anti_join_rewrite@duckdb` | `new_issue/duckdb_cte_boolean_inner_join_empty_result.md` | - | - |
| `groupby_aggregation@polars,polars_lazy` | `new_issue/polars_27672_grouped_max_sort.md` | - | - |
| `polars_reflected_arithmetic_operand_order@polars` | `new_issue/polars_series_reflected_arithmetic_operand_order.md` | `new_issue/polars_series_rtruediv_operand_order.md` | `new_issue/polars_series_rtruediv_operand_order.md` |
| `polars_vector_division_rounding@polars` | `new_issue/polars_vector_division_rounding.md` | - | - |
| `pyarrow_sliced_bool_groupby_any_all@pyarrow` | `new_issue/pyarrow_slice_groupby_any.md` | - | - |

## Reproducers

| Issue | Family | Status | Reproducer | Compile | Run |
| --- | --- | --- | --- | --- | --- |
| `new_issue/datafusion_22190_grouped_null_topk.md` | `grouped_topk_null_sort_key@datafusion` | `already_submitted_or_confirmed` | `new_issue/generated/issue-bundles/reproducers/datafusion_22190_grouped_null_topk.py` | `ok` | `exit 1` |
| `new_issue/datafusion_22489_grouped_limit_offset.md` | `groupby_aggregation@datafusion` | `already_submitted_or_confirmed` | `new_issue/generated/issue-bundles/reproducers/datafusion_22489_grouped_limit_offset.py` | `ok` | `exit 1` |
| `new_issue/datafusion_22490_negative_zero.md` | `negative_zero_comparison@datafusion` | `already_submitted_or_confirmed` | `new_issue/generated/issue-bundles/reproducers/datafusion_22490_negative_zero.py` | `ok` | `exit 1` |
| `new_issue/datafusion_distinct_null_topk.md` | `distinct_null_topk@datafusion` | `already_submitted_or_confirmed` | `new_issue/generated/issue-bundles/reproducers/datafusion_distinct_null_topk.py` | `ok` | `exit 1` |
| `new_issue/datafusion_limit_idempotence.md` | `datafusion_limit_idempotence@datafusion` | `already_submitted_or_confirmed` | `new_issue/generated/issue-bundles/reproducers/datafusion_limit_idempotence.py` | `ok` | `exit 0` |
| `new_issue/duckdb_cte_boolean_inner_join_empty_result.md` | `metamorphic_semi_anti_join_rewrite@duckdb` | `already_submitted_or_confirmed` | `new_issue/generated/issue-bundles/reproducers/duckdb_cte_boolean_inner_join_empty_result.py` | `ok` | `exit 0` |
| `new_issue/polars_27672_grouped_max_sort.md` | `groupby_aggregation@polars,polars_lazy` | `already_submitted_or_confirmed` | `new_issue/generated/issue-bundles/reproducers/polars_27672_grouped_max_sort.py` | `ok` | `exit 1` |
| `new_issue/polars_series_reflected_arithmetic_operand_order.md` | `polars_reflected_arithmetic_operand_order@polars` | `already_submitted_or_confirmed` | `new_issue/generated/issue-bundles/reproducers/polars_series_reflected_arithmetic_operand_order.py` | `ok` | `exit 0` |
| `new_issue/polars_vector_division_rounding.md` | `polars_vector_division_rounding@polars` | `already_submitted_or_confirmed` | `new_issue/generated/issue-bundles/reproducers/polars_vector_division_rounding.py` | `ok` | `exit 0` |
| `new_issue/pyarrow_slice_groupby_any.md` | `pyarrow_sliced_bool_groupby_any_all@pyarrow` | `already_submitted_or_confirmed` | `new_issue/generated/issue-bundles/reproducers/pyarrow_slice_groupby_any.py` | `ok` | `exit 0` |
