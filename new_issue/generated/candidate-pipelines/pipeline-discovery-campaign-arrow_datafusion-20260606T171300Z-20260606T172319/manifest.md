# DataDiffFuzz Candidate Pipeline

- Generated at: `2026-06-06T17:25:19Z`
- Frozen candidates: `11`
- Rechecked candidates: `11`
- Reproduced candidates: `11`
- Reduced artifacts: `11`
- Candidate-bug triage verdicts: `3`
- Local duplicate families: `8`
- Issue drafts: `3`
- Needs dedup check: `3`

## Candidates

| Candidate | Family | Score | True bug probability | Reproduced | Triage | Dedup | Issue readiness |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `running_sum_precision-duckdb-case-00001568-discovery-fresh` | `running_sum_precision@duckdb` | `5.53` | `1.00` | `true` | `not_reproduced` | `needs_final_upstream_dedup` | `` |
| `running_sum_precision-duckdb-case-00001640-discovery-fresh` | `running_sum_precision@duckdb` | `5.53` | `1.00` | `true` | `semantic_divergence_needs_confirmation` | `duplicate_pipeline_family` | `` |
| `running_sum_precision-duckdb-case-00001158-discovery-fresh` | `running_sum_precision@duckdb` | `5.47` | `1.00` | `true` | `candidate_implementation_bug` | `duplicate_pipeline_family` | `needs_dedup_check` |
| `running_sum_precision-duckdb-case-00000981-discovery-fresh` | `running_sum_precision@duckdb` | `5.45` | `1.00` | `true` | `candidate_implementation_bug` | `duplicate_pipeline_family` | `needs_dedup_check` |
| `conditional_expression-duckdb-case-00000882-discovery-fresh-champion-mut-2708-champion-mut-2795` | `conditional_expression@duckdb` | `5.41` | `1.00` | `true` | `not_reproduced` | `needs_final_upstream_dedup` | `` |
| `conditional_expression-duckdb-case-00000882-discovery-fresh-champion-mut-2708-champion-mut-2758` | `conditional_expression@duckdb` | `5.40` | `1.00` | `true` | `not_reproduced` | `duplicate_pipeline_family` | `` |
| `conditional_expression-duckdb-case-00000882-discovery-fresh-champion-mut-2708` | `conditional_expression@duckdb` | `5.38` | `1.00` | `true` | `semantic_divergence_needs_confirmation` | `duplicate_pipeline_family` | `` |
| `conditional_expression-duckdb-case-00000882-discovery-fresh` | `conditional_expression@duckdb` | `5.37` | `1.00` | `true` | `candidate_implementation_bug` | `duplicate_pipeline_family` | `needs_dedup_check` |
| `groupby_aggregation-duckdb-case-00000882-discovery-fresh-champion-mut-2708-champion-mut-2795-mut-2802` | `groupby_aggregation@duckdb` | `5.35` | `1.00` | `true` | `semantic_divergence_needs_confirmation` | `needs_final_upstream_dedup` | `` |
| `groupby_aggregation-duckdb-case-00000882-discovery-fresh-mut-2829-mut-2836` | `groupby_aggregation@duckdb` | `5.35` | `1.00` | `true` | `semantic_divergence_needs_confirmation` | `duplicate_pipeline_family` | `` |
| `groupby_aggregation-duckdb-case-00000882-discovery-fresh-mut-2829` | `groupby_aggregation@duckdb` | `5.34` | `1.00` | `true` | `not_reproduced` | `duplicate_pipeline_family` | `` |
