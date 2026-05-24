# Bug Artifact: case-00009762-bughunt-pyarrow-groupby-filter-cast-membership

Seed: `9762`

## Program

```json
{
  "program_id": "prog-00009762-pyarrow-groupby-filter-cast-membership",
  "seed": 9762,
  "operations": [
    {
      "op": "groupby",
      "keys": [
        "bucket_code",
        "amount_code"
      ],
      "aggs": [
        {
          "column": "bucket_code",
          "func": "min",
          "as": "agg_min_bucket"
        },
        {
          "column": "amount_code",
          "func": "max",
          "as": "agg_max_amount"
        }
      ]
    },
    {
      "op": "filter",
      "column": "agg_min_bucket",
      "cmp": "in_set",
      "value": [
        0.5,
        -1.0,
        10.0
      ]
    }
  ]
}
```

## Findings
- **semantic_output_mismatch** severity=critical root=groupby_aggregation oracle=differential confidence=high triage=candidate_implementation_bug triage_confidence=high suspicious=['pyarrow']: Backends returned different canonical tables; mismatch_class=row_count; shapes={'pandas': (0, 4), 'duckdb': (0, 4), 'pyarrow': (1, 4)}
  - triage_evidence=Independent DSL reference agrees with ['duckdb', 'pandas'] and disagrees with ['pyarrow'].

## Reproduce

```bash
python reproduce.py
```

## Environment

```json
{
  "python": "3.12.3 (main, Mar 23 2026, 19:04:32) [GCC 13.3.0]",
  "platform": "Linux-6.17.0-29-generic-x86_64-with-glibc2.39",
  "pandas": "3.0.3",
  "polars": "1.40.1",
  "duckdb": "1.5.3",
  "sqlite": "3.51.1",
  "sqlite_runtime": "pysqlite3",
  "pysqlite3_binary": "0.5.4.post2",
  "datadiff_fuzz_lab": "0.1.0"
}
```
