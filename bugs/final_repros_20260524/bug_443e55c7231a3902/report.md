# Bug Artifact: case-00011843-bughunt

Seed: `11843`

## Program

```json
{
  "program_id": "prog-00011843",
  "seed": 11843,
  "operations": [
    {
      "op": "join",
      "table": "t1",
      "left_on": "id",
      "right_on": "id",
      "how": "left"
    },
    {
      "op": "groupby",
      "keys": [
        "tag"
      ],
      "aggs": [
        {
          "column": "z",
          "func": "sum",
          "as": "sum_z"
        },
        {
          "column": "y",
          "func": "count",
          "as": "count_y"
        }
      ]
    },
    {
      "op": "sort",
      "keys": [
        {
          "column": "sum_z",
          "ascending": false,
          "nulls": "first"
        },
        {
          "column": "count_y",
          "ascending": false,
          "nulls": "first"
        },
        {
          "column": "tag",
          "ascending": false,
          "nulls": "last"
        }
      ]
    },
    {
      "op": "limit",
      "n": 12
    },
    {
      "op": "sort",
      "columns": [
        "count_y",
        "sum_z"
      ],
      "ascending": false
    },
    {
      "op": "offset",
      "n": 1
    }
  ]
}
```

## Findings
- **semantic_output_mismatch** severity=critical root=grouped_topk_null_sort_key oracle=differential confidence=high triage=candidate_implementation_bug triage_confidence=high suspicious=['datafusion']: Backends returned different canonical tables; mismatch_class=row_count; shapes={'pandas': (1, 3), 'duckdb': (1, 3), 'datafusion': (0, 3)}
  - triage_evidence=Independent DSL reference agrees with ['duckdb', 'pandas'] and disagrees with ['datafusion'].

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
