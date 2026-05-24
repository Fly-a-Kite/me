# Bug Artifact: case-00141204-bughunt

Seed: `141204`

## Program

```json
{
  "program_id": "prog-00141204",
  "seed": 141204,
  "operations": [
    {
      "op": "join",
      "table": "t1",
      "left_on": "id",
      "right_on": "id",
      "how": "left"
    },
    {
      "op": "mutate",
      "column": "m_0",
      "expr": {
        "kind": "arith_const",
        "source": "y",
        "op": "mul",
        "value": -1
      }
    },
    {
      "op": "filter",
      "column": "m_0",
      "cmp": "ge_is_not_true",
      "value": 0.0
    },
    {
      "op": "sort",
      "columns": [
        "flag",
        "g",
        "id",
        "j",
        "m_0",
        "s",
        "tag",
        "x",
        "y",
        "z"
      ],
      "ascending": false
    },
    {
      "op": "groupby",
      "keys": [
        "tag"
      ],
      "aggs": [
        {
          "column": "z",
          "func": "nunique",
          "as": "nunique_z"
        },
        {
          "column": "j",
          "func": "count",
          "as": "count_j"
        },
        {
          "column": "y",
          "func": "min",
          "as": "min_y"
        }
      ]
    },
    {
      "op": "select",
      "columns": [
        "count_j",
        "min_y",
        "nunique_z"
      ]
    },
    {
      "op": "sort",
      "columns": [
        "nunique_z",
        "count_j",
        "min_y"
      ],
      "ascending": false
    },
    {
      "op": "limit",
      "n": 13
    }
  ]
}
```

## Findings
- **semantic_output_mismatch** severity=critical root=outer_join_truth_filter oracle=differential confidence=high triage=candidate_implementation_bug triage_confidence=high suspicious=['datafusion']: Backends returned different canonical tables; mismatch_class=row_count; shapes={'pandas': (6, 3), 'duckdb': (6, 3), 'datafusion': (7, 3)}
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
