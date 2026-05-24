# Bug Artifact: case-00095328-bughunt-mut-95415

Seed: `95415`

## Program

```json
{
  "program_id": "prog-00095328-mut-95415",
  "seed": 95415,
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
        "kind": "add_const",
        "source": "z",
        "value": 2
      }
    },
    {
      "op": "filter",
      "column": "s",
      "cmp": "!=",
      "value": "中文"
    },
    {
      "op": "sort",
      "columns": [
        "z",
        "flag",
        "g",
        "id",
        "j",
        "m_0",
        "s",
        "tag",
        "x",
        "y"
      ],
      "ascending": true
    },
    {
      "op": "offset",
      "n": 11
    },
    {
      "op": "groupby",
      "keys": [
        "flag"
      ],
      "aggs": [
        {
          "column": "y",
          "func": "sum",
          "as": "sum_y"
        },
        {
          "column": "m_0",
          "func": "min",
          "as": "min_m_0"
        },
        {
          "column": "x",
          "func": "sum",
          "as": "sum_x"
        }
      ]
    },
    {
      "op": "sort",
      "columns": [
        "sum_y",
        "flag",
        "min_m_0",
        "sum_x"
      ],
      "ascending": false
    },
    {
      "op": "limit",
      "n": 4
    }
  ]
}
```

## Findings
- **semantic_output_mismatch** severity=critical root=groupby_aggregation oracle=differential confidence=high triage=candidate_implementation_bug triage_confidence=high suspicious=['datafusion']: Backends returned different canonical tables; mismatch_class=value; shapes={'pandas': (2, 4), 'duckdb': (2, 4), 'datafusion': (2, 4)}
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
