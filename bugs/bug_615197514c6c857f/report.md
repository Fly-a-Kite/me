# Bug Artifact: case-00022234-bughunt

Seed: `22234`

## Program

```json
{
  "program_id": "prog-00022234",
  "seed": 22234,
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
        "kind": "string_length",
        "source": "tag"
      }
    },
    {
      "op": "filter",
      "column": "z",
      "cmp": "<=",
      "value": 1.0
    },
    {
      "op": "groupby",
      "keys": [
        "id"
      ],
      "aggs": [
        {
          "column": "j",
          "func": "nunique",
          "as": "nunique_j"
        },
        {
          "column": "id",
          "func": "count",
          "as": "count_id"
        }
      ]
    },
    {
      "op": "sort",
      "keys": [
        {
          "column": "id",
          "ascending": false,
          "nulls": "last"
        },
        {
          "column": "count_id",
          "ascending": false,
          "nulls": "last"
        },
        {
          "column": "nunique_j",
          "ascending": false,
          "nulls": "last"
        }
      ]
    },
    {
      "op": "limit",
      "n": 8
    },
    {
      "op": "sort",
      "keys": [
        {
          "column": "id",
          "ascending": false,
          "nulls": "last"
        },
        {
          "column": "count_id",
          "ascending": false,
          "nulls": "last"
        },
        {
          "column": "nunique_j",
          "ascending": true,
          "nulls": "last"
        }
      ]
    },
    {
      "op": "select",
      "columns": [
        "count_id"
      ]
    },
    {
      "op": "offset",
      "n": 1
    }
  ]
}
```

## Findings
- **semantic_output_mismatch** severity=critical root=groupby_aggregation oracle=differential confidence=high triage=candidate_implementation_bug triage_confidence=high suspicious=['datafusion']: Backends returned different canonical tables; mismatch_class=row_count; shapes={'pandas': (1, 1), 'duckdb': (1, 1), 'datafusion': (0, 1)}
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
