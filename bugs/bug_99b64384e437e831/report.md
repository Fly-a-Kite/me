# Bug Artifact: case-00104257-bughunt-mut-109514

Seed: `109514`

## Program

```json
{
  "program_id": "prog-00104257-mut-109514",
  "seed": 109514,
  "operations": [
    {
      "op": "join",
      "table": "t1",
      "left_on": "id",
      "right_on": "id",
      "how": "inner"
    },
    {
      "op": "mutate",
      "column": "m_0",
      "expr": {
        "kind": "add_const",
        "source": "id",
        "value": -2
      }
    },
    {
      "op": "sort",
      "keys": [
        {
          "column": "id",
          "ascending": true,
          "nulls": "first"
        },
        {
          "column": "flag",
          "ascending": true,
          "nulls": "last"
        },
        {
          "column": "g",
          "ascending": true,
          "nulls": "first"
        },
        {
          "column": "j",
          "ascending": false,
          "nulls": "first"
        },
        {
          "column": "m_0",
          "ascending": true,
          "nulls": "last"
        },
        {
          "column": "s",
          "ascending": true,
          "nulls": "first"
        },
        {
          "column": "tag",
          "ascending": true,
          "nulls": "last"
        },
        {
          "column": "x",
          "ascending": false,
          "nulls": "first"
        },
        {
          "column": "y",
          "ascending": false,
          "nulls": "first"
        },
        {
          "column": "z",
          "ascending": true,
          "nulls": "last"
        }
      ]
    },
    {
      "op": "select",
      "columns": [
        "g",
        "j",
        "s"
      ]
    },
    {
      "op": "offset",
      "n": 1
    },
    {
      "op": "sort",
      "keys": [
        {
          "column": "s",
          "ascending": false,
          "nulls": "first"
        },
        {
          "column": "g",
          "ascending": false,
          "nulls": "first"
        },
        {
          "column": "j",
          "ascending": true,
          "nulls": "last"
        }
      ]
    },
    {
      "op": "select",
      "columns": [
        "g"
      ]
    },
    {
      "op": "limit",
      "n": 7
    }
  ]
}
```

## Findings
- **semantic_output_mismatch** severity=critical root=joined_order_offset_projection oracle=differential confidence=high triage=candidate_implementation_bug triage_confidence=high suspicious=['datafusion']: Backends returned different canonical tables; mismatch_class=value; shapes={'pandas': (7, 1), 'duckdb': (7, 1), 'datafusion': (7, 1)}
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
