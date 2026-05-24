# Bug Artifact: case-00100081-bughunt-mut-100630

Seed: `100630`

## Program

```json
{
  "program_id": "prog-00100081-mut-100630",
  "seed": 100630,
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
        "source": "z",
        "op": "mul",
        "value": 10
      }
    },
    {
      "op": "filter",
      "column": "m_0",
      "cmp": "le_is_not_false",
      "value": 0.5
    },
    {
      "op": "sort",
      "columns": [
        "tag",
        "flag",
        "g",
        "id",
        "j",
        "m_0",
        "s",
        "x",
        "y",
        "z"
      ],
      "ascending": true
    },
    {
      "op": "select",
      "columns": [
        "z"
      ]
    },
    {
      "op": "offset",
      "n": 2
    },
    {
      "op": "sort",
      "columns": [
        "z"
      ],
      "ascending": true
    }
  ]
}
```

## Findings
- **semantic_output_mismatch** severity=critical root=outer_join_truth_filter oracle=differential confidence=high triage=candidate_implementation_bug triage_confidence=high suspicious=['datafusion']: Backends returned different canonical tables; mismatch_class=value; shapes={'pandas': (22, 1), 'duckdb': (22, 1), 'datafusion': (22, 1)}
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
