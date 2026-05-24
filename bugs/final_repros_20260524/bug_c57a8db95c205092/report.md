# Bug Artifact: case-00009513-bughunt-row-value-absence-filter

Seed: `9513`

## Program

```json
{
  "program_id": "prog-00009513-row-value-absence-filter",
  "seed": 9513,
  "operations": [
    {
      "op": "tuple_absence_filter",
      "columns": [
        "left_a",
        "left_b"
      ],
      "table": "t1",
      "right_columns": [
        "right_a",
        "right_b"
      ]
    }
  ]
}
```

## Findings
- **semantic_output_mismatch** severity=critical root=tuple_absence_null_filter oracle=differential confidence=medium triage=candidate_implementation_bug triage_confidence=high suspicious=['duckdb']: Backends returned different canonical tables; mismatch_class=row_count; shapes={'duckdb': (0, 2), 'sqlite': (1, 2)}
  - triage_evidence=Independent DSL reference agrees with ['sqlite'] and disagrees with ['duckdb'].

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
