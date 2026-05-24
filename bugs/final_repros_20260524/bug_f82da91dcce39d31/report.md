# Bug Artifact: case-00009532-bughunt-polars-reverse-division-columns

Seed: `9532`

## Program

```json
{
  "program_id": "prog-00009532-polars-reverse-division-columns",
  "seed": 9532,
  "operations": [
    {
      "op": "mutate",
      "column": "ratio_value",
      "expr": {
        "kind": "reverse_division_columns",
        "source": "divisor_value",
        "numerator": "numerator_value"
      }
    }
  ]
}
```

## Findings
- **semantic_output_mismatch** severity=critical root=reverse_division_operand_order oracle=differential confidence=medium triage=candidate_implementation_bug triage_confidence=high suspicious=['polars']: Backends returned different canonical tables; mismatch_class=value; shapes={'polars': (1, 3), 'polars_lazy': (1, 3)}
  - triage_evidence=Independent DSL reference agrees with ['polars_lazy'] and disagrees with ['polars'].

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
