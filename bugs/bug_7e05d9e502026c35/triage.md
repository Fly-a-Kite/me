# Triage Report: case-00095328-bughunt-mut-95415

- Verdict: `candidate_implementation_bug`
- Paper status: `candidate_bug_needs_external_confirmation`
- Confidence: `high`
- Generator profile: `bughunt`
- Backends: pandas, duckdb, datafusion
- Rows: 3
- Operations: 5
- Reduced: True
- Original kinds: semantic_output_mismatch
- Reproduced kinds: semantic_output_mismatch
- Reproduced roots: groupby_aggregation
- Suspicious backends: datafusion

## Features

- contains_null: True
- contains_nan: False
- contains_inf: False
- contains_non_ascii_string: True
- uses_filter: False
- uses_mutate: True
- uses_modulo: False
- uses_string_lower: False
- uses_groupby: True
- uses_sort: True
- uses_limit: False
- uses_offset: True
- operation_sequence: ['join', 'mutate', 'sort', 'offset', 'groupby']

## Recommendation

- Minimize the artifact and create a backend-specific reproduction script.
- Check documentation and release notes, then submit an upstream issue.
- Count as confirmed only after maintainer acknowledgement, fix, or clear spec violation.
