# Triage Report: case-00022234-bughunt

- Verdict: `candidate_implementation_bug`
- Paper status: `candidate_bug_needs_external_confirmation`
- Confidence: `high`
- Generator profile: `bughunt`
- Backends: pandas, duckdb, datafusion
- Rows: 2
- Operations: 6
- Reduced: True
- Original kinds: semantic_output_mismatch
- Reproduced kinds: semantic_output_mismatch
- Reproduced roots: groupby_aggregation
- Suspicious backends: datafusion

## Features

- contains_null: False
- contains_nan: False
- contains_inf: False
- contains_non_ascii_string: False
- uses_filter: False
- uses_mutate: False
- uses_modulo: False
- uses_string_lower: False
- uses_groupby: True
- uses_sort: True
- uses_limit: True
- uses_offset: True
- operation_sequence: ['join', 'groupby', 'sort', 'limit', 'sort', 'offset']

## Recommendation

- Minimize the artifact and create a backend-specific reproduction script.
- Check documentation and release notes, then submit an upstream issue.
- Count as confirmed only after maintainer acknowledgement, fix, or clear spec violation.
