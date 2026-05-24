# Triage Report: case-00009513-bughunt-row-value-absence-filter

- Verdict: `candidate_implementation_bug`
- Paper status: `candidate_bug_needs_external_confirmation`
- Confidence: `high`
- Generator profile: `bughunt`
- Backends: duckdb, sqlite
- Rows: 1
- Operations: 1
- Reduced: True
- Original kinds: semantic_output_mismatch
- Reproduced kinds: semantic_output_mismatch
- Reproduced roots: tuple_absence_null_filter
- Suspicious backends: duckdb

## Features

- contains_null: True
- contains_nan: False
- contains_inf: False
- contains_non_ascii_string: False
- uses_filter: False
- uses_mutate: False
- uses_modulo: False
- uses_string_lower: False
- uses_groupby: False
- uses_sort: False
- uses_limit: False
- uses_offset: False
- operation_sequence: ['tuple_absence_filter']

## Recommendation

- Minimize the artifact and create a backend-specific reproduction script.
- Check documentation and release notes, then submit an upstream issue.
- Count as confirmed only after maintainer acknowledgement, fix, or clear spec violation.
