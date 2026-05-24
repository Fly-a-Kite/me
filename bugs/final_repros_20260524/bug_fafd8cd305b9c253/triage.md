# Triage Report: case-00009762-bughunt-pyarrow-groupby-filter-cast-membership

- Verdict: `not_reproduced`
- Paper status: `not_usable_until_reproduced`
- Confidence: `low`
- Generator profile: `bughunt`
- Backends: pandas, duckdb, pyarrow
- Rows: 1
- Operations: 2
- Reduced: False
- Original kinds: semantic_output_mismatch
- Reproduced kinds: none
- Reproduced roots: none
- Suspicious backends: none

## Features

- contains_null: False
- contains_nan: False
- contains_inf: False
- contains_non_ascii_string: False
- uses_filter: True
- uses_mutate: False
- uses_modulo: False
- uses_string_lower: False
- uses_groupby: True
- uses_sort: False
- uses_limit: False
- uses_offset: False
- operation_sequence: ['groupby', 'filter']

## Recommendation

- Do not count this artifact as a bug until reproduction succeeds.
- Check dependency versions and backend list.
