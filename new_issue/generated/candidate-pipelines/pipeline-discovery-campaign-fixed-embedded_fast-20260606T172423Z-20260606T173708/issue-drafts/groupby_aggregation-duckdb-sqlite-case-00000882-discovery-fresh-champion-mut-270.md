# Auto Issue Draft: groupby_aggregation@duckdb,sqlite

## Discovery Record

| Item | Value |
| --- | --- |
| Project family labels observed | `groupby_aggregation@duckdb,sqlite` |
| First DataDiffFuzz signal | 2026-06-06T20:48:41Z |
| How found | Automated candidate pipeline from `runs/run-20260606T172954-1780766994673624503.jsonl.gz` |
| Current status | Needs final upstream dedup before submission |

## Environment

- Target backend/version: see `bugs/bug_f7e852706cde3d74/environment.json`
- Candidate artifact: `bugs/bug_f7e852706cde3d74`
- Pipeline manifest directory: `new_issue/generated/candidate-pipelines/pipeline-discovery-campaign-fixed-embedded_fast-20260606T172423Z-20260606T173708`

## Reproducer

```python
#!/usr/bin/env python3
from datadiff.config import ExperimentConfig
from datadiff.dsl import Case
from datadiff.runner import run_loaded_case
from datadiff.util import load_json

here = __import__("pathlib").Path(__file__).parent
case = Case.from_dict(load_json(here / "reduced_case.json"))
config_data = load_json(here / "config.json")
config = ExperimentConfig.from_payload(config_data)
result = run_loaded_case(case, backends=['duckdb', 'pandas', 'sqlite'], config=config, save_artifact=False)
print(result["status"])
for finding in result["findings"]:
    print(finding)
```

## Expected Output

```text
Backends should agree on the final normalized result for this case.
```

## Actual Output

```text
Triage verdict: candidate_implementation_bug
Suspicious backends: duckdb, sqlite
Reproduced roots: groupby_aggregation
Local duplicate drafts: none
```

## DataDiffFuzz Evidence

- Frozen candidate evidence: `new_issue/generated/candidate-pipelines/pipeline-discovery-campaign-fixed-embedded_fast-20260606T172423Z-20260606T173708/frozen-candidates.json`
- Source evidence file: `new_issue/generated/discovery-campaign-fixed-embedded_fast-20260606T172423Z-embedded_sql-seed511-fresh-candidates.json`
- Source run log: `runs/run-20260606T172954-1780766994673624503.jsonl.gz`
- Candidate artifact: `bugs/bug_f7e852706cde3d74`
- Triage JSON: `bugs/bug_f7e852706cde3d74/triage.json`
- Triage markdown: `bugs/bug_f7e852706cde3d74/triage.md`
- Reduced case: `bugs/bug_f7e852706cde3d74/reduced_case.json`
- Reduced reproducer: `bugs/bug_f7e852706cde3d74/reproduce_reduced.py`
