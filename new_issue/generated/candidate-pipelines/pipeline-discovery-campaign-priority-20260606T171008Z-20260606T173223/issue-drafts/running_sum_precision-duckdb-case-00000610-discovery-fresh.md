# Auto Issue Draft: running_sum_precision@duckdb

## Discovery Record

| Item | Value |
| --- | --- |
| Project family labels observed | `running_sum_precision@duckdb` |
| First DataDiffFuzz signal | 2026-06-06T17:38:23Z |
| How found | Automated candidate pipeline from `runs/run-20260606T172152-1780766512875628479.jsonl.gz` |
| Current status | Needs final upstream dedup before submission |

## Environment

- Target backend/version: see `bugs/bug_907b8db166e0872f/environment.json`
- Candidate artifact: `bugs/bug_907b8db166e0872f`
- Pipeline manifest directory: `new_issue/generated/candidate-pipelines/pipeline-discovery-campaign-priority-20260606T171008Z-20260606T173223`

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
Suspicious backends: duckdb
Reproduced roots: running_sum_precision
Local duplicate drafts: none
```

## DataDiffFuzz Evidence

- Frozen candidate evidence: `new_issue/generated/candidate-pipelines/pipeline-discovery-campaign-priority-20260606T171008Z-20260606T173223/frozen-candidates.json`
- Source evidence file: `new_issue/generated/discovery-campaign-priority-20260606T171008Z-embedded_sql-seed501-fresh-candidates.json`
- Source run log: `runs/run-20260606T172152-1780766512875628479.jsonl.gz`
- Candidate artifact: `bugs/bug_907b8db166e0872f`
- Triage JSON: `bugs/bug_907b8db166e0872f/triage.json`
- Triage markdown: `bugs/bug_907b8db166e0872f/triage.md`
- Reduced case: `bugs/bug_907b8db166e0872f/reduced_case.json`
- Reduced reproducer: `bugs/bug_907b8db166e0872f/reproduce_reduced.py`
