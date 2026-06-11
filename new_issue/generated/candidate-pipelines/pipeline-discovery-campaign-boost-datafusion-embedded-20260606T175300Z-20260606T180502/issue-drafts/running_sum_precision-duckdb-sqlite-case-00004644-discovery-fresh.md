# Auto Issue Draft: running_sum_precision@duckdb,sqlite

## Discovery Record

| Item | Value |
| --- | --- |
| Project family labels observed | `running_sum_precision@duckdb,sqlite` |
| First DataDiffFuzz signal | 2026-06-06T21:48:17Z |
| How found | Automated candidate pipeline from `runs/run-20260606T175236-1780768356140820033.jsonl.gz` |
| Current status | Needs final upstream dedup before submission |

## Environment

- Target backend/version: see `bugs/bug_3eff1eb0e0141d5d/environment.json`
- Candidate artifact: `bugs/bug_3eff1eb0e0141d5d`
- Pipeline manifest directory: `new_issue/generated/candidate-pipelines/pipeline-discovery-campaign-boost-datafusion-embedded-20260606T175300Z-20260606T180502`

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
Reproduced roots: running_sum_precision
Local duplicate drafts: none
```

## DataDiffFuzz Evidence

- Frozen candidate evidence: `new_issue/generated/candidate-pipelines/pipeline-discovery-campaign-boost-datafusion-embedded-20260606T175300Z-20260606T180502/frozen-candidates.json`
- Source evidence file: `new_issue/generated/discovery-campaign-boost-datafusion-embedded-20260606T175300Z-embedded_sql-seed4501-fresh-candidates.json`
- Source run log: `runs/run-20260606T175236-1780768356140820033.jsonl.gz`
- Candidate artifact: `bugs/bug_3eff1eb0e0141d5d`
- Triage JSON: `bugs/bug_3eff1eb0e0141d5d/triage.json`
- Triage markdown: `bugs/bug_3eff1eb0e0141d5d/triage.md`
- Reduced case: `bugs/bug_3eff1eb0e0141d5d/reduced_case.json`
- Reduced reproducer: `bugs/bug_3eff1eb0e0141d5d/reproduce_reduced.py`
