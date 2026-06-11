# Auto Issue Draft: groupby_aggregation@duckdb

## Discovery Record

| Item | Value |
| --- | --- |
| Project family labels observed | `groupby_aggregation@duckdb` |
| First DataDiffFuzz signal | 2026-06-06T18:06:13Z |
| How found | Automated candidate pipeline from `runs/run-20260606T172519-1780766719919044625.jsonl.gz` |
| Current status | Needs final upstream dedup before submission |

## Environment

- Target backend/version: see `bugs/bug_ac1df63653882ba5/environment.json`
- Candidate artifact: `bugs/bug_ac1df63653882ba5`
- Pipeline manifest directory: `new_issue/generated/candidate-pipelines/pipeline-discovery-campaign-arrow_datafusion-20260606T171300Z-20260606T173813`

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
result = run_loaded_case(case, backends=['duckdb', 'pandas', 'pyarrow'], config=config, save_artifact=False)
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
Reproduced roots: groupby_aggregation
Local duplicate drafts: none
```

## DataDiffFuzz Evidence

- Frozen candidate evidence: `new_issue/generated/candidate-pipelines/pipeline-discovery-campaign-arrow_datafusion-20260606T171300Z-20260606T173813/frozen-candidates.json`
- Source evidence file: `new_issue/generated/discovery-campaign-arrow_datafusion-20260606T171300Z-arrow_layout-seed501-fresh-candidates.json`
- Source run log: `runs/run-20260606T172519-1780766719919044625.jsonl.gz`
- Candidate artifact: `bugs/bug_ac1df63653882ba5`
- Triage JSON: `bugs/bug_ac1df63653882ba5/triage.json`
- Triage markdown: `bugs/bug_ac1df63653882ba5/triage.md`
- Reduced case: `bugs/bug_ac1df63653882ba5/reduced_case.json`
- Reduced reproducer: `bugs/bug_ac1df63653882ba5/reproduce_reduced.py`
