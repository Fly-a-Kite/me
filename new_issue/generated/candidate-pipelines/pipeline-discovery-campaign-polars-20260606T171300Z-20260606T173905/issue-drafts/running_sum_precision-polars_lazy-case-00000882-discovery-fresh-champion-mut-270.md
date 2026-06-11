# Auto Issue Draft: running_sum_precision@polars_lazy

## Discovery Record

| Item | Value |
| --- | --- |
| Project family labels observed | `running_sum_precision@polars_lazy` |
| First DataDiffFuzz signal | 2026-06-06T17:43:13Z |
| How found | Automated candidate pipeline from `runs/run-20260606T172840-1780766920545021491.jsonl.gz` |
| Current status | Needs final upstream dedup before submission |

## Environment

- Target backend/version: see `bugs/bug_d9b56efa4a29884d/environment.json`
- Candidate artifact: `bugs/bug_d9b56efa4a29884d`
- Pipeline manifest directory: `new_issue/generated/candidate-pipelines/pipeline-discovery-campaign-polars-20260606T171300Z-20260606T173905`

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
result = run_loaded_case(case, backends=['pandas', 'polars', 'polars_lazy'], config=config, save_artifact=False)
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
Suspicious backends: polars_lazy
Reproduced roots: running_sum_precision
Local duplicate drafts: none
```

## DataDiffFuzz Evidence

- Frozen candidate evidence: `new_issue/generated/candidate-pipelines/pipeline-discovery-campaign-polars-20260606T171300Z-20260606T173905/frozen-candidates.json`
- Source evidence file: `new_issue/generated/discovery-campaign-polars-20260606T171300Z-polars_lazy-seed501-fresh-candidates.json`
- Source run log: `runs/run-20260606T172840-1780766920545021491.jsonl.gz`
- Candidate artifact: `bugs/bug_d9b56efa4a29884d`
- Triage JSON: `bugs/bug_d9b56efa4a29884d/triage.json`
- Triage markdown: `bugs/bug_d9b56efa4a29884d/triage.md`
- Reduced case: `bugs/bug_d9b56efa4a29884d/reduced_case.json`
- Reduced reproducer: `bugs/bug_d9b56efa4a29884d/reproduce_reduced.py`
