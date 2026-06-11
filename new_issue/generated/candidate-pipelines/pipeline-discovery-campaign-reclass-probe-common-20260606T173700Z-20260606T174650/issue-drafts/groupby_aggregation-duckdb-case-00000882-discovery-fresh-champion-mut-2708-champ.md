# Auto Issue Draft: groupby_aggregation@duckdb

## Discovery Record

| Item | Value |
| --- | --- |
| Project family labels observed | `groupby_aggregation@duckdb` |
| First DataDiffFuzz signal | 2026-06-06T21:01:13Z |
| How found | Automated candidate pipeline from `runs/run-20260606T173702-1780767422560088555.jsonl.gz` |
| Current status | Needs final upstream dedup before submission |

## Environment

- Target backend/version: see `bugs/bug_e4af4153bd26eb57/environment.json`
- Candidate artifact: `bugs/bug_e4af4153bd26eb57`
- Pipeline manifest directory: `new_issue/generated/candidate-pipelines/pipeline-discovery-campaign-reclass-probe-common-20260606T173700Z-20260606T174650`

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

- Frozen candidate evidence: `new_issue/generated/candidate-pipelines/pipeline-discovery-campaign-reclass-probe-common-20260606T173700Z-20260606T174650/frozen-candidates.json`
- Source evidence file: `new_issue/generated/discovery-campaign-reclass-probe-common-20260606T173700Z-arrow_probe_stress-seed3001-fresh-candidates.json`
- Source run log: `runs/run-20260606T173702-1780767422560088555.jsonl.gz`
- Candidate artifact: `bugs/bug_e4af4153bd26eb57`
- Triage JSON: `bugs/bug_e4af4153bd26eb57/triage.json`
- Triage markdown: `bugs/bug_e4af4153bd26eb57/triage.md`
- Reduced case: `bugs/bug_e4af4153bd26eb57/reduced_case.json`
- Reduced reproducer: `bugs/bug_e4af4153bd26eb57/reproduce_reduced.py`
