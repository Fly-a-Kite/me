# Auto Issue Draft: conditional_expression@duckdb

## Discovery Record

| Item | Value |
| --- | --- |
| Project family labels observed | `conditional_expression@duckdb` |
| First DataDiffFuzz signal | 2026-06-06T17:44:54Z |
| How found | Automated candidate pipeline from `runs/run-20260606T172519-1780766719919044625.jsonl.gz` |
| Current status | Needs final upstream dedup before submission |

## Environment

- Target backend/version: see `bugs/bug_f58eceac0a548b48/environment.json`
- Candidate artifact: `bugs/bug_f58eceac0a548b48`
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
Reproduced roots: conditional_expression
Local duplicate drafts: none
```

## DataDiffFuzz Evidence

- Frozen candidate evidence: `new_issue/generated/candidate-pipelines/pipeline-discovery-campaign-arrow_datafusion-20260606T171300Z-20260606T173813/frozen-candidates.json`
- Source evidence file: `new_issue/generated/discovery-campaign-arrow_datafusion-20260606T171300Z-arrow_layout-seed501-fresh-candidates.json`
- Source run log: `runs/run-20260606T172519-1780766719919044625.jsonl.gz`
- Candidate artifact: `bugs/bug_f58eceac0a548b48`
- Triage JSON: `bugs/bug_f58eceac0a548b48/triage.json`
- Triage markdown: `bugs/bug_f58eceac0a548b48/triage.md`
- Reduced case: `bugs/bug_f58eceac0a548b48/reduced_case.json`
- Reduced reproducer: `bugs/bug_f58eceac0a548b48/reproduce_reduced.py`
