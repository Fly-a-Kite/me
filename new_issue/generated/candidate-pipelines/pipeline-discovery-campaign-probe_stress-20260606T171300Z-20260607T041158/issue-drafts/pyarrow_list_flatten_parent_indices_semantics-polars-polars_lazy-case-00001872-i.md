# Auto Issue Draft: pyarrow_list_flatten_parent_indices_semantics@polars,polars_lazy

## Discovery Record

| Item | Value |
| --- | --- |
| Project family labels observed | `pyarrow_list_flatten_parent_indices_semantics@polars,polars_lazy` |
| First DataDiffFuzz signal | 2026-06-07T04:12:20Z |
| How found | Automated candidate pipeline from `runs/run-20260607T021439-1780798479224195303.jsonl.gz` |
| Current status | Needs final upstream dedup before submission |

## Environment

- Target backend/version: see `bugs/bug_342547b6ee046401/environment.json`
- Candidate artifact: `bugs/bug_342547b6ee046401`
- Pipeline manifest directory: `new_issue/generated/candidate-pipelines/pipeline-discovery-campaign-probe_stress-20260606T171300Z-20260607T041158`

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
Suspicious backends: polars, polars_lazy
Reproduced roots: pyarrow_list_flatten_parent_indices_semantics
Local duplicate drafts: none
```

## DataDiffFuzz Evidence

- Frozen candidate evidence: `new_issue/generated/candidate-pipelines/pipeline-discovery-campaign-probe_stress-20260606T171300Z-20260607T041158/frozen-candidates.json`
- Source evidence file: `new_issue/generated/discovery-campaign-probe_stress-20260606T171300Z-polars_probe_stress-seed1001-fresh-candidates.json`
- Source run log: `runs/run-20260607T021439-1780798479224195303.jsonl.gz`
- Candidate artifact: `bugs/bug_342547b6ee046401`
- Triage JSON: `bugs/bug_342547b6ee046401/triage.json`
- Triage markdown: `bugs/bug_342547b6ee046401/triage.md`
- Reduced case: `bugs/bug_342547b6ee046401/reduced_case.json`
- Reduced reproducer: `bugs/bug_342547b6ee046401/reproduce_reduced.py`
