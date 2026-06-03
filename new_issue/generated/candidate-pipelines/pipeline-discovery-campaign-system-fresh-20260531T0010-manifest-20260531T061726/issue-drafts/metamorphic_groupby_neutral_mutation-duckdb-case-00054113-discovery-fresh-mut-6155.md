# Auto Issue Draft: metamorphic_groupby_neutral_mutation@duckdb

## Discovery Record

| Item | Value |
| --- | --- |
| Project family labels observed | `metamorphic_groupby_neutral_mutation@duckdb` |
| First DataDiffFuzz signal | 2026-05-31T06:19:41Z |
| How found | Automated candidate pipeline from `runs/run-20260530T165542-1780160142319932143.jsonl.gz` |
| Current status | Needs final upstream dedup before submission |

## Environment

- Target backend/version: see `bugs/bug_6106a2d4ad837c90/environment.json`
- Bug artifact: `bugs/bug_6106a2d4ad837c90`
- Pipeline manifest directory: `new_issue/generated/candidate-pipelines/pipeline-discovery-campaign-system-fresh-20260531T0010-manifest-20260531T061726`

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
config = ExperimentConfig(**config_data) if config_data else ExperimentConfig()
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
Reproduced roots: metamorphic_groupby_neutral_mutation
Local duplicate drafts: none
```

## DataDiffFuzz Evidence

- Frozen candidate evidence: `new_issue/generated/candidate-pipelines/pipeline-discovery-campaign-system-fresh-20260531T0010-manifest-20260531T061726/frozen-candidates.json`
- Source evidence file: `new_issue/generated/discovery-campaign-system-fresh-20260531T0010-manifest-embedded_sql-seed53001-fresh-candidates.json`
- Source run log: `runs/run-20260530T165542-1780160142319932143.jsonl.gz`
- Bug artifact: `bugs/bug_6106a2d4ad837c90`
- Triage JSON: `bugs/bug_6106a2d4ad837c90/triage.json`
- Triage markdown: `bugs/bug_6106a2d4ad837c90/triage.md`
- Reduced case: `bugs/bug_6106a2d4ad837c90/reduced_case.json`
- Reduced reproducer: `bugs/bug_6106a2d4ad837c90/reproduce_reduced.py`
