from __future__ import annotations

from pathlib import Path
from typing import Any

from datadiff.dsl import Case
from datadiff.env import collect_environment
from datadiff.oracle import Finding
from datadiff.targets import target_context
from datadiff.util import BUGS_DIR, dump_json


def save_issue_artifact(
    case: Case,
    raw_results: dict[str, dict[str, Any]],
    normalized: dict[str, dict[str, Any]],
    findings: list[Finding],
    config: dict[str, Any] | None = None,
) -> Path:
    sig = findings[0].signature if findings else case.case_id
    bug_dir = BUGS_DIR / f"bug_{sig}"
    bug_dir.mkdir(parents=True, exist_ok=True)
    dump_json(case.to_dict(), bug_dir / "case.json")
    dump_json(raw_results, bug_dir / "results.json")
    dump_json(normalized, bug_dir / "normalized.json")
    dump_json([f.to_dict() for f in findings], bug_dir / "findings.json")
    dump_json(config or {}, bug_dir / "config.json")
    if isinstance(config, dict):
        strategy_manifest = {
            "strategy_snapshot_path": str(config.get("strategy_snapshot_path", "") or ""),
            "strategy_learning_path": str(config.get("strategy_learning_path", "") or ""),
            "freeze_strategy_snapshot": bool(config.get("freeze_strategy_snapshot", False)),
        }
        if any(strategy_manifest.values()):
            dump_json(strategy_manifest, bug_dir / "strategy.json")
    context = target_context(list(raw_results))
    dump_json(context.target_dicts(), bug_dir / "targets.json")
    dump_json(context.to_dict(), bug_dir / "target_context.json")
    dump_json(collect_environment(), bug_dir / "environment.json")
    repro = f'''#!/usr/bin/env python3
from datadiff.config import ExperimentConfig
from datadiff.dsl import Case
from datadiff.runner import run_loaded_case
from datadiff.util import load_json

here = __import__("pathlib").Path(__file__).parent
case = Case.from_dict(load_json(here / "case.json"))
config_data = load_json(here / "config.json")
config = ExperimentConfig(**config_data) if config_data else ExperimentConfig()
result = run_loaded_case(case, backends={list(raw_results)!r}, config=config, save_artifact=False)
print(result["status"])
for f in result["findings"]:
    print(f)
'''
    (bug_dir / "reproduce.py").write_text(repro, encoding="utf-8")
    (bug_dir / "report.md").write_text(_render_issue_artifact_report(case, findings), encoding="utf-8")
    return bug_dir


def _render_issue_artifact_report(case: Case, findings: list[Finding]) -> str:
    lines = [f"# Bug Artifact: {case.case_id}", "", f"Seed: `{case.seed}`", "", "## Program", "", "```json"]
    import json
    lines.append(json.dumps(case.program.to_dict(), ensure_ascii=False, indent=2))
    lines.extend(["```", "", "## Findings"])
    for f in findings:
        lines.append(
            f"- **{f.kind}** severity={f.severity} root={f.root_cause} "
            f"oracle={f.oracle} confidence={f.confidence} "
            f"triage={f.triage_verdict} triage_confidence={f.triage_confidence} "
            f"suspicious={f.suspicious_backends}: {f.evidence}"
        )
        if f.false_positive_reason:
            lines.append(f"  - false_positive_reason={f.false_positive_reason}")
        if f.triage_evidence:
            lines.append(f"  - triage_evidence={f.triage_evidence}")
    lines.extend(["", "## Reproduce", "", "```bash", "python reproduce.py", "```"])
    lines.extend(["", "## Environment", "", "```json"])
    lines.append(json.dumps(collect_environment(), ensure_ascii=False, indent=2))
    lines.append("```")
    return "\n".join(lines) + "\n"


save_bug_artifact = save_issue_artifact
_bug_report = _render_issue_artifact_report
