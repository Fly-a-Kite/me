#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from datadiff.util import REPORTS_DIR, load_json, read_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATADIFF = PROJECT_ROOT / ".venv" / "bin" / "datadiff"

DEFAULT_EXPERIMENTS = [
    ("core_ablation", PROJECT_ROOT / "runs" / "experiment-20260511T193546.json"),
    ("dataframe_generalization", PROJECT_ROOT / "runs" / "experiment-20260511T193709.json"),
    ("embedded_sql_generalization", PROJECT_ROOT / "runs" / "experiment-20260511T193901.json"),
    ("edge_float_boundary", PROJECT_ROOT / "runs" / "experiment-20260511T195036.json"),
    ("reducer_validation", PROJECT_ROOT / "runs" / "experiment-20260511T194215.json"),
]

NOISY_ABLATION_PRESETS = {"no_type_aware", "no_normalizer"}
EXPERIMENT_PRIORITY = {
    "core_ablation": 0,
    "dataframe_generalization": 1,
    "embedded_sql_generalization": 2,
    "reducer_validation": 3,
    "edge_float_boundary": 4,
}
PRESET_PRIORITY = {
    "baseline": 0,
    "guided": 1,
    "no_feedback": 2,
    "metamorphic": 3,
    "reducer": 4,
    "edge_float": 8,
    "no_type_aware": 9,
    "no_normalizer": 9,
}
ROOT_PRIORITY = {
    "filter_predicate": 0,
    "arithmetic_expression": 1,
    "groupby_aggregation": 2,
    "exception_taxonomy": 3,
    "type_cast": 4,
    "join_semantics": 5,
    "string_expression": 6,
    "ordering_or_limit": 7,
    "unknown": 8,
}


@dataclass(slots=True)
class Candidate:
    experiment: str
    target_suite: str
    preset: str
    seed: int
    run_file: str
    row_index: int
    case_id: str
    case_seed: int
    bug_dir: str
    kind: str
    root_cause: str
    signature: str
    suspicious_backends: tuple[str, ...]
    triage_verdict: str
    paper_status: str
    confidence: str
    evidence: str
    validate_returncode: int | None = None
    validate_status: str = ""
    validate_stdout: str = ""
    validate_stderr: str = ""
    triage_returncode: int | None = None
    triage_stdout: str = ""
    triage_stderr: str = ""
    final_verdict: str = ""
    final_paper_status: str = ""
    final_confidence: str = ""
    reduced: bool = False
    reduced_rows: int | None = None
    reduced_operations: int | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def key(self) -> tuple[Any, ...]:
        return (self.kind, self.root_cause, self.signature, self.suspicious_backends)

    @property
    def priority(self) -> tuple[int, int, int, int, int, str]:
        noisy = 1 if self.preset in NOISY_ABLATION_PRESETS else 0
        return (
            noisy,
            EXPERIMENT_PRIORITY.get(self.experiment, 99),
            PRESET_PRIORITY.get(self.preset, 99),
            ROOT_PRIORITY.get(self.root_cause, 99),
            self.seed,
            self.case_id,
        )


def main() -> int:
    args = parse_args()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    manifests = parse_manifests(args.manifest, use_defaults=not args.run_file)
    candidates = collect_candidates(manifests, include_noisy_ablations=args.include_noisy_ablations)
    candidates.extend(collect_run_file_candidates([Path(value) for value in args.run_file]))
    candidates = dedupe_candidates(candidates)
    selected = select_candidates(candidates, limit=args.limit, per_root=args.per_root)
    if args.execute:
        execute_triage(selected, reduce=args.reduce, timeout_s=args.timeout_s)
    write_triage_outputs(selected, all_candidates=candidates)
    print(REPORTS_DIR / "candidate-bug-artifacts.csv")
    print(REPORTS_DIR / "candidate-bug-triage.md")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select and optionally validate representative candidate implementation-bug artifacts.",
    )
    parser.add_argument(
        "--manifest",
        action="append",
        default=[],
        help="experiment manifest path; defaults to the current medium-experiment manifest set",
    )
    parser.add_argument(
        "--run-file",
        action="append",
        default=[],
        help="single run log to scan directly; may be repeated",
    )
    parser.add_argument("--limit", type=int, default=12, help="maximum selected artifacts")
    parser.add_argument("--per-root", type=int, default=3, help="soft cap per root cause before filling leftovers")
    parser.add_argument(
        "--include-noisy-ablations",
        action="store_true",
        help="include no_type_aware and no_normalizer candidates in the selection pool",
    )
    parser.add_argument("--execute", action="store_true", help="run validate-artifact and triage-artifact")
    parser.add_argument("--reduce", action="store_true", help="reduce selected artifacts during triage")
    parser.add_argument("--timeout-s", type=float, default=240.0, help="timeout per validation/triage command")
    return parser.parse_args()


def parse_manifests(values: list[str], *, use_defaults: bool) -> list[tuple[str, Path]]:
    if values:
        return [(Path(value).stem, Path(value)) for value in values]
    return DEFAULT_EXPERIMENTS if use_defaults else []


def collect_candidates(
    manifests: list[tuple[str, Path]],
    *,
    include_noisy_ablations: bool,
) -> list[Candidate]:
    by_key: dict[tuple[Any, ...], Candidate] = {}
    for experiment, manifest_path in manifests:
        manifest = load_json(manifest_path)
        target_suite = str(manifest.get("target_suite", ""))
        for run in manifest.get("runs", []):
            preset = str(run.get("preset", ""))
            if preset in NOISY_ABLATION_PRESETS and not include_noisy_ablations:
                continue
            run_file = Path(run["run_file"])
            for row_index, row in enumerate(read_jsonl(run_file)):
                bug_dir = str(row.get("bug_dir", ""))
                if not bug_dir:
                    continue
                case = row.get("case", {})
                for finding in row.get("findings", []):
                    if finding.get("triage_verdict") != "candidate_implementation_bug":
                        continue
                    candidate = Candidate(
                        experiment=experiment,
                        target_suite=target_suite,
                        preset=preset,
                        seed=int(run.get("seed", 0)),
                        run_file=str(run_file),
                        row_index=row_index,
                        case_id=str(case.get("case_id", "")),
                        case_seed=int(case.get("seed", 0)),
                        bug_dir=bug_dir,
                        kind=str(finding.get("kind", "")),
                        root_cause=str(finding.get("root_cause", "unknown")),
                        signature=str(finding.get("signature", "")),
                        suspicious_backends=tuple(sorted(finding.get("suspicious_backends", []))),
                        triage_verdict=str(finding.get("triage_verdict", "")),
                        paper_status=str(finding.get("paper_status", "")),
                        confidence=str(finding.get("confidence", "")),
                        evidence=str(finding.get("evidence", "")),
                    )
                    existing = by_key.get(candidate.key)
                    if existing is None or candidate.priority < existing.priority:
                        by_key[candidate.key] = candidate
    return sorted(by_key.values(), key=lambda item: item.priority)


def collect_run_file_candidates(run_files: list[Path]) -> list[Candidate]:
    candidates = []
    for run_file in run_files:
        for row_index, row in enumerate(read_jsonl(run_file)):
            bug_dir = str(row.get("bug_dir", ""))
            if not bug_dir:
                continue
            case = row.get("case", {})
            config = row.get("config", {})
            for finding in row.get("findings", []):
                if finding.get("triage_verdict") != "candidate_implementation_bug":
                    continue
                candidates.append(
                    Candidate(
                        experiment=run_file.stem,
                        target_suite="run_file",
                        preset=str(config.get("generator_profile", "longrun")),
                        seed=int(case.get("seed", 0)),
                        run_file=str(run_file),
                        row_index=row_index,
                        case_id=str(case.get("case_id", "")),
                        case_seed=int(case.get("seed", 0)),
                        bug_dir=bug_dir,
                        kind=str(finding.get("kind", "")),
                        root_cause=str(finding.get("root_cause", "unknown")),
                        signature=str(finding.get("signature", "")),
                        suspicious_backends=tuple(sorted(finding.get("suspicious_backends", []))),
                        triage_verdict=str(finding.get("triage_verdict", "")),
                        paper_status=str(finding.get("paper_status", "")),
                        confidence=str(finding.get("confidence", "")),
                        evidence=str(finding.get("evidence", "")),
                    )
                )
    return candidates


def dedupe_candidates(candidates: list[Candidate]) -> list[Candidate]:
    by_key: dict[tuple[Any, ...], Candidate] = {}
    for candidate in candidates:
        existing = by_key.get(candidate.key)
        if existing is None or candidate.priority < existing.priority:
            by_key[candidate.key] = candidate
    return sorted(by_key.values(), key=lambda item: item.priority)


def select_candidates(candidates: list[Candidate], *, limit: int, per_root: int) -> list[Candidate]:
    selected: list[Candidate] = []
    seen_bug_dirs: set[str] = set()
    root_counts: Counter[str] = Counter()
    for candidate in candidates:
        if len(selected) >= limit:
            break
        if candidate.bug_dir in seen_bug_dirs:
            continue
        if root_counts[candidate.root_cause] >= per_root:
            continue
        selected.append(candidate)
        seen_bug_dirs.add(candidate.bug_dir)
        root_counts[candidate.root_cause] += 1

    for candidate in candidates:
        if len(selected) >= limit:
            break
        if candidate.bug_dir in seen_bug_dirs:
            continue
        selected.append(candidate)
        seen_bug_dirs.add(candidate.bug_dir)
    return selected


def execute_triage(candidates: list[Candidate], *, reduce: bool, timeout_s: float) -> None:
    for candidate in candidates:
        validate = run_command(
            [str(DATADIFF), "validate-artifact", "--bug", candidate.bug_dir],
            timeout_s=timeout_s,
        )
        candidate.validate_returncode = validate.returncode
        candidate.validate_stdout = validate.stdout
        candidate.validate_stderr = validate.stderr
        candidate.validate_status = parse_validate_status(validate.stdout)

        triage_cmd = [str(DATADIFF), "triage-artifact", "--bug", candidate.bug_dir]
        if reduce:
            triage_cmd.append("--reduce")
        triage = run_command(triage_cmd, timeout_s=timeout_s)
        candidate.triage_returncode = triage.returncode
        candidate.triage_stdout = triage.stdout
        candidate.triage_stderr = triage.stderr
        load_final_triage(candidate)
        if validate.returncode != 0:
            candidate.notes.append("validation command returned non-zero")
        if triage.returncode != 0:
            candidate.notes.append("triage command returned non-zero")


def run_command(argv: list[str], *, timeout_s: float) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            argv,
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            argv,
            124,
            stdout=exc.stdout or "",
            stderr=(exc.stderr or "") + f"\ntimeout after {timeout_s}s",
        )


def parse_validate_status(stdout: str) -> str:
    for line in stdout.splitlines():
        if line.startswith("status="):
            return line.split("=", 1)[1].strip()
    return ""


def load_final_triage(candidate: Candidate) -> None:
    triage_path = Path(candidate.bug_dir) / "triage.json"
    if not triage_path.exists():
        return
    report = load_json(triage_path)
    candidate.final_verdict = str(report.get("verdict", ""))
    candidate.final_paper_status = str(report.get("paper_status", ""))
    candidate.final_confidence = str(report.get("triage_confidence", ""))
    candidate.reduced = bool(report.get("reduced", False))
    candidate.reduced_rows = _optional_int(report.get("rows"))
    candidate.reduced_operations = _optional_int(report.get("operations"))


def write_triage_outputs(selected: list[Candidate], *, all_candidates: list[Candidate]) -> None:
    write_csv(REPORTS_DIR / "candidate-bug-artifacts.csv", selected)
    write_markdown(REPORTS_DIR / "candidate-bug-triage.md", selected, all_candidates=all_candidates)


def write_outputs(selected: list[Candidate], *, all_candidates: list[Candidate]) -> None:
    # Compatibility alias for older callers.
    write_triage_outputs(selected, all_candidates=all_candidates)


def write_csv(path: Path, candidates: list[Candidate]) -> None:
    fieldnames = [
        "experiment",
        "target_suite",
        "preset",
        "seed",
        "case_id",
        "case_seed",
        "kind",
        "root_cause",
        "signature",
        "suspicious_backends",
        "initial_verdict",
        "initial_paper_status",
        "final_verdict",
        "final_paper_status",
        "final_confidence",
        "validate_status",
        "validate_returncode",
        "triage_returncode",
        "reduced",
        "reduced_rows",
        "reduced_operations",
        "bug_dir",
        "run_file",
        "row_index",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(
                {
                    "experiment": candidate.experiment,
                    "target_suite": candidate.target_suite,
                    "preset": candidate.preset,
                    "seed": candidate.seed,
                    "case_id": candidate.case_id,
                    "case_seed": candidate.case_seed,
                    "kind": candidate.kind,
                    "root_cause": candidate.root_cause,
                    "signature": candidate.signature,
                    "suspicious_backends": ",".join(candidate.suspicious_backends),
                    "initial_verdict": candidate.triage_verdict,
                    "initial_paper_status": candidate.paper_status,
                    "final_verdict": candidate.final_verdict,
                    "final_paper_status": candidate.final_paper_status,
                    "final_confidence": candidate.final_confidence,
                    "validate_status": candidate.validate_status,
                    "validate_returncode": candidate.validate_returncode,
                    "triage_returncode": candidate.triage_returncode,
                    "reduced": candidate.reduced,
                    "reduced_rows": candidate.reduced_rows,
                    "reduced_operations": candidate.reduced_operations,
                    "bug_dir": candidate.bug_dir,
                    "run_file": candidate.run_file,
                    "row_index": candidate.row_index,
                    "notes": "; ".join(candidate.notes),
                }
            )


def write_markdown(path: Path, selected: list[Candidate], *, all_candidates: list[Candidate]) -> None:
    final_verdicts = Counter(candidate.final_verdict or "not_executed" for candidate in selected)
    roots = Counter(candidate.root_cause for candidate in selected)
    source_counts = Counter(f"{candidate.experiment}/{candidate.preset}" for candidate in selected)
    all_roots = Counter(candidate.root_cause for candidate in all_candidates)
    lines = [
        "# Candidate Bug Artifact Triage",
        "",
        "## Selection",
        "",
        f"- Candidate pool after deduplication: {len(all_candidates)}",
        f"- Selected artifacts: {len(selected)}",
        f"- Selection excludes noisy ablations by default: {', '.join(sorted(NOISY_ABLATION_PRESETS))}",
        f"- Selected roots: {_counter_summary(roots)}",
        f"- Candidate-pool roots: {_counter_summary(all_roots)}",
        f"- Sources: {_counter_summary(source_counts)}",
        f"- Final verdicts: {_counter_summary(final_verdicts)}",
        "",
        "## Artifacts",
        "",
        "| # | source | root | suspicious | validate | final verdict | reduced | artifact |",
        "|---:|---|---|---|---|---|---:|---|",
    ]
    for idx, candidate in enumerate(selected, start=1):
        source = f"{candidate.experiment}/{candidate.preset}/seed={candidate.seed}"
        validate = candidate.validate_status or "not_run"
        final = candidate.final_verdict or "not_run"
        reduced = "yes" if candidate.reduced else "no"
        lines.append(
            f"| {idx} | {source} | `{candidate.root_cause}` | "
            f"`{','.join(candidate.suspicious_backends) or 'none'}` | `{validate}` | "
            f"`{final}` | {reduced} | `{candidate.bug_dir}` |"
        )
    lines.extend(["", "## Interpretation", ""])
    lines.extend(interpretation_notes(selected))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def interpretation_notes(selected: list[Candidate]) -> list[str]:
    if not selected:
        return ["- No candidate implementation-bug artifacts were selected."]
    verdicts = Counter(candidate.final_verdict or "not_executed" for candidate in selected)
    notes = []
    if verdicts.get("candidate_implementation_bug"):
        notes.append(
            "- At least one selected artifact remains a candidate implementation bug after reproduction triage; these are the next cases for backend-specific documentation checks and upstream issue confirmation."
        )
    if verdicts.get("semantic_divergence_needs_confirmation"):
        notes.append(
            "- Some initial candidate bugs downgraded to semantic divergences after current triage rules; do not count them as confirmed bugs in the paper table."
        )
    if verdicts.get("documented_semantic_divergence"):
        notes.append(
            "- Documented semantic divergences should stay in the boundary-semantics experiment family, not in confirmed bug counts."
        )
    if verdicts.get("not_reproduced"):
        notes.append(
            "- Non-reproduced artifacts need dependency/version checks before they can support any claim."
        )
    if verdicts == Counter({"not_executed": len(selected)}):
        notes.append("- Run this script with `--execute --reduce` to validate and classify the selected artifacts.")
    return notes or ["- No additional interpretation notes."]


def _counter_summary(counter: Counter[str], limit: int = 8) -> str:
    return "; ".join(f"{key}:{value}" for key, value in counter.most_common(limit)) or "none"


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
