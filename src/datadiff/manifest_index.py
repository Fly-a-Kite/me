from __future__ import annotations

from pathlib import Path
from typing import Any

from datadiff.util import load_json


def manifest_index_files(
    index_files: list[str] | tuple[str, ...],
) -> tuple[list[Path], list[Path], list[Path]]:
    manifest_files: list[Path] = []
    extra_manifest_files: list[Path] = []
    paper_run_journal_files: list[Path] = []
    for raw_path in index_files or []:
        index_path = Path(raw_path)
        data = load_json(index_path)
        if not isinstance(data, dict):
            raise ValueError(f"manifest index is not a JSON object: {index_path}")
        manifest_files.extend(paths_from_index_value(data.get("manifest_files", [])))
        extra_manifest_files.extend(paths_from_index_value(data.get("extra_manifest_files", [])))
        for command in data.get("commands", []) or []:
            if not isinstance(command, dict):
                continue
            manifest_files.extend(paths_from_index_value(command.get("manifest_files", [])))
            extra_manifest_files.extend(paths_from_index_value(command.get("extra_manifest_files", [])))
            paper_run_journal_files.extend(paths_from_index_value(command.get("paper_run_journal_files", [])))
    return (
        dedupe_paths(manifest_files),
        dedupe_paths(extra_manifest_files),
        dedupe_paths(paper_run_journal_files),
    )


def paths_from_index_value(value: Any) -> list[Path]:
    if isinstance(value, str):
        items = [part.strip() for part in value.split(",")]
    elif isinstance(value, (list, tuple, set)):
        items = [str(item).strip() for item in value]
    else:
        items = []
    return [Path(item) for item in items if item]


def dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    deduped: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


_final_readiness_manifest_index_files = manifest_index_files
_paths_from_index_value = paths_from_index_value
_dedupe_paths = dedupe_paths
