#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "reports" / "ubuntu-datafusion-artifact-manifest.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "ubuntu-datafusion-artifact-bundle.tar.gz"
MANIFEST_PATH_SECTIONS = (
    "summary_document",
    "methodology_roadmap",
    "environment",
    "primary_artifact",
    "ablation_audit",
    "pattern_analyses",
    "experiments",
    "harness_evidence",
)


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    output_path = Path(args.output)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = collect_manifest_paths(manifest)
    members = bundle_members(paths, root=PROJECT_ROOT)
    validate_paths(paths, root=PROJECT_ROOT)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_bundle(paths, output_path, root=PROJECT_ROOT)
    digest = sha256_file(output_path)
    digest_path = output_path.with_suffix(output_path.suffix + ".sha256")
    digest_path.write_text(f"{digest}  {output_path.name}\n", encoding="utf-8")
    print(f"bundle={output_path}")
    print(f"sha256={digest_path}")
    print(f"files={len(members)}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Ubuntu DataFusion evidence artifact bundle.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def collect_manifest_paths(manifest: dict[str, Any]) -> list[str]:
    paths: set[str] = set()
    for section in MANIFEST_PATH_SECTIONS:
        paths.update(_collect_paths(manifest.get(section)))
    return sorted(paths)


def _collect_paths(value: Any) -> set[str]:
    if isinstance(value, str):
        candidate = value.strip()
        return {candidate} if _looks_like_repo_path(candidate) else set()
    if isinstance(value, dict):
        paths: set[str] = set()
        for item in value.values():
            paths.update(_collect_paths(item))
        return paths
    if isinstance(value, list):
        paths: set[str] = set()
        for item in value:
            paths.update(_collect_paths(item))
        return paths
    return set()


def _looks_like_repo_path(value: str) -> bool:
    return bool(value) and "://" not in value and "/" in value


def validate_paths(paths: list[str], *, root: Path) -> None:
    missing = []
    empty = []
    for rel_path in paths:
        path = root / rel_path
        if not path.exists():
            missing.append(rel_path)
        elif path.is_file() and path.stat().st_size == 0:
            empty.append(rel_path)
    if missing or empty:
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if empty:
            details.append("empty=" + ",".join(empty))
        raise FileNotFoundError("; ".join(details))


def write_bundle(paths: list[str], output_path: Path, *, root: Path) -> None:
    members = bundle_members(paths, root=root)
    with output_path.open("wb") as handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=handle, mtime=0) as gz_handle:
            with tarfile.open(fileobj=gz_handle, mode="w") as tar:
                for rel_path in members:
                    tar.add(
                        root / rel_path,
                        arcname=rel_path,
                        recursive=False,
                        filter=normalize_tarinfo,
                    )


def bundle_members(paths: list[str], *, root: Path) -> list[str]:
    members: set[str] = set()
    for rel_path in paths:
        path = root / rel_path
        if path.is_dir():
            members.update(
                str(child.relative_to(root))
                for child in path.rglob("*")
                if child.is_file()
            )
        else:
            members.add(rel_path)
    return sorted(members)


def normalize_tarinfo(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo:
    tarinfo.uid = 0
    tarinfo.gid = 0
    tarinfo.uname = ""
    tarinfo.gname = ""
    tarinfo.mtime = 0
    tarinfo.mode = 0o755 if tarinfo.mode & 0o111 else 0o644
    return tarinfo


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
